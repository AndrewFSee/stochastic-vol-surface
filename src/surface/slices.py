"""Per-expiry surface construction: parity forwards, OTM Black IVs, SVI slices.

This is the pipeline :meth:`VolSurface.from_chain` uses.  It replaced an
older pooled-bin approach (since removed; see commit cf8bbea) that merged
every expiry within ±30% of a target tenor into one SVI fit and gave them all
the same time-to-expiry.  That mixing was the dominant source of day-to-day
noise in the stored surfaces.

Pipeline, per expiry
--------------------
1. **Clean quotes** — two-sided markets only (``bid > 0``, ``ask > bid``) with
   a bounded relative spread.
2. **Implied forward** — from put-call parity on the near-the-money strikes,
   ``F = K + (C - P) / D``.  This absorbs dividends, borrow and any rate
   mismatch, none of which ``S·e^{rT}`` accounts for (SPY's naive forward is
   ~0.9% too high at two years; XLF's ~2%).
3. **OTM Black IVs** — puts below the forward, calls above, inverted from the
   mid with Black-76 on that forward.  In-the-money quotes are dropped: their
   spreads are wide relative to time value, and they carry the early-exercise
   premium of American options.
4. **SVI fit** — quasi-explicit (Zeliade): for fixed ``(m, sigma)`` raw SVI is
   linear in ``(a, b(1+rho), b(1-rho))``, solved by weighted least squares
   under ``a >= 0`` (positive variance) and wing slopes ``<= 2`` (Lee's
   moment bound).  Residuals are weighted to approximate IV error scaled by
   the quote's bid-ask spread in vol terms.  One round of outlier rejection.

Across expiries
---------------
Total variance is interpolated **linearly in T at fixed forward
log-moneyness** between the two expiries bracketing each grid tenor (the same
construction VIX uses for its constant 30-day maturity).  Every grid cell is
flagged *observed* only when it lies inside the quoted strike range of both
bracketing expiries; everything else is extrapolation and is marked as such.
"""

from __future__ import annotations

import logging
import math
from dataclasses import asdict, dataclass
from typing import Callable, Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import ndtr

from src.surface.implied_vol import black_implied_vol, black_vega

logger = logging.getLogger(__name__)

#: Stamped into every saved surface so corpora built by different
#: construction methods are never silently mixed.
BUILDER_VERSION = "expiry-svi/1"

#: Below this fraction of rows with a usable two-sided quote, the chain is
#: treated as price-less and the quoted IV column is used instead (some
#: historical sessions publish settled IVs with a blank bid-ask).
MIN_PRICED_FRACTION = 0.10

#: Lee's moment bound: the asymptotic slope of total variance in k is <= 2.
MAX_WING_SLOPE = 2.0

#: Floor on a quote's bid-ask spread in vol terms, so a handful of very tight
#: ATM quotes cannot take all the weight in the fit.
MIN_SPREAD_IV = 0.005

RateFn = Callable[[float], float]

#: Integration grid in forward log-moneyness for variance-swap integrals.
_K_INTEGRATION = np.linspace(-2.5, 1.5, 4001)


# ────────────────────────────────────────────────────────────────────────────
# Slice result
# ────────────────────────────────────────────────────────────────────────────


@dataclass
class SliceFit:
    """A fitted raw-SVI smile for one expiry."""

    T: float
    forward: float
    discount: float
    forward_source: str          # "parity" | "carry"
    n_quotes: int                # quotes used in the final fit
    n_outliers: int
    k_min: float                 # quoted log-moneyness range
    k_max: float
    a: float
    b: float
    rho: float
    m: float
    sigma: float
    rmse_iv: float               # unweighted IV RMSE over the quotes used
    expiration: Optional[str] = None

    def total_variance(self, k: np.ndarray) -> np.ndarray:
        """Raw-SVI total variance at forward log-moneyness *k*."""
        return svi_total_variance(np.asarray(k, dtype=float),
                                  self.a, self.b, self.rho, self.m, self.sigma)

    def iv(self, k: np.ndarray) -> np.ndarray:
        """Implied vol at forward log-moneyness *k* (pure SVI)."""
        return np.sqrt(np.maximum(self.total_variance(k), 0.0) / self.T)

    def total_variance_extrapolated(self, k: np.ndarray) -> np.ndarray:
        """SVI inside the quoted range, tangent-line continuation outside.

        Beyond the last quoted strike the SVI wing is pure extrapolation, and
        on a sparse expiry it can be wild (a 7-quote XLF slice extrapolated to
        230% vol at k=-0.4).  Continuing total variance along the tangent at
        the boundary keeps the smile convex, while clamping the slope so total
        variance never *decreases* moving away from the money (flat-vol floor)
        keeps it positive.
        """
        k = np.asarray(k, dtype=float)
        w = self.total_variance(k)
        lo, hi = self.k_min, self.k_max
        slope_lo = min(self._slope(lo), 0.0)
        slope_hi = max(self._slope(hi), 0.0)
        w_lo, w_hi = float(self.total_variance(lo)), float(self.total_variance(hi))
        w = np.where(k < lo, w_lo + slope_lo * (k - lo), w)
        w = np.where(k > hi, w_hi + slope_hi * (k - hi), w)
        return w

    def variance_swap_total_variance(self) -> float:
        """Fair variance-swap total variance ``σ²_VS·T`` for this expiry.

        ``2 ∫ OTM(k)·e^{-k} dk`` with OTM prices normalised to a unit
        forward, integrated over the (extrapolated) smile.
        """
        k = _K_INTEGRATION
        sw = np.sqrt(np.maximum(self.total_variance_extrapolated(k), 1e-12))
        d1 = -k / sw + 0.5 * sw
        d2 = d1 - sw
        call = ndtr(d1) - np.exp(k) * ndtr(d2)
        put = np.exp(k) * ndtr(-d2) - ndtr(-d1)
        otm = np.where(k >= 0, call, put)
        return float(2.0 * np.trapezoid(otm * np.exp(-k), k))

    def _slope(self, k: float) -> float:
        """dw/dk of the SVI slice."""
        y = k - self.m
        return float(self.b * (self.rho + y / math.sqrt(y * y + self.sigma ** 2)))

    def to_dict(self) -> dict:
        """Plain-float dict, safe for JSON metadata."""
        return {key: (float(v) if isinstance(v, (int, float, np.floating, np.integer))
                      and not isinstance(v, bool) else v)
                for key, v in asdict(self).items()}


def svi_total_variance(k, a, b, rho, m, sigma):
    """Gatheral raw SVI: ``w(k) = a + b(rho(k-m) + sqrt((k-m)^2 + sigma^2))``."""
    y = k - m
    return a + b * (rho * y + np.sqrt(y * y + sigma * sigma))


# ────────────────────────────────────────────────────────────────────────────
# Quotes → OTM implied vols
# ────────────────────────────────────────────────────────────────────────────


def _constant_rate(r: float) -> RateFn:
    return lambda _T: r


def implied_forward(
    expiry_quotes: pd.DataFrame,
    discount: float,
    carry_forward: float,
    T: float,
    n_pairs: int = 6,
) -> tuple[float, str]:
    """Forward for one expiry from put-call parity, ``F = K + (C - P) / D``.

    Uses the *n_pairs* strikes with both a call and a put quoted whose
    ``|C - P|`` is smallest (i.e. nearest the forward), and takes the median.
    Falls back to *carry_forward* when fewer than two pairs exist or the
    parity forward is implausibly far from it.
    """
    q = expiry_quotes
    calls = q.loc[q["is_call"]].groupby("strike")["mid"].mean()
    puts = q.loc[~q["is_call"]].groupby("strike")["mid"].mean()
    both = pd.concat([calls.rename("c"), puts.rename("p")], axis=1, join="inner")
    if len(both) < 2:
        return carry_forward, "carry"

    both = both.assign(gap=(both["c"] - both["p"]).abs()).nsmallest(n_pairs, "gap")
    fwd = both.index.to_numpy() + (both["c"] - both["p"]).to_numpy() / discount
    F = float(np.median(fwd))

    # Dividends/borrow move the forward by a few percent a year at most; a
    # larger gap means the parity pairs themselves are bad quotes.
    if not np.isfinite(F) or F <= 0 or abs(math.log(F / carry_forward)) > 0.05 + 0.05 * T:
        return carry_forward, "carry"
    return F, "parity"


def otm_quotes(
    chain: pd.DataFrame,
    spot: float,
    rate_fn: RateFn,
    *,
    min_T: float = 2 / 365,
    max_rel_spread: float = 0.6,
    min_delta: float = 0.01,
) -> tuple[pd.DataFrame, dict[float, tuple[float, float, str]]]:
    """Clean a raw chain into OTM quotes with Black-76 implied vols.

    Returns ``(quotes, forwards)`` where *quotes* has one row per usable OTM
    option (columns ``T, strike, is_call, k, iv, spread_iv, forward,
    discount`` and ``expiration`` when known) and *forwards* maps each
    expiry's ``T`` to ``(forward, discount, source)``.
    """
    df = chain.copy()
    if "T" not in df.columns:
        raise ValueError("chain must have column 'T' (time-to-expiry in years)")
    df = df.dropna(subset=["strike", "T", "option_type"])
    df = df[(df["T"] >= min_T - 1e-9) & (df["strike"] > 0)]
    df["is_call"] = df["option_type"].astype(str).str.lower().str.startswith("c")

    has_ba = {"bid", "ask"} <= set(df.columns)
    if has_ba:
        two_sided = df["bid"].gt(0) & df["ask"].gt(df["bid"])
    else:
        two_sided = pd.Series(False, index=df.index)

    iv_col = next((c for c in ("implied_volatility_market", "implied_volatility")
                   if c in df.columns and df[c].gt(0).any()), None)
    use_prices = two_sided.mean() >= MIN_PRICED_FRACTION if len(df) else False
    if not use_prices and "mid" in df.columns and not has_ba:
        # A mid with no bid/ask (e.g. a pre-aggregated source) is still a price.
        use_prices = df["mid"].gt(0).mean() >= MIN_PRICED_FRACTION and iv_col is None

    if use_prices:
        if has_ba:
            df = df[two_sided].copy()
            df["mid"] = 0.5 * (df["bid"] + df["ask"])
            df = df[(df["ask"] - df["bid"]) <= max_rel_spread * df["mid"]]
        else:
            df = df[df["mid"] > 0].copy()
    elif iv_col is None:
        raise ValueError("chain has neither usable bid/ask quotes nor a quoted IV")
    else:
        logger.info("Only %.1f%% of rows are two-sided — using quoted IV column %r",
                    100 * float(two_sided.mean()), iv_col)
        df = df[df[iv_col] > 0].copy()

    # Group by expiry.  T is unique per expiry within one snapshot, and is
    # available even for chains without an `expiration` column.
    df["T_key"] = df["T"].round(8)
    frames: list[pd.DataFrame] = []
    forwards: dict[float, tuple[float, float, str]] = {}

    for T, g in df.groupby("T_key"):
        T = float(T)
        r = float(rate_fn(T))
        D = math.exp(-r * T)
        carry = spot * math.exp(r * T)
        if use_prices:
            F, source = implied_forward(g, D, carry, T)
        else:
            F, source = carry, "carry"
        forwards[T] = (F, D, source)

        otm = g[g["is_call"] == (g["strike"] >= F)].copy()
        if otm.empty:
            continue
        K = otm["strike"].to_numpy()
        is_call = otm["is_call"].to_numpy()

        if use_prices:
            iv = black_implied_vol(otm["mid"].to_numpy(), F, K, T, D, is_call)
            vega = black_vega(F, K, T, np.nan_to_num(iv, nan=0.2), D)
            if has_ba:
                with np.errstate(divide="ignore", invalid="ignore"):
                    spread_iv = (otm["ask"] - otm["bid"]).to_numpy() / vega
            else:
                spread_iv = np.full(len(otm), np.nan)
        else:
            iv = otm[iv_col].to_numpy(dtype=float)
            spread_iv = np.full(len(otm), np.nan)

        k = np.log(K / F)
        # OTM forward delta; drops strikes so far out that the quote is noise.
        with np.errstate(divide="ignore", invalid="ignore"):
            d1 = (-k + 0.5 * iv ** 2 * T) / (iv * math.sqrt(T))
        delta = np.where(is_call, ndtr(d1), ndtr(-d1))

        out = pd.DataFrame({
            "T": T, "strike": K, "is_call": is_call, "k": k, "iv": iv,
            "spread_iv": spread_iv, "forward": F, "discount": D,
        })
        if "expiration" in otm.columns:
            out["expiration"] = otm["expiration"].to_numpy()
        keep = np.isfinite(iv) & (iv > 0) & (delta >= min_delta)
        frames.append(out[keep])

    quotes = (pd.concat(frames, ignore_index=True) if frames
              else pd.DataFrame(columns=["T", "strike", "is_call", "k", "iv",
                                         "spread_iv", "forward", "discount"]))
    return quotes, forwards


# ────────────────────────────────────────────────────────────────────────────
# SVI slice fit
# ────────────────────────────────────────────────────────────────────────────


_LOWER = np.zeros(3)
_UPPER = np.array([np.inf, MAX_WING_SLOPE, MAX_WING_SLOPE])

def _active_sets() -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Every way of pinning (a, u, v) free / at lower / at upper bound.

    Returned as ``(free_idx, fixed_idx, fixed_values)``, most-free first and
    with ``a`` pinned at zero early — in practice that is where the optimum
    usually sits.
    """
    out = []
    for sa in ("free", "lo"):
        for su in ("free", "lo", "hi"):
            for sv in ("free", "lo", "hi"):
                states = (sa, su, sv)
                free = np.array([i for i, s in enumerate(states) if s == "free"], dtype=int)
                fixed = np.array([i for i, s in enumerate(states) if s != "free"], dtype=int)
                vals = np.array([_LOWER[i] if states[i] == "lo" else _UPPER[i] for i in fixed])
                out.append((free, fixed, vals))
    out.sort(key=lambda p: (-len(p[0]), 0 if 0 in p[1] else 1))
    return out


_ACTIVE_SETS = _active_sets()


def _box_lsq3(G: np.ndarray, c: np.ndarray, bb: float) -> tuple[np.ndarray, float]:
    """Minimise ``x'Gx - 2c'x + bb`` over the (a, u, v) box, exactly.

    ``G = A'A`` and ``c = A'b`` are the 3x3 normal equations.  Candidates are
    tried active set by active set; the first one that is feasible *and*
    satisfies the KKT sign conditions is the global optimum (the problem is a
    convex QP), so this usually stops after one or two 3x3 solves.  That
    matters because it runs on the order of a thousand times per slice.
    """
    tol = 1e-10 * (float(np.abs(c).max()) + 1.0)
    best_x, best_f = None, np.inf
    for free, fixed, vals in _ACTIVE_SETS:
        x = np.zeros(3)
        x[fixed] = vals
        if len(free):
            rhs = c[free] - G[np.ix_(free, fixed)] @ vals if len(fixed) else c[free]
            try:
                x[free] = np.linalg.solve(G[np.ix_(free, free)], rhs)
            except np.linalg.LinAlgError:
                continue
            if np.any(x < _LOWER - 1e-12) or np.any(x > _UPPER + 1e-12):
                continue
        grad = G @ x - c
        at_lo = x[fixed] <= _LOWER[fixed]
        kkt = np.all(np.where(at_lo, grad[fixed] >= -tol, grad[fixed] <= tol))
        f = float(x @ G @ x - 2 * c @ x + bb)
        if kkt:
            return x, max(f, 0.0)
        if f < best_f:  # feasible but not provably optimal; keep as fallback
            best_x, best_f = x, f
    return best_x, max(best_f, 0.0)


def _svi_inner(y: np.ndarray, w: np.ndarray, sw: np.ndarray, sigma: float):
    """Best ``(a, u, v)`` for fixed ``(m, sigma)``; ``u, v = b(1±rho)``.

    Returns ``(x, sse)`` under ``a >= 0`` and ``0 <= u, v <= 2``.
    """
    s = np.sqrt(y * y + sigma * sigma)
    A = np.column_stack([np.ones_like(y), 0.5 * (s + y), 0.5 * (s - y)])
    Aw = A * sw[:, None]
    bw = w * sw
    return _box_lsq3(Aw.T @ Aw, Aw.T @ bw, float(bw @ bw))


def fit_svi_quasi_explicit(
    k: np.ndarray,
    w: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> dict[str, float]:
    """Fit raw SVI to total variance *w* at log-moneyness *k*.

    Outer search over ``(m, sigma)`` — a coarse grid, then Nelder-Mead from
    the two best grid points — with the remaining three parameters solved
    exactly at each step.  The result always satisfies ``a >= 0``,
    ``b >= 0``, ``|rho| <= 1`` and wing slopes ``<= 2``.
    """
    k = np.asarray(k, dtype=float)
    w = np.asarray(w, dtype=float)
    sw = np.ones_like(w) if weights is None else np.sqrt(np.asarray(weights, dtype=float))

    k_lo, k_hi = float(k.min()), float(k.max())
    width = max(k_hi - k_lo, 0.05)
    m_bounds = (k_lo - 0.5 * width, k_hi + 0.5 * width)
    s_bounds = (1e-3, max(2.0 * width, 0.05))

    def obj(p):
        m, sig = p
        if not (m_bounds[0] <= m <= m_bounds[1] and s_bounds[0] <= sig <= s_bounds[1]):
            return 1e30
        return _svi_inner(k - m, w, sw, sig)[1]

    grid = [(m, s)
            for m in np.linspace(k_lo, k_hi, 7)
            for s in np.geomspace(s_bounds[0] * 10, s_bounds[1], 7)]
    scored = sorted(grid, key=obj)
    best_f0 = obj(scored[0])  # sets the scale for the convergence test

    best_p, best_f = None, np.inf
    for start in scored[:2]:
        res = minimize(obj, start, method="Nelder-Mead",
                       options={"xatol": 1e-5, "fatol": 1e-12 * max(best_f0, 1e-30),
                                "maxfev": 300})
        if res.fun < best_f:
            best_p, best_f = res.x, res.fun

    m, sig = best_p
    (a, u, v), _ = _svi_inner(k - m, w, sw, sig)
    b = 0.5 * (u + v)
    rho = 0.0 if b <= 0 else (u - v) / (u + v)
    return dict(a=float(a), b=float(b), rho=float(rho), m=float(m), sigma=float(sig))


def fit_slice(
    q: pd.DataFrame,
    T: float,
    forward: float,
    discount: float,
    forward_source: str,
    *,
    min_quotes: int = 6,
) -> Optional[SliceFit]:
    """Fit one expiry's OTM quotes; ``None`` if there are too few."""
    if len(q) < min_quotes:
        return None
    q = q.sort_values("k")
    k = q["k"].to_numpy()
    iv = q["iv"].to_numpy()
    spread = np.nan_to_num(q["spread_iv"].to_numpy(), nan=2 * MIN_SPREAD_IV)
    spread = np.maximum(spread, MIN_SPREAD_IV)

    w = iv ** 2 * T
    # A total-variance residual dw is an IV error of dw / (2·iv·T); dividing
    # by the spread in vol terms turns that into "spreads of mispricing".
    weights = 1.0 / (2.0 * iv * T * spread) ** 2

    keep = np.ones(len(q), dtype=bool)
    p = fit_svi_quasi_explicit(k, w, weights)

    # One round of outlier rejection.  A quote is never an outlier while the
    # fit is within its own bid-ask spread.
    resid = np.sqrt(np.maximum(svi_total_variance(k, **p), 0) / T) - iv
    scale = 1.4826 * np.median(np.abs(resid - np.median(resid)))
    bad = np.abs(resid) > np.maximum.reduce([np.full_like(resid, 4 * scale), spread,
                                             np.full_like(resid, 0.01)])
    if bad.any() and (~bad).sum() >= min_quotes:
        keep = ~bad
        p = fit_svi_quasi_explicit(k[keep], w[keep], weights[keep])
        resid = np.sqrt(np.maximum(svi_total_variance(k, **p), 0) / T) - iv

    exp = None
    if "expiration" in q.columns and q["expiration"].notna().any():
        exp = pd.Timestamp(q["expiration"].iloc[0]).strftime("%Y-%m-%d")

    return SliceFit(
        T=T, forward=forward, discount=discount, forward_source=forward_source,
        n_quotes=int(keep.sum()), n_outliers=int((~keep).sum()),
        k_min=float(k[keep].min()), k_max=float(k[keep].max()),
        rmse_iv=float(np.sqrt(np.mean(resid[keep] ** 2))),
        expiration=exp, **p,
    )


def fit_slices(
    chain: pd.DataFrame,
    spot: float,
    rate: float | RateFn = 0.05,
    *,
    min_T: float = 2 / 365,
    max_rel_spread: float = 0.6,
    min_quotes: int = 6,
) -> list[SliceFit]:
    """Fit an SVI smile to every expiry in *chain*, sorted by ``T``."""
    rate_fn = rate if callable(rate) else _constant_rate(float(rate))
    quotes, forwards = otm_quotes(chain, spot, rate_fn, min_T=min_T,
                                  max_rel_spread=max_rel_spread)
    fits: list[SliceFit] = []
    for T, g in quotes.groupby("T"):
        F, D, source = forwards[float(T)]
        try:
            fit = fit_slice(g, float(T), F, D, source, min_quotes=min_quotes)
        except Exception as exc:  # one bad expiry must not sink the surface
            logger.warning("SVI fit failed for T=%.4f: %s", T, exc)
            continue
        if fit is not None:
            fits.append(fit)
    return sorted(fits, key=lambda s: s.T)


# ────────────────────────────────────────────────────────────────────────────
# Slices → standard grid
# ────────────────────────────────────────────────────────────────────────────


def slices_to_grid(
    slices: list[SliceFit],
    k_grid: np.ndarray,
    t_grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample fitted slices onto the ``(k, T)`` grid.

    Total variance is linear in ``T`` between the two bracketing expiries at
    fixed forward log-moneyness.  Outside the expiry range implied vol is held
    flat (total variance scales with ``T``).  Outside each expiry's quoted
    strikes, see :meth:`SliceFit.total_variance_extrapolated`.

    Returns ``(iv_grid, observed)``, both shaped ``(n_k, n_t)``; *observed*
    is True where the cell lies inside the quoted strike range of both
    bracketing expiries.
    """
    surf = ExpirySurface(slices)
    k_grid = np.asarray(k_grid, dtype=float)
    iv = np.empty((len(k_grid), len(t_grid)))
    observed = np.zeros_like(iv, dtype=bool)
    for j, T in enumerate(t_grid):
        iv[:, j] = surf.iv(k_grid, T)
        observed[:, j] = surf.is_observed(k_grid, T)
    return iv, observed


# ────────────────────────────────────────────────────────────────────────────
# Continuous surface over the fitted expiries
# ────────────────────────────────────────────────────────────────────────────


class ExpirySurface:
    """A continuous ``(k, T)`` surface built directly on the per-expiry fits.

    Same interpolation as :func:`slices_to_grid` — total variance linear in
    ``T`` between bracketing expiries, flat vol outside the expiry range,
    tangent-line wings beyond quoted strikes — but queryable at any tenor and
    strike.  Features use this rather than the stored 25x8 grid, which cannot
    resolve short tenors (it starts at 1M) or exact delta strikes.

    *slices* may be :class:`SliceFit` objects or the dicts stored in a saved
    surface's metadata (``VolSurface.slices``).
    """

    def __init__(self, slices):
        fits = [s if isinstance(s, SliceFit) else SliceFit(**s) for s in slices]
        if not fits:
            raise ValueError("no fitted slices")
        self.slices: list[SliceFit] = sorted(fits, key=lambda s: s.T)
        self.Ts = np.array([s.T for s in self.slices])

    @property
    def t_min(self) -> float:
        return float(self.Ts[0])

    @property
    def t_max(self) -> float:
        return float(self.Ts[-1])

    def brackets(self, T: float) -> bool:
        """True if *T* lies within the fitted expiry range (no T extrapolation)."""
        return bool(self.Ts[0] - 1e-12 <= T <= self.Ts[-1] + 1e-12)

    def _bracket(self, T: float) -> tuple[int, int, float]:
        """Indices of the bracketing slices and the weight on the later one.

        Outside the expiry range both indices are the nearest edge slice.
        """
        if T <= self.Ts[0]:
            return 0, 0, 0.0
        if T >= self.Ts[-1]:
            n = len(self.Ts) - 1
            return n, n, 0.0
        j = int(np.searchsorted(self.Ts, T))
        i = j - 1
        return i, j, float((T - self.Ts[i]) / (self.Ts[j] - self.Ts[i]))

    def total_variance(self, k, T: float) -> np.ndarray:
        """Total implied variance at log-moneyness *k* and tenor *T*."""
        k = np.asarray(k, dtype=float)
        i, j, x = self._bracket(T)
        if i == j:  # outside the expiry range: hold implied vol flat
            w = self.slices[i].total_variance_extrapolated(k) * T / self.Ts[i]
        else:
            w = ((1 - x) * self.slices[i].total_variance_extrapolated(k)
                 + x * self.slices[j].total_variance_extrapolated(k))
        return np.maximum(w, 1e-10)

    def iv(self, k, T: float) -> np.ndarray:
        """Implied vol at log-moneyness *k* and tenor *T*."""
        return np.sqrt(self.total_variance(k, T) / T)

    def quoted_range(self, T: float) -> tuple[float, float]:
        """k-range quoted by *both* bracketing expiries."""
        i, j, _ = self._bracket(T)
        a, b = self.slices[i], self.slices[j]
        return max(a.k_min, b.k_min), min(a.k_max, b.k_max)

    def is_observed(self, k, T: float) -> np.ndarray:
        """True where ``(k, T)`` is inside quoted strikes of bracketing expiries.

        Outside the expiry range only a tenor that coincides with the edge
        expiry counts as observed.
        """
        k = np.asarray(k, dtype=float)
        i, j, _ = self._bracket(T)
        in_t = i != j or bool(np.isclose(T, self.Ts[i]))
        lo, hi = self.quoted_range(T)
        return in_t & (k >= lo) & (k <= hi)

    def delta_strike(self, delta: float, T: float) -> float:
        """Log-moneyness with forward Black delta *delta* at tenor *T*.

        ``delta > 0`` is a call delta (``0.25`` = 25-delta call), ``delta < 0``
        a put delta.  Uses the smile's own vol at the strike, solved by
        bracketing — call delta ``N(d1)`` is monotone in k on an
        arbitrage-free smile, so this cannot oscillate the way a fixed-point
        iteration on sigma can on a steep skew.
        """
        from scipy.optimize import brentq

        target = delta if delta > 0 else 1.0 + delta   # put Δ = call Δ - 1

        def f(k):
            w = float(self.total_variance(k, T))
            sw = math.sqrt(w)
            return float(ndtr(-k / sw + 0.5 * sw)) - target

        lo, hi = -4.0, 3.0
        if f(lo) * f(hi) > 0:
            return float("nan")
        return float(brentq(f, lo, hi, xtol=1e-8))

    def variance_swap_vol(self, T: float) -> float:
        """Model-free (VIX-style) vol at tenor *T*; NaN unless bracketed.

        Each expiry's fair variance is integrated from its smile, then
        interpolated linearly in total variance — CBOE's VIX construction.
        """
        if not self.brackets(T) or len(self.slices) < 2:
            return float("nan")
        i, j, x = self._bracket(T)
        wi = self.slices[i].variance_swap_total_variance()
        wj = self.slices[j].variance_swap_total_variance()
        return float(math.sqrt(((1 - x) * wi + x * wj) / T))
