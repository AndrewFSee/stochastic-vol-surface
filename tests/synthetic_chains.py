"""Internally consistent synthetic option chains for surface tests.

Prices are Black-76 on a dividend-adjusted forward, so the construction
pipeline has to recover the forward from put-call parity and the vols from
the prices — exactly as it must on real data.  A chain whose prices do not
match its quoted IVs can only test that the pipeline reads a column.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from src.surface.implied_vol import black_price


def smile_iv(k: np.ndarray, T: float, atm: float = 0.20) -> np.ndarray:
    """A realistic equity smile: put skew that flattens with maturity."""
    k = np.asarray(k, dtype=float)
    skew = -0.25 / np.sqrt(max(T, 1 / 52))
    return atm + 0.01 * T + skew * k * 0.1 + 0.6 * k ** 2 / np.sqrt(max(T, 1 / 52)) * 0.1


def make_chain(
    *,
    ticker: str = "TEST",
    as_of: date = date(2026, 1, 6),
    spot: float = 100.0,
    r: float = 0.04,
    q: float = 0.02,
    tenors: tuple[float, ...] = (0.0833, 0.25, 0.5, 1.0, 2.0),
    k_range: tuple[float, float] = (-0.35, 0.18),
    n_strikes: int = 22,
    rel_spread: float = 0.04,
    atm: float = 0.20,
) -> pd.DataFrame:
    """Calls and puts at every strike, priced off ``F = S·e^{(r-q)T}``."""
    rows = []
    for T in tenors:
        expiry = pd.Timestamp(as_of) + pd.Timedelta(days=int(round(T * 365)))
        T_eff = (expiry - pd.Timestamp(as_of)).days / 365.0
        F = spot * np.exp((r - q) * T_eff)
        D = np.exp(-r * T_eff)
        for k in np.linspace(*k_range, n_strikes):
            K = float(F * np.exp(k))
            iv = float(smile_iv(np.log(K / F), T_eff, atm))
            for opt in ("call", "put"):
                mid = float(black_price(F, K, T_eff, iv, D, opt == "call"))
                half = max(0.5 * rel_spread * mid, 0.005)
                rows.append({
                    "ticker": ticker,
                    "as_of": pd.Timestamp(as_of),
                    "expiration": expiry,
                    "strike": K,
                    "option_type": opt,
                    "bid": max(mid - half, 0.0),
                    "ask": mid + half,
                    "mid": mid,
                    "last_price": mid,
                    "volume": 100.0,
                    "open_interest": 500.0,
                    "implied_volatility_market": iv,
                    "T": T_eff,
                    "underlying_price": spot,
                })
    return pd.DataFrame(rows)


def gbm_prices(sigma=0.2, n=600, seed=0, start="2025-01-02"):
    """Daily OHLC from a GBM sampled intraday, so OHLC estimators have signal."""
    rng = np.random.default_rng(seed)
    steps = 390  # one per minute; coarser sampling understates high-low ranges
    dt = 1 / (252 * steps)
    log_p = np.cumsum(rng.normal(-0.5 * sigma ** 2 * dt, sigma * np.sqrt(dt), n * steps))
    path = 100 * np.exp(log_p).reshape(n, steps)
    idx = pd.bdate_range(start, periods=n, name="date")
    return pd.DataFrame({
        "open": path[:, 0], "high": path.max(axis=1), "low": path.min(axis=1),
        "close": path[:, -1], "adj_close": path[:, -1], "volume": 1e6,
    }, index=idx)
