"""Walk-forward evaluation of volatility forecasts.

No look-ahead: at each refit date *t₀* a model trains only on rows whose
*h*-day target window had closed by *t₀* (row *r* with ``r + h ≤ t₀``).
With overlapping 21-day targets, training on the most recent rows would
otherwise leak up to 20 days of the future into every fit.

Scoring
-------
* **QLIKE** ``y/f − log(y/f) − 1`` on variance — the loss that ranks
  forecasts correctly when realised variance is a noisy proxy (Patton 2011).
* RMSE, MAE and mean bias of the implied *volatility*, in vol points.
* Out-of-sample R² of log variance.
* Diebold-Mariano test of QLIKE against a baseline, with Newey-West
  (Bartlett, *h* lags) standard errors for the overlapping targets.
"""

from __future__ import annotations

import math
from typing import Callable, Mapping, Optional

import numpy as np
import pandas as pd


def walk_forward(
    ds: pd.DataFrame,
    factory: Callable[[], object],
    h: int,
    *,
    start: str,
    end: Optional[str] = None,
    refit_every: int = 21,
) -> pd.Series:
    """Out-of-sample forecasts for every row in ``[start, end]``.

    Refits every *refit_every* rows on the targets known at that point.
    """
    target = f"y_{h}"
    dates = ds.index
    mask = dates >= pd.Timestamp(start)
    if end is not None:
        mask &= dates <= pd.Timestamp(end)
    positions = np.flatnonzero(mask)
    out = pd.Series(np.nan, index=ds.index[positions], dtype=float)

    for b in range(0, len(positions), refit_every):
        block = positions[b:b + refit_every]
        p0 = block[0]
        train = ds.iloc[: max(p0 - h + 1, 0)]
        train = train[train[target].notna()]
        if len(train) < 250:
            continue
        model = factory().fit(train, h)
        rows = ds.iloc[block]
        out.loc[rows.index] = model.predict(rows, h, ds.iloc[: block[-1] + 1]).to_numpy()
    return out


def scores(y: pd.Series, f: pd.Series) -> dict[str, float]:
    """Forecast accuracy of variance forecasts *f* against realised *y*."""
    ok = np.isfinite(y) & np.isfinite(f) & (y > 0) & (f > 0)
    y, f = y[ok], f[ok]
    r = y / f
    vy, vf = np.sqrt(y), np.sqrt(f)
    ly, lf = np.log(y), np.log(f)
    return {
        "n": int(ok.sum()),
        "qlike": float(np.mean(r - np.log(r) - 1)),
        "rmse_vol": float(100 * np.sqrt(np.mean((vf - vy) ** 2))),
        "mae_vol": float(100 * np.mean(np.abs(vf - vy))),
        "bias_vol": float(100 * np.mean(vf - vy)),
        "r2_log": float(1 - np.sum((ly - lf) ** 2) / np.sum((ly - ly.mean()) ** 2)),
    }


def qlike_series(y: pd.Series, f: pd.Series) -> pd.Series:
    r = y / f
    return r - np.log(r) - 1


def diebold_mariano(loss_base: pd.Series, loss_model: pd.Series, lag: int) -> tuple[float, float]:
    """DM statistic and two-sided p-value for ``mean(loss_base − loss_model)``.

    Positive means the model beats the baseline.  Newey-West variance with a
    Bartlett kernel over *lag* autocovariances.
    """
    d = (loss_base - loss_model).dropna().to_numpy()
    n = len(d)
    if n < 10:
        return float("nan"), float("nan")
    dc = d - d.mean()
    var = dc @ dc / n
    for k in range(1, lag + 1):
        var += 2 * (1 - k / (lag + 1)) * (dc[k:] @ dc[:-k]) / n
    stat = d.mean() / math.sqrt(max(var, 1e-300) / n)
    p = math.erfc(abs(stat) / math.sqrt(2))
    return float(stat), float(p)


def compare(
    ds: pd.DataFrame,
    factories: Mapping[str, Callable[[], object]],
    h: int,
    *,
    start: str,
    end: Optional[str] = None,
    refit_every: int = 21,
    baseline: str = "HAR",
    progress: Optional[Callable[[str], None]] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run every model walk-forward; score them on their common dates.

    Returns ``(forecasts, report)``: forecasts has the target column ``y`` and
    one column per model; report has one row per model with :func:`scores`
    and the Diebold-Mariano test against *baseline*.
    """
    fc = {}
    for name, factory in factories.items():
        if progress:
            progress(name)
        fc[name] = walk_forward(ds, factory, h, start=start, end=end, refit_every=refit_every)
    forecasts = pd.DataFrame(fc)
    forecasts.insert(0, "y", ds[f"y_{h}"].reindex(forecasts.index))
    common = forecasts.dropna()

    rows = []
    base_loss = qlike_series(common["y"], common[baseline]) if baseline in common else None
    for name in factories:
        row = {"model": name, **scores(common["y"], common[name])}
        if base_loss is not None and name != baseline:
            row["dm_stat"], row["dm_p"] = diebold_mariano(
                base_loss, qlike_series(common["y"], common[name]), lag=h)
        rows.append(row)
    report = pd.DataFrame(rows).set_index("model")
    return forecasts, report


def by_regime(forecasts: pd.DataFrame, regime: pd.Series) -> pd.DataFrame:
    """QLIKE per model within each regime label (e.g. VIX buckets)."""
    common = forecasts.dropna()
    lab = regime.reindex(common.index)
    out = {}
    for g, idx in common.groupby(lab, observed=True).groups.items():
        sub = common.loc[idx]
        out[g] = {m: scores(sub["y"], sub[m])["qlike"] for m in common.columns if m != "y"}
        out[g]["n"] = len(sub)
    return pd.DataFrame(out)


def predictive_regression(x: pd.Series, y: pd.Series, lag: int) -> dict[str, float]:
    """Univariate predictive regression of *y* on standardised *x*.

    Newey-West (Bartlett, *lag*) t-statistic for overlapping targets.
    Returns slope per one standard deviation of *x*, t, p, R² and n.
    """
    d = pd.concat([x, y], axis=1).dropna()
    xv, yv = d.iloc[:, 0].to_numpy(), d.iloc[:, 1].to_numpy()
    xv = (xv - xv.mean()) / xv.std()
    X = np.column_stack([np.ones_like(xv), xv])
    b = np.linalg.lstsq(X, yv, rcond=None)[0]
    e = yv - X @ b
    n = len(yv)
    S = (X * e[:, None]).T @ (X * e[:, None]) / n
    for k in range(1, lag + 1):
        G = (X[k:] * e[k:, None]).T @ (X[:-k] * e[:-k, None]) / n
        S += (1 - k / (lag + 1)) * (G + G.T)
    Q = np.linalg.inv(X.T @ X / n)
    t = b[1] / math.sqrt((Q @ S @ Q / n)[1, 1])
    return {"slope_per_sd": float(b[1]), "t": float(t),
            "p": float(math.erfc(abs(t) / math.sqrt(2))),
            "r2": float(1 - e @ e / np.sum((yv - yv.mean()) ** 2)), "n": n}
