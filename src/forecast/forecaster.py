"""Production volatility forecasts: HAR + implied vol, pooled across tickers.

Chosen by walk-forward evaluation on SPY 2013–2023 (``scripts/evaluate_forecasts.py``):
HAR + implied was best or tied-best at both horizons and significantly better
than HAR at 5 days; GARCH, gradient boosting, LSTM and LSTM-GARCH all did
worse.  Extra surface features (skew, term structure, VVIX...) added nothing
significant beyond the implied level, so the production model leaves them out.

* Inputs: HAR terms plus each ticker's *own* implied vol — 7-day ATM for the
  5-day horizon (14-day ATM where no 7-day expiry is listed, flagged), the
  30-day variance swap for 21 days.
* Coefficients are pooled across tickers.  SPY's 2010–2023 history dominates
  the pool; fitting a ticker on its own ~7 months of live data was much worse
  out of sample.
* Built walk-forward with monthly refits, so every stored forecast — including
  the historical ones shown as a track record — is out-of-sample and uses only
  information available on its date.

Output: one row per (ticker, date, horizon) with the forecast vol, an 80%
prediction interval, the implied vol it used, and the realised vol once the
horizon has passed.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from src.forecast.dataset import DEFAULT_FEATURE_PATHS, build_dataset
from src.forecast.models import HAR_TERMS, LogLinear

logger = logging.getLogger(__name__)

FORECAST_VERSION = "forecast/1"
MODEL_NAME = "HAR + implied (pooled)"
DEFAULT_FORECASTS_PATH = "data/forecasts/vol_forecasts.parquet"
HORIZONS = (5, 21)

#: Implied-vol input per horizon: (primary, fallback when the primary is missing).
IMPLIED_INPUT = {5: ("atm_7d", "atm_14d"), 21: ("vs_30d", "atm_30d")}


def production_model() -> LogLinear:
    return LogLinear(MODEL_NAME, {**HAR_TERMS, "log_iv": lambda d, h: np.log(d["iv_in"] ** 2)})


def _with_implied(ds: pd.DataFrame, h: int) -> pd.DataFrame:
    primary, fallback = IMPLIED_INPUT[h]
    d = ds.copy()
    p = d[primary] if primary in d else pd.Series(np.nan, index=d.index)
    f = d[fallback] if fallback in d else pd.Series(np.nan, index=d.index)
    d["iv_in"] = p.fillna(f)
    d["iv_src"] = np.where(p.notna(), primary, np.where(f.notna(), fallback, None))
    return d


def _known(ds: pd.DataFrame, as_of: pd.Timestamp, h: int) -> pd.DataFrame:
    """Rows whose h-day target had closed by *as_of* (no look-ahead)."""
    pos = ds.index.searchsorted(as_of, side="right") - 1
    rows = ds.iloc[: max(pos - h + 1, 0)]
    return rows[rows[f"y_{h}"].notna()]


def build_forecasts(
    tickers: Optional[Sequence[str]] = None,
    *,
    start: str = "2013-01-01",
    underlying_dir: str = "data/underlying",
    feature_paths: Sequence[str] = DEFAULT_FEATURE_PATHS,
    horizons: Sequence[int] = HORIZONS,
) -> pd.DataFrame:
    """Walk-forward forecasts for every ticker and horizon (see module docstring)."""
    if tickers is None:
        tickers = sorted({t for p in feature_paths if Path(p).exists()
                          for t in pd.read_parquet(p, columns=["ticker"])["ticker"].unique()})
    raw = {}
    for t in tickers:
        try:
            raw[t] = build_dataset(t, underlying_dir=underlying_dir, feature_paths=feature_paths,
                                   horizons=horizons)
        except ValueError as exc:
            logger.warning("Skipping %s: %s", t, exc)
    if not raw:
        return pd.DataFrame()

    out = []
    last = max(ds.index.max() for ds in raw.values())
    months = pd.date_range(pd.Timestamp(start), last, freq="MS")
    for h in horizons:
        panel = {t: _with_implied(ds, h) for t, ds in raw.items()}
        for m0, m1 in zip(months, list(months[1:]) + [last + pd.Timedelta(days=1)]):
            train = pd.concat([_known(ds, m0 - pd.Timedelta(days=1), h) for ds in panel.values()])
            ok = train["iv_in"].notna()
            if ok.sum() < 250:
                continue
            model = production_model().fit(train[ok], h)
            for t, ds in panel.items():
                rows = ds[(ds.index >= m0) & (ds.index < m1)]
                rows = rows[rows["iv_in"].notna() & rows["rv_m"].notna()]
                if rows.empty:
                    continue
                var = model.predict(rows, h)
                lo, hi = model.interval(rows, h)
                out.append(pd.DataFrame({
                    "ticker": t, "date": rows.index, "horizon": h,
                    "forecast_vol": np.sqrt(var.to_numpy()),
                    "lo80_vol": np.sqrt(lo.to_numpy()), "hi80_vol": np.sqrt(hi.to_numpy()),
                    "implied_vol": rows["iv_in"].to_numpy(),
                    "implied_input": rows["iv_src"].to_numpy(),
                    "trailing_vol": np.sqrt(rows["rv_m"].to_numpy()),
                    "realised_vol": np.sqrt(rows[f"y_{h}"].to_numpy()),
                    "refit_date": m0, "n_train": int(ok.sum()),
                }))
    if not out:
        return pd.DataFrame()
    df = pd.concat(out, ignore_index=True)
    df["model"] = MODEL_NAME
    df["forecast_version"] = FORECAST_VERSION
    return df.sort_values(["ticker", "horizon", "date"]).reset_index(drop=True)


def save_forecasts(df: pd.DataFrame, path: str = DEFAULT_FORECASTS_PATH) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False, engine="pyarrow")
    logger.info("Saved %d forecasts -> %s", len(df), out)
    return out


def load_forecasts(path: str = DEFAULT_FORECASTS_PATH) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        logger.warning("No forecasts at %s — run scripts/build_forecasts.py", p)
        return pd.DataFrame()
    return pd.read_parquet(p)
