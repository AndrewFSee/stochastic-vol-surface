"""Production volatility forecasts: HAR + implied vol, pooled across tickers.

Chosen by walk-forward evaluation on SPY 2013–2025 (``scripts/evaluate_forecasts.py``):
HAR + implied was best or tied-best at both horizons and significantly better
than HAR at 5 days; GARCH, gradient boosting, LSTM and LSTM-GARCH all did
worse.  Extra surface features (skew, term structure, VVIX...) added nothing
significant beyond the implied level, so the production model leaves them out.

* Inputs: HAR terms plus each ticker's *own* implied vol — 7-day ATM for the
  5-day horizon (14-day ATM where no 7-day expiry is listed, flagged), the
  30-day variance swap for 21 days.
* Coefficients are pooled across tickers.  SPY's 2010–2025 history dominates
  the pool; fitting a ticker on its own ~7 months of live data was much worse
  out of sample.
* Built walk-forward with monthly refits, so every stored forecast — including
  the historical ones shown as a track record — is out-of-sample and uses only
  information available on its date.
* Earnings (single stocks): the model runs on ex-earnings inputs and targets,
  and the event's variance is added back when a release falls in the window
  — the implied earnings move from the surface, else the stock's historical
  average move (:func:`_with_events`).  On 14 stocks over 2013–2026 this cut
  QLIKE by about a quarter (``scripts/evaluate_earnings.py``,
  ``docs/earnings_evaluation.md``).  Funds and indices are unaffected.

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

from src.forecast.dataset import DEFAULT_FEATURE_PATHS, TRADING_DAYS, build_dataset
from src.forecast.models import HAR_TERMS, LogLinear, har

logger = logging.getLogger(__name__)

FORECAST_VERSION = "forecast/2"
MODEL_NAME = "HAR + implied (pooled)"
DEFAULT_FORECASTS_PATH = "data/forecasts/vol_forecasts.parquet"
HORIZONS = (5, 21)

#: Implied-vol input per horizon: (primary, fallback when the primary is missing).
IMPLIED_INPUT = {5: ("atm_7d", "atm_14d"), 21: ("vs_30d", "atm_30d")}
#: Calendar days each implied input covers, to strip an earnings event inside it.
TENOR_DAYS = {"atm_7d": 7, "atm_14d": 14, "vs_30d": 30, "atm_30d": 30}
EVENT_MODEL_NAME = "HAR + implied, earnings-adjusted (pooled)"


def production_model() -> LogLinear:
    return LogLinear(MODEL_NAME, {**HAR_TERMS, "log_iv": lambda d, h: np.log(d["iv_in"] ** 2)})


def _log_col(col: str):
    return lambda d, h: np.log(d[col])


def event_model(use_implied: bool = True) -> LogLinear:
    """HAR (+ implied) on ex-earnings inputs, fitted to ex-earnings variance.

    The earnings jump is added back separately (:func:`_with_events`).
    """
    terms = {f"log_rv_{k}": _log_col(f"rv_{k}_ex") for k in ("d", "w", "m")}
    if use_implied:
        terms["log_iv"] = lambda d, h: np.log(d["iv_ex"] ** 2)
    return LogLinear(EVENT_MODEL_NAME if use_implied else "HAR, earnings-adjusted", terms)


def event_variance(d: pd.DataFrame) -> pd.Series:
    """Event-day jump variance e²: the implied earnings move where the surface
    gives one, else the historical average excess move."""
    nan = pd.Series(np.nan, index=d.index)
    imp = d["earn_implied_move"] ** 2 if "earn_implied_move" in d else nan
    hist = d["ev_hist_var"] if "ev_hist_var" in d else nan
    return imp.fillna(hist)


def _with_events(ds: pd.DataFrame, h: int) -> pd.DataFrame:
    """Strip earnings from the implied input and the target; price it separately.

    * ``iv_ex``: implied vol without the event, ``iv² − e²·365/tenor`` when
      the next event falls inside the implied input's tenor (floored at half
      the raw vol);
    * ``ev_add``: ``252/h · e²`` when an event falls in ``(t, t+h]``;
    * ``y_<h>`` becomes the ex-event target; the true one moves to ``y_true``.
    """
    d = ds.copy()
    e2 = event_variance(d).fillna(0.0)
    if "iv_in" in d:
        tenor = d["iv_src"].map(TENOR_DAYS).astype(float)
        days = (pd.to_datetime(d["ev_next"]) - d.index.to_series()).dt.days
        inside = (days <= tenor).fillna(False).to_numpy(dtype=bool)
        iv2 = d["iv_in"] ** 2
        strip = np.where(inside, e2 * 365.0 / tenor, 0.0)
        d["iv_ex"] = np.sqrt(np.maximum(iv2 - strip, 0.25 * iv2))
    in_window = (d[f"ev_in_{h}"].astype(bool) if f"ev_in_{h}" in d
                 else pd.Series(False, index=d.index)).to_numpy()
    d["ev_add"] = np.where(in_window, TRADING_DAYS / h * e2, 0.0)
    d["ev_move"] = np.where(in_window, np.sqrt(e2), np.nan)
    d["y_true"] = d[f"y_{h}"]
    d[f"y_{h}"] = d[f"y_{h}_ex"]
    return d


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
    event_adjusted: bool = True,
    use_implied: bool = True,
    events_dir: str = "data/events",
) -> pd.DataFrame:
    """Walk-forward forecasts for every ticker and horizon (see module docstring).

    *event_adjusted* (the production setting) forecasts single stocks'
    earnings separately (see :func:`_with_events`); False gives the earlier
    model.  *use_implied* False drops the implied input (plain HAR), for
    backtests where no surfaces exist.
    """
    if tickers is None:
        tickers = sorted({t for p in feature_paths if Path(p).exists()
                          for t in pd.read_parquet(p, columns=["ticker"])["ticker"].unique()})
    raw = {}
    for t in tickers:
        try:
            raw[t] = build_dataset(t, underlying_dir=underlying_dir, feature_paths=feature_paths,
                                   horizons=horizons, events_dir=events_dir)
        except ValueError as exc:
            logger.warning("Skipping %s: %s", t, exc)
    if not raw:
        return pd.DataFrame()

    out = []
    last = max(ds.index.max() for ds in raw.values())
    months = pd.date_range(pd.Timestamp(start), last, freq="MS")
    rv_col = "rv_m_ex" if event_adjusted else "rv_m"
    name = MODEL_NAME
    for h in horizons:
        panel = {t: _with_implied(ds, h) for t, ds in raw.items()}
        if event_adjusted:
            panel = {t: _with_events(ds, h) for t, ds in panel.items()}
        for m0, m1 in zip(months, list(months[1:]) + [last + pd.Timedelta(days=1)]):
            train = pd.concat([_known(ds, m0 - pd.Timedelta(days=1), h) for ds in panel.values()])
            ok = train["iv_in"].notna() if use_implied else train[rv_col].notna()
            if ok.sum() < 250:
                continue
            if event_adjusted:
                model = event_model(use_implied)
            else:
                model = production_model() if use_implied else har()
            model.fit(train[ok], h)
            name = model.name
            for t, ds in panel.items():
                rows = ds[(ds.index >= m0) & (ds.index < m1)]
                keep = rows[rv_col].notna()
                if use_implied:
                    keep &= rows["iv_in"].notna()
                rows = rows[keep]
                if rows.empty:
                    continue
                add = rows["ev_add"].to_numpy() if event_adjusted else 0.0
                var = model.predict(rows, h).to_numpy() + add
                lo, hi = (x.to_numpy() + add for x in model.interval(rows, h))
                y = rows["y_true"] if event_adjusted else rows[f"y_{h}"]
                frame = pd.DataFrame({
                    "ticker": t, "date": rows.index, "horizon": h,
                    "forecast_vol": np.sqrt(var),
                    "lo80_vol": np.sqrt(lo), "hi80_vol": np.sqrt(hi),
                    "implied_vol": rows["iv_in"].to_numpy(),
                    "implied_input": rows["iv_src"].to_numpy(),
                    "trailing_vol": np.sqrt(rows["rv_m"].to_numpy()),
                    "realised_vol": np.sqrt(y.to_numpy()),
                    "refit_date": m0, "n_train": int(ok.sum()),
                })
                if event_adjusted:
                    frame["earnings_in_window"] = rows["ev_add"].to_numpy() > 0
                    frame["earnings_move"] = rows["ev_move"].to_numpy()
                    frame["earnings_date"] = pd.to_datetime(rows["ev_next"]).where(
                        rows["ev_add"] > 0).to_numpy()
                out.append(frame)
    if not out:
        return pd.DataFrame()
    df = pd.concat(out, ignore_index=True)
    df["model"] = name
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
