"""The structured snapshot an interpretation is written from.

Everything the model may say must come from here: the ticker's surface and
realised-vol features on the as-of date, how they rank against the ticker's
own history, the volatility forecast and its track record, data-quality
fields, and peer tickers for context.  Vols are in percent and spreads in
vol points, labelled as such, so the model never has to guess units.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Optional

import numpy as np
import pandas as pd


def _num(x, scale: float = 1.0, nd: int = 2) -> Optional[float]:
    """A JSON-safe rounded number, or None for missing values."""
    if x is None:
        return None
    try:
        v = float(x) * scale
    except (TypeError, ValueError):
        return None
    return None if not math.isfinite(v) else round(v, nd)


def _pct(row, col):
    """A trailing percentile rank as 0–100."""
    return _num(row.get(col), 100, 0)


def build_snapshot(features: pd.DataFrame, forecasts: pd.DataFrame, ticker: str,
                   as_of, track_record_days: int = 182) -> dict[str, Any]:
    """Assemble the interpretation input for (*ticker*, *as_of*)."""
    from src.dashboard.data import forecast_history, latest_forecasts, overview, track_record

    as_of = pd.Timestamp(as_of)
    hist = features[(features["ticker"] == ticker) & (features["date"] <= as_of)].sort_values("date")
    if hist.empty:
        raise ValueError(f"no features for {ticker} on or before {as_of.date()}")
    r = hist.iloc[-1].to_dict()
    V, P = 100.0, 100.0  # vol → percent, spread → vol points

    snap: dict[str, Any] = {
        "ticker": ticker,
        "as_of": str(r["date"].date()),
        "history_available_days": int(len(hist)),
        "history_note": (
            "Percentiles rank today within the trailing 252 trading days, or within all "
            f"available history when shorter ({len(hist)} days here)."
        ),
        "spot": _num(r.get("spot")),
        "implied_vol_pct": {
            "atm_7d": _num(r.get("atm_7d"), V), "atm_30d": _num(r.get("atm_30d"), V),
            "atm_91d": _num(r.get("atm_91d"), V), "atm_365d": _num(r.get("atm_365d"), V),
            "variance_swap_30d": _num(r.get("vs_30d"), V),
        },
        "implied_vol_changes_pts": {
            "atm_30d_1d": _num(r.get("atm_30d_d1"), P), "atm_30d_5d": _num(r.get("atm_30d_d5"), P),
            "variance_swap_30d_1d": _num(r.get("vs_30d_d1"), P),
        },
        "smile_30d": {
            "iv_10d_put_pct": _num(r.get("iv_p10_30d"), V), "iv_25d_put_pct": _num(r.get("iv_p25_30d"), V),
            "iv_25d_call_pct": _num(r.get("iv_c25_30d"), V), "iv_10d_call_pct": _num(r.get("iv_c10_30d"), V),
            "risk_reversal_25d_pts": _num(r.get("rr25_30d"), P),
            "butterfly_25d_pts": _num(r.get("bf25_30d"), P),
            "risk_reversal_25d_5d_change_pts": _num(r.get("rr25_30d_d5"), P),
        },
        "smile_91d": {"risk_reversal_25d_pts": _num(r.get("rr25_91d"), P),
                      "butterfly_25d_pts": _num(r.get("bf25_91d"), P)},
        "term_structure_pts": {
            "atm_30d_minus_7d": _num(r.get("ts_7_30"), P),
            "atm_91d_minus_30d": _num(r.get("ts_30_91"), P),
            "atm_365d_minus_30d": _num(r.get("ts_30_365"), P),
        },
        "percentiles_0_to_100": {
            "atm_30d": _pct(r, "atm_30d_pct252"), "variance_swap_30d": _pct(r, "vs_30d_pct252"),
            "risk_reversal_25d_30d": _pct(r, "rr25_30d_pct252"),
            "butterfly_25d_30d": _pct(r, "bf25_30d_pct252"),
            "term_91d_minus_30d": _pct(r, "ts_30_91_pct252"),
            "variance_risk_premium_30d": _pct(r, "vrp_30d_pct252"),
        },
        "realised": {
            "vol_21d_close_to_close_pct": _num(r.get("rv_cc_21d"), V),
            "vol_21d_yang_zhang_pct": _num(r.get("rv_yz_21d"), V),
            "vol_63d_pct": _num(r.get("rv_cc_63d"), V),
            "return_5d_pct": _num(r.get("ret_5d"), 100),
            "return_21d_pct": _num(r.get("ret_21d"), 100),
        },
        "variance_risk_premium_30d_pts": _num(r.get("vrp_30d"), P),
        "market": {
            "vix": _num(r.get("mkt_vix")),
            "vix3m_over_vix": _num((r.get("mkt_vix3m") or np.nan) / (r.get("mkt_vix") or np.nan), 1, 3),
            "vvix": _num(r.get("mkt_vvix")),
            "cboe_skew": _num(r.get("mkt_skew")),
            "move_treasury_vol": _num(r.get("mkt_move")),
            "high_yield_spread_pct": _num(r.get("macro_hy_oas")),
            "chicago_fed_nfci": _num(r.get("macro_nfci")),
        },
        "data_quality": {
            "expiries_fitted": _num(r.get("n_expiries"), 1, 0),
            "nearest_expiry_days": _num(r.get("nearest_expiry_days"), 1, 0),
            "median_fit_error_pts": _num(r.get("fit_rmse"), P),
        },
    }

    if pd.notna(r.get("earn_next_date")):
        snap["earnings"] = {
            "next_release_session": str(pd.Timestamp(r["earn_next_date"]).date()),
            "trading_days_until": _num(r.get("earn_days_to"), 1, 0),
            "implied_move_pct": _num(r.get("earn_implied_move"), 100),
            "historical_average_move_pct": _num(r.get("earn_hist_move"), 100),
        }

    # ── Forecast and its track record ────────────────────────────────────
    fc_out: dict[str, Any] = {}
    if forecasts is not None and not forecasts.empty:
        latest = latest_forecasts(forecasts, ticker, as_of)
        for h in (5, 21):
            if h not in latest.index:
                continue
            f = latest.loc[h]
            tr = track_record(forecast_history(
                forecasts, ticker, h, as_of - pd.Timedelta(days=track_record_days), as_of))
            fc_out[f"{h}_trading_days"] = {
                "forecast_vol_pct": _num(f["forecast_vol"], V),
                "range_80pct": [_num(f["lo80_vol"], V), _num(f["hi80_vol"], V)],
                "implied_vol_used_pct": _num(f["implied_vol"], V),
                "implied_input": str(f["implied_input"]),
                "forecast_date": str(pd.Timestamp(f["date"]).date()),
                "track_record_last_6m": {
                    "n_scored": tr.get("n", 0),
                    "rmse_forecast_pts": _num(tr.get("rmse_forecast")),
                    "rmse_implied_pts": _num(tr.get("rmse_implied")),
                    "share_inside_80pct_range": _num(tr.get("coverage80"), 1, 2),
                },
            }
    snap["volatility_forecast"] = fc_out or None

    # ── Peers on the same date ───────────────────────────────────────────
    peers = overview(features, as_of)
    snap["peers"] = [
        {"ticker": p["ticker"], "atm_30d_pct": _num(p.get("atm_30d"), V),
         "atm_30d_percentile": _pct(p, "atm_30d_pct252"),
         "risk_reversal_25d_30d_pts": _num(p.get("rr25_30d"), P),
         "variance_risk_premium_30d_pts": _num(p.get("vrp_30d"), P)}
        for _, p in peers.iterrows() if p["ticker"] != ticker
    ]
    return snap


def snapshot_hash(snapshot: dict, *extra: str) -> str:
    """Stable digest of a snapshot (plus prompt/model identifiers) for caching."""
    payload = json.dumps(snapshot, sort_keys=True, separators=(",", ":")) + "|" + "|".join(extra)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]
