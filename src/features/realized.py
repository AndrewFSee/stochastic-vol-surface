"""Realised-volatility and return features from daily underlying prices.

All windows are trailing and include the current day, so a value dated *t*
uses only prices up to the close of *t*.  Vols are annualised decimals
(0.15 = 15%), the same units as implied vol on the surfaces.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252
RV_WINDOWS = (10, 21, 63)


def close_to_close_vol(adj_close: pd.Series, window: int) -> pd.Series:
    """Zero-mean close-to-close realised vol, ``sqrt(252 · mean(r²))``.

    Zero-mean (rather than a sample standard deviation) is the variance-swap
    convention, which keeps it directly comparable with implied variance.
    """
    r = np.log(adj_close).diff()
    return np.sqrt(TRADING_DAYS * (r ** 2).rolling(window, min_periods=window).mean())


def yang_zhang_vol(df: pd.DataFrame, window: int) -> pd.Series:
    """Yang-Zhang (2000) OHLC realised vol: overnight + open-to-close + Rogers-Satchell.

    Several times more efficient than close-to-close on the same window, so
    a 21-day estimate is much less noisy.  Uses as-traded OHLC; the small
    overnight drop on ex-dividend days is not adjusted out.
    """
    o, h, l, c = (np.log(df[col]) for col in ("open", "high", "low", "close"))
    overnight = o - c.shift(1)
    open_close = c - o
    rs = (h - c) * (h - o) + (l - c) * (l - o)

    n = window
    k = 0.34 / (1.34 + (n + 1) / (n - 1))
    var_o = overnight.rolling(n, min_periods=n).var()
    var_c = open_close.rolling(n, min_periods=n).var()
    var_rs = rs.rolling(n, min_periods=n).mean()
    return np.sqrt(TRADING_DAYS * (var_o + k * var_c + (1 - k) * var_rs))


def realized_features(prices: pd.DataFrame) -> pd.DataFrame:
    """Return and realised-vol features for one ticker.

    *prices* is indexed by date with ``open, high, low, close, adj_close``.
    """
    p = prices.sort_index().copy()
    # Yahoo's newest bar often has no adjusted close yet.  Adjustments are
    # rebased so recent bars have factor 1, so carrying the last known
    # adj/close factor forward is exact unless today is an ex-dividend date.
    factor = (p["adj_close"] / p["close"]).ffill().fillna(1.0)
    p["adj_close"] = p["adj_close"].fillna(p["close"] * factor)
    logp = np.log(p["adj_close"])
    out = pd.DataFrame(index=p.index)
    out["ret_1d"] = logp.diff(1)
    out["ret_5d"] = logp.diff(5)
    out["ret_21d"] = logp.diff(21)
    for w in RV_WINDOWS:
        out[f"rv_cc_{w}d"] = close_to_close_vol(p["adj_close"], w)
    out["rv_yz_21d"] = yang_zhang_vol(p, 21)
    return out
