"""Earnings-event features: timing, implied move and historical move.

Implied earnings move
---------------------
Around a scheduled release the ATM total variance of an expiry *after* the
event is ``w = σ²·n + e²``: diffusive variance σ² per session over its *n*
remaining sessions, plus the event-day jump variance e², which every later
expiry carries once.  Time is counted in NYSE sessions, not calendar days,
because a weekend between two weekly expiries adds almost no variance.  Two
expiries give two equations:

* one before and one after the event (preferred):
  ``σ² = w₀/n₀``, ``e² = w₁ − σ²·n₁``;
* otherwise the first two after it: ``σ² = (w₂ − w₁)/(n₂ − n₁)``,
  ``e² = w₁ − σ²·n₁``.

``e = √e²`` is the market's standard deviation of the event-day log return
(risk-neutral, so it carries a premium).  It is NaN when the event is more
than ~3 months out, when the surface lacks the expiries, or when the
estimate is not positive (the term structure was too noisy to separate the
event).

Historical earnings move
------------------------
``earn_hist_move``: the root-mean-square event-day close-to-close return
over the last 8 releases (at least 4), in excess of a normal day's variance
(trailing 63 sessions excluding events).  Point-in-time and available on
every date with prices, unlike the implied move.
"""

from __future__ import annotations

import json
from typing import Optional

import numpy as np
import pandas as pd

#: Furthest first post-event expiry (years) for which the implied move is read.
MAX_EVENT_T = 0.30
HIST_EVENTS = 8
MIN_HIST_EVENTS = 4


def atm_term(slices: list[dict]) -> str:
    """JSON ``[[expiration, T, w_atm], ...]`` from per-expiry SVI fits.

    Stored with each cached surface row, so the implied move can be
    recomputed against updated earnings dates without re-reading surfaces.
    """
    from src.surface.slices import svi_total_variance

    term = []
    for s in slices:
        try:
            w = float(svi_total_variance(0.0, s["a"], s["b"], s["rho"], s["m"], s["sigma"]))
        except (KeyError, TypeError):
            continue
        if np.isfinite(w) and w > 0 and s.get("expiration"):
            term.append([str(s["expiration"])[:10], float(s["T"]), w])
    return json.dumps(sorted(term, key=lambda r: r[1]))


def implied_event_variance(term: list, event: pd.Timestamp, as_of: pd.Timestamp,
                           sessions: pd.DatetimeIndex) -> float:
    """Jump variance e² of a release moving session *event*, read from the
    surface of *as_of* (see module docstring).  *sessions* is the NYSE calendar."""
    if not term:
        return float("nan")
    ev = pd.Timestamp(event).strftime("%Y-%m-%d")
    base = sessions.searchsorted(pd.Timestamp(as_of), side="right")

    def n(expiry: str) -> int:
        return int(sessions.searchsorted(pd.Timestamp(expiry), side="right") - base)

    pre = [(n(e), w) for e, t, w in term if e < ev and n(e) >= 1]
    post = [(n(e), w) for e, t, w in term if e >= ev]
    first_t = next((t for e, t, w in term if e >= ev), None)
    if not post or first_t > MAX_EVENT_T:
        return float("nan")
    n1, w1 = post[0]
    if pre:
        n0, w0 = pre[-1]
        e2 = w1 - w0 / n0 * n1
    elif len(post) >= 2 and post[1][0] > n1:
        n2, w2 = post[1]
        e2 = w1 - (w2 - w1) / (n2 - n1) * n1
    else:
        return float("nan")
    return float(e2) if e2 > 0 else float("nan")


def calendar_columns(dates: pd.DatetimeIndex, events: pd.DatetimeIndex,
                     sessions: pd.DatetimeIndex) -> pd.DataFrame:
    """``earn_next_date``, ``earn_days_to`` and ``earn_days_since`` for *dates*.

    ``earn_days_to`` counts sessions in ``(t, next]``: 1 means the release
    moves tomorrow's session.  ``earn_days_since`` is 0 on the event session
    itself.  Both are NaN when there is no event on that side.
    """
    dates = pd.DatetimeIndex(dates)
    out = pd.DataFrame(index=dates)
    if len(events) == 0:
        out["earn_next_date"] = pd.NaT
        out["earn_days_to"] = np.nan
        out["earn_days_since"] = np.nan
        return out
    events = pd.DatetimeIndex(events).sort_values()
    sessions = pd.DatetimeIndex(sessions).sort_values()
    i_next = events.searchsorted(dates, side="right")
    has_next = i_next < len(events)
    nxt = events[np.minimum(i_next, len(events) - 1)].where(has_next)
    i_last = i_next - 1
    has_last = i_last >= 0
    last = events[np.maximum(i_last, 0)].where(has_last)

    pos_t = sessions.searchsorted(dates, side="right")
    pos_n = sessions.searchsorted(nxt.fillna(dates[0]), side="right")
    pos_l = sessions.searchsorted(last.fillna(dates[0]), side="right")
    out["earn_next_date"] = nxt
    out["earn_days_to"] = np.where(has_next, pos_n - pos_t, np.nan)
    out["earn_days_since"] = np.where(has_last, pos_t - pos_l, np.nan)
    return out


def historical_event_variance(close: pd.Series, events: pd.DatetimeIndex) -> pd.Series:
    """Excess event-day variance from past releases, as of each date of *close*.

    ``mean(r²) over the last 8 event sessions − a normal day's r²`` (trailing
    63 non-event sessions), floored at 0; NaN before 4 releases are seen.
    """
    close = close.dropna().sort_index()
    r = np.log(close).diff()
    is_ev = r.index.isin(pd.DatetimeIndex(events))
    normal = (r ** 2).where(~is_ev).rolling(63, min_periods=40).mean()
    ev_sq = (r ** 2)[is_ev].dropna()
    excess = ev_sq - normal.shift(1).reindex(ev_sq.index)
    mean_ex = excess.rolling(HIST_EVENTS, min_periods=MIN_HIST_EVENTS).mean()
    out = mean_ex.reindex(r.index).ffill()
    return out.clip(lower=0.0).rename("earn_hist_var")


def _scheduled_next(dates: pd.DatetimeIndex, snaps: pd.DataFrame,
                    sessions: pd.DatetimeIndex) -> pd.Series:
    """Next release session as scheduled in the latest snapshot on or before
    each date (NaT where no snapshot covers the date)."""
    from src.data.events import event_sessions

    out = pd.Series(pd.NaT, index=dates, dtype="datetime64[ns]")
    if snaps is None or snaps.empty:
        return out
    by_day = {d: event_sessions(g.assign(ticker="_"), sessions).get("_", pd.DatetimeIndex([]))
              for d, g in snaps.groupby("snapshot_date")}
    days = pd.DatetimeIndex(sorted(by_day))
    pos = days.searchsorted(dates, side="right") - 1
    for i, (dt, k) in enumerate(zip(dates, pos)):
        if k < 0:
            continue
        ev = by_day[days[k]]
        nxt = ev[ev > dt]
        if len(nxt):
            out.iloc[i] = nxt[0]
    return out


def earnings_features(
    df: pd.DataFrame,
    earnings: pd.DataFrame,
    prices: Optional[pd.DataFrame] = None,
    snapshots: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Add the earnings columns to a feature frame with ``ticker, date`` and
    (optionally) the ``_atm_term`` JSON of each surface.

    Where a schedule *snapshot* covers a date, the next release is the one
    scheduled then (point-in-time); otherwise the actual date is used.
    Tickers without releases (funds, indices) get NaN everywhere.
    """
    from src.data.events import event_sessions
    from src.features.table import trading_calendar

    cols = ["earn_next_date", "earn_days_to", "earn_days_since",
            "earn_implied_move", "earn_hist_move"]
    out = df.copy()
    for c in cols:
        out[c] = pd.NaT if c == "earn_next_date" else np.nan
    if earnings is None or earnings.empty or out.empty:
        return out

    lo = min(out["date"].min(), pd.Timestamp("2000-01-01"))
    hi = out["date"].max() + pd.Timedelta(days=400)
    sessions = trading_calendar(lo, hi)
    ev = event_sessions(earnings[earnings["ticker"].isin(out["ticker"].unique())], sessions)

    for t, idx in out.groupby("ticker").groups.items():
        if t not in ev or len(ev[t]) == 0:
            continue
        rows = out.loc[idx]
        dates = pd.DatetimeIndex(rows["date"])
        cal = calendar_columns(dates, ev[t], sessions)
        if snapshots is not None and not snapshots.empty:
            sched = _scheduled_next(dates, snapshots[snapshots["ticker"] == t], sessions)
            use = sched.notna().to_numpy()
            if use.any():
                cal.loc[use, "earn_next_date"] = sched[use].to_numpy()
                pos_t = sessions.searchsorted(dates[use], side="right")
                pos_n = sessions.searchsorted(pd.DatetimeIndex(sched[use]), side="right")
                cal.loc[use, "earn_days_to"] = pos_n - pos_t
        out.loc[idx, "earn_next_date"] = cal["earn_next_date"].to_numpy()
        out.loc[idx, "earn_days_to"] = cal["earn_days_to"].to_numpy()
        out.loc[idx, "earn_days_since"] = cal["earn_days_since"].to_numpy()
        if "_atm_term" in rows:
            out.loc[idx, "earn_implied_move"] = [
                np.sqrt(implied_event_variance(json.loads(term), nd, dt, sessions))
                if isinstance(term, str) and pd.notna(nd) else np.nan
                for term, nd, dt in zip(rows["_atm_term"], cal["earn_next_date"], rows["date"])]
        if prices is not None and not prices.empty and t in prices.index.get_level_values("ticker"):
            p = prices.xs(t, level="ticker")
            close = p["adj_close"].fillna(p["close"]) if "adj_close" in p else p["close"]
            hv = historical_event_variance(close, ev[t])
            out.loc[idx, "earn_hist_move"] = np.sqrt(hv.reindex(pd.DatetimeIndex(rows["date"])).to_numpy())
    out["earn_next_date"] = pd.to_datetime(out["earn_next_date"])
    return out
