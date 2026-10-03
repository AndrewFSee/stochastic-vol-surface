"""Streamlit dashboard over the feature table and the per-expiry surface fits.

    streamlit run src/dashboard/app.py

Views
-----
Overview        every ticker on one row: level, change, percentile, trend
Smile           fitted smiles drawn over the market quotes they came from
Term structure  ATM vol by expiry, today against a week and a month ago
Surface         implied vol on a delta × tenor grid, and its change
History         implied vs realised, VRP, skew and term spread through time
Quality         fit error and expiry coverage; SPY's variance swap vs VIX

Everything reads from the stores the daily job writes; nothing is computed
from live market data here.  Values outside the quoted region are blank,
never extrapolated.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# `streamlit run src/dashboard/app.py` puts src/dashboard on the path, not the
# project root that the `src.` imports need.
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from src.dashboard import data as D
from src.dashboard.theme import Theme, colorscale, style, theme_for

# The store to read; override with VSS_DATA_DIR (e.g. for a test store).
# Defaults to the project's own data folder, so the app finds it whatever
# directory it is launched from.
DATA_DIR = os.environ.get("VSS_DATA_DIR", str(_ROOT / "data"))
FEATURES_PATH = f"{DATA_DIR}/features/surface_features.parquet"
SURFACES_DIR = f"{DATA_DIR}/surfaces"
OPTIONS_DIR = f"{DATA_DIR}/options"

WINDOWS = {"3M": 91, "6M": 182, "1Y": 365, "All": None}
COMPARE = {"1 day": 1, "1 week": 7, "1 month": 30}


# ── Cached access ────────────────────────────────────────────────────────
# Paths are explicit arguments (no defaults): Streamlit's caches key only on
# the arguments actually passed, so a default would let one store's cached
# data answer for another.


@st.cache_data(ttl=300, show_spinner=False)
def _features(path: str) -> pd.DataFrame:
    from src.features import load_feature_table

    return load_feature_table(path)


@st.cache_resource(ttl=300, show_spinner=False)
def _surface(ticker: str, as_of: str, surfaces_dir: str):
    return D.load_vol_surface(ticker, as_of, surfaces_dir)


@st.cache_data(ttl=300, show_spinner=False)
def _quotes(ticker: str, as_of: str, expirations: tuple[str, ...],
            surfaces_dir: str, options_dir: str) -> pd.DataFrame:
    vs = _surface(ticker, as_of, surfaces_dir)
    return D.smile_quotes(vs, ticker, as_of, list(expirations), options_dir)


@st.cache_data(ttl=300, show_spinner=False)
def _grid(ticker: str, as_of: str, surfaces_dir: str) -> pd.DataFrame:
    return D.delta_tenor_grid(_surface(ticker, as_of, surfaces_dir))


def _chart(fig) -> None:
    st.plotly_chart(fig, width="stretch", theme=None, config={"displaylogo": False})


def _table(df: pd.DataFrame, label: str = "Show data") -> None:
    """Every chart's table-view twin, so no value is reachable only by hover."""
    with st.expander(label):
        st.dataframe(df, width="stretch", hide_index=True)


# ── Views ────────────────────────────────────────────────────────────────


def view_overview(feats: pd.DataFrame, as_of: pd.Timestamp) -> None:
    ov = D.overview(feats, as_of)
    if ov.empty:
        st.info("No tickers have data on or shortly before this date.")
        return

    def p(col):
        return 100 * ov[col] if col in ov else np.nan

    table = pd.DataFrame({
        "Ticker": ov["ticker"],
        "Spot": ov["spot"],
        "ATM 30d": p("atm_30d"),
        "1d change": p("atm_30d_d1"),
        "Percentile": p("atm_30d_pct252"),
        "ATM 30d, last 3M": ov["atm_30d_trend"].apply(lambda v: [100 * x for x in v]),
        "Var swap 30d": p("vs_30d"),
        "Realised 21d": p("rv_cc_21d"),
        "VRP 30d": p("vrp_30d"),
        "RR 25Δ 30d": p("rr25_30d"),
        "BF 25Δ 30d": p("bf25_30d"),
        "Term 30→91d": p("ts_30_91"),
        "Fit error": p("fit_rmse"),
        "As of": ov["date"].dt.date,
    })
    vol = st.column_config.NumberColumn(format="%.1f%%")
    spread = st.column_config.NumberColumn(format="%+.2f pts")
    st.dataframe(
        table, width="stretch", hide_index=True,
        column_config={
            "Spot": st.column_config.NumberColumn(format="%.2f"),
            "ATM 30d": vol, "Var swap 30d": vol, "Realised 21d": vol,
            "1d change": spread, "VRP 30d": spread, "RR 25Δ 30d": spread,
            "BF 25Δ 30d": spread, "Term 30→91d": spread,
            "Fit error": st.column_config.NumberColumn(format="%.2f pts"),
            "Percentile": st.column_config.ProgressColumn(
                format="%.0f", min_value=0, max_value=100,
                help="Rank of today's 30d ATM vol within the trailing year "
                     "(or all history, where shorter)"),
            "ATM 30d, last 3M": st.column_config.LineChartColumn(width="medium"),
        },
    )
    st.caption(
        "Vols in %, spreads in vol points. RR = 25Δ call − 25Δ put (negative = "
        "puts richer). VRP = 30d ATM implied − 21d realised. None = not quoted "
        "(no expiry near that tenor, or strikes outside the listed range)."
    )


def view_smile(t: Theme, ticker: str, as_of: str) -> None:
    vs = _surface(ticker, as_of, SURFACES_DIR)
    if vs is None:
        st.info("No fitted surface for this date.")
        return
    tbl = D.expiry_table(vs)
    labels = {e: f"{e}  ({d}d)" for e, d in zip(tbl["expiration"], tbl["days"])}
    by_label = {v: k for k, v in labels.items()}
    picked = st.multiselect(
        "Expiries (up to 4)", list(by_label),
        default=[labels[e] for e in D.default_expiries(vs)], max_selections=4,
        key=f"smile_expiries_{ticker}_{as_of}",
    )
    chosen = [by_label[p] for p in picked]
    if not chosen:
        return
    chosen = sorted(chosen, key=lambda e: tbl.set_index("expiration").loc[e, "T"])
    curves = D.smile_curves(vs, chosen)
    quotes = _quotes(ticker, as_of, tuple(chosen), SURFACES_DIR, OPTIONS_DIR)

    cols = 2 if len(chosen) > 1 else 1
    rows = int(np.ceil(len(chosen) / cols))
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=[labels[e] for e in chosen],
                        horizontal_spacing=0.08, vertical_spacing=0.14)
    fig.update_annotations(font=dict(size=13, color=t.ink_secondary))  # panel titles
    for i, e in enumerate(chosen):
        r, c = i // cols + 1, i % cols + 1
        q = quotes[quotes["expiration"] == e] if not quotes.empty else quotes
        if not q.empty:
            fig.add_trace(go.Scatter(
                x=q["moneyness"], y=100 * q["iv"], mode="markers", name="Market quote (mid ± ½ spread)",
                legendgroup="q", showlegend=i == 0,
                marker=dict(size=8, color=t.series[0], line=dict(width=2, color=t.surface)),
                error_y=dict(type="data", array=100 * q["half_spread"], thickness=1,
                             width=0, color=t.series[0]),
                customdata=np.column_stack([q["strike"], 100 * q["half_spread"]]),
                hovertemplate="<b>%{y:.2f}%</b> at %{x:.1f}% of fwd<br>strike %{customdata[0]:.1f}"
                              "  ±%{customdata[1]:.2f} pts<extra>quote</extra>",
            ), row=r, col=c)
        cv = curves[curves["expiration"] == e]
        fig.add_trace(go.Scatter(
            x=cv["moneyness"], y=100 * cv["iv"], mode="lines", name="SVI fit",
            legendgroup="fit", showlegend=i == 0,
            line=dict(width=2, color=t.series[1]),
            hovertemplate="<b>%{y:.2f}%</b> at %{x:.1f}% of fwd<extra>fit</extra>",
        ), row=r, col=c)
        for name, m in D.delta_markers(vs, e).items():
            if np.isfinite(m) and cv["moneyness"].min() <= m <= cv["moneyness"].max():
                fig.add_vline(x=m, line=dict(width=1, color=t.axis), row=r, col=c)
                # Inside the plot's top edge, clear of the panel title above it.
                fig.add_annotation(x=m, y=1, yref=f"y{i + 1} domain" if i else "y domain",
                                   xref=f"x{i + 1}" if i else "x", text=name, showarrow=False,
                                   yanchor="top", xanchor="left", xshift=3,
                                   font=dict(size=11, color=t.ink_muted))
    style(fig, t, height=380 * rows + 40, unified_hover=False)
    # Legend above the panel titles rather than on top of them.
    fig.update_layout(margin=dict(t=96), legend=dict(y=1.0, yref="container", yanchor="top"))
    fig.update_xaxes(title_text="Strike, % of forward")
    fig.update_yaxes(title_text="Implied vol (%)")
    _chart(fig)
    st.caption("Out-of-the-money quotes only (puts below the forward, calls above). "
               "The fit is drawn across the quoted range; vertical rules mark the "
               "25-delta strikes.")
    _table(tbl.assign(T=tbl["T"].round(4), forward=tbl["forward"].round(2),
                      fit_rmse_pts=tbl["fit_rmse_pts"].round(3)), "Show all expiries")


def view_term(t: Theme, ticker: str, as_of: pd.Timestamp, dates: list) -> None:
    targets = [("As of", as_of), ("1 week earlier", as_of - pd.Timedelta(days=7)),
               ("1 month earlier", as_of - pd.Timedelta(days=30))]
    fig = go.Figure()
    rows = []
    for i, (name, target) in enumerate(targets):
        d = D.on_or_before(dates, target)
        vs = _surface(ticker, str(d.date()), SURFACES_DIR) if d is not None else None
        if vs is None:
            continue
        curve, points = D.term_structure(vs)
        label = f"{name} ({d.date()})"
        fig.add_trace(go.Scatter(
            x=curve["days"], y=100 * curve["iv"], mode="lines", name=label,
            line=dict(width=2, color=t.series[i]),
            hovertemplate="%{y:.2f}%<extra>" + label + "</extra>",
        ))
        if i == 0:
            fig.add_trace(go.Scatter(
                x=points["days"], y=100 * points["iv"], mode="markers", name="Listed expiries",
                marker=dict(size=8, color=t.series[0], line=dict(width=2, color=t.surface)),
                hoverinfo="skip",
            ))
        rows.append(points.assign(series=label))
    if not rows:
        st.info("No fitted surface for this date.")
        return
    style(fig, t, height=420, y_title="ATM implied vol (%)", x_title="Days to expiry (log scale)")
    fig.update_xaxes(type="log", tickvals=[7, 14, 30, 60, 91, 182, 365, 730],
                     ticktext=["7", "14", "30", "60", "91", "182", "365", "730"])
    _chart(fig)
    st.caption("Forward-ATM vol interpolated in total variance between listed "
               "expiries — the same construction as VIX's constant 30 days.")
    tbl = pd.concat(rows)
    _table(tbl.assign(days=tbl["days"].round(0), iv=(100 * tbl["iv"]).round(2))
              .rename(columns={"iv": "ATM vol (%)"}))


def view_surface(t: Theme, ticker: str, as_of: pd.Timestamp, dates: list, compare_days: int,
                 compare_label: str) -> None:
    g = _grid(ticker, str(as_of.date()), SURFACES_DIR)
    if g.isna().all().all():
        st.info("No fitted surface for this date.")
        return
    prev = D.on_or_before(dates, as_of - pd.Timedelta(days=compare_days))
    g_prev = (_grid(ticker, str(prev.date()), SURFACES_DIR)
              if prev is not None and prev < as_of else None)

    c1, c2 = st.columns(2)
    with c1:
        z = 100 * g.to_numpy(dtype=float)
        fig = go.Figure(go.Heatmap(
            z=z, x=list(g.columns), y=list(g.index), colorscale=colorscale(t.sequential),
            colorbar=dict(title=dict(text="IV %", font=dict(color=t.ink_secondary)),
                          tickfont=dict(color=t.ink_muted), outlinewidth=0, thickness=12),
            xgap=2, ygap=2, hoverongaps=False,
            hovertemplate="%{y} · %{x}<br><b>%{z:.1f}%</b><extra></extra>",
        ))
        style(fig, t, height=400, title=f"Implied vol, {as_of.date()}",
              legend=False, unified_hover=False)
        fig.update_xaxes(showgrid=False, showline=False)
        fig.update_yaxes(showgrid=False, showline=False, autorange="reversed")
        _chart(fig)
    with c2:
        if g_prev is None:
            st.info(f"No earlier surface {compare_label} before this date.")
        else:
            dz = 100 * (g - g_prev).to_numpy(dtype=float)
            lim = float(np.nanmax(np.abs(dz))) if np.isfinite(dz).any() else 1.0
            fig = go.Figure(go.Heatmap(
                z=dz, x=list(g.columns), y=list(g.index), colorscale=colorscale(t.diverging),
                zmid=0, zmin=-lim, zmax=lim,
                colorbar=dict(title=dict(text="Δ pts", font=dict(color=t.ink_secondary)),
                              tickfont=dict(color=t.ink_muted), outlinewidth=0, thickness=12),
                xgap=2, ygap=2, hoverongaps=False,
                hovertemplate="%{y} · %{x}<br><b>%{z:+.2f} pts</b><extra></extra>",
            ))
            style(fig, t, height=400, title=f"Change since {prev.date()} ({compare_label})",
                  legend=False, unified_hover=False)
            fig.update_xaxes(showgrid=False, showline=False)
            fig.update_yaxes(showgrid=False, showline=False, autorange="reversed")
            _chart(fig)
    st.caption("Rows are constant maturities, columns forward-delta strikes from "
               "10Δ put to 10Δ call. Red = vol up, blue = vol down. Blank cells are "
               "outside the listed expiries or quoted strikes.")

    table = (100 * g).round(2).reset_index().rename(columns={"index": "Tenor"})
    if g_prev is not None:
        chg = (100 * (g - g_prev)).round(2).add_suffix(" Δ").reset_index(drop=True)
        table = pd.concat([table, chg], axis=1)
    _table(table)

    with st.expander("3D view"):
        z = 100 * g.to_numpy(dtype=float)
        fig = go.Figure(go.Surface(
            z=z, x=list(range(len(g.columns))), y=[D.TENORS_DAYS[n] for n in g.index],
            colorscale=colorscale(t.sequential), showscale=False,
            hovertemplate="%{y}d<br><b>%{z:.1f}%</b><extra></extra>",
        ))
        axis = dict(gridcolor=t.grid, color=t.ink_muted, backgroundcolor="rgba(0,0,0,0)")
        fig.update_layout(
            height=520, margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor="rgba(0,0,0,0)",
            scene=dict(
                xaxis=dict(axis, title="Delta", tickvals=list(range(len(g.columns))),
                           ticktext=list(g.columns)),
                yaxis=dict(axis, title="Days", type="log"),
                zaxis=dict(axis, title="IV %"),
            ),
        )
        _chart(fig)


def _line(t: Theme, df: pd.DataFrame, series: list[tuple[str, str]], title: str,
          y_title: str, zero: bool = False):
    fig = go.Figure()
    for i, (col, name) in enumerate(series):
        fig.add_trace(go.Scatter(
            x=df["date"], y=100 * df[col], mode="lines", name=name,
            line=dict(width=2, color=t.series[i]),
            hovertemplate="%{y:.2f}<extra>" + name + "</extra>",
        ))
    style(fig, t, height=320, title=title, y_title=y_title, legend=len(series) > 1)
    if zero:
        fig.add_hline(y=0, line=dict(width=1, color=t.axis))
    return fig


def view_history(t: Theme, feats: pd.DataFrame, ticker: str, as_of: pd.Timestamp,
                 window_days) -> None:
    start = None if window_days is None else as_of - pd.Timedelta(days=window_days)
    h = D.history(feats, ticker, start, as_of)
    if h.empty:
        st.info("No history in this window.")
        return
    c1, c2 = st.columns(2)
    with c1:
        _chart(_line(t, h, [("atm_30d", "30d ATM implied"), ("rv_cc_21d", "21d realised")],
                     "Implied vs realised vol", "Vol (%)"))
        _chart(_line(t, h, [("rr25_30d", "RR 25Δ")], "30d 25Δ risk reversal (call − put)",
                     "Vol points", zero=True))
    with c2:
        _chart(_line(t, h, [("vrp_30d", "VRP")], "Variance risk premium (30d implied − 21d realised)",
                     "Vol points", zero=True))
        _chart(_line(t, h, [("ts_30_91", "Term")], "Term spread (91d − 30d ATM)",
                     "Vol points", zero=True))
    cols = ["date", "atm_30d", "rv_cc_21d", "vrp_30d", "rr25_30d", "ts_30_91", "bf25_30d", "vs_30d"]
    tbl = h[[c for c in cols if c in h]].copy()
    num = tbl.columns.drop("date")
    tbl[num] = (100 * tbl[num]).round(2)
    tbl["date"] = tbl["date"].dt.date
    _table(tbl.iloc[::-1])


def view_quality(t: Theme, feats: pd.DataFrame, ticker: str, as_of: pd.Timestamp,
                 window_days) -> None:
    from src.surface.diagnostics import benchmark_against

    start = None if window_days is None else as_of - pd.Timedelta(days=window_days)
    spy = D.history(feats, "SPY", start, as_of)
    if not spy.empty and spy["mkt_vix"].notna().any():
        st.markdown("**Accuracy check: SPY 30d variance swap vs VIX**")
        b = benchmark_against(100 * spy.set_index("date")["vs_30d"],
                              spy.set_index("date")["mkt_vix"])
        k1, k2, k3 = st.columns(3)
        k1.metric("Daily-change correlation", f"{b.get('change_corr', np.nan):.3f}",
                  help="Our variance swap and VIX share a definition; near 1 means the "
                       "surfaces move with the market, not with noise.")
        k2.metric("Mean |difference|", f"{b.get('mean_abs_diff', np.nan):.2f} pts")
        k3.metric("Days compared", f"{b.get('n', 0)}")
        fig = go.Figure()
        for i, (y, name) in enumerate([(100 * spy["vs_30d"], "SPY 30d variance swap (ours)"),
                                       (spy["mkt_vix"], "VIX (CBOE)")]):
            fig.add_trace(go.Scatter(x=spy["date"], y=y, mode="lines", name=name,
                                     line=dict(width=2, color=t.series[i]),
                                     hovertemplate="%{y:.2f}<extra>" + name + "</extra>"))
        style(fig, t, height=320, y_title="Vol (%)")
        _chart(fig)

    h = D.history(feats, ticker, start, as_of)
    if not h.empty:
        c1, c2 = st.columns(2)
        with c1:
            _chart(_line(t, h, [("fit_rmse", "Fit error")],
                         f"{ticker} median fit error per expiry", "Vol points"))
        with c2:
            fig = go.Figure(go.Scatter(x=h["date"], y=h["n_expiries"], mode="lines",
                                       line=dict(width=2, color=t.series[0], shape="hv"),
                                       hovertemplate="%{y:.0f}<extra>expiries</extra>"))
            style(fig, t, height=320, title=f"{ticker} expiries fitted", y_title="Count",
                  legend=False)
            _chart(fig)

    latest = D.overview(feats, as_of)
    if not latest.empty:
        q = pd.DataFrame({
            "Ticker": latest["ticker"],
            "Expiries fitted": latest["n_expiries"].astype("Int64"),
            "Nearest expiry (days)": latest["nearest_expiry_days"].round(0),
            "Median fit error (pts)": (100 * latest["fit_rmse"]).round(3),
            "Forwards from parity": (100 * latest["parity_fraction"]).round(0),
        })
        st.markdown("**Latest surface quality**")
        st.dataframe(q, width="stretch", hide_index=True,
                     column_config={"Forwards from parity": st.column_config.NumberColumn(format="%.0f%%")})


# ── App ──────────────────────────────────────────────────────────────────


def run_dashboard() -> None:
    st.set_page_config(page_title="Vol Surface Dashboard", layout="wide")
    t = theme_for(getattr(getattr(st.context, "theme", None), "type", None))

    feats = _features(FEATURES_PATH)
    if feats.empty:
        st.title("Vol Surface Dashboard")
        st.info("No feature table yet. Build it with:\n\n"
                "```\npython scripts/build_surfaces.py\npython scripts/build_features.py\n```")
        return

    st.title("Vol Surface Dashboard")

    # ── One filter row, scoping everything below ─────────────────────────
    tickers = sorted(feats["ticker"].unique())
    f1, f2, f3, f4 = st.columns([1, 1, 1.4, 1.4])
    ticker = f1.selectbox("Ticker", tickers, key="ticker",
                          index=tickers.index("SPY") if "SPY" in tickers else 0)
    dates = D.dates_for(feats, ticker)
    as_of = f2.selectbox("As of", dates, format_func=lambda d: str(d.date()))
    window = f3.segmented_control("History", list(WINDOWS), default="6M") or "6M"
    compare = f4.segmented_control("Compare with", list(COMPARE), default="1 week") or "1 week"

    # ── Market context ───────────────────────────────────────────────────
    h = D.history(feats, ticker, end=as_of)
    row, prev = h.iloc[-1], (h.iloc[-2] if len(h) > 1 else None)

    def delta(col, scale=1.0, fmt="{:+.2f}"):
        if prev is None or not np.isfinite(row.get(col, np.nan)) or not np.isfinite(prev.get(col, np.nan)):
            return None
        return fmt.format(scale * (row[col] - prev[col]))

    k = st.columns(5)
    k[0].metric(f"{ticker} 30d ATM", D.pct(row["atm_30d"]), delta("atm_30d", 100, "{:+.2f} pts"),
                delta_color="off")
    pct = row.get("atm_30d_pct252", np.nan)
    k[1].metric("Percentile (1y)", "–" if not np.isfinite(pct) else f"{100 * pct:.0f}")
    k[2].metric(f"{ticker} VRP 30d", D.pts(row.get("vrp_30d", np.nan)))
    k[3].metric("VIX", "–" if not np.isfinite(row.get("mkt_vix", np.nan)) else f"{row['mkt_vix']:.2f}",
                delta("mkt_vix"), delta_color="off")
    ratio = row.get("mkt_vix3m", np.nan) / row.get("mkt_vix", np.nan)
    k[4].metric("VIX3M / VIX", "–" if not np.isfinite(ratio) else f"{ratio:.3f}",
                help="Above 1 = contango (calm); below 1 = backwardation (stress).")

    tabs = st.tabs(["Overview", "Smile", "Term structure", "Surface", "History", "Quality"])
    with tabs[0]:
        view_overview(feats, as_of)
    with tabs[1]:
        view_smile(t, ticker, str(as_of.date()))
    with tabs[2]:
        view_term(t, ticker, as_of, dates)
    with tabs[3]:
        view_surface(t, ticker, as_of, dates, COMPARE[compare], compare)
    with tabs[4]:
        view_history(t, feats, ticker, as_of, WINDOWS[window])
    with tabs[5]:
        view_quality(t, feats, ticker, as_of, WINDOWS[window])


if __name__ == "__main__":
    run_dashboard()
