"""Streamlit dashboard: 3-D surface, VIX term structure, skew panel, signal panel.

Reads the built surface corpus (``data/surfaces``) and the consolidated VIX
history.  Falls back to a synthetic surface when nothing has been built yet, so
the app is always runnable.
"""

from __future__ import annotations

try:
    import streamlit as st
    _HAS_ST = True
except ImportError:  # pragma: no cover
    _HAS_ST = False

try:
    import plotly.graph_objects as go
    _HAS_PLOTLY = True
except ImportError:  # pragma: no cover
    _HAS_PLOTLY = False

import numpy as np

DEFAULT_SURFACES_DIR = "data/surfaces"
DEFAULT_VIX_DIR = "data/vix"


def _make_demo_surface() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a demo IV surface for display when no real data is loaded."""
    k = np.linspace(-0.4, 0.2, 25)
    T = np.array([1/12, 3/12, 6/12, 1.0, 2.0])
    KK, TT = np.meshgrid(k, T, indexing="ij")
    # Rough smile + term structure
    iv = 0.18 + 0.05 * np.exp(-2 * KK ** 2) + 0.02 * np.log1p(TT) - 0.03 * KK
    return k, T, iv


# ── Data access (cached) ─────────────────────────────────────────────────


def _list_tickers(surfaces_dir: str = DEFAULT_SURFACES_DIR) -> list[str]:
    """Return tickers that have at least one built surface."""
    from pathlib import Path

    base = Path(surfaces_dir)
    if not base.exists():
        return []
    return sorted(
        p.name.split("=", 1)[1]
        for p in base.iterdir()
        if p.is_dir() and p.name.startswith("ticker=") and any(p.glob("date=*/surface.parquet"))
    )


def _load_history(ticker: str, surfaces_dir: str = DEFAULT_SURFACES_DIR):
    """Load a ticker's surface history: ``(dates, k_grid, t_grid, grids)``."""
    from src.surface.batch import load_surface_history

    return load_surface_history(ticker, surfaces_dir=surfaces_dir)


def _load_vix(vix_dir: str = DEFAULT_VIX_DIR):
    """Load the consolidated VIX-family history with derived ratios."""
    from src.data.vix_family import vix_term_structure_signal

    return vix_term_structure_signal(vix_dir)


if _HAS_ST:
    _list_tickers = st.cache_data(ttl=300)(_list_tickers)
    _load_history = st.cache_data(ttl=300)(_load_history)
    _load_vix = st.cache_data(ttl=300)(_load_vix)


# ── App ──────────────────────────────────────────────────────────────────


def run_dashboard() -> None:
    if not _HAS_ST:
        raise ImportError("streamlit is required: pip install streamlit")

    st.set_page_config(page_title="Vol Surface Dashboard", layout="wide")
    st.title("🌊 Deep Stochastic Volatility Surface Modeler")

    # ── Sidebar ───────────────────────────────────────────────────────────
    st.sidebar.header("Configuration")

    tickers = _list_tickers()
    using_demo = not tickers

    if using_demo:
        st.sidebar.warning(
            "No built surfaces found.\n\n"
            "Run `python scripts/build_surfaces.py` to populate `data/surfaces`."
        )
        ticker = "DEMO"
        k, T, iv = _make_demo_surface()
        as_of_label = "synthetic"
        dates = []
    else:
        ticker = st.sidebar.selectbox("Ticker", tickers)
        dates, k, T, grids = _load_history(ticker)
        if not dates:
            st.error(f"No readable surfaces for {ticker}.")
            return
        idx = st.sidebar.select_slider(
            "As-of date",
            options=list(range(len(dates))),
            value=len(dates) - 1,
            format_func=lambda i: str(dates[i]),
        )
        iv = grids[idx]
        as_of_label = str(dates[idx])
        st.sidebar.caption(f"{len(dates)} snapshots: {dates[0]} → {dates[-1]}")

    tenor_idx = st.sidebar.slider("Tenor slice index", 0, len(T) - 1,
                                  min(2, len(T) - 1))
    atm_idx = int(np.argmin(np.abs(k)))
    title_suffix = " (Demo)" if using_demo else f" — {as_of_label}"

    tabs = st.tabs(["3D Surface", "Skew", "Term Structure", "History", "Signals"])

    # --- 3D Surface ---
    with tabs[0]:
        st.subheader("Implied Volatility Surface")
        if _HAS_PLOTLY:
            KK, TT = np.meshgrid(k, T, indexing="ij")
            fig = go.Figure(data=[go.Surface(
                x=TT, y=KK, z=iv,
                colorscale="Viridis",
                colorbar=dict(title="IV"),
            )])
            fig.update_layout(
                scene=dict(
                    xaxis_title="Tenor (years)",
                    yaxis_title="Log-moneyness",
                    zaxis_title="Implied Vol",
                ),
                height=600,
                title=f"{ticker} IV Surface{title_suffix}",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Install plotly for interactive charts.")

    # --- Skew ---
    with tabs[1]:
        st.subheader("Vol Skew Slice")
        if _HAS_PLOTLY:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=k, y=iv[:, tenor_idx], mode="lines+markers",
                                     name=f"T={T[tenor_idx]:.2f}y"))
            fig.update_layout(xaxis_title="Log-moneyness", yaxis_title="IV",
                              height=400, title=f"{ticker}{title_suffix}")
            st.plotly_chart(fig, use_container_width=True)

    # --- Term Structure ---
    with tabs[2]:
        st.subheader("ATM Term Structure")
        atm_ivs = iv[atm_idx, :]
        if _HAS_PLOTLY:
            fig = go.Figure(go.Scatter(x=T, y=atm_ivs, mode="lines+markers"))
            fig.update_layout(xaxis_title="Tenor (years)", yaxis_title="ATM IV",
                              height=400, title=f"{ticker}{title_suffix}")
            st.plotly_chart(fig, use_container_width=True)

    # --- History ---
    with tabs[3]:
        st.subheader("ATM Vol Through Time")
        if using_demo:
            st.info("Build the surface corpus to see historical ATM vol.")
        elif _HAS_PLOTLY:
            _dates, _k, _T, grids = _load_history(ticker)
            _atm = int(np.argmin(np.abs(_k)))
            fig = go.Figure()
            for ti, tv in enumerate(_T):
                fig.add_trace(go.Scatter(
                    x=_dates, y=grids[:, _atm, ti], mode="lines",
                    name=f"{tv:.2f}y",
                ))
            fig.update_layout(xaxis_title="Date", yaxis_title="ATM IV",
                              height=450, title=f"{ticker} ATM term structure history")
            st.plotly_chart(fig, use_container_width=True)

    # --- Signals ---
    with tabs[4]:
        st.subheader("Trading Signals")
        from src.signals.regime_vol import classify_regime

        vix_df = _load_vix()
        if vix_df is not None and not vix_df.empty and "VIX" in vix_df:
            latest_vix = float(vix_df["VIX"].dropna().iloc[-1])
            vix_date = vix_df["VIX"].dropna().index[-1].date()
            st.caption(f"Latest VIX close: {latest_vix:.2f} ({vix_date})")
        else:
            latest_vix = 18.0
            st.caption("No VIX history found — using a default of 18.0")

        vix_input = st.number_input("VIX level", value=latest_vix, step=0.5)
        regime = classify_regime(float(vix_input))

        c1, c2, c3 = st.columns(3)
        c1.metric("Vol Regime", regime.value)

        # 25-delta proxy: read the grid a few knots either side of ATM.
        off = min(3, atm_idx, len(k) - 1 - atm_idx)
        skew_val = float(iv[atm_idx - off, tenor_idx] - iv[atm_idx + off, tenor_idx])
        c2.metric(f"Skew (±{off} knots)", f"{skew_val:.4f}")
        c3.metric("ATM IV", f"{iv[atm_idx, tenor_idx]:.4f}")

        if vix_df is not None and not vix_df.empty and "vix_ts_slope" in vix_df:
            slope = vix_df["vix_ts_slope"].dropna()
            if not slope.empty:
                val = float(slope.iloc[-1])
                st.metric(
                    "VIX3M / VIX", f"{val:.3f}",
                    help="Above 1.0 is contango (calm); below 1.0 is backwardation (stress).",
                )


if __name__ == "__main__":
    run_dashboard()
