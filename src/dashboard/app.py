"""Streamlit dashboard: 3-D surface, VIX term structure, skew panel, signal panel."""

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


def _make_demo_surface() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a demo IV surface for display when no real data is loaded."""
    k = np.linspace(-0.4, 0.2, 25)
    T = np.array([1/12, 3/12, 6/12, 1.0, 2.0])
    KK, TT = np.meshgrid(k, T, indexing="ij")
    # Rough smile + term structure
    iv = 0.18 + 0.05 * np.exp(-2 * KK ** 2) + 0.02 * np.log1p(TT) - 0.03 * KK
    return k, T, iv


def run_dashboard() -> None:
    if not _HAS_ST:
        raise ImportError("streamlit is required: pip install streamlit")

    st.set_page_config(page_title="Vol Surface Dashboard", layout="wide")
    st.title("🌊 Deep Stochastic Volatility Surface Modeler")

    # Sidebar controls
    st.sidebar.header("Configuration")
    ticker = st.sidebar.selectbox("Ticker", ["SPY", "QQQ", "AAPL", "MSFT"])
    tenor_idx = st.sidebar.slider("Tenor slice index", 0, 4, 2)

    # Load or generate surface
    k, T, iv = _make_demo_surface()

    tabs = st.tabs(["3D Surface", "Skew", "Term Structure", "Signals"])

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
                title=f"{ticker} IV Surface (Demo)",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Install plotly for interactive charts.")

    # --- Skew ---
    with tabs[1]:
        st.subheader("Vol Skew Slice")
        if _HAS_PLOTLY:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=k, y=iv[:, tenor_idx],
                                     name=f"T={T[tenor_idx]:.2f}y"))
            fig.update_layout(xaxis_title="Log-moneyness", yaxis_title="IV", height=400)
            st.plotly_chart(fig, use_container_width=True)

    # --- Term Structure ---
    with tabs[2]:
        st.subheader("ATM Term Structure")
        atm_idx = len(k) // 2
        atm_ivs = iv[atm_idx, :]
        if _HAS_PLOTLY:
            fig = go.Figure(go.Scatter(x=T, y=atm_ivs, mode="lines+markers"))
            fig.update_layout(xaxis_title="Tenor (years)", yaxis_title="ATM IV", height=400)
            st.plotly_chart(fig, use_container_width=True)

    # --- Signals ---
    with tabs[3]:
        st.subheader("Trading Signals")
        from src.signals.regime_vol import classify_regime
        demo_vix = st.number_input("Current VIX", value=18.0, step=0.5)
        regime = classify_regime(float(demo_vix))
        st.metric("Vol Regime", regime.value)

        skew_val = float(iv[atm_idx - 3, tenor_idx] - iv[atm_idx + 3, tenor_idx])
        st.metric("25-delta Skew (demo)", f"{skew_val:.4f}")
        st.metric("ATM IV (demo)", f"{iv[atm_idx, tenor_idx]:.4f}")


if __name__ == "__main__":
    run_dashboard()
