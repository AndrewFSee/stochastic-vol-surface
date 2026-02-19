"""
Streamlit dashboard main entrypoint for live vol surface monitoring.

Run with:
    streamlit run src/dashboard/app.py

Phase 2 dashboard features:
- 3D interactive vol surface (Plotly)
- VIX term structure with regime shading
- 25Δ skew time series with Z-score bands
- Live signal monitoring panel
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

try:
    import streamlit as st
    import pandas as pd
    import numpy as np

    from src.dashboard.surface_3d import plot_vol_surface_3d
    from src.dashboard.term_structure_chart import plot_term_structure
    from src.dashboard.skew_chart import plot_skew_timeseries
    from src.dashboard.signals_panel import render_signals_panel

    st.set_page_config(
        page_title="Vol Surface Modeler",
        page_icon="📈",
        layout="wide",
    )

    st.title("🌊 Deep Stochastic Volatility Surface Modeler")
    st.caption("Phase 2: Live Monitoring Dashboard")

    # Sidebar controls
    st.sidebar.header("Settings")
    ticker = st.sidebar.selectbox("Ticker", ["SPY", "QQQ", "AAPL", "MSFT"])
    tenor_label = st.sidebar.selectbox(
        "Reference Tenor",
        ["1M", "3M", "6M", "1Y"],
        index=1,
    )
    tenor_map = {"1M": 1 / 12, "3M": 3 / 12, "6M": 6 / 12, "1Y": 1.0}
    tenor = tenor_map[tenor_label]

    # Load data
    st.sidebar.markdown("---")
    if st.sidebar.button("🔄 Refresh Data"):
        st.cache_data.clear()

    # Main layout
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader(f"📊 Vol Surface — {ticker}")
        st.info("Load a surface snapshot from storage to visualize it here.")
        # TODO: Load latest surface from storage and render 3D plot

    with col2:
        st.subheader("📡 Signal Monitor")
        render_signals_panel(ticker=ticker, tenor=tenor)

    st.subheader("📈 VIX Term Structure")
    plot_term_structure()

    st.subheader("📉 25Δ Skew — Z-Score")
    plot_skew_timeseries(ticker=ticker, tenor=tenor)

except ImportError:
    # Running without streamlit (e.g., in tests)
    pass
