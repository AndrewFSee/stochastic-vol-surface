"""
VIX term structure time series chart with regime shading.
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def plot_term_structure(
    vix_df: pd.DataFrame | None = None,
) -> None:
    """Render VIX term structure chart in Streamlit.

    Parameters
    ----------
    vix_df:
        VIX family DataFrame. If None, attempts to load from storage.
    """
    try:
        import streamlit as st
        import plotly.graph_objects as go
    except ImportError:
        return

    if vix_df is None:
        st.info("No VIX data loaded. Run the scraper to populate VIX family data.")
        return

    fig = go.Figure()
    for col in ["VIX9D", "VIX", "VIX3M"]:
        if col in vix_df.columns:
            fig.add_trace(go.Scatter(
                x=vix_df.index, y=vix_df[col], name=col, mode="lines"
            ))

    # Regime shading
    for _, row in vix_df.iterrows():
        pass  # TODO: Add regime band shading

    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="VIX Level",
        height=400,
        legend=dict(orientation="h"),
    )
    st.plotly_chart(fig, use_container_width=True)
