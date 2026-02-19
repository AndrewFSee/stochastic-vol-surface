"""
25-delta skew time series chart with Z-score bands.
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def plot_skew_timeseries(
    ticker: str = "SPY",
    tenor: float = 3 / 12,
    skew_series: pd.Series | None = None,
) -> None:
    """Render 25Δ skew time series with Z-score bands in Streamlit.

    Parameters
    ----------
    ticker:
        Ticker symbol (for chart title).
    tenor:
        Tenor in years.
    skew_series:
        Pre-computed skew series. If None, shows a placeholder.
    """
    try:
        import streamlit as st
        import plotly.graph_objects as go
        import numpy as np
    except ImportError:
        return

    if skew_series is None:
        st.info(f"No skew data available for {ticker} at {tenor:.0%} tenor.")
        return

    from src.signals.skew_signals import skew_zscore

    z = skew_zscore(skew_series)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=skew_series.index, y=skew_series.values,
                             name="25Δ Skew", line=dict(color="blue")))
    fig.add_hline(y=skew_series.mean(), line_dash="dash", line_color="gray",
                  annotation_text="Mean")

    fig.update_layout(
        title=f"25Δ Skew — {ticker} {tenor:.0%} Tenor",
        xaxis_title="Date",
        yaxis_title="Skew (vol pts)",
        height=350,
    )
    st.plotly_chart(fig, use_container_width=True)
