"""
Live signal monitoring panel for the Streamlit dashboard.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def render_signals_panel(
    ticker: str = "SPY",
    tenor: float = 3 / 12,
) -> None:
    """Render the signal monitoring panel in Streamlit.

    Displays the latest skew, term structure, butterfly, and composite signals.
    """
    try:
        import streamlit as st
    except ImportError:
        return

    st.metric("Skew Signal", "N/A", help="25Δ put-call skew Z-score")
    st.metric("Term Structure", "N/A", help="ATM term structure slope")
    st.metric("Butterfly", "N/A", help="Butterfly mispricing")
    st.metric("Vol Regime", "N/A", help="VIX-based regime classifier")
    st.caption("💡 Run the daily scraper to populate live signals.")
