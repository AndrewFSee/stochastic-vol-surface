"""
Interactive 3D vol surface visualization using Plotly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def plot_vol_surface_3d(
    surface_grid: pd.DataFrame,
    ticker: str = "",
    as_of: str = "",
    colorscale: str = "Viridis",
) -> "plotly.graph_objects.Figure":  # type: ignore[name-defined]  # noqa: F821
    """Create an interactive 3D vol surface plot.

    Parameters
    ----------
    surface_grid:
        DataFrame with log-moneyness index and tenor (years) columns.
    ticker:
        Ticker label for the plot title.
    as_of:
        Date string for the plot title.
    colorscale:
        Plotly colorscale name.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    import plotly.graph_objects as go

    moneyness = np.array(surface_grid.index.astype(float))
    tenors = np.array(surface_grid.columns.astype(float))
    iv_matrix = surface_grid.values.astype(float) * 100  # Convert to percentage

    fig = go.Figure(data=[go.Surface(
        x=tenors,
        y=moneyness,
        z=iv_matrix,
        colorscale=colorscale,
        colorbar=dict(title="IV (%)"),
    )])

    fig.update_layout(
        title=f"Implied Vol Surface — {ticker} ({as_of})",
        scene=dict(
            xaxis_title="Tenor (years)",
            yaxis_title="Log-Moneyness",
            zaxis_title="Implied Vol (%)",
        ),
        height=600,
    )
    return fig
