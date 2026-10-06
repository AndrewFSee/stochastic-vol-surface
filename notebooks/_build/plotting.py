"""Matplotlib styling for the notebooks: the dashboard's validated light palette.

Series colours come from ``src.dashboard.theme.LIGHT`` (validated for colour
vision deficiency, all pairs, for up to three series). Charts use at most
three series; ranked comparisons use one colour.
"""

from __future__ import annotations

import matplotlib as mpl

from src.dashboard.theme import LIGHT as THEME


def setup_matplotlib() -> None:
    mpl.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 110,
        "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"],
        "font.size": 10,
        "axes.edgecolor": THEME.axis,
        "axes.linewidth": 0.8,
        "axes.labelcolor": THEME.ink_secondary,
        "axes.titlecolor": THEME.ink,
        "axes.prop_cycle": mpl.cycler(color=list(THEME.series)),
        "xtick.color": THEME.ink_muted,
        "ytick.color": THEME.ink_muted,
        "grid.color": THEME.grid,
        "grid.linewidth": 0.8,
        "lines.linewidth": 1.8,
        "legend.frameon": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def style_axes(ax, title=None, xlabel=None, ylabel=None) -> None:
    """Hairline grid, no top/right spines, left-aligned title."""
    ax.grid(True, axis="y")
    ax.spines[["top", "right"]].set_visible(False)
    if title:
        ax.set_title(title, loc="left", fontsize=11)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
