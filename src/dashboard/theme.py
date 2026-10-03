"""Chart palette and Plotly styling for the dashboard, in light and dark modes.

Every colour here is a documented palette step; dark mode uses its own
validated steps rather than an automatic inversion.  The three categorical
slots were checked all-pairs (they overlap in scatter and line charts) in
both modes:

    light  #2a78d6,#eb6834,#1baf7a   worst CVD ΔE 9.2, normal-vision ΔE 24.0
    dark   #3987e5,#d95926,#199e70   worst CVD ΔE 9.4, normal-vision ΔE 20.9

Light-mode aqua sits below 3:1 contrast, so every chart pairs colour with a
legend or direct labels and offers a table view.

Charts use at most three categorical series.  Past that, facet into small
multiples rather than adding hues.
"""

from __future__ import annotations

from dataclasses import dataclass

FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'


@dataclass(frozen=True)
class Theme:
    mode: str
    series: tuple[str, str, str]
    ink: str
    ink_secondary: str
    ink_muted: str
    grid: str
    axis: str
    surface: str
    #: Sequential ramp for magnitude, low → high.  Its low end recedes into
    #: the surface, so the anchor flips between modes.
    sequential: tuple[str, ...]
    #: Diverging blue ↔ red through a neutral gray (vol down ↔ vol up).
    diverging: tuple[str, str, str]


_BLUE_RAMP = ("#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#1c5cab", "#104281")

LIGHT = Theme(
    mode="light",
    series=("#2a78d6", "#eb6834", "#1baf7a"),
    ink="#0b0b0b", ink_secondary="#52514e", ink_muted="#898781",
    grid="#e1e0d9", axis="#c3c2b7", surface="#fcfcfb",
    sequential=_BLUE_RAMP,
    diverging=("#256abf", "#f0efec", "#e34948"),
)

DARK = Theme(
    mode="dark",
    series=("#3987e5", "#d95926", "#199e70"),
    ink="#ffffff", ink_secondary="#c3c2b7", ink_muted="#898781",
    grid="#2c2c2a", axis="#383835", surface="#1a1a19",
    sequential=tuple(reversed(_BLUE_RAMP)),
    diverging=("#3987e5", "#383835", "#e66767"),
)


def theme_for(mode: str | None) -> Theme:
    """The theme for Streamlit's reported mode (``"light"``/``"dark"``)."""
    return DARK if mode == "dark" else LIGHT


def colorscale(stops: tuple[str, ...]) -> list[list]:
    """Evenly spaced Plotly colourscale from a tuple of hex stops."""
    n = len(stops) - 1
    return [[i / n, c] for i, c in enumerate(stops)]


def style(fig, t: Theme, *, height: int = 380, title: str | None = None,
          y_title: str | None = None, x_title: str | None = None,
          legend: bool = True, unified_hover: bool = True):
    """Apply the shared chart chrome: hairline grid, recessive axes, ink text.

    Backgrounds are transparent so charts sit on Streamlit's own surface.
    Line charts get a unified hover (a crosshair that lists every series).
    """
    axis = dict(
        showgrid=True, gridcolor=t.grid, gridwidth=1, zeroline=False,
        linecolor=t.axis, linewidth=1, showline=True, ticks="", automargin=True,
        tickfont=dict(color=t.ink_muted, size=12),
        title_font=dict(color=t.ink_secondary, size=12),
    )
    fig.update_layout(
        height=height,
        # Room for tick labels and axis titles; automargin grows it further
        # when labels are longer.
        margin=dict(l=64, r=16, t=64 if title else 40, b=52),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, color=t.ink, size=13),
        title=dict(text=title, x=0, xanchor="left", font=dict(size=15, color=t.ink)) if title else None,
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0,
                    font=dict(color=t.ink_secondary, size=12), bgcolor="rgba(0,0,0,0)"),
        hovermode="x unified" if unified_hover else "closest",
        hoverlabel=dict(font=dict(family=FONT, size=12)),
    )
    fig.update_xaxes(**axis, title_text=x_title)
    fig.update_yaxes(**axis, title_text=y_title)
    if unified_hover:
        fig.update_xaxes(showspikes=True, spikemode="across", spikesnap="cursor",
                         spikethickness=1, spikecolor=t.axis, spikedash="solid")
    return fig


def rgba(hex_color: str, alpha: float) -> str:
    """``#rrggbb`` → ``rgba(r,g,b,alpha)``, for washes such as interval bands."""
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"
