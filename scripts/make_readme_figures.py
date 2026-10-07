#!/usr/bin/env python
"""CLI: render the README figures from the stores, in light and dark versions.

Writes docs/images/<name>-light.png and <name>-dark.png, which the README
shows through <picture> so each matches the reader's GitHub theme.  Colours
come from the dashboard's validated palette (src/dashboard/theme.py).

Usage:
    python scripts/make_readme_figures.py            # every figure
    python scripts/make_readme_figures.py -f surface  # one of: surface, vix,
                                                      # forecast, models, cross
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import click
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from src.dashboard.theme import DARK, LIGHT  # noqa: E402

OUT = ROOT / "docs" / "images"
DPI = 200


def _style(t) -> None:
    plt.rcParams.update({
        "font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 13,
        "figure.facecolor": t.surface, "axes.facecolor": t.surface, "savefig.facecolor": t.surface,
        "text.color": t.ink, "axes.labelcolor": t.ink_secondary, "axes.edgecolor": t.axis,
        "xtick.color": t.ink_muted, "ytick.color": t.ink_muted,
        "axes.grid": True, "grid.color": t.grid, "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "legend.labelcolor": t.ink_secondary,
        "axes.titlesize": 15, "axes.titleweight": "semibold", "axes.titlecolor": t.ink,
        "axes.titlelocation": "left", "axes.titlepad": 10,
    })


def _title(fig, t, title: str, subtitle: str) -> None:
    fig.text(0.012, 0.975, title, fontsize=20, fontweight="bold", color=t.ink, va="top")
    fig.text(0.012, 0.905, subtitle, fontsize=13.5, color=t.ink_secondary, va="top", wrap=True)


def _save(fig, name: str, mode: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}-{mode}.png"
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    print("wrote", path.relative_to(ROOT))


def _features(path: str) -> pd.DataFrame:
    return pd.read_parquet(ROOT / path)


# ── 1. The surface ──────────────────────────────────────────────────────────

def fig_surface(t) -> None:
    from src.surface.slices import ExpirySurface
    from src.surface.surface import VolSurface

    d = sorted((ROOT / "data/surfaces/ticker=SPY").glob("date=*"))[-1]
    es = ExpirySurface(VolSurface.load(d / "surface.parquet").slices)
    # Delta space, as vol desks quote it: 10-delta put on the left, 10-delta
    # call on the right.  Every tenor quotes this whole range, so nothing is
    # extrapolated (cells outside a tenor's quotes would be left blank).
    call_delta = np.linspace(0.90, 0.10, 61)
    days = np.unique(np.round(np.geomspace(7, 730, 40)))
    days = days[[es.brackets(x / 365) for x in days]]
    Z = np.full((len(days), len(call_delta)), np.nan)
    for i, dd in enumerate(days):
        T = dd / 365
        ks = np.array([es.delta_strike(c, T) for c in call_delta])
        Z[i] = np.where(es.is_observed(ks, T), 100 * es.iv(ks, T), np.nan)
    X, Y = np.meshgrid(1 - call_delta, np.log(days))

    cmap = LinearSegmentedColormap.from_list("vol", ("#cde2fb", "#6da7ec", "#256abf", "#104281"))
    lo, hi = np.nanmin(Z), np.nanmax(Z)
    fig = plt.figure(figsize=(11, 7))
    ax = fig.add_axes([-0.12, -0.03, 1.24, 1.0], projection="3d")
    ax.set_facecolor(t.surface)
    floor = np.floor(lo) - 1
    ax.plot_surface(X, Y, np.ma.masked_invalid(Z), cmap=cmap, vmin=lo, vmax=hi,
                    linewidth=0.25, edgecolor=t.surface, antialiased=True,
                    rstride=1, cstride=1, alpha=0.97)
    ax.set_zlim(floor, hi + 0.5)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color((0, 0, 0, 0))
        axis._axinfo["grid"]["color"] = t.grid
        axis._axinfo["grid"]["linewidth"] = 0.6
        axis.label.set_color(t.ink_secondary)
    ax.set_xticks([0.1, 0.25, 0.5, 0.75, 0.9], ["10Δ put", "25Δ put", "ATM", "25Δ call", "10Δ call"])
    ticks = [d_ for d_ in (7, 30, 91, 182, 365, 730) if days[0] <= d_ <= days[-1]]
    ax.set_yticks(np.log(ticks), [f"{d_}d" if d_ < 365 else f"{d_ // 365}y" for d_ in ticks])
    ax.set_zlabel("Implied vol (%)", labelpad=8)
    ax.set_ylabel("Tenor", labelpad=10)
    ax.tick_params(colors=t.ink_muted, labelsize=11.5)
    ax.view_init(elev=22, azim=-62)
    ax.set_box_aspect((1.3, 1.1, 0.72), zoom=1.0)
    atm30, p25, c25 = (100 * float(es.iv(es.delta_strike(x, 30 / 365), 30 / 365)) for x in (0.5, -0.25, 0.25))
    _title(fig, t, f"SPY implied-volatility surface · {d.name[5:]}",
           f"Per-expiry SVI fits, 7 days to {int(days[-1]) // 365} years. 30-day ATM {atm30:.1f}%; "
           f"25Δ put {p25:.1f}% vs call {c25:.1f}%: the equity skew")
    _save(fig, "surface", t.mode)


# ── 2. Validation against VIX ───────────────────────────────────────────────

def fig_vix(t) -> None:
    from src.surface.diagnostics import benchmark_against

    f = pd.concat([_features("data/historical/features/surface_features.parquet"),
                   _features("data/features/surface_features.parquet")])
    s = f[f.ticker == "SPY"].drop_duplicates("date", keep="last").set_index("date").sort_index()
    ours, vix = 100 * s["vs_30d"], s["mkt_vix"]
    b = benchmark_against(ours, vix)
    gap = ours.index.to_series().diff() > pd.Timedelta(days=10)     # don't bridge data gaps
    o_plot = ours.where(~gap)

    fig = plt.figure(figsize=(12, 5.8))
    gs = fig.add_gridspec(1, 3, width_ratios=[2.6, 1, 1], left=0.055, right=0.99,
                          bottom=0.12, top=0.76, wspace=0.2)
    main = fig.add_subplot(gs[0])
    main.plot(vix.index, vix, color=t.series[1], lw=1.0, label="VIX (CBOE)")
    main.plot(o_plot.index, o_plot, color=t.series[0], lw=1.0, label="Our SPY 30d variance swap")
    main.set_ylabel("Volatility (%)")
    main.set_title("2010–2026")
    main.legend(loc="upper right")
    for i, (a, z, ttl) in enumerate([("2020-02-10", "2020-05-15", "March 2020 crash"),
                                     ("2025-03-15", "2025-05-20", "April 2025 sell-off")]):
        ax = fig.add_subplot(gs[i + 1], sharey=None)
        ax.plot(vix.loc[a:z], color=t.series[1], lw=2.0)
        ax.plot(ours.loc[a:z], color=t.series[0], lw=2.0)
        ax.set_title(ttl)
        ax.tick_params(axis="x", rotation=30, labelsize=11)
        ax.xaxis.set_major_locator(matplotlib.dates.AutoDateLocator(minticks=3, maxticks=5))
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b %d"))
    _title(fig, t, "Validated against VIX",
           f"Rebuilt from our SPY smiles, independently of CBOE: correlation "
           f"{b['level_corr']:.3f}, mean gap {b['mean_abs_diff']:.2f} vol pts over "
           f"{int(b['n']):,} days")
    _save(fig, "vix_validation", t.mode)


# ── 3. Forecast track record ────────────────────────────────────────────────

def fig_forecast(t) -> None:
    from src.dashboard.data import window_end

    fc = pd.read_parquet(ROOT / "data/forecasts/vol_forecasts.parquet")
    z = fc[(fc.ticker == "SPY") & (fc.horizon == 21) & (fc.date >= "2024-06-01")].sort_values("date")
    done = z.dropna(subset=["realised_vol"])
    x = window_end(pd.DatetimeIndex(done["date"]), 21)
    gap = pd.Series(x).diff() > pd.Timedelta(days=10)

    def br(v):
        return np.where(gap.to_numpy(), np.nan, 100 * v.to_numpy())

    fig, ax = plt.subplots(figsize=(12, 5.6))
    fig.subplots_adjust(left=0.06, right=0.99, bottom=0.08, top=0.72)
    ax.fill_between(x, br(done["lo80_vol"]), br(done["hi80_vol"]), color=t.series[0],
                    alpha=0.16, lw=0, label="80% forecast range")
    ax.plot(x, br(done["implied_vol"]), color=t.series[1], lw=1.2, label="Implied vol at the time")
    ax.plot(x, br(done["forecast_vol"]), color=t.series[0], lw=1.6, label="Model forecast")
    ax.plot(x, br(done["realised_vol"]), color=t.series[2], lw=1.6, label="Realised")
    ax.set_ylabel("21-day volatility (%)")
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=4, fontsize=11.5)
    inside = ((done.realised_vol >= done.lo80_vol) & (done.realised_vol <= done.hi80_vol)).mean()
    err_m = 100 * np.sqrt(((done.forecast_vol - done.realised_vol) ** 2).mean())
    err_i = 100 * np.sqrt(((done.implied_vol - done.realised_vol) ** 2).mean())
    _title(fig, t, "Out-of-sample volatility forecasts · SPY, 21 trading days",
           f"Walk-forward HAR + implied vol, plotted when each window closed. RMSE "
           f"{err_m:.1f} vs {err_i:.1f} pts for raw implied; the 80% range held {100 * inside:.0f}% of outcomes")
    _save(fig, "forecast", t.mode)


# ── 4. Model comparison ─────────────────────────────────────────────────────

def _qlike_table(md: str, horizon: str) -> pd.Series:
    block = md.split(f"## {horizon}-day horizon")[1].split("QLIKE by VIX regime")[0]
    rows = [r.split("|")[1:3] for r in block.splitlines() if r.startswith("| ") and "qlike" not in r]
    return pd.Series({m.strip(): float(q) for m, q in rows})


def fig_models(t) -> None:
    md = (ROOT / "docs/forecast_evaluation.md").read_text(encoding="utf-8")
    period = re.search(r"Walk-forward, (\d{4})-\d\d-\d\d to (\d{4})", md).groups()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.8))
    fig.subplots_adjust(left=0.18, right=0.98, bottom=0.11, top=0.76, wspace=0.62)
    for ax, h in zip(axes, ("5", "21")):
        q = _qlike_table(md, h).sort_values(ascending=False)
        colors = [t.series[0] if m == "HAR + implied" else t.ink_muted for m in q.index]
        ax.barh(q.index, q.values, color=colors, height=0.62)
        for y, v in enumerate(q.values):
            ax.text(v + 0.004, y, f"{v:.3f}", va="center", fontsize=11, color=t.ink_secondary)
        ax.set_title(f"{h}-day horizon")
        ax.set_xlabel("QLIKE loss (lower is better)")
        ax.set_xlim(0.3, q.max() * 1.1)
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", labelsize=11.5, colors=t.ink_secondary)
    _title(fig, t, "Ten models, one honest backtest",
           f"SPY {period[0]}–{period[1]}, walk-forward, no look-ahead. Production model in blue: "
           "beats HAR at 5 days (p < 0.001)")
    _save(fig, "models", t.mode)


# ── 5. Cross-asset snapshot ─────────────────────────────────────────────────

def fig_cross(t) -> None:
    from src.data.universe import GROUPS, group_of

    f = _features("data/features/surface_features.parquet")
    day = f["date"].max()
    s = f[f.date == day].set_index("ticker")
    order = [tk for ts in GROUPS.values() for tk in ts if tk in s.index]
    s = s.loc[order]

    fig, ax = plt.subplots(figsize=(12, 6.2))
    fig.subplots_adjust(left=0.06, right=0.99, bottom=0.15, top=0.72)
    x = np.arange(len(order))
    iv, rv = 100 * s["atm_30d"], 100 * s["rv_cc_21d"]
    ax.vlines(x, np.minimum(iv, rv), np.maximum(iv, rv), color=t.axis, lw=1.4, zorder=1)
    ax.scatter(x, rv, s=34, facecolor=t.surface, edgecolor=t.series[1], lw=1.6, zorder=2,
               label="Realised vol, last 21 days")
    ax.scatter(x, iv, s=38, color=t.series[0], zorder=3, label="30-day implied vol (ATM)")
    ax.set_xticks(x, order, rotation=90, fontsize=10.5)
    ax.set_xlim(-0.8, len(order) - 0.2)
    ax.set_ylabel("Volatility (%)")
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2, fontsize=11.5)
    ax.grid(axis="x", visible=False)
    ymax = np.nanmax([iv.max(), rv.max()])
    start = 0
    for g, ts in GROUPS.items():
        n = sum(tk in s.index for tk in ts)
        if not n:
            continue
        if start:
            ax.axvline(start - 0.5, color=t.grid, lw=1.2, zorder=0)
        short = {"US equity indices": "Indices", "Rates and credit": "Rates & credit",
                 "International": "Intl", "Crypto and VIX": "Crypto, VIX"}.get(g, g)
        ax.text(start + (n - 1) / 2, ymax * 1.06, short, ha="center", fontsize=10.5,
                color=t.ink_secondary)
        start += n
    ax.set_ylim(0, ymax * 1.14)
    _title(fig, t, f"{len(order)} underlyings, one daily pipeline · {day:%Y-%m-%d}",
           "Implied above realised is the volatility risk premium. Credit and Treasuries sit "
           "low; crypto and single stocks high")
    _save(fig, "cross_asset", t.mode)


FIGURES = {"surface": fig_surface, "vix": fig_vix, "forecast": fig_forecast,
           "models": fig_models, "cross": fig_cross}


@click.command()
@click.option("--figure", "-f", "names", multiple=True, type=click.Choice(list(FIGURES)))
def main(names):
    """Render the README figures."""
    for name in names or FIGURES:
        for t in (LIGHT, DARK):
            _style(t)
            FIGURES[name](t)


if __name__ == "__main__":
    main()
