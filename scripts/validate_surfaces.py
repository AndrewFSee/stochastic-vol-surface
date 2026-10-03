#!/usr/bin/env python
"""CLI: check a built surface corpus against independent benchmarks.

For SPY, the 30-day variance-swap vol integrated from our fitted smiles is
compared with VIX, which CBOE computes the same way from SPX options.  For
every ticker, daily-change statistics flag measurement noise: real implied
vol changes are roughly uncorrelated day to day, while noise shows up as a
strongly negative lag-1 autocorrelation (jumps that revert the next day).

Usage:
    python scripts/validate_surfaces.py
    python scripts/validate_surfaces.py -t SPY -t QQQ --surfaces-dir data/surfaces
    python scripts/validate_surfaces.py --strict      # exit 1 if SPY fails
"""

import logging
import sys

import click

logging.basicConfig(level=logging.WARNING, stream=sys.stdout)

#: Pass thresholds for the SPY-vs-VIX check under --strict.
MIN_CHANGE_CORR = 0.80
MAX_MEAN_ABS_DIFF = 1.5     # vol points


@click.command()
@click.option("--tickers", "-t", multiple=True, default=None,
              help="Tickers to check (default: all in the surface store)")
@click.option("--surfaces-dir", default="data/surfaces", show_default=True)
@click.option("--vix-dir", default="data/vix", show_default=True)
@click.option("--start", default=None, help="Start date YYYY-MM-DD")
@click.option("--end", default=None, help="End date YYYY-MM-DD")
@click.option("--strict", is_flag=True,
              help=f"Exit 1 unless SPY 30d vol tracks VIX (change corr >= "
                   f"{MIN_CHANGE_CORR}, mean |diff| <= {MAX_MEAN_ABS_DIFF} pts)")
def main(tickers, surfaces_dir, vix_dir, start, end, strict):
    """Benchmark surfaces against VIX and report noise statistics."""
    import pandas as pd

    from src.surface.batch import available_tickers
    from src.surface.diagnostics import benchmark_against, noise_stats, surface_series

    tickers = list(tickers) or available_tickers(surfaces_dir)
    if not tickers:
        click.echo(f"No surfaces found in {surfaces_dir}", err=True)
        raise SystemExit(1)

    rows = []
    spy = None
    for tkr in tickers:
        s = surface_series(tkr, surfaces_dir=surfaces_dir, start=start, end=end)
        if s.empty:
            continue
        if tkr == "SPY":
            spy = s
        for col in ("atm_1m", "atm_3m", "vs_30d"):
            st = noise_stats(s[col])
            if st["n"]:
                rows.append({"ticker": tkr, "series": col, **st})
        builders = ", ".join(sorted(s["builder"].unique()))
        quality = (f"fit_rmse median {s['fit_rmse'].median():.2f} pts | observed "
                   f"{s['observed_fraction'].median():.0%} | " if s["fit_rmse"].notna().any()
                   else "no fit diagnostics | ")
        rows.append({"ticker": tkr, "series": quality + builders})

    click.echo("\nDaily-change noise (vol points; lag1_autocorr near 0 is healthy):")
    click.echo(pd.DataFrame(rows).round(3).to_string(index=False))

    if spy is None:
        return

    from src.data.vix_family import load_vix_history

    vix = load_vix_history(vix_dir)
    if vix.empty or "VIX" not in vix:
        click.echo("\nNo VIX history — skipping the SPY benchmark.")
        return
    vix_s = vix["VIX"]
    vix_s.index = pd.to_datetime(vix_s.index)

    click.echo("\nSPY vs VIX (vol points):")
    bench = {}
    for col in ("vs_30d", "atm_1m"):
        bench[col] = benchmark_against(spy[col], vix_s)
        click.echo(f"  {col:7s} " + "  ".join(
            f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
            for k, v in bench[col].items()))
    click.echo("  (variance-swap vol is VIX's own definition; ATM sits below it "
               "by the skew premium)")

    if strict:
        b = bench["vs_30d"]
        ok = (b.get("change_corr", 0) >= MIN_CHANGE_CORR
              and b.get("mean_abs_diff", 99) <= MAX_MEAN_ABS_DIFF)
        click.echo(f"\nStrict check: {'PASS' if ok else 'FAIL'}")
        if not ok:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
