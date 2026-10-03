#!/usr/bin/env python
"""CLI: build the historical volatility-surface corpus from stored options chains.

Reads   data/options/ticker={T}/date={D}/chain.parquet
Writes  data/surfaces/ticker={T}/date={D}/surface.parquet

Usage:
    # Build everything that does not exist yet (safe to re-run daily)
    python scripts/build_surfaces.py

    # One ticker, a date range, forcing a rebuild
    python scripts/build_surfaces.py -t SPY --start 2026-06-01 --overwrite

    # Save the per-surface QC report
    python scripts/build_surfaces.py --report data/logs/surface_build.csv
"""

import logging
import sys

import click

logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
# Per-surface stage logging is far too chatty for a corpus sweep; the batch
# layer reports one line per ticker plus a QC report at the end.
for _noisy in ("src.surface.grid_builder", "src.surface.surface",
               "src.surface.filters", "src.data.storage", "src.data.rates"):
    logging.getLogger(_noisy).setLevel(logging.ERROR)


@click.command()
@click.option("--tickers", "-t", multiple=True, default=None,
              help="Tickers to build (default: all in the options store)")
@click.option("--options-dir", default="data/options", show_default=True)
@click.option("--surfaces-dir", default="data/surfaces", show_default=True)
@click.option("--start", default=None, help="Start date YYYY-MM-DD")
@click.option("--end", default=None, help="End date YYYY-MM-DD")
@click.option("--method", default="svi", type=click.Choice(["svi", "rbf"]),
              show_default=True)
@click.option("--overwrite", is_flag=True, help="Rebuild existing surfaces")
@click.option("--report", default=None,
              help="Write the per-surface QC report to this CSV path")
def main(tickers, options_dir, surfaces_dir, start, end, method, overwrite, report):
    """Build volatility surfaces for every stored options chain."""
    from src.surface.batch import build_corpus

    rep = build_corpus(
        tickers=list(tickers) or None,
        options_dir=options_dir,
        surfaces_dir=surfaces_dir,
        method=method,
        overwrite=overwrite,
        start=start,
        end=end,
    )

    click.echo("")
    click.echo(rep.summary())

    df = rep.to_frame()
    if not df.empty:
        bad = df[df.status.isin(["failed", "rejected"])]
        if not bad.empty:
            click.echo(f"\n{len(bad)} problem surfaces:")
            click.echo(
                bad[["ticker", "as_of", "status", "message"]]
                .head(30).to_string(index=False)
            )

    if report and not df.empty:
        from pathlib import Path

        Path(report).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(report, index=False)
        click.echo(f"\nQC report written to {report}")

    # Non-zero exit if nothing at all was produced, so cron/CI notices.
    counts = rep.counts()
    if not counts.get("built") and not counts.get("skipped"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
