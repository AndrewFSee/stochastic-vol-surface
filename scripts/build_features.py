#!/usr/bin/env python
"""CLI: build the point-in-time feature table from the surface store.

Reads   data/surfaces, data/underlying/prices.parquet, data/vix
Writes  data/features/surface_features.parquet  (one row per ticker x date)

The table is rebuilt in full every time, so it is deterministic and always
consistent with the current surfaces.

Usage:
    python scripts/build_features.py
    python scripts/build_features.py -t SPY -t QQQ --out data/features/test.parquet
    python scripts/build_features.py --surfaces-dir data/historical/surfaces \
        --out data/historical/features/surface_features.parquet
"""

import logging
from pathlib import Path
import sys

import click

logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)


@click.command()
@click.option("--tickers", "-t", multiple=True, default=None,
              help="Tickers to include (default: all in the surface store)")
@click.option("--surfaces-dir", default="data/surfaces", show_default=True)
@click.option("--underlying-dir", default="data/underlying", show_default=True)
@click.option("--vix-dir", default="data/vix", show_default=True)
@click.option("--macro-dir", default="data/macro", show_default=True)
@click.option("--events-dir", default="data/events", show_default=True)
@click.option("--no-cache", is_flag=True, help="Recompute every surface's features from scratch")
@click.option("--out", default="data/features/surface_features.parquet", show_default=True)
@click.option("--start", default=None, help="Start date YYYY-MM-DD")
@click.option("--end", default=None, help="End date YYYY-MM-DD")
def main(tickers, surfaces_dir, underlying_dir, vix_dir, macro_dir, events_dir, no_cache, out,
         start, end):
    """Assemble and save the feature table."""
    from src.features.table import build_feature_table, save_feature_table

    df = build_feature_table(
        list(tickers) or None,
        surfaces_dir=surfaces_dir, underlying_dir=underlying_dir,
        vix_dir=vix_dir, macro_dir=macro_dir, events_dir=events_dir, start=start, end=end,
        cache_path=None if no_cache else str(Path(out).parent / "_surface_rows_cache.parquet"),
    )
    if df.empty:
        click.echo(f"No surfaces with stored fits in {surfaces_dir}.", err=True)
        raise SystemExit(1)

    save_feature_table(df, out)

    summary = df.groupby("ticker").agg(
        rows=("date", "size"), first=("date", "min"), last=("date", "max"),
        atm_30d=("atm_30d", lambda s: s.notna().mean()),
        rr25_30d=("rr25_30d", lambda s: s.notna().mean()),
        vrp_30d=("vrp_30d", lambda s: s.notna().mean()),
    )
    click.echo(f"\n{len(df)} rows x {df.shape[1]} columns -> {out}")
    click.echo("Coverage (non-null fraction) of key features:")
    click.echo(summary.to_string())


if __name__ == "__main__":
    main()
