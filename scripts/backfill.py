#!/usr/bin/env python
"""CLI: load Kaggle historical SPY IV data into the Parquet store."""

import click
import logging

logging.basicConfig(level=logging.INFO)


@click.command()
@click.argument("filepath", type=click.Path(exists=True))
@click.option("--output-dir", default="data/options", show_default=True)
def main(filepath, output_dir):
    """Load and normalise a Kaggle SPY IV dataset file (CSV or Parquet)."""
    from src.data.kaggle_loader import load_kaggle_spy_iv
    from src.data.storage import save_options_chain

    click.echo(f"Loading {filepath} …")
    df = load_kaggle_spy_iv(filepath)
    click.echo(f"Loaded {len(df)} rows.")
    paths = save_options_chain(df, base_dir=output_dir)
    click.echo(f"Saved to {len(paths)} partition(s) in {output_dir}.")


if __name__ == "__main__":
    main()
