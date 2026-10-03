#!/usr/bin/env python
"""CLI: backfill historical SPY option chains from Kaggle into a Parquet store.

Historical data is written to a **separate store** (``data/historical/options``
by default) rather than mixed into the live yfinance corpus.  The two differ in
source, quote timing, and IV convention, so keeping them apart preserves
provenance — and keeps the daily freshness check in ``scripts/data_health.py``
meaningful, which a 2010-dated partition would otherwise break.

Usage:
    # List what a dataset contains without downloading it
    python scripts/backfill_kaggle.py --list

    # Download every year and load it
    python scripts/backfill_kaggle.py --download

    # Just a few years
    python scripts/backfill_kaggle.py --download --years 2018 2020 2022

    # Load files already sitting in data/kaggle/
    python scripts/backfill_kaggle.py

Then build surfaces over the historical store:
    python scripts/build_surfaces.py \
        --options-dir data/historical/options \
        --surfaces-dir data/historical/surfaces
"""

import logging
import re
import sys
from pathlib import Path

import click

logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logging.getLogger("src.data.storage").setLevel(logging.WARNING)
logger = logging.getLogger("backfill_kaggle")

# Year-partitioned SPY EOD chains with IV and Greeks, 2010-2023.
DEFAULT_KAGGLE_DATASET = "dudesurfin/spy-options-eod-volatility-surface-2010-2023"
DEFAULT_DOWNLOAD_DIR = "data/kaggle"
DEFAULT_OUTPUT_DIR = "data/historical/options"


def _api():
    """Return an authenticated Kaggle API client."""
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        click.echo(
            "The `kaggle` package is not installed.  pip install kaggle\n"
            "Then set KAGGLE_USERNAME and KAGGLE_KEY (e.g. in .env).",
            err=True,
        )
        raise SystemExit(1)

    api = KaggleApi()
    try:
        api.authenticate()
    except Exception as exc:
        click.echo(f"Kaggle authentication failed: {exc}", err=True)
        raise SystemExit(1)
    return api


def _year_of(name: str):
    """Extract a 4-digit year from a filename, if present."""
    m = re.search(r"(19|20)\d{2}", str(name))
    return int(m.group(0)) if m else None


@click.command()
@click.argument("filepath", required=False, type=click.Path())
@click.option("--download", is_flag=True, help="Download from Kaggle before loading")
@click.option("--list", "list_only", is_flag=True, help="List dataset files and exit")
@click.option("--dataset", default=DEFAULT_KAGGLE_DATASET, show_default=True)
@click.option("--download-dir", default=DEFAULT_DOWNLOAD_DIR, show_default=True)
@click.option("--output-dir", default=DEFAULT_OUTPUT_DIR, show_default=True)
@click.option("--years", multiple=True, type=int,
              help="Restrict to these years (repeatable)")
@click.option("--ticker", default="SPY", show_default=True,
              help="Ticker label to store the data under")
def main(filepath, download, list_only, dataset, download_dir, output_dir, years, ticker):
    """Normalise historical Kaggle option chains into the Parquet store."""
    from src.data.kaggle_loader import load_kaggle_spy_iv
    from src.data.storage import save_options_chain

    dl_dir = Path(download_dir)

    # ── List mode ─────────────────────────────────────────────────────────
    if list_only:
        api = _api()
        files = api.dataset_list_files(dataset).files
        click.echo(f"{dataset}: {len(files)} file(s)")
        for f in files:
            click.echo(f"  {f.name}")
        return

    # ── Acquire files ─────────────────────────────────────────────────────
    if filepath:
        targets = [Path(filepath)]
    else:
        if download:
            api = _api()
            dl_dir.mkdir(parents=True, exist_ok=True)
            names = [f.name for f in api.dataset_list_files(dataset).files]
            names = [n for n in names if n.endswith((".csv", ".parquet", ".pq"))]
            if years:
                names = [n for n in names if _year_of(n) in set(years)]
            if not names:
                click.echo("No matching files in the dataset.", err=True)
                raise SystemExit(1)

            for n in names:
                if (dl_dir / n).exists():
                    click.echo(f"  already have {n}")
                    continue
                click.echo(f"  downloading {n} …")
                api.dataset_download_file(dataset, n, path=str(dl_dir), force=False)

        if not dl_dir.exists():
            click.echo(f"{dl_dir} does not exist — use --download.", err=True)
            raise SystemExit(1)

        targets = sorted(
            p for p in dl_dir.iterdir()
            if p.suffix in (".csv", ".parquet", ".pq")
        )
        if years:
            targets = [p for p in targets if _year_of(p.name) in set(years)]

    if not targets:
        click.echo("Nothing to load.", err=True)
        raise SystemExit(1)

    # ── Load file by file (a full decade will not fit in memory at once) ──
    click.echo(f"\nLoading {len(targets)} file(s) into {output_dir}/ as {ticker}\n")
    total_rows = 0
    total_parts = 0
    failures = []

    for path in targets:
        try:
            df = load_kaggle_spy_iv(path)
        except Exception as exc:
            logger.error("Failed to load %s: %s", path.name, exc)
            failures.append((path.name, str(exc)))
            continue

        if df.empty:
            failures.append((path.name, "no usable rows"))
            continue

        df["ticker"] = ticker
        paths = save_options_chain(df, base_dir=output_dir)
        total_rows += len(df)
        total_parts += len(paths)
        click.echo(
            f"  {path.name:<28} {len(df):>10,} rows  "
            f"{df['as_of'].min().date()} -> {df['as_of'].max().date()}  "
            f"{len(paths):>4} partitions"
        )

    click.echo(f"\nBackfilled {total_rows:,} rows into {total_parts} partitions "
               f"at {output_dir}/")
    if failures:
        click.echo(f"\n{len(failures)} file(s) failed:", err=True)
        for name, err in failures:
            click.echo(f"  {name}: {err}", err=True)

    click.echo(
        "\nNext: build surfaces over the historical store\n"
        f"  python scripts/build_surfaces.py "
        f"--options-dir {output_dir} "
        f"--surfaces-dir {Path(output_dir).parent / 'surfaces'}"
    )


if __name__ == "__main__":
    main()
