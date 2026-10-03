#!/usr/bin/env python
"""CLI: report on the health of the collected data corpus.

Checks freshness (is the scraper still running?), per-ticker coverage and
gaps, snapshot quality, surface-build status, and the VIX / rates stores.

Usage:
    python scripts/data_health.py
    python scripts/data_health.py --csv-dir data/logs/health
    python scripts/data_health.py --strict      # exit 1 if data is stale
"""

import logging
import sys

import click

logging.basicConfig(level=logging.WARNING, stream=sys.stdout)


@click.command()
@click.option("--options-dir", default="data/options", show_default=True)
@click.option("--vix-dir", default="data/vix", show_default=True)
@click.option("--rates-dir", default="data/rates", show_default=True)
@click.option("--surfaces-dir", default="data/surfaces", show_default=True)
@click.option("--csv-dir", default=None,
              help="Also write coverage/quality/surface tables as CSVs here")
@click.option("--strict", is_flag=True,
              help="Exit non-zero if data is stale or a store is missing")
@click.option("--max-stale-days", default=1, show_default=True,
              help="Trading days behind before --strict fails")
def main(options_dir, vix_dir, rates_dir, surfaces_dir, csv_dir, strict, max_stale_days):
    """Print a data-health report for the collected corpus."""
    from src.data.health import full_report

    rep = full_report(
        options_dir=options_dir,
        vix_dir=vix_dir,
        rates_dir=rates_dir,
        surfaces_dir=surfaces_dir,
    )
    click.echo(rep.render())

    if csv_dir:
        from pathlib import Path

        out = Path(csv_dir)
        out.mkdir(parents=True, exist_ok=True)
        rep.coverage.to_csv(out / "coverage.csv", index=False)
        rep.quality.to_csv(out / "quality.csv", index=False)
        rep.surfaces.to_csv(out / "surfaces.csv", index=False)
        click.echo(f"CSV tables written to {out}")

    if strict:
        problems = []
        stale = rep.freshness.get("trading_days_stale")
        if stale is None:
            problems.append("no options data found")
        elif stale > max_stale_days:
            problems.append(f"data is {stale} trading days stale")
        if not rep.rates.get("n_dates"):
            problems.append("rate history missing")
        if not rep.vix.get("n_dates"):
            problems.append("VIX history missing")

        if problems:
            click.echo("\nFAILED: " + "; ".join(problems), err=True)
            raise SystemExit(1)
        click.echo("\nAll strict checks passed.")


if __name__ == "__main__":
    main()
