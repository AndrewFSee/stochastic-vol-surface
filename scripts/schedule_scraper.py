#!/usr/bin/env python
"""Daily scheduler: full data-collection pipeline (options + VIX + rates).

Uses the ``src.data.scheduler.Scheduler`` module underneath.
Reads ticker list and settings from ``config/default.yaml`` by default;
CLI flags override any config-file values.

Usage:
    python scripts/schedule_scraper.py                # default 4:30 PM ET
    python scripts/schedule_scraper.py --time 17:00   # 5:00 PM ET
    python scripts/schedule_scraper.py --once          # run once immediately then exit
    python scripts/schedule_scraper.py --no-vix        # skip VIX family
    python scripts/schedule_scraper.py --preclose      # thin ETFs only, 15:30-16:00 ET

Runs forever (Ctrl+C to stop).  Designed to be launched via:
    - Task Scheduler (Windows)
    - systemd timer / cron (Linux / macOS)
    - or simply ``nohup python scripts/schedule_scraper.py &``
"""

import click
import logging
from pathlib import Path

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger("scheduler")

# ── Load defaults from config/default.yaml ────────────────────────────────
_CFG_PATH = Path(__file__).resolve().parent.parent / "config" / "default.yaml"


def _load_yaml() -> dict:
    """Read config/default.yaml."""
    try:
        with open(_CFG_PATH) as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        return {}


_yaml_all = _load_yaml()
_yaml_cfg = _yaml_all.get("scraper", {})


@click.command()
@click.option("--tickers", "-t", multiple=True,
              default=_yaml_cfg.get("tickers", ["SPY", "QQQ", "AAPL", "MSFT"]),
              show_default=True)
@click.option("--time", "schedule_time",
              default=_yaml_cfg.get("schedule_time", "16:30"), show_default=True,
              help="Daily trigger time HH:MM (in --timezone)")
@click.option("--timezone",
              default=_yaml_cfg.get("timezone", "America/New_York"),
              show_default=True)
@click.option("--output-dir", default="data/options", show_default=True)
@click.option("--vix/--no-vix", default=True, show_default=True,
              help="Collect VIX family snapshot")
@click.option("--rates/--no-rates", default=True, show_default=True,
              help="Collect FRED rates (needs FRED_API_KEY)")
@click.option("--surfaces/--no-surfaces", default=True, show_default=True,
              help="Build vol surfaces from the chains collected in this run")
@click.option("--prices/--no-prices", default=True, show_default=True,
              help="Refresh daily OHLC for the tickers (realised-vol inputs)")
@click.option("--features/--no-features", default=True, show_default=True,
              help="Rebuild the feature table after the surfaces")
@click.option("--once", is_flag=True, help="Run once immediately then exit")
@click.option("--preclose", is_flag=True,
              help="Collect only the pre-close tickers (run at ~15:45 ET), then exit")
@click.option("--force", is_flag=True,
              help="Collect even before the options close (stores intraday quotes as the close)")
def main(tickers, schedule_time, timezone, output_dir, vix, rates, surfaces,
         prices, features, once, preclose, force):
    """Launch the daily data-collection scheduler (options + VIX + rates + surfaces)."""
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    from src.data.schema import ScraperConfig
    from src.data.scheduler import Scheduler, run_collection

    cfg = ScraperConfig(
        tickers=list(tickers),
        schedule_time=schedule_time,
        timezone=timezone,
        output_dir=output_dir,
        collect_vix=vix,
        collect_rates=rates,
        build_surfaces=surfaces,
        collect_underlying=prices,
        build_features=features,
        backup_dir=(_yaml_all.get("backup") or {}).get("dir"),
        inter_ticker_delay=_yaml_cfg.get("inter_ticker_delay", 1.5),
        preclose_tickers=_yaml_cfg.get("preclose_tickers") or [],
    )

    if preclose:
        from src.data.scheduler import run_preclose

        result = run_preclose(cfg, force=force)
        click.echo(f"Pre-close: {result.total_rows:,} rows -> {result.partitions_written} partitions")
        if result.errors:
            click.echo(f"  Errors: {result.errors}", err=True)
        return

    if once:
        click.echo(f"Running single collection for {cfg.tickers} …")
        result = run_collection(cfg, force=force)
        click.echo(f"  Options:  {result.total_rows:,} rows → {result.partitions_written} partitions")
        if result.vix_snapshot:
            click.echo(f"  VIX snap: {result.vix_snapshot}")
        if result.rates_snapshot:
            click.echo(f"  Rates:    {result.rates_snapshot}")
        if result.errors:
            click.echo(f"  Errors:   {result.errors}", err=True)
        return

    sched = Scheduler(cfg)
    click.echo(
        f"Scheduler started — collecting {cfg.tickers} "
        f"at {schedule_time} {timezone} Mon–Fri.\n"
        f"  VIX: {'ON' if vix else 'OFF'}  |  Rates: {'ON' if rates else 'OFF'}"
        f"  |  Surfaces: {'ON' if surfaces else 'OFF'}"
        f"  |  Prices: {'ON' if prices else 'OFF'}"
        f"  |  Features: {'ON' if features else 'OFF'}\n"
        f"  Ctrl+C to stop."
    )
    sched.start()


if __name__ == "__main__":
    main()
