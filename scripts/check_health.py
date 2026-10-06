#!/usr/bin/env python
"""CLI: run the post-run health checks now (the daily job runs them automatically).

Checks every stage produced the latest session's output, that SPY's surfaces
still agree with VIX, and that the backup is recent.  See src/data/monitor.py.

Usage:
    python scripts/check_health.py            # print results; exit 1 on failure
    python scripts/check_health.py --notify   # also raise a desktop alert on failure
    python scripts/check_health.py --test-notification
"""

import logging
import sys
import warnings

import click
import yaml

logging.basicConfig(level=logging.WARNING, stream=sys.stdout)


@click.command()
@click.option("--notify", "do_notify", is_flag=True, help="Desktop alert if any check fails")
@click.option("--test-notification", is_flag=True, help="Send a test alert and exit")
@click.option("--max-stale", default=1, show_default=True,
              help="Sessions the newest chain may lag (1 allows a run before tonight's scrape)")
def main(do_notify, test_notification, max_stale):
    """Run the health checks."""
    from src.data.monitor import alert_if_failing, notify, run_checks

    if test_notification:
        ok = notify("Vol surface: test notification", "Alerts are working. No action needed.")
        raise SystemExit(0 if ok else 1)
    warnings.filterwarnings("ignore")
    cfg = yaml.safe_load(open("config/default.yaml"))
    checks = run_checks(tickers=cfg["scraper"]["tickers"],
                        backup_dir=(cfg.get("backup") or {}).get("dir"), max_stale=max_stale)
    for c in checks:
        click.echo(f"{'OK  ' if c.ok else 'FAIL'}  {c.name:10s} {c.message}")
    failed = [c for c in checks if not c.ok]
    if do_notify:
        alert_if_failing(checks)
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
