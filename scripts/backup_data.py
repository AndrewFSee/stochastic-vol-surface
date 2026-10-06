#!/usr/bin/env python
"""CLI: incremental backup of data/ to the backup drive (also run daily by the job).

Copies new or changed files to <dir>/data, never deletes from the backup, and
verifies data/options (the irreplaceable chains) afterwards.

Usage:
    python scripts/backup_data.py                       # dir from config/default.yaml
    python scripts/backup_data.py --dir E:/other-backup
"""

import json
import logging
import sys
from pathlib import Path

import click
import yaml

logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s")


def _default_dir():
    cfg = yaml.safe_load(open(Path(__file__).resolve().parents[1] / "config" / "default.yaml"))
    return (cfg.get("backup") or {}).get("dir")


@click.command()
@click.option("--dir", "backup_dir", default=_default_dir, show_default="from config")
@click.option("--data-dir", default="data", show_default=True)
def main(backup_dir, data_dir):
    """Back up the data folder."""
    from src.data.backup import BackupError, backup_data

    if not backup_dir:
        raise SystemExit("No backup dir: set backup.dir in config/default.yaml or pass --dir")
    try:
        res = backup_data(data_dir, backup_dir)
    except BackupError as exc:
        raise SystemExit(f"Backup failed: {exc}")
    click.echo(f"{res.files_copied:,} of {res.files_total:,} files copied "
               f"({res.bytes_copied / 1e6:,.1f} MB) in {res.seconds:.0f}s -> {res.target}")
    click.echo("verified: " + json.dumps(res.verified))


if __name__ == "__main__":
    main()
