"""Tests for the incremental data backup (src.data.backup)."""

import os
import time

import pytest

from src.data.backup import BackupError, backup_data


def _write(p, text):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)


@pytest.fixture
def data(tmp_path):
    d = tmp_path / "data"
    _write(d / "options" / "ticker=SPY" / "date=2026-01-05" / "chain.parquet", "chain-1")
    _write(d / "features" / "surface_features.parquet", "feat-v1")
    return d


def test_first_backup_copies_everything_and_verifies(data, tmp_path):
    res = backup_data(data, tmp_path / "bk")
    assert res.files_copied == res.files_total == 2
    assert (tmp_path / "bk" / "data" / "options" / "ticker=SPY" / "date=2026-01-05" / "chain.parquet").read_text() == "chain-1"
    assert res.verified["options"]["backup_files"] == 1
    assert (tmp_path / "bk" / "last_backup.json").exists()


def test_second_backup_copies_only_changes(data, tmp_path):
    backup_data(data, tmp_path / "bk")
    f = data / "features" / "surface_features.parquet"
    f.write_text("feat-v2-longer")
    os.utime(f, (time.time() + 10, time.time() + 10))
    res = backup_data(data, tmp_path / "bk")
    assert res.files_copied == 1
    assert (tmp_path / "bk" / "data" / "features" / "surface_features.parquet").read_text() == "feat-v2-longer"


def test_backup_never_deletes(data, tmp_path):
    backup_data(data, tmp_path / "bk")
    (data / "options" / "ticker=SPY" / "date=2026-01-05" / "chain.parquet").unlink()
    backup_data(data, tmp_path / "bk")
    assert (tmp_path / "bk" / "data" / "options" / "ticker=SPY" / "date=2026-01-05" / "chain.parquet").exists()


def test_missing_drive_raises(data):
    with pytest.raises(BackupError, match="not available"):
        backup_data(data, "Q:/definitely-not-mounted")
