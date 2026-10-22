"""Tests for the post-run health checks (src.data.monitor)."""

import json
from datetime import date, datetime, timedelta

import pandas as pd

from src.data import monitor as M


def test_run_check_ignores_market_closed():
    assert M._check_run(["Market closed — skipped"]).ok
    assert not M._check_run(["Options: timeout"]).ok


def test_accuracy_passes_when_tracking_vix():
    dates = pd.bdate_range("2026-01-05", periods=80)
    vix = pd.Series(15 + pd.Series(range(80)).mod(7).to_numpy() * 0.8, index=dates)
    f = pd.DataFrame({"ticker": "SPY", "date": dates, "mkt_vix": vix.to_numpy(),
                      "vs_30d": (vix.to_numpy() + 0.2) / 100})
    assert M._check_accuracy(f).ok


def test_accuracy_fails_when_surfaces_break():
    dates = pd.bdate_range("2026-01-05", periods=80)
    vix = 15 + pd.Series(range(80)).mod(7).to_numpy() * 0.8
    noise = pd.Series(range(80)).mod(5).to_numpy() * 2.0
    f = pd.DataFrame({"ticker": "SPY", "date": dates, "mkt_vix": vix, "vs_30d": (vix + noise) / 100})
    c = M._check_accuracy(f)
    assert not c.ok and "corr" in c.message


def test_backup_check(tmp_path):
    assert M._check_backup(None, datetime.now()) is None
    assert not M._check_backup(str(tmp_path), datetime.now()).ok          # no record
    (tmp_path / "last_backup.json").write_text(json.dumps(
        {"finished_at": (datetime.now() - timedelta(days=5)).isoformat()}))
    assert not M._check_backup(str(tmp_path), datetime.now()).ok          # too old
    (tmp_path / "last_backup.json").write_text(json.dumps({"finished_at": datetime.now().isoformat()}))
    assert M._check_backup(str(tmp_path), datetime.now()).ok


def test_record_and_latest(tmp_path):
    log = str(tmp_path / "h.jsonl")
    M.record([M.Check("run", True, "fine")], date(2026, 1, 5), log)
    M.record([M.Check("run", False, "broke")], date(2026, 1, 6), log)
    last = M.latest_record(log)
    assert last["as_of"] == "2026-01-06" and not last["ok"]


def test_run_checks_never_raises(tmp_path):
    checks = M.run_checks(tickers=["SPY"], options_dir=str(tmp_path / "nope"))
    assert any(c.name == "freshness" and not c.ok for c in checks)


def test_alert_only_on_failure(monkeypatch):
    sent = []
    monkeypatch.setattr(M, "notify", lambda t, b: sent.append((t, b)) or True)
    assert not M.alert_if_failing([M.Check("run", True, "ok")])
    assert M.alert_if_failing([M.Check("backup", False, "drive missing")])
    assert len(sent) == 1 and "backup" in sent[0][1]


def test_disk_check_flags_low_free_space(tmp_path, monkeypatch):
    import shutil
    from collections import namedtuple

    from src.data.monitor import _check_disk

    usage = namedtuple("usage", "total used free")
    monkeypatch.setattr(shutil, "disk_usage", lambda p: usage(100e9, 97e9, 3e9))
    c = _check_disk([str(tmp_path), None])
    assert not c.ok and "3.0 GB free" in c.message
    monkeypatch.setattr(shutil, "disk_usage", lambda p: usage(100e9, 50e9, 50e9))
    assert _check_disk([str(tmp_path)]).ok


def test_best_effort_surface_warns_until_missing_three_sessions(tmp_path):
    """A best-effort ticker's missing surface is a warning; three sessions in
    a row without one is a failure.  Other tickers fail on any miss."""
    from src.data.monitor import _check_surfaces
    from src.surface.batch import BUILDER_VERSION

    opt, srf = tmp_path / "options", tmp_path / "surfaces"
    days = ["2026-10-19", "2026-10-20", "2026-10-21"]

    def chain(t, d):
        (opt / f"ticker={t}" / f"date={d}").mkdir(parents=True, exist_ok=True)
        (opt / f"ticker={t}" / f"date={d}" / "chain.parquet").write_bytes(b"")

    def surface(t, d):
        import pyarrow as pa
        import pyarrow.parquet as pq
        p = srf / f"ticker={t}" / f"date={d}"
        p.mkdir(parents=True, exist_ok=True)
        meta = {b"vol_surface": json.dumps({"builder": BUILDER_VERSION}).encode()}
        tbl = pa.table({"x": [1]}).replace_schema_metadata(meta)
        pq.write_table(tbl, p / "surface.parquet")

    for d in days:
        chain("SPY", d); surface("SPY", d); chain("HYG", d)
    surface("HYG", days[0]); surface("HYG", days[1])           # missing only on the last day

    c = _check_surfaces(str(opt), str(srf), days[-1], best_effort=["HYG"])
    assert c.ok and "HYG (1 session)" in c.message and "1 of 2 surfaces" in c.message
    assert not _check_surfaces(str(opt), str(srf), days[-1]).ok   # not best-effort: fails

    import shutil
    shutil.rmtree(srf / "ticker=HYG")                         # missing all three sessions
    c = _check_surfaces(str(opt), str(srf), days[-1], best_effort=["HYG"])
    assert not c.ok and "no surface for 2026-10-21: HYG" in c.message
