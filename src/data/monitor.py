"""Post-run health checks and alerts.

After each daily run, :func:`run_checks` verifies that every stage produced
today's output and that the surfaces still agree with VIX.  Results are
appended to ``data/logs/health_checks.jsonl`` (the dashboard reads the latest
line), and any failure raises a Windows desktop notification.  Without this,
a broken run is only visible in a log nobody reads.

Checks
------
run          the collection steps reported no errors
freshness    every configured ticker has the latest session's chain
surfaces     every chain from the latest session has a surface
features     the feature table covers the latest session
forecasts    the forecasts cover the latest feature date
accuracy     SPY 30d variance swap vs VIX over the last 63 sessions:
             daily-change corr >= 0.8, mean |diff| <= 1.5 pts,
             latest |diff| <= 3 pts
backup       the last backup finished within 2 days
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_LOG = "data/logs/health_checks.jsonl"
MIN_CHANGE_CORR = 0.80
MAX_MEAN_ABS_DIFF = 1.5
MAX_LATEST_DIFF = 3.0
ACCURACY_WINDOW = 63
MAX_BACKUP_AGE = timedelta(days=2)


@dataclass
class Check:
    name: str
    ok: bool
    message: str


def _check_run(run_errors: Iterable[str]) -> Check:
    errs = [e for e in run_errors if "Market closed" not in e]
    return Check("run", not errs, "; ".join(errs) if errs else "no errors")


def _check_freshness(options_dir: str, tickers: Sequence[str], as_of: date, max_stale: int) -> Check:
    from src.data.health import freshness

    f = freshness(options_dir, as_of)
    if f["latest"] is None:
        return Check("freshness", False, "no option chains found")
    per = f["per_ticker"]
    missing = [t for t in tickers if t not in per]
    lagging = [t for t in tickers if t in per and per[t] < f["latest"].isoformat()]
    problems = []
    if f["trading_days_stale"] > max_stale:
        problems.append(f"newest chain is {f['trading_days_stale']} session(s) old ({f['latest']})")
    if missing:
        problems.append(f"no chains at all for {', '.join(missing)}")
    if lagging:
        problems.append(f"behind the latest session: {', '.join(lagging)}")
    return Check("freshness", not problems,
                 "; ".join(problems) or f"all {len(per)} tickers current ({f['latest']})")


def _latest_options_date(options_dir: str) -> Optional[str]:
    from src.data.health import freshness

    f = freshness(options_dir)
    return f["latest"].isoformat() if f["latest"] else None


def _check_surfaces(options_dir: str, surfaces_dir: str, latest: str) -> Check:
    from src.surface.batch import BUILDER_VERSION, stored_builder, surface_path

    tickers = [p.name.split("=", 1)[1] for p in Path(options_dir).glob("ticker=*")
               if (p / f"date={latest}" / "chain.parquet").exists()]
    missing = [t for t in tickers
               if stored_builder(surface_path(t, latest, surfaces_dir)) != BUILDER_VERSION]
    return Check("surfaces", not missing,
                 f"no surface for {latest}: {', '.join(sorted(missing))}" if missing
                 else f"{len(tickers)} surfaces built for {latest}")


def _check_features(features_path: str, latest: str):
    if not Path(features_path).exists():
        return Check("features", False, f"missing {features_path}"), None
    f = pd.read_parquet(features_path, columns=["ticker", "date", "vs_30d", "mkt_vix"])
    newest = f["date"].max().date().isoformat()
    ok = newest >= latest
    return Check("features", ok, f"feature table ends {newest}" + ("" if ok else f", chains go to {latest}")), f


def _check_forecasts(forecasts_path: str, features: Optional[pd.DataFrame]) -> Check:
    if not Path(forecasts_path).exists():
        return Check("forecasts", False, f"missing {forecasts_path}")
    fc = pd.read_parquet(forecasts_path, columns=["date"])
    newest = fc["date"].max()
    want = features["date"].max() if features is not None else newest
    return Check("forecasts", newest >= want,
                 f"forecasts end {newest.date()}" + ("" if newest >= want else f", features go to {want.date()}"))


def _check_accuracy(features: Optional[pd.DataFrame]) -> Check:
    from src.surface.diagnostics import benchmark_against

    if features is None:
        return Check("accuracy", False, "no feature table")
    spy = features[features["ticker"] == "SPY"].sort_values("date").set_index("date").tail(ACCURACY_WINDOW)
    spy = spy.dropna(subset=["vs_30d", "mkt_vix"])
    if len(spy) < 20:
        return Check("accuracy", True, f"only {len(spy)} SPY days with VIX; check skipped")
    b = benchmark_against(100 * spy["vs_30d"], spy["mkt_vix"])
    latest_gap = abs(100 * spy["vs_30d"].iloc[-1] - spy["mkt_vix"].iloc[-1])
    problems = []
    if b["change_corr"] < MIN_CHANGE_CORR:
        problems.append(f"daily-change corr {b['change_corr']:.2f} < {MIN_CHANGE_CORR}")
    if b["mean_abs_diff"] > MAX_MEAN_ABS_DIFF:
        problems.append(f"mean |diff| {b['mean_abs_diff']:.2f} pts > {MAX_MEAN_ABS_DIFF}")
    if latest_gap > MAX_LATEST_DIFF:
        problems.append(f"latest gap {latest_gap:.2f} pts > {MAX_LATEST_DIFF}")
    msg = (f"SPY 30d VS vs VIX, last {len(spy)} days: corr {b['change_corr']:.3f}, "
           f"mean |diff| {b['mean_abs_diff']:.2f} pts, latest {latest_gap:.2f} pts")
    return Check("accuracy", not problems, "; ".join(problems) + " — " + msg if problems else msg)


def _check_backup(backup_dir: Optional[str], now: datetime) -> Optional[Check]:
    if not backup_dir:
        return None
    p = Path(backup_dir) / "last_backup.json"
    if not p.exists():
        return Check("backup", False, f"no backup record at {p} (drive unplugged?)")
    try:
        done = datetime.fromisoformat(json.loads(p.read_text())["finished_at"])
    except (ValueError, KeyError):
        return Check("backup", False, f"unreadable backup record {p}")
    age = now - done
    return Check("backup", age <= MAX_BACKUP_AGE,
                 f"last backup {done:%Y-%m-%d %H:%M}" + ("" if age <= MAX_BACKUP_AGE
                                                         else f", {age.days} days ago"))


def run_checks(
    *,
    tickers: Sequence[str],
    run_errors: Iterable[str] = (),
    as_of: Optional[date] = None,
    options_dir: str = "data/options",
    surfaces_dir: str = "data/surfaces",
    features_path: str = "data/features/surface_features.parquet",
    forecasts_path: str = "data/forecasts/vol_forecasts.parquet",
    backup_dir: Optional[str] = None,
    max_stale: int = 0,
) -> list[Check]:
    """Run every check; never raises (a crashing check becomes a failed check)."""
    as_of = as_of or date.today()
    out: list[Check] = []

    def guard(name, fn, *a):
        try:
            return fn(*a)
        except Exception as exc:  # a broken check must still be reported
            return Check(name, False, f"check crashed: {type(exc).__name__}: {exc}")

    out.append(_check_run(run_errors))
    out.append(guard("freshness", _check_freshness, options_dir, tickers, as_of, max_stale))
    latest = guard("surfaces", _latest_options_date, options_dir)
    if isinstance(latest, str):
        out.append(guard("surfaces", _check_surfaces, options_dir, surfaces_dir, latest))
        res = guard("features", _check_features, features_path, latest)
        feat_check, features = res if isinstance(res, tuple) else (res, None)
        out.append(feat_check)
        out.append(guard("forecasts", _check_forecasts, forecasts_path, features))
        out.append(guard("accuracy", _check_accuracy, features))
    b = guard("backup", _check_backup, backup_dir, datetime.now())
    if b is not None:
        out.append(b)
    return out


def record(checks: Sequence[Check], as_of: date, path: str = DEFAULT_LOG) -> None:
    """Append one JSON line with every check's result."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    line = {"as_of": as_of.isoformat(), "checked_at": datetime.now().isoformat(timespec="seconds"),
            "ok": all(c.ok for c in checks), "checks": [asdict(c) for c in checks]}
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(line) + "\n")


def latest_record(path: str = DEFAULT_LOG) -> Optional[dict]:
    p = Path(path)
    if not p.exists():
        return None
    lines = [l for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    return json.loads(lines[-1]) if lines else None


def _xml_escape(s: str) -> str:
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
             .replace('"', "&quot;").replace("'", "&apos;"))


def notify(title: str, body: str) -> bool:
    """Show a Windows desktop notification; returns False where unsupported.

    Uses the Windows Runtime toast API through Windows PowerShell, which needs
    no extra modules.  The scheduled task runs in the logged-in session, so
    the notification appears on screen.
    """
    if sys.platform != "win32":
        logger.warning("ALERT: %s — %s", title, body)
        return False
    xml = (f"<toast><visual><binding template='ToastGeneric'><text>{_xml_escape(title)}</text>"
           f"<text>{_xml_escape(body[:250])}</text></binding></visual></toast>")
    ps = (
        "[Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType = WindowsRuntime] | Out-Null;"
        "[Windows.Data.Xml.Dom.XmlDocument, Windows.Data.Xml.Dom.XmlDocument, ContentType = WindowsRuntime] | Out-Null;"
        "$x = New-Object Windows.Data.Xml.Dom.XmlDocument; $x.LoadXml($env:VSS_TOAST_XML);"
        "$t = [Windows.UI.Notifications.ToastNotification]::new($x);"
        "[Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier("
        "'{1AC14E77-02E7-4E5D-B744-2EB1AE5198B7}\\WindowsPowerShell\\v1.0\\powershell.exe').Show($t)"
    )
    import os

    try:
        r = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", ps],
                           env={**os.environ, "VSS_TOAST_XML": xml},
                           capture_output=True, text=True, timeout=30)
        if r.returncode != 0:
            logger.warning("Notification failed: %s", r.stderr.strip()[:300])
        return r.returncode == 0
    except Exception as exc:
        logger.warning("Notification failed: %s", exc)
        return False


def alert_if_failing(checks: Sequence[Check]) -> bool:
    """Notify about failed checks; returns True if any failed."""
    failed = [c for c in checks if not c.ok]
    if failed:
        notify(f"Vol surface: {len(failed)} check(s) failed",
               "; ".join(f"{c.name}: {c.message}" for c in failed))
    return bool(failed)
