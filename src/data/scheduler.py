"""Autonomous daily scheduler for the full data-collection pipeline.

Orchestrates:
  1. Options-chain scrape  (yfinance → Parquet)
  2. VIX-family snapshot   (yfinance → Parquet)
  3. Risk-free rates       (FRED → Parquet, if API key present)
  4. Schema validation on every write
  5. Lightweight integrity report logged after each run

Can be used as:
  • Imported module   — ``Scheduler(cfg).start()``
  • CLI               — ``python -m src.data.scheduler``
  • Windows Task Scheduler / cron target (via scripts/schedule_scraper.py)
"""

from __future__ import annotations

import json
import logging
import os
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

from src.data.schema import ScraperConfig, ScrapeResult

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────────────
# Market-calendar helper
# ────────────────────────────────────────────────────────────────────────────


def is_market_open(d: date) -> bool:
    """Return True if *d* is a regular NYSE trading day (not a weekend or holiday)."""
    try:
        import pandas_market_calendars as mcal

        nyse = mcal.get_calendar("NYSE")
        valid = nyse.valid_days(d.isoformat(), d.isoformat())
        return len(valid) > 0
    except ImportError:
        logger.warning("pandas_market_calendars not installed — skipping holiday check")
        # Fall back to weekday-only check (no holiday awareness)
        return d.weekday() < 5


# ────────────────────────────────────────────────────────────────────────────
# Core job: one complete collection cycle
# ────────────────────────────────────────────────────────────────────────────


#: Equity-ETF options trade until 16:15 ET, a quarter hour after the stock
#: close; collect only once they have stopped moving.
OPTIONS_CLOSE_LAG = timedelta(minutes=15)


def session_close(d: date, tz: str = "America/New_York") -> datetime:
    """The NYSE close on *d* as an aware datetime (13:00 on half days)."""
    from zoneinfo import ZoneInfo

    try:
        import pandas_market_calendars as mcal

        sched = mcal.get_calendar("NYSE").schedule(d.isoformat(), d.isoformat())
        if len(sched):
            return sched["market_close"].iloc[0].to_pydatetime().astimezone(ZoneInfo(tz))
    except ImportError:
        pass
    return datetime(d.year, d.month, d.day, 16, 0, tzinfo=ZoneInfo(tz))


def collection_allowed(as_of: date, now: datetime) -> tuple[bool, str]:
    """Whether a scrape at *now* yields closing quotes for *as_of*.

    yfinance returns whatever is quoted at the moment of the call, stamped
    with *as_of*.  So a run before that session's options close stores
    intraday quotes as the close, and a run on a later day (e.g. a missed
    16:30 task caught up the next morning) stores the wrong day's quotes
    under *as_of*.  Both corrupt the history silently.
    """
    if now.date() != as_of:
        return False, (f"as-of {as_of} is not today ({now.date()}); quotes now "
                       f"would not be {as_of}'s close")
    ready = session_close(as_of, str(now.tzinfo)) + OPTIONS_CLOSE_LAG
    if now < ready:
        return False, f"before the options close ({ready:%H:%M %Z}); quotes are intraday"
    return True, ""


#: The pre-close snapshot runs in the last half hour before the close.
PRECLOSE_WINDOW = timedelta(minutes=30)


def preclose_allowed(as_of: date, now: datetime) -> tuple[bool, str]:
    """Whether *now* is inside the pre-close window of *as_of*'s session.

    Thin ETFs' option quotes are pulled at the 16:00 close, so their chains
    are collected in the last half hour while the quotes are still live.
    """
    if now.date() != as_of:
        return False, f"as-of {as_of} is not today ({now.date()})"
    close = session_close(as_of, str(now.tzinfo))
    if not (close - PRECLOSE_WINDOW <= now < close):
        return False, (f"outside the pre-close window "
                       f"({close - PRECLOSE_WINDOW:%H:%M}-{close:%H:%M %Z})")
    return True, ""


def _has_chain(output_dir: str, ticker: str, as_of: date) -> bool:
    return (Path(output_dir) / f"ticker={ticker}" / f"date={as_of.isoformat()}").exists()


def run_preclose(
    cfg: ScraperConfig,
    as_of: Optional[date] = None,
    *,
    force: bool = False,
) -> ScrapeResult:
    """Collect the chains of ``cfg.preclose_tickers`` before the close.

    Rows are tagged ``snapshot = "preclose"``.  The 16:30 run then skips
    these tickers' chains (keeping the live quotes) but builds their surfaces
    and features with everyone else's.  Logged to
    ``logs/preclose_runs.jsonl``.
    """
    from zoneinfo import ZoneInfo

    from src.data.scraper import scrape_all
    from src.data.storage import save_options_chain

    now = datetime.now(ZoneInfo(cfg.timezone))
    as_of = as_of or now.date()
    tickers = list(cfg.preclose_tickers)
    if not tickers or not is_market_open(as_of):
        return ScrapeResult(as_of=as_of, tickers=tickers, total_rows=0, partitions_written=0,
                            errors=[] if tickers else ["No pre-close tickers configured"])
    ok, why = preclose_allowed(as_of, now)
    if not ok and not force:
        logger.warning("Not collecting pre-close chains: %s.", why)
        return ScrapeResult(as_of=as_of, tickers=tickers, total_rows=0,
                            partitions_written=0, errors=[f"Skipped: {why}"])

    errors: list[str] = []
    rows = parts = 0
    try:
        df = scrape_all(tickers, as_of=as_of, inter_ticker_delay=cfg.inter_ticker_delay)
        if df.empty:
            errors.append("Pre-close scrape returned nothing")
        else:
            df = _apply_quality_filters(df, cfg)
            df["snapshot"] = "preclose"
            missing = sorted(set(tickers) - set(df["ticker"]))
            if missing:
                errors.append(f"No pre-close chain for {', '.join(missing)}")
            parts = len(save_options_chain(df, base_dir=cfg.output_dir))
            rows = len(df)
            logger.info("Pre-close options: %d rows -> %d partitions", rows, parts)
    except Exception as exc:
        logger.exception("Pre-close scrape failed")
        errors.append(f"Options: {exc}")

    result = ScrapeResult(as_of=as_of, tickers=tickers, total_rows=rows,
                          partitions_written=parts, errors=errors)
    log = Path(cfg.output_dir).parent / "logs" / "preclose_runs.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "a") as f:
        f.write(result.model_dump_json() + "\n")
    return result


def run_collection(
    cfg: ScraperConfig,
    as_of: Optional[date] = None,
    *,
    force: bool = False,
) -> ScrapeResult:
    """Execute a single collection cycle (options + VIX + rates).

    Skips the run entirely if the NYSE is closed (holiday / weekend), or if
    it is not after that session's options close (see
    :func:`collection_allowed`; *force* overrides this), and returns a
    zero-row ``ScrapeResult``.

    Parameters
    ----------
    cfg : ScraperConfig
        Validated configuration (tickers, paths, flags).
    as_of : date or None
        Stamp for the run; defaults to today in ``cfg.timezone``.
    force : bool
        Collect even before the close or for another day.  For deliberate
        manual runs only — the quotes are stored as *as_of*'s close.

    Returns
    -------
    ScrapeResult   with metadata about the run.
    """
    from zoneinfo import ZoneInfo

    from src.data.scraper import scrape_all
    from src.data.storage import save_options_chain

    now = datetime.now(ZoneInfo(cfg.timezone))
    as_of = as_of or now.date()

    # ── Holiday guard ─────────────────────────────────────────────────────
    if not is_market_open(as_of):
        logger.info("Market closed on %s — skipping collection.", as_of)
        return ScrapeResult(
            as_of=as_of,
            tickers=cfg.tickers,
            total_rows=0,
            partitions_written=0,
            errors=["Market closed — skipped"],
        )

    # ── Timing guard ──────────────────────────────────────────────────────
    ok, why = collection_allowed(as_of, now)
    if not ok and not force:
        logger.warning("Not collecting: %s. Use --force to override.", why)
        return ScrapeResult(as_of=as_of, tickers=cfg.tickers, total_rows=0,
                            partitions_written=0, errors=[f"Skipped: {why}"])

    errors: list[str] = []
    total_rows = 0
    partitions = 0
    vix_snap: Optional[dict[str, float]] = None
    rates_snap: Optional[dict[str, float]] = None

    # ── 1. Options chains ─────────────────────────────────────────────────
    logger.info("=== Collection cycle %s ===", as_of)
    # Thin ETFs already collected before the close keep those live quotes.
    done = [t for t in cfg.preclose_tickers
            if t in cfg.tickers and _has_chain(cfg.output_dir, t, as_of)]
    if done:
        logger.info("Keeping pre-close chains for %s", ", ".join(done))
    to_scrape = [t for t in cfg.tickers if t not in done]
    try:
        df = scrape_all(to_scrape, as_of=as_of, inter_ticker_delay=cfg.inter_ticker_delay)
        if df.empty:
            errors.append("Options scrape returned empty DataFrame")
        else:
            # Optional: filter by spread / OI from config
            df = _apply_quality_filters(df, cfg)

            # Validate before writing
            try:
                from src.data.schema import validate_raw_chain
                df = validate_raw_chain(df, lazy=True)
            except Exception as exc:
                logger.warning("Schema validation warnings: %s", exc)

            paths = save_options_chain(df, base_dir=cfg.output_dir)
            total_rows = len(df)
            partitions = len(paths)
            logger.info("Options: %d rows → %d partitions", total_rows, partitions)
    except Exception as exc:
        logger.exception("Options scrape failed")
        errors.append(f"Options: {exc}")

    # ── 2. VIX family ─────────────────────────────────────────────────────
    if cfg.collect_vix:
        try:
            vix_snap = _collect_vix(as_of, cfg.vix_dir)
            logger.info("VIX snapshot: %s", vix_snap)
        except Exception as exc:
            logger.exception("VIX collection failed")
            errors.append(f"VIX: {exc}")

    # ── 3. Risk-free rates ────────────────────────────────────────────────
    if cfg.collect_rates:
        try:
            rates_snap = _collect_rates(as_of, cfg.rates_dir)
            logger.info("Rates snapshot: %s", rates_snap)
        except Exception as exc:
            logger.warning("Rates collection failed (FRED_API_KEY set?): %s", exc)
            errors.append(f"Rates: {exc}")

    # ── 3a. Macro and credit series (FRED) ────────────────────────────────
    if cfg.collect_macro:
        try:
            _collect_macro(as_of, cfg.macro_dir)
        except Exception as exc:
            logger.warning("Macro collection failed (FRED_API_KEY set?): %s", exc)
            errors.append(f"Macro: {exc}")

    # ── 3b. Underlying prices ─────────────────────────────────────────────
    if cfg.collect_underlying:
        try:
            n = _collect_underlying(cfg)
            logger.info("Underlying prices: %d rows refreshed", n)
        except Exception as exc:
            logger.exception("Underlying price collection failed")
            errors.append(f"Underlying: {exc}")

    # ── 3c. Earnings dates ────────────────────────────────────────────────
    if cfg.collect_earnings:
        try:
            n = _collect_earnings(as_of, cfg)
            logger.info("Earnings dates: %d tickers refreshed", n)
        except Exception as exc:
            logger.exception("Earnings-date collection failed")
            errors.append(f"Earnings: {exc}")

    # ── 4. Volatility surfaces ────────────────────────────────────────────
    # Runs last so it consumes the chains and rates written above.
    if cfg.build_surfaces and total_rows > 0:
        try:
            counts = _build_surfaces(as_of, cfg)
            logger.info("Surfaces: %s", counts)
            failed = counts.get("failed", 0) + counts.get("rejected", 0)
            if failed:
                errors.append(f"Surfaces: {failed} failed/rejected")
        except Exception as exc:
            logger.exception("Surface build failed")
            errors.append(f"Surfaces: {exc}")

    # ── 5. Feature table ──────────────────────────────────────────────────
    # Rebuilt in full from the stores, so it always matches the surfaces.
    if cfg.build_features and total_rows > 0:
        try:
            from src.features.table import build_feature_table, save_feature_table

            feats = build_feature_table(
                cfg.tickers, surfaces_dir=cfg.surfaces_dir,
                underlying_dir=cfg.underlying_dir, vix_dir=cfg.vix_dir,
                macro_dir=cfg.macro_dir, events_dir=cfg.events_dir,
                cache_path=str(Path(cfg.features_path).parent / "_surface_rows_cache.parquet"),
            )
            if feats.empty:
                errors.append("Features: no rows built")
            else:
                save_feature_table(feats, cfg.features_path)
                logger.info("Features: %d rows", len(feats))
        except Exception as exc:
            logger.exception("Feature build failed")
            errors.append(f"Features: {exc}")

    # ── 6. Volatility forecasts ───────────────────────────────────────────
    # After the features they read; rebuilt in full (walk-forward, ~15 s).
    if cfg.build_features and total_rows > 0:
        try:
            from src.forecast.forecaster import build_forecasts, save_forecasts

            fc = build_forecasts(underlying_dir=cfg.underlying_dir)
            if fc.empty:
                errors.append("Forecasts: none built")
            else:
                save_forecasts(fc)
                logger.info("Forecasts: %d rows", len(fc))
        except Exception as exc:
            logger.exception("Forecast build failed")
            errors.append(f"Forecasts: {exc}")

    # ── 7. Backup ─────────────────────────────────────────────────────────
    # Last, so it captures everything this run wrote.
    if cfg.backup_dir:
        try:
            from src.data.backup import backup_data

            b = backup_data(Path(cfg.output_dir).parent, cfg.backup_dir)
            logger.info("Backup: %d files copied to %s", b.files_copied, b.target)
        except Exception as exc:
            logger.exception("Backup failed")
            errors.append(f"Backup: {exc}")

    # ── 8. Health checks and alerts ───────────────────────────────────────
    # Verify every stage produced today's output and the surfaces still agree
    # with VIX; record the result and raise a desktop alert on any failure.
    try:
        from src.data.monitor import alert_if_failing, record, run_checks

        checks = run_checks(
            tickers=cfg.tickers, run_errors=errors, as_of=as_of,
            options_dir=cfg.output_dir, surfaces_dir=cfg.surfaces_dir,
            features_path=cfg.features_path, backup_dir=cfg.backup_dir,
        )
        record(checks, as_of, str(Path(cfg.output_dir).parent / "logs" / "health_checks.jsonl"))
        if alert_if_failing(checks):
            errors.append("Health: " + "; ".join(c.name for c in checks if not c.ok) + " failed")
    except Exception as exc:
        logger.exception("Health checks failed to run")
        errors.append(f"Health checks: {exc}")

    result = ScrapeResult(
        as_of=as_of,
        tickers=cfg.tickers,
        total_rows=total_rows,
        partitions_written=partitions,
        vix_snapshot=vix_snap,
        rates_snapshot=rates_snap,
        errors=errors,
    )

    # Persist run metadata
    _save_run_log(result, cfg.output_dir)
    return result


# ────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ────────────────────────────────────────────────────────────────────────────


def _apply_quality_filters(df: pd.DataFrame, cfg: ScraperConfig) -> pd.DataFrame:
    """Drop rows that fail open-interest, volume, or spread-ratio filters."""
    n_before = len(df)

    if cfg.min_open_interest > 0 and "open_interest" in df.columns:
        df = df[df["open_interest"].fillna(0) >= cfg.min_open_interest]

    if cfg.min_volume > 0 and "volume" in df.columns:
        df = df[df["volume"].fillna(0) >= cfg.min_volume]

    if cfg.max_spread_ratio < 1.0 and "bid" in df.columns and "ask" in df.columns:
        mid = df["mid"] if "mid" in df.columns else (df["bid"] + df["ask"]) / 2
        spread_ratio = (df["ask"] - df["bid"]) / mid.replace(0, float("nan"))
        df = df[spread_ratio.fillna(1.0) <= cfg.max_spread_ratio]

    n_after = len(df)
    if n_before != n_after:
        logger.info("Quality filters: %d → %d rows (dropped %d)",
                     n_before, n_after, n_before - n_after)
    return df.copy()


def _collect_vix(as_of: date, vix_dir: str) -> dict[str, float]:
    """Fetch latest VIX-family values and persist to Parquet."""
    from src.data.vix_family import fetch_vix_family

    df = fetch_vix_family(period="5d")
    if df.empty:
        return {}

    # Save full recent window
    out_dir = Path(vix_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"vix_{as_of.isoformat()}.parquet"
    df.to_parquet(out_path, engine="pyarrow")

    latest = df.iloc[-1]
    return {col: round(float(latest[col]), 4)
            for col in df.columns if pd.notna(latest[col])}


def _collect_rates(as_of: date, rates_dir: str) -> dict[str, float]:
    """Fetch recent FRED rates and merge them into the consolidated history.

    Only a short trailing window is requested — the merge in
    :func:`~src.data.rates.save_rates_history` keeps everything already stored,
    so a daily run stays cheap.  Use ``scripts/backfill_rates.py`` to repair a
    longer gap.
    """
    from src.data.rates import fetch_rates, save_rates_history

    start = (as_of - timedelta(days=30)).isoformat()
    df = fetch_rates(start=start)
    if df.empty:
        return {}

    save_rates_history(df, rates_dir=rates_dir)

    latest = df.iloc[-1]
    return {col: round(float(latest[col]), 6)
            for col in df.columns if pd.notna(latest[col])}


def _collect_macro(as_of: date, macro_dir: str) -> None:
    """Refresh the trailing two months of FRED macro series into the history."""
    from src.data.macro import fetch_macro, save_macro_history

    df = fetch_macro(start=(as_of - timedelta(days=60)).isoformat())
    if not df.empty:
        save_macro_history(df, macro_dir)


def _collect_earnings(as_of: date, cfg: ScraperConfig) -> int:
    """Refresh earnings dates: the stocks already known every day (dates get
    confirmed or moved), every configured ticker on Mondays (new tickers)."""
    from src.data.events import (fetch_earnings, load_earnings, save_earnings,
                                 save_earnings_snapshot)

    known = set(load_earnings(cfg.events_dir)["ticker"])
    tickers = cfg.tickers if (as_of.weekday() == 0 or not known) else \
        [t for t in cfg.tickers if t in known]
    df = fetch_earnings(tickers)
    if not df.empty:
        save_earnings(df, cfg.events_dir)
        save_earnings_snapshot(df, as_of, cfg.events_dir)
    return int(df["ticker"].nunique()) if not df.empty else 0


def _collect_underlying(cfg: ScraperConfig) -> int:
    """Refresh the trailing month of daily OHLC into the price history.

    A month rather than a day so that provisional bars from earlier runs
    (Yahoo's same-day bar often has no close yet) get replaced by final ones.
    """
    from src.data.underlying import fetch_underlying, save_underlying_history

    df = fetch_underlying(cfg.tickers, period="1mo")
    if df.empty:
        return 0
    save_underlying_history(df, cfg.underlying_dir)
    return len(df)


def _build_surfaces(as_of: date, cfg: ScraperConfig) -> dict[str, int]:
    """Build volatility surfaces for the snapshots just collected."""
    from src.surface.batch import build_corpus

    report = build_corpus(
        tickers=cfg.tickers,
        dates=[as_of],
        options_dir=cfg.output_dir,
        surfaces_dir=cfg.surfaces_dir,
        progress=False,
        workers=max(1, min(len(cfg.tickers), (os.cpu_count() or 2) - 1)),
    )
    return report.counts()


def _save_run_log(result: ScrapeResult, base_dir: str) -> None:
    """Append run metadata to a JSONL log file."""
    log_dir = Path(base_dir).parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "scrape_runs.jsonl"
    with open(log_path, "a") as f:
        f.write(result.model_dump_json() + "\n")
    logger.info("Run log appended to %s", log_path)


# ────────────────────────────────────────────────────────────────────────────
# Scheduler class
# ────────────────────────────────────────────────────────────────────────────


class Scheduler:
    """Wraps APScheduler to run :func:`run_collection` on a cron trigger.

    Usage::

        from src.data.scheduler import Scheduler
        from src.data.schema import ScraperConfig

        sched = Scheduler(ScraperConfig())
        sched.start()          # blocking — runs forever
    """

    def __init__(self, cfg: Optional[ScraperConfig] = None):
        self.cfg = cfg or ScraperConfig()

    def start(self) -> None:
        """Start the blocking scheduler (Mon–Fri at configured time)."""
        from apscheduler.schedulers.blocking import BlockingScheduler
        from apscheduler.triggers.cron import CronTrigger

        hour, minute = self.cfg.schedule_time.split(":")
        scheduler = BlockingScheduler()
        trigger = CronTrigger(
            day_of_week="mon-fri",
            hour=int(hour),
            minute=int(minute),
            timezone=self.cfg.timezone,
        )
        scheduler.add_job(
            run_collection,
            trigger=trigger,
            args=[self.cfg],
            id="daily_collection",
            name=f"Daily scrape ({', '.join(self.cfg.tickers)})",
            misfire_grace_time=3600,  # allow up to 1 hr late
        )

        logger.info(
            "Scheduler started — collecting %s at %s %s Mon–Fri",
            self.cfg.tickers, self.cfg.schedule_time, self.cfg.timezone,
        )
        try:
            scheduler.start()
        except (KeyboardInterrupt, SystemExit):
            logger.info("Scheduler stopped.")

    def run_now(self) -> ScrapeResult:
        """Run a single collection cycle immediately (no scheduling)."""
        return run_collection(self.cfg)


# ────────────────────────────────────────────────────────────────────────────
# Module-level CLI
# ────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    )

    cfg = ScraperConfig()

    if "--once" in sys.argv:
        result = run_collection(cfg)
        print(result.model_dump_json(indent=2))
    else:
        sched = Scheduler(cfg)
        sched.start()
