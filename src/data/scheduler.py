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


def run_collection(cfg: ScraperConfig, as_of: Optional[date] = None) -> ScrapeResult:
    """Execute a single collection cycle (options + VIX + rates).

    Skips the run entirely if the NYSE is closed (holiday / weekend)
    and returns a zero-row ``ScrapeResult``.

    Parameters
    ----------
    cfg : ScraperConfig
        Validated configuration (tickers, paths, flags).
    as_of : date or None
        Stamp for the run; defaults to today.

    Returns
    -------
    ScrapeResult   with metadata about the run.
    """
    from src.data.scraper import scrape_all
    from src.data.storage import save_options_chain

    as_of = as_of or date.today()

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

    errors: list[str] = []
    total_rows = 0
    partitions = 0
    vix_snap: Optional[dict[str, float]] = None
    rates_snap: Optional[dict[str, float]] = None

    # ── 1. Options chains ─────────────────────────────────────────────────
    logger.info("=== Collection cycle %s ===", as_of)
    try:
        df = scrape_all(cfg.tickers, as_of=as_of, inter_ticker_delay=cfg.inter_ticker_delay)
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

    # ── 3b. Underlying prices ─────────────────────────────────────────────
    if cfg.collect_underlying:
        try:
            n = _collect_underlying(cfg)
            logger.info("Underlying prices: %d rows refreshed", n)
        except Exception as exc:
            logger.exception("Underlying price collection failed")
            errors.append(f"Underlying: {exc}")

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
            )
            if feats.empty:
                errors.append("Features: no rows built")
            else:
                save_feature_table(feats, cfg.features_path)
                logger.info("Features: %d rows", len(feats))
        except Exception as exc:
            logger.exception("Feature build failed")
            errors.append(f"Features: {exc}")

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
