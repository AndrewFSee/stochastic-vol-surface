"""Data contracts for the options-chain pipeline.

Uses **Pandera** for DataFrame validation and **Pydantic** for config/metadata.
Every DataFrame that enters or leaves the storage layer must pass through these
schemas so that downstream modules (surface construction, features) never
encounter surprise columns or dtypes.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Literal, Optional

import os as _os
_os.environ.setdefault("DISABLE_PANDERA_IMPORT_WARNING", "True")

import pandera as pa
from pandera.typing import Series
from pydantic import BaseModel, Field

# ────────────────────────────────────────────────────────────────────────────
# Pandera: DataFrame-level contracts
# ────────────────────────────────────────────────────────────────────────────


class RawOptionsChainSchema(pa.DataFrameModel):
    """Schema for raw options-chain data as it arrives from the scraper.

    Every row is one option contract on one as-of date.
    """

    ticker: Series[str] = pa.Field(nullable=False)
    as_of: Series[pa.DateTime] = pa.Field(nullable=False)
    expiration: Series[pa.DateTime] = pa.Field(nullable=False)
    strike: Series[float] = pa.Field(gt=0, nullable=False)
    option_type: Series[str] = pa.Field(isin=["call", "put"], nullable=False)

    # Pricing — at least one of mid / (bid+ask) / last_price should be present.
    # We mark them nullable because individual sources may lack some.
    bid: Series[float] = pa.Field(ge=0, nullable=True, coerce=True)
    ask: Series[float] = pa.Field(ge=0, nullable=True, coerce=True)
    mid: Series[float] = pa.Field(ge=0, nullable=True, coerce=True)
    last_price: Series[float] = pa.Field(ge=0, nullable=True, coerce=True)

    volume: Series[float] = pa.Field(ge=0, nullable=True, coerce=True)
    open_interest: Series[float] = pa.Field(ge=0, nullable=True, coerce=True)
    implied_volatility_market: Series[float] = pa.Field(
        ge=0, le=10.0, nullable=True, coerce=True,
        description="Market-quoted IV (annualised), 0-10 range",
    )
    T: Series[float] = pa.Field(
        gt=0, nullable=False,
        description="Time-to-expiry in years (ACT/365)",
    )
    underlying_price: Series[float] = pa.Field(gt=0, nullable=True, coerce=True)

    class Config:
        name = "RawOptionsChain"
        strict = False          # allow extra columns (e.g. contractSymbol)
        coerce = True           # auto-cast compatible dtypes
        ordered = False


class NormalisedChainSchema(pa.DataFrameModel):
    """Schema after Kaggle / scraper normalisation — ready for storage.

    Same as RawOptionsChainSchema but also requires ``implied_volatility``
    (which may be identical to ``implied_volatility_market`` or computed later).
    """

    ticker: Series[str] = pa.Field(nullable=False)
    as_of: Series[pa.DateTime] = pa.Field(nullable=False)
    expiration: Series[pa.DateTime] = pa.Field(nullable=False)
    strike: Series[float] = pa.Field(gt=0, nullable=False)
    option_type: Series[str] = pa.Field(isin=["call", "put"], nullable=False)
    T: Series[float] = pa.Field(gt=0, nullable=False)
    underlying_price: Series[float] = pa.Field(gt=0, nullable=True, coerce=True)

    # At least one pricing field
    mid: Series[float] = pa.Field(ge=0, nullable=True, coerce=True)

    # IV must be present
    implied_volatility: Series[float] = pa.Field(
        gt=0, le=10.0, nullable=True, coerce=True,
    )

    class Config:
        name = "NormalisedChain"
        strict = False
        coerce = True
        ordered = False


class SurfaceGridSchema(pa.DataFrameModel):
    """Schema for a long-format surface grid (log-moneyness, tenor, IV)."""

    log_moneyness: Series[float] = pa.Field(nullable=False)
    T: Series[float] = pa.Field(gt=0, nullable=False)
    implied_volatility: Series[float] = pa.Field(gt=0, le=5.0, nullable=False)
    option_type: Series[str] = pa.Field(isin=["call", "put"], nullable=False)
    strike: Series[float] = pa.Field(gt=0, nullable=False)

    class Config:
        name = "SurfaceGrid"
        strict = False
        coerce = True
        ordered = False


class VixFamilySchema(pa.DataFrameModel):
    """Schema for the VIX-family daily snapshot stored alongside chains."""

    date: Series[pa.DateTime] = pa.Field(nullable=False)
    VIX: Series[float] = pa.Field(ge=0, nullable=True, coerce=True)
    VIX3M: Series[float] = pa.Field(ge=0, nullable=True, coerce=True)
    VIX9D: Series[float] = pa.Field(ge=0, nullable=True, coerce=True)
    SKEW: Series[float] = pa.Field(ge=0, nullable=True, coerce=True)
    VVIX: Series[float] = pa.Field(ge=0, nullable=True, coerce=True)

    class Config:
        name = "VixFamily"
        strict = False
        coerce = True
        ordered = False


class RatesSchema(pa.DataFrameModel):
    """Schema for the FRED risk-free rates snapshot."""

    date: Series[pa.DateTime] = pa.Field(nullable=False)

    class Config:
        name = "Rates"
        strict = False      # columns depend on which tenors are fetched
        coerce = True
        ordered = False


# ────────────────────────────────────────────────────────────────────────────
# Pydantic: config / metadata models
# ────────────────────────────────────────────────────────────────────────────


class ScraperConfig(BaseModel):
    """Configuration for the daily scraper job."""

    tickers: list[str] = Field(default=["SPY", "QQQ", "IWM", "GLD", "AAPL", "MSFT", "TSLA", "XLF"])
    schedule_time: str = Field(default="16:30", description="HH:MM in timezone")
    timezone: str = Field(default="America/New_York")
    min_open_interest: int = Field(default=10, ge=0)
    min_volume: int = Field(default=0, ge=0)
    max_spread_ratio: float = Field(
        default=0.5, ge=0, le=1.0,
        description="Max (ask-bid)/mid to keep a row",
    )
    inter_ticker_delay: float = Field(
        default=1.5, ge=0,
        description="Seconds to sleep between tickers to avoid rate-limits",
    )
    output_dir: str = Field(default="data/options")
    vix_dir: str = Field(default="data/vix")
    rates_dir: str = Field(default="data/rates")
    surfaces_dir: str = Field(default="data/surfaces")
    collect_vix: bool = Field(default=True)
    collect_rates: bool = Field(default=True, description="Requires FRED_API_KEY")
    build_surfaces: bool = Field(
        default=True,
        description="Build vol surfaces from the chains collected in this run",
    )
    underlying_dir: str = Field(default="data/underlying")
    collect_underlying: bool = Field(
        default=True,
        description="Refresh daily OHLC for the tickers (realised-vol inputs)",
    )
    features_path: str = Field(default="data/features/surface_features.parquet")
    macro_dir: str = Field(default="data/macro")
    collect_macro: bool = Field(default=True, description="FRED credit/macro series; needs FRED_API_KEY")
    events_dir: str = Field(default="data/events")
    collect_earnings: bool = Field(default=True, description="Earnings dates for single stocks")
    backup_dir: Optional[str] = Field(
        default=None,
        description="Copy data/ here after each run (see src.data.backup); None disables",
    )
    build_features: bool = Field(
        default=True,
        description="Rebuild the feature table after the surfaces",
    )


class ScrapeResult(BaseModel):
    """Metadata returned after a successful scrape run."""

    as_of: date
    tickers: list[str]
    total_rows: int
    partitions_written: int
    vix_snapshot: Optional[dict[str, float]] = None
    rates_snapshot: Optional[dict[str, float]] = None
    errors: list[str] = Field(default_factory=list)


class StorageIntegrityReport(BaseModel):
    """Returned by integrity-check utilities."""

    ticker: str
    date_range: tuple[str, str]
    n_dates: int
    n_rows: int
    missing_dates: list[str] = Field(default_factory=list)
    duplicate_dates: list[str] = Field(default_factory=list)
    schema_errors: list[str] = Field(default_factory=list)


# ────────────────────────────────────────────────────────────────────────────
# Convenience validators
# ────────────────────────────────────────────────────────────────────────────


def validate_raw_chain(df, *, lazy: bool = True):
    """Validate *df* against :class:`RawOptionsChainSchema`.

    Returns the validated DataFrame (potentially coerced) or raises
    ``pandera.errors.SchemaErrors`` if *lazy=True*.
    """
    return RawOptionsChainSchema.validate(df, lazy=lazy)


def validate_normalised_chain(df, *, lazy: bool = True):
    """Validate *df* against :class:`NormalisedChainSchema`."""
    return NormalisedChainSchema.validate(df, lazy=lazy)


def validate_surface_grid(df, *, lazy: bool = True):
    """Validate *df* against :class:`SurfaceGridSchema`."""
    return SurfaceGridSchema.validate(df, lazy=lazy)
