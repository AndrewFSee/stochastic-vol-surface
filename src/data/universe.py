"""Asset-class groups for the ticker universe.

Used to order and label tickers in the dashboard and to pick comparable
peers for the interpretation.  Tickers not listed fall into "Other".
"""

from __future__ import annotations

GROUPS: dict[str, tuple[str, ...]] = {
    "US equity indices": ("SPY", "QQQ", "IWM", "DIA"),
    "Sectors": ("XLF", "XLE", "XLK", "XLV", "XLI", "XLU", "SMH", "KRE", "XBI"),
    "Rates and credit": ("TLT", "IEF", "HYG", "LQD"),
    "International": ("EEM", "EFA", "EWZ"),
    "Commodities": ("GLD", "SLV", "USO", "UNG", "GDX"),
    "Crypto and VIX": ("IBIT", "VXX"),
    "Single stocks": ("AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "AVGO", "AMD",
                      "TSLA", "NFLX", "JPM", "BAC", "COIN", "PLTR"),
}
OTHER = "Other"
_GROUP_OF = {t: g for g, ts in GROUPS.items() for t in ts}
_ORDER = {t: i for i, t in enumerate(t for ts in GROUPS.values() for t in ts)}


def group_of(ticker: str) -> str:
    return _GROUP_OF.get(ticker, OTHER)


def sort_tickers(tickers) -> list[str]:
    """Tickers in group order (as listed in :data:`GROUPS`), unknown ones last."""
    return sorted(tickers, key=lambda t: (_ORDER.get(t, len(_ORDER)), t))
