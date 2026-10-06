"""Shared fixtures."""

from __future__ import annotations

import pandas as pd
import pytest

from tests.synthetic_chains import gbm_prices, make_chain


@pytest.fixture
def stores(tmp_path):
    """Options, surfaces, prices and VIX for one ticker over five days."""
    from src.data.underlying import save_underlying_history
    from src.surface.batch import build_corpus

    dates = list(pd.bdate_range("2026-03-02", periods=5).date)
    opt = tmp_path / "options"
    for i, d in enumerate(dates):
        part = opt / "ticker=TEST" / f"date={d.isoformat()}"
        part.mkdir(parents=True)
        make_chain(as_of=d, atm=0.20 + 0.01 * i,
                   tenors=(0.06, 0.12, 0.25, 0.5, 1.0)).to_parquet(
            part / "chain.parquet", index=False)
    build_corpus(options_dir=str(opt), surfaces_dir=str(tmp_path / "surfaces"),
                 progress=False)

    prices = gbm_prices(n=80, start="2025-11-03").loc[:pd.Timestamp(dates[-1])]
    prices["ticker"] = "TEST"
    save_underlying_history(prices.reset_index().set_index(["date", "ticker"]),
                            str(tmp_path / "underlying"))

    vix_dir = tmp_path / "vix"
    vix_dir.mkdir()
    pd.DataFrame({"VIX": [15.0 + i for i in range(5)]},
                 index=pd.DatetimeIndex(pd.to_datetime(dates), name="date")).to_parquet(
        vix_dir / f"vix_{dates[-1].isoformat()}.parquet")
    return {"root": str(tmp_path),
            "options_dir": str(opt),
            "surfaces_dir": str(tmp_path / "surfaces"),
            "underlying_dir": str(tmp_path / "underlying"),
            "vix_dir": str(vix_dir), "macro_dir": str(tmp_path / "macro"),
            "dates": dates}
