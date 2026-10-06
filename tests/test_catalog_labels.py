"""The feature catalog stays in sync with the table; labels are aligned forward."""

from __future__ import annotations

import numpy as np
import pandas as pd


def test_every_built_column_is_documented(stores):
    from src.features.catalog import undocumented
    from src.features.table import build_feature_table

    df = build_feature_table(surfaces_dir=stores["surfaces_dir"],
                             underlying_dir=stores["underlying_dir"],
                             vix_dir=stores["vix_dir"], macro_dir=stores["macro_dir"])
    assert undocumented(df.columns) == []


def test_every_configured_source_column_is_documented():
    """Columns the synthetic store does not produce: all vol indices, macro
    series, earnings and dynamics."""
    from src.data.macro import MACRO_SERIES
    from src.features.catalog import undocumented
    from src.features.table import _VIX_COLUMNS, DYNAMIC_COLUMNS

    cols = (list(_VIX_COLUMNS.values()) + [f"macro_{n.lower()}" for n in MACRO_SERIES]
            + ["earn_next_date", "earn_days_to", "earn_days_since", "earn_implied_move",
               "earn_hist_move"]
            + [f"{c}{s}" for c in DYNAMIC_COLUMNS for s in ("_d1", "_d5", "_z63", "_pct252")])
    assert undocumented(cols) == []
    assert undocumented(["made_up_feature"]) == ["made_up_feature"]


def test_labels_look_forward_on_the_trading_calendar():
    from src.features.labels import add_labels

    dates = pd.bdate_range("2026-03-02", periods=40)
    close = pd.Series(100 * np.exp(np.cumsum(np.full(40, 0.01))), index=dates)  # +1% a day
    prices = pd.DataFrame({"close": close, "adj_close": close})
    prices["ticker"] = "AAA"
    prices = prices.rename_axis("date").set_index("ticker", append=True)
    feats = pd.DataFrame({"ticker": "AAA", "date": dates,
                          "atm_30d": np.linspace(0.20, 0.59, 40)})
    out = add_labels(feats, prices, horizons=(5,)).set_index("date")

    t = dates[10]
    assert abs(out.loc[t, "label_ret_5"] - 0.05) < 1e-12
    assert abs(out.loc[t, "label_rv_5"] - np.sqrt(252 * 0.01 ** 2)) < 1e-12
    assert abs(out.loc[t, "label_atm30_chg_5"] - 0.05) < 1e-12
    assert np.isnan(out.loc[dates[-1], "label_ret_5"])           # window not closed yet
    assert np.isnan(out.loc[dates[-3], "label_atm30_chg_5"])
