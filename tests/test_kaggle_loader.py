"""Tests for normalising third-party historical option-chain files."""

import numpy as np
import pandas as pd
import pytest

from src.data.kaggle_loader import (
    _normalise_name,
    _resolve_columns,
    load_kaggle_spy_iv,
)


# ── Column-name normalisation ────────────────────────────────────────────


@pytest.mark.parametrize("raw,expected", [
    ("[C_IV]", "c_iv"),
    ("  [QUOTE_DATE]  ", "quote_date"),
    ("UNDERLYING_LAST", "underlying_last"),
    ("Strike", "strike"),
    ("expire date", "expire_date"),
    ("expire-date", "expire_date"),
    ("(P_BID)", "p_bid"),
])
def test_normalise_name(raw, expected):
    assert _normalise_name(raw) == expected


def test_resolve_columns_handles_bracketed_headers():
    """The real dataset ships headers wrapped in square brackets."""
    df = pd.DataFrame(columns=[
        "[QUOTE_DATE]", "[EXPIRE_DATE]", "[STRIKE]",
        "[UNDERLYING_LAST]", "[C_IV]", "[P_IV]", "[C_BID]", "[C_ASK]",
    ])
    m = _resolve_columns(df)
    assert m["date"] == "[QUOTE_DATE]"
    assert m["expiration"] == "[EXPIRE_DATE]"
    assert m["strike"] == "[STRIKE]"
    assert m["underlying_price"] == "[UNDERLYING_LAST]"
    assert m["call_iv"] == "[C_IV]"
    assert m["put_iv"] == "[P_IV]"


def test_resolve_columns_handles_plain_headers():
    df = pd.DataFrame(columns=["date", "expiration", "strike", "call_iv", "put_iv"])
    m = _resolve_columns(df)
    assert m["date"] == "date" and m["call_iv"] == "call_iv"


# ── Fixtures ─────────────────────────────────────────────────────────────


def _wide_bracketed(tmp_path, n_strikes=6):
    """A miniature file in the real dataset's wide, bracketed format."""
    rows = []
    for quote in ("2020-03-16", "2020-03-17"):
        for exp in ("2020-04-17", "2020-06-19"):
            for i in range(n_strikes):
                strike = 250.0 + 5 * i
                rows.append({
                    "[QUOTE_DATE]": quote,
                    "[EXPIRE_DATE]": exp,
                    "[STRIKE]": strike,
                    "[UNDERLYING_LAST]": 265.0,
                    "[C_IV]": 0.55 + 0.01 * i,
                    "[P_IV]": 0.60 + 0.01 * i,
                    "[C_BID]": 10.0, "[C_ASK]": 10.5,
                    "[P_BID]": 8.0, "[P_ASK]": 8.4,
                    "[C_VOLUME]": 100.0, "[P_VOLUME]": 90.0,
                })
    path = tmp_path / "spy_eod_2020.parquet"
    pd.DataFrame(rows).to_parquet(path)
    return path


# ── Loading ──────────────────────────────────────────────────────────────


def test_load_produces_canonical_columns(tmp_path):
    df = load_kaggle_spy_iv(_wide_bracketed(tmp_path))
    for col in ("ticker", "as_of", "expiration", "strike", "option_type",
                "implied_volatility_market", "T", "underlying_price", "mid"):
        assert col in df.columns


def test_load_explodes_wide_into_long(tmp_path):
    """Each wide row carries a call and a put; both must become rows."""
    path = _wide_bracketed(tmp_path, n_strikes=6)
    raw = pd.read_parquet(path)
    df = load_kaggle_spy_iv(path)
    assert len(df) == 2 * len(raw)
    assert set(df.option_type) == {"call", "put"}


def test_load_computes_time_to_expiry(tmp_path):
    df = load_kaggle_spy_iv(_wide_bracketed(tmp_path))
    assert (df["T"] > 0).all()
    row = df.iloc[0]
    expected = (row["expiration"] - row["as_of"]).days / 365.0
    assert row["T"] == pytest.approx(expected)


def test_load_computes_mid_from_bid_ask(tmp_path):
    df = load_kaggle_spy_iv(_wide_bracketed(tmp_path))
    calls = df[df.option_type == "call"]
    assert calls["mid"].iloc[0] == pytest.approx((10.0 + 10.5) / 2)


def test_load_parses_dates(tmp_path):
    df = load_kaggle_spy_iv(_wide_bracketed(tmp_path))
    assert pd.api.types.is_datetime64_any_dtype(df["as_of"])
    assert pd.api.types.is_datetime64_any_dtype(df["expiration"])


def test_load_drops_nonpositive_iv(tmp_path):
    rows = pd.read_parquet(_wide_bracketed(tmp_path))
    rows.loc[0, "[C_IV]"] = 0.0
    rows.loc[1, "[C_IV]"] = np.nan
    path = tmp_path / "patched.parquet"
    rows.to_parquet(path)

    df = load_kaggle_spy_iv(path)
    assert (df["implied_volatility_market"] > 0).all()
    assert df["implied_volatility_market"].notna().all()


def test_load_sets_both_iv_columns(tmp_path):
    """Downstream code reads either name depending on provenance."""
    df = load_kaggle_spy_iv(_wide_bracketed(tmp_path))
    pd.testing.assert_series_equal(
        df["implied_volatility"], df["implied_volatility_market"],
        check_names=False,
    )


def test_load_rejects_unrecognisable_file(tmp_path):
    path = tmp_path / "junk.parquet"
    pd.DataFrame({"foo": [1], "bar": [2]}).to_parquet(path)
    with pytest.raises(ValueError):
        load_kaggle_spy_iv(path)


def test_loaded_frame_is_storable(tmp_path):
    """The output must satisfy the storage layer's partition contract."""
    from src.data.storage import load_options_chain, save_options_chain

    df = load_kaggle_spy_iv(_wide_bracketed(tmp_path))
    store = tmp_path / "store"
    paths = save_options_chain(df, base_dir=str(store))
    assert len(paths) == df["as_of"].nunique()

    back = load_options_chain("SPY", base_dir=str(store))
    assert len(back) == len(df)


# ── Streamed JSON chains (2024–2025 backfill) ────────────────────────────────


def _json_record(date, exp, strike, kind, bid, ask):
    return {"contractID": f"SPY{kind}{strike}", "symbol": "SPY", "expiration": exp,
            "strike": f"{strike:.2f}", "type": kind, "bid": f"{bid:.2f}", "ask": f"{ask:.2f}",
            "volume": "10", "open_interest": "5", "date": date, "implied_volatility": "0.2"}


@pytest.mark.parametrize("layout", ["lists", "by_date"])
@pytest.mark.parametrize("chunk_size", [7, 64, 1 << 20])
def test_json_days_stream_across_chunk_boundaries(tmp_path, chunk_size, layout):
    import json

    from src.data.kaggle_loader import iter_json_days

    days = [[_json_record(d, "2024-02-16", k, t, 1.0, 1.1)
             for k in (480, 490) for t in ("call", "put")]
            for d in ("2024-01-02", "2024-01-03", "2024-01-04")]
    p = tmp_path / "chains.json"
    # 2024 file: a list of per-day lists; 2025 file: an object keyed by date,
    # with empty lists on holidays.
    body = days if layout == "lists" else {"2024-01-01": [], **{d[0]["date"]: d for d in days}}
    p.write_text(json.dumps(body), encoding="utf-8")
    got = list(iter_json_days(p, chunk_size=chunk_size))
    assert got == days


def test_json_records_get_spot_from_closes_and_canonical_columns():
    from src.data.kaggle_loader import normalise_json_records

    recs = [_json_record("2024-01-02", "2024-02-16", 480, "call", 9.0, 9.2),
            _json_record("2024-01-02", "2024-02-16", 480, "put", 2.0, 2.1),
            _json_record("2024-01-03", "2024-02-16", 480, "put", 2.0, 2.1)]   # no close
    spot = pd.Series([475.3], index=pd.to_datetime(["2024-01-02"]))
    df = normalise_json_records(recs, spot)
    assert len(df) == 2 and (df["underlying_price"] == 475.3).all()
    assert df["mid"].tolist() == pytest.approx([9.1, 2.05])
    assert df["T"].iloc[0] == pytest.approx(45 / 365)
    assert set(df["option_type"]) == {"call", "put"} and (df["ticker"] == "SPY").all()
