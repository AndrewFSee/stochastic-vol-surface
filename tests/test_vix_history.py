"""Tests for consolidating the overlapping daily VIX snapshot windows."""

import numpy as np
import pandas as pd
import pytest

from src.data import vix_family as V

COLS = ["VIX", "VIX3M", "VIX9D", "SKEW", "VVIX"]


def _write_snapshot(vix_dir, as_of, frame):
    """Write one ``vix_YYYY-MM-DD.parquet`` snapshot file."""
    frame = frame.rename_axis("date")
    frame.to_parquet(vix_dir / f"vix_{as_of}.parquet", engine="pyarrow")


@pytest.fixture
def vix_dir(tmp_path):
    """Two overlapping snapshots mimicking the real collector.

    Each run stores a 5-day window; on the run date itself SKEW has not
    settled, and VIX3M/VIX9D are only populated for the newest row.
    """
    d = tmp_path / "vix"
    d.mkdir()

    day1 = pd.DataFrame(
        {
            "VIX":   [20.0, 21.0],
            "VIX3M": [np.nan, 22.0],
            "VIX9D": [np.nan, 19.0],
            "SKEW":  [140.0, np.nan],   # run-date SKEW not settled
            "VVIX":  [100.0, 101.0],
        },
        index=pd.to_datetime(["2026-01-05", "2026-01-06"]),
    )
    _write_snapshot(d, "2026-01-06", day1)

    # Next run: overlaps 01-06 and now carries its settled values.
    day2 = pd.DataFrame(
        {
            "VIX":   [21.0, 22.0],
            "VIX3M": [22.5, 23.0],       # 01-06 backfilled
            "VIX9D": [19.5, 20.0],
            "SKEW":  [141.0, np.nan],    # 01-06 SKEW now settled
            "VVIX":  [101.0, 102.0],
        },
        index=pd.to_datetime(["2026-01-06", "2026-01-07"]),
    )
    _write_snapshot(d, "2026-01-07", day2)
    return d


# ── Consolidation ────────────────────────────────────────────────────────


def test_history_unions_all_dates(vix_dir):
    h = V.load_vix_history(str(vix_dir))
    assert list(h.index.strftime("%Y-%m-%d")) == ["2026-01-05", "2026-01-06", "2026-01-07"]


def test_history_is_sorted(vix_dir):
    h = V.load_vix_history(str(vix_dir))
    assert h.index.is_monotonic_increasing


def test_later_snapshot_backfills_missing_values(vix_dir):
    """The key behaviour: overlap recovers values that were NaN when first seen."""
    h = V.load_vix_history(str(vix_dir))
    assert h.loc["2026-01-06", "SKEW"] == pytest.approx(141.0)
    assert h.loc["2026-01-06", "VIX3M"] == pytest.approx(22.5)
    assert h.loc["2026-01-06", "VIX9D"] == pytest.approx(19.5)


def test_values_present_in_only_one_snapshot_survive(vix_dir):
    h = V.load_vix_history(str(vix_dir))
    assert h.loc["2026-01-05", "SKEW"] == pytest.approx(140.0)


def test_unsettled_latest_value_stays_null(vix_dir):
    """Nothing fabricates a value that no snapshot ever carried."""
    h = V.load_vix_history(str(vix_dir))
    assert pd.isna(h.loc["2026-01-07", "SKEW"])


def test_consolidation_beats_any_single_snapshot(vix_dir):
    """Consolidated history has strictly fewer nulls than the raw concat."""
    h = V.load_vix_history(str(vix_dir))
    raw = pd.concat(
        [pd.read_parquet(p) for p in sorted(vix_dir.glob("vix_*.parquet"))]
    )
    assert h.isna().sum().sum() < raw.isna().sum().sum()


def test_dropna_vix_filters_headline_nulls(tmp_path):
    d = tmp_path / "vix"
    d.mkdir()
    _write_snapshot(d, "2026-01-06", pd.DataFrame(
        {"VIX": [np.nan, 21.0], "VVIX": [100.0, 101.0]},
        index=pd.to_datetime(["2026-01-05", "2026-01-06"]),
    ))
    assert len(V.load_vix_history(str(d))) == 2
    assert len(V.load_vix_history(str(d), dropna_vix=True)) == 1


def test_empty_dir_returns_empty_frame(tmp_path):
    empty = tmp_path / "none"
    empty.mkdir()
    assert V.load_vix_history(str(empty)).empty


def test_save_history_roundtrips(vix_dir):
    path = V.save_vix_history(str(vix_dir))
    assert path.exists()
    saved = pd.read_parquet(path)
    assert len(saved) == 3


# ── Derived signals ──────────────────────────────────────────────────────


def test_term_structure_signal_columns(vix_dir):
    s = V.vix_term_structure_signal(str(vix_dir))
    for col in ("vix_ts_slope", "vix_short_ts", "vvix_vix"):
        assert col in s.columns


def test_term_structure_slope_is_ratio(vix_dir):
    s = V.vix_term_structure_signal(str(vix_dir))
    row = s.loc["2026-01-07"]
    assert row["vix_ts_slope"] == pytest.approx(row["VIX3M"] / row["VIX"])


def test_contango_gives_slope_above_one(vix_dir):
    s = V.vix_term_structure_signal(str(vix_dir))
    # VIX3M (23.0) > VIX (22.0) on 01-07 → contango
    assert s.loc["2026-01-07", "vix_ts_slope"] > 1.0


def test_signal_on_empty_dir_is_empty(tmp_path):
    empty = tmp_path / "none"
    empty.mkdir()
    assert V.vix_term_structure_signal(str(empty)).empty


def test_snapshots_take_precedence_over_backfill(tmp_path):
    """The backfill fills history; collected snapshots win where they overlap."""
    import pandas as pd

    from src.data.vix_family import BACKFILL_FILENAME, load_vix_history

    idx = pd.DatetimeIndex(pd.to_datetime(["2026-01-05", "2026-01-06"]), name="date")
    pd.DataFrame({"VIX": [10.0, 11.0], "SKEW": [130.0, 131.0]}, index=idx).to_parquet(
        tmp_path / BACKFILL_FILENAME)
    pd.DataFrame({"VIX": [99.0], "SKEW": [None]}, index=idx[1:]).to_parquet(
        tmp_path / "vix_2026-01-06.parquet")

    h = load_vix_history(str(tmp_path))
    assert h.loc["2026-01-05", "VIX"] == 10.0     # only in the backfill
    assert h.loc["2026-01-06", "VIX"] == 99.0     # snapshot wins
    assert h.loc["2026-01-06", "SKEW"] == 131.0   # backfill fills the snapshot's gap


def test_backfill_merges_rather_than_replaces(tmp_path, monkeypatch):
    """A short re-run must not erase a longer backfill already stored."""
    import pandas as pd

    from src.data import vix_family as V

    def fake_fetch(start, end):
        idx = pd.bdate_range(start, end, inclusive="left", name="date")
        return pd.DataFrame({"VIX": 20.0}, index=idx)

    monkeypatch.setattr(V, "fetch_vix_family", fake_fetch)
    V.backfill_vix_history("2010-01-04", "2010-01-09", str(tmp_path))
    V.backfill_vix_history("2024-01-02", "2024-01-05", str(tmp_path))
    h = pd.read_parquet(tmp_path / V.BACKFILL_FILENAME)
    assert pd.Timestamp("2010-01-04") in h.index and pd.Timestamp("2024-01-02") in h.index
