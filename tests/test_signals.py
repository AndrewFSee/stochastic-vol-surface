"""Signal logic tests."""

import numpy as np
import pytest

from src.signals.skew_signals import (
    SkewSignal,
    SkewSnapshot,
    compute_skew_signal,
)
from src.signals.term_structure import (
    TermStructureSignal,
    TermStructureSnapshot,
    detect_inversion,
)
from src.signals.regime_vol import VolRegime, classify_regime, regime_time_series
from src.signals.composite import (
    CompositeSignal,
    TradeRecommendation,
    aggregate_signals,
)

import pandas as pd


# ── Skew signals ──────────────────────────────────────────────────────────────

def test_skew_steepening():
    """Above-average skew should return STEEPENING."""
    rng = np.random.default_rng(42)
    history = rng.normal(0.03, 0.005, 100)  # history with some variance
    current = float(np.mean(history) + 3.0 * np.std(history, ddof=1))  # z ≈ +3
    sig = compute_skew_signal(current, history, z_threshold=1.5)
    assert sig == SkewSignal.STEEPENING


def test_skew_flattening():
    """Below-average skew should return FLATTENING."""
    rng = np.random.default_rng(42)
    history = rng.normal(0.05, 0.005, 100)
    current = float(np.mean(history) - 3.0 * np.std(history, ddof=1))  # z ≈ -3
    sig = compute_skew_signal(current, history, z_threshold=1.5)
    assert sig == SkewSignal.FLATTENING


def test_skew_neutral():
    """Near-average skew should return NEUTRAL."""
    history = np.random.default_rng(42).normal(0.03, 0.005, 100)
    current = float(np.mean(history))  # exactly at mean
    sig = compute_skew_signal(current, history, z_threshold=1.5)
    assert sig == SkewSignal.NEUTRAL


def test_skew_snapshot_properties():
    snap = SkewSnapshot(tenor=0.25, put_25d_iv=0.22, atm_iv=0.18, call_25d_iv=0.16)
    assert snap.put_call_skew == pytest.approx(0.06)
    assert snap.risk_reversal == pytest.approx(-0.06)
    assert snap.butterfly == pytest.approx(0.01)


def test_skew_neutral_short_history():
    """Should return NEUTRAL when history has fewer than 2 points."""
    sig = compute_skew_signal(0.05, np.array([0.03]))
    assert sig == SkewSignal.NEUTRAL


# ── Term structure ────────────────────────────────────────────────────────────

def test_term_structure_normal():
    tenors = np.array([0.25, 0.5, 1.0, 2.0])
    ivs = np.array([0.15, 0.17, 0.19, 0.21])  # upward sloping
    snap = TermStructureSnapshot(tenors=tenors, atm_ivs=ivs)
    assert snap.classify() == TermStructureSignal.NORMAL


def test_term_structure_inverted():
    # Use evenly spaced tenors with linearly decreasing IVs → near-zero curvature
    tenors = np.array([0.5, 1.0, 1.5, 2.0])
    ivs = np.array([0.30, 0.26, 0.22, 0.18])  # exactly linear → curvature ≈ 0
    snap = TermStructureSnapshot(tenors=tenors, atm_ivs=ivs)
    assert snap.classify() == TermStructureSignal.INVERTED


def test_detect_inversion():
    tenors = np.array([0.25, 0.5, 1.0])
    ivs = np.array([0.30, 0.20, 0.22])  # 0.25y > 0.5y → inversion
    inversions = detect_inversion(tenors, ivs, threshold=0.01)
    assert len(inversions) >= 1
    assert inversions[0] == (0.25, 0.5)


def test_term_structure_slope_sign():
    tenors = np.array([0.25, 0.5, 1.0])
    ivs_up = np.array([0.15, 0.18, 0.21])
    ivs_down = np.array([0.21, 0.18, 0.15])
    assert TermStructureSnapshot(tenors, ivs_up).slope > 0
    assert TermStructureSnapshot(tenors, ivs_down).slope < 0


# ── Vol regime ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("vix,expected", [
    (10.0, VolRegime.LOW),
    (15.0, VolRegime.LOW),    # equal to threshold → LOW
    (20.0, VolRegime.NORMAL),
    (25.0, VolRegime.HIGH),
    (35.0, VolRegime.CRISIS),
    (50.0, VolRegime.CRISIS),
])
def test_classify_regime(vix, expected):
    assert classify_regime(vix) == expected


def test_regime_time_series():
    vix = pd.Series([12.0, 18.0, 28.0, 40.0], index=pd.date_range("2024-01-01", periods=4))
    result = regime_time_series(vix)
    assert list(result) == [VolRegime.LOW, VolRegime.NORMAL, VolRegime.HIGH, VolRegime.CRISIS]


# ── Composite signal ──────────────────────────────────────────────────────────

def _make_composite(vix: float = 18.0, atm_z: float = 0.0, skew: float = 0.03) -> CompositeSignal:
    history_skews = np.full(100, 0.03)
    history_vols = np.full(100, 0.18)
    tenors = np.array([0.25, 0.5, 1.0])
    ivs = np.array([0.16, 0.17, 0.18])
    ts_snap = TermStructureSnapshot(tenors=tenors, atm_ivs=ivs)
    current_atm = 0.18 + atm_z * 0.01  # small variation
    return aggregate_signals(
        current_skew=skew,
        historical_skews=history_skews,
        ts_snapshot=ts_snap,
        current_vix=vix,
        current_atm_vol=current_atm,
        historical_atm_vols=history_vols,
    )


def test_composite_returns_valid_recommendation():
    sig = _make_composite()
    assert isinstance(sig.recommendation, TradeRecommendation)


def test_composite_crisis_regime_buys_vol():
    """High VIX (crisis) should tilt towards BUY_VOL."""
    sig = _make_composite(vix=45.0)
    assert sig.vol_regime == VolRegime.CRISIS
    # The dominant contribution should favour buying vol
    assert sig.recommendation in (TradeRecommendation.BUY_VOL, TradeRecommendation.NEUTRAL)


def test_composite_confidence_in_range():
    sig = _make_composite()
    assert 0.0 <= sig.confidence <= 1.0


def test_composite_rationale_nonempty():
    sig = _make_composite()
    assert len(sig.rationale) > 0
