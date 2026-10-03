"""Tests for the volatility-forecasting package (src.forecast)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.forecast import dataset as DS
from src.forecast.evaluate import diebold_mariano, qlike_series, scores, walk_forward
from src.forecast.models import HAR_TERMS, LogLinear
from tests.synthetic_chains import gbm_prices


# ── Targets and inputs ───────────────────────────────────────────────────


def test_forward_realised_variance_uses_only_the_next_h_returns():
    ret = pd.Series([0.0, 0.01, -0.02, 0.03, 0.0, 0.0])
    y = DS.forward_realised_variance(ret, 2)
    assert y.iloc[0] == pytest.approx(252 / 2 * (0.01 ** 2 + 0.02 ** 2))
    assert y.iloc[1] == pytest.approx(252 / 2 * (0.02 ** 2 + 0.03 ** 2))
    assert y.iloc[-2:].isna().all()                  # incomplete windows


def test_daily_variance_matches_close_to_close_on_average():
    p = gbm_prices(sigma=0.25, n=400, seed=3)
    gk = DS.daily_variance(p).dropna()
    c2c = 252 * np.log(p["adj_close"]).diff().dropna() ** 2
    assert (gk >= 0).all()
    assert gk.mean() == pytest.approx(c2c.mean(), rel=0.25)
    assert gk.std() < c2c.std()                      # far less noisy


def test_daily_variance_ignores_dividend_adjustment_gaps():
    p = gbm_prices(n=60, seed=1)
    p["adj_close"] = p["close"] * np.where(np.arange(len(p)) < 30, 0.99, 1.0)  # ex-div at day 30
    jump = DS.daily_variance(p).iloc[30]
    assert jump < 5 * DS.daily_variance(p).median()


# ── Walk-forward: no look-ahead ──────────────────────────────────────────


class _Recorder:
    """A model that remembers the last training row it was given."""

    seen: list = []

    def fit(self, train, h):
        _Recorder.seen.append(train.index.max())
        return self

    def predict(self, rows, h, history=None):
        return pd.Series(1.0, index=rows.index)


def test_walk_forward_trains_only_on_closed_targets():
    idx = pd.bdate_range("2015-01-01", periods=600)
    ds = pd.DataFrame({"y_21": 1.0}, index=idx)
    _Recorder.seen = []
    walk_forward(ds, _Recorder, 21, start=str(idx[400].date()), refit_every=50)
    refit_positions = range(400, 600, 50)
    for last_seen, p0 in zip(_Recorder.seen, refit_positions):
        # The newest training row's 21-day window must have closed by the refit date.
        assert idx.get_loc(last_seen) + 21 <= p0


# ── Models ───────────────────────────────────────────────────────────────


def _synthetic(n=1500, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2010-01-01", periods=n)
    lv = np.cumsum(rng.normal(0, 0.05, n)) * 0.2 + np.log(0.03)
    d = pd.DataFrame({"rv_d": np.exp(lv + rng.normal(0, 0.5, n)),
                      "rv_w": np.exp(lv + rng.normal(0, 0.2, n)),
                      "rv_m": np.exp(lv + rng.normal(0, 0.1, n))}, index=idx)
    d["y_21"] = np.exp(0.2 * np.log(d.rv_w) + 0.7 * np.log(d.rv_m) - 0.3 + rng.normal(0, 0.3, n))
    return d


def test_loglinear_recovers_a_log_linear_relation():
    d = _synthetic()
    m = LogLinear("HAR", dict(HAR_TERMS)).fit(d, 21)
    raw = m.coef_[1:] / m.sd_                        # back to unstandardised slopes
    assert raw[1] == pytest.approx(0.2, abs=0.08) and raw[2] == pytest.approx(0.7, abs=0.08)


def test_prediction_interval_is_calibrated_in_sample():
    d = _synthetic(seed=1)
    m = LogLinear("HAR", dict(HAR_TERMS)).fit(d, 21)
    lo, hi = m.interval(d, 21)
    inside = ((d.y_21 >= lo) & (d.y_21 <= hi)).mean()
    assert inside == pytest.approx(0.8, abs=0.03)


def test_smearing_makes_forecasts_mean_unbiased():
    d = _synthetic(seed=2)
    m = LogLinear("HAR", dict(HAR_TERMS)).fit(d, 21)
    assert m.predict(d, 21).mean() == pytest.approx(d.y_21.mean(), rel=0.05)


# ── Scoring ──────────────────────────────────────────────────────────────


def test_qlike_is_zero_for_a_perfect_forecast():
    y = pd.Series([0.01, 0.04, 0.09])
    assert scores(y, y)["qlike"] == pytest.approx(0.0)
    assert scores(y, y)["rmse_vol"] == pytest.approx(0.0)


def test_diebold_mariano_sign():
    rng = np.random.default_rng(0)
    y = pd.Series(np.exp(rng.normal(np.log(0.03), 0.4, 800)))
    good = y * np.exp(rng.normal(0, 0.1, 800))
    bad = y * np.exp(rng.normal(0, 0.6, 800))
    stat, p = diebold_mariano(qlike_series(y, bad), qlike_series(y, good), lag=5)
    assert stat > 3 and p < 0.01                     # positive = second argument better


# ── Production forecaster helpers ────────────────────────────────────────


def test_implied_input_falls_back_and_is_flagged():
    from src.forecast.forecaster import _with_implied

    d = pd.DataFrame({"atm_7d": [0.2, np.nan, np.nan], "atm_14d": [0.21, 0.22, np.nan]})
    out = _with_implied(d, 5)
    assert out["iv_in"].tolist()[:2] == [0.2, 0.22] and np.isnan(out["iv_in"].iloc[2])
    assert out["iv_src"].tolist() == ["atm_7d", "atm_14d", None]


def test_known_rows_exclude_open_targets():
    from src.forecast.forecaster import _known

    idx = pd.bdate_range("2020-01-01", periods=100)
    d = pd.DataFrame({"y_21": 1.0}, index=idx)
    k = _known(d, idx[60], 21)
    assert k.index.max() == idx[60 - 21]
