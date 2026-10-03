"""Volatility forecasting models with one interface.

Every model forecasts annualised realised variance over the next *h* trading
days (``y_<h>`` in :mod:`src.forecast.dataset`):

    model.fit(train, h)                  # rows whose target is fully known
    model.predict(rows, h, history)      # -> pd.Series of variance forecasts

``history`` is every dataset row up to the last row being predicted; only
models that filter through time (GARCH) use it.  A forecast is NaN where a
model's inputs are missing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def _target(h: int) -> str:
    return f"y_{h}"


def implied_column(h: int) -> str:
    """The implied-variance input that matches a horizon of *h* trading days."""
    return "mkt_vix9d" if h <= 7 else "vs_30d"


def _implied_var(rows: pd.DataFrame, h: int) -> pd.Series:
    col = implied_column(h)
    iv = rows[col] / 100.0 if col.startswith("mkt_") else rows[col]
    return iv ** 2


# ────────────────────────────────────────────────────────────────────────────
# Implied variance
# ────────────────────────────────────────────────────────────────────────────


class ImpliedVariance:
    """The market's own forecast: implied variance, unadjusted.

    Includes the variance risk premium, so it is biased upward by design;
    included as the reference any model should beat.
    """

    name = "Implied (raw)"

    def fit(self, train: pd.DataFrame, h: int) -> "ImpliedVariance":
        return self

    def predict(self, rows: pd.DataFrame, h: int, history=None) -> pd.Series:
        return _implied_var(rows, h)


# ────────────────────────────────────────────────────────────────────────────
# Log-linear regressions (HAR family)
# ────────────────────────────────────────────────────────────────────────────


def _log_var(col: str) -> Callable[[pd.DataFrame, int], pd.Series]:
    return lambda d, h: np.log(d[col])


def _log_implied(d: pd.DataFrame, h: int) -> pd.Series:
    return np.log(_implied_var(d, h))


def _raw(col: str) -> Callable[[pd.DataFrame, int], pd.Series]:
    return lambda d, h: d[col]


HAR_TERMS = {"log_rv_d": _log_var("rv_d"), "log_rv_w": _log_var("rv_w"),
             "log_rv_m": _log_var("rv_m")}
IV_TERMS = {"log_iv": _log_implied}
SURFACE_TERMS = {
    "log_vs_91d": lambda d, h: np.log(d["vs_91d"] ** 2),
    "ts_30_91": _raw("ts_30_91"),
    "rr25_30d": _raw("rr25_30d"),
    "bf25_30d": _raw("bf25_30d"),
    "atm_skew_30d": _raw("atm_skew_30d"),
    "log_vix_ts": lambda d, h: np.log(d["mkt_vix3m"] / d["mkt_vix"]),
    "log_vvix": lambda d, h: np.log(d["mkt_vvix"]),
    "mkt_skew": _raw("mkt_skew"),
    "vrp_30d": _raw("vrp_30d"),
}


@dataclass
class LogLinear:
    """Ridge regression of log realised variance on transformed inputs.

    With the HAR terms alone this is the HAR-RV model (Corsi 2009) in logs.
    Forecasts are de-biased with Duan's smearing estimator,
    ``E[y] = exp(x'β) · mean(exp(residuals))``.
    """

    name: str
    terms: dict
    alpha: float = 0.0
    coef_: Optional[np.ndarray] = field(default=None, repr=False)
    mu_: Optional[np.ndarray] = field(default=None, repr=False)
    sd_: Optional[np.ndarray] = field(default=None, repr=False)
    smear_: float = 1.0
    #: Training residual quantiles in log variance, for prediction intervals.
    resid_q_: dict = field(default_factory=dict, repr=False)

    def design(self, d: pd.DataFrame, h: int) -> pd.DataFrame:
        with np.errstate(divide="ignore", invalid="ignore"):
            X = pd.DataFrame({k: f(d, h) for k, f in self.terms.items()}, index=d.index)
        return X.replace([np.inf, -np.inf], np.nan)

    def fit(self, train: pd.DataFrame, h: int) -> "LogLinear":
        X = self.design(train, h)
        y = np.log(train[_target(h)])
        ok = X.notna().all(axis=1) & np.isfinite(y)
        X, y = X[ok].to_numpy(), y[ok].to_numpy()
        self.mu_, self.sd_ = X.mean(0), X.std(0) + 1e-12
        Z = np.column_stack([np.ones(len(X)), (X - self.mu_) / self.sd_])
        P = self.alpha * np.eye(Z.shape[1])
        P[0, 0] = 0.0  # never shrink the intercept
        self.coef_ = np.linalg.solve(Z.T @ Z + P, Z.T @ y)
        resid = y - Z @ self.coef_
        self.smear_ = float(np.mean(np.exp(resid)))
        self.resid_q_ = {q: float(np.quantile(resid, q)) for q in (0.05, 0.1, 0.5, 0.9, 0.95)}
        return self

    def predict_log(self, rows: pd.DataFrame, h: int) -> pd.Series:
        """Fitted log variance (no bias correction)."""
        X = self.design(rows, h)
        Z = np.column_stack([np.ones(len(X)), (X.to_numpy() - self.mu_) / self.sd_])
        return pd.Series(Z @ self.coef_, index=rows.index)

    def predict(self, rows: pd.DataFrame, h: int, history=None) -> pd.Series:
        return np.exp(self.predict_log(rows, h)) * self.smear_

    def interval(self, rows: pd.DataFrame, h: int, lo: float = 0.1, hi: float = 0.9):
        """Prediction interval for realised variance from training residual quantiles."""
        base = self.predict_log(rows, h)
        return np.exp(base + self.resid_q_[lo]), np.exp(base + self.resid_q_[hi])

    def coefficients(self) -> pd.Series:
        """Standardised coefficients (effect of a one-sd move, in log variance)."""
        return pd.Series(self.coef_[1:], index=list(self.terms))


def har() -> LogLinear:
    return LogLinear("HAR", dict(HAR_TERMS))


def implied_regression() -> LogLinear:
    return LogLinear("Implied (recalibrated)", dict(IV_TERMS))


def har_iv() -> LogLinear:
    return LogLinear("HAR + implied", {**HAR_TERMS, **IV_TERMS})


def har_x(alpha: float = 5.0) -> LogLinear:
    return LogLinear("HAR + surface", {**HAR_TERMS, **IV_TERMS, **SURFACE_TERMS}, alpha=alpha)


# ────────────────────────────────────────────────────────────────────────────
# GARCH
# ────────────────────────────────────────────────────────────────────────────


@dataclass
class Garch:
    """GARCH(1,1) or GJR-GARCH(1,1,1) with Student-t errors, on daily returns.

    Parameters are re-estimated at each refit; between refits the filter runs
    forward with fixed parameters, so each forecast uses returns up to its
    own date only.
    """

    kind: str = "gjr"
    params_: Optional[pd.Series] = field(default=None, repr=False)

    @property
    def name(self) -> str:
        return "GJR-GARCH" if self.kind == "gjr" else "GARCH"

    def _model(self, ret: pd.Series):
        from arch import arch_model

        return arch_model(100 * ret.dropna(), mean="Constant", vol="GARCH", p=1,
                          o=1 if self.kind == "gjr" else 0, q=1, dist="t", rescale=False)

    def fit(self, train: pd.DataFrame, h: int) -> "Garch":
        res = self._model(train["ret"]).fit(disp="off", show_warning=False)
        self.params_ = res.params
        return self

    def predict(self, rows: pd.DataFrame, h: int, history: pd.DataFrame) -> pd.Series:
        fixed = self._model(history["ret"]).fix(self.params_)
        fc = fixed.forecast(horizon=h, start=rows.index[0], reindex=False)
        total = fc.variance.sum(axis=1) / 1e4          # percent² → decimal²
        return (TRADING_DAYS / h * total).reindex(rows.index)


# ────────────────────────────────────────────────────────────────────────────
# Gradient boosting
# ────────────────────────────────────────────────────────────────────────────


@dataclass
class Boosting:
    """Gradient-boosted trees on the HAR and surface inputs (log target)."""

    name: str = "Gradient boosting"
    terms: dict = field(default_factory=lambda: {**HAR_TERMS, **IV_TERMS, **SURFACE_TERMS})
    model_: object = field(default=None, repr=False)
    smear_: float = 1.0

    def _X(self, d, h):
        with np.errstate(divide="ignore", invalid="ignore"):
            X = pd.DataFrame({k: f(d, h) for k, f in self.terms.items()}, index=d.index)
        return X.replace([np.inf, -np.inf], np.nan)

    def fit(self, train: pd.DataFrame, h: int) -> "Boosting":
        from sklearn.ensemble import HistGradientBoostingRegressor

        X, y = self._X(train, h), np.log(train[_target(h)])
        ok = X.notna().all(axis=1) & np.isfinite(y)
        self.model_ = HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.03, max_leaf_nodes=15,
            min_samples_leaf=50, l2_regularization=1.0, random_state=0,
        ).fit(X[ok], y[ok])
        self.smear_ = float(np.mean(np.exp(y[ok] - self.model_.predict(X[ok]))))
        return self

    def predict(self, rows: pd.DataFrame, h: int, history=None) -> pd.Series:
        X = self._X(rows, h)
        out = pd.Series(np.nan, index=rows.index)
        ok = X.notna().all(axis=1)
        if ok.any():
            out[ok] = np.exp(self.model_.predict(X[ok])) * self.smear_
        return out


def standard_models() -> dict[str, Callable[[], object]]:
    """Factories for every core model, keyed by display name."""
    return {
        "Implied (raw)": ImpliedVariance,
        "Implied (recalibrated)": implied_regression,
        "GARCH": lambda: Garch("garch"),
        "GJR-GARCH": lambda: Garch("gjr"),
        "HAR": har,
        "HAR + implied": har_iv,
        "HAR + surface": har_x,
        "Gradient boosting": Boosting,
    }
