"""Calendar spread arbitrage detection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CalendarArbitrageAlert:
    tenor_near: float
    tenor_far: float
    log_moneyness: float
    tv_near: float
    tv_far: float
    violation_bps: float  # (tv_near - tv_far) * 10000


def detect_calendar_arbitrage(
    tenors: np.ndarray,
    total_var_grid: np.ndarray,  # (n_tenors, n_k)
    k_grid: np.ndarray,
    threshold_bps: float = 30.0,
) -> list[CalendarArbitrageAlert]:
    """Find calendar spread arbitrage (total variance not monotone in tenor).

    Parameters
    ----------
    tenors         : (n_tenors,) sorted array of tenors in years
    total_var_grid : (n_tenors, n_k) total variance w = sigma^2 * T
    k_grid         : (n_k,) log-moneyness grid
    threshold_bps  : minimum violation (in bps of total variance) to flag

    Returns
    -------
    List of CalendarArbitrageAlert for each (tenor_pair, k_point) violation
    """
    n_tenors = len(tenors)
    alerts: list[CalendarArbitrageAlert] = []

    for i in range(n_tenors - 1):
        T1, T2 = tenors[i], tenors[i + 1]
        w1, w2 = total_var_grid[i], total_var_grid[i + 1]
        violations = w1 - w2  # positive = violation
        for j in range(len(k_grid)):
            v = violations[j]
            if v > threshold_bps / 10_000:
                alerts.append(
                    CalendarArbitrageAlert(
                        tenor_near=float(T1),
                        tenor_far=float(T2),
                        log_moneyness=float(k_grid[j]),
                        tv_near=float(w1[j]),
                        tv_far=float(w2[j]),
                        violation_bps=float(v * 10_000),
                    )
                )
    return alerts


def calendar_spread_value(
    iv_near: float,
    iv_far: float,
    T_near: float,
    T_far: float,
) -> float:
    """Calendar spread value in total variance space: w_far - w_near.

    Positive = no arbitrage (far > near).
    Negative = potential calendar arbitrage.
    """
    w_near = iv_near ** 2 * T_near
    w_far = iv_far ** 2 * T_far
    return w_far - w_near
