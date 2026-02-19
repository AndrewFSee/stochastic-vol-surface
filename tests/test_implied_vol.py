"""IV round-trip tests: compute BS price then invert to recover IV."""

import math
import numpy as np
import pytest

from src.surface.implied_vol import bs_price, implied_vol, implied_vol_vectorized


# ── Smoke tests ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("option_type", ["call", "put"])
@pytest.mark.parametrize("sigma", [0.10, 0.20, 0.50, 0.80])
def test_iv_round_trip(option_type, sigma):
    """Inverting BS price should recover the original sigma."""
    S, K, T, r = 100.0, 100.0, 0.25, 0.05
    price = bs_price(S, K, T, r, sigma, option_type)
    recovered = implied_vol(price, S, K, T, r, option_type)
    assert math.isfinite(recovered), f"IV not finite for sigma={sigma}"
    assert abs(recovered - sigma) < 1e-4, f"Round-trip failed: {recovered:.6f} != {sigma}"


def test_bs_price_put_call_parity():
    """Black-Scholes should satisfy put-call parity: C - P = S - K*exp(-rT)."""
    S, K, T, r, sigma = 100.0, 105.0, 0.5, 0.04, 0.25
    call = bs_price(S, K, T, r, sigma, "call")
    put = bs_price(S, K, T, r, sigma, "put")
    parity = S - K * math.exp(-r * T)
    assert abs((call - put) - parity) < 1e-8


def test_bs_price_positive():
    """BS price should be non-negative."""
    assert bs_price(100, 100, 0.25, 0.05, 0.20, "call") > 0
    assert bs_price(100, 100, 0.25, 0.05, 0.20, "put") > 0


def test_iv_nan_for_below_intrinsic():
    """implied_vol should return NaN when price is below intrinsic value."""
    S, K, T, r = 100.0, 90.0, 0.25, 0.05
    intrinsic = max(S - K * math.exp(-r * T), 0)
    iv = implied_vol(intrinsic - 1.0, S, K, T, r, "call")
    assert math.isnan(iv)


def test_iv_nan_for_negative_T():
    """implied_vol should return NaN for expired option."""
    iv = implied_vol(5.0, 100.0, 100.0, 0.0, 0.05, "call")
    assert math.isnan(iv)


def test_implied_vol_vectorized():
    """Vectorized IV should return correct shape and values."""
    S = 100.0
    K = np.array([95.0, 100.0, 105.0])
    T = 0.25
    r = 0.05
    sigma = 0.20
    option_types = ["call", "call", "put"]

    prices = np.array([bs_price(S, k, T, r, sigma, ot) for k, ot in zip(K, option_types)])
    ivs = implied_vol_vectorized(prices, S, K, T, r, option_types)

    assert ivs.shape == (3,)
    for iv in ivs:
        assert math.isfinite(iv), f"Non-finite IV: {iv}"
        assert abs(iv - sigma) < 1e-4


@pytest.mark.parametrize("sigma", [0.01, 0.05, 0.15, 0.30, 0.60, 1.00])
def test_iv_round_trip_wide_sigma(sigma):
    """Round-trip should work across a wide range of volatilities."""
    S, K, T, r = 100.0, 100.0, 1.0, 0.03
    price = bs_price(S, K, T, r, sigma, "call")
    recovered = implied_vol(price, S, K, T, r, "call")
    assert abs(recovered - sigma) < 1e-4


def test_bs_price_otm_call_low_vol():
    """Deep OTM call with low vol should have near-zero price."""
    price = bs_price(S=100, K=200, T=0.1, r=0.05, sigma=0.10, option_type="call")
    assert price < 1e-5


def test_bs_price_deep_itm_call():
    """Deep ITM call price should be close to S - K*exp(-rT)."""
    S, K, T, r, sigma = 200.0, 100.0, 0.25, 0.05, 0.20
    price = bs_price(S, K, T, r, sigma, "call")
    intrinsic = S - K * math.exp(-r * T)
    assert abs(price - intrinsic) < 0.5
