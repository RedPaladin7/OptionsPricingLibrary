"""
tests/test_black_scholes.py
---------------------------
Unit tests for the Black-Scholes pricer and Greeks.

Every test verifies against a known analytical value or a
fundamental mathematical identity. This is how you know
your implementation is correct — not by running it and
looking at the output, but by checking it against ground truth.

Run with:  pytest tests/test_black_scholes.py -v
"""

import pytest
import numpy as np
from options_lib.models.black_scholes import BlackScholes
from options_lib.models.implied_vol import implied_vol_bs
from options_lib.instruments.european import EuropeanOption
from options_lib.instruments.base import MarketData, OptionType


# ------------------------------------------------------------------
# Fixtures — reusable test setups
# ------------------------------------------------------------------

@pytest.fixture
def atm_call():
    """ATM call: S=K=100, T=1yr, r=5%, q=0, sigma=20%"""
    return EuropeanOption(strike=100.0, expiry=1.0, option_type=OptionType.CALL)

@pytest.fixture
def atm_put():
    return EuropeanOption(strike=100.0, expiry=1.0, option_type=OptionType.PUT)

@pytest.fixture
def mkt():
    return MarketData(spot=100.0, rate=0.05, div_yield=0.0)

@pytest.fixture
def model():
    return BlackScholes(sigma=0.20)


# ------------------------------------------------------------------
# Pricing tests
# ------------------------------------------------------------------

class TestPricing:

    def test_atm_call_known_value(self, atm_call, mkt, model):
        """
        ATM BS call with S=K=100, T=1, r=5%, σ=20% has a known value.
        Reference: 10.4506 (verified against standard BS tables).
        """
        price = model.price(atm_call, mkt)
        assert abs(price - 10.4506) < 0.001, f"Expected ~10.4506, got {price:.4f}"

    def test_put_call_parity(self, mkt, model):
        """
        Put-call parity: C - P = S*e^{-qT} - K*e^{-rT}
        This is a model-independent identity. If it fails, the pricer
        has an error in either the call or put formula.
        """
        S, K, T, r, q = 100, 100, 1.0, 0.05, 0.0
        call = EuropeanOption(strike=K, expiry=T, option_type=OptionType.CALL)
        put  = EuropeanOption(strike=K, expiry=T, option_type=OptionType.PUT)
        C = model.price(call, mkt)
        P = model.price(put,  mkt)
        lhs = C - P
        rhs = S * np.exp(-q * T) - K * np.exp(-r * T)
        assert abs(lhs - rhs) < 1e-10, f"Put-call parity violated: {lhs:.6f} != {rhs:.6f}"

    def test_intrinsic_value_floor(self, mkt, model):
        """
        Option price must be >= its intrinsic value.
        Deep ITM call: C >= max(S - K, 0)
        """
        deep_itm = EuropeanOption(strike=50.0, expiry=1.0, option_type=OptionType.CALL)
        price = model.price(deep_itm, mkt)
        intrinsic = max(mkt.spot - deep_itm.strike, 0)
        assert price >= intrinsic - 1e-10

    def test_deep_otm_call_near_zero(self, mkt, model):
        """Deep OTM call should have near-zero value."""
        otm = EuropeanOption(strike=500.0, expiry=0.1, option_type=OptionType.CALL)
        price = model.price(otm, mkt)
        assert price < 0.01, f"Deep OTM call price should be ~0, got {price:.6f}"

    def test_zero_vol_call(self, mkt):
        """
        With σ→0, call price = max(S*e^{-qT} - K*e^{-rT}, 0).
        The option has no time value — only intrinsic.
        """
        model_low = BlackScholes(sigma=1e-6)
        call = EuropeanOption(strike=90.0, expiry=1.0, option_type=OptionType.CALL)
        price = model_low.price(call, mkt)
        intrinsic = max(mkt.spot * np.exp(0) - 90 * np.exp(-mkt.rate * 1.0), 0)
        assert abs(price - intrinsic) < 0.01

    def test_call_price_increases_with_vol(self, atm_call, mkt):
        """
        Call price is monotonically increasing in σ.
        This is because higher vol increases the probability of
        large moves, which benefits the option holder (asymmetric payoff).
        """
        prices = [BlackScholes(sigma=s).price(atm_call, mkt) for s in [0.1, 0.2, 0.3, 0.4]]
        assert all(prices[i] < prices[i+1] for i in range(len(prices)-1))

    def test_call_price_increases_with_spot(self, model):
        """Call price is monotonically increasing in spot."""
        call = EuropeanOption(strike=100, expiry=1.0, option_type=OptionType.CALL)
        prices = [model.price(call, MarketData(spot=s, rate=0.05)) for s in [80, 90, 100, 110, 120]]
        assert all(prices[i] < prices[i+1] for i in range(len(prices)-1))

    def test_dividend_reduces_call(self, atm_call, model):
        """
        Higher dividend yield reduces call value (stock expected to
        drop by dividend amount, reducing expected terminal price).
        """
        no_div  = model.price(atm_call, MarketData(spot=100, rate=0.05, div_yield=0.0))
        with_div = model.price(atm_call, MarketData(spot=100, rate=0.05, div_yield=0.03))
        assert with_div < no_div


# ------------------------------------------------------------------
# Greeks tests
# ------------------------------------------------------------------

class TestGreeks:

    def test_call_delta_bounds(self, atm_call, mkt, model):
        """Call delta must be in (0, 1)."""
        d = model.delta(atm_call, mkt)
        assert 0 < d < 1, f"Call delta out of bounds: {d}"

    def test_put_delta_bounds(self, atm_put, mkt, model):
        """Put delta must be in (-1, 0)."""
        d = model.delta(atm_put, mkt)
        assert -1 < d < 0, f"Put delta out of bounds: {d}"

    def test_atm_call_delta_approx_half(self, atm_call, mkt, model):
        """
        ATM call delta ≈ 0.5 (slightly above due to the vol adjustment in d1).
        Intuition: at the money, the option has ~50% chance of expiring ITM.
        """
        d = model.delta(atm_call, mkt)
        assert 0.5 < d < 0.65, f"ATM call delta should be ~0.5-0.6, got {d:.4f}"

    def test_call_put_delta_relationship(self, mkt, model):
        """
        Call delta - Put delta = e^{-qT}  (from put-call parity differentiated w.r.t. S)
        """
        call = EuropeanOption(strike=100, expiry=1.0, option_type=OptionType.CALL)
        put  = EuropeanOption(strike=100, expiry=1.0, option_type=OptionType.PUT)
        d_call = model.delta(call, mkt)
        d_put  = model.delta(put,  mkt)
        T, q = call.expiry, mkt.div_yield
        expected = np.exp(-q * T)
        assert abs((d_call - d_put) - expected) < 1e-10

    def test_gamma_positive(self, atm_call, mkt, model):
        """Gamma is always positive for long options."""
        g = model.gamma(atm_call, mkt)
        assert g > 0

    def test_call_put_gamma_equal(self, mkt, model):
        """Call and put with same parameters have identical gamma."""
        call = EuropeanOption(strike=100, expiry=1.0, option_type=OptionType.CALL)
        put  = EuropeanOption(strike=100, expiry=1.0, option_type=OptionType.PUT)
        assert abs(model.gamma(call, mkt) - model.gamma(put, mkt)) < 1e-12

    def test_vega_positive(self, atm_call, mkt, model):
        """Vega is always positive for long options."""
        v = model.vega(atm_call, mkt)
        assert v > 0

    def test_call_put_vega_equal(self, mkt, model):
        """Call and put with same parameters have identical vega."""
        call = EuropeanOption(strike=100, expiry=1.0, option_type=OptionType.CALL)
        put  = EuropeanOption(strike=100, expiry=1.0, option_type=OptionType.PUT)
        assert abs(model.vega(call, mkt) - model.vega(put, mkt)) < 1e-10

    def test_theta_negative(self, atm_call, mkt, model):
        """Theta is negative for long options (time value decays)."""
        th = model.theta(atm_call, mkt)
        assert th < 0, f"Theta should be negative, got {th}"

    def test_bs_pde_residual(self, atm_call, mkt, model):
        """
        The BS PDE must hold: Θ + ½σ²S²Γ + (r-q)SΔ - rV = 0
        A non-zero residual means the Greeks are internally inconsistent.
        """
        residual = model.verify_pde(atm_call, mkt)
        assert abs(residual) < 1e-8, f"BS PDE not satisfied. Residual: {residual:.2e}"

    def test_delta_matches_finite_difference(self, atm_call, mkt, model):
        """
        Analytical delta should match numerical (bump-and-reprice) delta.
        If they diverge, there's a bug in the analytical formula.
        """
        h = 0.01
        analytical = model.delta(atm_call, mkt)
        up   = model.price(atm_call, MarketData(mkt.spot + h, mkt.rate, mkt.div_yield))
        down = model.price(atm_call, MarketData(mkt.spot - h, mkt.rate, mkt.div_yield))
        numerical  = (up - down) / (2 * h)
        assert abs(analytical - numerical) < 1e-6, \
            f"Delta mismatch: analytical={analytical:.6f}, numerical={numerical:.6f}"

    def test_gamma_matches_finite_difference(self, atm_call, mkt, model):
        """Analytical gamma vs numerical second derivative."""
        h = 0.5
        analytical = model.gamma(atm_call, mkt)
        mid  = model.price(atm_call, mkt)
        up   = model.price(atm_call, MarketData(mkt.spot + h, mkt.rate, mkt.div_yield))
        down = model.price(atm_call, MarketData(mkt.spot - h, mkt.rate, mkt.div_yield))
        numerical  = (up - 2 * mid + down) / h**2
        assert abs(analytical - numerical) < 1e-5, \
            f"Gamma mismatch: analytical={analytical:.6f}, numerical={numerical:.6f}"

    def test_vega_matches_finite_difference(self, atm_call, mkt, model):
        """Analytical vega vs bump-and-reprice in sigma."""
        h = 0.001
        analytical = model.vega(atm_call, mkt)
        up   = BlackScholes(sigma=model.sigma + h).price(atm_call, mkt)
        down = BlackScholes(sigma=model.sigma - h).price(atm_call, mkt)
        numerical  = (up - down) / (2 * h)
        # Tolerance of 1e-4: central difference with h=0.001 has O(h²) error
        # which for vega ~37 gives ~3e-5. This is numerical precision, not a bug.
        assert abs(analytical - numerical) < 1e-4, \
            f"Vega mismatch: analytical={analytical:.6f}, numerical={numerical:.6f}"


# ------------------------------------------------------------------
# Implied vol tests
# ------------------------------------------------------------------

class TestImpliedVol:

    @pytest.mark.parametrize("sigma_true", [0.10, 0.20, 0.30, 0.50, 0.80])
    def test_round_trip(self, sigma_true):
        """
        Round-trip test: price at sigma_true, then recover sigma_true from price.
        This tests both the pricer and the IV solver simultaneously.
        """
        call  = EuropeanOption(strike=100, expiry=1.0, option_type=OptionType.CALL)
        mkt   = MarketData(spot=100, rate=0.05)
        model = BlackScholes(sigma=sigma_true)
        price = model.price(call, mkt)
        iv    = implied_vol_bs(market_price=price, instrument=call, market=mkt)
        assert abs(iv - sigma_true) < 1e-5, \
            f"IV round-trip failed: true={sigma_true}, recovered={iv:.6f}"

    @pytest.mark.parametrize("strike", [80, 90, 100, 110, 120])
    def test_round_trip_across_strikes(self, strike):
        """IV inversion works for ITM, ATM, and OTM options."""
        call  = EuropeanOption(strike=strike, expiry=0.5, option_type=OptionType.CALL)
        mkt   = MarketData(spot=100, rate=0.05)
        sigma = 0.25
        price = BlackScholes(sigma=sigma).price(call, mkt)
        iv    = implied_vol_bs(market_price=price, instrument=call, market=mkt, sigma_init=0.3)
        assert abs(iv - sigma) < 1e-5

    def test_put_implied_vol(self):
        """IV inversion works for puts too."""
        put   = EuropeanOption(strike=100, expiry=1.0, option_type=OptionType.PUT)
        mkt   = MarketData(spot=100, rate=0.05)
        sigma = 0.20
        price = BlackScholes(sigma=sigma).price(put, mkt)
        iv    = implied_vol_bs(market_price=price, instrument=put, market=mkt)
        assert abs(iv - sigma) < 1e-5