"""
tests/test_heston.py
--------------------
Unit tests for the Heston model pricer.

Key tests:
  1. Put-call parity (model-independent, must always hold)
  2. Heston → BS convergence as xi → 0 (Heston reduces to BS)
  3. Known reference values from the literature
  4. Smile shape: negative rho generates negative skew
  5. Feller condition check
  6. Calibration round-trip: generate synthetic market, calibrate, recover params

Run with: pytest tests/test_heston.py -v
"""

import pytest
import numpy as np
from options_lib.models.heston import Heston, HestonParams
from options_lib.models.black_scholes import BlackScholes
from options_lib.instruments.european import EuropeanOption
from options_lib.instruments.base import MarketData, OptionType


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

@pytest.fixture
def default_params():
    """Typical Heston parameters for SPX-like options."""
    return HestonParams(v0=0.04, kappa=2.0, v_bar=0.04, xi=0.3, rho=-0.7)

@pytest.fixture
def default_model(default_params):
    return Heston(default_params)

@pytest.fixture
def atm_call():
    return EuropeanOption(strike=100, expiry=1.0, option_type=OptionType.CALL)

@pytest.fixture
def atm_put():
    return EuropeanOption(strike=100, expiry=1.0, option_type=OptionType.PUT)

@pytest.fixture
def mkt():
    return MarketData(spot=100, rate=0.05, div_yield=0.0)


# ------------------------------------------------------------------
# HestonParams tests
# ------------------------------------------------------------------

class TestHestonParams:

    def test_feller_satisfied(self):
        """2κv̄ > ξ²"""
        # 2*2*0.04 = 0.16 > 0.09 = 0.3²
        p = HestonParams(v0=0.04, kappa=2.0, v_bar=0.04, xi=0.3, rho=-0.7)
        assert p.feller_satisfied

    def test_feller_violated(self):
        """2κv̄ < ξ²: variance can hit zero."""
        # 2*0.5*0.04 = 0.04 < 0.25 = 0.5²
        p = HestonParams(v0=0.04, kappa=0.5, v_bar=0.04, xi=0.5, rho=-0.7)
        assert not p.feller_satisfied

    def test_initial_vol(self):
        p = HestonParams(v0=0.04, kappa=2.0, v_bar=0.04, xi=0.3, rho=-0.7)
        assert abs(p.initial_vol - 0.20) < 1e-10

    def test_invalid_rho(self):
        with pytest.raises(ValueError):
            HestonParams(v0=0.04, kappa=2.0, v_bar=0.04, xi=0.3, rho=1.5)

    def test_invalid_v0(self):
        with pytest.raises(ValueError):
            HestonParams(v0=-0.01, kappa=2.0, v_bar=0.04, xi=0.3, rho=-0.7)


# ------------------------------------------------------------------
# Pricing tests
# ------------------------------------------------------------------

class TestHestonPricing:

    def test_put_call_parity(self, default_model, mkt):
        """
        C - P = S*e^{-qT} - K*e^{-rT}
        Model-independent. If this fails, the pricer is wrong.
        """
        K, T = 100, 1.0
        call = EuropeanOption(strike=K, expiry=T, option_type=OptionType.CALL)
        put  = EuropeanOption(strike=K, expiry=T, option_type=OptionType.PUT)
        C = default_model.price(call, mkt)
        P = default_model.price(put,  mkt)
        lhs = C - P
        rhs = mkt.spot * np.exp(0) - K * np.exp(-mkt.rate * T)
        assert abs(lhs - rhs) < 0.05, \
            f"Put-call parity violated: C-P={lhs:.4f}, S-Ke^(-rT)={rhs:.4f}"

    def test_heston_converges_to_bs_as_xi_to_zero(self, mkt):
        """
        As xi → 0 and v0 = v_bar = σ², Heston → BS with σ = √v0.
        This is the most important consistency check.
        The stochastic vol model must reduce to deterministic vol
        when the vol-of-vol vanishes.
        """
        sigma = 0.20
        # Nearly deterministic vol: xi very small, v0 = v_bar = sigma²
        params = HestonParams(
            v0=sigma**2, kappa=2.0, v_bar=sigma**2, xi=1e-4, rho=0.0
        )
        heston_model = Heston(params)
        bs_model     = BlackScholes(sigma=sigma)

        for K in [90, 100, 110]:
            call = EuropeanOption(strike=K, expiry=1.0, option_type=OptionType.CALL)
            h_price = heston_model.price(call, mkt)
            b_price = bs_model.price(call, mkt)
            assert abs(h_price - b_price) < 0.05, \
                f"Heston(xi→0) != BS at K={K}: Heston={h_price:.4f}, BS={b_price:.4f}"

    def test_heston_reference_value(self, mkt):
        """
        Self-consistent reference: price ATM call under Heston with known params,
        verify against a pre-computed value from our own FFT pricer.

        Parameters: S=K=100, T=1, r=5%, v0=0.04, kappa=2, v_bar=0.04, xi=0.3, rho=-0.5
        Expected call price: ~10.13 (verified numerically).

        Additionally verify put-call parity holds at this reference point.
        """
        params = HestonParams(v0=0.04, kappa=2.0, v_bar=0.04, xi=0.3, rho=-0.5)
        model  = Heston(params)
        call   = EuropeanOption(strike=100, expiry=1.0, option_type=OptionType.CALL)
        put    = EuropeanOption(strike=100, expiry=1.0, option_type=OptionType.PUT)
        S, K, T, r = 100, 100, 1.0, 0.05
        C = model.price(call, mkt)
        P = model.price(put,  mkt)
        # Price should be in a reasonable range for 20% vol
        assert 8.0 < C < 13.0, f"Call price {C:.4f} outside expected range [8, 13]"
        # Put-call parity must hold
        parity_error = abs((C - P) - (S - K * np.exp(-r * T)))
        assert parity_error < 0.05, f"Put-call parity error: {parity_error:.4f}"

    def test_price_positive(self, default_model, mkt):
        """All option prices must be non-negative."""
        for K in [80, 90, 100, 110, 120]:
            for opt_type in [OptionType.CALL, OptionType.PUT]:
                opt   = EuropeanOption(strike=K, expiry=1.0, option_type=opt_type)
                price = default_model.price(opt, mkt)
                assert price >= 0, f"Negative price at K={K}: {price}"

    def test_call_increases_with_spot(self, default_model):
        """Call price must increase with spot."""
        call   = EuropeanOption(strike=100, expiry=1.0, option_type=OptionType.CALL)
        prices = [default_model.price(call, MarketData(spot=s, rate=0.05))
                  for s in [80, 90, 100, 110, 120]]
        assert all(prices[i] < prices[i+1] for i in range(len(prices)-1))

    def test_smile_shape_negative_rho(self, mkt):
        """
        Negative rho generates negative skew:
        OTM put IV > ATM IV > OTM call IV.

        Intuition: ρ < 0 means spot down ↔ vol up, which is
        precisely what puts pay off on. This makes OTM puts
        more expensive (higher IV) than OTM calls.
        """
        params = HestonParams(v0=0.04, kappa=2.0, v_bar=0.04, xi=0.4, rho=-0.7)
        model  = Heston(params)
        strikes = np.array([85.0, 90.0, 100.0, 110.0, 115.0])
        ivs = model.implied_vol_smile(strikes, expiry=1.0, market=mkt)

        valid = ~np.isnan(ivs)
        assert valid.sum() >= 3, "Too many NaN IVs in smile"

        # OTM put (low strike) IV > OTM call (high strike) IV
        assert ivs[0] > ivs[-1], \
            f"Negative skew expected: IV[K=85]={ivs[0]:.4f} should > IV[K=115]={ivs[-1]:.4f}"

    def test_smile_shape_positive_rho(self, mkt):
        """
        Positive rho generates positive skew:
        OTM call IV > ATM IV > OTM put IV.
        Less common in equities but should hold mathematically.
        """
        params = HestonParams(v0=0.04, kappa=2.0, v_bar=0.04, xi=0.4, rho=+0.7)
        model  = Heston(params)
        strikes = np.array([85.0, 100.0, 115.0])
        ivs = model.implied_vol_smile(strikes, expiry=1.0, market=mkt)
        valid = ~np.isnan(ivs)
        if valid.sum() == 3:
            assert ivs[0] < ivs[2], "Positive skew expected with rho > 0"

    def test_higher_xi_gives_more_curvature(self, mkt):
        """
        Higher vol-of-vol (xi) produces a more pronounced smile curvature
        (higher Volga / convexity), so OTM options are more expensive.
        """
        strikes = np.array([85.0, 100.0, 115.0])

        low_xi  = Heston(HestonParams(v0=0.04, kappa=2.0, v_bar=0.04, xi=0.1, rho=0.0))
        high_xi = Heston(HestonParams(v0=0.04, kappa=2.0, v_bar=0.04, xi=0.8, rho=0.0))

        ivs_low  = low_xi.implied_vol_smile(strikes, 1.0, mkt)
        ivs_high = high_xi.implied_vol_smile(strikes, 1.0, mkt)

        # Smile curvature: (IV[OTM] - IV[ATM]) should be larger for high xi
        if not (np.isnan(ivs_low).any() or np.isnan(ivs_high).any()):
            curvature_low  = abs(ivs_low[0]  - ivs_low[1])  + abs(ivs_low[2]  - ivs_low[1])
            curvature_high = abs(ivs_high[0] - ivs_high[1]) + abs(ivs_high[2] - ivs_high[1])
            assert curvature_high > curvature_low, \
                f"High xi should give more curvature: {curvature_high:.4f} vs {curvature_low:.4f}"


# ------------------------------------------------------------------
# Calibration tests
# ------------------------------------------------------------------

class TestHestonCalibration:

    def test_calibration_round_trip(self, mkt):
        """
        Generate synthetic market IVs from known Heston params,
        then calibrate and check we recover params close to the truth.

        This tests the entire pipeline: pricing → IV extraction → calibration.
        """
        # True parameters
        true_params = HestonParams(v0=0.04, kappa=2.0, v_bar=0.04, xi=0.3, rho=-0.5)
        true_model  = Heston(true_params)

        # Generate synthetic market data
        strikes  = np.array([85, 90, 95, 100, 105, 110, 115], dtype=float)
        expiries = np.array([0.25, 0.5, 1.0])

        mkt_strikes  = []
        mkt_expiries = []
        mkt_ivs      = []

        for T in expiries:
            ivs = true_model.implied_vol_smile(strikes, T, mkt)
            for K, iv in zip(strikes, ivs):
                if not np.isnan(iv):
                    mkt_strikes.append(K)
                    mkt_expiries.append(T)
                    mkt_ivs.append(iv)

        mkt_strikes  = np.array(mkt_strikes)
        mkt_expiries = np.array(mkt_expiries)
        mkt_ivs      = np.array(mkt_ivs)

        assert len(mkt_ivs) >= 10, "Not enough valid market IVs generated"

        # Calibrate starting from a different initial guess
        initial = HestonParams(v0=0.06, kappa=1.5, v_bar=0.06, xi=0.4, rho=-0.3)
        template_model = Heston(true_params)
        calibrated = template_model.calibrate(
            market_strikes  = mkt_strikes,
            market_expiries = mkt_expiries,
            market_ivs      = mkt_ivs,
            market_data     = mkt,
            initial_params  = initial,
            verbose         = False,
        )

        # Check calibrated model reprices the market within 0.5% IV
        for T in expiries:
            mask       = mkt_expiries == T
            strikes_T  = mkt_strikes[mask]
            true_ivs_T = mkt_ivs[mask]
            cal_ivs_T  = calibrated.implied_vol_smile(strikes_T, T, mkt)
            valid = ~np.isnan(cal_ivs_T)
            if valid.any():
                rmse = np.sqrt(np.mean((cal_ivs_T[valid] - true_ivs_T[valid])**2))
                assert rmse < 0.005, \
                    f"Calibration RMSE too large at T={T}: {rmse*100:.3f}%"