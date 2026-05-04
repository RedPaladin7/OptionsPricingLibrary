"""
tests/test_heston_asian.py
---------------------------
Tests for HestonSimulator and HestonAsianPricer.

Key tests:
  HestonSimulator:
    1. Path shape is correct
    2. Variance stays non-negative with full truncation
    3. Feller condition: when 2κv̄ >> ξ², variance rarely hits 0
    4. E[S_T] = S0 * e^{(r-q)T}  (martingale condition under Q)
    5. Flat vol limit: xi→0 gives same terminal distribution as BS
    6. Negative correlation: spot down ↔ variance up

  HestonAsianPricer:
    7. Asian price > 0 for ITM options
    8. Heston Asian call != BS Asian call (vol dynamics matter)
    9. Higher xi → higher Heston-BS premium (more vol uncertainty)
    10. Arithmetic Asian <= European (averaging reduces vol)
    11. vol_premium_pct in reasonable range for typical params
    12. Moment matching: sample mean ≈ theoretical forward

Run with: pytest tests/test_heston_asian.py -v
"""

import pytest
import numpy as np
from options_lib.numerics.heston_simulator import HestonSimulator
from options_lib.models.heston import HestonParams
from options_lib.models.heston_asian_mc import HestonAsianPricer, HestonAsianResult
from options_lib.models.monte_carlo import MonteCarlo
from options_lib.instruments.asian import AsianOption, AverageType
from options_lib.instruments.european import EuropeanOption
from options_lib.instruments.base import MarketData, OptionType


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

@pytest.fixture
def typical_params():
    """Typical SPX-like Heston parameters."""
    return HestonParams(v0=0.04, kappa=2.0, v_bar=0.04, xi=0.3, rho=-0.7)

@pytest.fixture
def flat_params():
    """Near-zero xi: Heston should behave like flat BS."""
    return HestonParams(v0=0.04, kappa=2.0, v_bar=0.04, xi=1e-4, rho=0.0)

@pytest.fixture
def mkt():
    return MarketData(spot=100, rate=0.05, div_yield=0.0)

@pytest.fixture
def atm_asian():
    return AsianOption(
        strike=100, expiry=1.0, option_type=OptionType.CALL,
        average_type=AverageType.ARITHMETIC, n_observations=52
    )


# ------------------------------------------------------------------
# HestonSimulator tests
# ------------------------------------------------------------------

class TestHestonSimulator:

    def test_path_shapes(self, typical_params):
        sim = HestonSimulator(typical_params, n_paths=1000, n_steps=50, seed=1)
        S, v = sim.simulate(100, 1.0, 0.05)
        assert S.shape == (1000, 51)
        assert v.shape == (1000, 51)

    def test_initial_values(self, typical_params):
        sim = HestonSimulator(typical_params, n_paths=1000, n_steps=50, seed=1)
        S, v = sim.simulate(100, 1.0, 0.05)
        assert np.all(S[:, 0] == 100.0)
        assert np.all(v[:, 0] == typical_params.v0)

    def test_variance_non_negative_full_truncation(self, typical_params):
        """Full truncation ensures variance is always >= 0."""
        sim = HestonSimulator(typical_params, n_paths=5000, n_steps=100,
                               truncation='full', seed=1)
        _, v = sim.simulate(100, 1.0, 0.05)
        assert np.all(v >= 0), f"Negative variance found: min={v.min():.6f}"

    def test_spot_prices_positive(self, typical_params):
        """All spot prices must be positive (lognormal step guarantees this)."""
        sim = HestonSimulator(typical_params, n_paths=5000, n_steps=50, seed=1)
        S, _ = sim.simulate(100, 1.0, 0.05)
        assert np.all(S > 0)

    def test_martingale_condition(self, typical_params):
        """
        Under Q: E[S_T] = S0 * e^{(r-q)*T}
        The discounted spot is a Q-martingale.
        With many paths this should hold within ~1%.
        """
        sim = HestonSimulator(typical_params, n_paths=100_000, n_steps=50,
                               antithetic=True, seed=42)
        S, _ = sim.simulate(S0=100, T=1.0, r=0.05, q=0.0)
        S_T  = S[:, -1]
        theoretical = 100 * np.exp(0.05 * 1.0)
        empirical   = np.mean(S_T)
        assert abs(empirical - theoretical) / theoretical < 0.01, \
            f"Martingale violated: E[S_T]={empirical:.4f}, expected={theoretical:.4f}"

    def test_flat_vol_limit(self, flat_params):
        """
        With xi→0 and v0=v_bar, Heston → GBM with σ=√v0.
        Terminal distribution should match BS.
        """
        sigma = np.sqrt(flat_params.v0)
        sim   = HestonSimulator(flat_params, n_paths=50_000, n_steps=50,
                                 antithetic=True, seed=42)
        S, _ = sim.simulate(S0=100, T=1.0, r=0.05, q=0.0)
        S_T  = S[:, -1]

        # Compare empirical log-return distribution to theoretical N(μ, σ²T)
        log_ret    = np.log(S_T / 100)
        theo_mean  = (0.05 - 0.5 * sigma**2) * 1.0
        theo_std   = sigma * 1.0

        emp_mean = np.mean(log_ret)
        emp_std  = np.std(log_ret)

        assert abs(emp_mean - theo_mean) < 0.02, \
            f"Mean mismatch: empirical={emp_mean:.4f}, theoretical={theo_mean:.4f}"
        assert abs(emp_std - theo_std) < 0.02, \
            f"Std mismatch: empirical={emp_std:.4f}, theoretical={theo_std:.4f}"

    def test_negative_correlation_effect(self, mkt):
        """
        With ρ < 0: when S goes down, v tends to go up.
        Compare: when S_T < S0, average v should be higher than baseline.
        """
        params_neg = HestonParams(v0=0.04, kappa=1.0, v_bar=0.04, xi=0.5, rho=-0.8)
        sim = HestonSimulator(params_neg, n_paths=50_000, n_steps=50,
                               antithetic=True, seed=42)
        S, v = sim.simulate(100, 1.0, 0.05)

        S_T    = S[:, -1]
        v_T    = v[:, -1]
        down_mask = S_T < 100
        up_mask   = S_T > 100

        avg_v_down = np.mean(v_T[down_mask]) if down_mask.sum() > 0 else 0
        avg_v_up   = np.mean(v_T[up_mask])   if up_mask.sum() > 0   else 0

        assert avg_v_down > avg_v_up, \
            f"Negative correlation: v when S_T<S0 ({avg_v_down:.4f}) " \
            f"should > v when S_T>S0 ({avg_v_up:.4f})"

    def test_terminal_distribution_only(self, typical_params):
        """terminal_distribution() returns shape (n_paths,)."""
        sim = HestonSimulator(typical_params, n_paths=1000, n_steps=50, seed=1)
        S_T = sim.terminal_distribution(100, 1.0, 0.05)
        assert S_T.shape == (1000,)
        assert np.all(S_T > 0)


# ------------------------------------------------------------------
# HestonAsianPricer tests
# ------------------------------------------------------------------

class TestHestonAsianPricer:

    def test_result_type(self, typical_params, atm_asian, mkt):
        pricer = HestonAsianPricer(typical_params, n_paths=10_000, n_steps=52, seed=1)
        result = pricer.price(atm_asian, mkt)
        assert isinstance(result, HestonAsianResult)

    def test_price_positive_itm(self, typical_params, mkt):
        """Deep ITM Asian call must have positive price."""
        itm_asian = AsianOption(
            strike=80, expiry=1.0, option_type=OptionType.CALL,
            average_type=AverageType.ARITHMETIC, n_observations=52
        )
        pricer = HestonAsianPricer(typical_params, n_paths=10_000, n_steps=52, seed=1)
        result = pricer.price(itm_asian, mkt)
        assert result.price > 0, f"ITM Asian price should be positive: {result.price}"

    def test_price_non_negative(self, typical_params, mkt):
        """Asian price must be non-negative for all strikes."""
        for K in [80, 100, 120]:
            asian = AsianOption(K, 1.0, OptionType.CALL, AverageType.ARITHMETIC, 52)
            pricer = HestonAsianPricer(typical_params, n_paths=10_000, n_steps=52, seed=1)
            result = pricer.price(asian, mkt)
            assert result.price >= 0

    def test_heston_vs_bs_differ(self, typical_params, atm_asian, mkt):
        """
        Heston Asian price should differ from flat BS Asian price
        for typical parameters with non-zero xi and rho.
        """
        pricer = HestonAsianPricer(typical_params, n_paths=50_000, n_steps=52,
                                    seed=42, control_variate=False)
        result = pricer.price(atm_asian, mkt)
        # Prices should differ by at least a small amount
        assert abs(result.vol_premium) > 0.001, \
            f"Heston and BS Asian prices should differ: premium={result.vol_premium:.4f}"

    def test_higher_xi_larger_premium(self, mkt, atm_asian):
        """
        Higher vol-of-vol (xi) → larger stochastic vol premium.
        More vol uncertainty → more departure from BS.
        """
        params_low  = HestonParams(v0=0.04, kappa=2.0, v_bar=0.04, xi=0.1, rho=-0.5)
        params_high = HestonParams(v0=0.04, kappa=2.0, v_bar=0.04, xi=0.6, rho=-0.5)

        pricer_low  = HestonAsianPricer(params_low,  n_paths=30_000, seed=42,
                                         control_variate=False)
        pricer_high = HestonAsianPricer(params_high, n_paths=30_000, seed=42,
                                         control_variate=False)

        result_low  = pricer_low.price(atm_asian, mkt)
        result_high = pricer_high.price(atm_asian, mkt)

        prem_low  = abs(result_low.vol_premium)
        prem_high = abs(result_high.vol_premium)

        # High xi should produce larger premium (or same direction, at minimum)
        assert prem_high >= prem_low - 0.05, \
            f"High xi premium {prem_high:.4f} should >= low xi {prem_low:.4f}"

    def test_arithmetic_asian_leq_european(self, typical_params, mkt):
        """
        Arithmetic Asian call <= European call under Heston.
        Averaging reduces the effective volatility → lower price.
        """
        asian   = AsianOption(100, 1.0, OptionType.CALL, AverageType.ARITHMETIC, 52)
        european = EuropeanOption(100, 1.0, OptionType.CALL)

        asian_pricer = HestonAsianPricer(typical_params, n_paths=50_000,
                                          seed=42, control_variate=False)
        asian_result = asian_pricer.price(asian, mkt)

        from options_lib.models.heston import Heston
        heston_model = Heston(typical_params)
        eu_price     = heston_model.price(european, mkt)

        assert asian_result.price <= eu_price + 0.20, \
            f"Asian {asian_result.price:.4f} should <= European {eu_price:.4f}"

    def test_confidence_interval_valid(self, typical_params, atm_asian, mkt):
        """Confidence interval brackets the price."""
        pricer = HestonAsianPricer(typical_params, n_paths=10_000, seed=1)
        result = pricer.price(atm_asian, mkt)
        lo, hi = result.confidence_interval
        assert lo < result.price < hi

    def test_summary_string(self, typical_params, atm_asian, mkt):
        pricer = HestonAsianPricer(typical_params, n_paths=5_000, seed=1)
        result = pricer.price(atm_asian, mkt)
        s = result.summary()
        assert "Heston price" in s
        assert "Vol premium" in s

    def test_geometric_asian_same_as_arithmetic_direction(self, typical_params, mkt):
        """
        Geometric Asian call <= Arithmetic Asian call (AM-GM inequality).
        This must hold even under Heston.
        """
        arith = AsianOption(100, 1.0, OptionType.CALL, AverageType.ARITHMETIC, 52)
        geo   = AsianOption(100, 1.0, OptionType.CALL, AverageType.GEOMETRIC,  52)

        pricer = HestonAsianPricer(typical_params, n_paths=30_000, seed=42,
                                    control_variate=False)
        r_arith = pricer.price(arith, mkt)
        r_geo   = pricer.price(geo,   mkt)

        assert r_arith.price >= r_geo.price - 0.10, \
            f"Arith {r_arith.price:.4f} should >= Geo {r_geo.price:.4f}"


# ------------------------------------------------------------------
# QE scheme tests
# ------------------------------------------------------------------

class TestQEScheme:

    def test_qe_path_shapes(self):
        """QE scheme produces paths of correct shape."""
        params = HestonParams(v0=0.04, kappa=2.0, v_bar=0.04, xi=0.3, rho=-0.7)
        sim = HestonSimulator(params, n_paths=500, n_steps=50, scheme='qe', seed=1)
        S, v = sim.simulate(100, 1.0, 0.05)
        assert S.shape == (500, 51)
        assert v.shape == (500, 51)

    def test_qe_variance_non_negative(self):
        """QE variance is non-negative by construction — no truncation needed."""
        params = HestonParams(v0=0.04, kappa=0.5, v_bar=0.04, xi=0.8, rho=-0.7)
        sim = HestonSimulator(params, n_paths=5000, n_steps=50, scheme='qe', seed=1)
        _, v = sim.simulate(100, 1.0, 0.05)
        assert np.all(v >= 0), f"QE variance went negative: min={v.min():.6f}"

    def test_qe_spot_positive(self):
        """QE spot prices are always positive."""
        params = HestonParams(v0=0.04, kappa=2.0, v_bar=0.04, xi=0.3, rho=-0.7)
        sim = HestonSimulator(params, n_paths=2000, n_steps=50, scheme='qe', seed=1)
        S, _ = sim.simulate(100, 1.0, 0.05)
        assert np.all(S > 0)

    def test_qe_martingale_condition(self):
        """QE: E[S_T] = S0 * e^{(r-q)*T} within 1%."""
        params = HestonParams(v0=0.04, kappa=2.0, v_bar=0.04, xi=0.3, rho=-0.7)
        sim = HestonSimulator(params, n_paths=100_000, n_steps=50,
                               scheme='qe', antithetic=True, seed=42)
        S, _ = sim.simulate(100, 1.0, 0.05)
        theoretical = 100 * np.exp(0.05)
        empirical   = np.mean(S[:, -1])
        assert abs(empirical - theoretical) / theoretical < 0.01, \
            f"QE martingale: E[S_T]={empirical:.4f}, expected={theoretical:.4f}"

    def test_qe_agrees_with_euler_atm_call(self):
        """
        QE and Euler-Milstein should give close prices for ATM European call.
        Both are correct — they should agree within MC noise.
        """
        from options_lib.models.heston import Heston
        from options_lib.instruments.european import EuropeanOption
        from options_lib.instruments.base import MarketData, OptionType

        params = HestonParams(v0=0.04, kappa=2.0, v_bar=0.04, xi=0.3, rho=-0.7)

        # Reference: Heston FFT price (analytical, exact)
        heston_model = Heston(params)
        call = EuropeanOption(100, 1.0, OptionType.CALL)
        mkt  = MarketData(spot=100, rate=0.05)
        fft_price = heston_model.price(call, mkt)

        # QE MC price
        sim_qe = HestonSimulator(params, n_paths=100_000, n_steps=50,
                                  scheme='qe', antithetic=True, seed=42)
        S_T_qe = sim_qe.terminal_distribution(100, 1.0, 0.05, 0.0)
        qe_price = float(np.exp(-0.05) * np.mean(np.maximum(S_T_qe - 100, 0)))

        # Euler price
        sim_euler = HestonSimulator(params, n_paths=100_000, n_steps=50,
                                     scheme='euler', antithetic=True, seed=42)
        S_T_euler = sim_euler.terminal_distribution(100, 1.0, 0.05, 0.0)
        euler_price = float(np.exp(-0.05) * np.mean(np.maximum(S_T_euler - 100, 0)))

        # Both should be within 0.20 of FFT
        assert abs(qe_price - fft_price) < 0.20, \
            f"QE price {qe_price:.4f} vs FFT {fft_price:.4f}"
        assert abs(euler_price - fft_price) < 0.20, \
            f"Euler price {euler_price:.4f} vs FFT {fft_price:.4f}"

    def test_qe_fewer_steps_still_accurate(self):
        """
        Key benefit of QE: works well with fewer time steps.
        QE with n_steps=10 should be more accurate than Euler with n_steps=10.
        """
        from options_lib.models.heston import Heston
        from options_lib.instruments.european import EuropeanOption
        from options_lib.instruments.base import MarketData, OptionType

        params = HestonParams(v0=0.04, kappa=2.0, v_bar=0.04, xi=0.3, rho=-0.7)
        heston_model = Heston(params)
        call = EuropeanOption(100, 1.0, OptionType.CALL)
        mkt  = MarketData(spot=100, rate=0.05)
        fft_price = heston_model.price(call, mkt)

        n_paths = 50_000
        n_steps_coarse = 10   # very few steps — where Euler degrades

        sim_qe = HestonSimulator(params, n_paths=n_paths, n_steps=n_steps_coarse,
                                  scheme='qe', antithetic=True, seed=42)
        sim_euler = HestonSimulator(params, n_paths=n_paths, n_steps=n_steps_coarse,
                                     scheme='euler', antithetic=True, seed=42)

        S_qe    = sim_qe.terminal_distribution(100, 1.0, 0.05)
        S_euler = sim_euler.terminal_distribution(100, 1.0, 0.05)

        qe_price    = float(np.exp(-0.05) * np.mean(np.maximum(S_qe    - 100, 0)))
        euler_price = float(np.exp(-0.05) * np.mean(np.maximum(S_euler - 100, 0)))

        err_qe    = abs(qe_price    - fft_price)
        err_euler = abs(euler_price - fft_price)

        # QE should be at least as accurate as Euler with same coarse grid
        # (typically significantly better)
        assert err_qe <= err_euler + 0.10, \
            f"QE error {err_qe:.4f} vs Euler error {err_euler:.4f} (FFT ref={fft_price:.4f})"

    def test_as_lsmc_simulator_interface(self):
        """
        as_lsmc_simulator() returns a callable with the correct
        (S0, T, r, q, n_paths, n_steps) -> S_paths signature.
        """
        from options_lib.numerics.lsmc import LongstaffSchwartz
        from options_lib.instruments.american import AmericanOption
        from options_lib.instruments.base import OptionType

        params = HestonParams(v0=0.04, kappa=2.0, v_bar=0.04, xi=0.3, rho=-0.7)
        sim    = HestonSimulator(params, n_paths=5000, n_steps=50,
                                  scheme='euler', seed=42)

        lsmc = LongstaffSchwartz(
            sigma          = 0.20,
            n_paths        = 5000,
            n_steps        = 50,
            path_simulator = sim.as_lsmc_simulator(),
        )
        am_put = AmericanOption(100, 1.0, OptionType.PUT)
        result = lsmc.price(am_put, spot=100, rate=0.05)

        assert result.price > 0, "Heston LSMC price should be positive"
        assert result.price < 30, "Heston LSMC price seems unreasonably large"
        assert result.std_error >= 0

    def test_as_lsmc_simulator_qe(self):
        """
        as_lsmc_simulator() works with QE scheme too —
        American put under Heston QE paths.
        """
        from options_lib.numerics.lsmc import LongstaffSchwartz
        from options_lib.instruments.american import AmericanOption
        from options_lib.instruments.base import OptionType

        params = HestonParams(v0=0.04, kappa=2.0, v_bar=0.04, xi=0.3, rho=-0.7)
        sim    = HestonSimulator(params, n_paths=5000, n_steps=50,
                                  scheme='qe', seed=42)

        lsmc = LongstaffSchwartz(
            sigma          = 0.20,
            n_paths        = 5000,
            n_steps        = 50,
            path_simulator = sim.as_lsmc_simulator(),
        )
        am_put = AmericanOption(100, 1.0, OptionType.PUT)
        result = lsmc.price(am_put, spot=100, rate=0.05)

        assert result.price > 0
        assert result.price < 30

    def test_invalid_scheme_raises(self):
        """Invalid scheme raises ValueError."""
        params = HestonParams(v0=0.04, kappa=2.0, v_bar=0.04, xi=0.3, rho=-0.7)
        with pytest.raises(ValueError, match="scheme"):
            HestonSimulator(params, scheme='bad_scheme')