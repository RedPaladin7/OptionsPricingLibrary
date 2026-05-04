"""
tests/test_monte_carlo.py
--------------------------
Tests for the Monte Carlo pricing engine.

Key tests:
  1. European call/put match BS (within statistical tolerance)
  2. Antithetic variates reduce standard error
  3. Control variate reduces standard error dramatically
  4. Barrier in/out parity: knock-in + knock-out = vanilla
  5. Asian geometric price matches Kemna-Vorst closed form
  6. Asian arithmetic < vanilla (averaging reduces vol)

Run with: pytest tests/test_monte_carlo.py -v
"""

import pytest
import numpy as np
from options_lib.models.monte_carlo import MonteCarlo, MonteCarloResult
from options_lib.models.black_scholes import BlackScholes
from options_lib.instruments.european import EuropeanOption
from options_lib.instruments.barrier import BarrierOption, BarrierType
from options_lib.instruments.asian import AsianOption, AverageType
from options_lib.instruments.base import OptionType, MarketData


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

@pytest.fixture
def mkt():
    return MarketData(spot=100, rate=0.05, div_yield=0.0)

@pytest.fixture
def model():
    # Fixed seed for reproducibility in tests
    return MonteCarlo(sigma=0.20, n_paths=50_000, antithetic=True,
                      control_variate=True, seed=42)

@pytest.fixture
def atm_call():
    return EuropeanOption(strike=100, expiry=1.0, option_type=OptionType.CALL)

@pytest.fixture
def atm_put():
    return EuropeanOption(strike=100, expiry=1.0, option_type=OptionType.PUT)


# ------------------------------------------------------------------
# European option tests
# ------------------------------------------------------------------

class TestMCEuropean:

    @pytest.mark.parametrize("K", [90, 100, 110])
    def test_european_call_matches_bs(self, mkt, K):
        """
        MC price for European call should be within 3 standard errors of BS.
        With 50k paths and control variate, std error is ~0.001.
        """
        call  = EuropeanOption(strike=K, expiry=1.0, option_type=OptionType.CALL)
        mc    = MonteCarlo(sigma=0.20, n_paths=50_000, antithetic=True,
                           control_variate=True, seed=42)
        bs    = BlackScholes(sigma=0.20)

        result   = mc.price_with_stats(call, mkt)
        bs_price = bs.price(call, mkt)

        # Within 5 std errors (very conservative — with CV this is easily met)
        assert abs(result.price - bs_price) < 5 * result.std_error + 0.01, \
            f"MC={result.price:.4f}, BS={bs_price:.4f}, SE={result.std_error:.4f}"

    def test_european_put_call_parity(self, mkt, model):
        """MC prices should satisfy put-call parity within statistical noise."""
        call = EuropeanOption(strike=100, expiry=1.0, option_type=OptionType.CALL)
        put  = EuropeanOption(strike=100, expiry=1.0, option_type=OptionType.PUT)
        C = model.price(call, mkt)
        P = model.price(put,  mkt)
        lhs = C - P
        rhs = mkt.spot - 100 * np.exp(-mkt.rate * 1.0)
        assert abs(lhs - rhs) < 0.10, \
            f"Put-call parity error: {abs(lhs-rhs):.4f}"

    def test_control_variate_reduces_std_error(self, mkt):
        """
        Control variate should give a lower standard error than plain MC.
        The reduction is typically 10-50x.
        """
        call = EuropeanOption(strike=100, expiry=1.0, option_type=OptionType.CALL)
        plain_mc = MonteCarlo(sigma=0.20, n_paths=10_000, antithetic=False,
                              control_variate=False, seed=42)
        cv_mc    = MonteCarlo(sigma=0.20, n_paths=10_000, antithetic=False,
                              control_variate=True,  seed=42)

        plain_result = plain_mc.price_with_stats(call, mkt)
        cv_result    = cv_mc.price_with_stats(call, mkt)

        assert cv_result.std_error < plain_result.std_error, \
            f"CV SE={cv_result.std_error:.6f} should be < plain SE={plain_result.std_error:.6f}"

    def test_antithetic_reduces_std_error(self, mkt):
        """Antithetic variates should reduce standard error vs plain MC."""
        call = EuropeanOption(strike=100, expiry=1.0, option_type=OptionType.CALL)
        plain_mc = MonteCarlo(sigma=0.20, n_paths=20_000, antithetic=False,
                              control_variate=False, seed=42)
        anti_mc  = MonteCarlo(sigma=0.20, n_paths=20_000, antithetic=True,
                              control_variate=False, seed=42)

        plain_result = plain_mc.price_with_stats(call, mkt)
        anti_result  = anti_mc.price_with_stats(call, mkt)

        assert anti_result.std_error < plain_result.std_error, \
            f"Antithetic SE={anti_result.std_error:.6f} should < plain SE={plain_result.std_error:.6f}"


# ------------------------------------------------------------------
# Barrier option tests
# ------------------------------------------------------------------

class TestMCBarrier:

    def test_barrier_inout_parity(self, mkt):
        """
        Knock-in + Knock-out = Vanilla option (same strike & expiry).
        This is a model-independent identity. Must hold.
        """
        S, K, H = 100, 100, 85   # down barrier at 85
        T = 1.0
        mc = MonteCarlo(sigma=0.20, n_paths=100_000, antithetic=True,
                        control_variate=False, seed=42, n_steps=252)

        down_out = BarrierOption(K, T, OptionType.CALL, H, BarrierType.DOWN_AND_OUT)
        down_in  = BarrierOption(K, T, OptionType.CALL, H, BarrierType.DOWN_AND_IN)
        vanilla  = EuropeanOption(strike=K, expiry=T, option_type=OptionType.CALL)

        p_out     = mc.price(down_out, mkt)
        p_in      = mc.price(down_in,  mkt)
        p_vanilla = mc.price(vanilla,  mkt)

        assert abs((p_out + p_in) - p_vanilla) < 0.15, \
            f"In-out parity violated: {p_out:.4f} + {p_in:.4f} = {p_out+p_in:.4f} vs vanilla {p_vanilla:.4f}"

    def test_knockin_cheaper_than_vanilla(self, mkt):
        """
        Knock-in option <= vanilla (only pays if barrier hit).
        """
        K, H, T = 100, 85, 1.0
        mc = MonteCarlo(sigma=0.20, n_paths=50_000, antithetic=True,
                        control_variate=False, seed=42, n_steps=252)

        down_in = BarrierOption(K, T, OptionType.CALL, H, BarrierType.DOWN_AND_IN)
        vanilla = EuropeanOption(strike=K, expiry=T, option_type=OptionType.CALL)

        p_in  = mc.price(down_in, mkt)
        p_van = mc.price(vanilla, mkt)

        assert p_in <= p_van + 0.10, \
            f"Knock-in {p_in:.4f} should be <= vanilla {p_van:.4f}"

    def test_knockout_closer_to_barrier_is_cheaper(self, mkt):
        """
        Down-and-out call with higher barrier (closer to spot) is cheaper:
        more likely to get knocked out.
        """
        K, T = 100, 1.0
        mc = MonteCarlo(sigma=0.20, n_paths=50_000, antithetic=True,
                        control_variate=False, seed=42, n_steps=252)

        far_barrier  = BarrierOption(K, T, OptionType.CALL, 70, BarrierType.DOWN_AND_OUT)
        near_barrier = BarrierOption(K, T, OptionType.CALL, 90, BarrierType.DOWN_AND_OUT)

        p_far  = mc.price(far_barrier,  mkt)
        p_near = mc.price(near_barrier, mkt)

        assert p_far > p_near, \
            f"Far barrier {p_far:.4f} should be > near barrier {p_near:.4f}"


# ------------------------------------------------------------------
# Asian option tests
# ------------------------------------------------------------------

class TestMCAsian:

    def test_geometric_asian_matches_kemna_vorst(self, mkt):
        """
        MC price for geometric Asian should match Kemna-Vorst closed form.
        This validates the path simulation and geometric average computation.
        """
        call = AsianOption(strike=100, expiry=1.0, option_type=OptionType.CALL,
                           average_type=AverageType.GEOMETRIC, n_observations=252)
        mc   = MonteCarlo(sigma=0.20, n_paths=50_000, antithetic=True,
                          control_variate=False, seed=42)

        mc_result = mc.price_with_stats(call, mkt)

        # Kemna-Vorst analytical price
        kv_price  = mc._kemna_vorst_price(call, mkt)

        assert abs(mc_result.price - kv_price) < 0.10, \
            f"Geometric Asian MC={mc_result.price:.4f}, KV={kv_price:.4f}"

    def test_arithmetic_asian_cheaper_than_vanilla(self, mkt):
        """
        Asian (average price) call is cheaper than vanilla call.
        Averaging reduces effective volatility: Var(avg S) < Var(S_T).
        Lower vol => lower option price.
        """
        mc      = MonteCarlo(sigma=0.20, n_paths=50_000, antithetic=True,
                             control_variate=True, seed=42)
        asian   = AsianOption(strike=100, expiry=1.0, option_type=OptionType.CALL,
                              average_type=AverageType.ARITHMETIC, n_observations=252)
        vanilla = EuropeanOption(strike=100, expiry=1.0, option_type=OptionType.CALL)

        p_asian   = mc.price(asian,   mkt)
        p_vanilla = mc.price(vanilla, mkt)

        assert p_asian < p_vanilla, \
            f"Asian {p_asian:.4f} should be < vanilla {p_vanilla:.4f}"

    def test_arithmetic_geq_geometric_asian(self, mkt):
        """
        Arithmetic average >= geometric average (AM-GM inequality).
        So arithmetic Asian call >= geometric Asian call.
        """
        mc   = MonteCarlo(sigma=0.20, n_paths=50_000, antithetic=True,
                          control_variate=False, seed=42)
        arith = AsianOption(strike=100, expiry=1.0, option_type=OptionType.CALL,
                            average_type=AverageType.ARITHMETIC, n_observations=252)
        geo   = AsianOption(strike=100, expiry=1.0, option_type=OptionType.CALL,
                            average_type=AverageType.GEOMETRIC,  n_observations=252)

        p_arith = mc.price(arith, mkt)
        p_geo   = mc.price(geo,   mkt)

        assert p_arith >= p_geo - 0.05, \
            f"Arithmetic Asian {p_arith:.4f} should >= geometric {p_geo:.4f}"

    def test_control_variate_reduces_asian_std_error(self, mkt):
        """
        Geometric Asian as control variate should dramatically reduce
        the standard error of the arithmetic Asian estimate.
        """
        n_paths = 10_000
        plain_mc = MonteCarlo(sigma=0.20, n_paths=n_paths, antithetic=False,
                              control_variate=False, seed=42)
        cv_mc    = MonteCarlo(sigma=0.20, n_paths=n_paths, antithetic=False,
                              control_variate=True, seed=42)

        asian = AsianOption(strike=100, expiry=1.0, option_type=OptionType.CALL,
                            average_type=AverageType.ARITHMETIC, n_observations=52)

        plain_result = plain_mc.price_with_stats(asian, mkt)
        cv_result    = cv_mc.price_with_stats(asian, mkt)

        assert cv_result.std_error < plain_result.std_error, \
            f"CV SE={cv_result.std_error:.6f} should < plain SE={plain_result.std_error:.6f}"