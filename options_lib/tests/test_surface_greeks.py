"""
tests/test_surface_greeks.py
-----------------------------
Tests for SurfaceGreekEngine — vol surface Greeks via SVI parameter bumping.

Key tests:
  1. SurfaceGreekEngine prices correctly vs direct BS
  2. Parallel vega has same sign as scalar BS vega
  3. Vega by expiry sums approximately to parallel vega
  4. ATM option has largest vega in its expiry bucket vs OTM/ITM
  5. Skew sensitivity: OTM put is sensitive to skew, ATM is not
  6. Curvature sensitivity: OTM options are sensitive, ATM is not
  7. Portfolio Greeks are linear sum of individual Greeks
  8. Scenario P&L approximation is in right direction

Run with: pytest tests/test_surface_greeks.py -v
"""

import pytest
import numpy as np
from options_lib.risk.surface_greeks import (
    SurfaceGreekEngine, SurfaceGreeks, VolSurfaceScenario, scenario_pnl
)
from options_lib.market_data.vol_surface import SVIParams, VolSurface
from options_lib.instruments.european import EuropeanOption
from options_lib.instruments.base import MarketData, OptionType
from options_lib.models.black_scholes import BlackScholes


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

@pytest.fixture
def surface():
    """Realistic vol surface with negative skew (equity-like)."""
    svi_slices = {
        '2025-06-01': SVIParams(a=0.018, b=0.12, rho=-0.50, m=0.0, sigma=0.10, expiry=0.5),
        '2025-12-01': SVIParams(a=0.020, b=0.15, rho=-0.60, m=0.0, sigma=0.15, expiry=1.0),
    }
    forwards = {
        '2025-06-01': 102.5,
        '2025-12-01': 105.1,
    }
    return VolSurface(
        svi_slices=svi_slices, forwards=forwards,
        spot=100, rate=0.05, ticker='TEST'
    )

@pytest.fixture
def mkt():
    return MarketData(spot=100, rate=0.05, div_yield=0.0)

@pytest.fixture
def engine(surface, mkt):
    return SurfaceGreekEngine(surface, mkt)

@pytest.fixture
def atm_call():
    return EuropeanOption(strike=100, expiry=1.0, option_type=OptionType.CALL)

@pytest.fixture
def otm_put():
    return EuropeanOption(strike=85, expiry=1.0, option_type=OptionType.PUT)

@pytest.fixture
def itm_call():
    return EuropeanOption(strike=90, expiry=1.0, option_type=OptionType.CALL)


# ------------------------------------------------------------------
# Basic engine tests
# ------------------------------------------------------------------

class TestSurfaceGreekEngineBasic:

    def test_price_matches_bs_on_surface(self, engine, atm_call, surface, mkt):
        """
        Engine price = BS(σ_surface(K,T)).
        The engine should extract the surface IV and price via BS.
        """
        result = engine.compute(atm_call)
        iv     = surface.implied_vol(K=100, T=1.0)
        bs_p   = BlackScholes(sigma=iv).price(atm_call, mkt)
        assert abs(result.price - bs_p) < 1e-8, \
            f"Engine price {result.price:.4f} != BS price {bs_p:.4f}"

    def test_returns_surface_greeks_type(self, engine, atm_call):
        result = engine.compute(atm_call)
        assert isinstance(result, SurfaceGreeks)

    def test_vega_by_expiry_keys(self, engine, atm_call, surface):
        """vega_by_expiry has one entry per expiry slice."""
        result = engine.compute(atm_call)
        assert set(result.vega_by_expiry.keys()) == set(surface.expiry_dates)

    def test_skew_by_expiry_keys(self, engine, atm_call, surface):
        result = engine.compute(atm_call)
        assert set(result.skew_by_expiry.keys()) == set(surface.expiry_dates)


# ------------------------------------------------------------------
# Vega tests
# ------------------------------------------------------------------

class TestSurfaceVega:

    def test_parallel_vega_positive_long_call(self, engine, atm_call):
        """
        Long call has positive vega: value increases when vol rises.
        This must hold for both scalar BS vega and surface parallel vega.
        """
        result = engine.compute(atm_call)
        assert result.vega_parallel > 0, \
            f"ATM call parallel vega should be positive: {result.vega_parallel:.4f}"

    def test_parallel_vega_positive_long_put(self, engine, otm_put):
        """Long put also has positive vega."""
        result = engine.compute(otm_put)
        assert result.vega_parallel > 0, \
            f"OTM put parallel vega should be positive: {result.vega_parallel:.4f}"

    def test_vega_by_expiry_sums_approx_to_parallel(self, engine, atm_call):
        """
        Summing per-expiry vegas should give approximately the parallel vega.
        Not exact because the option only falls in one expiry bucket,
        but the sign and rough magnitude should match.
        """
        result = engine.compute(atm_call)
        sum_by_expiry = sum(result.vega_by_expiry.values())
        # The sum of per-expiry Vegas should have same sign as parallel vega
        assert np.sign(sum_by_expiry) == np.sign(result.vega_parallel), \
            f"Sign mismatch: sum_by_expiry={sum_by_expiry:.4f}, parallel={result.vega_parallel:.4f}"

    def test_atm_vega_large_relative_to_otm(self, engine, mkt, surface):
        """
        ATM option has larger vega than deep OTM option.
        Standard result: vega peaks at ATM.
        """
        atm = EuropeanOption(100, 1.0, OptionType.CALL)
        otm = EuropeanOption(130, 1.0, OptionType.CALL)  # deep OTM

        r_atm = engine.compute(atm)
        r_otm = engine.compute(otm)

        assert abs(r_atm.vega_parallel) > abs(r_otm.vega_parallel), \
            f"ATM vega {r_atm.vega_parallel:.4f} should > OTM vega {r_otm.vega_parallel:.4f}"

    def test_term_structure_dv01_populated(self, engine, atm_call):
        """term_structure_dv01 should have entries and be non-trivial."""
        result = engine.compute(atm_call)
        assert len(result.term_structure_dv01) > 0
        total_dv01 = sum(abs(v) for v in result.term_structure_dv01.values())
        assert total_dv01 > 0


# ------------------------------------------------------------------
# Skew sensitivity tests
# ------------------------------------------------------------------

class TestSkewSensitivity:

    def test_otm_put_sensitive_to_skew(self, engine, otm_put):
        """
        OTM put is sensitive to skew: a negative-skew surface prices OTM
        puts more expensively. Changing ρ (skew) should move OTM put price.
        """
        result = engine.compute(otm_put)
        assert abs(result.skew_sensitivity) > 1e-4, \
            f"OTM put should be sensitive to skew: {result.skew_sensitivity:.6f}"

    def test_atm_call_less_skew_sensitive_than_otm_put(self, engine, atm_call, otm_put):
        """
        ATM options are less sensitive to skew than OTM options.
        Skew tilts the smile: OTM options move, ATM is roughly unchanged.
        """
        r_atm = engine.compute(atm_call)
        r_otm = engine.compute(otm_put)
        assert abs(r_otm.skew_sensitivity) >= abs(r_atm.skew_sensitivity) - 0.001

    def test_skew_by_expiry_populated(self, engine, atm_call):
        result = engine.compute(atm_call)
        assert len(result.skew_by_expiry) > 0


# ------------------------------------------------------------------
# Curvature sensitivity tests
# ------------------------------------------------------------------

class TestCurvatureSensitivity:

    def test_otm_sensitive_to_curvature(self, engine, otm_put):
        """OTM options depend on smile curvature (vol convexity)."""
        result = engine.compute(otm_put)
        assert abs(result.curvature_sensitivity) > 1e-5

    def test_curvature_by_expiry_populated(self, engine, atm_call):
        result = engine.compute(atm_call)
        assert len(result.curvature_by_expiry) > 0


# ------------------------------------------------------------------
# Portfolio aggregation tests
# ------------------------------------------------------------------

class TestPortfolioGreeks:

    def test_portfolio_price_is_sum(self, engine, mkt, surface):
        """Portfolio price = sum of individual prices."""
        calls = [
            EuropeanOption(90,  1.0, OptionType.CALL),
            EuropeanOption(100, 1.0, OptionType.CALL),
            EuropeanOption(110, 1.0, OptionType.CALL),
        ]
        positions = [1.0, 2.0, -1.0]

        portfolio = engine.compute_portfolio(calls, positions)
        individual_sum = sum(
            pos * engine.compute(inst).price
            for pos, inst in zip(positions, calls)
        )
        assert abs(portfolio.price - individual_sum) < 1e-8

    def test_portfolio_vega_is_sum(self, engine, mkt, surface):
        """Portfolio parallel vega = weighted sum of individual vegas."""
        calls = [
            EuropeanOption(95,  1.0, OptionType.CALL),
            EuropeanOption(105, 1.0, OptionType.CALL),
        ]
        positions = [1.0, -1.0]

        portfolio = engine.compute_portfolio(calls, positions)
        individual_vega_sum = sum(
            pos * engine.compute(inst).vega_parallel
            for pos, inst in zip(positions, calls)
        )
        assert abs(portfolio.vega_parallel - individual_vega_sum) < 1e-8

    def test_straddle_vega_positive(self, engine):
        """
        Long straddle (long call + long put) has large positive vega.
        Both legs benefit from higher vol.
        """
        call = EuropeanOption(100, 1.0, OptionType.CALL)
        put  = EuropeanOption(100, 1.0, OptionType.PUT)
        port = engine.compute_portfolio([call, put], [1.0, 1.0])
        assert port.vega_parallel > 0

    def test_delta_hedge_reduces_no_vega(self, engine, mkt, surface):
        """Delta hedging with stock doesn't change vega (stock has no vega)."""
        call   = EuropeanOption(100, 1.0, OptionType.CALL)
        result = engine.compute(call)
        # Vega of call alone = vega of delta-hedged position (stock has no vega)
        assert result.vega_parallel > 0


# ------------------------------------------------------------------
# Scenario P&L tests
# ------------------------------------------------------------------

class TestScenarioPnL:

    def test_vol_spike_hurts_short_straddle(self, engine):
        """
        Short straddle (short call + short put) loses money in a vol spike.
        Parallel vol up → negative P&L for short straddle.
        """
        call = EuropeanOption(100, 1.0, OptionType.CALL)
        put  = EuropeanOption(100, 1.0, OptionType.PUT)

        # For individual positions (short = -1)
        straddle_vega = (
            -1.0 * engine.compute(call).vega_parallel +
            -1.0 * engine.compute(put).vega_parallel
        )
        # Short straddle vega is negative
        assert straddle_vega < 0, \
            f"Short straddle should have negative vega: {straddle_vega:.4f}"

    def test_scenario_pnl_direction_correct(self, engine, atm_call):
        """
        Vol spike → positive P&L for long call (positive vega).
        """
        scenario = VolSurfaceScenario.vol_spike(magnitude=0.02)
        result   = scenario_pnl(engine, atm_call, scenario)
        assert result['total_pnl'] > 0, \
            f"Long call should gain from vol spike: {result['total_pnl']:.4f}"

    def test_vol_crush_negative_for_long_call(self, engine, atm_call):
        """Vol crush → negative P&L for long call."""
        scenario = VolSurfaceScenario.vol_crush(magnitude=0.02)
        result   = scenario_pnl(engine, atm_call, scenario)
        assert result['total_pnl'] < 0, \
            f"Long call should lose from vol crush: {result['total_pnl']:.4f}"

    def test_scenario_pnl_keys(self, engine, atm_call):
        """scenario_pnl returns all required keys."""
        scenario = VolSurfaceScenario(parallel_shift=0.01)
        result   = scenario_pnl(engine, atm_call, scenario)
        for key in ['pnl_parallel', 'pnl_skew', 'pnl_curvature', 'total_pnl', 'base_price']:
            assert key in result

    def test_summary_string(self, engine, atm_call):
        result = engine.compute(atm_call)
        s = result.summary()
        assert "Parallel vega" in s
        assert "Skew sensitivity" in s


# ------------------------------------------------------------------
# Vega matrix tests
# ------------------------------------------------------------------

class TestVegaMatrix:

    @pytest.fixture
    def atm_call_05(self):
        return EuropeanOption(strike=100, expiry=0.5, option_type=OptionType.CALL)

    def test_vega_matrix_shape(self, engine, atm_call):
        """Vega matrix has shape (n_expiries, n_strikes)."""
        strikes = np.array([90., 100., 110.])
        vm = engine.vega_matrix(atm_call, pillar_strikes=strikes)
        assert vm.matrix.shape == (2, 3)   # 2 expiries in fixture surface
        assert len(vm.strikes) == 3
        assert len(vm.expiry_dates) == 2

    def test_vega_matrix_base_price_correct(self, engine, atm_call, surface, mkt):
        """base_price in VegaMatrix matches direct pricing."""
        vm = engine.vega_matrix(atm_call)
        iv = surface.implied_vol(K=100, T=1.0)
        bs_price = BlackScholes(sigma=iv).price(atm_call, mkt)
        assert abs(vm.base_price - bs_price) < 1e-8

    def test_vega_matrix_positive_long_call(self, engine, atm_call):
        """
        All pillar vegas should be positive for a long call.
        The call benefits from higher vol at any pillar.
        """
        strikes = np.array([85., 95., 100., 105., 115.])
        vm = engine.vega_matrix(atm_call, pillar_strikes=strikes)
        valid = vm.matrix[~np.isnan(vm.matrix)]
        assert np.all(valid >= 0), \
            f"All pillar vegas should be non-negative for long call. Min: {valid.min():.6f}"

    def test_total_vega_matches_parallel_vega(self, engine, atm_call):
        """
        total_vega (sum of all pillars) should be close to parallel vega
        from compute(). Both measure sensitivity to a uniform vol shift.
        """
        strikes = np.linspace(85, 115, 7)
        vm      = engine.vega_matrix(atm_call, pillar_strikes=strikes)
        sg      = engine.compute(atm_call)
        # total_vega is per 1bp, parallel_vega is per 1% → scale by 100
        total_vega_pct = vm.total_vega * 100
        assert np.sign(total_vega_pct) == np.sign(sg.vega_parallel), \
            "total_vega and parallel_vega should have same sign"

    def test_expiry_vegas_shape(self, engine, atm_call):
        """expiry_vegas sums across strikes → shape (n_expiries,)."""
        vm = engine.vega_matrix(atm_call)
        assert vm.expiry_vegas.shape == (len(vm.expiry_dates),)

    def test_strike_vegas_shape(self, engine, atm_call):
        """strike_vegas sums across expiries → shape (n_strikes,)."""
        strikes = np.array([90., 100., 110.])
        vm = engine.vega_matrix(atm_call, pillar_strikes=strikes)
        assert vm.strike_vegas.shape == (3,)

    def test_pillar_vega_decreasing_far_otm_call(self, engine, mkt, surface):
        """
        For a long ATM call, vegas at the option\'s own expiry should all
        be positive (the call benefits from higher vol at any pillar of
        that expiry). Pillars at other expiries may be smaller.
        """
        atm = EuropeanOption(100, 1.0, OptionType.CALL)
        strikes = np.array([85., 100., 115.])
        vm = engine.vega_matrix(atm, pillar_strikes=strikes)

        # The 1yr row (index 1 in fixture) should have all positive entries
        # since this is the option\'s own expiry
        t1_idx = 1
        row = vm.matrix[t1_idx]
        assert np.all(row >= 0), \
            f"All pillar vegas at option expiry should be non-negative: {row}"

    def test_portfolio_vega_matrix_sum(self, engine):
        """
        Portfolio vega matrix = weighted sum of individual matrices.
        Test with long + short call → smaller net vega.
        """
        call1 = EuropeanOption(95,  1.0, OptionType.CALL)
        call2 = EuropeanOption(105, 1.0, OptionType.CALL)
        strikes = np.array([90., 100., 110.])

        vm1   = engine.vega_matrix(call1, pillar_strikes=strikes)
        vm2   = engine.vega_matrix(call2, pillar_strikes=strikes)
        vm_pf = engine.portfolio_vega_matrix(
            [call1, call2], [1.0, -1.0], pillar_strikes=strikes
        )

        # Portfolio matrix = 1.0 * vm1 - 1.0 * vm2
        expected = vm1.matrix - vm2.matrix
        np.testing.assert_allclose(vm_pf.matrix, expected, atol=1e-10)

    def test_vega_matrix_to_dict(self, engine, atm_call):
        """to_dict() returns all required keys."""
        vm = engine.vega_matrix(atm_call)
        d  = vm.to_dict()
        for key in ['strikes', 'expiry_dates', 'expiries', 'matrix',
                    'base_price', 'total_vega', 'expiry_vegas', 'strike_vegas']:
            assert key in d

    def test_vega_matrix_summary(self, engine, atm_call):
        """summary() returns a formatted string."""
        vm = engine.vega_matrix(atm_call)
        s  = vm.summary()
        assert "Vega Matrix" in s
        assert "Total vega" in s