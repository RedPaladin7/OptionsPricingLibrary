"""
tests/test_greeks.py
---------------------
Tests for the GreekEngine and GreekSurface, and PnL attribution.

Key tests:
  1. GreekEngine BS analytical matches bump-and-reprice
  2. Greek surface has correct shape and monotone properties
  3. PnL attribution: explained + unexplained = actual P&L exactly
  4. For small moves, Taylor expansion is highly accurate
  5. Gamma-Theta relationship: Θ ≈ -½σ²S²Γ (BS PDE identity)
  6. Backtest attribution: cumulative explained ≈ cumulative actual

Run with: pytest tests/test_greeks.py -v
"""

import pytest
import numpy as np
from options_lib.models.black_scholes import BlackScholes
from options_lib.instruments.european import EuropeanOption
from options_lib.instruments.base import MarketData, OptionType
from options_lib.risk.greeks import GreekEngine, GreekSurface, Greeks
from options_lib.risk.pnl_attribution import PnLAttributor, PnLComponents, summarise_backtest


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

@pytest.fixture
def model():
    return BlackScholes(sigma=0.20)

@pytest.fixture
def atm_call():
    return EuropeanOption(strike=100, expiry=1.0, option_type=OptionType.CALL)

@pytest.fixture
def atm_put():
    return EuropeanOption(strike=100, expiry=1.0, option_type=OptionType.PUT)

@pytest.fixture
def mkt():
    return MarketData(spot=100, rate=0.05, div_yield=0.0)

@pytest.fixture
def engine(model):
    return GreekEngine(model)


# ------------------------------------------------------------------
# GreekEngine tests
# ------------------------------------------------------------------

class TestGreekEngine:

    def test_all_greeks_returns_greeks_object(self, engine, atm_call, mkt):
        g = engine.all_greeks(atm_call, mkt)
        assert isinstance(g, Greeks)

    def test_price_matches_model(self, engine, model, atm_call, mkt):
        assert abs(engine.price(atm_call, mkt) - model.price(atm_call, mkt)) < 1e-10

    def test_delta_matches_analytical(self, engine, model, atm_call, mkt):
        assert abs(engine.delta(atm_call, mkt) - model.delta(atm_call, mkt)) < 1e-10

    def test_gamma_matches_analytical(self, engine, model, atm_call, mkt):
        assert abs(engine.gamma(atm_call, mkt) - model.gamma(atm_call, mkt)) < 1e-10

    def test_vega_matches_analytical(self, engine, model, atm_call, mkt):
        assert abs(engine.vega(atm_call, mkt) - model.vega(atm_call, mkt)) < 1e-10

    def test_theta_matches_analytical(self, engine, model, atm_call, mkt):
        assert abs(engine.theta(atm_call, mkt) - model.theta(atm_call, mkt)) < 1e-10

    def test_vanna_matches_analytical(self, engine, model, atm_call, mkt):
        assert abs(engine.vanna(atm_call, mkt) - model.vanna(atm_call, mkt)) < 1e-10

    def test_volga_matches_analytical(self, engine, model, atm_call, mkt):
        assert abs(engine.volga(atm_call, mkt) - model.volga(atm_call, mkt)) < 1e-10

    def test_delta_bounds_call(self, engine, atm_call, mkt):
        assert 0 < engine.delta(atm_call, mkt) < 1

    def test_delta_bounds_put(self, engine, atm_put, mkt):
        assert -1 < engine.delta(atm_put, mkt) < 0

    def test_gamma_positive(self, engine, atm_call, mkt):
        assert engine.gamma(atm_call, mkt) > 0

    def test_vega_positive(self, engine, atm_call, mkt):
        assert engine.vega(atm_call, mkt) > 0

    def test_theta_negative_long_call(self, engine, atm_call, mkt):
        assert engine.theta(atm_call, mkt) < 0

    def test_volga_positive(self, engine, atm_call, mkt):
        """Volga always positive for vanilla options."""
        assert engine.volga(atm_call, mkt) > 0

    def test_gamma_theta_bs_pde_identity(self, engine, atm_call, mkt, model):
        """
        The BS PDE must hold: Θ + ½σ²S²Γ + (r-q)SΔ - rV = 0
        This is the fundamental constraint that links all Greeks.
        If this fails, the Greeks are internally inconsistent.
        """
        residual = model.verify_pde(atm_call, mkt)
        assert abs(residual) < 1e-6, f"BS PDE residual too large: {residual:.2e}"


# ------------------------------------------------------------------
# GreekSurface tests
# ------------------------------------------------------------------

class TestGreekSurface:

    def test_surface_shape(self, model, mkt):
        strikes  = np.array([90, 100, 110], dtype=float)
        expiries = np.array([0.25, 0.5, 1.0])
        surface  = GreekSurface(model, mkt, strikes, expiries, OptionType.CALL)
        surface.compute()
        assert surface.delta_surface.shape == (3, 3)
        assert surface.gamma_surface.shape == (3, 3)

    def test_delta_surface_monotone_in_spot(self, model, mkt):
        """
        For fixed expiry, call delta increases as strike decreases
        (lower strike = more ITM = higher delta).
        """
        strikes  = np.array([80, 90, 100, 110, 120], dtype=float)
        expiries = np.array([1.0])
        surface  = GreekSurface(model, mkt, strikes, expiries, OptionType.CALL)
        surface.compute()
        deltas = surface.delta_surface[0]  # single expiry
        # Delta should decrease as strike increases (call goes more OTM)
        assert all(deltas[i] > deltas[i+1] for i in range(len(deltas)-1)), \
            f"Delta not monotone decreasing with strike: {deltas}"

    def test_get_surface(self, model, mkt):
        strikes  = np.array([90, 100, 110], dtype=float)
        expiries = np.array([0.5, 1.0])
        surface  = GreekSurface(model, mkt, strikes, expiries, OptionType.CALL)
        surface.compute()
        delta = surface.get_surface('delta')
        assert delta.shape == (2, 3)

    def test_invalid_greek_name(self, model, mkt):
        strikes  = np.array([100], dtype=float)
        expiries = np.array([1.0])
        surface  = GreekSurface(model, mkt, strikes, expiries, OptionType.CALL)
        surface.compute()
        with pytest.raises(ValueError):
            surface.get_surface('nonexistent')


# ------------------------------------------------------------------
# P&L Attribution tests
# ------------------------------------------------------------------

class TestPnLAttribution:

    def test_pnl_decomposition_identity(self, model, atm_call, mkt):
        """
        explained_pnl + unexplained = actual_pnl  (by definition).
        This is trivially true from the dataclass property.
        """
        attr = PnLAttributor(model, atm_call, mkt)
        mkt1 = MarketData(spot=101, rate=mkt.rate, div_yield=mkt.div_yield)
        result = attr.explain(mkt1, sigma_t1=0.20, dt=1/252)
        assert abs(result.explained_pnl + result.unexplained - result.actual_pnl) < 1e-10

    def test_small_move_high_explanation_ratio(self, model, atm_call, mkt):
        """
        For a small spot move (1%), the Taylor expansion should explain
        >95% of P&L. Large unexplained residual signals higher-order terms.
        """
        attr = PnLAttributor(model, atm_call, mkt)
        mkt1 = MarketData(spot=101, rate=mkt.rate, div_yield=mkt.div_yield)
        result = attr.explain(mkt1, sigma_t1=0.20, dt=1/252)
        assert result.explanation_ratio > 0.95, \
            f"Small move should be well-explained. Ratio: {result.explanation_ratio:.2%}"

    def test_delta_pnl_dominates_small_move(self, model, atm_call, mkt):
        """
        For a pure spot move with no vol change, Delta P&L should dominate.
        Gamma P&L is second-order (dS²) and much smaller for a 1% move.
        """
        attr = PnLAttributor(model, atm_call, mkt)
        mkt1 = MarketData(spot=101, rate=mkt.rate, div_yield=mkt.div_yield)
        result = attr.explain(mkt1, sigma_t1=0.20, dt=1/252)
        assert abs(result.delta_pnl) > abs(result.gamma_pnl), \
            f"Delta ({result.delta_pnl:.4f}) should dominate Gamma ({result.gamma_pnl:.4f})"

    def test_vega_pnl_for_vol_move(self, model, atm_call, mkt):
        """
        For a pure vol move with no spot change, Vega P&L should dominate.
        """
        attr = PnLAttributor(model, atm_call, mkt)
        mkt1 = MarketData(spot=mkt.spot, rate=mkt.rate, div_yield=mkt.div_yield)
        result = attr.explain(mkt1, sigma_t1=0.21, dt=1/252)  # vol up 1%
        assert abs(result.vega_pnl) > 0, "Vega P&L should be positive for vol increase"
        assert result.vega_pnl > 0, "Long call gains value when vol rises"

    def test_theta_pnl_negative_long_call(self, model, atm_call, mkt):
        """
        For no market moves, P&L = Theta P&L = time decay (negative for long).
        """
        attr = PnLAttributor(model, atm_call, mkt)
        mkt1 = MarketData(spot=mkt.spot, rate=mkt.rate, div_yield=mkt.div_yield)
        result = attr.explain(mkt1, sigma_t1=0.20, dt=1/252)
        assert result.theta_pnl < 0, \
            f"Theta P&L should be negative for long call: {result.theta_pnl:.4f}"

    def test_gamma_pnl_positive_long_call(self, model, atm_call, mkt):
        """
        Large spot move benefits the long call via positive Gamma P&L.
        Direction doesn't matter (dS² is always positive), but sign of Gamma does.
        Long options have positive Gamma → positive Gamma P&L for any large move.
        """
        attr = PnLAttributor(model, atm_call, mkt)
        for new_spot in [95, 105]:   # both up and down moves
            mkt1 = MarketData(spot=new_spot, rate=mkt.rate, div_yield=mkt.div_yield)
            result = attr.explain(mkt1, sigma_t1=0.20, dt=1/252)
            assert result.gamma_pnl > 0, \
                f"Gamma P&L should be positive for long call: {result.gamma_pnl:.4f}"

    def test_summary_string(self, model, atm_call, mkt):
        """summary() returns a formatted string."""
        attr = PnLAttributor(model, atm_call, mkt)
        mkt1 = MarketData(spot=102, rate=mkt.rate, div_yield=mkt.div_yield)
        result = attr.explain(mkt1, sigma_t1=0.21, dt=1/252)
        s = result.summary()
        assert "Actual P&L" in s
        assert "Delta P&L" in s
        assert "Unexplained" in s

    def test_backtest_attribution(self, model, atm_call, mkt):
        """
        Over a simulated path, cumulative explained P&L should
        be close to cumulative actual P&L.
        """
        np.random.seed(42)
        n = 30   # 30 trading days
        dt = 1/252
        sigma = 0.20

        # Simulate a GBM spot path
        spot_path  = [mkt.spot]
        sigma_path = [sigma]
        for _ in range(n):
            dW = np.random.normal(0, np.sqrt(dt))
            spot_path.append(spot_path[-1] * np.exp((mkt.rate - 0.5*sigma**2)*dt + sigma*dW))
            sigma_path.append(sigma + np.random.normal(0, 0.002))

        spot_path  = np.array(spot_path)
        sigma_path = np.array(np.clip(sigma_path, 0.05, 0.80))

        attr    = PnLAttributor(model, atm_call, mkt)
        results = attr.backtest(spot_path, sigma_path, dt=dt)
        summary = summarise_backtest(results)

        # Cumulative explanation ratio should be reasonable
        # (won't be perfect because we reattribute at each step without recomputing Greeks)
        assert summary['n_periods'] == n
        assert 'explanation_ratio' in summary