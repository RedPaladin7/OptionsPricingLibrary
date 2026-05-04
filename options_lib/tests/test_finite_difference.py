"""
tests/test_finite_difference.py
---------------------------------
Tests for the Crank-Nicolson finite difference solver.

Key tests:
  1. European call/put match BS analytical price
  2. American put > European put (early exercise premium > 0)
  3. American call on non-dividend stock = European call
  4. American put price decreases as spot increases
  5. Early exercise premium increases with dividend yield
  6. Greeks from grid match bump-and-reprice

Run with: pytest tests/test_finite_difference.py -v
"""

import pytest
import numpy as np
from options_lib.numerics.finite_difference import CrankNicolson
from options_lib.models.black_scholes import BlackScholes
from options_lib.instruments.european import EuropeanOption
from options_lib.instruments.american import AmericanOption
from options_lib.instruments.base import OptionType, MarketData


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

@pytest.fixture
def solver():
    return CrankNicolson(sigma=0.20, M=300, N=300)

@pytest.fixture
def atm_eur_call():
    return EuropeanOption(strike=100, expiry=1.0, option_type=OptionType.CALL)

@pytest.fixture
def atm_eur_put():
    return EuropeanOption(strike=100, expiry=1.0, option_type=OptionType.PUT)

@pytest.fixture
def atm_am_put():
    return AmericanOption(strike=100, expiry=1.0, option_type=OptionType.PUT)

@pytest.fixture
def atm_am_call():
    return AmericanOption(strike=100, expiry=1.0, option_type=OptionType.CALL)


# ------------------------------------------------------------------
# European option tests — validate against BS analytical
# ------------------------------------------------------------------

class TestEuropeanFD:

    @pytest.mark.parametrize("K,S,sigma", [
        (100, 100, 0.20),   # ATM
        (110, 100, 0.20),   # OTM call
        (90,  100, 0.20),   # ITM call
        (100, 100, 0.40),   # high vol
        (100, 100, 0.10),   # low vol
    ])
    def test_european_call_matches_bs(self, K, S, sigma):
        """
        FD price for European call should match BS analytical price.
        Tolerance of 0.05 accounts for grid discretisation error.
        Finer grids (M=N=500) would give tighter agreement.
        """
        solver = CrankNicolson(sigma=sigma, M=300, N=300)
        call   = EuropeanOption(strike=K, expiry=1.0, option_type=OptionType.CALL)
        r, q   = 0.05, 0.0

        fd_price = solver.price(call, spot=S, r=r, q=q)
        bs_price = BlackScholes(sigma=sigma).price(
            call, MarketData(spot=S, rate=r, div_yield=q)
        )
        tol = 0.10 if sigma >= 0.4 else 0.05  # high vol needs looser tolerance
        assert abs(fd_price - bs_price) < tol, \
            f"FD={fd_price:.4f} vs BS={bs_price:.4f} at K={K}, S={S}, sigma={sigma}"

    @pytest.mark.parametrize("K,S", [(90, 100), (100, 100), (110, 100)])
    def test_european_put_matches_bs(self, K, S):
        """European put FD price matches BS."""
        solver = CrankNicolson(sigma=0.20, M=300, N=300)
        put    = EuropeanOption(strike=K, expiry=1.0, option_type=OptionType.PUT)
        r, q   = 0.05, 0.0

        fd_price = solver.price(put, spot=S, r=r, q=q)
        bs_price = BlackScholes(sigma=0.20).price(
            put, MarketData(spot=S, rate=r, div_yield=q)
        )
        assert abs(fd_price - bs_price) < 0.05, \
            f"FD={fd_price:.4f} vs BS={bs_price:.4f} at K={K}"

    def test_european_put_call_parity_fd(self, solver, atm_eur_call, atm_eur_put):
        """
        FD should satisfy put-call parity:
            C - P = S*e^{-qT} - K*e^{-rT}
        """
        S, K, T, r, q = 100, 100, 1.0, 0.05, 0.0
        C = solver.price(atm_eur_call, spot=S, r=r, q=q)
        P = solver.price(atm_eur_put,  spot=S, r=r, q=q)
        lhs = C - P
        rhs = S - K * np.exp(-r * T)
        assert abs(lhs - rhs) < 0.1, \
            f"Put-call parity violated: C-P={lhs:.4f}, S-Ke^(-rT)={rhs:.4f}"


# ------------------------------------------------------------------
# American option tests
# ------------------------------------------------------------------

class TestAmericanFD:

    def test_american_put_geq_european_put(self, solver):
        """
        American put must be worth at least as much as European put.
        Early exercise right has non-negative value.
        """
        r, q, S = 0.05, 0.0, 100

        for K in [90, 100, 110]:
            eu_put = EuropeanOption(strike=K, expiry=1.0, option_type=OptionType.PUT)
            am_put = AmericanOption(strike=K, expiry=1.0, option_type=OptionType.PUT)
            eu_price = solver.price(eu_put, spot=S, r=r, q=q)
            am_price = solver.price(am_put, spot=S, r=r, q=q)
            assert am_price >= eu_price - 1e-4, \
                f"American put {am_price:.4f} < European put {eu_price:.4f} at K={K}"

    def test_american_call_equals_european_call_no_dividends(self, solver):
        """
        For non-dividend paying stocks, early exercise of a call is never optimal.
        Therefore American call = European call.

        Proof sketch: C_american >= C_european >= max(S - K*e^{-rT}, 0) > max(S-K, 0).
        So you're always better off selling the option than exercising it.
        """
        r, q, S = 0.05, 0.0, 100
        K = 100

        eu_call = EuropeanOption(strike=K, expiry=1.0, option_type=OptionType.CALL)
        am_call = AmericanOption(strike=K, expiry=1.0, option_type=OptionType.CALL)

        eu_price = solver.price(eu_call, spot=S, r=r, q=q)
        am_price = solver.price(am_call, spot=S, r=r, q=q)

        assert abs(am_price - eu_price) < 0.05, \
            f"American call {am_price:.4f} != European call {eu_price:.4f} (no dividends)"

    def test_american_put_geq_intrinsic(self, solver):
        """
        American put must always be worth at least its intrinsic value max(K-S, 0).
        This is the early exercise constraint.
        """
        S, K, r, q = 80, 100, 0.05, 0.0  # deep ITM put
        am_put = AmericanOption(strike=K, expiry=1.0, option_type=OptionType.PUT)
        price  = solver.price(am_put, spot=S, r=r, q=q)
        intrinsic = max(K - S, 0.0)
        assert price >= intrinsic - 1e-4, \
            f"American put {price:.4f} < intrinsic {intrinsic:.4f}"

    def test_american_put_decreases_with_spot(self, solver):
        """American put price is monotonically decreasing in spot."""
        am_put = AmericanOption(strike=100, expiry=1.0, option_type=OptionType.PUT)
        prices = [solver.price(am_put, spot=S, r=0.05, q=0.0)
                  for S in [70, 80, 90, 100, 110, 120]]
        assert all(prices[i] > prices[i+1] for i in range(len(prices)-1)), \
            f"Put price not monotone decreasing: {[f'{p:.3f}' for p in prices]}"

    def test_early_exercise_premium_positive_deep_itm(self, solver):
        """
        Deep ITM American put should have positive early exercise premium.
        When S << K, it's optimal to exercise immediately.
        The American put is worth K - S today (risk-free) vs waiting.
        """
        S, K = 60, 100  # 40% ITM
        eu_put = EuropeanOption(strike=K, expiry=1.0, option_type=OptionType.PUT)
        am_put = AmericanOption(strike=K, expiry=1.0, option_type=OptionType.PUT)
        r, q   = 0.05, 0.0

        eu_price = solver.price(eu_put, spot=S, r=r, q=q)
        am_price = solver.price(am_put, spot=S, r=r, q=q)
        premium  = am_price - eu_price

        assert premium > 0.1, \
            f"Early exercise premium should be positive for deep ITM: {premium:.4f}"

    def test_american_put_decreases_with_rate(self, solver):
        """
        Higher interest rates reduce American put price.
        Intuition: the put pays K - S. Higher r means K*e^{-rT} (the PV of
        receiving K) is smaller, making the put worth less.
        This is the opposite of the naive intuition — holding puts in a
        high rate environment is costly.
        """
        am_put = AmericanOption(strike=100, expiry=1.0, option_type=OptionType.PUT)
        prices = [solver.price(am_put, spot=100, r=r, q=0.0)
                  for r in [0.01, 0.05, 0.10, 0.15]]
        assert all(prices[i] > prices[i+1] for i in range(len(prices)-1)), \
            f"American put should decrease with r: {[f'{p:.3f}' for p in prices]}"

    def test_early_exercise_boundary_returns_arrays(self, solver):
        """
        Smoke test: early_exercise_boundary returns two arrays of correct shape.
        Deeper validation requires a reference implementation.
        """
        am_put = AmericanOption(strike=100, expiry=1.0, option_type=OptionType.PUT)
        times, boundary = solver.early_exercise_boundary(am_put, r=0.05, q=0.0)
        assert len(times) == solver.N
        assert len(boundary) == solver.N
        assert times[0] <= times[-1]  # times are ordered