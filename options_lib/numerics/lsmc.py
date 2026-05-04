"""
numerics/lsmc.py
-----------------
Longstaff-Schwartz Monte Carlo (LSM) for American option pricing.

The Problem with American Options
----------------------------------
American options can be exercised at any time t in [0, T]. The holder
faces an optimal stopping problem at each time step: is it better to
exercise now (take the intrinsic value) or continue holding (take the
continuation value)?

    V(S, t) = max(exercise_value, continuation_value)

The continuation value E[V(S_{t+1}, t+1) | S_t] requires knowing the
future — which is unknown. This is what makes American options hard.

The Longstaff-Schwartz Algorithm (2001)
-----------------------------------------
Key insight: estimate the continuation value by regressing future
discounted payoffs against basis functions of the current spot price,
using only in-the-money paths (where exercise is relevant).

Algorithm:
  1. Simulate N paths of S_t under Q (standard GBM)
  2. At expiry T: set cash flow = payoff(S_T) for each path
  3. Work BACKWARD from T-1 to t=1:
       a. Select only in-the-money (ITM) paths at time step j
       b. Regress the discounted future cash flows Y against
          basis functions of S_j: [1, S, S², ...] (Laguerre polynomials)
       c. Fitted values = estimated continuation value C(S_j)
       d. On ITM paths where intrinsic > C(S_j): exercise now
          Update cash flow to intrinsic value at time j
  4. Discount all path cash flows back to t=0 and average

Why Regression Works
--------------------
The continuation value is an expectation conditional on the current state:
    C(S_j) = E[e^{-r*dt} * V_{j+1} | S_j]

By the Markov property, this is a function of S_j only. Least-squares
regression approximates this function within the span of the chosen
basis functions. As N → ∞, the approximation converges to the true
conditional expectation (by the projection theorem in L² space).

The regression is only done on ITM paths because:
  - OTM paths have zero intrinsic value — no exercise decision needed
  - Including OTM paths would dilute the regression with zeros
  - The continuation value only matters where exercise is relevant

Basis Functions
---------------
We use the first K Laguerre polynomials (standard in LSM literature):
    L_0(x) = 1
    L_1(x) = 1 - x
    L_2(x) = 1 - 2x + x²/2
    ...

x = S/K (normalised spot) ensures numerical stability in the regression.
Alternatively, a simple polynomial basis [1, x, x², x³] works well.

LSM vs Finite Difference
------------------------
                FD (Crank-Nicolson)    LSM
Dimensionality  1D only (per asset)    Any dimension (multi-asset!)
Vol model       Constant BS            Any simulatable model
Speed           Fast for 1D            Slow (needs many paths)
Accuracy        High (deterministic)   Stochastic (std error)
Early boundary  Extracted directly     Requires extra work

For single-asset Americans under BS: FD is faster and more accurate.
For multi-asset (basket options, convertible bonds) or under Heston/LV:
LSM is the only tractable approach. This is why it's the industry standard.

References
----------
Longstaff, F.A. and Schwartz, E.S. (2001). "Valuing American Options by
Simulation: A Simple Least-Squares Approach." Review of Financial Studies,
14(1), 113-147.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Callable

from options_lib.instruments.base import OptionType
from options_lib.instruments.american import AmericanOption


# ------------------------------------------------------------------
# Basis Functions
# ------------------------------------------------------------------

def laguerre_basis(x: np.ndarray, degree: int = 4) -> np.ndarray:
    """
    Laguerre polynomial basis functions for LSM regression.

    Parameters
    ----------
    x : np.ndarray, shape (N,)
        Input values (normalised spot S/K for numerical stability).
    degree : int
        Number of basis functions (polynomial degree + 1).

    Returns
    -------
    np.ndarray, shape (N, degree)
        Basis matrix. Column j = L_j(x).

    Notes
    -----
    Laguerre polynomials are defined on [0, ∞) which matches the
    domain of spot prices. They are the canonical choice in the
    original LSM paper.
    """
    x = np.asarray(x, dtype=float)
    basis = np.zeros((len(x), degree))
    basis[:, 0] = np.exp(-x / 2)                          # L_0
    if degree >= 2:
        basis[:, 1] = np.exp(-x / 2) * (1 - x)           # L_1
    if degree >= 3:
        basis[:, 2] = np.exp(-x / 2) * (1 - 2*x + 0.5*x**2)  # L_2
    if degree >= 4:
        basis[:, 3] = np.exp(-x / 2) * (1 - 3*x + 1.5*x**2 - x**3/6)  # L_3
    if degree >= 5:
        basis[:, 4] = np.exp(-x / 2) * (
            1 - 4*x + 3*x**2 - 2*x**3/3 + x**4/24
        )  # L_4
    for k in range(5, degree):
        # Fill remaining with simple polynomial terms for flexibility
        basis[:, k] = x**k
    return basis


def monomial_basis(x: np.ndarray, degree: int = 4) -> np.ndarray:
    """
    Simple monomial basis [1, x, x², ..., x^{degree-1}].
    Faster to compute but numerically less stable than Laguerre.
    """
    x = np.asarray(x, dtype=float)
    return np.column_stack([x**k for k in range(degree)])


# ------------------------------------------------------------------
# LSMC Result
# ------------------------------------------------------------------

@dataclass
class LSMCResult:
    """
    Results from Longstaff-Schwartz pricing.

    Attributes
    ----------
    price       : float   American option price
    std_error   : float   Monte Carlo standard error
    n_paths     : int     Number of simulated paths
    n_steps     : int     Number of time steps
    exercise_boundary : np.ndarray or None
        Estimated early exercise boundary S*(t) at each time step.
        Shape: (n_steps,). Only computed if extract_boundary=True.
    """
    price             : float
    std_error         : float
    n_paths           : int
    n_steps           : int
    exercise_boundary : Optional[np.ndarray] = None

    @property
    def confidence_interval(self) -> tuple:
        return (
            self.price - 1.96 * self.std_error,
            self.price + 1.96 * self.std_error
        )

    def __repr__(self) -> str:
        lo, hi = self.confidence_interval
        return (
            f"LSMC Price: {self.price:.4f} "
            f"± {1.96*self.std_error:.4f} "
            f"(95% CI: [{lo:.4f}, {hi:.4f}]) "
            f"[N={self.n_paths:,}, steps={self.n_steps}]"
        )


# ------------------------------------------------------------------
# LSMC Pricer
# ------------------------------------------------------------------

@dataclass
class LongstaffSchwartz:
    """
    Longstaff-Schwartz Monte Carlo pricer for American options.

    Parameters
    ----------
    sigma : float
        Volatility (constant BS — for stochastic vol, override
        path_simulator with a custom simulator).
    n_paths : int
        Number of simulation paths. More paths → lower std error.
        Recommended: 50,000 – 200,000 for production accuracy.
    n_steps : int
        Number of time steps per path. More steps → better approximation
        of the continuous exercise decision. 50–252 is typical.
    degree : int
        Number of regression basis functions. 3–5 is standard.
        More basis functions → better continuation value approximation
        but higher variance (bias-variance tradeoff).
    basis : str
        'laguerre' (default, more stable) or 'monomial'.
    antithetic : bool
        Use antithetic variates for variance reduction.
    seed : int, optional
        Random seed for reproducibility.
    path_simulator : callable, optional
        Custom path simulator: (S0, T, r, q, n_paths, n_steps) -> paths
        of shape (n_paths, n_steps+1). If None, uses standard GBM.
        Override this for Heston or local vol dynamics.

    Notes
    -----
    The standard error of LSMC is approximately:
        SE ≈ std(discounted_payoffs) / sqrt(n_paths)

    With antithetic variates, effective variance is reduced by ~30-50%.
    Increasing n_paths by 4x halves the standard error.
    """
    sigma            : float
    n_paths          : int   = 100_000
    n_steps          : int   = 100
    degree           : int   = 4
    basis            : str   = 'laguerre'
    antithetic       : bool  = True
    seed             : Optional[int] = None
    path_simulator   : Optional[Callable] = None

    def __post_init__(self):
        if self.sigma <= 0:
            raise ValueError(f"sigma must be positive, got {self.sigma}")
        if self.n_paths < 1000:
            raise ValueError(f"n_paths should be at least 1000 for reliable results")
        if self.degree < 2:
            raise ValueError(f"degree must be at least 2")

    def _simulate_paths(
        self,
        S0      : float,
        T       : float,
        r       : float,
        q       : float,
    ) -> np.ndarray:
        """
        Simulate GBM paths under the risk-neutral measure Q.

        S_{t+dt} = S_t * exp[(r - q - σ²/2)*dt + σ*sqrt(dt)*Z]

        This is the EXACT log-normal transition — no discretisation error.
        (Unlike Euler-Maruyama, which has O(dt) error.)

        Returns
        -------
        paths : np.ndarray, shape (n_paths, n_steps + 1)
            paths[:, 0] = S0 (initial spot)
            paths[:, -1] = S_T (terminal spot)
        """
        if self.path_simulator is not None:
            return self.path_simulator(S0, T, r, q, self.n_paths, self.n_steps)

        if self.seed is not None:
            np.random.seed(self.seed)

        dt    = T / self.n_steps
        drift = (r - q - 0.5 * self.sigma**2) * dt
        vol   = self.sigma * np.sqrt(dt)

        if self.antithetic:
            half    = self.n_paths // 2
            Z_half  = np.random.standard_normal((half, self.n_steps))
            Z       = np.concatenate([Z_half, -Z_half], axis=0)
        else:
            Z = np.random.standard_normal((self.n_paths, self.n_steps))

        # Cumulative log-returns
        log_returns = drift + vol * Z            # (n_paths, n_steps)
        log_S       = np.log(S0) + np.cumsum(log_returns, axis=1)
        log_S0      = np.full((self.n_paths, 1), np.log(S0))

        return np.exp(np.concatenate([log_S0, log_S], axis=1))

    def _basis_matrix(self, S: np.ndarray, K: float) -> np.ndarray:
        """
        Compute regression basis matrix for normalised spot x = S/K.

        Normalising by K ensures x is O(1), improving regression stability.
        """
        x = S / K
        if self.basis == 'laguerre':
            return laguerre_basis(x, self.degree)
        else:
            return monomial_basis(x, self.degree)

    def price(
        self,
        instrument         : AmericanOption,
        spot               : float,
        rate               : float,
        div_yield          : float = 0.0,
        extract_boundary   : bool  = False,
    ) -> LSMCResult:
        """
        Price an American option using Longstaff-Schwartz.

        Parameters
        ----------
        instrument : AmericanOption
        spot : float   Current spot price S_0.
        rate : float   Risk-free rate r.
        div_yield : float   Continuous dividend yield q.
        extract_boundary : bool
            If True, extract the early exercise boundary S*(t) at each step.
            Adds minor computational overhead.

        Returns
        -------
        LSMCResult
        """
        S0 = spot
        K  = instrument.strike
        T  = instrument.expiry
        r  = rate
        q  = div_yield
        dt = T / self.n_steps

        # ------------------------------------------------------------------
        # Step 1: Simulate paths
        # ------------------------------------------------------------------
        paths = self._simulate_paths(S0, T, r, q)
        # paths shape: (n_paths, n_steps + 1)

        # ------------------------------------------------------------------
        # Step 2: Terminal payoff at expiry
        # ------------------------------------------------------------------
        cash_flows = instrument.payoff(paths[:, -1]).copy()
        # cash_flows[i] = payoff if exercised at T for path i

        # Storage for early exercise boundary
        boundary = np.full(self.n_steps, np.nan) if extract_boundary else None

        # ------------------------------------------------------------------
        # Step 3: Backward induction
        # Iterate from t = T-dt back to t = dt (step indices n_steps-1 to 1)
        # We skip t=0 because we always hold at t=0 (we just started)
        # ------------------------------------------------------------------
        for j in range(self.n_steps - 1, 0, -1):
            S_j   = paths[:, j]             # spot at time step j
            t_j   = j * dt                  # calendar time at step j

            # Intrinsic value at this step
            intrinsic = instrument.payoff(S_j)

            # Select in-the-money (ITM) paths — exercise is only relevant here
            itm_mask = intrinsic > 0
            n_itm    = itm_mask.sum()

            if n_itm < self.degree + 1:
                # Too few ITM paths to regress — skip this step
                # (Discount cash flows forward — handled implicitly)
                continue

            # ------------------------------------------------------------------
            # Step 3a: Build regression basis for ITM paths
            # ------------------------------------------------------------------
            X = self._basis_matrix(S_j[itm_mask], K)  # (n_itm, degree)

            # Discounted future cash flows (the "Y" in the regression)
            # Discount from step j+1 onward (already accumulated in cash_flows)
            Y = np.exp(-r * dt) * cash_flows[itm_mask]

            # ------------------------------------------------------------------
            # Step 3b: Least-squares regression: Y ≈ X @ beta
            # Continuation value estimate = X @ beta_hat
            # ------------------------------------------------------------------
            # np.linalg.lstsq is numerically stable and handles rank deficiency
            beta_hat, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
            continuation = X @ beta_hat              # shape: (n_itm,)

            # ------------------------------------------------------------------
            # Step 3c: Exercise decision
            # Exercise where intrinsic > estimated continuation value
            # ------------------------------------------------------------------
            exercise_now = intrinsic[itm_mask] > continuation

            # Update cash flows: exercise paths get intrinsic now (not later)
            # Non-exercise paths keep their future cash flow (discounted one step)
            itm_indices = np.where(itm_mask)[0]
            exercise_indices   = itm_indices[exercise_now]
            no_exercise_indices = itm_indices[~exercise_now]

            cash_flows[exercise_indices]    = intrinsic[itm_mask][exercise_now]
            cash_flows[no_exercise_indices] = np.exp(-r * dt) * cash_flows[no_exercise_indices]
            # OTM paths: just discount forward
            cash_flows[~itm_mask] = np.exp(-r * dt) * cash_flows[~itm_mask]

            # ------------------------------------------------------------------
            # Extract exercise boundary (optional)
            # S*(t_j) = boundary between exercise and continuation regions
            # ------------------------------------------------------------------
            if extract_boundary and n_itm > 0:
                if exercise_now.any() and (~exercise_now).any():
                    ex_S   = S_j[itm_mask][exercise_now]
                    no_ex_S = S_j[itm_mask][~exercise_now]
                    if instrument.option_type == OptionType.PUT:
                        # For puts: exercise below the boundary
                        # boundary ≈ max(S where we exercise)
                        boundary[j] = float(np.percentile(ex_S, 90))
                    else:
                        # For calls: exercise above the boundary
                        boundary[j] = float(np.percentile(ex_S, 10))

        # ------------------------------------------------------------------
        # Step 4: Discount all remaining cash flows to t=0
        # ------------------------------------------------------------------
        # At this point, cash_flows[i] holds the discounted cash flow
        # for path i, expressed at t=dt. Discount one more step to t=0.
        pv_cash_flows = np.exp(-r * dt) * cash_flows

        # ------------------------------------------------------------------
        # Step 5: Average across paths
        # ------------------------------------------------------------------
        price     = float(np.mean(pv_cash_flows))
        std_error = float(np.std(pv_cash_flows) / np.sqrt(self.n_paths))

        return LSMCResult(
            price             = max(price, 0.0),
            std_error         = std_error,
            n_paths           = self.n_paths,
            n_steps           = self.n_steps,
            exercise_boundary = boundary,
        )

    def compare_to_european(
        self,
        instrument : AmericanOption,
        spot       : float,
        rate       : float,
        div_yield  : float = 0.0,
    ) -> dict:
        """
        Price American option and compute early exercise premium
        vs the European counterpart (priced analytically via BS).

        Returns
        -------
        dict with keys: american_price, european_price, early_exercise_premium
        """
        from options_lib.models.black_scholes import BlackScholes
        from options_lib.instruments.european import EuropeanOption
        from options_lib.instruments.base import MarketData

        am_result = self.price(instrument, spot, rate, div_yield)

        eu_opt = EuropeanOption(
            strike=instrument.strike,
            expiry=instrument.expiry,
            option_type=instrument.option_type,
        )
        mkt = MarketData(spot=spot, rate=rate, div_yield=div_yield)
        eu_price = BlackScholes(sigma=self.sigma).price(eu_opt, mkt)

        return {
            'american_price'         : am_result.price,
            'american_std_error'     : am_result.std_error,
            'european_price'         : eu_price,
            'early_exercise_premium' : max(am_result.price - eu_price, 0.0),
        }

    def __repr__(self) -> str:
        return (
            f"LongstaffSchwartz(sigma={self.sigma}, "
            f"n_paths={self.n_paths:,}, n_steps={self.n_steps}, "
            f"degree={self.degree}, basis={self.basis})"
        )