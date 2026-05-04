"""
numerics/finite_difference.py
------------------------------
Crank-Nicolson finite difference solver for the Black-Scholes PDE.

The Black-Scholes PDE (backward in time):
    dV/dt + (r-q)S dV/dS + (1/2)sigma^2 S^2 d^2V/dS^2 - rV = 0

with terminal condition V(S, T) = payoff(S).

We solve this on a grid of (S, t) values, marching backward from T to 0.

Space discretisation
--------------------
Replace S with a uniform grid: S_i = i * dS, i = 0, 1, ..., M
    dV/dS  ~ (V_{i+1} - V_{i-1}) / (2 dS)    [central difference]
    d^2V/dS^2 ~ (V_{i+1} - 2V_i + V_{i-1}) / dS^2

Time discretisation — Crank-Nicolson
--------------------------------------
The key choice. Two extreme options:

    Explicit (forward Euler):
        [V^{n+1} - V^n] / dt = L[V^n]
        Directly solvable but conditionally stable: requires dt < dS^2 / sigma^2.
        For fine grids this means millions of time steps — very slow.

    Implicit (backward Euler):
        [V^{n+1} - V^n] / dt = L[V^{n+1}]
        Unconditionally stable but first-order accurate in time.

    Crank-Nicolson (average of explicit and implicit):
        [V^{n+1} - V^n] / dt = (1/2)(L[V^n] + L[V^{n+1}])
        Unconditionally stable AND second-order accurate in both space and time.
        This is the standard method for option pricing PDEs.

Each Crank-Nicolson step solves a tridiagonal linear system:
    A * V^{n+1} = B * V^n
where A and B are tridiagonal matrices. Solved via the Thomas algorithm
(forward/backward substitution) in O(M) time.

American Options — Early Exercise Constraint
--------------------------------------------
At each time step, after solving the linear system, enforce:
    V^{n+1}_i = max(V^{n+1}_i, intrinsic(S_i))

This is the "operator splitting" or "penalty" approach. Simple but effective:
it ensures the option is never worth less than immediate exercise.

References
----------
Wilmott, P., Dewynne, J., Howison, S. (1993). Option Pricing: Mathematical
Models and Computation. Oxford Financial Press.
"""

import numpy as np
from scipy.linalg import solve_banded
from dataclasses import dataclass
from typing import Optional

from options_lib.instruments.base import Instrument, ExerciseStyle
from options_lib.instruments.european import EuropeanOption
from options_lib.instruments.american import AmericanOption


@dataclass
class CrankNicolson:
    """
    Crank-Nicolson finite difference solver for the BS PDE.

    Handles both European and American exercise styles.
    For American options, applies the early exercise constraint
    at each time step via operator splitting.

    Parameters
    ----------
    sigma : float
        Volatility (constant, BS assumption).
    M : int
        Number of spot grid points. Default 200.
        More points = finer spatial resolution = more accurate.
    N : int
        Number of time steps. Default 200.
        More steps = finer time resolution. CN is stable for any N.
    S_max_multiplier : float
        The spot grid runs from 0 to S_max = S_max_multiplier * K.
        Default 4.0 (grid extends to 4x the strike).
        Too small: boundary condition error for large spots.
        Too large: wastes grid points in irrelevant region.

    Notes
    -----
    The solver prices over the ENTIRE spot grid [0, S_max] in one pass.
    This means after one solve, you have V(S, 0) for all S simultaneously —
    Greeks are essentially free (just read off the grid).
    """

    sigma            : float
    M                : int   = 200
    N                : int   = 200
    S_max_multiplier : float = 4.0

    def __post_init__(self):
        if self.sigma <= 0:
            raise ValueError(f"sigma must be positive, got {self.sigma}")
        if self.M < 10:
            raise ValueError(f"M must be at least 10, got {self.M}")
        if self.N < 10:
            raise ValueError(f"N must be at least 10, got {self.N}")

    def solve(
        self,
        instrument : Instrument,
        r          : float,
        q          : float,
        S_grid     : Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Solve the BS PDE for the given instrument.

        Parameters
        ----------
        instrument : Instrument
            EuropeanOption or AmericanOption.
        r : float
            Risk-free rate.
        q : float
            Dividend yield.
        S_grid : np.ndarray, optional
            Custom spot grid. If None, uses uniform grid [0, S_max].

        Returns
        -------
        S_grid : np.ndarray, shape (M+1,)
            Spot prices at each grid node.
        V : np.ndarray, shape (M+1,)
            Option values V(S, 0) at each grid node.
        """
        K = instrument.strike
        T = instrument.expiry
        is_american = instrument.exercise_style == ExerciseStyle.AMERICAN

        # ------------------------------------------------------------------
        # Step 1: Build the grids
        # ------------------------------------------------------------------
        S_max = self.S_max_multiplier * K
        if S_grid is None:
            S_grid = np.linspace(0, S_max, self.M + 1)

        dS = S_grid[1] - S_grid[0]
        dt = T / self.N

        # Interior nodes only (exclude boundaries at i=0 and i=M)
        S_int = S_grid[1:-1]          # shape (M-1,)
        i_int = np.arange(1, self.M)  # indices 1, ..., M-1

        # ------------------------------------------------------------------
        # Step 2: Terminal condition V(S, T) = payoff(S)
        # ------------------------------------------------------------------
        V = instrument.payoff(S_grid).copy()  # shape (M+1,)

        # ------------------------------------------------------------------
        # Step 3: Build the Crank-Nicolson coefficient matrices
        # ------------------------------------------------------------------
        # At each interior node i, the BS PDE gives three coefficients:
        #   alpha_i = (1/4) dt [-sigma^2 i^2 + (r-q) i]   (subdiagonal)
        #   beta_i  = (1/2) dt [ sigma^2 i^2 + r]          (diagonal)
        #   gamma_i = (1/4) dt [-sigma^2 i^2 - (r-q) i]   (superdiagonal)
        #
        # Note: using i (grid index) rather than S because S = i*dS and
        # the coefficients scale with i^2 (from S^2 d^2V/dS^2 = i^2 dS^2 * V''/dS^2).

        i = i_int.astype(float)
        alpha = 0.25 * dt * (-self.sigma**2 * i**2 + (r - q) * i)
        beta  = 0.50 * dt * ( self.sigma**2 * i**2 + r)
        gamma = 0.25 * dt * (-self.sigma**2 * i**2 - (r - q) * i)

        # ------------------------------------------------------------------
        # Step 4: Assemble tridiagonal matrices A (implicit) and B (explicit)
        # ------------------------------------------------------------------
        # Crank-Nicolson:  (I + L_implicit) V^{n+1} = (I - L_explicit) V^n
        # => A V^{n+1} = B V^n
        #
        # A = I + theta * L  (with theta=1/2 for CN)
        # B = I - theta * L
        #
        # In tridiagonal form:
        #   A: subdiag = -alpha, diag = 1+beta, superdiag = -gamma
        #   B: subdiag =  alpha, diag = 1-beta, superdiag =  gamma

        n_int = self.M - 1  # number of interior nodes

        # A matrix (used in LHS: solve A * V_new = rhs)
        # A = I + dt/2 * L  (implicit side)
        # A[i,i-1] = alpha_i, A[i,i] = 1+beta_i, A[i,i+1] = gamma_i
        A_sub   = alpha[1:]          # below diagonal, length n_int-1
        A_diag  =  1 + beta          # diagonal, length n_int
        A_super = gamma[:-1]         # above diagonal, length n_int-1

        # B = I - dt/2 * L  (explicit side)
        # B[i,i-1] = -alpha_i, B[i,i] = 1-beta_i, B[i,i+1] = -gamma_i
        B_sub   = -alpha[1:]
        B_diag  =  1 - beta
        B_super = -gamma[:-1]

        # Pack A into scipy's banded format (3 x n_int):
        # Row 0: superdiagonal (first element unused)
        # Row 1: main diagonal
        # Row 2: subdiagonal (last element unused)
        A_banded = np.zeros((3, n_int))
        A_banded[0, 1:] = A_super   # superdiagonal
        A_banded[1, :]  = A_diag    # diagonal
        A_banded[2, :-1] = A_sub    # subdiagonal

        # ------------------------------------------------------------------
        # Step 5: Time-stepping loop — march backward from T to 0
        # ------------------------------------------------------------------
        for n in range(self.N):
            V_old = V[1:-1].copy()  # interior values at current time step

            # RHS = B * V_old (tridiagonal matrix-vector multiply)
            rhs = B_diag * V_old
            rhs[1:]  += B_sub   * V_old[:-1]   # subdiagonal contribution
            rhs[:-1] += B_super * V_old[1:]    # superdiagonal contribution

            # Add boundary contributions to the RHS
            # At i=0 (S=0): V = K*exp(-r*(T-t)) for puts, 0 for calls
            # At i=M (S=S_max): V = 0 for puts, S_max - K*exp(-r*(T-t)) for calls
            tau = (n + 1) * dt   # time elapsed from T (so current time = T - tau)

            V_lower, V_upper = self._boundary_conditions(
                instrument, S_max, r, q, T - tau
            )

            # Boundary terms appear in the first and last rows of A*V = rhs
            # They are split 50/50 between old and new time steps (CN)
            rhs[0]  -= alpha[0]  * V_lower    # lower BC contribution to row 1
            rhs[-1] -= gamma[-1] * V_upper    # upper BC contribution to row M-1

            # Solve A * V_new = rhs (tridiagonal system via banded solver)
            V_new = solve_banded((1, 1), A_banded, rhs)

            # Apply new boundary conditions
            V[0]  = V_lower
            V[-1] = V_upper
            V[1:-1] = V_new

            # ------------------------------------------------------------------
            # American early exercise constraint: V >= intrinsic value
            # ------------------------------------------------------------------
            if is_american:
                intrinsic = instrument.payoff(S_grid)
                V = np.maximum(V, intrinsic)

        return S_grid, V

    def _boundary_conditions(
        self,
        instrument : Instrument,
        S_max      : float,
        r          : float,
        q          : float,
        t          : float,
    ) -> tuple[float, float]:
        """
        Boundary conditions at S=0 and S=S_max.

        S=0 boundary:
            Call: worthless (stock can never go up from 0)    -> V = 0
            Put:  certain to be exercised -> V = K*e^{-r*t}   (PV of strike)

        S=S_max boundary:
            Call: deep ITM, delta -> 1                        -> V = S_max*e^{-q*t} - K*e^{-r*t}
            Put:  deep OTM, worthless                          -> V = 0

        Parameters
        ----------
        t : float
            Current time (not time to expiry — actual calendar time).
        """
        K = instrument.strike
        from options_lib.instruments.base import OptionType

        if instrument.option_type == OptionType.CALL:
            V_lower = 0.0
            V_upper = max(S_max * np.exp(-q * t) - K * np.exp(-r * t), 0.0)
        else:
            V_lower = K * np.exp(-r * t)
            V_upper = 0.0

        return V_lower, V_upper

    def price(
        self,
        instrument : Instrument,
        spot       : float,
        r          : float,
        q          : float = 0.0,
    ) -> float:
        """
        Price a single option at a given spot price.

        Runs the full PDE solve, then interpolates V(S=spot, t=0).

        Parameters
        ----------
        instrument : Instrument
        spot : float
            Current spot price.
        r : float
            Risk-free rate.
        q : float
            Dividend yield.

        Returns
        -------
        float
            Option price.
        """
        S_grid, V = self.solve(instrument, r, q)
        return float(np.interp(spot, S_grid, V))

    def greeks(
        self,
        instrument : Instrument,
        spot       : float,
        r          : float,
        q          : float = 0.0,
    ) -> dict:
        """
        Compute Delta and Gamma from the PDE solution using finite differences
        on the V-grid. This is essentially free — the grid gives us V(S) for
        all S, so derivatives are just grid differences.

        Returns
        -------
        dict with keys: price, delta, gamma
        """
        S_grid, V = self.solve(instrument, r, q)

        # Find the grid index closest to spot
        idx = int(np.searchsorted(S_grid, spot))
        idx = np.clip(idx, 1, len(S_grid) - 2)

        dS = S_grid[idx + 1] - S_grid[idx - 1]
        dS_sq = (S_grid[idx + 1] - S_grid[idx - 1]) / 2

        price = float(np.interp(spot, S_grid, V))
        delta = float((V[idx + 1] - V[idx - 1]) / dS)
        gamma = float((V[idx + 1] - 2 * V[idx] + V[idx - 1]) / dS_sq**2)

        return {"price": price, "delta": delta, "gamma": gamma}

    def early_exercise_boundary(
        self,
        instrument : AmericanOption,
        r          : float,
        q          : float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Extract the early exercise boundary S*(t) for an American option.

        S*(t) is the critical spot price at which early exercise is optimal.
        For a put: exercise if S < S*(t). For a call: exercise if S > S*(t).

        This is found by comparing the PDE solution V(S, t) to the intrinsic
        value at each time step: S*(t) = the spot where V = intrinsic.

        Returns
        -------
        times : np.ndarray
            Time values from 0 to T.
        boundary : np.ndarray
            Critical spot S*(t) at each time.
        """
        K = instrument.strike
        T = instrument.expiry
        S_max = self.S_max_multiplier * K

        S_grid = np.linspace(0, S_max, self.M + 1)
        dt = T / self.N

        V = instrument.payoff(S_grid).copy()
        boundaries = []
        times = []

        i_int = np.arange(1, self.M)
        i = i_int.astype(float)
        alpha = 0.25 * dt * (-self.sigma**2 * i**2 + (r - q) * i)
        beta  = 0.50 * dt * ( self.sigma**2 * i**2 + r)
        gamma = 0.25 * dt * (-self.sigma**2 * i**2 - (r - q) * i)

        n_int = self.M - 1
        A_banded = np.zeros((3, n_int))
        A_banded[0, 1:]  = gamma[:-1]
        A_banded[1, :]   =  1 + beta
        A_banded[2, :-1] = alpha[1:]

        B_sub   = -alpha[1:]
        B_diag  =  1 - beta
        B_super = -gamma[:-1]

        for n in range(self.N):
            V_old = V[1:-1].copy()
            rhs = B_diag * V_old
            rhs[1:]  += B_sub   * V_old[:-1]
            rhs[:-1] += B_super * V_old[1:]

            tau = (n + 1) * dt
            V_lower, V_upper = self._boundary_conditions(instrument, S_max, r, q, T - tau)
            rhs[0]  -= alpha[0]  * V_lower
            rhs[-1] -= gamma[-1] * V_upper

            V_new = solve_banded((1, 1), A_banded, rhs)
            V[0]  = V_lower
            V[-1] = V_upper
            V[1:-1] = V_new

            intrinsic = instrument.payoff(S_grid)
            V = np.maximum(V, intrinsic)

            # Find boundary: where V equals intrinsic (exercise is optimal)
            diff = V - intrinsic
            from options_lib.instruments.base import OptionType
            if instrument.option_type == OptionType.PUT:
                # For put: boundary is highest S where V = intrinsic
                exercise_region = np.where(diff < 1e-6)[0]
                if len(exercise_region) > 0:
                    boundaries.append(S_grid[exercise_region[-1]])
                else:
                    boundaries.append(0.0)
            else:
                # For call: boundary is lowest S where V = intrinsic
                exercise_region = np.where(diff < 1e-6)[0]
                if len(exercise_region) > 0:
                    boundaries.append(S_grid[exercise_region[0]])
                else:
                    boundaries.append(S_max)

            times.append(T - tau)

        return np.array(times[::-1]), np.array(boundaries[::-1])

    def __repr__(self) -> str:
        return f"CrankNicolson(sigma={self.sigma}, M={self.M}, N={self.N})"