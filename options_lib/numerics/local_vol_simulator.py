"""
numerics/local_vol_simulator.py
--------------------------------
Monte Carlo path simulation under the Dupire local volatility model.

Stock process under local vol:
    dS = (r - q) S dt + σ_loc(S, t) S dW^Q

σ_loc(S, t) is a deterministic function of spot and time, extracted
from the market implied vol surface via Dupire's formula.

Why Local Vol Paths Are Different
-----------------------------------
Under flat BS vol, the diffusion coefficient is constant: σ S dW.
Under local vol, the coefficient changes at every point in time and space:
σ_loc(S_t, t) S_t dW. As the spot moves, the vol automatically adjusts.

This means:
  - The vol surface is automatically reproduced at t=0 (calibration exact)
  - The forward smile is determined by the surface slope: skew → skew
  - Spot moves toward the barrier → vol changes according to σ_loc(S, t)

This last point is crucial for barriers: as the spot approaches the barrier,
the local vol (extracted from the market smile near that level) governs
the probability of hitting the barrier. Flat BS vol misses this entirely.

Discretisation
--------------
We use the Euler-Maruyama scheme with log-transformation for stability:

    log(S_{t+dt}) = log(S_t) + (r - q - σ_loc²/2)*dt + σ_loc*√dt*Z

where σ_loc = σ_loc(S_t, t) is evaluated at the CURRENT spot and time.

This is the Euler scheme applied to log(S), which is more stable than
applying it to S directly (avoids negative spots).

For accuracy, a smaller time step is recommended:
  - Barrier options: n_steps ≥ 252 (daily) to avoid discretisation bias
  - The Broadie-Glasserman-Kou (1997) correction adjusts for the fact that
    continuous barrier monitoring differs from discrete monitoring.

Milstein Correction
--------------------
The Milstein scheme adds a correction term for better strong convergence:

    log(S_{t+dt}) = log(S_t) + (r - q - σ²/2)*dt + σ*√dt*Z
                    + (1/2) * (dσ/dS * S) * σ * (Z²-1)*dt

This O(dt) correction reduces strong error from O(√dt) to O(dt).
We implement the simpler Euler scheme by default (sufficient for pricing)
and offer Milstein as an option.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional

from options_lib.market_data.local_vol import LocalVolSurface


@dataclass
class LocalVolSimulator:
    """
    Monte Carlo path simulator under Dupire local volatility.

    Parameters
    ----------
    lv_surface : LocalVolSurface
        Calibrated local vol surface σ_loc(S, t).
    n_paths : int
        Number of simulation paths.
    n_steps : int
        Number of time steps. For barriers, use ≥ 252.
    antithetic : bool
        Use antithetic variates for variance reduction.
    milstein : bool
        Use Milstein correction for better strong convergence.
        Default False (Euler is sufficient for pricing).
    seed : int, optional
        Random seed.

    Notes
    -----
    The local vol surface is evaluated by interpolation at each
    (S_t, t) point along the path. This means one surface lookup
    per path per time step — O(n_paths * n_steps) evaluations.

    For n_paths=100,000 and n_steps=252, this is ~25 million lookups.
    The bicubic spline interpolant in LocalVolSurface is fast enough
    (~1 microsecond per call) but the total is ~25 seconds.

    For production: parallelise with numpy vectorisation over paths
    at each time step (evaluate all path spots simultaneously).
    """

    lv_surface : LocalVolSurface
    n_paths    : int   = 50_000
    n_steps    : int   = 252
    antithetic : bool  = True
    milstein   : bool  = False
    seed       : Optional[int] = None

    def simulate(
        self,
        S0 : float,
        T  : float,
        r  : float,
        q  : float = 0.0,
    ) -> np.ndarray:
        """
        Simulate stock price paths under local volatility.

        Parameters
        ----------
        S0 : float   Initial spot price.
        T  : float   Time horizon in years.
        r  : float   Risk-free rate.
        q  : float   Continuous dividend yield.

        Returns
        -------
        paths : np.ndarray, shape (n_paths, n_steps + 1)
            Simulated spot prices. Column 0 = S0, column -1 = S_T.
        """
        if self.seed is not None:
            np.random.seed(self.seed)

        dt      = T / self.n_steps
        n_paths = self.n_paths

        # Generate standard normal increments
        if self.antithetic:
            half   = n_paths // 2
            Z_half = np.random.standard_normal((half, self.n_steps))
            Z      = np.concatenate([Z_half, -Z_half], axis=0)
        else:
            Z = np.random.standard_normal((n_paths, self.n_steps))

        # Initialise path array
        paths        = np.zeros((n_paths, self.n_steps + 1))
        paths[:, 0]  = S0

        # ------------------------------------------------------------------
        # Time-stepping loop
        # At each step j, evaluate σ_loc at every current spot simultaneously
        # ------------------------------------------------------------------
        for j in range(self.n_steps):
            t_j = j * dt
            S_j = paths[:, j]

            # Vectorised local vol lookup at all spots S_j simultaneously
            # Clip spots to the grid range for stability
            S_j_clipped = np.clip(
                S_j,
                self.lv_surface.S_grid[0],
                self.lv_surface.S_grid[-1]
            )
            t_clipped = np.clip(t_j, self.lv_surface.T_grid[0], self.lv_surface.T_grid[-1])

            # Batch evaluate using the 2D spline
            sigma_loc = self._batch_local_vol(S_j_clipped, t_clipped)

            # Euler-Maruyama in log-space
            log_drift = (r - q - 0.5 * sigma_loc**2) * dt
            diffusion = sigma_loc * np.sqrt(dt) * Z[:, j]

            if self.milstein:
                # Milstein correction: (1/2) * σ * dσ/dS * S * (Z² - 1) * dt
                # Approximate dσ/dS by finite difference on σ_loc surface
                dS      = S_j_clipped * 0.01
                S_up    = np.clip(S_j_clipped + dS, self.lv_surface.S_grid[0],
                                  self.lv_surface.S_grid[-1])
                sig_up  = self._batch_local_vol(S_up, t_clipped)
                dsig_dS = (sig_up - sigma_loc) / dS
                milstein_corr = (0.5 * sigma_loc * dsig_dS * S_j_clipped
                                 * (Z[:, j]**2 - 1) * dt)
                diffusion = diffusion + milstein_corr

            paths[:, j + 1] = S_j * np.exp(log_drift + diffusion)

        return paths

    def _batch_local_vol(
        self, S_array: np.ndarray, t: float
    ) -> np.ndarray:
        """
        Evaluate local vol for all spots in S_array at time t.
        Uses the spline interpolant for vectorised evaluation.

        Parameters
        ----------
        S_array : np.ndarray   Spot prices, shape (n_paths,).
        t : float              Current time.

        Returns
        -------
        np.ndarray, shape (n_paths,)   Local vol at each spot.
        """
        if self.lv_surface._interpolant is not None:
            # Vectorised spline evaluation: ev(t, S_array) returns shape (n_paths,)
            sigma_loc = self.lv_surface._interpolant.ev(
                np.full_like(S_array, t), S_array
            )
        else:
            # Fallback: loop (slow)
            sigma_loc = np.array([
                self.lv_surface.local_vol(float(s), t) for s in S_array
            ])

        return np.clip(sigma_loc, 0.01, 5.0)

    def __repr__(self) -> str:
        return (
            f"LocalVolSimulator(n_paths={self.n_paths:,}, "
            f"n_steps={self.n_steps}, antithetic={self.antithetic})"
        )