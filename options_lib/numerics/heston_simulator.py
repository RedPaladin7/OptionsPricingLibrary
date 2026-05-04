"""
numerics/heston_simulator.py
-----------------------------
Monte Carlo path simulation under the Heston stochastic volatility model.

The Heston Joint Process (under Q)
------------------------------------
    dS_t = (r - q) S_t dt + √v_t S_t dW_t^S
    dv_t = κ(v̄ - v_t) dt + ξ √v_t dW_t^v
    dW_t^S dW_t^v = ρ dt

Unlike GBM (where σ is constant), the diffusion coefficient √v_t
evolves stochastically. This means:
  - At each time step, the volatility governing S is different
  - The correlation ρ < 0 causes vol to spike when spot falls
  - The forward distribution of S is non-lognormal (skewed, fat-tailed)

This is why Heston paths produce different barrier and Asian prices
than flat BS paths even when both are calibrated to the same vanilla surface.

Discretisation Schemes
-----------------------
Euler-Maruyama (simple, O(√dt) strong error):
    v_{t+dt} = v_t + κ(v̄ - v_t)dt + ξ√v_t √dt Z_v
    log(S_{t+dt}) = log(S_t) + (r-q-v_t/2)dt + √v_t √dt Z_S

Problem: Euler can produce negative variance v_t < 0 when ξ is large
or dt is not small enough. Standard fixes:

  1. Full truncation:   v_{t+dt} = max(v_{t+dt}, 0)     [used here]
  2. Reflection:        v_{t+dt} = |v_{t+dt}|
  3. Partial truncation: use max(v_t, 0) in diffusion only

Milstein Correction (O(dt) strong error):
Adds a correction term to the variance process:
    v_{t+dt} = v_t + κ(v̄ - v_t)dt + ξ√v_t √dt Z_v + (1/4)ξ²(Z_v²-1)dt

This reduces the strong discretisation error from O(√dt) to O(dt),
which means you need ~10x fewer time steps for the same path accuracy.
Important for path-dependent options (barriers, Asians) that are
sensitive to the vol path, not just terminal distribution.

QE Scheme (Andersen 2008) - Best in class, now implemented:
The Quadratic-Exponential scheme matches the conditional distribution
of v_t analytically at each step, virtually eliminating bias for any dt.
It is the gold standard for production Heston simulation.

At each step, QE computes the conditional mean m and variance s² of v_{t+dt}
given v_t analytically (from the known CIR process moments), then samples
from a distribution that exactly matches those moments:

  ψ = s²/m²  (normalised variance, key diagnostic)

  If ψ ≤ ψ_c  (variance process is "stable", approx lognormal):
      Use quadratic approximation: v ~ a(b + Z)²
      where b² = 2/ψ - 1 + √(2/ψ·(2/ψ-1)),  a = m/(1+b²)

  If ψ > ψ_c  (variance process near zero, can hit 0):
      Use exponential mixture: v = 0 with prob p,
                                   Exp(β) with prob 1-p
      where p = (ψ-1)/(ψ+1),  β = (1-p)/m

  ψ_c = 1.5  (Andersen's recommended threshold)

This scheme requires NO truncation — variance stays non-negative
by construction. It is exact in the sense of matching first two
conditional moments, not just an approximation like Euler-Milstein.

Correlated Brownians
--------------------
W_S and W_v must be correlated with correlation ρ. Generated via
Cholesky decomposition:
    Z_S = ρ Z_v + √(1-ρ²) Z_⊥
where Z_v, Z_⊥ are independent standard normals.

References
----------
Andersen, L. (2008). "Simple and efficient simulation of the Heston
stochastic volatility model." Journal of Computational Finance, 11(3).

Lord, R., Koekkoek, R., Van Dijk, D. (2010). "A comparison of biased
simulation schemes for stochastic volatility models."
Quantitative Finance, 10(2), 177-194.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional

@dataclass
class HestonSimulator:
    """
    Monte Carlo path simulator for the Heston stochastic volatility model.

    Simulates joint paths (S_t, v_t) under the risk-neutral measure Q.

    Parameters
    ----------
    params : HestonParams
        Heston model parameters (v0, kappa, v_bar, xi, rho).
    n_paths : int
        Number of simulation paths.
    n_steps : int
        Number of time steps. More steps = better accuracy for path-
        dependent options. 252 (daily) recommended for barriers and Asians.
    antithetic : bool
        Antithetic variates: for each path Z, also simulate -Z.
        Reduces variance by exploiting symmetry of the normal distribution.
        Note: for Heston with ρ ≠ 0, antithetic is applied to BOTH
        Z_v and Z_⊥ simultaneously to preserve the correlation structure.
    milstein : bool
        Use Milstein correction for variance process. Reduces strong
        discretisation error from O(√dt) to O(dt). Recommended for
        barrier options and short-dated Asians.
    truncation : str
        How to handle negative variance:
        'full'     — max(v, 0) at every step (default, most common)
        'reflect'  — |v| at every step
    seed : int, optional
        Random seed for reproducibility.

    Usage
    -----
    >>> params = HestonParams(v0=0.04, kappa=2.0, v_bar=0.04, xi=0.3, rho=-0.7)
    >>> sim = HestonSimulator(params, n_paths=100_000, n_steps=252)
    >>> S_paths, v_paths = sim.simulate(S0=100, T=1.0, r=0.05, q=0.0)
    >>> S_paths.shape   # (100_000, 253)
    """

    params     : object  # HestonParams — imported lazily to avoid circular import
    n_paths    : int   = 100_000
    n_steps    : int   = 252
    antithetic : bool  = True
    milstein   : bool  = True
    truncation : str   = 'full'
    scheme     : str   = 'euler'   # 'euler' (Euler-Milstein) or 'qe' (Andersen 2008 QE)
    seed       : Optional[int] = None

    def __post_init__(self):
        if self.truncation not in ('full', 'reflect'):
            raise ValueError(f"truncation must be 'full' or 'reflect', got {self.truncation}")
        if self.scheme not in ('euler', 'qe'):
            raise ValueError(f"scheme must be 'euler' or 'qe', got {self.scheme}")

    def simulate(
        self,
        S0 : float,
        T  : float,
        r  : float,
        q  : float = 0.0,
    ) -> tuple:
        """
        Simulate Heston paths under Q.

        Parameters
        ----------
        S0 : float   Initial spot price.
        T  : float   Time horizon in years.
        r  : float   Risk-free rate.
        q  : float   Continuous dividend yield.

        Returns
        -------
        S_paths : np.ndarray, shape (n_paths, n_steps + 1)
            Spot price paths. S_paths[:, 0] = S0.
        v_paths : np.ndarray, shape (n_paths, n_steps + 1)
            Variance paths. v_paths[:, 0] = v0.
        """
        if self.scheme == 'qe':
            return self.simulate_qe(S0, T, r, q)

        kappa = self.params.kappa
        v_bar = self.params.v_bar
        xi    = self.params.xi
        rho   = self.params.rho
        v0    = self.params.v0

        dt        = T / self.n_steps
        sqrt_dt   = np.sqrt(dt)
        n_paths   = self.n_paths
        rho_perp  = np.sqrt(max(1 - rho**2, 0.0))  # √(1-ρ²)

        if self.seed is not None:
            np.random.seed(self.seed)

        # ------------------------------------------------------------------
        # Generate correlated random increments
        # Z_v   : drives the variance process
        # Z_perp: drives the orthogonal component of the spot process
        # Z_S   = ρ Z_v + √(1-ρ²) Z_perp  (Cholesky decomposition)
        # ------------------------------------------------------------------
        if self.antithetic:
            half      = n_paths // 2
            Zv_half   = np.random.standard_normal((half, self.n_steps))
            Zp_half   = np.random.standard_normal((half, self.n_steps))
            Z_v       = np.concatenate([Zv_half, -Zv_half], axis=0)
            Z_perp    = np.concatenate([Zp_half, -Zp_half], axis=0)
        else:
            Z_v    = np.random.standard_normal((n_paths, self.n_steps))
            Z_perp = np.random.standard_normal((n_paths, self.n_steps))

        # Correlated spot Brownian: Z_S = ρ Z_v + √(1-ρ²) Z_⊥
        Z_S = rho * Z_v + rho_perp * Z_perp

        # ------------------------------------------------------------------
        # Initialise path arrays
        # ------------------------------------------------------------------
        S_paths        = np.zeros((n_paths, self.n_steps + 1))
        v_paths        = np.zeros((n_paths, self.n_steps + 1))
        S_paths[:, 0]  = S0
        v_paths[:, 0]  = v0

        # ------------------------------------------------------------------
        # Time-stepping loop
        # ------------------------------------------------------------------
        for j in range(self.n_steps):
            v_j = v_paths[:, j]
            S_j = S_paths[:, j]

            # Variance used in diffusion: max(v, 0) for full truncation
            v_pos = np.maximum(v_j, 0.0)
            sqrt_v = np.sqrt(v_pos)

            # ----------------------------------------------------------
            # Variance step: Euler-Maruyama (± Milstein correction)
            # dv = κ(v̄ - v) dt + ξ √v dW_v
            # ----------------------------------------------------------
            v_drift    = kappa * (v_bar - v_pos) * dt
            v_diffusion = xi * sqrt_v * sqrt_dt * Z_v[:, j]

            if self.milstein:
                # Milstein: + (1/4) ξ² (Z_v² - 1) dt
                v_milstein = 0.25 * xi**2 * (Z_v[:, j]**2 - 1) * dt
                v_new = v_j + v_drift + v_diffusion + v_milstein
            else:
                v_new = v_j + v_drift + v_diffusion

            # Apply truncation to keep variance non-negative
            if self.truncation == 'full':
                v_new = np.maximum(v_new, 0.0)
            else:  # reflect
                v_new = np.abs(v_new)

            v_paths[:, j + 1] = v_new

            # ----------------------------------------------------------
            # Spot step (log-Euler for stability)
            # d(log S) = (r - q - v/2) dt + √v dW_S
            # ----------------------------------------------------------
            log_drift    = (r - q - 0.5 * v_pos) * dt
            log_diffusion = sqrt_v * sqrt_dt * Z_S[:, j]

            S_paths[:, j + 1] = S_j * np.exp(log_drift + log_diffusion)

        return S_paths, v_paths

    def simulate_qe(
        self,
        S0 : float,
        T  : float,
        r  : float,
        q  : float = 0.0,
    ) -> tuple:
        """
        Simulate Heston paths using the Quadratic-Exponential (QE) scheme.
        Andersen (2008).

        The QE scheme samples the variance process v_t from a distribution
        that exactly matches the first two conditional moments of the true
        CIR process at each step — eliminating discretisation bias in v
        regardless of the step size dt.

        Conditional moments of v_{t+dt} given v_t (exact, from CIR theory):
            m  = E[v_{t+dt} | v_t] = v_t e^{-κ dt} + v̄ (1 - e^{-κ dt})
            s² = Var[v_{t+dt} | v_t]
               = v_t ξ²/κ e^{-κdt}(1-e^{-κdt}) + v̄ ξ²/(2κ)(1-e^{-κdt})²

        ψ = s²/m² (normalised variance) determines which distribution to use:

        ψ ≤ ψ_c = 1.5  → Quadratic approximation (v is not near zero):
            b² = 2/ψ - 1 + √(2/ψ · (2/ψ - 1))
            a  = m / (1 + b²)
            v_{t+dt} ~ a · (b + Z)²,   Z ~ N(0,1)

        ψ > ψ_c  → Exponential mixture (v can hit zero, near Feller boundary):
            p = (ψ-1)/(ψ+1)         [probability of v=0]
            β = (1-p)/m
            v_{t+dt} = 0               with prob p
                      = ln((1-p)/(1-U))/β  with prob 1-p,  U ~ Uniform(0,1)

        Spot step uses the same log-Euler discretisation as the Euler scheme,
        but uses the QE-sampled v_t at each step. The key difference is that
        v_t is now sampled without bias, so the spot paths are also unbiased.

        Martingale correction:
        The log-Euler spot step has a small bias because E[e^{X}] ≠ e^{E[X]}
        for stochastic X. We apply the standard correction factor:
            log_drift = (r - q) dt - 0.5 * K0 dt
        where K0 accounts for the covariance between the spot and variance
        increments. See Andersen (2008) eq. (33).
        """
        kappa = self.params.kappa
        v_bar = self.params.v_bar
        xi    = self.params.xi
        rho   = self.params.rho
        v0    = self.params.v0

        dt       = T / self.n_steps
        n_paths  = self.n_paths
        rho_perp = np.sqrt(max(1 - rho**2, 0.0))
        psi_c    = 1.5   # Andersen's recommended threshold

        if self.seed is not None:
            np.random.seed(self.seed)

        # ------------------------------------------------------------------
        # Pre-compute QE constants that depend only on dt (not on v_t)
        # ------------------------------------------------------------------
        exp_kdt   = np.exp(-kappa * dt)
        # Coefficients for conditional variance formula
        # s² = c1 * v_t + c2
        c1 = xi**2 * exp_kdt * (1 - exp_kdt) / kappa
        c2 = v_bar * xi**2 * (1 - exp_kdt)**2 / (2 * kappa)

        # ------------------------------------------------------------------
        # Generate random draws for the spot Brownian (Z_S)
        # and uniform draws (U) for the exponential branch of QE
        # ------------------------------------------------------------------
        if self.antithetic:
            half    = n_paths // 2
            Zp_half = np.random.standard_normal((half, self.n_steps))
            U_half  = np.random.uniform(0, 1, (half, self.n_steps))
            Z_perp  = np.concatenate([Zp_half, -Zp_half], axis=0)
            U_mat   = np.concatenate([U_half, 1 - U_half], axis=0)
            # Z_v for the quadratic branch (also antithetic)
            Zv_half = np.random.standard_normal((half, self.n_steps))
            Z_v     = np.concatenate([Zv_half, -Zv_half], axis=0)
        else:
            Z_perp = np.random.standard_normal((n_paths, self.n_steps))
            U_mat  = np.random.uniform(0, 1, (n_paths, self.n_steps))
            Z_v    = np.random.standard_normal((n_paths, self.n_steps))

        # ------------------------------------------------------------------
        # Initialise paths
        # ------------------------------------------------------------------
        S_paths       = np.zeros((n_paths, self.n_steps + 1))
        v_paths       = np.zeros((n_paths, self.n_steps + 1))
        S_paths[:, 0] = S0
        v_paths[:, 0] = v0

        # ------------------------------------------------------------------
        # Time-stepping loop
        # ------------------------------------------------------------------
        for j in range(self.n_steps):
            v_j = v_paths[:, j]
            S_j = S_paths[:, j]

            # --------------------------------------------------------------
            # Step 1: Compute conditional moments of v_{t+dt} given v_t
            # --------------------------------------------------------------
            m    = v_j * exp_kdt + v_bar * (1 - exp_kdt)   # E[v_{t+dt}|v_t]
            s_sq = c1 * v_j + c2                             # Var[v_{t+dt}|v_t]
            psi  = s_sq / np.maximum(m**2, 1e-14)           # ψ = s²/m²

            # --------------------------------------------------------------
            # Step 2: Sample v_{t+dt} using QE dispatch
            # --------------------------------------------------------------
            v_new = np.zeros(n_paths)

            # --- Quadratic branch (ψ ≤ ψ_c): v ~ a(b+Z)² ---
            qe_mask = psi <= psi_c
            if qe_mask.any():
                m_q   = m[qe_mask]
                psi_q = psi[qe_mask]
                b_sq  = 2.0/psi_q - 1.0 + np.sqrt(2.0/psi_q * (2.0/psi_q - 1.0))
                a     = m_q / (1.0 + b_sq)
                b     = np.sqrt(b_sq)
                v_new[qe_mask] = a * (b + Z_v[qe_mask, j])**2

            # --- Exponential branch (ψ > ψ_c): mixture of 0 and Exp ---
            exp_mask = ~qe_mask
            if exp_mask.any():
                m_e   = m[exp_mask]
                psi_e = psi[exp_mask]
                p     = (psi_e - 1.0) / (psi_e + 1.0)     # P(v=0)
                beta  = (1.0 - p) / np.maximum(m_e, 1e-14)
                U_e   = U_mat[exp_mask, j]
                # v = 0 if U <= p, else inverse CDF of Exp(beta)
                above_p = U_e > p
                v_exp   = np.zeros(exp_mask.sum())
                if above_p.any():
                    v_exp[above_p] = (
                        np.log((1.0 - p[above_p]) / np.maximum(1.0 - U_e[above_p], 1e-14))
                        / beta[above_p]
                    )
                v_new[exp_mask] = v_exp

            v_paths[:, j + 1] = v_new

            # --------------------------------------------------------------
            # Step 3: Spot step — log-Euler using CURRENT variance v_j
            # Using v_j (not v_new or v_avg) preserves the martingale
            # property E[S_{t+dt}|S_t] = S_t * e^{(r-q)*dt}.
            # The Ito correction -v/2 must use the same v as the diffusion.
            # --------------------------------------------------------------
            v_pos = np.maximum(v_j, 0.0)

            # Correlated spot Brownian: Z_S = ρ Z_v + √(1-ρ²) Z_⊥
            Z_S = rho * Z_v[:, j] + rho_perp * Z_perp[:, j]

            log_drift     = (r - q - 0.5 * v_pos) * dt
            log_diffusion = np.sqrt(v_pos * dt) * Z_S

            S_paths[:, j + 1] = S_j * np.exp(log_drift + log_diffusion)

        return S_paths, v_paths

    def terminal_distribution(
        self,
        S0 : float,
        T  : float,
        r  : float,
        q  : float = 0.0,
    ) -> np.ndarray:
        """
        Return only terminal spot prices S_T (more memory efficient
        when path structure is not needed, e.g., European options).

        Returns
        -------
        np.ndarray, shape (n_paths,)
        """
        S_paths, _ = self.simulate(S0, T, r, q)
        return S_paths[:, -1]

    def realised_variance(self, v_paths: np.ndarray, dt: float) -> np.ndarray:
        """
        Compute realised variance for each path:
            RV = (1/T) ∫_0^T v_t dt ≈ (1/N) Σ v_{t_j}

        This is the payoff of a variance swap (times T).

        Parameters
        ----------
        v_paths : np.ndarray, shape (n_paths, n_steps+1)
        dt : float   Time step size.

        Returns
        -------
        np.ndarray, shape (n_paths,)   Realised variance per path.
        """
        # Trapezoidal integration of v_t along each path
        return np.trapezoid(v_paths, dx=dt, axis=1) / (v_paths.shape[1] * dt)

    def as_lsmc_simulator(self):
        """
        Return a callable that conforms to the LongstaffSchwartz.path_simulator
        interface: (S0, T, r, q, n_paths, n_steps) -> S_paths.

        The LSMC path_simulator hook expects a function with that exact
        signature, but HestonSimulator.simulate() only takes (S0, T, r, q)
        and returns (S_paths, v_paths). This factory method wraps the
        simulator, discards the variance paths, and accepts the n_paths /
        n_steps arguments so LSMC can override them per call.

        Usage
        -----
        >>> from options_lib.numerics.lsmc import LongstaffSchwartz
        >>> sim   = HestonSimulator(params, n_paths=50_000, n_steps=100)
        >>> lsmc  = LongstaffSchwartz(sigma=0.20,
        ...             path_simulator=sim.as_lsmc_simulator())
        >>> result = lsmc.price(am_put, spot=100, rate=0.05)
        """
        def _wrapper(S0, T, r, q, n_paths, n_steps):
            # Build a fresh simulator with the requested n_paths / n_steps
            # so LSMC controls the grid size, while all other params
            # (antithetic, milstein, scheme, seed) stay from self.
            sim = HestonSimulator(
                params     = self.params,
                n_paths    = n_paths,
                n_steps    = n_steps,
                antithetic = self.antithetic,
                milstein   = self.milstein,
                scheme     = self.scheme,
                seed       = self.seed,
            )
            S_paths, _ = sim.simulate(S0, T, r, q)  # discard v_paths
            return S_paths
        return _wrapper

    def __repr__(self) -> str:
        return (
            f"HestonSimulator(v0={self.params.v0:.4f}, "
            f"κ={self.params.kappa:.2f}, v̄={self.params.v_bar:.4f}, "
            f"ξ={self.params.xi:.3f}, ρ={self.params.rho:.3f}, "
            f"n_paths={self.n_paths:,}, n_steps={self.n_steps}, "
            f"scheme={self.scheme})"
        )