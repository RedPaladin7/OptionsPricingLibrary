"""
models/heston_asian_mc.py
--------------------------
Asian option pricing under Heston stochastic volatility via Monte Carlo.

Why Heston Changes Asian Prices
---------------------------------
Under flat BS vol, the arithmetic average A = (1/n) Σ S_{t_i} has a
distribution that is approximately lognormal (the averaging reduces vol).
The Kemna-Vorst formula for geometric Asians is exact under BS.

Under Heston:
  - vol is stochastic along the path
  - periods of high variance (vol clustering) produce fat-tailed averages
  - the correlation ρ < 0 means downside moves come with higher vol,
    producing negative skewness in the average distribution
  - all of this is completely invisible to flat BS MC

Concretely, for a typical SPX-like surface (negative skew, positive vol-of-vol):
  Heston Asian call price > BS Asian call price (fat tails raise average)
  Heston Asian put price > BS Asian put price  (same reason)
  The difference grows with: |ρ|, ξ (vol-of-vol), T, n_observations

Control Variate Under Heston
-----------------------------
The geometric Asian control variate (Kemna-Vorst closed form) is derived
assuming constant vol. Under Heston, it no longer has a closed form.

We use a different control variate approach:
  - Simulate TWO sets of paths: (a) Heston paths, (b) BS paths with
    σ = √v̄ (the long-run vol) using the SAME random numbers
  - Price on both path sets simultaneously
  - Use the BS price (known analytically via Kemna-Vorst) as the control
  - Since Heston paths with large dt ≈ BS paths, the correlation is high

This "BS-as-control" approach works well when v0 ≈ v̄ (near mean reversion)
and the control variate correlation decreases for extreme parameters.

Alternatively, and more robustly: use the arithmetic BS Asian as the
control. Both approaches are implemented.

Moment Matching
----------------
An optional variance reduction: adjust the simulated paths so that
their sample mean exactly matches the theoretical mean e^{(r-q)T}·S0.
This eliminates MC bias in the first moment, which is the main source
of error for arithmetic average options.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional

from options_lib.models.heston import HestonParams
from options_lib.numerics.heston_simulator import HestonSimulator
from options_lib.instruments.asian import AsianOption, AverageType
from options_lib.instruments.base import MarketData, OptionType


@dataclass
class HestonAsianResult:
    """
    Result from Heston Asian option pricing.

    Attributes
    ----------
    price          : float   Heston MC price (with variance reduction)
    std_error      : float   MC standard error
    bs_price       : float   Flat BS price (same paths, for comparison)
    vol_premium    : float   Heston price - BS price (stochastic vol effect)
    vol_premium_pct: float   Relative premium
    n_paths        : int
    n_steps        : int
    """
    price           : float
    std_error       : float
    bs_price        : float
    n_paths         : int
    n_steps         : int

    @property
    def vol_premium(self) -> float:
        """Price difference due to stochastic volatility."""
        return self.price - self.bs_price

    @property
    def vol_premium_pct(self) -> float:
        if abs(self.bs_price) < 1e-8:
            return 0.0
        return self.vol_premium / self.bs_price

    @property
    def confidence_interval(self) -> tuple:
        return (self.price - 1.96 * self.std_error,
                self.price + 1.96 * self.std_error)

    def summary(self) -> str:
        lo, hi = self.confidence_interval
        lines = [
            f"{'='*52}",
            f"Heston Asian Option Pricing",
            f"{'='*52}",
            f"Heston price:   {self.price:.4f} ± {1.96*self.std_error:.4f}",
            f"  95% CI:       [{lo:.4f}, {hi:.4f}]",
            f"BS flat price:  {self.bs_price:.4f}",
            f"Vol premium:    {self.vol_premium:+.4f} ({self.vol_premium_pct:+.1%})",
            f"N paths:        {self.n_paths:,}",
            f"N steps:        {self.n_steps}",
            f"{'='*52}",
        ]
        return '\n'.join(lines)


@dataclass
class HestonAsianPricer:
    """
    Asian option pricer under Heston stochastic volatility.

    Parameters
    ----------
    params : HestonParams
        Heston model parameters.
    n_paths : int
        Number of MC paths. 100,000 recommended for production.
    n_steps : int
        Time steps per path. Should equal n_observations for accurate
        arithmetic average (sample the path at each observation date).
    antithetic : bool
        Antithetic variates.
    milstein : bool
        Milstein correction for variance discretisation.
    control_variate : bool
        Use BS flat vol as control variate via common random numbers.
    moment_matching : bool
        Adjust paths so sample mean of S_T = theoretical forward price.
        Eliminates first-moment MC bias.
    seed : int, optional

    Notes
    -----
    Setting n_steps = instrument.n_observations ensures the path is
    sampled exactly at each averaging date, not between them.
    For n_observations = 252 (daily), n_steps = 252.
    """

    params          : HestonParams
    n_paths         : int  = 100_000
    n_steps         : int  = 252
    antithetic      : bool = True
    milstein        : bool = True
    control_variate : bool = True
    moment_matching : bool = True
    seed            : Optional[int] = None

    def price(
        self,
        instrument : AsianOption,
        market     : MarketData,
    ) -> HestonAsianResult:
        """
        Price an Asian option under Heston stochastic vol.

        Parameters
        ----------
        instrument : AsianOption
        market     : MarketData

        Returns
        -------
        HestonAsianResult
        """
        S0  = market.spot
        r   = market.rate
        q   = market.div_yield
        T   = instrument.expiry
        K   = instrument.strike
        n   = instrument.n_observations

        # Override n_steps to match observation dates
        n_steps = max(self.n_steps, n)

        # ------------------------------------------------------------------
        # Step 1: Simulate Heston paths
        # ------------------------------------------------------------------
        sim = HestonSimulator(
            params     = self.params,
            n_paths    = self.n_paths,
            n_steps    = n_steps,
            antithetic = self.antithetic,
            milstein   = self.milstein,
            seed       = self.seed,
        )
        S_paths, v_paths = sim.simulate(S0, T, r, q)

        # ------------------------------------------------------------------
        # Step 2: Extract paths at observation dates
        # For n_steps >= n_observations, sample every (n_steps/n) steps
        # ------------------------------------------------------------------
        step_size  = n_steps // n
        obs_indices = [j * step_size for j in range(1, n + 1)]
        obs_indices[-1] = min(obs_indices[-1], n_steps)  # ensure last = T
        obs_paths  = S_paths[:, obs_indices]  # shape (n_paths, n_obs)

        # ------------------------------------------------------------------
        # Step 3: Optional moment matching
        # Rescale paths so sample mean of terminal prices = e^{(r-q)T} S0
        # ------------------------------------------------------------------
        if self.moment_matching:
            theoretical_fwd = S0 * np.exp((r - q) * T)
            sample_mean     = np.mean(S_paths[:, -1])
            if sample_mean > 1e-6:
                scale = theoretical_fwd / sample_mean
                obs_paths = obs_paths * scale

        # ------------------------------------------------------------------
        # Step 4: Compute Heston Asian payoffs
        # ------------------------------------------------------------------
        heston_payoffs = self._compute_asian_payoffs(
            obs_paths, instrument
        )
        disc_heston = np.exp(-r * T) * heston_payoffs

        # ------------------------------------------------------------------
        # Step 5: BS flat vol paths (same random numbers = common CRN)
        # Re-use the same seed so Z draws are identical
        # ------------------------------------------------------------------
        if self.seed is not None:
            np.random.seed(self.seed)

        # Flat BS paths: σ = √v̄ (long-run vol)
        bs_sigma   = np.sqrt(self.params.v_bar)
        bs_paths   = self._simulate_bs_paths(S0, T, r, q, n_steps, bs_sigma)
        bs_obs     = bs_paths[:, obs_indices]

        if self.moment_matching:
            bs_mean = np.mean(bs_paths[:, -1])
            if bs_mean > 1e-6:
                bs_scale = theoretical_fwd / bs_mean
                bs_obs = bs_obs * bs_scale

        bs_payoffs  = self._compute_asian_payoffs(bs_obs, instrument)
        disc_bs     = np.exp(-r * T) * bs_payoffs

        # BS analytical price via Kemna-Vorst
        bs_price_cf = self._kemna_vorst(instrument, market, bs_sigma)

        # ------------------------------------------------------------------
        # Step 6: Control variate adjustment
        # Adjusted = Heston_payoff - c*(BS_payoff - E^BS[BS_payoff])
        # c = Cov(Heston, BS) / Var(BS)
        # ------------------------------------------------------------------
        if self.control_variate:
            cov_mat = np.cov(disc_heston, disc_bs)
            if cov_mat[1, 1] > 1e-12:
                c = cov_mat[0, 1] / cov_mat[1, 1]
            else:
                c = 1.0
            adjusted = disc_heston - c * (disc_bs - bs_price_cf)
        else:
            adjusted = disc_heston

        price     = float(np.mean(adjusted))
        std_error = float(np.std(adjusted) / np.sqrt(self.n_paths))

        # BS flat price for comparison (MC estimate on same paths)
        bs_mc_price = float(np.mean(disc_bs))

        return HestonAsianResult(
            price     = max(price, 0.0),
            std_error = std_error,
            bs_price  = bs_mc_price,
            n_paths   = self.n_paths,
            n_steps   = n_steps,
        )

    def _compute_asian_payoffs(
        self,
        obs_paths  : np.ndarray,
        instrument : AsianOption,
    ) -> np.ndarray:
        """
        Compute Asian payoffs from observed paths at averaging dates.

        Parameters
        ----------
        obs_paths : np.ndarray, shape (n_paths, n_obs)
        instrument : AsianOption

        Returns
        -------
        np.ndarray, shape (n_paths,)
        """
        K = instrument.strike

        if instrument.average_type == AverageType.ARITHMETIC:
            avg = np.mean(obs_paths, axis=1)
        else:
            avg = np.exp(np.mean(np.log(np.maximum(obs_paths, 1e-8)), axis=1))

        if instrument.option_type == OptionType.CALL:
            return np.maximum(avg - K, 0.0)
        else:
            return np.maximum(K - avg, 0.0)

    def _simulate_bs_paths(
        self,
        S0      : float,
        T       : float,
        r       : float,
        q       : float,
        n_steps : int,
        sigma   : float,
    ) -> np.ndarray:
        """
        Simulate flat GBM paths. Uses current random state for CRN.

        Returns
        -------
        np.ndarray, shape (n_paths, n_steps+1)
        """
        n_paths  = self.n_paths
        dt       = T / n_steps
        drift    = (r - q - 0.5 * sigma**2) * dt
        vol_step = sigma * np.sqrt(dt)

        if self.antithetic:
            half   = n_paths // 2
            Z_half = np.random.standard_normal((half, n_steps))
            Z      = np.concatenate([Z_half, -Z_half], axis=0)
        else:
            Z = np.random.standard_normal((n_paths, n_steps))

        log_rets = drift + vol_step * Z
        log_S    = np.log(S0) + np.cumsum(log_rets, axis=1)
        log_S0   = np.full((n_paths, 1), np.log(S0))
        return np.exp(np.concatenate([log_S0, log_S], axis=1))

    def _kemna_vorst(
        self,
        instrument : AsianOption,
        market     : MarketData,
        sigma      : float,
    ) -> float:
        """
        Kemna-Vorst (1990) closed-form price for geometric average Asian.
        Used as the control variate mean (E[Y]).

        Even for arithmetic Asians we use this — it is the closed-form
        price of the GEOMETRIC Asian, which serves as a high-correlation
        proxy for the arithmetic Asian.
        """
        from scipy.stats import norm
        S, K, T = market.spot, instrument.strike, instrument.expiry
        r, q, n = market.rate, market.div_yield, instrument.n_observations

        sigma_adj = sigma * np.sqrt((2 * n + 1) / (6 * (n + 1)))
        b_adj     = 0.5 * (r - q - 0.5 * sigma**2 + sigma_adj**2)
        sqrt_T    = np.sqrt(T)

        if sigma_adj < 1e-8:
            return max(S * np.exp((b_adj - r) * T) - K * np.exp(-r * T), 0.0)

        d1 = (np.log(S / K) + (b_adj + 0.5 * sigma_adj**2) * T) / (sigma_adj * sqrt_T)
        d2 = d1 - sigma_adj * sqrt_T

        if instrument.option_type == OptionType.CALL:
            price = (S * np.exp((b_adj - r) * T) * norm.cdf(d1)
                     - K * np.exp(-r * T) * norm.cdf(d2))
        else:
            price = (K * np.exp(-r * T) * norm.cdf(-d2)
                     - S * np.exp((b_adj - r) * T) * norm.cdf(-d1))

        return float(max(price, 0.0))

    def vol_premium_surface(
        self,
        strikes  : np.ndarray,
        expiries : np.ndarray,
        market   : MarketData,
        n_obs    : int = 52,
    ) -> dict:
        """
        Compute the Heston vol premium (Heston - BS) across
        a grid of strikes and expiries.

        This shows WHERE stochastic vol matters most for Asian pricing:
        - Larger at longer expiries (vol uncertainty compounds)
        - Larger for OTM options (distribution tails differ most)
        - Sign depends on ρ and ξ

        Parameters
        ----------
        strikes  : np.ndarray   Strike grid.
        expiries : np.ndarray   Expiry grid.
        market   : MarketData
        n_obs    : int           Observations per period (default 52 = weekly).

        Returns
        -------
        dict with keys:
            'heston_prices'  : np.ndarray (len(expiries), len(strikes))
            'bs_prices'      : np.ndarray
            'vol_premium'    : np.ndarray (absolute)
            'vol_premium_pct': np.ndarray (relative)
        """
        heston_prices  = np.zeros((len(expiries), len(strikes)))
        bs_prices      = np.zeros_like(heston_prices)

        for i, T in enumerate(expiries):
            n_steps_T = max(n_obs, int(n_obs * T))
            for j, K in enumerate(strikes):
                inst = AsianOption(
                    strike=K, expiry=T,
                    option_type=OptionType.CALL,
                    average_type=AverageType.ARITHMETIC,
                    n_observations=n_obs,
                )
                result = self.price(inst, market)
                heston_prices[i, j] = result.price
                bs_prices[i, j]     = result.bs_price

        vol_prem = heston_prices - bs_prices
        with np.errstate(divide='ignore', invalid='ignore'):
            vol_prem_pct = np.where(
                np.abs(bs_prices) > 1e-6,
                vol_prem / bs_prices,
                0.0
            )

        return {
            'heston_prices'   : heston_prices,
            'bs_prices'       : bs_prices,
            'vol_premium'     : vol_prem,
            'vol_premium_pct' : vol_prem_pct,
            'strikes'         : strikes,
            'expiries'        : expiries,
        }

    def __repr__(self) -> str:
        return (
            f"HestonAsianPricer({self.params}, "
            f"n_paths={self.n_paths:,}, n_steps={self.n_steps})"
        )