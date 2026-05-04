"""
models/local_vol_mc.py
-----------------------
Barrier option pricer under Dupire local volatility via Monte Carlo.

The Model Risk Problem for Barriers
--------------------------------------
Consider a down-and-out call with barrier H < S_0. Its price depends on:
  1. The probability of hitting H before expiry
  2. The payoff conditional on NOT hitting H

Both depend critically on how volatility evolves as the spot approaches H.

Three models, all calibrated to the SAME vanilla surface (same σ_BS(K,T)):

  Flat BS vol:    σ = const. As S falls toward H, vol is fixed.
                  → Transition probabilities don't adjust for the smile.

  Local vol:      σ_loc(S, t). As S falls, vol increases (negative skew
                  means lower spots have higher vol). The barrier is
                  approached with higher vol → higher knockout probability
                  → barrier option CHEAPER than flat vol suggests.

  Heston:         Stochastic vol with correlation ρ < 0. When S falls,
                  v tends to rise. But the spot-vol correlation mechanism
                  is different from local vol's deterministic adjustment.
                  → Different forward vol distribution → different barrier price.

Even though all three models produce identical European vanilla prices
at t=0, they produce DIFFERENT barrier prices because barriers depend
on the PATH of the spot — and the vol dynamics along that path differ.

This is "model risk": a real, daily concern on exotic desks.

Key Result (Empirical)
----------------------
For typical negative-skew equity surfaces:
  - Down-and-out calls: Local vol price < Flat vol price
    (LV generates higher vol near the barrier → higher knockout prob)
  - Up-and-out puts: Opposite effect
  - The difference is largest when the barrier is close to the spot

This module quantifies exactly this difference using the real calibrated
local vol surface, and compares it to flat BS vol pricing.

Continuity Correction (Broadie-Glasserman-Kou 1997)
------------------------------------------------------
Continuous barriers are monitored at all times in theory. MC monitors
at discrete time steps. The probability of hitting a continuous barrier
is HIGHER than discrete monitoring suggests.

The BGK correction shifts the barrier slightly to correct for this:
  - Down barrier: H_adj = H * exp(-0.5826 * σ * sqrt(dt))
  - Up barrier:   H_adj = H * exp(+0.5826 * σ * sqrt(dt))

This brings discrete MC results closer to continuous barrier prices.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional

from options_lib.instruments.barrier import BarrierOption, BarrierType
from options_lib.instruments.european import EuropeanOption
from options_lib.instruments.base import MarketData, OptionType
from options_lib.models.black_scholes import BlackScholes
from options_lib.market_data.local_vol import LocalVolSurface
from options_lib.numerics.local_vol_simulator import LocalVolSimulator


@dataclass
class BarrierPricingResult:
    """
    Full barrier option pricing result with model comparison.

    Attributes
    ----------
    lv_price     : float   Price under local vol MC
    lv_std_error : float   Monte Carlo standard error
    bs_price     : float   Price under flat BS vol MC
    bs_std_error : float   Standard error for BS MC
    model_risk   : float   Absolute price difference: |LV - BS|
    model_risk_pct : float Relative price difference: |LV - BS| / BS
    n_paths      : int
    n_steps      : int
    """
    lv_price       : float
    lv_std_error   : float
    bs_price       : float
    bs_std_error   : float
    n_paths        : int
    n_steps        : int

    @property
    def model_risk(self) -> float:
        return abs(self.lv_price - self.bs_price)

    @property
    def model_risk_pct(self) -> float:
        if abs(self.bs_price) < 1e-8:
            return 0.0
        return self.model_risk / abs(self.bs_price)

    def summary(self) -> str:
        lines = [
            f"{'='*50}",
            f"Barrier Option Pricing — Model Comparison",
            f"{'='*50}",
            f"Local vol price:  {self.lv_price:.4f} ± {1.96*self.lv_std_error:.4f}",
            f"Flat BS vol price:{self.bs_price:.4f} ± {1.96*self.bs_std_error:.4f}",
            f"{'─'*50}",
            f"Model risk:       {self.model_risk:.4f} ({self.model_risk_pct:.1%})",
            f"Direction:        LV {'cheaper' if self.lv_price < self.bs_price else 'more expensive'} than BS",
            f"N paths:          {self.n_paths:,}",
            f"N steps:          {self.n_steps}",
            f"{'='*50}",
        ]
        return '\n'.join(lines)


@dataclass
class LocalVolBarrierPricer:
    """
    Barrier option pricer under Dupire local volatility.

    Uses Monte Carlo simulation with local vol paths, and
    simultaneously prices under flat BS vol for model risk comparison.

    Parameters
    ----------
    lv_surface : LocalVolSurface
        Calibrated Dupire local vol surface.
    bs_sigma : float
        ATM implied vol for flat BS pricing (for comparison).
        Typically the ATM IV from the same vol surface.
    n_paths : int
        Number of MC paths. 100,000 recommended.
    n_steps : int
        Time steps per path. 252 (daily) for barrier options.
    antithetic : bool
        Antithetic variates variance reduction.
    continuity_correction : bool
        Apply BGK (1997) correction for discrete monitoring bias.
    seed : int, optional

    Notes
    -----
    Running both local vol and BS simultaneously uses the SAME random
    numbers (common random numbers technique). This means the model risk
    estimate (LV price - BS price) has much lower variance than if we
    ran the two simulations independently. The variance of the DIFFERENCE
    is what matters for model risk estimation.
    """

    lv_surface             : LocalVolSurface
    bs_sigma               : float
    n_paths                : int   = 100_000
    n_steps                : int   = 252
    antithetic             : bool  = True
    continuity_correction  : bool  = True
    seed                   : Optional[int] = None

    def price(
        self,
        instrument : BarrierOption,
        market     : MarketData,
    ) -> BarrierPricingResult:
        """
        Price a barrier option under local vol and flat BS vol.

        Uses common random numbers: the SAME simulated paths are used
        for both LV and BS pricing, so the model risk estimate is precise.

        Parameters
        ----------
        instrument : BarrierOption
        market : MarketData

        Returns
        -------
        BarrierPricingResult
        """
        S0 = market.spot
        r  = market.rate
        q  = market.div_yield
        T  = instrument.expiry
        K  = instrument.strike
        H  = instrument.barrier
        dt = T / self.n_steps

        # ------------------------------------------------------------------
        # Simulate local vol paths
        # ------------------------------------------------------------------
        lv_sim = LocalVolSimulator(
            lv_surface = self.lv_surface,
            n_paths    = self.n_paths,
            n_steps    = self.n_steps,
            antithetic = self.antithetic,
            seed       = self.seed,
        )
        lv_paths = lv_sim.simulate(S0, T, r, q)

        # ------------------------------------------------------------------
        # Simulate flat BS paths using SAME random numbers
        # Re-seed to get identical random draws
        # ------------------------------------------------------------------
        if self.seed is not None:
            np.random.seed(self.seed)

        bs_paths = self._simulate_bs_paths(S0, T, r, q, dt)

        # ------------------------------------------------------------------
        # Apply barrier logic and compute payoffs for both path sets
        # ------------------------------------------------------------------
        lv_payoffs = self._compute_barrier_payoffs(
            lv_paths, instrument, r, T, dt
        )
        bs_payoffs = self._compute_barrier_payoffs(
            bs_paths, instrument, r, T, dt
        )

        # Discount
        disc_lv = np.exp(-r * T) * lv_payoffs
        disc_bs = np.exp(-r * T) * bs_payoffs

        lv_price = float(np.mean(disc_lv))
        bs_price = float(np.mean(disc_bs))
        lv_se    = float(np.std(disc_lv) / np.sqrt(self.n_paths))
        bs_se    = float(np.std(disc_bs) / np.sqrt(self.n_paths))

        return BarrierPricingResult(
            lv_price     = max(lv_price, 0.0),
            lv_std_error = lv_se,
            bs_price     = max(bs_price, 0.0),
            bs_std_error = bs_se,
            n_paths      = self.n_paths,
            n_steps      = self.n_steps,
        )

    def _simulate_bs_paths(
        self,
        S0 : float,
        T  : float,
        r  : float,
        q  : float,
        dt : float,
    ) -> np.ndarray:
        """
        Simulate flat GBM paths. Uses the current np.random state
        so it draws the same normals as the LV simulator when seeded.
        """
        n_paths = self.n_paths
        sigma   = self.bs_sigma

        if self.antithetic:
            half   = n_paths // 2
            Z_half = np.random.standard_normal((half, self.n_steps))
            Z      = np.concatenate([Z_half, -Z_half], axis=0)
        else:
            Z = np.random.standard_normal((n_paths, self.n_steps))

        drift       = (r - q - 0.5 * sigma**2) * dt
        diffusion   = sigma * np.sqrt(dt) * Z
        log_returns = drift + diffusion
        log_S       = np.log(S0) + np.cumsum(log_returns, axis=1)
        log_S0      = np.full((n_paths, 1), np.log(S0))

        return np.exp(np.concatenate([log_S0, log_S], axis=1))

    def _compute_barrier_payoffs(
        self,
        paths      : np.ndarray,
        instrument : BarrierOption,
        r          : float,
        T          : float,
        dt         : float,
    ) -> np.ndarray:
        """
        Compute payoff for each path, applying barrier logic.

        For each path:
          - Check if barrier was breached at any time step
          - Apply payoff or rebate accordingly
          - Apply continuity correction if enabled

        Parameters
        ----------
        paths : np.ndarray, shape (n_paths, n_steps + 1)

        Returns
        -------
        np.ndarray, shape (n_paths,)   Undiscounted payoffs.
        """
        H    = instrument.barrier
        K    = instrument.strike
        btype = instrument.barrier_type
        otype = instrument.option_type

        # Continuity correction (BGK 1997)
        # Adjusts the barrier to account for discrete monitoring bias
        if self.continuity_correction:
            avg_sigma = self.bs_sigma  # use ATM vol as approximation
            correction = 0.5826 * avg_sigma * np.sqrt(dt)
            if btype in (BarrierType.DOWN_AND_OUT, BarrierType.DOWN_AND_IN):
                H_eff = H * np.exp(-correction)  # shift down barrier down
            else:
                H_eff = H * np.exp(+correction)  # shift up barrier up
        else:
            H_eff = H

        # Check barrier breach for each path
        # paths[:, 1:] excludes the initial spot (barriers checked from t > 0)
        S_T  = paths[:, -1]          # terminal spot
        path_min = np.min(paths[:, 1:], axis=1)   # minimum spot along path
        path_max = np.max(paths[:, 1:], axis=1)   # maximum spot along path

        if btype in (BarrierType.DOWN_AND_OUT, BarrierType.DOWN_AND_IN):
            barrier_hit = path_min <= H_eff
        else:
            barrier_hit = path_max >= H_eff

        # Terminal vanilla payoff
        if otype == OptionType.CALL:
            vanilla = np.maximum(S_T - K, 0.0)
        else:
            vanilla = np.maximum(K - S_T, 0.0)

        # Apply barrier logic
        if btype in (BarrierType.DOWN_AND_OUT, BarrierType.UP_AND_OUT):
            # Knock-out: barrier hit → rebate, else vanilla
            payoffs = np.where(barrier_hit, instrument.rebate, vanilla)
        else:
            # Knock-in: barrier hit → vanilla, else rebate
            payoffs = np.where(barrier_hit, vanilla, instrument.rebate)

        return payoffs

    def model_risk_surface(
        self,
        barriers  : np.ndarray,
        market    : MarketData,
        instrument_template : BarrierOption,
    ) -> dict:
        """
        Compute local vol vs flat BS price difference across a range of barriers.

        This produces the "model risk curve" — showing how much the two models
        disagree as the barrier moves closer to or further from the spot.

        Parameters
        ----------
        barriers : np.ndarray
            Array of barrier levels to test.
        market : MarketData
        instrument_template : BarrierOption
            Template instrument — barrier will be varied.

        Returns
        -------
        dict with keys: barriers, lv_prices, bs_prices, model_risk, model_risk_pct
        """
        lv_prices      = []
        bs_prices      = []
        model_risks    = []
        model_risk_pcts = []

        for H in barriers:
            inst = BarrierOption(
                strike       = instrument_template.strike,
                expiry       = instrument_template.expiry,
                option_type  = instrument_template.option_type,
                barrier      = float(H),
                barrier_type = instrument_template.barrier_type,
                rebate       = instrument_template.rebate,
            )
            result = self.price(inst, market)
            lv_prices.append(result.lv_price)
            bs_prices.append(result.bs_price)
            model_risks.append(result.model_risk)
            model_risk_pcts.append(result.model_risk_pct)

        return {
            'barriers'        : barriers,
            'lv_prices'       : np.array(lv_prices),
            'bs_prices'       : np.array(bs_prices),
            'model_risk'      : np.array(model_risks),
            'model_risk_pct'  : np.array(model_risk_pcts),
        }

    def __repr__(self) -> str:
        return (
            f"LocalVolBarrierPricer("
            f"bs_sigma={self.bs_sigma:.2f}, "
            f"n_paths={self.n_paths:,}, "
            f"n_steps={self.n_steps})"
        )