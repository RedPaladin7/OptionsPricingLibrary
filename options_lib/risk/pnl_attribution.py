"""
risk/pnl_attribution.py
------------------------
Greek P&L attribution — decomposing option P&L into its Greek components.

Every trading desk does this every morning: take yesterday's P&L and
explain it using the Greeks. If the Greeks explain 99%+ of the P&L,
the model is working. If there's a large unexplained residual, something
is wrong — model error, data error, or a gap in the Greek coverage.

The Taylor Expansion
--------------------
Over a small time step dt with spot move dS and vol move dσ:

    dV ≈ Δ·dS + ½Γ·dS² + V·dσ + Θ·dt
         + Vanna·dS·dσ + ½·Volga·dσ²

Each term is a named P&L component:

    Delta P&L  = Δ · dS                (directional)
    Gamma P&L  = ½ · Γ · dS²           (convexity — large moves)
    Vega P&L   = V · dσ                (vol level change)
    Theta P&L  = Θ · dt                (time decay, always negative for long)
    Vanna P&L  = Vanna · dS · dσ       (spot × vol cross)
    Volga P&L  = ½ · Volga · dσ²      (vol convexity)

    Total explained = sum of above
    Unexplained     = actual dV − total explained  (model/higher-order error)

Why This Matters
----------------
1. Risk management: know exactly why you made or lost money
2. Hedging validation: if Gamma P&L is large but you thought you hedged gamma,
   something is wrong
3. Model validation: large unexplained P&L means the Taylor expansion breaks
   down (large moves, model misspecification)
4. Greeks quality: a good set of Greeks has small unexplained P&L

The Gamma-Theta Tradeoff
------------------------
The most important relationship: Gamma P&L and Theta P&L are always
in opposition. The BS PDE says:

    Θ + ½σ²S²Γ + (r-q)SΔ − rV = 0

Rearranging: Θ ≈ −½σ²S²Γ  (roughly, for small r)

So: positive Gamma → negative Theta (you pay time decay to own convexity)
    negative Gamma → positive Theta (you earn time decay but bleed on large moves)

This is the core trade-off of options market making.

Usage
-----
>>> from options_lib.risk.pnl_attribution import PnLAttributor
>>> attr = PnLAttributor(model, instrument, market_t0)
>>> result = attr.explain(market_t1)
>>> result.summary()
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional

from options_lib.models.base import Model
from options_lib.models.black_scholes import BlackScholes
from options_lib.instruments.base import Instrument, MarketData
from options_lib.instruments.european import EuropeanOption
from options_lib.risk.greeks import GreekEngine, Greeks


@dataclass
class PnLComponents:
    """
    Full P&L breakdown for one period.

    All values in dollar terms (same units as the option price).

    Attributes
    ----------
    actual_pnl    : float   True P&L = V(t1) - V(t0)
    delta_pnl     : float   Δ · dS
    gamma_pnl     : float   ½ · Γ · dS²
    vega_pnl      : float   Vega · dσ
    theta_pnl     : float   Θ · dt
    vanna_pnl     : float   Vanna · dS · dσ
    volga_pnl     : float   ½ · Volga · dσ²
    explained_pnl : float   Sum of all Greek components
    unexplained   : float   actual_pnl − explained_pnl
    """
    actual_pnl    : float
    delta_pnl     : float
    gamma_pnl     : float
    vega_pnl      : float
    theta_pnl     : float
    vanna_pnl     : float
    volga_pnl     : float

    dS   : float   # spot move
    dSigma : float # vol move
    dt   : float   # time elapsed

    @property
    def explained_pnl(self) -> float:
        return (self.delta_pnl + self.gamma_pnl + self.vega_pnl
                + self.theta_pnl + self.vanna_pnl + self.volga_pnl)

    @property
    def unexplained(self) -> float:
        return self.actual_pnl - self.explained_pnl

    @property
    def explanation_ratio(self) -> float:
        """
        Fraction of P&L explained by Greeks. Should be close to 1.0.
        Values far from 1.0 indicate model error or large moves
        where the Taylor expansion breaks down.
        """
        if abs(self.actual_pnl) < 1e-8:
            return 1.0
        return self.explained_pnl / self.actual_pnl

    def summary(self) -> str:
        """Pretty-print the P&L breakdown."""
        lines = [
            f"{'='*50}",
            f"P&L Attribution Summary",
            f"{'='*50}",
            f"Market moves:  dS={self.dS:+.4f}  dσ={self.dSigma:+.4f}  dt={self.dt:.4f}yr",
            f"{'─'*50}",
            f"Actual P&L:    {self.actual_pnl:+.4f}",
            f"{'─'*50}",
            f"Delta P&L:     {self.delta_pnl:+.4f}",
            f"Gamma P&L:     {self.gamma_pnl:+.4f}",
            f"Vega P&L:      {self.vega_pnl:+.4f}",
            f"Theta P&L:     {self.theta_pnl:+.4f}",
            f"Vanna P&L:     {self.vanna_pnl:+.4f}",
            f"Volga P&L:     {self.volga_pnl:+.4f}",
            f"{'─'*50}",
            f"Explained:     {self.explained_pnl:+.4f}",
            f"Unexplained:   {self.unexplained:+.4f}",
            f"Expl. ratio:   {self.explanation_ratio:.1%}",
            f"{'='*50}",
        ]
        return '\n'.join(lines)

    def to_dict(self) -> dict:
        return {
            'actual_pnl'    : self.actual_pnl,
            'delta_pnl'     : self.delta_pnl,
            'gamma_pnl'     : self.gamma_pnl,
            'vega_pnl'      : self.vega_pnl,
            'theta_pnl'     : self.theta_pnl,
            'vanna_pnl'     : self.vanna_pnl,
            'volga_pnl'     : self.volga_pnl,
            'explained_pnl' : self.explained_pnl,
            'unexplained'   : self.unexplained,
            'explanation_ratio': self.explanation_ratio,
        }


class PnLAttributor:
    """
    Compute and decompose P&L into Greek components for one instrument.

    Parameters
    ----------
    model      : Model
        Pricing model used at t=0 to compute Greeks.
    instrument : Instrument
        The option being attributed.
    market_t0  : MarketData
        Market data at the start of the period (spot, rate, div yield).
    sigma_t0   : float
        Implied vol at t=0 (used to compute Vega and vol-related Greeks).
        For BS model: same as model.sigma.
        For Heston: ATM implied vol from the model.
    """

    def __init__(
        self,
        model      : Model,
        instrument : Instrument,
        market_t0  : MarketData,
        sigma_t0   : Optional[float] = None,
    ):
        self.model      = model
        self.instrument = instrument
        self.market_t0  = market_t0

        # Extract sigma_t0 from model if not provided
        if sigma_t0 is None:
            if isinstance(model, BlackScholes):
                self.sigma_t0 = model.sigma
            else:
                raise ValueError(
                    "sigma_t0 must be provided for non-BS models. "
                    "Use the ATM implied vol from your vol surface."
                )
        else:
            self.sigma_t0 = sigma_t0

        # Compute Greeks at t=0 once — reused for all attributions
        self.engine   = GreekEngine(model)
        self.greeks_0 = self.engine.all_greeks(instrument, market_t0)
        self.price_0  = self.greeks_0.price

    def explain(
        self,
        market_t1  : MarketData,
        sigma_t1   : Optional[float] = None,
        dt         : float = 1/252,
    ) -> PnLComponents:
        """
        Attribute the P&L between t=0 and t=1 into Greek components.

        Parameters
        ----------
        market_t1 : MarketData
            Market data at the end of the period.
        sigma_t1 : float, optional
            Implied vol at t=1. If None, assumes no vol move (dσ = 0).
        dt : float
            Time elapsed in years. Default 1/252 (one trading day).

        Returns
        -------
        PnLComponents
            Full breakdown of P&L by Greek.
        """
        if sigma_t1 is None:
            sigma_t1 = self.sigma_t0

        # Market moves
        dS     = market_t1.spot - self.market_t0.spot
        dSigma = sigma_t1 - self.sigma_t0

        # Reprice at t=1 to get actual P&L
        # The instrument at t=1 has reduced expiry
        inst_t1    = self.instrument.with_expiry(max(self.instrument.expiry - dt, 1e-6))
        model_t1   = self._model_at_sigma(sigma_t1)
        price_t1   = model_t1.price(inst_t1, market_t1)
        actual_pnl = price_t1 - self.price_0

        # ------------------------------------------------------------------
        # Taylor expansion P&L components
        # ------------------------------------------------------------------

        # Delta P&L = Δ · dS
        delta_pnl = self.greeks_0.delta * dS

        # Gamma P&L = ½ · Γ · dS²
        gamma_pnl = 0.5 * self.greeks_0.gamma * dS**2

        # Vega P&L = Vega · dσ
        # Note: Vega from BS is per unit vol (e.g. per 100%).
        # dSigma is in the same units (decimal), so this is correct.
        vega_pnl = self.greeks_0.vega * dSigma

        # Theta P&L = Θ · dt (Θ is already per calendar day, so × 365*dt days)
        # theta is per calendar day, dt is in years
        theta_pnl = self.greeks_0.theta * dt * 365

        # Vanna P&L = Vanna · dS · dσ
        vanna_pnl = self.greeks_0.vanna * dS * dSigma

        # Volga P&L = ½ · Volga · dσ²
        volga_pnl = 0.5 * self.greeks_0.volga * dSigma**2

        return PnLComponents(
            actual_pnl = actual_pnl,
            delta_pnl  = delta_pnl,
            gamma_pnl  = gamma_pnl,
            vega_pnl   = vega_pnl,
            theta_pnl  = theta_pnl,
            vanna_pnl  = vanna_pnl,
            volga_pnl  = volga_pnl,
            dS         = dS,
            dSigma     = dSigma,
            dt         = dt,
        )

    def _model_at_sigma(self, sigma: float) -> Model:
        """Return a model instance with updated vol."""
        if isinstance(self.model, BlackScholes):
            return BlackScholes(sigma=sigma)
        return self.model   # for Heston, vol changes come via market_t1

    def backtest(
        self,
        spot_path  : np.ndarray,
        sigma_path : np.ndarray,
        dt         : float = 1/252,
    ) -> list:
        """
        Run P&L attribution over a historical spot and vol path.

        Parameters
        ----------
        spot_path  : np.ndarray, shape (N,)
            Spot prices at each time step.
        sigma_path : np.ndarray, shape (N,)
            Implied vols at each time step.
        dt : float
            Time step size in years.

        Returns
        -------
        list of PnLComponents
            One entry per time step.
        """
        results = []
        model   = self.model
        inst    = self.instrument

        for n in range(1, len(spot_path)):
            # Update instrument expiry for current time
            T_remaining = max(inst.expiry - n * dt, 1e-6)
            inst_n = inst.with_expiry(T_remaining + dt)   # yesterday's expiry

            mkt_yesterday = MarketData(
                spot=spot_path[n-1],
                rate=self.market_t0.rate,
                div_yield=self.market_t0.div_yield,
            )
            mkt_today = MarketData(
                spot=spot_path[n],
                rate=self.market_t0.rate,
                div_yield=self.market_t0.div_yield,
            )

            attributor = PnLAttributor(
                model=self._model_at_sigma(sigma_path[n-1]),
                instrument=inst_n,
                market_t0=mkt_yesterday,
                sigma_t0=sigma_path[n-1],
            )
            result = attributor.explain(mkt_today, sigma_t1=sigma_path[n], dt=dt)
            results.append(result)

        return results


def summarise_backtest(results: list) -> dict:
    """
    Aggregate a list of PnLComponents from a backtest.

    Returns
    -------
    dict
        Total and average P&L for each component.
        Includes the explanation ratio over the whole backtest.
    """
    keys = ['actual_pnl', 'delta_pnl', 'gamma_pnl', 'vega_pnl',
            'theta_pnl', 'vanna_pnl', 'volga_pnl', 'explained_pnl', 'unexplained']

    totals = {k: sum(getattr(r, k) for r in results) for k in keys}
    totals['explanation_ratio'] = (
        totals['explained_pnl'] / totals['actual_pnl']
        if abs(totals['actual_pnl']) > 1e-8 else 1.0
    )

    n = len(results)
    averages = {f"avg_{k}": totals[k] / n for k in keys}

    return {**totals, **averages, 'n_periods': n}