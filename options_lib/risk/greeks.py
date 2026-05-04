"""
risk/greeks.py
--------------
Unified Greek computation across all models and instruments.

Two modes:
  1. Analytical  — calls the model's own analytical formula (BS only)
  2. Bump-and-reprice — perturbs market inputs and finite-differences the price
                        Works for ANY model, including Heston and MC.

Greek Surface
-------------
Rather than computing one Greek at one point, we often want the full
surface: e.g. Delta(K, T) for all strikes and expiries in a portfolio.
This is what traders look at on their risk screens.

The GreekSurface class computes and stores:
    Delta, Gamma, Vega, Theta, Vanna, Volga
across a grid of (strike, expiry) pairs for a given model and market.

Usage
-----
>>> from options_lib.risk.greeks import GreekEngine, GreekSurface
>>> model  = BlackScholes(sigma=0.20)
>>> engine = GreekEngine(model)
>>> call   = EuropeanOption(strike=100, expiry=1.0, option_type=OptionType.CALL)
>>> mkt    = MarketData(spot=100, rate=0.05)
>>> engine.all_greeks(call, mkt)
{'price': 10.45, 'delta': 0.636, 'gamma': 0.019, 'vega': 37.5, ...}
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from options_lib.models.base import Model
from options_lib.models.black_scholes import BlackScholes
from options_lib.instruments.base import Instrument, MarketData, OptionType
from options_lib.instruments.european import EuropeanOption


@dataclass
class Greeks:
    """
    Container for all Greeks at a single (instrument, market) point.

    All values are in dollar terms (not percentage) unless noted.

    Attributes
    ----------
    price  : float   Option price V
    delta  : float   dV/dS                  ($ per $ spot move)
    gamma  : float   d²V/dS²                ($ per $² spot move)
    vega   : float   dV/dσ                  ($ per 1.0 vol move, i.e. 100%)
    theta  : float   dV/dt                  ($ per calendar day)
    rho    : float   dV/dr                  ($ per 1% rate move)
    vanna  : float   d²V/(dS dσ)            ($ per $ spot per 1.0 vol)
    volga  : float   d²V/dσ²               ($ per 1.0² vol²)
    charm  : float   d²V/(dS dt)            ($ per $ spot per day)
    """
    price  : float
    delta  : float
    gamma  : float
    vega   : float
    theta  : float
    rho    : float   = 0.0
    vanna  : float   = 0.0
    volga  : float   = 0.0
    charm  : float   = 0.0

    def to_dict(self) -> dict:
        return {
            'price' : self.price,
            'delta' : self.delta,
            'gamma' : self.gamma,
            'vega'  : self.vega,
            'theta' : self.theta,
            'rho'   : self.rho,
            'vanna' : self.vanna,
            'volga' : self.volga,
            'charm' : self.charm,
        }

    def __repr__(self) -> str:
        return (
            f"Greeks(\n"
            f"  price={self.price:.4f}\n"
            f"  delta={self.delta:.4f}  gamma={self.gamma:.4f}\n"
            f"  vega={self.vega:.4f}   theta={self.theta:.4f}/day\n"
            f"  vanna={self.vanna:.4f} volga={self.volga:.4f}\n"
            f"  charm={self.charm:.6f}/day\n"
            f")"
        )


class GreekEngine:
    """
    Computes Greeks for any model via analytical formulas or bump-and-reprice.

    Strategy:
      - If model is BlackScholes: use analytical formulas (exact, fast)
      - Otherwise: bump-and-reprice numerically (works for Heston, MC, etc.)

    Bump sizes are chosen to balance truncation error (too large)
    and cancellation error (too small). Standard choices:
      dS    = 0.01 * S     (1% spot bump)
      dsigma = 0.001       (0.1% vol bump — only for BS)
      dt    = 1/365        (1 day time bump)
      dr    = 0.0001       (1 basis point rate bump)

    Parameters
    ----------
    model : Model
        Any pricing model (BlackScholes, Heston, MonteCarlo, etc.)
    """

    def __init__(self, model: Model):
        self.model = model

    def price(self, instrument: Instrument, market: MarketData) -> float:
        return self.model.price(instrument, market)

    def delta(self, instrument: Instrument, market: MarketData) -> float:
        """dV/dS via central difference on spot."""
        if isinstance(self.model, BlackScholes):
            return self.model.delta(instrument, market)

        h = market.spot * 0.01
        up   = self.model.price(instrument, MarketData(market.spot + h, market.rate, market.div_yield))
        down = self.model.price(instrument, MarketData(market.spot - h, market.rate, market.div_yield))
        return (up - down) / (2 * h)

    def gamma(self, instrument: Instrument, market: MarketData) -> float:
        """d²V/dS² via central second difference on spot."""
        if isinstance(self.model, BlackScholes):
            return self.model.gamma(instrument, market)

        h   = market.spot * 0.01
        mid  = self.model.price(instrument, market)
        up   = self.model.price(instrument, MarketData(market.spot + h, market.rate, market.div_yield))
        down = self.model.price(instrument, MarketData(market.spot - h, market.rate, market.div_yield))
        return (up - 2 * mid + down) / h**2

    def vega(self, instrument: Instrument, market: MarketData) -> float:
        """
        dV/dσ via bump-and-reprice on sigma.

        Only meaningful for models with an explicit sigma parameter.
        For Heston, 'vega' w.r.t. the initial vol sqrt(v0) is computed.
        """
        if isinstance(self.model, BlackScholes):
            return self.model.vega(instrument, market)

        # For Heston or other stochastic vol models:
        # bump v0 by a small amount and finite-difference
        from options_lib.models.heston import Heston
        if isinstance(self.model, Heston):
            h = 0.001   # bump in vol space
            params = self.model.params
            import dataclasses
            from options_lib.models.heston import HestonParams

            v0_up   = max((np.sqrt(params.v0) + h)**2, 1e-6)
            v0_down = max((np.sqrt(params.v0) - h)**2, 1e-6)

            params_up   = HestonParams(v0=v0_up,   kappa=params.kappa,
                                        v_bar=params.v_bar, xi=params.xi, rho=params.rho)
            params_down = HestonParams(v0=v0_down, kappa=params.kappa,
                                        v_bar=params.v_bar, xi=params.xi, rho=params.rho)

            m_up   = Heston(params_up,   alpha=self.model.alpha, N=self.model.N, eta=self.model.eta)
            m_down = Heston(params_down, alpha=self.model.alpha, N=self.model.N, eta=self.model.eta)

            p_up   = m_up.price(instrument, market)
            p_down = m_down.price(instrument, market)
            return (p_up - p_down) / (2 * h)

        raise NotImplementedError(f"Vega not implemented for {type(self.model).__name__}")

    def theta(self, instrument: Instrument, market: MarketData) -> float:
        """
        dV/dt per calendar day via reprice at T - 1/365.

        Note: we reduce expiry by 1 day and compare prices.
        This is the theta decay the holder experiences overnight.
        """
        if isinstance(self.model, BlackScholes):
            return self.model.theta(instrument, market)

        dt = 1 / 365
        if instrument.expiry <= dt:
            return 0.0

        inst_tomorrow = instrument.with_expiry(instrument.expiry - dt)
        p_today    = self.model.price(instrument, market)
        p_tomorrow = self.model.price(inst_tomorrow, market)
        return p_tomorrow - p_today     # negative for long options

    def rho(self, instrument: Instrument, market: MarketData) -> float:
        """dV/dr per 1% rate move."""
        if isinstance(self.model, BlackScholes):
            return self.model.rho(instrument, market)

        h = 0.0001  # 1 basis point
        up   = self.model.price(instrument, MarketData(market.spot, market.rate + h, market.div_yield))
        down = self.model.price(instrument, MarketData(market.spot, market.rate - h, market.div_yield))
        return (up - down) / (2 * h) / 100   # per 1%

    def vanna(self, instrument: Instrument, market: MarketData) -> float:
        """
        d²V/(dS dσ) — cross derivative of Delta w.r.t. vol.

        Computed as: [Delta(S, σ+h) - Delta(S, σ-h)] / (2h)
        i.e. how much Delta changes when vol moves by h.

        For BS: use analytical formula.
        For other models: double bump (spot then vol).
        """
        if isinstance(self.model, BlackScholes):
            return self.model.vanna(instrument, market)

        # Bump vol (via sigma for BS, or v0 for Heston) and re-compute delta
        from options_lib.models.heston import Heston, HestonParams
        if isinstance(self.model, Heston):
            h = 0.01
            params = self.model.params
            v0_up   = max((np.sqrt(params.v0) + h)**2, 1e-6)
            v0_down = max((np.sqrt(params.v0) - h)**2, 1e-6)

            m_up   = Heston(HestonParams(v0=v0_up,   kappa=params.kappa,
                                          v_bar=params.v_bar, xi=params.xi, rho=params.rho))
            m_down = Heston(HestonParams(v0=v0_down, kappa=params.kappa,
                                          v_bar=params.v_bar, xi=params.xi, rho=params.rho))

            eng_up   = GreekEngine(m_up)
            eng_down = GreekEngine(m_down)
            delta_up   = eng_up.delta(instrument, market)
            delta_down = eng_down.delta(instrument, market)
            return (delta_up - delta_down) / (2 * h)

        raise NotImplementedError(f"Vanna not implemented for {type(self.model).__name__}")

    def volga(self, instrument: Instrument, market: MarketData) -> float:
        """
        d²V/dσ² — second derivative of price w.r.t. vol.
        Computed as: [Vega(σ+h) - 2*Vega(σ) + Vega(σ-h)] / h²
        """
        if isinstance(self.model, BlackScholes):
            return self.model.volga(instrument, market)

        from options_lib.models.heston import Heston, HestonParams
        if isinstance(self.model, Heston):
            h = 0.01
            params = self.model.params

            def make_model(dv):
                v0 = max((np.sqrt(params.v0) + dv)**2, 1e-6)
                return Heston(HestonParams(v0=v0, kappa=params.kappa,
                                           v_bar=params.v_bar, xi=params.xi, rho=params.rho))

            p_up   = make_model(+h).price(instrument, market)
            p_mid  = self.model.price(instrument, market)
            p_down = make_model(-h).price(instrument, market)
            return (p_up - 2*p_mid + p_down) / h**2

        raise NotImplementedError(f"Volga not implemented for {type(self.model).__name__}")

    def charm(self, instrument: Instrument, market: MarketData) -> float:
        """
        d²V/(dS dt) — rate of change of Delta with respect to time.
        Computed as: [Delta(T - dt) - Delta(T)] / dt
        """
        if isinstance(self.model, BlackScholes):
            return self.model.charm(instrument, market)

        dt = 1 / 365
        if instrument.expiry <= dt:
            return 0.0

        inst_tomorrow = instrument.with_expiry(instrument.expiry - dt)
        d_today    = self.delta(instrument, market)
        d_tomorrow = self.delta(inst_tomorrow, market)
        return (d_tomorrow - d_today)   # per calendar day

    def all_greeks(self, instrument: Instrument, market: MarketData) -> Greeks:
        """
        Compute all Greeks in one call.

        For BS: fully analytical, essentially instant.
        For other models: multiple bump-and-reprice evaluations.
        """
        return Greeks(
            price = self.price(instrument, market),
            delta = self.delta(instrument, market),
            gamma = self.gamma(instrument, market),
            vega  = self.vega(instrument, market),
            theta = self.theta(instrument, market),
            rho   = self.rho(instrument, market),
            vanna = self.vanna(instrument, market),
            volga = self.volga(instrument, market),
            charm = self.charm(instrument, market),
        )


@dataclass
class GreekSurface:
    """
    Greek surface across a grid of strikes and expiries.

    This is what a trader's risk screen shows: for every option in
    the book, what are the Greeks? Aggregated, this gives portfolio-level
    risk exposures.

    Parameters
    ----------
    model  : Model
        Pricing model (typically BlackScholes for speed).
    market : MarketData
        Current market data.
    strikes  : np.ndarray   Grid of strikes.
    expiries : np.ndarray   Grid of expiries (in years).
    option_type : OptionType

    Usage
    -----
    >>> surface = GreekSurface(model, market, strikes, expiries, OptionType.CALL)
    >>> surface.compute()
    >>> surface.delta_surface   # shape (len(expiries), len(strikes))
    """

    model       : Model
    market      : MarketData
    strikes     : np.ndarray
    expiries    : np.ndarray
    option_type : OptionType

    # Computed surfaces — filled by compute()
    price_surface : np.ndarray = field(init=False, repr=False)
    delta_surface : np.ndarray = field(init=False, repr=False)
    gamma_surface : np.ndarray = field(init=False, repr=False)
    vega_surface  : np.ndarray = field(init=False, repr=False)
    theta_surface : np.ndarray = field(init=False, repr=False)
    vanna_surface : np.ndarray = field(init=False, repr=False)
    volga_surface : np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        n_T = len(self.expiries)
        n_K = len(self.strikes)
        self.price_surface = np.zeros((n_T, n_K))
        self.delta_surface = np.zeros((n_T, n_K))
        self.gamma_surface = np.zeros((n_T, n_K))
        self.vega_surface  = np.zeros((n_T, n_K))
        self.theta_surface = np.zeros((n_T, n_K))
        self.vanna_surface = np.zeros((n_T, n_K))
        self.volga_surface = np.zeros((n_T, n_K))

    def compute(self) -> "GreekSurface":
        """
        Compute all Greek surfaces.

        For BS with analytical Greeks, this is fast (microseconds per point).
        For Heston, each point requires an FFT call — use a coarser grid.

        Returns self for method chaining.
        """
        engine = GreekEngine(self.model)

        for i, T in enumerate(self.expiries):
            for j, K in enumerate(self.strikes):
                try:
                    inst = EuropeanOption(strike=K, expiry=T, option_type=self.option_type)
                    g = engine.all_greeks(inst, self.market)
                    self.price_surface[i, j] = g.price
                    self.delta_surface[i, j] = g.delta
                    self.gamma_surface[i, j] = g.gamma
                    self.vega_surface[i, j]  = g.vega
                    self.theta_surface[i, j] = g.theta
                    self.vanna_surface[i, j] = g.vanna
                    self.volga_surface[i, j] = g.volga
                except Exception:
                    # Leave as NaN for problem spots
                    for arr in [self.price_surface, self.delta_surface,
                                self.gamma_surface, self.vega_surface,
                                self.theta_surface, self.vanna_surface,
                                self.volga_surface]:
                        arr[i, j] = np.nan

        return self

    def get_surface(self, greek: str) -> np.ndarray:
        """
        Retrieve a named Greek surface.

        Parameters
        ----------
        greek : str
            One of: 'price', 'delta', 'gamma', 'vega', 'theta', 'vanna', 'volga'

        Returns
        -------
        np.ndarray, shape (n_expiries, n_strikes)
        """
        surfaces = {
            'price' : self.price_surface,
            'delta' : self.delta_surface,
            'gamma' : self.gamma_surface,
            'vega'  : self.vega_surface,
            'theta' : self.theta_surface,
            'vanna' : self.vanna_surface,
            'volga' : self.volga_surface,
        }
        if greek not in surfaces:
            raise ValueError(f"Unknown Greek '{greek}'. Choose from {list(surfaces.keys())}")
        return surfaces[greek]