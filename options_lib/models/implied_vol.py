"""
models/implied_vol.py
---------------------
Implied volatility computation using the Black-Scholes model.

Given a market-observed option price, find the volatility σ_imp
that makes the BS model price equal to the market price:

    BS(S, K, T, r, q, σ_imp) = C_market

This is a 1D root-finding problem (not analytically solvable in σ).

Usage
-----
>>> from options_lib.models.implied_vol import implied_vol_bs
>>> from options_lib.instruments.european import EuropeanOption
>>> from options_lib.instruments.base import MarketData, OptionType
>>> mkt  = MarketData(spot=100, rate=0.05)
>>> call = EuropeanOption(strike=100, expiry=1.0, option_type=OptionType.CALL)
>>> iv   = implied_vol_bs(market_price=10.45, instrument=call, market=mkt)
>>> round(iv, 4)
0.2
"""

import numpy as np
from options_lib.models.black_scholes import BlackScholes
from options_lib.instruments.european import EuropeanOption
from options_lib.instruments.base import MarketData
from options_lib.numerics.root_finding import implied_vol, ConvergenceError


def implied_vol_bs(
    market_price: float,
    instrument: EuropeanOption,
    market: MarketData,
    sigma_init: float = 0.20,
    tol: float = 1e-6
) -> float:
    """
    Compute Black-Scholes implied volatility from a market price.

    Parameters
    ----------
    market_price : float
        Observed market mid-price of the option.
    instrument : EuropeanOption
        The option contract.
    market : MarketData
        Current market data (spot, rate, div yield).
    sigma_init : float
        Initial vol guess for Newton-Raphson (default: 20%).
    tol : float
        Convergence tolerance.

    Returns
    -------
    float
        Implied volatility as a decimal (e.g., 0.20 = 20%).

    Notes
    -----
    The pricer and vega function are partially applied with all
    parameters fixed except sigma. This turns the multi-argument
    BS formula into a single-variable function ready for the
    root-finder.
    """

    def pricer(sigma: float) -> float:
        model = BlackScholes(sigma=sigma)
        return model.price(instrument, market)

    def vega_fn(sigma: float) -> float:
        model = BlackScholes(sigma=sigma)
        return model.vega(instrument, market)

    return implied_vol(
        market_price=market_price,
        pricer=pricer,
        vega_fn=vega_fn,
        sigma_init=sigma_init,
        tol=tol
    )