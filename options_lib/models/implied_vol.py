"""
file: models/implied_vol.py 
Calculates implied volatility using the black scholes model.
Finding the value of sigma which makes the BS option price 
equal to the market price.
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
    Compute black scholes implied volatility from market data.
    Pricer and vega fn are partially applied with all variables 
    except sigma. This converts the BS formula into a single variable function ready for the root finder.
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