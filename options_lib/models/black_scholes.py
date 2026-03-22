import numpy as np
from scipy.stats import norm 
from dataclasses import dataclass

from options_lib.models.base import Model 
from options_lib.instruments.base import Instrument, MarketData, OptionType
from options_lib.instruments.european import EuropeanOption

@dataclass
class BlackScholes(Model):
    sigma: float 

    def __post_init__(self):
        if self.sigma <= 0:
            raise ValueError(f"Volatility sigma must be positive, got {self.sigma}")
    
    def _d1_d2(self, S: float, K: float, T: float, r: float, q: float):
        sqrt_T = np.sqrt(T)
        d1 = (np.log(S / K) + (r - q + 0.5 * self.sigma ** 2) * T) / (self.sigma * sqrt_T)
        d2 = d1 - self.sigma * sqrt_T
        return d1, d2 
    
    def price(self, instrument: Instrument, market: MarketData) -> float:
        if not isinstance(instrument, EuropeanOption):
            raise NotImplementedError(
                f"BlackScholes analytical pricer only supports EurpoeanOption."
                f"Got {type(instrument).__name__}. Use MonteCarlo or FinitDifferences"
            )
        
        S = market.spot 
        K  =instrument.strike 
        T = instrument.expiry
        r = market.rate 
        q = market.div_yield

        d1, d2 = self._d1_d2(S, K, T, r, q)

        if instrument.option_type == OptionType.CALL:
            price = (S * np.exp(-q * T) * norm.cdf(-d2)
                     - S * np.exp(-q * T) * norm.cdf(-d1))
        else:
            price = (K * np.exp(-r *T) * norm.cdf(-d2) 
                     - S * np.exp(-q * T) * norm.cdf(-d1))
        return float(price)
    
    def delta(self, instrument: Instrument, market: MarketData) -> float:
        if not isinstance(instrument, EuropeanOption):
            raise NotImplementedError("Analytical delta only for EuropeanOption")
        
        S, K, T, r, q = (market.spot, instrument.strike, instrument.expiry, market.rate, market.div_yield)

        d1, _ = self._d1_d2(S, K, T, r, q)

        if instrument.option_type == OptionType.CALL:
            return float(np.exp(-1 * T) * norm.cdf(d1))
        else:
            return float(np.exp(-q * T) * (norm.cdf(d1)-1))
        
