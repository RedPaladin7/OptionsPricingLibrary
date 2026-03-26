"""
file: models/base.py
---
Model defines how to evolve the underlying for computing expectation. 
Knows nothing about the specific payoff structure of the instrument being priced.
"""

from abc import ABC, abstractmethod
from options_lib.instruments.base import Instrument, MarketData 

class Model(ABC):
    """
    Abstract base class for all pricing models.
    
    All models must implement: 
    price(instrument, market) -> float

    Optionally override: 
    delta, gamma, vega, theta for analytical greeks
    """
    @abstractmethod
    def price(self, instrument: Instrument, market: MarketData) -> float:
        """
        Compute fair value of a given instrument given market data.
        ---
        Input: instrument being priced (payoff structure and expiry)
        Output: No arbitrage price of the instrument.
        """
        ... 

    def delta(self, instrument: Instrument, market: MarketData) -> float:
        """
        dV/dS - sensitivity of option price to spot price
        """
        h = market.spot * 0.01 
        up = self.price(instrument, MarketData(market.spot + h, market.rate, market.div_yield))
        down = self.price(instrument, MarketData(market.spot - h, market.rate, market.div_yield))
        return (up - down) / (2 * h)
    
    def gamma(self, instrument: Instrument, market: MarketData) -> float:
        """
        d^2V/dS^2 - sensitivity of delta to spot price 
        Default: bump-and-reprice (central difference)
        """
        h = market.spot * 0.01 
        mid = self.price(instrument, market)
        up = self.price(instrument, MarketData(market.spot + h, market.rate, market.div_yield))
        down = self.price(instrument, MarketData(market.spot - h, market.rate, market.div_yield))

        return (up - 2 * mid + down) / (h ** 2)
    
    def vega(self, instrument: Instrument, market: MarketData) -> float:
        """
        dV/d(sigma) — sensitivity to volatility.
        Base class cannot implement this without knowing the value for sigma.
        Subclasses with sigma parameter must override this. 
        """
        raise NotImplementedError(
            f'{type(self.__name__)} does not expose a sigma parameter. Override vega() in the subclass'
        )
    
    def theta(self, instrument: Instrument, market: MarketData) -> float:
        """
        dV/dt - time decay
        Default: reprice with expiry reduced by 1/365
        """
        from dataclasses import replace 
        import copy 

        dt = 1 / 365 
        if instrument.expiry <= dt:
            return 0.0 
        raise NotImplementedError(
            'theta() requires instrument to be repriced at T - dt. Implement in each model-instrument pair or use the risk module.'
        )
    
    def __repr__(self) -> str:
        return f'{type(self).__name__}()'