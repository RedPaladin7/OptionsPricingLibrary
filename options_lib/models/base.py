from abc import ABC, abstractmethod
from options_lib.instruments.base import Instrument, MarketData 

class Model(ABC):

    @abstractmethod
    def price(self, instrument: Instrument, market: MarketData) -> float:
        ... 

    def delta(self, instrument: Instrument, market: MarketData) -> float:
        h = market.spot * 0.01 
        up = self.price(instrument, MarketData(market.spot + h, market.rate, market.div_yield))
        down = self.price(instrument, MarketData(market.spot - h, market.rate, market.div_yield))
        return (up - down) / (2 * h)
    
    def evga(self, instrument: Instrument, market: MarketData) -> float:
        raise NotImplementedError(
            f'{type(self.__name__)} does not expose a sigma parameter. Override vega() in the subclass'
        )
    
    def theta(self, instrument: Instrument, market: MarketData) -> float:
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