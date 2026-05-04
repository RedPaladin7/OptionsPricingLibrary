"""
models/base.py
--------------
Abstract base class for all pricing models.

A model defines HOW to evolve the underlying — its dynamics, parameters,
and numerical method for computing expectations. It knows nothing about
the specific payoff structure of the instrument being priced.

The pricing interface is always:
    model.price(instrument, market_data) -> float
"""

from abc import ABC, abstractmethod
from options_lib.instruments.base import Instrument, MarketData


class Model(ABC):
    """
    Abstract base class for all pricing models.

    Every concrete model must implement:
      - price(instrument, market) -> float

    Optionally override:
      - delta, gamma, vega, theta for analytical Greeks.
        If not overridden, the risk module will use bump-and-reprice.
    """

    @abstractmethod
    def price(self, instrument: Instrument, market: MarketData) -> float:
        """
        Compute the fair value of an instrument given market data.

        Parameters
        ----------
        instrument : Instrument
            The contract being priced (defines payoff and expiry).
        market : MarketData
            Current market inputs (spot, rate, div yield).

        Returns
        -------
        float
            The no-arbitrage price of the instrument.
        """
        ...

    def delta(self, instrument: Instrument, market: MarketData) -> float:
        """
        dV/dS — sensitivity to spot price.
        Default: bump-and-reprice with 1% of spot bump.
        Override in subclasses for analytical formulae.
        """
        h = market.spot * 0.01
        up   = self.price(instrument, MarketData(market.spot + h, market.rate, market.div_yield))
        down = self.price(instrument, MarketData(market.spot - h, market.rate, market.div_yield))
        return (up - down) / (2 * h)

    def gamma(self, instrument: Instrument, market: MarketData) -> float:
        """
        d²V/dS² — second-order sensitivity to spot.
        Default: bump-and-reprice (central difference).
        """
        h = market.spot * 0.01
        mid  = self.price(instrument, market)
        up   = self.price(instrument, MarketData(market.spot + h, market.rate, market.div_yield))
        down = self.price(instrument, MarketData(market.spot - h, market.rate, market.div_yield))
        return (up - 2 * mid + down) / (h ** 2)

    def vega(self, instrument: Instrument, market: MarketData) -> float:
        """
        dV/dσ — sensitivity to volatility.
        Base class cannot implement this without knowing sigma.
        Subclasses with a sigma parameter must override this.
        Raises NotImplementedError by default.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not expose a sigma parameter. "
            "Override vega() in the subclass."
        )

    def theta(self, instrument: Instrument, market: MarketData) -> float:
        """
        dV/dt — time decay (per calendar day).
        Default: reprice with expiry reduced by 1/365.
        """
        from dataclasses import replace
        import copy

        dt = 1 / 365
        if instrument.expiry <= dt:
            return 0.0

        # Create a copy of the instrument with reduced expiry
        # Concrete instruments must support this via their constructor
        raise NotImplementedError(
            "theta() requires instrument to be repriced at T - dt. "
            "Implement in each model-instrument pair or use the risk module."
        )

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"