"""
instruments/base.py
-------------------
Abstract base class for all financial instruments.

An instrument defines WHAT the contract is — its payoff structure,
exercise rights, and expiry. It knows nothing about how to price itself.
That is the model's job.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
import numpy as np


class OptionType(Enum):
    CALL = "call"
    PUT  = "put"


class ExerciseStyle(Enum):
    EUROPEAN = "european"   # exercise only at expiry
    AMERICAN = "american"   # exercise any time up to expiry


@dataclass
class MarketData:
    """
    All market inputs needed to price an option.

    Attributes
    ----------
    spot : float
        Current underlying price S_0.
    rate : float
        Continuously compounded risk-free rate r.
    div_yield : float
        Continuous dividend yield q. Modifies drift to (r - q).
        For non-dividend paying stocks, set to 0.
    """
    spot      : float
    rate      : float
    div_yield : float = 0.0


class Instrument(ABC):
    """
    Abstract base class for all instruments.

    Every concrete instrument must implement:
      - payoff(spots)  : vectorised payoff at expiry given an array of spot prices
      - expiry         : time to expiry in years (property)
      - exercise_style : European or American (property)
    """

    @abstractmethod
    def payoff(self, spots: np.ndarray) -> np.ndarray:
        """
        Compute the terminal payoff for an array of spot prices.

        Parameters
        ----------
        spots : np.ndarray
            Array of underlying prices at expiry.

        Returns
        -------
        np.ndarray
            Payoff at each spot price. Must be non-negative.
        """
        ...

    @property
    @abstractmethod
    def expiry(self) -> float:
        """Time to expiry T in years."""
        ...

    @property
    @abstractmethod
    def exercise_style(self) -> ExerciseStyle:
        """European or American exercise."""
        ...

    def intrinsic_value(self, spot: float) -> float:
        """
        Intrinsic value = immediate exercise value.
        Used by American option solvers to enforce early exercise constraint.
        """
        return float(self.payoff(np.array([spot]))[0])