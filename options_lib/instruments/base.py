"""
file: instruments/base.py
Abstract base class for all financial instruments.

Instrument defines what the contract is:
Payoff structure, exercise rights, expiry. 
Knows nothing about how to price itself.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum 
import numpy as np

# grouping related constants under same namespace
class OptionType(Enum):
    CALL = 'call'
    PUT = 'put'

class ExerciseStyle(Enum):
    EUROPEAN = 'european' # can be exercised only at expiry
    AMERICAN = 'american' # can be exercised at any time up to expiry

# dataclass is used because this is a class used primarily to store data rather than perform any compelx logic
# option holder does not receive the dividens (stock holder does) so it effectively reduces the drift
@dataclass
class MarketData:
    """
    Market inputs needed to price an option
    ---
    spot: float 
        Current price of the underlying asset S_0
    rate: float
        Risk free interest rate r 
    div_yield: float:
        Continuous dividend yield 
        Set to 0 for non dividend paying stocks
    """
    spot:       float 
    rate:       float 
    div_yield:  float = 0.0 

# property tags makes the method a getter (allows you to call the method as it were a simple attribute)
# any class that inherits from instruments must implement its own version of the abstract method
# This is a abstract base class, you cannot create an instance of this
class Instrument(ABC):
    @abstractmethod
    def payoff(self, sports:np.ndarray) -> np.ndarray:
        """
        Terminal payoff for an array of spot prices.
        ---
        spots: np.ndarray
            Array of spot prices.
        ---
        Returns 
        np.ndarray
            Payoff at each spot price.
        """
        ... 
    
    @property
    @abstractmethod
    def expiry(self) -> float:
        """
        Time to expiry (T) in years
        """
        ... 
    
    @property
    @abstractmethod
    def exercise_style(self) -> ExerciseStyle:
        """
        European or American Exercise
        """
        ... 
    
    def intrinsic_value(self, spot: float) -> float:
        """
        Immediate exercise value
        """
        return float(self.payoff(np.array([spot]))[0])
