"""
file: instruments/european.py
Vanilla European call and put option 

Can only exercised at expiry T.
Payoff:
    Call: max(S_T - K, 0)
    Put: max(K - S_T, 0)
"""

import numpy as np 
from dataclasses import dataclass 
from options_lib.instruments.base import Instrument, OptionType, ExerciseStyle

@dataclass
class EuropeanOption(Instrument):
    """
    Vanilla european option
    ---
    strike: float
    expiry: float => Time expiry in years
    option_type: OptionType => can be put or call
    """
    strike: float 
    _expiry: float 
    option_type: OptionType

    def __init__(self, strike: float, expiry: float, option_type: OptionType):
        if strike <= 0:
            raise ValueError(f'Strike must be positive, got {strike}')
        if expiry <= 0:
            raise ValueError(f'Expiry must be positive, got {expiry}')
        self.strike = strike
        self._expiry = expiry 
        self.option_type = option_type

    def payoff(self, spots: np.ndarray) -> np.ndarray:
        """
        Terminal payoff for an array of spot prices.
        """
        spots = np.asarray(spots, dtype=float)
        if self.option_type == OptionType.CALL:
            return np.maximum(spots - self.strike, 0.0)
        else:
            return np.maximum(self.strike - spots, 0.0)
        
    @property
    def expiry(self) -> float:
        return self._expiry
    
    @property
    def exercise_style(self):
        return ExerciseStyle.EUROPEAN
    
    def with_expiry(self, new_expiry: float) -> "EuropeanOption":
        """
        Returns a copy of this option with a different expiry. 
        """
        return EuropeanOption(
            strike=self.strike,
            expiry=new_expiry,
            option_type=self.option_type
        )
    
    def __repr__(self) -> str:
        return (
            f'EuropeanOption(strike={self.strike}, expiry={self.expiry}, type={self.option_type.value})'
        )