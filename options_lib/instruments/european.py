import numpy as np 
from dataclasses import dataclass 
from options_lib.instruments.base import Instrument, OptionType, ExerciseStyle

@dataclass
class EuropeanOption(Instrument):
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
    
    def with_expiry(self, new_expiry: float) -> 'EuropeanOption':
        return EuropeanOption(
            strike=self.strike,
            expiry=new_expiry,
            option_type=self.option_type
        )
    
    def __repr__(self) -> str:
        return (
            f'EuropeanOption(strike={self.strike}, expiry={self.expiry}, type={self.option_type.value})'
        )