import numpy as np 
from dataclasses import dataclass
from enum import Enum 
from options_lib.instruments.base import Instrument, OptionType, ExerciseStyle

class AverageType(Enum):
    ARITHMETIC = 'arithmetic'
    GEOMETRIC = 'geometric'

@dataclass
class AsianOption(Instrument):
    strike: float 
    _expiry: float 
    option_type: OptionType
    average_type: AverageType
    n_observations: int 

    def __init__(
        self, 
        strike: float,
        expiry: float,
        option_type: OptionType,
        average_type: AverageType = AverageType.ARITHMETIC,
        n_observations: int = 252
    ):
        if strike <= 0:
            raise ValueError(f'Strike must be positive, got {strike}')
        if expiry <= 0:
            raise ValueError(f'Expiry must be positive, got {expiry}')
        if n_observations < 2:
            raise ValueError(f'Need at least 2 observations')
        self.strike = strike
        self._expiry = expiry 
        self.option_type = option_type
        self.average_type = average_type
        self.n_observations = n_observations

    def compute_average(self, path: np.ndarray) -> float:
        if self.average_type == AverageType.ARITHMETIC:
            return float(np.mean(path))
        else:
            return float(np.exp(np.mean(np.log(path))))
        
    def path_payoff(self, path: np.ndarray) -> float:
        A = self.compute_average(path)
        if self.option_type == OptionType.CALL:
            return max(A - self.strike, 0.0)
        else:
            return max(self.strike - A, 0.0)
        
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
    def exercise_style(self) -> ExerciseStyle:
        return ExerciseStyle.EUROPEAN
    
    def __repr__(self) -> str:
        return f'AsianOption(strike={self.strike}, expiry={self._expiry}, type={self.option_type.value})'