from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum 
import numpy as np

class OptionType(Enum):
    CALL = 'call'
    PUT = 'put'

class ExerciseStyle(Enum):
    EUROPEAN = 'european'
    AMERICAN = 'american'

@dataclass
class MarketData:
    spot:       float 
    rate:       float 
    div_yield:  float = 0.0 

class Instrument(ABC):
    @abstractmethod
    def payoff(self, sports:np.ndarray) -> np.ndarray:
        ... 
    
    @property
    @abstractmethod
    def expiry(self) -> float:
        ... 
    
    @property
    @abstractmethod
    def exercise_style(self) -> ExerciseStyle:
        ... 
    
    def intrinsic_value(self, spot: float) -> float:
        return float(self.payoff(np.array([spot]))[0])
