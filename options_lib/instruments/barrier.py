import numpy as np 
from dataclasses import dataclass
from enum import Enum 
from options_lib.instruments.base import Instrument, OptionType, ExerciseStyle

class BarrierType(Enum):
    DOWN_AND_OUT = 'down_and_out'
    DOWN_AND_IN = 'down_and_in'
    UP_AND_OUT = 'up_and_out'
    UP_AND_IN = 'up_and_in'


@dataclass
class BarrierOption(Instrument):
    strike: float 
    _expiry: float 
    option_type: OptionType
    barrier_type: BarrierType
    barrier: float
    rebate: float = 0.0 

    def __init__(
        self, 
        strike: float,
        expiry: float,
        option_type: OptionType,
        barrier_type: BarrierType,
        barrier: float,
        rebate: float = 0.0
    ):
        if strike <= 0:
            raise ValueError(f'Strike must be positive, got {strike}')
        if expiry <= 0:
            raise ValueError(f'Expiry must be positive, got {expiry}')
        if barrier <= 0:
            raise ValueError(f'Barrier must be positive, got {barrier}')
        self.strike = strike
        self._expiry = expiry 
        self.option_type = option_type
        self.barrier_type = barrier_type
        self.barrier = barrier
        self.rebate = rebate

    def is_knocked_out(self, path: np.ndarray) -> bool:
        if self.barrier_type in (BarrierType.DOWN_AND_OUT, BarrierType.DOWN_AND_IN):
            return bool(np.any(path <= self.barrier))
        else:
            return bool(np.any(path >= self.barrier))
        
    def is_knocked_in(self, path: np.ndarray) -> bool:
        return self.is_knocked_out(path)
    
    def path_payoff(self, path: np.ndarray) -> float:
        s_t = path[-1]
        barrier_hit = self.is_knocked_out(path)

        if self.option_type == OptionType.CALL:
            vanilla = max(s_t - self.strike, 0.0)
        else:
            vanilla = max(self.strike - s_t, 0.0)

        if self.barrier_type in (BarrierType.DOWN_AND_OUT, BarrierType.UP_AND_OUT):
            return self.rebate if barrier_hit else vanilla
        else:
            return vanilla if barrier_hit else self.rebate 
        
    def payoff(self, spots:np.ndarray) -> np.ndarray:
        spots = np.asarray(spots, dtype=float)
        if self.option_type == OptionType.CALL:
            vanilla = np.maximum(spots - self.strike, 0.0)
        else:
            vanilla = np.maximum(self.strike - spots, 0.0) 
        return vanilla
    
    @property
    def expiry(self) -> float:
        return self.expiry
    
    @property
    def exercise_style(self) -> float:
        return ExerciseStyle.EUROPEAN
    
    def __repr__(self) -> str:
        return (
            f'BarrierOption(strike={self.strike}, expiry={self._expiry}, type={self.option_type.value}, barrier={self.barrier}, rebate={self.rebate})'
        )
    

