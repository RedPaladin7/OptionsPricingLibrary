import numpy as np 
from dataclasses import dataclass
from options_lib.instruments.base import Instrument, OptionType, ExerciseStyle

@dataclass
class AmericanOption(Instrument):
    strike: float
    _expiry: float 
    option_type: OptionType

    def __init__(self, strike: float, expiry: float, option_type: OptionType):
        if strike <= 0:
            raise ValueError(f"Strike must be positive, got {strike}")
        if expiry <= 0:
            raise ValueError(f"Expiry must be positive, got {expiry}")
        self.strike      = strike
        self._expiry     = expiry
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
    def exercise_style(self) -> ExerciseStyle:
        return ExerciseStyle.AMERICAN
    
    def with_expiry(self, new_expiry: float) -> 'AmericanOption':
        return AmericanOption(
            strike=self.strike,
            expiry=new_expiry,
            option_type=self.option_type
        )
    
    def early_exercise_premium(
            self, 
            american_price: float,
            european_price: float
    ) -> float:
        premium = american_price - european_price
        if premium < -1e-6:
            raise ValueError(
                f"American price {american_price:.4f} < European price "
                f"{european_price:.4f}. This violates no-arbitrage."
            )
        return max(premium, 0.0)
    
    def __repr__(self) -> str:
        return (
            f'AmericanOption(strike={self.strike}, expiry={self._expiry}, type={self.option_type.value})'
        )