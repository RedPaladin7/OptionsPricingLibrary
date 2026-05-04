"""
instruments/european.py
-----------------------
European vanilla call and put options.

A European option can only be exercised at expiry T.
Payoff:
    Call: max(S_T - K, 0)
    Put:  max(K - S_T, 0)

These are the building blocks of the entire options market.
All exotic options are in some sense deformations of these.
"""

import numpy as np
from dataclasses import dataclass
from options_lib.instruments.base import Instrument, OptionType, ExerciseStyle


@dataclass
class EuropeanOption(Instrument):
    """
    European vanilla option.

    Parameters
    ----------
    strike : float
        Strike price K.
    expiry : float
        Time to expiry T in years. E.g. 0.25 = 3 months.
    option_type : OptionType
        OptionType.CALL or OptionType.PUT.

    Examples
    --------
    >>> call = EuropeanOption(strike=100, expiry=1.0, option_type=OptionType.CALL)
    >>> call.payoff(np.array([90, 100, 110]))
    array([ 0.,  0., 10.])

    >>> put = EuropeanOption(strike=100, expiry=1.0, option_type=OptionType.PUT)
    >>> put.payoff(np.array([90, 100, 110]))
    array([10.,  0.,  0.])
    """

    strike      : float
    _expiry     : float
    option_type : OptionType

    def __init__(self, strike: float, expiry: float, option_type: OptionType):
        if strike <= 0:
            raise ValueError(f"Strike must be positive, got {strike}")
        if expiry <= 0:
            raise ValueError(f"Expiry must be positive, got {expiry}")
        self.strike      = strike
        self._expiry     = expiry
        self.option_type = option_type

    def payoff(self, spots: np.ndarray) -> np.ndarray:
        """
        Terminal payoff for an array of spot prices at expiry.

        Call payoff: max(S - K, 0)
        Put payoff:  max(K - S, 0)
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
    def exercise_style(self) -> ExerciseStyle:
        return ExerciseStyle.EUROPEAN

    def with_expiry(self, new_expiry: float) -> "EuropeanOption":
        """Return a copy of this option with a different expiry. Used for theta."""
        return EuropeanOption(
            strike=self.strike,
            expiry=new_expiry,
            option_type=self.option_type
        )

    def __repr__(self) -> str:
        return (
            f"EuropeanOption(strike={self.strike}, expiry={self._expiry}, "
            f"type={self.option_type.value})"
        )