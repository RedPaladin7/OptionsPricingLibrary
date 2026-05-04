"""
instruments/american.py
-----------------------
American vanilla option — early exercise at any time up to expiry.

The key difference from European: the holder can exercise at any t in [0, T].
This means the option value must always satisfy:

    V(S, t) >= intrinsic_value(S)   for all t

This constraint creates a "free boundary" S*(t) — the critical spot price
at which early exercise is optimal. Below S*(t) (for puts), it's better
to exercise immediately than hold the option.

For American CALLS on non-dividend paying stocks:
    Early exercise is NEVER optimal (proven via Jensen's inequality).
    American call = European call.

For American PUTS (or calls on dividend-paying stocks):
    Early exercise CAN be optimal when deep ITM.
    The option picks up the intrinsic value now vs waiting for more
    time value that might erode anyway.

Pricing:
    No closed-form solution exists. Must use numerical methods:
    - Finite difference (Crank-Nicolson + early exercise constraint)
    - Binomial tree
    - Monte Carlo with Longstaff-Schwartz (least-squares regression)
"""

import numpy as np
from dataclasses import dataclass
from options_lib.instruments.base import Instrument, OptionType, ExerciseStyle


@dataclass
class AmericanOption(Instrument):
    """
    American vanilla option with early exercise.

    Parameters
    ----------
    strike : float
        Strike price K.
    expiry : float
        Time to expiry T in years.
    option_type : OptionType
        CALL or PUT.
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
        Intrinsic value payoff. Same formula as European payoff.
        For American options this is also the early exercise value at any t < T.
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
        return ExerciseStyle.AMERICAN

    def with_expiry(self, new_expiry: float) -> "AmericanOption":
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
        """
        Extra value of American over European due to early exercise right.
        Always >= 0 by no-arbitrage: more rights = more value.
        """
        premium = american_price - european_price
        if premium < -1e-6:
            raise ValueError(
                f"American price {american_price:.4f} < European price "
                f"{european_price:.4f}. This violates no-arbitrage."
            )
        return max(premium, 0.0)

    def __repr__(self) -> str:
        return (
            f"AmericanOption(strike={self.strike}, expiry={self._expiry}, "
            f"type={self.option_type.value})"
        )