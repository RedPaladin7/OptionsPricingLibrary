"""
instruments/barrier.py
----------------------
Barrier options — path-dependent options that activate or deactivate
when the underlying crosses a barrier level H.

Types
-----
Knock-Out: option CEASES to exist if S crosses the barrier.
    - Down-and-Out: knocked out if S falls below H (H < S0)
    - Up-and-Out:   knocked out if S rises above H (H > S0)

Knock-In: option COMES INTO existence only if S crosses the barrier.
    - Down-and-In:  activated if S falls below H
    - Up-and-In:    activated if S rises above H

Key relationship (in/out parity):
    Knock-In + Knock-Out = Vanilla option (same strike, expiry)
    This must hold — the two barriers cover all possible paths.

Why barriers matter for pricing:
    - Much cheaper than vanillas (barrier may never be hit -> zero payoff)
    - Sensitive to the volatility SMILE near the barrier level, not just ATM vol
    - Cannot be priced correctly with flat vol — you need the full vol surface
    - This is why local vol and stochastic vol models price barriers differently
      even when calibrated to the same vanilla prices

Pricing:
    - Under flat BS vol: closed-form formulas exist (Reiner & Rubinstein 1991)
    - Under stochastic vol: Monte Carlo or 2D PDE required
    - We implement MC pricing via the MonteCarlo model
"""

import numpy as np
from dataclasses import dataclass
from enum import Enum
from options_lib.instruments.base import Instrument, OptionType, ExerciseStyle


class BarrierType(Enum):
    DOWN_AND_OUT = "down_and_out"
    DOWN_AND_IN  = "down_and_in"
    UP_AND_OUT   = "up_and_out"
    UP_AND_IN    = "up_and_in"


@dataclass
class BarrierOption(Instrument):
    """
    Single-barrier European option.

    Parameters
    ----------
    strike : float
        Strike price K.
    expiry : float
        Time to expiry T in years.
    option_type : OptionType
        CALL or PUT.
    barrier : float
        Barrier level H. Typically H < S for down barriers, H > S for up.
    barrier_type : BarrierType
        One of DOWN_AND_OUT, DOWN_AND_IN, UP_AND_OUT, UP_AND_IN.
    rebate : float
        Cash amount paid if the option is knocked out. Default 0.
        (Some products pay a consolation rebate on knockout.)

    Notes
    -----
    The payoff() method here computes the TERMINAL payoff given only the
    final spot price. This is only correct for European-style barriers that
    only check at expiry (a "European barrier").

    For the more common CONTINUOUS barrier (checked at all times), we need
    the full path — this is handled by the MonteCarlo model's path simulator,
    which calls is_knocked_out() along each simulated path.
    """

    strike       : float
    _expiry      : float
    option_type  : OptionType
    barrier      : float
    barrier_type : BarrierType
    rebate       : float = 0.0

    def __init__(
        self,
        strike       : float,
        expiry       : float,
        option_type  : OptionType,
        barrier      : float,
        barrier_type : BarrierType,
        rebate       : float = 0.0
    ):
        if strike <= 0:
            raise ValueError(f"Strike must be positive, got {strike}")
        if expiry <= 0:
            raise ValueError(f"Expiry must be positive, got {expiry}")
        if barrier <= 0:
            raise ValueError(f"Barrier must be positive, got {barrier}")

        self.strike       = strike
        self._expiry      = expiry
        self.option_type  = option_type
        self.barrier      = barrier
        self.barrier_type = barrier_type
        self.rebate       = rebate

    def is_knocked_out(self, path: np.ndarray) -> bool:
        """
        Check if a price path triggers the knockout barrier.

        Parameters
        ----------
        path : np.ndarray
            Array of spot prices along a path (shape: n_steps).

        Returns
        -------
        bool
            True if the barrier was breached at any point along the path.
        """
        if self.barrier_type in (BarrierType.DOWN_AND_OUT, BarrierType.DOWN_AND_IN):
            return bool(np.any(path <= self.barrier))
        else:  # UP_AND_OUT, UP_AND_IN
            return bool(np.any(path >= self.barrier))

    def is_knocked_in(self, path: np.ndarray) -> bool:
        """True if the knock-in barrier was triggered."""
        return self.is_knocked_out(path)  # same barrier logic, opposite effect

    def path_payoff(self, path: np.ndarray) -> float:
        """
        Compute the payoff for a single simulated path.
        This is the method the Monte Carlo engine calls.

        Parameters
        ----------
        path : np.ndarray
            Spot prices at each time step along a path.
            Final element path[-1] = S_T.

        Returns
        -------
        float
            Payoff for this path, accounting for barrier event.
        """
        S_T = path[-1]
        barrier_hit = self.is_knocked_out(path)

        # Compute vanilla payoff
        if self.option_type == OptionType.CALL:
            vanilla = max(S_T - self.strike, 0.0)
        else:
            vanilla = max(self.strike - S_T, 0.0)

        if self.barrier_type in (BarrierType.DOWN_AND_OUT, BarrierType.UP_AND_OUT):
            # Knock-out: if barrier hit, pay rebate; else pay vanilla
            return self.rebate if barrier_hit else vanilla
        else:
            # Knock-in: if barrier hit, pay vanilla; else pay rebate
            return vanilla if barrier_hit else self.rebate

    def payoff(self, spots: np.ndarray) -> np.ndarray:
        """
        Terminal-only payoff (ignores path). Used as a fallback.
        For proper continuous barrier pricing, use path_payoff() via MC.
        """
        spots = np.asarray(spots, dtype=float)
        if self.option_type == OptionType.CALL:
            vanilla = np.maximum(spots - self.strike, 0.0)
        else:
            vanilla = np.maximum(self.strike - spots, 0.0)
        return vanilla

    @property
    def expiry(self) -> float:
        return self._expiry

    @property
    def exercise_style(self) -> ExerciseStyle:
        return ExerciseStyle.EUROPEAN

    def __repr__(self) -> str:
        return (
            f"BarrierOption(K={self.strike}, T={self._expiry}, "
            f"type={self.option_type.value}, "
            f"H={self.barrier}, barrier={self.barrier_type.value})"
        )