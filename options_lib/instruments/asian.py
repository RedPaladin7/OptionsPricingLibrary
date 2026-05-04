"""
instruments/asian.py
--------------------
Asian options — payoff depends on the AVERAGE price of the underlying
over the option's life, not just the terminal price.

Why average?
    - Harder to manipulate than a single terminal price
    - Cheaper than vanilla (averaging reduces volatility: Var(avg) < Var(S_T))
    - Common in commodity and FX markets where spot can be illiquid at expiry

Types by average:
    Arithmetic average:  A = (1/n) Σ S_{t_i}   — no closed form under GBM
    Geometric average:   G = (Π S_{t_i})^{1/n} — has closed form under GBM
                                                   (Kemna & Vorst 1990)

Types by what's averaged:
    Average price:  payoff = max(A - K, 0)   — strike is fixed
    Average strike: payoff = max(S_T - A, 0) — strike is the average itself

We implement average-price options (most common).

Pricing:
    Geometric: closed-form via Kemna-Vorst, because geometric average of
               lognormals is lognormal.
    Arithmetic: no closed form — use Monte Carlo. The geometric price
                serves as an excellent CONTROL VARIATE for MC.
"""

import numpy as np
from dataclasses import dataclass
from enum import Enum
from options_lib.instruments.base import Instrument, OptionType, ExerciseStyle


class AverageType(Enum):
    ARITHMETIC = "arithmetic"
    GEOMETRIC  = "geometric"


@dataclass
class AsianOption(Instrument):
    """
    Asian (average price) option.

    Parameters
    ----------
    strike : float
        Strike price K.
    expiry : float
        Time to expiry T in years.
    option_type : OptionType
        CALL or PUT.
    average_type : AverageType
        ARITHMETIC or GEOMETRIC.
    n_observations : int
        Number of averaging dates (evenly spaced from 0 to T).
        E.g. 252 = daily averaging over 1 year.

    Notes
    -----
    The payoff() method is ill-defined for Asians because it needs the
    full path (to compute the average), not just the terminal spot.
    Use path_payoff() which the MC engine calls.

    payoff() returns vanilla payoff as a fallback — do not use it
    directly for Asian pricing.
    """

    strike         : float
    _expiry        : float
    option_type    : OptionType
    average_type   : AverageType
    n_observations : int

    def __init__(
        self,
        strike         : float,
        expiry         : float,
        option_type    : OptionType,
        average_type   : AverageType  = AverageType.ARITHMETIC,
        n_observations : int          = 252
    ):
        if strike <= 0:
            raise ValueError(f"Strike must be positive, got {strike}")
        if expiry <= 0:
            raise ValueError(f"Expiry must be positive, got {expiry}")
        if n_observations < 2:
            raise ValueError(f"Need at least 2 observation dates")

        self.strike         = strike
        self._expiry        = expiry
        self.option_type    = option_type
        self.average_type   = average_type
        self.n_observations = n_observations

    def compute_average(self, path: np.ndarray) -> float:
        """
        Compute the relevant average of a price path.

        Parameters
        ----------
        path : np.ndarray
            Spot prices at each observation date. Shape: (n_observations,).

        Returns
        -------
        float
            Arithmetic or geometric average.
        """
        if self.average_type == AverageType.ARITHMETIC:
            return float(np.mean(path))
        else:
            # Geometric average: exp(mean of log prices)
            return float(np.exp(np.mean(np.log(path))))

    def path_payoff(self, path: np.ndarray) -> float:
        """
        Compute payoff from a simulated path.
        Called by the Monte Carlo engine.

        Parameters
        ----------
        path : np.ndarray
            Spot prices at each observation date.

        Returns
        -------
        float
            max(A - K, 0) for calls, max(K - A, 0) for puts.
        """
        A = self.compute_average(path)
        if self.option_type == OptionType.CALL:
            return max(A - self.strike, 0.0)
        else:
            return max(self.strike - A, 0.0)

    def payoff(self, spots: np.ndarray) -> np.ndarray:
        """
        Vanilla payoff at terminal spot (fallback only — not path-aware).
        Use path_payoff() for actual Asian pricing.
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

    def __repr__(self) -> str:
        return (
            f"AsianOption(K={self.strike}, T={self._expiry}, "
            f"type={self.option_type.value}, "
            f"avg={self.average_type.value}, n={self.n_observations})"
        )