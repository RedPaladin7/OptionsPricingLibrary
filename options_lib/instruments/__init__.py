"""
options_lib.instruments
------------------------
Financial contracts. Each instrument defines WHAT the payoff is —
it knows nothing about how to price itself.

Instruments
-----------
EuropeanOption    Vanilla call/put, exercise at expiry only
AmericanOption    Early exercise at any t in [0, T]
BarrierOption     Knock-in / knock-out with optional rebate
AsianOption       Arithmetic or geometric average-price option

Base classes
------------
Instrument        Abstract base: payoff(), expiry, exercise_style
MarketData        Spot, risk-free rate, dividend yield
OptionType        CALL or PUT
ExerciseStyle     EUROPEAN or AMERICAN
"""

from options_lib.instruments.base     import (
    Instrument,
    MarketData,
    OptionType,
    ExerciseStyle,
)
from options_lib.instruments.european import EuropeanOption
from options_lib.instruments.american import AmericanOption
from options_lib.instruments.barrier  import BarrierOption, BarrierType
from options_lib.instruments.asian    import AsianOption, AverageType

__all__ = [
    "Instrument",
    "MarketData",
    "OptionType",
    "ExerciseStyle",
    "EuropeanOption",
    "AmericanOption",
    "BarrierOption",
    "BarrierType",
    "AsianOption",
    "AverageType",
]