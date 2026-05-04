"""
options_lib
-----------
A from-scratch options pricing library.

Covers: Black-Scholes, Heston, Monte Carlo, Longstaff-Schwartz,
Crank-Nicolson FD, Dupire local vol, SVI vol surface calibration,
Greeks (scalar + surface), and P&L attribution.

Quick start
-----------
>>> from options_lib import BlackScholes, EuropeanOption, MarketData, OptionType
>>> model = BlackScholes(sigma=0.20)
>>> call  = EuropeanOption(strike=100, expiry=1.0, option_type=OptionType.CALL)
>>> mkt   = MarketData(spot=100, rate=0.05)
>>> model.price(call, mkt)
10.450...

Sub-packages
------------
options_lib.models          Pricing models (BS, Heston, MC)
options_lib.instruments     Contracts (European, American, Barrier, Asian)
options_lib.numerics        Numerical methods (FFT, FD, LSMC)
options_lib.market_data     Live data + vol surface (SVI, Dupire)
options_lib.risk            Greeks + P&L attribution (scalar + surface)

Extended models — import directly
----------------------------------
from options_lib.numerics.lsmc          import LongstaffSchwartz
from options_lib.models.local_vol_mc    import LocalVolBarrierPricer
from options_lib.models.heston_asian_mc import HestonAsianPricer
from options_lib.numerics.heston_simulator    import HestonSimulator
from options_lib.numerics.local_vol_simulator import LocalVolSimulator
"""

from options_lib.instruments.base     import MarketData, OptionType, ExerciseStyle
from options_lib.instruments.european import EuropeanOption
from options_lib.instruments.american import AmericanOption
from options_lib.instruments.barrier  import BarrierOption, BarrierType
from options_lib.instruments.asian    import AsianOption, AverageType

from options_lib.models.black_scholes import BlackScholes
from options_lib.models.implied_vol   import implied_vol_bs
from options_lib.models.heston        import Heston, HestonParams
from options_lib.models.monte_carlo   import MonteCarlo

__all__ = [
    # Market data
    "MarketData",
    "OptionType",
    "ExerciseStyle",
    # Instruments
    "EuropeanOption",
    "AmericanOption",
    "BarrierOption",
    "BarrierType",
    "AsianOption",
    "AverageType",
    # Core models
    "BlackScholes",
    "implied_vol_bs",
    "Heston",
    "HestonParams",
    "MonteCarlo",
]