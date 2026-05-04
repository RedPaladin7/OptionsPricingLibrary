"""
options_lib.models
------------------
Pricing models. Each model implements Model.price(instrument, market)
and optionally analytical Greeks.

Core models (safe to import at top level)
-----------------------------------------
BlackScholes        Analytical pricer for European options + full Greeks
implied_vol_bs      BS implied vol inversion (Newton-Raphson + Brent)
Heston              Char fn + Carr-Madan FFT + calibration
MonteCarlo          Generic MC (European, barrier, Asian) + variance reduction

Extended models (import directly to avoid circular dependencies)
---------------------------------------------------------------
These depend on market_data or numerics modules that depend back on models.
Import them directly from their modules:

    from options_lib.models.local_vol_mc    import LocalVolBarrierPricer
    from options_lib.models.heston_asian_mc import HestonAsianPricer
    from options_lib.numerics.lsmc          import LongstaffSchwartz
"""

from options_lib.models.base          import Model
from options_lib.models.black_scholes import BlackScholes
from options_lib.models.implied_vol   import implied_vol_bs
from options_lib.models.heston        import Heston, HestonParams
from options_lib.models.monte_carlo   import MonteCarlo, MonteCarloResult

__all__ = [
    "Model",
    "BlackScholes",
    "implied_vol_bs",
    "Heston",
    "HestonParams",
    "MonteCarlo",
    "MonteCarloResult",
    # Extended — import directly:
    # from options_lib.models.local_vol_mc    import LocalVolBarrierPricer
    # from options_lib.models.heston_asian_mc import HestonAsianPricer
]