"""
options_lib.market_data
------------------------
Live market data ingestion and vol surface calibration.

Pipeline
--------
1. fetch_option_chain()     Download + clean options from yfinance
2. calibrate_vol_surface()  Fit SVI per expiry slice -> VolSurface
3. build_local_vol_surface() Extract Dupire local vol from SVI surface

Classes
-------
OptionChain         Clean option quotes ready for surface fitting
OptionQuote         Single cleaned option quote with stripped IV
VolSurface          Calibrated SVI surface: IV(K, T), RND, arb checks
SVIParams           SVI slice parameters (a, b, rho, m, sigma)
LocalVolSurface     Dupire local vol surface sigma_loc(S, t)
"""

from options_lib.market_data.option_chain import (
    fetch_option_chain,
    OptionChain,
    OptionQuote,
)
from options_lib.market_data.vol_surface  import (
    SVIParams,
    VolSurface,
    calibrate_svi_slice,
    calibrate_vol_surface,
)
from options_lib.market_data.local_vol    import (
    LocalVolSurface,
    build_local_vol_surface,
)

__all__ = [
    # Option chain
    "fetch_option_chain",
    "OptionChain",
    "OptionQuote",
    # Vol surface
    "SVIParams",
    "VolSurface",
    "calibrate_svi_slice",
    "calibrate_vol_surface",
    # Local vol
    "LocalVolSurface",
    "build_local_vol_surface",
]