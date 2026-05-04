"""
options_lib.risk
-----------------
Greeks and P&L attribution.

Two levels of Greeks
--------------------
Scalar Greeks (greeks.py)
    Classic Delta, Gamma, Vega, Theta, Rho, Vanna, Volga, Charm.
    Analytical for BlackScholes, bump-and-reprice for Heston/MC.
    Vega = sensitivity to a single flat sigma — the standard textbook Greek.

Surface Greeks (surface_greeks.py)
    Sensitivities to movements in the entire SVI vol surface:
    - vega_parallel       dV per 1% parallel shift of all slices
    - vega_by_expiry      per-expiry vega bucketing (term structure)
    - skew_sensitivity    dV/d(rho) — sensitivity to smile tilt
    - curvature_sensitivity dV/d(sigma_svi) — sensitivity to smile convexity
    This is what production risk systems actually compute.

P&L Attribution (pnl_attribution.py)
    Taylor expansion decomposition:
    dV ≈ Delta*dS + 0.5*Gamma*dS^2 + Vega*dsigma + Theta*dt
         + Vanna*dS*dsigma + 0.5*Volga*dsigma^2
    With backtest support over historical paths.
"""

from options_lib.risk.greeks import (
    GreekEngine,
    GreekSurface,
    Greeks,
)
from options_lib.risk.surface_greeks import (
    SurfaceGreekEngine,
    SurfaceGreeks,
    VolSurfaceScenario,
    scenario_pnl,
    VegaMatrix,
)
from options_lib.risk.pnl_attribution import (
    PnLAttributor,
    PnLComponents,
    summarise_backtest,
)

__all__ = [
    # Scalar Greeks
    "GreekEngine",
    "GreekSurface",
    "Greeks",
    # Surface Greeks
    "SurfaceGreekEngine",
    "SurfaceGreeks",
    "VolSurfaceScenario",
    "scenario_pnl",
    # P&L attribution
    "PnLAttributor",
    "PnLComponents",
    "summarise_backtest",
    "VegaMatrix",
]