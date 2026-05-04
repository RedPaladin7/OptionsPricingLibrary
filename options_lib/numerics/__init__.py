"""
options_lib.numerics
---------------------
Pure numerical methods. No finance logic — each module solves a
well-defined mathematical problem.

Modules
-------
root_finding         Brent's method + Newton-Raphson (used for IV inversion)
fft                  Carr-Madan FFT: characteristic function -> call prices O(N log N)
finite_difference    Crank-Nicolson PDE solver for European + American options
lsmc                 Longstaff-Schwartz MC for American options (backward induction)
heston_simulator     Euler-Milstein path simulation for the Heston (S, v) process
local_vol_simulator  Euler-Maruyama path simulation under Dupire local vol

Note: HestonSimulator and LocalVolSimulator are not imported here to avoid
circular imports (they depend on models/market_data respectively).
Import them directly:
    from options_lib.numerics.heston_simulator import HestonSimulator
    from options_lib.numerics.local_vol_simulator import LocalVolSimulator
"""

from options_lib.numerics.root_finding      import brent, newton_raphson, implied_vol
from options_lib.numerics.fft               import carr_madan_fft, interpolate_call_price
from options_lib.numerics.finite_difference import CrankNicolson
from options_lib.numerics.lsmc              import (
    LongstaffSchwartz,
    LSMCResult,
    laguerre_basis,
    monomial_basis,
)
from options_lib.numerics.heston_simulator import HestonSimulator

__all__ = [
    # Root finding
    "brent",
    "newton_raphson",
    "implied_vol",
    # FFT
    "carr_madan_fft",
    "interpolate_call_price",
    # PDE
    "CrankNicolson",
    # LSMC
    "LongstaffSchwartz",
    "LSMCResult",
    "laguerre_basis",
    "monomial_basis",
    "HestonSimulator"
    # Path simulators (import directly to avoid circular imports)
    # from options_lib.numerics.heston_simulator import HestonSimulator
    # from options_lib.numerics.local_vol_simulator import LocalVolSimulator
]