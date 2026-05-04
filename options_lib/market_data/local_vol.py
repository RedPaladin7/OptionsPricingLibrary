"""
market_data/local_vol.py
-------------------------
Dupire local volatility surface extracted from the SVI vol surface.

What is Local Vol?
------------------
Black-Scholes assumes σ is constant. Heston makes it stochastic.
Local vol (Dupire 1994) takes a third approach: find a deterministic
function σ_loc(S, t) such that the model exactly reproduces ALL
market option prices simultaneously.

The stock process under local vol:
    dS = (r - q) S dt + σ_loc(S, t) S dW

Unlike BS (flat vol) and Heston (vol is stochastic), local vol is:
  - Fully consistent with all market vanillas (exact calibration)
  - A deterministic function of spot and time (not stochastic)
  - The "most parsimonious" model that fits the smile

Dupire's Formula
----------------
Given the implied vol surface σ_BS(K, T), Dupire showed that:

    σ_loc²(K, T) = [∂C/∂T + (r-q)K ∂C/∂K + qC] /
                   [½ K² ∂²C/∂K²]

In terms of total implied variance w(k, T) where k = ln(K/F):

    σ_loc²(K, T) = (∂w/∂T) /
                   [1 - k/w * ∂w/∂k
                    + ¼(-¼ - 1/w + k²/w²)(∂w/∂k)² + ½ ∂²w/∂k²]

This is Gatheral's (2006) formulation in terms of SVI total variance —
it's numerically more stable than differentiating call prices directly.

Why Local Vol for Barrier Options?
-----------------------------------
Local vol and Heston can both be calibrated to the same vanilla surface,
but they price barriers differently:

  - Local vol: the vol "smile" is baked into σ(S,t) — as S moves,
    you automatically reprice using the correct local vol at each point.
    The forward smile (conditional vol distribution) is different.

  - Heston: vol is stochastic independently of S. The forward smile
    dynamics differ from local vol.

  - Empirically: barrier options are sensitive to the "forward smile" —
    what the smile looks like in the future conditional on the spot level.
    Local vol and Heston give different answers here even with identical
    vanilla prices.

This model risk is real and is what sell-side desks manage daily.

Numerical Implementation
------------------------
We compute σ_loc(K, T) on a grid by:
1. Evaluating w(k, T) from the calibrated SVI surface
2. Numerically differentiating w w.r.t. T (central difference)
3. Numerically differentiating w w.r.t. k (central difference)
4. Applying Dupire's formula

Edge effects near the boundaries of the grid need care —
we use central differences in the interior and one-sided at edges.

References
----------
Dupire, B. (1994). Pricing with a smile. Risk, 7(1), 18-20.
Gatheral, J. (2006). The Volatility Surface. Wiley Finance.
"""

import numpy as np
from scipy.interpolate import RectBivariateSpline
from dataclasses import dataclass
from typing import Optional
import warnings

from options_lib.market_data.vol_surface import VolSurface


@dataclass
class LocalVolSurface:
    """
    Dupire local volatility surface σ_loc(S, t).

    Computed from a calibrated SVI surface and stored as a 2D
    interpolant over a (spot, time) grid.

    Parameters
    ----------
    vol_surface : VolSurface
        Calibrated SVI implied vol surface.
    S_grid : np.ndarray
        Spot grid for local vol computation.
    T_grid : np.ndarray
        Time grid for local vol computation.
    local_vol_grid : np.ndarray, shape (len(T_grid), len(S_grid))
        Local vol values σ_loc(S, T) on the grid.

    Usage
    -----
    >>> lv = build_local_vol_surface(vol_surface)
    >>> lv.local_vol(S=105, T=0.5)   # σ_loc at S=105, T=0.5yr
    """
    vol_surface     : VolSurface
    S_grid          : np.ndarray
    T_grid          : np.ndarray
    local_vol_grid  : np.ndarray   # shape: (len(T_grid), len(S_grid))
    _interpolant    : object = None

    def __post_init__(self):
        # Build 2D interpolant for fast queries at arbitrary (S, T)
        # RectBivariateSpline requires increasing axes
        valid = np.isfinite(self.local_vol_grid)
        if valid.sum() < 10:
            warnings.warn("Very few valid local vol grid points.")

        # Replace NaN/inf with nearest valid value (simple fill)
        lv_clean = self.local_vol_grid.copy()
        lv_clean = np.where(np.isfinite(lv_clean), lv_clean,
                            np.nanmedian(lv_clean))
        lv_clean = np.clip(lv_clean, 0.01, 5.0)

        try:
            self._interpolant = RectBivariateSpline(
                self.T_grid, self.S_grid, lv_clean,
                kx=3, ky=3   # cubic spline in both dimensions
            )
        except Exception as e:
            warnings.warn(f"Spline interpolant failed: {e}. Using linear fallback.")
            self._interpolant = None

        self._lv_clean = lv_clean

    def local_vol(self, S: float, T: float) -> float:
        """
        Interpolated local vol at spot S and time T.

        Parameters
        ----------
        S : float   Current spot price.
        T : float   Time in years.

        Returns
        -------
        float   Local vol σ_loc(S, T).
        """
        T = np.clip(T, self.T_grid[0], self.T_grid[-1])
        S = np.clip(S, self.S_grid[0], self.S_grid[-1])

        if self._interpolant is not None:
            val = float(self._interpolant.ev(T, S))
        else:
            # Linear fallback
            i_T = np.searchsorted(self.T_grid, T)
            i_S = np.searchsorted(self.S_grid, S)
            i_T = np.clip(i_T, 0, len(self.T_grid) - 1)
            i_S = np.clip(i_S, 0, len(self.S_grid) - 1)
            val = float(self._lv_clean[i_T, i_S])

        return float(np.clip(val, 0.01, 5.0))

    def local_vol_grid_surface(self) -> np.ndarray:
        """Return the raw local vol grid (T × S)."""
        return self._lv_clean

    def compare_to_implied_vol(
        self, strikes: np.ndarray, expiries: np.ndarray
    ) -> dict:
        """
        Compare local vol σ_loc(K, T) to implied vol σ_BS(K, T).

        The relationship between the two:
          E[σ_loc²(S_t, t) | S_T = K] = σ_BS²(K, T)

        So local vol is the expectation of local variance conditional
        on terminal spot = K. They agree at a single point but diverge
        for the full surface.

        Returns dict with 'local_vol', 'implied_vol', 'ratio' arrays.
        """
        spot = self.vol_surface.spot
        lvs  = np.zeros((len(expiries), len(strikes)))
        ivs  = np.zeros((len(expiries), len(strikes)))

        for i, T in enumerate(expiries):
            for j, K in enumerate(strikes):
                lvs[i, j] = self.local_vol(K, T)
                ivs[i, j] = self.vol_surface.implied_vol(K, T)

        return {
            'local_vol'    : lvs,
            'implied_vol'  : ivs,
            'ratio'        : lvs / np.maximum(ivs, 0.001),
            'strikes'      : strikes,
            'expiries'     : expiries,
        }


def build_local_vol_surface(
    vol_surface    : VolSurface,
    n_S            : int   = 100,
    n_T            : int   = 50,
    S_min_fraction : float = 0.5,
    S_max_fraction : float = 2.0,
    dT             : float = 1 / 365,
    dk             : float = 0.01,
) -> LocalVolSurface:
    """
    Build Dupire local vol surface from a calibrated SVI surface.

    Parameters
    ----------
    vol_surface : VolSurface
        Calibrated SVI surface.
    n_S : int
        Number of spot grid points.
    n_T : int
        Number of time grid points.
    S_min_fraction : float
        S_grid starts at spot * S_min_fraction.
    S_max_fraction : float
        S_grid ends at spot * S_max_fraction.
    dT : float
        Time step for numerical differentiation of w w.r.t. T.
    dk : float
        Log-moneyness step for numerical differentiation of w w.r.t. k.

    Returns
    -------
    LocalVolSurface
    """
    spot = vol_surface.spot
    rate = vol_surface.rate

    # Grids
    S_grid = np.linspace(spot * S_min_fraction, spot * S_max_fraction, n_S)
    T_min  = vol_surface.expiries.min()
    T_max  = vol_surface.expiries.max()
    T_grid = np.linspace(max(T_min * 0.5, dT * 2), T_max, n_T)

    lv_grid = np.full((len(T_grid), len(S_grid)), np.nan)

    for i, T in enumerate(T_grid):
        for j, S in enumerate(S_grid):
            try:
                lv = _dupire_local_vol(
                    S=S, T=T,
                    vol_surface=vol_surface,
                    rate=rate,
                    dT=dT, dk=dk
                )
                lv_grid[i, j] = lv
            except Exception:
                continue

    return LocalVolSurface(
        vol_surface    = vol_surface,
        S_grid         = S_grid,
        T_grid         = T_grid,
        local_vol_grid = lv_grid,
    )


def _dupire_local_vol(
    S           : float,
    T           : float,
    vol_surface : VolSurface,
    rate        : float,
    dT          : float = 1/365,
    dk          : float = 0.01,
) -> float:
    """
    Compute Dupire local vol at a single (S, T) point using Gatheral's
    formulation in total variance space.

    Dupire formula in (k, T) space:
        σ_loc² = (∂w/∂T) / D(k, w)

    where D(k, w) is the denominator:
        D = 1 - (k/w)(∂w/∂k) + ¼[-(¼ + 1/w) + k²/w²](∂w/∂k)² + ½(∂²w/∂k²)

    All derivatives computed numerically via central differences.
    """
    # Find nearest expiry slice for computing k and F
    dates    = vol_surface.expiry_dates
    expiries = vol_surface.expiries
    idx      = np.argmin(np.abs(expiries - T))
    exp_date = dates[idx]
    T_slice  = expiries[idx]

    F = vol_surface.forwards.get(
        exp_date,
        vol_surface.spot * np.exp(rate * T_slice)
    )
    # Use current T's forward for log-moneyness
    F_T = vol_surface.spot * np.exp(rate * T)
    k   = np.log(S / F_T)

    # Helper: total variance at (k, T)
    def w(k_val, T_val):
        K_val = F_T * np.exp(k_val)
        iv    = vol_surface.implied_vol(K_val, T_val)
        return max(iv**2 * T_val, 1e-8)

    w0 = w(k, T)

    # ------------------------------------------------------------------
    # Numerical derivatives
    # ------------------------------------------------------------------

    # ∂w/∂T — forward difference (we're always at T >= T_min)
    T_up   = min(T + dT, expiries[-1] * 0.999)
    T_down = max(T - dT, expiries[0]  * 1.001)
    if T_up > T_down:
        dwdT = (w(k, T_up) - w(k, T_down)) / (T_up - T_down)
    else:
        dwdT = 1e-4   # fallback

    if dwdT < 1e-6:
        return np.nan   # degenerate (flat term structure at this point)

    # ∂w/∂k — central difference
    dwdk = (w(k + dk, T) - w(k - dk, T)) / (2 * dk)

    # ∂²w/∂k² — central second difference
    d2wdk2 = (w(k + dk, T) - 2 * w0 + w(k - dk, T)) / dk**2

    # ------------------------------------------------------------------
    # Dupire denominator
    # ------------------------------------------------------------------
    D = (1.0
         - (k / w0) * dwdk
         + 0.25 * (-0.25 - 1.0 / w0 + k**2 / w0**2) * dwdk**2
         + 0.5 * d2wdk2)

    if D <= 1e-4:
        return np.nan   # butterfly arbitrage region — density near zero

    lv_sq = dwdT / D

    if lv_sq <= 0:
        return np.nan

    return float(np.sqrt(lv_sq))