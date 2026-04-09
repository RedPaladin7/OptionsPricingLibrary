import numpy as np
from scipy.interpolate import RectBivariateSpline
from dataclasses import dataclass
from typing import Optional
import warnings 

from options_lib.market_data.vol_surface import VolSurface

@dataclass
class LocalVolSurface:
    vol_surface: VolSurface
    S_grid: np.ndarray
    T_grid: np.ndarray
    local_vol_grid: np.ndarray
    _interpolan: object = None 

    def __post_init__(self):
        valid = np.isfinite(self.local_vol_grid)
        if valid.sum() < 10:
            warnings.warn("Very few valid local vol grid points.")
 
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
        T = np.clip(T, self.T_grid[0], self.T_grid[-1])
        S = np.clip(S, self.S_grid[0], self.S_grid[-1])

        if self._interpolant is not None:
            val = float(self._interpolant.ev(T, S))
        else:
            i_T = np.searchsorted(self.T_grid, T)
            i_S = np.searchsorted(self.S_grid, S)
            i_T = np.clip(i_T, 0, len(self.T_grid) -1)
            i_S = np.clip(i_S, 0, len(self.S_grid) -1)
            val = float(self._lv_clean[i_T, i_S])
        return float(np.clip(val, 0.01, 5.0))
    
    def local_vol_grid_surface(self) -> np.ndarray:
        return self._lv_clean
    
    def compare_to_implied_vol(self, strikes: np.ndarray, expiries: np.ndarray) -> dict:
        spot = self.vol_surface.spot 
        lvs = np.zeros((len(expiries), len(strikes)))
        ivs = np.zeros((len(expiries), len(strikes)))

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
    

