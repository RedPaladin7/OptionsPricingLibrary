import numpy as np
from dataclasses import dataclass
from typing import Optional
from options_lib.market_data.local_vol import LocalVolSurface

@dataclass
class LocalVolSimulator:
    lv_surface: LocalVolSurface
    n_paths: int = 50_000
    n_steps: int = 252 
    antithetic: bool = True 
    milstein: bool = False 
    seed: Optional[int] = None 

    def simulate(
        self, 
        S0: float,
        T: float, 
        r: float,
        q: float = 0.0
    ) -> np.ndarray:
        if self.seed is not None:
            np.random.seed(self.seed)
        dt = T / self.n_steps
        n_paths = self.n_paths

        if self.antithetic:
            half = n_paths // 2
            z_half = np.random.standard_normal((half, self.n_steps))
            z = np.concatenate([z_half, -z_half], axis=0)
        else:
            z = np.random.standard_normal((n_paths, self.n_steps))
        
        paths = np.zeros((n_paths, self.n_steps+1))
        paths[:, 0] = S0

        for j in range(self.n_steps):
            t_j  = j * dt 
            s_j = paths[:, j]

            s_j_clipped = np.clip(
                s_j, 
                self.lv_surface.S_grid[0], 
                self.lv_surface.S_grid[-1]
            )
            t_clipped = np.clip(t_j, self.lv_surface.T_grid[0], self.lv_surface.T_grid[-1])

            sigma_loc = self._batch_local_vol(s_j_clipped, t_clipped)

            log_drift = (r - q - 0.5 * sigma_loc**2) * dt 
            diffusion = sigma_loc * np.sqrt(dt) * z[:, j]

            if self.milstein:
                ds = s_j_clipped * 0.01 
                s_ip = np.clip(s_j_clipped + ds, self.lv_surface.S_grid[0], self.lv_surface.S_grid[-1])
                sig_up = self._batch_local_vol(s_ip, t_clipped)
                dsig_ds = (sig_up - sigma_loc) / ds 
                milstein_corr = (0.5 * sigma_loc * dsig_ds * s_j_clipped * (z[:, j]**2 - 1)*dt)
                diffusion = diffusion + milstein_corr
            paths[:, j+1] = s_j * np.exp(log_drift + diffusion)
        return paths 

    def _batch_local_vol(self, s_array: np.ndarray, t: float) -> np.ndarray:
        if self.lv_surface._interpolant is not None:
            sigma_loc = self.lv_surface._interpolant.ev(
                np.full_like(s_array, t), s_array
            )
        else:
            sigma_loc = np.array([
                self.lv_surface.local_vol(float(s), t) for s in s_array
            ])
        return np.clip(sigma_loc, 0.01, 5.0)
    
    def __repr__(self) -> str:
        return (
            f"LocalVolSimulator(n_paths={self.n_paths:,}, "
            f"n_steps={self.n_steps}, antithetic={self.antithetic})"
        )