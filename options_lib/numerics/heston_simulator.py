import numpy as np 
from dataclasses import dataclass
from typing import Optional

from options_lib.models.heston import HestonParams

@dataclass 
class HestonSimulator:
    params: object
    n_paths: int = 100_000 
    n_steps: int = 252 
    antithetic: bool = True 
    milstein: bool = True 
    truncation: str = 'full'
    scheme: str = 'euler'
    seed: Optional[int] = None

    def __post_init__(self):
        if self.truncation not in ('full', 'reflect'):
            raise ValueError(f"truncation must be 'full' or 'reflect', got {self.truncation}")
        if self.scheme not in ('euler', 'qe'):
            raise ValueError(f"scheme must be 'euler' or 'qe', got {self.scheme}")
        
    def simulate(
        self, 
        S0: float, 
        T: float, 
        r: float, 
        q: float = 0.0
    ) -> tuple:
        kappa = self.params.kappa
        v_bar = self.params.v_bar
        xi = self.params.xi
        rho   = self.params.rho
        v0    = self.params.v0

        dt = T / self.n_steps
        sqrt_dt = np.sqrt(dt)
        n_paths = self.n_paths
        rho_perp = np.sqrt(max(1 - rho**2, 0.0))

        if self.seed is not None:
            np.random.seed(self.seed)

        if self.antithetic:
            half = n_paths // 2
            zv_half = np.random.standard_normal((half, self.n_steps))
            zp_half = np.random.standard_normal((half, self.n_steps))
            z_v = np.concatenate([zv_half, -zv_half], axis=0)
            z_perp = np.concatenate([zp_half, -zp_half], axis=0)
        else:
            zv = np.random.standard_normal((n_paths, self.n_steps))
            zp = np.random.standard_normal((n_paths, self.n_steps))

        z_s = rho * z_v + rho_perp * z_perp

        s_paths = np.zeros((n_paths, self.n_steps + 1))
        v_paths = np.zeros((n_paths, self.n_steps + 1))

        s_paths[:, 0] = S0
        v_paths[:, 0] = v0

        for j in range(self.n_steps):
            v_j = v_paths[:, j]
            s_j = s_paths[:, j]

            v_pos = np.maximum(v_j, 0.0)
            sqrt_v = np.sqrt(v_pos)

            v_drift = kappa * (v_bar - v_pos) * dt 
            v_diffusion = xi * sqrt_v * sqrt_dt * z_v[:, j]

            if self.milstein:
                v_milstein = 0.25 * xi**2 * (z_v[:, j]**2 - 1) * dt 
                v_new = v_j + v_drift + v_diffusion + v_milstein
            else:
                v_new = v_j + v_drift + v_diffusion

            if self.truncation == 'full':
                v_new = np.maximum(v_new, 0.0)
            elif self.truncation == 'reflect':
                v_new = np.abs(v_new)

            v_paths[:, j+1] = v_new 

            log_drift = (r - q - 0.5 * v_pos) * dt 
            log_diffusion = sqrt_v * sqrt_dt * z_s[:, j]

            s_paths[:, j+1] = s_j * np.exp(log_drift + log_diffusion)
        return s_paths, v_paths 
    
    def terminal_distribution(
        self, 
        S0: float, 
        T: float, 
        r: float, 
        q: float = 0.0
    ) -> np.ndarray:
        s_paths, _ = self.simulate(S0, T, r, q)
        return s_paths[:, -1]
    
    def realized_variance(self, v_paths: np.ndarray, dt: float) -> np.ndarray:
        return np.trapezoid(v_paths, dx=dt, axis=1) / (v_paths.shape[1] * dt)
    
    def __repr__(self) -> str:
        return (
            f"HestonSimulator(v0={self.params.v0:.4f}, "
            f"κ={self.params.kappa:.2f}, v̄={self.params.v_bar:.4f}, "
            f"ξ={self.params.xi:.3f}, ρ={self.params.rho:.3f}, "
            f"n_paths={self.n_paths:,}, n_steps={self.n_steps}, "
            f"milstein={self.milstein})"
        )