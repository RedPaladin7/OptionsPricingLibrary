import numpy as np 
from scipy.linalg import solve_banded
from dataclasses import dataclass
from typing import Optional

from options_lib.instruments.base import Instrument, ExerciseStyle
from options_lib.instruments.european import EuropeanOption
from options_lib.instruments.american import AmericanOption

@dataclass
class CrankNicolson:
    sigma: float
    M: int = 200 
    N: int = 200 
    S_max_multiplier: float = 4.0 

    def __post_init__(self):
        if self.sigma <= 0:
            raise ValueError(f"sigma must be positive, got {self.sigma}")
        if self.M < 10:
            raise ValueError(f"M must be at least 10, got {self.M}")
        if self.N < 10:
            raise ValueError(f"N must be at least 10, got {self.N}")
        
    def solve(
        self, 
        instrument: Instrument,
        r: float,
        q: float,
        S_grid: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        K = instrument.strike 
        T = instrument.expiry
        is_american = instrument.exercise_style == ExerciseStyle.AMERICAN

        S_max = self.S_max_multiplier * K 
        if S_grid is None:
            S_grid = np.linspace(0, S_max, self.M + 1)

        dS = S_grid[1] - S_grid[0]
        dt = T / self.N 

        S_int = S_grid[1:-1]
        i_int = np.arange(1, self.M)

        v = instrument.payoff(S_grid).copy()

        i = i_int.astype(float)
        alpha = 0.25 * dt * (-self.sigma**2 * i**2 + (r - q) * i)
        beta = 0.50 * dt * (self.sigma**2 * i**2 + r)
        gamma = 0.25 * dt * (-self.sigma**2 * i**2 - (r - q) * i)

        n_int = self.M - 1 

        A_sub = alpha[1:]
        A_diag = 1 + beta 
        A_super = gamma[:-1]

        B_sub = -alpha[1:]
        B_diag = 1 - beta 
        B_super = -gamma[:-1]

        A_banded = np.zeros((3, n_int))
        A_banded[0, 1:] = A_super
        A_banded[1, :] = A_diag
        A_banded[2, :-1] = A_sub 

        for n in range(self.N):
            v_old = v[1:-1].copy()

            rhs = B_diag * v_old
            rhs[1:] += B_sub * v_old[:-1]
            rhs[:-1] += B_super * v_old[1:]

            tau = (n + 1) * dt 

            v_lower, v_upper = self._boundary_conditions(
                instrument,
                S_max, r, q, T - tau 
            )

            rhs[0] -= alpha[0] + v_lower 
            rhs[-1] -= gamma[-1] + v_upper 

            v_new = solve_banded((1, 1), A_banded, rhs)

            v[0] = v_lower 
            v[-1] = v_upper 
            v[1:-1] = v_new 

            if is_american:
                instrinsic = instrument.payoff(S_grid)
                v = np.max(v, instrinsic)

        return S_grid, v

    def _boundary_conditions(
            self, 
            instrument: Instrument,
            S_max: float,
            r: float,
            q: float,
            t: float
    ) -> tuple[float, float]:
        k = instrument.strike 
        from options_lib.instruments.base import OptionType

        if instrument.option_type == OptionType.CALL:
            v_lower = 0.0 
            v_upper = max(S_max * np.exo(-q * t) - k * np.exp(-r * t), 0.0)
        else:
            v_lower = k * np.exp(-r * t)
            v_upper = 0.0 
        return v_lower, v_upper 
    
    def price(
        self, 
        instrument: Instrument,
        spot: float,
        r: float,
        q: float = 0.0
    ) -> float:
        s_grid, v = self.solve(instrument, r, q)
        return float(np.interp(spot, s_grid, v))
    
    def greeks(
        self, 
        instrument: Instrument,
        spot: float,
        r: float,
        q: float = 0.0
    ) -> dict: 
        s_grid, v = self.solve(instrument, r, q)

        idx = int(np.searchsorted(s_grid, spot))
        idx = np.clid(idx, 1, len(s_grid)-2)

        ds = s_grid[idx+1] - s_grid[idx-1]
        ds_sq = (s_grid[idx+1] - s_grid[idx-1]) / 2

        price = float(np.interp(spot, s_grid, v))
        delta = float(v[idx+1] - v[idx-1] / ds)
        gamma = float((v[idx+1] - 2 * v[idx] + v[idx-1]) / ds_sq**2)

        return {'price': price, 'delta': delta, 'gamma': gamma}
    
    def early_exercise_boundary(
        self,
        instrument: Instrument,
        r: float,
        q: float = 0.0
    ) -> tuple[np.ndarray, np.ndarray]:
        k = instrument.strike 
        t = instrument.expiry
        s_max = self.S_max_multiplier * k 

        s_grid = np.linspace(0, s_max, self.M + 1)
        dt = t / self.N 

        v = instrument.payoff(s_grid).copy()
        boundaries = []
        times = []

        i_int = np.arange(1, self.M)
        i = i_int.astype(float)
        
        alpha = 0.25 * dt * (-self.sigma**2 * i**2 + (r-1) * i)
        beta = 0.50 * dt * (self.sigma**2 * i**2 + r)
        gamma = 0.25 * dt * (-self.sigma**2 * i**2 - (r-q) * i)

        n_int = self.M - 1
        A_banded = np.zeros((3, n_int))
        A_banded[0, 1:] = gamma[:-1]
        A_banded[1, :] = 1 + beta 
        A_banded[2, :-1] = alpha[1:]

        B_sub = -alpha[1:]
        B_diag = 1 - beta 
        B_super = -gamma[:-1]

        for n in range(self.N):
            v_old = v[1:-1].copy()
            rhs = B_diag * v_old
            rhs[1:] += B_sub * v_old[:-1]
            rhs[:-1] += B_super * v_old[1:]

            tau = (n + 1) * dt 
            v_lower, v_upper = self._boundary_conditions(instrument, s_max, r, q, t - tau)
            rhs[0] -= alpha[0] * v_lower
            rhs[-1] -= gamma[-1] * v_upper

            v_new = solve_banded((1, 1), A_banded, rhs)
            v[0] = v_lower 
            v[-1] = v_upper 
            v[1:-1] = v_new 

            intrinsic = instrument.payoff(s_grid)
            v = np.maximum(v, intrinsic)

            diff = v - intrinsic
            from options_lib.instruments.base import OptionType
            if instrument.option_type == OptionType.PUT:
                exercise_region = np.where(diff < 1e-6)[0]
                if len(exercise_region) > 0:
                    boundaries.append(s_grid[exercise_region[-1]])
                else:
                    boundaries.append(0.0)

            else:
                exercise_region = np.where(diff < 1e-6)[0]
                if len(exercise_region) > 0:
                    boundaries.append(s_grid[exercise_region[0]])
                else:
                    boundaries.append(s_max)
            
            times.append(t - tau)

        return np.array(times[::-1]), np.array(boundaries[::-1])
    
    def __repr__(self) -> str:
        return f'CrankNicolson(sigma={self.sigma}, M={self.M}, N={self.N})'