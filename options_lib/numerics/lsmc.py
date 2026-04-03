import numpy as np 
from dataclasses import dataclass
from typing import Optional, Callable

from options_lib.instruments.base import OptionType
from options_lib.instruments.american import AmericanOption

def laguerre_basis(x: np.ndarray, degree: int = 4) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    basis = np.zeros((len(x), degree))
    basis[:, 0] = np.exp(-x / 2)
    if degree >= 2:
        basis[:, 1] = np.exp(-x / 2) * (1 - x)
    if degree >= 3:
        basis[:, 2] = np.exp(-x / 2) * (1 - 2*x + 0.5*x**2)
    if degree >= 4:
        basis[:, 3] = np.exp(-x / 2) * (1 - 3*x + 1.5*x**2 - x**3/6)
    if degree >= 5:
        basis[:, 4] = np.exp(-x / 2) * (1 - 4*x + 3*x**2 - 2*x**3/3 + x**4/24)

    for k in range(5, degree):
        basis[:, k] = x**k 
    return basis 

def monomial_basis(x: np.ndarray, degree: int = 4) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return np.column_stack([x**k for k in range(degree)])

@dataclass
class LSMCResult:
    price: float 
    std_error: float 
    n_paths: int 
    n_steps: int 
    exercise_boundary: Optional[np.ndarray] = None 

    @property
    def confidence_interval(self) -> tuple:
        return (
            self.price - 1.96 * self.std_error,
            self.price + 1.96 * self.std_error
        )
 
    def __repr__(self) -> str:
        lo, hi = self.confidence_interval
        return (
            f"LSMC Price: {self.price:.4f} "
            f"± {1.96*self.std_error:.4f} "
            f"(95% CI: [{lo:.4f}, {hi:.4f}]) "
            f"[N={self.n_paths:,}, steps={self.n_steps}]"
        )
    
@dataclass
class LongstaffSchwartz:
    sigma: float 
    n_paths: int = 100_000 
    n_steps: int = 100 
    degree: int = 4 
    basis: str = 'laguerre'
    antithetic: bool = True 
    seed: Optional[int] = None 
    path_simulator: Optional[Callable] = None 

    def __post_init__(self):
        if self.sigma <= 0:
            raise ValueError(f"sigma must be positive, got {self.sigma}")
        if self.n_paths < 1000:
            raise ValueError(f"n_paths should be at least 1000 for reliable results")
        if self.degree < 2:
            raise ValueError(f"degree must be at least 2")
        
    def _simulate_paths(
        self, 
        S0: float, 
        T: float, 
        r: float, 
        q: float
    ) -> np.ndarray:
        if self.path_simulator is not None:
            return self.path_simulator(S0, T, r, q, self.n_paths, self.n_steps)
        if self.seed is not None:
            np.random.seed(self.seed)
        
        dt = T / self.n_steps
        drift = (r - q - 0.5 * self.sigma**2) * dt
        vol   = self.sigma * np.sqrt(dt)
 
        if self.antithetic:
            half    = self.n_paths // 2
            Z_half  = np.random.standard_normal((half, self.n_steps))
            Z       = np.concatenate([Z_half, -Z_half], axis=0)
        else:
            Z = np.random.standard_normal((self.n_paths, self.n_steps))
        
        log_returns = drift + vol * Z           
        log_S       = np.log(S0) + np.cumsum(log_returns, axis=1)
        log_S0      = np.full((self.n_paths, 1), np.log(S0))
 
        return np.exp(np.concatenate([log_S0, log_S], axis=1))
    
    def _basis_matrix(self, S: np.ndarray, K: float) -> np.ndarray:
        x = S / K 
        if self.basis == 'laguerre':
            return laguerre_basis(x, self.degree)
        else:
            return monomial_basis(x, self.degree)
        
    def price(
        self, 
        instrument: AmericanOption,
        spot: float,
        rate: float,
        div_yield: float = 0.0,
        extract_boundary: bool = False
    ) -> LSMCResult:
        S0 = spot 
        T = instrument.expiry
        K = instrument.strike 
        r = rate 
        q = div_yield
        dt = T / self.n_steps

        paths = self._simulate_paths(S0, T, r, q)
        cash_flows = instrument.payoff(paths[:, -1]).copy()

        boundary = np.full(self.n_steps, np.nan) if extract_boundary else None 

        for j in range(self.n_steps - 1, 0, -1):
            s_j = paths[:, j]
            t_j = j * dt 

            intrinsic = instrument.payoff(s_j)

            itm_mask = intrinsic > 0 
            n_itm = itm_mask.sum() 

            if n_itm < self.degree + 1:
                continue 

            x = self._basis_matrix(s_j[itm_mask], K)
            y = np.exp(-r * dt) * cash_flows[itm_mask]

            beta_hat, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
            continuation = x @ beta_hat

            exercise_now = intrinsic[itm_mask] > continuation

            itm_indices = np.where(itm_mask)[0]
            exercsise_indices = itm_indices[exercise_now]
            no_exercise_indices = itm_indices[~exercise_now]

            cash_flows[exercsise_indices] = intrinsic[itm_mask][exercise_now]
            cash_flows[no_exercise_indices] = np.exp(-r * dt) * cash_flows[no_exercise_indices]

            cash_flows[~itm_mask] = np.exp(-r * dt) * cash_flows[~itm_mask]

            if extract_boundary and n_itm > 0:
                if exercise_now.any() and (~exercise_now).any():
                    ex_s = s_j[itm_mask][exercise_now]
                    no_ex_s = s_j[itm_mask][~exercise_now]
                    if instrument.option_type == OptionType.PUT:
                        boundary[j] = float(np.percentile(ex_s, 90))
                    else:
                        boundary[j] = float(np.percentile(ex_s, 10))
            pv_cash_flows = np.exp(-r * dt) * cash_flows

            price = float(np.mean(pv_cash_flows))
            std_error = float(np.std(pv_cash_flows) / np.sqrt(self.n_paths))

        return LSMCResult(
            price=max(price, 0.0),
            std_error=std_error,
            n_paths=self.n_paths,
            n_steps=self.n_steps,
            exercise_boundary=boundary
        )
    
    def compare_to_european(
        self, 
        instrument: AmericanOption,
        spot: float,
        rate: float,
        div_yield: float = 0.0
    ) -> dict:
        from options_lib.models.black_scholes import BlackScholes
        from options_lib.instruments.european import EuropeanOption
        from options_lib.instruments.base import MarketData

        am_result = self.price(instrument, spot, rate, div_yield)

        eu_opt = EuropeanOption(
            strike=instrument.strike,
            expiry=instrument.expiry,
            option_type=instrument.option_type,
        )
        mkt =  MarketData(spot=spot, rate=rate, div_yield=div_yield)
        eu_price = BlackScholes(sigma=self.sigma).price(eu_opt, mkt)

        return {
            'american_price'         : am_result.price,
            'american_std_error'     : am_result.std_error,
            'european_price'         : eu_price,
            'early_exercise_premium' : max(am_result.price - eu_price, 0.0),
        }
    
    def __repr__(self) -> str:
        return (
            f"LongstaffSchwartz(sigma={self.sigma}, "
            f"n_paths={self.n_paths:,}, n_steps={self.n_steps}, "
            f"degree={self.degree}, basis={self.basis})"
        )
