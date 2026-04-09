import numpy as np
from scipy.optimize import minimize, Bounds 
from scipy.interpolate import interp1d
from dataclasses import dataclass, field
from typing import Optional 
import warnings 

from options_lib.models.black_scholes import BlackScholes
from options_lib.instruments.european import EuropeanOption
from options_lib.instruments.base import MarketData, OptionType

@dataclass
class SVIParams:
    a: float 
    b: float 
    rho: float 
    m: float 
    sigma: float 
    expiry: float 

    def total_variance(self, k: np.ndarray) -> np.ndarray:
        k = np.asarray(k, dtype=float)
        d = k - self.m 
        w = self.a + self.b * (self.rho * d + np.sqrt(d**2 + self.sigma**2))
        return np.maximum(w, 0.0)
    
    def implied_vol(self, k: np.ndarray) -> np.ndarray:
        w = self.total_variance(k)
        return np.sqrt(np.maximum(w / self.expiry, 1e-8))
    
    def implied_vol_from_strike(self, K: np.ndarray, F: float) -> np.ndarray:
        K = np.asarray(K, dtype=float)
        k = np.log(K / F)
        return self.implied_vol(k)
    
    def butterfly_density(self, k: np.ndarray) -> np.ndarray:
        k = np.asarray(k, dtype=float)
        w = self.total_variance(k)
        w = np.maximum(w, 1e-8)

        d = k - self.m 
        denom = np.sqrt(d**2 + self.sigma**2)
        dwdk = self.b * (self.rho + d / denom)
        d2wdk2 = self.b + self.sigma**2 / (denom**3)

        g = (1 - k *dwdk / (2 * w)) ** 2 - (dwdk**2 / 4) * (1/w + 1/4) + d2wdk2 / 2

        return g
    
    def is_butterfly_free(self, k_grid: Optional[np.ndarray]=None) -> bool:
        if k_grid is None:
            k_grid = np.linspace(-1.5, 1.5, 200)
        g = self.butterfly_density(k_grid)
        return bool(np.all(g >= 1e-6))
    
    def __repr__(self) -> str:
        iv_atm = float(self.implied_vol(np.array([self.m])))
        return (
            f"SVIParams(T={self.expiry:.3f}, "
            f"a={self.a:.4f}, b={self.b:.4f}, ρ={self.rho:.3f}, "
            f"m={self.m:.3f}, σ={self.sigma:.4f} | "
            f"ATM_vol={iv_atm:.1%})"
        )
    
def calibrate_svi_slice(
    log_moneyness: np.ndarray,
    market_ivs: np.ndarray,
    expiry: float,
    weights: Optional[np.ndarray]=None,
    n_restarts: int = 5
) -> SVIParams:
    if len(log_moneyness) < 5:
        raise ValueError('Need atleast 5 quotes to calibrate SVI')
    
    w_market = market_ivs**2 * expiry
    if weights is None:
        weights = np.one(len(log_moneyness))
    weights = weights / weights.sum()

    def objective(params):
        a, b, rho, m, sigma = params 
        if b <= 0 or sigma <= 0 or abs(rho) >= 1:
            return 1e6
        d = log_moneyness - m 
        w = a + b * (rho * d + np.sqrt(d**2 + sigma**2))
        if np.any(w <= 0):
            return 1e6 
        iv_model = np.sqrt(w / expiry)
        iv_market = np.sqrt(np.maximum(w_market/ expiry, 0))
        rmse = np.sqrt(np.sum(weights * (iv_model - iv_market)**2))
        return float(rmse)
    
    bounds = Bounds(
        lb=[-0.5, 1e-4, -0.99, -1.0, 1e-4],
        ub=[1.0, 2.0, 0.99, 1.0, 2.0]
    )

    atm_iv = float(np.interp(0, log_moneyness, market_ivs))
    w_atm = atm_iv**2 * expiry

    best_result = None 
    best_val = np.inf 

    np.random.seed(0)
    initial_guesses = [
        [w_atm * 0.8, 0.1, -0.3, 0.0, 0.1]
    ] + [
        [
            w_atm * np.random.uniform(0.5, 1.5),
            np.random.uniform(0.01, 0.5),
            np.random.uniform(-0.8, 0.2),
            np.random.uniform(-0.2, 0.2),
            np.random.uniform(0.01, 0.5)
        ]
        for _ in range(n_restarts - 1)
    ]

    for x0 in initial_guesses:
        try:
            result = minimize(
                objective, x0, 
                method='L-BFGS-B',
                bounds=bounds, 
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if result.fun < best_val:
                best_val = result.fun
                best_result = result 
        except Exception:
            continue

    if best_result is None:
        raise RuntimeError('SVI calibration failed for all restarts')
    a, b, rho, m, sigma = best_result.x 
    return SVIParams(a=a, b=b, rho=rho, m=m, sigma=sigma, expiry=expiry)

@dataclass
class VolSurface:
    svi_slices: dict 
    fowards: dict 
    spot: float 
    rate: float 
    ticker: str = ''

    @property
    def expiry_dates(self) -> list:
        return sorted(self.svi_slices.keys())
    
    @property
    def expiries(self) -> np.ndarray:
        return np.array([self.svi_slices[e].expiry for e in self.expiry_dates])
    
    def implied_vol(self, K: float, T: float) -> float:
        expiries = self.expiries
        dates = self.expiry_dates

        if T < expiries[0]:
            svi = self.svi_slices[dates[0]]
            F = self.fowards.get(dates[0], self.spot * np.exp(self.rate * T))
            k = np.log(K/F)
            return float(svi.implied_vol(np.array([k]))[0])
        if T >= expiries[-1]:
            svi = self.svi_slices[dates[-1]]
            F = self.fowards.get(dates[-1], self.spot * np.exp(self.rate * T))
            k = np.log(K/F)
            return float(svi.implied_vol(np.array([k]))[0])
        
        idx = np.searchsorted(expiries, T)
        T_lo, T_hi = expiries[idx-1], expiries[idx]
        d_lo, d_hi = dates[idx-1], dates[idx]

        F_lo = self.fowards.get(d_lo, self.spot * np.exp(self.rate * T_lo))
        F_hi = self.fowards.get(d_hi, self.spot * np.exp(self.rate * T_hi))

        k_lo = np.log(K/F_lo)
        k_hi = np.log(K/F_hi)

        w_lo = float(self.svi_slices[d_lo].total_variance(np.array([k_lo]))[0])
        w_hi = float(self.svi_slices[d_hi].total_variance(np.array([k_hi]))[0])

        alpha = (T - T_lo) / (T_hi - T_lo)
        w = (1 - alpha) * w_lo + alpha * w_hi 
        return float(np.sqrt(max(w/T, 1e-8)))
    
    def implied_vol_surface(self, strikes: np.ndarray, expiries: np.ndarray) -> np.ndarray:
        surface = np.zeros((len(expiries), len(strikes)))
        for i, T in enumerate(expiries):
            for j, K in enumerate(strikes):
                try:
                    surface[i, j] = self.implied_vol(K, T)
                except Exception:
                    surface[i, j] = np.nan 
        return surface 
    
    def check_calender_arbitrage(self) -> dict:
        k_grid = np.linspace(-1.0, 1.0, 100)
        dates = self.expiry_dates
        violations = []

        for i in range(len(dates)-1):
            d1, d2 = dates[i], dates[i+1]
            svi1 = self.svi_slices[d1]
            svi2 = self.svi_slices[d2]
            
            w1 = svi1.total_variance(k_grid)
            w2 = svi2.total_variance(k_grid)

            bad = k_grid[w2 < w1 - 1e-6]
            for k in bad:
                violations.append({
                    'T1': svi1.expiry, 'T2': svi2.expiry,
                    'k': k,
                    'w1': float(svi1.total_variance(np.array([k]))[0]),
                    'w2': float(svi2.total_variance(np.array([k]))[0])
                })
        return {
            'is_arbitrage_free': len(violations) == 0,
            'n_violations': len(violations),
            'violations': violations
        }
    
    def check_butterfly_arbitrage(self) -> dict:
        k_grid = np.linspace(-1.5, 1.5, 100)
        results = {}

        for d, svi in self.svi_slices.items():
            results[d] = svi.is_butterfly_free(k_grid)
        
        return {
            'is_arbitrage_free': all(results.values()),
            'slice_results': results
        }
    
    def risk_neutral_density(
        self, 
        expiry_date: str,
        K_grid: Optional[np.ndarray]=None
    ) -> tuple:
        svi = self.svi_slices[expiry_date]
        F = self.fowards.get(expiry_date, self.spot * np.exp(self.rate * svi.expiry))
        T = svi.expiry 

        atm_vol = float(svi.implied_vol(np.array([0.0]))[0])

        if K_grid is None:
            k_min = -3.0 * atm_vol * np.sqrt(T)
            k_max = 3.0 * atm_vol * np.sqrt(T)
            k_grid = np.linspace(k_min, k_max, 100)
            K_grid = F * np.exp(k_grid)
        else:
            k_grid = np.log(K_grid / F)

        g = svi.butterfly_density(k_grid)

        w = svi.total_variance(k_grid)
        w = np.maximum(w, 1e-8)
        sig  = np.sqrt(w / T)
        dens = g / (K_grid * sig * np.sqrt(T))
        dens = np.maximum(dens, 0.0)

        dk = np.diff(K_grid)
        norm = np.sum(0.5 * (dens[:-1] + dens[1:]) * dk)
        if norm > 1e-8:
            dens /= norm 
        return K_grid, dens
    
    def surface_summary(self) -> str:
        lines = [
            f"VolSurface: {self.ticker}",
            f"  Spot:    {self.spot:.2f}",
            f"  Slices:  {len(self.svi_slices)} expiries",
        ]
        for d, svi in self.svi_slices.items():
            atm_iv = float(svi.implied_vol(np.array([0.0]))[0])
            arb_ok = "✓" if svi.is_butterfly_free() else "✗"
            lines.append(
                f"    {d}: ATM={atm_iv:.1%}, ρ={svi.rho:+.2f}, "
                f"butterfly={arb_ok}"
            )
        return '\n'.join(lines)
    
def calibrate_vol_surface(
    option_chain, 
    min_quotes_per_slice: int = 5,
    use_vega_weights: bool = True, 
    verbose: bool = False
) -> VolSurface:
    svi_slices = {}
    forwards = option_chain.forwards.copy()

    for exp_date in option_chain.expiry_dates:
        quotes = option_chain.get_slice(exp_date)
        if len(quotes) < min_quotes_per_slice:
            if verbose:
                print(f'Skipping due to less quotes')
            continue 
        T = quotes[0].expiry 
        F = forwards.get(exp_date, option_chain.spot * np.exp(option_chain.rate * T))

        log_m = np.array([np.log(q.strike / F) for q in quotes])
        ivs = np.array([q.iv for q in quotes])

        if use_vega_weights:
            mkt = MarketData(
                spot=option_chain.spot,
                rate=option_chain.rate,
                div_yield=option_chain.div_yield
            )
            weights = np.array([
                BlackScholes(sigma=q.iv).vega(
                    EuropeanOption(
                        strike=q.strike, expiry=T,
                        option_type=OptionType.CALL
                    ), mkt 
                )
                for q in quotes
            ])
            weights = np.maximum(weights, 1e-6)
        else:
            weights = np.ones(len(quotes))

        try:
            svi = calibrate_svi_slice(log_m, ivs, T, weights=weights)
            svi_slices[exp_date] = svi 
            if verbose:
                atm_iv  = float(svi.implied_vol(np.array([0.0]))[0])
                mkt_atm = float(np.interp(0, log_m, ivs))
                arb     = "✓" if svi.is_butterfly_free() else "⚠"
                print(
                    f"  {exp_date} (T={T:.3f}): "
                    f"ATM_fit={atm_iv:.1%} mkt={mkt_atm:.1%} "
                    f"butterfly={arb} "
                    f"n={len(quotes)}"
                )
        except Exception as e:
            warnings.warn(f'SVI calibration failed for {exp_date}: {e}')
            continue 

    if not svi_slices:
        raise RuntimeError('SVI calibration failed for all expiry dates')
    
    surface = VolSurface(
        svi_slices=svi_slices,
        forwards=forwards,
        spot=option_chain.spot,
        rate=option_chain.rate,
        ticker=option_chain.ticker
    )

    if verbose:
        cal_check = surface.check_calendar_arbitrage()
        but_check = surface.check_butterfly_arbitrage()
        print(f"\nCalendar arbitrage free: {cal_check['is_arbitrage_free']}")
        print(f"Butterfly arbitrage free: {but_check['is_arbitrage_free']}")
 
    return surface



        
                        
                

        