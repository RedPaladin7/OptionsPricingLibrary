import numpy as np
from dataclasses import dataclass, field 
from typing import Optional, Callable

from options_lib.models.base import Model
from options_lib.instruments.base import Instrument, MarketData, ExerciseStyle
from options_lib.instruments.european import EuropeanOption
from options_lib.instruments.barrier import BarrierOption
from options_lib.instruments.asian import AsianOption

@dataclass
class MonteCarloResult:
    price: float 
    std_error: float 
    n_paths: int 

    @property
    def confidence_interval(self) -> tuple:
        return (
            self.price - 1.96 * self.std_error,
            self.price + 1.96 * self.std_error
        )
    
    def __repr__(self) -> str:
        lo, hi = self.confidence_interval
        return (
            f"MC Price: {self.price:.4f} "
            f"± {1.96*self.std_error:.4f} (95% CI: [{lo:.4f}, {hi:.4f}]) "
            f"[N={self.n_paths:,}]"
        )
    
@dataclass
class MonteCarlo(Model):
    sigma: float 
    n_paths: int = 100_000 
    n_steps: int = 252 
    antithetic: bool = True 
    control_variate: bool = True 
    seed: Optional[int] = None 

    def __post_init__(self):
        if self.sigma <= 0:
            raise ValueError(f"sigma must be positive, got {self.sigma}")
        if self.n_paths < 100:
            raise ValueError(f"n_paths must be at least 100, got {self.n_paths}")
        
    def simulate_paths(
        self, 
        S0: float, 
        T: float, 
        r: float, 
        q: float, 
        n_paths: Optional[int] = None,
        n_steps: Optional[int] = None
    ) -> np.ndarray:
        n_paths = n_paths or self.n_paths
        n_steps = n_steps or self.n_steps

        if self.seed is not None:
            np.random.seed(self.seed)
        

        dt = T / n_steps
        drift = (r - q - 0.5 * self.sigma ** 2) * dt
        vol = self.sigma * np.sqrt(dt)

        if self.antithetic:
            half = n_paths // 2
            z_half = np.random.standard_normal((half, n_steps))
            z = np.concatenate((z_half, -z_half))
        else:
            z = np.random.standard_normal((n_paths, n_steps))

        log_returns = drift + vol * z 
        log_S = np.log(S0) + np.cumsum(log_returns, axis=1)

        log_S0 = np.full((n_paths, 1), np.log(S0))
        log_paths = np.concatenate([log_S0, log_S], axis=1)

        return np.exp(log_paths)
    
    def price(self, instrument: Instrument, market: MarketData) -> float:
        return self.price_with_stats(instrument, market).price

    def price_with_stats(
        self, 
        instrument: Instrument, 
        market: MarketData
    ) -> MonteCarloResult:
        if isinstance(instrument, BarrierOption):
            return self._price_barrier(instrument, market)
        elif isinstance(instrument, AsianOption):
            return self._price_asian(instrument, market)
        elif isinstance(instrument, EuropeanOption):
            return self._price_european(instrument, market)
        else:
            return self._price_generic(instrument, market)

    def _price_european(
        self, 
        instrument: EuropeanOption, 
        market: MarketData
    ) -> MonteCarloResult:
        S0, r, q = market.spot, market.rate, market.div_yield
        T = instrument.expiry

        paths = self.simulate_paths(S0, T, r, q, n_steps=1)
        S_T = paths[:, -1]

        payoffs = instrument.payoff(S_T)
        disc_payoffs = np.exp(-r * T) * payoffs

        if self.control_variate:
            from options_lib.models.black_scholes import BlackScholes
            from options_lib.instruments.base import MarketData as MD 

            bs_model = BlackScholes(sigma=self.sigma)
            bs_price = bs_model.price(instrument, market)
            cv_payoffs = instrument.payoff(S_T)
            disc_cv = np.exp(-r * T) * cv_payoffs

            cov_matrix = np.cov(disc_payoffs, disc_cv)
            if cov_matrix[1, 1] > 1e-12:
                c = cov_matrix[0, 1] / cov_matrix[1, 1]
            else:
                c = 1.0 
            
            adjusted = disc_payoffs - c * (disc_cv - bs_price)
            price = float(np.mean(disc_payoffs))
            se = float(np.std(adjusted) / np.sqrt(self.n_paths))
        else:
            price = float(np.mean(disc_payoffs))
            se = float(np.std(disc_payoffs) / np.sqrt(self.n_paths))
        
        return MonteCarloResult(price=price, std_error=se, n_paths=self.n_paths)
    
    def _price_barrier(
        self, 
        instrument: BarrierOption, 
        market: MarketData
    ) -> MonteCarloResult:
        S0, r, q = market.spot, market.rate, market.div_yield
        T = instrument.expiry

        paths = self.simulate_paths(S0, T, r, q)

        payoffs = np.array([
            instrument.path_payoff(paths[i]) 
            for i in range(self.n_paths)
        ])

        disc_payoffs = np.exp(-r * T) * payoffs
        price = float(np.mean(disc_payoffs))
        se = float(np.std(disc_payoffs) / np.sqrt(self.n_paths))

        return MonteCarloResult(price=price, std_error=se, n_paths=self.n_paths)
    
    def _price_asian(
        self, 
        instrument: AsianOption, 
        market: MarketData
    ) -> MonteCarloResult:
        S0, r, q = market.spot, market.rate, market.div_yield
        T = instrument.expiry
        n_obs = instrument.n_observations

        paths = self.simulate_paths(S0, T, r, q, n_steps=n_obs)
        obs_paths = paths[:, 1:]

        geo_avg = np.exp(np.mean(np.log(obs_paths), axis=1))
        if instrument.option_type.value == 'call':
            geo_payoffs = np.maximum(geo_avg - instrument.strike, 0)
        else:
            geo_payoffs = np.maximum(instrument.strike - geo_avg, 0)

        disc_geo = np.exp(-r * T) * geo_payoffs
        geo_price_cf = self._kemna_vorst_price(instrument, market)

        from options_lib.instruments.asian import AverageType
        if instrument.average_type == AverageType.GEOMETRIC:
            price = float(np.mean(disc_geo))
            se = float(np.std(disc_geo) / np.sqrt(self.n_paths)) 
        else:
            arith_avg = np.mean(obs_paths, axis=1)
            if instrument.option_type.value == 'call':
                arith_payoffs = np.maximum(arith_avg - instrument.strike, 0)
            else:
                arith_payoffs = np.maximum(instrument.strike - arith_avg, 0)
            disc_arith = np.exp(-r * T) * arith_payoffs
            
            if self.control_variate:
                cov_matrix = np.cov(disc_arith, disc_geo)
                if cov_matrix[1, 1] < 1e-12:
                    c = cov_matrix[0, 1] / cov_matrix[1, 1]
                else:
                    c = 1.0 
                adjusted = disc_arith - c * (disc_geo - geo_price_cf)
                price = float(np.mean(adjusted))
                se = float(np.std(adjusted) / np.sqrt(self.n_paths))
            else:
                price = float(np.mean(disc_arith))
                se = float(np.std(disc_arith) / np.sqrt(self.n_paths))
        return MonteCarloResult(price=price, std_error=se, n_paths=self.n_paths)
        
    def _kemna_vorst_price(
        self, 
        instrument: AsianOption, 
        market: MarketData
    ) -> float:
        from scipy.stats import norm 
        S, K, T = market.spot, instrument.strike, instrument.expiry
        r, q, n = market.rate, market.div_yield, instrument.n_observations
        sigma = self.sigma 

        sigma_adj = sigma * np.sqrt((2 * n + 1) / (6 * (n + 1)))
        b_adj = 0.5 * (r - q - 0.5 * sigma**2 + sigma_adj**2)

        sqrt_T = np.sqrt(T)
        d1 = (np.log(S / K) + (b_adj + 0.5 * sigma_adj**2) * T) / (sigma_adj * sqrt_T)
        d2 = d1 - sigma_adj * sqrt_T

        if instrument.option_type.value == 'call':
            price = (S * np.exp(b_adj - r) * T * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))
        else:
            price = (K* np.exp(-r * T) * norm.cdf(-d2) - S * np.exp((b_adj - r) * T) * norm.cdf(-d1))
        
        return float(max(price, 0.0))
    
    def _price_generic(
        self, 
        instrument: Instrument, 
        market: MarketData,
    ) -> MonteCarloResult:
        S0, r, q = market.spot, market.rate, market.div_yield
        T = instrument.expiry

        paths = self.simulate_paths(S0, T, r, q, n_steps=1)
        S_T = paths[:, -1]
        payoffs = instrument.payoff(S_T)
        disc_payoffs = np.exp(-r * T) * payoffs

        price = float(np.mean(disc_payoffs))
        se = float(np.std(disc_payoffs) / np.sqrt(self.n_paths))

        return MonteCarloResult(price=price, std_error=se, n_paths=self.n_paths)
    
    def variance_reduction_summary(
        self, 
        instrument: Instrument, 
        market: MarketData
    ) -> dict:
        results = {}

        plain_mc = MonteCarlo(
            sigma=self.sigma, n_paths=self.n_paths,
            antithetic=False, control_variate=False, seed=self.seed
        )
        results["plain"] = plain_mc.price_with_stats(instrument, market)
 
        anti_mc = MonteCarlo(
            sigma=self.sigma, n_paths=self.n_paths,
            antithetic=True, control_variate=False, seed=self.seed
        )
        results["antithetic"] = anti_mc.price_with_stats(instrument, market)
 
        cv_mc = MonteCarlo(
            sigma=self.sigma, n_paths=self.n_paths,
            antithetic=True, control_variate=True, seed=self.seed
        )
        results["control_variate"] = cv_mc.price_with_stats(instrument, market)
 
        return results
    
    def __repr__(self) -> str:
        return (
            f"MonteCarlo(sigma={self.sigma}, n_paths={self.n_paths:,}, "
            f"n_steps={self.n_steps}, antithetic={self.antithetic}, "
            f"control_variate={self.control_variate})"
        )

