import numpy as np
from scipy.stats import norm 
from dataclasses import dataclass

from options_lib.models.base import Model 
from options_lib.instruments.base import Instrument, MarketData, OptionType
from options_lib.instruments.european import EuropeanOption

@dataclass
class BlackScholes(Model):
    sigma: float 

    def __post_init__(self):
        if self.sigma <= 0:
            raise ValueError(f"Volatility sigma must be positive, got {self.sigma}")
    
    def _d1_d2(self, S: float, K: float, T: float, r: float, q: float):
        sqrt_T = np.sqrt(T)
        d1 = (np.log(S / K) + (r - q + 0.5 * self.sigma ** 2) * T) / (self.sigma * sqrt_T)
        d2 = d1 - self.sigma * sqrt_T
        return d1, d2 
    
    def price(self, instrument: Instrument, market: MarketData) -> float:
        if not isinstance(instrument, EuropeanOption):
            raise NotImplementedError(
                f"BlackScholes analytical pricer only supports EurpoeanOption."
                f"Got {type(instrument).__name__}. Use MonteCarlo or FinitDifferences"
            )
        
        S = market.spot 
        K  =instrument.strike 
        T = instrument.expiry
        r = market.rate 
        q = market.div_yield

        d1, d2 = self._d1_d2(S, K, T, r, q)

        if instrument.option_type == OptionType.CALL:
            price = (S * np.exp(-q * T) * norm.cdf(-d2)
                     - S * np.exp(-q * T) * norm.cdf(-d1))
        else:
            price = (K * np.exp(-r *T) * norm.cdf(-d2) 
                     - S * np.exp(-q * T) * norm.cdf(-d1))
        return float(price)
    
    def delta(self, instrument: Instrument, market: MarketData) -> float:
        if not isinstance(instrument, EuropeanOption):
            raise NotImplementedError("Analytical delta only for EuropeanOption")
        
        S, K, T, r, q = (market.spot, instrument.strike, instrument.expiry, market.rate, market.div_yield)

        d1, _ = self._d1_d2(S, K, T, r, q)

        if instrument.option_type == OptionType.CALL:
            return float(np.exp(-1 * T) * norm.cdf(d1))
        else:
            return float(np.exp(-q * T) * (norm.cdf(d1)-1))
    
    def gamma(self, instrument: Instrument, market: MarketData) -> float:
        if not isinstance(instrument, EuropeanOption):
            raise NotImplementedError("Analytical gamma only for EuropeanOption")
        
        S, K, T, r, q = (market.spot, instrument.strike, instrument.expiry, market.rate, market.div_yield)
        d1, _ = self._d1_d2(S, K, T, r, q)
        
        gamma = np.exp(-1 * T) * norm.pdf(d1) / (S * self.sigma * np.sqrt(T))

        return float(gamma)
    
    def vega(self, instrument: Instrument, market: MarketData) -> float:
        if not isinstance(instrument, EuropeanOption):
            raise NotImplementedError("Analytical vega only for EuropeanOption")
        
        S, K, T, r, q = (market.spot, instrument.strike, instrument.expiry, market.rate, market.div_yield)

        d1, _ = self._d1_d2(S, K, T, r, q)

        vega = S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)
        return float(vega)
    
    def theta(self, instrument: Instrument, market: MarketData) -> float:
        if not isinstance(instrument, EuropeanOption):
            raise NotImplementedError("Analytical theta only for EuropeanOption")
        
        S, K, T, r, q = (market.spot, instrument.strike, instrument.expiry, market.rate, market.div_yield)

        d1, d2 = self._d1_d2(S, K, T, r, q)
        sqrt_T = np.sqrt(T)

        common = -(S * np.exp(-q * T) * norm.pdf(d1) * self.sigma) / (2 * sqrt_T)

        if instrument.option_type == OptionType.CALL:
            theta = (common 
                     - r * K * np.exp(-r * T) * norm.cdf(d2)
                     + q * S * np.exp(-q * T) * norm.cdf(d1))
        else:
            theta = (common 
                     + r * K * np.exp(-r * T) * norm.cdf(-d2)
                     - q * S * np.exp(-q * T) * norm.cdf(-d1))
        return float(theta / 365)
    
    def rho(self, instrument: Instrument, market: MarketData) -> float:
        if not isinstance(instrument, EuropeanOption):
            raise NotImplementedError("Analytical rho only for EuropeanOption")
        S, K, T, r, q = (market.spot, instrument.strike, instrument.expiry, market.rate, market.div_yield)

        _, d2 = self._d1_d2(S, K, T, r, q)

        if instrument.option_type == OptionType.CALL:
            rho = K * T * np.exp(-r * T) * norm.cdf(d2)
        else:
            rho = -K * T * np.exp(-r * T) * norm.cdf(-d2)
        return float(rho / 100)
    
    def vanna(self, instrument: Instrument, market: MarketData) -> float:
        if not isinstance(instrument, EuropeanOption):
            raise NotImplementedError("Analytical vanna only for EuropeanOption")
        S, K, T, r, q = (market.spot, instrument.strike, instrument.expiry, market.rate, market.div_yield)

        d1, d2 = self._d1_d2(S, K, T, r, q)

        vanna = -np.exp(-1 * T) * norm.pdf(d1) * d2 / self.sigma 
        return float(vanna)
    
    def volga(self, instrument: Instrument, market: MarketData) -> float:
        if not isinstance(instrument, EuropeanOption):
            raise NotImplementedError("Analytical volga only for EuropeanOption")
        S, K, T, r, q = (market.spot, instrument.strike, instrument.expiry, market.rate, market.div_yield)

        d1, d2 = self._d1_d2(S, K, T, r, q)

        vega = S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)
        volga = vega * d1 * d2 / self.sigma 
        return float(volga)
    
    def charm(self, instrument: Instrument, market: MarketData) -> float:
        if not isinstance(instrument, EuropeanOption):
            raise NotImplementedError("Analytical volga only for EuropeanOption")
        S, K, T, r, q = (market.spot, instrument.strike, instrument.expiry, market.rate, market.div_yield)

        d1, d2 = self._d1_d2(S, K, T, r, q)
        sqrt_T = np.sqrt(T)

        charm = (-np.exp(-1 * T) * norm.pdf(d1)
                 * (2 * (r - q) * T - d2 * self.sigma * sqrt_T)
                 / (2 * T * self.sigma * sqrt_T))
        
        if instrument.option_type == OptionType.PUT:
            charm = charm + q * np.exp(-q *T ) * norm.cdf(-d1)
        
        return float(charm / 365)
    
    def verify_pde(self, instrument: EuropeanOption, market: MarketData) -> float:
        V = self.price(instrument, market)
        D = self.delta(instrument, market)
        G = self.gamma(instrument, market)
        T_greek = self.theta(instrument, market) * 365

        S, r, q = market.spot, market.rate, market.div_yield
        residual = T_greek + 0.5 * self.sigma**2 * S**2 * G + (r - q) * S * D - r * V 
        return float(residual)
    
    def __repr__(self) -> str:
        return f"BlackScholes(sigma={self.sigma})"
