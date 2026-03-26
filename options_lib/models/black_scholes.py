"""
file: models/black_scholes.py
---
Analytical pricing model with full closed-form Greeks.
Black Scholes model assumes the stock follows GBM under probability measure Q:
dS = (r - q) S dt + (sigma) S dW^Q

Solving the BS PDE with terminal condition V(S, T) = payoff(S) gives closed form prices for European options.

Call price:
C = S e^(-qT) N(d1) - K e^(-rT) N(d2)
Put price:
P = K e^(-rT) N(-d2) - S e^(-qT) N(-d1)

where:
d1 = [ln(S/K) + (r - q + 0.5 * sigma^2) / (sigma * sqrt(T))]
d2 = d1 - sigma sqrt(T)

ALl greeks are derived analytically from the above formula.
"""
import numpy as np
from scipy.stats import norm 
from dataclasses import dataclass

from options_lib.models.base import Model 
from options_lib.instruments.base import Instrument, MarketData, OptionType
from options_lib.instruments.european import EuropeanOption

@dataclass
class BlackScholes(Model):
    """
    Black Scholes analytical pricer for European Options
    ---
    Parameters:
    sigma: float => constant volatility 
    Volatility is the single free parameter that calibrates the model to market price.
    """
    sigma: float 

    def __post_init__(self):
        if self.sigma <= 0:
            raise ValueError(f"Volatility sigma must be positive, got {self.sigma}")
    
    def _d1_d2(self, S: float, K: float, T: float, r: float, q: float):
        """
        Compute d1 and d2 from BS formula.
        d1 = [ln(S/K) + (r - q + 0.5 * sigma^2) / (sigma * sqrt(T))]
        d2 = d1 - sigma sqrt(T)
        """
        sqrt_T = np.sqrt(T)
        d1 = (np.log(S / K) + (r - q + 0.5 * self.sigma ** 2) * T) / (self.sigma * sqrt_T)
        d2 = d1 - self.sigma * sqrt_T
        return d1, d2 
    
    def price(self, instrument: Instrument, market: MarketData) -> float:
        """
        Price European option analytically. 
        Use different model for other options.
        ---
        Input: instrument and market data
        Output: fair value according to BS formula.
        """
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
            # C = S e^(-qT) N(d1) - K e^(-rT) N(d2)
            price = (S * np.exp(-q * T) * norm.cdf(-d2)
                     - S * np.exp(-q * T) * norm.cdf(-d1))
        else:
            # P = K e^(-rT) N(-d2) - S e^(-qT) N(-d1)
            price = (K * np.exp(-r *T) * norm.cdf(-d2) 
                     - S * np.exp(-q * T) * norm.cdf(-d1))
        return float(price)
    
    def delta(self, instrument: Instrument, market: MarketData) -> float:
        """
        Delta = dV/dS
        Call: e^(qT) * N(d1)  
        Put: e^(qT) * (N(d1) - 1)
        ---
        Detla shares of stock replicate the option's instantenous exposure to S. Hedge ratio.
        """
        if not isinstance(instrument, EuropeanOption):
            raise NotImplementedError("Analytical delta only for EuropeanOption")
        
        S, K, T, r, q = (market.spot, instrument.strike, instrument.expiry, market.rate, market.div_yield)

        d1, _ = self._d1_d2(S, K, T, r, q)

        if instrument.option_type == OptionType.CALL:
            return float(np.exp(-1 * T) * norm.cdf(d1))
        else:
            return float(np.exp(-q * T) * (norm.cdf(d1)-1))
    
    def gamma(self, instrument: Instrument, market: MarketData) -> float:
        """
        Gamma = d^V/dS^2
              = e^(-qT) N'(d1) / (S * sigma * sqrt(T))
        Same for calls and puts 
        Rate of change of delta with respect to S. Convexity profit from large moves. 
        High gamma => rebalance the hedge frequently.
        """
        if not isinstance(instrument, EuropeanOption):
            raise NotImplementedError("Analytical gamma only for EuropeanOption")
        
        S, K, T, r, q = (market.spot, instrument.strike, instrument.expiry, market.rate, market.div_yield)
        d1, _ = self._d1_d2(S, K, T, r, q)
        
        gamma = np.exp(-1 * T) * norm.pdf(d1) / (S * self.sigma * np.sqrt(T))

        return float(gamma)
    
    def vega(self, instrument: Instrument, market: MarketData) -> float:
        """
        Vega = dV/d(sigma)
        V = S e^(-qT) * N'(d1) * sqrt(T)
        Same for calls and puts
        ---
        Dollar change in option value per 1-unit increase in vol.
        Higher for ATM options and decays towards zero for deep ITM and OTM options. 

        """
        if not isinstance(instrument, EuropeanOption):
            raise NotImplementedError("Analytical vega only for EuropeanOption")
        
        S, K, T, r, q = (market.spot, instrument.strike, instrument.expiry, market.rate, market.div_yield)

        d1, _ = self._d1_d2(S, K, T, r, q)

        vega = S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)
        return float(vega)
    
    def theta(self, instrument: Instrument, market: MarketData) -> float:
        """
        Theta = dV/dt 
        
        Call theta:
            Θ = -[S e^{-qT} N'(d1) sigma / (2√T)]
                - r K e^{-rT} N(d2)
                + q S e^{-qT} N(d1)
 
        Put theta:
            Θ = -[S e^{-qT} N'(d1) sigma / (2√T)]
                + r K e^{-rT} N(-d2)
                - q S e^{-qT} N(-d1)

        Cost of owning time. Long options have negative theta - they lose value each just from time passing.

        Gamma Theta tradeoff:
        theta + 0.5 * sigma^2 * S^2 * gamma + (r-q)*S*delta - rV = 0
 
        """
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
        # per calender day
        return float(theta / 365)
    
    def rho(self, instrument: Instrument, market: MarketData) -> float:
        """
        rho = dV/dr 
        call: K T e^(-rT) N(d2) / 100 
        put: -K T e^(-rT) N(-d2) / 100 

        Becomes important for long dated options and for rate sensitive underlyings.
        """
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
        """
        Vanna = d^2 V / (dSdSigma) = dDelta / dSigma = dVega / dS
              = -e^(-qT) N'(d1) d2 / sigma 

        Same for calls and puts 

        Cross derivate: how delta changes with vol and how vega changes with spot prices.
        To be delta neutral through vol moves, you must also neutralize vanna. 
        Assymetrical structure of ITM and OTM calls.
        """
        if not isinstance(instrument, EuropeanOption):
            raise NotImplementedError("Analytical vanna only for EuropeanOption")
        S, K, T, r, q = (market.spot, instrument.strike, instrument.expiry, market.rate, market.div_yield)

        d1, d2 = self._d1_d2(S, K, T, r, q)

        vanna = -np.exp(-q * T) * norm.pdf(d1) * d2 / self.sigma 
        return float(vanna)
    
    def volga(self, instrument: Instrument, market: MarketData) -> float:
        """
        Volga = d^2V/dSigma^2 = dVega/dSigma 
              = Vega * d1 * d2 / sigma 

        Convexity of option value with respect to vol. 
        Positive value means option benefits from large moves in vol. 
        Used in vanna-volga approximation for pricing of exotics.
        Always positive for vanilla options.
        """
        if not isinstance(instrument, EuropeanOption):
            raise NotImplementedError("Analytical volga only for EuropeanOption")
        S, K, T, r, q = (market.spot, instrument.strike, instrument.expiry, market.rate, market.div_yield)

        d1, d2 = self._d1_d2(S, K, T, r, q)

        vega = S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)
        volga = vega * d1 * d2 / self.sigma 
        return float(volga)
    
    def charm(self, instrument: Instrument, market: MarketData) -> float:
        """
        Charm = d^V/dSdt = dDelta/dt = dTheta/dS
              = e^(qT) N'(d1) [2(r-q)T - d2 * sigma sqrt(T)] / (2T * sigma sqrt(T))

        Measure of how delta changes over time.
        Important for managing overnight risk.
        """
        if not isinstance(instrument, EuropeanOption):
            raise NotImplementedError("Analytical volga only for EuropeanOption")
        S, K, T, r, q = (market.spot, instrument.strike, instrument.expiry, market.rate, market.div_yield)

        d1, d2 = self._d1_d2(S, K, T, r, q)
        sqrt_T = np.sqrt(T)

        charm = (-np.exp(-q * T) * norm.pdf(d1)
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
