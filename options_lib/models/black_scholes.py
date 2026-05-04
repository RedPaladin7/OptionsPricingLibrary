"""
models/black_scholes.py
-----------------------
Black-Scholes analytical pricing model with full closed-form Greeks.

The Black-Scholes model assumes the stock follows GBM under Q:
    dS = (r - q) S dt + σ S dW^Q

where r is the risk-free rate and q is the continuous dividend yield.

Solving the BS PDE with terminal condition V(S,T) = payoff(S) gives
closed-form prices for European options.

Call price:  C = S e^{-qT} N(d1) - K e^{-rT} N(d2)
Put price:   P = K e^{-rT} N(-d2) - S e^{-qT} N(-d1)

where:
    d1 = [ln(S/K) + (r - q + σ²/2) T] / (σ √T)
    d2 = d1 - σ √T

All Greeks are derived analytically from these formulae.
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
    Black-Scholes analytical pricer for European options.

    Parameters
    ----------
    sigma : float
        Constant volatility σ (annualised, e.g. 0.20 = 20%).

    Notes
    -----
    Volatility is a model parameter, not market data, because in BS
    it is the single free parameter that calibrates the model to
    market prices (implied volatility).

    Examples
    --------
    >>> from options_lib.instruments.european import EuropeanOption
    >>> from options_lib.instruments.base import MarketData, OptionType
    >>> model = BlackScholes(sigma=0.20)
    >>> call  = EuropeanOption(strike=100, expiry=1.0, option_type=OptionType.CALL)
    >>> mkt   = MarketData(spot=100, rate=0.05, div_yield=0.0)
    >>> model.price(call, mkt)
    10.4506...
    """

    sigma: float

    def __post_init__(self):
        if self.sigma <= 0:
            raise ValueError(f"Volatility sigma must be positive, got {self.sigma}")

    # ------------------------------------------------------------------
    # Core d1, d2 computation
    # ------------------------------------------------------------------

    def _d1_d2(self, S: float, K: float, T: float, r: float, q: float):
        """
        Compute d1 and d2 from the BS formula.

        d1 = [ln(S/K) + (r - q + σ²/2) T] / (σ √T)
        d2 = d1 - σ √T

        Intuition:
        - d2: risk-neutral probability (in log-space) that S_T > K.
              N(d2) = Q(S_T > K), i.e. prob of call finishing ITM.
        - d1: d2 shifted by σ√T, accounts for the fact that we receive
              the stock (not just $1) on exercise. N(d1) = Delta of call.
        """
        sqrt_T = np.sqrt(T)
        d1 = (np.log(S / K) + (r - q + 0.5 * self.sigma ** 2) * T) / (self.sigma * sqrt_T)
        d2 = d1 - self.sigma * sqrt_T
        return d1, d2

    # ------------------------------------------------------------------
    # Pricing
    # ------------------------------------------------------------------

    def price(self, instrument: Instrument, market: MarketData) -> float:
        """
        Price a European option analytically.

        Only EuropeanOption instruments are supported. For other instrument
        types, raise NotImplementedError to force use of a different model.

        Parameters
        ----------
        instrument : EuropeanOption
            The European call or put to price.
        market : MarketData
            Spot, rate, and dividend yield.

        Returns
        -------
        float
            The BS price of the option.
        """
        if not isinstance(instrument, EuropeanOption):
            raise NotImplementedError(
                f"BlackScholes analytical pricer only supports EuropeanOption. "
                f"Got {type(instrument).__name__}. Use MonteCarlo or FiniteDifference."
            )

        S = market.spot
        K = instrument.strike
        T = instrument.expiry
        r = market.rate
        q = market.div_yield

        d1, d2 = self._d1_d2(S, K, T, r, q)

        if instrument.option_type == OptionType.CALL:
            # C = S e^{-qT} N(d1) - K e^{-rT} N(d2)
            price = (S * np.exp(-q * T) * norm.cdf(d1)
                     - K * np.exp(-r * T) * norm.cdf(d2))
        else:
            # P = K e^{-rT} N(-d2) - S e^{-qT} N(-d1)
            price = (K * np.exp(-r * T) * norm.cdf(-d2)
                     - S * np.exp(-q * T) * norm.cdf(-d1))

        return float(price)

    # ------------------------------------------------------------------
    # First-order Greeks
    # ------------------------------------------------------------------

    def delta(self, instrument: Instrument, market: MarketData) -> float:
        """
        Delta = dV/dS

        Call delta: e^{-qT} N(d1)         ∈ (0, 1)
        Put delta:  e^{-qT} (N(d1) - 1)   ∈ (-1, 0)

        Interpretation: delta shares of stock replicate the option's
        instantaneous exposure to S. This is the hedge ratio in the
        Black-Scholes delta-hedging argument.
        """
        if not isinstance(instrument, EuropeanOption):
            raise NotImplementedError("Analytical delta only for EuropeanOption.")

        S, K, T, r, q = (market.spot, instrument.strike, instrument.expiry,
                          market.rate, market.div_yield)
        d1, _ = self._d1_d2(S, K, T, r, q)

        if instrument.option_type == OptionType.CALL:
            return float(np.exp(-q * T) * norm.cdf(d1))
        else:
            return float(np.exp(-q * T) * (norm.cdf(d1) - 1))

    def gamma(self, instrument: Instrument, market: MarketData) -> float:
        """
        Gamma = d²V/dS²

        Γ = e^{-qT} N'(d1) / (S σ √T)

        Same formula for calls and puts (put-call parity implies equal gamma).

        Interpretation: Gamma measures the rate of change of Delta with
        respect to S. High gamma = Delta changes rapidly = hedger must
        rebalance frequently. Gamma is the "curvature" of the option price
        curve with respect to spot.

        Gamma P&L: ½ Γ (δS)² — the convexity profit from large moves.
        This is what you PAY for when you buy an option (via theta decay).
        """
        if not isinstance(instrument, EuropeanOption):
            raise NotImplementedError("Analytical gamma only for EuropeanOption.")

        S, K, T, r, q = (market.spot, instrument.strike, instrument.expiry,
                          market.rate, market.div_yield)
        d1, _ = self._d1_d2(S, K, T, r, q)

        gamma = np.exp(-q * T) * norm.pdf(d1) / (S * self.sigma * np.sqrt(T))
        return float(gamma)

    def vega(self, instrument: Instrument, market: MarketData) -> float:
        """
        Vega = dV/dσ

        V = S e^{-qT} N'(d1) √T

        Same formula for calls and puts.

        Interpretation: Vega measures the dollar change in option value
        per 1-unit (100%) increase in vol. In practice, quoted per 1%
        move in vol (divide by 100).

        Note: Vega is highest for ATM options and decays toward zero
        for deep ITM and OTM options. This makes sense — if an option is
        deep ITM, extra vol doesn't change much whether it expires ITM.
        """
        if not isinstance(instrument, EuropeanOption):
            raise NotImplementedError("Analytical vega only for EuropeanOption.")

        S, K, T, r, q = (market.spot, instrument.strike, instrument.expiry,
                          market.rate, market.div_yield)
        d1, _ = self._d1_d2(S, K, T, r, q)

        vega = S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)
        return float(vega)

    def theta(self, instrument: Instrument, market: MarketData) -> float:
        """
        Theta = dV/dt (per calendar day, sign convention: negative = decay)

        Call theta:
            Θ = -[S e^{-qT} N'(d1) σ / (2√T)]
                - r K e^{-rT} N(d2)
                + q S e^{-qT} N(d1)

        Put theta:
            Θ = -[S e^{-qT} N'(d1) σ / (2√T)]
                + r K e^{-rT} N(-d2)
                - q S e^{-qT} N(-d1)

        Divided by 365 to express per calendar day.

        Interpretation: Theta is the cost of owning time. Long options
        have negative theta — they lose value each day just from time
        passing (all else equal). This is the price you pay for Gamma.

        The fundamental BS relationship:
            Θ + ½ σ² S² Γ + (r-q) S Δ - r V = 0
        shows that Theta and Gamma are always in tension: high Gamma
        (good for the option holder) comes with high negative Theta (bad).
        """
        if not isinstance(instrument, EuropeanOption):
            raise NotImplementedError("Analytical theta only for EuropeanOption.")

        S, K, T, r, q = (market.spot, instrument.strike, instrument.expiry,
                          market.rate, market.div_yield)
        d1, d2 = self._d1_d2(S, K, T, r, q)
        sqrt_T = np.sqrt(T)

        # Common term: the N'(d1) term (always negative)
        common = -(S * np.exp(-q * T) * norm.pdf(d1) * self.sigma) / (2 * sqrt_T)

        if instrument.option_type == OptionType.CALL:
            theta = (common
                     - r * K * np.exp(-r * T) * norm.cdf(d2)
                     + q * S * np.exp(-q * T) * norm.cdf(d1))
        else:
            theta = (common
                     + r * K * np.exp(-r * T) * norm.cdf(-d2)
                     - q * S * np.exp(-q * T) * norm.cdf(-d1))

        return float(theta / 365)  # per calendar day

    def rho(self, instrument: Instrument, market: MarketData) -> float:
        """
        Rho = dV/dr (per 1% change in interest rate)

        Call rho:  K T e^{-rT} N(d2)  / 100
        Put rho:  -K T e^{-rT} N(-d2) / 100

        Interpretation: Rho is the least used first-order Greek for
        short-dated equity options because rates move slowly relative
        to spot and vol. It becomes important for long-dated options
        and for rate-sensitive underlyings.
        """
        if not isinstance(instrument, EuropeanOption):
            raise NotImplementedError("Analytical rho only for EuropeanOption.")

        S, K, T, r, q = (market.spot, instrument.strike, instrument.expiry,
                          market.rate, market.div_yield)
        _, d2 = self._d1_d2(S, K, T, r, q)

        if instrument.option_type == OptionType.CALL:
            rho = K * T * np.exp(-r * T) * norm.cdf(d2)
        else:
            rho = -K * T * np.exp(-r * T) * norm.cdf(-d2)

        return float(rho / 100)  # per 1% rate move

    # ------------------------------------------------------------------
    # Second-order / cross Greeks
    # ------------------------------------------------------------------

    def vanna(self, instrument: Instrument, market: MarketData) -> float:
        """
        Vanna = d²V / (dS dσ) = dDelta/dσ = dVega/dS

        Vanna = -e^{-qT} N'(d1) d2 / σ

        Same sign for calls and puts.

        Interpretation: Vanna measures how Delta changes when vol changes,
        and equivalently, how Vega changes when spot changes.

        Practical use: A portfolio that is delta-hedged but has non-zero
        Vanna will see its delta change when vol moves. To be delta-neutral
        through vol moves, you must also neutralise Vanna. This is the
        "vanna-volga" hedging method used by FX desks.

        Sign intuition: For a call, if vol increases (d1 increases toward
        N(d1)=1), the delta of a deep OTM call rises, but for a deep ITM
        call it's already near 1 and can't rise much. Vanna captures this
        asymmetry.
        """
        if not isinstance(instrument, EuropeanOption):
            raise NotImplementedError("Analytical vanna only for EuropeanOption.")

        S, K, T, r, q = (market.spot, instrument.strike, instrument.expiry,
                          market.rate, market.div_yield)
        d1, d2 = self._d1_d2(S, K, T, r, q)

        vanna = -np.exp(-q * T) * norm.pdf(d1) * d2 / self.sigma
        return float(vanna)

    def volga(self, instrument: Instrument, market: MarketData) -> float:
        """
        Volga (Vomma) = d²V/dσ² = dVega/dσ

        Volga = Vega * d1 * d2 / σ

        Interpretation: Volga measures the convexity of option value
        with respect to vol. Positive Volga means the option benefits
        from large moves in vol (like Gamma benefits from large moves
        in spot).

        An option with positive Volga gets more valuable when vol-of-vol
        is high. This is used in the vanna-volga approximation for pricing
        exotics: the cost of Volga exposure is priced in by comparing
        against vanilla options.

        Note: Volga is always positive for vanilla options (both calls and
        puts), which makes sense — options always benefit from more
        uncertainty (Jensen's inequality).
        """
        if not isinstance(instrument, EuropeanOption):
            raise NotImplementedError("Analytical volga only for EuropeanOption.")

        S, K, T, r, q = (market.spot, instrument.strike, instrument.expiry,
                          market.rate, market.div_yield)
        d1, d2 = self._d1_d2(S, K, T, r, q)

        vega  = S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)
        volga = vega * d1 * d2 / self.sigma
        return float(volga)

    def charm(self, instrument: Instrument, market: MarketData) -> float:
        """
        Charm = d²V / (dS dt) = dDelta/dt = dTheta/dS

        Call charm: e^{-qT} N'(d1) [2(r-q)T - d2 σ√T] / (2T σ√T)
        Put charm:  same formula, different sign convention via put-call parity.

        Interpretation: Charm measures how Delta changes over time.
        Important for managing overnight risk — if you delta-hedge at
        close, Charm tells you how much your delta will have drifted
        by the next morning purely from time passing.

        Sign: For an OTM call, charm is typically negative — delta
        decreases toward zero as time passes and the option is less
        likely to end ITM. For an ITM call, charm is positive — delta
        moves toward 1.
        """
        if not isinstance(instrument, EuropeanOption):
            raise NotImplementedError("Analytical charm only for EuropeanOption.")

        S, K, T, r, q = (market.spot, instrument.strike, instrument.expiry,
                          market.rate, market.div_yield)
        d1, d2 = self._d1_d2(S, K, T, r, q)
        sqrt_T = np.sqrt(T)

        charm = (-np.exp(-q * T) * norm.pdf(d1)
                 * (2 * (r - q) * T - d2 * self.sigma * sqrt_T)
                 / (2 * T * self.sigma * sqrt_T))

        if instrument.option_type == OptionType.PUT:
            charm = charm + q * np.exp(-q * T) * norm.cdf(-d1)

        return float(charm / 365)  # per calendar day

    # ------------------------------------------------------------------
    # Utility: BS PDE sanity check
    # ------------------------------------------------------------------

    def verify_pde(self, instrument: EuropeanOption, market: MarketData) -> float:
        """
        Verify the Black-Scholes PDE:
            Θ + ½ σ² S² Γ + (r-q) S Δ - r V = 0

        Returns the residual (should be ~0 up to numerical precision).
        Useful as a unit test sanity check.
        """
        V = self.price(instrument, market)
        D = self.delta(instrument, market)
        G = self.gamma(instrument, market)
        T_greek = self.theta(instrument, market) * 365  # convert back to per-year

        S, r, q = market.spot, market.rate, market.div_yield
        residual = T_greek + 0.5 * self.sigma**2 * S**2 * G + (r - q) * S * D - r * V
        return float(residual)

    def __repr__(self) -> str:
        return f"BlackScholes(sigma={self.sigma})"
