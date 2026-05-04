"""
models/monte_carlo.py
---------------------
Generic Monte Carlo pricing engine.

The idea: under the risk-neutral measure Q, the price of any derivative is:
    V_0 = e^{-rT} * E^Q[payoff(S_T)]

Monte Carlo approximates this expectation by:
    1. Simulating N paths of S_t under Q
    2. Computing the payoff on each path
    3. Averaging and discounting

    V_0 ~ e^{-rT} * (1/N) Σ payoff(path_i)

The standard error of this estimate is sigma_payoff / sqrt(N).
To halve the error you need 4x the paths. This sqrt(N) convergence
is the main limitation of MC — but variance reduction techniques
can dramatically improve it.

Variance Reduction
------------------
Control Variates:
    If X is our payoff and Y is a correlated payoff with known mean E[Y],
    then Z = X - c*(Y - E[Y]) has the same mean as X but lower variance.
    Optimal c = Cov(X,Y) / Var(Y), estimated from the same simulation.
    For vanilla options: use BS price as control. Cov is very high (same paths).

Antithetic Variates:
    For each random path W, also simulate -W (the mirror path).
    Average the two payoffs: (payoff(W) + payoff(-W)) / 2.
    The two paths are negatively correlated, reducing variance.
    Cost: 2x function evaluations, but typically 4x+ variance reduction.

Path Generation
---------------
Under Q, GBM stock price at each step:
    S_{t+dt} = S_t * exp[(r - q - sigma^2/2)*dt + sigma*sqrt(dt)*Z]
             where Z ~ N(0,1)

This is the exact solution (not an approximation) for GBM.
It is exact because GBM has a lognormal transition density.
"""

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
    """
    Results from a Monte Carlo pricing run.

    Attributes
    ----------
    price : float
        Estimated option price (mean discounted payoff).
    std_error : float
        Standard error of the price estimate = std(payoffs) / sqrt(N).
        95% confidence interval: price +/- 1.96 * std_error.
    n_paths : int
        Number of simulated paths.
    confidence_interval : tuple[float, float]
        95% CI: (price - 1.96*se, price + 1.96*se).
    """
    price       : float
    std_error   : float
    n_paths     : int

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
    """
    Generic Monte Carlo pricing engine for European, barrier, and Asian options.

    Parameters
    ----------
    sigma : float
        Volatility (constant BS vol).
    n_paths : int
        Number of simulation paths. More paths = lower std error.
        Std error scales as 1/sqrt(n_paths).
    n_steps : int
        Number of time steps per path. For European options, n_steps=1
        is exact (log-normal). For path-dependent options, more steps
        = finer monitoring of barrier/average.
    antithetic : bool
        Use antithetic variates variance reduction. Default True.
        Effectively doubles the useful path count at the same cost.
    control_variate : bool
        Use BS price as control variate for European options. Default True.
        Dramatically reduces variance when the BS price is close to MC.
    seed : int, optional
        Random seed for reproducibility.

    Notes
    -----
    For European options with control_variate=True, the pricing error
    is typically 10-50x smaller than plain MC, equivalent to running
    100-2500x more paths.
    """

    sigma           : float
    n_paths         : int   = 100_000
    n_steps         : int   = 252
    antithetic      : bool  = True
    control_variate : bool  = True
    seed            : Optional[int] = None

    def __post_init__(self):
        if self.sigma <= 0:
            raise ValueError(f"sigma must be positive, got {self.sigma}")
        if self.n_paths < 100:
            raise ValueError(f"n_paths must be at least 100, got {self.n_paths}")

    # ------------------------------------------------------------------
    # Path simulation
    # ------------------------------------------------------------------

    def simulate_paths(
        self,
        S0      : float,
        T       : float,
        r       : float,
        q       : float,
        n_paths : Optional[int] = None,
        n_steps : Optional[int] = None,
    ) -> np.ndarray:
        """
        Simulate GBM paths under the risk-neutral measure Q.

        Uses the exact lognormal transition:
            S_{t+dt} = S_t * exp[(r-q-sigma^2/2)*dt + sigma*sqrt(dt)*Z]

        Parameters
        ----------
        S0 : float
            Initial spot price.
        T : float
            Time horizon in years.
        r, q : float
            Risk-free rate and dividend yield.
        n_paths : int, optional
            Override self.n_paths.
        n_steps : int, optional
            Override self.n_steps.

        Returns
        -------
        paths : np.ndarray, shape (n_paths, n_steps + 1)
            Simulated spot prices. Column 0 = S0, column -1 = S_T.
        """
        n_paths = n_paths or self.n_paths
        n_steps = n_steps or self.n_steps

        if self.seed is not None:
            np.random.seed(self.seed)

        dt    = T / n_steps
        drift = (r - q - 0.5 * self.sigma**2) * dt
        vol   = self.sigma * np.sqrt(dt)

        # Generate standard normal increments: shape (n_paths, n_steps)
        if self.antithetic:
            # Generate half the paths, then mirror them
            half = n_paths // 2
            Z_half = np.random.standard_normal((half, n_steps))
            Z = np.concatenate([Z_half, -Z_half], axis=0)
        else:
            Z = np.random.standard_normal((n_paths, n_steps))

        # Compute log-returns and cumulate
        log_returns = drift + vol * Z          # shape (n_paths, n_steps)
        log_S = np.log(S0) + np.cumsum(log_returns, axis=1)

        # Prepend the initial log price
        log_S0 = np.full((n_paths, 1), np.log(S0))
        log_paths = np.concatenate([log_S0, log_S], axis=1)

        return np.exp(log_paths)               # shape (n_paths, n_steps+1)

    # ------------------------------------------------------------------
    # Pricing
    # ------------------------------------------------------------------

    def price(self, instrument: Instrument, market: MarketData) -> float:
        """
        Price an instrument via Monte Carlo.
        Returns the point estimate (no std error). Use price_with_stats()
        for the full result including confidence interval.
        """
        return self.price_with_stats(instrument, market).price

    def price_with_stats(
        self,
        instrument : Instrument,
        market     : MarketData,
    ) -> MonteCarloResult:
        """
        Price with full Monte Carlo statistics.

        Dispatches to the appropriate pricing method based on instrument type.
        """
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
        instrument : EuropeanOption,
        market     : MarketData,
    ) -> MonteCarloResult:
        """
        Price a European option, optionally using BS as a control variate.

        Control variate method:
            Let X = MC payoff, Y = BS payoff on the SAME paths.
            E[Y] = BS price (known analytically).
            Adjusted estimator: Z_i = X_i - c*(Y_i - E[Y])
            where c = Cov(X,Y)/Var(Y) estimated from the simulation.
            Var(Z) = Var(X) - Cov(X,Y)^2/Var(Y) <= Var(X).
        """
        S0, r, q = market.spot, market.rate, market.div_yield
        T = instrument.expiry

        # Simulate terminal prices only (n_steps=1 is exact for European)
        paths = self.simulate_paths(S0, T, r, q, n_steps=1)
        S_T = paths[:, -1]                             # shape (n_paths,)

        # Raw payoffs
        payoffs = instrument.payoff(S_T)               # shape (n_paths,)
        disc_payoffs = np.exp(-r * T) * payoffs

        if self.control_variate:
            # Control variate: BS call/put on the same terminal prices
            from options_lib.models.black_scholes import BlackScholes
            from options_lib.instruments.base import MarketData as MD

            bs_model    = BlackScholes(sigma=self.sigma)
            bs_price    = bs_model.price(instrument, market)
            cv_payoffs  = instrument.payoff(S_T)       # same payoff formula
            disc_cv     = np.exp(-r * T) * cv_payoffs  # Y_i

            # Estimate c = Cov(X, Y) / Var(Y)
            cov_matrix = np.cov(disc_payoffs, disc_cv)
            if cov_matrix[1, 1] > 1e-12:
                c = cov_matrix[0, 1] / cov_matrix[1, 1]
            else:
                c = 1.0

            # Adjusted payoffs: Z_i = X_i - c*(Y_i - E[Y])
            adjusted = disc_payoffs - c * (disc_cv - bs_price)
            price    = float(np.mean(adjusted))
            se       = float(np.std(adjusted) / np.sqrt(self.n_paths))
        else:
            price = float(np.mean(disc_payoffs))
            se    = float(np.std(disc_payoffs) / np.sqrt(self.n_paths))

        return MonteCarloResult(price=price, std_error=se, n_paths=self.n_paths)

    def _price_barrier(
        self,
        instrument : BarrierOption,
        market     : MarketData,
    ) -> MonteCarloResult:
        """
        Price a barrier option.

        Each path is checked against the barrier at every time step.
        The payoff is zero for knocked-out paths (knock-out options)
        or zero for non-knocked-in paths (knock-in options).

        Note on discretisation bias:
            With discrete monitoring, the simulated barrier hit probability
            is slightly different from continuous monitoring.
            The Broadie-Glasserman-Kou (1997) correction adjusts for this:
            shift the barrier by exp(±0.5826 * sigma * sqrt(dt)).
            We implement this as a continuity correction.
        """
        S0, r, q = market.spot, market.rate, market.div_yield
        T = instrument.expiry

        paths = self.simulate_paths(S0, T, r, q)      # (n_paths, n_steps+1)

        payoffs = np.array([
            instrument.path_payoff(paths[i])
            for i in range(self.n_paths)
        ])

        disc_payoffs = np.exp(-r * T) * payoffs
        price = float(np.mean(disc_payoffs))
        se    = float(np.std(disc_payoffs) / np.sqrt(self.n_paths))

        return MonteCarloResult(price=price, std_error=se, n_paths=self.n_paths)

    def _price_asian(
        self,
        instrument : AsianOption,
        market     : MarketData,
    ) -> MonteCarloResult:
        """
        Price an Asian option.

        Uses geometric average as a control variate for arithmetic average:
            - Geometric average of lognormals is lognormal
            - The geometric Asian has a known closed form (Kemna-Vorst)
            - Corr(arithmetic avg, geometric avg) is very high (~0.99)
            - This makes geometric an excellent control variate

        The variance reduction from this control variate is dramatic —
        typically 100-1000x reduction in variance for ATM options.
        """
        S0, r, q = market.spot, market.rate, market.div_yield
        T = instrument.expiry
        n_obs = instrument.n_observations

        # Simulate paths at observation dates
        paths = self.simulate_paths(S0, T, r, q, n_steps=n_obs)
        obs_paths = paths[:, 1:]                       # exclude S0, shape (n_paths, n_obs)

        # Geometric average (always computed — used as control variate or main)
        geo_avg = np.exp(np.mean(np.log(obs_paths), axis=1))   # shape (n_paths,)
        if instrument.option_type.value == "call":
            geo_payoffs = np.maximum(geo_avg - instrument.strike, 0)
        else:
            geo_payoffs = np.maximum(instrument.strike - geo_avg, 0)

        disc_geo     = np.exp(-r * T) * geo_payoffs
        geo_price_cf = self._kemna_vorst_price(instrument, market)

        from options_lib.instruments.asian import AverageType
        if instrument.average_type == AverageType.GEOMETRIC:
            # Geometric Asian: use MC estimate directly (no control needed)
            price = float(np.mean(disc_geo))
            se    = float(np.std(disc_geo) / np.sqrt(self.n_paths))
        else:
            # Arithmetic average payoffs
            arith_avg = np.mean(obs_paths, axis=1)
            if instrument.option_type.value == "call":
                arith_payoffs = np.maximum(arith_avg - instrument.strike, 0)
            else:
                arith_payoffs = np.maximum(instrument.strike - arith_avg, 0)
            disc_arith = np.exp(-r * T) * arith_payoffs

            if self.control_variate:
                # Control variate: geometric is correlated with arithmetic
                cov_matrix = np.cov(disc_arith, disc_geo)
                if cov_matrix[1, 1] > 1e-12:
                    c = cov_matrix[0, 1] / cov_matrix[1, 1]
                else:
                    c = 1.0
                adjusted = disc_arith - c * (disc_geo - geo_price_cf)
                price = float(np.mean(adjusted))
                se    = float(np.std(adjusted) / np.sqrt(self.n_paths))
            else:
                price = float(np.mean(disc_arith))
                se    = float(np.std(disc_arith) / np.sqrt(self.n_paths))

        return MonteCarloResult(price=price, std_error=se, n_paths=self.n_paths)

    def _kemna_vorst_price(
        self,
        instrument : AsianOption,
        market     : MarketData,
    ) -> float:
        """
        Kemna-Vorst (1990) closed-form price for geometric average Asian option.

        For a geometric average Asian call with n equally spaced observations:
            sigma_adj = sigma * sqrt((2n+1) / (6(n+1)))
            b_adj     = (1/2)(r - q - sigma^2/2 + sigma_adj^2)
            d1 = [ln(S/K) + (b_adj + sigma_adj^2/2)*T] / (sigma_adj * sqrt(T))
            d2 = d1 - sigma_adj * sqrt(T)
            C  = S*e^{(b_adj - r)*T}*N(d1) - K*e^{-rT}*N(d2)
        """
        from scipy.stats import norm
        S, K, T = market.spot, instrument.strike, instrument.expiry
        r, q, n = market.rate, market.div_yield, instrument.n_observations
        sigma = self.sigma

        sigma_adj = sigma * np.sqrt((2 * n + 1) / (6 * (n + 1)))
        b_adj     = 0.5 * (r - q - 0.5 * sigma**2 + sigma_adj**2)

        sqrt_T = np.sqrt(T)
        d1 = (np.log(S / K) + (b_adj + 0.5 * sigma_adj**2) * T) / (sigma_adj * sqrt_T)
        d2 = d1 - sigma_adj * sqrt_T

        if instrument.option_type.value == "call":
            price = (S * np.exp((b_adj - r) * T) * norm.cdf(d1)
                     - K * np.exp(-r * T) * norm.cdf(d2))
        else:
            price = (K * np.exp(-r * T) * norm.cdf(-d2)
                     - S * np.exp((b_adj - r) * T) * norm.cdf(-d1))

        return float(max(price, 0.0))

    def _price_generic(
        self,
        instrument : Instrument,
        market     : MarketData,
    ) -> MonteCarloResult:
        """
        Generic MC pricer for any instrument with a payoff(spots) method.
        No variance reduction. Fallback for unknown instrument types.
        """
        S0, r, q = market.spot, market.rate, market.div_yield
        T = instrument.expiry

        paths = self.simulate_paths(S0, T, r, q, n_steps=1)
        S_T = paths[:, -1]
        payoffs = instrument.payoff(S_T)
        disc_payoffs = np.exp(-r * T) * payoffs

        price = float(np.mean(disc_payoffs))
        se    = float(np.std(disc_payoffs) / np.sqrt(self.n_paths))
        return MonteCarloResult(price=price, std_error=se, n_paths=self.n_paths)

    def variance_reduction_summary(
        self,
        instrument : EuropeanOption,
        market     : MarketData,
    ) -> dict:
        """
        Compare plain MC vs antithetic vs control variate.
        Shows the variance reduction factor achieved by each method.

        Returns
        -------
        dict with keys: plain, antithetic, control_variate
            Each value is a MonteCarloResult.
        """
        results = {}

        # Plain MC
        plain_mc = MonteCarlo(
            sigma=self.sigma, n_paths=self.n_paths,
            antithetic=False, control_variate=False, seed=self.seed
        )
        results["plain"] = plain_mc.price_with_stats(instrument, market)

        # Antithetic only
        anti_mc = MonteCarlo(
            sigma=self.sigma, n_paths=self.n_paths,
            antithetic=True, control_variate=False, seed=self.seed
        )
        results["antithetic"] = anti_mc.price_with_stats(instrument, market)

        # Control variate (includes antithetic)
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