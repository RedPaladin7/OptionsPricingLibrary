"""
models/heston.py
----------------
Heston (1993) stochastic volatility model.

The Heston model extends Black-Scholes by making volatility stochastic:

    dS_t = (r - q) S_t dt + √v_t S_t dW_t^S          (stock)
    dv_t = κ(v̄ - v_t) dt + ξ √v_t dW_t^v             (variance)
    dW_t^S dW_t^v = ρ dt                               (correlation)

Parameters
----------
v0    : float  Initial variance (σ₀² e.g. 0.04 = 20% vol)
kappa : float  Mean reversion speed κ (how fast vol reverts to v̄)
v_bar : float  Long-run variance v̄ (long-run vol² e.g. 0.04)
xi    : float  Vol of vol ξ (volatility of the variance process)
rho   : float  Spot-vol correlation ρ ∈ (-1, 1)

Key properties
--------------
- Feller condition: 2κv̄ > ξ²  ensures variance stays strictly positive
- ρ < 0 (typical for equities): generates negative skew — large down
  moves in S coincide with vol spikes, making OTM puts expensive
- ξ large: generates a pronounced smile (fat tails)
- κ large: vol mean-reverts quickly, flattening the term structure

Pricing
-------
The Heston model has a semi-closed-form solution via the characteristic
function of ln(S_T). We price via the Carr-Madan FFT method:
    1. Evaluate the Heston characteristic function φ(u) analytically
    2. Feed φ into the FFT pricer to get call prices at all strikes

Calibration
-----------
Given market option prices {C_i^mkt} at strikes {K_i} and expiries {T_i},
we find model parameters that minimise the sum of squared IV errors:
    min_{v0,κ,v̄,ξ,ρ}  Σ_i (σ_imp^model(K_i,T_i) - σ_imp^mkt(K_i,T_i))²

References
----------
Heston, S.L. (1993). "A closed-form solution for options with stochastic
volatility with applications to bond and currency options."
Review of Financial Studies, 6(2), 327-343.
"""

import numpy as np
from scipy.optimize import minimize, Bounds
from scipy.interpolate import interp1d
from dataclasses import dataclass, field
from typing import Optional

from options_lib.models.base import Model
from options_lib.models.black_scholes import BlackScholes
from options_lib.models.implied_vol import implied_vol_bs
from options_lib.instruments.base import Instrument, MarketData, OptionType
from options_lib.instruments.european import EuropeanOption
from options_lib.numerics.fft import carr_madan_fft, interpolate_call_price


@dataclass
class HestonParams:
    """
    Heston model parameters.

    Attributes
    ----------
    v0    : Initial variance v_0. Note: this is variance, not vol.
            e.g. v0=0.04 corresponds to initial vol of 20%.
    kappa : Mean reversion speed κ > 0.
    v_bar : Long-run variance v̄ > 0. Long-run vol = √v̄.
    xi    : Vol of vol ξ > 0.
    rho   : Spot-vol correlation ρ ∈ (-1, 1).
            Typically negative for equity indices (leverage effect).

    Properties
    ----------
    feller_satisfied : bool
        True if 2κv̄ > ξ² (variance process stays strictly positive).
        If False, variance can hit zero and the model may misbehave.
    """
    v0    : float
    kappa : float
    v_bar : float
    xi    : float
    rho   : float

    def __post_init__(self):
        if self.v0 <= 0:
            raise ValueError(f"v0 must be positive, got {self.v0}")
        if self.kappa <= 0:
            raise ValueError(f"kappa must be positive, got {self.kappa}")
        if self.v_bar <= 0:
            raise ValueError(f"v_bar must be positive, got {self.v_bar}")
        if self.xi <= 0:
            raise ValueError(f"xi must be positive, got {self.xi}")
        if not -1 < self.rho < 1:
            raise ValueError(f"rho must be in (-1, 1), got {self.rho}")

    @property
    def feller_satisfied(self) -> bool:
        """2κv̄ > ξ²: variance process stays strictly positive."""
        return 2 * self.kappa * self.v_bar > self.xi ** 2

    @property
    def initial_vol(self) -> float:
        """Initial volatility (square root of initial variance)."""
        return np.sqrt(self.v0)

    @property
    def long_run_vol(self) -> float:
        """Long-run volatility (square root of long-run variance)."""
        return np.sqrt(self.v_bar)

    def __repr__(self) -> str:
        feller = "✓" if self.feller_satisfied else "✗"
        return (
            f"HestonParams("
            f"v0={self.v0:.4f} [vol={self.initial_vol:.1%}], "
            f"kappa={self.kappa:.3f}, "
            f"v_bar={self.v_bar:.4f} [vol={self.long_run_vol:.1%}], "
            f"xi={self.xi:.3f}, "
            f"rho={self.rho:.3f}, "
            f"Feller={feller})"
        )


@dataclass
class Heston(Model):
    """
    Heston stochastic volatility model pricer.

    Uses the Heston characteristic function with Carr-Madan FFT
    to price European options.

    Parameters
    ----------
    params : HestonParams
        The 5 Heston model parameters.
    alpha : float
        FFT dampening factor. Default 1.5.
    N : int
        FFT grid size. Default 4096.
    eta : float
        FFT frequency grid spacing. Default 0.25.

    Examples
    --------
    >>> params = HestonParams(v0=0.04, kappa=2.0, v_bar=0.04, xi=0.3, rho=-0.7)
    >>> model  = Heston(params)
    >>> call   = EuropeanOption(strike=100, expiry=1.0, option_type=OptionType.CALL)
    >>> mkt    = MarketData(spot=100, rate=0.05)
    >>> model.price(call, mkt)
    ~10.3...  (slightly different from BS due to stochastic vol)
    """

    params : HestonParams
    alpha  : float = 1.5
    N      : int   = 4096
    eta    : float = 0.25

    # ------------------------------------------------------------------
    # Characteristic Function
    # ------------------------------------------------------------------

    def characteristic_function(
        self,
        u  : np.ndarray,
        T  : float,
        S  : float,
        r  : float,
        q  : float,
    ) -> np.ndarray:
        """
        Heston characteristic function φ(u) of ln(S_T) under Q.

        Derivation sketch
        -----------------
        The Heston PDE for the characteristic function has affine structure:
            φ(u, t) = exp(A(u, τ) + B(u, τ) v_t + iu ln(S_t))

        where τ = T - t and A, B satisfy Riccati ODEs:
            dB/dτ = ½(-u² - iu) + (ρξiu - κ)B + ½ξ²B²
            dA/dτ = κv̄ B + (r-q)iu - ½ξ²... (integrated separately)

        The Riccati ODE for B is solved analytically, giving:

            d  = √((ρξiu - κ)² + ξ²(u² + iu))
            g  = (κ - ρξiu - d) / (κ - ρξiu + d)

            B(τ) = (κ - ρξiu - d)(1 - e^{-dτ}) / (ξ²(1 - g e^{-dτ}))

            A(τ) = (r-q)iuτ + (κv̄/ξ²)[(κ - ρξiu - d)τ - 2ln((1-ge^{-dτ})/(1-g))]

        Then: φ(u) = exp(A(τ) + B(τ)v₀ + iu ln(S))

        Parameters
        ----------
        u : np.ndarray (complex)
            Frequency argument. Can be complex (for Carr-Madan, the input
            is u - (α+1)i which is complex).
        T : float
            Time to expiry.
        S : float
            Current spot price.
        r, q : float
            Risk-free rate and dividend yield.

        Returns
        -------
        np.ndarray (complex)
            φ(u) — the characteristic function evaluated at each u.

        Notes on numerical stability
        ----------------------------
        The "original" Heston formula has a discontinuity in the complex
        log (branch cut issue) for long maturities or extreme parameters.
        We use the "Little Heston Trap" formulation (Albrecher et al. 2007)
        which avoids this by reformulating g to keep the argument of the
        complex log away from the branch cut.
        """
        kappa = self.params.kappa
        v_bar = self.params.v_bar
        xi    = self.params.xi
        rho   = self.params.rho
        v0    = self.params.v0

        # ----------------------------------------------------------
        # Standard Heston characteristic function (Gatheral 2006)
        # Uses e^{-dτ} instead of e^{dτ} to keep magnitudes bounded.
        # ----------------------------------------------------------

        # d = √((κ - ρξiu)² + ξ²(iu + u²))
        d = np.sqrt(
            (kappa - rho * xi * 1j * u) ** 2
            + xi**2 * (u**2 + 1j * u)
        )

        # g = (κ - ρξiu - d) / (κ - ρξiu + d)
        numer_g = kappa - rho * xi * 1j * u - d
        denom_g = kappa - rho * xi * 1j * u + d
        g = numer_g / denom_g

        # B(τ) = (κ - ρξiu - d)(1 - e^{-dτ}) / (ξ²(1 - g e^{-dτ}))
        exp_neg_dT = np.exp(-d * T)
        B = (numer_g * (1 - exp_neg_dT)
             / (xi**2 * (1 - g * exp_neg_dT)))

        # A(τ) = (r-q)iuτ + (κv̄/ξ²)[(κ - ρξiu - d)τ - 2ln((1-ge^{-dτ})/(1-g))]
        log_term = np.log((1 - g * exp_neg_dT) / (1 - g))
        A = ((r - q) * 1j * u * T
             + (kappa * v_bar / xi**2)
             * (numer_g * T - 2 * log_term))

        # φ(u) = exp(A + B*v0 + iu*ln(S))
        phi = np.exp(A + B * v0 + 1j * u * np.log(S))

        return phi

    # ------------------------------------------------------------------
    # Pricing
    # ------------------------------------------------------------------

    def price(self, instrument: Instrument, market: MarketData) -> float:
        """
        Price a European option using the Heston characteristic function
        and Carr-Madan FFT.

        For puts: use put-call parity after pricing the call.
            P = C - S*e^{-qT} + K*e^{-rT}

        Parameters
        ----------
        instrument : EuropeanOption
        market : MarketData

        Returns
        -------
        float
            Heston model price.
        """
        if not isinstance(instrument, EuropeanOption):
            raise NotImplementedError(
                "Heston analytical pricer only supports EuropeanOption. "
                "Use MonteCarlo for path-dependent instruments."
            )

        S = market.spot
        K = instrument.strike
        T = instrument.expiry
        r = market.rate
        q = market.div_yield

        # Bind all parameters except u into the characteristic function
        def char_fn(u):
            return self.characteristic_function(u, T, S, r, q)

        # Run FFT — get call prices at a grid of strikes
        strikes, call_prices = carr_madan_fft(
            char_fn=char_fn,
            S=S, T=T, r=r, q=q,
            alpha=self.alpha,
            N=self.N,
            eta=self.eta,
        )

        # Interpolate at the target strike
        call_price = interpolate_call_price(strikes, call_prices, K)

        # For puts, use put-call parity
        if instrument.option_type == OptionType.PUT:
            put_price = call_price - S * np.exp(-q * T) + K * np.exp(-r * T)
            return float(max(put_price, 0.0))

        return float(max(call_price, 0.0))

    def price_smile(
        self,
        strikes    : np.ndarray,
        expiry     : float,
        market     : MarketData,
        option_type: OptionType = OptionType.CALL
    ) -> np.ndarray:
        """
        Price calls (or puts) at multiple strikes in one FFT call.

        This is much faster than calling price() in a loop because
        the FFT prices ALL strikes simultaneously.

        Parameters
        ----------
        strikes : np.ndarray
            Array of strike prices.
        expiry : float
            Time to expiry (same for all strikes).
        market : MarketData
        option_type : OptionType

        Returns
        -------
        np.ndarray
            Prices at each strike.
        """
        S = market.spot
        T = expiry
        r = market.rate
        q = market.div_yield

        def char_fn(u):
            return self.characteristic_function(u, T, S, r, q)

        fft_strikes, fft_prices = carr_madan_fft(
            char_fn=char_fn,
            S=S, T=T, r=r, q=q,
            alpha=self.alpha,
            N=self.N,
            eta=self.eta,
        )

        # Interpolate at all target strikes at once
        call_prices = np.interp(strikes, fft_strikes, fft_prices)
        call_prices = np.maximum(call_prices, 0.0)

        if option_type == OptionType.PUT:
            put_prices = call_prices - S * np.exp(-q * T) + strikes * np.exp(-r * T)
            return np.maximum(put_prices, 0.0)

        return call_prices

    def implied_vol_smile(
        self,
        strikes : np.ndarray,
        expiry  : float,
        market  : MarketData,
    ) -> np.ndarray:
        """
        Compute the implied volatility smile produced by the Heston model.

        Steps:
          1. Price calls at all strikes via FFT (one shot)
          2. Invert each price to get BS implied vol

        This is the key diagnostic: if calibrated to market prices, the
        Heston IV smile should match the market IV smile.

        Parameters
        ----------
        strikes : np.ndarray
        expiry  : float
        market  : MarketData

        Returns
        -------
        np.ndarray
            Implied volatilities at each strike. NaN where inversion fails
            (e.g., deep OTM options with near-zero price).
        """
        call_prices = self.price_smile(strikes, expiry, market, OptionType.CALL)

        iv_smile = np.full(len(strikes), np.nan)
        for i, (K, C) in enumerate(zip(strikes, call_prices)):
            try:
                opt = EuropeanOption(strike=K, expiry=expiry, option_type=OptionType.CALL)
                iv_smile[i] = implied_vol_bs(
                    market_price=C,
                    instrument=opt,
                    market=market,
                    sigma_init=np.sqrt(self.params.v0)
                )
            except Exception:
                pass  # Leave as NaN for problematic strikes

        return iv_smile

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def calibrate(
        self,
        market_strikes  : np.ndarray,
        market_expiries : np.ndarray,
        market_ivs      : np.ndarray,
        market_data     : MarketData,
        initial_params  : Optional[HestonParams] = None,
        verbose         : bool = False,
    ) -> "Heston":
        """
        Calibrate Heston parameters to market implied volatilities.

        Minimises the root mean squared error between model IVs and
        market IVs across all (strike, expiry) pairs.

        Objective:
            min_{v0,κ,v̄,ξ,ρ}  √[ (1/N) Σ_i (σ_model(K_i,T_i) - σ_mkt_i)² ]

        We minimise in IV space (not price space) because:
          - IVs are roughly the same order of magnitude across strikes/expiries
          - Price differences are dominated by ATM options (high vega)
          - IV errors treat OTM and ITM options equally

        Parameters
        ----------
        market_strikes : np.ndarray, shape (N,)
            Strike prices of market options.
        market_expiries : np.ndarray, shape (N,)
            Expiries (in years) of each market option.
        market_ivs : np.ndarray, shape (N,)
            Market implied volatilities (as decimals, e.g. 0.20 = 20%).
        market_data : MarketData
            Current spot, rate, dividend yield.
        initial_params : HestonParams, optional
            Starting point for optimisation. If None, uses a reasonable
            default (ATM BS vol ≈ mean market IV).
        verbose : bool
            If True, print calibration progress.

        Returns
        -------
        Heston
            A new Heston instance with calibrated parameters.
            (Does not modify self — returns a new object.)

        Notes
        -----
        Optimiser: L-BFGS-B (gradient-based, handles box constraints).
        Parameter bounds:
            v0    ∈ (1e-4, 1.0)     variance
            kappa ∈ (0.1,  10.0)    mean reversion
            v_bar ∈ (1e-4, 1.0)     long-run variance
            xi    ∈ (0.01, 2.0)     vol of vol
            rho   ∈ (-0.99, 0.99)   correlation
        """
        if initial_params is None:
            # Use mean market IV squared as initial variance guess
            mean_iv = float(np.nanmean(market_ivs))
            initial_params = HestonParams(
                v0    = mean_iv**2,
                kappa = 2.0,
                v_bar = mean_iv**2,
                xi    = 0.3,
                rho   = -0.7
            )

        # Pack parameters into a vector for the optimiser
        x0 = np.array([
            initial_params.v0,
            initial_params.kappa,
            initial_params.v_bar,
            initial_params.xi,
            initial_params.rho,
        ])

        # Parameter bounds
        bounds = Bounds(
            lb=[1e-4,  0.1,  1e-4, 0.01, -0.99],
            ub=[1.0,  10.0,  1.0,  2.0,   0.99],
        )

        iteration = [0]

        def objective(x):
            v0, kappa, v_bar, xi, rho = x
            try:
                params = HestonParams(v0=v0, kappa=kappa, v_bar=v_bar, xi=xi, rho=rho)
                model  = Heston(params, alpha=self.alpha, N=self.N, eta=self.eta)
            except ValueError:
                return 1e6  # Invalid parameters

            # Compute model IVs at all market (strike, expiry) pairs
            errors = []
            unique_expiries = np.unique(market_expiries)

            for T in unique_expiries:
                mask    = market_expiries == T
                strikes = market_strikes[mask]
                mkt_ivs = market_ivs[mask]

                try:
                    model_ivs = model.implied_vol_smile(strikes, T, market_data)
                    valid = ~np.isnan(model_ivs)
                    if valid.any():
                        errors.extend((model_ivs[valid] - mkt_ivs[valid]) ** 2)
                except Exception:
                    errors.append(1.0)  # Penalise failures

            if not errors:
                return 1e6

            rmse = np.sqrt(np.mean(errors))

            iteration[0] += 1
            if verbose and iteration[0] % 20 == 0:
                print(f"  Iter {iteration[0]:4d}: RMSE = {rmse*100:.4f}%  "
                      f"params = v0={v0:.4f}, κ={kappa:.3f}, "
                      f"v̄={v_bar:.4f}, ξ={xi:.3f}, ρ={rho:.3f}")

            return float(rmse)

        if verbose:
            print("Starting Heston calibration...")
            print(f"  {len(market_strikes)} market quotes, "
                  f"{len(np.unique(market_expiries))} expiries")

        result = minimize(
            objective,
            x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8}
        )

        v0, kappa, v_bar, xi, rho = result.x
        calibrated_params = HestonParams(
            v0=v0, kappa=kappa, v_bar=v_bar, xi=xi, rho=rho
        )

        if verbose:
            print(f"\nCalibration {'converged' if result.success else 'did not converge'}.")
            print(f"  Final RMSE: {result.fun * 100:.4f}%")
            print(f"  {calibrated_params}")

        return Heston(calibrated_params, alpha=self.alpha, N=self.N, eta=self.eta)

    def __repr__(self) -> str:
        return f"Heston({self.params})"