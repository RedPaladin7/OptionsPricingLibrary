"""
market_data/vol_surface.py
---------------------------
Volatility surface construction using the SVI parametrisation.

The Problem
-----------
Market option prices imply different BS vols at different strikes and
expiries — the "vol surface" σ(K, T). We need to:
  1. Fit a smooth, arbitrage-free parametric surface to market IV data
  2. Interpolate/extrapolate at any (K, T) pair
  3. Use it to price any option consistently with market vanillas

Why SVI (Stochastic Volatility Inspired)?
-----------------------------------------
Gatheral (2004) proposed fitting each expiry slice with:

    w(k) = a + b[ρ(k-m) + √((k-m)² + σ²)]

where:
  k = ln(K/F) is the log-moneyness (F = forward price)
  w = σ²T is the total implied variance (NOT vol — variance × time)
  a = overall level of variance
  b = slope parameter (controls ATM vol level)
  ρ = skew parameter (-1 < ρ < 1, typically negative for equities)
  m = ATM point (the log-moneyness at minimum variance)
  σ = curvature (controls smile width / convexity)

SVI is popular because:
  - 5 parameters, interpretable, calibrates quickly
  - Generates realistic smiles (negative skew from ρ, smile from σ)
  - Has known closed-form arbitrage conditions
  - Used heavily in practice by vol desks

Arbitrage Conditions
--------------------
A vol surface must satisfy two no-arbitrage conditions:

1. Calendar spread arbitrage (static arbitrage):
   Total variance w(k,T) must be non-decreasing in T for fixed k.
   Violation means: a longer-dated option is cheaper than a shorter-dated
   one at the same log-moneyness → arbitrage (sell short, buy long).

2. Butterfly arbitrage (dynamic arbitrage):
   The risk-neutral density implied by the surface must be positive:
   g(k) = (1 - k*d1/(2*w) - d2²/(4*w) + d2/(2*w) + d1²/(4*w)) >= 0
   Violation means negative risk-neutral probabilities → arbitrage.

   In practice, this means the smile cannot be too steep or convex.

References
----------
Gatheral, J. (2004). A parsimonious arbitrage-free implied volatility
parametrization with application to the valuation of volatility derivatives.
Presentation at Global Derivatives, Madrid.

Gatheral, J. and Jacquier, A. (2014). Arbitrage-free SVI volatility surfaces.
Quantitative Finance, 14(1), 59-71.
"""

import numpy as np
from scipy.optimize import minimize, Bounds
from scipy.interpolate import interp1d
from dataclasses import dataclass, field
from typing import Optional
import warnings


# ------------------------------------------------------------------
# SVI Parametrisation
# ------------------------------------------------------------------

@dataclass
class SVIParams:
    """
    SVI parameters for one expiry slice.

    The SVI raw parametrisation:
        w(k) = a + b[ρ(k-m) + √((k-m)² + σ²)]

    where w = total implied variance = σ_BS² × T.

    Attributes
    ----------
    a   : float  Level shift. Must satisfy a + b*σ*√(1-ρ²) >= 0.
    b   : float  Slope/curvature. b >= 0.
    rho : float  Skew. -1 < ρ < 1.
    m   : float  ATM shift (log-moneyness at minimum variance).
    sigma : float  Curvature (smile width). σ > 0.
    expiry : float  Time to expiry T (for converting w to vol).
    """
    a      : float
    b      : float
    rho    : float
    m      : float
    sigma  : float
    expiry : float

    def total_variance(self, k: np.ndarray) -> np.ndarray:
        """
        w(k) = a + b[ρ(k-m) + √((k-m)² + σ²)]

        Parameters
        ----------
        k : np.ndarray
            Log-moneyness = ln(K/F).

        Returns
        -------
        np.ndarray
            Total implied variance w = σ_BS² × T. Must be >= 0.
        """
        k  = np.asarray(k, dtype=float)
        d  = k - self.m
        w  = self.a + self.b * (self.rho * d + np.sqrt(d**2 + self.sigma**2))
        return np.maximum(w, 0.0)   # floor at 0 for numerical safety

    def implied_vol(self, k: np.ndarray) -> np.ndarray:
        """
        BS implied vol from SVI total variance.
        σ_BS(k) = √(w(k) / T)
        """
        w = self.total_variance(k)
        return np.sqrt(np.maximum(w / self.expiry, 1e-8))

    def implied_vol_from_strike(
        self, K: np.ndarray, F: float
    ) -> np.ndarray:
        """
        Implied vol as a function of strike K and forward F.
        k = ln(K/F).
        """
        K = np.asarray(K, dtype=float)
        k = np.log(K / F)
        return self.implied_vol(k)

    def butterfly_density(self, k: np.ndarray) -> np.ndarray:
        """
        Compute the Lee (2004) density proxy g(k).
        g(k) >= 0 is required for no butterfly arbitrage.

        g(k) = (1 - k*d1/(2w))² - d1²/4*(1/w + 1/4) + d2²/2

        where d1 = -k/√w + √w/2, d2 = -k/√w - √w/2  (Breeden-Litzenberger style)
        Simplified form from Gatheral & Jacquier (2014).
        """
        k = np.asarray(k, dtype=float)
        w = self.total_variance(k)
        w = np.maximum(w, 1e-8)

        # SVI derivative dw/dk
        d    = k - self.m
        denom = np.sqrt(d**2 + self.sigma**2)
        dwdk = self.b * (self.rho + d / denom)

        # Second derivative d²w/dk²
        d2wdk2 = self.b * self.sigma**2 / (denom**3)

        # Density proxy (from Gatheral & Jacquier eq 2.3)
        g = (1 - k * dwdk / (2 * w))**2 \
            - (dwdk**2 / 4) * (1/w + 1/4) \
            + d2wdk2 / 2

        return g

    def is_butterfly_free(self, k_grid: Optional[np.ndarray] = None) -> bool:
        """
        Check butterfly arbitrage over a grid of log-moneynesses.
        Returns True if g(k) >= 0 everywhere on the grid.
        """
        if k_grid is None:
            k_grid = np.linspace(-1.5, 1.5, 200)
        g = self.butterfly_density(k_grid)
        return bool(np.all(g >= -1e-6))

    def __repr__(self) -> str:
        iv_atm = float(self.implied_vol(np.array([self.m])))
        return (
            f"SVIParams(T={self.expiry:.3f}, "
            f"a={self.a:.4f}, b={self.b:.4f}, ρ={self.rho:.3f}, "
            f"m={self.m:.3f}, σ={self.sigma:.4f} | "
            f"ATM_vol={iv_atm:.1%})"
        )


# ------------------------------------------------------------------
# SVI Slice Calibration
# ------------------------------------------------------------------

def calibrate_svi_slice(
    log_moneyness : np.ndarray,
    market_ivs    : np.ndarray,
    expiry        : float,
    weights       : Optional[np.ndarray] = None,
    n_restarts    : int = 5,
) -> SVIParams:
    """
    Calibrate SVI parameters to one expiry slice of market IVs.

    Minimises weighted RMSE between SVI implied vols and market IVs.

    Parameters
    ----------
    log_moneyness : np.ndarray
        k = ln(K/F) for each option in this slice.
    market_ivs : np.ndarray
        Market implied volatilities (decimals) for each option.
    expiry : float
        Time to expiry T in years.
    weights : np.ndarray, optional
        Per-observation weights. Default: uniform.
        In practice, weight by vega (ATM options more important).
    n_restarts : int
        Number of random restarts to avoid local minima.

    Returns
    -------
    SVIParams
        Calibrated SVI parameters for this expiry slice.

    Notes
    -----
    We calibrate in total variance space (w = σ²T), not vol space.
    This is more numerically stable and the SVI formula is designed for it.

    The objective is RMSE in implied vol space (not w space) because
    that's what traders care about — vol differences in bp.
    """
    if len(log_moneyness) < 5:
        raise ValueError(
            f"Need at least 5 quotes to calibrate SVI, got {len(log_moneyness)}."
        )

    # Convert to total variance
    w_market = market_ivs**2 * expiry   # shape (N,)

    if weights is None:
        weights = np.ones(len(log_moneyness))
    weights = weights / weights.sum()

    def objective(params):
        a, b, rho, m, sigma = params
        if b <= 0 or sigma <= 0 or abs(rho) >= 1:
            return 1e6
        # SVI total variance
        d    = log_moneyness - m
        w    = a + b * (rho * d + np.sqrt(d**2 + sigma**2))
        if np.any(w <= 0):
            return 1e6
        # RMSE in vol space
        iv_model  = np.sqrt(w / expiry)
        iv_market = np.sqrt(np.maximum(w_market / expiry, 0))
        rmse = np.sqrt(np.sum(weights * (iv_model - iv_market)**2))
        return float(rmse)

    # SVI constraints:
    # b >= 0
    # -1 < rho < 1
    # sigma > 0
    # a + b*sigma*sqrt(1-rho^2) >= 0  (positivity)
    bounds = Bounds(
        lb=[-0.5,  1e-4, -0.99, -1.0,  1e-4],
        ub=[ 1.0,  2.0,   0.99,  1.0,  2.0 ],
    )

    # ATM vol as initial guess for a
    atm_iv  = float(np.interp(0, log_moneyness, market_ivs))
    w_atm   = atm_iv**2 * expiry

    best_result = None
    best_val    = np.inf

    # Multiple random restarts
    np.random.seed(0)
    initial_guesses = [
        [w_atm * 0.8, 0.1, -0.3, 0.0, 0.1],  # typical starting point
    ] + [
        [
            w_atm * np.random.uniform(0.5, 1.5),
            np.random.uniform(0.01, 0.5),
            np.random.uniform(-0.8, 0.2),
            np.random.uniform(-0.2, 0.2),
            np.random.uniform(0.01, 0.5),
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
                best_val    = result.fun
                best_result = result
        except Exception:
            continue

    if best_result is None:
        raise RuntimeError("SVI calibration failed for all restarts.")

    a, b, rho, m, sigma = best_result.x
    return SVIParams(a=a, b=b, rho=rho, m=m, sigma=sigma, expiry=expiry)


# ------------------------------------------------------------------
# Full Vol Surface
# ------------------------------------------------------------------

@dataclass
class VolSurface:
    """
    Calibrated implied volatility surface.

    Holds one SVIParams per expiry slice and provides:
      - IV interpolation at any (K, T)
      - Arbitrage checks across the entire surface
      - Total variance surface w(k, T)

    Parameters
    ----------
    svi_slices : dict
        {expiry_date_str: SVIParams} for each calibrated slice.
    forwards : dict
        {expiry_date_str: forward_price} per expiry.
    spot : float
        Underlying spot price.
    rate : float
        Risk-free rate.
    """
    svi_slices : dict         # {expiry_str: SVIParams}
    forwards   : dict         # {expiry_str: forward}
    spot       : float
    rate       : float
    ticker     : str = ""

    @property
    def expiry_dates(self) -> list:
        return sorted(self.svi_slices.keys())

    @property
    def expiries(self) -> np.ndarray:
        return np.array([self.svi_slices[d].expiry for d in self.expiry_dates])

    def implied_vol(self, K: float, T: float) -> float:
        """
        Interpolated implied vol at strike K and expiry T.

        Interpolation in total variance space (linear in T for fixed k),
        then convert back to vol. This ensures calendar spread arbitrage
        is preserved across the interpolated surface.

        Parameters
        ----------
        K : float   Strike price.
        T : float   Time to expiry in years.

        Returns
        -------
        float   Implied volatility.
        """
        expiries = self.expiries
        dates    = self.expiry_dates

        if T <= expiries[0]:
            # Extrapolate flat to the nearest slice
            svi = self.svi_slices[dates[0]]
            F   = self.forwards.get(dates[0], self.spot * np.exp(self.rate * T))
            k   = np.log(K / F)
            return float(svi.implied_vol(np.array([k]))[0])

        if T >= expiries[-1]:
            svi = self.svi_slices[dates[-1]]
            F   = self.forwards.get(dates[-1], self.spot * np.exp(self.rate * T))
            k   = np.log(K / F)
            return float(svi.implied_vol(np.array([k]))[0])

        # Find surrounding slices
        idx  = np.searchsorted(expiries, T)
        T_lo, T_hi = expiries[idx-1], expiries[idx]
        d_lo, d_hi = dates[idx-1],    dates[idx]

        F_lo = self.forwards.get(d_lo, self.spot * np.exp(self.rate * T_lo))
        F_hi = self.forwards.get(d_hi, self.spot * np.exp(self.rate * T_hi))

        k_lo = np.log(K / F_lo)
        k_hi = np.log(K / F_hi)

        w_lo = float(self.svi_slices[d_lo].total_variance(np.array([k_lo]))[0])
        w_hi = float(self.svi_slices[d_hi].total_variance(np.array([k_hi]))[0])

        # Linear interpolation of total variance
        alpha = (T - T_lo) / (T_hi - T_lo)
        w     = (1 - alpha) * w_lo + alpha * w_hi
        return float(np.sqrt(max(w / T, 1e-8)))

    def implied_vol_surface(
        self,
        strikes  : np.ndarray,
        expiries : np.ndarray,
    ) -> np.ndarray:
        """
        Compute IV surface on a (expiries × strikes) grid.

        Returns
        -------
        np.ndarray, shape (len(expiries), len(strikes))
        """
        surface = np.zeros((len(expiries), len(strikes)))
        for i, T in enumerate(expiries):
            for j, K in enumerate(strikes):
                try:
                    surface[i, j] = self.implied_vol(K, T)
                except Exception:
                    surface[i, j] = np.nan
        return surface

    def check_calendar_arbitrage(self) -> dict:
        """
        Check calendar spread arbitrage across all slices.

        For each pair of consecutive expiries (T1, T2) with T1 < T2:
        Total variance w(k, T2) >= w(k, T1) for all k.
        Otherwise, the longer-dated option is cheaper → arbitrage.

        Returns
        -------
        dict with keys:
            'is_arbitrage_free' : bool
            'violations'        : list of (T1, T2, k, w1, w2) tuples
        """
        k_grid     = np.linspace(-1.0, 1.0, 100)
        dates      = self.expiry_dates
        violations = []

        for i in range(len(dates) - 1):
            d1, d2   = dates[i], dates[i+1]
            svi1     = self.svi_slices[d1]
            svi2     = self.svi_slices[d2]

            w1 = svi1.total_variance(k_grid)
            w2 = svi2.total_variance(k_grid)

            bad = k_grid[w2 < w1 - 1e-6]
            for k in bad:
                violations.append({
                    'T1': svi1.expiry, 'T2': svi2.expiry,
                    'k': k,
                    'w1': float(svi1.total_variance(np.array([k]))[0]),
                    'w2': float(svi2.total_variance(np.array([k]))[0]),
                })

        return {
            'is_arbitrage_free': len(violations) == 0,
            'n_violations'     : len(violations),
            'violations'       : violations,
        }

    def check_butterfly_arbitrage(self) -> dict:
        """
        Check butterfly arbitrage for each SVI slice.

        Returns True if g(k) >= 0 for all k in each slice.

        Returns
        -------
        dict with keys:
            'is_arbitrage_free' : bool
            'slice_results'     : {expiry_date: bool}
        """
        k_grid  = np.linspace(-1.5, 1.5, 200)
        results = {}

        for d, svi in self.svi_slices.items():
            results[d] = svi.is_butterfly_free(k_grid)

        return {
            'is_arbitrage_free': all(results.values()),
            'slice_results'    : results,
        }

    def risk_neutral_density(
        self,
        expiry_date : str,
        K_grid      : Optional[np.ndarray] = None,
    ) -> tuple:
        """
        Compute the risk-neutral density q(K) for one expiry
        using the Breeden-Litzenberger formula:

            q(K) = e^{rT} * d²C/dK²

        In terms of the vol surface:
            q(K) ∝ g(k) / (F * σ_BS * √T)

        where g(k) is the butterfly density proxy from the SVI slice.

        Parameters
        ----------
        expiry_date : str
        K_grid : np.ndarray, optional
            Strike grid for the density. Defaults to F * exp([-3σ, 3σ]).

        Returns
        -------
        K_grid : np.ndarray
        density : np.ndarray   Risk-neutral density q(K)
        """
        svi = self.svi_slices[expiry_date]
        F   = self.forwards.get(expiry_date,
                                self.spot * np.exp(self.rate * svi.expiry))
        T   = svi.expiry

        atm_vol = float(svi.implied_vol(np.array([0.0]))[0])

        if K_grid is None:
            # Grid spanning ~3 standard deviations of log-returns
            k_min  = -3.0 * atm_vol * np.sqrt(T)
            k_max  =  3.0 * atm_vol * np.sqrt(T)
            k_grid = np.linspace(k_min, k_max, 300)
            K_grid = F * np.exp(k_grid)
        else:
            k_grid = np.log(K_grid / F)

        # Butterfly density g(k) from SVI
        g = svi.butterfly_density(k_grid)

        # Convert to density in K-space:
        # q(K) ∝ g(k) / (K * σ(k) * √T)
        w    = svi.total_variance(k_grid)
        w    = np.maximum(w, 1e-8)
        sig  = np.sqrt(w / T)
        dens = g / (K_grid * sig * np.sqrt(T))
        dens = np.maximum(dens, 0.0)

        # Normalise to integrate to 1
        dk   = np.diff(K_grid)
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


# ------------------------------------------------------------------
# Full Surface Calibration
# ------------------------------------------------------------------

def calibrate_vol_surface(
    option_chain,
    min_quotes_per_slice : int   = 5,
    use_vega_weights     : bool  = True,
    verbose              : bool  = False,
) -> VolSurface:
    """
    Calibrate a full SVI vol surface from a cleaned option chain.

    Calibrates one SVI slice per expiry independently, then
    checks the full surface for calendar and butterfly arbitrage.

    Parameters
    ----------
    option_chain : OptionChain
        Cleaned option chain from fetch_option_chain().
    min_quotes_per_slice : int
        Skip expiry slices with fewer quotes than this.
    use_vega_weights : bool
        Weight observations by BS vega. ATM options (high vega) have
        tighter bid-ask spreads and more reliable IVs — upweight them.
    verbose : bool
        Print calibration progress.

    Returns
    -------
    VolSurface
        Fully calibrated vol surface.
    """
    from options_lib.models.black_scholes import BlackScholes
    from options_lib.instruments.european import EuropeanOption
    from options_lib.instruments.base import MarketData, OptionType

    svi_slices = {}
    forwards   = option_chain.forwards.copy()

    for exp_date in option_chain.expiry_dates:
        quotes = option_chain.get_slice(exp_date)

        if len(quotes) < min_quotes_per_slice:
            if verbose:
                print(f"  Skipping {exp_date}: only {len(quotes)} quotes")
            continue

        T = quotes[0].expiry
        F = forwards.get(exp_date,
                         option_chain.spot * np.exp(option_chain.rate * T))

        # Log-moneyness and IVs
        log_m = np.array([np.log(q.strike / F) for q in quotes])
        ivs   = np.array([q.iv for q in quotes])

        # Vega weights: higher weight to ATM options
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
            warnings.warn(f"SVI calibration failed for {exp_date}: {e}")
            continue

    if not svi_slices:
        raise RuntimeError("SVI calibration failed for all expiry slices.")

    surface = VolSurface(
        svi_slices = svi_slices,
        forwards   = forwards,
        spot       = option_chain.spot,
        rate       = option_chain.rate,
        ticker     = option_chain.ticker,
    )

    if verbose:
        cal_check = surface.check_calendar_arbitrage()
        but_check = surface.check_butterfly_arbitrage()
        print(f"\nCalendar arbitrage free: {cal_check['is_arbitrage_free']}")
        print(f"Butterfly arbitrage free: {but_check['is_arbitrage_free']}")

    return surface