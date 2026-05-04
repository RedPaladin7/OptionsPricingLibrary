"""
numerics/root_finding.py
------------------------
Numerical root-finding methods for implied volatility inversion.

The BS formula C = f(σ) is not analytically invertible in σ.
Given a market price C_mkt, we need to find σ_imp such that:
    f(σ_imp) = C_mkt

Two methods are implemented:

1. Brent's method  — bracketed, guaranteed to converge, robust
2. Newton-Raphson  — fast near solution (uses Vega as derivative)

Strategy: use Newton-Raphson first (fast), fall back to Brent if it
fails to converge (e.g., near-zero Vega for deep OTM options).
"""

import numpy as np
from typing import Callable, Tuple


class ConvergenceError(Exception):
    """Raised when root-finding fails to converge."""
    pass


# ------------------------------------------------------------------
# Brent's Method
# ------------------------------------------------------------------

def brent(
    f: Callable[[float], float],
    a: float,
    b: float,
    tol: float = 1e-8,
    max_iter: int = 100
) -> float:
    """
    Brent's method for root finding on a bracketed interval [a, b].

    Combines bisection (safe, slow) with secant/inverse quadratic
    interpolation (fast near root). Guaranteed to converge if f(a)
    and f(b) have opposite signs.

    Parameters
    ----------
    f : callable
        The function whose root we seek. f(a) and f(b) must have
        opposite signs (i.e., [a,b] brackets a root).
    a, b : float
        Initial bracket. Must satisfy f(a) * f(b) < 0.
    tol : float
        Convergence tolerance on |b - a|.
    max_iter : int
        Maximum number of iterations.

    Returns
    -------
    float
        The root x* such that f(x*) ≈ 0.

    Raises
    ------
    ValueError
        If [a, b] does not bracket a root.
    ConvergenceError
        If max_iter is reached without convergence.

    Notes
    -----
    Why Brent over Newton-Raphson for implied vol?
    Newton-Raphson can fail when Vega ≈ 0 (deep OTM options), causing
    division by near-zero and divergence. Brent's method never evaluates
    the derivative and always converges within the bracket — it just
    finds where the BS price crosses the market price.
    """
    fa, fb = f(a), f(b)

    if fa * fb > 0:
        raise ValueError(
            f"Brent's method requires f(a)*f(b) < 0. "
            f"Got f({a:.4f})={fa:.6f}, f({b:.4f})={fb:.6f}. "
            "The market price may be outside the model's range."
        )

    if abs(fa) < abs(fb):
        a, b = b, a
        fa, fb = fb, fa

    c, fc = a, fa
    mflag = True
    s = 0.0
    d = 0.0

    for _ in range(max_iter):
        if abs(b - a) < tol:
            return b

        if fa != fc and fb != fc:
            # Inverse quadratic interpolation
            s = (a * fb * fc / ((fa - fb) * (fa - fc))
                 + b * fa * fc / ((fb - fa) * (fb - fc))
                 + c * fa * fb / ((fc - fa) * (fc - fb)))
        else:
            # Secant method
            s = b - fb * (b - a) / (fb - fa)

        # Conditions for bisection fallback
        cond1 = not ((3 * a + b) / 4 < s < b or b < s < (3 * a + b) / 4)
        cond2 = mflag and abs(s - b) >= abs(b - c) / 2
        cond3 = not mflag and abs(s - b) >= abs(c - d) / 2
        cond4 = mflag and abs(b - c) < tol
        cond5 = not mflag and abs(c - d) < tol

        if cond1 or cond2 or cond3 or cond4 or cond5:
            s = (a + b) / 2  # bisection
            mflag = True
        else:
            mflag = False

        fs = f(s)
        d, c = c, b
        fc = fb

        if fa * fs < 0:
            b, fb = s, fs
        else:
            a, fa = s, fs

        if abs(fa) < abs(fb):
            a, b = b, a
            fa, fb = fb, fa

    raise ConvergenceError(
        f"Brent's method did not converge in {max_iter} iterations. "
        f"Final bracket: [{a:.6f}, {b:.6f}], width: {abs(b-a):.2e}"
    )


# ------------------------------------------------------------------
# Newton-Raphson
# ------------------------------------------------------------------

def newton_raphson(
    f: Callable[[float], float],
    df: Callable[[float], float],
    x0: float,
    tol: float = 1e-8,
    max_iter: int = 50
) -> float:
    """
    Newton-Raphson method for root finding.

    Iteration: x_{n+1} = x_n - f(x_n) / f'(x_n)

    Quadratic convergence near the root — very fast when it works.
    Can diverge if f'(x) ≈ 0 (Vega ≈ 0 for deep OTM options).

    Parameters
    ----------
    f  : callable   — function whose root we seek
    df : callable   — derivative of f (for implied vol: this is Vega)
    x0 : float      — initial guess
    tol : float     — convergence tolerance |f(x)| < tol
    max_iter : int  — maximum iterations

    Returns
    -------
    float
        Root x* such that f(x*) ≈ 0.

    Raises
    ------
    ConvergenceError
        If NR diverges or derivative is too small.
    """
    x = x0
    for i in range(max_iter):
        fx  = f(x)
        if abs(fx) < tol:
            return x
        dfx = df(x)
        if abs(dfx) < 1e-12:
            raise ConvergenceError(
                f"Newton-Raphson: derivative too small ({dfx:.2e}) at x={x:.6f}. "
                "Likely near zero-Vega region. Switch to Brent's method."
            )
        x = x - fx / dfx
        # Keep vol in reasonable bounds
        x = max(x, 1e-6)
        x = min(x, 10.0)

    raise ConvergenceError(
        f"Newton-Raphson did not converge in {max_iter} iterations. "
        f"Last value: x={x:.6f}, f(x)={f(x):.2e}"
    )


# ------------------------------------------------------------------
# Implied Volatility Solver
# ------------------------------------------------------------------

def implied_vol(
    market_price: float,
    pricer: Callable[[float], float],
    vega_fn: Callable[[float], float],
    sigma_init: float = 0.20,
    tol: float = 1e-6
) -> float:
    """
    Solve for implied volatility given a market price.

    Strategy:
      1. Try Newton-Raphson starting from sigma_init (fast, usually works)
      2. If NR fails, fall back to Brent's method on [1e-4, 5.0] (robust)

    Parameters
    ----------
    market_price : float
        Observed market price of the option.
    pricer : callable
        Function sigma -> model_price. Should be the BS formula with
        all other parameters (S, K, T, r, q) pre-bound.
    vega_fn : callable
        Function sigma -> vega. Used as the NR derivative.
        For BS: vega = S * exp(-qT) * N'(d1) * sqrt(T)
    sigma_init : float
        Initial guess for NR. Default 20% is reasonable for equities.
    tol : float
        Convergence tolerance on |model_price - market_price|.

    Returns
    -------
    float
        Implied volatility σ_imp such that pricer(σ_imp) ≈ market_price.

    Raises
    ------
    ConvergenceError
        If both NR and Brent fail.
    ValueError
        If market_price is outside the no-arbitrage range.

    Notes
    -----
    No-arbitrage bounds on the option price constrain the search:
    For a call: max(S*exp(-qT) - K*exp(-rT), 0) ≤ C ≤ S*exp(-qT)
    A price outside these bounds implies arbitrage — no finite IV exists.
    """
    # Objective: find sigma s.t. pricer(sigma) - market_price = 0
    objective = lambda sigma: pricer(sigma) - market_price
    derivative = lambda sigma: vega_fn(sigma)

    # 1. Try Newton-Raphson first
    try:
        iv = newton_raphson(objective, derivative, x0=sigma_init, tol=tol)
        if 1e-6 < iv < 10.0:
            return iv
    except ConvergenceError:
        pass

    # 2. Fall back to Brent's method
    # Check that [lo, hi] brackets the root
    lo, hi = 1e-4, 5.0
    f_lo = objective(lo)
    f_hi = objective(hi)

    if f_lo * f_hi > 0:
        raise ValueError(
            f"Market price {market_price:.4f} is outside the no-arbitrage range. "
            f"Model price at σ={lo}: {pricer(lo):.4f}, at σ={hi}: {pricer(hi):.4f}. "
            "Check for stale/erroneous market data."
        )

    return brent(objective, lo, hi, tol=tol)