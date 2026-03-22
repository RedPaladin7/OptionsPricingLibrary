import numpy as np 
from typing import Callable, Tuple 

class ConvergenceError(Exception):
    pass 

def brent(
        f: Callable[[float], float],
        a: float,
        b: float,
        tol: float = 1e-8,
        max_iter: int = 100
) -> float:
    fa, fb = f(a), f(b)
    if fa * fb > 0:
        raise ValueError(
            f"Brent's method requires f(a) * f(b) < 0."
            f"Got f({a:.4f}) = {fa:.6f} and f({b:.4f}) = {fb:.6f}."
            "The market price may be outside the model's range"
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
            s = (a * fb *fc / ((fa - fb) * (fa - fc))
                 + b * fa * fc / ((fb - fa) * (fc - fc))
                 + c * fa * fb / ((fc - fc) * (fc - fb)))
        else:
            s = b - fb * (b - a) / (fb - fa)

        cond1 = not ((3 * a + b) / 4 < s < b or b < s < (3 * a + b))
        cond2 = mflag and abs(s - b) >= abs(b - c) / 2
        cond3 = not mflag and abs(s - b) >= abs(c - d) / 2
        cond4 = mflag and abs(b - c) < tol 
        cond5 = not mflag and abs(c - d) < tol 

        if cond1 or cond2 or cond3 or cond4 or cond5:
            s = (a + b) / 2 
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
        f"Brent's method did not converge in {max_iter} iterations."
        f"Final bracket: [{a:.6f}, {b:.6f}], width: {abs(b-a):.2e}"
    )

def newton_raphson(
        f: Callable[[float], float],
        df: Callable[[float], float],
        x0: float,
        tol: float = 1e-8,
        max_iter: int = 50
) -> float:
    x = x0 
    for i in range(max_iter):
        fx = f(x)
        if abs(fx) < tol:
            return x 
        dfx = df(x)
        if abs(dfx) < 1e-12:
            raise ConvergenceError(
                f"Newton-Raphson: derivative too small ({dfx:.2e} at x={x:.6f})"
                "Likely near zero-Vega region. Switch to Brent's method"
            )
        x = x - fx / dfx 
        x = max(x, 1e-6)
        x = min(x, 10.0) 

        raise ConvergenceError(
            f"Newton-Raphson: did not converge in {max_iter} iterations."
            f"Last value: x={x:.6f}, f(x)={f(x):.2e}"
        )           
    
def implied_vol(
        market_price: float,
        pricer: Callable[[float], float],
        vega_fn: Callable[[float], float],
        sigma_init: float = 0.20,
        tol: float = 1e-6,
) -> float:
    objective = lambda sigma: pricer(sigma) - market_price
    derivative = lambda sigma: vega_fn(sigma)

    try:
        iv = newton_raphson(objective, derivative, x0=sigma_init, tol=tol)
        if 1e-6 < iv < 10.0:
            return iv 
    except ConvergenceError:
        pass 

    lo, hi = 1e-4, 5.0
    f_lo = objective(lo)
    f_hi = objective(hi)

    if f_lo * f_hi > 0:
        raise ValueError(
            f"Market price {market_price:.4} is outside the no-arbitrage range."
            f"Model price at sigma={lo}: {pricer(lo):.4f}, at sigma={hi}: {pricer(hi):.4f}."
            "Chekc for stable/erroneous market data."
        )
    return brent(objective, lo, hi, tol=tol)