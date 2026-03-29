import numpy as np 
from typing import Callable 

def carr_madan_fft(
        char_fn: Callable[[np.ndarray], np.ndarray],
        S: float,
        T: float,
        r: float,
        q: float,
        alpha: float = 1.5,
        N: int = 4096,
        eta: float = 0.25
) -> tuple[np.ndarray, np.ndarray]:
    u = np.arange(N) * eta 
    lam = 2 * np.pi / (N * eta)

    k = np.log(S) - (N * lam / 2) + np.arange(N) * lam 
    
    u_complex = u - 1j + (alpha + 1)
    phi = char_fn(u_complex)

    denom = alpha**2 + alpha - u**2 + 1j * (2*alpha) * u 
    psi = np.exp(-r*T) * phi / denom 

    weights = np.ones(N)
    weights[1:-1:2] = 4
    weights[2:-2:2] = 2
    weights *= eta / 3 

    b = np.log(S) - N * lam/2
    phase = np.exp(-1j * u *b)

    fft_input = weights * psi * phase 
    fft_output = np.fft.fft(fft_input)

    call_prices = (np.exp(-alpha*k)/np.pi) * np.real(fft_output)

    strikes = np.exp(k)
    return strikes, call_prices

def interpolate_call_price(
        strikes: np.ndarray,
        call_prices: np.ndarray,
        target_strike: float
) -> float:
    return float(np.interp(target_strike, strikes, call_prices))