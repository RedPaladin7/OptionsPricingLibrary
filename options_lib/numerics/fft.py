"""
numerics/fft.py
---------------
Carr-Madan FFT method for pricing European options.

Given a model with a known characteristic function φ(u) of ln(S_T),
we can price European calls across ALL strikes simultaneously in O(N log N).

The key insight (Carr & Madan 1999):
    The Fourier transform of a dampened call price has a closed form
    in terms of the characteristic function φ(u). Inverting via FFT
    recovers call prices at N strikes in one shot.

Setup
-----
Define the dampened call price (dampening needed for integrability):
    c_T(k) = e^{αk} C(k)     where k = ln(K), α > 0

Its Fourier transform is:
    ψ(u) = e^{-rT} φ(u - (α+1)i) / (α² + α - u² + i(2α+1)u)

where φ(u) is the characteristic function of ln(S_T) under Q.

Then by the inverse Fourier transform:
    C(k) = e^{-αk} / π  ∫₀^∞  Re[e^{-iuk} ψ(u)] du

This integral is discretised and evaluated via FFT.

References
----------
Carr, P. and Madan, D. (1999). "Option valuation using the fast Fourier
transform." Journal of Computational Finance, 2(4), 61-73.
"""

import numpy as np
from typing import Callable


def carr_madan_fft(
    char_fn      : Callable[[np.ndarray], np.ndarray],
    S            : float,
    T            : float,
    r            : float,
    q            : float,
    alpha        : float = 1.5,
    N            : int   = 4096,
    eta          : float = 0.25,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Price European calls across a grid of log-strikes using the Carr-Madan FFT.

    Parameters
    ----------
    char_fn : callable
        The characteristic function of ln(S_T) under Q.
        Takes a complex array u and returns complex array φ(u).
        All model parameters (v0, kappa, etc.) should be pre-bound.
    S : float
        Current spot price.
    T : float
        Time to expiry in years.
    r : float
        Risk-free rate.
    q : float
        Continuous dividend yield.
    alpha : float
        Dampening factor. Must satisfy α > 0 and α+1 < domain of φ.
        Typical range: 1.0 to 2.0. Too small → oscillatory integrand.
        Too large → numerical instability in φ. Default 1.5 works well.
    N : int
        Number of FFT grid points. Must be a power of 2.
        More points = finer strike grid. Default 4096 is sufficient.
    eta : float
        Spacing of the integration grid in frequency space.
        Smaller eta = wider strike range covered.
        Strike grid spacing λ = 2π / (N * eta).

    Returns
    -------
    strikes : np.ndarray, shape (N,)
        Array of strike prices K corresponding to each call price.
    call_prices : np.ndarray, shape (N,)
        Call prices at each strike. Values outside a reasonable range
        (e.g. deep OTM) may be numerically unreliable.

    Notes
    -----
    The FFT computes N output values from N input values in O(N log N).
    Compared to direct integration at M strikes (O(M * N_quad)), FFT
    is dramatically faster for dense strike grids.

    The log-strike grid is centred at ln(S) (ATM). Strikes are:
        k_j = -N*λ/2 + j*λ,  j = 0, ..., N-1
    where λ = 2π / (N * eta).
    """
    # ------------------------------------------------------------------
    # Step 1: Set up the grids
    # ------------------------------------------------------------------

    # Frequency (integration) grid: u_j = j * eta, j = 0, ..., N-1
    u = np.arange(N) * eta                          # shape (N,)

    # Log-strike grid spacing (determined by FFT duality)
    lam = 2 * np.pi / (N * eta)                     # λ

    # Log-strike grid centred at ln(S)
    # k_j = ln(S) - N*λ/2 + j*λ
    k = np.log(S) - (N * lam / 2) + np.arange(N) * lam   # shape (N,)

    # ------------------------------------------------------------------
    # Step 2: Evaluate the modified integrand ψ(u)
    # ------------------------------------------------------------------
    # ψ(u) = e^{-rT} φ(u - (α+1)i) / (α² + α - u² + i(2α+1)u)
    #
    # The denominator comes from the Fourier transform of e^{αk} max(e^k - K, 0).
    # The shift u - (α+1)i moves the characteristic function evaluation
    # into the complex plane, which is valid as long as φ is defined there.

    # Evaluate char fn at complex argument: u - (α+1)i
    u_complex = u - 1j * (alpha + 1)                # complex array
    phi = char_fn(u_complex)                         # φ(u - (α+1)i)

    # Denominator: α² + α - u² + i(2α+1)u
    denom = alpha**2 + alpha - u**2 + 1j * (2*alpha + 1) * u

    # Full integrand
    psi = np.exp(-r * T) * phi / denom               # shape (N,)

    # ------------------------------------------------------------------
    # Step 3: Apply the Simpson's rule weights for numerical integration
    # ------------------------------------------------------------------
    # Simpson's rule: weights are [1, 4, 2, 4, 2, ..., 4, 1] / 3
    # Applied here to improve accuracy of the discretised integral.
    # The factor eta/3 comes from the integration step size.

    weights = np.ones(N)
    weights[1:-1:2] = 4   # odd indices
    weights[2:-2:2] = 2   # even indices (excluding endpoints)
    weights *= eta / 3

    # ------------------------------------------------------------------
    # Step 4: Construct the FFT input
    # ------------------------------------------------------------------
    # The FFT computes: X[j] = Σ_k x[k] * exp(-2πi j k / N)
    # We need:          Σ_k e^{-i u_k k_j} ψ(u_k) * weight_k
    #
    # Factor out the k-grid offset ln(S) - N*λ/2 by multiplying by
    # the corresponding phase shift.

    # b = left endpoint of log-strike grid: k_j = b + j*lam
    # Phase factor e^{-i u_n b} aligns the FFT with the log-strike grid.
    # With u_n = n*eta, eta*lam = 2π/N, so e^{-i u_n k_j} = e^{-i u_n b} * e^{-2πi nj/N}
    # which is exactly what np.fft.fft computes.
    b = np.log(S) - N * lam / 2
    phase = np.exp(-1j * u * b)

    fft_input = weights * psi * phase                # shape (N,)

    # ------------------------------------------------------------------
    # Step 5: Execute the FFT
    # ------------------------------------------------------------------
    fft_output = np.fft.fft(fft_input)               # O(N log N)

    # ------------------------------------------------------------------
    # Step 6: Recover call prices from FFT output
    # ------------------------------------------------------------------
    # C(k_j) = e^{-α k_j} / π * Re[fft_output_j]
    #
    # The 1/π factor (not 1/(2π)) because we integrate over [0,∞) only
    # (the integrand is even in its real part).

    call_prices = (np.exp(-alpha * k) / np.pi) * np.real(fft_output)

    # Convert log-strikes back to strikes
    strikes = np.exp(k)

    return strikes, call_prices


def interpolate_call_price(
    strikes     : np.ndarray,
    call_prices : np.ndarray,
    target_strike : float
) -> float:
    """
    Linearly interpolate a call price at a specific strike from the FFT grid.

    Parameters
    ----------
    strikes : np.ndarray
        Strike grid from carr_madan_fft.
    call_prices : np.ndarray
        Call prices on the grid from carr_madan_fft.
    target_strike : float
        The strike at which we want the call price.

    Returns
    -------
    float
        Interpolated call price.
    """
    return float(np.interp(target_strike, strikes, call_prices))