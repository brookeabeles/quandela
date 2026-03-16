"""
Fixed-point iteration for the true saddle y_0^*(γ) (BM24 Algorithm 2).

Saddle equation: y_α^* / 2 = √c_α ∑_s w_s(y^*) A_αs.
Update: y_new_α = 2 √c_α E_w[A_α], then damp: y = ρ y + (1-ρ) y_new.
"""

import numpy as np
from .core import softmax_weights


def saddle_fixed_point(
    A: np.ndarray,
    b_s: np.ndarray,
    c_alpha: np.ndarray,
    rho: float = 0.5,
    max_iter: int = 2000,
    tol: float = 1e-12,
) -> tuple[np.ndarray, bool, float]:
    """
    Find y_0^* such that y_α/2 = √c_α ∑_s w_s(y) A_αs.

    Returns (y_star, converged, residual).
    For large γ use higher damping (e.g. rho=0.8 or 0.9).
    """
    d, n_s = A.shape
    sqrt_c = np.sqrt(c_alpha + 0j)
    y = np.zeros(d, dtype=complex)

    for it in range(max_iter):
        w = softmax_weights(y, A, b_s, sqrt_c)
        # E_w[A_α] = sum_s w_s A[α, s]
        E_A = w @ A.T  # (d,)
        y_new = 2.0 * sqrt_c * E_A
        residual = np.max(np.abs(y - y_new))
        y = rho * y + (1.0 - rho) * y_new
        if residual < tol:
            return y, True, float(residual)

    return y, False, float(residual)


def saddle_with_adaptive_damping(
    A: np.ndarray,
    b_s: np.ndarray,
    c_alpha: np.ndarray,
    rho_init: float = 0.5,
    max_iter: int = 3000,
    tol: float = 1e-12,
) -> tuple[np.ndarray, bool, float]:
    """
    Try fixed-point with given rho; if not converged, retry with higher damping.
    """
    for rho in [rho_init, 0.7, 0.85, 0.95]:
        y, ok, res = saddle_fixed_point(A, b_s, c_alpha, rho=rho, max_iter=max_iter, tol=tol)
        if ok:
            return y, True, res
    return y, False, res
