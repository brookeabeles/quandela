"""
Fixed-point iteration for the true saddle y_0^*(γ) (BM24 Algorithm 2).

Saddle equation: y_α^* / 2 = √c_α ∑_s w_s(y^*) A_αs.
Update: y_new_α = 2 √c_α E_w[A_α], then damp: y = ρ y + (1-ρ) y_new.
"""

import numpy as np
from .core import bm24_clause_arity, softmax_weights, softmax_weights_with_coeff


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


def saddle_fixed_point_q(
    A: np.ndarray,
    b_s: np.ndarray,
    coeff_alpha: np.ndarray,
    q: int,
    rho: float = 0.5,
    max_iter: int = 2000,
    tol: float = 1e-12,
    y_init: np.ndarray | None = None,
) -> tuple[np.ndarray, bool, float]:
    """
    Fixed point for BM24 Proposition 1 (Eq. (20)) for general q>=1.

    Let F(y) = log ∑_s b_s exp(∑_α coeff_α A_{αs} y_α) where coeff_α = r * (-c_α)^{1/(2^q)}.
    Then:
        ∂_α F(y) = coeff_α E_w[A_α]
    and the fixed point is (BM24 Eq. (20), k = 2^q):
        y_α = -k * ( ∂_α F(y) )^{k-1}.

    For q=1 (k=2), this reduces to y_α = -2 ∂_α F(y), matching the Gaussian case.
    """
    if q < 1:
        raise ValueError("q must be >= 1")

    d, _ = A.shape
    coeff_alpha = np.asarray(coeff_alpha, dtype=complex)
    if coeff_alpha.shape[0] != d:
        raise ValueError("coeff_alpha has wrong length")

    if y_init is not None:
        y = np.copy(np.asarray(y_init, dtype=complex))
        if y.shape != (d,):
            raise ValueError("y_init has wrong shape")
    else:
        y = np.zeros(d, dtype=complex)

    for _it in range(max_iter):
        w = softmax_weights_with_coeff(y, A, b_s, coeff_alpha)
        E_A = w @ A.T  # (d,)
        gradF = coeff_alpha * E_A  # (d,)
        k = bm24_clause_arity(q)
        y_new = -float(k) * np.power(gradF, k - 1)

        # Guard against overflow / NaN propagation
        if not np.all(np.isfinite(y_new)):
            residual = float("inf")
            continue

        residual = float(np.max(np.abs(y - y_new)))
        y = rho * y + (1.0 - rho) * y_new
        if residual < tol:
            return y, True, residual

    return y, False, residual


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


def saddle_with_adaptive_damping_q(
    A: np.ndarray,
    b_s: np.ndarray,
    coeff_alpha: np.ndarray,
    q: int,
    rho_init: float = 0.5,
    max_iter: int = 3000,
    tol: float = 1e-12,
    y_init: np.ndarray | None = None,
) -> tuple[np.ndarray, bool, float]:
    """
    Adaptive damping wrapper for saddle_fixed_point_q (general q).
    """
    for rho in [rho_init, 0.7, 0.85, 0.95]:
        y, ok, res = saddle_fixed_point_q(
            A,
            b_s,
            coeff_alpha,
            q=q,
            rho=rho,
            max_iter=max_iter,
            tol=tol,
            y_init=y_init,
        )
        if ok:
            return y, True, res
    return y, False, res


def _homotopy_subdivide(
    A: np.ndarray,
    b_s: np.ndarray,
    coeff_alpha_fn,
    q: int,
    y_init: np.ndarray,
    r_lo: float,
    r_hi: float,
    rho: float,
    max_iter: int,
    tol: float,
    max_depth: int,
    verbose: bool,
):
    """
    Recursively subdivide the interval [r_lo, r_hi] to find a convergent path.
    Returns (y_converged, residual) or None if max_depth exhausted.
    """
    if max_depth <= 0:
        return None
    r_mid = (r_lo + r_hi) / 2.0

    coeff_mid = coeff_alpha_fn(r_mid)
    y_mid, conv_mid, res_mid = saddle_fixed_point_q(
        A,
        b_s,
        coeff_mid,
        q=q,
        rho=rho,
        max_iter=max_iter,
        tol=tol,
        y_init=y_init,
    )
    if not conv_mid:
        result = _homotopy_subdivide(
            A,
            b_s,
            coeff_alpha_fn,
            q,
            y_init,
            r_lo=r_lo,
            r_hi=r_mid,
            rho=rho,
            max_iter=max_iter,
            tol=tol,
            max_depth=max_depth - 1,
            verbose=verbose,
        )
        if result is None:
            return None
        y_mid, res_mid = result

    if verbose:
        print(f"    subdivide: r_mid={r_mid:.4f}, residual={res_mid:.3e}")

    coeff_hi = coeff_alpha_fn(r_hi)
    y_hi, conv_hi, res_hi = saddle_fixed_point_q(
        A,
        b_s,
        coeff_hi,
        q=q,
        rho=rho,
        max_iter=max_iter,
        tol=tol,
        y_init=y_mid,
    )
    if conv_hi:
        return y_hi, res_hi
    return _homotopy_subdivide(
        A,
        b_s,
        coeff_alpha_fn,
        q,
        y_mid,
        r_lo=r_mid,
        r_hi=r_hi,
        rho=rho,
        max_iter=max_iter,
        tol=tol,
        max_depth=max_depth - 1,
        verbose=verbose,
    )


def saddle_fixed_point_q_homotopy(
    A: np.ndarray,
    b_s: np.ndarray,
    coeff_alpha_fn,
    q: int,
    r_target: float,
    r_start: float = 1.0,
    step_factor: float = 1.05,
    rho: float = 0.95,
    max_iter_per_step: int = 20000,
    tol: float = 1e-6,
    verbose: bool = False,
) -> tuple[np.ndarray, bool, float]:
    """
    Homotopy continuation in r for the q>1 fixed point (BM24 Eq. 20).

    The fixed-point map has Lipschitz constant ~ r^{k-1} with k=2^q, making direct iteration
    divergent for large r (e.g. r=176.54 for 8-SAT when q=3). This solver starts at
    r_start and ramps geometrically to r_target, warmstarting each step from the
    previous converged solution.
    """
    if r_start <= 0 or r_target <= 0:
        raise ValueError("r_start and r_target must be positive")
    if step_factor <= 1.0:
        raise ValueError("step_factor must be > 1")

    d, _ = A.shape
    y = np.zeros(d, dtype=complex)

    # Adaptive schedule with backtracking (more robust than deep recursion):
    # propose r_next = r_cur * step_factor; if step fails, bisect r interval until it works.
    r_cur = float(r_start)
    last_residual = 0.0
    converged_global = True

    step_idx = 0
    while r_cur < float(r_target) - 1e-15:
        step_idx += 1
        r_hi = min(float(r_target), r_cur * float(step_factor))
        r_next = r_hi

        # Backtrack by bisection until converged (or give up)
        ok = False
        res = float("inf")
        for _bt in range(30):
            coeff = coeff_alpha_fn(r_next)
            y_step, conv, res = saddle_fixed_point_q(
                A,
                b_s,
                coeff,
                q=q,
                rho=rho,
                max_iter=max_iter_per_step,
                tol=tol,
                y_init=y,
            )
            if conv:
                y = y_step
                last_residual = res
                r_cur = r_next
                ok = True
                if verbose:
                    print(f"  homotopy step {step_idx}: r={r_cur:.6f}, residual={res:.3e}")
                break
            # bisect interval
            r_next = 0.5 * (r_cur + r_next)
            if (r_next - r_cur) / max(r_cur, 1e-12) < 1e-6:
                break

        if not ok:
            converged_global = False
            last_residual = res
            if verbose:
                print(f"  WARNING: homotopy stalled near r={r_cur:.6f} (last res={res:.3e})")
            break

    return y, converged_global, float(last_residual)
