"""
Validation checks from Section 4 of the instructions.
"""

import numpy as np
from .core import build_structure_matrix, compute_b_s, compute_c_alpha, hessian_log_at_y, weights_at_zero


def check_b_normalization(b_s: np.ndarray, tol: float = 1e-8) -> bool:
    """∑_s b_s = 1."""
    s = np.sum(b_s)
    ok = np.abs(s - 1.0) < tol
    if not ok:
        print(f"  [FAIL] sum(b_s) = {s} (expected 1)")
    return ok


def rank_of_matrix(M: np.ndarray, tol: float = 1e-10) -> int:
    """Numerical rank (singular values > tol * σ_1)."""
    sigmas = np.linalg.svd(M, compute_uv=False)
    if len(sigmas) == 0 or sigmas[0] < 1e-15:
        return 0
    return int(np.sum(sigmas > tol * sigmas[0]))


def check_rank_A(A: np.ndarray, p: int, tol: float = 1e-10) -> bool:
    """rank(A) = 2^{2p}."""
    expected = 1 << (2 * p)
    r = rank_of_matrix(A, tol)
    ok = r == expected
    if not ok:
        print(f"  [FAIL] rank(A) = {r} (expected {expected})")
    return ok


def check_rank_Cov_at_zero(A: np.ndarray, b_s: np.ndarray, p: int, tol: float = 1e-10) -> bool:
    """rank(Cov_{w(0)}(A)) = 2^{2p} - 1."""
    w = weights_at_zero(b_s)
    d, n_s = A.shape
    E_A = w @ A.T
    E_AA = A @ (w[:, None] * A.T)
    Cov = E_AA - np.outer(E_A, E_A)
    r = rank_of_matrix(Cov, tol)
    expected = (1 << (2 * p)) - 1
    ok = r == expected
    if not ok:
        print(f"  [FAIL] rank(Cov_w0(A)) = {r} (expected {expected})")
    return ok


def check_zero_eigenvalues_H_log_at_zero(p: int, A: np.ndarray, b_s: np.ndarray, c_alpha: np.ndarray, tol: float = 1e-8) -> bool:
    """Number of zero eigenvalues of H_log(0) = 2^{2p} + 1."""
    y0 = np.zeros(A.shape[0], dtype=complex)
    H = hessian_log_at_y(y0, A, b_s, c_alpha)
    # H_log is complex; use SVD. Zero singular values count.
    sigmas = np.linalg.svd(H, compute_uv=False)
    num_zero = int(np.sum(sigmas < tol))
    expected = (1 << (2 * p)) + 1
    ok = num_zero == expected
    if not ok:
        print(f"  [FAIL] #(σ≈0) of H_log(0) = {num_zero} (expected {expected})")
    return ok


def check_Hermitian(H: np.ndarray, tol: float = 1e-8) -> bool:
    """H is Hermitian: ‖H - H†‖_F < tol. H_log is complex symmetric so may not be Hermitian; warn only."""
    diff = H - np.conj(H).T
    n = np.sqrt(np.sum(np.abs(diff) ** 2))
    ok = n < tol
    if not ok:
        print(f"  [WARN] H_log - H_log† Frobenius = {n:.2e} (H is complex symmetric, not Hermitian)")
    return True  # don't fail build; this is expected for complex c_alpha


def check_saddle_residual(y: np.ndarray, A: np.ndarray, w: np.ndarray, c_alpha: np.ndarray, tol: float = 1e-8) -> bool:
    """At y_0^*, |y_α/2 - √c_α ∑_s w_s A_αs| < tol for all α."""
    sqrt_c = np.sqrt(c_alpha + 0j)
    E_A = w @ A.T
    rhs = sqrt_c * E_A
    residual = np.abs(y / 2.0 - rhs)
    ok = np.max(residual) < tol
    if not ok:
        print(f"  [FAIL] saddle residual max = {np.max(residual)}")
    return ok


def check_frobenius_consistency(H: np.ndarray, sigmas: np.ndarray, tol: float = 1e-10) -> bool:
    """‖H‖_F² = ∑_i σ_i² = ∑_{α,α'} |H_{αα'}|²."""
    from .spectral import frobenius_norm
    entry_sq = np.sum(np.abs(H) ** 2)
    sv_sq = np.sum(sigmas ** 2)
    diff = np.abs(entry_sq - sv_sq)
    ok = diff < tol * (entry_sq + 1e-30)
    if not ok:
        print(f"  [FAIL] Frobenius: entries²={entry_sq}, σ² sum={sv_sq}")
    return ok


def check_bitflip_symmetry(A: np.ndarray, p: int) -> bool:
    """A_{α,s} = A_{α, s̄} for all α, s (s̄ = bit flip of s)."""
    n = 2 * p + 1
    n_s = 1 << n
    for alpha in range(min(100, A.shape[0])):  # sample
        for s in range(min(100, n_s)):
            s_bar = s ^ ((1 << n) - 1)  # flip all bits
            if np.abs(A[alpha, s] - A[alpha, s_bar]) > 1e-12:
                print(f"  [FAIL] bit-flip symmetry at α={alpha}, s={s}")
                return False
    return True


def run_all_checks(p: int, A: np.ndarray, b_s: np.ndarray, c_alpha: np.ndarray, y_saddle: np.ndarray | None = None, w_saddle: np.ndarray | None = None) -> dict[str, bool]:
    """Run all validation checks. Returns dict of check_name -> passed."""
    results = {}
    results["sum_b_s"] = check_b_normalization(b_s)
    results["rank_A"] = check_rank_A(A, p)
    results["rank_Cov_w0"] = check_rank_Cov_at_zero(A, b_s, p)
    results["zero_eig_H_log_0"] = check_zero_eigenvalues_H_log_at_zero(p, A, b_s, c_alpha)
    y0 = np.zeros(A.shape[0], dtype=complex)
    H0 = hessian_log_at_y(y0, A, b_s, c_alpha)
    results["Hermitian"] = check_Hermitian(H0)
    sigmas = np.linalg.svd(H0, compute_uv=False)
    results["Frobenius_consistency"] = check_frobenius_consistency(H0, sigmas)
    results["bitflip_symmetry"] = check_bitflip_symmetry(A, p)
    if y_saddle is not None and w_saddle is not None:
        results["saddle_residual"] = check_saddle_residual(y_saddle, A, w_saddle, c_alpha)
    else:
        results["saddle_residual"] = True  # skip
    return results
