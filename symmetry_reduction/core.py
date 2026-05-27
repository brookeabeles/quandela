"""
Core definitions for QAOA Hessian spectral concentration (BM24).

Implements: index sets S, A; structure matrix A_αs; mixing coefficients b_s;
phase coefficients c_α; Hessian perturbation H_log(y).
Notation follows Boulebnane & Montanaro, PRX Quantum 5, 030348 (2024).
"""

import numpy as np
from collections import defaultdict

# Default: β_j = π/4 for all j; clause-to-variable ratio r = 1
DEFAULT_BETA = np.pi / 4
DEFAULT_R = 1.0


def bm24_clause_arity(q: int) -> int:
    """BM24 clause arity k = 2^q (not 2*q). Used in Eq. (20) fixed-point powers."""
    return 1 << int(q)


def popcount(n):
    """Number of 1-bits in integer n (|α| for subset α as bitmask)."""
    return bin(n).count("1")


def indices_by_size(p):
    """Group subset indices α in [0, 2^{2p+1}) by |α|. Returns dict size -> list of alpha."""
    d = 1 << (2 * p + 1)
    groups = defaultdict(list)
    for alpha in range(d):
        groups[popcount(alpha)].append(alpha)
    return dict(groups)


def structure_matrix_entry(alpha: int, s: int, n: int, q: int = 1) -> float:
    """
    Structure matrix element A_αs used in the generalized multinomial sum framework of BM24.

    In the notation of Appendix A, this implements Eq. (A35):

        A_αs := (1/2) * 1[∀ j, j' ∈ α : s_j = s_{j'}],

    i.e. it is 1/2 when all bits of s on the positions in α are equal, and 0 otherwise.

    IMPORTANT:  Eq. (A35) is *independent* of q (and therefore of k = 2^q).
    The dependence on q enters only through the 2q-th power in the generalized
    multinomial sum (Eq. (A30)), not through A_αs itself.  We nevertheless accept
    a `q` parameter for API uniformity with k = 2^q, so callers can pass q=3 for
    8-SAT, but the definition of A_αs is the same for all q.

    For |α| ≤ 1: return 1/2 (vacuous). For |α| ≥ 2: 1/2 iff all bits of s at positions in α agree.
    n = 2p+1 = number of bits. The parameter q is currently ignored by design.
    """
    k = popcount(alpha)
    if k <= 1:
        return 0.5
    # Get bits of s at positions in alpha
    bits = []
    for j in range(n):
        if (alpha >> j) & 1:
            bits.append((s >> j) & 1)
    return 0.5 if all(b == bits[0] for b in bits) else 0.0


def build_structure_matrix(p: int, q: int = 1) -> np.ndarray:
    """
    Build A ∈ R^{d×d} with d = 2^{2p+1}.

    Rows = α (subset index), columns = s (configuration index).
    For all q ≥ 1 (k = 2^q), BM24 Appendix A Example 1 fixes A_αs to Eq. (A35),
    which is independent of q.  The `q` parameter is accepted so that higher-level
    code can be explicit about the targeted k-SAT regime (e.g. q=3 for k=8),
    but the actual entries follow the universal definition

        A_αs = (1/2) * 1[∀ j, j' ∈ α : s_j = s_{j'}].

    This matches Eq. (A35) exactly, and the q-dependence instead appears in
    the 2q-th power in the generalized multinomial sum (Eq. (A30)).
    A[alpha, s] = (1/2) * 1[s constant on alpha].
    Vectorized over s for each alpha (O(d) numpy work per row).
    """
    n = 2 * p + 1
    d = 1 << n
    A = np.zeros((d, d))
    s_grid = np.arange(d, dtype=np.int64)
    for alpha in range(d):
        positions = [j for j in range(n) if (alpha >> j) & 1]
        if len(positions) <= 1:
            A[alpha, :] = 0.5
            continue
        pos_arr = np.array(positions, dtype=np.int64)
        # bits[s, k] = bit at positions[k] of s
        bits = ((s_grid[:, None] >> pos_arr) & 1).astype(np.int8)
        agree = (bits == bits[:, 0:1]).all(axis=1)
        A[alpha, :] = 0.5 * agree
    return A


def compute_b_s(p: int, betas: np.ndarray) -> np.ndarray:
    """
    Mixing coefficients b_s from BM24 Eq. (A5) / Eq. (17).

    b_s = (1/2) * ∏_{j=0}^{p-1} ⟨s_j | e^{iβ_j X} | s_{j+1}⟩ · ⟨s_{2p-j-1} | e^{-iβ_j X} | s_{2p-j}⟩
    with ⟨s'| e^{iβ X} |s⟩ = cos(β) if s'=s, i·sin(β) if s'≠s.

    s is represented as integer in [0, 2^{2p+1}); bit j = (s >> j) & 1.
    Returns complex array of length 2^{2p+1}. Sum should equal 1.
    """
    n = 2 * p + 1
    n_s = 1 << n
    betas = np.asarray(betas, dtype=complex)
    if len(betas) < p:
        betas = np.resize(betas, p)
    betas = betas[:p]

    b = np.zeros(n_s, dtype=complex)
    for s in range(n_s):
        val = 0.5
        for j in range(p):
            s_j = (s >> j) & 1
            s_jp1 = (s >> (j + 1)) & 1
            if s_j == s_jp1:
                val *= np.cos(betas[j])
            else:
                val *= 1j * np.sin(betas[j])
            # backward factor
            idx_lo = 2 * p - j - 1
            idx_hi = 2 * p - j
            s_lo = (s >> idx_lo) & 1
            s_hi = (s >> idx_hi) & 1
            if s_lo == s_hi:
                val *= np.cos(betas[j])
            else:
                val *= -1j * np.sin(betas[j])
        b[s] = val
    return b


def compute_c_alpha(p: int, gammas: np.ndarray, r: float = DEFAULT_R) -> np.ndarray:
    """
    Coefficients c_α used by the q=1 Gaussian-integral (quadratic) saddle implementation.

    This follows BM24 Eq. (18) and incorporates the clause-to-variable ratio `r` as an
    overall multiplicative prefactor (matching Example 1 Eq. (A34) in Appendix A).

        c_α := r * (-1)^{1[p∈α]} *
               ∏_{j∈α, j<p} (e^{-iγ_j/2} - 1) *
               ∏_{j∈α, j>p} (e^{ iγ_{2p-j}/2} - 1).

    Index p contributes only the sign; no exponential factor for j=p.
    α is subset as bitmask: j ∈ α iff (alpha >> j) & 1.
    Returns complex array of length d = 2^{2p+1}.
    """
    n = 2 * p + 1
    d = 1 << n
    gammas = np.asarray(gammas, dtype=complex)
    if len(gammas) < p:
        gammas = np.resize(gammas, p)
    gammas = gammas[:p]

    c = np.zeros(d, dtype=complex)
    for alpha in range(d):
        sign = -1.0 if (alpha >> p) & 1 else 1.0
        prod = r * sign
        for j in range(n):
            if not ((alpha >> j) & 1):
                continue
            if j < p:
                prod *= np.exp(-1j * gammas[j] / 2) - 1
            elif j > p:
                prod *= np.exp(1j * gammas[2 * p - j] / 2) - 1
        c[alpha] = prod
    return c


def compute_c_alpha_phase(p: int, gammas: np.ndarray) -> np.ndarray:
    """
    BM24 Eq. (18) phase-only coefficient (no r prefactor).

        c_α^(phase) := (-1)^{1[p∈α]} *
                       ∏_{j∈α, j<p} (e^{-iγ_j/2} - 1) *
                       ∏_{j∈α, j>p} (e^{ iγ_{2p-j}/2} - 1).

    This is the coefficient that appears inside (-c_α)^{1/(2^q)} in Proposition 1 / Eq. (16),
    with the separate global factor r handled explicitly.
    """
    n = 2 * p + 1
    d = 1 << n
    gammas = np.asarray(gammas, dtype=complex)
    if len(gammas) < p:
        gammas = np.resize(gammas, p)
    gammas = gammas[:p]

    c_phase = np.zeros(d, dtype=complex)
    for alpha in range(d):
        sign = -1.0 if (alpha >> p) & 1 else 1.0
        prod = sign
        for j in range(n):
            if not ((alpha >> j) & 1):
                continue
            if j < p:
                prod *= np.exp(-1j * gammas[j] / 2) - 1
            elif j > p:
                prod *= np.exp(1j * gammas[2 * p - j] / 2) - 1
        c_phase[alpha] = prod
    return c_phase


def alpha_linear_coefficients(p: int, gammas: np.ndarray, q: int = 1, r: float = DEFAULT_R) -> np.ndarray:
    """
    Coefficients multiplying A_{αs} y_α in the BM24 Proposition 1 definition of F (Eq. (16)).

    Eq. (16) uses the combination:
        (r/2) * 1[const on α] * (-c_α)^{1/(2^q)} * z_α
    and with Eq. (A35): A_{αs} = (1/2) * 1[const on α],
    this becomes:
        r * A_{αs} * (-c_α)^{1/(2^q)} * z_α.

    Therefore the linear coefficient we need is:
        coeff_α := r * (-c_phase_α)^{1/(2^q)}.

    For q=1 this is r * sqrt(-c_phase). Note: this differs by a phase convention from the
    older q=1 implementation in this repository (which uses sqrt(c_α) with c including r).
    Use this helper consistently with the q-dependent fixed point (Eq. (20), k = 2^q).
    """
    if q < 1:
        raise ValueError("q must be >= 1")
    c_phase = compute_c_alpha_phase(p, gammas)
    k = bm24_clause_arity(q)
    # principal branch of the complex power
    coeff = r * np.power(-c_phase + 0j, 1.0 / float(k))
    return coeff


def softmax_weights_with_coeff(y: np.ndarray, A: np.ndarray, b_s: np.ndarray, coeff_alpha: np.ndarray) -> np.ndarray:
    """
    Generalized weight function for BM24 Eq. (16):

        w_s(y) ∝ b_s * exp(∑_α coeff_α A_{αs} y_α)

    where coeff_α = r * (-c_phase_α)^{1/(2^q)} for q-SAT.
    """
    coeff_alpha = np.asarray(coeff_alpha)
    return softmax_weights(y, A, b_s, coeff_alpha)


def hessian_F_at_y(
    y: np.ndarray,
    A: np.ndarray,
    b_s: np.ndarray,
    coeff_alpha: np.ndarray,
) -> np.ndarray:
    """
    Hessian of F (the log-partition part) for the generalized BM24 setup:

        F(y) = log ∑_s b_s exp(∑_α coeff_α A_{αs} y_α).

    Then:
        ∂_α F = coeff_α E_w[A_α]
        ∂_{α,α'}^2 F = coeff_α coeff_{α'} Cov_w(A_α, A_{α'}).
    """
    d, _ = A.shape
    coeff_alpha = np.asarray(coeff_alpha, dtype=complex)
    if coeff_alpha.shape[0] != d:
        raise ValueError("coeff_alpha has wrong length")

    w = softmax_weights_with_coeff(y, A, b_s, coeff_alpha)
    E_A = w @ A.T
    E_AA = A @ (w[:, None] * A.T)
    Cov = E_AA - np.outer(E_A, E_A)
    return np.outer(coeff_alpha, coeff_alpha) * Cov


def softmax_weights(y: np.ndarray, A: np.ndarray, b_s: np.ndarray, sqrt_c: np.ndarray) -> np.ndarray:
    """
    w_s(y) = b_s * exp(∑_α √c_α A_αs y_α) / Z.
    Use log-sum-exp for numerical stability.
    A is (d, n_s), sqrt_c is (d,), y is (d,). Returns (n_s,) nonnegative and summing to 1.
    """
    # log numerator: log(b_s) + (A.T @ (sqrt_c * y))[s]  -> for each s: log_b[s] + (sqrt_c * y) @ A[:, s]
    # (A.T @ diag(sqrt_c) @ y) = (sqrt_c * y) @ A  per column: A[:, s] . (sqrt_c * y)
    log_b = np.log(b_s + 1e-300 * (1 - np.sign(np.abs(b_s))))
    log_b = np.where(np.isfinite(log_b), log_b, -700.0)
    # linear part: for each s, sum_alpha sqrt_c[alpha] * A[alpha, s] * y[alpha]
    # = (A.T * (sqrt_c * y)[None, :]).sum(axis=1)  but A is (d, n_s), so (sqrt_c * y) @ A gives (n_s,)
    linear = (sqrt_c * y) @ A  # (n_s,)
    log_w = log_b + linear
    log_w = np.where(np.isfinite(log_w), log_w, -700.0)
    shift = np.max(log_w.real)
    log_w = log_w - shift
    w = np.exp(log_w)
    w = w / np.sum(w)
    return w


def hessian_log_at_y(
    y: np.ndarray,
    A: np.ndarray,
    b_s: np.ndarray,
    c_alpha: np.ndarray,
) -> np.ndarray:
    """
    H_log(y) = diag(√c) Cov_{w(y)}(A) diag(√c).

    Cov_αα' = E_w[A_α A_α'] - E_w[A_α] E_w[A_α'].
    w = softmax_weights(y). Returns complex matrix (d, d).
    """
    d, n_s = A.shape
    sqrt_c = np.sqrt(c_alpha + 0j)
    w = softmax_weights(y, A, b_s, sqrt_c)
    # E_w[A_α] = sum_s w_s A[α, s]
    Cov = covariance_from_complex_weights(A, w)
    H_log = np.outer(sqrt_c, sqrt_c) * Cov
    return H_log


def weights_at_zero(b_s: np.ndarray) -> np.ndarray:
    """At y=0, w_s(0) = b_s / ∑_s' b_s'."""
    z = np.sum(b_s)
    if np.abs(z) < 1e-300:
        return np.ones_like(b_s) / len(b_s)
    return b_s / z


def realify_weights(w: np.ndarray, eps: float = 0.0) -> np.ndarray:
    """
    Track B (interpretive proxy): turn complex softmax weights into a real probability vector.

    We take the real part, clip to >= eps, and renormalize to sum 1. If the total mass is
    numerically zero, fall back to uniform weights.
    """
    w = np.real(np.asarray(w).ravel())
    w = np.maximum(w, float(eps))
    total = float(np.sum(w))
    if total < 1e-300:
        return np.ones_like(w, dtype=float) / len(w)
    return w / total


def covariance_from_complex_weights(A: np.ndarray, w: np.ndarray) -> np.ndarray:
    """
    Track A (proof object): Cov_w(A) using the exact (generally complex) saddle softmax weights.

        Cov_αα' = E_w[A_α A_α'] - E_w[A_α] E_w[A_α'].

    This returns a complex matrix in general.
    """
    w = np.asarray(w).ravel()
    E_A = w @ A.T
    E_AA = A @ (w[:, None] * A.T)
    return E_AA - np.outer(E_A, E_A)


def covariance_from_real_weights(A: np.ndarray, w: np.ndarray) -> np.ndarray:
    """
    Track B (interpretive proxy): Cov_w(A) for a real probability vector w (nonnegative, sum 1).

    This returns a real matrix.
    """
    w = np.asarray(w).ravel()
    if np.max(np.abs(np.imag(w))) > 1e-12:
        raise ValueError("covariance_from_real_weights expects real weights (use realify_weights first).")
    w = np.real(w)
    if np.min(w) < -1e-12:
        raise ValueError("covariance_from_real_weights expects nonnegative weights.")
    s = float(np.sum(w))
    if not np.isfinite(s) or abs(s - 1.0) > 1e-6:
        raise ValueError(f"covariance_from_real_weights expects weights summing to 1 (got sum={s}).")
    E_A = w @ A.T
    E_AA = A @ (w[:, None] * A.T)
    return E_AA - np.outer(E_A, E_A)


# Backwards-compatible alias (historical name). Prefer the explicit Track A/B APIs above.
covariance_from_weights = covariance_from_real_weights
