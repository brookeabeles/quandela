"""
Frobenius energy decomposition by subset size (|α|, |α'|) and mechanism isolation.
"""

import numpy as np
from collections import defaultdict
from .core import popcount
from .core import realify_weights


def _popcount_array(d: int, n_bits: int) -> np.ndarray:
    """Array of popcount(alpha) for alpha in [0, d). Vectorized over alpha."""
    # (d,) indices; extract n_bits bits; sum gives popcount
    alpha_grid = np.arange(d, dtype=np.int64)[:, None]
    bits = (alpha_grid >> np.arange(n_bits, dtype=np.int64)) & 1
    return bits.sum(axis=1).astype(np.intp)


def energy_by_block(H_log: np.ndarray, p: int) -> tuple[np.ndarray, np.ndarray]:
    """
    E_{m,m'} = ∑_{|α|=m, |α'|=m'} |H_log[α,α']|².
    Returns (E_mm: (2p+2, 2p+2), f_mm: same shape, fractional contribution).
    Vectorized via precomputed popcount and np.add.at.
    """
    d = H_log.shape[0]
    n = 2 * p + 1
    E_mm = np.zeros((n + 1, n + 1))
    sizes = _popcount_array(d, n)
    vals_sq = np.abs(H_log) ** 2
    row_idx = np.repeat(sizes, d)
    col_idx = np.tile(sizes, d)
    np.add.at(E_mm, (row_idx, col_idx), vals_sq.ravel())
    total = np.sum(E_mm)
    f_mm = E_mm / total if total > 1e-30 else E_mm
    return E_mm, f_mm


def block_norms(H_log: np.ndarray, p: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Per-block (m, m') Frobenius and operator norms for Direction 2 block-structure study.
    Returns (frobenius_mm: (2p+2, 2p+2), operator_mm: same shape).
    Frobenius of block (m,m') = sqrt(E_mm[m,m']). Operator = largest singular value of that block.
    """
    d = H_log.shape[0]
    n = 2 * p + 1
    sizes = _popcount_array(d, n)
    E_mm, _ = energy_by_block(H_log, p)
    frob_mm = np.sqrt(np.maximum(E_mm, 0.0))
    op_mm = np.zeros((n + 1, n + 1))
    for m in range(n + 1):
        for mp in range(n + 1):
            row_idx = np.where(sizes == m)[0]
            col_idx = np.where(sizes == mp)[0]
            if len(row_idx) == 0 or len(col_idx) == 0:
                op_mm[m, mp] = 0.0
                continue
            block = H_log[np.ix_(row_idx, col_idx)]
            s = np.linalg.svd(block, compute_uv=False)
            op_mm[m, mp] = float(s[0]) if len(s) else 0.0
    return frob_mm, op_mm


def energy_marginal_max_size(H_log: np.ndarray, p: int) -> np.ndarray:
    """E_{≥m} = ∑_{max(|α|,|α'|) ≥ m} |H_log[α,α']|². Index m from 0 to 2p+1."""
    E_mm, _ = energy_by_block(H_log, p)
    n = 2 * p + 1
    E_ge = np.zeros(n + 2)
    for m in range(n + 1):
        for mp in range(n + 1):
            m_max = max(m, mp)
            E_ge[m_max] += E_mm[m, mp]
    # E_ge[m] = sum over (m',m'') with max(m',m'') >= m
    cum = np.cumsum(E_ge[::-1])[::-1]
    return cum[: n + 1]


def mechanism_per_alpha(
    p: int,
    A: np.ndarray,
    w: np.ndarray,
    c_alpha: np.ndarray,
) -> tuple[dict, np.ndarray]:
    """
    For each α: |c_α|, w(E_α) = agreement probability, Var_w(A_α).
    Combined factor sqrt(|c_α|) * sqrt(Var_w(A_α)).
    w(E_α) = 2 * E_w[A_α] since A_αs = 1/2 when s constant on α. Var_w(A_α) = (1/4) w(E_α)(1 - w(E_α)).
    Returns (dict with arrays keyed by size, sizes array).
    """
    d, n_s = A.shape
    c_mag = np.abs(c_alpha)
    # Track B (interpretive proxy): physical agreement probability from realified weights
    w_real = realify_weights(w, eps=0.0)
    # E_w[A_α] is then real and in [0, 1/2]
    E_A = w_real @ A.T
    # w(E_α) = 2 * E_w[A_α] in [0, 1]
    w_E_alpha = np.clip(2.0 * E_A, 0.0, 1.0)
    # Var_w(A_α) = (1/4) * w(E_α) * (1 - w(E_α))
    Var_A = 0.25 * w_E_alpha * (1.0 - w_E_alpha)
    combined = np.sqrt(c_mag + 1e-300) * np.sqrt(Var_A + 1e-300)

    by_size = defaultdict(list)
    for alpha in range(d):
        m = popcount(alpha)
        if m >= 2:
            by_size[m].append((c_mag[alpha], w_E_alpha[alpha], Var_A[alpha], combined[alpha]))
    # mean by size
    sizes = sorted(by_size.keys())
    mean_c = np.array([np.mean([x[0] for x in by_size[m]]) for m in sizes])
    mean_wE = np.array([np.mean([x[1] for x in by_size[m]]) for m in sizes])
    mean_var = np.array([np.mean([x[2] for x in by_size[m]]) for m in sizes])
    mean_combined = np.array([np.mean([x[3] for x in by_size[m]]) for m in sizes])
    return {
        "sizes": np.array(sizes),
        "mean_c_mag": mean_c,
        "mean_w_E_alpha": mean_wE,
        "mean_Var_A": mean_var,
        "mean_combined": mean_combined,
        "c_mag_all": c_mag,
        "w_E_alpha_all": w_E_alpha,
        "Var_A_all": Var_A,
        "combined_all": combined,
    }, np.array(sizes)


def phase_bound_Gamma(gamma: float) -> float:
    """Γ = 2|sin(γ/4)|. Upper bound: |c_α| ≤ Γ^{|α|-1}."""
    return 2.0 * np.abs(np.sin(gamma / 4))


def analytical_bounds(p: int, gamma: float) -> tuple[np.ndarray, np.ndarray]:
    """Phase bound: Γ^{m-1} for m≥1. Agreement bound: 2^{1-m} for m≥1."""
    Gamma = phase_bound_Gamma(gamma)
    sizes = np.arange(1, 2 * p + 2)
    phase_bound = np.power(Gamma, np.maximum(sizes - 1, 0))
    agreement_bound = np.power(0.5, np.maximum(sizes - 1, 0))  # 2^{1-m}
    return phase_bound, agreement_bound
