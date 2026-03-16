"""
Spectral analysis of H_log: SVD, Frobenius/spectral norms, stable rank, energy capture.

Stable rank universality (Direction 1): r_s = ‖H‖_F²/‖H‖² is conjectured ≤ C (independent
of p and γ); see STABLE_RANK_THEOREM.md for the theorem statement and proof strategy.
"""

import numpy as np

EPS_MACHINE = 1e-12


def singular_values(H: np.ndarray) -> np.ndarray:
    """Singular values of H (descending). H may be complex; use SVD."""
    return np.linalg.svd(H, compute_uv=False)


def frobenius_norm(H: np.ndarray) -> float:
    """‖H‖_F = sqrt(∑_ij |H_ij|²)."""
    return float(np.sqrt(np.sum(np.abs(H) ** 2)))


def spectral_norm(H: np.ndarray) -> float:
    """‖H‖ = σ_1 (largest singular value)."""
    sigmas = singular_values(H)
    return float(sigmas[0]) if len(sigmas) else 0.0


def stable_rank(H: np.ndarray) -> float:
    """r_s = ‖H‖_F² / ‖H‖². Universality: r_s ≤ C for H_log(0) (see STABLE_RANK_THEOREM.md)."""
    s1 = spectral_norm(H)
    if s1 < 1e-300:
        return 0.0
    return frobenius_norm(H) ** 2 / (s1 ** 2)


def exact_rank(H: np.ndarray, eps: float = EPS_MACHINE) -> int:
    """Number of singular values > eps."""
    sigmas = singular_values(H)
    return int(np.sum(sigmas > eps))


def cumulative_energy_fraction(sigmas: np.ndarray) -> np.ndarray:
    """E(k) = (∑_{i=1}^k σ_i²) / ‖H‖_F² for k = 1..d. Returns array of length d."""
    sigmas = np.asarray(sigmas)
    total_sq = np.sum(sigmas ** 2)
    if total_sq < 1e-30:
        return np.ones_like(sigmas, dtype=float)
    cum_sq = np.cumsum(sigmas ** 2)
    return cum_sq / total_sq


def effective_dimensions(
    sigmas: np.ndarray,
    eta_values: tuple[float, ...] = (0.9, 0.95, 0.99, 0.999),
) -> dict[float, int]:
    """k_η = smallest k such that E(k) ≥ η. Returns {η: k_η}."""
    E = cumulative_energy_fraction(sigmas)
    out = {}
    for eta in eta_values:
        idx = np.searchsorted(E, eta)
        k = min(idx + 1, len(sigmas))
        out[eta] = k
    return out


def tail_mass_fraction(sigmas: np.ndarray, k: int) -> float:
    """
    Fraction of Frobenius energy in the tail beyond k: 1 - E(k).
    Relevant if saddle-point error scales like sum_{j>k} σ_j² (favors stable rank story).
    """
    sigmas = np.asarray(sigmas)
    total_sq = np.sum(sigmas ** 2)
    if total_sq < 1e-30 or k >= len(sigmas):
        return 0.0
    head_sq = np.sum(sigmas[:k] ** 2)
    return float(1.0 - head_sq / total_sq)


def spectral_tail_ratio(sigmas: np.ndarray, k: int) -> float:
    """
    σ_{k+1} / σ_1 (first dropped singular value, normalized).
    Relevant if saddle-point error scales like max_{j>k} σ_j (e.g. inverse approximation; favors k_99).
    """
    sigmas = np.asarray(sigmas)
    if len(sigmas) == 0 or sigmas[0] < 1e-300:
        return 0.0
    if k >= len(sigmas):
        return 0.0
    return float(sigmas[k] / sigmas[0])


def determinant_ratio_probe(
    H: np.ndarray,
    tol: float = 1e-12,
    tol_denom: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Determinant-ratio probe for saddle-point integral truncation error.
    H_log is complex symmetric; ∇²F = -I/2 + H_log. Get eigenvalues λ_k of H_log
    via eig (not SVD). Sort by |λ_k| descending; restrict to non-null (|λ_k| > tol).
    R(k₀) = ∏_{k>k₀} (½) / (½ − λ_k) = ratio of truncated Gaussian integral (treating
    directions beyond k₀ as free) to full. Returns (k0_axis, err_array) where
    err_array[i] = |R(i)| − 1 (relative error when truncating at k₀ = i).
    Null directions (|λ| ≤ tol) and near-singular (|½−λ| < tol_denom) are excluded
    from the product so R is well-defined.
    """
    lam, _ = np.linalg.eig(H)
    lam = np.asarray(lam)
    # Sort by |λ| descending (match SVD ordering)
    idx = np.argsort(np.abs(lam))[::-1]
    lam = lam[idx]
    # Restrict to non-null and safe denominator
    mask = (np.abs(lam) > tol) & (np.abs(0.5 - lam) > tol_denom)
    lam = lam[mask]
    n = len(lam)
    if n == 0:
        return np.arange(0, dtype=float), np.zeros(0, dtype=float)
    # R(k0) = ∏_{k>k0} (0.5)/(0.5 - lam[k]); err[k0] = |R(k0)| - 1
    # Compute cumulative product from the tail: for k0 = 0..n-1, R(k0) = prod over k in [k0+1, n)
    err = np.zeros(n)
    for k0 in range(n):
        # product over k = k0+1 .. n-1
        r = 1.0 + 0.0j
        for k in range(k0 + 1, n):
            r *= 0.5 / (0.5 - lam[k])
        err[k0] = float(np.abs(r) - 1.0)
    k0_axis = np.arange(n, dtype=float)
    return k0_axis, err


def determinant_ratio_and_tail_metrics(
    H: np.ndarray,
    tol: float = 1e-12,
    tol_denom: float = 1e-10,
) -> dict:
    """
    Determinant-ratio and spectral tail metrics for saddle truncation analysis (Direction 1).

    Returns eigenvalues λ_j (sorted by |λ_j| desc), and for each truncation level k₀:
    - R(k₀): ratio ∏_{j>k₀} (½)/(½ − λ_j)
    - log_R(k₀): log R(k₀) = Σ_{j>k₀} −log(1 − 2λ_j)
    - err(k₀): |R(k₀) − 1|
    - nuclear_tail: Σ_{j>k₀} |λ_j|
    - frobenius_tail: Σ_{j>k₀} |λ_j|²
    - operator_tail: max_{j>k₀} |λ_j|

    Used to test which spectral tail quantity controls |R(k₀) − 1|.
    """
    lam, _ = np.linalg.eig(H)
    lam = np.asarray(lam)
    idx = np.argsort(np.abs(lam))[::-1]
    lam = lam[idx]
    mask = (np.abs(lam) > tol) & (np.abs(0.5 - lam) > tol_denom)
    lam = lam[mask]
    n = len(lam)
    if n == 0:
        return {
            "eigenvalues": np.array([], dtype=complex),
            "k0_axis": np.array([], dtype=float),
            "R": np.array([]),
            "log_R": np.array([]),
            "err": np.array([]),
            "nuclear_tail": np.array([]),
            "frobenius_tail": np.array([]),
            "operator_tail": np.array([]),
        }
    k0_axis = np.arange(n, dtype=float)
    R_arr = np.zeros(n, dtype=complex)
    log_R_arr = np.zeros(n, dtype=float)
    for k0 in range(n):
        r = 1.0 + 0.0j
        log_r = 0.0
        for k in range(k0 + 1, n):
            factor = 0.5 / (0.5 - lam[k])
            r *= factor
            log_r += np.log(factor)
        R_arr[k0] = r
        log_R_arr[k0] = np.real(log_r)
    err = np.abs(R_arr - 1.0)
    # Tail metrics: for index k0 we keep directions 0..k0, tail is j in [k0+1, n)
    nuclear_tail = np.zeros(n)
    frobenius_tail = np.zeros(n)
    operator_tail = np.zeros(n)
    abs_lam = np.abs(lam)
    for k0 in range(n):
        if k0 + 1 >= n:
            nuclear_tail[k0] = 0.0
            frobenius_tail[k0] = 0.0
            operator_tail[k0] = 0.0
        else:
            tail_lam = abs_lam[k0 + 1 :]
            nuclear_tail[k0] = float(np.sum(tail_lam))
            frobenius_tail[k0] = float(np.sum(tail_lam**2))
            operator_tail[k0] = float(np.max(tail_lam))
    return {
        "eigenvalues": lam,
        "k0_axis": k0_axis,
        "R": R_arr,
        "log_R": log_R_arr,
        "err": err,
        "nuclear_tail": nuclear_tail,
        "frobenius_tail": frobenius_tail,
        "operator_tail": operator_tail,
    }


def approximation_error_proxies(
    sigmas: np.ndarray,
    k_99: int,
    stable_rank_val: float,
) -> dict[str, float]:
    """
    Proxies for saddle-point approximation error when truncating to k_99 vs k ≈ stable_rank.
    Used to investigate whether k_99 or stable rank controls actual approximation error
    (Direction 1 in QAOA Hessian spectral findings).

    Returns:
        tail_mass_at_k99: 1 - E(k_99) (by definition ≈ 0.01)
        tail_mass_at_k_sr: 1 - E(k_sr) when k_sr = round(stable_rank)
        spectral_tail_at_k99: σ_{k_99+1} / σ_1
        spectral_tail_at_k_sr: σ_{k_sr+1} / σ_1 when k_sr = round(stable_rank)
    """
    sigmas = np.maximum(np.asarray(sigmas), 0.0)
    k_sr = max(1, int(round(stable_rank_val)))
    k_sr = min(k_sr, len(sigmas) - 1) if len(sigmas) > 1 else 0
    k99_safe = min(k_99, len(sigmas) - 1) if len(sigmas) > 0 else 0
    return {
        "tail_mass_at_k99": tail_mass_fraction(sigmas, k_99),
        "tail_mass_at_k_sr": tail_mass_fraction(sigmas, k_sr),
        "spectral_tail_at_k99": spectral_tail_ratio(sigmas, k99_safe),
        "spectral_tail_at_k_sr": spectral_tail_ratio(sigmas, k_sr),
    }


def kept_dropped_alpha_analysis(
    H: np.ndarray,
    k: int,
    p: int,
    top_n_alpha: int = 20,
):
    """
    Which α (parameter dimensions) contribute to the kept vs dropped subspace when
    truncating to the top k singular directions.

    Performs full SVD (compute_uv=True). For each α, computes contribution to the
    kept subspace (first k right singular vectors) and to the dropped subspace (rest).
    Returns block-level summary by |α| and optional top-α indices for inspection.

    Returns:
        dict with:
        - block_weights_kept: (2p+2,) fraction of kept-subspace energy in each |α| block
        - block_weights_dropped: (2p+2,) same for dropped subspace
        - top_alpha_kept: length-min(top_n_alpha, d) indices α ordered by contribution to kept
        - top_alpha_dropped: same for dropped
        - alpha_weights_kept: (d,) per-α weight in kept subspace (only if d <= 2^14 to limit memory)
    """
    from .mechanisms import _popcount_array

    d = H.shape[0]
    n_bits = 2 * p + 1
    k = min(max(0, k), d)
    # Full SVD; V is (d, d), columns are right singular vectors
    U, sigmas, Vh = np.linalg.svd(H, full_matrices=True)
    V = Vh.T  # (d, d), V[:, j] = j-th right singular vector
    # weight_kept[α] = sum_{j=0}^{k-1} |V[α,j]|^2
    kept = np.abs(V[:, :k]) ** 2
    weight_kept = np.sum(kept, axis=1)
    weight_dropped = np.sum(np.abs(V[:, k:]) ** 2, axis=1)
    total_kept = np.sum(weight_kept)
    total_dropped = np.sum(weight_dropped)
    if total_kept < 1e-30:
        total_kept = 1.0
    if total_dropped < 1e-30:
        total_dropped = 1.0
    weight_kept = weight_kept / total_kept
    weight_dropped = weight_dropped / total_dropped

    sizes = _popcount_array(d, n_bits)  # (d,) with values in [0, 2p+1]
    n_blocks = n_bits + 1
    block_weights_kept = np.zeros(n_blocks)
    block_weights_dropped = np.zeros(n_blocks)
    for m in range(n_blocks):
        mask = sizes == m
        block_weights_kept[m] = np.sum(weight_kept[mask])
        block_weights_dropped[m] = np.sum(weight_dropped[mask])

    top_alpha_kept = np.argsort(weight_kept)[::-1][:top_n_alpha]
    top_alpha_dropped = np.argsort(weight_dropped)[::-1][:top_n_alpha]

    out = {
        "block_weights_kept": block_weights_kept,
        "block_weights_dropped": block_weights_dropped,
        "top_alpha_kept": top_alpha_kept,
        "top_alpha_dropped": top_alpha_dropped,
    }
    if d <= 2**14:
        out["alpha_weights_kept"] = weight_kept
        out["alpha_weights_dropped"] = weight_dropped
    return out


def effective_dimension_block_structure(H: np.ndarray, k: int, p: int):
    """
    Structure and patterns within the effective-dimension matrix (rank-k approximation).

    The "effective" matrix is H_eff = U[:,:k] @ diag(σ_1..σ_k) @ Vh[:k,:], i.e. the
    best rank-k approximation in Frobenius norm. This routine returns:

    1. Block (|α|, |α'|) structure of H_eff: where the 99% Frobenius mass sits in
       subset-size space.
    2. Per-singular-vector block composition: for each of the k kept directions,
       how much of that direction's mass lies in each block |α| (so patterns across
       singular vectors).

    Returns:
        dict with:
        - f_mm_eff: (2p+2, 2p+2) fractional Frobenius energy of H_eff in each (m, m') block
        - E_mm_eff: same shape, raw squared Frobenius in each block
        - per_vector_block: (k, 2p+2) for each j in 0..k-1, fraction of |v_j|² in each block m
        - diagonal_fraction: fraction of f_mm_eff on the diagonal (m=m')
        - off_diagonal_fraction: fraction on off-diagonal
    """
    from .mechanisms import _popcount_array, energy_by_block

    d = H.shape[0]
    n_bits = 2 * p + 1
    k = min(max(0, k), d)
    U, sigmas, Vh = np.linalg.svd(H, full_matrices=False)
    # H_eff = U[:,:k] @ diag(s[:k]) @ Vh[:k,:]
    H_eff = (U[:, :k] * sigmas[:k]) @ Vh[:k, :]
    E_mm_eff, f_mm_eff = energy_by_block(H_eff, p)

    sizes = _popcount_array(d, n_bits)
    n_blocks = n_bits + 1
    per_vector_block = np.zeros((k, n_blocks))
    V = Vh.T  # (d, k) if full_matrices=False then Vh is (k, d), V is (d, k)
    for j in range(k):
        w = np.abs(V[:, j]) ** 2
        total = np.sum(w)
        if total < 1e-30:
            total = 1.0
        for m in range(n_blocks):
            per_vector_block[j, m] = np.sum(w[sizes == m]) / total

    diag = np.trace(f_mm_eff)
    off_diag = float(np.sum(f_mm_eff) - diag)
    total_f = float(np.sum(f_mm_eff))
    if total_f < 1e-30:
        total_f = 1.0
    return {
        "f_mm_eff": f_mm_eff,
        "E_mm_eff": E_mm_eff,
        "per_vector_block": per_vector_block,
        "diagonal_fraction": diag / total_f,
        "off_diagonal_fraction": off_diag / total_f,
    }


def spectral_summary(H: np.ndarray, eta_values=(0.9, 0.95, 0.99, 0.999)):
    """
    Return dict with: singular_values, frobenius_norm, spectral_norm, stable_rank,
    exact_rank, k_90, k_95, k_99, k_999, cumulative_energy (E(k) array).
    """
    sigmas = singular_values(H)
    sigmas = np.maximum(sigmas, 0.0)
    k_eta = effective_dimensions(sigmas, eta_values=eta_values)
    E = cumulative_energy_fraction(sigmas)
    return {
        "singular_values": sigmas,
        "frobenius_norm": frobenius_norm(H),
        "spectral_norm": spectral_norm(H),
        "stable_rank": stable_rank(H),
        "exact_rank": exact_rank(H),
        "k_90": k_eta.get(0.9),
        "k_95": k_eta.get(0.95),
        "k_99": k_eta.get(0.99),
        "k_999": k_eta.get(0.999),
        "cumulative_energy": E,
        **{f"k_{int(eta*100)}": k_eta.get(eta) for eta in eta_values},
    }
