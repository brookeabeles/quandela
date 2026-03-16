#!/usr/bin/env python3
"""
Symmetry Reduction of the QAOA Action for k-SAT
================================================

IMPORTANT — Relation to full BM24 A36:
--------------------------------------
This script uses a SIMPLIFIED single-variable QAOA model:

  - Trajectory space: S = {0,1}^{p+1}  (one bit per layer, p+1 layers)
  - Interaction set:  |A| = p  (one channel per QAOA round)
  - Feature matrix:  V[s, j] = √c_j · (1 - s_j), so rank(V) = p

In this simplified model, the effective dimension is p and G(y) can be
computed in O(p) via a 2×2 transfer matrix — so the script's findings
(rank = p, O(p) saddle, transfer-matrix equivalence) are correct
*for this model*.

The FULL BM24 A36 (Example 1, PRX Quantum 5, 030348) uses:

  - S = {0,1}^{2p+1}   (trajectories of length 2p+1)
  - A = { J ⊆ [2p+1] : |J| ≥ 2 }
  - A_{αs} = (1/2) · 1[s constant on α]

There, |A| = 2^{2p+1} - 2p - 2 is EXPONENTIAL; there is NO O(p) dimension
collapse (each α has a distinct feature vector). BM24 nevertheless evaluate
the saddle map in O(p·4^p) time via sum-over-subsets DP, without enumerating
|A|. See BM24_A36_symmetry_reduction.md for the full derivation and
saddle equations.

Investigation (this script):
  Does the action F: C^{|A|} -> C have effective dimension O(p)
  in the SIMPLIFIED model (|S|=2^{p+1}, |A|=p)?

  F(y) = -1/4 ||y||^2 + log G(y)
  G(y) = sum_s b_s exp(v_s . y)
  v_s = (sqrt(c_alpha) * A_{alpha,s})_alpha

  Effective dimension = rank of the feature matrix V (rows v_s).
  In the simplified model: rank(V) = p.
"""

import numpy as np
from itertools import product as iterproduct
import warnings
warnings.filterwarnings('ignore')

def _svdvals(M):
    """Singular values of M; uses numpy so scipy is optional."""
    return np.linalg.svd(M, compute_uv=False)


# ============================================================
# TRUE BM24 feature matrix and Hessian (single pipeline)
# ============================================================
#
# Two index-set conventions:
# - BM24 Example 1 (A32): A = { J ⊆ [2p+1] : |J| ≥ 2 },  |A| = 2^{2p+1} - 2p - 2
# - Hessian paper / full: A = 2^{[2p+1]} (all subsets),   |A| = 2^{2p+1}
# Subsets with |J| ≤ 1 give constant rows A_{αs} = 1/2 for all s, so zero covariance;
# they don't affect H_log spectrum but change the printed dimension.
#

def _all_subsets(n):
    """All subsets of [n] = {0,...,n-1}. Order: by size then lex. Size 2^n."""
    from itertools import combinations
    return [frozenset(c) for size in range(n + 1) for c in combinations(range(n), size)]


def _all_subsets_size_at_least_2(n):
    """Subsets of [n] with |J| >= 2. BM24 Example 1 (A32)."""
    from itertools import combinations
    return [frozenset(c) for size in range(2, n + 1) for c in combinations(range(n), size)]


def _bitstring_to_tuple(b, n):
    """Integer b in [0, 2^n) -> tuple of n bits (s_0, ..., s_{n-1}), s_j = (b >> j) & 1."""
    return tuple((b >> j) & 1 for j in range(n))


def _is_constant_on(s_tuple, alpha):
    """True iff s_tuple[j] is the same for all j in alpha. Empty alpha -> True."""
    if not alpha:
        return True
    vals = [s_tuple[j] for j in alpha]
    return all(v == vals[0] for v in vals)


def compute_b_s_BM24(p, betas):
    """
    BM24 mixing coefficients b_s for all s in {0,1}^{2p+1} (Example 1, Eq. (A5)).

    b_s = (1/2) * prod_{j=0}^{p-1} <s_j|e^{i*beta_j*X}|s_{j+1}> *
                         <s_{2p-j-1}|e^{-i*beta_j*X}|s_{2p-j}>

    with <s'|e^{i beta X}|s> = cos(beta) if s' = s,  i sin(beta) otherwise.

    Returns complex array of shape (2^{2p+1},).
    """
    import numpy as np  # local to avoid confusion with callers

    n = 2 * p + 1
    n_s = 1 << n
    betas = np.asarray(betas, dtype=complex)
    if betas.size < p:
        betas = np.resize(betas, p)
    betas = betas[:p]

    b = np.zeros(n_s, dtype=complex)
    for s_int in range(n_s):
        # bits s_0,...,s_{2p}
        s_bits = tuple((s_int >> j) & 1 for j in range(n))
        val = 0.5 + 0j
        for j in range(p):
            beta_j = betas[j]
            # forward factor <s_j | e^{i beta_j X} | s_{j+1}>
            if s_bits[j] == s_bits[j + 1]:
                val *= np.cos(beta_j)
            else:
                val *= 1j * np.sin(beta_j)
            # backward factor <s_{2p-j-1} | e^{-i beta_j X} | s_{2p-j}>
            if s_bits[2 * p - j - 1] == s_bits[2 * p - j]:
                val *= np.cos(beta_j)
            else:
                val *= -1j * np.sin(beta_j)
        b[s_int] = val

    # Normalize so that sum_s b_s = 1 (BM24 prior)
    Z = np.sum(b)
    if np.abs(Z) > 1e-300:
        b = b / Z
    return b


def compute_c_alpha_BM24(list_alpha, p, gammas, r=1.0):
    """
    BM24 (A34): c_α = r (-1)^{1[p∈α]} ∏_{j∈α,j<p}(e^{-iγ_j/2}-1) ∏_{j∈α,j>p}(e^{iγ_{2p-j}/2}-1).
    Index p contributes only the sign; no element-wise factor for j=p.
    Returns array of length len(list_alpha). Used by build and Hessian (no recovery from V).
    """
    gammas = np.asarray(gammas, dtype=complex)
    n_alpha = len(list_alpha)
    c_alpha = np.zeros(n_alpha, dtype=complex)
    for idx_alpha, alpha in enumerate(list_alpha):
        sign = -1.0 if p in alpha else 1.0
        prod = r * sign
        for j in alpha:
            if j < p:
                prod *= (np.exp(-1j * gammas[j] / 2) - 1)
            elif j > p:
                prod *= (np.exp(1j * gammas[2*p - j] / 2) - 1)
        c_alpha[idx_alpha] = prod
    return c_alpha


def build_BM24_true_feature_matrix(p, gammas=None, r=1.0, index_set="full"):
    """
    TRUE BM24 feature matrix (Eq. A31, A33–A35). c_α from (A34) via compute_c_alpha_BM24.

    index_set: "full" -> A = 2^{[2p+1]}, |A| = 2^{2p+1} (Hessian paper convention).
               "J>=2"  -> A = {J : |J| ≥ 2}, |A| = 2^{2p+1} - 2p - 2 (BM24 Example 1 A32).

    Returns: V (n_s × n_alpha), list_alpha, c_alpha (so Hessian code never recovers c from V).
    """
    n = 2 * p + 1
    if gammas is None:
        gammas = np.ones(p) * 0.5
    gammas = np.asarray(gammas, dtype=complex)
    assert len(gammas) == p

    list_alpha = _all_subsets(n) if index_set == "full" else _all_subsets_size_at_least_2(n)
    n_s = 2**n
    n_alpha = len(list_alpha)

    c_alpha = compute_c_alpha_BM24(list_alpha, p, gammas, r)

    V = np.zeros((n_s, n_alpha), dtype=complex)
    for s_int in range(n_s):
        s_tuple = _bitstring_to_tuple(s_int, n)
        for idx_alpha, alpha in enumerate(list_alpha):
            if _is_constant_on(s_tuple, alpha):
                sqrt_c = np.sqrt(c_alpha[idx_alpha] + 0j)
                V[s_int, idx_alpha] = 0.5 * sqrt_c
    return V, list_alpha, c_alpha


def analyze_true_BM24_rank(p_max=4, gammas=None, index_set="full"):
    """
    Build TRUE BM24 feature matrix and report rank.
    index_set: "full" -> |A| = 2^{2p+1}; "J>=2" -> |A| = 2^{2p+1} - 2p - 2 (BM24 A32).
    """
    label = "A = 2^{[2p+1]}" if index_set == "full" else "A = {J : |J|≥2}"
    print("=" * 70)
    print(f"TRUE BM24 FEATURE MATRIX — {label}")
    print("=" * 70)
    print()
    print(f"{'p':>2} | {'|S|=2^(2p+1)':>14} | {'|A|':>8} | {'shape':>14} | {'rank(V)':>8}")
    print("-" * 70)

    for p in range(1, p_max + 1):
        V, list_alpha, c_alpha = build_BM24_true_feature_matrix(p, gammas=gammas, index_set=index_set)
        n_s, n_alpha = V.shape
        sv = _svdvals(V)
        tol = 1e-10 * (sv[0] if sv[0] > 1e-15 else 1.0)
        rank = int(np.sum(sv > tol))
        print(f"{p:>2} | {n_s:>14} | {n_alpha:>8} | {str(V.shape):>14} | {rank:>8}")

    print()
    print("Conclusion: rank(V) = 2^{2p} (exponential in p, not O(p)).")


def build_BM24_A_matrix(p, list_alpha=None):
    """Build the 0/1 constancy matrix A[s,α] = (1/2)*1[s constant on α]. Shape (n_s, n_alpha)."""
    n = 2 * p + 1
    n_s = 2**n
    if list_alpha is None:
        list_alpha = _all_subsets_size_at_least_2(n)
    n_alpha = len(list_alpha)
    A = np.zeros((n_s, n_alpha))
    for s_int in range(n_s):
        s_tuple = _bitstring_to_tuple(s_int, n)
        for idx_alpha, alpha in enumerate(list_alpha):
            if _is_constant_on(s_tuple, alpha):
                A[s_int, idx_alpha] = 0.5
    return A


def hessian_Phi_at_y(p, y, V, c_alpha, b_s):
    """
    Hessian of Φ(y) = -¼|y|² + log ∑_s b_s exp(⟨v_s,y⟩) w.r.t. y (complex).

    ∂²Φ/∂y_α ∂y_β = -½ δ_αβ + √(c_α c_β) * Cov_μ(A_αs, A_βs)
    with μ_y(s) = b_s exp(⟨v_s,y⟩) / Z(y), and A_αs = (1/2)*1[s const on α].
    """
    n_alpha = len(c_alpha)
    n_s = len(b_s)
    # μ_y(s) = b_s * exp(V[s,:] @ y) / Z
    log_weights = V @ np.asarray(y, dtype=complex)
    w = b_s * np.exp(log_weights)
    Z = np.sum(w)
    if np.abs(Z) < 1e-300:
        return np.eye(n_alpha) * (-0.5)  # fallback
    mu = w / Z
    # E_μ[A_αs] = sum_s μ(s) A_αs; we need A matrix: A[s,α] = (1/2)*1[s const on α]
    # V[s,α] = √c_α A_αs  =>  A[s,α] = V[s,α] / √c_α (for c_α ≠ 0). Safer: pass A or recompute.
    # Build A from V: A[s,α] = V[s,α] / sqrt(c_alpha[α]), but V[s,α] can be 0.
    sqrt_c = np.sqrt(c_alpha + 0j)
    A = np.zeros((n_s, n_alpha), dtype=complex)
    for idx_alpha in range(n_alpha):
        if np.abs(sqrt_c[idx_alpha]) > 1e-15:
            A[:, idx_alpha] = V[:, idx_alpha] / sqrt_c[idx_alpha]
        # else column stays 0
    E_A = mu @ A   # (n_alpha,)  E_μ[A_αs]
    # E_μ[A_αs A_βs]: A_αs A_βs = 1/4 if s const on α and β, else 0. So (A A^T)[s,s] not what we need.
    # (A @ A.T) would be sum_α A[s,α]^2. We need E_μ[A_αs A_βs] = sum_s μ(s) A[s,α] A[s,β].
    # So E_AA = A.T @ (mu * A)  gives (A.T @ diag(mu) @ A)[α,β] = sum_s μ(s) A[s,α] A[s,β].
    E_AA = (A.T * mu) @ A   # (n_alpha, n_alpha)
    Cov = E_AA - np.outer(E_A, E_A)
    sqrt_c_outer = np.outer(sqrt_c, sqrt_c)
    H = -0.5 * np.eye(n_alpha) + sqrt_c_outer * Cov
    return H


def effective_dimension_H_log_frobenius(H_log, eta_values=(0.9, 0.95, 0.99)):
    """
    H_log = ∇²Φ + ½I = diag(√c) Cov_μ(A) diag(√c). Effective dimension by Frobenius energy.

    SVD of H_log → singular values σ_1 ≥ σ_2 ≥ ... . Frobenius norm² = ∑ σ_i².
    k_η = smallest k such that ∑_{i=1}^k σ_i² ≥ η * ∑_i σ_i² (fraction of Frobenius norm).
    Returns: dict {η: k_η}, and sorted singular values (descending).
    """
    sigmas = _svdvals(H_log)  # already nonnegative, descending
    sigmas = np.maximum(sigmas, 0.0)
    total_sq = np.sum(sigmas**2)
    if total_sq < 1e-30:
        return {e: 0 for e in eta_values}, sigmas
    cum_sq = np.cumsum(sigmas**2)
    out = {}
    for eta in eta_values:
        k = np.searchsorted(cum_sq, eta * total_sq) + 1
        out[eta] = min(k, len(sigmas))
    return out, sigmas


def analyze_true_BM24_hessian(p_max=4, at_y_zero=True, index_set="full", eta_values=(0.9, 0.95, 0.99)):
    """
    Single pipeline: build A, c_α, V; Hessian at y; H_log = H + ½I; SVD(H_log); k_η by Frobenius.

    Reproduces Tables 3–5 style: k_90%%, k_95%%, k_99%% = # singular values to capture that
    fraction of ‖H_log‖_F². Uses explicit c_α from compute_c_alpha_BM24 (no recovery from V).
    index_set: "full" (d = 2^{2p+1}) or "J>=2".
    """
    print("=" * 72)
    print("HESSIAN EFFECTIVE DIMENSION — H_log = ∇²Φ + ½I, SVD, k_η by Frobenius")
    print("=" * 72)
    print()
    print("Φ(y) = -¼|y|² + log ∑_s b_s exp(⟨v_s,y⟩). H = -½I + √c Cov √c, H_log = H + ½I = √c Cov √c.")
    print("k_η = smallest k s.t. sum of top-k σ_i² ≥ η × ‖H_log‖_F².")
    print()
    # Header: p | |A| | k_90% | k_95% | k_99% | ‖H_log‖_F | σ_1
    eta_str = " ".join(f"k_{int(e*100)}%" for e in eta_values)
    print(f"{'p':>2} | {'|A|':>6} | {eta_str} | {'‖H_log‖_F':>10} | {'σ_1':>10}")
    print("-" * 72)

    gammas = np.ones(p_max) * 0.5
    betas = np.ones(p_max) * (np.pi / 4)
    for p in range(1, p_max + 1):
        V, list_alpha, c_alpha = build_BM24_true_feature_matrix(p, gammas=gammas[:p], index_set=index_set)
        n_s, n_alpha = V.shape
        b_s = compute_b_s_BM24(p, betas[:p])
        y0 = np.zeros(n_alpha, dtype=complex) if at_y_zero else (np.random.randn(n_alpha) + 1j * np.random.randn(n_alpha)) * 0.01
        H = hessian_Phi_at_y(p, y0, V, c_alpha, b_s)
        H_log = H + 0.5 * np.eye(n_alpha)
        k_eta, sigmas = effective_dimension_H_log_frobenius(H_log, eta_values=eta_values)
        frob = np.sqrt(np.sum(sigmas**2))
        sigma_1 = float(sigmas[0]) if len(sigmas) else 0.0
        k_parts = " ".join(f"{k_eta[e]:>6}" for e in eta_values)
        print(f"{p:>2} | {n_alpha:>6} | {k_parts} | {frob:>10.4f} | {sigma_1:>10.4f}")

    print()
    print("k_η: # singular values of H_log needed to capture η of Frobenius norm².")


# ============================================================
# Part 1: QAOA single-variable evolution at depth p
# ============================================================

def mixer_matrix(beta):
    """2x2 mixer unitary e^{-i beta X/2} in computational basis."""
    c, s = np.cos(beta/2), -1j * np.sin(beta/2)
    return np.array([[c, s], [s, c]])

def phase_matrix(gamma, k):
    """
    (Unused: simplified model encodes phase via c_j in the feature matrix, not this helper.)
    Phase contribution from one clause on one variable.
    For a k-SAT clause, the phase separator acts as:
      e^{-i gamma C_clause} where C_clause is the clause projector.

    A variable in the clause gets phase e^{-i gamma / 2^k} when it's in
    the "wrong" state (violating configuration for that literal).
    After averaging over the other k-1 variables uniformly:
      - with prob 1/2^{k-1}, ALL other variables violate -> full clause violation
      - the net effect on one variable depends on the fraction of
        configurations of other variables that lead to violation

    For the single-variable reduction (BM24), the key is the
    "effective single-variable phase" which has the form:
      |0> -> e^{-i gamma * f_0} |0>
      |1> -> e^{-i gamma * f_1} |1>
    where f_0, f_1 depend on how the variable's state affects
    clause satisfaction.

    For a random clause where the variable appears positive:
      f_1 = 0 (variable = 1 satisfies the literal, clause has fewer violations)
      f_0 = 1/2^k (variable = 0 means the literal is violated; clause is
             violated when ALL k literals are violated, prob 1/2^{k-1})
    """
    # Phase factors for the two basis states
    # Using BM24's convention (Appendix A, equation before A16)
    phase_0 = np.exp(-1j * gamma / (2**k))  # variable in "violating" state
    phase_1 = 1.0                             # variable in "satisfying" state
    return np.array([[phase_0, 0], [0, phase_1]])


def all_trajectories(p):
    """
    Generate all 2^{p+1} binary trajectories s = (s_0, s_1, ..., s_p).

    SIMPLIFIED MODEL: We track state only at "mixer" steps (where basis
    can change). So trajectory length = p+1, not 2p+1 as in full BM24.

    Convention: s_j = computational basis state after the j-th mixer.
    """
    L = p + 1
    return list(iterproduct([0, 1], repeat=L))


def compute_amplitude_b(s, betas):
    """
    Compute the mixer amplitude b_s for trajectory s = (s_0, ..., s_p).

    b_s = (1/2) * prod_{j=1}^{p} <s_j| e^{-i beta_j X/2} |s_{j-1}>

    The 1/2 comes from the initial uniform superposition <+|s_0> = 1/sqrt(2)
    and the measurement <s_p|...> contributing another 1/sqrt(2).
    """
    p = len(betas)
    assert len(s) == p + 1

    amp = 0.5  # from |+> initial state decomposition
    for j in range(p):
        M = mixer_matrix(betas[j])
        amp *= M[s[j+1], s[j]]
    return amp


def compute_phase_factor(s, gammas, k):
    """
    Compute the phase factor for trajectory s from the phase separators.

    For each round j, the variable gets a phase depending on its state s_j.
    Returns the log of the phase factor, decomposed into components
    that will form the feature vectors.
    """
    p = len(gammas)
    phases = []
    for j in range(p):
        gamma_j = gammas[j]
        if s[j] == 0:  # "violating" state for this literal
            phases.append(-1j * gamma_j / (2**k))
        else:  # "satisfying" state
            phases.append(0.0)
    return phases


# ============================================================
# Part 2: Construct the feature matrix V and examine its rank
# ============================================================

def build_feature_matrix_direct(p, betas, gammas, k):
    """
    Build the action components for the SIMPLIFIED model.

    Trajectories: s in {0,1}^{p+1}, so |S| = 2^{p+1}.
    Channels: one per round, so |A| = p.
    V[s, j] = sqrt(c_j) * A_{j,s} with A_{j,s} = 1 if s_j=0 else 0.
    """
    trajectories = all_trajectories(p)
    n_traj = len(trajectories)  # 2^{p+1}

    b = np.array([compute_amplitude_b(s, betas) for s in trajectories])

    V = np.zeros((n_traj, p), dtype=complex)
    for idx, s in enumerate(trajectories):
        for j in range(p):
            gamma_j = gammas[j]
            c_j = (np.exp(-1j * gamma_j / 2**k) - 1)
            A_js = 1.0 if s[j] == 0 else 0.0
            V[idx, j] = np.sqrt(c_j + 0j) * A_js

    return trajectories, b, V


def build_full_transfer_matrix_representation(p, betas, gammas, k):
    """
    Compute G(y) = sum_s b_s exp(v_s . y) via a product of 2x2 transfer matrices.

    This shows that in the SIMPLIFIED model, G depends on y through exactly
    p numbers -> effective dimension = p. Cost O(p) per evaluation.
    """
    def G_of_y(y_vec):
        assert len(y_vec) == p
        state = np.array([1.0, 1.0]) / np.sqrt(2)
        for j in range(p):
            gamma_j = gammas[j]
            c_j = np.exp(-1j * gamma_j / 2**k) - 1
            phase_diag = np.array([
                np.exp(np.sqrt(c_j + 0j) * y_vec[j]),
                1.0
            ])
            state = phase_diag * state
            M = mixer_matrix(betas[j])
            state = M @ state
        final = np.array([1.0, 1.0]) / np.sqrt(2)
        return np.dot(final, state)

    return G_of_y


# ============================================================
# Part 3: Examine the structure for the FULL BM24 parametrization
# ============================================================

def build_BM24_full_matrix(p, betas, gammas, k):
    """
    Build the feature matrix in the SIMPLIFIED model (same as build_feature_matrix_direct).

    Naming kept for compatibility with the rest of the script. This is NOT
    the full BM24 matrix from Example 1 (which would be 2^{2p+1} x (2^{2p+1}-2p-2)).
    """
    trajectories = all_trajectories(p)
    n_traj = len(trajectories)

    V_physical = np.zeros((n_traj, p), dtype=complex)
    b = np.zeros(n_traj, dtype=complex)

    for idx, s in enumerate(trajectories):
        b[idx] = compute_amplitude_b(s, betas)
        for j in range(p):
            gamma_j = gammas[j]
            c_j = (np.exp(-1j * gamma_j / 2**k) - 1)
            A_js = 1 - s[j]
            V_physical[idx, j] = np.sqrt(c_j + 0j) * A_js

    return trajectories, b, V_physical


def verify_transfer_matrix_equivalence(p, betas, gammas, k):
    """Verify that the transfer matrix computation of G(y) matches the direct sum."""
    trajectories, b, V = build_BM24_full_matrix(p, betas, gammas, k)
    G_transfer = build_full_transfer_matrix_representation(p, betas, gammas, k)

    print(f"\n  Transfer matrix vs direct sum (p={p}):")
    max_err = 0
    for trial in range(5):
        y = np.random.randn(p) + 1j * np.random.randn(p) * 0.3
        G_direct = np.sum(b * np.exp(V @ y))
        G_tm = G_transfer(y)
        err = abs(G_direct - G_tm) / max(abs(G_direct), 1e-15)
        max_err = max(max_err, err)
    print(f"  Max relative error: {max_err:.2e}")
    return max_err < 1e-10


# ============================================================
# Part 4: Rank analysis — the main investigation
# ============================================================

def analyze_rank_structure(p_max=8, k=3):
    """
    For the SIMPLIFIED model: |S|=2^{p+1}, |A|=p, rank(V)=p.
    """
    print("=" * 75)
    print(f"RANK ANALYSIS (SIMPLIFIED MODEL: S={{0,1}}^{{p+1}}, |A|=p)")
    print("=" * 75)
    print()
    print(f"{'p':>3} | {'|S|=2^(p+1)':>12} | {'|A|_BM24=2^(2p+1)':>18} | "
          f"{'|A|_this_model':>14} | {'rank(V)':>8} | {'eff. dim':>9}")
    print("-" * 75)

    for p in range(1, p_max + 1):
        betas = np.random.uniform(-np.pi/2, np.pi/2, p)
        gammas = np.random.uniform(0.5, 3.0, p)

        trajectories, b, V = build_BM24_full_matrix(p, betas, gammas, k)
        n_traj = len(trajectories)
        A_naive = 2**(2*p + 1)
        A_physical = p

        if V.shape[0] > 0 and V.shape[1] > 0:
            sv = _svdvals(V)
            rank = np.sum(sv > 1e-12 * sv[0]) if sv[0] > 1e-15 else 0
        else:
            rank = 0

        print(f"{p:>3} | {n_traj:>12} | {A_naive:>18} | "
              f"{A_physical:>14} | {rank:>8} | {rank:>9}")

    print()
    print("In this SIMPLIFIED model: rank(V) = p. (Full BM24 A36 has |A| exponential.)")


# ============================================================
# Part 5: Detailed analysis of WHY the reduction works
# ============================================================

def explain_reduction_mechanism(p=3, k=3):
    """Explain why rank(V)=p in the SIMPLIFIED model (V[s,j] depends only on s_j)."""
    print("\n" + "=" * 75)
    print(f"MECHANISM IN THE SIMPLIFIED MODEL (p={p}, k={k})")
    print("=" * 75)

    betas = np.array([-np.pi/4] * p)
    gammas = np.array([2.0] * p)

    trajectories, b, V = build_BM24_full_matrix(p, betas, gammas, k)
    n_traj = len(trajectories)

    print(f"\n1. TRAJECTORY SPACE (this model): |S| = 2^(p+1) = {n_traj}, s ∈ {{0,1}}^{p+1}")
    print(f"\n2. FEATURE MATRIX V[s, j] = √c_j · (1 - s_j)  =>  column j depends only on s_j")
    print(f"\n3. So the p columns are linearly independent => rank(V) = p")
    sv = _svdvals(V)
    rank = np.sum(sv > 1e-12 * sv[0])
    print(f"\n   Numerical rank(V) = {rank}")
    print(f"\n4. Saddle problem in this model reduces to C^p. (Full BM24: no such collapse.)")


# ============================================================
# Part 6: The deeper question — what is |A| really?
# ============================================================

def investigate_BM24_index_set(p_max=6):
    """
    Index-set conventions: A = 2^{[2p+1]} (full) vs A = {J : |J|≥2} (BM24 A32).
    Simplified model (this script): S = {0,1}^{p+1}, |A| = p.
    """
    print("\n" + "=" * 75)
    print("INDEX SET CONVENTIONS")
    print("=" * 75)
    print("""
Simplified model: S = {0,1}^{p+1}, |A| = p  =>  rank(V)=p, O(p) evaluation.
True BM24:        S = {0,1}^{2p+1}. Two conventions:
  - Full (Hessian paper): A = 2^{[2p+1]},  |A| = 2^{2p+1} = d
  - BM24 Example 1 (A32): A = {J : |J|≥2}, |A| = 2^{2p+1} - 2p - 2
""")
    print(f"{'p':>3} | {'|S|_simpl':>10} | {'|A|_simpl':>10} | {'|A|_full':>10} | {'|A|_J>=2':>10}")
    print("-" * 55)
    for p in range(1, p_max + 1):
        n_simpl = 2**(p+1)
        A_full = 2**(2*p+1)
        A_J2 = 2**(2*p+1) - 2*p - 2
        print(f"{p:>3} | {n_simpl:>10} | {p:>10} | {A_full:>10} | {A_J2:>10}")


# ============================================================
# Part 7: Implications for the saddle-point problem
# ============================================================

def implications_for_saddle_analysis():
    """
    In the SIMPLIFIED model, the saddle lives in C^p. In full BM24, saddle
    is in C^{|A|} but computable via O(p·4^p) recursion.
    """
    print("\n" + "=" * 75)
    print("IMPLICATIONS (SIMPLIFIED MODEL vs FULL BM24)")
    print("=" * 75)
    print("""
SIMPLIFIED MODEL (this script):
  - Saddle equations in C^p; transfer matrix gives O(p) evaluation.
  - Rank(V)=p => effective dimension p.

FULL BM24 A36:
  - No dimension collapse to O(p); |A| = 2^{2p+1} - 2p - 2.
  - Saddle map: y_α = 2√c_α E_μ_y[A_{αs}]; computable in O(p·4^p) via
    sum-over-subsets (constancy events), see BM24_A36_symmetry_reduction.md.
""")


# ============================================================
# Part 8: Numerical verification of the count
# ============================================================

def count_saddles_small_p(p_max=5, k=3):
    """Find saddles of the SIMPLIFIED p-dimensional action. Requires scipy for fsolve."""
    print("\n" + "=" * 75)
    print(f"SADDLE COUNT IN THE SIMPLIFIED MODEL (dim = p)")
    print("=" * 75)
    try:
        from scipy.optimize import fsolve
    except ImportError:
        print("  (Skipped: scipy not installed.)")
        return

    beta_val = -np.pi / 4

    for p in range(1, p_max + 1):
        betas = np.array([beta_val] * p)
        gammas = np.array([2.0] * p)

        trajectories = all_trajectories(p)
        b_arr = np.array([compute_amplitude_b(s, betas) for s in trajectories])

        V = np.zeros((len(trajectories), p), dtype=complex)
        for idx, s in enumerate(trajectories):
            for j in range(p):
                c_j = np.exp(-1j * gammas[j] / 2**k) - 1
                V[idx, j] = np.sqrt(c_j + 0j) * (1 - s[j])

        def action_gradient(z_real_imag):
            z = z_real_imag[:p] + 1j * z_real_imag[p:]
            exponents = V @ z
            weights = b_arr * np.exp(exponents)
            G = np.sum(weights)
            if abs(G) < 1e-30:
                return np.ones(2 * p) * 1e10
            grad_log_G = np.sum(weights[:, None] * V, axis=0) / G
            grad_F = -z / 2 + grad_log_G
            return np.concatenate([grad_F.real, grad_F.imag])

        found_saddles = []
        n_trials = 500 * (2 ** p)

        for trial in range(n_trials):
            z0 = np.random.randn(2 * p) * (0.5 + 0.3 * p)
            try:
                sol = fsolve(action_gradient, z0, full_output=True)  # type: ignore
                z_sol = sol[0]
                if np.max(np.abs(action_gradient(z_sol))) < 1e-10:
                    z_complex = z_sol[:p] + 1j * z_sol[p:]
                    is_new = True
                    for z_prev in found_saddles:
                        if np.max(np.abs(z_complex - z_prev)) < 1e-6:
                            is_new = False
                            break
                    if is_new:
                        found_saddles.append(z_complex)
            except Exception:
                pass

        saddle_values = []
        for z in found_saddles:
            exponents = V @ z
            G = np.sum(b_arr * np.exp(exponents))
            if abs(G) > 1e-30:
                F_val = -np.sum(z**2) / 4 + np.log(G)
                saddle_values.append(F_val.real)
        saddle_values.sort(reverse=True)

        n_saddles = len(found_saddles)
        print(f"\n  p = {p}: found {n_saddles} saddles in C^{p}")
        if saddle_values:
            print(f"    Re[F] (top): {['%.4f' % v for v in saddle_values[:6]]}")
            if len(saddle_values) > 1:
                print(f"    Dominance gap δ = {saddle_values[0] - saddle_values[1]:.4f}")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("╔" + "═" * 73 + "╗")
    print("║  QAOA ACTION: TRUE BM24 PIPELINE (A=2^{[2p+1]}, H_log, k_η) + SIMPLIFIED MODEL  ║")
    print("╚" + "═" * 73 + "╝")

    # --- Primary: TRUE BM24 pipeline (single pipeline, full index set) ---
    print("\n\n" + "▶ " * 3 + "TRUE BM24 — FEATURE MATRIX RANK (A = 2^{[2p+1]}, d = 2^{2p+1})" + " ◀" * 3)
    analyze_true_BM24_rank(p_max=4, index_set="full")

    print("\n\n" + "▶ " * 3 + "TRUE BM24 — HESSIAN H_log = ∇²Φ + ½I, SVD, k_η by Frobenius" + " ◀" * 3)
    analyze_true_BM24_hessian(p_max=4, index_set="full")

    # --- Pedagogical contrast: simplified model (S={0,1}^{p+1}, |A|=p) ---
    print("\n\n" + "--- PEDAGOGICAL CONTRAST: SIMPLIFIED MODEL ---")
    print("\n" + "▶ " * 3 + "PART 1: RANK (simplified: S={0,1}^{p+1}, |A|=p)" + " ◀" * 3)
    analyze_rank_structure(p_max=10, k=3)

    print("\n\n" + "▶ " * 3 + "PART 2: TRANSFER MATRIX VERIFICATION" + " ◀" * 3)
    for p in range(1, 6):
        betas = np.random.uniform(-np.pi/2, np.pi/2, p)
        gammas = np.random.uniform(0.5, 3.0, p)
        verify_transfer_matrix_equivalence(p, betas, gammas, k=3)

    print("\n\n" + "▶ " * 3 + "PART 3: WHY rank(V)=p IN THIS MODEL" + " ◀" * 3)
    explain_reduction_mechanism(p=3, k=3)

    print("\n\n" + "▶ " * 3 + "PART 4: THIS MODEL vs FULL BM24 INDEX SET" + " ◀" * 3)
    investigate_BM24_index_set(p_max=8)

    print("\n\n" + "▶ " * 3 + "PART 5: IMPLICATIONS" + " ◀" * 3)
    implications_for_saddle_analysis()

    print("\n\n" + "▶ " * 3 + "PART 6: SADDLE COUNT (SIMPLIFIED MODEL)" + " ◀" * 3)
    count_saddles_small_p(p_max=4, k=3)

    print("\n\n" + "=" * 75)
    print("DONE")
    print("=" * 75)
