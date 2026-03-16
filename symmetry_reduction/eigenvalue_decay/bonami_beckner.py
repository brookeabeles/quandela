"""
Bonami–Beckner comparison (Phase 1, Task 1.4).

Compare eigenvalue decay of:
(a) H_log = diag(√c) Cov_w(A) diag(√c)
(b) diag(√c) Cov^{(ρ)}(A) diag(√c) with Cov^{(ρ)}_αα' = ∑_{S ⊆ α∩α', |S| even} ρ^{|S|} 2^{-|α|} 2^{-|α'|}
(c) diag(√c) Cov_uniform(A) diag(√c)

Cov_uniform uses w_s = 1/2^n. Cov^{(ρ)} is the "noise operator" version: Fourier-level damping by ρ^{|S|}.
Effective ρ ≈ Γ/√2 with Γ = 2|sin(γ/4)|.
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Any

import sys
if __name__ == "__main__":
    _root = Path(__file__).resolve().parent.parent.parent
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

from symmetry_reduction.core import (
    build_structure_matrix,
    compute_b_s,
    compute_c_alpha,
    hessian_log_at_y,
    weights_at_zero,
    DEFAULT_BETA,
)
from symmetry_reduction.mechanisms import phase_bound_Gamma
from .eigenvalue_decay import _eigenvalues_sorted


def _popcount_arr(d: int, n_bits: int) -> np.ndarray:
    """Popcount for each index in [0, d)."""
    alpha_grid = np.arange(d, dtype=np.int64)[:, None]
    bits = (alpha_grid >> np.arange(n_bits, dtype=np.int64)) & 1
    return bits.sum(axis=1).astype(np.intp)


def cov_uniform_agreement(A: np.ndarray, n: int) -> np.ndarray:
    """Cov_uniform(A): covariance of rows of A under uniform weights w_s = 1/2^n."""
    n_s = A.shape[1]
    w = np.ones(n_s) / n_s
    E_A = w @ A.T
    E_AA = A @ (w[:, None] * A.T)
    return E_AA - np.outer(E_A, E_A)


def cov_rho_agreement(d: int, n: int, rho: float) -> np.ndarray:
    """
    Cov^{(ρ)}_αα' = ∑_{S ⊆ α∩α', |S| even} ρ^{|S|} · 2^{-|α|} · 2^{-|α'|}.
    (Excluding S=∅ gives covariance; including S=∅ gives second moment. We use even S including ∅
    so E[f_α] = 2^{-|α|} and Cov = E[f_α f_α'] - E[f_α]E[f_α'] = sum_{S≠∅, S⊆α∩α', |S| even} ρ^{|S|} 2^{-|α|-|α'|}.)
    Actually E_unif[f_α] = 2^{-|α|} (mean of agreement function). So
    Cov_unif(f_α, f_α') = ∑_{S≠∅} f̂_α(S) f̂_α'(S) = ∑_{S ⊆ α∩α', |S| even, S≠∅} 2^{-|α|} 2^{-|α'|}.
    The ρ-weighted version: Cov^{(ρ)} = ∑_{S≠∅, S⊆α∩α', |S| even} ρ^{|S|} 2^{-|α|} 2^{-|α'|}.
    """
    sizes = _popcount_arr(d, n)
    Cov = np.zeros((d, d))
    for alpha in range(d):
        for alpha_p in range(d):
            m = sizes[alpha]
            mp = sizes[alpha_p]
            inter = alpha & alpha_p  # S ⊆ α∩α' iff S ⊆ (alpha & alpha_p)
            t = bin(inter).count("1")  # |α ∩ α'|
            # number of S ⊆ α∩α' with |S| even, S≠∅: sum_{j=1,3,5,...} binom(t,j) = 2^{t-1} - 1? No: even means 0,2,4,... sum_{j even, j>=2} binom(t,j) = 2^{t-1} - 1 for t>=1. For t=0: 0.
            # Actually even S with S≠∅: j=2,4,..., so sum_{j even, j>0} binom(t,j). Sum_{j even} binom(t,j) = 2^{t-1}. So sum_{j even, j>0} = 2^{t-1} - 1 (if t>=1, the 1 is from j=0).
            total = 0.0
            for j in range(0, t + 1, 2):  # j even including 0
                if j == 0:
                    continue
                from math import comb
                total += (rho ** j) * comb(t, j)
            total *= (2.0 ** (-m - mp))
            Cov[alpha, alpha_p] = total
    return Cov


def H_rho_matrix(p: int, gamma: float, rho: float | None = None) -> np.ndarray:
    """H_ρ = diag(√c) Cov^{(ρ)} diag(√c). If rho is None, use Γ/√2."""
    if rho is None:
        rho = phase_bound_Gamma(gamma) / np.sqrt(2)
    n = 2 * p + 1
    d = 1 << n
    gammas = np.full(p, gamma)
    c_alpha = compute_c_alpha(p, gammas)
    sqrt_c = np.sqrt(c_alpha + 0j)
    Cov = cov_rho_agreement(d, n, rho)
    return np.outer(sqrt_c, sqrt_c) * Cov


def bonami_beckner_comparison(
    p: int,
    gamma: float,
    at_saddle: bool = False,
    rho_values: tuple[float, ...] = (0.2, 0.4, 0.6, 0.8),
) -> dict[str, Any]:
    """
    Compute eigenvalue decay (sorted |λ_k|) for:
    (a) H_log at y=0 or saddle
    (b) H_ρ for ρ = Γ/√2 and for a few rho values to find best fit
    (c) H_uniform = diag(√c) Cov_uniform(A) diag(√c)

    Returns dict with eigenvalues (or abs) for each, and optional best-fit ρ.
    """
    from symmetry_reduction.saddle import saddle_with_adaptive_damping
    from symmetry_reduction.core import softmax_weights
    n = 2 * p + 1
    d = 1 << n
    A = build_structure_matrix(p)
    b_s = compute_b_s(p, np.full(p, DEFAULT_BETA))
    c_alpha = compute_c_alpha(p, np.full(p, gamma))
    sqrt_c = np.sqrt(c_alpha + 0j)

    # (a) H_log
    if at_saddle:
        y_star, _, _ = saddle_with_adaptive_damping(A, b_s, c_alpha)
        H_log = hessian_log_at_y(y_star, A, b_s, c_alpha)
    else:
        y0 = np.zeros(d, dtype=complex)
        H_log = hessian_log_at_y(y0, A, b_s, c_alpha)
    lam_log = _eigenvalues_sorted(H_log)
    abs_log = np.abs(lam_log)

    # (c) H_uniform
    Cov_unif = cov_uniform_agreement(A, n)
    H_unif = np.outer(sqrt_c, sqrt_c) * Cov_unif
    lam_unif = _eigenvalues_sorted(H_unif)
    abs_unif = np.abs(lam_unif)

    # (b) H_ρ for several ρ
    Gamma = phase_bound_Gamma(gamma)
    rho_eff = Gamma / np.sqrt(2)
    H_rho_eff = H_rho_matrix(p, gamma, rho=rho_eff)
    lam_rho_eff = _eigenvalues_sorted(H_rho_eff)
    abs_rho_eff = np.abs(lam_rho_eff)

    result = {
        "p": p,
        "gamma": gamma,
        "at_saddle": at_saddle,
        "abs_eigenvalues_H_log": abs_log,
        "abs_eigenvalues_H_uniform": abs_unif,
        "abs_eigenvalues_H_rho_eff": abs_rho_eff,
        "rho_eff": rho_eff,
        "Gamma": Gamma,
    }
    # Optional: fit ρ to match decay of H_log (e.g. match first 20 eigenvalues in log scale)
    rho_results = {}
    for rho in rho_values:
        H_rho = H_rho_matrix(p, gamma, rho=rho)
        lam_rho = _eigenvalues_sorted(H_rho)
        rho_results[rho] = np.abs(lam_rho)
    result["abs_eigenvalues_by_rho"] = rho_results
    return result


def plot_bonami_comparison(
    p: int = 3,
    gamma: float = 1.0,
    at_saddle: bool = False,
    out_path: Path | None = None,
) -> None:
    """Plot log|λ_k| vs k for H_log, H_uniform, H_ρ (effective ρ)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = bonami_beckner_comparison(p, gamma, at_saddle=at_saddle)
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    k_max = min(100, len(out["abs_eigenvalues_H_log"]))
    k_axis = np.arange(k_max)
    ax.semilogy(k_axis, np.maximum(out["abs_eigenvalues_H_log"][:k_max], 1e-20), "b.-", markersize=2, label="H_log")
    abs_unif = out["abs_eigenvalues_H_uniform"]
    ax.semilogy(np.arange(min(k_max, len(abs_unif))), np.maximum(abs_unif[:k_max], 1e-20), "g.-", markersize=2, label="H_uniform")
    abs_rho = out["abs_eigenvalues_H_rho_eff"]
    ax.semilogy(np.arange(min(k_max, len(abs_rho))), np.maximum(abs_rho[:k_max], 1e-20), "r.-", markersize=2, label=f"H_ρ (ρ=Γ/√2={out['rho_eff']:.3f})")
    ax.set_xlabel("k (eigenvalue index)")
    ax.set_ylabel("|λ_k|")
    ax.set_title(f"Eigenvalue decay: H_log vs noise-operator model (p={p}, γ={gamma:.2f})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path = out_path or Path(__file__).resolve().parent.parent / "figures" / "eigenvalue_decay" / f"bonami_comparison_p{p}_gamma{gamma:.2f}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_bonami_comparison_all(
    p_values: tuple[int, ...] = (2, 3),
    gamma_values: tuple[float, ...] = (0.3, 1.0, np.pi),
    at_saddle: bool = False,
    out_path: Path | None = None,
) -> None:
    """One figure with 2×3 subplots: (p, γ) grid of H_log vs H_uniform vs H_ρ."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_p = len(p_values)
    n_g = len(gamma_values)
    fig, axes = plt.subplots(n_p, n_g, figsize=(4 * n_g, 4 * n_p), sharex=True, sharey=True)
    if n_p == 1 and n_g == 1:
        axes = np.array([[axes]])
    elif n_p == 1:
        axes = axes.reshape(1, -1)
    elif n_g == 1:
        axes = axes.reshape(-1, 1)
    k_max = 80
    for i, p in enumerate(p_values):
        for j, gamma in enumerate(gamma_values):
            ax = axes[i, j]
            out = bonami_beckner_comparison(p, gamma, at_saddle=at_saddle)
            abs_log = out["abs_eigenvalues_H_log"]
            n_plot = min(k_max, len(abs_log))
            k_axis = np.arange(n_plot)
            ax.semilogy(k_axis, np.maximum(abs_log[:n_plot], 1e-20), "b.-", markersize=1.5, label="H_log")
            abs_unif = out["abs_eigenvalues_H_uniform"]
            n_u = min(n_plot, len(abs_unif))
            ax.semilogy(np.arange(n_u), np.maximum(abs_unif[:n_u], 1e-20), "g.-", markersize=1.5, label="H_unif")
            abs_rho = out["abs_eigenvalues_H_rho_eff"]
            n_r = min(n_plot, len(abs_rho))
            ax.semilogy(np.arange(n_r), np.maximum(abs_rho[:n_r], 1e-20), "r.-", markersize=1.5, label="H_ρ")
            ax.set_title(f"p={p}, γ={gamma:.2f}")
            ax.legend(fontsize=6)
            ax.grid(True, alpha=0.3)
    for ax in axes[-1, :]:
        ax.set_xlabel("k")
    for ax in axes[:, 0]:
        ax.set_ylabel("|λ_k|")
    fig.suptitle("Bonami–Beckner: H_log vs H_uniform vs H_ρ" + (" (saddle)" if at_saddle else " (y=0)"))
    fig.tight_layout()
    if out_path is None:
        out_path = Path(__file__).resolve().parent.parent / "figures" / "eigenvalue_decay" / "bonami_comparison_all.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    plot_bonami_comparison(p=3, gamma=1.0)
