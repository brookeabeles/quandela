"""
Eigenvalue-by-index decay of H_log: numerical characterization (Phase 1).

Tasks 1.1–1.3: compute eigenvalues (eig, not eigh), fit exponential/power-law/stretched-exponential,
test c'(p) scaling, and decompose eigenvector energy by block level |α|.
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Any

# Import from parent package (symmetry_reduction)
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
    softmax_weights,
    weights_at_zero,
    DEFAULT_BETA,
)
from symmetry_reduction.saddle import saddle_with_adaptive_damping
from symmetry_reduction.mechanisms import _popcount_array

OUT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = OUT_DIR / "data"
FIG_DIR = OUT_DIR / "figures" / "eigenvalue_decay"


def _get_H_log(p: int, gamma: float, at_saddle: bool, A: np.ndarray | None = None, b_s: np.ndarray | None = None):
    """Build H_log at y=0 or at saddle for (p, gamma). Returns (H, ...)."""
    d = 1 << (2 * p + 1)
    betas = np.full(p, DEFAULT_BETA)
    gammas = np.full(p, gamma)
    if A is None:
        A = build_structure_matrix(p)
    if b_s is None:
        b_s = compute_b_s(p, betas)
    c_alpha = compute_c_alpha(p, gammas)
    if at_saddle:
        y_star, converged, _ = saddle_with_adaptive_damping(A, b_s, c_alpha)
        H = hessian_log_at_y(y_star, A, b_s, c_alpha)
        return H, A, b_s, c_alpha, y_star
    y0 = np.zeros(d, dtype=complex)
    H = hessian_log_at_y(y0, A, b_s, c_alpha)
    return H, A, b_s, c_alpha, None


def _eigenvalues_sorted(H: np.ndarray, tol: float = 1e-14) -> np.ndarray:
    """Eigenvalues of H (complex symmetric). Sorted by |λ_k| descending."""
    lam, _ = np.linalg.eig(H)
    lam = np.asarray(lam)
    idx = np.argsort(np.abs(lam))[::-1]
    lam = lam[idx]
    # optionally drop negligible
    lam = lam[np.abs(lam) > tol]
    return lam


def fit_exponential_log(abs_lam: np.ndarray, k_start: int = 0, k_end: int | None = None) -> dict[str, float]:
    """Fit |λ_k| = M exp(-c' k) via log|λ_k| = log M - c' k. Returns M, c', R²."""
    k_end = k_end or len(abs_lam)
    use = np.arange(k_start, min(k_end, len(abs_lam)), dtype=float)
    if len(use) < 2:
        return {"M": np.nan, "c_prime": np.nan, "r2": np.nan}
    y = np.log(np.maximum(abs_lam[k_start:k_end], 1e-30))
    x = use
    n = len(x)
    sxy = np.sum((x - np.mean(x)) * (y - np.mean(y)))
    sxx = np.sum((x - np.mean(x)) ** 2)
    if sxx < 1e-30:
        return {"M": np.nan, "c_prime": np.nan, "r2": np.nan}
    c_prime = -sxy / sxx
    log_M = np.mean(y) + c_prime * np.mean(x)
    M = np.exp(log_M)
    y_pred = log_M - c_prime * x
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 1e-30 else 0.0
    return {"M": float(M), "c_prime": float(c_prime), "r2": float(r2)}


def fit_power_law(abs_lam: np.ndarray, k_start: int = 1, k_end: int | None = None) -> dict[str, float]:
    """Fit |λ_k| = M k^{-α} via log|λ_k| = log M - α log k (k≥1). Returns M, α, R²."""
    k_end = k_end or len(abs_lam)
    use_idx = np.arange(max(1, k_start), min(k_end, len(abs_lam)))
    if len(use_idx) < 2:
        return {"M": np.nan, "alpha": np.nan, "r2": np.nan}
    k_vals = use_idx.astype(float) + 0.5  # avoid log(0)
    y = np.log(np.maximum(abs_lam[use_idx], 1e-30))
    x = np.log(k_vals)
    sxy = np.sum((x - np.mean(x)) * (y - np.mean(y)))
    sxx = np.sum((x - np.mean(x)) ** 2)
    if sxx < 1e-30:
        return {"M": np.nan, "alpha": np.nan, "r2": np.nan}
    alpha = -sxy / sxx
    log_M = np.mean(y) + alpha * np.mean(x)
    M = np.exp(log_M)
    y_pred = log_M - alpha * x
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 1e-30 else 0.0
    return {"M": float(M), "alpha": float(alpha), "r2": float(r2)}


def fit_stretched_exponential(
    abs_lam: np.ndarray, beta: float = 0.5, k_start: int = 0, k_end: int | None = None
) -> dict[str, float]:
    """Fit |λ_k| = M exp(-c' k^β). Fix β, fit log|λ_k| = log M - c' k^β. Returns M, c_prime, R²."""
    k_end = k_end or len(abs_lam)
    use_idx = np.arange(k_start, min(k_end, len(abs_lam)), dtype=float)
    if len(use_idx) < 2:
        return {"M": np.nan, "c_prime": np.nan, "beta": beta, "r2": np.nan}
    x = np.power(use_idx + 0.5, beta)
    y = np.log(np.maximum(abs_lam[k_start:k_end], 1e-30))
    sxy = np.sum((x - np.mean(x)) * (y - np.mean(y)))
    sxx = np.sum((x - np.mean(x)) ** 2)
    if sxx < 1e-30:
        return {"M": np.nan, "c_prime": np.nan, "beta": beta, "r2": np.nan}
    c_prime = -sxy / sxx
    log_M = np.mean(y) + c_prime * np.mean(x)
    M = np.exp(log_M)
    y_pred = log_M - c_prime * x
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 1e-30 else 0.0
    return {"M": float(M), "c_prime": float(c_prime), "beta": beta, "r2": float(r2)}


def eigenvalue_decay_analysis(
    p: int,
    gamma: float,
    at_saddle: bool = False,
    k_fit_max: int | None = None,
) -> dict[str, Any]:
    """
    Compute eigenvalues of H_log, sort by |λ_k| descending, fit decay models.

    Returns:
        eigenvalues: complex array sorted by magnitude descending
        abs_eigenvalues: |λ_k|
        fits: dict with 'exponential', 'power_law', 'stretched_exponential'
              (M, c' or α, R²)
    """
    H, *_ = _get_H_log(p, gamma, at_saddle)
    lam = _eigenvalues_sorted(H)
    abs_lam = np.abs(lam)
    n = len(abs_lam)
    k_fit_max = min(k_fit_max or n, n)

    fits = {
        "exponential": fit_exponential_log(abs_lam, k_start=0, k_end=k_fit_max),
        "power_law": fit_power_law(abs_lam, k_start=1, k_end=k_fit_max),
        "stretched_exponential": fit_stretched_exponential(abs_lam, beta=0.5, k_start=0, k_end=k_fit_max),
    }
    return {
        "eigenvalues": lam,
        "abs_eigenvalues": abs_lam,
        "fits": fits,
        "p": p,
        "gamma": gamma,
        "at_saddle": at_saddle,
    }


def decay_rate_vs_p(
    gamma: float,
    p_values: tuple[int, ...] = (2, 3, 4, 5),
    at_saddle: bool = False,
) -> dict[str, Any]:
    """
    For fixed gamma, compute c'(p) and M(p) from exponential fit.
    If c'(p) → const > 0 as p grows, eigenvalue decay is uniform in p.
    """
    c_prime_list = []
    M_list = []
    r2_list = []
    for p in p_values:
        out = eigenvalue_decay_analysis(p, gamma, at_saddle=at_saddle)
        f = out["fits"]["exponential"]
        c_prime_list.append(f["c_prime"])
        M_list.append(f["M"])
        r2_list.append(f["r2"])
    return {
        "gamma": gamma,
        "p_values": list(p_values),
        "c_prime": np.array(c_prime_list),
        "M": np.array(M_list),
        "r2": np.array(r2_list),
        "at_saddle": at_saddle,
    }


def eigenvalue_block_decomposition(
    p: int,
    gamma: float,
    at_saddle: bool = False,
    n_eig: int | None = None,
) -> dict[str, Any]:
    """
    For each eigenvector v_k of H_log, compute fraction of |v_k|² in each block level m = |α|.
    Returns heatmap data: (eigenvalue index k, block level m) → energy fraction.
    """
    H, A, b_s, c_alpha, _ = _get_H_log(p, gamma, at_saddle)
    d, n_bits = H.shape[0], 2 * p + 1
    sizes = _popcount_array(d, n_bits)  # size[alpha] = |alpha|

    lam, v = np.linalg.eig(H)
    lam = np.asarray(lam)
    v = np.asarray(v)
    idx = np.argsort(np.abs(lam))[::-1]
    lam = lam[idx]
    v = v[:, idx]
    n_eig = n_eig or min(d, 200)  # limit for large d
    v = v[:, :n_eig]

    n_blocks = n_bits + 1
    block_energy = np.zeros((n_eig, n_blocks))  # [k, m]
    for k in range(n_eig):
        vk = v[:, k]
        norm_sq = np.sum(np.abs(vk) ** 2)
        if norm_sq < 1e-30:
            continue
        for m in range(n_blocks):
            mask = sizes == m
            block_energy[k, m] = np.sum(np.abs(vk[mask]) ** 2) / norm_sq

    return {
        "block_energy": block_energy,
        "eigenvalues": lam[:n_eig],
        "p": p,
        "gamma": gamma,
        "at_saddle": at_saddle,
        "n_blocks": n_blocks,
        "k_indices": np.arange(n_eig),
        "m_levels": np.arange(n_blocks),
    }


def block_max_eigenvalue_analysis(
    p: int,
    gamma: float,
    at_saddle: bool = False,
) -> dict[str, Any]:
    """
    Extract the maximum absolute eigenvalue per Fourier block level m.
    For each eigenvector k, primary block m = argmax_m block_energy[k, m].
    For each m, max |λ^(m)| = max{ |λ_k| : primary_m[k] == m }.
    Returns m_levels (0..2p+1) and block_max (max |λ| per m; NaN/0 if no eigenvector has primary m).
    """
    data = eigenvalue_block_decomposition(p, gamma, at_saddle=at_saddle)
    block_energy = data["block_energy"]
    lam = data["eigenvalues"]
    n_blocks = data["n_blocks"]
    abs_lam = np.abs(lam)
    # primary_m[k] = block level m where eigenvector k has maximum energy
    primary_m = np.argmax(block_energy, axis=1)
    block_max = np.full(n_blocks, np.nan, dtype=float)
    for m in range(n_blocks):
        mask = primary_m == m
        if np.any(mask):
            block_max[m] = float(np.max(abs_lam[mask]))
        else:
            block_max[m] = 0.0  # or leave np.nan
    m_levels = np.arange(n_blocks, dtype=float)
    return {
        "m_levels": m_levels,
        "block_max": block_max,
        "primary_m": primary_m,
        "p": p,
        "gamma": gamma,
        "at_saddle": at_saddle,
        "n_blocks": n_blocks,
    }


def fit_exponential_block_max(
    m_levels: np.ndarray,
    block_max: np.ndarray,
    m_min: int = 0,
    m_max: int | None = None,
) -> dict[str, float]:
    """
    Fit |λ^(m)| ≈ M exp(-c_m m) to block maximums vs block level m.
    Uses only valid (finite, positive) block_max entries. Returns M, c_m, R².
    """
    m_levels = np.asarray(m_levels)
    block_max = np.asarray(block_max)
    valid = np.isfinite(block_max) & (block_max > 1e-30)
    m_max = m_max if m_max is not None else len(block_max) - 1
    for m in range(len(block_max)):
        if m < m_min or m > m_max:
            valid[m] = False
    x = m_levels[valid]
    y = np.log(block_max[valid])
    if len(x) < 2:
        return {"M": np.nan, "c_m": np.nan, "r2": np.nan}
    sxy = np.sum((x - np.mean(x)) * (y - np.mean(y)))
    sxx = np.sum((x - np.mean(x)) ** 2)
    if sxx < 1e-30:
        return {"M": np.nan, "c_m": np.nan, "r2": np.nan}
    c_m = -sxy / sxx
    log_M = np.mean(y) + c_m * np.mean(x)
    M = np.exp(log_M)
    y_pred = log_M - c_m * x
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 1e-30 else 0.0
    return {"M": float(M), "c_m": float(c_m), "r2": float(r2)}


def plot_decay_and_c_prime(
    p_values: tuple[int, ...] = (2, 3, 4, 5, 6),
    gamma_values: tuple[float, ...] = (np.pi / 4, np.pi / 2, np.pi, 2 * np.pi),
    at_saddle: bool = False,
    out_dir: Path | None = None,
) -> None:
    """
    Produce:
      (1) Eigenvalue-decay panels: log|λ_k| vs index k for a few (p, γ).
      (2) Grid of c'(p, γ): depths on horizontal lines with 4 γ values.
      (3) Separate c'(p) vs p plot (one curve per γ).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = out_dir or FIG_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- (1) Eigenvalue-decay panels: log|λ_k| vs index k ---
    fig_ev, axes_ev = plt.subplots(1, 3, figsize=(15, 4.5))
    decay_panels = [(2, gamma_values[0]), (4, gamma_values[1]), (6, gamma_values[2])]
    for ax, (p, gamma) in zip(axes_ev, decay_panels):
        out = eigenvalue_decay_analysis(p, gamma, at_saddle=at_saddle)
        k_axis = np.arange(len(out["abs_eigenvalues"]))
        ax.semilogy(
            k_axis,
            np.maximum(out["abs_eigenvalues"], 1e-20),
            "b.",
            markersize=1.5,
            label=r"$|\lambda_k|$",
        )
        f = out["fits"]["exponential"]
        if np.isfinite(f["M"]) and np.isfinite(f["c_prime"]) and f["c_prime"] > 0:
            pred = f["M"] * np.exp(-f["c_prime"] * k_axis)
            ax.semilogy(
                k_axis,
                pred,
                "r-",
                alpha=0.8,
                label=rf"exp fit $c'={f['c_prime']:.3f}$, $R^2={f['r2']:.2f}$",
            )
        ax.set_xlabel("Eigenvalue index $k$")
        ax.set_ylabel(r"$|\lambda_k|$ (absolute value)")
        ax.set_title(rf"Index-wise eigenvalue decay (p={p}, $\gamma$={gamma:.2f})")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    fig_ev.suptitle(
        f"Index-wise eigenvalue decay |λ_k| vs k {'(saddle)' if at_saddle else '(y=0)'}",
        y=1.02,
    )
    fig_ev.tight_layout(rect=[0, 0, 1, 0.96])
    suffix = "saddle" if at_saddle else "y0"
    fig_ev.savefig(
        out_dir / f"eigenvalue_decay_{suffix}.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig_ev)

    # --- (2) Grid of c'(p, γ): depths as horizontal lines, γ values along x ---
    p_list = list(p_values)
    gamma_list = list(gamma_values)
    cprime_mat = np.zeros((len(p_list), len(gamma_list)))
    for j, gamma in enumerate(gamma_list):
        out = decay_rate_vs_p(gamma, p_values=p_values, at_saddle=at_saddle)
        cprime_mat[:, j] = out["c_prime"]

    fig_grid, ax_grid = plt.subplots(1, 1, figsize=(6, 4.5))
    im = ax_grid.imshow(
        cprime_mat,
        aspect="auto",
        origin="lower",
        cmap="viridis",
    )
    ax_grid.set_xticks(np.arange(len(gamma_list)))
    # Use symbolic labels for the common 4-angle grid, otherwise fall back to numeric labels.
    if len(gamma_list) == 4 and np.allclose(
        gamma_list,
        [np.pi / 4, np.pi / 2, np.pi, 2 * np.pi],
        rtol=1e-6,
        atol=1e-8,
    ):
        ax_grid.set_xticklabels([r"$\pi/4$", r"$\pi/2$", r"$\pi$", r"$2\pi$"])
    else:
        ax_grid.set_xticklabels([f"{g:.2f}" for g in gamma_list])
    ax_grid.set_yticks(np.arange(len(p_list)))
    ax_grid.set_yticklabels([str(p) for p in p_list])
    ax_grid.set_xlabel(r"Problem angle $\gamma$")
    ax_grid.set_ylabel(r"QAOA depth $p$")
    ax_grid.set_title(r"Exponential decay rate $c'(p,\gamma)$")
    cbar = fig_grid.colorbar(im, ax=ax_grid)
    cbar.set_label(r"$c'(p,\gamma)$ (eigenvalue decay rate)")
    fig_grid.tight_layout()
    fig_grid.savefig(
        out_dir / f"eigenvalue_cprime_grid_{suffix}.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig_grid)

    # --- (3) Separate c'(p) vs p plot (one curve per γ) ---
    fig_cp, ax_cp = plt.subplots(1, 1, figsize=(6, 4.5))
    for j, gamma in enumerate(gamma_list):
        ax_cp.plot(
            p_list,
            cprime_mat[:, j],
            "o-",
            label=rf"$\gamma$={gamma:.2f}",
        )
    ax_cp.set_xlabel(r"QAOA depth $p$")
    ax_cp.set_ylabel(r"$c'(p,\gamma)$ (eigenvalue decay rate)")
    ax_cp.set_title(r"Index-wise decay rate $c'(p,\gamma)$ vs $p$")
    ax_cp.legend(fontsize=7)
    ax_cp.grid(True, alpha=0.3)
    fig_cp.tight_layout()
    fig_cp.savefig(
        out_dir / f"eigenvalue_cprime_vs_p_{suffix}.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig_cp)


def plot_block_decomposition_heatmap(
    p: int = 3,
    gamma: float = 1.0,
    at_saddle: bool = False,
    out_dir: Path | None = None,
) -> None:
    """Heatmap: rows = eigenvalue index k, columns = block level m, color = energy fraction."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = out_dir or FIG_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    data = eigenvalue_block_decomposition(p, gamma, at_saddle=at_saddle)
    BE = data["block_energy"]
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    im = ax.imshow(BE, aspect="auto", origin="lower", cmap="viridis", vmin=0, vmax=1)
    ax.set_xlabel("Block level m = |α|")
    ax.set_ylabel("Eigenvalue index k")
    ax.set_title(f"Eigenvector energy by block (p={p}, γ={gamma:.2f}, {'saddle' if at_saddle else 'y=0'})")
    plt.colorbar(im, ax=ax, label="Fraction of |v_k|²")
    fig.tight_layout()
    fig.savefig(out_dir / f"eigenvalue_block_decomposition_p{p}_gamma{gamma:.2f}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_block_decomposition_combined(
    p: int = 3,
    gamma: float = 1.0,
    out_dir: Path | None = None,
) -> None:
    """One figure: block decomposition heatmaps for y=0 (left) and saddle (right)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = out_dir or FIG_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, at_saddle in zip(axes, [False, True]):
        data = eigenvalue_block_decomposition(p, gamma, at_saddle=at_saddle)
        BE = data["block_energy"]
        im = ax.imshow(BE, aspect="auto", origin="lower", cmap="viridis", vmin=0, vmax=1)
        ax.set_xlabel("Block level m = |α|")
        ax.set_ylabel("Eigenvalue index k")
        ax.set_title(f"p={p}, γ={gamma:.2f} — {'saddle' if at_saddle else 'y=0'}")
    fig.colorbar(im, ax=axes, label="Fraction of |v_k|²", shrink=0.6)
    fig.suptitle("Eigenvector energy by block")
    fig.tight_layout(rect=[0, 0, 0.92, 0.96])
    fig.savefig(out_dir / f"eigenvalue_block_decomposition_p{p}_gamma{gamma:.2f}_combined.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_block_max_decay(
    p_values: tuple[int, ...] = (2, 3, 4, 5),
    gamma_values: tuple[float, ...] = (0.3, 1.0, np.pi),
    at_saddle: bool = False,
    out_dir: Path | None = None,
) -> None:
    """
    Block-wise spectral decay: log(max |λ^(m)|) vs m for selected (p, γ), with exponential fit.
    Fourth panel: block-decay rate c_m(p) vs p for different γ to check if it stabilizes with depth.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = out_dir or FIG_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    panels = axes.ravel()
    # First three panels: selected (p, gamma) — log(max |λ^(m)|) vs m with fit
    decay_panels = [(2, 0.3), (3, 1.0), (4, np.pi)]
    for idx, (p, gamma) in enumerate(decay_panels):
        ax = panels[idx]
        out = block_max_eigenvalue_analysis(p, gamma, at_saddle=at_saddle)
        m_lev = out["m_levels"]
        bmax = out["block_max"]
        valid = np.isfinite(bmax) & (bmax > 1e-20)
        ax.semilogy(m_lev[valid], bmax[valid], "b.", markersize=4, label=r"$\max|\lambda^{(m)}|$")
        fit = fit_exponential_block_max(out["m_levels"], out["block_max"])
        if np.isfinite(fit["M"]) and np.isfinite(fit["c_m"]) and fit["c_m"] > 0:
            pred = fit["M"] * np.exp(-fit["c_m"] * m_lev)
            ax.semilogy(m_lev, np.maximum(pred, 1e-20), "r-", alpha=0.8, label=rf"fit $c_m$={fit['c_m']:.3f} R²={fit['r2']:.2f}")
        ax.set_xlabel("Block level m")
        ax.set_ylabel(r"$\max|\lambda^{(m)}|$")
        ax.set_title(f"Block-wise eigenvalue decay via max |λ^(m)| (p={p}, γ={gamma:.2f})")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    # Fourth panel: c_m(p) vs p for each gamma
    ax_c = panels[3]
    for gamma in gamma_values:
        c_m_vals = []
        for p in p_values:
            out = block_max_eigenvalue_analysis(p, gamma, at_saddle=at_saddle)
            fit = fit_exponential_block_max(out["m_levels"], out["block_max"])
            c_m_vals.append(fit["c_m"] if np.isfinite(fit["c_m"]) else np.nan)
        ax_c.plot(p_values, c_m_vals, "o-", label=f"γ={gamma:.2f}")
    ax_c.set_xlabel("p")
    ax_c.set_ylabel(r"$c_m$ (block decay rate)")
    ax_c.set_title(r"Block-wise decay rate $c_m(p)$ vs $p$")
    ax_c.legend(fontsize=7)
    ax_c.grid(True, alpha=0.3)
    fig.suptitle(
        f"Block-wise eigenvalue decay from primary-block maxima |λ^(m)| vs m "
        f"{'(saddle)' if at_saddle else '(y=0)'}",
        y=1.01,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    suffix = "saddle" if at_saddle else "y0"
    fig.savefig(out_dir / f"eigenvalue_block_max_decay_{suffix}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    """
    CLI entry point.

    Examples
    --------
    - Default (y=0 data, p=2..5, 4 gamma values):
        python -m symmetry_reduction.eigenvalue_decay.eigenvalue_decay

    - At the saddle point y*_0 (same grids):
        python -m symmetry_reduction.eigenvalue_decay.eigenvalue_decay --at-saddle

    - Restrict to smaller p range for speed (useful with --at-saddle):
        python -m symmetry_reduction.eigenvalue_decay.eigenvalue_decay --at-saddle --p-max 4
    """
    import argparse
    import numpy as np

    parser = argparse.ArgumentParser(description="Eigenvalue decay diagnostics for H_log.")
    parser.add_argument(
        "--at-saddle",
        action="store_true",
        help="Evaluate at the true saddle y*_0 instead of at y=0.",
    )
    parser.add_argument(
        "--p-max",
        type=int,
        default=5,
        help="Maximum QAOA depth p to include in grids (min p is fixed at 2). "
             "Smaller values speed up runs, especially with --at-saddle.",
    )
    args = parser.parse_args()

    at_saddle = bool(args.at_saddle)
    p_max = max(2, int(args.p_max))
    p_values = tuple(range(2, p_max + 1))

    # Quick one-point fit printout (re-uses same at_saddle flag and does not call any grids).
    p_quick, gamma_quick = 3, 1.0
    if p_quick <= p_max:
        where_str = "saddle" if at_saddle else "y=0"
        out = eigenvalue_decay_analysis(p_quick, gamma_quick, at_saddle=at_saddle)
        print(f"Eigenvalue decay analysis p={p_quick} gamma={gamma_quick:.2f} {where_str}")
        print(
            "  Exponential fit: M=%.4e c'=%.4f R²=%.4f"
            % (
                out["fits"]["exponential"]["M"],
                out["fits"]["exponential"]["c_prime"],
                out["fits"]["exponential"]["r2"],
            )
        )
        print(
            "  Power-law fit:   M=%.4e α=%.4f R²=%.4f"
            % (
                out["fits"]["power_law"]["M"],
                out["fits"]["power_law"]["alpha"],
                out["fits"]["power_law"]["r2"],
            )
        )

    # Grids and summary plots. These are the heavy parts; allow p_max to cap work.
    gamma_values = (0.30, 1.00, float(np.pi), float(2 * np.pi))
    plot_decay_and_c_prime(p_values=p_values, gamma_values=gamma_values, at_saddle=at_saddle)

    # A single representative block-decomposition heatmap (defaults kept for compatibility).
    try:
        plot_block_decomposition_heatmap(p=3, gamma=1.0, at_saddle=at_saddle)
    except Exception:
        # If p=3 data is unavailable (e.g., very small p_max), skip without failing the run.
        pass
