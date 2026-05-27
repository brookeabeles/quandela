"""
Direction 2: Blockwise spectral decay in subset-size coordinates (|α|, |α'|).

- Build Hessian blocks by (m, m') = (|α|, |α'|); compute Frobenius and operator norm per block.
- Heatmaps of spectral energy across block coordinates.
- Compare normal truncation (|α| ≤ m) vs complement (|α| ≥ n−m): Frobenius energy captured,
  determinant-ratio truncation error.
- Track how dominant blocks shift with γ; optional decay-law fits.
"""

from __future__ import annotations

import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from pathlib import Path

_repo_root = Path(__file__).resolve().parent.parent.parent
if str(_repo_root) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(_repo_root))

from symmetry_reduction.core import (
    build_structure_matrix,
    compute_b_s,
    compute_c_alpha,
    alpha_linear_coefficients,
    hessian_log_at_y,
    hessian_F_at_y,
    softmax_weights,
    covariance_from_complex_weights,
)
from symmetry_reduction.saddle import (
    saddle_with_adaptive_damping,
)
from symmetry_reduction.spectral import determinant_ratio_probe, spectral_summary
from symmetry_reduction.mechanisms import energy_by_block, block_norms, _popcount_array
from symmetry_reduction.run_sweep import DEFAULT_BETA

OUT_DIR = Path(__file__).resolve().parent
FIG_DIR = OUT_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)
RESULTS_CACHE = FIG_DIR / "direction2_complexCov_sweep_results.pkl"

# Warmstart cache for the q>1 Newton solver (per p, betas, q, r).
_NEWTON_WARMSTART: dict[tuple, tuple[float, np.ndarray]] = {}

def compute_cov_block_structure(
    p: int,
    gamma: float,
    betas: np.ndarray | None = None,
    gammas: np.ndarray | None = None,
    q: int = 1,
    r: float = 1.0,
) -> dict:
    """
    Compute the block structure of Cov_w(A) at the q=1 saddle.

    This isolates the universal part of the Hessian independent of the coeff-weighting
    (and empirically close to γ-independent for fixed (p,β,r)).

    Returns dict with:
      - "f_mm_cov": fractional block energy of Cov_w(A)
      - "f_mm_H":   fractional block energy of full H_log = diag(√c)·Cov·diag(√c)
      - "converged": bool
      - "saddle_residual": float
    """
    if betas is None:
        betas = np.full(p, DEFAULT_BETA, dtype=float)
    if gammas is None:
        gammas = np.full(p, float(gamma), dtype=float)

    A = build_structure_matrix(p, q=q)
    b_s = compute_b_s(p, betas)

    c_alpha = compute_c_alpha(p, gammas, r=r)
    y_star, converged, residual = saddle_with_adaptive_damping(A, b_s, c_alpha)

    H_log = hessian_log_at_y(y_star, A, b_s, c_alpha)
    _, f_mm_H = energy_by_block(H_log, p)

    sqrt_c = np.sqrt(c_alpha + 0j)
    w = softmax_weights(y_star, A, b_s, sqrt_c)
    Cov = covariance_from_complex_weights(A, w)
    _, f_mm_cov = energy_by_block(Cov, p)

    return {
        "f_mm_cov": f_mm_cov,
        "f_mm_H": f_mm_H,
        "converged": bool(converged),
        "saddle_residual": float(residual),
        "p": p,
        "gamma": float(gamma),
        "q": int(q),
        "r": float(r),
    }


def _get_saddle_hessian(
    p: int,
    gamma: float,
    A=None,
    b_s=None,
    betas: np.ndarray | None = None,
    gammas: np.ndarray | None = None,
    q: int = 1,
    r: float = 1.0,
):
    """
    Compute H_log at the BM24 saddle for a given (p, γ) and clause-arity parameter q (k = 2^q).

    - A is built from build_structure_matrix(p, q) if not supplied.
    - b_s uses layer-dependent betas if provided, otherwise a uniform DEFAULT_BETA.
    - c_alpha uses layer-dependent gammas if provided, otherwise a uniform scalar γ.

    The structure matrix uses the q-independent Eq. (A35); the q-dependence of 2^q-SAT
    enters through the generalized multinomial exponent in the analytic theory, not A.

    NOTE (q > 1): The returned Hessian is ∇²F (the log-partition Hessian),
    not the full action Hessian ∇²Φ* from BM24 Eq. (15). For q > 1, ∇²Φ*
    contains additional terms involving (∂_α F)^{2q-2} · ∇²F and higher-order
    cross terms. These corrections share the same underlying covariance/block
    structure (determined by A_{αs} and the softmax covariance), so the
    qualitative (|α|,|α'|) energy distribution is typically preserved, but
    magnitudes may differ.
    """
    if A is None:
        A = build_structure_matrix(p, q=q)
    if betas is None:
        betas = np.full(p, DEFAULT_BETA, dtype=float)
    if b_s is None:
        b_s = compute_b_s(p, betas)
    if gammas is None:
        gammas = np.full(p, float(gamma), dtype=float)

    # q-dependent setup (BM24 Proposition 1 / Eq. (16),(20)):
    # coeff_α := r * (-c_phase_α)^{1/(2q)}
    if q == 1:
        # Backward-compatible path (Gaussian / quadratic case used elsewhere in repo)
        c_alpha = compute_c_alpha(p, gammas, r=r)
        y_star, converged, residual = saddle_with_adaptive_damping(A, b_s, c_alpha)
        H = hessian_log_at_y(y_star, A, b_s, c_alpha)
        return H, converged, residual

    # q > 1: Newton solver in u-space with γ-homotopy.
    # The fixed-point iteration (Eq. 20) cannot converge for large r because
    # its Lipschitz constant scales as r^{2q-1} (≈10^11 for 8-SAT at r=176.54).
    from symmetry_reduction.newton_saddle_q import solve_8sat_saddle
    from symmetry_reduction.newton_saddle_q import active_indices_for_p

    cache_key = (int(p), int(q), float(r), tuple(np.asarray(betas, dtype=float).round(12)))
    u_init_active = None
    gamma_start_override = None
    if cache_key in _NEWTON_WARMSTART:
        gamma_prev, u_prev = _NEWTON_WARMSTART[cache_key]
        if float(gamma_prev) < float(gamma):
            u_init_active = u_prev
            gamma_start_override = float(gamma_prev)

    y_star, coeff_alpha, converged, residual = solve_8sat_saddle(
        p=p,
        gamma_target=float(gamma),
        betas=np.asarray(betas, dtype=float),
        q=q,
        r=float(r),
        gamma_homotopy_factor=2.0,
        gamma_max_steps=120,
        target_residual=3e-3,
        newton_max_iter=200,
        u_init_active=u_init_active,
        gamma_start_override=gamma_start_override,
        max_newton_calls=200,
        verbose=False,
    )

    if converged:
        active_idx = active_indices_for_p(p)
        _NEWTON_WARMSTART[cache_key] = (float(gamma), coeff_alpha[active_idx] * y_star[active_idx])

    H = hessian_F_at_y(y_star, A, b_s, coeff_alpha)
    return H, converged, residual


def compute_block_structure(H: np.ndarray, p: int) -> dict:
    """E_mm, f_mm, Frobenius and operator norm per block (m, m')."""
    E_mm, f_mm = energy_by_block(H, p)
    frob_mm, op_mm = block_norms(H, p)
    return {
        "E_mm": E_mm,
        "f_mm": f_mm,
        "frobenius_mm": frob_mm,
        "operator_mm": op_mm,
        "p": p,
    }


def truncation_metrics(H_full: np.ndarray, H_sub: np.ndarray, p: int, at_saddle: bool = True) -> dict:
    """Frobenius energy captured by H_sub; determinant-ratio error for H_sub (err at k0=0)."""
    frob_full_sq = np.sum(np.abs(H_full) ** 2)
    frob_sub_sq = np.sum(np.abs(H_sub) ** 2)
    energy_captured = float(frob_sub_sq / (frob_full_sq + 1e-30))
    k0_axis, err_array = determinant_ratio_probe(H_sub)
    # Truncation error: use |R(0)-1| (product over all tail directions of submatrix)
    det_err = float(err_array[0]) if len(err_array) > 0 else np.nan
    return {"energy_captured": energy_captured, "determinant_ratio_err": det_err}


def run_block_and_truncation(
    p: int,
    gamma: float,
    max_m: int | None = None,
    A=None,
    b_s=None,
    at_saddle: bool = True,
    betas: np.ndarray | None = None,
    gammas: np.ndarray | None = None,
    q: int = 1,
    r: float = 1.0,
) -> dict:
    """Block structure and normal/complement truncation for one (p, γ, q, r, β⃗, γ⃗)."""
    H, converged, residual = _get_saddle_hessian(
        p, gamma, A, b_s, betas=betas, gammas=gammas, q=q, r=r
    )
    n = 2 * p + 1
    if max_m is None:
        max_m = n
    block_data = compute_block_structure(H, p)
    d = H.shape[0]
    sizes = _popcount_array(d, n)
    results = {
        "p": p,
        "gamma": gamma,
        "q": q,
        "r": float(r),
        "converged": converged,
        "saddle_residual": float(residual),
        **block_data,
        "normal": {},
        "complement": {},
    }
    for m in range(max_m + 1):
        # Normal: |α| ≤ m
        normal_mask = sizes <= m
        n_kept = int(np.sum(normal_mask))
        if n_kept > 0 and n_kept < d:
            idx = np.where(normal_mask)[0]
            H_sub = H[np.ix_(idx, idx)]
            results["normal"][m] = {
                "n_kept": n_kept,
                **truncation_metrics(H, H_sub, p, at_saddle),
            }
        # Complement: |α| ≥ n - m  (i.e. |α^c| ≤ m)
        comp_mask = sizes >= (n - m)
        n_kept_c = int(np.sum(comp_mask))
        if n_kept_c > 0 and n_kept_c < d:
            idx_c = np.where(comp_mask)[0]
            H_sub_c = H[np.ix_(idx_c, idx_c)]
            results["complement"][m] = {
                "n_kept": n_kept_c,
                **truncation_metrics(H, H_sub_c, p, at_saddle),
            }
    return results


def plot_block_heatmaps(
    block_data: dict,
    gamma: float,
    out_dir: Path | None = None,
) -> list[Path]:
    """Heatmaps of f_mm (fractional energy) and operator norm per block."""
    if out_dir is None:
        out_dir = FIG_DIR
    p = block_data["p"]
    f_mm = block_data["f_mm"]
    op_mm = block_data["operator_mm"]
    paths = []
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    im0 = axes[0].imshow(f_mm, aspect="auto", cmap="viridis")
    axes[0].set_title(rf"Fractional Frobenius energy $f_{{m,m'}}$ (p={p}, $\gamma$={gamma:.3f})")
    axes[0].set_xlabel(r"$|\alpha'|$")
    axes[0].set_ylabel(r"$|\alpha|$")
    plt.colorbar(im0, ax=axes[0])
    im1 = axes[1].imshow(np.log10(op_mm + 1e-20), aspect="auto", cmap="viridis")
    axes[1].set_title(rf"$\log_{{10}}$ block operator norm (p={p})")
    axes[1].set_xlabel(r"$|\alpha'|$")
    axes[1].set_ylabel(r"$|\alpha|$")
    plt.colorbar(im1, ax=axes[1])
    plt.tight_layout()
    path = out_dir / f"direction2_block_heatmap_complexCov_p{p}_g{gamma:.3f}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    paths.append(path)
    return paths


def plot_truncation_comparison(
    results: dict,
    out_path: Path | None = None,
) -> Path:
    """Plot energy captured and determinant-ratio error: normal vs complement by m."""
    p, gamma = results["p"], results["gamma"]
    n = 2 * p + 1
    m_vals = sorted(set(results["normal"].keys()) | set(results["complement"].keys()))
    norm_energy = [results["normal"].get(m, {}).get("energy_captured", np.nan) for m in m_vals]
    comp_energy = [results["complement"].get(m, {}).get("energy_captured", np.nan) for m in m_vals]
    norm_err = [results["normal"].get(m, {}).get("determinant_ratio_err", np.nan) for m in m_vals]
    comp_err = [results["complement"].get(m, {}).get("determinant_ratio_err", np.nan) for m in m_vals]
    fig, axes = plt.subplots(2, 1, figsize=(6, 6), sharex=True)
    ax0, ax1 = axes
    ax0.plot(m_vals, np.array(norm_energy) * 100, "o-", label="Normal |α|≤m")
    ax0.plot(m_vals, np.array(comp_energy) * 100, "s-", label="Complement |α|≥n−m")
    ax0.set_ylabel("Frobenius energy captured (%)")
    ax0.legend()
    ax0.grid(True, alpha=0.3)
    ax0.set_title(rf"Block truncation (p={p}, $\gamma$={gamma:.3f})")
    ax1.semilogy(m_vals, np.maximum(np.abs(norm_err), 1e-16), "o-", label="Normal |α|≤m")
    ax1.semilogy(m_vals, np.maximum(np.abs(comp_err), 1e-16), "s-", label="Complement |α|≥n−m")
    ax1.set_xlabel("m")
    ax1.set_ylabel(r"$|R(k_0{=}0)-1|$ (det-ratio error)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    if out_path is None:
        out_path = FIG_DIR / f"direction2_truncation_comparison_complexCov_p{p}_g{gamma:.3f}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return out_path


def _get_result_for_heatmap(
    p: int,
    gamma: float,
    all_results: list[dict],
    A=None,
    b_s=None,
    compute_on_fly: bool = True,
) -> dict | None:
    """Get block structure for (p, gamma) from all_results or compute on the fly if compute_on_fly."""
    r = next((x for x in all_results if x["p"] == p and np.isclose(x["gamma"], gamma)), None)
    if r is not None:
        return r
    r = next((x for x in all_results if x["p"] == p), None)
    if r is not None and np.isclose(r["gamma"], gamma):
        return r
    if not compute_on_fly:
        return None
    try:
        return run_block_and_truncation(p, float(gamma), A=A, b_s=b_s, at_saddle=True)
    except Exception:
        return None


def plot_block_heatmap_summary(
    all_results: list[dict],
    choices: list[tuple[int, float]] | None = None,
    out_path: Path | None = None,
    compute_on_fly: bool = True,
) -> Path:
    """One figure: 4 rows x 4 columns. Rows p=2,3,4,5; each row γ = π/4, π/2, π, 2π.
    Shared color scale (vmin=0, vmax=1) for all panels.
    If compute_on_fly is False, missing (p,γ) are shown as 'no data' (faster when only partial data)."""
    # 16 panels: (p, γ) for p in (2,3,4,5) and γ in (π/4, π/2, π, 2π)
    gammas = (np.pi / 4, np.pi / 2, np.pi, 2.0 * np.pi)
    p_values = (2, 3, 4, 5)
    if choices is not None:
        target = choices[:16]
    else:
        target = [(p, g) for p in p_values for g in gammas]

    fig, axes = plt.subplots(4, 4, figsize=(14, 14))
    vmin, vmax = 0.0, 1.0  # shared scale for fractional energy
    # PowerNorm gamma < 1 stretches low values so color differences are more visible
    norm = mcolors.PowerNorm(gamma=0.45, vmin=vmin, vmax=vmax)
    cache_p = {}  # p -> (A, b_s) for reuse
    first_im = None
    for i, (p, gamma) in enumerate(target):
        ax = axes.flat[i]
        if p not in cache_p:
            cache_p[p] = (build_structure_matrix(p), compute_b_s(p, np.full(p, DEFAULT_BETA)))
        A, b_s = cache_p[p]
        r = _get_result_for_heatmap(p, gamma, all_results, A=A, b_s=b_s, compute_on_fly=compute_on_fly)
        if r is None:
            ax.text(0.5, 0.5, f"p={p}, γ={gamma:.3f}\n(no data)", ha="center", va="center", transform=ax.transAxes)
            ax.set_xlabel(r"$|\alpha'|$")
            ax.set_ylabel(r"$|\alpha|$")
            continue
        f_mm = r["f_mm"]
        im = ax.imshow(f_mm, aspect="auto", cmap="plasma", norm=norm)
        if first_im is None:
            first_im = im
        # Label γ as π/4, π/2, π, 2π where possible
        g_frac = gamma / np.pi
        if np.isclose(g_frac, 0.25):
            g_label = r"$\pi/4$"
        elif np.isclose(g_frac, 0.5):
            g_label = r"$\pi/2$"
        elif np.isclose(g_frac, 1.0):
            g_label = r"$\pi$"
        elif np.isclose(g_frac, 2.0):
            g_label = r"$2\pi$"
        else:
            g_label = rf"{g_frac:.2f}$\pi$"
        ax.set_title(rf"p={p}, $\gamma$={g_label}")
        ax.set_xlabel(r"$|\alpha'|$")
        ax.set_ylabel(r"$|\alpha|$")
    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.94, 0.08, 0.02, 0.82])
    if first_im is not None:
        fig.colorbar(first_im, cax=cbar_ax, label=r"Block fractional energy $f_{m,m'}$")
    fig.suptitle(r"Block fractional energy $f_{m,m'}$ — rows: p=2,3,4,5; $\gamma$ = $\pi/4$, $\pi/2$, $\pi$, $2\pi$", y=1.01)
    plt.tight_layout(rect=[0, 0, 0.92, 0.98])
    if out_path is None:
        out_path = FIG_DIR / "direction2_block_heatmap_summary_complexCov.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return out_path


def plot_truncation_summary(
    all_results: list[dict],
    choices: list[tuple[int, float]] | None = None,
    out_path: Path | None = None,
) -> Path:
    """One figure: energy captured and det-ratio error (normal vs complement) for a few (p,γ)."""
    if choices is None:
        choices = [(2, 1.0), (3, 1.0), (4, np.pi)]
    fig, axes = plt.subplots(2, 1, figsize=(7, 6), sharex=False)
    ax0, ax1 = axes
    for p, gamma in choices:
        r = next((x for x in all_results if x["p"] == p and np.isclose(x["gamma"], gamma)), None)
        if r is None:
            r = next((x for x in all_results if x["p"] == p), None)
        if r is None:
            continue
        m_vals = sorted(set(r["normal"].keys()) | set(r["complement"].keys()))
        norm_energy = [r["normal"].get(m, {}).get("energy_captured", np.nan) * 100 for m in m_vals]
        comp_energy = [r["complement"].get(m, {}).get("energy_captured", np.nan) * 100 for m in m_vals]
        norm_err = [np.maximum(np.abs(r["normal"].get(m, {}).get("determinant_ratio_err", np.nan)), 1e-16) for m in m_vals]
        comp_err = [np.maximum(np.abs(r["complement"].get(m, {}).get("determinant_ratio_err", np.nan)), 1e-16) for m in m_vals]
        label = rf"p={p}, $\gamma$={r['gamma']:.2f}"
        ax0.plot(m_vals, norm_energy, "o-", markersize=3, label=f"{label} norm")
        ax0.plot(m_vals, comp_energy, "s--", markersize=3, label=f"{label} compl")
        ax1.semilogy(m_vals, norm_err, "o-", markersize=3, label=f"{label} norm")
        ax1.semilogy(m_vals, comp_err, "s--", markersize=3, label=f"{label} compl")
    ax0.set_ylabel("Frobenius energy captured (%)")
    ax0.legend(loc="best", fontsize=7)
    ax1.legend(loc="best", fontsize=7)
    ax0.grid(True, alpha=0.3)
    ax0.set_title("Block truncation: normal |α|≤m vs complement |α|≥n−m (summary)")
    ax1.set_xlabel("m")
    ax1.set_ylabel(r"$|R(0)-1|$ (det-ratio error)")
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    if out_path is None:
        out_path = FIG_DIR / "direction2_truncation_summary_complexCov.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return out_path


def fit_block_decay(frob_mm: np.ndarray, p: int) -> dict:
    """
    Empirical decay: fit block Frobenius norms. Optional helper for blockwise decay law.
    Returns simple stats: max norm by diagonal band (m-m' = const) and by m.
    """
    n = frob_mm.shape[0] - 1
    diag_band_max = []
    for d in range(-n, n + 1):
        vals = []
        for m in range(n + 1):
            mp = m - d
            if 0 <= mp <= n:
                vals.append(frob_mm[m, mp])
        diag_band_max.append((d, max(vals)) if vals else (d, 0.0))
    return {"diag_band_max": diag_band_max, "p": p}


def load_sweep_results() -> list[dict] | None:
    """Load cached sweep results from RESULTS_CACHE. Returns None if missing or invalid."""
    if not RESULTS_CACHE.exists():
        return None
    try:
        with open(RESULTS_CACHE, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


def run_sweep(
    p_values: tuple[int, ...] = (2, 3, 4, 5),
    gamma_values: tuple[float, ...] = (0.3, 1.0, np.pi, 2 * np.pi),
    summary_only: bool = True,
    plot_all: bool = False,
    save_results: bool = True,
):
    """Run Direction 2 over (p, γ). By default only 2 summary figures; use plot_all for per-(p,γ)."""
    all_results = []
    for p in p_values:
        A = build_structure_matrix(p)
        b_s = compute_b_s(p, np.full(p, DEFAULT_BETA))
        for gamma in gamma_values:
            r = run_block_and_truncation(p, float(gamma), A=A, b_s=b_s, at_saddle=True)
            all_results.append(r)
            if plot_all:
                plot_block_heatmaps(r, r["gamma"], FIG_DIR)
                plot_truncation_comparison(r, FIG_DIR / f"direction2_truncation_p{p}_g{gamma:.3f}.png")
    if save_results and all_results:
        with open(RESULTS_CACHE, "wb") as f:
            pickle.dump(all_results, f)
    if summary_only and all_results:
        # Heatmap summary: use only results we have (no on-the-fly compute) so the figure saves quickly.
        plot_block_heatmap_summary(all_results, compute_on_fly=False)
        plot_truncation_summary(all_results)
    return all_results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Direction 2: block structure and truncation")
    parser.add_argument("--p", type=int, nargs="+", default=[2, 3, 4, 5], help="QAOA depths (default: 2,3,4,5 for 4×4 heatmap)")
    parser.add_argument("--gamma", type=float, nargs="+", default=[np.pi / 4, np.pi / 2, np.pi, 2 * np.pi])
    parser.add_argument("--all", action="store_true", help="Also generate per-(p,γ) figures (many files)")
    parser.add_argument("--plot-only", action="store_true", help="Redraw summary figures from cached results (no recompute)")
    args = parser.parse_args()
    if args.plot_only:
        all_results = load_sweep_results()
        if not all_results:
            print("No cached results at", RESULTS_CACHE, "- run without --plot-only first.")
            return
        plot_block_heatmap_summary(all_results, compute_on_fly=False)
        plot_truncation_summary(all_results)
        print("Direction 2: summary figures redrawn from cache →", FIG_DIR)
        return
    run_sweep(
        p_values=tuple(args.p),
        gamma_values=tuple(args.gamma),
        summary_only=True,
        plot_all=args.all,
    )
    print("Direction 2: 2 summary figures saved to", FIG_DIR, "(use --all for all per-(p,γ) plots)")


if __name__ == "__main__":
    main()
