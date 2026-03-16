"""
Direction 1: Bridge saddle integral error to spectral tail quantities.

Computes for saddle Hessian H_log:
- Eigenvalues λ_j, determinant-ratio R(k₀), log R(k₀), |R(k₀)−1|
- Spectral tail metrics: nuclear Σ_{j>k₀}|λ_j|, Frobenius Σ|λ_j|², operator max|λ_j|
Plots |R(k₀)−1| vs each tail to identify which controls truncation error.
Scaling over p and γ.
"""

from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path

# Ensure repo root on path for "from symmetry_reduction ..."
import sys
# Parent of symmetry_reduction (repo root) so that "from symmetry_reduction ..." works
_repo_root = Path(__file__).resolve().parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))
from symmetry_reduction.core import (
    build_structure_matrix,
    compute_b_s,
    compute_c_alpha,
    hessian_log_at_y,
)
from symmetry_reduction.saddle import saddle_with_adaptive_damping
from symmetry_reduction.spectral import (
    determinant_ratio_and_tail_metrics,
    spectral_summary,
)
from symmetry_reduction.run_sweep import DEFAULT_BETA

OUT_DIR = Path(__file__).resolve().parent
FIG_DIR = OUT_DIR / "figures"
DATA_DIR = OUT_DIR / "data"
FIG_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)


def _set_gamma_axis_pi_fractions(ax):
    """Set x-axis to γ/π with tick labels as π fractions."""
    ax.set_xlabel(r"$\gamma/\pi$")
    ticks = [0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
    labels = ["0", r"$\pi/4$", r"$\pi/2$", r"$3\pi/4$", r"$\pi$", r"$5\pi/4$", r"$3\pi/2$", r"$7\pi/4$", r"$2\pi$"]
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)


def run_single(p: int, gamma: float, A=None, b_s=None):
    """Compute saddle Hessian and full tail + R(k₀) metrics for one (p, γ)."""
    if A is None:
        A = build_structure_matrix(p)
    if b_s is None:
        b_s = compute_b_s(p, np.full(p, DEFAULT_BETA))
    gammas = np.full(p, float(gamma))
    c_alpha = compute_c_alpha(p, gammas)
    y_star, converged, _ = saddle_with_adaptive_damping(A, b_s, c_alpha)
    H = hessian_log_at_y(y_star, A, b_s, c_alpha)
    spec = spectral_summary(H)
    data = determinant_ratio_and_tail_metrics(H)
    data["p"] = p
    data["gamma"] = gamma
    data["converged"] = converged
    data["k_99"] = spec["k_99"]
    data["stable_rank"] = spec["stable_rank"]
    return data


def run_sweep(
    p_values: tuple[int, ...] = (2, 3, 4, 5),
    gamma_values: tuple[float, ...] = (0.3, 1.0, np.pi, 2 * np.pi),
):
    """Run Direction 1 over (p, γ); return list of result dicts."""
    results = []
    A_cache = {}
    b_s_cache = {}
    for p in p_values:
        A_cache[p] = build_structure_matrix(p)
        b_s_cache[p] = compute_b_s(p, np.full(p, DEFAULT_BETA))
    for p in p_values:
        for gamma in gamma_values:
            data = run_single(p, gamma, A=A_cache[p], b_s=b_s_cache[p])
            results.append(data)
    return results


def _scatter_tail_vs_err(ax, k0_axis, err, nuclear, frob, op_tail, label_prefix="", max_points=200):
    """Plot err vs each tail; use k0 as implicit (one point per k0)."""
    n = len(k0_axis)
    step = max(1, n // max_points) if n > max_points else 1
    idx = np.arange(0, n, step)
    if len(idx) == 0:
        return
    e = err[idx]
    nu = nuclear[idx]
    fr = frob[idx]
    op = op_tail[idx]
    # Avoid log(0)
    e_plot = np.maximum(e, 1e-16)
    ax.scatter(nu, e_plot, s=8, alpha=0.6, label=label_prefix + "nuclear Σ|λ|", c="C0")
    ax.scatter(fr, e_plot, s=8, alpha=0.6, label=label_prefix + "Frobenius Σ|λ|²", c="C1")
    ax.scatter(op, e_plot, s=8, alpha=0.6, label=label_prefix + "operator max|λ|", c="C2")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Spectral tail quantity")
    ax.set_ylabel(r"$|R(k_0) - 1|$")
    ax.legend(loc="best", fontsize=7)
    ax.grid(True, alpha=0.3)


def plot_err_vs_tails_single(data: dict, out_path: Path | None = None):
    """Single (p, γ): one figure with err vs nuclear, Frobenius, operator tail."""
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    k0 = data["k0_axis"]
    err = data["err"]
    nuclear = data["nuclear_tail"]
    frob = data["frobenius_tail"]
    op = data["operator_tail"]
    p, g = data["p"], data["gamma"]
    _scatter_tail_vs_err(ax, k0, err, nuclear, frob, op)
    ax.set_title(rf"$|R(k_0)-1|$ vs spectral tails (p={p}, $\gamma$={g:.3f})")
    plt.tight_layout()
    if out_path is None:
        out_path = FIG_DIR / f"direction1_err_vs_tails_p{p}_g{data['gamma']:.3f}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return out_path


def plot_err_vs_tails_combined(results: list[dict], out_path: Path | None = None):
    """Combined plot: all (p, γ) points; color by p or gamma."""
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    for d in results:
        k0 = d["k0_axis"]
        err = d["err"]
        nuc = d["nuclear_tail"]
        fr = d["frobenius_tail"]
        op = d["operator_tail"]
        n = len(k0)
        step = max(1, n // 150)
        idx = np.arange(0, n, step)
        e = np.maximum(err[idx], 1e-16)
        ax.scatter(nuc[idx], e, s=4, alpha=0.4, c="C0")
        ax.scatter(fr[idx], e, s=4, alpha=0.4, c="C1")
        ax.scatter(op[idx], e, s=4, alpha=0.4, c="C2")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Spectral tail quantity")
    ax.set_ylabel(r"$|R(k_0) - 1|$")
    ax.set_title("Truncation error vs tail metrics (all p, γ)")
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="C0", markersize=6),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="C1", markersize=6),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="C2", markersize=6),
    ]
    labels = [r"nuclear $\sum|\lambda|$", r"Frobenius $\sum|\lambda|^2$", r"operator $\max|\lambda|$"]
    ax.legend(handles=handles, labels=labels, loc="best")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if out_path is None:
        out_path = FIG_DIR / "direction1_err_vs_tails_combined.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return out_path


def plot_logR_vs_nuclear(results: list[dict], out_path: Path | None = None):
    """Theory: |log R(k₀)| ≤ C Σ_{j>k₀} |λ_j|. Plot log R vs nuclear tail. Colors = p."""
    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    p_vals = sorted(set(d["p"] for d in results))
    cmap = plt.get_cmap("tab10")
    colors = {p: cmap(i % 10) for i, p in enumerate(p_vals)}
    for p in p_vals:
        subset = [d for d in results if d["p"] == p]
        for d in subset:
            k0 = d["k0_axis"]
            logR = np.abs(d["log_R"])
            nuc = d["nuclear_tail"]
            n = len(k0)
            step = max(1, n // 120)
            idx = np.arange(0, n, step)
            ax.scatter(nuc[idx], np.maximum(logR[idx], 1e-16), s=6, alpha=0.5, c=[colors[p]])
    ax.plot([1e-6, 1], [1e-6, 1], "k--", alpha=0.5, label="y=x")
    # One legend entry per p: use proxy artists (same color as scatter)
    legend_handles = [Line2D([0], [0], marker="o", linestyle="", color=colors[p], label=rf"$p={p}$", markersize=6) for p in p_vals]
    legend_handles.append(Line2D([0], [0], color="k", linestyle="--", alpha=0.5, label="y=x"))
    ax.legend(handles=legend_handles)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Spectral tail: $\sum_{j>k_0} |\lambda_j|$")
    ax.set_ylabel(r"$|\log R(k_0)|$")
    ax.set_title(r"$|\log R(k_0)|$ vs nuclear tail (theory: $\lesssim$ tail)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if out_path is None:
        out_path = FIG_DIR / "direction1_logR_vs_nuclear_tail.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return out_path


def plot_scaling_by_gamma(results: list[dict], p_fixed: int = 5, out_path: Path | None = None):
    """For fixed p, plot tail metrics and err at k₀ = k_99 vs γ."""
    by_p = [d for d in results if d["p"] == p_fixed]
    if not by_p:
        return None
    gammas = [d["gamma"] for d in by_p]
    k99_list = [d["k_99"] for d in by_p]
    err_at_k99 = []
    nuc_at_k99 = []
    frob_at_k99 = []
    op_at_k99 = []
    for d in by_p:
        k99 = min(d["k_99"], len(d["k0_axis"]) - 1)
        if k99 < 0:
            err_at_k99.append(np.nan)
            nuc_at_k99.append(np.nan)
            frob_at_k99.append(np.nan)
            op_at_k99.append(np.nan)
        else:
            err_at_k99.append(d["err"][k99])
            nuc_at_k99.append(d["nuclear_tail"][k99])
            frob_at_k99.append(d["frobenius_tail"][k99])
            op_at_k99.append(d["operator_tail"][k99])
    x = np.asarray(gammas) / np.pi
    fig, axes = plt.subplots(2, 1, figsize=(6, 6), sharex=True)
    ax0, ax1 = axes
    ax0.semilogy(x, np.maximum(err_at_k99, 1e-16), "o-", label=r"$|R(k_{99})-1|$")
    ax0.set_ylabel(r"$|R(k_{99})-1|$")
    ax0.legend()
    ax0.grid(True, alpha=0.3)
    ax0.set_title(rf"Truncation error at $k_0=k_{{99}}$ (p={p_fixed})")
    ax1.semilogy(x, nuc_at_k99, "s-", label="nuclear tail")
    ax1.semilogy(x, frob_at_k99, "^-", label="Frobenius tail")
    ax1.semilogy(x, op_at_k99, "d-", label="operator tail")
    _set_gamma_axis_pi_fractions(ax1)
    ax1.set_ylabel("Tail at k₀=k₉₉")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    if out_path is None:
        out_path = FIG_DIR / f"direction1_scaling_gamma_p{p_fixed}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return out_path


def plot_scaling_summary(results: list[dict], out_path: Path | None = None):
    """One figure: truncation error and nuclear tail at k₀=k₉₉ vs γ, all p on same axes."""
    p_vals = sorted(set(d["p"] for d in results))
    fig, axes = plt.subplots(2, 1, figsize=(6, 5), sharex=True)
    ax0, ax1 = axes
    for p in p_vals:
        by_p = [d for d in results if d["p"] == p]
        if not by_p:
            continue
        gammas = np.array([d["gamma"] for d in by_p])
        x = gammas / np.pi
        err_at_k99 = []
        nuc_at_k99 = []
        for d in by_p:
            k99 = min(d["k_99"], len(d["k0_axis"]) - 1)
            if k99 < 0:
                err_at_k99.append(np.nan)
                nuc_at_k99.append(np.nan)
            else:
                err_at_k99.append(d["err"][k99])
                nuc_at_k99.append(d["nuclear_tail"][k99])
        err_at_k99 = np.maximum(np.array(err_at_k99), 1e-16)
        ax0.semilogy(x, err_at_k99, "o-", label=f"p={p}", markersize=4)
        ax1.semilogy(x, nuc_at_k99, "s-", label=f"p={p}", markersize=4)
    ax0.set_ylabel(r"$|R(k_{99})-1|$")
    ax0.set_title("Truncation error at k₀=k₉₉ vs γ/π (all p)")
    ax0.legend()
    ax0.grid(True, alpha=0.3)
    _set_gamma_axis_pi_fractions(ax1)
    ax1.set_ylabel(r"Nuclear tail $\sum_{j>k_{99}}|\lambda_j|$")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    if out_path is None:
        out_path = FIG_DIR / "direction1_scaling_summary.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return out_path


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Direction 1: spectral tail vs truncation error")
    parser.add_argument("--p", type=int, nargs="+", default=[2, 3, 4, 5], help="QAOA depths")
    parser.add_argument("--gamma", type=float, nargs="+", default=[0.3, 1.0, np.pi, 2 * np.pi], help="γ values")
    parser.add_argument("--single", action="store_true", help="Single (p,γ) then exit")
    parser.add_argument("--no-plot", action="store_true", help="Skip plots")
    parser.add_argument("--all", action="store_true", help="Also generate per-p scaling figures (4 extra)")
    args = parser.parse_args()
    p_vals = tuple(args.p)
    g_vals = tuple(args.gamma)
    if args.single:
        p, g = p_vals[0], g_vals[0]
        print(f"Running single (p={p}, γ={g})...")
        data = run_single(p, g)
        print(f"  k_99={data['k_99']}, stable_rank={data['stable_rank']:.2f}, n_ev={len(data['eigenvalues'])}")
        if not args.no_plot:
            plot_err_vs_tails_single(data)
            print("  Plot saved to", FIG_DIR)
        return
    print("Running sweep over p=", p_vals, "gamma=", g_vals)
    results = run_sweep(p_values=p_vals, gamma_values=g_vals)
    if not args.no_plot:
        plot_err_vs_tails_combined(results)
        plot_logR_vs_nuclear(results)
        plot_scaling_summary(results)
        if args.all:
            for p in p_vals:
                plot_scaling_by_gamma(results, p_fixed=p)
        print("Plots saved to", FIG_DIR, "(3 summary figures; use --all for per-p scaling)")
    print("Done.")


if __name__ == "__main__":
    main()
