"""
Plots for QAOA Hessian spectral concentration (Section 5 of instructions).
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent
DATA_DIR = OUT_DIR / "data"
FIG_DIR = OUT_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)


def _set_gamma_axis_pi_fractions(ax):
    """Set x-axis to γ/π with tick labels as π fractions (0, π/4, π/2, …, 2π)."""
    ax.set_xlabel(r"$\gamma/\pi$")
    ticks = [0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
    labels = ["0", r"$\pi/4$", r"$\pi/2$", r"$3\pi/4$", r"$\pi$", r"$5\pi/4$", r"$3\pi/2$", r"$7\pi/4$", r"$2\pi$"]
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)


def load_data(p: int, saddle: bool = False):
    """Load npz for given p (y0 or saddle)."""
    suffix = "_saddle" if saddle else "_y0"
    # Prefer Track A complex-covariance filename; fall back to legacy name.
    path_trackA = DATA_DIR / f"spectral_data_p{p}_complexCov{suffix}.npz"
    path_legacy = DATA_DIR / f"spectral_data_p{p}{suffix}.npz"
    path = path_trackA if path_trackA.exists() else path_legacy
    if not path.exists():
        return None
    return dict(np.load(path, allow_pickle=True))


def load_probe1(p: int = 5) -> list[dict] | None:
    """Load Probe 1 results from data/probe1_saddle_p{p}.npz. Returns list of dicts for plot_probe1_saddle_determinant_ratio (determinant-ratio probe)."""
    path = DATA_DIR / f"probe1_saddle_p{p}.npz"
    if not path.exists():
        return None
    d = dict(np.load(path, allow_pickle=True))
    n = len(d["gamma"])
    if "err_array" not in d:
        return None
    return [
        {
            "gamma": float(d["gamma"][i]),
            "k_99": int(d["k99"][i]),
            "stable_rank": float(d["stable_rank"][i]),
            "k0_axis": np.asarray(d["k0_axis"][i]),
            "err_array": np.asarray(d["err_array"][i]),
        }
        for i in range(n)
    ]


def plot1_k99_vs_gamma(results_by_p: dict, at_saddle: bool, out_path: Path | None = None):
    """Plot 1: k_99 vs γ for each p. X-axis in γ/π."""
    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    for p, data in results_by_p.items():
        if data is None:
            continue
        x = np.asarray(data["gamma_values"]) / np.pi
        k = data["k99"]
        ax.plot(x, k, "o-", label=f"p={p}", markersize=3)
    _set_gamma_axis_pi_fractions(ax)
    ax.set_ylabel(r"$k_{99}$")
    ax.set_title(r"$k_{99}$ vs $\gamma$" + (" (at saddle $y_0^*$)" if at_saddle else " (at $y=0$)"))
    ax.legend()
    ax.grid(True, alpha=0.3)
    if out_path is None:
        out_path = FIG_DIR / f"k99_vs_gamma_{'saddle' if at_saddle else 'y0'}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot2_stable_rank_vs_gamma(results_by_p: dict, at_saddle: bool, out_path: Path | None = None):
    """Plot 2: Stable rank r_s vs γ for each p. X-axis in γ/π."""
    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    for p, data in results_by_p.items():
        if data is None:
            continue
        x = np.asarray(data["gamma_values"]) / np.pi
        r = data["stable_rank"]
        ax.plot(x, r, "o-", label=f"p={p}", markersize=3)
    _set_gamma_axis_pi_fractions(ax)
    ax.set_ylabel(r"Stable rank $r_s$")
    ax.set_title("Stable rank vs " + r"$\gamma$" + (" (at saddle)" if at_saddle else " (at y=0)"))
    ax.legend()
    ax.grid(True, alpha=0.3)
    if out_path is None:
        out_path = FIG_DIR / f"stable_rank_vs_gamma_{'saddle' if at_saddle else 'y0'}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# Shared style for combined saddle + y0 plots: same color per p, saddle solid, y0 dashed and fainter.
Y0_ALPHA = 0.5


def _plot_saddle_and_y0_overlay(
    ax,
    results_saddle: dict,
    results_y0: dict,
    x_key: str,
    y_key: str,
    markersize: int = 3,
    gamma_in_pi_fractions: bool = True,
):
    """Overlay saddle (solid) and y0 (dashed, fainter) for each p with consistent colors.
    If gamma_in_pi_fractions is True and x_key is 'gamma_values', x-axis is plotted as γ/π."""
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    ps = sorted(set(results_saddle.keys()) | set(results_y0.keys()))
    for i, p in enumerate(ps):
        color = colors[i % len(colors)]
        if results_saddle.get(p) is not None:
            d = results_saddle[p]
            x_vals = np.asarray(d[x_key]) / np.pi if (gamma_in_pi_fractions and x_key == "gamma_values") else d[x_key]
            ax.plot(
                x_vals,
                d[y_key],
                color=color,
                label=f"p={p}",
                markersize=markersize,
                linestyle="-",
                marker="o",
            )
        if results_y0.get(p) is not None:
            d = results_y0[p]
            x_vals = np.asarray(d[x_key]) / np.pi if (gamma_in_pi_fractions and x_key == "gamma_values") else d[x_key]
            ax.plot(
                x_vals,
                d[y_key],
                color=color,
                markersize=markersize,
                linestyle="--",
                marker="o",
                alpha=Y0_ALPHA,
            )
    handles, labels = ax.get_legend_handles_labels()
    handles.append(
        Line2D(
            [0],
            [0],
            color="gray",
            linestyle="--",
            alpha=Y0_ALPHA,
            label="Dashed: y=0",
        )
    )
    ax.legend(handles=handles)


def plot1_k99_vs_gamma_combined(
    results_saddle: dict,
    results_y0: dict,
    out_path: Path | None = None,
):
    """Plot 1 combined: k_99 vs γ for each p; saddle solid, y=0 dashed. X-axis in γ/π."""
    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    _plot_saddle_and_y0_overlay(ax, results_saddle, results_y0, "gamma_values", "k99")
    _set_gamma_axis_pi_fractions(ax)
    ax.set_ylabel(r"$k_{99}$")
    ax.set_title(r"$k_{99}$ vs $\gamma$")
    ax.grid(True, alpha=0.3)
    if out_path is None:
        out_path = FIG_DIR / "k99_vs_gamma_combined.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot2_stable_rank_vs_gamma_combined(
    results_saddle: dict,
    results_y0: dict,
    out_path: Path | None = None,
):
    """Plot 2 combined: Stable rank vs γ for each p; saddle solid, y=0 dashed. X-axis in γ/π."""
    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    _plot_saddle_and_y0_overlay(
        ax, results_saddle, results_y0, "gamma_values", "stable_rank"
    )
    _set_gamma_axis_pi_fractions(ax)
    ax.set_ylabel(r"Stable rank $r_s$")
    ax.set_title("Stable rank vs " + r"$\gamma$")
    ax.grid(True, alpha=0.3)
    if out_path is None:
        out_path = FIG_DIR / "stable_rank_vs_gamma_combined.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot3_frobenius_heatmap(f_mm: np.ndarray, p: int, gamma: float, out_path: Path | None = None):
    """Plot 3: Heatmap of f_{m,m'} for selected (p, γ)."""
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    im = ax.imshow(f_mm, aspect="auto", cmap="viridis")
    ax.set_xlabel(r"$|\alpha'|$")
    ax.set_ylabel(r"$|\alpha|$")
    ax.set_title(rf"Frobenius energy fraction $f_{{m,m'}}$ (p={p}, $\gamma$={gamma:.2f})")
    plt.colorbar(im, ax=ax)
    if out_path is None:
        out_path = FIG_DIR / f"frobenius_heatmap_p{p}_g{gamma:.2f}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_analyze_kept_dims(
    out_y0: dict,
    out_saddle: dict | None,
    p: int,
    gamma: float,
    out_dir: Path | None = None,
):
    """
    Plot kept/dropped α analysis and effective-dimension block structure.
    Expects out_y0 (and optionally out_saddle) with keys "kept_dropped" and
    "effective_block_structure" (from run_single_p_gamma(..., analyze_kept_dims=True)).
    Saves figures to out_dir (default FIG_DIR).
    """
    if out_dir is None:
        out_dir = FIG_DIR
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    gamma_str = f"{gamma:.4f}".replace(".", "_")

    # 1) Effective matrix block structure: f_mm_eff heatmaps (y0 and saddle)
    eff0 = out_y0.get("effective_block_structure")
    if eff0 is not None:
        fig, ax = plt.subplots(1, 1, figsize=(5.5, 5))
        im = ax.imshow(eff0["f_mm_eff"], aspect="auto", cmap="viridis")
        ax.set_xlabel(r"$|\alpha'|$")
        ax.set_ylabel(r"$|\alpha|$")
        ax.set_title(rf"Effective matrix $f_{{m,m'}}^{{\mathrm{{eff}}}}$ (y=0, p={p}, $\gamma$={gamma:.2f})")
        plt.colorbar(im, ax=ax, label="Fraction")
        fig.savefig(out_dir / f"kept_dims_eff_heatmap_y0_p{p}_g{gamma_str}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    if out_saddle is not None:
        eff_s = out_saddle.get("effective_block_structure")
        if eff_s is not None:
            fig, ax = plt.subplots(1, 1, figsize=(5.5, 5))
            im = ax.imshow(eff_s["f_mm_eff"], aspect="auto", cmap="viridis")
            ax.set_xlabel(r"$|\alpha'|$")
            ax.set_ylabel(r"$|\alpha|$")
            ax.set_title(rf"Effective matrix $f_{{m,m'}}^{{\mathrm{{eff}}}}$ (saddle, p={p}, $\gamma$={gamma:.2f})")
            plt.colorbar(im, ax=ax, label="Fraction")
            fig.savefig(out_dir / f"kept_dims_eff_heatmap_saddle_p{p}_g{gamma_str}.png", dpi=150, bbox_inches="tight")
            plt.close(fig)

    # 2) Block weights kept vs dropped (y0 and saddle)
    kd0 = out_y0.get("kept_dropped")
    if kd0 is not None:
        n_blocks = len(kd0["block_weights_kept"])
        fig, ax = plt.subplots(1, 1, figsize=(6, 3.5))
        x = np.arange(n_blocks)
        w = 0.38
        ax.bar(x - w / 2, kd0["block_weights_kept"], width=w, label="Kept", color="C0", alpha=0.9)
        ax.bar(x + w / 2, kd0["block_weights_dropped"], width=w, label="Dropped", color="C1", alpha=0.9)
        ax.set_xlabel(r"$|\alpha|$")
        ax.set_ylabel("Fraction")
        ax.set_title(rf"Block weights (y=0, k_99={out_y0.get('k_99', '?')}, p={p}, $\gamma$={gamma:.2f})")
        ax.legend()
        ax.set_xticks(x)
        fig.savefig(out_dir / f"kept_dims_block_weights_y0_p{p}_g{gamma_str}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    if out_saddle is not None:
        kd_s = out_saddle.get("kept_dropped")
        if kd_s is not None:
            n_blocks = len(kd_s["block_weights_kept"])
            fig, ax = plt.subplots(1, 1, figsize=(6, 3.5))
            x = np.arange(n_blocks)
            w = 0.38
            ax.bar(x - w / 2, kd_s["block_weights_kept"], width=w, label="Kept", color="C0", alpha=0.9)
            ax.bar(x + w / 2, kd_s["block_weights_dropped"], width=w, label="Dropped", color="C1", alpha=0.9)
            ax.set_xlabel(r"$|\alpha|$")
            ax.set_ylabel("Fraction")
            ax.set_title(rf"Block weights (saddle, k_99={out_saddle.get('k_99', '?')}, p={p}, $\gamma$={gamma:.2f})")
            ax.legend()
            ax.set_xticks(x)
            fig.savefig(out_dir / f"kept_dims_block_weights_saddle_p{p}_g{gamma_str}.png", dpi=150, bbox_inches="tight")
            plt.close(fig)

    print("Plots saved to", out_dir)


def plot4_mechanism_decomposition(
    sizes: np.ndarray,
    mean_c: np.ndarray,
    mean_wE: np.ndarray,
    mean_combined: np.ndarray,
    gamma: float,
    Gamma: float,
    out_path: Path | None = None,
):
    """Plot 4: Mean |c_α|, w(E_α), combined factor vs |α|; analytical bounds. Log y."""
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.semilogy(sizes, mean_c, "o-", label=r"Mean $|c_α|$ (M2)", markersize=4)
    ax.semilogy(sizes, mean_wE, "s-", label=r"Mean $w(E_α)$ (M1+3)", markersize=4)
    ax.semilogy(sizes, mean_combined, "^-", label=r"Mean $\sqrt{|c_α|\cdot\mathrm{Var}}$", markersize=4)
    m = np.arange(1, len(sizes) + 2)
    ax.semilogy(m, np.power(Gamma, m - 1), "--", label=rf"Phase bound $\Gamma^{{m-1}}$, $\Gamma$={Gamma:.3f}")
    ax.semilogy(m, np.power(0.5, m - 1), "--", label=r"Agreement bound $2^{1-m}$")
    ax.set_xlabel(r"$|\alpha|$")
    ax.set_ylabel("Magnitude")
    ax.set_title(rf"Mechanism decomposition vs subset size ($\gamma$={gamma:.2f})")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    if out_path is None:
        out_path = FIG_DIR / f"mechanism_vs_size_g{gamma:.2f}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot5_mechanism_crossover(gamma_values: np.ndarray, crossover_sizes: np.ndarray, out_path: Path | None = None):
    """Plot 5: Crossover size |α|* vs γ. X-axis in γ/π."""
    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    x = np.asarray(gamma_values) / np.pi
    ax.plot(x, crossover_sizes, "o-")
    _set_gamma_axis_pi_fractions(ax)
    ax.set_ylabel(r"Crossover size $|\alpha|^*$")
    ax.set_title("Mechanism crossover size vs " + r"$\gamma$")
    ax.grid(True, alpha=0.3)
    if out_path is None:
        out_path = FIG_DIR / "mechanism_crossover.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot6_softmax_diagnostics(gamma_values: np.ndarray, kappa: np.ndarray, ipr: np.ndarray, entropy: np.ndarray, k99: np.ndarray, out_path: Path | None = None):
    """Plot 6: Entropy and IPR vs γ/π (saddle). Optional kappa, k99 are accepted but not plotted."""
    x = np.asarray(gamma_values) / np.pi
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.semilogy(x, np.maximum(ipr, 1e-2), "s-", color="C1", label="IPR", markersize=3)
    ax1.set_ylabel("IPR (log)")
    ax2 = ax1.twinx()
    ax2.plot(x, entropy, "^-", color="green", label="Entropy", markersize=3)
    ax2.set_ylabel("Entropy")
    _set_gamma_axis_pi_fractions(ax1)
    fig.legend(loc="upper right", fontsize=8)
    ax1.set_title("Entropy and IPR vs " + r"$\gamma/\pi$" + " (saddle)")
    ax1.grid(True, alpha=0.3)
    if out_path is None:
        out_path = FIG_DIR / "softmax_diagnostics.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_softmax_weight_distribution(
    p: int = 5,
    gamma_list: list[float] | None = None,
    out_path: Path | None = None,
):
    """
    Plot sorted |w_s(y₀*)| vs rank (log y) for several γ; 1/d reference line;
    compute and print/annotate number of weights holding 90% mass per γ.
    """
    if gamma_list is None:
        gamma_list = [0.30, 1.00, 2.00, 3.14, 6.28]
    from .core import (
        build_structure_matrix,
        compute_b_s,
        compute_c_alpha,
        softmax_weights,
        DEFAULT_BETA,
    )
    from .saddle import saddle_with_adaptive_damping

    betas = np.full(p, DEFAULT_BETA)
    A = build_structure_matrix(p)
    b_s = compute_b_s(p, betas)
    d = A.shape[1]  # n_s = number of configurations
    uniform_level = 1.0 / d

    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(gamma_list)))
    k90_list = []
    w_max_global = 0.0

    for gamma, color in zip(gamma_list, colors):
        gammas = np.full(p, float(gamma))
        c_alpha = compute_c_alpha(p, gammas)
        y_star, _, _ = saddle_with_adaptive_damping(A, b_s, c_alpha)
        w = softmax_weights(y_star, A, b_s, np.sqrt(c_alpha + 0j))
        w_abs = np.abs(w)
        w_sorted = np.sort(w_abs)[::-1]
        w_max_global = max(w_max_global, float(w_sorted[0]) if len(w_sorted) else 0.0)
        rank = np.arange(1, len(w_sorted) + 1, dtype=float)
        total_mass = np.sum(w_abs)
        cum = np.cumsum(w_sorted)
        k90 = int(np.searchsorted(cum, 0.9 * total_mass) + 1) if total_mass > 0 else 0
        k90 = min(k90, len(w_sorted))
        k90_list.append((gamma, k90))
        ax.semilogy(rank, np.maximum(w_sorted, 1e-20), "-", color=color, linewidth=2, label=rf"$\gamma={gamma:.2f}$ ($k_{{90\%}}={k90}$)")

    ax.axhline(uniform_level, color="gray", linestyle="--", alpha=0.7, linewidth=1.5, label=rf"$1/d$ ($d={d}$)")
    ax.set_xlabel("Rank index", fontsize=11)
    ax.set_ylabel(r"$|w_s|$ (sorted descending)", fontsize=11)
    ax.set_title(rf"Softmax weight distribution at saddle $y_0^*$ (p={p})", fontsize=12)
    # Y-axis: from 1e-4 to just above max weight
    y_top = min(1.0, 2.0 * max(w_max_global, uniform_level))
    ax.set_ylim(1e-4, y_top)
    # X-axis: focus on the regime where most decay happens (first ~1024 ranks for large d)
    x_max = min(1024, d) if d > 256 else d
    ax.set_xlim(1, x_max)
    ax.tick_params(axis="both", labelsize=10)
    ax.legend(fontsize=9, loc="lower right", bbox_to_anchor=(1.0, 0.06), framealpha=0.95)
    ax.grid(True, alpha=0.4, which="both")
    if out_path is None:
        out_path = FIG_DIR / f"softmax_weight_distribution_p{p}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("Weights holding 90% of total mass (by γ):")
    for gamma, k90 in k90_list:
        print(f"  γ = {gamma:.2f}: k_90% = {k90}")


def plot7_y0_vs_saddle(gamma_values: np.ndarray, k99_y0: np.ndarray, k99_saddle: np.ndarray, stable_rank_y0: np.ndarray, stable_rank_saddle: np.ndarray, p: int, out_path: Path | None = None):
    """Plot 7: k_99(0) vs k_99(y_0^*) and stable rank; shade where saddle is better. X-axis in γ/π."""
    x = np.asarray(gamma_values) / np.pi
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    ax1.plot(x, k99_y0, "o-", label=r"$k_{99}(0)$", markersize=3)
    ax1.plot(x, k99_saddle, "s-", label=r"$k_{99}(y_0^*)$", markersize=3)
    better = k99_saddle < k99_y0
    if np.any(better):
        ax1.fill_between(x, 0, np.maximum(k99_y0, k99_saddle), where=better, alpha=0.2, color="green", label="Saddle better")
    ax1.set_ylabel(r"$k_{99}$")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_title(rf"$y=0$ vs saddle $y_0^*$ (p={p})")
    ax2.plot(x, stable_rank_y0, "o-", label=r"$r_s(0)$", markersize=3)
    ax2.plot(x, stable_rank_saddle, "s-", label=r"$r_s(y_0^*)$", markersize=3)
    _set_gamma_axis_pi_fractions(ax2)
    ax2.set_ylabel("Stable rank")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    if out_path is None:
        out_path = FIG_DIR / f"y0_vs_saddle_p{p}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot8_cumulative_energy(E_k: np.ndarray, k_max: int | None = None, gamma_label: str = "", out_path: Path | None = None):
    """Plot 8: E(k) vs k with 90%, 95%, 99% thresholds. E_k is cumulative energy fraction array."""
    if k_max is None:
        k_max = len(E_k)
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    k_axis = np.arange(1, min(k_max, len(E_k)) + 1)
    ax.plot(k_axis, E_k[: len(k_axis)], "b-")
    for eta, name in [(0.9, "90%"), (0.95, "95%"), (0.99, "99%")]:
        idx = np.searchsorted(E_k, eta)
        if idx < len(E_k):
            ax.axhline(eta, color="gray", linestyle="--", alpha=0.7)
            ax.axvline(idx + 1, color="gray", linestyle=":", alpha=0.7)
    ax.set_xlabel("$k$")
    ax.set_ylabel(r"$E(k)$ (cumulative energy fraction)")
    ax.set_title("Cumulative energy " + (gamma_label or ""))
    ax.grid(True, alpha=0.3)
    if out_path is None:
        out_path = FIG_DIR / "cumulative_energy.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot9_singular_value_decay(sigmas: np.ndarray, gamma_label: str = "", out_path: Path | None = None):
    """Plot 9: σ_k vs k on log scale."""
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    k = np.arange(1, len(sigmas) + 1)
    ax.semilogy(k, np.maximum(sigmas, 1e-16), "o-", markersize=2)
    ax.set_xlabel("$k$")
    ax.set_ylabel(r"$\sigma_k$")
    ax.set_title("Singular value decay " + (gamma_label or ""))
    ax.grid(True, alpha=0.3)
    if out_path is None:
        out_path = FIG_DIR / "singular_value_decay.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_scaling_exponent_vs_gamma(gamma_values, alpha_y0, alpha_saddle, out_path: Path | None = None):
    """Plot the power-law exponent alpha as a function of gamma. X-axis in γ/π."""
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.asarray(gamma_values) / np.pi
    ax.plot(x, alpha_y0, "o-", label=r"$\alpha$ at $y=0$", markersize=4)
    ax.plot(x, alpha_saddle, "s-", label=r"$\alpha$ at saddle", markersize=4)
    ax.axhline(2, color="red", linestyle="--", alpha=0.5, label=r"$\alpha=2$ (tractability threshold)")
    _set_gamma_axis_pi_fractions(ax)
    ax.set_ylabel(r"Scaling exponent $\alpha$")
    ax.set_title(r"$k_{99} \sim C \cdot p^\alpha$: exponent vs $\gamma$")
    ax.legend()
    ax.grid(True, alpha=0.3)
    if out_path is None:
        out_path = FIG_DIR / "scaling_exponent_vs_gamma.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_scaling_exponent_vs_gamma_with_uncertainty(
    gamma_values,
    alpha_y0,
    alpha_y0_min,
    alpha_y0_max,
    alpha_saddle,
    alpha_saddle_min,
    alpha_saddle_max,
    out_path: Path | None = None,
):
    """Plot alpha vs gamma with leave-one-out uncertainty bands for y=0 and saddle."""
    fig, ax = plt.subplots(figsize=(7, 4))

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    c_y0, c_sad = colors[0], colors[1]

    x = np.asarray(gamma_values) / np.pi
    ax.plot(x, alpha_y0, "o-", color=c_y0, label=r"$\alpha$ at $y=0$", markersize=4)
    ax.fill_between(
        x,
        alpha_y0_min,
        alpha_y0_max,
        color=c_y0,
        alpha=0.2,
    )

    ax.plot(
        x,
        alpha_saddle,
        "s-",
        color=c_sad,
        label=r"$\alpha$ at saddle",
        markersize=4,
    )
    ax.fill_between(
        x,
        alpha_saddle_min,
        alpha_saddle_max,
        color=c_sad,
        alpha=0.2,
    )

    ax.axhline(2, color="red", linestyle="--", alpha=0.5, label=r"$\alpha=2$ (tractability threshold)")
    _set_gamma_axis_pi_fractions(ax)
    ax.set_ylabel(r"Scaling exponent $\alpha$")
    ax.set_title(r"$k_{99} \sim C \cdot p^\alpha$: exponent vs $\gamma$")
    ax.legend()
    ax.grid(True, alpha=0.3)
    if out_path is None:
        out_path = FIG_DIR / "scaling_exponent_vs_gamma_with_uncertainty.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_k99_cov_vs_hessian(gamma_values, k99_hessian, k99_cov, p: int, saddle: bool = False, out_path: Path | None = None):
    """Plot k99 from Cov vs k99 from H_log as a function of gamma. X-axis in γ/π."""
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.asarray(gamma_values) / np.pi
    ax.plot(x, k99_hessian, "o-", label=r"$k_{99}$ (Hessian $H_{\log}$)", markersize=3)
    ax.plot(x, k99_cov, "s-", label=r"$k_{99}$ (Cov only)", markersize=3)
    _set_gamma_axis_pi_fractions(ax)
    ax.set_ylabel(r"$k_{99}$")
    ax.set_title(rf"$k_{{99}}^{{\mathrm{{Cov}}}}$ vs $k_{{99}}^{{\mathrm{{Hessian}}}}$ (p={p})" + (" saddle" if saddle else " y=0"))
    ax.legend()
    ax.grid(True, alpha=0.3)
    if out_path is None:
        out_path = FIG_DIR / f"k99_cov_vs_hessian_p{p}_{'saddle' if saddle else 'y0'}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_k99_cov_vs_hessian_combined(
    d_saddle: dict,
    d_y0: dict,
    p: int,
    out_path: Path | None = None,
):
    """Plot k99 Cov vs Hessian for both saddle (solid) and y0 (dashed). X-axis in γ/π."""
    fig, ax = plt.subplots(figsize=(7, 4))
    # Colors: Hessian = first color, Cov = second (match original blue/orange)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    c_hess, c_cov = colors[0], colors[1]
    x_sad = np.asarray(d_saddle["gamma_values"]) / np.pi
    x_y0 = np.asarray(d_y0["gamma_values"]) / np.pi
    # Saddle: solid
    ax.plot(
        x_sad,
        d_saddle["k99"],
        "o-",
        color=c_hess,
        label=r"$k_{99}$ (Hessian $H_{\log}$)",
        markersize=3,
    )
    ax.plot(
        x_sad,
        d_saddle["k99_cov"],
        "s-",
        color=c_cov,
        label=r"$k_{99}$ (Cov only)",
        markersize=3,
    )
    # y0: dashed, fainter
    ax.plot(
        x_y0,
        d_y0["k99"],
        "o--",
        color=c_hess,
        markersize=3,
        alpha=Y0_ALPHA,
    )
    ax.plot(
        x_y0,
        d_y0["k99_cov"],
        "s--",
        color=c_cov,
        markersize=3,
        alpha=Y0_ALPHA,
    )
    handles, labels = ax.get_legend_handles_labels()
    handles.append(
        Line2D(
            [0],
            [0],
            color="gray",
            linestyle="--",
            alpha=Y0_ALPHA,
            label="Dashed: y=0",
        )
    )
    ax.legend(handles=handles)
    _set_gamma_axis_pi_fractions(ax)
    ax.set_ylabel(r"$k_{99}$")
    ax.set_title(rf"$k_{{99}}^{{\mathrm{{Cov}}}}$ vs $k_{{99}}^{{\mathrm{{Hessian}}}}$ (p={p})")
    ax.grid(True, alpha=0.3)
    if out_path is None:
        out_path = FIG_DIR / f"k99_cov_vs_hessian_p{p}_combined.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_approximation_error_metric_investigation(
    data: dict,
    p: int,
    at_saddle: bool,
    out_path: Path | None = None,
):
    """
    Investigation: which metric (k_99 or stable rank) controls saddle-point approximation error?

    The saddle-point integral involves the Hessian; truncating to k dimensions gives error that may
    scale like (A) sum_{j>k} σ_j² (Frobenius tail mass — favors stable rank) or (B) max_{j>k} σ_j
    (spectral tail — favors k_99). This plot compares:
    - Tail mass 1-E(k): when using k_99 it is ~1% by definition; when using k≈stable_rank it varies.
    - Spectral tail σ_{k+1}/σ_1: size of first dropped singular value for k=k_99 vs k=round(r_s).
    """
    from .spectral import approximation_error_proxies

    gamma_values = data["gamma_values"]
    n = len(gamma_values)
    tail_k99 = np.zeros(n)
    tail_k_sr = np.zeros(n)
    spec_tail_k99 = np.zeros(n)
    spec_tail_k_sr = np.zeros(n)
    for i in range(n):
        sigmas = data["singular_values"][i]
        k99 = int(data["k99"][i])
        rs = float(data["stable_rank"][i])
        proxies = approximation_error_proxies(sigmas, k99, rs)
        tail_k99[i] = proxies["tail_mass_at_k99"]
        tail_k_sr[i] = proxies["tail_mass_at_k_sr"]
        spec_tail_k99[i] = proxies["spectral_tail_at_k99"]
        spec_tail_k_sr[i] = proxies["spectral_tail_at_k_sr"]

    x = np.asarray(gamma_values) / np.pi
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    label_sad = " (at saddle)" if at_saddle else " (at y=0)"
    ax1.plot(x, tail_k99, "o-", label=r"Tail mass when $k=k_{99}$ (≈1%)", markersize=3)
    ax1.plot(x, tail_k_sr, "s-", label=r"Tail mass when $k=\lceil r_s \rceil$", markersize=3)
    ax1.set_ylabel(r"Frobenius tail mass $1-E(k)$")
    ax1.set_title("Saddle-point error proxy: Frobenius tail" + label_sad + r" (p=" + str(p) + ")")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(0.01, color="gray", linestyle=":", alpha=0.5)

    ax2.semilogy(x, np.maximum(spec_tail_k99, 1e-16), "o-", label=r"Spectral tail when $k=k_{99}$", markersize=3)
    ax2.semilogy(x, np.maximum(spec_tail_k_sr, 1e-16), "s-", label=r"Spectral tail when $k=\lceil r_s \rceil$", markersize=3)
    _set_gamma_axis_pi_fractions(ax2)
    ax2.set_ylabel(r"$\sigma_{k+1} / \sigma_1$")
    ax2.set_title("Saddle-point error proxy: spectral tail (inverse / max σ)" + label_sad)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    if out_path is None:
        out_path = FIG_DIR / f"approximation_error_metric_investigation_p{p}_{'saddle' if at_saddle else 'y0'}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_probe1_saddle_determinant_ratio(
    probe_results: list[dict],
    p: int = 5,
    out_path: Path | None = None,
):
    """
    Plot Probe 1: determinant-ratio |R(k₀)| − 1 vs k₀ at the saddle. R(k₀) is the
    ratio of truncated to full Gaussian integral when treating directions beyond k₀
    as free. Mark k₀ = ⌈r_s⌉ and k₀ = k_99 on each curve. If |R−1| ≪ 1 at ⌈r_s⌉ then
    stable rank wins; if we need k_99 for that, k_99 wins.
    """
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(probe_results)))
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    for i, r in enumerate(probe_results):
        k0 = np.asarray(r["k0_axis"])
        err = np.asarray(r["err_array"])
        if len(k0) == 0 or len(err) == 0:
            continue
        gamma = r["gamma"]
        k_99 = r["k_99"]
        r_s = r["stable_rank"]
        k_sr = max(0, min(int(round(r_s)), len(err) - 1))
        k99_clip = max(0, min(k_99, len(err) - 1))
        err_pos = np.maximum(np.abs(err), 1e-16)
        ax.semilogy(k0, err_pos, "-", color=colors[i], label=rf"$\gamma$={gamma:.2f}")
        ax.scatter([k_sr], [err_pos[k_sr]], color=colors[i], marker="s", s=60, zorder=5, edgecolors="black", linewidths=0.5)
        ax.scatter([k99_clip], [err_pos[k99_clip]], color=colors[i], marker="o", s=60, zorder=5, edgecolors="black", linewidths=0.5)
    ax.axhline(1.0, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel(r"$k_0$ (number of directions kept)")
    ax.set_ylabel(r"$||R(k_0)| - 1|$ (magnitude of integral ratio error)")
    ax.set_title(rf"Probe 1 (saddle, p={p}): determinant-ratio error. Square = $\lceil r_s \rceil$, circle = $k_{{99}}$")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if out_path is None:
        out_path = FIG_DIR / f"probe1_saddle_determinant_ratio_p{p}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def generate_mechanism_plots(p=5, gamma_values=None):
    """Generate mechanism decomposition 4-panel figure at y=0 and at saddle (recompute from scratch)."""
    if gamma_values is None:
        gamma_values = [0.3, 1.0, np.pi, 2 * np.pi]
    from .core import (
        build_structure_matrix,
        compute_b_s,
        compute_c_alpha,
        softmax_weights,
        weights_at_zero,
        DEFAULT_BETA,
    )
    from .saddle import saddle_with_adaptive_damping
    from .mechanisms import mechanism_per_alpha, phase_bound_Gamma

    betas = np.full(p, DEFAULT_BETA)
    A = build_structure_matrix(p)
    b_s = compute_b_s(p, betas)
    w0 = weights_at_zero(b_s)

    for at_saddle, label in [(False, "y=0"), (True, "saddle")]:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.ravel()
        for i, gamma in enumerate(gamma_values):
            gammas = np.full(p, gamma)
            c_alpha = compute_c_alpha(p, gammas)
            if at_saddle:
                y_star, _, _ = saddle_with_adaptive_damping(A, b_s, c_alpha)
                w = softmax_weights(y_star, A, b_s, np.sqrt(c_alpha + 0j))
            else:
                w = w0
            mech, _ = mechanism_per_alpha(p, A, w, c_alpha)
            Gamma = phase_bound_Gamma(gamma)
            ax = axes[i]
            ax.semilogy(
                mech["sizes"],
                mech["mean_c_mag"],
                "o-",
                color="red",
                label=r"Mean $|c_\alpha|$ (Mech 2)",
                markersize=4,
            )
            ax.semilogy(
                mech["sizes"],
                mech["mean_w_E_alpha"],
                "s-",
                color="blue",
                label=r"Mean $w(E_\alpha)$ (Mech 1+3)",
                markersize=4,
            )
            ax.semilogy(
                mech["sizes"],
                np.maximum(mech["mean_combined"], 1e-20),
                "^-",
                color="green",
                label=r"Combined $\sqrt{|c|\cdot\mathrm{Var}}$",
                markersize=4,
            )
            m_range = np.arange(2, 2 * p + 2)
            ax.semilogy(m_range, np.power(Gamma, m_range - 1), "--", color="red", alpha=0.5, label=rf"$\Gamma^{{m-1}}$, $\Gamma$={Gamma:.2f}")
            ax.semilogy(m_range, np.power(0.5, m_range - 1), "--", color="blue", alpha=0.5, label=r"$2^{1-m}$")
            ax.set_xlabel(r"$|\alpha|$")
            ax.set_ylabel("Magnitude")
            ax.set_title(rf"$\gamma = {gamma:.2f}$" + (rf" ($\Gamma = {Gamma:.2f}$)"))
            ax.legend(fontsize=6)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(bottom=1e-10)
        fig.suptitle(f"Mechanism decomposition (p={p}, {label})", fontsize=14)
        fig.tight_layout()
        fig.savefig(FIG_DIR / f"mechanism_decomposition_p{p}_{label.replace(' ', '_')}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


def plot_cumulative_energy_overlay(p=5, gamma_values=None):
    """Overlay cumulative energy curves for several gamma values (y=0 and saddle)."""
    if gamma_values is None:
        gamma_values = [0.3, 1.0, np.pi, 2 * np.pi]
    data_y0 = load_data(p, saddle=False)
    data_saddle = load_data(p, saddle=True)
    if data_y0 is None:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(gamma_values)))
    for gamma, color in zip(gamma_values, colors):
        idx = np.argmin(np.abs(data_y0["gamma_values"] - gamma))
        g_val = float(data_y0["gamma_values"][idx])
        sigmas = data_y0["singular_values"][idx]
        total_sq = np.sum(sigmas**2) + 1e-30
        cum = np.cumsum(sigmas**2) / total_sq
        k = np.arange(1, len(cum) + 1)
        ax.semilogx(k, cum, "-", color=color, label=rf"$\gamma={g_val:.2f}$ (y=0)")
        if data_saddle is not None:
            sigmas_s = data_saddle["singular_values"][idx]
            total_sq_s = np.sum(sigmas_s**2) + 1e-30
            cum_s = np.cumsum(sigmas_s**2) / total_sq_s
            ax.semilogx(k, cum_s, "--", color=color, label=rf"$\gamma={g_val:.2f}$ (saddle)")
    for eta in [0.9, 0.95, 0.99]:
        ax.axhline(eta, color="gray", linestyle=":", alpha=0.5)
        ax.text(1.5, eta + 0.005, f"{int(eta*100)}%", fontsize=8, color="gray")
    ax.set_xlabel("k (number of singular values)")
    ax.set_ylabel("Cumulative energy fraction E(k)")
    ax.set_title(f"Cumulative energy (p={p})")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, 2 ** (2 * p + 1))
    fig.savefig(FIG_DIR / f"cumulative_energy_overlay_p{p}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_sv_decay_overlay(p=5, gamma_values=None):
    """σ_k / σ_1 vs k on log-log for several gamma values."""
    if gamma_values is None:
        gamma_values = [0.3, 1.0, np.pi]
    data = load_data(p, saddle=False)
    if data is None:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    for gamma in gamma_values:
        idx = np.argmin(np.abs(data["gamma_values"] - gamma))
        sigmas = data["singular_values"][idx]
        sigmas = sigmas[sigmas > 1e-15]
        if len(sigmas) == 0:
            continue
        k = np.arange(1, len(sigmas) + 1)
        ax.loglog(k, sigmas / sigmas[0], "o-", markersize=2, label=rf"$\gamma={data['gamma_values'][idx]:.2f}$")
    ax.set_xlabel("k")
    ax.set_ylabel(r"$\sigma_k / \sigma_1$")
    ax.set_title(f"Singular value decay (p={p}, y=0)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(FIG_DIR / f"sv_decay_p{p}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot10_scaling_with_p(p_values: np.ndarray, k99_values: np.ndarray, gamma_label: str = "", out_path: Path | None = None):
    """Plot 10: k_99 vs p on log-log; fit power law k_99 = c p^α."""
    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    p_values = np.asarray(p_values)
    k99_values = np.asarray(k99_values)
    mask = (p_values >= 1) & (k99_values > 0)
    if np.sum(mask) < 2:
        plt.close(fig)
        return
    log_p = np.log(p_values[mask])
    log_k = np.log(k99_values[mask])
    coeffs = np.polyfit(log_p, log_k, 1)
    alpha = coeffs[0]
    ax.loglog(p_values, k99_values, "o-", label="data")
    ax.loglog(p_values[mask], np.exp(np.polyval(coeffs, log_p)), "--", label=rf"fit $k_{{99}} \propto p^{{\alpha}}$, $\alpha$={alpha:.3f}")
    ax.set_xlabel("$p$")
    ax.set_ylabel(r"$k_{99}$")
    ax.set_title("Scaling with $p$ " + (gamma_label or ""))
    ax.legend()
    ax.grid(True, alpha=0.3)
    if out_path is None:
        out_path = FIG_DIR / "k99_vs_p.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def generate_all_plots_from_saved(p_max: int = 5):
    """Generate plots from saved npz files (after run_sweep has been run).
    Use p_max=6 to include p=6 if spectral_data_p6_*.npz exist."""
    results_y0 = {p: load_data(p, saddle=False) for p in range(1, p_max + 1)}
    results_saddle = {p: load_data(p, saddle=True) for p in range(1, p_max + 1)}

    # Combined plots: saddle solid, y0 dashed and fainter, same color per p
    plot1_k99_vs_gamma_combined(results_saddle, results_y0)
    plot2_stable_rank_vs_gamma_combined(results_saddle, results_y0)

    # Frobenius heatmap: only for p that have energy_by_size (p=6 run_p6 does not save it)
    for p in range(4, p_max + 1):
        d = load_data(p, saddle=False)
        if d is not None and "energy_by_size" in d:
            gammas_sel = [0.3, 1.0, np.pi]
            for g in gammas_sel:
                idx = np.argmin(np.abs(d["gamma_values"] - g))
                plot3_frobenius_heatmap(d["energy_by_size"][idx] / (np.sum(d["energy_by_size"][idx]) + 1e-30), p, float(d["gamma_values"][idx]))

    for p in [5]:
        d_s = load_data(p, saddle=True)
        if d_s is not None:
            for g in [0.3, 1.0, np.pi]:
                idx = np.argmin(np.abs(d_s["gamma_values"] - g))
                # Need mechanism data; not saved in npz. Skip plot4 from saved or compute on fly.
                pass

    # y0 vs saddle: all p with both y0 and saddle data
    for p in range(4, p_max + 1):
        d_y0 = load_data(p, saddle=False)
        d_s = load_data(p, saddle=True)
        if d_y0 is not None and d_s is not None:
            plot7_y0_vs_saddle(
                d_y0["gamma_values"],
                d_y0["k99"],
                d_s["k99"],
                d_y0["stable_rank"],
                d_s["stable_rank"],
                p,
            )

    # Softmax diagnostics (saddle): all p with saddle data and kappa/ipr/entropy
    for p in range(4, p_max + 1):
        d = load_data(p, saddle=True)
        if d is not None and "kappa" in d and "ipr" in d and "entropy" in d:
            plot6_softmax_diagnostics(
                d["gamma_values"],
                d["kappa"],
                d["ipr"],
                d["entropy"],
                d["k99"],
                out_path=FIG_DIR / f"softmax_diagnostics_realified_p{p}.png",
            )

    # Scaling exponent vs gamma (use p up to p_max when data available, with leave-one-out bands)
    try:
        from .scaling import get_alphas_with_leave_one_out
        p_vals = tuple(p for p in range(2, p_max + 1))
        g_y0, a_y0, a_y0_min, a_y0_max = get_alphas_with_leave_one_out(p_values=p_vals, saddle=False)
        g_sad, a_sad, a_sad_min, a_sad_max = get_alphas_with_leave_one_out(p_values=p_vals, saddle=True)
        if len(g_y0) > 0 and len(g_y0) == len(g_sad):
            plot_scaling_exponent_vs_gamma_with_uncertainty(
                g_y0,
                a_y0,
                a_y0_min,
                a_y0_max,
                a_sad,
                a_sad_min,
                a_sad_max,
            )
    except Exception:
        pass

    # Mechanism decomposition (recompute from scratch) for p=5 and p=6
    for p in [5, 6]:
        if load_data(p, saddle=True) is not None:
            generate_mechanism_plots(p=p)

    # Cumulative energy and SV decay overlays for p=5 and p=6
    for p in [5, 6]:
        if load_data(p, saddle=False) is not None:
            plot_cumulative_energy_overlay(p=p)
            plot_sv_decay_overlay(p=p)

    # k99 Cov vs Hessian (combined saddle + y0)
    for p in range(4, p_max + 1):
        d0 = load_data(p, saddle=False)
        d1 = load_data(p, saddle=True)
        if d0 is not None and d1 is not None and "k99_cov" in d0 and "k99_cov" in d1:
            plot_k99_cov_vs_hessian_combined(d1, d0, p)

    # Approximation error metric investigation (k_99 vs stable rank)
    for p in range(4, p_max + 1):
        d0 = load_data(p, saddle=False)
        d1 = load_data(p, saddle=True)
        if d0 is not None:
            plot_approximation_error_metric_investigation(d0, p, at_saddle=False)
        if d1 is not None:
            plot_approximation_error_metric_investigation(d1, p, at_saddle=True)

    print("Plots saved to", FIG_DIR)


if __name__ == "__main__":
    import sys
    p_max = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    generate_all_plots_from_saved(p_max=p_max)
