"""
Direction 1: compare tail metrics as predictors of truncation error.

Given the per-(p, γ) spectral-tail data from `direction1_spectral_tail`,
this script generates a 1×4 panel figure:
    - |R(k₀)−1| vs nuclear tail   Σ|λ|
    - |R(k₀)−1| vs Frobenius tail Σ|λ|²
    - |R(k₀)−1| vs operator tail  max|λ|
    - |R(k₀)−1| vs k₀ / r_s       (stable-rank-normalized truncation level)

Each panel is log–log with a least-squares line in log space and its R²,
so it is visually clear which metric is the tightest predictor. The
stable-rank panel probes the hypothesis that the *number of directions
kept relative to the stable rank* controls truncation error.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.linear_model import LinearRegression  # type: ignore[import-untyped]

from direction1_spectral_tail import (  # type: ignore[import]
    FIG_DIR,
    run_sweep,
)


def _flatten_results(
    results: list[dict],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Collect all (metric, err) pairs across (p, γ, k₀) into flat arrays.

    Returns:
        nuclear_all, frobenius_all, operator_all, k_over_rs_all, err_all
    where k_over_rs_all = (k₀+1)/r_s for each sample (stable-rank-normalized
    truncation level).
    """
    err_list: list[np.ndarray] = []
    nuc_list: list[np.ndarray] = []
    frob_list: list[np.ndarray] = []
    op_list: list[np.ndarray] = []
    k_over_rs_list: list[np.ndarray] = []

    for d in results:
        err = np.asarray(d["err"], dtype=float)
        nuc = np.asarray(d["nuclear_tail"], dtype=float)
        frob = np.asarray(d["frobenius_tail"], dtype=float)
        op = np.asarray(d["operator_tail"], dtype=float)
        k0_axis = np.asarray(d["k0_axis"], dtype=float)
        rs = float(d.get("stable_rank", 0.0))

        # Avoid zeros (problematic on log scales); also require positive stable rank.
        if rs <= 0:
            continue
        k_over_rs = (k0_axis + 1.0) / rs
        mask = (err > 0) & (nuc > 0) & (frob > 0) & (op > 0) & (k_over_rs > 0)

        if not np.any(mask):
            continue

        err_list.append(err[mask])
        nuc_list.append(nuc[mask])
        frob_list.append(frob[mask])
        op_list.append(op[mask])
        k_over_rs_list.append(k_over_rs[mask])

    if not err_list:
        return (
            np.array([], dtype=float),
            np.array([], dtype=float),
            np.array([], dtype=float),
            np.array([], dtype=float),
            np.array([], dtype=float),
        )

    err_all = np.concatenate(err_list)
    nuc_all = np.concatenate(nuc_list)
    frob_all = np.concatenate(frob_list)
    op_all = np.concatenate(op_list)
    k_over_rs_all = np.concatenate(k_over_rs_list)
    return nuc_all, frob_all, op_all, k_over_rs_all, err_all


def _scatter_with_fit(ax: plt.Axes, x: np.ndarray, y: np.ndarray, label: str, color: str) -> None:
    """Scatter log–log data and overlay a least-squares line in log space."""
    if x.size == 0 or y.size == 0:
        ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center", va="center")
        ax.set_xlabel(label)
        ax.set_xscale("log")
        ax.set_yscale("log")
        return

    # Downsample for plotting if enormous
    n = x.size
    if n > 5000:
        idx = np.linspace(0, n - 1, 5000, dtype=int)
        x_plot = x[idx]
        y_plot = y[idx]
    else:
        x_plot = x
        y_plot = y

    ax.scatter(x_plot, y_plot, s=4, alpha=0.4, color=color)

    # Fit line in log10 space
    X = np.log10(x.reshape(-1, 1))
    Y = np.log10(y)
    reg = LinearRegression().fit(X, Y)
    r2 = reg.score(X, Y)

    xs = np.logspace(np.min(np.log10(x)), np.max(np.log10(x)), 200)
    Ys = reg.predict(np.log10(xs).reshape(-1, 1))
    ax.plot(xs, 10.0**Ys, color="k", lw=2)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(label)
    ax.set_title(f"{label}\nR² = {r2:.3f}")
    ax.grid(True, alpha=0.3)


def plot_tail_metric_comparison(results: list[dict], out_path: Path | None = None) -> Path:
    """Create 4-panel figure comparing nuclear / Frobenius / operator / stable-rank metrics."""
    nuc_all, frob_all, op_all, k_over_rs_all, err_all = _flatten_results(results)
    err_all = np.maximum(err_all, 1e-16)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=True)
    ax_nuc, ax_frob, ax_op, ax_sr = axes

    _scatter_with_fit(ax_nuc, nuc_all, err_all, r"nuclear tail $\sum|\lambda|$", "C0")
    _scatter_with_fit(ax_frob, frob_all, err_all, r"Frobenius tail $\sum|\lambda|^2$", "C1")
    _scatter_with_fit(ax_op, op_all, err_all, r"operator tail $\max|\lambda|$", "C2")
    _scatter_with_fit(ax_sr, k_over_rs_all, err_all, r"stable-rank metric $(k_0+1)/r_s$", "C3")

    ax_nuc.set_ylabel(r"$|R(k_0) - 1|$")
    fig.suptitle("Which spectral tail best predicts truncation error?", y=1.02)
    fig.tight_layout()

    if out_path is None:
        out_path = FIG_DIR / "direction1_tail_metric_comparison.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Compare nuclear, Frobenius, and operator spectral tails as predictors of truncation error.",
    )
    parser.add_argument("--p", type=int, nargs="+", default=[2, 3, 4, 5], help="QAOA depths")
    parser.add_argument(
        "--gamma",
        type=float,
        nargs="+",
        default=[0.3, 1.0, float(np.pi), 2.0 * float(np.pi)],
        help="γ values",
    )
    args = parser.parse_args()

    results = run_sweep(p_values=tuple(args.p), gamma_values=tuple(args.gamma))
    out_path = plot_tail_metric_comparison(results)
    print("Tail-metric comparison figure saved to", out_path)


if __name__ == "__main__":
    main()

