from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from symmetry_reduction.core import (
    build_structure_matrix,
    compute_b_s,
    covariance_from_real_weights,
    realify_weights,
    softmax_weights_with_coeff,
)
from symmetry_reduction.newton_saddle_q import solve_8sat_saddle


def run_sweep(
    p_values: list[int],
    gamma_values: np.ndarray,
    q: int = 3,
    r: float = 176.54,
) -> dict[str, np.ndarray]:
    n_p = len(p_values)
    n_g = len(gamma_values)

    entropy = np.full((n_p, n_g), np.nan, dtype=float)
    max_weight = np.full((n_p, n_g), np.nan, dtype=float)
    cov_trace = np.full((n_p, n_g), np.nan, dtype=float)
    cov_fro = np.full((n_p, n_g), np.nan, dtype=float)
    residual = np.full((n_p, n_g), np.nan, dtype=float)
    converged = np.zeros((n_p, n_g), dtype=float)

    for i, p in enumerate(p_values):
        A = build_structure_matrix(p, q=q)
        b_s = compute_b_s(p, np.full(p, np.pi / 4))
        u_warm = None

        for j, gamma in enumerate(gamma_values):
            y_star, coeff_alpha, conv, res = solve_8sat_saddle(
                p=p,
                gamma_target=float(gamma),
                betas=np.full(p, np.pi / 4),
                q=q,
                r=r,
                gamma_homotopy_factor=1.3,
                gamma_max_steps=80,
                target_residual=1e-2,
                newton_tol=1e-8,
                newton_max_iter=80,
                max_newton_calls=60,
                u_init_active=u_warm,
            )

            d = A.shape[0]
            active_idx = np.where(np.array([bin(a).count("1") for a in range(d)]) >= 2)[0]
            u_warm = coeff_alpha[active_idx] * y_star[active_idx]

            w = softmax_weights_with_coeff(y_star, A, b_s, coeff_alpha)
            # Track B (interpretive proxy): realified weights for covariance diagnostics
            w_real = realify_weights(w, eps=0.0)

            Cov = covariance_from_real_weights(A, w_real)

            entropy[i, j] = float(-np.sum(w_real * np.log(w_real + 1e-300)))
            max_weight[i, j] = float(np.max(w_real))
            cov_trace[i, j] = float(np.real(np.trace(Cov)))
            cov_fro[i, j] = float(np.linalg.norm(Cov, ord="fro"))
            residual[i, j] = float(res)
            converged[i, j] = 1.0 if conv else 0.0

            print(
                f"p={p:2d}, gamma/pi={gamma/np.pi:6.3f}, "
                f"conv={conv}, res={res:.2e}, "
                f"H={entropy[i,j]:.4f}, wmax={max_weight[i,j]:.4e}, "
                f"trCov={cov_trace[i,j]:.4e}"
            , flush=True)

    return {
        "entropy": entropy,
        "max_weight": max_weight,
        "cov_trace": cov_trace,
        "cov_fro": cov_fro,
        "residual": residual,
        "converged": converged,
    }


def plot_heatmap(
    data: np.ndarray,
    p_values: list[int],
    gamma_values: np.ndarray,
    title: str,
    cbar_label: str,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    extent = [
        gamma_values[0] / np.pi,
        gamma_values[-1] / np.pi,
        min(p_values) - 0.5,
        max(p_values) + 0.5,
    ]
    im = ax.imshow(
        data,
        aspect="auto",
        origin="lower",
        extent=extent,
        interpolation="nearest",
    )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)
    ax.set_xlabel(r"$\gamma / \pi$")
    ax.set_ylabel("depth p")
    ax.set_yticks(p_values)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    out_dir = Path("multivariable/figures/covariance_softmax_8sat")
    out_dir.mkdir(parents=True, exist_ok=True)

    p_values = [1, 2, 3]
    gamma_values = np.linspace(0.05 * np.pi, 2.0 * np.pi, 12)

    results = run_sweep(p_values, gamma_values, q=3, r=176.54)

    plot_heatmap(
        results["cov_trace"],
        p_values,
        gamma_values,
        "8-SAT covariance trace over depth and gamma",
        "trace(Cov)",
        out_dir / "cov_trace_heatmap_realified.png",
    )
    plot_heatmap(
        results["cov_fro"],
        p_values,
        gamma_values,
        "8-SAT covariance Frobenius norm over depth and gamma",
        "||Cov||_F",
        out_dir / "cov_fro_heatmap_realified.png",
    )
    plot_heatmap(
        results["entropy"],
        p_values,
        gamma_values,
        "8-SAT softmax entropy over depth and gamma",
        "entropy",
        out_dir / "softmax_entropy_heatmap_realified.png",
    )
    plot_heatmap(
        results["max_weight"],
        p_values,
        gamma_values,
        "8-SAT max softmax weight over depth and gamma",
        "max_s w_s",
        out_dir / "softmax_max_weight_heatmap_realified.png",
    )
    plot_heatmap(
        results["converged"],
        p_values,
        gamma_values,
        "8-SAT Newton convergence map",
        "1 = converged",
        out_dir / "solver_convergence_heatmap.png",
    )
    plot_heatmap(
        np.log10(results["residual"] + 1e-30),
        p_values,
        gamma_values,
        "8-SAT Newton residual map",
        "log10 residual",
        out_dir / "solver_residual_heatmap.png",
    )

    print(f"Saved figures to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
