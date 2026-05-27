import numpy as np

from symmetry_reduction.core import build_structure_matrix, compute_b_s, compute_c_alpha, hessian_log_at_y
from symmetry_reduction.saddle import saddle_with_adaptive_damping


def nondegeneracy_scan(p: int, r: float = 1.0, n_gamma: int = 40) -> list[tuple[float, float, float, bool]]:
    """Track min|eig(∇²Ψ)| and min distance of eig(H_log) from 1/2 across gamma."""
    betas = np.full(p, np.pi / 4)
    A = build_structure_matrix(p)
    b_s = compute_b_s(p, betas)
    d = A.shape[0]
    gamma_grid = np.linspace(0.01, 2 * np.pi, n_gamma)
    results = []

    for gv in gamma_grid:
        gammas = np.full(p, gv)
        c_alpha = compute_c_alpha(p, gammas, r=r)
        y_star, conv, _ = saddle_with_adaptive_damping(A, b_s, c_alpha)
        if not conv:
            results.append((float(gv), np.nan, np.nan, False))
            continue

        h_log = hessian_log_at_y(y_star, A, b_s, c_alpha)
        h_full = -0.5 * np.eye(d) + h_log
        eigvals_full = np.linalg.eigvals(h_full)
        eigvals_h = np.linalg.eigvals(h_log)
        min_abs_eig = float(np.min(np.abs(eigvals_full)))
        min_dist_half = float(np.min(np.abs(eigvals_h - 0.5)))
        results.append((float(gv), min_abs_eig, min_dist_half, True))

    print(f"p={p}, r={r:.2f}")
    valid = [x for x in results if x[3]]
    if valid:
        print(f"  min_gamma min|eig(∇²Ψ)| = {min(v[1] for v in valid):.4e}")
        print(f"  min_gamma dist(eig(H_log), 1/2) = {min(v[2] for v in valid):.4e}")
    else:
        print("  no converged points")
    return results


def main() -> None:
    for p in [1, 2, 3, 4]:
        nondegeneracy_scan(p, r=1.0, n_gamma=40)
        nondegeneracy_scan(p, r=176.54, n_gamma=40)
        print()


if __name__ == "__main__":
    main()
