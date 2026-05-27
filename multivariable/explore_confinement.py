import numpy as np

from symmetry_reduction.core import build_structure_matrix, compute_b_s, compute_c_alpha


def confinement_analysis(p: int, r: float = 1.0) -> None:
    betas = np.full(p, np.pi / 4)
    A = build_structure_matrix(p)
    _ = compute_b_s(p, betas)  # kept for parity with other probes
    d = A.shape[0]

    for gamma in [np.pi / 4, np.pi, 2 * np.pi]:
        gammas = np.full(p, gamma)
        c_alpha = compute_c_alpha(p, gammas, r=r)
        sqrt_c = np.sqrt(c_alpha + 0j)

        # max_s Σ_α |sqrt(c_α)| |A_{αs}|
        linear_rate = float(np.max(np.abs(sqrt_c) @ np.abs(A)))
        r_confine = 4.0 * linear_rate
        print(
            f"p={p}, gamma={gamma/np.pi:.2f}pi, r={r:.2f}: "
            f"linear_rate={linear_rate:.4e}, R_confine={r_confine:.4e}, d={d}"
        )


def main() -> None:
    for p in [1, 2, 3, 4, 5]:
        confinement_analysis(p, r=1.0)
        confinement_analysis(p, r=176.54)
        print()


if __name__ == "__main__":
    main()
