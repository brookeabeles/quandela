import numpy as np

from symmetry_reduction.core import build_structure_matrix, compute_b_s, compute_c_alpha
from symmetry_reduction.saddle import saddle_fixed_point


def check_entirety(p: int, gamma: float, r: float = 1.0, n_rays: int = 400, n_t: int = 40) -> float:
    """Probe |Z(y)| near the q=1 saddle along random complex directions."""
    betas = np.full(p, np.pi / 4)
    A = build_structure_matrix(p)
    b_s = compute_b_s(p, betas)
    gammas = np.full(p, gamma)
    c_alpha = compute_c_alpha(p, gammas, r=r)
    sqrt_c = np.sqrt(c_alpha + 0j)

    y_star, conv, res = saddle_fixed_point(A, b_s, c_alpha, rho=0.5, max_iter=5000, tol=1e-12)

    def Z(y: np.ndarray) -> complex:
        exponents = (sqrt_c * y) @ A
        return np.sum(b_s * np.exp(exponents))

    z0 = Z(y_star)
    min_abs_z = float(np.abs(z0))
    rng = np.random.default_rng(0)
    for _ in range(n_rays):
        direction = rng.normal(size=A.shape[0]) + 1j * rng.normal(size=A.shape[0])
        direction /= np.linalg.norm(direction) + 1e-300
        for t in np.linspace(0.0, 5.0, n_t):
            zt = Z(y_star + t * direction)
            min_abs_z = min(min_abs_z, float(np.abs(zt)))

    gpi = gamma / np.pi
    print(
        f"p={p}, gamma={gpi:.2f}pi, r={r:.2f}: "
        f"|Z(y*)|={abs(z0):.6e}, min|Z|={min_abs_z:.6e}, conv={conv}, res={res:.2e}"
    )
    return min_abs_z


def main() -> None:
    for p in [1, 2, 3]:
        for gamma in [np.pi / 4, np.pi, 2 * np.pi]:
            check_entirety(p, gamma, r=1.0)
        print()


if __name__ == "__main__":
    main()
