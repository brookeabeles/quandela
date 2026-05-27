import numpy as np

from symmetry_reduction.core import (
    build_structure_matrix,
    compute_b_s,
    compute_c_alpha,
    softmax_weights,
)
from symmetry_reduction.saddle import saddle_with_adaptive_damping


def enumerate_saddles(
    p: int, gamma: float, r: float = 1.0, n_trials: int = 120, max_scale: float = 12.0
) -> list[tuple[np.ndarray, complex]]:
    """Try random starts and collect distinct fixed points for q=1 map."""
    betas = np.full(p, np.pi / 4)
    A = build_structure_matrix(p)
    b_s = compute_b_s(p, betas)
    d = A.shape[0]
    c_alpha = compute_c_alpha(p, np.full(p, gamma), r=r)
    sqrt_c = np.sqrt(c_alpha + 0j)

    def action(y: np.ndarray) -> complex:
        z = np.sum(b_s * np.exp((sqrt_c * y) @ A))
        return -np.sum(y ** 2) / 4 + np.log(z)

    y0, _, _ = saddle_with_adaptive_damping(A, b_s, c_alpha)
    saddles: list[tuple[np.ndarray, complex]] = [(y0, action(y0))]
    rng = np.random.default_rng(42)

    for trial in range(n_trials):
        scale = 0.3 + (trial / max(1, n_trials - 1)) * max_scale
        y = (rng.normal(size=d) + 1j * rng.normal(size=d)) * scale
        res = np.inf
        for _ in range(4000):
            w = softmax_weights(y, A, b_s, sqrt_c)
            y_new = 2.0 * sqrt_c * (w @ A.T)
            res = float(np.max(np.abs(y - y_new)))
            y = 0.95 * y + 0.05 * y_new
            if res < 1e-10:
                break

        if res < 1e-6:
            psi = action(y)
            if all(np.max(np.abs(y - yp)) > 1e-3 for yp, _ in saddles):
                saddles.append((y, psi))
    return saddles


def analyze_dominance(p: int, r: float = 1.0) -> None:
    print(f"\np={p}, r={r:.2f}")
    print(f"{'gamma/pi':>8s} {'#sad':>5s} {'bestRe':>12s} {'gap':>10s} {'stokes?':>8s}")
    for gv in np.linspace(0.1, 2 * np.pi, 8):
        saddles = enumerate_saddles(p, gv, r=r)
        re_vals = sorted((float(np.real(psi)) for _, psi in saddles), reverse=True)
        im_vals = [float(np.imag(psi)) for _, psi in saddles]
        n_s = len(saddles)
        gap = re_vals[0] - re_vals[1] if n_s > 1 else float("inf")
        stokes = any(abs(im_vals[i] - im_vals[j]) < 0.05 for i in range(n_s) for j in range(i + 1, n_s))
        print(f"{gv/np.pi:8.3f} {n_s:5d} {re_vals[0]:+12.6f} {gap:10.6f} {('YES' if stokes else 'no'):>8s}")


def main() -> None:
    for p in [1, 2]:
        analyze_dominance(p, r=1.0)
        analyze_dominance(p, r=176.54)


if __name__ == "__main__":
    main()
