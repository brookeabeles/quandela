import numpy as np

from symmetry_reduction.core import (
    build_structure_matrix,
    compute_b_s,
    compute_c_alpha,
    softmax_weights,
)
from symmetry_reduction.saddle import saddle_with_adaptive_damping


def action_decomposition(p: int, gamma: float, r: float = 1.0) -> dict:
    betas = np.full(p, np.pi / 4)
    A = build_structure_matrix(p)
    b_s = compute_b_s(p, betas)
    d = A.shape[0]
    c_alpha = compute_c_alpha(p, np.full(p, gamma), r=r)
    sqrt_c = np.sqrt(c_alpha + 0j)

    y, conv, res = saddle_with_adaptive_damping(A, b_s, c_alpha)
    kinetic = float(np.real(-np.sum(y ** 2) / 4))
    z = np.sum(b_s * np.exp((sqrt_c * y) @ A))
    log_partition = float(np.real(np.log(z)))
    total = kinetic + log_partition

    w = softmax_weights(y, A, b_s, sqrt_c)
    w_real = np.maximum(np.real(w), 0.0)
    w_real = w_real / (np.sum(w_real) + 1e-300)
    entropy = float(-np.sum(w_real * np.log(w_real + 1e-300)))
    return {
        "kinetic": kinetic,
        "log_partition": log_partition,
        "total": total,
        "entropy": entropy,
        "max_entropy": float(np.log(d)),
        "y_norm": float(np.linalg.norm(y)),
        "conv": bool(conv),
        "res": float(res),
    }


def main() -> None:
    for p in [1, 2, 3]:
        print(f"\np={p}, r=1.0")
        print(f"{'gamma/pi':>8s} {'kinetic':>11s} {'logZ':>11s} {'RePsi':>11s} {'entropy':>9s} {'||y*||':>9s}")
        for gv in np.linspace(0.1, 2 * np.pi, 10):
            d = action_decomposition(p, gv, r=1.0)
            print(
                f"{gv/np.pi:8.3f} {d['kinetic']:+11.6f} {d['log_partition']:+11.6f} "
                f"{d['total']:+11.6f} {d['entropy']:9.5f} {d['y_norm']:9.5f}"
            )


if __name__ == "__main__":
    main()
