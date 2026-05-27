import numpy as np

from symmetry_reduction.core import (
    build_structure_matrix,
    compute_b_s,
    compute_c_alpha,
    hessian_log_at_y,
    softmax_weights,
)
from symmetry_reduction.saddle import saddle_with_adaptive_damping


def newton_continuation(
    p: int, gamma_target: float, r: float = 1.0, delta_gamma: float = 0.05
) -> tuple[np.ndarray, float, int]:
    """Track q=1 saddle from small gamma to gamma_target with Newton updates."""
    betas = np.full(p, np.pi / 4)
    A = build_structure_matrix(p)
    b_s = compute_b_s(p, betas)
    d = A.shape[0]

    gamma_cur = max(delta_gamma, 1e-3)
    c_alpha = compute_c_alpha(p, np.full(p, gamma_cur), r=r)
    y, _, _ = saddle_with_adaptive_damping(A, b_s, c_alpha)

    steps = 0
    while gamma_cur < gamma_target - 1e-12:
        gamma_next = min(gamma_target, gamma_cur + delta_gamma)
        c_alpha = compute_c_alpha(p, np.full(p, gamma_next), r=r)
        sqrt_c = np.sqrt(c_alpha + 0j)

        for _ in range(50):
            w = softmax_weights(y, A, b_s, sqrt_c)
            e_a = w @ A.T
            grad = -y / 2 + sqrt_c * e_a
            res = float(np.max(np.abs(grad)))
            if res < 1e-12:
                break

            h_log = hessian_log_at_y(y, A, b_s, c_alpha)
            j = -0.5 * np.eye(d) + h_log
            delta = np.linalg.solve(j, -grad)

            step = 1.0
            for _ in range(30):
                y_trial = y + step * delta
                w_t = softmax_weights(y_trial, A, b_s, sqrt_c)
                grad_t = -y_trial / 2 + sqrt_c * (w_t @ A.T)
                if float(np.max(np.abs(grad_t))) < res:
                    break
                step *= 0.5
            y = y + step * delta

        gamma_cur = gamma_next
        steps += 1

    return y, gamma_cur, steps


def main() -> None:
    for p in [1, 2, 3]:
        for r_val in [1.0, 176.54]:
            y_newton, g_final, n_steps = newton_continuation(p, 2 * np.pi, r=r_val, delta_gamma=0.05)
            A = build_structure_matrix(p)
            b_s = compute_b_s(p, np.full(p, np.pi / 4))
            c_alpha = compute_c_alpha(p, np.full(p, 2 * np.pi), r=r_val)
            y_fp, conv_fp, res_fp = saddle_with_adaptive_damping(A, b_s, c_alpha)
            diff = float(np.max(np.abs(y_newton - y_fp)))
            print(
                f"p={p}, r={r_val:>7.2f}: continuation_steps={n_steps}, gamma_final={g_final/np.pi:.2f}pi, "
                f"max_diff_vs_fp={diff:.3e}, fp_conv={conv_fp}, fp_res={res_fp:.2e}"
            )


if __name__ == "__main__":
    main()
