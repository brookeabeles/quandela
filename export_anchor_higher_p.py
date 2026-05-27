"""
Export anchors for p=2 and p=3 certification runs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from symmetry_reduction.core import (
    alpha_linear_coefficients,
    bm24_clause_arity,
    build_structure_matrix,
    compute_b_s,
)
from symmetry_reduction.newton_saddle_q import (
    _compute_weights_from_u,
    _get_active_indices,
    newton_solve_q,
)


def export_anchor(
    p: int,
    betas: np.ndarray,
    gammas: np.ndarray,
    q: int = 3,
    r: float = 176.54,
    outfile: str | Path | None = None,
) -> dict:
    p = int(p)
    betas = np.asarray(betas, dtype=float)
    gammas = np.asarray(gammas, dtype=float)
    if betas.shape != (p,):
        raise ValueError("betas must have shape (p,)")
    if gammas.shape != (p,):
        raise ValueError("gammas must have shape (p,)")

    A = build_structure_matrix(p, q=q)
    b_s = compute_b_s(p, betas)
    d = int(A.shape[0])
    active_idx = _get_active_indices(d)
    n_act = int(len(active_idx))
    log_b = np.log(b_s + 1e-300 * (1 - np.sign(np.abs(b_s))))
    log_b = np.where(np.isfinite(log_b), log_b, -700.0)
    coeff = alpha_linear_coefficients(p, gammas, q=q, r=r)

    # t-homotopy on full gamma vector
    u = np.zeros(n_act, dtype=complex)
    t = 1e-6
    while t < 1.0 - 1e-14:
        t_next = min(1.0, t * 1.08)
        gammas_step = t_next * gammas
        coeff_step = alpha_linear_coefficients(p, gammas_step, q=q, r=r)
        c_2q_step = np.power(coeff_step[active_idx], bm24_clause_arity(q))
        u_new, conv, res, _ = newton_solve_q(
            u_init=u,
            c_2q_active=c_2q_step,
            active_idx=active_idx,
            A=A,
            log_b=log_b,
            d=d,
            q=q,
            tol=1e-12,
            max_iter=800,
        )
        if conv or (np.isfinite(res) and float(res) < 1e-5):
            u = u_new
            t = t_next
        else:
            t_next = float(np.sqrt(t * t_next))
            if t_next - t < 1e-12:
                break

    coeff_act = coeff[active_idx]
    y0 = np.zeros(n_act, dtype=complex)
    for i in range(n_act):
        if abs(coeff_act[i]) > 1e-14:
            y0[i] = u[i] / coeff_act[i]

    k = bm24_clause_arity(q)
    c_2q = np.power(coeff[active_idx], k)
    w = _compute_weights_from_u(u, active_idx, A, log_b, d)
    A_act = A[active_idx]
    E = w @ A_act.T
    g = u + float(k) * c_2q * np.power(E, k - 1)
    residual = float(np.max(np.abs(g)))

    anchor = {
        "p": int(p),
        "q": int(q),
        "r": float(r),
        "betas": [float(b) for b in betas],
        "gammas": [float(g) for g in gammas],
        "active_idx": [int(a) for a in active_idx],
        "n_active": int(n_act),
        "d": int(d),
        "y0_real": y0.real.tolist(),
        "y0_imag": y0.imag.tolist(),
        "coeff_real": coeff.real.tolist(),
        "coeff_imag": coeff.imag.tolist(),
        "log_b_real": np.real(log_b).tolist(),
        "log_b_imag": np.imag(log_b).tolist(),
        "anchor_residual_inf": residual,
    }

    if outfile is None:
        outfile = f"anchor_p{p}.json"
    outfile = Path(outfile)
    outfile.parent.mkdir(parents=True, exist_ok=True)
    with outfile.open("w", encoding="utf-8") as f:
        json.dump(anchor, f, indent=2)

    print(f"Exported anchor to {outfile}")
    print(f"  p={p}, n_active={n_act}, d={d}")
    print(f"  residual={residual:.3e}")
    print(f"  max |y0| = {np.max(np.abs(y0)):.6e}")
    print(f"  nonzero components (|y|>1): {np.sum(np.abs(y0) > 1)}/{n_act}")
    return anchor


def main() -> int:
    parser = argparse.ArgumentParser(description="Export higher-p anchors for certification.")
    parser.add_argument("--p", type=int, required=True, choices=[2, 3])
    parser.add_argument("--q", type=int, default=3)
    parser.add_argument("--r", type=float, default=176.54)
    parser.add_argument("--beta", type=float, default=-float(np.pi / 2))
    parser.add_argument("--gamma", type=float, default=0.14)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    betas = np.full(args.p, args.beta, dtype=float)
    gammas = np.full(args.p, args.gamma, dtype=float)
    out = args.out
    if out is None:
        out = Path(f"anchor_p{args.p}_uniform_gamma{args.gamma:.6f}.json")
    export_anchor(
        p=args.p,
        betas=betas,
        gammas=gammas,
        q=args.q,
        r=args.r,
        outfile=out,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
