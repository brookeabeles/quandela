"""
Redo Hessian/spectral analysis using Phasecraft formulas as ground truth.

For each requested depth p:
1) Load Phasecraft optimal angles (betas, gammas) for (k, r).
2) Build BM24 structure matrix A.
3) Use Phasecraft b_s and c_alpha.
4) Solve saddle and compute Hessian/spectral summaries.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from symmetry_reduction.core import build_structure_matrix, hessian_log_at_y
from symmetry_reduction.phasecraft_bridge import (
    get_phasecraft_angles,
    phasecraft_b_s,
    phasecraft_c_alpha,
)
from symmetry_reduction.saddle import saddle_with_adaptive_damping
from symmetry_reduction.spectral import determinant_ratio_and_tail_metrics, spectral_summary


OUT_DIR = Path(__file__).resolve().parent / "data"
OUT_DIR.mkdir(exist_ok=True)


def run_one(k: int, r: float, p: int, biased: bool = False) -> dict:
    betas, gammas = get_phasecraft_angles(k, r, p, biased=biased)
    A = build_structure_matrix(p)
    b_s = phasecraft_b_s(betas)
    c_alpha = phasecraft_c_alpha(gammas, r)
    y_star, converged, residual = saddle_with_adaptive_damping(A, b_s, c_alpha)
    H = hessian_log_at_y(y_star, A, b_s, c_alpha)
    spec = spectral_summary(H)
    det = determinant_ratio_and_tail_metrics(H)
    k99 = spec["k_99"]
    k99 = min(k99, len(det["k0_axis"]) - 1) if len(det["k0_axis"]) else 0
    return {
        "k": k,
        "r": r,
        "p": p,
        "converged": bool(converged),
        "residual": float(residual),
        "stable_rank": float(spec["stable_rank"]),
        "k_95": int(spec["k_95"]),
        "k_99": int(spec["k_99"]),
        "k_999": int(spec["k_999"]),
        "fro_norm": float(spec["frobenius_norm"]),
        "op_norm": float(spec["spectral_norm"]),
        "err_at_k99": float(det["err"][k99]) if len(det["err"]) else np.nan,
        "nuclear_tail_at_k99": float(det["nuclear_tail"][k99]) if len(det["nuclear_tail"]) else np.nan,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--r", type=float, default=176.54)
    parser.add_argument("--p", type=int, nargs="+", default=[2, 3, 4, 5, 6, 7, 8])
    parser.add_argument("--biased", action="store_true")
    args = parser.parse_args()

    rows = []
    for p in args.p:
        rows.append(run_one(args.k, args.r, p, biased=args.biased))

    out_path = OUT_DIR / f"phasecraft_hessian_spectral_k{args.k}_r{args.r}.npz"
    np.savez_compressed(out_path, rows=np.array(rows, dtype=object))

    for row in rows:
        print(
            f"p={row['p']:>2} conv={row['converged']} "
            f"res={row['residual']:.3e} rs={row['stable_rank']:.3f} "
            f"k99={row['k_99']:>4} err@k99={row['err_at_k99']:.3e}"
        )
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
