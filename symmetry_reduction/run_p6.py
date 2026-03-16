#!/usr/bin/env python3
"""
Run p=6 for selected gamma values only (d=8192). Full SVD ~30–60s per (gamma, saddle).
Use: python -m symmetry_reduction.run_p6
"""

import numpy as np
from pathlib import Path

from .core import build_structure_matrix, compute_b_s, DEFAULT_BETA
from .run_sweep import run_single_p_gamma, DATA_DIR

GAMMA_SELECT = [0.3, 1.0, np.pi, 2 * np.pi]


def main():
    p = 6
    d = 2 ** (2 * p + 1)
    print(f"Running p={p} (d={d}) for γ ∈ {[f'{g:.2f}' for g in GAMMA_SELECT]}...")
    print("  Building A and b_s once (shared across γ)...", flush=True)
    A = build_structure_matrix(p)
    betas = np.full(p, DEFAULT_BETA)
    b_s = compute_b_s(p, betas)
    results_y0 = []
    results_saddle = []
    for gamma in GAMMA_SELECT:
        print(f"  γ={gamma:.2f}...", flush=True)
        out_y0, out_saddle = run_single_p_gamma(p, gamma, run_saddle=True, A=A, b_s=b_s)
        results_y0.append(out_y0)
        results_saddle.append(out_saddle)
        print(
            f"    k99(y=0)={out_y0['k_99']}, k99(saddle)={out_saddle['k_99']}, "
            f"rs(y=0)={out_y0['stable_rank']:.2f}, rs(saddle)={out_saddle['stable_rank']:.2f}"
        )

    # Save as single npz for p=6 (one row per gamma)
    gamma_arr = np.array(GAMMA_SELECT)
    np.savez_compressed(
        DATA_DIR / "spectral_data_p6_y0.npz",
        gamma_values=gamma_arr,
        singular_values=np.array([r["singular_values"] for r in results_y0]),
        k99=np.array([r["k_99"] for r in results_y0]),
        k90=np.array([r["k_90"] for r in results_y0]),
        k95=np.array([r["k_95"] for r in results_y0]),
        k999=np.array([r["k_999"] for r in results_y0]),
        stable_rank=np.array([r["stable_rank"] for r in results_y0]),
        frobenius_norm=np.array([r["frobenius_norm"] for r in results_y0]),
        spectral_norm=np.array([r["spectral_norm"] for r in results_y0]),
        k99_cov=np.array([r["k99_cov"] for r in results_y0]),
    )
    np.savez_compressed(
        DATA_DIR / "spectral_data_p6_saddle.npz",
        gamma_values=gamma_arr,
        singular_values=np.array([r["singular_values"] for r in results_saddle]),
        k99=np.array([r["k_99"] for r in results_saddle]),
        k90=np.array([r["k_90"] for r in results_saddle]),
        k95=np.array([r["k_95"] for r in results_saddle]),
        k999=np.array([r["k_999"] for r in results_saddle]),
        stable_rank=np.array([r["stable_rank"] for r in results_saddle]),
        frobenius_norm=np.array([r["frobenius_norm"] for r in results_saddle]),
        spectral_norm=np.array([r["spectral_norm"] for r in results_saddle]),
        k99_cov=np.array([r["k99_cov"] for r in results_saddle]),
        saddle_converged=np.array([r["converged"] for r in results_saddle]),
        saddle_residual=np.array([r["saddle_residual"] for r in results_saddle]),
        kappa=np.array([r["kappa"] for r in results_saddle]),
        ipr=np.array([r["ipr"] for r in results_saddle]),
        entropy=np.array([r["entropy"] for r in results_saddle]),
    )
    print("Saved", DATA_DIR / "spectral_data_p6_y0.npz", "and spectral_data_p6_saddle.npz")
    print("Run scaling_table(p_values=(2,3,4,5,6)) to include p=6 in scaling fits.")


if __name__ == "__main__":
    main()
