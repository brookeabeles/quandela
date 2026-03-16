"""
Run eigenvalue decay analyses and save figures/data into symmetry_reduction/eigenvalue_decay/ and figures.
"""

from __future__ import annotations

import numpy as np
from pathlib import Path

# Ensure package is importable when run as script
import sys
ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from symmetry_reduction.eigenvalue_decay.eigenvalue_decay import (
    eigenvalue_decay_analysis,
    decay_rate_vs_p,
    plot_decay_and_c_prime,
    plot_block_decomposition_combined,
    plot_block_max_decay,
)
from symmetry_reduction.eigenvalue_decay.fourier_agreement import run_verification as run_fourier_verification
from symmetry_reduction.eigenvalue_decay.bonami_beckner import plot_bonami_comparison_all

OUT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = OUT_DIR / "data"
FIG_DIR = OUT_DIR / "figures" / "eigenvalue_decay"
DATA_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


def run_all():
    """Run Phase 1 numerics and save plots."""
    print("=== Fourier agreement verification ===")
    run_fourier_verification([2, 3])

    p_max = (2, 3, 4, 5, 6)
    print("\n=== Eigenvalue decay + c'(p) (combined figures, p up to 6) ===")
    plot_decay_and_c_prime(
        p_values=p_max,
        gamma_values=(0.3, 1.0, np.pi, 2 * np.pi),
        at_saddle=False,
        out_dir=FIG_DIR,
    )
    plot_decay_and_c_prime(
        p_values=p_max,
        gamma_values=(0.3, 1.0, np.pi),
        at_saddle=True,
        out_dir=FIG_DIR,
    )
    print("Saved eigenvalue_decay_and_c_prime_y0.png, eigenvalue_decay_and_c_prime_saddle.png")

    print("\n=== Block decomposition (y=0 + saddle combined) ===")
    plot_block_decomposition_combined(p=3, gamma=1.0, out_dir=FIG_DIR)
    print("Saved eigenvalue_block_decomposition_p3_gamma1.00_combined.png")

    print("\n=== Block-wise spectral decay (max |λ^(m)| vs m, c_m(p) vs p) ===")
    plot_block_max_decay(
        p_values=(2, 3, 4, 5),
        gamma_values=(0.3, 1.0, np.pi),
        at_saddle=False,
        out_dir=FIG_DIR,
    )
    plot_block_max_decay(
        p_values=(2, 3, 4, 5),
        gamma_values=(0.3, 1.0, np.pi),
        at_saddle=True,
        out_dir=FIG_DIR,
    )
    print("Saved eigenvalue_block_max_decay_y0.png, eigenvalue_block_max_decay_saddle.png")

    print("\n=== Bonami–Beckner comparison (p=2,4,6 × 3 γ, one figure) ===")
    plot_bonami_comparison_all(
        p_values=(2, 4, 6),
        gamma_values=(0.3, 1.0, np.pi),
        at_saddle=False,
        out_path=FIG_DIR / "bonami_comparison_all.png",
    )
    print("Saved bonami_comparison_all.png")

    # Save decay rate vs p for a couple of gamma values
    print("\n=== Decay rate c'(p) (summary, p=2..6) ===")
    for gamma in [0.3, 1.0, np.pi]:
        out = decay_rate_vs_p(gamma, p_values=p_max, at_saddle=False)
        print(f"  γ={gamma:.2f}: c' = {out['c_prime']}, R² = {out['r2']}")

    print("\nDone. Figures in:", FIG_DIR)


if __name__ == "__main__":
    run_all()
