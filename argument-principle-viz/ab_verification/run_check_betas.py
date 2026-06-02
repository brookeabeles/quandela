"""
Batch non-degeneracy verification over multiple beta values.

This is a lightweight wrapper around `run_full_check.py`'s verification
functions, intended to quickly scan beta in (-pi, pi).
"""

from __future__ import annotations

import sys
from typing import List

import mpmath as mp

# Allow importing sibling module when run as `python ab_verification/run_check_betas.py`
if __name__ == "__main__":
    # Ensure this directory is on sys.path
    from pathlib import Path

    HERE = Path(__file__).resolve().parent
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))

from run_full_check import verify_nondegeneracy


def linspace_betas(n: int, lo: mp.mpf, hi: mp.mpf) -> List[mp.mpf]:
    if n < 2:
        return [lo]
    step = (hi - lo) / (n - 1)
    return [lo + i * step for i in range(n)]


def main() -> None:
    mp.mp.dps = 70

    beta_values = linspace_betas(20, -0.95 * mp.pi, 0.95 * mp.pi)
    gamma_max = 2 * mp.pi
    delta_gamma = mp.pi / 25

    R_cutoff = mp.mpf(35)
    k_cap = 70
    tol_dphi = mp.mpf("1e-26")
    newton_max_iter = 50

    nondeg_epsilon = mp.mpf("1e-14")
    small_threshold = mp.mpf("1e-12")

    lip_refine = True
    lip_samples = 9
    lip_safety_factor = 2.0

    for beta in beta_values:
        report = verify_nondegeneracy(
            beta=beta,
            gamma_tilde_max=gamma_max,
            delta_gamma=delta_gamma,
            R_cutoff=R_cutoff,
            k_cap=k_cap,
            grid_tol_dphi=tol_dphi,
            nondeg_epsilon=nondeg_epsilon,
            small_threshold=small_threshold,
            newton_max_iter=newton_max_iter,
            lip_refine=lip_refine,
            lip_samples=lip_samples,
            lip_safety_factor=lip_safety_factor,
        )

        status = "PASS" if report.passed else "FAIL"
        print(
            "beta="
            + mp.nstr(beta, 10, strip_zeros=False)
            + "  "
            + status
            + "  "
            + "min|Phi''|="
            + mp.nstr(report.min_abs_d2phi, 8)
            + "  "
            + "cert_min="
            + mp.nstr(report.lip_refined_min_bound, 8)
            + "  "
            + "lip_ok="
            + str(report.lip_refine_passed)
        )


if __name__ == "__main__":
    main()

