"""
Find the optimal (beta, gamma) maximizing the asymptotic exponent
    lambda_inf(beta, gamma) = phi_pref + Re Phi_seed(beta, gamma)

This is the large-n limit of lambda_abs = (1/n) log P_n, i.e. definition 2
("asymptotic exact optimum").

Strategy:
1. Fine 2D grid search around the coarse-sweep best (beta~0.7, gamma~-1.0).
2. scipy Nelder-Mead polish from the best grid point.
3. Report optimal (beta, gamma), lambda_inf, and comparison to BM24's beta=0.5434.

Runtime: ~60 s for the fine grid (no competitor search, no Krawczyk).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from phasecraft.bm24_saddle_audit_p1.audit import (
    bm24_seed_z,
    _scipy_polish_to_tol,
    _mpmath_newton_polish,
    _x_to_z,
)
from phasecraft.bm24_saddle_audit_p1.continue_seed_branch_certified import (
    conv2_full_exponent,
    conv2_pref,
    polish_to_residual,
)
from phasecraft.lib.saddles.krawczyk_p1_roots import SaddleSystem
from phasecraft.lib.saddles.picard_lefschetz import compute_phi

Q = 3
K_CLAUSE = 8
R = 176.54
PHI_PREF = conv2_pref(K_CLAUSE, R)
MIN_RESIDUAL = 1e-10
DPS = 50


def _get_seed_re_phi(beta: float, gamma: float) -> float | None:
    """Return phi_pref + Re Phi_seed, or None if the saddle can't be found."""
    try:
        z_it, phi_it, conv = bm24_seed_z(Q, R, beta, gamma)
        # proceed even if conv=False; the iterator may still have a good estimate
        if z_it is None:
            return None
        sys_ = SaddleSystem.build(
            q=Q, r=R,
            betas=np.array([beta]),
            gammas=np.array([gamma]),
        )
        x0 = np.concatenate([z_it.real, z_it.imag])
        x_pol, res = polish_to_residual(sys_, x0, min_residual=MIN_RESIDUAL, dps=DPS)
        if res > 1e-4:
            return None
        z_pol = _x_to_z(x_pol, sys_.nvars)
        phi = compute_phi(z_pol, q=Q, r=R, betas=sys_.betas, gammas=sys_.gammas)
        return conv2_full_exponent(float(phi.real), K_CLAUSE, R)
    except Exception:
        return None


def objective(params: np.ndarray) -> float:
    """Negative lambda_inf for minimization; large penalty on failure."""
    beta, gamma = float(params[0]), float(params[1])
    val = _get_seed_re_phi(beta, gamma)
    if val is None:
        return 1000.0
    return -val


def main() -> None:
    print(f"phi_pref = {PHI_PREF:.8f}")
    print()

    # ------------------------------------------------------------------ #
    # Step 1: fine grid in the interesting region beta in [0.4, 0.9],     #
    #         gamma in [-0.5, -2.0], 30x30                                #
    # ------------------------------------------------------------------ #
    beta_vals = np.linspace(0.40, 0.90, 30)
    gamma_vals = np.linspace(-0.30, -2.00, 30)

    best_lam = -np.inf
    best_bg = (0.7, -1.0)
    n_ok = 0

    print("Fine grid sweep (30x30) ...")
    results = []
    for beta in beta_vals:
        for gamma in gamma_vals:
            lam = _get_seed_re_phi(beta, gamma)
            if lam is None:
                continue
            n_ok += 1
            results.append((beta, gamma, lam))
            if lam > best_lam:
                best_lam = lam
                best_bg = (beta, gamma)

    print(f"  {n_ok}/{30*30} points succeeded")
    print(f"  Grid best: beta={best_bg[0]:.4f}  gamma={best_bg[1]:.4f}  lambda_inf={best_lam:.8f}")

    # Print top-10
    results.sort(key=lambda x: -x[2])
    print("\nTop-10 grid points:")
    print(f"  {'beta':>7} {'gamma':>8} {'lambda_inf':>14}")
    for b, g, lam in results[:10]:
        print(f"  {b:7.4f} {g:8.4f} {lam:14.8f}")

    # ------------------------------------------------------------------ #
    # Step 2: Nelder-Mead polish from best grid point                     #
    # ------------------------------------------------------------------ #
    print(f"\nPolishing with Nelder-Mead from ({best_bg[0]:.4f}, {best_bg[1]:.4f}) ...")
    res = minimize(
        objective,
        x0=np.array(best_bg),
        method="Nelder-Mead",
        options={"xatol": 1e-5, "fatol": 1e-8, "maxiter": 1000, "disp": True},
    )

    opt_beta, opt_gamma = float(res.x[0]), float(res.x[1])
    opt_lam = -float(res.fun)
    print(f"\nNelder-Mead optimum:")
    print(f"  beta*  = {opt_beta:.8f}")
    print(f"  gamma* = {opt_gamma:.8f}")
    print(f"  lambda_inf* = {opt_lam:.10f}")

    # ------------------------------------------------------------------ #
    # Step 3: BM24's cited beta=0.5434 — optimize gamma only (beta fixed)#
    # ------------------------------------------------------------------ #
    print("\nBM24 comparison at beta=0.5434 (gamma optimized, beta fixed):")
    bm24_beta = 0.5434
    bm24_best_lam = -np.inf
    bm24_best_gamma = None
    for gamma_test in np.linspace(-0.3, -2.5, 100):
        v = _get_seed_re_phi(bm24_beta, gamma_test)
        if v is not None and v > bm24_best_lam:
            bm24_best_lam = v
            bm24_best_gamma = gamma_test

    # Polish gamma only (beta=0.5434 fixed)
    def obj_fixed_beta(gamma_arr: np.ndarray) -> float:
        v = _get_seed_re_phi(bm24_beta, float(gamma_arr[0]))
        return -v if v is not None else 1000.0

    res_bm24 = minimize(
        obj_fixed_beta,
        x0=np.array([bm24_best_gamma]),
        method="Nelder-Mead",
        options={"xatol": 1e-6, "fatol": 1e-9, "maxiter": 500},
    )
    bm24_opt_gamma = float(res_bm24.x[0])
    bm24_opt_lam = -float(res_bm24.fun)

    print(f"  Best gamma for beta=0.5434: {bm24_opt_gamma:.8f}")
    print(f"  lambda_inf at (0.5434, {bm24_opt_gamma:.6f}): {bm24_opt_lam:.10f}")
    print(f"\n  Global optimum:              lambda_inf* = {opt_lam:.10f}  (beta={opt_beta:.6f}, gamma={opt_gamma:.6f})")
    print(f"  BM24 fixed-beta optimum:     lambda_inf  = {bm24_opt_lam:.10f}  (beta=0.5434,   gamma={bm24_opt_gamma:.6f})")
    print(f"  Difference (global - BM24):  {opt_lam - bm24_opt_lam:+.10f}")
    print(f"\n  BM24's beta=0.5434 is {'NOT ' if opt_lam > bm24_opt_lam + 1e-7 else ''}the global optimum")

    # ------------------------------------------------------------------ #
    # Summary                                                             #
    # ------------------------------------------------------------------ #
    print("\n" + "="*60)
    print("RESULT: Asymptotic optimal (beta*, gamma*)")
    print("="*60)
    print(f"  beta*   = {opt_beta:.6f}")
    print(f"  gamma*  = {opt_gamma:.6f}")
    print(f"  lambda_inf* = {opt_lam:.10f}")
    print(f"  phi_pref    = {PHI_PREF:.10f}")
    print(f"  Re Phi_seed = {opt_lam - PHI_PREF:.10f}")
    print()
    print(f"  BM24 fixed-beta optimum: beta=0.5434, gamma={bm24_opt_gamma:.6f}: {bm24_opt_lam:.10f}")
    print(f"  Improvement: {opt_lam - bm24_opt_lam:+.10f}")


if __name__ == "__main__":
    main()
