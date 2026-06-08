"""
Joint (beta, gamma) sweep: BM24 seed saddle dominance phase diagram.

For a 2D grid of (beta, gamma), certifies the seed saddle, computes
Re Phi and exact lambda_abs, and runs a quick competitor search to find
the max algebraically dominant competing saddle.

Writes results to phasecraft/results/bm24_saddle_audit_p1/sweep_beta_gamma/

Extended sweep: beta in [0, 2pi], gamma in [0, -2pi].
"""

from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from phasecraft.bm24_saddle_audit_p1.audit import (
    bm24_seed_z,
    bm24_prefactor_exponent_ksat_all_subsets,
    _adaptive_krawczyk_with_diagnostics,
    _mpmath_newton_polish,
    _scipy_polish_to_tol,
    _x_to_z,
)
from phasecraft.bm24_saddle_audit_p1.continue_seed_branch_certified import (
    conv2_full_exponent,
    conv2_pref,
    lambda_abs_n_max,
    certify_z,
    polish_to_residual,
    step_from_previous_z,
    discover_competitors_at_gamma,
)
from phasecraft.lib.saddles.krawczyk_p1_roots import SaddleSystem
from phasecraft.lib.saddles.picard_lefschetz import compute_phi

Q = 3
K_CLAUSE = 8
R = 176.54

# Beta values: full QAOA-relevant range [0, 2pi], sampling at physically meaningful intervals.
# BM24 iterator may fail for large beta (beta > ~1.5); cold-start failures degrade gracefully.
import math as _math
BETA_GRID = [
    0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.5434,
    0.60, 0.70, 0.80, 0.90, 1.00,
    1.10, 1.20, 1.30, _math.pi / 2,  # ~1.5708
    1.60, 1.70, 1.80, 1.90, 2.00,
    2.20, 2.50, _math.pi,             # ~3.1416
    3.50, 4.00, 3 * _math.pi / 2,     # ~4.7124
    5.00, 5.50, 2 * _math.pi,         # ~6.2832
]

# Gamma grid: from near-zero to -2pi, with extra density near crossover (~-1.8)
GAMMA_GRID = [
    -0.05, -0.30, -0.60, -1.00, -1.30,
    -1.50, -1.60, -1.70, -1.80, -1.83, -1.90,
    -2.00, -2.50, -3.00, -3.50,
    -4.00, -4.50, -5.00, -5.50,
    -2 * _math.pi,  # ~-6.2832
]

N_VALUES = list(range(18, 25))       # n=18..24 for lambda_abs (slightly wider for robustness)
N_COMPETITOR_STARTS = 250
DPS = 80
BRANCH_TOL = 1e-4
MIN_RESIDUAL = 1e-10

OUT_DIR = REPO_ROOT / "phasecraft" / "results" / "bm24_saddle_audit_p1" / "sweep_beta_gamma_full"


def _init_seed(beta: float, gamma: float) -> tuple[np.ndarray | None, float, bool]:
    """Cold-start seed via BM24 iterator at (beta, gamma). Returns (z, re_phi, ok)."""
    try:
        z_it, phi_it, conv = bm24_seed_z(Q, R, beta, gamma)
        if not conv:
            return None, float("nan"), False
        sys_ = SaddleSystem.build(
            q=Q, r=R,
            betas=np.array([beta]),
            gammas=np.array([gamma]),
        )
        x0 = np.concatenate([z_it.real, z_it.imag])
        x_pol, res = polish_to_residual(sys_, x0, min_residual=MIN_RESIDUAL, dps=DPS)
        if res > 1e-6:
            return None, float("nan"), False
        z_pol = _x_to_z(x_pol, sys_.nvars)
        ok, _ = certify_z(sys_, z_pol, dps=DPS)
        phi = compute_phi(
            z_pol, q=Q, r=R,
            betas=sys_.betas, gammas=sys_.gammas,
        )
        return z_pol, float(phi.real), bool(ok)
    except Exception as exc:
        print(f"    init_seed failed beta={beta:.4f} gamma={gamma:.4f}: {exc}")
        return None, float("nan"), False


def _step_seed(
    z_prev: np.ndarray,
    beta: float,
    gamma: float,
) -> tuple[np.ndarray | None, float, bool]:
    """One continuation step via step_from_previous_z. Returns (z, re_phi, ok)."""
    try:
        z_new, row = step_from_previous_z(
            q=Q, K_clause=K_CLAUSE, r=R, beta=beta,
            gamma=gamma, z_prev=z_prev,
            step_index=0,
            n_values=N_VALUES,
            match_tol=0.05, min_residual=MIN_RESIDUAL, dps=DPS,
        )
        if row.failed or not row.certified:
            return None, float("nan"), False
        return z_new, row.re_phi_m, row.certified
    except Exception as exc:
        print(f"    step_seed failed beta={beta:.4f} gamma={gamma:.4f}: {exc}")
        return None, float("nan"), False


def sweep_point(
    beta: float,
    gamma: float,
    seed_z: np.ndarray,
) -> dict:
    """Compute seed action, lambda_abs, and quick competitor search at one (beta,gamma)."""
    sys_ = SaddleSystem.build(
        q=Q, r=R, betas=np.array([beta]), gammas=np.array([gamma])
    )
    phi = compute_phi(seed_z, q=Q, r=R, betas=sys_.betas, gammas=sys_.gammas)
    re_phi_seed = float(phi.real)
    im_phi_seed = float(phi.imag)
    full_seed = conv2_full_exponent(re_phi_seed, K_CLAUSE, R)
    lam = lambda_abs_n_max(Q, K_CLAUSE, R, beta, gamma, N_VALUES)
    seed_gap = full_seed - lam

    comps = discover_competitors_at_gamma(
        Q, K_CLAUSE, R, beta, gamma, seed_z,
        num_starts=N_COMPETITOR_STARTS,
        seed=int(1000 * abs(gamma) + 1000 * beta),
        branch_tol=BRANCH_TOL,
        min_residual=MIN_RESIDUAL,
        dps=DPS,
    )
    cert_comps = [c for c in comps if c.get("certified")]
    if cert_comps:
        max_comp = max(cert_comps, key=lambda c: float(c["full_conv2_exponent"]))
        max_delta = float(max_comp["full_conv2_exponent"]) - full_seed
        best_match = min(
            cert_comps,
            key=lambda c: abs(float(c["full_conv2_exponent"]) - lam),
        )
        best_match_gap = abs(float(best_match["full_conv2_exponent"]) - lam)
        seed_is_best = abs(seed_gap) <= best_match_gap
    else:
        max_delta = 0.0
        seed_is_best = True
        best_match_gap = float("nan")

    return {
        "beta": beta,
        "gamma": gamma,
        "re_phi_seed": re_phi_seed,
        "im_phi_seed": im_phi_seed,
        "full_conv2_seed": full_seed,
        "lambda_abs": lam,
        "seed_gap": seed_gap,
        "n_certified_competitors": len(cert_comps),
        "max_competitor_delta_re": max_delta,
        "seed_algebraically_dominates": max_delta <= 0.0,
        "seed_is_best_match_to_exact": seed_is_best,
        "best_match_abs_gap": best_match_gap,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    phi_pref = conv2_pref(K_CLAUSE, R)
    print(f"phi_pref_conv2 = {phi_pref:.6f}")
    print(f"Beta grid:  {BETA_GRID}")
    print(f"Gamma grid: {GAMMA_GRID}")
    print(f"Competitor starts: {N_COMPETITOR_STARTS}")
    print()

    all_rows: list[dict] = []
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")

    for beta in BETA_GRID:
        print(f"=== beta = {beta:.4f} ===")
        z_current: np.ndarray | None = None
        gamma_prev: float | None = None

        for gamma in GAMMA_GRID:
            print(f"  gamma={gamma:6.3f} ...", end=" ", flush=True)

            # Get seed z at this (beta, gamma)
            if z_current is None or gamma_prev is None:
                # Cold start
                z, re_phi, ok = _init_seed(beta, gamma)
            else:
                # Continuation step
                z, re_phi, ok = _step_seed(z_current, beta, gamma)
                if z is None:
                    # Fallback: cold start
                    print("(step failed, retrying cold)", end=" ", flush=True)
                    z, re_phi, ok = _init_seed(beta, gamma)

            if z is None:
                print("FAILED (no seed)")
                row = {
                    "beta": beta, "gamma": gamma,
                    "re_phi_seed": float("nan"), "im_phi_seed": float("nan"),
                    "full_conv2_seed": float("nan"), "lambda_abs": float("nan"),
                    "seed_gap": float("nan"), "n_certified_competitors": 0,
                    "max_competitor_delta_re": float("nan"),
                    "seed_algebraically_dominates": None,
                    "seed_is_best_match_to_exact": None,
                    "best_match_abs_gap": float("nan"),
                    "status": "seed_failed",
                }
                all_rows.append(row)
                z_current = None
                gamma_prev = None
                continue

            z_current = z
            gamma_prev = gamma

            row = sweep_point(beta, gamma, z)
            row["certified_seed"] = ok
            row["status"] = "ok"
            all_rows.append(row)

            seed_dom = "seed_dom" if row["seed_algebraically_dominates"] else f"comp_dom(Δ={row['max_competitor_delta_re']:+.3f})"
            best = "seed_best" if row["seed_is_best_match_to_exact"] else "comp_best"
            print(
                f"Re={row['re_phi_seed']:+.4f}  gap={row['seed_gap']:+.5f}  "
                f"{seed_dom}  {best}  n_comp={row['n_certified_competitors']}"
            )

        print()

    # Save raw results
    out_path = OUT_DIR / "sweep_beta_gamma_full.json"
    with open(out_path, "w") as f:
        json.dump(
            {
                "metadata": {
                    "q": Q, "K_clause": K_CLAUSE, "r": R,
                    "phi_pref_conv2": phi_pref,
                    "beta_grid": BETA_GRID,
                    "gamma_grid": GAMMA_GRID,
                    "n_values": N_VALUES,
                    "n_competitor_starts": N_COMPETITOR_STARTS,
                    "timestamp": ts,
                },
                "rows": all_rows,
            },
            f, indent=2,
        )
    print(f"\nWrote {len(all_rows)} rows to {out_path}")

    # Summary table
    print("\n=== SUMMARY: seed_gap and max_competitor_delta_re ===")
    print(f"{'beta':>6} {'gamma':>7} | {'seed_gap':>9} | {'max_Δ':>8} | {'seed_dom':>9} | {'seed_best':>9}")
    print("-" * 65)
    for row in all_rows:
        if row.get("status") != "ok":
            continue
        sd = "YES" if row["seed_algebraically_dominates"] else "NO "
        sb = "YES" if row["seed_is_best_match_to_exact"] else "NO "
        print(
            f"{row['beta']:6.4f} {row['gamma']:7.3f} | "
            f"{row['seed_gap']:+9.5f} | "
            f"{row['max_competitor_delta_re']:+8.4f} | "
            f"{sd:>9} | {sb:>9}"
        )

    # Save summary table
    summary = {
        "crossover_candidates": [
            r for r in all_rows
            if r.get("status") == "ok"
            and (not r["seed_algebraically_dominates"] or not r["seed_is_best_match_to_exact"])
        ]
    }
    with open(OUT_DIR / "sweep_summary_full.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote summary to {OUT_DIR / 'sweep_summary_full.json'}")


if __name__ == "__main__":
    main()
