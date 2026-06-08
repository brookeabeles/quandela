"""
Ring 4: Saturated 2000-start competitor exhaustion in the seed-dominant zone.

Design principles (correcting naive algebraic comparison):
- Many roots of G(z)=0 are "phantom saddles" on thimbles that don't contribute
  to the physical integral. Some phantoms have Re Phi >> Re Phi_seed or even > 0.
- The correct physical criterion is: does any competitor have full_conv2 CLOSER to
  the exact lambda_abs(n) than the seed? Only PHYSICAL competitors matter.
- Seed initialization uses continuation from gamma=-0.01 to handle large |gamma|.

At each (beta, gamma) point in the seed-dominant zone (seed_is_best=True in sweep):
  1. Continue seed z* from gamma=-0.01 using step_from_previous_z
  2. Run discover_roots with N_STARTS=2000
  3. For each root: compute full_conv2 and gap to exact lambda_abs
  4. Report: any competitor CLOSER to lambda_abs than seed? -> new physical competitor?

Key claim: with 2000 starts, no NEW physical competitor is found that beats the seed
in the seed-dominant zone. The 250-start sweep is already saturated.

Results: phasecraft/results/bm24_saddle_audit_p1/paper_results/ring4_*.{json,png}
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from phasecraft.bm24_saddle_audit_p1.audit import (
    bm24_seed_z,
    _x_to_z,
    finite_n_exponent_grid,
)
from phasecraft.bm24_saddle_audit_p1.continue_seed_branch_certified import (
    conv2_pref,
    conv2_full_exponent,
    certify_z,
    polish_to_residual,
    step_from_previous_z,
)
from phasecraft.lib.saddles.krawczyk_p1_roots import SaddleSystem, discover_roots
from phasecraft.lib.saddles.picard_lefschetz import compute_phi

Q = 3
K_CLAUSE = 8
R = 176.54
PHI_PREF = conv2_pref(K_CLAUSE, R)
DPS = 80
MIN_RESIDUAL = 1e-10
N_STARTS_RING4 = 2000
SEED_RNG = 7777
N_VALS = [24, 30, 40]      # n values for lambda_abs comparison

# Points from the seed-dominant zone.
# Strategy: use (beta, gamma) values where the 250-start sweep already shows
# seed_is_best_match_to_exact=True. Ring 4 tests whether 2000 starts changes this.
RING4_POINTS = [
    # (beta, gamma)  — all have seed_is_best=True in the 250-start sweep
    (0.5434, -0.300),   # far from crossing, seed dominant (sweep: max_comp_delta=-0.12)
    (0.5434, -1.000),   # mid zone (sweep: max_comp_delta=-0.03)
    (0.5434, -1.500),   # near crossing (sweep: seed_alg_dom=True)
    (0.5434, -1.700),   # very near crossing
    (0.5434, -1.800),   # just before crossing at -1.83
    (0.5434, -1.830),   # exactly at crossing
    # Other betas
    (0.30,   -1.000),
    (0.40,   -1.000),
    (0.50,   -1.000),
    (0.60,   -1.000),
]

# Reference sweep data (from 250-start sweep): gamma_cross ≈ -1.83 for beta=0.5434
# seed_is_best=True for gamma > -1.83 at beta=0.5434
SEED_IS_BEST_SWEEP = {  # from the 250-start sweep (approximate)
    (0.5434, -0.300): True,
    (0.5434, -1.000): True,
    (0.5434, -1.500): True,
    (0.5434, -1.700): True,
    (0.5434, -1.800): True,
    (0.5434, -1.830): True,   # boundary point
    (0.30,   -1.000): True,
    (0.40,   -1.000): True,
    (0.50,   -1.000): True,
    (0.60,   -1.000): True,
}

OUT_DIR = REPO_ROOT / "phasecraft" / "results" / "bm24_saddle_audit_p1" / "paper_results"


def continue_seed_to_gamma(beta: float, gamma_target: float) -> tuple[np.ndarray | None, dict]:
    """
    Continue seed z from gamma=-0.01 to gamma_target.
    Returns (z_seed, info_dict) or (None, {'ok': False}).
    """
    gamma_start = -0.01
    z_it, _, conv = bm24_seed_z(Q, R, beta, gamma_start)
    if not conv:
        return None, {"ok": False, "reason": "bm24_iterator_failed_at_start"}

    x0 = np.concatenate([z_it.real, z_it.imag])
    sys0 = SaddleSystem.build(q=Q, r=R, betas=np.array([beta]), gammas=np.array([gamma_start]))
    x_pol, _ = polish_to_residual(sys0, x0, min_residual=MIN_RESIDUAL, dps=DPS)
    z_prev = _x_to_z(x_pol, sys0.nvars)

    # Intermediate continuation steps (step size ≤ 0.2)
    n_steps = max(3, int(abs(gamma_target - gamma_start) / 0.15)) + 1
    gammas_chain = np.linspace(gamma_start, gamma_target, n_steps + 1)[1:]

    for g_step in gammas_chain:
        z_new, row = step_from_previous_z(
            Q, K_CLAUSE, R, beta, float(g_step), z_prev,
            step_index=0, n_values=[24], match_tol=0.05,
            min_residual=MIN_RESIDUAL, dps=DPS,
        )
        if row.failed:
            return None, {"ok": False, "reason": f"step_failed_at_gamma={g_step:.3f}"}
        z_prev = z_new

    # Final step: compute full statistics at gamma_target
    z_final, row_final = step_from_previous_z(
        Q, K_CLAUSE, R, beta, gamma_target, z_prev,
        step_index=0, n_values=N_VALS, match_tol=0.05,
        min_residual=MIN_RESIDUAL, dps=DPS,
    )
    if row_final.failed:
        return None, {"ok": False, "reason": "final_step_failed"}

    return z_final, {
        "ok": True,
        "full_conv2": row_final.full_conv2_exponent,
        "lambda_abs": row_final.lambda_abs_n_max,
        "box_radius": row_final.box_radius,
        "gap": row_final.gap_to_exact,
        "certified": row_final.certified,
    }


def run_point(beta: float, gamma: float) -> dict:
    """Run 2000-start search at (beta, gamma). Returns analysis dict."""
    t0 = time.time()
    betas_np = np.array([beta])
    gammas_np = np.array([gamma])

    # Get seed via continuation
    z_seed, seed_info = continue_seed_to_gamma(beta, gamma)
    if z_seed is None:
        print(f"  Seed continuation failed: {seed_info['reason']}")
        return {"beta": beta, "gamma": gamma, "status": "seed_failed",
                "reason": seed_info["reason"]}

    full_conv2_seed = seed_info["full_conv2"]
    lambda_abs_ref = seed_info["lambda_abs"]
    gap_seed = abs(seed_info["gap"])
    print(f"  Seed: full_conv2={full_conv2_seed:.6f}  λ_abs={lambda_abs_ref:.6f}  "
          f"|gap|={gap_seed:.4f}  cert={seed_info['certified']}")

    # Discover all roots with 2000 starts
    sys = SaddleSystem.build(q=Q, r=R, betas=betas_np, gammas=gammas_np)
    roots = discover_roots(sys, N_STARTS_RING4, SEED_RNG, 1e-10)
    t_discover = time.time() - t0
    print(f"  discover_roots: {len(roots)} roots in {t_discover:.1f}s")

    # Evaluate each root
    competitors = []
    n_skip_seed = 0
    for x_root in roots:
        z = _x_to_z(x_root, sys.nvars)
        # Exclude the seed
        if np.linalg.norm(z - z_seed, ord=np.inf) < 1e-4:
            n_skip_seed += 1
            continue
        phi = compute_phi(z, q=Q, r=R, betas=betas_np, gammas=gammas_np)
        full_comp = conv2_full_exponent(phi.real, K_CLAUSE, R)
        ok, proof = certify_z(sys, z, dps=DPS)
        radii = proof.get("box_radius", [])
        box_r_c = float(min(radii)) if radii else float("nan")
        gap_comp = abs(full_comp - lambda_abs_ref)
        competitors.append({
            "re_phi_m": float(phi.real),
            "full_conv2": float(full_comp),
            "delta_re_vs_seed": float(full_comp - full_conv2_seed),
            "gap_to_lambda": float(gap_comp),
            "beats_seed": float(gap_comp) < gap_seed,   # physically beats seed
            "certified": bool(ok),
            "box_radius": float(box_r_c),
            "in_physical_range": abs(float(full_comp) - lambda_abs_ref) < 1.0,
        })

    # Sort by gap to lambda_abs (closest first)
    competitors.sort(key=lambda c: c["gap_to_lambda"])

    # Key statistics
    n_certified = sum(1 for c in competitors if c["certified"])
    n_physical = sum(1 for c in competitors if c["in_physical_range"] and c["certified"])
    n_beats_seed = sum(1 for c in competitors if c["beats_seed"] and c["certified"])
    seed_still_best = n_beats_seed == 0

    # Best competitor (by physical criterion)
    best_comp = next((c for c in competitors if c["certified"] and c["in_physical_range"]),
                     None)
    best_gap = best_comp["gap_to_lambda"] if best_comp else float("inf")

    t_total = time.time() - t0
    print(f"  {len(competitors)} competitors ({n_certified} certified, {n_physical} physical-range)")
    print(f"  Seed |gap|={gap_seed:.5f}  Best competitor |gap|={best_gap:.5f}")
    print(f"  Seed still best: {seed_still_best}  (beats_seed: {n_beats_seed})  [{t_total:.1f}s]")

    return {
        "beta": float(beta),
        "gamma": float(gamma),
        "status": "ok",
        "n_starts": N_STARTS_RING4,
        "n_roots_found": len(roots),
        "n_seeds_identified": n_skip_seed,
        "n_competitors": len(competitors),
        "n_certified_competitors": n_certified,
        "n_physical_range_competitors": n_physical,
        "n_beats_seed_certified": n_beats_seed,
        "seed_still_best_2000_starts": bool(seed_still_best),
        "seed_was_best_250_starts": bool(SEED_IS_BEST_SWEEP.get((beta, gamma), True)),
        "full_conv2_seed": float(full_conv2_seed),
        "lambda_abs_ref": float(lambda_abs_ref),
        "gap_seed_abs": float(gap_seed),
        "best_competitor_gap_abs": float(best_gap),
        "t_seconds": float(t_total),
        "top_competitors": competitors[:10],
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t_global = time.time()

    print(f"Ring 4: {N_STARTS_RING4}-start competitor saturation (physical criterion)")
    print(f"Points: {len(RING4_POINTS)}  n_vals for λ_abs: {N_VALS}\n")

    results = []
    for beta, gamma in RING4_POINTS:
        print(f"Point (β={beta:.4f}, γ={gamma:.3f}):")
        res = run_point(beta, gamma)
        results.append(res)
        print()

    # Save JSON
    out_json = OUT_DIR / "ring4_competitor_saturation.json"
    with open(out_json, "w") as f:
        json.dump({"results": results, "n_starts": N_STARTS_RING4}, f, indent=2)
    print(f"Saved {out_json}")

    ok_results = [r for r in results if r.get("status") == "ok"]

    # -----------------------------------------------------------------------
    # Figure 1: Gap ratio (seed gap / best competitor gap) vs gamma
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(2, 1, figsize=(10, 9), sharex=False)
    ax1, ax2 = axes

    # At beta_opt: vary gamma
    beta_opt_r = [r for r in ok_results if abs(r["beta"] - 0.5434) < 0.01]
    beta_opt_r.sort(key=lambda r: r["gamma"])

    if beta_opt_r:
        gammas = [r["gamma"] for r in beta_opt_r]
        gap_seed = np.array([r["gap_seed_abs"] for r in beta_opt_r])
        gap_best = np.array([r["best_competitor_gap_abs"] for r in beta_opt_r])
        n_beats = [r["n_beats_seed_certified"] for r in beta_opt_r]

        # Gap ratio: < 1 means seed is closer to lambda_abs
        ratio = np.where(gap_best > 0, gap_seed / gap_best, np.nan)

        ax1.axhline(1.0, color="k", lw=1.0, ls="--", label="Tie (ratio=1)")
        ax1.axhline(0.1, color="gray", lw=0.8, ls=":", alpha=0.5, label="Seed 10× better")
        ax1.plot(gammas, ratio, "o-", color="C0", lw=2, ms=6,
                 label=f"seed|gap| / best_comp|gap|  ({N_STARTS_RING4} starts)")
        for i, (g, n, r_val) in enumerate(zip(gammas, n_beats, ratio)):
            if n > 0:
                ax1.annotate(f"BEAT: {n}", (g, r_val), fontsize=8, color="red",
                             ha="center", va="bottom")
        ax1.set_ylabel("Seed gap / Best competitor gap\n(< 1: seed wins, > 1: beaten)")
        ax1.set_title(
            rf"Ring 4 ({N_STARTS_RING4} starts): physical competitor gap ratio at $\beta_{{opt}}=0.5434$"
        )
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlabel(r"$\gamma$")
        ax1.set_ylim(0, None)

    # All points: n_beats
    if ok_results:
        labels = [f"β={r['beta']:.2f}\nγ={r['gamma']:.2f}" for r in ok_results]
        n_beats_all = [r["n_beats_seed_certified"] for r in ok_results]
        still_best = [r["seed_still_best_2000_starts"] for r in ok_results]
        colors = ["C2" if sb else "C3" for sb in still_best]

        x_pos = np.arange(len(ok_results))
        ax2.bar(x_pos, n_beats_all, color=colors, alpha=0.8)
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(labels, fontsize=7, rotation=45, ha="right")
        ax2.set_ylabel("# certified competitors beating seed")
        ax2.set_title("Physical competitors beating seed (green=0, red>0)")
        ax2.axhline(0.5, color="k", lw=0.8, ls="--")
        ax2.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "ring4_fig1_physical_gaps.png", dpi=150)
    plt.close(fig)
    print("Saved ring4_fig1_physical_gaps.png")

    # -----------------------------------------------------------------------
    # Figure 2: Number of roots found vs gamma (coverage landscape)
    # -----------------------------------------------------------------------
    if beta_opt_r:
        fig, ax = plt.subplots(figsize=(9, 5))
        gammas = [r["gamma"] for r in beta_opt_r]
        n_roots = [r["n_roots_found"] for r in beta_opt_r]
        n_phys = [r["n_physical_range_competitors"] for r in beta_opt_r]
        n_cert = [r["n_certified_competitors"] for r in beta_opt_r]

        ax.bar(gammas, n_roots, width=0.08, color="C7", alpha=0.6, label="All roots found")
        ax.bar(gammas, n_cert, width=0.06, color="C0", alpha=0.7, label="Certified")
        ax.bar(gammas, n_phys, width=0.04, color="C2", alpha=0.9, label="Physical-range certified")
        ax.set_xlabel(r"$\gamma$")
        ax.set_ylabel("# roots")
        ax.set_title(
            rf"Ring 4: saddle landscape coverage at $\beta={{0.5434}}$, {N_STARTS_RING4} starts"
        )
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")
        fig.tight_layout()
        fig.savefig(OUT_DIR / "ring4_fig2_coverage.png", dpi=150)
        plt.close(fig)
        print("Saved ring4_fig2_coverage.png")

    # -----------------------------------------------------------------------
    # Summary table
    # -----------------------------------------------------------------------
    print("\n=== Ring 4 Summary ===")
    print(f"{'beta':>7} {'gamma':>7} | {'roots':>5} {'phys':>5} | {'gap_seed':>9} {'gap_best':>9} | seed_best")
    print("-" * 65)
    for r in results:
        if r.get("status") != "ok":
            print(f"{r['beta']:7.4f} {r['gamma']:7.3f} | {'FAILED':>35}")
            continue
        sb = "YES" if r["seed_still_best_2000_starts"] else "NO ← NEW COMPETITOR"
        print(f"{r['beta']:7.4f} {r['gamma']:7.3f} | {r['n_roots_found']:5d} "
              f"{r['n_physical_range_competitors']:5d} | "
              f"{r['gap_seed_abs']:9.4f} {r['best_competitor_gap_abs']:9.4f} | {sb}")

    n_changed = sum(
        1 for r in ok_results
        if not r["seed_still_best_2000_starts"] and r.get("seed_was_best_250_starts", True)
    )
    print(f"\nPoints where 2000-start search found new physical competitor: {n_changed}/{len(ok_results)}")
    print(f"Total time: {time.time()-t_global:.1f}s")


if __name__ == "__main__":
    main()
