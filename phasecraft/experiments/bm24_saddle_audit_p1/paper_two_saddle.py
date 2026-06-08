"""
B: Two-saddle uniform asymptotics at the anti-Stokes crossing.

At beta=0.5434 (BM24 optimal), near gamma=-1.83 where Re(Phi_seed)~Re(Phi_43),
the single-saddle approximation has residual error. This script:

1. Tracks the seed saddle via continuation from gamma=-0.01
2. Discovers the dominant competitor (branch 43) at each gamma via multistart Newton
3. Evaluates the two-saddle sum log|exp(n*E_seed)+exp(n*E_comp)| / n for n=18..40
4. Compares: seed alone, competitor alone, two-saddle sum, exact lambda_abs(n)
5. Shows the anti-Stokes crossing Re Phi_seed = Re Phi_comp and the transition
   zone width as a function of n

Results: phasecraft/results/bm24_saddle_audit_p1/paper_results/two_saddle_*.{json,png}
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from phasecraft.bm24_saddle_audit_p1.audit import (
    bm24_seed_z,
    finite_n_exponent_grid,
    _x_to_z,
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
BETA = 0.5434
PHI_PREF = conv2_pref(K_CLAUSE, R)   # -0.6896...
DPS = 80
MIN_RESIDUAL = 1e-10

# Dense grid near crossing; coarser away
GAMMA_GRID = [
    -1.50, -1.60, -1.70, -1.75,
    -1.80, -1.83, -1.85, -1.90,
    -2.00, -2.50,
]

N_EXACT = list(range(18, 41))         # n=18..40 for two-saddle vs exact comparison
N_COMP_STARTS = 100                   # multistart budget for competitor search

OUT_DIR = REPO_ROOT / "phasecraft" / "results" / "bm24_saddle_audit_p1" / "paper_results"


# ---------------------------------------------------------------------------
# Two-saddle approximation (equal prefactors, ratio=1)
# ---------------------------------------------------------------------------

def two_saddle_lam(re_seed, re_comp, phi_pref, n_values):
    """
    Two-saddle approximation assuming equal Gaussian prefactors (ratio=1):
        lambda_two(n) = phi_pref + logaddexp(n*re_seed, n*re_comp) / n

    This is the leading-order two-saddle formula. The Hessian-based prefactor
    ratio requires knowledge of the full BM24 generating function measure and
    cannot be computed from the auxiliary z-coordinates alone.
    """
    out = []
    for n in n_values:
        E_s = n * re_seed
        E_c = n * re_comp
        lam = phi_pref + np.logaddexp(E_s, E_c) / n
        out.append({"n": n, "lam_two": float(lam)})
    return out


# ---------------------------------------------------------------------------
# Seed continuation
# ---------------------------------------------------------------------------

def get_seed_z_chain(gammas_target, n_vals):
    """Continuation from gamma=-0.01 through all targets. Returns dict gamma -> z."""
    beta = BETA
    gamma_start = -0.01
    sys0 = SaddleSystem.build(q=Q, r=R, betas=np.array([beta]), gammas=np.array([gamma_start]))
    z_it, _, conv = bm24_seed_z(Q, R, beta, gamma_start)
    if not conv:
        return {}
    x0 = np.concatenate([z_it.real, z_it.imag])
    x_pol, res = polish_to_residual(sys0, x0, min_residual=MIN_RESIDUAL, dps=DPS)
    if res > 1e-6:
        return {}
    z = _x_to_z(x_pol, sys0.nvars)

    z_by_gamma = {}
    g_prev = float(gamma_start)
    targets = sorted(gammas_target, reverse=True)  # least-negative first

    for g_target in targets:
        step = -0.03
        g = g_prev
        while g > float(g_target) + 1e-9:
            g_next = max(g + step, float(g_target))
            z_next, row = step_from_previous_z(
                Q, K_CLAUSE, R, beta, g_next, z,
                step_index=0, n_values=n_vals,
                match_tol=0.1, min_residual=MIN_RESIDUAL, dps=DPS,
            )
            if z_next is None:
                step *= 0.5
                if abs(step) < 5e-4:
                    z = None
                    break
                continue
            z = z_next
            g = g_next
        if z is not None:
            z_by_gamma[g_target] = z.copy()
            g_prev = g_target
        else:
            print(f"    Seed continuation failed at gamma={g_target:.3f}")
            break

    return z_by_gamma


# ---------------------------------------------------------------------------
# Competitor discovery
# ---------------------------------------------------------------------------

def find_best_competitor_z(gamma, z_seed, lam_24):
    """Run multistart Newton and return (z_comp, re_phi_comp) for best certified competitor."""
    sys_ = SaddleSystem.build(q=Q, r=R, betas=np.array([BETA]), gammas=np.array([gamma]))
    roots_x = discover_roots(sys_, num_starts=N_COMP_STARTS,
                             seed=int(1234 * abs(gamma)), tol=1e-12)
    best_z = None
    best_gap = float("inf")
    best_re = float("nan")
    best_im = float("nan")

    for x in roots_x:
        z_cand = _x_to_z(x, sys_.nvars)
        if np.linalg.norm(z_cand - z_seed, ord=np.inf) < 1e-4:
            continue  # same branch as seed
        x_pol, res = polish_to_residual(sys_, x, min_residual=1e-8, dps=DPS)
        if res > 1e-4:
            continue
        z_pol = _x_to_z(x_pol, sys_.nvars)
        ok, _ = certify_z(sys_, z_pol, dps=DPS)
        if not ok:
            continue
        phi = compute_phi(z_pol, Q, R, sys_.betas, sys_.gammas)
        full = float(phi.real) + PHI_PREF
        gap = abs(full - lam_24)
        if gap < best_gap:
            best_gap = gap
            best_z = z_pol.copy()
            best_re = float(phi.real)
            best_im = float(phi.imag)

    return best_z, best_re, best_im


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    n_vals_step = list(range(18, 25))

    print(f"beta={BETA}, phi_pref={PHI_PREF:.6f}")
    print(f"Running at {len(GAMMA_GRID)} gamma values...\n")

    # Get seed z* at all target gammas
    print("Continuing seed branch...")
    z_by_gamma = get_seed_z_chain(GAMMA_GRID, n_vals_step)
    print(f"  Got seed z* at {len(z_by_gamma)}/{len(GAMMA_GRID)} gammas\n")

    results = []
    for gamma in GAMMA_GRID:
        print(f"  gamma={gamma:.3f}", end=" ", flush=True)

        # 1) Seed at this gamma
        z_seed = z_by_gamma.get(gamma)
        if z_seed is None:
            print("  -> NO SEED")
            continue

        sys_ = SaddleSystem.build(q=Q, r=R, betas=np.array([BETA]), gammas=np.array([gamma]))
        phi_s = compute_phi(z_seed, Q, R, sys_.betas, sys_.gammas)
        re_phi_seed = float(phi_s.real)
        im_phi_seed = float(phi_s.imag)
        full_seed = PHI_PREF + re_phi_seed
        print(f"Re Phi_seed={re_phi_seed:+.4f}", end="  ", flush=True)

        # 2) Exact lambda_abs for n=18..40
        fn = finite_n_exponent_grid(K_CLAUSE, Q, R, BETA, gamma, N_EXACT)
        lam_exact = {int(n): float(fn["lambda_abs"][str(n)]) for n in N_EXACT}
        lam_24 = lam_exact[24]

        # 3) Best competitor z*
        print("searching competitor...", end=" ", flush=True)
        z_comp, re_phi_comp, im_phi_comp = find_best_competitor_z(gamma, z_seed, lam_24)

        if z_comp is None:
            print("  -> NO COMP")
            results.append({
                "gamma": gamma, "re_phi_seed": re_phi_seed,
                "im_phi_seed": im_phi_seed, "full_seed": full_seed,
                "lambda_abs_24": lam_24, "lam_exact": lam_exact,
                "competitor_found": False,
            })
            continue

        full_comp = PHI_PREF + re_phi_comp
        print(f"Re Phi_comp={re_phi_comp:+.4f}", end="  ", flush=True)

        # 4) Two-saddle sum for n=18..40
        two_saddle = two_saddle_lam(re_phi_seed, re_phi_comp, PHI_PREF, N_EXACT)

        # Gap summary at n=24
        gap_seed = full_seed - lam_24
        gap_comp = full_comp - lam_24
        gap_two = two_saddle[N_EXACT.index(24)]["lam_two"] - lam_24
        print(f"gap_seed={gap_seed:+.4f}  gap_comp={gap_comp:+.4f}  gap_two={gap_two:+.4f}")

        results.append({
            "gamma": gamma,
            "re_phi_seed": re_phi_seed, "im_phi_seed": im_phi_seed,
            "re_phi_comp": re_phi_comp, "im_phi_comp": im_phi_comp,
            "full_seed": full_seed, "full_comp": full_comp,
            "lambda_abs_24": lam_24, "lam_exact": lam_exact,
            "two_saddle": two_saddle,
            "competitor_found": True,
        })

    # Save JSON
    out_json = OUT_DIR / "two_saddle_crossing.json"
    with open(out_json, "w") as f:
        json.dump({"beta": BETA, "phi_pref": PHI_PREF, "results": results}, f, indent=2)
    print(f"\nWrote {out_json}")

    # -----------------------------------------------------------------------
    # Plots
    # -----------------------------------------------------------------------
    rows_ok = [r for r in results if r.get("competitor_found")]
    if not rows_ok:
        print("No valid rows to plot.")
        return

    gammas = [r["gamma"] for r in rows_ok]
    lam_24_arr = np.array([r["lambda_abs_24"] for r in rows_ok])
    seed_L0 = np.array([r["full_seed"] for r in rows_ok])
    comp_L0 = np.array([r["full_comp"] for r in rows_ok])

    def get_n_pred(key, rows, n_target=24):
        out = []
        for r in rows:
            lv = {lv["n"]: lv for lv in r["two_saddle"]}
            out.append(lv[n_target][key])
        return np.array(out)

    two_24 = get_n_pred("lam_two", rows_ok, n_target=24)
    two_40 = get_n_pred("lam_two", rows_ok, n_target=40)

    # Figure 1: Re Phi trajectory + all approximation levels at n=24
    fig, axes = plt.subplots(2, 1, figsize=(9, 9), sharex=True)

    ax = axes[0]
    re_seed_arr = np.array([r["re_phi_seed"] for r in rows_ok])
    re_comp_arr = np.array([r["re_phi_comp"] for r in rows_ok])
    ax.plot(gammas, re_seed_arr, "o-", color="C0", lw=2, label=r"$\mathrm{Re}\,\Phi_{\rm seed}$")
    ax.plot(gammas, re_comp_arr, "s-", color="C1", lw=2, label=r"$\mathrm{Re}\,\Phi_{\rm comp}$ (branch 43)")
    ax.axhline(0, color="k", lw=0.7, ls="--", alpha=0.5)
    ax.axvline(-1.83, color="gray", lw=0.8, ls=":", label=r"$\gamma_{\rm cross}\approx-1.83$")
    # Mark crossing: where curves cross
    ax.set_ylabel(r"$\mathrm{Re}\,\Phi$")
    ax.set_title(
        r"Two-saddle crossing analysis: $\beta=0.5434$, $q=3$, $k=8$, $r=176.54$"
        "\n" r"Top: saddle actions. Bottom: approximation levels vs exact."
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(gammas, seed_L0,  "o-",  color="C0",  lw=2.0, ms=6, label=r"Seed: $\phi_{\rm pref}+\mathrm{Re}\,\Phi_{\rm seed}$")
    ax.plot(gammas, comp_L0,  "s-",  color="C1",  lw=2.0, ms=6, label=r"Branch 43: $\phi_{\rm pref}+\mathrm{Re}\,\Phi_{\rm comp}$")
    ax.plot(gammas, two_24,   "^-",  color="C2",  lw=2.5, ms=7, label=r"Two-saddle ($n=24$)")
    ax.plot(gammas, two_40,   "v--", color="C3",  lw=1.5, ms=5, label=r"Two-saddle ($n=40$)")
    ax.plot(gammas, lam_24_arr, "kD-", lw=1.5, ms=6, label=r"Exact $\lambda_{\rm abs}(n=24)$")
    ax.axvline(-1.83, color="gray", lw=0.8, ls=":")
    ax.set_xlabel(r"$\gamma$")
    ax.set_ylabel(r"$\log|P_n|/n$")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "two_saddle_fig1_overview.png", dpi=150)
    plt.close(fig)
    print("Saved two_saddle_fig1_overview.png")

    # Figure 2: Residual gaps vs gamma at n=24
    fig, ax = plt.subplots(figsize=(9, 5))

    lam_40_arr = np.array([r["lam_exact"][40] for r in rows_ok])

    ax.plot(gammas, seed_L0 - lam_24_arr, "o-",  color="C0", lw=2, label="Seed gap (n=24)")
    ax.plot(gammas, comp_L0 - lam_24_arr, "s-",  color="C1", lw=2, label="Branch-43 gap (n=24)")
    ax.plot(gammas, two_24  - lam_24_arr, "^-",  color="C2", lw=2.5, ms=7, label="Two-saddle gap (n=24)")
    ax.plot(gammas, two_40  - lam_40_arr, "v--", color="C3", lw=1.5, ms=5, label="Two-saddle gap (n=40)")

    ax.axhline(0, color="k", lw=0.8)
    ax.axvline(-1.83, color="gray", lw=0.8, ls=":", label=r"$\gamma_{\rm cross}\approx-1.83$")
    ax.fill_between(gammas, -0.02, 0.02, alpha=0.08, color="green", label=r"$|\mathrm{gap}|<0.02$")

    ax.set_xlabel(r"$\gamma$")
    ax.set_ylabel(r"approximation $- \lambda_{\rm abs}(n)$")
    ax.set_title(
        r"Residual gaps: two-saddle reduces error near anti-Stokes crossing ($\beta=0.5434$)"
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "two_saddle_fig2_gaps.png", dpi=150)
    plt.close(fig)
    print("Saved two_saddle_fig2_gaps.png")

    # Figure 3: Gap vs n at gamma=-1.83 (anti-Stokes crossing)
    cross_rows = [r for r in rows_ok if abs(r["gamma"] - (-1.83)) < 0.005]
    if cross_rows:
        r_cross = cross_rows[0]
        ns = np.array(N_EXACT, dtype=float)
        exact_n = np.array([r_cross["lam_exact"][str(n)] for n in N_EXACT])
        seed_n = np.full_like(ns, r_cross["full_seed"])
        comp_n = np.full_like(ns, r_cross["full_comp"])
        two_n = np.array([lv["lam_two"] for lv in r_cross["two_saddle"]])

        # Also add log(2)/n reference
        log2_n = PHI_PREF + r_cross["re_phi_seed"] + np.log(2) / ns

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        # Left: absolute predictions vs exact
        axes[0].plot(ns, exact_n, "kD-", lw=1.5, ms=5, label=r"Exact $\lambda_{\rm abs}(n)$")
        axes[0].plot(ns, seed_n, "--", color="C0", lw=1.5, label="Seed (leading order)")
        axes[0].plot(ns, comp_n, "--", color="C1", lw=1.5, label="Branch 43 (leading order)")
        axes[0].plot(ns, two_n,  "-",  color="C2", lw=2.5, label="Two-saddle sum")
        axes[0].plot(ns, log2_n, ":",  color="C4", lw=1.5, label=r"Seed + $\log 2 / n$")
        axes[0].set_xlabel(r"$n$")
        axes[0].set_ylabel(r"$\log|P_n|/n$")
        axes[0].set_title(r"Anti-Stokes crossing $\gamma=-1.83$: predictions vs exact")
        axes[0].legend(fontsize=9)
        axes[0].grid(True, alpha=0.3)

        # Right: gaps vs n
        axes[1].plot(ns, seed_n - exact_n, "--", color="C0", lw=1.5, label="Seed gap")
        axes[1].plot(ns, comp_n - exact_n, "--", color="C1", lw=1.5, label="Branch-43 gap")
        axes[1].plot(ns, two_n  - exact_n, "-",  color="C2", lw=2.5, label="Two-saddle gap")
        axes[1].plot(ns, log2_n - exact_n, ":",  color="C4", lw=1.5, label=r"Seed + $\log 2/n$ gap")
        axes[1].axhline(0, color="k", lw=0.8)
        axes[1].set_xlabel(r"$n$")
        axes[1].set_ylabel(r"approximation $- \lambda_{\rm abs}(n)$")
        axes[1].set_title(r"Gap vs $n$ at anti-Stokes crossing $\gamma=-1.83$")
        axes[1].legend(fontsize=9)
        axes[1].grid(True, alpha=0.3)

        fig.suptitle(
            r"Anti-Stokes crossing: $\beta=0.5434$, $\gamma=-1.83$"
            "\n" + r"Two-saddle sum = $\phi_{\rm pref} + \log(e^{n\,\mathrm{Re}\Phi_{\rm seed}} + e^{n\,\mathrm{Re}\Phi_{\rm comp}})/n$"
        )
        fig.tight_layout()
        fig.savefig(OUT_DIR / "two_saddle_fig3_crossing_n_convergence.png", dpi=150)
        plt.close(fig)
        print("Saved two_saddle_fig3_crossing_n_convergence.png")

    # Figure 4: ΔRe Phi vs gamma — shows crossing geometry
    fig, ax = plt.subplots(figsize=(8, 4))
    delta_re = np.array([r["re_phi_comp"] - r["re_phi_seed"] for r in rows_ok])
    ax.plot(gammas, delta_re, "o-", color="C2", lw=2, ms=6)
    ax.axhline(0, color="k", lw=1.5, label="Anti-Stokes line (ΔRe Φ = 0)")
    ax.axvline(-1.83, color="gray", lw=0.8, ls=":", label=r"$\gamma_{\rm cross}$")
    ax.fill_between(gammas, -1/np.array([24]*len(gammas)), 1/np.array([24]*len(gammas)),
                    alpha=0.15, color="gray", label=r"$|\Delta\mathrm{Re}\Phi|<1/n$ (n=24)")
    ax.set_xlabel(r"$\gamma$")
    ax.set_ylabel(r"$\Delta\mathrm{Re}\,\Phi = \mathrm{Re}\,\Phi_{\rm comp} - \mathrm{Re}\,\Phi_{\rm seed}$")
    ax.set_title(r"Anti-Stokes crossing: $\Delta\mathrm{Re}\,\Phi$ changes sign near $\gamma=-1.83$")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "two_saddle_fig4_delta_re_phi.png", dpi=150)
    plt.close(fig)
    print("Saved two_saddle_fig4_delta_re_phi.png")

    print("\nDone.")


if __name__ == "__main__":
    main()
