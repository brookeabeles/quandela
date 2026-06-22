#!/usr/bin/env python3
r"""
compare_z_wu_saddle.py
======================

Probe the ADVANTAGE of the w,u-saddle formulation over the z-saddle for q=3.

Background
----------
Both formulations target the BM24 p=1 QAOA saddle equations for random k-SAT
with k = 2^q, but approach the problem differently:

  z-saddle  (SaddleSystem, krawczyk_p1_roots.py)
    Variables  : z ∈ C^8  (all 2^{2p+1}=8 states for p=1)
    Reduction  : c_root = (-c)^{1/k}  — a FRACTIONAL k-th root, k=8 for q=3.
                 numpy uses the PRINCIPAL BRANCH; other branches give different
                 saddle points.  Numerical discovery works but Krawczyk
                 certification fails for q≥2 because the 8D Jacobian bounds
                 blow up on the large-|z| regime where the dominant saddle lives.
    Action     : Phi_z = F - (1-2^{-q}) Σ z_s dF_s   (per compute_phi)

  w,u-saddle (WSaddleSystem, krawczyk_w_saddle_q.py)
    Variables  : w ∈ C^4  (nontrivial-index chart, p=1)
    Reduction  : u eliminated via eq(II): u = i g(w)  — NO fractional root.
    Action     : Phi_wu = Σ_a [c_a u_a^{2q} + i w_a u_a] + log Δ(w)
    Note       : Phi_wu ≠ Phi_z in general (different parent actions, different
                 scales); the wu action covers the NONTRIVIAL-INDEX orbit subspace.

Key questions probed
--------------------
1. CERTIFICATION GAP  (critical):  does z-system Krawczyk certify roots for q=3?
   Wu-system should certify; z-system should fail.
2. q SWEEP: does the certification gap appear at q=2? q=4?
3. GAMMA REACH: how far in |gamma| can wu certify vs z certify?
4. ROOT LANDSCAPE: how many competitors does each system find across gamma?
5. DIMENSION / TIMING: is the 4D system meaningfully faster than the 8D?

Run
---
    python -m phasecraft.w_saddle.compare_z_wu_saddle
    python -m phasecraft.w_saddle.compare_z_wu_saddle --output my_output.png
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from phasecraft.w_saddle.krawczyk_w_saddle_q import (
    WSaddleSystem,
    discover_w_roots,
    krawczyk_certify_reduced_escalate,
    solve_w_from_init,
)
from phasecraft.lib.saddles.krawczyk_p1_roots import (
    SaddleSystem,
    discover_roots,
    krawczyk_certify_with_escalation,
)
from phasecraft.lib.saddles.picard_lefschetz import compute_phi

# ---------------------------------------------------------------------------
# Standard physical parameters (BM24 k=8 SAT, p=1)
# ---------------------------------------------------------------------------
Q3 = 3        # q=3 → k=8
K3 = 8
R = 176.54    # 8-SAT satisfiability threshold (BM24 Table I)
BETA_SLICE = -np.pi / 2          # nontrivial-index slice (core.py default)
BETA_OPT = 0.5433996420760803    # optimal p=1 QAOA β for k=8

CONV2_PREF_K8 = -(R / (2 ** K3))   # ≈ -0.6896  (BM24 Eq A41)

GAMMA_MESH = np.concatenate([
    np.linspace(-0.01, -0.40, 20),
    np.linspace(-0.40, -1.00, 13)[1:],
])

Z_STARTS = 300
W_STARTS = 150
CERTIFY_MAX = 8    # max roots to attempt certification per (gamma, system)
DPS = 60
RNG_SEED = 42


# ---------------------------------------------------------------------------
# Point-level certification helpers
# ---------------------------------------------------------------------------

def certify_z(sys_z: SaddleSystem, z: np.ndarray, dps: int = DPS) -> bool:
    try:
        ok, _ = krawczyk_certify_with_escalation(sys_z, z, dps, 4)
        return bool(ok)
    except Exception:
        return False


def certify_wu(sys_w: WSaddleSystem, w: np.ndarray, dps: int = DPS) -> bool:
    try:
        ok, _ = krawczyk_certify_reduced_escalate(sys_w, w, dps_start=dps, max_levels=3)
        return bool(ok)
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Per-gamma scan
# ---------------------------------------------------------------------------

@dataclass
class GammaPoint:
    gamma: float
    # z system
    z_n_roots: int = 0
    z_n_certified: int = 0
    z_n_attempted: int = 0
    z_time: float = 0.0
    z_phis: list[complex] = field(default_factory=list)
    z_error: Optional[str] = None
    # wu system
    wu_n_roots: int = 0
    wu_n_certified: int = 0
    wu_n_attempted: int = 0
    wu_time: float = 0.0
    wu_phis: list[complex] = field(default_factory=list)
    wu_error: Optional[str] = None


def probe_z_system(gamma: float, q: int, beta: float,
                   z_starts: int = Z_STARTS, dps: int = DPS) -> tuple[list[complex], int, int, float, Optional[str]]:
    t0 = time.perf_counter()
    try:
        sys_z = SaddleSystem.build(q=q, r=R, betas=np.array([beta]), gammas=np.array([gamma]))
        raw = discover_roots(sys_z, num_starts=z_starts, seed=RNG_SEED, tol=1e-12)
        n = sys_z.nvars
        phis = []
        z_roots = []
        for xr in raw:
            z = xr[:n] + 1j * xr[n:]
            phi = compute_phi(z, q=q, r=R, betas=np.array([beta]), gammas=np.array([gamma]))
            if np.isfinite(phi.real) and np.isfinite(phi.imag):
                phis.append(complex(phi))
                z_roots.append(z)
        # Attempt certification on the highest-Re(Phi) roots first
        order = np.argsort([-p.real for p in phis])
        n_cert = 0
        n_att = min(len(z_roots), CERTIFY_MAX)
        for idx in order[:n_att]:
            if certify_z(sys_z, z_roots[idx], dps=dps):
                n_cert += 1
        elapsed = time.perf_counter() - t0
        return phis, n_att, n_cert, elapsed, None
    except Exception as e:
        return [], 0, 0, time.perf_counter() - t0, str(e)


def probe_wu_system(gamma: float, q: int, beta: float,
                    wu_starts: int = W_STARTS, dps: int = DPS) -> tuple[list[complex], int, int, float, Optional[str]]:
    t0 = time.perf_counter()
    try:
        sys_w = WSaddleSystem(r=R, gamma=gamma, q=q, beta=beta)
        raw = discover_w_roots(sys_w, num_starts=wu_starts, seed=RNG_SEED, tol=1e-12)
        # Augment with leading-order warm start
        w0, ok, _ = solve_w_from_init(sys_w, sys_w.leading_seed())
        if ok and not any(np.linalg.norm(w0 - np.asarray(w, dtype=complex), ord=np.inf) < 1e-5 for w in raw):
            raw.append(w0)
        phis = []
        w_roots = []
        for w in raw:
            w = np.asarray(w, dtype=complex)
            phi = sys_w.Phi_eff(w)
            if np.isfinite(phi.real) and np.isfinite(phi.imag):
                phis.append(complex(phi))
                w_roots.append(w)
        # Certify highest-Re(Phi_wu) roots first
        order = np.argsort([-p.real for p in phis])
        n_cert = 0
        n_att = min(len(w_roots), CERTIFY_MAX)
        for idx in order[:n_att]:
            if certify_wu(sys_w, w_roots[idx], dps=dps):
                n_cert += 1
        elapsed = time.perf_counter() - t0
        return phis, n_att, n_cert, elapsed, None
    except Exception as e:
        return [], 0, 0, time.perf_counter() - t0, str(e)


def scan_gamma(gammas: np.ndarray, q: int, beta: float, verbose: bool = True,
               z_starts: int = Z_STARTS, wu_starts: int = W_STARTS,
               dps: int = DPS) -> list[GammaPoint]:
    results = []
    for i, gamma in enumerate(gammas):
        if verbose:
            print(f"  γ={gamma:+.3f} ({i+1}/{len(gammas)})", end="  ", flush=True)

        z_phis, z_att, z_cert, z_t, z_err = probe_z_system(gamma, q, beta, z_starts=z_starts, dps=dps)
        wu_phis, wu_att, wu_cert, wu_t, wu_err = probe_wu_system(gamma, q, beta, wu_starts=wu_starts, dps=dps)

        pt = GammaPoint(gamma=float(gamma))
        pt.z_n_roots = len(z_phis)
        pt.z_n_attempted = z_att
        pt.z_n_certified = z_cert
        pt.z_time = z_t
        pt.z_phis = z_phis
        pt.z_error = z_err
        pt.wu_n_roots = len(wu_phis)
        pt.wu_n_attempted = wu_att
        pt.wu_n_certified = wu_cert
        pt.wu_time = wu_t
        pt.wu_phis = wu_phis
        pt.wu_error = wu_err
        results.append(pt)

        if verbose:
            z_cert_str = f"{z_cert}/{z_att}"
            wu_cert_str = f"{wu_cert}/{wu_att}"
            z_phi_str = f"max={max(p.real for p in z_phis):+.3f}" if z_phis else "no roots"
            wu_phi_str = f"max={max(p.real for p in wu_phis):+.3f}" if wu_phis else "no roots"
            print(f"z cert={z_cert_str} {z_phi_str}  |  wu cert={wu_cert_str} {wu_phi_str}")
    return results


# ---------------------------------------------------------------------------
# q-sweep certification test
# ---------------------------------------------------------------------------

def q_certification_sweep(
    gamma: float = -0.05,
    beta: float = BETA_SLICE,
    q_values: list[int] = None,
    z_starts: int = Z_STARTS,
    wu_starts: int = W_STARTS,
    dps: int = DPS,
) -> dict:
    """For each q in q_values, find roots and try to certify. Returns summary."""
    if q_values is None:
        q_values = [1, 2, 3, 4]
    results = {}
    print(f"\n=== q-certification sweep at gamma={gamma}, beta={'SLICE' if np.isclose(beta, BETA_SLICE) else f'{beta:.4f}'} ===")
    for q in q_values:
        k = 2 ** q
        # For q-sweep use SAME r=176.54 to be apples-to-apples about k-SAT diff
        z_phis, z_att, z_cert, z_t, z_err = probe_z_system(gamma, q, beta, z_starts=z_starts, dps=dps)
        wu_phis, wu_att, wu_cert, wu_t, wu_err = probe_wu_system(gamma, q, beta, wu_starts=wu_starts, dps=dps)
        results[q] = {
            "k": k,
            "z": {"n_roots": len(z_phis), "n_cert": z_cert, "n_att": z_att,
                  "max_phi": max((p.real for p in z_phis), default=float("nan"))},
            "wu": {"n_roots": len(wu_phis), "n_cert": wu_cert, "n_att": wu_att,
                   "max_phi": max((p.real for p in wu_phis), default=float("nan"))},
        }
        z_str = f"{z_cert}/{z_att}  max={results[q]['z']['max_phi']:+.4f}"
        wu_str = f"{wu_cert}/{wu_att}  max={results[q]['wu']['max_phi']:+.4f}"
        print(f"  q={q} (k={k}): z cert={z_str}  |  wu cert={wu_str}")
    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_comparison(results: list[GammaPoint], q: int, beta: float,
                    q_sweep: dict, out_path: Path) -> None:
    gammas = np.array([r.gamma for r in results])

    # Certify fractions
    z_cert_frac = np.array([
        r.z_n_certified / r.z_n_attempted if r.z_n_attempted > 0 else float("nan")
        for r in results
    ])
    wu_cert_frac = np.array([
        r.wu_n_certified / r.wu_n_attempted if r.wu_n_attempted > 0 else float("nan")
        for r in results
    ])

    # Root counts
    z_counts = np.array([r.z_n_roots for r in results])
    wu_counts = np.array([r.wu_n_roots for r in results])

    # Max Re(Phi) per system
    z_max_phi = np.array([
        max((p.real for p in r.z_phis), default=float("nan")) for r in results
    ])
    wu_max_phi = np.array([
        max((p.real for p in r.wu_phis), default=float("nan")) for r in results
    ])

    # Timing
    z_times = np.array([r.z_time for r in results])
    wu_times = np.array([r.wu_time for r in results])

    beta_label = "SLICE(-π/2)" if np.isclose(beta, BETA_SLICE) else f"OPT({beta:.4f})"
    fig = plt.figure(figsize=(15, 12))
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.42, wspace=0.32)

    ax_cert = fig.add_subplot(gs[0, :])       # certification — full width top
    ax_phi_z = fig.add_subplot(gs[1, 0])
    ax_phi_wu = fig.add_subplot(gs[1, 1])
    ax_cnt = fig.add_subplot(gs[2, 0])
    ax_qsweep = fig.add_subplot(gs[2, 1])

    # ---- panel 1: certification fraction vs gamma (the KEY panel) ----
    ax_cert.plot(gammas, z_cert_frac * 100, "C3o-", ms=5, lw=1.8,
                 label=f"z-saddle  (8D, c_root=(-c)^{{1/{2**q}}})")
    ax_cert.plot(gammas, wu_cert_frac * 100, "C0s-", ms=5, lw=1.8,
                 label=f"w,u-saddle (4D, no fractional root)")
    ax_cert.axhline(100, color="k", lw=0.5, ls=":")
    ax_cert.axhline(0, color="k", lw=0.5, ls=":")
    ax_cert.set_ylim(-5, 110)
    ax_cert.set_xlabel("γ")
    ax_cert.set_ylabel("% roots certified (Krawczyk)")
    ax_cert.set_title(
        f"Krawczyk certification rate: w,u vs z  |  q={q} (k={2**q}), r={R}, β={beta_label}\n"
        f"ADVANTAGE: w,u avoids fractional {2**q}-th root → certifies where z fails",
        fontsize=10,
    )
    ax_cert.legend(fontsize=9, loc="upper right")
    ax_cert.grid(True, alpha=0.25)

    # ---- panel 2: max Re(Phi_z) vs gamma ----
    ax_phi_z.plot(gammas, z_max_phi, "C3o-", ms=4, lw=1.5)
    ax_phi_z.axhline(0, color="k", lw=0.5, ls="--")
    ax_phi_z.set_xlabel("γ")
    ax_phi_z.set_ylabel("max Re(Φ_z)")
    ax_phi_z.set_title(f"z-saddle dominant action (8D)")
    ax_phi_z.grid(True, alpha=0.25)

    # ---- panel 3: max Re(Phi_wu) vs gamma ----
    ax_phi_wu.plot(gammas, wu_max_phi, "C0s-", ms=4, lw=1.5)
    ax_phi_wu.axhline(0, color="k", lw=0.5, ls="--")
    ax_phi_wu.set_xlabel("γ")
    ax_phi_wu.set_ylabel("max Re(Φ_wu)")
    ax_phi_wu.set_title(f"w,u-saddle dominant action (4D chart)")
    ax_phi_wu.grid(True, alpha=0.25)

    # ---- panel 4: root counts ----
    ax_cnt.plot(gammas, z_counts, "C3o-", ms=4, lw=1.5, label="z (8D)")
    ax_cnt.plot(gammas, wu_counts, "C0s--", ms=4, lw=1.5, label="w,u (4D)")
    ax_cnt.set_xlabel("γ")
    ax_cnt.set_ylabel("# distinct roots found")
    ax_cnt.set_title("Root discovery counts")
    ax_cnt.legend(fontsize=9)
    ax_cnt.grid(True, alpha=0.25)

    # ---- panel 5: q-sweep certification bar chart ----
    q_vals = sorted(q_sweep.keys())
    x = np.arange(len(q_vals))
    bar_w = 0.35
    z_cert_pcts = [100 * q_sweep[q]["z"]["n_cert"] / max(q_sweep[q]["z"]["n_att"], 1) for q in q_vals]
    wu_cert_pcts = [100 * q_sweep[q]["wu"]["n_cert"] / max(q_sweep[q]["wu"]["n_att"], 1) for q in q_vals]
    ax_qsweep.bar(x - bar_w/2, z_cert_pcts, bar_w, label="z-saddle", color="C3", alpha=0.75)
    ax_qsweep.bar(x + bar_w/2, wu_cert_pcts, bar_w, label="w,u-saddle", color="C0", alpha=0.75)
    ax_qsweep.set_xticks(x)
    ax_qsweep.set_xticklabels([f"q={q}\n(k={2**q})" for q in q_vals])
    ax_qsweep.set_ylabel("% certified (γ=-0.05)")
    ax_qsweep.set_ylim(0, 115)
    ax_qsweep.set_title(f"Certification rate vs q (γ=-0.05)\n(k=2^q; wu avoids k-th root)")
    ax_qsweep.legend(fontsize=9)
    ax_qsweep.grid(True, alpha=0.25, axis="y")
    for i, (z_v, wu_v) in enumerate(zip(z_cert_pcts, wu_cert_pcts)):
        ax_qsweep.text(i - bar_w/2, z_v + 2, f"{z_v:.0f}%", ha="center", va="bottom", fontsize=8, color="C3")
        ax_qsweep.text(i + bar_w/2, wu_v + 2, f"{wu_v:.0f}%", ha="center", va="bottom", fontsize=8, color="C0")

    fig.suptitle(
        f"w,u-saddle vs z-saddle: certification advantage for q=3 (k=8-SAT)\n"
        f"r={R}, β={beta_label}  |  z: {Z_STARTS} starts in C^8  |  wu: {W_STARTS} starts in C^4",
        fontsize=11,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved → {out_path}")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(results: list[GammaPoint]) -> None:
    print("\n" + "="*92)
    print(f"{'γ':>7}  {'z_roots':>7}  {'z_cert%':>8}  {'wu_roots':>8}  {'wu_cert%':>9}  "
          f"{'z_maxΦ':>8}  {'wu_maxΦ':>8}")
    print("-"*92)
    for r in results:
        z_pct = f"{100*r.z_n_certified/r.z_n_attempted:.0f}%" if r.z_n_attempted else "---"
        wu_pct = f"{100*r.wu_n_certified/r.wu_n_attempted:.0f}%" if r.wu_n_attempted else "---"
        z_max = f"{max((p.real for p in r.z_phis), default=float('nan')):+.4f}" if r.z_phis else "  n/a  "
        wu_max = f"{max((p.real for p in r.wu_phis), default=float('nan')):+.4f}" if r.wu_phis else "  n/a  "
        print(f"{r.gamma:+7.3f}  {r.z_n_roots:7d}  {z_pct:>8}  {r.wu_n_roots:8d}  {wu_pct:>9}  "
              f"{z_max:>8}  {wu_max:>8}")
    print("="*92)

    # Summary statistics
    z_cert_total = sum(r.z_n_certified for r in results)
    z_att_total = sum(r.z_n_attempted for r in results)
    wu_cert_total = sum(r.wu_n_certified for r in results)
    wu_att_total = sum(r.wu_n_attempted for r in results)
    print(f"\nOverall certification: z = {z_cert_total}/{z_att_total} "
          f"({100*z_cert_total/max(z_att_total,1):.1f}%)  |  "
          f"wu = {wu_cert_total}/{wu_att_total} "
          f"({100*wu_cert_total/max(wu_att_total,1):.1f}%)")
    if z_cert_total == 0 and wu_cert_total > 0:
        print("  → z-system: 0% certification  (fractional root breaks Krawczyk for q≥2)")
        print("  → wu-system: certifies rigorously (no fractional root in F_tilde)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--q", type=int, default=3, help="QAOA parameter q (k=2^q)")
    parser.add_argument("--beta", type=float, default=BETA_SLICE,
                        help=f"QAOA beta angle (default: SLICE_BETA={BETA_SLICE:.4f})")
    parser.add_argument("--gamma-start", type=float, default=-0.01)
    parser.add_argument("--gamma-stop", type=float, default=-0.80)
    parser.add_argument("--num-points", type=int, default=30)
    parser.add_argument("--z-starts", type=int, default=Z_STARTS)
    parser.add_argument("--wu-starts", type=int, default=W_STARTS)
    parser.add_argument("--dps", type=int, default=DPS)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parent / "runs" / "compare_z_wu.png")
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--skip-q-sweep", action="store_true")
    args = parser.parse_args()

    beta_label = "SLICE" if np.isclose(args.beta, BETA_SLICE) else f"{args.beta:.4f}"
    print(f"Comparing z-saddle vs w,u-saddle: q={args.q}, k={2**args.q}, r={R}, β={beta_label}")
    print(f"  z: {args.z_starts} starts in C^8 (16 real) | wu: {args.wu_starts} starts in C^4 (8 real)")
    print(f"  Krawczyk dps={args.dps}, max_roots_certified={CERTIFY_MAX}\n")

    gammas = np.linspace(args.gamma_start, args.gamma_stop, args.num_points)
    print(f"Gamma sweep: {len(gammas)} points from {args.gamma_start} to {args.gamma_stop}")
    results = scan_gamma(gammas, q=args.q, beta=args.beta, verbose=not args.quiet,
                         z_starts=args.z_starts, wu_starts=args.wu_starts, dps=args.dps)
    print_summary(results)

    q_sweep = {}
    if not args.skip_q_sweep:
        q_sweep = q_certification_sweep(gamma=-0.05, beta=args.beta,
                                        z_starts=args.z_starts, wu_starts=args.wu_starts,
                                        dps=args.dps)

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        dump = {
            "params": {"q": args.q, "k": 2**args.q, "r": R, "beta": args.beta},
            "conv2_prefactor_k8": CONV2_PREF_K8,
            "results": [
                {
                    "gamma": r.gamma,
                    "z": {"n_roots": r.z_n_roots, "n_cert": r.z_n_certified,
                          "n_att": r.z_n_attempted, "time": r.z_time,
                          "phis_real": [p.real for p in r.z_phis],
                          "phis_imag": [p.imag for p in r.z_phis]},
                    "wu": {"n_roots": r.wu_n_roots, "n_cert": r.wu_n_certified,
                           "n_att": r.wu_n_attempted, "time": r.wu_time,
                           "phis_real": [p.real for p in r.wu_phis],
                           "phis_imag": [p.imag for p in r.wu_phis]},
                }
                for r in results
            ],
            "q_sweep": q_sweep,
        }
        args.json.write_text(json.dumps(dump, indent=2))
        print(f"JSON → {args.json}")

    plot_comparison(results, q=args.q, beta=args.beta, q_sweep=q_sweep, out_path=args.output)


if __name__ == "__main__":
    main()
