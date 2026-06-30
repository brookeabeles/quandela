"""Homotopy-style Picard-Lefschetz branch tracker for BM24 p=1.

This script follows the next PL step after the local probe:

1. Anchor on the best exact-matching competitor just after the beta-opt crossing.
2. Continue that same competitor branch backward toward small Gamma and forward
   to larger Gamma.
3. Track seed-vs-competitor Delta Re(Phi), unwrapped Delta Im(Phi), exact
   finite-n exponent gaps, and near-Stokes/anti-Stokes windows.

This still does not compute rigorous intersection numbers.  It tells us where
an intersection-number jump is possible along the homotopy from small Gamma,
and whether that jump window coincides with the physical dominance transition.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-phasecraft")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
PACKAGE_ROOT = HERE.parents[2]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from phasecraft.bm24_saddle_audit_p1 import refined_transition_scan as refined
from phasecraft.bm24_saddle_audit_p1.audit import _x_to_z
from phasecraft.bm24_saddle_audit_p1.continue_seed_branch_certified import (
    certify_z,
    conv2_full_exponent,
    conv2_pref,
    lambda_abs_n_max,
    polish_to_residual,
)
from phasecraft.bm24_saddle_audit_p1.picard_lefschetz_probe import (
    discover_relevant_saddles,
    flow_probe_for_saddle,
)
from phasecraft.lib.saddles.krawczyk_p1_roots import SaddleSystem
from phasecraft.lib.saddles.picard_lefschetz import compute_phi


Q = 3
K_CLAUSE = 8
R = 176.54
BETA = 0.5433996420760803
N_VALUES = list(range(18, 25))
PHI_PREF = conv2_pref(K_CLAUSE, R)

OUT = HERE / "results" / "pl_homotopy_tracker"


def default_Gamma_grid() -> list[float]:
    vals = [
        0.05, 0.30, 0.60, 1.00, 1.30, 1.50, 1.60, 1.70, 1.75,
        1.80, 1.825, 1.85, 1.875, 1.90, 1.95, 2.00, 2.05, 2.10,
    ]
    return vals


def seed_row(beta: float, gamma: float) -> tuple[np.ndarray, dict[str, Any]]:
    z_seed, ok = refined.warm_seed_to(beta, gamma)
    if z_seed is None or not ok:
        raise RuntimeError(f"seed continuation failed at gamma={gamma}")
    sys = SaddleSystem.build(q=Q, r=R, betas=np.array([beta]), gammas=np.array([gamma]))
    phi = compute_phi(z_seed, q=Q, r=R, betas=sys.betas, gammas=sys.gammas)
    full = conv2_full_exponent(float(phi.real), K_CLAUSE, R)
    lam = lambda_abs_n_max(Q, K_CLAUSE, R, beta, gamma, N_VALUES)
    return z_seed, {
        "re_phi": float(phi.real),
        "im_phi": float(phi.imag),
        "full_exponent": float(full),
        "lambda_abs": float(lam),
        "gap_abs": float(abs(full - lam)),
        "residual_inf": float(np.linalg.norm(sys.G_complex(z_seed), ord=np.inf)),
    }


def seed_row_from_z(beta: float, gamma: float, z_seed: np.ndarray) -> dict[str, Any]:
    sys = SaddleSystem.build(q=Q, r=R, betas=np.array([beta]), gammas=np.array([gamma]))
    phi = compute_phi(z_seed, q=Q, r=R, betas=sys.betas, gammas=sys.gammas)
    full = conv2_full_exponent(float(phi.real), K_CLAUSE, R)
    lam = lambda_abs_n_max(Q, K_CLAUSE, R, beta, gamma, N_VALUES)
    return {
        "re_phi": float(phi.real),
        "im_phi": float(phi.imag),
        "full_exponent": float(full),
        "lambda_abs": float(lam),
        "gap_abs": float(abs(full - lam)),
        "residual_inf": float(np.linalg.norm(sys.G_complex(z_seed), ord=np.inf)),
    }


def build_seed_rows(beta: float, gammas: list[float]) -> dict[float, dict[str, Any]]:
    out: dict[float, dict[str, Any]] = {}
    z_current: np.ndarray | None = None
    for gamma in sorted(gammas, reverse=True):
        if z_current is not None:
            z_next, ok = refined.seed_from_previous(z_current, beta, gamma)
            if z_next is None or not ok:
                z_next, ok = refined.warm_seed_to(beta, gamma)
        else:
            z_next, ok = refined.warm_seed_to(beta, gamma)
        if z_next is None or not ok:
            out[gamma] = {"failed": True, "failure_reason": "seed_continuation_failed"}
            z_current = None
            continue
        z_current = z_next
        out[gamma] = {"failed": False, "z": z_current, **seed_row_from_z(beta, gamma, z_current)}
    return out


def choose_anchor_competitor(beta: float, gamma: float, starts: int) -> dict[str, Any]:
    _z_seed, saddles, lam = discover_relevant_saddles(beta, gamma, starts)
    comps = saddles[1:]
    if not comps:
        raise RuntimeError(f"no certified competitors found at anchor gamma={gamma}")
    best = min(comps, key=lambda s: s["exact_gap_abs"])
    return {
        "z": best["z"],
        "re_phi": best["re_phi"],
        "im_phi": best["im_phi"],
        "full_exponent": best["full_exponent"],
        "exact_gap_abs": best["exact_gap_abs"],
        "lambda_abs": lam,
    }


def continue_competitor_to(beta: float, gamma: float, z_prev: np.ndarray) -> tuple[np.ndarray | None, dict[str, Any]]:
    sys = SaddleSystem.build(q=Q, r=R, betas=np.array([beta]), gammas=np.array([gamma]))
    x_prev = np.concatenate([z_prev.real, z_prev.imag])
    x_pol, res = polish_to_residual(sys, x_prev, min_residual=1e-10, dps=80)
    z_pol = _x_to_z(x_pol, sys.nvars)
    ok, proof = certify_z(sys, z_pol, dps=80)
    phi = compute_phi(z_pol, q=Q, r=R, betas=sys.betas, gammas=sys.gammas)
    full = conv2_full_exponent(float(phi.real), K_CLAUSE, R)
    lam = lambda_abs_n_max(Q, K_CLAUSE, R, beta, gamma, N_VALUES)
    failed = (not ok) or (not np.isfinite(res)) or res > 1e-7
    row = {
        "certified": bool(ok) and not failed,
        "failed": bool(failed),
        "failure_reason": "" if not failed else str(proof.get("reason", "cert_or_residual_failed")),
        "residual_inf": float(res),
        "krawczyk_contraction": float(proof.get("contraction_bound", math.inf)),
        "re_phi": float(phi.real),
        "im_phi": float(phi.imag),
        "full_exponent": float(full),
        "lambda_abs": float(lam),
        "gap_abs": float(abs(full - lam)),
        "z_step_dist": float(np.linalg.norm(z_pol - z_prev, ord=np.inf)),
        "z_real": z_pol.real.tolist(),
        "z_imag": z_pol.imag.tolist(),
    }
    return (None if failed else z_pol), row


def continue_branch(beta: float, anchor_gamma: float, z_anchor: np.ndarray, gammas: list[float]) -> dict[float, dict[str, Any]]:
    by_gamma: dict[float, dict[str, Any]] = {}
    by_gamma[anchor_gamma] = {"z": z_anchor, "source": "anchor"}

    # Toward zero: increasing gamma from anchor to least negative.
    z = z_anchor
    for gamma in sorted([g for g in gammas if g > anchor_gamma]):
        z_next, row = continue_competitor_to(beta, gamma, z)
        row["source"] = "continued_toward_zero"
        if z_next is None:
            by_gamma[gamma] = row
            break
        row["z"] = z_next
        by_gamma[gamma] = row
        z = z_next

    # Toward larger Gamma: decreasing gamma from anchor.
    z = z_anchor
    for gamma in sorted([g for g in gammas if g < anchor_gamma], reverse=True):
        z_next, row = continue_competitor_to(beta, gamma, z)
        row["source"] = "continued_toward_large_Gamma"
        if z_next is None:
            by_gamma[gamma] = row
            break
        row["z"] = z_next
        by_gamma[gamma] = row
        z = z_next

    return by_gamma


def strip_z(row: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in row.items() if k != "z"}


def im_mod_distance(x: float) -> float:
    y = (x + math.pi) % (2.0 * math.pi) - math.pi
    return abs(y)


def build_rows(
    beta: float,
    gammas: list[float],
    branch: dict[float, dict[str, Any]],
    *,
    stokes_tol: float,
    anti_tol: float,
) -> list[dict[str, Any]]:
    rows = []
    raw_delta_ims = []
    temp_rows = []
    seed_by_gamma = build_seed_rows(beta, gammas)
    for gamma in sorted(gammas, reverse=True):
        srow = seed_by_gamma.get(gamma, {"failed": True, "failure_reason": "seed_missing"})
        if srow.get("failed"):
            temp_rows.append({
                "gamma": gamma,
                "Gamma": -gamma,
                "seed_failed": True,
                "competitor_failed": True,
                "seed": srow,
            })
            raw_delta_ims.append(np.nan)
            continue
        crow = branch.get(gamma)
        if crow is None or crow.get("failed"):
            temp_rows.append({"gamma": gamma, "Gamma": -gamma, "seed": strip_z(srow), "competitor_failed": True})
            raw_delta_ims.append(np.nan)
            continue
        z_seed = srow.get("z")
        z_comp = crow.get("z")
        z_dist_seed = float(np.linalg.norm(z_comp - z_seed, ord=np.inf)) if z_seed is not None and z_comp is not None else math.inf
        if z_dist_seed < 1e-4:
            temp_rows.append({
                "gamma": gamma,
                "Gamma": -gamma,
                "seed": strip_z(srow),
                "competitor_failed": True,
                "competitor_merged_with_seed": True,
                "z_distance_from_seed": z_dist_seed,
            })
            raw_delta_ims.append(np.nan)
            continue
        delta_re = float(crow["re_phi"] - srow["re_phi"])
        delta_im = float(crow["im_phi"] - srow["im_phi"])
        raw_delta_ims.append(delta_im)
        temp_rows.append(
            {
                "gamma": gamma,
                "Gamma": -gamma,
                "seed": strip_z(srow),
                "competitor": strip_z(crow),
                "z_distance_from_seed": z_dist_seed,
                "delta_re_phi": delta_re,
                "delta_full_exponent": float(crow["full_exponent"] - srow["full_exponent"]),
                "delta_im_phi_raw": delta_im,
                "seed_best_exact": bool(srow["gap_abs"] <= crow["gap_abs"]),
                "competitor_best_exact": bool(crow["gap_abs"] < srow["gap_abs"]),
                "competitor_failed": False,
            }
        )

    finite = np.array([0.0 if np.isnan(x) else x for x in raw_delta_ims], dtype=float)
    unwrapped = np.unwrap(finite)
    for row, dim_unwrapped in zip(temp_rows, unwrapped):
        if row.get("competitor_failed"):
            rows.append(row)
            continue
        row["delta_im_phi_unwrapped"] = float(dim_unwrapped)
        row["delta_im_mod_2pi_abs"] = im_mod_distance(float(dim_unwrapped))
        row["near_stokes"] = bool(row["delta_im_mod_2pi_abs"] <= stokes_tol)
        row["near_anti_stokes"] = bool(abs(float(row["delta_re_phi"])) <= anti_tol)
        row["candidate_pl_jump_window"] = bool(row["near_stokes"] and row["near_anti_stokes"])
        rows.append(row)
    return rows


def estimate_crossings(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [r for r in rows if not r.get("competitor_failed")]
    out: dict[str, Any] = {"anti_stokes_intervals": [], "physical_switch_intervals": [], "candidate_pl_jump_points": []}
    for a, b in zip(ok[:-1], ok[1:]):
        dra = float(a["delta_re_phi"])
        drb = float(b["delta_re_phi"])
        if dra == 0.0 or dra * drb < 0.0:
            out["anti_stokes_intervals"].append([a["Gamma"], b["Gamma"]])
        pa = bool(a["competitor_best_exact"])
        pb = bool(b["competitor_best_exact"])
        if pa != pb:
            out["physical_switch_intervals"].append([a["Gamma"], b["Gamma"]])
    for r in ok:
        if r["candidate_pl_jump_window"]:
            out["candidate_pl_jump_points"].append(r["Gamma"])
    return out


def plot(rows: list[dict[str, Any]], crossings: dict[str, Any]) -> list[str]:
    OUT.mkdir(parents=True, exist_ok=True)
    ok = [r for r in rows if not r.get("competitor_failed")]
    G = np.array([r["Gamma"] for r in ok])
    dRe = np.array([r["delta_re_phi"] for r in ok])
    dIm = np.array([r["delta_im_mod_2pi_abs"] for r in ok])
    seed_gap = np.array([r["seed"]["gap_abs"] for r in ok])
    comp_gap = np.array([r["competitor"]["gap_abs"] for r in ok])

    paths = []
    fig, axes = plt.subplots(2, 1, figsize=(9.5, 7.0), sharex=True)
    axes[0].axhline(0.0, color="#4a5568", lw=1.0)
    axes[0].plot(G, dRe, "o-", color="#6b46c1", label=r"$\Delta Re\,\Phi$ competitor - seed")
    for lo, hi in crossings["anti_stokes_intervals"]:
        axes[0].axvspan(lo, hi, color="#dd6b20", alpha=0.18, label="anti-Stokes bracket")
    axes[0].set_ylabel(r"$\Delta Re\,\Phi$")
    axes[0].legend(loc="best", fontsize=9)
    axes[0].grid(True, alpha=0.25)

    axes[1].axhline(0.05, color="#c53030", lw=1.0, ls="--", label="Stokes tol")
    axes[1].plot(G, dIm, "o-", color="#2b6cb0", label=r"$|\Delta Im\,\Phi|$ mod $2\pi$")
    for g in crossings["candidate_pl_jump_points"]:
        axes[1].axvline(g, color="#c53030", alpha=0.35)
    axes[1].set_xlabel(r"$\Gamma=-\gamma$")
    axes[1].set_ylabel(r"phase distance")
    axes[1].legend(loc="best", fontsize=9)
    axes[1].grid(True, alpha=0.25)
    fig.suptitle("Homotopy PL necessary conditions along one continued competitor branch")
    fig.tight_layout()
    p = OUT / "pl_homotopy_stokes_anti_stokes.png"
    fig.savefig(p, dpi=180, bbox_inches="tight")
    plt.close(fig)
    paths.append(str(p))

    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    ax.plot(G, seed_gap, "o-", color="#2b6cb0", label="seed exact gap")
    ax.plot(G, comp_gap, "o-", color="#dd6b20", label="continued competitor exact gap")
    for lo, hi in crossings["physical_switch_intervals"]:
        ax.axvspan(lo, hi, color="#dd6b20", alpha=0.18, label="exact-match switch")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\Gamma=-\gamma$")
    ax.set_ylabel(r"$|E-\lambda_n|$")
    ax.set_title("Physical exact-exponent switch along the same branch")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    p = OUT / "pl_homotopy_exact_gap_switch.png"
    fig.savefig(p, dpi=180, bbox_inches="tight")
    plt.close(fig)
    paths.append(str(p))
    return paths


def write_report(result: dict[str, Any]) -> str:
    rows = [r for r in result["rows"] if not r.get("competitor_failed")]
    lines = [
        "# PL homotopy branch tracker",
        "",
        "## Verdict",
        "",
        "This is the first practical homotopy-style PL diagnostic: it follows one competitor branch from the post-crossing region back toward small Gamma and asks where a Stokes jump could plausibly turn on its intersection number.  It is still not a rigorous intersection-number computation.",
        "",
        "## Setup",
        "",
        f"- beta: {result['beta']}",
        f"- anchor Gamma: {-result['anchor_gamma']}",
        f"- anchor competitor search starts: {result['starts']}",
        f"- Stokes tolerance: {result['stokes_tol']}",
        f"- anti-Stokes tolerance: {result['anti_tol']}",
        "",
        "## Crossing Summary",
        "",
        f"- anti-Stokes brackets: {result['crossings']['anti_stokes_intervals']}",
        f"- physical exact-match switch brackets: {result['crossings']['physical_switch_intervals']}",
        f"- candidate simultaneous PL jump points: {result['crossings']['candidate_pl_jump_points']}",
        "",
        "## Branch Table",
        "",
        "| Gamma | Delta Re Phi | |Delta Im Phi| mod 2pi | seed gap | comp gap | exact winner | PL candidate? |",
        "|---:|---:|---:|---:|---:|---|---|",
    ]
    for r in rows:
        winner = "competitor" if r["competitor_best_exact"] else "seed"
        lines.append(
            f"| {r['Gamma']:.6f} | {r['delta_re_phi']:+.6g} | {r['delta_im_mod_2pi_abs']:.6g} | "
            f"{r['seed']['gap_abs']:.6g} | {r['competitor']['gap_abs']:.6g} | {winner} | "
            f"{'yes' if r['candidate_pl_jump_window'] else ''} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "- Start of homotopy assumption: at small Gamma, seed intersection number is 1 and this competitor branch is off.",
        "- A necessary place for the competitor intersection number to change is a Stokes condition, `Delta Im Phi = 0 mod 2pi`.",
        "- Exponential dominance changes at anti-Stokes, `Delta Re Phi = 0`.",
        "- If the Stokes and anti-Stokes windows coincide, that is the strongest numerical evidence for the PL mechanism behind the observed physical transition.",
        "",
        "## Figures",
        "",
    ]
    lines += [f"- `{Path(p).name}`" for p in result["figures"]]
    lines.append("")
    path = OUT / "pl_homotopy_tracker_report.md"
    path.write_text("\n".join(lines))
    return str(path)


def run(args: argparse.Namespace) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    Gammas = args.Gammas if args.Gammas else default_Gamma_grid()
    gammas = sorted([-float(G) for G in Gammas], reverse=True)
    anchor_gamma = -float(args.anchor_Gamma)
    if anchor_gamma not in gammas:
        gammas.append(anchor_gamma)
        gammas = sorted(set(gammas), reverse=True)

    print(f"Anchoring competitor at beta={args.beta:.6f}, Gamma={-anchor_gamma:.6f}")
    anchor = choose_anchor_competitor(args.beta, anchor_gamma, args.starts)
    branch = continue_branch(args.beta, anchor_gamma, anchor["z"], gammas)
    branch[anchor_gamma] = {
        **branch[anchor_gamma],
        **{k: v for k, v in anchor.items() if k != "z"},
        "certified": True,
        "failed": False,
        "residual_inf": 0.0,
        "gap_abs": anchor["exact_gap_abs"],
    }
    rows = build_rows(args.beta, gammas, branch, stokes_tol=args.stokes_tol, anti_tol=args.anti_tol)
    crossings = estimate_crossings(rows)
    figures = plot(rows, crossings)

    result = {
        "metadata": {
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "caveat": "Necessary-condition homotopy tracker, not rigorous PL intersections.",
        },
        "beta": args.beta,
        "anchor_gamma": anchor_gamma,
        "starts": args.starts,
        "stokes_tol": args.stokes_tol,
        "anti_tol": args.anti_tol,
        "crossings": crossings,
        "rows": rows,
        "figures": figures,
    }
    report = write_report(result)
    result["report"] = report
    out_json = OUT / "pl_homotopy_tracker.json"
    with out_json.open("w") as f:
        json.dump(result, f, indent=2)

    print("\nWrote:")
    print(f"  {out_json}")
    print(f"  {report}")
    for fig in figures:
        print(f"  {fig}")


def parse_csv(value: str) -> list[float]:
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--beta", type=float, default=BETA)
    parser.add_argument("--anchor-Gamma", type=float, default=1.85)
    parser.add_argument("--starts", type=int, default=200)
    parser.add_argument("--stokes-tol", type=float, default=0.08)
    parser.add_argument("--anti-tol", type=float, default=0.02)
    parser.add_argument("--Gammas", type=parse_csv, default=None)
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
