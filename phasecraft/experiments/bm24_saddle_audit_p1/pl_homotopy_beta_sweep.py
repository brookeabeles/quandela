"""Run the PL homotopy necessary-condition test across refined beta transitions."""

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

from phasecraft.bm24_saddle_audit_p1 import pl_homotopy_tracker as ht


RESULTS = HERE / "results"
REFINED_SUMMARY = RESULTS / "refined_transition_scan" / "refined_transition_scan_summary.json"
OUT = RESULTS / "pl_homotopy_beta_sweep"


def load_refined_transitions() -> list[dict[str, Any]]:
    with REFINED_SUMMARY.open() as f:
        data = json.load(f)
    return data["transitions"]


def is_robust_single(t: dict[str, Any]) -> bool:
    trs = t.get("transitions", [])
    return (
        len(trs) == 1
        and trs[0].get("from") == "seed_or_decoy"
        and trs[0].get("to") == "competitor"
    )


def gamma_grid_around(lo: float, hi: float) -> list[float]:
    vals = ht.default_Gamma_grid()
    vals += [lo, hi, 0.5 * (lo + hi)]
    vals += [round(lo - 0.025, 10), round(hi + 0.025, 10)]
    return sorted({v for v in vals if 0.05 <= v <= 2.15})


def interval_overlap(a: list[float], b: list[float]) -> bool:
    alo, ahi = sorted(a)
    blo, bhi = sorted(b)
    return max(alo, blo) <= min(ahi, bhi)


def run_one(beta: float, transition: dict[str, Any], starts: int, stokes_tol: float, anti_tol: float) -> dict[str, Any]:
    lo = float(transition["Gamma_low"])
    hi = float(transition["Gamma_high"])
    anchor_G = hi
    Gammas = gamma_grid_around(lo, hi)
    gammas = sorted([-g for g in Gammas], reverse=True)
    anchor_gamma = -anchor_G
    if anchor_gamma not in gammas:
        gammas.append(anchor_gamma)
        gammas = sorted(set(gammas), reverse=True)

    anchor = ht.choose_anchor_competitor(beta, anchor_gamma, starts)
    branch = ht.continue_branch(beta, anchor_gamma, anchor["z"], gammas)
    branch[anchor_gamma] = {
        **branch[anchor_gamma],
        **{k: v for k, v in anchor.items() if k != "z"},
        "certified": True,
        "failed": False,
        "residual_inf": 0.0,
        "gap_abs": anchor["exact_gap_abs"],
    }
    rows = ht.build_rows(beta, gammas, branch, stokes_tol=stokes_tol, anti_tol=anti_tol)
    crossings = ht.estimate_crossings(rows)
    physical = crossings["physical_switch_intervals"]
    anti = crossings["anti_stokes_intervals"]
    overlaps = [
        {"physical": p, "anti_stokes": a}
        for p in physical
        for a in anti
        if interval_overlap(p, a)
    ]
    ok_rows = [r for r in rows if not r.get("competitor_failed")]
    near_transition = [
        r for r in ok_rows
        if lo - 0.05 <= float(r["Gamma"]) <= hi + 0.05
    ]
    max_phase_near = max((float(r["delta_im_mod_2pi_abs"]) for r in near_transition), default=float("nan"))
    min_abs_dre_near = min((abs(float(r["delta_re_phi"])) for r in near_transition), default=float("nan"))
    return {
        "beta": beta,
        "refined_transition": transition,
        "anchor_Gamma": anchor_G,
        "crossings": crossings,
        "overlaps": overlaps,
        "alignment": bool(overlaps),
        "n_valid_rows": len(ok_rows),
        "n_merged_seed_rows": sum(1 for r in rows if r.get("competitor_merged_with_seed")),
        "max_phase_distance_near_refined_transition": max_phase_near,
        "min_abs_delta_re_near_refined_transition": min_abs_dre_near,
        "rows": rows,
    }


def plot_summary(results: list[dict[str, Any]]) -> str:
    OUT.mkdir(parents=True, exist_ok=True)
    betas = np.array([r["beta"] for r in results])
    refined_mid = np.array([r["refined_transition"]["Gamma_mid"] for r in results])
    refined_err = np.array([r["refined_transition"]["uncertainty"] for r in results])
    anti_mid = []
    anti_err = []
    aligned = []
    for r in results:
        if r["overlaps"]:
            a = r["overlaps"][0]["anti_stokes"]
            anti_mid.append(0.5 * (a[0] + a[1]))
            anti_err.append(0.5 * abs(a[1] - a[0]))
            aligned.append(True)
        else:
            anti_mid.append(np.nan)
            anti_err.append(np.nan)
            aligned.append(False)
    anti_mid = np.array(anti_mid, dtype=float)
    anti_err = np.array(anti_err, dtype=float)

    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    ax.errorbar(betas, refined_mid, yerr=refined_err, fmt="o-", color="#2b6cb0", label="exact-exponent switch")
    ax.errorbar(betas, anti_mid, yerr=anti_err, fmt="s--", color="#dd6b20", label="continued-branch anti-Stokes")
    for beta, ok in zip(betas, aligned):
        if not ok:
            ax.scatter([beta], [refined_mid[np.where(betas == beta)[0][0]]], marker="x", s=80, color="#c53030")
    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel(r"$\Gamma=-\gamma$")
    ax.set_title("Do PL anti-Stokes brackets align with refined physical switches?")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    path = OUT / "pl_beta_sweep_alignment.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def write_report(results: list[dict[str, Any]], skipped: list[dict[str, Any]], figure: str, starts: int) -> str:
    aligned = [r for r in results if r["alignment"]]
    lines = [
        "# PL homotopy beta sweep",
        "",
        "## Question",
        "",
        "Does the PL/Stokes alignment seen at beta_opt also occur at the other refined robust transition points?",
        "",
        "## Method",
        "",
        "For each robust single-switch beta slice, anchor on the best exact-matching competitor just after the refined transition, continue the same branch backward/forward in Gamma, remove points where it merges into the seed, and compare the continued-branch anti-Stokes bracket to the refined exact-exponent switch bracket.",
        "",
        f"- competitor starts at each anchor: {starts}",
        f"- tested robust single-switch slices: {len(results)}",
        f"- aligned slices: {len(aligned)}",
        f"- skipped ambiguous/nonmonotone slices: {len(skipped)}",
        "",
        "## Results",
        "",
        "| beta | refined switch | anti-Stokes brackets | overlap? | max phase dist near switch | merged small-Gamma rows |",
        "|---:|---|---|---|---:|---:|",
    ]
    for r in results:
        refined = r["refined_transition"]
        anti = r["crossings"]["anti_stokes_intervals"]
        lines.append(
            f"| {r['beta']:.6f} | [{refined['Gamma_low']:.4f}, {refined['Gamma_high']:.4f}] | "
            f"{anti} | {'yes' if r['alignment'] else 'no'} | "
            f"{r['max_phase_distance_near_refined_transition']:.3g} | {r['n_merged_seed_rows']} |"
        )
    lines += [
        "",
        "## Skipped",
        "",
    ]
    if skipped:
        for s in skipped:
            lines.append(f"- beta={s['beta']:.6f}: {len(s.get('transitions', []))} transition(s), not a robust single switch")
    else:
        lines.append("- none")
    lines += [
        "",
        "## Interpretation",
        "",
        "- `overlap=yes` means the continued branch's anti-Stokes bracket intersects the exact-exponent switch bracket from the refined scan.",
        "- This is still a necessary-condition PL test, not an intersection-number proof.",
        "- Rows merged into the seed at small Gamma are explicitly filtered because they otherwise create fake Stokes/anti-Stokes events.",
        "",
        "## Figure",
        "",
        f"- `{Path(figure).name}`",
        "",
    ]
    path = OUT / "pl_beta_sweep_report.md"
    path.write_text("\n".join(lines))
    return str(path)


def run(args: argparse.Namespace) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    transitions = load_refined_transitions()
    robust = [t for t in transitions if is_robust_single(t)]
    skipped = [t for t in transitions if not is_robust_single(t)]
    if args.max_betas is not None:
        robust = robust[: args.max_betas]
    results = []
    for i, t in enumerate(robust, start=1):
        beta = float(t["beta"])
        tr = t["transitions"][0]
        print(f"[{i}/{len(robust)}] beta={beta:.6f}, transition=[{tr['Gamma_low']:.4f},{tr['Gamma_high']:.4f}]", flush=True)
        try:
            results.append(run_one(beta, tr, args.starts, args.stokes_tol, args.anti_tol))
        except Exception as exc:
            results.append({
                "beta": beta,
                "refined_transition": tr,
                "anchor_Gamma": tr["Gamma_high"],
                "crossings": {"anti_stokes_intervals": [], "physical_switch_intervals": [], "candidate_pl_jump_points": []},
                "overlaps": [],
                "alignment": False,
                "error": str(exc),
                "n_valid_rows": 0,
                "n_merged_seed_rows": 0,
                "max_phase_distance_near_refined_transition": float("nan"),
                "min_abs_delta_re_near_refined_transition": float("nan"),
                "rows": [],
            })
            print(f"  failed: {exc}", flush=True)
    figure = plot_summary(results)
    report = write_report(results, skipped, figure, args.starts)
    out_json = OUT / "pl_beta_sweep.json"
    with out_json.open("w") as f:
        json.dump(
            {
                "metadata": {
                    "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "starts": args.starts,
                    "stokes_tol": args.stokes_tol,
                    "anti_tol": args.anti_tol,
                },
                "results": results,
                "skipped": skipped,
                "figure": figure,
                "report": report,
            },
            f,
            indent=2,
        )
    print("\nWrote:")
    print(f"  {out_json}")
    print(f"  {report}")
    print(f"  {figure}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--starts", type=int, default=80)
    parser.add_argument("--stokes-tol", type=float, default=0.08)
    parser.add_argument("--anti-tol", type=float, default=0.02)
    parser.add_argument("--max-betas", type=int, default=None)
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
