"""Run PL homotopy diagnostics across all refined beta slices.

For each beta in the refined transition scan, choose an anchor Gamma from the
longest contiguous competitor-best run, continue that same competitor branch,
and compare:

* anti-Stokes brackets, Delta Re Phi = 0
* Stokes phase proximity, Delta Im Phi = 0 mod 2pi
* exact finite-n winner switch

Outputs per-beta tracker files plus a summary figure/table.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
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

from phasecraft.bm24_saddle_audit_p1 import pl_homotopy_tracker as tracker


RESULTS = HERE / "results"
REFINED = RESULTS / "refined_transition_scan" / "refined_transition_scan.json"
OUT = RESULTS / "pl_homotopy_all_slices"

PURPLE = "#6b46c1"
ORANGE = "#dd6b20"
BLUE = "#2b6cb0"
GREEN = "#2f855a"
RED = "#c53030"
GRAY = "#4a5568"


@dataclass
class AnchorChoice:
    beta: float
    anchor_Gamma: float
    run_start: float
    run_end: float
    run_length: int
    refined_regime: str


def load_json(path: Path) -> Any:
    with path.open() as f:
        return json.load(f)


def refined_class(row: dict[str, Any], decoy_margin: float = 0.02) -> str:
    if not bool(row.get("seed_is_best_match_to_exact")):
        return "competitor"
    if float(row.get("max_competitor_delta_re", 0.0)) > decoy_margin:
        return "decoy"
    return "seed"


def choose_anchor(rows_b: list[dict[str, Any]]) -> AnchorChoice:
    beta = float(rows_b[0]["beta"])
    rows = sorted(rows_b, key=lambda r: -float(r["gamma"]))
    comp_G = [-float(r["gamma"]) for r in rows if refined_class(r) == "competitor"]
    if not comp_G:
        raise RuntimeError(f"no competitor-best refined points for beta={beta}")
    comp_G = sorted(comp_G)
    runs: list[list[float]] = []
    cur = [comp_G[0]]
    for a, b in zip(comp_G[:-1], comp_G[1:]):
        if b - a <= 0.051:
            cur.append(b)
        else:
            runs.append(cur)
            cur = [b]
    runs.append(cur)
    # Prefer longest stable run; if tied, choose the later-Gamma run.
    run = max(runs, key=lambda xs: (len(xs), xs[0]))
    seed_flags = [refined_class(r) != "competitor" for r in rows]
    transitions = sum(1 for a, b in zip(seed_flags[:-1], seed_flags[1:]) if a != b)
    regime = "robust_single_switch" if transitions == 1 else "ambiguous_or_nonmonotone"
    return AnchorChoice(
        beta=beta,
        anchor_Gamma=float(run[0]),
        run_start=float(run[0]),
        run_end=float(run[-1]),
        run_length=len(run),
        refined_regime=regime,
    )


def intervals_overlap(a: list[float], b: list[float]) -> bool:
    lo1, hi1 = sorted(map(float, a))
    lo2, hi2 = sorted(map(float, b))
    return max(lo1, lo2) <= min(hi1, hi2) + 1e-12


def run_one(choice: AnchorChoice, args: argparse.Namespace) -> dict[str, Any]:
    beta_dir = OUT / f"beta_{choice.beta:.6f}".replace(".", "p")
    beta_dir.mkdir(parents=True, exist_ok=True)
    existing = beta_dir / "pl_homotopy_tracker.json"
    if args.reuse_existing and existing.is_file():
        with existing.open() as f:
            result = json.load(f)
        result["json"] = str(existing)
        result["status"] = "ok"
        return result
    old_out = tracker.OUT
    tracker.OUT = beta_dir
    try:
        Gammas = tracker.default_Gamma_grid()
        for G in [choice.anchor_Gamma, choice.run_start, choice.run_end]:
            if G not in Gammas:
                Gammas.append(G)
        gammas = sorted([-float(G) for G in sorted(set(Gammas))], reverse=True)
        anchor_gamma = -choice.anchor_Gamma

        print(
            f"beta={choice.beta:.6f}: anchor Gamma={choice.anchor_Gamma:.6f} "
            f"run=[{choice.run_start:.3f},{choice.run_end:.3f}] len={choice.run_length}",
            flush=True,
        )
        anchor = tracker.choose_anchor_competitor(choice.beta, anchor_gamma, args.starts)
        branch = tracker.continue_branch(choice.beta, anchor_gamma, anchor["z"], gammas)
        branch[anchor_gamma] = {
            **branch[anchor_gamma],
            **{k: v for k, v in anchor.items() if k != "z"},
            "certified": True,
            "failed": False,
            "residual_inf": 0.0,
            "gap_abs": anchor["exact_gap_abs"],
        }
        rows = tracker.build_rows(
            choice.beta,
            gammas,
            branch,
            stokes_tol=args.stokes_tol,
            anti_tol=args.anti_tol,
        )
        crossings = tracker.estimate_crossings(rows)
        figures = tracker.plot(rows, crossings)
        result = {
            "metadata": {
                "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "caveat": "Necessary-condition homotopy tracker, not rigorous PL intersections.",
            },
            "beta": choice.beta,
            "anchor_Gamma": choice.anchor_Gamma,
            "anchor_gamma": anchor_gamma,
            "anchor_run_start": choice.run_start,
            "anchor_run_end": choice.run_end,
            "anchor_run_length": choice.run_length,
            "refined_regime": choice.refined_regime,
            "starts": args.starts,
            "stokes_tol": args.stokes_tol,
            "anti_tol": args.anti_tol,
            "crossings": crossings,
            "rows": rows,
            "figures": figures,
        }
        report = tracker.write_report(result)
        result["report"] = report
        out_json = beta_dir / "pl_homotopy_tracker.json"
        with out_json.open("w") as f:
            json.dump(result, f, indent=2)
        result["json"] = str(out_json)
        result["status"] = "ok"
        return result
    except Exception as exc:
        result = {
            "beta": choice.beta,
            "anchor_Gamma": choice.anchor_Gamma,
            "refined_regime": choice.refined_regime,
            "status": "failed",
            "error": repr(exc),
        }
        with (beta_dir / "failure.json").open("w") as f:
            json.dump(result, f, indent=2)
        print(f"  FAILED beta={choice.beta:.6f}: {exc}", flush=True)
        return result
    finally:
        tracker.OUT = old_out


def summarize_result(result: dict[str, Any]) -> dict[str, Any]:
    if result.get("status") != "ok":
        return {
            "beta": result["beta"],
            "status": result["status"],
            "refined_regime": result.get("refined_regime", ""),
            "anchor_Gamma": result.get("anchor_Gamma", math.nan),
            "pl_aligned": False,
            "reason": result.get("error", ""),
        }
    crossings = result["crossings"]
    anti = crossings["anti_stokes_intervals"]
    phys = crossings["physical_switch_intervals"]
    overlap = any(intervals_overlap(a, p) for a in anti for p in phys)
    stokes_near_overlap = False
    overlap_points: list[float] = []
    for r in result["rows"]:
        if r.get("competitor_failed"):
            continue
        G = float(r["Gamma"])
        if r.get("candidate_pl_jump_window"):
            for p in phys:
                lo, hi = sorted(map(float, p))
                if lo - 0.051 <= G <= hi + 0.051:
                    stokes_near_overlap = True
                    overlap_points.append(G)
    valid = [r for r in result["rows"] if not r.get("competitor_failed")]
    right_censored_supportive = False
    if not overlap and valid:
        first = min(valid, key=lambda r: float(r["Gamma"]))
        right_censored_supportive = bool(
            result["refined_regime"] == "robust_single_switch"
            and first.get("competitor_best_exact")
            and first.get("candidate_pl_jump_window")
            and sum(1 for r in result["rows"] if r.get("competitor_merged_with_seed")) > 0
        )
    pl_aligned = bool((overlap and stokes_near_overlap) or right_censored_supportive)
    return {
        "beta": float(result["beta"]),
        "status": "ok",
        "refined_regime": result["refined_regime"],
        "anchor_Gamma": float(result["anchor_Gamma"]),
        "anchor_run": [result["anchor_run_start"], result["anchor_run_end"]],
        "anti_stokes_intervals": anti,
        "physical_switch_intervals": phys,
        "candidate_pl_jump_points": crossings["candidate_pl_jump_points"],
        "pl_aligned": pl_aligned,
        "right_censored_supportive": right_censored_supportive,
        "overlap_stokes_points": overlap_points,
        "n_merged_small_gamma": sum(1 for r in result["rows"] if r.get("competitor_merged_with_seed")),
    }


def make_summary_figure(summaries: list[dict[str, Any]]) -> list[str]:
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 8.2), sharex=True)
    for s in summaries:
        if s["status"] == "ok":
            color = GREEN if s["pl_aligned"] else (RED if s["refined_regime"] != "robust_single_switch" else ORANGE)
            marker = "o" if s["pl_aligned"] else "x"
            phys = s["physical_switch_intervals"]
            anti = s["anti_stokes_intervals"]
            for lo, hi in phys:
                axes[0].plot([s["beta"], s["beta"]], [lo, hi], color=BLUE, lw=5, alpha=0.35)
            for lo, hi in anti:
                axes[0].plot([s["beta"], s["beta"]], [lo, hi], color=ORANGE, lw=2.2, alpha=0.9)
            y = np.mean(phys[0]) if phys else s["anchor_Gamma"]
            axes[0].scatter(s["beta"], y, marker=marker, s=70, color=color, zorder=4)
        else:
            axes[0].scatter(s["beta"], s.get("anchor_Gamma", np.nan), marker="x", s=80, color=RED, linewidths=2.2, zorder=4)

    axes[0].set_ylabel(r"$\Gamma=-\gamma$")
    axes[0].set_title("All-slice PL homotopy: physical switches vs anti-Stokes brackets")
    axes[0].grid(True, alpha=0.25)
    axes[0].plot([], [], color=BLUE, lw=5, alpha=0.35, label="exact-switch bracket")
    axes[0].plot([], [], color=ORANGE, lw=2.2, label="anti-Stokes bracket")
    axes[0].scatter([], [], color=GREEN, s=70, label="PL-aligned")
    axes[0].scatter([], [], color=RED, marker="x", s=70, label="not clean / ambiguous")
    axes[0].legend(loc="best", fontsize=9)

    labels = []
    scores = []
    colors = []
    for s in summaries:
        labels.append(f"{s['beta']:.3f}")
        score = 1 if s["pl_aligned"] else 0
        scores.append(score)
        colors.append(GREEN if score else RED)
    axes[1].bar(np.arange(len(summaries)), scores, color=colors, alpha=0.85)
    axes[1].set_xticks(np.arange(len(summaries)))
    axes[1].set_xticklabels(labels)
    axes[1].set_yticks([0, 1])
    axes[1].set_yticklabels(["not clean", "aligned"])
    axes[1].set_xlabel(r"$\beta$")
    axes[1].set_title("Does the same-branch PL necessary condition align with exact switch?")
    axes[1].grid(True, axis="y", alpha=0.25)

    fig.tight_layout()
    png = OUT / "pl_homotopy_all_slices_summary.png"
    pdf = OUT / "pl_homotopy_all_slices_summary.pdf"
    fig.savefig(png, dpi=190, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return [str(png), str(pdf)]


def write_report(summaries: list[dict[str, Any]], figures: list[str]) -> str:
    lines = [
        "# PL homotopy diagnostics across refined beta slices",
        "",
        "This batch applies the beta-opt PL homotopy diagnostic to every refined beta slice.  Each slice uses the longest contiguous competitor-best run as its anchor branch.  The result is still a necessary-condition PL diagnostic, not rigorous intersection numbers.",
        "",
        "## Summary",
        "",
        "| beta | refined regime | anchor Gamma | anti-Stokes brackets | exact-switch brackets | PL aligned? | notes |",
        "|---:|---|---:|---|---|---|---|",
    ]
    for s in summaries:
        notes = ""
        if s["status"] != "ok":
            notes = s.get("reason", "failed")
        elif s.get("right_censored_supportive"):
            notes = f"right-censored: {s['n_merged_small_gamma']} pre-onset branch-merge points filtered"
        elif s.get("n_merged_small_gamma", 0):
            notes = f"{s['n_merged_small_gamma']} small-Gamma branch-merge points filtered"
        lines.append(
            f"| {s['beta']:.6f} | {s.get('refined_regime','')} | {s.get('anchor_Gamma', math.nan):.6f} | "
            f"{s.get('anti_stokes_intervals', [])} | {s.get('physical_switch_intervals', [])} | "
            f"{'yes' if s.get('pl_aligned') else 'no'} | {notes} |"
        )
    lines += [
        "",
        "## Figures",
        "",
    ]
    lines += [f"- `{Path(p).name}`" for p in figures]
    lines.append("")
    path = OUT / "pl_homotopy_all_slices_report.md"
    path.write_text("\n".join(lines))
    return str(path)


def run(args: argparse.Namespace) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    refined = load_json(REFINED)
    rows = [r for r in refined["rows"] if r.get("status") == "ok"]
    by_beta: dict[float, list[dict[str, Any]]] = {}
    for r in rows:
        by_beta.setdefault(round(float(r["beta"]), 10), []).append(r)
    choices = [choose_anchor(v) for _, v in sorted(by_beta.items())]
    if args.limit:
        choices = choices[: args.limit]

    results = []
    for i, choice in enumerate(choices, start=1):
        print(f"\n=== slice {i}/{len(choices)} ===", flush=True)
        results.append(run_one(choice, args))

    summaries = [summarize_result(r) for r in results]
    figures = make_summary_figure(summaries)
    report = write_report(summaries, figures)
    combined = {
        "metadata": {
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "starts": args.starts,
            "stokes_tol": args.stokes_tol,
            "anti_tol": args.anti_tol,
        },
        "summaries": summaries,
        "results": results,
        "figures": figures,
        "report": report,
    }
    out_json = OUT / "pl_homotopy_all_slices.json"
    with out_json.open("w") as f:
        json.dump(combined, f, indent=2)
    print("\nWrote:")
    print(f"  {out_json}")
    print(f"  {report}")
    for fig in figures:
        print(f"  {fig}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--starts", type=int, default=80)
    parser.add_argument("--stokes-tol", type=float, default=0.08)
    parser.add_argument("--anti-tol", type=float, default=0.02)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--reuse-existing", action="store_true")
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
