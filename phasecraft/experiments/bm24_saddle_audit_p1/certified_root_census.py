"""Certified root census for BM24 p=1 saddle candidates.

This is a stricter wrapper around the existing competitor diagnostic.  It
repeats the certified root search at increasing random-start budgets and asks:

* do certified root counts saturate?
* are the high-Re decoys stable as the search budget grows?
* which certified root best matches the exact finite-n exponent?

This is still a numerical root census, not a Picard-Lefschetz contour proof.
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

from phasecraft.bm24_saddle_audit_p1.competitor_dominance_diagnostic import (
    DEFAULT_CONTINUATION,
    DEFAULT_PRIOR_COMPETITORS,
    _load_continuation,
    _prior_root_z_list,
    interpolate_seed_z,
    run_gamma_diagnostic,
)
from phasecraft.bm24_saddle_audit_p1.continue_seed_branch_certified import (
    lambda_abs_n_max,
)


OUT = HERE / "results" / "certified_root_census"
RAW = OUT / "certified_root_census.json"

Q = 3
K_CLAUSE = 8
R = 176.54
BETA_OPT = 0.5433996420760803
N_VALUES = list(range(18, 25))


def z_of(entry: dict[str, Any]) -> np.ndarray:
    return np.asarray(entry["z_real"], dtype=float) + 1j * np.asarray(entry["z_imag"], dtype=float)


def count_new_roots(
    reference: list[dict[str, Any]],
    previous: list[dict[str, Any]],
    *,
    tol: float,
) -> int:
    prev_z = [z_of(r) for r in previous]
    n_new = 0
    for r in reference:
        z = z_of(r)
        if not any(np.linalg.norm(z - pz, ord=np.inf) < tol for pz in prev_z):
            n_new += 1
    return n_new


def best_exact_match(row: dict[str, Any], lambda_abs: float) -> dict[str, Any]:
    candidates = [
        {
            "kind": "seed",
            "full_conv2_exponent": float(row["seed_full_conv2_exponent"]),
            "gap_to_lambda_abs": abs(float(row["seed_full_conv2_exponent"]) - lambda_abs),
            "signed_re_phi_gap_vs_seed": 0.0,
        }
    ]
    for i, comp in enumerate(row["competitors"]):
        candidates.append(
            {
                "kind": "competitor",
                "index": i,
                "full_conv2_exponent": float(comp["full_conv2_exponent"]),
                "gap_to_lambda_abs": abs(float(comp["full_conv2_exponent"]) - lambda_abs),
                "signed_re_phi_gap_vs_seed": float(comp["signed_re_phi_gap_vs_seed"]),
                "re_phi_m": float(comp["re_phi_m"]),
                "im_phi_m": float(comp["im_phi_m"]),
                "z_distance_from_seed": float(comp["z_distance_from_seed"]),
            }
        )
    return min(candidates, key=lambda c: float(c["gap_to_lambda_abs"]))


def summarize_gamma(rows_for_gamma: list[dict[str, Any]], *, match_tol: float) -> dict[str, Any]:
    rows = sorted(rows_for_gamma, key=lambda r: int(r["budget"]))
    final = rows[-1]
    prev = rows[-2] if len(rows) >= 2 else None
    gamma = float(final["gamma"])
    lambda_abs = lambda_abs_n_max(Q, K_CLAUSE, R, BETA_OPT, gamma, N_VALUES)
    best = best_exact_match(final, lambda_abs)
    comps = final["competitors"]
    high_re = [c for c in comps if float(c["signed_re_phi_gap_vs_seed"]) > 0.02]
    exact_decoys = [
        c for c in high_re
        if abs(float(c["full_conv2_exponent"]) - lambda_abs) > float(best["gap_to_lambda_abs"]) + match_tol
    ]
    new_vs_prev = None
    if prev is not None:
        new_vs_prev = count_new_roots(final["competitors"], prev["competitors"], tol=1e-5)

    return {
        "gamma": gamma,
        "Gamma": -gamma,
        "lambda_abs": lambda_abs,
        "budgets": [int(r["budget"]) for r in rows],
        "num_certified_competitors_by_budget": [
            int(r["num_certified_competitors"]) for r in rows
        ],
        "max_signed_gap_by_budget": [
            float(r["max_signed_gap_vs_seed"]) for r in rows
        ],
        "final_num_certified_competitors": int(final["num_certified_competitors"]),
        "final_new_roots_vs_previous_budget": new_vs_prev,
        "final_max_signed_gap_vs_seed": float(final["max_signed_gap_vs_seed"]),
        "final_nearest_signed_gap_vs_seed": float(final["nearest_signed_gap_vs_seed"]),
        "best_exact_match": best,
        "n_high_re_competitors": len(high_re),
        "n_high_re_exact_decoys": len(exact_decoys),
        "seed_exact_gap": abs(float(final["seed_full_conv2_exponent"]) - lambda_abs),
        "census_status": "near_saturated" if new_vs_prev == 0 else "not_saturated",
    }


def plot_census(summary: list[dict[str, Any]]) -> list[str]:
    OUT.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []

    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    for s in summary:
        ax.plot(
            s["budgets"],
            s["num_certified_competitors_by_budget"],
            "o-",
            label=rf"$\Gamma={s['Gamma']:.3f}$",
        )
    ax.set_xscale("log")
    ax.set_xlabel("random starts")
    ax.set_ylabel("certified competitor roots found")
    ax.set_title("Certified root-count saturation test")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=True, fontsize=9)
    fig.tight_layout()
    path = OUT / "root_count_saturation.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    paths.append(str(path))

    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    for s in summary:
        ax.plot(
            s["budgets"],
            s["max_signed_gap_by_budget"],
            "o-",
            label=rf"$\Gamma={s['Gamma']:.3f}$",
        )
    ax.set_xscale("log")
    ax.axhline(0.0, color="#4a5568", lw=1.0)
    ax.set_xlabel("random starts")
    ax.set_ylabel(r"max discovered competitor $\Delta\mathrm{Re}\Phi$ vs seed")
    ax.set_title("High-Re challenger stability with search budget")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=True, fontsize=9)
    fig.tight_layout()
    path = OUT / "max_re_gap_saturation.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    paths.append(str(path))

    return paths


def write_report(
    metadata: dict[str, Any],
    summary: list[dict[str, Any]],
    figures: list[str],
) -> str:
    lines = [
        "# Certified root census",
        "",
        "## Scope",
        "",
        "This is a repeated-budget numerical census of certified roots of the BM24 p=1 saddle equations. "
        "Each discovered root is polished and Krawczyk-certified.  The test is strong evidence for local "
        "root-discovery stability, but it is not a Picard-Lefschetz contour/intersection proof and therefore "
        "not a mathematical completeness theorem.",
        "",
        "## Parameters",
        "",
        f"- beta: `{metadata['beta']}`",
        f"- Gamma values: `{metadata['Gamma_values']}`",
        f"- budgets: `{metadata['budgets']}`",
        f"- n values for exact exponent: `{metadata['n_values']}`",
        "",
        "## Results",
        "",
        "| Gamma | certified competitors by budget | new roots at final budget | max gap vs seed | best exact match | seed exact gap | best exact gap | high-Re decoys | status |",
        "|---:|---:|---:|---:|---|---:|---:|---:|---|",
    ]
    for s in summary:
        best = s["best_exact_match"]
        new = s["final_new_roots_vs_previous_budget"]
        new_s = "-" if new is None else str(new)
        lines.append(
            "| "
            f"{s['Gamma']:.6f} | "
            f"{s['num_certified_competitors_by_budget']} | "
            f"{new_s} | "
            f"{s['final_max_signed_gap_vs_seed']:+.6f} | "
            f"{best['kind']} | "
            f"{s['seed_exact_gap']:.6g} | "
            f"{float(best['gap_to_lambda_abs']):.6g} | "
            f"{s['n_high_re_exact_decoys']} / {s['n_high_re_competitors']} | "
            f"{s['census_status']} |"
        )

    lines += [
        "",
        "## Interpretation",
        "",
        "The important column is not the largest discovered real action by itself.  The exact-exponent match "
        "separates physical candidates from high-Re decoys.  If root counts and top gaps stabilize with "
        "budget, the census supports the claim that the observed seed/competitor transition is not a random-start artifact.",
        "",
        "## Figures",
        "",
    ]
    lines += [f"- `{Path(p).name}`" for p in figures]
    lines.append("")

    path = OUT / "certified_root_census_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return str(path)


def parse_csv_floats(value: str) -> list[float]:
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def parse_csv_ints(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gammas", type=parse_csv_floats, default="-1.75,-1.825,-1.85,-2.0")
    p.add_argument("--budgets", type=parse_csv_ints, default="300,800,1600")
    p.add_argument("--seed", type=int, default=1729)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--match-tol", type=float, default=0.0)
    p.add_argument("--dps", type=int, default=80)
    args = p.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    cached: dict[str, dict[str, Any]] = {}
    if args.resume and RAW.exists():
        old = json.loads(RAW.read_text(encoding="utf-8"))
        for row in old.get("runs", []):
            cached[f"{float(row['gamma']):.10f}|{int(row['budget'])}"] = row

    cont_rows = _load_continuation(Path(DEFAULT_CONTINUATION))
    runs: list[dict[str, Any]] = []
    for gamma in args.gammas:
        z_init = interpolate_seed_z(cont_rows, gamma)
        prior_z = _prior_root_z_list(Path(DEFAULT_PRIOR_COMPETITORS), gamma)
        for budget in args.budgets:
            key = f"{float(gamma):.10f}|{int(budget)}"
            if key in cached:
                row = cached[key]
                print(
                    f"gamma={gamma:.6f} budget={budget}: cached "
                    f"cert_comp={row['num_certified_competitors']}",
                    flush=True,
                )
                runs.append(row)
                continue
            print(f"gamma={gamma:.6f} budget={budget}: running", flush=True)
            row = run_gamma_diagnostic(
                q=Q,
                K_clause=K_CLAUSE,
                r=R,
                beta=BETA_OPT,
                gamma=gamma,
                seed_z_init=z_init,
                prior_z=prior_z,
                num_random_starts=int(budget),
                seed=int(args.seed + 100000 * abs(gamma) + budget),
                seed_branch_tol=1e-6,
                dedup_tol=1e-5,
                min_residual=1e-10,
                dps=args.dps,
            )
            row["budget"] = int(budget)
            runs.append(row)
            RAW.write_text(
                json.dumps(
                    {
                        "metadata": {
                            "beta": BETA_OPT,
                            "gammas": args.gammas,
                            "Gamma_values": [-g for g in args.gammas],
                            "budgets": args.budgets,
                            "n_values": N_VALUES,
                            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                        },
                        "runs": runs,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            print(
                f"  roots={row['num_roots_found']} certified_comp={row['num_certified_competitors']} "
                f"max_gap={row['max_signed_gap_vs_seed']:+.4f}",
                flush=True,
            )

    metadata = {
        "beta": BETA_OPT,
        "gammas": args.gammas,
        "Gamma_values": [-g for g in args.gammas],
        "budgets": args.budgets,
        "n_values": N_VALUES,
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    grouped = []
    for gamma in args.gammas:
        rows_g = [r for r in runs if math.isclose(float(r["gamma"]), float(gamma))]
        grouped.append(summarize_gamma(rows_g, match_tol=args.match_tol))
    figures = plot_census(grouped)
    report = write_report(metadata, grouped, figures)
    summary = {
        "metadata": metadata,
        "gamma_summary": grouped,
        "figures": figures,
        "report": report,
    }
    (OUT / "certified_root_census_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    RAW.write_text(json.dumps({"metadata": metadata, "runs": runs}, indent=2), encoding="utf-8")

    print("\nWrote:")
    print(RAW)
    print(OUT / "certified_root_census_summary.json")
    print(report)
    for fig in figures:
        print(fig)


if __name__ == "__main__":
    main()
