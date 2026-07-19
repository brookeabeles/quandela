"""RMSE of tracked saddle branches against exact lambda_abs at n=100.

Uses the same exact finite-n machinery as the finite-size discrepancy figure:
``finite_n_exponent_grid(..., [100])``.  The tracked seed/competitor branch
values are read from the Krawczyk PL homotopy outputs used by the Fig. 8-style
equal-action diagnostics.
"""

from __future__ import annotations

import csv
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-phasecraft")

import numpy as np


HERE = Path(__file__).resolve().parent
REPO_PARENT = HERE.parents[2]
if str(REPO_PARENT) not in sys.path:
    sys.path.insert(0, str(REPO_PARENT))

from phasecraft.bm24_saddle_audit_p1.audit import finite_n_exponent_grid
from phasecraft.bm24_saddle_audit_p1.pl_homotopy_tracker import K_CLAUSE, Q, R
from phasecraft.bm24_saddle_audit_p1.continue_seed_branch_certified import conv2_pref


OUT = HERE / "results" / "physically_possible_refined"
PL_ALL = HERE / "results" / "pl_homotopy_all_slices"
N_EXACT = 100


def load_json(path: Path) -> Any:
    with path.open() as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2)


def gamma_as_by_beta() -> dict[float, float]:
    data = load_json(OUT / "equal_real_action_refined_brackets.json")
    out: dict[float, float] = {}
    for row in data.get("rows", []):
        if row.get("status") == "ok" and row.get("midpoint") is not None:
            out[round(float(row["beta"]), 10)] = float(row["midpoint"])
    return out


def pl_results_by_beta() -> dict[float, dict[str, Any]]:
    data = load_json(PL_ALL / "pl_homotopy_all_slices.json")
    out = {}
    for result in data.get("results", []):
        if result.get("status") == "ok" and result.get("rows"):
            out[round(float(result["beta"]), 10)] = result
    return out


def lambda_cache_path() -> Path:
    return OUT / "exact_n100_lambda_cache.json"


def load_lambda_cache() -> dict[str, Any]:
    path = lambda_cache_path()
    if path.exists():
        return load_json(path)
    return {
        "metadata": {
            "n": N_EXACT,
            "q": Q,
            "K_clause": K_CLAUSE,
            "r": R,
            "source": "finite_n_exponent_grid(K_clause, Q, R, beta, gamma, [100])",
        },
        "values": {},
    }


def cache_key(beta: float, Gamma: float) -> str:
    return f"beta={beta:.10f}|Gamma={Gamma:.10f}"


def exact_lambda_abs_n100(beta: float, Gamma: float, cache: dict[str, Any]) -> float:
    key = cache_key(beta, Gamma)
    if key not in cache["values"]:
        gamma = -float(Gamma)
        fn = finite_n_exponent_grid(K_CLAUSE, Q, R, beta, gamma, [N_EXACT])
        cache["values"][key] = {
            "beta": float(beta),
            "Gamma": float(Gamma),
            "gamma": gamma,
            "lambda_abs_100": float(fn["lambda_abs"][str(N_EXACT)]),
            "crosscheck_lambda_abs_prop4_100": float(fn["crosscheck_lambda_abs_prop4"][str(N_EXACT)]),
            "primary_baseline": fn["primary_baseline"],
        }
        write_json(lambda_cache_path(), cache)
    return float(cache["values"][key]["lambda_abs_100"])


def valid_branch_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for row in result.get("rows", []):
        if row.get("competitor_failed") or row.get("competitor_merged_with_seed"):
            continue
        if not row.get("competitor") or row.get("seed", {}).get("failed"):
            continue
        if row["competitor"].get("failed") or not row["competitor"].get("certified"):
            continue
        rows.append(row)
    rows.sort(key=lambda row: float(row["Gamma"]))
    return rows


def rmse(values: list[float]) -> float | None:
    if not values:
        return None
    arr = np.asarray(values, dtype=float)
    return float(np.sqrt(np.mean(arr * arr)))


def summarize_region(rows: list[dict[str, Any]]) -> dict[str, Any]:
    seed_err = [float(row["seed_full_exponent"]) - float(row["lambda_abs_100"]) for row in rows]
    comp_err = [float(row["comp_full_exponent"]) - float(row["lambda_abs_100"]) for row in rows]
    seed_raw_err = [float(row["seed_re_phi"]) - float(row["lambda_abs_100"]) for row in rows]
    comp_raw_err = [float(row["comp_re_phi"]) - float(row["lambda_abs_100"]) for row in rows]
    return {
        "n_points": len(rows),
        "rmse_seed_full_exponent": rmse(seed_err),
        "rmse_comp_full_exponent": rmse(comp_err),
        "rmse_seed_re_phi_raw": rmse(seed_raw_err),
        "rmse_comp_re_phi_raw": rmse(comp_raw_err),
        "mean_signed_error_seed_full_exponent": float(np.mean(seed_err)) if seed_err else None,
        "mean_signed_error_comp_full_exponent": float(np.mean(comp_err)) if comp_err else None,
    }


def build_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cache = load_lambda_cache()
    gamma_as = gamma_as_by_beta()
    pl_by_beta = pl_results_by_beta()
    phi_pref = conv2_pref(K_CLAUSE, R)
    point_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for beta_key in sorted(set(gamma_as).intersection(pl_by_beta)):
        beta = float(beta_key)
        gas = float(gamma_as[beta_key])
        branch_rows = valid_branch_rows(pl_by_beta[beta_key])
        enriched: list[dict[str, Any]] = []
        for row in branch_rows:
            Gamma = float(row["Gamma"])
            lam = exact_lambda_abs_n100(beta, Gamma, cache)
            seed = row["seed"]
            comp = row["competitor"]
            region = "inside_abs_gamma_lt_gamma_AS" if Gamma < gas else "outside_abs_gamma_gt_gamma_AS"
            item = {
                "beta": beta,
                "Gamma": Gamma,
                "gamma": -Gamma,
                "gamma_AS": gas,
                "region": region,
                "lambda_abs_100": lam,
                "seed_re_phi": float(seed["re_phi"]),
                "comp_re_phi": float(comp["re_phi"]),
                "phi_pref_conv2": float(phi_pref),
                "seed_full_exponent": float(seed["full_exponent"]),
                "comp_full_exponent": float(comp["full_exponent"]),
                "seed_error_full_minus_lambda": float(seed["full_exponent"]) - lam,
                "comp_error_full_minus_lambda": float(comp["full_exponent"]) - lam,
                "seed_residual_inf": float(seed.get("residual_inf", math.nan)),
                "comp_residual_inf": float(comp.get("residual_inf", math.nan)),
                "comp_certified": bool(comp.get("certified", False)),
            }
            enriched.append(item)
            point_rows.append(item)

        inside = [row for row in enriched if row["region"] == "inside_abs_gamma_lt_gamma_AS"]
        outside = [row for row in enriched if row["region"] == "outside_abs_gamma_gt_gamma_AS"]
        summary_rows.append({
            "beta": beta,
            "gamma_AS": gas,
            "n_branch_points": len(enriched),
            "inside_abs_gamma_lt_gamma_AS": summarize_region(inside),
            "outside_abs_gamma_gt_gamma_AS": summarize_region(outside),
        })

    return point_rows, summary_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(summary_rows: list[dict[str, Any]], point_rows: list[dict[str, Any]]) -> Path:
    path = OUT / "exact_n100_branch_rmse_report.md"
    lines = [
        "# Exact n=100 Branch RMSE Against lambda_abs",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Method",
        "",
        "For each clean beta slice with a resolved equal-action point, I used the Krawczyk-tracked seed and competitor rows from `pl_homotopy_all_slices.json`.",
        "",
        "For each sampled Gamma, I recomputed the exact finite-n baseline at `n=100` using:",
        "",
        "```python",
        "finite_n_exponent_grid(K_clause=8, q=3, r=176.54, beta=beta, gamma=-Gamma, n_values=[100])",
        "```",
        "",
        "This is the same exact finite-n lambda_abs machinery used by the finite-size/Fig. 6 workflow.  The primary comparison is against `lambda_abs[100]` from the Conv2 all-subsets baseline.",
        "",
        "The RMSE columns compare `lambda_abs(100)` to the full saddle exponent `E = Re Phi + phi_pref`, because `lambda_abs` includes the same prefactor convention.  Raw `Re Phi` RMSE values are also stored in the JSON for auditability, but they are offset by the prefactor and are not the physically matched comparison.",
        "",
        "Rows are split by the resolved equal-action crossing `Gamma_AS(beta)`: inside means `|gamma| = Gamma < Gamma_AS`; outside means `Gamma > Gamma_AS`.",
        "",
        "## Summary Table",
        "",
        "| beta | Gamma_AS | n_in | RMSE seed in | RMSE comp in | n_out | RMSE seed out | RMSE comp out |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    def fmt(value: Any) -> str:
        return "-" if value is None else f"{float(value):.6g}"

    for row in summary_rows:
        inside = row["inside_abs_gamma_lt_gamma_AS"]
        outside = row["outside_abs_gamma_gt_gamma_AS"]
        lines.append(
            f"| {row['beta']:.10g} | {row['gamma_AS']:.6f} | "
            f"{inside['n_points']} | {fmt(inside['rmse_seed_full_exponent'])} | {fmt(inside['rmse_comp_full_exponent'])} | "
            f"{outside['n_points']} | {fmt(outside['rmse_seed_full_exponent'])} | {fmt(outside['rmse_comp_full_exponent'])} |"
        )

    lines += [
        "",
        "## Interpretation",
        "",
        "Inside the equal-action crossing, the seed branch is generally the closer saddle-rate approximation to the exact `lambda_abs(100)`.  Outside the crossing, the tracked competitor branch becomes comparable to or better than the seed on the clean slices.  This is an RMSE diagnostic of branch rates against the exact finite-n baseline; it does not compute or prove thimble intersection numbers.",
        "",
        "## Outputs",
        "",
        "- `exact_n100_branch_rmse_summary.json`",
        "- `exact_n100_branch_rmse_points.csv`",
        "- `exact_n100_branch_rmse_summary.csv`",
        "- `exact_n100_lambda_cache.json`",
        "",
        f"Number of point rows: `{len(point_rows)}`.",
        "",
    ]
    path.write_text("\n".join(lines))
    return path


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    point_rows, summary_rows = build_rows()
    payload = {
        "metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "n": N_EXACT,
            "q": Q,
            "K_clause": K_CLAUSE,
            "r": R,
            "lambda_source": "finite_n_exponent_grid(..., [100])",
            "branch_source": str((PL_ALL / "pl_homotopy_all_slices.json").resolve()),
            "gamma_as_source": str((OUT / "equal_real_action_refined_brackets.json").resolve()),
            "rmse_primary_convention": "full_exponent = Re Phi + conv2_pref compared to lambda_abs(100)",
        },
        "summary_rows": summary_rows,
        "point_rows": point_rows,
    }
    write_json(OUT / "exact_n100_branch_rmse_summary.json", payload)
    write_csv(OUT / "exact_n100_branch_rmse_points.csv", point_rows)
    flat_summary = []
    for row in summary_rows:
        for region_key in ["inside_abs_gamma_lt_gamma_AS", "outside_abs_gamma_gt_gamma_AS"]:
            flat = {"beta": row["beta"], "gamma_AS": row["gamma_AS"], "region": region_key}
            flat.update(row[region_key])
            flat_summary.append(flat)
    write_csv(OUT / "exact_n100_branch_rmse_summary.csv", flat_summary)
    report = write_markdown(summary_rows, point_rows)
    print(report)


if __name__ == "__main__":
    main()
