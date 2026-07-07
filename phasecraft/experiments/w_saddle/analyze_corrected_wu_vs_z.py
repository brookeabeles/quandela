#!/usr/bin/env python3
"""Comparison-grade audit for the corrected BM24 q=3 (u,w) saddle run.

This script deliberately separates three notions that were easy to blur in the
exploratory runs:

1. certified local algebraic saddles in the corrected integer-power (u,w) chart;
2. archived z-chart algebraic saddles, where Conv2 finite-n normalization is known;
3. physical plausibility screens.  For (u,w) the latter is only a proxy until the
   parent-action normalization and contour/thimble contribution are derived.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from phasecraft.bm24_saddle_audit_p1.audit import finite_n_exponent_grid


R = 176.54
Q = 3
K_CLAUSE = 8
BETA = 0.5433996420760803
PHI_PREF_CONV2 = -(R / (2**K_CLAUSE))


def _load_json(path: Path) -> Any:
    with path.open() as f:
        return json.load(f)


def _isfinite(x: Any) -> bool:
    try:
        return math.isfinite(float(x))
    except Exception:
        return False


def _interp(xs: list[float], ys: list[float], x: float) -> float:
    order = np.argsort(xs)
    x_arr = np.asarray(xs, dtype=float)[order]
    y_arr = np.asarray(ys, dtype=float)[order]
    return float(np.interp(float(x), x_arr, y_arr))


def _points_by_gamma(payload: dict[str, Any]) -> dict[float, dict[str, Any]]:
    return {round(float(p["gamma"]), 12): p for p in payload["points"]}


def _row_series(payload: dict[str, Any], lo: float, hi: float) -> list[dict[str, Any]]:
    out = []
    for row in payload["table"]:
        gamma = float(row["gamma"])
        if lo <= gamma <= hi:
            out.append(row)
    return sorted(out, key=lambda r: float(r["gamma"]))


def _competitor_re_values(point: dict[str, Any]) -> list[float]:
    sweep = point.get("competitor_sweep") or {}
    vals = []
    for comp in sweep.get("competitors") or []:
        re_phi = comp.get("Phi_eff_real")
        if _isfinite(re_phi):
            vals.append(float(re_phi))
    return vals


def _all_certified_re_values(point: dict[str, Any]) -> list[float]:
    sweep = point.get("competitor_sweep") or {}
    vals = []
    for root in sweep.get("certified_roots") or []:
        re_phi = root.get("Phi_eff_real")
        if _isfinite(re_phi):
            vals.append(float(re_phi))
    seed = point.get("Phi_eff_real")
    if _isfinite(seed):
        vals.append(float(seed))
    return vals


def _branch_gap_stats(resolved: dict[str, Any]) -> dict[str, Any]:
    branches = resolved.get("branches", [])
    n_points = [int(b.get("num_points", len(b.get("points", [])))) for b in branches]
    danger = 0
    nonseed = 0
    max_re = -math.inf
    min_re = math.inf
    max_delta = -math.inf
    min_delta = math.inf
    for branch in branches:
        if branch.get("is_seed_sheet"):
            continue
        nonseed += 1
        deltas = []
        for pt in branch.get("points", []):
            re_phi = pt.get("Phi_eff_real")
            delta = pt.get("DeltaRe_vs_seed")
            if _isfinite(re_phi):
                max_re = max(max_re, float(re_phi))
                min_re = min(min_re, float(re_phi))
            if _isfinite(delta):
                deltas.append(float(delta))
        if deltas:
            max_delta = max(max_delta, max(deltas))
            min_delta = min(min_delta, min(deltas))
            # DeltaRe_vs_seed is seed - competitor in these archives; negative
            # means this branch beats the seed in raw Re(Phi).
            if min(deltas) < 0:
                danger += 1
    return {
        "num_branches": int(resolved.get("num_branches", len(branches))),
        "num_nonseed_branches": nonseed,
        "num_danger_branches_seed_minus_comp_negative": danger,
        "branch_point_count_min": int(min(n_points)) if n_points else 0,
        "branch_point_count_max": int(max(n_points)) if n_points else 0,
        "branch_point_count_mean": float(np.mean(n_points)) if n_points else float("nan"),
        "nonseed_re_min": min_re,
        "nonseed_re_max": max_re,
        "delta_seed_minus_comp_min": min_delta,
        "delta_seed_minus_comp_max": max_delta,
        "mesh_delta_re_sign_changes": len(resolved.get("delta_re_crossing_intervals_mesh") or []),
        "crossing_certificates": len(resolved.get("crossing_certificates") or []),
    }


def _wu_crossing_proxy_stats(resolved: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for cert in resolved.get("crossing_certificates") or []:
        comp_iv = cert.get("action_interval_comp", {}).get("RePhi_interval")
        seed_iv = cert.get("action_interval_seed", {}).get("RePhi_interval")
        if not comp_iv or not seed_iv:
            continue
        comp_re = 0.5 * (float(comp_iv[0]) + float(comp_iv[1]))
        seed_re = 0.5 * (float(seed_iv[0]) + float(seed_iv[1]))
        gamma_interval = cert.get("gamma_interval") or [cert.get("gamma_left"), cert.get("gamma_right")]
        gamma_mid = 0.5 * (float(gamma_interval[0]) + float(gamma_interval[1]))
        rows.append(
            {
                "branch_id": cert.get("branch_id"),
                "gamma_mid": gamma_mid,
                "Gamma_mid": -gamma_mid,
                "gamma_interval": gamma_interval,
                "competitor_re": comp_re,
                "seed_re": seed_re,
                "naive_prefactor_possible": bool(PHI_PREF_CONV2 + comp_re <= 0),
                "box_disjoint": bool(cert.get("box_disjoint")),
            }
        )
    possible = [r for r in rows if r["naive_prefactor_possible"]]
    impossible = [r for r in rows if not r["naive_prefactor_possible"]]
    return {
        "total_refined_re_gap_zero_brackets": len(rows),
        "naive_prefactor_possible_crossings": len(possible),
        "naive_prefactor_impossible_crossings": len(impossible),
        "box_disjoint_crossings": sum(1 for r in rows if r["box_disjoint"]),
        "possible_Gamma_range": [min((r["Gamma_mid"] for r in possible), default=None), max((r["Gamma_mid"] for r in possible), default=None)],
        "impossible_Gamma_range": [min((r["Gamma_mid"] for r in impossible), default=None), max((r["Gamma_mid"] for r in impossible), default=None)],
        "rows": rows,
    }


def _summarize_system(name: str, comp: dict[str, Any], resolved: dict[str, Any], lo: float, hi: float) -> dict[str, Any]:
    rows = _row_series(comp, lo, hi)
    point_map = _points_by_gamma(comp)
    total_comp = 0
    proxy_possible = 0
    proxy_impossible = 0
    raw_positive = 0
    max_comp_re = -math.inf
    all_comp_re = []
    best_gap_seed_minus_comp = []
    for row in rows:
        point = point_map.get(round(float(row["gamma"]), 12))
        comp_re = _competitor_re_values(point or {})
        total_comp += len(comp_re)
        for val in comp_re:
            all_comp_re.append(val)
            max_comp_re = max(max_comp_re, val)
            if val > 0:
                raw_positive += 1
            if PHI_PREF_CONV2 + val <= 0:
                proxy_possible += 1
            else:
                proxy_impossible += 1
        gap = row.get("min_certified_gap_re")
        if _isfinite(gap):
            best_gap_seed_minus_comp.append(float(gap))
    cert_roots = [float(r["num_certified_roots"]) for r in rows if _isfinite(r.get("num_certified_roots"))]
    cert_comp = [float(r["num_certified_competitors"]) for r in rows if _isfinite(r.get("num_certified_competitors"))]
    return {
        "name": name,
        "window": [lo, hi],
        "num_gamma_slices": len(rows),
        "certified_roots_min": int(min(cert_roots)) if cert_roots else 0,
        "certified_roots_max": int(max(cert_roots)) if cert_roots else 0,
        "certified_roots_mean": float(np.mean(cert_roots)) if cert_roots else float("nan"),
        "certified_competitors_min": int(min(cert_comp)) if cert_comp else 0,
        "certified_competitors_max": int(max(cert_comp)) if cert_comp else 0,
        "certified_competitors_mean": float(np.mean(cert_comp)) if cert_comp else float("nan"),
        "total_certified_competitor_hits": total_comp,
        "competitor_raw_re_min": float(min(all_comp_re)) if all_comp_re else None,
        "competitor_raw_re_max": float(max(all_comp_re)) if all_comp_re else None,
        "competitor_raw_re_mean": float(np.mean(all_comp_re)) if all_comp_re else None,
        "competitor_raw_re_positive_hits": raw_positive,
        "naive_prefactor_proxy_possible_hits": proxy_possible,
        "naive_prefactor_proxy_impossible_hits": proxy_impossible,
        "naive_prefactor_proxy_threshold_re_phi": -PHI_PREF_CONV2,
        "seed_minus_best_comp_gap_min": float(min(best_gap_seed_minus_comp)) if best_gap_seed_minus_comp else None,
        "seed_minus_best_comp_gap_max": float(max(best_gap_seed_minus_comp)) if best_gap_seed_minus_comp else None,
        "branch_stats": _branch_gap_stats(resolved),
    }


def _finite_n_sparse(cache_path: Path, gammas: list[float], n_values: list[int], force: bool) -> dict[str, Any]:
    if cache_path.exists() and not force:
        return _load_json(cache_path)
    rows = []
    for gamma in gammas:
        exact = finite_n_exponent_grid(K_CLAUSE, Q, R, BETA, gamma, n_values)
        row = {"gamma": float(gamma)}
        for n in n_values:
            row[f"lambda_abs_n{n}"] = float(exact["lambda_abs"][str(n)])
        rows.append(row)
        print(f"finite-n exact gamma={gamma:+.3f}: " + ", ".join(f"n={n} {row[f'lambda_abs_n{n}']:+.9f}" for n in n_values))
    payload = {
        "definition": "Exact BM24 Conv2 all-subsets finite-n exponent lambda_abs=(1/n)log|S_n|.",
        "q": Q,
        "K_clause": K_CLAUSE,
        "r": R,
        "beta": BETA,
        "n_values": n_values,
        "rows": rows,
    }
    cache_path.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def _plot_dashboard(out_dir: Path, wu: dict[str, Any], z: dict[str, Any], finite: dict[str, Any]) -> None:
    wu_rows = _row_series(wu["comp"], -1.0, -0.01)
    z_rows = _row_series(z["comp"], -1.0, -0.01)
    wu_points = _points_by_gamma(wu["comp"])
    z_points = _points_by_gamma(z["comp"])

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.5), constrained_layout=True)
    ax = axes[0, 0]
    for label, rows, color in [("(u,w), corrected", wu_rows, "#005f73"), ("z archive", z_rows, "#9b2226")]:
        gamma_abs = [-float(r["gamma"]) for r in rows]
        roots = [float(r["num_certified_roots"]) for r in rows]
        comps = [float(r["num_certified_competitors"]) for r in rows]
        ax.plot(gamma_abs, roots, lw=1.8, color=color, label=f"{label}: roots")
        ax.plot(gamma_abs, comps, lw=1.2, ls="--", color=color, alpha=0.75, label=f"{label}: competitors")
    ax.set_title("Certified local roots in common window")
    ax.set_xlabel(r"$\Gamma=-\gamma$")
    ax.set_ylabel("count per slice")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    ax = axes[0, 1]
    for label, rows, color in [("(u,w), corrected", wu_rows, "#005f73"), ("z archive", z_rows, "#9b2226")]:
        xs, ys = [], []
        for r in rows:
            if _isfinite(r.get("min_certified_gap_re")):
                xs.append(-float(r["gamma"]))
                ys.append(float(r["min_certified_gap_re"]))
        ax.plot(xs, ys, lw=1.8, color=color, label=label)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title("Seed minus best certified competitor")
    ax.set_xlabel(r"$\Gamma=-\gamma$")
    ax.set_ylabel(r"$\Re\Phi_{seed}-\max\Re\Phi_{comp}$")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    ax = axes[1, 0]
    for label, comp, rows, point_map, color in [
        ("(u,w)", wu["comp"], wu_rows, wu_points, "#005f73"),
        ("z", z["comp"], z_rows, z_points, "#9b2226"),
    ]:
        vals = []
        for r in rows:
            vals.extend(_competitor_re_values(point_map.get(round(float(r["gamma"]), 12), {})))
        if vals:
            ax.hist(vals, bins=80, histtype="step", lw=1.6, color=color, label=f"{label} competitors")
    ax.axvline(-PHI_PREF_CONV2, color="#ca6702", ls=":", lw=2, label="naive Conv2 p_succ=1 threshold")
    ax.set_title("Raw competitor action distribution")
    ax.set_xlabel(r"raw $\Re\Phi$")
    ax.set_ylabel("certified competitor hits")
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)

    ax = axes[1, 1]
    z_seed_x = [-float(r["gamma"]) for r in z_rows]
    z_seed_e = [PHI_PREF_CONV2 + float(r["Re_Phi"]) for r in z_rows]
    z_env_e = []
    for r in z_rows:
        p = z_points.get(round(float(r["gamma"]), 12), {})
        vals = _all_certified_re_values(p)
        z_env_e.append(PHI_PREF_CONV2 + max(vals) if vals else PHI_PREF_CONV2 + float(r["Re_Phi"]))
    ax.plot(z_seed_x, z_seed_e, color="#9b2226", lw=1.3, label="z seed exponent")
    ax.plot(z_seed_x, z_env_e, color="#9b2226", lw=2.0, ls="--", label="z certified envelope")
    finite_rows = finite.get("rows", [])
    if finite_rows:
        xs = [-float(r["gamma"]) for r in finite_rows]
        for key, marker in [("lambda_abs_n40", "o"), ("lambda_abs_n100", "s")]:
            if key in finite_rows[0]:
                ax.scatter(xs, [float(r[key]) for r in finite_rows], s=28, marker=marker, label=f"exact {key.replace('lambda_abs_', '')}")
    ax.set_title("Finite-n tracking belongs to the z/Conv2 normalization")
    ax.set_xlabel(r"$\Gamma=-\gamma$")
    ax.set_ylabel(r"Conv2 exponent")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    fig.suptitle("Corrected BM24 q=3 (u,w) vs z saddle audit", fontsize=14)
    fig.savefig(out_dir / "corrected_wu_vs_z_dashboard.png", dpi=220)
    plt.close(fig)


def _plot_wu_proxy(out_dir: Path, wu: dict[str, Any]) -> None:
    rows = _row_series(wu["comp"], -1.0, -0.01)
    point_map = _points_by_gamma(wu["comp"])
    xs = []
    total = []
    proxy_possible = []
    proxy_impossible = []
    best_re = []
    seed_re = []
    for row in rows:
        gamma = float(row["gamma"])
        point = point_map.get(round(gamma, 12), {})
        vals = _competitor_re_values(point)
        xs.append(-gamma)
        total.append(len(vals))
        proxy_possible.append(sum(1 for v in vals if PHI_PREF_CONV2 + v <= 0))
        proxy_impossible.append(sum(1 for v in vals if PHI_PREF_CONV2 + v > 0))
        best_re.append(max(vals) if vals else float("nan"))
        seed_re.append(float(row["Re_Phi"]))

    fig, axes = plt.subplots(2, 1, figsize=(11.5, 8.0), sharex=True, constrained_layout=True)
    ax = axes[0]
    ax.plot(xs, total, color="#001219", lw=1.6, label="all certified competitors")
    ax.plot(xs, proxy_possible, color="#0a9396", lw=1.8, label="naive-prefactor possible")
    ax.plot(xs, proxy_impossible, color="#ae2012", lw=1.8, label="naive-prefactor impossible")
    ax.set_ylabel("count per slice")
    ax.set_title("(u,w) proxy physical screen")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    ax = axes[1]
    ax.plot(xs, seed_re, color="#005f73", lw=1.6, label="seed raw RePhi")
    ax.plot(xs, best_re, color="#ae2012", lw=1.6, label="best competitor raw RePhi")
    ax.axhline(-PHI_PREF_CONV2, color="#ca6702", ls=":", lw=2, label="naive p_succ=1 threshold")
    ax.set_xlabel(r"$\Gamma=-\gamma$")
    ax.set_ylabel(r"raw $\Re\Phi_{uw}$")
    ax.set_ylim(min(min(seed_re), -0.2) - 0.1, min(max(v for v in best_re if math.isfinite(v)), 8.0) + 0.4)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.savefig(out_dir / "corrected_wu_proxy_physical_screen.png", dpi=220)
    plt.close(fig)


def _plot_wu_plausible_tracks(out_dir: Path, wu: dict[str, Any], resolved: dict[str, Any], crossing_stats: dict[str, Any]) -> None:
    rows = _row_series(wu["comp"], -1.0, -0.01)
    seed_x = [-float(r["gamma"]) for r in rows]
    seed_y = [float(r["Re_Phi"]) for r in rows]

    fig, axes = plt.subplots(2, 1, figsize=(12.5, 8.8), sharex=True, constrained_layout=True)
    ax = axes[0]
    ax.plot(seed_x, seed_y, color="#001219", lw=2.0, label="seed raw RePhi")
    ax.axhline(-PHI_PREF_CONV2, color="#ca6702", ls=":", lw=2, label="naive p_succ=1 threshold")

    branch_scores = []
    for branch in resolved.get("branches", []):
        if branch.get("is_seed_sheet"):
            continue
        pts = []
        best_delta = math.inf
        for pt in branch.get("points", []):
            gamma = float(pt["gamma"])
            re_phi = float(pt["Phi_eff_real"])
            if -1.0 <= gamma <= -0.01 and PHI_PREF_CONV2 + re_phi <= 0:
                pts.append((-gamma, re_phi))
                if _isfinite(pt.get("DeltaRe_vs_seed")):
                    best_delta = min(best_delta, float(pt["DeltaRe_vs_seed"]))
        if pts:
            branch_scores.append((best_delta, int(branch["branch_id"]), pts))

    branch_scores.sort(key=lambda x: x[0])
    for best_delta, branch_id, pts in branch_scores:
        pts = sorted(pts)
        x = [p[0] for p in pts]
        y = [p[1] for p in pts]
        if len(pts) >= 2:
            ax.plot(x, y, color="#0a9396", alpha=0.20, lw=1.0)
        else:
            ax.scatter(x, y, color="#0a9396", alpha=0.25, s=8)
    for best_delta, branch_id, pts in branch_scores[:8]:
        pts = sorted(pts)
        ax.plot([p[0] for p in pts], [p[1] for p in pts], lw=2.0, alpha=0.9, label=f"branch {branch_id}" if branch_id else None)
    ax.set_title("(u,w) proxy-plausible branch tracks")
    ax.set_ylabel(r"raw $\Re\Phi_{uw}$")
    ax.set_ylim(-35, 2.0)
    ax.legend(fontsize=8, ncol=3)
    ax.grid(alpha=0.25)

    ax = axes[1]
    poss = [r for r in crossing_stats["rows"] if r["naive_prefactor_possible"]]
    imp = [r for r in crossing_stats["rows"] if not r["naive_prefactor_possible"]]
    bins = np.linspace(0.0, 1.0, 41)
    ax.hist([r["Gamma_mid"] for r in poss], bins=bins, alpha=0.8, color="#0a9396", label="proxy-possible crossings")
    ax.hist([r["Gamma_mid"] for r in imp], bins=bins, alpha=0.65, color="#ae2012", label="proxy-impossible crossings")
    ax.set_title("Corrected (u,w) Re-gap zero brackets by proxy screen")
    ax.set_xlabel(r"$\Gamma=-\gamma$")
    ax.set_ylabel("refined brackets")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.savefig(out_dir / "corrected_wu_plausible_tracks_and_crossings.png", dpi=220)
    plt.close(fig)


def _write_csv(out_dir: Path, wu: dict[str, Any], z: dict[str, Any], finite: dict[str, Any]) -> None:
    finite_by_gamma = {round(float(r["gamma"]), 12): r for r in finite.get("rows", [])}
    with (out_dir / "corrected_wu_vs_z_common_window.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "gamma",
                "Gamma",
                "wu_seed_re",
                "wu_cert_roots",
                "wu_cert_competitors",
                "wu_seed_minus_best_comp",
                "wu_best_comp_re",
                "wu_naive_proxy_possible_comps",
                "wu_naive_proxy_impossible_comps",
                "z_nearest_gamma",
                "z_seed_re",
                "z_seed_conv2_exponent",
                "z_cert_roots",
                "z_cert_competitors",
                "z_seed_minus_best_comp",
                "z_best_comp_re",
                "z_envelope_conv2_exponent",
                "exact_lambda_abs_n40",
                "exact_lambda_abs_n100",
            ],
        )
        writer.writeheader()
        wu_rows = _row_series(wu["comp"], -1.0, -0.01)
        z_rows = _row_series(z["comp"], -1.0, -0.01)
        z_gamma_keys = [round(float(r["gamma"]), 12) for r in z_rows]
        z_rows_by_gamma = {round(float(r["gamma"]), 12): r for r in z_rows}
        wu_points = _points_by_gamma(wu["comp"])
        z_points = _points_by_gamma(z["comp"])
        for wr in sorted(wu_rows, key=lambda r: float(r["gamma"]), reverse=True):
            gamma_key = round(float(wr["gamma"]), 12)
            nearest_z_key = min(z_gamma_keys, key=lambda g: abs(g - gamma_key))
            zr = z_rows_by_gamma[nearest_z_key]
            wvals = _competitor_re_values(wu_points.get(gamma_key, {}))
            zvals = _all_certified_re_values(z_points.get(nearest_z_key, {}))
            fr = finite_by_gamma.get(gamma_key, {})
            writer.writerow(
                {
                    "gamma": gamma_key,
                    "Gamma": -gamma_key,
                    "wu_seed_re": wr["Re_Phi"],
                    "wu_cert_roots": wr["num_certified_roots"],
                    "wu_cert_competitors": wr["num_certified_competitors"],
                    "wu_seed_minus_best_comp": wr["min_certified_gap_re"],
                    "wu_best_comp_re": max(wvals) if wvals else "",
                    "wu_naive_proxy_possible_comps": sum(1 for v in wvals if PHI_PREF_CONV2 + v <= 0),
                    "wu_naive_proxy_impossible_comps": sum(1 for v in wvals if PHI_PREF_CONV2 + v > 0),
                    "z_nearest_gamma": nearest_z_key,
                    "z_seed_re": zr["Re_Phi"],
                    "z_seed_conv2_exponent": PHI_PREF_CONV2 + float(zr["Re_Phi"]),
                    "z_cert_roots": zr["num_certified_roots"],
                    "z_cert_competitors": zr["num_certified_competitors"],
                    "z_seed_minus_best_comp": zr["min_certified_gap_re"],
                    "z_best_comp_re": max(zvals) if zvals else "",
                    "z_envelope_conv2_exponent": PHI_PREF_CONV2 + max(zvals) if zvals else "",
                    "exact_lambda_abs_n40": fr.get("lambda_abs_n40", ""),
                    "exact_lambda_abs_n100": fr.get("lambda_abs_n100", ""),
                }
            )


def _write_report(out_dir: Path, summary: dict[str, Any]) -> None:
    wu = summary["systems"]["wu_corrected_common_window"]
    z = summary["systems"]["z_archive_common_window"]
    phys = summary["z_physical_layer"]
    wu_as = summary["wu_anti_stokes_proxy_layer"]
    lines = [
        "# Corrected BM24 q=3 complete (u,w) saddle audit vs z saddle",
        "",
        "## Corrected formalism",
        "",
        "For BM24 q=3 the clause degree is `m = 2**q = 8`.  The parent-action saddle equations used here are",
        "",
        "`Phi(u,w) = sum_a c_a u_a^m + i sum_a w_a u_a + log Delta(w)`,",
        "",
        "`m c_a u_a^(m-1) + i w_a = 0`, and `i u_a + g_a(w) = 0`.",
        "",
        "Eliminating through the second equation gives `u_a = i g_a(w)` and the integer-power reduced system",
        "",
        "`F_a(w) = w_a + kappa_q c_a g_a(w)^(m-1) = 0`, with `kappa_q = -m i^m`.",
        "",
        "Thus for q=3: `m=8`, `kappa_q=-8`, and `F_a(w)=w_a - 8 c_a g_a(w)^7`.  The discarded stale run solved a degree-six surrogate and is not used below.",
        "",
        "## Common-window algebraic comparison",
        "",
        f"Window: `gamma in [{wu['window'][0]}, {wu['window'][1]}]`, i.e. `Gamma=-gamma in [0.01, 1.0]`.",
        "",
        f"- Corrected `(u,w)` certified roots per slice: min {wu['certified_roots_min']}, max {wu['certified_roots_max']}, mean {wu['certified_roots_mean']:.2f}.",
        f"- Archived `z` certified roots per slice in the same window: min {z['certified_roots_min']}, max {z['certified_roots_max']}, mean {z['certified_roots_mean']:.2f}.",
        f"- Corrected `(u,w)` resolved branches: {wu['branch_stats']['num_branches']} total, {wu['branch_stats']['num_danger_branches_seed_minus_comp_negative']} that beat the seed somewhere.",
        f"- Archived `z` resolved branches: {z['branch_stats']['num_branches']} total, {z['branch_stats']['num_danger_branches_seed_minus_comp_negative']} that beat the seed somewhere.",
        f"- `(u,w)` seed-minus-best gap range: {wu['seed_minus_best_comp_gap_min']:.6g} to {wu['seed_minus_best_comp_gap_max']:.6g}.",
        f"- `z` seed-minus-best gap range in the same window: {z['seed_minus_best_comp_gap_min']:.6g} to {z['seed_minus_best_comp_gap_max']:.6g}.",
        "",
        "Interpretation: the corrected `(u,w)` chart does not reduce the number of local algebraic objects in this run.  It finds more certified local roots and more branch-resolved competitors than the archived `z` run on the overlapping interval.  Its advantage is formal cleanliness and direct Krawczyk certification in the integer-power chart, not an automatic collapse of the saddle landscape.",
        "",
        "## Physical plausibility",
        "",
        f"The known BM24 Conv2 prefactor for k=8, r=176.54 is `{PHI_PREF_CONV2:.9f}`.  In the `z` system, the physical exponent screen is `E = {PHI_PREF_CONV2:.9f} + RePhi_z <= 0`.",
        "",
        f"For `(u,w)`, applying the same number is only a naive proxy until the parent-action normalization and contour map are derived. Under this proxy, the p_succ=1 threshold is raw `RePhi <= {-PHI_PREF_CONV2:.9f}`.",
        "",
        f"- `(u,w)` certified competitor hits in the common window: {wu['total_certified_competitor_hits']}.",
        f"- `(u,w)` naive-prefactor possible hits: {wu['naive_prefactor_proxy_possible_hits']}.",
        f"- `(u,w)` naive-prefactor impossible hits: {wu['naive_prefactor_proxy_impossible_hits']}.",
        f"- `(u,w)` raw competitor RePhi range: {wu['competitor_raw_re_min']:.6g} to {wu['competitor_raw_re_max']:.6g}.",
        "",
        "So yes: many high-Re `(u,w)` saddles are exactly the kind of physically implausible `p_succ > 1` decoys you were worried about, at least under the natural Conv2-prefactor sanity screen.  The low-Re subset is the plausible candidate pool; it still needs contour/thimble evidence.",
        "",
        "For comparison, the later `z` physical layer reports:",
        "",
        f"- certified competitors recomputed: {phys['certified_competitors']}; possible kept: {phys['possible_competitors']}; impossible removed: {phys['impossible_competitors']}.",
        f"- beta={BETA:.10f} possible-exponent transition bracket: Gamma {phys['beta_opt_transition_low']} to {phys['beta_opt_transition_high']}.",
        f"- finite-size/anti-Stokes push crossing: gamma {summary['z_finite_size']['anti_stokes_gamma']:.12f}.",
        "",
        "## Anti-Stokes status",
        "",
        f"The corrected `(u,w)` branch resolver reports {summary['wu_corrected_full_run']['refined_re_gap_brackets']} refined Re-gap zero brackets and {summary['wu_corrected_full_run']['refined_re_gap_box_disjoint']} box-disjoint brackets on `Gamma <= 1`. These are anti-Stokes candidates in the local action sense, not certified Picard-Lefschetz contour crossings.",
        "",
        f"Under the naive-prefactor physical screen, {wu_as['naive_prefactor_possible_crossings']} of those refined `(u,w)` Re-gap brackets have proxy-possible competitor action and {wu_as['naive_prefactor_impossible_crossings']} have proxy-impossible competitor action. This is the cleanest current split between plausible transition candidates and high-Re decoy crossings.",
        "",
        "The physical `z` anti-Stokes evidence lives farther out: at beta approximately 0.5434, the possible-exponent scan has candidate Gamma estimates near 1.465816, 1.649362, and 1.650757, while the finite-size pair push has gamma approximately -1.830842.  The corrected `(u,w)` run stops at gamma=-1, so it does not yet cover the known physical transition window.",
        "",
        "## Finite-n tracking",
        "",
        "Finite-n tracking is currently rigorous for the `z`/Conv2 normalization: exact all-subsets `lambda_abs=(1/n)log|S_n|` can be compared with `phi_pref + RePhi_z`.  The new dashboard overlays sparse exact finite-n points against the z seed and certified envelope on the common window.",
        "",
        "For `(u,w)`, the correct next experiment is not to overlay raw `RePhi_uw` as if it were already a finite-n exponent.  The right workflow is: derive or fit the `(u,w)` normalization, filter the naive-impossible saddles, then test whether the remaining branch envelope tracks exact finite-n rates and whether PL continuation selects those saddles.",
        "",
        "## Artifacts",
        "",
        "- `corrected_wu_vs_z_dashboard.png`",
        "- `corrected_wu_proxy_physical_screen.png`",
        "- `corrected_wu_plausible_tracks_and_crossings.png`",
        "- `corrected_wu_vs_z_common_window.csv`",
        "- `corrected_wu_vs_z_summary.json`",
    ]
    (out_dir / "CORRECTED_UW_VS_Z_COMPARISON_ANALYSIS.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=Path("experiments/w_saddle/runs/wu_bm24_q3_correct_20260706"))
    ap.add_argument("--force-finite-n", action="store_true")
    args = ap.parse_args()
    out_dir = args.out_dir

    wu_comp = _load_json(out_dir / "competitors_robust_wu.json")
    wu_resolved = _load_json(out_dir / "resolved_robust_wu.json")
    wu_summary = _load_json(out_dir / "corrected_wu_summary.json")

    z_root = Path("results/bm24_saddle_audit_p1/run_seed_branch_g-2pi/archive/competitor_dominance/z_robust_full")
    z_comp = _load_json(z_root / "competitors_robust_z.json")
    z_resolved = _load_json(z_root / "resolved_robust_z.json")
    z_model = _load_json(z_root / "dominance/model_comparison_summary.json")
    z_phys = _load_json(Path("results/bm24_saddle_audit_p1/physically_possible_refined/physically_possible_refined_summary.json"))
    z_finite = _load_json(Path("results/bm24_saddle_audit_p1/anti_stokes_finite_size_push/anti_stokes_finite_size_summary.json"))
    z_possible_as = _load_json(Path("results/bm24_saddle_audit_p1/possible_anti_stokes/possible_anti_stokes_lines.json"))

    finite = _finite_n_sparse(
        out_dir / "z_exact_finite_n_sparse_common_window.json",
        [-0.1, -0.2, -0.3, -0.5, -0.7, -0.9, -1.0],
        [40, 100],
        args.force_finite_n,
    )

    beta_opt_transition = next(t for t in z_phys["transitions"] if abs(float(t["beta"]) - 0.5433996421) < 1e-8)
    beta_opt_candidates = [
        c for c in z_possible_as["fixed_beta_candidates"]
        if abs(float(c["beta"]) - 0.5433996421) < 1e-8
    ]
    wu_crossing_stats = _wu_crossing_proxy_stats(wu_resolved)

    summary = {
        "definition": {
            "q": Q,
            "m": 2**Q,
            "r": R,
            "beta": BETA,
            "corrected_wu_equation": "F_a(w)=w_a - 8 c_a g_a(w)^7 for q=3/k=8",
            "discarded_bad_equation": "degree-six surrogate w + 6 c g^5 from stale code path",
            "conv2_prefactor_z": PHI_PREF_CONV2,
            "wu_physical_caveat": "Raw RePhi_uw is not yet a certified BM24 finite-n exponent; same-prefactor filtering is a sanity proxy only.",
        },
        "wu_corrected_full_run": wu_summary,
        "systems": {
            "wu_corrected_common_window": _summarize_system("wu", wu_comp, wu_resolved, -1.0, -0.01),
            "z_archive_common_window": _summarize_system("z", z_comp, z_resolved, -1.0, -0.01),
        },
        "z_archive_full": {
            "model_comparison": z_model,
            "resolved_branch_stats": _branch_gap_stats(z_resolved),
        },
        "z_physical_layer": {
            "certified_competitors": z_phys["counts"]["certified_competitors"],
            "possible_competitors": z_phys["counts"]["possible_competitors"],
            "impossible_competitors": z_phys["counts"]["impossible_competitors"],
            "beta_opt_transition_low": beta_opt_transition["transitions"][0]["Gamma_low"],
            "beta_opt_transition_high": beta_opt_transition["transitions"][0]["Gamma_high"],
            "possible_anti_stokes_beta_opt_candidates": beta_opt_candidates,
        },
        "wu_anti_stokes_proxy_layer": wu_crossing_stats,
        "z_finite_size": {
            "anti_stokes_gamma": z_finite["anti_stokes_gamma"],
            "finite_size_crossings": z_finite["finite_size_crossings"],
            "residual_convergence": z_finite["residual_convergence"],
        },
        "new_sparse_finite_n_common_window": finite,
    }

    (out_dir / "corrected_wu_vs_z_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=True) + "\n")
    _write_csv(out_dir, {"comp": wu_comp}, {"comp": z_comp}, finite)
    _plot_dashboard(out_dir, {"comp": wu_comp}, {"comp": z_comp}, finite)
    _plot_wu_proxy(out_dir, {"comp": wu_comp})
    _plot_wu_plausible_tracks(out_dir, {"comp": wu_comp}, wu_resolved, wu_crossing_stats)
    _write_report(out_dir, summary)
    print(f"wrote comparison artifacts under {out_dir}")


if __name__ == "__main__":
    main()
