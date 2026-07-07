"""Rebuild refined transition figures after filtering impossible saddle exponents.

The refined scan already separates high-Re algebraic competitors from saddles
that match the exact finite-n exponent.  This stricter audit asks a simpler
question first: if a saddle were taken literally, would it predict
``p_succ ~ exp(n E)`` with ``E > 0``?  Such a saddle cannot be a standalone
physical probability contribution, so it is excluded from the candidate
controller set before redrawing the transition map.
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
import matplotlib.colors as mcolors
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
PACKAGE_ROOT = HERE.parents[2]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from phasecraft.bm24_saddle_audit_p1.continue_seed_branch_certified import (
    discover_competitors_at_gamma,
)
from phasecraft.bm24_saddle_audit_p1.refined_transition_scan import (
    BETA_OPT,
    K_CLAUSE,
    Q,
    R,
    conv2_full_exponent,
    conv2_pref,
    default_betas,
    default_gammas,
    lambda_abs_n_max,
    pcolor_edges,
    row_key,
    seed_from_previous,
    warm_seed_to,
)
from phasecraft.bm24_saddle_audit_p1 import sweep_beta_gamma as sweep
from phasecraft.bm24_saddle_audit_p1.audit import finite_n_exponent_grid
from phasecraft.bm24_saddle_audit_p1.pl_homotopy_tracker import (
    continue_competitor_to,
    seed_row as certified_seed_row,
)
from phasecraft.lib.saddles.krawczyk_p1_roots import SaddleSystem
from phasecraft.lib.saddles.picard_lefschetz import compute_phi


RESULTS = HERE / "results"
REFINED = RESULTS / "refined_transition_scan"
PL_ALL = RESULTS / "pl_homotopy_all_slices"
OUT = RESULTS / "physically_possible_refined"
RAW = OUT / "physically_possible_refined.json"

BLUE = "#2b6cb0"
ORANGE = "#dd6b20"
GREEN = "#2f855a"
RED = "#c53030"
PURPLE = "#6b46c1"
GRAY = "#4a5568"


def load_json(path: Path) -> Any:
    with path.open() as f:
        return json.load(f)


def load_cached_rows() -> dict[str, dict[str, Any]]:
    if not RAW.exists():
        return {}
    data = load_json(RAW)
    return {
        row_key(float(r["beta"]), float(r["gamma"])): r
        for r in data.get("rows", [])
    }


def save_rows(rows: list[dict[str, Any]], metadata: dict[str, Any]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with RAW.open("w") as f:
        json.dump({"metadata": metadata, "rows": rows}, f, indent=2)


def merge_rows(cached: dict[str, dict[str, Any]], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged = dict(cached)
    for row in rows:
        if "beta" in row and "gamma" in row:
            merged[row_key(float(row["beta"]), float(row["gamma"]))] = row
    return sorted(merged.values(), key=lambda r: (float(r["beta"]), -float(r["gamma"])))


def physically_possible(full_exponent: float, tol: float) -> bool:
    return math.isfinite(full_exponent) and full_exponent <= tol


def classify_possible(row: dict[str, Any], decoy_margin: float) -> str:
    if row.get("status") != "ok":
        return "failed"
    if not bool(row.get("seed_is_best_match_possible")):
        return "possible_competitor"
    if float(row.get("max_possible_competitor_delta", float("-inf"))) > decoy_margin:
        return "possible_decoy"
    if int(row.get("n_impossible_competitors", 0)) and float(row.get("max_all_competitor_full", float("-inf"))) > 0.0:
        return "impossible_only_decoy"
    return "seed"


def possible_transition_table(rows: list[dict[str, Any]], decoy_margin: float) -> list[dict[str, Any]]:
    out = []
    for beta in sorted({float(r["beta"]) for r in rows if r.get("status") == "ok"}):
        rows_b = sorted(
            [r for r in rows if r.get("status") == "ok" and math.isclose(float(r["beta"]), beta)],
            key=lambda r: -float(r["gamma"]),
        )
        flags = [classify_possible(r, decoy_margin) != "possible_competitor" for r in rows_b]
        gamma_pos = [-float(r["gamma"]) for r in rows_b]
        transitions = []
        for a, b, Ga, Gb in zip(flags[:-1], flags[1:], gamma_pos[:-1], gamma_pos[1:]):
            if a != b:
                transitions.append(
                    {
                        "Gamma_low": Ga,
                        "Gamma_high": Gb,
                        "Gamma_mid": 0.5 * (Ga + Gb),
                        "uncertainty": 0.5 * abs(Gb - Ga),
                        "from": "seed_or_decoy" if a else "possible_competitor",
                        "to": "seed_or_decoy" if b else "possible_competitor",
                    }
                )
        classes = [classify_possible(r, decoy_margin) for r in rows_b]
        out.append(
            {
                "beta": beta,
                "n_points": len(rows_b),
                "n_seed": classes.count("seed"),
                "n_possible_decoy": classes.count("possible_decoy"),
                "n_impossible_only_decoy": classes.count("impossible_only_decoy"),
                "n_possible_competitor": classes.count("possible_competitor"),
                "transitions": transitions,
            }
        )
    return out


def compute_row(beta: float, gamma: float, seed_z: np.ndarray, starts: int, possible_tol: float) -> dict[str, Any]:
    sys_ = SaddleSystem.build(q=Q, r=R, betas=np.array([beta]), gammas=np.array([gamma]))
    phi_seed = compute_phi(seed_z, q=Q, r=R, betas=sys_.betas, gammas=sys_.gammas)
    re_phi_seed = float(phi_seed.real)
    im_phi_seed = float(phi_seed.imag)
    seed_full = conv2_full_exponent(re_phi_seed, K_CLAUSE, R)
    lam = lambda_abs_n_max(Q, K_CLAUSE, R, beta, gamma, sweep.N_VALUES)
    seed_gap = seed_full - lam
    seed_gap_abs = abs(seed_gap)

    comps = discover_competitors_at_gamma(
        Q,
        K_CLAUSE,
        R,
        beta,
        gamma,
        seed_z,
        num_starts=starts,
        seed=int(8101 + 1000 * abs(gamma) + 1000 * beta),
        branch_tol=sweep.BRANCH_TOL,
        min_residual=sweep.MIN_RESIDUAL,
        dps=sweep.DPS,
    )
    cert = [c for c in comps if c.get("certified")]
    possible = [
        c for c in cert
        if physically_possible(float(c["full_conv2_exponent"]), possible_tol)
    ]
    impossible = [
        c for c in cert
        if not physically_possible(float(c["full_conv2_exponent"]), possible_tol)
    ]

    if cert:
        max_all = max(cert, key=lambda c: float(c["full_conv2_exponent"]))
        max_all_full = float(max_all["full_conv2_exponent"])
        max_all_delta = max_all_full - seed_full
    else:
        max_all_full = float("nan")
        max_all_delta = 0.0

    if possible:
        max_possible = max(possible, key=lambda c: float(c["full_conv2_exponent"]))
        max_possible_full = float(max_possible["full_conv2_exponent"])
        max_possible_delta = max_possible_full - seed_full
        best_possible = min(
            possible,
            key=lambda c: abs(float(c["full_conv2_exponent"]) - lam),
        )
        best_possible_gap = abs(float(best_possible["full_conv2_exponent"]) - lam)
        seed_is_best_possible = seed_gap_abs <= best_possible_gap
        best_possible_full = float(best_possible["full_conv2_exponent"])
        best_possible_z_real = best_possible.get("z_real", [])
        best_possible_z_imag = best_possible.get("z_imag", [])
    else:
        max_possible_full = float("nan")
        max_possible_delta = float("-inf")
        best_possible_gap = float("nan")
        seed_is_best_possible = True
        best_possible_full = float("nan")
        best_possible_z_real = []
        best_possible_z_imag = []

    row = {
        "beta": beta,
        "gamma": gamma,
        "re_phi_seed": re_phi_seed,
        "im_phi_seed": im_phi_seed,
        "full_conv2_seed": seed_full,
        "lambda_abs": lam,
        "seed_gap": seed_gap,
        "n_certified_competitors": len(cert),
        "max_competitor_delta_re": max_all_delta,
        "seed_algebraically_dominates": max_all_delta <= 0.0,
        "seed_is_best_match_to_exact": seed_is_best_possible,
        "best_match_abs_gap": best_possible_gap,
        "status": "ok",
        "n_starts_possible_audit": starts,
        "possible_exponent_tol": possible_tol,
        "n_certified_competitors_recomputed": len(cert),
        "n_possible_competitors": len(possible),
        "n_impossible_competitors": len(impossible),
        "max_all_competitor_full": max_all_full,
        "max_all_competitor_delta": max_all_delta,
        "max_possible_competitor_full": max_possible_full,
        "max_possible_competitor_delta": max_possible_delta,
        "best_possible_match_abs_gap": best_possible_gap,
        "best_possible_full": best_possible_full,
        "best_possible_z_real": best_possible_z_real,
        "best_possible_z_imag": best_possible_z_imag,
        "seed_is_best_match_possible": seed_is_best_possible,
        "removed_impossible_high_re_controller": bool(max_all_full > possible_tol and max_all_delta > max_possible_delta),
    }
    return row


def grid_from_rows(rows: list[dict[str, Any]], decoy_margin: float) -> dict[str, np.ndarray]:
    ok = [r for r in rows if r.get("status") == "ok"]
    betas = np.array(sorted({float(r["beta"]) for r in ok}))
    Gammas = np.array(sorted({-float(r["gamma"]) for r in ok}))
    cls = np.full((len(Gammas), len(betas)), np.nan)
    n_possible = np.full_like(cls, np.nan, dtype=float)
    n_impossible = np.full_like(cls, np.nan, dtype=float)
    max_possible = np.full_like(cls, np.nan, dtype=float)
    max_all = np.full_like(cls, np.nan, dtype=float)
    gap = np.full_like(cls, np.nan, dtype=float)
    residual_delta = np.full_like(cls, np.nan, dtype=float)
    code = {
        "seed": 0.0,
        "impossible_only_decoy": 1.0,
        "possible_decoy": 2.0,
        "possible_competitor": 3.0,
        "failed": np.nan,
    }
    for r in ok:
        i = int(np.where(np.isclose(Gammas, -float(r["gamma"])))[0][0])
        j = int(np.where(np.isclose(betas, float(r["beta"])))[0][0])
        cls[i, j] = code[classify_possible(r, decoy_margin)]
        n_possible[i, j] = float(r.get("n_possible_competitors", 0))
        n_impossible[i, j] = float(r.get("n_impossible_competitors", 0))
        max_possible[i, j] = float(r.get("max_possible_competitor_full", float("nan")))
        max_all[i, j] = float(r.get("max_all_competitor_full", float("nan")))
        seed_gap_abs = abs(float(r.get("seed_gap", float("nan"))))
        best_possible_gap = float(r.get("best_possible_match_abs_gap", float("nan")))
        if int(r.get("n_possible_competitors", 0)) > 0 and math.isfinite(best_possible_gap):
            residual_delta[i, j] = best_possible_gap - seed_gap_abs
        if bool(r.get("seed_is_best_match_possible")) or math.isnan(float(r.get("best_possible_match_abs_gap", float("nan")))):
            gap[i, j] = abs(float(r.get("seed_gap", float("nan"))))
        else:
            gap[i, j] = float(r["best_possible_match_abs_gap"])
    return {
        "betas": betas,
        "Gammas": Gammas,
        "class": cls,
        "n_possible": n_possible,
        "n_impossible": n_impossible,
        "max_possible": max_possible,
        "max_all": max_all,
        "gap": gap,
        "residual_delta": residual_delta,
    }


def robust_points(transitions: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    bx, by, be, ax, ay = [], [], [], [], []
    for t in transitions:
        trs = t.get("transitions", [])
        if (
            len(trs) == 1
            and trs[0]["from"] == "seed_or_decoy"
            and trs[0]["to"] == "possible_competitor"
        ):
            bx.append(float(t["beta"]))
            by.append(float(trs[0]["Gamma_mid"]))
            be.append(float(trs[0]["uncertainty"]))
        elif trs:
            ax.append(float(t["beta"]))
            ay.append(float(trs[0]["Gamma_mid"]))
    return np.array(bx), np.array(by), np.array(be), np.array(ax), np.array(ay)


def _plot_guideline_segments(axis: Any, bx: np.ndarray, by: np.ndarray, **kwargs: Any) -> None:
    left = (bx > 0.350001) & (bx < 0.60)
    right = bx > 0.60
    if int(np.sum(left)) >= 2:
        axis.plot(bx[left], by[left], **kwargs)
    if int(np.sum(right)) >= 2:
        axis.plot(bx[right], by[right], **kwargs)


def lambda_abs_by_n(
    q: int,
    k_clause: int,
    r: float,
    beta: float,
    gamma: float,
    n_values: list[int],
) -> dict[int, float]:
    result = finite_n_exponent_grid(k_clause, q, r, beta, gamma, n_values)
    return {
        int(n): float(result["lambda_abs"][str(n)])
        for n in result["n_values"]
    }


def pl_summary() -> list[dict[str, Any]]:
    path = PL_ALL / "pl_homotopy_all_slices.json"
    if not path.exists():
        return []
    return load_json(path)["summaries"]


def pl_results() -> list[dict[str, Any]]:
    path = PL_ALL / "pl_homotopy_all_slices.json"
    if not path.exists():
        return []
    return load_json(path).get("results", [])


def zero_root_candidates(xs: np.ndarray, ys: np.ndarray, tol: float = 1e-10) -> list[dict[str, float]]:
    roots: list[dict[str, float]] = []
    finite = np.isfinite(xs) & np.isfinite(ys)
    xs = xs[finite]
    ys = ys[finite]
    if len(xs) < 2:
        return roots
    order = np.argsort(xs)
    xs = xs[order]
    ys = ys[order]
    for idx, (x, y) in enumerate(zip(xs, ys)):
        if abs(float(y)) <= tol:
            roots.append({"root": float(x), "lo": float(x), "hi": float(x), "interpolated": False})
    for x0, x1, y0, y1 in zip(xs[:-1], xs[1:], ys[:-1], ys[1:]):
        if y0 == 0.0 or y1 == 0.0 or y0 * y1 > 0.0:
            continue
        root = float(x0 - y0 * (x1 - x0) / (y1 - y0))
        roots.append({"root": root, "lo": float(min(x0, x1)), "hi": float(max(x0, x1)), "interpolated": True})
    dedup: dict[float, dict[str, float]] = {}
    for root in roots:
        dedup.setdefault(round(float(root["root"]), 10), root)
    return [dedup[k] for k in sorted(dedup)]


def blockwise_unwrap(raw_values: np.ndarray) -> np.ndarray:
    raw = np.asarray(raw_values, dtype=float)
    unwrapped = np.full_like(raw, np.nan)
    valid = np.flatnonzero(np.isfinite(raw))
    blocks = np.split(valid, np.where(np.diff(valid) > 1)[0] + 1)
    for block in blocks:
        if len(block):
            unwrapped[block] = np.unwrap(raw[block])
    return unwrapped


def _append_root(
    out: dict[str, list[float]],
    prefix: str,
    beta: float,
    candidate: dict[str, float],
) -> None:
    root = float(candidate["root"])
    out[f"{prefix}_beta"].append(beta)
    out[f"{prefix}_gamma"].append(root)
    out[f"{prefix}_err_low"].append(max(0.0, root - float(candidate["lo"])))
    out[f"{prefix}_err_high"].append(max(0.0, float(candidate["hi"]) - root))


def _nearest_stokes_family(
    roots_by_beta: dict[float, list[dict[str, float]]],
    anti_by_beta: dict[float, list[dict[str, float]]],
    *,
    anchor_beta_target: float = BETA_OPT,
    max_step: float = 0.18,
) -> set[tuple[float, int]]:
    if not roots_by_beta:
        return set()
    anchor_beta = min(roots_by_beta, key=lambda b: abs(b - anchor_beta_target))
    anti_anchor = anti_by_beta.get(anchor_beta, [])
    target = anti_anchor[0]["root"] if anti_anchor else np.mean([r["root"] for r in roots_by_beta[anchor_beta]])
    anchor_idx = min(range(len(roots_by_beta[anchor_beta])), key=lambda i: abs(roots_by_beta[anchor_beta][i]["root"] - target))
    selected: set[tuple[float, int]] = {(anchor_beta, anchor_idx)}

    def walk(betas: list[float], start_beta: float, start_root: float) -> None:
        prev = start_root
        for beta in betas:
            candidates = roots_by_beta.get(beta, [])
            if not candidates:
                continue
            idx = min(range(len(candidates)), key=lambda i: abs(candidates[i]["root"] - prev))
            if abs(candidates[idx]["root"] - prev) <= max_step:
                selected.add((beta, idx))
                prev = candidates[idx]["root"]

    betas = sorted(roots_by_beta)
    anchor_root = roots_by_beta[anchor_beta][anchor_idx]["root"]
    walk([b for b in betas if b > anchor_beta], anchor_beta, anchor_root)
    walk(list(reversed([b for b in betas if b < anchor_beta])), anchor_beta, anchor_root)
    return selected


def condition_candidates_by_beta() -> tuple[dict[float, list[dict[str, float]]], dict[float, list[dict[str, float]]]]:
    anti_by_beta: dict[float, list[dict[str, float]]] = {}
    stokes_by_beta: dict[float, list[dict[str, float]]] = {}
    for result in pl_results():
        beta = float(result.get("beta", math.nan))
        rows = [r for r in result.get("rows", []) if not r.get("competitor_failed")]
        if not math.isfinite(beta) or not rows:
            continue
        gammas = np.array([float(r["Gamma"]) for r in rows], dtype=float)
        delta_re = np.array([float(r.get("delta_re_phi", float("nan"))) for r in rows], dtype=float)
        delta_im_raw = np.array([float(r.get("delta_im_phi_raw", float("nan"))) for r in rows], dtype=float)
        delta_im = blockwise_unwrap(delta_im_raw)
        anti_by_beta[beta] = zero_root_candidates(gammas, delta_re)
        stokes_by_beta[beta] = zero_root_candidates(gammas, delta_im)
    return anti_by_beta, stokes_by_beta


def pl_condition_points(*, show_family: bool = True) -> dict[str, np.ndarray]:
    out: dict[str, list[float]] = {
        "anti_beta": [], "anti_gamma": [], "anti_err_low": [], "anti_err_high": [],
        "stokes_family_beta": [], "stokes_family_gamma": [], "stokes_family_err_low": [], "stokes_family_err_high": [],
        "stokes_other_beta": [], "stokes_other_gamma": [], "stokes_other_err_low": [], "stokes_other_err_high": [],
    }
    anti_by_beta, stokes_by_beta = condition_candidates_by_beta()
    family = _nearest_stokes_family(stokes_by_beta, anti_by_beta) if show_family else set()
    for beta, candidates in anti_by_beta.items():
        for candidate in candidates:
            _append_root(out, "anti", beta, candidate)
    for beta, candidates in stokes_by_beta.items():
        for idx, candidate in enumerate(candidates):
            prefix = "stokes_family" if (beta, idx) in family else "stokes_other"
            _append_root(out, prefix, beta, candidate)

    return {key: np.array(value) for key, value in out.items()}


def phase_anchor_robustness(anchor_targets: list[float] | None = None) -> dict[str, Any]:
    anchor_targets = anchor_targets or [0.50, BETA_OPT, 0.575]
    anti_by_beta, stokes_by_beta = condition_candidates_by_beta()
    families: dict[str, set[tuple[float, int]]] = {}
    roots: dict[str, dict[float, float]] = {}
    for target in anchor_targets:
        key = f"{target:.10f}"
        family = _nearest_stokes_family(stokes_by_beta, anti_by_beta, anchor_beta_target=target)
        families[key] = family
        roots[key] = {
            float(beta): float(stokes_by_beta[beta][idx]["root"])
            for beta, idx in family
            if beta in stokes_by_beta and idx < len(stokes_by_beta[beta])
        }
    common_betas = sorted(set.intersection(*(set(value) for value in roots.values()))) if roots else []
    rows = []
    robust = bool(common_betas)
    for beta in common_betas:
        selected = [roots[f"{target:.10f}"][beta] for target in anchor_targets]
        spread = float(max(selected) - min(selected))
        same = spread <= 1e-8
        robust = robust and same
        rows.append({
            "beta": float(beta),
            "selected_Gammas": selected,
            "spread": spread,
            "same": same,
        })
    if not common_betas:
        robust = False
    return {
        "anchor_targets": [float(v) for v in anchor_targets],
        "robust": bool(robust),
        "common_betas": common_betas,
        "rows": rows,
    }


def phase_alignment_report_rows() -> list[dict[str, float]]:
    cond = pl_condition_points()
    rows = []
    for beta in sorted(set(cond["anti_beta"]).intersection(set(cond["stokes_family_beta"]))):
        anti = cond["anti_gamma"][np.where(np.isclose(cond["anti_beta"], beta))[0]]
        stokes = cond["stokes_family_gamma"][np.where(np.isclose(cond["stokes_family_beta"], beta))[0]]
        if len(anti) and len(stokes):
            rows.append({
                "beta": float(beta),
                "Gamma_AS": float(anti[0]),
                "Gamma_S": float(stokes[0]),
                "delta": float(abs(stokes[0] - anti[0])),
            })
    return rows


def _complex_from_row(row: dict[str, Any], prefix: str) -> np.ndarray | None:
    re = row.get(f"{prefix}_z_real")
    im = row.get(f"{prefix}_z_imag")
    if not re or not im:
        return None
    return np.asarray(re, dtype=float) + 1j * np.asarray(im, dtype=float)


def _tracked_competitor_near(beta: float, Gamma: float) -> tuple[np.ndarray | None, float | None]:
    best_row = None
    best_dist = math.inf
    for result in pl_results():
        if not math.isclose(float(result.get("beta", math.nan)), beta, abs_tol=5e-9):
            continue
        for row in result.get("rows", []):
            if row.get("competitor_failed"):
                continue
            comp = row.get("competitor", {})
            if not comp.get("z_real") or not comp.get("z_imag"):
                continue
            dist = abs(float(row["Gamma"]) - Gamma)
            if dist < best_dist:
                best_dist = dist
                best_row = row
    if best_row is None:
        return None, None
    comp = best_row["competitor"]
    z = np.asarray(comp["z_real"], dtype=float) + 1j * np.asarray(comp["z_imag"], dtype=float)
    return z, float(best_row["Gamma"])


def competitor_identity_report_rows(rows: list[dict[str, Any]], transitions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = {(round(float(r["beta"]), 10), round(-float(r["gamma"]), 10)): r for r in rows if r.get("status") == "ok"}
    out = []
    for t in transitions:
        beta = float(t["beta"])
        trs = [
            tr for tr in t.get("transitions", [])
            if tr.get("from") == "seed_or_decoy" and tr.get("to") == "possible_competitor"
        ]
        if not trs:
            continue
        tr = trs[0]
        Gamma_probe = float(tr["Gamma_high"])
        row = by_key.get((round(beta, 10), round(Gamma_probe, 10)))
        z_best = _complex_from_row(row or {}, "best_possible")
        z_track, Gamma_track = _tracked_competitor_near(beta, Gamma_probe)
        if z_best is None or z_track is None:
            out.append({
                "beta": beta,
                "Gamma_rate": Gamma_probe,
                "Gamma_tracked": Gamma_track,
                "d_z": float("nan"),
                "classification": "needs recomputation with stored z-coordinates",
            })
            continue
        d_z = float(np.linalg.norm(z_best - z_track, ord=np.inf))
        if d_z < 1e-5:
            cls = "same numerical root"
        elif d_z < 1e-3:
            cls = "probably same branch; inspect"
        else:
            cls = "different competitor"
        out.append({
            "beta": beta,
            "Gamma_rate": Gamma_probe,
            "Gamma_tracked": Gamma_track,
            "d_z": d_z,
            "classification": cls,
        })
    return out


def _nearest_pl_result(beta: float) -> dict[str, Any] | None:
    candidates = [
        result for result in pl_results()
        if result.get("status") == "ok" and result.get("rows")
    ]
    if not candidates:
        return None
    best = min(candidates, key=lambda result: abs(float(result.get("beta", math.nan)) - beta))
    if abs(float(best.get("beta", math.nan)) - beta) > 5e-4:
        return None
    return best


def _crossing_from_series(points: list[tuple[float, float]]) -> dict[str, Any]:
    finite = [(float(g), float(d)) for g, d in points if math.isfinite(g) and math.isfinite(d)]
    finite.sort(key=lambda item: item[0])
    if not finite:
        return {"status": "missing", "Gamma_cross": None, "lo": None, "hi": None}

    sign_changes: list[dict[str, float]] = []
    for (g0, d0), (g1, d1) in zip(finite[:-1], finite[1:]):
        if d0 > 0.0 and d1 <= 0.0:
            if d1 == d0:
                root = 0.5 * (g0 + g1)
            else:
                root = g0 - d0 * (g1 - g0) / (d1 - d0)
            sign_changes.append({
                "Gamma_cross": float(root),
                "lo": float(g0),
                "hi": float(g1),
            })
    if sign_changes:
        first = sign_changes[0]
        first["status"] = "bracketed" if len(sign_changes) == 1 else "multiple_sign_changes"
        first["n_sign_changes"] = len(sign_changes)
        return first
    if all(d <= 0.0 for _, d in finite):
        return {
            "status": "left_censored",
            "Gamma_cross": float(finite[0][0]),
            "lo": None,
            "hi": float(finite[0][0]),
        }
    if all(d > 0.0 for _, d in finite):
        return {
            "status": "right_censored",
            "Gamma_cross": float(finite[-1][0]),
            "lo": float(finite[-1][0]),
            "hi": None,
        }
    return {"status": "no_positive_to_negative_change", "Gamma_cross": None, "lo": None, "hi": None}


def finite_n_stability_rows(transitions: list[dict[str, Any]], n_values: list[int]) -> tuple[list[dict[str, Any]], list[float]]:
    bx, _by, _be, ax, _ay = robust_points(transitions)
    clean_betas = [float(b) for b in bx]
    nonmonotone_betas = [float(b) for b in ax]
    out: list[dict[str, Any]] = []
    lambda_cache: dict[tuple[float, float], dict[int, float]] = {}

    for beta in clean_betas:
        result = _nearest_pl_result(beta)
        if result is None:
            out.append({
                "beta": beta,
                "status": "missing_pl_branch",
                "per_n": {},
                "range": None,
                "n_valid_Gamma": 0,
            })
            continue

        series_by_n: dict[int, list[tuple[float, float]]] = {int(n): [] for n in n_values}
        for row in sorted(result.get("rows", []), key=lambda r: float(r.get("Gamma", math.nan))):
            if row.get("competitor_failed") or row.get("competitor_merged_with_seed") or not row.get("competitor"):
                continue
            if row.get("seed", {}).get("failed"):
                continue
            Gamma = float(row["Gamma"])
            gamma = float(row["gamma"])
            seed_E = float(row["seed"]["full_exponent"])
            comp_E = float(row["competitor"]["full_exponent"])
            cache_key = (round(float(result["beta"]), 10), round(gamma, 10))
            if cache_key not in lambda_cache:
                lambda_cache[cache_key] = lambda_abs_by_n(Q, K_CLAUSE, R, float(result["beta"]), gamma, n_values)
            lam_by_n = lambda_cache[cache_key]
            for n in n_values:
                lam = float(lam_by_n[int(n)])
                Dn = abs(comp_E - lam) - abs(seed_E - lam)
                series_by_n[int(n)].append((Gamma, float(Dn)))

        per_n = {int(n): _crossing_from_series(series_by_n[int(n)]) for n in n_values}
        bracketed = [
            float(item["Gamma_cross"])
            for item in per_n.values()
            if item.get("status") in {"bracketed", "multiple_sign_changes"} and item.get("Gamma_cross") is not None
        ]
        out.append({
            "beta": beta,
            "matched_pl_beta": float(result["beta"]),
            "status": "ok",
            "per_n": per_n,
            "range": (float(max(bracketed) - min(bracketed)) if len(bracketed) == len(n_values) else None),
            "n_valid_Gamma": max((len(v) for v in series_by_n.values()), default=0),
        })

    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "finite_n_crossover_stability.json").open("w") as f:
        json.dump({
            "n_values": [int(n) for n in n_values],
            "clean_betas": clean_betas,
            "nonmonotone_betas": nonmonotone_betas,
            "rows": out,
        }, f, indent=2)
    return out, nonmonotone_betas


def _valid_branch_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    return sorted(
        [
            row for row in result.get("rows", [])
            if not row.get("competitor_failed")
            and not row.get("competitor_merged_with_seed")
            and row.get("competitor")
            and not row.get("seed", {}).get("failed")
        ],
        key=lambda row: float(row["Gamma"]),
    )


def _interp_from_rows(branch_rows: list[dict[str, Any]], Gamma: float, selector: str) -> float:
    if not branch_rows:
        return float("nan")
    xs = np.array([float(row["Gamma"]) for row in branch_rows], dtype=float)
    if selector == "seed_E":
        ys = np.array([float(row["seed"]["full_exponent"]) for row in branch_rows], dtype=float)
    elif selector == "comp_E":
        ys = np.array([float(row["competitor"]["full_exponent"]) for row in branch_rows], dtype=float)
    elif selector == "delta_re":
        ys = np.array([float(row["delta_re_phi"]) for row in branch_rows], dtype=float)
    elif selector == "delta_im_raw":
        ys = np.array([float(row["delta_im_phi_raw"]) for row in branch_rows], dtype=float)
    else:
        raise ValueError(selector)
    order = np.argsort(xs)
    return float(np.interp(float(Gamma), xs[order], ys[order]))


def _dn_mechanism(seed_E: float, comp_E: float, lam: float) -> tuple[str, str, bool]:
    lo = min(seed_E, comp_E)
    hi = max(seed_E, comp_E)
    if lam < lo:
        ordering = "lambda_n below both saddle rates"
        mechanism = "D_n = E_comp - E_seed"
        algebraic = True
    elif lam > hi:
        ordering = "lambda_n above both saddle rates"
        mechanism = "D_n = E_seed - E_comp"
        algebraic = True
    else:
        ordering = "lambda_n between the saddle rates"
        if seed_E <= lam <= comp_E:
            mechanism = "D_n = E_comp + E_seed - 2 lambda_n"
        else:
            mechanism = "D_n = 2 lambda_n - E_comp - E_seed"
        algebraic = False
    return ordering, mechanism, algebraic


def finite_n_mechanism_audit_rows(finite_n_rows: list[dict[str, Any]], n_values: list[int]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for stab in finite_n_rows:
        if stab.get("status") != "ok":
            continue
        beta = float(stab["beta"])
        result = _nearest_pl_result(beta)
        if result is None:
            continue
        branch_rows = _valid_branch_rows(result)
        for n in n_values:
            item = stab.get("per_n", {}).get(int(n), {})
            if item.get("status") not in {"bracketed", "multiple_sign_changes"}:
                continue
            Gamma = float(item["Gamma_cross"])
            gamma = -Gamma
            seed_E = _interp_from_rows(branch_rows, Gamma, "seed_E")
            comp_E = _interp_from_rows(branch_rows, Gamma, "comp_E")
            lam = lambda_abs_by_n(Q, K_CLAUSE, R, float(result["beta"]), gamma, [int(n)])[int(n)]
            ordering, mechanism, algebraic = _dn_mechanism(seed_E, comp_E, lam)
            out.append({
                "beta": beta,
                "n": int(n),
                "Gamma_cross": Gamma,
                "E_seed": seed_E,
                "E_comp": comp_E,
                "lambda_n": float(lam),
                "min_E": float(min(seed_E, comp_E)),
                "max_E": float(max(seed_E, comp_E)),
                "ordering": ordering,
                "D_n_mechanism": mechanism,
                "algebraic_equal_action": bool(algebraic),
            })
    with (OUT / "finite_n_mechanism_audit.json").open("w") as f:
        json.dump({"rows": out}, f, indent=2)
    return out


def interval_text(lo: float | None, hi: float | None) -> str:
    if lo is None and hi is None:
        return "-"
    if lo is None:
        return f"(-inf,{float(hi):.4f}]"
    if hi is None:
        return f"[{float(lo):.4f},+inf)"
    return f"[{float(lo):.4f},{float(hi):.4f}]"


def interval_overlap(a_lo: float | None, a_hi: float | None, b_lo: float | None, b_hi: float | None) -> bool | None:
    if a_lo is None or a_hi is None or b_lo is None or b_hi is None:
        return None
    return max(float(a_lo), float(b_lo)) <= min(float(a_hi), float(b_hi))


def interval_midpoint_distance(a_lo: float | None, a_hi: float | None, b_lo: float | None, b_hi: float | None) -> float | None:
    if a_lo is None or a_hi is None or b_lo is None or b_hi is None:
        return None
    return abs(0.5 * (float(a_lo) + float(a_hi)) - 0.5 * (float(b_lo) + float(b_hi)))


def interval_separation(a_lo: float | None, a_hi: float | None, b_lo: float | None, b_hi: float | None) -> float | None:
    if a_lo is None or a_hi is None or b_lo is None or b_hi is None:
        return None
    if interval_overlap(a_lo, a_hi, b_lo, b_hi):
        return 0.0
    return min(abs(float(a_lo) - float(b_hi)), abs(float(b_lo) - float(a_hi)))


def interval_comparison_rows(finite_n_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    anti_by_beta, stokes_by_beta = condition_candidates_by_beta()
    family = _nearest_stokes_family(stokes_by_beta, anti_by_beta)
    rate_rows = []
    phase_rows = []
    for stab in finite_n_rows:
        if stab.get("status") != "ok":
            continue
        beta = float(stab["beta"])
        anti = anti_by_beta.get(beta, [])
        if not anti:
            continue
        anti_item = anti[0]
        cross = stab.get("per_n", {}).get(24, {})
        rate_rows.append({
            "beta": beta,
            "I_AS_lo": float(anti_item["lo"]),
            "I_AS_hi": float(anti_item["hi"]),
            "I_cross_lo": cross.get("lo"),
            "I_cross_hi": cross.get("hi"),
            "overlap": interval_overlap(float(anti_item["lo"]), float(anti_item["hi"]), cross.get("lo"), cross.get("hi")),
            "midpoint_distance": interval_midpoint_distance(float(anti_item["lo"]), float(anti_item["hi"]), cross.get("lo"), cross.get("hi")),
        })
        selected = [
            stokes_by_beta[b][idx]
            for b, idx in family
            if math.isclose(float(b), beta, abs_tol=1e-9) and idx < len(stokes_by_beta[b])
        ]
        if selected:
            stokes_item = selected[0]
            phase_rows.append({
                "beta": beta,
                "I_AS_lo": float(anti_item["lo"]),
                "I_AS_hi": float(anti_item["hi"]),
                "I_S_lo": float(stokes_item["lo"]),
                "I_S_hi": float(stokes_item["hi"]),
                "overlap": interval_overlap(float(anti_item["lo"]), float(anti_item["hi"]), float(stokes_item["lo"]), float(stokes_item["hi"])),
                "interval_separation": interval_separation(float(anti_item["lo"]), float(anti_item["hi"]), float(stokes_item["lo"]), float(stokes_item["hi"])),
            })
    with (OUT / "interval_comparisons.json").open("w") as f:
        json.dump({"rate_vs_anti_stokes": rate_rows, "stokes_vs_anti_stokes": phase_rows}, f, indent=2)
    return rate_rows, phase_rows


def _complex_from_lists(re: list[float], im: list[float]) -> np.ndarray:
    return np.asarray(re, dtype=float) + 1j * np.asarray(im, dtype=float)


def _row_comp_z(row: dict[str, Any]) -> np.ndarray | None:
    comp = row.get("competitor", {})
    if not comp.get("z_real") or not comp.get("z_imag"):
        return None
    return _complex_from_lists(comp["z_real"], comp["z_imag"])


def _seed_eval_from_hint(beta: float, gamma: float, seed_hint: np.ndarray | None) -> tuple[np.ndarray, dict[str, Any]]:
    if seed_hint is not None:
        z_seed, ok = seed_from_previous(seed_hint, beta, gamma)
        if z_seed is not None and ok:
            sys_ = SaddleSystem.build(q=Q, r=R, betas=np.array([beta]), gammas=np.array([gamma]))
            phi = compute_phi(z_seed, q=Q, r=R, betas=sys_.betas, gammas=sys_.gammas)
            full = conv2_full_exponent(float(phi.real), K_CLAUSE, R)
            return z_seed, {
                "failed": False,
                "re_phi": float(phi.real),
                "im_phi": float(phi.imag),
                "full_exponent": float(full),
                "residual_inf": float(np.linalg.norm(sys_.G_complex(z_seed), ord=np.inf)),
            }
    z_seed, srow = certified_seed_row(beta, gamma)
    return z_seed, srow


def _evaluate_equal_action_endpoint(
    beta: float,
    Gamma: float,
    comp_hint: np.ndarray,
    seed_hint: np.ndarray | None = None,
) -> tuple[np.ndarray | None, np.ndarray | None, dict[str, Any]]:
    gamma = -float(Gamma)
    try:
        z_seed, srow = _seed_eval_from_hint(beta, gamma, seed_hint)
        z_comp, crow = continue_competitor_to(beta, gamma, comp_hint)
        if z_comp is None or crow.get("failed"):
            return None, z_seed, {
                "Gamma": float(Gamma),
                "failed": True,
                "failure_reason": crow.get("failure_reason", "competitor_continuation_failed"),
                "seed_residual_inf": float(srow.get("residual_inf", math.nan)),
                "comp_residual_inf": float(crow.get("residual_inf", math.nan)),
                "comp_certified": bool(crow.get("certified", False)),
                "z_step_dist": float(crow.get("z_step_dist", math.inf)),
            }
        z_dist_seed = float(np.linalg.norm(z_comp - z_seed, ord=np.inf))
        f_as = float(crow["re_phi"] - srow["re_phi"])
        return z_comp, z_seed, {
            "Gamma": float(Gamma),
            "failed": False,
            "f_AS": f_as,
            "seed_E": float(srow["full_exponent"]),
            "comp_E": float(crow["full_exponent"]),
            "seed_residual_inf": float(srow["residual_inf"]),
            "comp_residual_inf": float(crow["residual_inf"]),
            "comp_certified": bool(crow.get("certified", False)),
            "z_step_dist": float(crow.get("z_step_dist", math.inf)),
            "z_distance_from_seed": z_dist_seed,
            "comp_z_real": z_comp.real.tolist(),
            "comp_z_imag": z_comp.imag.tolist(),
        }
    except Exception as exc:
        return None, None, {
            "Gamma": float(Gamma),
            "failed": True,
            "failure_reason": repr(exc),
        }


def _find_initial_as_bracket(result: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]] | None:
    rows = _valid_branch_rows(result)
    for lo_row, hi_row in zip(rows[:-1], rows[1:]):
        flo = float(lo_row["delta_re_phi"])
        fhi = float(hi_row["delta_re_phi"])
        if flo == 0.0 or flo * fhi <= 0.0:
            return lo_row, hi_row
    return None


def refine_equal_action_brackets(target_betas: list[float], width_tol: float = 1e-3) -> list[dict[str, Any]]:
    refined_rows: list[dict[str, Any]] = []
    for target_beta in target_betas:
        result = _nearest_pl_result(target_beta)
        if result is None:
            refined_rows.append({"beta": float(target_beta), "status": "missing_pl_branch"})
            continue
        beta = float(result["beta"])
        bracket = _find_initial_as_bracket(result)
        if bracket is None:
            refined_rows.append({"beta": beta, "status": "missing_sign_change_bracket"})
            continue
        lo_row, hi_row = bracket
        lo = float(lo_row["Gamma"])
        hi = float(hi_row["Gamma"])
        z_lo_hint = _row_comp_z(lo_row)
        z_hi_hint = _row_comp_z(hi_row)
        if z_lo_hint is None or z_hi_hint is None:
            refined_rows.append({"beta": beta, "status": "missing_endpoint_z"})
            continue
        z_lo, seed_lo, lo_eval = _evaluate_equal_action_endpoint(beta, lo, z_lo_hint)
        z_hi, seed_hi, hi_eval = _evaluate_equal_action_endpoint(beta, hi, z_hi_hint)
        if z_lo is None or seed_lo is None or z_hi is None or seed_hi is None or lo_eval.get("failed") or hi_eval.get("failed"):
            refined_rows.append({"beta": beta, "status": "endpoint_certification_failed", "lo": lo_eval, "hi": hi_eval})
            continue
        if float(lo_eval["f_AS"]) * float(hi_eval["f_AS"]) > 0.0:
            refined_rows.append({"beta": beta, "status": "endpoint_signs_do_not_bracket", "lo": lo_eval, "hi": hi_eval})
            continue

        left = {"Gamma": lo, "z": z_lo, "seed_z": seed_lo, "eval": lo_eval}
        right = {"Gamma": hi, "z": z_hi, "seed_z": seed_hi, "eval": hi_eval}
        iterations = 0
        failures: list[dict[str, Any]] = []
        while abs(float(right["Gamma"]) - float(left["Gamma"])) > width_tol and iterations < 32:
            mid = 0.5 * (float(left["Gamma"]) + float(right["Gamma"]))
            hint_side = left if abs(mid - float(left["Gamma"])) <= abs(float(right["Gamma"]) - mid) else right
            z_mid, seed_mid, mid_eval = _evaluate_equal_action_endpoint(beta, mid, hint_side["z"], hint_side["seed_z"])
            iterations += 1
            if z_mid is None or seed_mid is None or mid_eval.get("failed"):
                failures.append(mid_eval)
                break
            if float(left["eval"]["f_AS"]) * float(mid_eval["f_AS"]) <= 0.0:
                right = {"Gamma": mid, "z": z_mid, "seed_z": seed_mid, "eval": mid_eval}
            else:
                left = {"Gamma": mid, "z": z_mid, "seed_z": seed_mid, "eval": mid_eval}

        status = "ok" if not failures and abs(float(right["Gamma"]) - float(left["Gamma"])) <= width_tol else "partial"
        refined_rows.append({
            "beta": beta,
            "target_beta": float(target_beta),
            "status": status,
            "Gamma_lo": float(left["Gamma"]),
            "Gamma_hi": float(right["Gamma"]),
            "midpoint": 0.5 * (float(left["Gamma"]) + float(right["Gamma"])),
            "width": abs(float(right["Gamma"]) - float(left["Gamma"])),
            "f_AS_lo": float(left["eval"]["f_AS"]),
            "f_AS_hi": float(right["eval"]["f_AS"]),
            "lo": {k: v for k, v in left["eval"].items() if not k.startswith("comp_z_")},
            "hi": {k: v for k, v in right["eval"].items() if not k.startswith("comp_z_")},
            "iterations": iterations,
            "failures": failures,
            "label": "equal-real-action brackets with Krawczyk-certified endpoint saddles",
        })

    with (OUT / "equal_real_action_refined_brackets.json").open("w") as f:
        json.dump({"width_tol": width_tol, "rows": refined_rows}, f, indent=2)
    return refined_rows


def resolve_left_censored_beta035() -> dict[str, Any]:
    result = _nearest_pl_result(0.35)
    if result is None:
        return {"beta": 0.35, "status": "missing_pl_branch"}
    branch_rows = _valid_branch_rows(result)
    start = next((row for row in branch_rows if math.isclose(float(row["Gamma"]), 1.70, abs_tol=1e-9)), None)
    if start is None:
        return {"beta": 0.35, "status": "missing_start_row"}
    z = _row_comp_z(start)
    if z is None:
        return {"beta": 0.35, "status": "missing_start_z"}
    points = []
    previous = z
    seed_previous: np.ndarray | None = None
    for Gamma in [1.65, 1.60, 1.55]:
        z_next, seed_next, row = _evaluate_equal_action_endpoint(0.35, Gamma, previous, seed_previous)
        points.append(row)
        if z_next is None or row.get("failed"):
            status = "continuation_failed"
            out = {"beta": 0.35, "status": status, "points": points, "failure": row}
            with (OUT / "beta035_left_censor_resolution.json").open("w") as f:
                json.dump(out, f, indent=2)
            return out
        previous = z_next
        seed_previous = seed_next
        if float(row.get("z_distance_from_seed", math.inf)) < 1e-4:
            out = {"beta": 0.35, "status": "continued_branch_merged_with_seed", "points": points}
            with (OUT / "beta035_left_censor_resolution.json").open("w") as f:
                json.dump(out, f, indent=2)
            return out
        if float(row["f_AS"]) >= 0.0:
            out = {"beta": 0.35, "status": "sign_change_found_or_touched", "points": points}
            with (OUT / "beta035_left_censor_resolution.json").open("w") as f:
                json.dump(out, f, indent=2)
            return out
    out = {"beta": 0.35, "status": "no_sign_change_through_1p55", "points": points}
    with (OUT / "beta035_left_censor_resolution.json").open("w") as f:
        json.dump(out, f, indent=2)
    return out


def format_crossing_cell(item: dict[str, Any]) -> str:
    status = item.get("status")
    value = item.get("Gamma_cross")
    if value is None:
        return "-"
    if status == "left_censored":
        return f"<= {float(value):.4f}"
    if status == "right_censored":
        return f">= {float(value):.4f}"
    if status == "multiple_sign_changes":
        return f"{float(value):.4f}*"
    if status == "bracketed":
        return f"{float(value):.4f}"
    return "-"


def plot_finite_n_stability(stability_rows: list[dict[str, Any]], n_values: list[int]) -> str:
    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    cmap = plt.get_cmap("tab10")
    for idx, row in enumerate(stability_rows):
        if row.get("status") != "ok":
            continue
        beta = float(row["beta"])
        xs: list[int] = []
        ys: list[float] = []
        upper_xs: list[int] = []
        upper_ys: list[float] = []
        for n in n_values:
            item = row["per_n"].get(int(n), {})
            value = item.get("Gamma_cross")
            if value is None:
                continue
            if item.get("status") == "left_censored":
                upper_xs.append(int(n))
                upper_ys.append(float(value))
            elif item.get("status") in {"bracketed", "multiple_sign_changes"}:
                xs.append(int(n))
                ys.append(float(value))
        color = cmap(idx % 10)
        if xs:
            ax.plot(xs, ys, marker="o", lw=1.5, color=color, label=rf"$\beta={beta:.4g}$")
        if upper_xs:
            ax.plot(
                upper_xs,
                upper_ys,
                marker="v",
                lw=1.0,
                ls=":",
                color=color,
                label=rf"$\beta={beta:.4g}$ upper bound",
            )
    ax.set_xlabel(r"$n$")
    ax.set_ylabel(r"$\Gamma_{\mathrm{cross}}(\beta;n)$")
    ax.set_title(r"Finite-$n$ crossover stability on the validated tracked branch")
    ax.grid(True, alpha=0.25)
    ax.set_xticks([int(n) for n in n_values])
    ax.legend(loc="best", fontsize=8, frameon=True, ncols=2)
    fig.tight_layout()
    path = OUT / "finite_n_crossover_stability.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def plot_separated_diagnostics(grid: dict[str, np.ndarray], transitions: list[dict[str, Any]]) -> str:
    OUT.mkdir(parents=True, exist_ok=True)
    bx, by, be, ax, ay = robust_points(transitions)
    phase_robust = phase_anchor_robustness()
    cond = pl_condition_points(show_family=bool(phase_robust["robust"]))
    beta060_roots_path = OUT / "beta060_equal_action_roots.json"
    beta060_roots: list[dict[str, Any]] = []
    beta060_scans: dict[str, list[dict[str, Any]]] = {}
    if beta060_roots_path.exists():
        beta060_data = load_json(beta060_roots_path)
        beta060_roots = beta060_data.get("roots", [])
        beta060_scans = beta060_data.get("scan_rows", {})
    refined_as_path = OUT / "equal_real_action_refined_brackets.json"
    refined_as_rows = []
    if refined_as_path.exists():
        refined_as_rows = [
            row for row in load_json(refined_as_path).get("rows", [])
            if row.get("status") in {"ok", "partial"} and row.get("Gamma_lo") is not None
        ]

    fig, axes = plt.subplots(1, 2, figsize=(14.2, 5.8), sharex=True, sharey=True)
    X = pcolor_edges(grid["betas"])
    Y = pcolor_edges(grid["Gammas"])

    D = grid["residual_delta"]
    finite_D = D[np.isfinite(D)]
    vmax = float(np.nanpercentile(np.abs(finite_D), 90)) if finite_D.size else 1.0
    vmax = max(vmax, 1e-3)
    norm = mcolors.TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax)
    mesh = axes[0].pcolormesh(X, Y, D, cmap="RdBu_r", norm=norm, shading="flat", alpha=0.82)
    axes[0].pcolormesh(X, Y, np.zeros_like(D), facecolor="none", edgecolor="#ffffff", linewidth=0.35, shading="flat")
    no_possible = np.where(grid["n_possible"] <= 0.0, 1.0, np.nan)
    axes[0].contourf(
        grid["betas"],
        grid["Gammas"],
        no_possible,
        levels=[0.5, 1.5],
        colors="none",
        hatches=["////"],
    )
    if len(bx):
        axes[0].errorbar(
            bx,
            by,
            yerr=be,
            fmt="o",
            color=PURPLE,
            ecolor="#2d3748",
            capsize=3,
            ms=5,
            label=r"equal-rate boundary from $D_{24}$",
            zorder=5,
        )
        _plot_guideline_segments(axes[0], bx, by, color=PURPLE, lw=1.4, alpha=0.7, zorder=4)
    ax_non = np.asarray([x for x in ax if not math.isclose(float(x), 0.60, abs_tol=5e-9)])
    ay_non = np.asarray([float(y) for x, y in zip(ax, ay) if not math.isclose(float(x), 0.60, abs_tol=5e-9)])
    if len(ax_non):
        axes[0].scatter(ax_non, ay_non, marker="x", color=RED, s=70, linewidths=2.0, label="nonmonotone slice", zorder=5)
    branch_markers = {"branch_left": "^", "branch_right": "D", "branch_A": "o"}
    for root in beta060_roots:
        marker = branch_markers.get(str(root.get("branch_id")), "s")
        midpoint = float(root["midpoint"])
        yerr = np.array([[midpoint - float(root["Gamma_lo"])], [float(root["Gamma_hi"]) - midpoint]])
        axes[0].errorbar(
            [0.60],
            [midpoint],
            yerr=yerr,
            fmt=marker,
            markerfacecolor="white",
            markeredgecolor=RED,
            ecolor=RED,
            capsize=2,
            ms=6,
            lw=1.0,
            label=f"{root.get('branch_id')} partial",
            zorder=6,
        )
    cbar = fig.colorbar(mesh, ax=axes[0], shrink=0.88)
    cbar.set_label(r"$D_{24}=|\lambda_\mathrm{comp}-\lambda_{24}|-|\lambda_\mathrm{seed}-\lambda_{24}|$")
    axes[0].legend(
        handles=[
            mpatches.Patch(facecolor="#b2182b", alpha=0.65, label="seed gives smaller rate residual"),
            mpatches.Patch(facecolor="#2166ac", alpha=0.65, label="competitor gives smaller rate residual"),
            mlines.Line2D([], [], color=PURPLE, marker="o", linestyle="-", markersize=5, label=r"equal-rate boundary from $D_{24}$"),
            mlines.Line2D([], [], color=RED, marker="D", markerfacecolor="white", linestyle="None", markersize=6, label=r"$\beta=0.60$ branch partial"),
            mpatches.Patch(facecolor="white", hatch="////", edgecolor="#4a5568", label="no admissible competitor found"),
        ],
        loc="lower left",
        fontsize=8,
        frameon=True,
    )
    axes[0].set_title(r"(a) Rate residuals of certified saddles")

    axes[1].set_facecolor("#f7fafc")
    for x in X:
        axes[1].axvline(float(x), color="#e2e8f0", lw=0.45, zorder=0)
    for y in Y:
        axes[1].axhline(float(y), color="#e2e8f0", lw=0.45, zorder=0)
    if refined_as_rows:
        as_beta = np.array([float(row["beta"]) for row in refined_as_rows])
        as_mid = np.array([float(row["midpoint"]) for row in refined_as_rows])
        as_err_low = as_mid - np.array([float(row["Gamma_lo"]) for row in refined_as_rows])
        as_err_high = np.array([float(row["Gamma_hi"]) for row in refined_as_rows]) - as_mid
        axes[1].errorbar(
            as_beta,
            as_mid,
            yerr=np.vstack([as_err_low, as_err_high]),
            fmt="o",
            color="#1a202c",
            ecolor="#1a202c",
            elinewidth=1.0,
            capsize=2,
            ms=4,
            label="equal-action bracket",
            zorder=4,
        )
    for root in beta060_roots:
        marker = branch_markers.get(str(root.get("branch_id")), "s")
        midpoint = float(root["midpoint"])
        yerr = np.array([[midpoint - float(root["Gamma_lo"])], [float(root["Gamma_hi"]) - midpoint]])
        axes[1].errorbar(
            [0.60],
            [midpoint],
            yerr=yerr,
            fmt=marker,
            color="#1a202c",
            markerfacecolor="white",
            ecolor="#1a202c",
            capsize=2,
            ms=5,
            label=f"{root.get('branch_id')} partial",
            zorder=5,
        )
    if len(cond["stokes_other_beta"]):
        axes[1].scatter(
            cond["stokes_other_beta"],
            cond["stokes_other_gamma"],
            marker="s",
            color="#a0aec0",
            s=16,
            alpha=0.55,
            label="phase candidates",
            zorder=2,
        )
    if len(cond["stokes_family_beta"]):
        order = np.argsort(cond["stokes_family_beta"])
        axes[1].errorbar(
            cond["stokes_family_beta"][order],
            cond["stokes_family_gamma"][order],
            yerr=np.vstack([cond["stokes_family_err_low"][order], cond["stokes_family_err_high"][order]]),
            fmt="s--",
            color=RED,
            ecolor="#9b2c2c",
            elinewidth=1.0,
            capsize=2,
            lw=1.2,
            ms=4,
            label="selected phase-alignment family (anchor-robust)",
            zorder=5,
        )
    if beta060_scans:
        axins = axes[1].inset_axes([0.08, 0.64, 0.38, 0.29])
        for idx, (branch_id, rows) in enumerate(beta060_scans.items()):
            valid = [
                row for row in rows
                if not row.get("failed") and math.isfinite(float(row.get("f_AS", float("nan"))))
            ]
            valid = [row for row in valid if 1.835 <= float(row["Gamma"]) <= 1.875]
            valid.sort(key=lambda row: float(row["Gamma"]))
            if not valid:
                continue
            color = RED if idx else PURPLE
            axins.plot(
                [float(row["Gamma"]) for row in valid],
                [float(row["f_AS"]) for row in valid],
                marker="o",
                ms=2,
                lw=1.0,
                color=color,
                label=str(branch_id).replace("branch_", ""),
            )
        axins.axhline(0.0, color="#1a202c", lw=0.8)
        axins.set_title(r"$\beta=0.60$", fontsize=8)
        axins.tick_params(labelsize=7, pad=1)
        axins.grid(True, alpha=0.2)
        axins.legend(fontsize=6, frameon=False, loc="best")
    axes[1].set_title("(b) Slice-wise PL necessary-condition candidates")
    axes[1].legend(loc="lower right", fontsize=8, frameon=True)

    for axis in axes:
        axis.axvline(BETA_OPT, color=GRAY, ls="--", lw=1.0)
        axis.text(BETA_OPT + 0.004, 2.095, r"$\beta_\mathrm{opt}$", color=GRAY, fontsize=9, va="top")
        axis.set_xlabel(r"$\beta$")
        axis.set_xlim(0.33, 0.67)
        axis.set_ylim(1.43, 2.12)
    axes[0].set_ylabel(r"$\Gamma \equiv -\gamma$")
    fig.suptitle(r"Saddle crossover beyond the small-$\gamma$ regime", y=0.99)
    fig.tight_layout()
    path = OUT / "separated_crossover_and_pl_conditions.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    fig.savefig(OUT / "separated_crossover_and_pl_conditions.pdf", bbox_inches="tight")
    plt.close(fig)
    return str(path)


def plot_map(grid: dict[str, np.ndarray], transitions: list[dict[str, Any]]) -> str:
    OUT.mkdir(parents=True, exist_ok=True)
    bx, by, be, ax, ay = robust_points(transitions)
    fig, ax0 = plt.subplots(figsize=(11.2, 6.4))
    cmap = mcolors.ListedColormap(["#cfe8ff", "#efe1b2", "#d8f0d2", "#ffd1a6"])
    norm = mcolors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)
    mesh = ax0.pcolormesh(
        pcolor_edges(grid["betas"]),
        pcolor_edges(grid["Gammas"]),
        grid["class"],
        cmap=cmap,
        norm=norm,
        shading="flat",
    )
    cbar = fig.colorbar(mesh, ax=ax0, ticks=[0, 1, 2, 3])
    cbar.ax.set_yticklabels([
        "seed smaller residual",
        "no admissible high-Re",
        "certified but rate-inconsistent",
        "competitor smaller residual",
    ])

    if len(bx):
        ax0.errorbar(bx, by, yerr=be, fmt="o-", color=PURPLE, ecolor="#718096", capsize=3, lw=1.6, ms=4, label=r"finite-$n$ seed-competitor crossover")
    if len(ax):
        ax0.scatter(ax, ay, marker="x", color=RED, s=70, linewidths=2.0, label=r"nonmonotone finite-$n$ slice", zorder=4)

    for s in pl_summary():
        beta = float(s["beta"])
        if s["status"] == "ok":
            for lo, hi in s.get("anti_stokes_intervals", []):
                ax0.plot([beta, beta], [lo, hi], color=ORANGE, lw=4.5, alpha=0.72, solid_capstyle="round")
            y = np.mean(s["physical_switch_intervals"][0]) if s.get("physical_switch_intervals") else float(s.get("anchor_Gamma", math.nan))
            marker = "o" if s.get("pl_aligned") else "x"
            color = GREEN if s.get("pl_aligned") else RED
            ax0.scatter(beta, y, marker=marker, color=color, s=80, edgecolor="white" if marker == "o" else color, linewidth=1.4, zorder=5)
        else:
            ax0.scatter(beta, float(s.get("anchor_Gamma", math.nan)), marker="x", color=RED, s=90, linewidths=2.4, zorder=5)

    ax0.plot([], [], color=ORANGE, lw=4.5, alpha=0.72, label=r"equal-real-action bracket")
    ax0.scatter([], [], color=GREEN, s=80, edgecolor="white", label="Stokes condition overlaps crossover")
    ax0.axvline(BETA_OPT, color=GRAY, ls="--", lw=1.0, label=r"$\beta_\mathrm{opt}$")
    ax0.set_xlabel(r"$\beta$")
    ax0.set_ylabel(r"$\Gamma\equiv-\gamma$")
    ax0.set_title(r"Filtered saddle-rate crossover")
    ax0.set_xlim(0.33, 0.67)
    ax0.set_ylim(1.43, 2.12)
    ax0.legend(loc="upper right", fontsize=8.5, frameon=True)
    fig.tight_layout()
    path = OUT / "possible_exponent_transition_map_with_pl.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def plot_boundary(transitions: list[dict[str, Any]]) -> str:
    bx, by, be, ax, ay = robust_points(transitions)
    fig, ax0 = plt.subplots(figsize=(9.4, 5.0))
    if len(bx):
        ax0.errorbar(bx, by, yerr=be, fmt="o", color=PURPLE, ecolor="#718096", capsize=3, ms=5, label=r"finite-$n$ crossover")
        ax0.plot(bx, by, color=PURPLE, lw=1.2, alpha=0.55)
    if len(ax):
        ax0.scatter(ax, ay, marker="x", color=RED, s=72, linewidths=2.0, label=r"nonmonotone finite-$n$ slice")

    for s in pl_summary():
        beta = float(s["beta"])
        if s["status"] != "ok":
            ax0.scatter(beta, float(s.get("anchor_Gamma", math.nan)), marker="x", color=RED, s=85, linewidths=2.2)
            continue
        for lo, hi in s.get("anti_stokes_intervals", []):
            ax0.plot([beta, beta], [lo, hi], color=ORANGE, lw=4.5, alpha=0.78, solid_capstyle="round")
        if s.get("pl_aligned"):
            y = np.mean(s["physical_switch_intervals"][0]) if s.get("physical_switch_intervals") else float(s["anchor_Gamma"])
            ax0.scatter(beta, y, color=GREEN, s=65, edgecolor="white", linewidth=1.1, zorder=5)

    ax0.plot([], [], color=ORANGE, lw=4.5, alpha=0.78, label=r"equal-real-action bracket")
    ax0.scatter([], [], color=GREEN, s=65, edgecolor="white", label="Stokes condition overlaps crossover")
    ax0.axvline(BETA_OPT, color=GRAY, ls="--", lw=1.0)
    ax0.set_xlabel(r"$\beta$")
    ax0.set_ylabel(r"transition $\Gamma\equiv-\gamma$")
    ax0.set_title(r"Filtered finite-$n$ crossover")
    ax0.grid(True, alpha=0.25)
    ax0.legend(loc="best", fontsize=9, frameon=True)
    fig.tight_layout()
    path = OUT / "possible_exponent_boundary_curve_with_pl.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def plot_counts(grid: dict[str, np.ndarray]) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.0), sharey=True)
    X = pcolor_edges(grid["betas"])
    Y = pcolor_edges(grid["Gammas"])
    im0 = axes[0].pcolormesh(X, Y, grid["n_possible"], cmap="Greens", shading="flat")
    axes[0].set_title(r"Certified competitors with $E \leq 0$")
    axes[0].set_xlabel(r"$\beta$")
    axes[0].set_ylabel(r"$\Gamma=-\gamma$")
    fig.colorbar(im0, ax=axes[0], label="# possible")
    im1 = axes[1].pcolormesh(X, Y, grid["n_impossible"], cmap="Reds", shading="flat")
    axes[1].set_title(r"Certified competitors removed by $E>0$")
    axes[1].set_xlabel(r"$\beta$")
    fig.colorbar(im1, ax=axes[1], label="# impossible")
    for ax in axes:
        ax.axvline(BETA_OPT, color=GRAY, ls="--", lw=1.0)
    fig.tight_layout()
    path = OUT / "possible_vs_impossible_competitor_counts.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def plot_gap(grid: dict[str, np.ndarray]) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.0), sharey=True)
    X = pcolor_edges(grid["betas"])
    Y = pcolor_edges(grid["Gammas"])
    diff = grid["max_all"] - grid["max_possible"]
    im0 = axes[0].pcolormesh(X, Y, diff, cmap="magma", shading="flat")
    axes[0].set_title("Rate removed by impossible high-Re filter")
    axes[0].set_xlabel(r"$\beta$")
    axes[0].set_ylabel(r"$\Gamma=-\gamma$")
    fig.colorbar(im0, ax=axes[0], label=r"$E_\mathrm{max\,all}-E_\mathrm{max\,possible}$")
    log_gap = np.log10(np.clip(grid["gap"], 1e-8, None))
    im1 = axes[1].pcolormesh(X, Y, log_gap, cmap="viridis_r", shading="flat")
    axes[1].set_title("Best exact match after possible-exponent filter")
    axes[1].set_xlabel(r"$\beta$")
    fig.colorbar(im1, ax=axes[1], label=r"$\log_{10}|E_\mathrm{best}-\lambda_n|$")
    for ax in axes:
        ax.axvline(BETA_OPT, color=GRAY, ls="--", lw=1.0)
    fig.tight_layout()
    path = OUT / "possible_filter_removed_rate_and_exact_gap.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def write_report(
    rows: list[dict[str, Any]],
    transitions: list[dict[str, Any]],
    figures: list[str],
    metadata: dict[str, Any],
    decoy_margin: float,
    finite_n_rows: list[dict[str, Any]],
    finite_n_nonmonotone_betas: list[float],
    n_values: list[int],
    mechanism_rows: list[dict[str, Any]],
    refined_as_rows: list[dict[str, Any]],
    rate_interval_rows: list[dict[str, Any]],
    phase_interval_rows: list[dict[str, Any]],
    phase_robustness: dict[str, Any],
    beta035_resolution: dict[str, Any],
) -> str:
    ok = [r for r in rows if r.get("status") == "ok"]
    classes = [classify_possible(r, decoy_margin) for r in ok]
    total_cert = sum(int(r.get("n_certified_competitors_recomputed", 0)) for r in ok)
    total_possible = sum(int(r.get("n_possible_competitors", 0)) for r in ok)
    total_impossible = sum(int(r.get("n_impossible_competitors", 0)) for r in ok)
    removed_controller = sum(1 for r in ok if r.get("removed_impossible_high_re_controller"))
    lines = [
        "# Rate-admissible refined saddle audit",
        "",
        "This redraws the refined transition figures after filtering certified competitors whose standalone conv2 exponent `E` is positive. Saddles with `E > 0` were excluded from the rate-admissible single-saddle set. Such saddles could only be compatible with a bounded observable if their thimble coefficient vanishes or exponentially precise cancellations occur.",
        "",
        "Important caveat: these figures compare certified saddle rates and necessary Stokes/anti-Stokes-style conditions.  They do not compute thimble intersection numbers or prove that a competitor controls the original BM24 contour.",
        "",
        "Panel (a) colors `D_24 = |lambda_comp - lambda_24| - |lambda_seed - lambda_24|`, using the existing `n=max(n_values)=24` finite-n reference from the refined scan. The finite-`n` stability section below recomputes `lambda_n` for `n=18..24` on the same tracked competitor branch.",
        "",
        "The purple connecting curve is a guide to the eye through separately bracketed finite-n crossover slices, not a certified two-parameter branch continuation.",
        "",
        "## Filter",
        "",
        f"- possible exponent criterion: `E <= {metadata['possible_exponent_tol']}`",
        f"- beta points: {len(metadata['beta_grid'])}",
        f"- Gamma points: {len(metadata['Gamma_grid'])}",
        f"- competitor starts per recomputed point: {metadata['starts']}",
        f"- certified competitor roots recomputed: {total_cert}",
        f"- possible-exponent competitors kept: {total_possible}",
        f"- positive-exponent competitors excluded as single-saddle controllers: {total_impossible}",
        f"- grid points where the all-root max-Re controller was removed by the filter: {removed_controller} / {len(ok)}",
        "",
        "Search completeness caveat: Krawczyk certification applies to roots that were found.  The finite random-start competitor search is not a proof that every saddle root was discovered.",
        "",
        "## Classification after filter",
        "",
        f"- seed gives smaller rate residual cleanly: {classes.count('seed')}",
        f"- seed gives smaller rate residual; only excluded high-Re saddles challenged it before filtering: {classes.count('impossible_only_decoy')}",
        f"- seed gives smaller rate residual despite certified rate-admissible challengers: {classes.count('possible_decoy')}",
        f"- rate-admissible competitor gives smaller rate residual: {classes.count('possible_competitor')}",
        "",
        "## Transition estimates after filter",
        "",
        "| beta | transition interval in Gamma | midpoint | notes |",
        "|---:|---:|---:|---|",
    ]
    for t in transitions:
        if not t["transitions"]:
            lines.append(f"| {t['beta']:.6f} | - | - | no possible-exponent transition on refined grid |")
            continue
        parts = [f"{tr['Gamma_low']:.4f}-{tr['Gamma_high']:.4f}" for tr in t["transitions"]]
        first = t["transitions"][0]
        note = (
            f"{t['n_possible_decoy']} possible-decoy, "
            f"{t['n_impossible_only_decoy']} impossible-only-decoy, "
            f"{t['n_possible_competitor']} possible-competitor points"
        )
        lines.append(f"| {t['beta']:.6f} | {', '.join(parts)} | {first['Gamma_mid']:.4f} | {note} |")
    identity_rows = competitor_identity_report_rows(rows, transitions)
    if identity_rows:
        lines += [
            "",
            "## Competitor identity check",
            "",
            "This compares panel (a)'s best discovered rate-admissible competitor on the competitor side of the crossover with panel (b)'s tracked competitor branch at the nearest stored Gamma.  The current distance is a direct z-coordinate sup-norm; explicit minimization over saddle-coordinate symmetries is not yet implemented.",
            "",
            "| beta | Gamma_rate | Gamma_tracked | d_z | classification |",
            "|---:|---:|---:|---:|---|",
        ]
        for row in identity_rows:
            d = float(row["d_z"])
            d_text = "nan" if not math.isfinite(d) else f"{d:.3e}"
            gt = row["Gamma_tracked"]
            gt_text = "nan" if gt is None or not math.isfinite(float(gt)) else f"{float(gt):.4f}"
            lines.append(
                f"| {row['beta']:.6f} | {row['Gamma_rate']:.4f} | {gt_text} | "
                f"{d_text} | {row['classification']} |"
            )
        if any(r["classification"] == "different competitor" for r in identity_rows):
            lines += [
                "",
                "Because at least one slice compares as a different competitor, the conservative interpretation is: the rate crossover and a slice-wise equal-action candidate occur at similar parameter values, although the globally consistent identity of the relevant competitor branch has not been established.",
            ]
    if finite_n_rows:
        header = "| beta | " + " | ".join(f"Gamma_cross({int(n)})" for n in n_values) + " | range |"
        sep = "|---:|" + "|".join("---:" for _ in n_values) + "|---:|"
        lines += [
            "",
            "## Finite-n crossover stability",
            "",
            "This uses the same validated tracked competitor branch for all `n`; it does not reselect the best competitor independently at each finite size and does not rerun saddle searches.  Only `lambda_n = n^{-1} log |P_n|` is recomputed.",
            "",
            header,
            sep,
        ]
        for row in finite_n_rows:
            if row.get("status") != "ok":
                values = ["-"] * len(n_values)
                range_text = row.get("status", "missing")
            else:
                values = [format_crossing_cell(row["per_n"].get(int(n), {})) for n in n_values]
                range_value = row.get("range")
                range_text = "not bracketed" if range_value is None else f"{float(range_value):.4f}"
            lines.append(f"| {float(row['beta']):.6f} | " + " | ".join(values) + f" | {range_text} |")
        if finite_n_nonmonotone_betas:
            beta_list = ", ".join(f"{b:.6f}" for b in finite_n_nonmonotone_betas)
            lines += [
                "",
                f"Nonmonotone rate slices kept separate from this table: `{beta_list}`.  No smooth crossover boundary is inferred through those slices.",
            ]
        lines += [
            "",
            "`<=` entries are left-censored by the tracked-branch cache: the first valid tracked-competitor row is already competitor-favored, so the crossing lies at or below the displayed Gamma.",
            "`*` marks a first positive-to-negative crossing in a series with multiple sign changes.",
        ]
    if mechanism_rows:
        all_algebraic = all(bool(row["algebraic_equal_action"]) for row in mechanism_rows)
        lines += [
            "",
            "## Absolute-value mechanism audit",
            "",
            "This audits why the finite-n rate-residual crossover agrees with the equal-real-action condition on the clean tracked branch.  At each displayed crossing, `lambda_n` lies outside the interval between the two saddle rates, so the absolute-value residual equation collapses algebraically to equality of the two saddle rates.  Thus the observed `Gamma_cross(n) = Gamma_AS` is an algebraic consequence in this regime, not an independent finite-n numerical coincidence." if all_algebraic else "At least one audited row has `lambda_n` between the saddle rates, so equality with the equal-real-action condition is not purely forced everywhere.",
            "",
            "| beta | n | E_seed | E_comp | lambda_n | ordering | D_n mechanism |",
            "|---:|---:|---:|---:|---:|---|---|",
        ]
        for row in mechanism_rows:
            lines.append(
                f"| {row['beta']:.6f} | {int(row['n'])} | {row['E_seed']:.8g} | "
                f"{row['E_comp']:.8g} | {row['lambda_n']:.8g} | {row['ordering']} | "
                f"{row['D_n_mechanism']} |"
            )
    if refined_as_rows:
        lines += [
            "",
            "## Equal-real-action endpoint refinement",
            "",
            "These are equal-real-action brackets with Krawczyk-certified endpoint saddles.  The saddle roots at the endpoints are certified; the scalar Gamma root is not claimed to be rigorously interval-certified because no interval proof in Gamma is implemented.",
            "",
            "| beta | Gamma_lo | Gamma_hi | midpoint | width | f_AS(lo) | f_AS(hi) | status |",
            "|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
        for row in refined_as_rows:
            if row.get("status") not in {"ok", "partial"}:
                lines.append(f"| {float(row.get('beta', row.get('target_beta', float('nan')))):.6f} | - | - | - | - | - | - | {row.get('status')} |")
                continue
            lines.append(
                f"| {row['beta']:.6f} | {row['Gamma_lo']:.6f} | {row['Gamma_hi']:.6f} | "
                f"{row['midpoint']:.6f} | {row['width']:.2e} | {row['f_AS_lo']:+.3e} | "
                f"{row['f_AS_hi']:+.3e} | {row['status']} |"
            )
    if rate_interval_rows:
        lines += [
            "",
            "## Interval comparisons",
            "",
            "Because `D_24 = 0` is algebraically equivalent to `Delta Re Phi = 0` in the audited ordering regime, no redundant second bisection was run for `I_cross,24`; the comparison below uses the local sign-change brackets.",
            "",
            "| beta | I_AS | I_cross,24 | overlap? | midpoint distance |",
            "|---:|---|---|---|---:|",
        ]
        for row in rate_interval_rows:
            dist = row["midpoint_distance"]
            dist_text = "-" if dist is None else f"{float(dist):.4g}"
            overlap = "unknown" if row["overlap"] is None else ("yes" if row["overlap"] else "no")
            lines.append(
                f"| {row['beta']:.6f} | {interval_text(row['I_AS_lo'], row['I_AS_hi'])} | "
                f"{interval_text(row['I_cross_lo'], row['I_cross_hi'])} | {overlap} | {dist_text} |"
            )
    if phase_interval_rows:
        lines += [
            "",
            "| beta | I_AS | I_S | overlap? | interval separation |",
            "|---:|---|---|---|---:|",
        ]
        for row in phase_interval_rows:
            sep = row["interval_separation"]
            sep_text = "-" if sep is None else f"{float(sep):.4g}"
            overlap = "unknown" if row["overlap"] is None else ("yes" if row["overlap"] else "no")
            lines.append(
                f"| {row['beta']:.6f} | {interval_text(row['I_AS_lo'], row['I_AS_hi'])} | "
                f"{interval_text(row['I_S_lo'], row['I_S_hi'])} | {overlap} | {sep_text} |"
            )
    if phase_robustness:
        anchors = ", ".join(f"{float(v):.6f}" for v in phase_robustness.get("anchor_targets", []))
        lines += [
            "",
            "## Phase-family anchor robustness",
            "",
            f"Anchors tested: `{anchors}`.",
            "The selected phase-alignment family is anchor-robust across three initialisations." if phase_robustness.get("robust") else "The three anchors do not select one common phase-alignment family, so the red connecting family should not be interpreted as established.",
            "Agreement at `beta_opt` is not independent because the original family was selected by proximity to the equal-real-action candidate there.",
            "",
            "| beta | selected Gamma spread across anchors | same family? |",
            "|---:|---:|---|",
        ]
        for row in phase_robustness.get("rows", []):
            lines.append(f"| {row['beta']:.6f} | {row['spread']:.3e} | {'yes' if row['same'] else 'no'} |")
    if beta035_resolution:
        lines += [
            "",
            "## Beta 0.35 left-censor check",
            "",
            f"Status: `{beta035_resolution.get('status')}`.",
        ]
        if beta035_resolution.get("points"):
            lines += [
                "",
                "| Gamma | f_AS | comp certified? | comp residual | z-step distance |",
                "|---:|---:|---|---:|---:|",
            ]
            for row in beta035_resolution["points"]:
                f_text = "-" if row.get("f_AS") is None else f"{float(row['f_AS']):+.3e}"
                lines.append(
                    f"| {float(row['Gamma']):.4f} | {f_text} | {'yes' if row.get('comp_certified') else 'no'} | "
                    f"{float(row.get('comp_residual_inf', math.nan)):.3e} | {float(row.get('z_step_dist', math.nan)):.3e} |"
                )
    phase_rows = phase_alignment_report_rows()
    if phase_rows and phase_robustness.get("robust"):
        gamma_grid = sorted(float(v) for v in metadata["Gamma_grid"])
        grid_steps = np.diff(gamma_grid)
        grid_resolution = float(np.median(grid_steps)) if len(grid_steps) else float("nan")
        lines += [
            "",
            "## Phase-alignment family check",
            "",
            "The selected phase-alignment family is chosen by nearest-neighbor continuation from the root nearest the equal-real-action candidate at `beta_opt`.  Other numerically unwrapped phase-alignment roots are shown as secondary candidates in the figure.",
            "",
            f"Median Gamma-grid step: `{grid_resolution:.4g}`.",
            "",
            "| beta | Gamma_AS | Gamma_S | |Gamma_S - Gamma_AS| |",
            "|---:|---:|---:|---:|",
        ]
        for row in phase_rows:
            lines.append(
                f"| {row['beta']:.6f} | {row['Gamma_AS']:.4f} | "
                f"{row['Gamma_S']:.4f} | {row['delta']:.4f} |"
            )
    elif phase_rows:
        lines += [
            "",
            "## Phase-alignment candidate check",
            "",
            "The red phase-family connection is suppressed because the selected phase family is not anchor-robust.  The figure therefore shows phase-alignment roots as pale gray candidates only.",
        ]
    lines += ["", "## Figures", ""]
    lines += [f"- `{Path(fig).name}`" for fig in figures]
    lines += [
        "",
        "## Final Figure Caption",
        "",
        "Saddle crossover beyond the small-gamma regime. Panel (a) shows the rate-residual diagnostic `D_24 = |E_comp-lambda_24| - |E_seed-lambda_24|`; plotted intervals are brackets from the sampled Gamma grid and lines are guides between branch-resolved slices. In the audited ordering regime, `D_24=0` coincides algebraically with equal real action. Panel (b) shows branch-resolved equal-real-action candidates and numerical phase-alignment candidates. The beta=0.60 inset shows `f_AS(Gamma)` for the continued side-anchored branches; the old nonmonotone marker is resolved as a branch-selection artifact, not a certified single-branch continuation. PL phase points are necessary-condition candidates only; no thimble intersection numbers are computed.",
    ]
    lines.append("")
    path = OUT / "physically_possible_refined_report.md"
    path.write_text("\n".join(lines))
    return str(path)


def parse_csv_floats(value: str) -> list[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def run(args: argparse.Namespace) -> None:
    refined_meta = load_json(REFINED / "refined_transition_scan.json")["metadata"]
    if args.betas:
        betas = sorted(set(round(float(v), 10) for v in args.betas))
    else:
        betas = [float(v) for v in refined_meta.get("beta_grid", default_betas("quick"))]
    if args.gammas:
        Gammas = sorted(set(round(float(v), 10) for v in args.gammas))
    else:
        Gammas = [float(v) for v in refined_meta.get("Gamma_grid", default_gammas("quick"))]
    gammas = [-G for G in Gammas]

    metadata = {
        "q": Q,
        "K_clause": K_CLAUSE,
        "r": R,
        "phi_pref_conv2": conv2_pref(K_CLAUSE, R),
        "beta_grid": betas,
        "Gamma_grid": Gammas,
        "gamma_grid": gammas,
        "starts": int(args.starts),
        "possible_exponent_tol": float(args.possible_tol),
        "source_refined_scan": str((REFINED / "refined_transition_scan.json").resolve()),
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    cached = load_cached_rows() if args.resume else {}
    rows_new: list[dict[str, Any]] = []
    print(f"Possible-exponent refined audit: {len(betas)} beta x {len(gammas)} Gamma")
    print(f"Competitor starts per point: {args.starts}")
    print(f"Output: {RAW}")
    for ib, beta in enumerate(betas, start=1):
        print(f"\n=== beta {ib}/{len(betas)} = {beta:.6f} ===", flush=True)
        z_current = None
        ok_current = False
        if all(row_key(beta, gamma) in cached for gamma in gammas):
            print("  full beta slice cached", flush=True)
        else:
            z_current, ok_current = warm_seed_to(beta, gammas[0])
            print(f"  warm start to Gamma={-gammas[0]:.4f}: {'ok' if z_current is not None else 'failed'}", flush=True)

        for ig, gamma in enumerate(gammas):
            key = row_key(beta, gamma)
            if key in cached:
                row = cached[key]
                rows_new.append(row)
                print(f"  Gamma={-gamma:.4f}: cached {classify_possible(row, args.decoy_margin)}", flush=True)
                z_current = None
                continue
            print(f"  Gamma={-gamma:.4f}: ", end="", flush=True)
            if ig == 0 and z_current is not None:
                z = z_current
            elif z_current is not None:
                z, ok_current = seed_from_previous(z_current, beta, gamma)
                if z is None:
                    z, ok_current = warm_seed_to(beta, gamma)
            else:
                z, ok_current = warm_seed_to(beta, gamma)
            if z is None:
                row = {"beta": beta, "gamma": gamma, "status": "seed_failed"}
                rows_new.append(row)
                print("seed failed", flush=True)
                continue
            z_current = z
            try:
                row = compute_row(beta, gamma, z, int(args.starts), float(args.possible_tol))
            except Exception as exc:
                row = {"beta": beta, "gamma": gamma, "status": "failed", "reason": repr(exc)}
                print(f"failed ({exc})", flush=True)
                rows_new.append(row)
                z_current = None
                continue
            row["certified_seed"] = bool(ok_current)
            rows_new.append(row)
            print(
                f"{classify_possible(row, args.decoy_margin)}; "
                f"possible={row['n_possible_competitors']}/{row['n_certified_competitors_recomputed']}; "
                f"impossible={row['n_impossible_competitors']}; "
                f"best_possible_gap={row['best_possible_match_abs_gap']:.4g}",
                flush=True,
            )
            if args.checkpoint_every and len(rows_new) % args.checkpoint_every == 0:
                save_rows(merge_rows(cached, rows_new), metadata)

    rows = merge_rows(cached, rows_new)
    save_rows(rows, metadata)
    transitions = possible_transition_table(rows, args.decoy_margin)
    grid = grid_from_rows(rows, args.decoy_margin)
    finite_n_rows, finite_n_nonmonotone_betas = finite_n_stability_rows(transitions, sweep.N_VALUES)
    mechanism_rows = finite_n_mechanism_audit_rows(finite_n_rows, sweep.N_VALUES)
    rate_interval_rows, phase_interval_rows = interval_comparison_rows(finite_n_rows)
    phase_robustness = phase_anchor_robustness()
    refined_as_rows = refine_equal_action_brackets([0.40, 0.45, 0.50, BETA_OPT, 0.575, 0.625, 0.65])
    beta035_resolution = resolve_left_censored_beta035()
    figures = [
        plot_separated_diagnostics(grid, transitions),
        str(OUT / "separated_crossover_and_pl_conditions.pdf"),
        plot_map(grid, transitions),
        plot_boundary(transitions),
        plot_counts(grid),
        plot_gap(grid),
        plot_finite_n_stability(finite_n_rows, sweep.N_VALUES),
    ]
    report = write_report(
        rows,
        transitions,
        figures,
        metadata,
        args.decoy_margin,
        finite_n_rows,
        finite_n_nonmonotone_betas,
        sweep.N_VALUES,
        mechanism_rows,
        refined_as_rows,
        rate_interval_rows,
        phase_interval_rows,
        phase_robustness,
        beta035_resolution,
    )
    summary = {
        "metadata": metadata,
        "counts": {
            "rows": len(rows),
            "ok": sum(1 for r in rows if r.get("status") == "ok"),
            "seed": sum(1 for r in rows if classify_possible(r, args.decoy_margin) == "seed"),
            "impossible_only_decoy": sum(1 for r in rows if classify_possible(r, args.decoy_margin) == "impossible_only_decoy"),
            "possible_decoy": sum(1 for r in rows if classify_possible(r, args.decoy_margin) == "possible_decoy"),
            "possible_competitor": sum(1 for r in rows if classify_possible(r, args.decoy_margin) == "possible_competitor"),
            "certified_competitors": sum(int(r.get("n_certified_competitors_recomputed", 0)) for r in rows if r.get("status") == "ok"),
            "possible_competitors": sum(int(r.get("n_possible_competitors", 0)) for r in rows if r.get("status") == "ok"),
            "impossible_competitors": sum(int(r.get("n_impossible_competitors", 0)) for r in rows if r.get("status") == "ok"),
        },
        "transitions": transitions,
        "finite_n_stability": {
            "json": str((OUT / "finite_n_crossover_stability.json").resolve()),
            "nonmonotone_betas": finite_n_nonmonotone_betas,
            "rows": finite_n_rows,
        },
        "finite_n_mechanism_audit": {
            "json": str((OUT / "finite_n_mechanism_audit.json").resolve()),
            "rows": mechanism_rows,
        },
        "equal_real_action_refined_brackets": {
            "json": str((OUT / "equal_real_action_refined_brackets.json").resolve()),
            "rows": refined_as_rows,
        },
        "interval_comparisons": {
            "json": str((OUT / "interval_comparisons.json").resolve()),
            "rate_vs_anti_stokes": rate_interval_rows,
            "stokes_vs_anti_stokes": phase_interval_rows,
        },
        "phase_anchor_robustness": phase_robustness,
        "beta035_left_censor_resolution": beta035_resolution,
        "figures": figures,
        "report": report,
    }
    with (OUT / "physically_possible_refined_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print("\nWrote:")
    print(f"  {RAW}")
    print(f"  {OUT / 'physically_possible_refined_summary.json'}")
    print(f"  {OUT / 'finite_n_crossover_stability.json'}")
    print(f"  {OUT / 'finite_n_mechanism_audit.json'}")
    print(f"  {OUT / 'equal_real_action_refined_brackets.json'}")
    print(f"  {OUT / 'interval_comparisons.json'}")
    print(f"  {OUT / 'beta035_left_censor_resolution.json'}")
    print(f"  {report}")
    for fig in figures:
        print(f"  {fig}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--starts", type=int, default=200)
    parser.add_argument("--possible-tol", type=float, default=0.0)
    parser.add_argument("--decoy-margin", type=float, default=0.02)
    parser.add_argument("--checkpoint-every", type=int, default=12)
    parser.add_argument("--no-resume", dest="resume", action="store_false", default=True)
    parser.add_argument("--betas", type=parse_csv_floats, default=None)
    parser.add_argument("--gammas", type=parse_csv_floats, default=None, help="comma-separated positive Gamma values")
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
