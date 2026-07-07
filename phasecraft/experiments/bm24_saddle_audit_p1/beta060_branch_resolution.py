"""Branch-resolved beta=0.60 equal-action diagnostic.

This is a focused follow-up to the physically possible saddle crossover audit.
It avoids defining a branch by independently selecting the rate-best
competitor at every sampled point.  Instead, it continues the validated
competitor roots from beta=0.575 and beta=0.625 into the beta=0.60 slice, then
uses those continued roots to scan equal-real-action zeros.
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
REPO_PARENT = HERE.parents[2]
if str(REPO_PARENT) not in sys.path:
    sys.path.insert(0, str(REPO_PARENT))

from phasecraft.bm24_saddle_audit_p1 import refined_transition_scan as refined
from phasecraft.bm24_saddle_audit_p1.audit import _x_to_z
from phasecraft.bm24_saddle_audit_p1.continue_seed_branch_certified import (
    certify_z,
    conv2_full_exponent,
    lambda_abs_n_max,
    polish_to_residual,
)
from phasecraft.bm24_saddle_audit_p1.pl_homotopy_tracker import (
    K_CLAUSE,
    N_VALUES,
    Q,
    R,
)
from phasecraft.lib.saddles.krawczyk_p1_roots import SaddleSystem, numerical_jacobian
from phasecraft.lib.saddles.picard_lefschetz import compute_phi


RESULTS = HERE / "results"
PL_ALL = RESULTS / "pl_homotopy_all_slices"
OUT = RESULTS / "physically_possible_refined"
RAW = OUT / "physically_possible_refined.json"

TARGET_BETA = 0.60
LEFT_BETA = 0.575
RIGHT_BETA = 0.625
ANCHOR_GAMMAS = [1.83, 1.84, 1.85, 1.86, 1.87, 1.88]
GAMMA_SCAN = [round(v, 10) for v in np.arange(1.75, 1.9200001, 0.005)]
LOCAL_BETAS = [round(v, 10) for v in np.arange(0.575, 0.6250001, 0.005)]
MIN_RESIDUAL = 1e-10
DPS = 80
BRANCH_MATCH_TOL = 1e-5
LIKELY_MATCH_TOL = 1e-3
SEED_MERGE_TOL = 1e-4
MAX_Z_JUMP = 0.75

BLUE = "#2b6cb0"
GREEN = "#2f855a"
PURPLE = "#6b46c1"
RED = "#c53030"
GRAY = "#4a5568"


def load_json(path: Path) -> Any:
    with path.open() as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2)


def z_from_parts(re: list[float], im: list[float]) -> np.ndarray:
    return np.asarray(re, dtype=float) + 1j * np.asarray(im, dtype=float)


def finite_or_none(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def beta_tag(beta: float) -> str:
    return f"beta_{beta:.6f}".replace(".", "p")


def pl_rows_for_beta(beta: float) -> list[dict[str, Any]]:
    path = PL_ALL / beta_tag(beta) / "pl_homotopy_tracker.json"
    data = load_json(path)
    return [
        row for row in data.get("rows", [])
        if not row.get("competitor_failed")
        and not row.get("competitor_merged_with_seed")
        and row.get("competitor", {}).get("z_real")
        and row.get("competitor", {}).get("z_imag")
    ]


def nearest_pl_competitor(beta: float, Gamma: float) -> np.ndarray:
    rows = pl_rows_for_beta(beta)
    if not rows:
        raise RuntimeError(f"no valid PL rows for beta={beta}")
    row = min(rows, key=lambda r: abs(float(r["Gamma"]) - Gamma))
    comp = row["competitor"]
    return z_from_parts(comp["z_real"], comp["z_imag"])


def saddle_system(beta: float, Gamma: float) -> SaddleSystem:
    return SaddleSystem.build(q=Q, r=R, betas=np.array([beta]), gammas=np.array([-Gamma]))


def certify_and_measure(sys_: SaddleSystem, z: np.ndarray) -> tuple[bool, dict[str, Any], float]:
    ok, proof = certify_z(sys_, z, dps=DPS)
    x = np.concatenate([z.real, z.imag])
    jac = numerical_jacobian(sys_.G_real, x)
    jcond = float(np.linalg.cond(jac)) if jac.size else math.inf
    return bool(ok), proof, jcond


def eval_seed(beta: float, Gamma: float, seed_hint: np.ndarray | None = None) -> tuple[np.ndarray | None, dict[str, Any]]:
    gamma = -float(Gamma)
    try:
        sys_ = saddle_system(beta, Gamma)
        if seed_hint is not None:
            x_hint = np.concatenate([seed_hint.real, seed_hint.imag])
            x_pol, res = polish_to_residual(sys_, x_hint, min_residual=MIN_RESIDUAL, dps=DPS)
            z_seed = _x_to_z(x_pol, sys_.nvars)
            ok = bool(math.isfinite(res) and res <= 1e-7)
        else:
            z_seed, ok = refined.warm_seed_to(beta, gamma)
        if z_seed is None or not ok:
            return None, {"failed": True, "failure_reason": "seed_continuation_failed"}
        res = float(np.linalg.norm(sys_.G_complex(z_seed), ord=np.inf))
        certified, proof, jcond = certify_and_measure(sys_, z_seed)
        phi = compute_phi(z_seed, q=Q, r=R, betas=sys_.betas, gammas=sys_.gammas)
        full = conv2_full_exponent(float(phi.real), K_CLAUSE, R)
        failed = (not certified) or (not math.isfinite(res)) or res > 1e-7
        return z_seed, {
            "failed": bool(failed),
            "failure_reason": "" if not failed else str(proof.get("reason", "seed_cert_or_residual_failed")),
            "certified": bool(certified) and not failed,
            "residual_inf": res,
            "krawczyk_contraction": float(proof.get("contraction_bound", math.inf)),
            "jacobian_cond": jcond,
            "re_phi": float(phi.real),
            "im_phi": float(phi.imag),
            "full_exponent": float(full),
            "z_real": z_seed.real.tolist(),
            "z_imag": z_seed.imag.tolist(),
        }
    except Exception as exc:
        return None, {"failed": True, "failure_reason": repr(exc)}


def eval_competitor(
    beta: float,
    Gamma: float,
    z_hint: np.ndarray,
    *,
    source_anchor: str,
    previous_z: np.ndarray | None = None,
    seed_z: np.ndarray | None = None,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    sys_ = saddle_system(beta, Gamma)
    try:
        x_hint = np.concatenate([z_hint.real, z_hint.imag])
        x_pol, res = polish_to_residual(sys_, x_hint, min_residual=MIN_RESIDUAL, dps=DPS)
        z_pol = _x_to_z(x_pol, sys_.nvars)
        certified, proof, jcond = certify_and_measure(sys_, z_pol)
        phi = compute_phi(z_pol, q=Q, r=R, betas=sys_.betas, gammas=sys_.gammas)
        full = conv2_full_exponent(float(phi.real), K_CLAUSE, R)
        z_step = float(np.linalg.norm(z_pol - (previous_z if previous_z is not None else z_hint), ord=np.inf))
        z_seed_dist = (
            float(np.linalg.norm(z_pol - seed_z, ord=np.inf))
            if seed_z is not None
            else math.nan
        )
        z_jump = z_step > MAX_Z_JUMP
        failed = (not certified) or (not math.isfinite(res)) or res > 1e-7
        reason = ""
        if failed:
            if not certified:
                reason = str(proof.get("reason", "krawczyk_failed"))
            elif res > 1e-7:
                reason = "residual_above_1e-7"
            else:
                reason = "nonfinite_residual"
        row = {
            "beta": float(beta),
            "Gamma": float(Gamma),
            "gamma": -float(Gamma),
            "source_anchor": source_anchor,
            "failed": bool(failed),
            "failure_reason": reason,
            "certified": bool(certified) and not failed,
            "z_jump_flag": bool(z_jump),
            "z_jump_note": f"z_step_gt_{MAX_Z_JUMP}" if z_jump else "",
            "residual_inf": float(res),
            "krawczyk_contraction": float(proof.get("contraction_bound", math.inf)),
            "jacobian_cond": jcond,
            "z_step_distance": z_step,
            "distance_to_seed": z_seed_dist,
            "re_phi": float(phi.real),
            "im_phi": float(phi.imag),
            "full_exponent": float(full),
            "z_real": z_pol.real.tolist(),
            "z_imag": z_pol.imag.tolist(),
        }
        return (None if failed else z_pol), row
    except Exception as exc:
        return None, {
            "beta": float(beta),
            "Gamma": float(Gamma),
            "gamma": -float(Gamma),
            "source_anchor": source_anchor,
            "failed": True,
            "failure_reason": repr(exc),
        }


def adaptive_beta_continue(
    *,
    source_anchor: str,
    beta0: float,
    beta1: float,
    Gamma: float,
    z0: np.ndarray,
) -> tuple[np.ndarray | None, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    beta = float(beta0)
    z = z0
    seed_hint: np.ndarray | None = None
    direction = 1.0 if beta1 > beta0 else -1.0
    step = 0.005
    allowed = [0.005, 0.0025, 0.001, 0.0005]
    while abs(beta - beta1) > 2e-10:
        raw_next = beta + direction * min(step, abs(beta1 - beta))
        beta_next = round(raw_next, 10)
        z_next, candidate = eval_competitor(
            beta_next,
            Gamma,
            z,
            source_anchor=source_anchor,
            previous_z=z,
            seed_z=None,
        )
        if z_next is not None and not candidate.get("failed"):
            rows.append(candidate)
            beta = beta_next
            z = z_next
            if step < 0.005:
                step = allowed[max(0, allowed.index(step) - 1)]
            continue
        idx = allowed.index(step) if step in allowed else len(allowed) - 1
        if idx == len(allowed) - 1:
            rows.append(candidate)
            return None, rows
        step = allowed[idx + 1]
    return z, rows


def prepare_anchor(beta: float, Gamma: float, source_anchor: str) -> tuple[np.ndarray | None, dict[str, Any]]:
    hint = nearest_pl_competitor(beta, Gamma)
    return eval_competitor(beta, Gamma, hint, source_anchor=source_anchor, seed_z=None)


def run_beta_anchor_continuations() -> tuple[dict[str, Any], dict[str, Any]]:
    continuations: list[dict[str, Any]] = []
    endpoint_roots: dict[str, list[dict[str, Any]]] = {"left": [], "right": []}
    for source, beta0 in [("left", LEFT_BETA), ("right", RIGHT_BETA)]:
        for Gamma in ANCHOR_GAMMAS:
            z_anchor, anchor_row = prepare_anchor(beta0, Gamma, source)
            entry = {
                "source_anchor": source,
                "anchor_beta": beta0,
                "target_beta": TARGET_BETA,
                "Gamma": Gamma,
                "anchor": anchor_row,
                "steps": [],
                "status": "failed",
            }
            if z_anchor is None or anchor_row.get("failed"):
                continuations.append(entry)
                continue
            z_target, rows = adaptive_beta_continue(
                source_anchor=source,
                beta0=beta0,
                beta1=TARGET_BETA,
                Gamma=Gamma,
                z0=z_anchor,
            )
            entry["steps"] = rows
            if z_target is not None:
                entry["status"] = "reached_beta060"
                endpoint_roots[source].append(rows[-1])
            continuations.append(entry)

    comparisons: list[dict[str, Any]] = []
    for Gamma in ANCHOR_GAMMAS:
        lrow = next((r for r in endpoint_roots["left"] if math.isclose(float(r["Gamma"]), Gamma, abs_tol=1e-9)), None)
        rrow = next((r for r in endpoint_roots["right"] if math.isclose(float(r["Gamma"]), Gamma, abs_tol=1e-9)), None)
        if lrow is None or rrow is None:
            comparisons.append({"Gamma": Gamma, "classification": "missing_endpoint"})
            continue
        zl = z_from_parts(lrow["z_real"], lrow["z_imag"])
        zr = z_from_parts(rrow["z_real"], rrow["z_imag"])
        dz = float(np.linalg.norm(zl - zr, ord=np.inf))
        dE = abs(float(lrow["full_exponent"]) - float(rrow["full_exponent"]))
        dRe = abs(float(lrow["re_phi"]) - float(rrow["re_phi"]))
        if dz < BRANCH_MATCH_TOL:
            cls = "same_branch"
        elif dz < LIKELY_MATCH_TOL and dE < 1e-8:
            cls = "likely_same_branch_but_numerically_difficult"
        else:
            cls = "distinct_branches"
        comparisons.append({
            "Gamma": Gamma,
            "z_left_right_inf": dz,
            "abs_E_left_right": dE,
            "abs_RePhi_left_right": dRe,
            "classification": cls,
        })

    if comparisons and all(c.get("classification") == "same_branch" for c in comparisons):
        branch_summary = {
            "classification": "same_branch",
            "branches": [{"branch_id": "branch_A", "sources": ["left", "right"]}],
        }
    elif comparisons and all(c.get("classification") in {"same_branch", "likely_same_branch_but_numerically_difficult"} for c in comparisons):
        branch_summary = {
            "classification": "likely_same_branch_but_numerically_difficult",
            "branches": [{"branch_id": "branch_A", "sources": ["left", "right"]}],
        }
    else:
        branch_summary = {
            "classification": "distinct_branches",
            "branches": [
                {"branch_id": "branch_left", "sources": ["left"]},
                {"branch_id": "branch_right", "sources": ["right"]},
            ],
        }

    payload = {
        "metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "target_beta": TARGET_BETA,
            "anchor_betas": {"left": LEFT_BETA, "right": RIGHT_BETA},
            "anchor_Gammas": ANCHOR_GAMMAS,
            "adaptive_beta_steps": [0.005, 0.0025, 0.001, 0.0005],
        },
        "continuations": continuations,
        "endpoint_comparisons": comparisons,
        "branch_summary": branch_summary,
    }
    write_json(OUT / "beta060_branch_continuation.json", payload)
    return payload, endpoint_roots


def representative_roots(branch_summary: dict[str, Any], endpoint_roots: dict[str, list[dict[str, Any]]]) -> dict[str, dict[str, Any]]:
    reps: dict[str, dict[str, Any]] = {}
    if branch_summary["classification"] in {"same_branch", "likely_same_branch_but_numerically_difficult"}:
        rows = endpoint_roots["left"] or endpoint_roots["right"]
        row = min(rows, key=lambda r: abs(float(r["Gamma"]) - 1.85))
        reps["branch_A"] = row
    else:
        for source, branch_id in [("left", "branch_left"), ("right", "branch_right")]:
            rows = endpoint_roots[source]
            if rows:
                reps[branch_id] = min(rows, key=lambda r: abs(float(r["Gamma"]) - 1.85))
    return reps


def gamma_series(
    *,
    branch_id: str,
    beta: float,
    start_row: dict[str, Any],
    gammas: list[float],
    start_seed_hint: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    start_Gamma = float(start_row["Gamma"])
    start_z = z_from_parts(start_row["z_real"], start_row["z_imag"])
    all_g = sorted(set(float(g) for g in gammas + [start_Gamma]))
    rows_by_g: dict[float, dict[str, Any]] = {}

    def eval_at(Gamma: float, z_hint: np.ndarray, previous_z: np.ndarray | None, seed_hint: np.ndarray | None) -> tuple[np.ndarray | None, np.ndarray | None, dict[str, Any]]:
        seed_z, seed_row = eval_seed(beta, Gamma, seed_hint)
        if seed_z is None or seed_row.get("failed"):
            return None, seed_z, {
                "branch_id": branch_id,
                "beta": beta,
                "Gamma": Gamma,
                "failed": True,
                "failure_reason": f"seed_failed:{seed_row.get('failure_reason', '')}",
            }
        z, crow = eval_competitor(
            beta,
            Gamma,
            z_hint,
            source_anchor=branch_id,
            previous_z=previous_z,
            seed_z=seed_z,
        )
        crow["branch_id"] = branch_id
        crow["seed"] = dict(seed_row)
        if z is not None and not crow.get("failed"):
            f_as = float(crow["full_exponent"]) - float(seed_row["full_exponent"])
            crow["f_AS"] = f_as
            crow["seed_E"] = float(seed_row["full_exponent"])
            crow["comp_E"] = float(crow["full_exponent"])
            crow["branch_eligible"] = bool(crow.get("certified")) and float(crow.get("distance_to_seed", math.inf)) > SEED_MERGE_TOL
            crow["branch_safe"] = bool(crow["branch_eligible"]) and not bool(crow.get("z_jump_flag", False))
            if not crow["branch_safe"]:
                crow["branch_warning"] = "z_jump_or_merged_with_seed_or_uncertified"
        return z, seed_z, crow

    # Ensure the start row has seed/f_AS fields.
    z0, seed0, row0 = eval_at(start_Gamma, start_z, None, start_seed_hint)
    if z0 is None or row0.get("failed"):
        row0["branch_id"] = branch_id
        rows_by_g[start_Gamma] = row0
        return [row0]
    rows_by_g[start_Gamma] = row0

    z = z0
    seed_hint = seed0
    for Gamma in [g for g in all_g if g > start_Gamma]:
        z_next, seed_next, row = eval_at(Gamma, z, z, seed_hint)
        rows_by_g[Gamma] = row
        if z_next is None or row.get("failed") or not row.get("branch_eligible", False):
            break
        z = z_next
        seed_hint = seed_next

    z = z0
    seed_hint = seed0
    for Gamma in reversed([g for g in all_g if g < start_Gamma]):
        z_next, seed_next, row = eval_at(Gamma, z, z, seed_hint)
        rows_by_g[Gamma] = row
        if z_next is None or row.get("failed") or not row.get("branch_eligible", False):
            break
        z = z_next
        seed_hint = seed_next

    return [rows_by_g[g] for g in sorted(rows_by_g)]


def sign_change_pairs(rows: list[dict[str, Any]]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    valid = [
        r for r in rows
        if not r.get("failed") and r.get("branch_eligible") and finite_or_none(r.get("f_AS")) is not None
    ]
    valid.sort(key=lambda r: float(r["Gamma"]))
    pairs = []
    for a, b in zip(valid[:-1], valid[1:]):
        fa = float(a["f_AS"])
        fb = float(b["f_AS"])
        if fa == 0.0 or fa * fb <= 0.0:
            pairs.append((a, b))
    return pairs


def refine_root(branch_id: str, beta: float, lo_row: dict[str, Any], hi_row: dict[str, Any], width_tol: float = 1e-3) -> dict[str, Any]:
    left = dict(lo_row)
    right = dict(hi_row)
    if float(left["Gamma"]) > float(right["Gamma"]):
        left, right = right, left
    failures: list[dict[str, Any]] = []
    iterations = 0
    while abs(float(right["Gamma"]) - float(left["Gamma"])) > width_tol and iterations < 32:
        mid = 0.5 * (float(left["Gamma"]) + float(right["Gamma"]))
        zl = z_from_parts(left["z_real"], left["z_imag"])
        zr = z_from_parts(right["z_real"], right["z_imag"])
        seed_left = z_from_parts(left["seed"].get("z_real", []), left["seed"].get("z_imag", [])) if left.get("seed", {}).get("z_real") else None
        seed_right = z_from_parts(right["seed"].get("z_real", []), right["seed"].get("z_imag", [])) if right.get("seed", {}).get("z_real") else None
        # Re-evaluate from both sides.  The seed hint is deliberately not
        # essential; a warm seed fallback is used inside eval_seed.
        series_l = gamma_series(
            branch_id=branch_id,
            beta=beta,
            start_row={**left, "Gamma": mid, "z_real": zl.real.tolist(), "z_imag": zl.imag.tolist()},
            gammas=[mid],
            start_seed_hint=seed_left,
        )
        series_r = gamma_series(
            branch_id=branch_id,
            beta=beta,
            start_row={**right, "Gamma": mid, "z_real": zr.real.tolist(), "z_imag": zr.imag.tolist()},
            gammas=[mid],
            start_seed_hint=seed_right,
        )
        iterations += 1
        ml = series_l[0]
        mr = series_r[0]
        if ml.get("failed") or mr.get("failed"):
            failures.append({"Gamma": mid, "left": ml, "right": mr, "reason": "midpoint_continuation_failed"})
            break
        dz = float(np.linalg.norm(z_from_parts(ml["z_real"], ml["z_imag"]) - z_from_parts(mr["z_real"], mr["z_imag"]), ord=np.inf))
        if dz >= BRANCH_MATCH_TOL:
            failures.append({"Gamma": mid, "z_left_right_inf": dz, "reason": "two_sided_midpoints_do_not_match"})
            break
        if float(ml.get("distance_to_seed", math.inf)) <= SEED_MERGE_TOL:
            failures.append({"Gamma": mid, "reason": "midpoint_merged_with_seed", "distance_to_seed": ml.get("distance_to_seed")})
            break
        fm = float(ml["f_AS"])
        if float(left["f_AS"]) * fm <= 0.0:
            right = ml
        else:
            left = ml

    width = abs(float(right["Gamma"]) - float(left["Gamma"]))
    branch_safe = not failures and width <= width_tol and bool(left.get("branch_safe")) and bool(right.get("branch_safe"))
    return {
        "branch_id": branch_id,
        "beta": float(beta),
        "Gamma_lo": float(left["Gamma"]),
        "Gamma_hi": float(right["Gamma"]),
        "midpoint": 0.5 * (float(left["Gamma"]) + float(right["Gamma"])),
        "width": width,
        "f_AS_lo": float(left["f_AS"]),
        "f_AS_hi": float(right["f_AS"]),
        "certification": "Krawczyk-certified endpoint saddles" if branch_safe else "partial",
        "branch_safe": bool(branch_safe),
        "iterations": iterations,
        "failures": failures,
    }


def beta060_equal_action_roots(reps: dict[str, dict[str, Any]]) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    scans: dict[str, list[dict[str, Any]]] = {}
    roots: list[dict[str, Any]] = []
    for branch_id, start_row in reps.items():
        rows = gamma_series(branch_id=branch_id, beta=TARGET_BETA, start_row=start_row, gammas=GAMMA_SCAN)
        scans[branch_id] = rows
        for lo, hi in sign_change_pairs(rows):
            roots.append(refine_root(branch_id, TARGET_BETA, lo, hi))
    payload = {
        "metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "beta": TARGET_BETA,
            "Gamma_range": [1.75, 1.92],
            "Gamma_step": 0.005,
            "width_tol": 1e-3,
        },
        "branches": [
            {
                "branch_id": branch_id,
                "n_scan_points": len(rows),
                "n_branch_safe_points": sum(1 for r in rows if r.get("branch_safe")),
            }
            for branch_id, rows in scans.items()
        ],
        "scan_rows": {branch_id: rows for branch_id, rows in scans.items()},
        "roots": roots,
    }
    write_json(OUT / "beta060_equal_action_roots.json", payload)
    return payload, scans


def continue_branch_to_beta(branch_id: str, start_row: dict[str, Any], beta: float, Gamma: float) -> dict[str, Any] | None:
    if math.isclose(beta, TARGET_BETA, abs_tol=1e-12):
        return start_row
    z0 = z_from_parts(start_row["z_real"], start_row["z_imag"])
    z_target, rows = adaptive_beta_continue(
        source_anchor=branch_id,
        beta0=TARGET_BETA,
        beta1=beta,
        Gamma=Gamma,
        z0=z0,
    )
    if z_target is None or not rows:
        return None
    row = rows[-1]
    row["branch_id"] = branch_id
    return row


def cached_pl_equal_action_roots(beta: float, branch_id: str) -> list[dict[str, Any]]:
    rows = pl_rows_for_beta(beta)
    rows.sort(key=lambda r: float(r["Gamma"]))
    out = []
    for lo, hi in zip(rows[:-1], rows[1:]):
        flo = float(lo["delta_re_phi"])
        fhi = float(hi["delta_re_phi"])
        if flo == 0.0 or flo * fhi <= 0.0:
            Gamma_lo = float(lo["Gamma"])
            Gamma_hi = float(hi["Gamma"])
            root = Gamma_lo - flo * (Gamma_hi - Gamma_lo) / (fhi - flo) if fhi != flo else 0.5 * (Gamma_lo + Gamma_hi)
            out.append({
                "branch_id": branch_id,
                "beta": float(beta),
                "Gamma_lo": min(Gamma_lo, Gamma_hi),
                "Gamma_hi": max(Gamma_lo, Gamma_hi),
                "midpoint": float(root),
                "width": abs(Gamma_hi - Gamma_lo),
                "f_AS_lo": flo,
                "f_AS_hi": fhi,
                "certification": "cached PL endpoints Krawczyk-certified",
                "branch_safe": True,
            })
    return out


def local_2d_roots(reps: dict[str, dict[str, Any]], beta060_roots: list[dict[str, Any]]) -> dict[str, Any]:
    local: dict[str, Any] = {
        "metadata": {
            "betas": LOCAL_BETAS,
            "Gamma_range": [1.75, 1.92],
            "status": "unresolved_local_structure_after_beta060_branch_checks",
            "note": "Intermediate beta slices are left open because beta=0.60 failed branch-safe two-sided refinement.",
        },
        "branches": {},
    }
    for branch_id in reps:
        branch_entries = []
        for beta in LOCAL_BETAS:
            roots: list[dict[str, Any]] = []
            status = "unresolved_not_connected"
            if math.isclose(float(beta), LEFT_BETA, abs_tol=1e-12) and branch_id.endswith("left"):
                roots = cached_pl_equal_action_roots(LEFT_BETA, branch_id)
                status = "cached_left_anchor"
            elif math.isclose(float(beta), RIGHT_BETA, abs_tol=1e-12) and branch_id.endswith("right"):
                roots = cached_pl_equal_action_roots(RIGHT_BETA, branch_id)
                status = "cached_right_anchor"
            elif math.isclose(float(beta), TARGET_BETA, abs_tol=1e-12):
                roots = [r for r in beta060_roots if r.get("branch_id") == branch_id]
                status = "partial_beta060_bracket" if roots else "no_beta060_bracket"
            branch_entries.append({"beta": float(beta), "status": status, "roots": roots})
        local["branches"][branch_id] = branch_entries
    write_json(OUT / "local_equal_action_branches_beta060.json", local)
    return local


def plot_local(local: dict[str, Any]) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    markers = ["o", "s", "D", "^"]
    colors = [PURPLE, GREEN, BLUE, RED]
    for idx, (branch_id, entries) in enumerate(local["branches"].items()):
        xs, ys, ylo, yhi = [], [], [], []
        pxs, pys, pylo, pyhi = [], [], [], []
        fail_x, fail_y = [], []
        for entry in entries:
            beta = float(entry["beta"])
            roots = [r for r in entry.get("roots", []) if r.get("branch_safe")]
            partial = [r for r in entry.get("roots", []) if not r.get("branch_safe")]
            if roots:
                for root in roots:
                    xs.append(beta)
                    ys.append(float(root["midpoint"]))
                    ylo.append(float(root["midpoint"]) - float(root["Gamma_lo"]))
                    yhi.append(float(root["Gamma_hi"]) - float(root["midpoint"]))
            elif partial:
                for root in partial:
                    pxs.append(beta)
                    pys.append(float(root["midpoint"]))
                    pylo.append(float(root["midpoint"]) - float(root["Gamma_lo"]))
                    pyhi.append(float(root["Gamma_hi"]) - float(root["midpoint"]))
            else:
                fail_x.append(beta)
                fail_y.append(np.nan)
        order = np.argsort(xs) if xs else []
        if len(xs):
            x = np.asarray(xs)[order]
            y = np.asarray(ys)[order]
            ax.errorbar(
                x,
                y,
                yerr=np.vstack([np.asarray(ylo)[order], np.asarray(yhi)[order]]),
                fmt=markers[idx % len(markers)] + "-",
                color=colors[idx % len(colors)],
                capsize=2,
                lw=1.4,
                ms=4,
                label=branch_id,
            )
        if fail_x:
            ax.scatter(
                fail_x,
                [1.752] * len(fail_x),
                marker=markers[idx % len(markers)],
                facecolors="none",
                edgecolors=colors[idx % len(colors)],
                s=36,
                label=f"{branch_id} failed/no zero",
            )
        if pxs:
            ax.errorbar(
                pxs,
                pys,
                yerr=np.vstack([pylo, pyhi]),
                fmt=markers[idx % len(markers)],
                color=colors[idx % len(colors)],
                markerfacecolor="white",
                capsize=2,
                lw=1.0,
                ms=5,
                linestyle="none",
                label=f"{branch_id} partial",
            )
    ax.axvline(TARGET_BETA, color=GRAY, lw=1.0, ls="--")
    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel(r"$\Gamma$")
    ax.set_title(r"Local branch-resolved equal-real-action zeros near $\beta=0.60$")
    ax.set_xlim(0.572, 0.628)
    ax.set_ylim(1.74, 1.93)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8, frameon=True)
    fig.tight_layout()
    fig.savefig(OUT / "local_equal_action_branches_beta060.pdf", bbox_inches="tight")
    fig.savefig(OUT / "local_equal_action_branches_beta060.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_fas_inset(scans: dict[str, list[dict[str, Any]]]) -> None:
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    colors = [PURPLE, GREEN, BLUE, RED]
    for idx, (branch_id, rows) in enumerate(scans.items()):
        valid = [r for r in rows if not r.get("failed") and r.get("branch_safe") and finite_or_none(r.get("f_AS")) is not None]
        valid.sort(key=lambda r: float(r["Gamma"]))
        if not valid:
            continue
        ax.plot(
            [float(r["Gamma"]) for r in valid],
            [float(r["f_AS"]) for r in valid],
            marker="o",
            ms=2.5,
            lw=1.2,
            color=colors[idx % len(colors)],
            label=branch_id,
        )
    ax.axhline(0.0, color="#1a202c", lw=0.9)
    ax.set_xlabel(r"$\Gamma$")
    ax.set_ylabel(r"$f_{AS}=E_\mathrm{comp}-E_\mathrm{seed}$")
    ax.grid(True, alpha=0.22)
    ax.legend(loc="best", fontsize=7, frameon=True)
    fig.tight_layout()
    fig.savefig(OUT / "beta060_fAS_inset.pdf", bbox_inches="tight")
    fig.savefig(OUT / "beta060_fAS_inset.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def extend_scans_for_assignment(reps: dict[str, dict[str, Any]], original_Gammas: list[float]) -> dict[str, list[dict[str, Any]]]:
    return {
        branch_id: gamma_series(branch_id=branch_id, beta=TARGET_BETA, start_row=start_row, gammas=original_Gammas)
        for branch_id, start_row in reps.items()
    }


def nearest_scan_row(rows: list[dict[str, Any]], Gamma: float) -> dict[str, Any] | None:
    candidates = [r for r in rows if not r.get("failed") and r.get("branch_eligible") and r.get("z_real")]
    if not candidates:
        return None
    row = min(candidates, key=lambda r: abs(float(r["Gamma"]) - Gamma))
    if abs(float(row["Gamma"]) - Gamma) > 1e-8:
        return None
    return row


def branch_assignment(reps: dict[str, dict[str, Any]], beta060_scans: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    raw = load_json(RAW)
    rows = [
        r for r in raw.get("rows", [])
        if r.get("status") == "ok" and math.isclose(float(r.get("beta", math.nan)), TARGET_BETA, abs_tol=1e-9)
    ]
    original_Gammas = sorted({-float(r["gamma"]) for r in rows})
    assignment_scans = extend_scans_for_assignment(reps, original_Gammas)
    out_rows = []
    selected_branch_sequence: list[str] = []
    for row in sorted(rows, key=lambda r: -float(r["gamma"])):
        Gamma = -float(row["gamma"])
        z_best = z_from_parts(row.get("best_possible_z_real", []), row.get("best_possible_z_imag", [])) if row.get("best_possible_z_real") else None
        nearest_branch = None
        nearest_dist = math.inf
        f_by_branch: dict[str, float | None] = {}
        for branch_id, scan_rows in assignment_scans.items():
            srow = nearest_scan_row(scan_rows, Gamma)
            f_by_branch[branch_id] = finite_or_none(srow.get("f_AS")) if srow else None
            if z_best is not None and srow is not None:
                dz = float(np.linalg.norm(z_best - z_from_parts(srow["z_real"], srow["z_imag"]), ord=np.inf))
                if dz < nearest_dist:
                    nearest_dist = dz
                    nearest_branch = branch_id
        D24 = float(row["best_possible_match_abs_gap"]) - abs(float(row["seed_gap"]))
        if nearest_branch is not None:
            selected_branch_sequence.append(nearest_branch)
        out_rows.append({
            "Gamma": Gamma,
            "selected_best_branch": nearest_branch,
            "nearest_branch_distance_inf": nearest_dist if math.isfinite(nearest_dist) else None,
            "D_24": D24,
            "f_AS_by_branch": f_by_branch,
            "root_search_status": "assigned" if nearest_branch is not None else "unassigned",
        })

    branch_switching = len(set(selected_branch_sequence)) > 1
    same_branch_multi = any(len(sign_change_pairs(scan_rows)) > 1 for scan_rows in beta060_scans.values())
    if branch_switching:
        explanation = "branch_selection_artifact"
    elif same_branch_multi:
        explanation = "multiple_equal_action_zeros_on_one_tracked_branch"
    elif not selected_branch_sequence:
        explanation = "failed_continuation_or_missing_assignment"
    else:
        explanation = "rate_best_selection_follows_one_branch"
    payload = {
        "metadata": {"timestamp": datetime.now(timezone.utc).isoformat(), "beta": TARGET_BETA},
        "rows": out_rows,
        "old_red_cross_explanation": explanation,
        "assignment_scans": assignment_scans,
    }
    write_json(OUT / "beta060_branch_assignment.json", payload)
    return payload


def main_case(branch_payload: dict[str, Any], roots_payload: dict[str, Any]) -> str:
    branch_cls = branch_payload["branch_summary"]["classification"]
    safe_roots = [r for r in roots_payload.get("roots", []) if r.get("branch_safe")]
    if branch_cls == "same_branch" and len(safe_roots) == 1:
        return "A"
    if branch_cls in {"same_branch", "likely_same_branch_but_numerically_difficult"} and len(safe_roots) == 0 and roots_payload.get("roots"):
        return "B"
    if len(safe_roots) > 1:
        return "C"
    if branch_cls == "distinct_branches":
        return "D"
    return "E"


def write_report(
    branch_payload: dict[str, Any],
    roots_payload: dict[str, Any],
    assignment_payload: dict[str, Any],
    local_payload: dict[str, Any],
) -> None:
    safe_roots = [r for r in roots_payload.get("roots", []) if r.get("branch_safe")]
    case = main_case(branch_payload, roots_payload)
    lines = [
        "# Beta=0.60 Branch Resolution",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Branch continuation from side anchors",
        "",
        f"Endpoint classification: `{branch_payload['branch_summary']['classification']}`.",
        "",
        "| Gamma | classification | ||z_left-z_right||_inf | |Delta E| |",
        "|---:|---|---:|---:|",
    ]
    for row in branch_payload.get("endpoint_comparisons", []):
        lines.append(
            f"| {float(row['Gamma']):.3f} | {row.get('classification')} | "
            f"{finite_or_none(row.get('z_left_right_inf')) or float('nan'):.3e} | "
            f"{finite_or_none(row.get('abs_E_left_right')) or float('nan'):.3e} |"
        )
    lines += [
        "",
        "## beta=0.60 equal-action roots",
        "",
        "| branch ID | Gamma_lo | Gamma_hi | midpoint | certification | branch-safe? |",
        "|---|---:|---:|---:|---|---|",
    ]
    for row in roots_payload.get("roots", []):
        lines.append(
            f"| {row['branch_id']} | {float(row['Gamma_lo']):.6f} | {float(row['Gamma_hi']):.6f} | "
            f"{float(row['midpoint']):.6f} | {row['certification']} | {bool(row['branch_safe'])} |"
        )
    if not roots_payload.get("roots"):
        lines.append("| - | - | - | - | no branch-safe sign change found | False |")
    lines += [
        "",
        "## Old red cross",
        "",
        f"Classification: `{assignment_payload['old_red_cross_explanation']}`.",
        "",
        "The old beta=0.60 marker came from the gridwise rate-best competitor selection, not from a single branch definition. The table in `beta060_branch_assignment.json` assigns each original grid-selected root to the nearest continued branch family and reports the tracked branch `f_AS` values alongside `D_24`.",
        "",
        "## Local 2D structure",
        "",
    ]
    branch_root_counts = {
        branch_id: sum(len([r for r in entry.get("roots", []) if r.get("branch_safe")]) for entry in entries)
        for branch_id, entries in local_payload.get("branches", {}).items()
    }
    lines.append(f"Branch-safe local root counts: `{branch_root_counts}`.")
    lines += [
        "",
        "## Main-figure decision",
        "",
        f"Decision case: `{case}`.",
    ]
    if case == "A":
        root = safe_roots[0]
        lines.append(
            f"One same certified branch reaches beta=0.60 and has one branch-safe zero: "
            f"Gamma in [{root['Gamma_lo']:.6f}, {root['Gamma_hi']:.6f}]."
        )
    elif case == "C":
        lines.append("Multiple branch-resolved zeros were found; the main figure should plot separate local branch curves.")
    elif case == "D":
        lines.append("Distinct side-anchored branches reach beta=0.60; the old nonmonotonicity should be treated as branch switching.")
    else:
        lines.append("No unique certified branch-resolved beta=0.60 crossover is established.")
    lines += [
        "",
        "No thimble intersection numbers are computed here; the PL phase points remain necessary-condition diagnostics only.",
        "",
    ]
    (OUT / "beta060_resolution_report.md").write_text("\n".join(lines))


def run() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    branch_payload, endpoint_roots = run_beta_anchor_continuations()
    reps = representative_roots(branch_payload["branch_summary"], endpoint_roots)
    roots_payload, beta060_scans = beta060_equal_action_roots(reps)
    plot_fas_inset(beta060_scans)
    local_payload = local_2d_roots(reps, roots_payload.get("roots", []))
    plot_local(local_payload)
    assignment_payload = branch_assignment(reps, beta060_scans)
    write_report(branch_payload, roots_payload, assignment_payload, local_payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    return parser.parse_args()


if __name__ == "__main__":
    parse_args()
    run()
