"""
wu_robust_workflow.py
=====================
Robust competitor pipeline for BM24 w,u-saddle, general q >= 1, p=1.

Mirrors ``phasecraft.bm24_saddle_audit_p1.z_robust_workflow`` but uses
``krawczyk_w_saddle_q.FullUWSaddleSystem`` and ``krawczyk_certify_full_escalate``
to certify the full (u,w) saddle equations rather than the reduced w-only form.

Three-step pipeline (same as z and q=1 w_saddle workflows):
  1. ``continue_wu_seed_branch``  – gamma continuation of the BM24 seed
  2. ``competitor_sweep_on_wu_payload``  – certified competitor discovery
  3. ``resolve_wu_branches``  – branch tracking + crossing refinement

Entry point: ``run_robust_pipeline``.
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from phasecraft.w_saddle.krawczyk_w_saddle_q import (
    SLICE_BETA,
    SLICE_P,
    FullUWSaddleSystem,
    WSaddleSystem,
    discover_w_roots,
    krawczyk_certify_full_escalate,
    solve_uw_from_w,
    solve_w_from_init,
)
from phasecraft.w_saddle.workflow import (
    BranchPoint,
    CompetitorBranch,
    SEED_DUPLICATE_W_FACTOR,
    STATUS_COMPETITOR_CERTIFIED_LOCAL,
    STATUS_COMPETITOR_PL_ACTIVE,
    STATUS_SEED_CERTIFIED_LOCAL,
    STATUS_SEED_DUPLICATE,
    PL_ACTIVE_NOTE,
    action_interval_from_info,
    build_compact_table,
    cluster_numerical_roots,
    dedupe_certified_by_action,
    delta_re_interval_from_action,
    detect_refine_gamma_window,
    diagnose_same_sheet_pairs,
    extract_gamma_nodes,
    find_delta_re_crossing_intervals,
    interpolate_w_seed_from_payload,
    interval_straddles_zero,
    krawczyk_box_record,
    krawczyk_boxes_disjoint,
    merge_same_sheet_branches,
    plot_competitor_analysis,
    resolve_plot_output_path,
    track_branches_from_nodes,
    unwrap_im_branch,
    unwrap_phase_diff,
    _dedupe_crossings_by_same_sheet,
)

DEFAULT_RUN_DIR = Path(__file__).resolve().parent / "runs"

DEFAULT_Q = 3
DEFAULT_K = 8          # k = 2^q = 2^3 = 8
DEFAULT_R = 176.54     # random 8-SAT satisfiability threshold (BM24 Table I)
DEFAULT_BETA = 0.5433996420760803  # optimal p=1 QAOA β for k=8 (matches z_robust_workflow)

WU_DISCLAIMER = (
    "BM24 w,u-saddle (general q) branch-resolved competitor diagnostic. "
    "Re Phi_M gaps; not PL dominance. "
    "Multiple certified w-roots may coexist on different log sheets."
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class WUConfig:
    """Configuration for the w,u-saddle robust pipeline.

    Defaults target random 8-SAT (k=2^q=8, q=3) at its satisfiability
    threshold r=176.54 (Boulebnane & Montanaro 2024, Table I).
    """

    q: int = DEFAULT_Q
    K_clause: int = DEFAULT_K   # k = 2^q (clause arity, must be power of 2)
    r: float = DEFAULT_R
    beta: float = DEFAULT_BETA

    def system(self, gamma: float) -> WSaddleSystem:
        return WSaddleSystem(r=self.r, gamma=gamma, q=self.q, beta=self.beta)

    def full_system(self, gamma: float) -> FullUWSaddleSystem:
        return FullUWSaddleSystem(r=self.r, gamma=gamma, q=self.q, beta=self.beta)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def make_gamma_mesh(gamma_start: float, gamma_stop: float, num_points: int) -> np.ndarray:
    return np.linspace(float(gamma_start), float(gamma_stop), int(num_points))


def w_from_record(rec: dict[str, Any]) -> np.ndarray:
    return np.asarray(rec["w_real"], dtype=float) + 1j * np.asarray(rec["w_imag"], dtype=float)


def wu_root_record(
    cfg: WUConfig,
    w: np.ndarray,
    gamma: float,
    *,
    certified: bool,
    info: dict[str, Any],
    residual: float,
) -> dict[str, Any]:
    phi = cfg.system(gamma).Phi_eff(w)
    return {
        "w_real": w.real.tolist(),
        "w_imag": w.imag.tolist(),
        "residual": float(residual),
        "Phi_eff_real": float(phi.real),
        "Phi_eff_imag": float(phi.imag),
        "delta_lower": float(info.get("delta_lower", 0.0)) if certified else None,
        "contraction_bound": float(info.get("contraction_bound", math.inf)),
        "krawczyk_certified": bool(certified),
        "krawczyk_info": info,
    }


def certify_wu_point(
    cfg: WUConfig,
    w: np.ndarray,
    gamma: float,
    *,
    dps: int,
    max_levels: int = 4,
) -> tuple[bool, dict[str, Any]]:
    sf = cfg.full_system(gamma)
    x, _, _ = solve_uw_from_w(sf, w)
    ok, info = krawczyk_certify_full_escalate(sf, x, dps_start=dps, max_levels=max_levels)
    return bool(ok), dict(info)


# ---------------------------------------------------------------------------
# Numerical root discovery at one gamma
# ---------------------------------------------------------------------------


def discover_wu_numerical(
    cfg: WUConfig,
    gamma: float,
    w_seed: np.ndarray,
    *,
    num_starts: int,
    seed: int,
    tol: float = 1e-12,
) -> list[np.ndarray]:
    sys = cfg.system(gamma)
    roots = discover_w_roots(sys, num_starts=num_starts, seed=seed, tol=tol)
    # Warm-start from seed as additional probe
    w_pol, ok, _ = solve_w_from_init(sys, w_seed, tol=tol)
    if ok and not any(np.linalg.norm(w_pol - r, ord=np.inf) < 1e-5 for r in roots):
        roots.append(w_pol)
    # Extra warm-starts around seed
    scale = max(float(np.max(np.abs(w_seed))) * 8.0, 0.5)
    rng = np.random.default_rng(seed + 17)
    extra: list[np.ndarray] = []
    for _ in range(max(0, num_starts // 4)):
        w_try, ok_try, _ = solve_w_from_init(
            sys,
            w_seed + rng.normal(0, scale / 4, size=4) + 1j * rng.normal(0, scale / 4, size=4),
            tol=tol,
        )
        if ok_try and not any(np.linalg.norm(w_try - r, ord=np.inf) < 1e-5 for r in roots + extra):
            extra.append(w_try)
    return cluster_numerical_roots(roots + extra, 1e-5, w_ref=w_seed)


# ---------------------------------------------------------------------------
# Sweep at a single gamma
# ---------------------------------------------------------------------------


def sweep_wu_at_gamma(
    cfg: WUConfig,
    w_seed: np.ndarray,
    gamma: float,
    *,
    num_starts: int,
    seed: int,
    dps: int,
    cluster_tol: float = 1e-5,
    seed_ident_tol: float = 1e-4,
) -> dict[str, Any]:
    numerical = discover_wu_numerical(
        cfg, gamma, w_seed, num_starts=num_starts, seed=seed, tol=1e-12
    )
    clustered = cluster_numerical_roots(numerical, cluster_tol, w_ref=w_seed)

    certified_rows: list[dict[str, Any]] = []
    sys = cfg.system(gamma)
    for w in clustered:
        res0 = float(np.linalg.norm(sys.F_complex(w), ord=np.inf))
        if not np.isfinite(res0) or res0 > 1e-7:
            continue
        w_pol, ok_pol, res = solve_w_from_init(sys, w)
        if ok_pol:
            w, res = w_pol, res
        ok, info = certify_wu_point(cfg, w, gamma, dps=dps)
        if not ok:
            continue
        certified_rows.append(wu_root_record(cfg, w, gamma, certified=True, info=info, residual=res))

    certified_rows = dedupe_certified_by_action(certified_rows)
    if not certified_rows:
        return {
            "num_numerical_roots": len(numerical),
            "num_clustered_roots": len(clustered),
            "num_certified_roots": 0,
            "certified_roots": [],
            "competitors": [],
            "min_certified_gap_re": None,
            "seed_index": None,
        }

    dists = [float(np.linalg.norm(w_from_record(r) - w_seed, ord=np.inf)) for r in certified_rows]
    seed_index = int(np.argmin(dists))
    for i, row in enumerate(certified_rows):
        row["distance_to_seed"] = dists[i]
        row["is_seed_branch"] = bool(i == seed_index and dists[i] < seed_ident_tol)

    phi_seed_re = certified_rows[seed_index]["Phi_eff_real"]
    phi_seed_im = certified_rows[seed_index]["Phi_eff_imag"]
    competitors = []
    for i, row in enumerate(certified_rows):
        if i == seed_index:
            continue
        gap_re = float(phi_seed_re - row["Phi_eff_real"])
        competitors.append(
            {**row, "gap_re": gap_re, "phase_diff": unwrap_phase_diff(phi_seed_im, row["Phi_eff_imag"])}
        )

    min_gap = min((c["gap_re"] for c in competitors), default=None)
    pos = [c["gap_re"] for c in competitors if c["gap_re"] > 0]
    return {
        "num_numerical_roots": len(numerical),
        "num_clustered_roots": len(clustered),
        "num_certified_roots": len(certified_rows),
        "num_certified_competitors": len(competitors),
        "certified_roots": certified_rows,
        "seed_index": seed_index,
        "min_certified_gap_re": float(min_gap) if min_gap is not None else None,
        "min_positive_gap_re": float(min(pos)) if pos else None,
        "best_competitor_re_phi": float(
            max((c["Phi_eff_real"] for c in competitors), default=phi_seed_re)
        ),
        "seed_dominates_all": bool(competitors and min_gap is not None and min_gap > 0),
        "closest_competitor": min(competitors, key=lambda c: c["gap_re"]) if competitors else None,
        "competitors": competitors,
    }


# ---------------------------------------------------------------------------
# Seed branch continuation
# ---------------------------------------------------------------------------


def continue_wu_seed_branch(
    cfg: WUConfig,
    gammas: np.ndarray,
    *,
    dps: int = 80,
    min_residual: float = 1e-10,
    match_tol: float = 0.02,
    gamma_step: float = 0.01,
    min_step: float = 0.0005,
) -> dict[str, Any]:
    """Gamma continuation of the BM24 seed branch (w,u general-q saddle)."""
    points: list[dict[str, Any]] = []
    w_prev: Optional[np.ndarray] = None

    for _j, gamma in enumerate(gammas):
        sys = cfg.system(float(gamma))
        if w_prev is None:
            w_prev = sys.leading_seed()

        w, ok, res = solve_w_from_init(sys, w_prev)
        if not ok:
            # Residual too large; try harder from leading seed
            w, ok, res = solve_w_from_init(sys, sys.leading_seed())
            if not ok:
                break

        ok_cert, info = certify_wu_point(cfg, w, float(gamma), dps=dps)
        phi = sys.Phi_eff(w)
        row = {
            "gamma": float(gamma),
            "w_star_real": w.real.tolist(),
            "w_star_imag": w.imag.tolist(),
            "Phi_eff_real": float(phi.real),
            "Phi_eff_imag": float(phi.imag),
            "Phi_eff_im_lifted": float(phi.imag),
            "Phi_eff_im_unwrapped": float(phi.imag),
            "residual": float(res),
            "residual_F_norm": float(res),
            "delta_lower": float(info.get("delta_lower", 0.0)) if ok_cert else 0.0,
            "contraction_bound": float(info.get("contraction_bound", math.inf)),
            "krawczyk_contraction_bound": float(info.get("contraction_bound", math.inf)),
            "krawczyk_certified": bool(ok_cert),
            "krawczyk_info": info,
            "box_radius": float(info.get("box_radius", 0.0)) if ok_cert else 0.0,
            "newton_ok": bool(ok),
            "full_conv2_exponent": 0.0,
        }
        points.append(row)

        if not ok_cert:
            break
        w_prev = w

    im_raw = [float(p["Phi_eff_imag"]) for p in points]
    im_u, unwrap_ok = unwrap_im_branch(im_raw)
    for p, imv in zip(points, im_u):
        p["Phi_eff_im_unwrapped"] = imv

    return {
        "r": cfg.r,
        "q": cfg.q,
        "beta": cfg.beta,
        "chart": "wu_saddle",
        "gamma_mesh": [float(g) for g in gammas],
        "num_points_requested": len(gammas),
        "num_points_completed": len(points),
        "completed_full_mesh": len(points) == len(gammas),
        "certificate_type": "wu_seed_branch_continuation",
        "points": points,
        "unwrap_continuous": bool(unwrap_ok),
        "table": build_compact_table(points),
    }


# ---------------------------------------------------------------------------
# Full competitor sweep over a payload
# ---------------------------------------------------------------------------


def competitor_sweep_on_wu_payload(
    payload: dict[str, Any],
    cfg: WUConfig,
    *,
    dps: int,
    competitor_starts: int,
    cluster_tol: float = 1e-5,
    seed: int = 0,
) -> dict[str, Any]:
    for j, row in enumerate(payload.get("points", [])):
        if not row.get("krawczyk_certified", True):
            continue
        gamma = float(row["gamma"])
        w_seed = np.asarray(row["w_star_real"]) + 1j * np.asarray(row["w_star_imag"])
        sweep = sweep_wu_at_gamma(
            cfg,
            w_seed,
            gamma,
            num_starts=competitor_starts,
            seed=seed + j + 1,
            dps=dps,
            cluster_tol=cluster_tol,
        )
        row["competitor_sweep"] = sweep
        row["num_numerical_roots"] = sweep["num_numerical_roots"]
        row["num_certified_roots"] = sweep["num_certified_roots"]
        row["num_certified_competitors"] = sweep.get("num_certified_competitors", 0)
        row["min_certified_gap_re"] = sweep["min_certified_gap_re"]
        row["min_positive_gap_re"] = sweep.get("min_positive_gap_re")
        row["best_competitor_re_phi"] = sweep.get("best_competitor_re_phi")
        row["seed_dominates_all"] = sweep.get("seed_dominates_all")
        row["closest_competitor"] = sweep.get("closest_competitor")
    payload["competitor_sweep_completed"] = True
    payload["table"] = build_compact_table(payload["points"])
    return payload


def _sweep_row_at_gamma_wu(
    payload: dict[str, Any],
    cfg: WUConfig,
    gamma: float,
    *,
    dps: int,
    competitor_starts: int,
    cluster_tol: float,
    seed: int,
) -> dict[str, Any]:
    w_guess = interpolate_w_seed_from_payload(payload, gamma)
    sys = cfg.system(gamma)
    w, ok, res = solve_w_from_init(sys, w_guess)
    if not ok:
        w = w_guess
    ok_cert, info = certify_wu_point(cfg, w, gamma, dps=dps)
    phi = sys.Phi_eff(w)
    sweep = sweep_wu_at_gamma(
        cfg,
        w,
        gamma,
        num_starts=competitor_starts,
        seed=seed,
        dps=dps,
        cluster_tol=cluster_tol,
    )
    return {
        "gamma": float(gamma),
        "w_star_real": w.real.tolist(),
        "w_star_imag": w.imag.tolist(),
        "Phi_eff_real": float(phi.real),
        "Phi_eff_imag": float(phi.imag),
        "Phi_eff_im_lifted": float(phi.imag),
        "Phi_eff_im_unwrapped": float(phi.imag),
        "residual": float(res),
        "residual_F_norm": float(res),
        "krawczyk_certified": bool(ok_cert),
        "krawczyk_info": info,
        "box_radius": float(info.get("box_radius", 0.0)) if ok_cert else 0.0,
        "competitor_sweep": sweep,
        "num_certified_roots": sweep["num_certified_roots"],
        "num_certified_competitors": sweep.get("num_certified_competitors", 0),
        "min_certified_gap_re": sweep["min_certified_gap_re"],
        "seed_dominates_all": sweep.get("seed_dominates_all"),
        "closest_competitor": sweep.get("closest_competitor"),
    }


# ---------------------------------------------------------------------------
# Dense-window refinement
# ---------------------------------------------------------------------------


def merge_refined_wu_window(
    payload: dict[str, Any],
    cfg: WUConfig,
    gamma_lo: float,
    gamma_hi: float,
    dense_gammas: np.ndarray,
    *,
    dps: int,
    competitor_starts: int,
    seed: int,
    cluster_tol: float = 1e-5,
) -> dict[str, Any]:
    gamma_lo, gamma_hi = float(gamma_lo), float(gamma_hi)
    if gamma_lo > gamma_hi:
        gamma_lo, gamma_hi = gamma_hi, gamma_lo
    kept = [
        p for p in payload["points"]
        if float(p["gamma"]) < gamma_lo or float(p["gamma"]) > gamma_hi
    ]
    refined = [
        _sweep_row_at_gamma_wu(
            payload,
            cfg,
            float(g),
            dps=dps,
            competitor_starts=competitor_starts,
            cluster_tol=cluster_tol,
            seed=seed + 10000 + j,
        )
        for j, g in enumerate(dense_gammas)
    ]
    merged = sorted(kept + refined, key=lambda p: float(p["gamma"]), reverse=True)
    out = dict(payload)
    out["points"] = merged
    out["gamma_mesh"] = [float(p["gamma"]) for p in merged]
    out["num_points_completed"] = len(merged)
    rw = dict(out.get("refined_window") or {})
    rw.update({"gamma_lo": gamma_lo, "gamma_hi": gamma_hi, "num_dense_points": len(dense_gammas)})
    out["refined_window"] = rw
    out["table"] = build_compact_table(merged)
    return out


# ---------------------------------------------------------------------------
# Crossing refinement (general-q w,u certifier)
# ---------------------------------------------------------------------------


def _certify_pair_at_gamma_wu(
    cfg: WUConfig,
    gamma: float,
    w_seed_init: np.ndarray,
    w_comp_init: np.ndarray,
    *,
    dps: int,
) -> dict[str, Any]:
    """Certify (seed, competitor) pair at gamma for DeltaRe bracketing."""
    out: dict[str, Any] = {"gamma": float(gamma)}
    for label, w_init in (("seed", w_seed_init), ("competitor", w_comp_init)):
        sys = cfg.system(gamma)
        w, ok, res = solve_w_from_init(sys, w_init)
        ok_cert, info = certify_wu_point(cfg, w, gamma, dps=dps)
        phi = sys.Phi_eff(w)
        rho = float(info.get("box_radius", 0.0)) if ok_cert else 0.0
        # Inject Phi fields so action_interval_from_info can find them
        info_aug = dict(info)
        info_aug["Phi_eff_real"] = float(phi.real)
        info_aug["Phi_eff_imag"] = float(phi.imag)
        act = action_interval_from_info(info_aug) if ok_cert else {}
        out[label] = {
            "w_star_real": w.real.tolist(),
            "w_star_imag": w.imag.tolist(),
            "Phi_eff_real": float(phi.real),
            "Phi_eff_imag": float(phi.imag),
            "krawczyk_certified": bool(ok_cert),
            "delta_lower": float(info.get("delta_lower", 0.0)) if ok_cert else None,
            "box_radius": rho,
            "box": krawczyk_box_record(w, rho),
            "residual": float(res),
            "newton_ok": bool(ok),
            **act,
        }
    seed_act = action_interval_from_info(out["seed"])
    comp_act = action_interval_from_info(out["competitor"])
    delta_iv = delta_re_interval_from_action(seed_act, comp_act)
    out["DeltaRe"] = float(out["seed"]["Phi_eff_real"] - out["competitor"]["Phi_eff_real"])
    out["DeltaRe_interval"] = delta_iv
    out["DeltaIm_wrapped"] = float(
        unwrap_phase_diff(out["seed"]["Phi_eff_imag"], out["competitor"]["Phi_eff_imag"])
    )
    out["both_certified"] = bool(
        out["seed"]["krawczyk_certified"] and out["competitor"]["krawczyk_certified"]
    )
    w_s = np.asarray(out["seed"]["w_star_real"]) + 1j * np.asarray(out["seed"]["w_star_imag"])
    w_c = np.asarray(out["competitor"]["w_star_real"]) + 1j * np.asarray(out["competitor"]["w_star_imag"])
    out["box_disjoint"] = out["both_certified"] and krawczyk_boxes_disjoint(
        w_s, out["seed"]["box_radius"], w_c, out["competitor"]["box_radius"]
    )
    return out


def refine_crossing_interval_wu(
    seed_payload: dict[str, Any],
    cfg: WUConfig,
    interval: dict[str, Any],
    w_comp_end_hi: np.ndarray,
    w_comp_end_lo: np.ndarray,
    *,
    dps: int,
    max_depth: int = 16,
    target_width: float = 0.008,
) -> dict[str, Any]:
    """Bisect gamma to bracket a DeltaRe=0 crossing (general-q w,u certifier)."""
    g_hi = float(interval["gamma_hi"])
    g_lo = float(interval["gamma_lo"])

    w_seed_hi = interpolate_w_seed_from_payload(seed_payload, g_hi)
    w_seed_lo = interpolate_w_seed_from_payload(seed_payload, g_lo)

    hi_pt = _certify_pair_at_gamma_wu(cfg, g_hi, w_seed_hi, w_comp_end_hi, dps=dps)
    lo_pt = _certify_pair_at_gamma_wu(cfg, g_lo, w_seed_lo, w_comp_end_lo, dps=dps)

    history = [hi_pt, lo_pt]
    g_left, g_right = g_lo, g_hi
    w_c_left, w_c_right = w_comp_end_lo, w_comp_end_hi
    left_pt, right_pt = lo_pt, hi_pt
    delta_left_iv = lo_pt["DeltaRe_interval"]
    delta_right_iv = hi_pt["DeltaRe_interval"]

    for _ in range(max_depth):
        if abs(g_right - g_left) <= target_width:
            break
        g_mid = 0.5 * (g_left + g_right)
        w_s_mid = interpolate_w_seed_from_payload(seed_payload, g_mid)
        t = (g_mid - g_left) / (g_right - g_left) if g_right != g_left else 0.5
        w_c_mid = (1 - t) * w_c_left + t * w_c_right
        mid_pt = _certify_pair_at_gamma_wu(cfg, g_mid, w_s_mid, w_c_mid, dps=dps)
        history.append(mid_pt)
        d_mid_iv = mid_pt["DeltaRe_interval"]
        if not mid_pt.get("both_certified"):
            break
        left_cross = interval_straddles_zero(delta_left_iv) and interval_straddles_zero(d_mid_iv)
        right_cross = interval_straddles_zero(d_mid_iv) and interval_straddles_zero(delta_right_iv)
        if left_cross or (not right_cross and left_pt["DeltaRe"] * mid_pt["DeltaRe"] <= 0):
            g_right, w_c_right, delta_right_iv, right_pt = g_mid, w_c_mid, d_mid_iv, mid_pt
        else:
            g_left, w_c_left, delta_left_iv, left_pt = g_mid, w_c_mid, d_mid_iv, mid_pt

    delta_re_iv_samples = [h["DeltaRe_interval"] for h in history if h.get("both_certified")]
    flat_lo = min(iv[0] for iv in delta_re_iv_samples) if delta_re_iv_samples else None
    flat_hi = max(iv[1] for iv in delta_re_iv_samples) if delta_re_iv_samples else None

    seed_act = action_interval_from_info(left_pt["seed"])
    comp_act = action_interval_from_info(left_pt["competitor"])

    return {
        "crossing_branch_id": int(interval["branch_id"]),
        "branch_id": int(interval["branch_id"]),
        "gamma_left": float(g_left),
        "gamma_right": float(g_right),
        "gamma_interval": [float(g_left), float(g_right)],
        "gamma_width": float(g_right - g_left),
        "DeltaRe_left_interval": left_pt.get("DeltaRe_interval"),
        "DeltaRe_right_interval": right_pt.get("DeltaRe_interval"),
        "DeltaRe_bracket": [flat_lo, flat_hi] if flat_lo is not None else None,
        "crossing_contains_zero_in_Re_gap": bool(
            flat_lo is not None and flat_lo <= 0 <= flat_hi
        ),
        "seed_box": left_pt["seed"]["box"],
        "competitor_box": left_pt["competitor"]["box"],
        "box_disjoint": bool(left_pt.get("box_disjoint") and right_pt.get("box_disjoint")),
        "delta_lower_seed": left_pt["seed"].get("delta_lower"),
        "delta_lower_comp": left_pt["competitor"].get("delta_lower"),
        "action_interval_seed": seed_act,
        "action_interval_comp": comp_act,
        "refinement_history": history,
        "target_width": target_width,
        **{k: interval[k] for k in ("gamma_hi", "gamma_lo", "DeltaRe_hi", "DeltaRe_lo") if k in interval},
    }


# ---------------------------------------------------------------------------
# Branch resolution
# ---------------------------------------------------------------------------


def resolve_wu_branches(
    sweep_payload: dict[str, Any],
    seed_payload: dict[str, Any],
    cfg: WUConfig,
    *,
    branch_step_tol: float = 0.35,
    merge_same_sheet: bool = True,
    refine_all_crossings: bool = True,
    dps: int = 80,
    target_width: float = 0.008,
) -> dict[str, Any]:
    slices = extract_gamma_nodes(sweep_payload, seed_payload)
    branches = track_branches_from_nodes(slices, branch_step_tol=branch_step_tol)
    same_sheet = diagnose_same_sheet_pairs(branches)
    if merge_same_sheet:
        branches = merge_same_sheet_branches(branches, same_sheet)

    intervals: list[dict[str, Any]] = []
    for br in branches:
        intervals.extend(find_delta_re_crossing_intervals(br))

    crossing_certs: list[dict[str, Any]] = []
    if refine_all_crossings:
        for iv in intervals:
            br = next((b for b in branches if b.branch_id == iv["branch_id"]), None)
            if br is None or br.is_seed_sheet:
                continue
            pt_hi = min(br.points, key=lambda p: abs(p.gamma - iv["gamma_hi"]))
            pt_lo = min(br.points, key=lambda p: abs(p.gamma - iv["gamma_lo"]))
            if not (pt_hi.box_disjoint_from_seed and pt_lo.box_disjoint_from_seed):
                continue
            w_hi = np.asarray(pt_hi.w_real) + 1j * np.asarray(pt_hi.w_imag)
            w_lo = np.asarray(pt_lo.w_real) + 1j * np.asarray(pt_lo.w_imag)
            try:
                crossing_certs.append(
                    refine_crossing_interval_wu(
                        seed_payload, cfg, iv, w_hi, w_lo, dps=dps, target_width=target_width
                    )
                )
            except Exception:
                pass

    crossing_certs = _dedupe_crossings_by_same_sheet(branches, crossing_certs)

    def br_dict(br: CompetitorBranch) -> dict[str, Any]:
        return {
            "branch_id": br.branch_id,
            "label": br.label,
            "is_seed_sheet": br.is_seed_sheet,
            "num_points": len(br.points),
            "points": [
                {
                    "branch_id": p.branch_id,
                    "gamma": p.gamma,
                    "w_real": p.w_real,
                    "w_imag": p.w_imag,
                    "Phi_eff_real": p.Phi_eff_real,
                    "Phi_eff_imag": p.Phi_eff_imag,
                    "Phi_eff_imag_unwrapped": p.Phi_eff_imag_unwrapped,
                    "box_radius": p.box_radius,
                    "status": p.status,
                    "DeltaRe_vs_seed": p.DeltaRe_vs_seed,
                    "distance_to_seed_w": p.distance_to_seed_w,
                    "box_disjoint_from_seed": p.box_disjoint_from_seed,
                    "krawczyk_certified": p.krawczyk_certified,
                }
                for p in sorted(br.points, key=lambda x: x.gamma, reverse=True)
            ],
        }

    return {
        "r": float(sweep_payload.get("r", cfg.r)),
        "q": cfg.q,
        "chart": "wu_saddle",
        "certificate_type": "wu_branch_resolved_competitors",
        "branch_step_tol": branch_step_tol,
        "num_branches": len(branches),
        "branches": [br_dict(b) for b in branches],
        "delta_re_crossing_intervals_mesh": intervals,
        "crossing_certificates": crossing_certs,
        "same_sheet_diagnoses": same_sheet,
        "theorem_hierarchy_note": WU_DISCLAIMER,
    }


# ---------------------------------------------------------------------------
# Master pipeline
# ---------------------------------------------------------------------------


def run_robust_pipeline(
    out_dir: Path,
    *,
    cfg: Optional[WUConfig] = None,
    gamma_start: float = -0.01,
    gamma_stop: float = -1.0,
    num_points: int = 100,
    competitor_starts: int = 500,
    refine_num_points: int = 119,
    dps: int = 80,
    seed: int = 0,
    skip_continue: bool = False,
    skip_sweep: bool = False,
    skip_resolve: bool = False,
    plots: bool = True,
    no_auto_refine: bool = False,
) -> dict[str, Path]:
    """Three-step robust competitor pipeline for the BM24 w,u-saddle (general q).

    Outputs
    -------
    seed_branch_wu.json        – certified seed continuation
    competitors_robust_wu.json – sweep (+ auto-refined window)
    resolved_robust_wu.json    – branch-tracked crossing certificates
    competitors_robust_analysis_wu.png – 2×3 summary dashboard
    """
    cfg = cfg or WUConfig()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {
        "seed_json": out_dir / "seed_branch_wu.json",
        "sweep_json": out_dir / "competitors_robust_wu.json",
        "resolved_json": out_dir / "resolved_robust_wu.json",
        "plot": out_dir / "competitors_robust_analysis_wu.png",
    }

    # Step 1 – continuation
    if not skip_continue:
        gammas = make_gamma_mesh(gamma_start, gamma_stop, num_points)
        seed_payload = continue_wu_seed_branch(cfg, gammas, dps=dps)
        seed_payload["metadata"] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "pipeline": "wu_robust",
            "cfg": {"q": cfg.q, "r": cfg.r, "beta": cfg.beta},
        }
        paths["seed_json"].write_text(json.dumps(seed_payload, indent=2), encoding="utf-8")

    # Step 2 – competitor sweep
    if not skip_sweep:
        seed_payload = json.loads(paths["seed_json"].read_text(encoding="utf-8"))
        sweep_payload = dict(seed_payload)
        sweep_payload = competitor_sweep_on_wu_payload(
            sweep_payload, cfg, dps=dps, competitor_starts=competitor_starts, seed=seed
        )
        if not no_auto_refine:
            detected = detect_refine_gamma_window(
                sweep_payload, gap_threshold=1.0, spike_threshold=2.0, margin=0.05
            )
            if detected is not None:
                glo, ghi, meta = detected
                dense = np.linspace(glo, ghi, int(refine_num_points))
                sweep_payload = merge_refined_wu_window(
                    sweep_payload,
                    cfg,
                    glo,
                    ghi,
                    dense,
                    dps=dps,
                    competitor_starts=competitor_starts,
                    seed=seed + 50000,
                )
                sweep_payload.setdefault("refined_window", {})["auto_detection"] = meta
            else:
                sweep_payload["refined_window"] = {
                    "method": "auto_skipped",
                    "reason": "no_competitor_activity",
                }
        sweep_payload["competitor_sweep_completed"] = True
        paths["sweep_json"].write_text(json.dumps(sweep_payload, indent=2), encoding="utf-8")

    # Step 3 – branch resolution
    if not skip_resolve:
        sweep_payload = json.loads(paths["sweep_json"].read_text(encoding="utf-8"))
        seed_payload = json.loads(paths["seed_json"].read_text(encoding="utf-8"))
        resolved = resolve_wu_branches(sweep_payload, seed_payload, cfg, dps=dps)
        paths["resolved_json"].write_text(json.dumps(resolved, indent=2), encoding="utf-8")

    # Plots
    if plots and paths["sweep_json"].is_file() and paths["resolved_json"].is_file():
        sweep_payload = json.loads(paths["sweep_json"].read_text(encoding="utf-8"))
        resolved = json.loads(paths["resolved_json"].read_text(encoding="utf-8"))
        plot_competitor_analysis(
            sweep_payload, paths["plot"], resolved_payload=resolved, allow_overwrite=True
        )

    return paths


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="BM24 w,u-saddle robust competitor pipeline (general q)")
    p.add_argument("--out-dir", type=str, default=str(DEFAULT_RUN_DIR / "wu_robust_run"))
    p.add_argument("--gamma-start", type=float, default=-0.01)
    p.add_argument("--gamma-stop", type=float, default=-1.0)
    p.add_argument("--num-points", type=int, default=100)
    p.add_argument("--competitor-starts", type=int, default=500)
    p.add_argument("--refine-num-points", type=int, default=119)
    p.add_argument("--dps", type=int, default=80)
    p.add_argument("--q", type=int, default=3)
    p.add_argument("--r", type=float, default=DEFAULT_R)
    p.add_argument("--beta", type=float, default=SLICE_BETA)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--quick", action="store_true", help="20 mesh pts, 80 starts, no auto-refine")
    p.add_argument("--plots", action="store_true", default=True)
    p.add_argument("--skip-continue", action="store_true")
    p.add_argument("--skip-sweep", action="store_true")
    p.add_argument("--skip-resolve", action="store_true")
    p.add_argument("--no-auto-refine", action="store_true")
    args = p.parse_args()

    cfg = WUConfig(q=args.q, r=args.r, beta=args.beta)
    num_points = 20 if args.quick else args.num_points
    competitor_starts = 80 if args.quick else args.competitor_starts
    refine_num = 30 if args.quick else args.refine_num_points

    paths = run_robust_pipeline(
        Path(args.out_dir),
        cfg=cfg,
        gamma_start=args.gamma_start,
        gamma_stop=args.gamma_stop,
        num_points=num_points,
        competitor_starts=competitor_starts,
        refine_num_points=refine_num,
        dps=args.dps,
        seed=args.seed,
        skip_continue=args.skip_continue,
        skip_sweep=args.skip_sweep,
        skip_resolve=args.skip_resolve,
        plots=args.plots,
        no_auto_refine=args.no_auto_refine or args.quick,
    )
    print(json.dumps({k: str(v) for k, v in paths.items()}, indent=2))


if __name__ == "__main__":
    main()
