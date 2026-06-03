"""
Full robust competitor pipeline for BM24 z-saddle (q=3, p=1).

Mirrors ``phasecraft.w_saddle`` continue → sweep → resolve → 2×3 dashboard,
using ``krawczyk_p1_roots.SaddleSystem`` and ``picard_lefschetz.compute_phi``.

Entry: ``python -m phasecraft.bm24_saddle_audit_p1.z_robust pipeline``
"""

from __future__ import annotations

import json
import math
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from phasecraft.bm24_saddle_audit_p1.audit import _x_to_z
from phasecraft.bm24_saddle_audit_p1.continue_seed_branch_certified import (
    IM_PHI_WARNING,
    certify_z,
    conv2_full_exponent,
    polish_to_residual,
    step_from_previous_z,
    initialize_seed_at_gamma,
)
from phasecraft.krawczyk_p1_roots import SaddleSystem, discover_roots
from phasecraft.picard_lefschetz import compute_phi
from phasecraft.w_saddle.workflow import (
    SEED_DUPLICATE_W_FACTOR,
    STATUS_COMPETITOR_CERTIFIED_LOCAL,
    STATUS_COMPETITOR_PL_ACTIVE,
    STATUS_SEED_CERTIFIED_LOCAL,
    STATUS_SEED_DUPLICATE,
    BranchPoint,
    CompetitorBranch,
    build_compact_table,
    dedupe_certified_by_action,
    detect_refine_gamma_window,
    diagnose_same_sheet_pairs,
    find_delta_re_crossing_intervals,
    merge_refined_gamma_window,
    merge_same_sheet_branches,
    plot_competitor_analysis,
    unwrap_im_branch,
    unwrap_phase_diff,
    _assign_seed_sheet_branch,
    _branch_label,
    _w_norm_scale,
)

DEFAULT_Q = 3
DEFAULT_K = 8
DEFAULT_R = 176.54
DEFAULT_BETA = 0.5433996420760803
DEFAULT_RUN_DIR = (
    REPO_ROOT / "phasecraft/experiments/bm24_saddle_audit_p1/results/run_seed_branch_g-2pi"
)

Z_DISCLAIMER = (
    "BM24 z-saddle branch-resolved competitor diagnostic. Re Phi_M gaps; not PL dominance. "
    "Multiple certified z-roots may coexist on different log sheets."
)


@dataclass
class ZConfig:
    q: int = DEFAULT_Q
    K_clause: int = DEFAULT_K
    r: float = DEFAULT_R
    beta: float = DEFAULT_BETA

    def saddle(self, gamma: float) -> SaddleSystem:
        return SaddleSystem.build(
            q=self.q,
            r=self.r,
            betas=np.array([self.beta]),
            gammas=np.array([gamma]),
        )


def make_gamma_mesh(gamma_start: float, gamma_stop: float, num_points: int) -> np.ndarray:
    return np.linspace(float(gamma_start), float(gamma_stop), int(num_points))


def _log(msg: str) -> None:
    print(msg, flush=True)


def _point_from_continuation_row(row: dict[str, Any], *, proof: Optional[dict] = None) -> dict[str, Any]:
    """Convert ``continue_seed_branch_certified`` JSON row to sweep payload point."""
    certified = bool(row.get("certified")) and not bool(row.get("failed"))
    return {
        "gamma": float(row["gamma"]),
        "w_star_real": row["z_real"],
        "w_star_imag": row["z_imag"],
        "Phi_eff_real": float(row["re_phi_m"]),
        "Phi_eff_imag": float(row["im_phi_m"]),
        "Phi_eff_im_lifted": float(row["im_phi_m"]),
        "Phi_eff_im_unwrapped": float(row["im_phi_m"]),
        "residual": float(row["residual_inf"]),
        "residual_F_norm": float(row["residual_inf"]),
        "delta_lower": 0.0,
        "contraction_bound": float(row["krawczyk_contraction"]),
        "krawczyk_contraction_bound": float(row["krawczyk_contraction"]),
        "krawczyk_certified": certified,
        "krawczyk_info": proof or {},
        "box_radius": float(row.get("box_radius", 0.0)),
        "newton_ok": True,
        "full_conv2_exponent": float(row["full_conv2_exponent"]),
    }


def load_seed_payload_from_continuation_json(
    path: Path,
    *,
    stride: int = 1,
    gamma_start: Optional[float] = None,
    gamma_stop: Optional[float] = None,
) -> dict[str, Any]:
    """Reuse dense certified seed branch from an existing audit run."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    meta = data.get("metadata") or {}
    rows = [r for r in data.get("continuation", []) if r.get("certified") and not r.get("failed")]
    if gamma_start is not None:
        rows = [r for r in rows if float(r["gamma"]) <= float(gamma_start) + 1e-12]
    if gamma_stop is not None:
        rows = [r for r in rows if float(r["gamma"]) >= float(gamma_stop) - 1e-12]
    rows = sorted(rows, key=lambda r: float(r["gamma"]), reverse=True)
    if stride > 1:
        rows = rows[:: int(stride)]
    points = [_point_from_continuation_row(r) for r in rows]
    im_raw = [float(p["Phi_eff_imag"]) for p in points]
    im_u, unwrap_ok = unwrap_im_branch(im_raw)
    for p, imv in zip(points, im_u):
        p["Phi_eff_im_unwrapped"] = imv
    gammas = [float(p["gamma"]) for p in points]
    return {
        "r": float(meta.get("r", DEFAULT_R)),
        "q": int(meta.get("q", DEFAULT_Q)),
        "K_clause": int(meta.get("K_clause", DEFAULT_K)),
        "beta": float(meta.get("beta", DEFAULT_BETA)),
        "chart": "z_saddle",
        "gamma_mesh": gammas,
        "num_points_requested": len(gammas),
        "num_points_completed": len(points),
        "completed_full_mesh": True,
        "certificate_type": "z_seed_branch_reused",
        "reused_from": str(path),
        "reuse_stride": int(stride),
        "im_phi_warning": IM_PHI_WARNING,
        "points": points,
        "unwrap_continuous": bool(unwrap_ok),
        "table": build_compact_table(points),
    }


def z_from_record(rec: dict[str, Any]) -> np.ndarray:
    if "z_real" in rec:
        return np.asarray(rec["z_real"], dtype=float) + 1j * np.asarray(rec["z_imag"], dtype=float)
    if "w_real" in rec:
        return np.asarray(rec["w_real"], dtype=float) + 1j * np.asarray(rec["w_imag"], dtype=float)
    return np.asarray(rec["w_star_real"], dtype=float) + 1j * np.asarray(rec["w_star_imag"], dtype=float)


def cluster_z_roots(roots: list[np.ndarray], z_ref: np.ndarray, cluster_tol: float) -> list[np.ndarray]:
    scale = max(float(np.max(np.abs(z_ref))), 1e-3)
    tol = float(cluster_tol) * scale
    out: list[np.ndarray] = []
    for z in roots:
        z = np.asarray(z, dtype=complex)
        if not any(np.linalg.norm(z - c, ord=np.inf) < tol for c in out):
            out.append(z)
    return out


def _box_radii(info: dict[str, Any]) -> list[float]:
    br = info.get("box_radius", 0.0)
    if isinstance(br, list):
        return [float(x) for x in br]
    return [float(br)] * 8


def z_boxes_disjoint(z_seed: np.ndarray, rho_seed: list[float], z: np.ndarray, rho: list[float]) -> bool:
    n = min(len(z_seed), len(z), len(rho_seed), len(rho))
    for i in range(n):
        if abs(z_seed[i] - z[i]) >= rho_seed[i] + rho[i]:
            return True
    return False


def classify_vs_seed_z(node: dict[str, Any], z_seed: np.ndarray, rho_seed: list[float]) -> dict[str, Any]:
    z = z_from_record(node)
    rho = _box_radii(node.get("krawczyk_info") or {"box_radius": node.get("box_radius", 0.0)})
    while len(rho) < len(z):
        rho.append(rho[-1] if rho else 0.0)
    while len(rho_seed) < len(z_seed):
        rho_seed.append(rho_seed[-1] if rho_seed else 0.0)
    z_dist = float(np.linalg.norm(z - z_seed, ord=np.inf))
    rho_s = float(min(rho_seed)) if rho_seed else 0.0
    rho_c = float(min(rho)) if rho else 0.0
    w_thresh = SEED_DUPLICATE_W_FACTOR * (rho_s + rho_c)
    overlap = not z_boxes_disjoint(z_seed, rho_seed, z, rho)
    is_seed = bool(node.get("is_seed_sheet"))
    is_dup = (z_dist < w_thresh or overlap) and not is_seed
    return {
        "z_distance_inf": z_dist,
        "w_distance_inf": z_dist,
        "z_duplicate_threshold": w_thresh,
        "box_disjoint_from_seed": not overlap,
        "is_seed_duplicate": is_dup,
    }


def z_root_record(
    cfg: ZConfig,
    z: np.ndarray,
    gamma: float,
    *,
    certified: bool,
    info: dict[str, Any],
    residual: float,
) -> dict[str, Any]:
    sys = cfg.saddle(gamma)
    phi = compute_phi(z, q=cfg.q, r=cfg.r, betas=sys.betas, gammas=sys.gammas)
    return {
        "z_real": z.real.tolist(),
        "z_imag": z.imag.tolist(),
        "w_real": z.real.tolist(),
        "w_imag": z.imag.tolist(),
        "residual": float(residual),
        "Phi_eff_real": float(phi.real),
        "Phi_eff_imag": float(phi.imag),
        "delta_lower": float(info.get("delta_lower", 0.0)) if certified else None,
        "contraction_bound": float(info.get("contraction_bound", math.inf)),
        "krawczyk_certified": bool(certified),
        "krawczyk_info": info,
        "full_conv2_exponent": conv2_full_exponent(phi.real, cfg.K_clause, cfg.r),
    }


def certify_z_point(cfg: ZConfig, z: np.ndarray, gamma: float, *, dps: int) -> tuple[bool, dict[str, Any]]:
    sys = cfg.saddle(gamma)
    ok, info = certify_z(sys, z, dps=dps)
    return bool(ok), dict(info)


def discover_z_numerical(
    cfg: ZConfig,
    gamma: float,
    z_seed: np.ndarray,
    *,
    num_starts: int,
    seed: int,
    tol: float = 1e-12,
) -> list[np.ndarray]:
    sys = cfg.saddle(gamma)
    roots_x = discover_roots(sys, num_starts=num_starts, seed=seed, tol=tol)
    n = sys.nvars
    zs = [_x_to_z(x, n) for x in roots_x]
    scale = max(float(np.max(np.abs(z_seed))) * 8.0, 0.5)
    rng = np.random.default_rng(seed + 17)
    x0 = np.concatenate([z_seed.real, z_seed.imag])
    sol_roots: list[np.ndarray] = []
    for _ in range(max(0, num_starts // 4)):
        x_pol, res = polish_to_residual(sys, x0 + rng.normal(0, scale, size=2 * n), min_residual=1e-8, dps=60)
        if res <= 1e-7:
            z = _x_to_z(x_pol, n)
            if not any(np.linalg.norm(z - s, ord=np.inf) < 1e-5 for s in zs + sol_roots):
                sol_roots.append(z)
    return cluster_z_roots(zs + sol_roots, z_seed, cluster_tol=1e-5)


def sweep_z_at_gamma(
    cfg: ZConfig,
    z_seed: np.ndarray,
    gamma: float,
    *,
    num_starts: int,
    seed: int,
    dps: int,
    cluster_tol: float = 1e-5,
    seed_ident_tol: float = 1e-4,
) -> dict[str, Any]:
    numerical = discover_z_numerical(cfg, gamma, z_seed, num_starts=num_starts, seed=seed)
    clustered = cluster_z_roots(numerical, z_seed, cluster_tol)
    certified_rows: list[dict[str, Any]] = []
    for z in clustered:
        sys = cfg.saddle(gamma)
        x_pol, res = polish_to_residual(
            sys, np.concatenate([z.real, z.imag]), min_residual=1e-8, dps=max(60, dps)
        )
        z_pol = _x_to_z(x_pol, sys.nvars)
        res = float(np.linalg.norm(sys.G_complex(z_pol), ord=np.inf))
        if not np.isfinite(res) or res > 1e-7:
            continue
        ok, info = certify_z_point(cfg, z_pol, gamma, dps=dps)
        if not ok:
            continue
        certified_rows.append(z_root_record(cfg, z_pol, gamma, certified=True, info=info, residual=res))
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
    dists = [float(np.linalg.norm(z_from_record(r) - z_seed, ord=np.inf)) for r in certified_rows]
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
        competitors.append({**row, "gap_re": gap_re, "phase_diff": unwrap_phase_diff(phi_seed_im, row["Phi_eff_imag"])})
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
        "best_competitor_re_phi": float(max((c["Phi_eff_real"] for c in competitors), default=phi_seed_re)),
        "seed_dominates_all": bool(competitors and min_gap is not None and min_gap > 0),
        "closest_competitor": min(competitors, key=lambda c: c["gap_re"]) if competitors else None,
        "competitors": competitors,
    }


def interpolate_z_from_payload(payload: dict[str, Any], gamma: float) -> np.ndarray:
    pts = sorted(payload.get("points", []), key=lambda p: float(p["gamma"]))
    if not pts:
        raise ValueError("empty seed payload")
    g = float(gamma)
    gs = [float(p["gamma"]) for p in pts]
    if g <= min(gs):
        return z_from_record(pts[0 if gs[0] <= gs[-1] else -1])
    if g >= max(gs):
        return z_from_record(pts[-1 if gs[0] <= gs[-1] else 0])
    for i in range(len(pts) - 1):
        g0, g1 = float(pts[i]["gamma"]), float(pts[i + 1]["gamma"])
        if (g0 <= g <= g1) or (g1 <= g <= g0):
            t = (g - g0) / (g1 - g0) if abs(g1 - g0) > 1e-15 else 0.0
            z0, z1 = z_from_record(pts[i]), z_from_record(pts[i + 1])
            return (1 - t) * z0 + t * z1
    ref = min(pts, key=lambda p: abs(float(p["gamma"]) - g))
    return z_from_record(ref)


def _continuation_row_to_point(row: Any, *, certified: bool, proof: dict) -> dict[str, Any]:
    d = row.to_row() if hasattr(row, "to_row") else dict(row)
    return {
        "gamma": float(d["gamma"]),
        "w_star_real": d["z_real"],
        "w_star_imag": d["z_imag"],
        "Phi_eff_real": float(d["re_phi_m"]),
        "Phi_eff_imag": float(d["im_phi_m"]),
        "Phi_eff_im_lifted": float(d["im_phi_m"]),
        "Phi_eff_im_unwrapped": float(d["im_phi_m"]),
        "residual": float(d["residual_inf"]),
        "residual_F_norm": float(d["residual_inf"]),
        "delta_lower": 0.0,
        "contraction_bound": float(d["krawczyk_contraction"]),
        "krawczyk_contraction_bound": float(d["krawczyk_contraction"]),
        "krawczyk_certified": bool(certified),
        "krawczyk_info": proof,
        "box_radius": float(d.get("box_radius", 0.0)),
        "newton_ok": True,
        "full_conv2_exponent": float(d["full_conv2_exponent"]),
    }


def continue_z_seed_on_mesh(
    cfg: ZConfig,
    gammas: np.ndarray,
    *,
    dps: int = 80,
    min_residual: float = 1e-10,
    match_tol: float = 0.02,
    gamma_step: float = 0.01,
    min_step: float = 0.0005,
    n_values: Optional[list[int]] = None,
) -> dict[str, Any]:
    """Certified seed branch at each mesh γ (w_saddle-style; not adaptive-only)."""
    n_values = n_values or list(range(12, 23))
    mesh = [float(g) for g in np.asarray(gammas, dtype=float)]
    if len(mesh) < 1:
        raise ValueError("empty gamma mesh")
    decreasing = mesh[-1] < mesh[0]
    points: list[dict[str, Any]] = []
    z_cur: Optional[np.ndarray] = None
    step_index = 0

    for j, g_target in enumerate(mesh):
        if j == 0:
            z_cur, row0, _ = initialize_seed_at_gamma(
                cfg.q,
                cfg.K_clause,
                cfg.r,
                cfg.beta,
                g_target,
                n_values=n_values,
                match_tol=match_tol,
                min_residual=min_residual,
                dps=dps,
                use_iterator=True,
            )
            points.append(_continuation_row_to_point(row0, certified=True, proof={}))
            step_index = 1
            _log(f"  mesh continue [{j + 1}/{len(mesh)}] γ={g_target:.6g} (init)")
            continue

        assert z_cur is not None
        gamma = float(points[-1]["gamma"])
        z_work = z_cur
        inner_step = min(abs(float(gamma_step)), abs(gamma - g_target) * 0.5)
        reached = False
        while not reached:
            trial = gamma - inner_step if decreasing else gamma + inner_step
            if decreasing:
                trial = max(g_target, trial)
            else:
                trial = min(g_target, trial)
            z_next, row = step_from_previous_z(
                cfg.q,
                cfg.K_clause,
                cfg.r,
                cfg.beta,
                trial,
                z_work,
                step_index=step_index,
                n_values=n_values,
                match_tol=match_tol,
                min_residual=min_residual,
                dps=dps,
            )
            if z_next is None:
                inner_step *= 0.5
                if inner_step < min_step:
                    points.append(_continuation_row_to_point(row, certified=False, proof={}))
                    _log(f"  mesh continue STOP at γ={g_target:.6g} (sub-step failed)")
                    break
                continue
            z_work = z_next
            gamma = float(trial)
            step_index += 1
            inner_step = min(abs(float(gamma_step)), inner_step * 1.25)
            if abs(gamma - g_target) < 1e-11:
                points.append(_continuation_row_to_point(row, certified=True, proof={}))
                z_cur = z_work
                reached = True
        if not reached:
            break
        _log(f"  mesh continue [{j + 1}/{len(mesh)}] γ={g_target:.6g}")

    im_raw = [float(p["Phi_eff_imag"]) for p in points]
    im_u, unwrap_ok = unwrap_im_branch(im_raw)
    for p, imv in zip(points, im_u):
        p["Phi_eff_im_unwrapped"] = imv

    return {
        "r": cfg.r,
        "q": cfg.q,
        "K_clause": cfg.K_clause,
        "beta": cfg.beta,
        "chart": "z_saddle",
        "gamma_mesh": mesh,
        "num_points_requested": len(mesh),
        "num_points_completed": len(points),
        "completed_full_mesh": len(points) == len(mesh),
        "certificate_type": "z_seed_branch_mesh_continuation",
        "im_phi_warning": IM_PHI_WARNING,
        "points": points,
        "unwrap_continuous": bool(unwrap_ok),
        "table": build_compact_table(points),
    }


# Back-compat alias
continue_z_seed_branch = continue_z_seed_on_mesh


def competitor_sweep_on_z_payload(
    payload: dict[str, Any],
    cfg: ZConfig,
    *,
    dps: int,
    competitor_starts: int,
    cluster_tol: float = 1e-5,
    seed: int = 0,
) -> dict[str, Any]:
    pts = payload.get("points", [])
    n_sweep = sum(1 for r in pts if r.get("krawczyk_certified", True))
    _log(f"  competitor sweep: {n_sweep} certified γ points × {competitor_starts} starts")
    done = 0
    t0 = time.time()
    for j, row in enumerate(pts):
        if not row.get("krawczyk_certified", True):
            continue
        gamma = float(row["gamma"])
        z_seed = z_from_record(row)
        sweep = sweep_z_at_gamma(
            cfg,
            z_seed,
            gamma,
            num_starts=competitor_starts,
            seed=seed + j + 1,
            dps=dps,
            cluster_tol=cluster_tol,
        )
        row["competitor_sweep"] = sweep
        row["num_numerical_roots"] = sweep["num_numerical_roots"]
        row["num_certified_roots"] = sweep["num_certified_roots"]
        row["num_certified_competitors"] = sweep["num_certified_competitors"]
        row["min_certified_gap_re"] = sweep["min_certified_gap_re"]
        row["min_positive_gap_re"] = sweep["min_positive_gap_re"]
        row["best_competitor_re_phi"] = sweep["best_competitor_re_phi"]
        row["seed_dominates_all"] = sweep["seed_dominates_all"]
        row["closest_competitor"] = sweep["closest_competitor"]
        done += 1
        if done == 1 or done % 5 == 0 or done == n_sweep:
            elapsed = time.time() - t0
            eta = (elapsed / done) * (n_sweep - done) if done else 0.0
            _log(
                f"    sweep {done}/{n_sweep} γ={gamma:.4g} "
                f"cert_roots={sweep['num_certified_roots']} "
                f"({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)"
            )
    payload["competitor_sweep_completed"] = True
    payload["table"] = build_compact_table(payload["points"])
    return payload


def _sweep_row_at_gamma(
    payload: dict[str, Any],
    cfg: ZConfig,
    gamma: float,
    *,
    dps: int,
    competitor_starts: int,
    cluster_tol: float,
    seed: int,
) -> dict[str, Any]:
    z_guess = interpolate_z_from_payload(payload, gamma)
    sys = cfg.saddle(gamma)
    x_pol, res = polish_to_residual(
        sys, np.concatenate([z_guess.real, z_guess.imag]), min_residual=1e-8, dps=dps
    )
    z_star = _x_to_z(x_pol, sys.nvars)
    ok, info = certify_z_point(cfg, z_star, gamma, dps=dps)
    phi = compute_phi(z_star, q=cfg.q, r=cfg.r, betas=sys.betas, gammas=sys.gammas)
    sweep = sweep_z_at_gamma(
        cfg,
        z_star,
        gamma,
        num_starts=competitor_starts,
        seed=seed,
        dps=dps,
        cluster_tol=cluster_tol,
    )
    return {
        "gamma": float(gamma),
        "w_star_real": z_star.real.tolist(),
        "w_star_imag": z_star.imag.tolist(),
        "Phi_eff_real": float(phi.real),
        "Phi_eff_imag": float(phi.imag),
        "residual": float(res),
        "residual_F_norm": float(res),
        "krawczyk_certified": bool(ok),
        "krawczyk_info": info,
        "box_radius": float(min(_box_radii(info)) if ok else 0.0),
        "competitor_sweep": sweep,
        "num_certified_roots": sweep["num_certified_roots"],
        "num_certified_competitors": sweep["num_certified_competitors"],
        "min_certified_gap_re": sweep["min_certified_gap_re"],
        "seed_dominates_all": sweep["seed_dominates_all"],
        "closest_competitor": sweep["closest_competitor"],
    }


def merge_refined_z_window(
    payload: dict[str, Any],
    cfg: ZConfig,
    gamma_lo: float,
    gamma_hi: float,
    dense_gammas: np.ndarray,
    *,
    dps: int,
    competitor_starts: int,
    seed: int,
) -> dict[str, Any]:
    gamma_lo, gamma_hi = float(gamma_lo), float(gamma_hi)
    if gamma_lo > gamma_hi:
        gamma_lo, gamma_hi = gamma_hi, gamma_lo
    kept = [p for p in payload["points"] if float(p["gamma"]) < gamma_lo or float(p["gamma"]) > gamma_hi]
    refined = [
        _sweep_row_at_gamma(
            payload,
            cfg,
            float(g),
            dps=dps,
            competitor_starts=competitor_starts,
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


def _node_from_z_record(
    rec: dict[str, Any],
    gamma: float,
    *,
    z_seed: np.ndarray,
    rho_seed: list[float],
    is_seed_sheet: bool,
) -> dict[str, Any]:
    info = rec.get("krawczyk_info") or {}
    rho = _box_radii(info if info else {"box_radius": rec.get("box_radius", 0.0)})
    node = {
        "gamma": float(gamma),
        "w_real": rec["w_real"],
        "w_imag": rec["w_imag"],
        "Phi_eff_real": float(rec["Phi_eff_real"]),
        "Phi_eff_imag": float(rec["Phi_eff_imag"]),
        "box_radius": float(min(rho)) if rho else 0.0,
        "delta_lower": float(rec.get("delta_lower") or 0.0),
        "krawczyk_certified": bool(rec.get("krawczyk_certified", True)),
        "distance_to_seed_w": float(np.linalg.norm(z_from_record(rec) - z_seed, ord=np.inf)),
        "is_seed_sheet": bool(is_seed_sheet),
        "krawczyk_info": info,
    }
    node.update(classify_vs_seed_z(node, z_seed, rho_seed))
    return node


def extract_gamma_nodes_z(
    sweep_payload: dict[str, Any],
    seed_payload: dict[str, Any],
    *,
    seed_ident_tol: float = 1e-3,
) -> list[dict[str, Any]]:
    r = float(sweep_payload.get("r", seed_payload.get("r", DEFAULT_R)))
    rows = sorted(sweep_payload.get("points", []), key=lambda p: float(p["gamma"]), reverse=True)
    out: list[dict[str, Any]] = []
    for row in rows:
        gamma = float(row["gamma"])
        z_seed = z_from_record(row)
        rho_seed = _box_radii(row.get("krawczyk_info") or {"box_radius": row.get("box_radius", 1e-16)})
        sweep = row.get("competitor_sweep") or {}
        roots = list(sweep.get("certified_roots") or [])
        if not roots:
            continue
        nodes = []
        for rec in roots:
            z = z_from_record(rec)
            dist = float(np.linalg.norm(z - z_seed, ord=np.inf))
            is_seed = bool(rec.get("is_seed_branch")) or dist < seed_ident_tol
            nodes.append(_node_from_z_record(rec, gamma, z_seed=z_seed, rho_seed=rho_seed, is_seed_sheet=is_seed))
        out.append(
            {
                "gamma": gamma,
                "r": r,
                "w_seed": z_seed,
                "rho_seed": rho_seed,
                "seed_Phi_real": float(row["Phi_eff_real"]),
                "seed_Phi_imag": float(row.get("Phi_eff_im_lifted", row["Phi_eff_imag"])),
                "seed_RePhi_interval": [float(row["Phi_eff_real"]), float(row["Phi_eff_real"])],
                "nodes": nodes,
            }
        )
    return out


def track_branches_z(
    gamma_slices: list[dict[str, Any]],
    *,
    branch_step_tol: float = 0.35,
    use_tangent: bool = True,
    re_weight: float = 0.15,
    im_weight: float = 0.10,
) -> list[CompetitorBranch]:
    """Match certified z-roots along gamma (w_saddle track_branches_from_nodes, principal Im)."""
    if not gamma_slices:
        return []

    branches: list[CompetitorBranch] = []
    next_id = 0

    def start_branch(_node: dict[str, Any]) -> CompetitorBranch:
        nonlocal next_id
        br = CompetitorBranch(branch_id=next_id, is_seed_sheet=False)
        next_id += 1
        return br

    def append_point(br: CompetitorBranch, sl: dict[str, Any], node: dict[str, Any]) -> BranchPoint:
        delta_re = float(sl["seed_Phi_real"]) - float(node["Phi_eff_real"])
        delta_im = float(unwrap_phase_diff(sl["seed_Phi_imag"], node["Phi_eff_imag"]))
        if node.get("is_seed_duplicate"):
            status = STATUS_SEED_DUPLICATE
        elif node.get("is_seed_sheet"):
            status = STATUS_SEED_CERTIFIED_LOCAL
        elif not node.get("box_disjoint_from_seed", True):
            status = STATUS_SEED_DUPLICATE
        elif delta_re < 0:
            status = STATUS_COMPETITOR_PL_ACTIVE
        else:
            status = STATUS_COMPETITOR_CERTIFIED_LOCAL
        seed_re_iv = sl.get("seed_RePhi_interval") or [sl["seed_Phi_real"], sl["seed_Phi_real"]]
        comp_re_iv = [node["Phi_eff_real"], node["Phi_eff_real"]]
        delta_re_iv = [float(seed_re_iv[0] - comp_re_iv[1]), float(seed_re_iv[1] - comp_re_iv[0])]
        return BranchPoint(
            branch_id=br.branch_id,
            gamma=float(node["gamma"]),
            w_real=list(node["w_real"]),
            w_imag=list(node["w_imag"]),
            Phi_eff_real=float(node["Phi_eff_real"]),
            Phi_eff_imag=float(node["Phi_eff_imag"]),
            Phi_eff_imag_unwrapped=float(node["Phi_eff_imag"]),
            box_radius=float(node["box_radius"]),
            delta_lower=float(node.get("delta_lower", 0.0)),
            status=status,
            distance_to_seed_w=float(node["distance_to_seed_w"]),
            DeltaRe_vs_seed=delta_re,
            DeltaIm_vs_seed=delta_im,
            krawczyk_certified=bool(node["krawczyk_certified"]),
            is_seed_duplicate=bool(node.get("is_seed_duplicate")),
            box_disjoint_from_seed=bool(node.get("box_disjoint_from_seed", True)),
            RePhi_interval=list(comp_re_iv),
            DeltaRe_interval_vs_seed=delta_re_iv,
            log_branch_method="principal_log_PhiM_z_chart",
        )

    def tracking_nodes(sl: dict[str, Any]) -> list[dict[str, Any]]:
        return [n for n in sl["nodes"] if not n.get("is_seed_duplicate")]

    def match_cost(br: CompetitorBranch, node: dict[str, Any], sl: dict[str, Any]) -> float:
        z_end = z_from_record({"w_real": br.points[-1].w_real, "w_imag": br.points[-1].w_imag})
        z_n = z_from_record(node)
        if use_tangent and len(br.points) >= 2:
            p1, p2 = br.points[-2], br.points[-1]
            z1 = z_from_record({"w_real": p1.w_real, "w_imag": p1.w_imag})
            dg = float(p1.gamma - p2.gamma)
            if abs(dg) > 1e-14:
                t = (float(sl["gamma"]) - float(p2.gamma)) / dg
                z_pred = z_end + t * (z_end - z1)
            else:
                z_pred = z_end
        else:
            z_pred = z_end
        scale = _w_norm_scale(z_end) * max(branch_step_tol, 0.05)
        d_z = float(np.linalg.norm(z_n - z_pred, ord=np.inf)) / scale
        d_re = abs(float(node["Phi_eff_real"]) - br.points[-1].Phi_eff_real) / max(abs(br.points[-1].Phi_eff_real), 1.0)
        d_im = abs(
            float(unwrap_phase_diff(br.points[-1].Phi_eff_imag_unwrapped, float(node["Phi_eff_imag"])))
        ) / (2.0 * np.pi)
        return d_z + re_weight * d_re + im_weight * d_im

    first = gamma_slices[0]
    for node in tracking_nodes(first):
        br = start_branch(node)
        br.points.append(append_point(br, first, node))
        branches.append(br)

    for sl in gamma_slices[1:]:
        open_branches = [b for b in branches if b.points]
        endpoints = [
            (b, z_from_record({"w_real": b.points[-1].w_real, "w_imag": b.points[-1].w_imag}))
            for b in open_branches
        ]
        nodes = tracking_nodes(sl)
        unused = list(range(len(nodes)))
        matches: list[tuple[float, int, int]] = []
        for bi, (br, z_end) in enumerate(endpoints):
            scale = _w_norm_scale(z_end) * max(branch_step_tol, 0.05)
            for ni in unused:
                z_n = z_from_record(nodes[ni])
                if float(np.linalg.norm(z_n - z_end, ord=np.inf)) <= scale:
                    matches.append((match_cost(br, nodes[ni], sl), bi, ni))
        matches.sort(key=lambda t: t[0])
        used_b: set[int] = set()
        used_n: set[int] = set()
        for _cost, bi, ni in matches:
            if bi in used_b or ni in used_n:
                continue
            used_b.add(bi)
            used_n.add(ni)
            open_branches[bi].points.append(append_point(open_branches[bi], sl, nodes[ni]))
        for ni in unused:
            if ni in used_n:
                continue
            br = start_branch(nodes[ni])
            br.points.append(append_point(br, sl, nodes[ni]))
            branches.append(br)

    _assign_seed_sheet_branch(branches)
    for br in branches:
        br.label = _branch_label(br)
    return branches


def resolve_z_branches(
    sweep_payload: dict[str, Any],
    seed_payload: dict[str, Any],
    *,
    branch_step_tol: float = 0.35,
    merge_same_sheet: bool = True,
    dps: int = 80,
) -> dict[str, Any]:
    slices = extract_gamma_nodes_z(sweep_payload, seed_payload)
    branches = track_branches_z(slices, branch_step_tol=branch_step_tol)
    same_sheet = diagnose_same_sheet_pairs(branches)
    if merge_same_sheet:
        branches = merge_same_sheet_branches(branches, same_sheet)
    intervals: list[dict[str, Any]] = []
    for br in branches:
        intervals.extend(find_delta_re_crossing_intervals(br))
    # Crossing bisection uses WSaddleSystem in w_saddle.workflow; z chart keeps mesh intervals only.
    crossing_certs: list[dict[str, Any]] = []

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
        "r": float(sweep_payload.get("r", DEFAULT_R)),
        "q": int(sweep_payload.get("q", DEFAULT_Q)),
        "chart": "z_saddle",
        "certificate_type": "z_branch_resolved_competitors",
        "branch_step_tol": branch_step_tol,
        "num_branches": len(branches),
        "branches": [br_dict(b) for b in branches],
        "delta_re_crossing_intervals_mesh": intervals,
        "crossing_certificates": crossing_certs,
        "same_sheet_diagnoses": same_sheet,
        "theorem_hierarchy_note": Z_DISCLAIMER,
    }


def run_robust_pipeline(
    out_dir: Path,
    *,
    cfg: Optional[ZConfig] = None,
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
    reuse_seed_json: Optional[Union[str, Path]] = None,
    sweep_stride: int = 1,
) -> dict[str, Path]:
    cfg = cfg or ZConfig()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "seed_json": out_dir / "seed_branch_z.json",
        "sweep_json": out_dir / "competitors_robust_z.json",
        "resolved_json": out_dir / "resolved_robust_z.json",
        "plot": out_dir / "competitors_robust_analysis_v1_z.png",
    }

    t_pipeline = time.time()

    if not skip_continue:
        if reuse_seed_json is not None:
            _log(f"[1/4] Reusing seed continuation: {reuse_seed_json}")
            seed_payload = load_seed_payload_from_continuation_json(
                Path(reuse_seed_json),
                stride=sweep_stride,
                gamma_start=gamma_start,
                gamma_stop=gamma_stop,
            )
        else:
            gammas = make_gamma_mesh(gamma_start, gamma_stop, num_points)
            _log(
                f"[1/4] Mesh continuation: {len(gammas)} γ from {gamma_start} to {gamma_stop} "
                f"(~{len(gammas) * 0.5:.0f}s certify + hours for sweep)"
            )
            seed_payload = continue_z_seed_on_mesh(cfg, gammas, dps=dps)
        seed_payload["metadata"] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "pipeline": "z_robust",
            "cfg": cfg.__dict__,
            "competitor_starts": competitor_starts,
            "sweep_stride": sweep_stride,
        }
        paths["seed_json"].write_text(json.dumps(seed_payload, indent=2), encoding="utf-8")
        _log(
            f"  wrote {paths['seed_json']} ({seed_payload['num_points_completed']} points, "
            f"full_mesh={seed_payload.get('completed_full_mesh')})"
        )

    if not skip_sweep:
        _log(f"[2/4] Multi-start competitor sweep (dps={dps})")
        seed_payload = json.loads(paths["seed_json"].read_text(encoding="utf-8"))
        sweep_payload = dict(seed_payload)
        sweep_payload = competitor_sweep_on_z_payload(
            sweep_payload, cfg, dps=dps, competitor_starts=competitor_starts, seed=seed
        )
        if not no_auto_refine:
            detected = detect_refine_gamma_window(
                sweep_payload,
                gap_threshold=1.0,
                spike_threshold=2.0,
                margin=0.05,
            )
            if detected is not None:
                glo, ghi, meta = detected
                dense = np.linspace(glo, ghi, int(refine_num_points))
                sweep_payload = merge_refined_z_window(
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
                sweep_payload["refined_window"] = {"method": "auto_skipped", "reason": "no_competitor_activity"}
        sweep_payload["competitor_sweep_completed"] = True
        paths["sweep_json"].write_text(json.dumps(sweep_payload, indent=2), encoding="utf-8")
        _log(f"  wrote {paths['sweep_json']}")

    if not skip_resolve:
        _log("[3/4] Branch resolve + crossing intervals")
        sweep_payload = json.loads(paths["sweep_json"].read_text(encoding="utf-8"))
        seed_payload = json.loads(paths["seed_json"].read_text(encoding="utf-8"))
        resolved = resolve_z_branches(sweep_payload, seed_payload, dps=dps)
        paths["resolved_json"].write_text(json.dumps(resolved, indent=2), encoding="utf-8")
        _log(f"  wrote {paths['resolved_json']} ({resolved.get('num_branches', 0)} branches)")

    if plots and paths["sweep_json"].is_file() and paths["resolved_json"].is_file():
        _log("[4/4] Plot 2×3 dashboard")
        sweep_payload = json.loads(paths["sweep_json"].read_text(encoding="utf-8"))
        resolved = json.loads(paths["resolved_json"].read_text(encoding="utf-8"))
        plot_competitor_analysis(
            sweep_payload,
            paths["plot"],
            resolved_payload=resolved,
            allow_overwrite=True,
        )
        _log(f"  wrote {paths['plot']}")

    elapsed = time.time() - t_pipeline
    _log(f"Done in {elapsed:.1f}s")
    if (
        not skip_sweep
        and competitor_starts >= 200
        and paths["sweep_json"].is_file()
        and elapsed < 120
    ):
        n_pts = len(json.loads(paths["sweep_json"].read_text())["points"])
        if n_pts >= 20:
            _log(
                "WARNING: finished very quickly for a heavy sweep. "
                "You may be viewing competitors_robust_analysis_z.png from "
                "plot_z_competitor_robust_dashboard (instant JSON replot), not this pipeline. "
                f"Check {paths['plot']} and {paths['sweep_json']}."
            )
    return paths


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="BM24 z-saddle robust competitor pipeline")
    p.add_argument("--out-dir", type=str, default=str(DEFAULT_RUN_DIR / "z_robust_run"))
    p.add_argument("--gamma-start", type=float, default=-0.01)
    p.add_argument("--gamma-stop", type=float, default=-1.0)
    p.add_argument("--num-points", type=int, default=100)
    p.add_argument("--competitor-starts", type=int, default=500)
    p.add_argument("--refine-num-points", type=int, default=119)
    p.add_argument("--dps", type=int, default=80)
    p.add_argument("--r", type=float, default=DEFAULT_R)
    p.add_argument("--beta", type=float, default=DEFAULT_BETA)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--quick", action="store_true", help="20 mesh pts, 80 starts, no wall refine")
    p.add_argument("--plots", action="store_true", default=True)
    p.add_argument("--skip-continue", action="store_true")
    p.add_argument("--skip-sweep", action="store_true")
    p.add_argument("--skip-resolve", action="store_true")
    p.add_argument("--no-auto-refine", action="store_true")
    p.add_argument(
        "--reuse-seed-json",
        type=str,
        default="",
        help="Skip continuation; load certified rows from seed_branch_continuation.json",
    )
    p.add_argument(
        "--sweep-stride",
        type=int,
        default=1,
        help="When reusing seed JSON, keep every Nth certified γ (reduces sweep cost)",
    )
    args = p.parse_args()

    _log(
        "z_robust_workflow (multi-start sweep + resolve) — NOT plot_z_competitor_robust_dashboard"
    )

    cfg = ZConfig(r=args.r, beta=args.beta)
    num_points = 20 if args.quick else args.num_points
    competitor_starts = 80 if args.quick else args.competitor_starts
    refine_num = 30 if args.quick else args.refine_num_points

    reuse = args.reuse_seed_json.strip() or None
    if reuse is None and not args.skip_continue:
        default_seed = DEFAULT_RUN_DIR / "seed_branch_continuation.json"
        if default_seed.is_file() and num_points >= 50:
            _log(
                f"Tip: dense seed branch exists at {default_seed}. "
                "Use --reuse-seed-json with --sweep-stride 5 to avoid re-continuing."
            )

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
        reuse_seed_json=reuse,
        sweep_stride=max(1, int(args.sweep_stride)),
    )
    print(json.dumps({k: str(v) for k, v in paths.items()}, indent=2))


if __name__ == "__main__":
    main()
