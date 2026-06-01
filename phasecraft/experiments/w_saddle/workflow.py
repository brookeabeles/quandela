"""
W-saddle proof pipeline: gamma continuation, competitor sweeps, branch resolution, plots.

Run: ``python -m phasecraft.w_saddle <subcommand> --help``
Core interval math: ``phasecraft.w_saddle.core``
Certificate wording: ``phasecraft.certificate_language``
Outputs: ``phasecraft/w_saddle/runs/``

Merges the former ``continue_w_saddle_branch`` and ``competitor_branch_tracking`` modules.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from phasecraft.certificate_language import (
    ALLOWED_GAMMA_CONVENTIONS,
    CERTIFICATE_TYPE_BRANCH_RESOLVED,
    CERTIFICATE_TYPE_DISCRETE,
    CERTIFICATE_TYPE_TRACKED_COMPETITOR,
    COMPETITOR_DISCLAIMER,
    finalize_payload_certificate,
    lean_proof_note,
)
from phasecraft.w_saddle.core import (
    WSaddleSystem,
    Delta,
    certify_log_delta_ratio_step,
    discover_w_roots,
    initial_log_delta_lift,
    krawczyk_certify_with_escalation,
    krawczyk_certify_w_root,
    phi_eff_with_lifted_log,
    solve_w_from_init,
)


def _attach_krawczyk_fields(row: dict[str, Any], info: dict[str, Any]) -> None:
    """Copy certified-root and optional action-interval fields onto a mesh row."""
    for key in (
        "action_center_real",
        "action_center_imag",
        "action_interval_real",
        "action_interval_imag",
        "action_width_real",
        "action_width_imag",
        "log_branch_method",
        "krawczyk_inclusion",
    ):
        if key in info:
            row[key] = info[key]


@dataclass
class StopReason:
    code: str
    message: str
    index: int
    gamma: float


def resolve_plot_output_path(plot_path: Path, *, allow_overwrite: bool = False) -> Path:
    """Return plot_path, or a non-colliding variant if the file already exists."""
    plot_path = Path(plot_path)
    if allow_overwrite or not plot_path.exists():
        return plot_path
    stem, suffix = plot_path.stem, plot_path.suffix
    for n in range(1, 1000):
        candidate = plot_path.with_name(f"{stem}_v{n}{suffix}")
        if not candidate.exists():
            print(
                f"Plot output exists; writing {candidate} (not overwriting {plot_path})",
                file=sys.stderr,
            )
            return candidate
    raise FileExistsError(f"Could not find free variant of {plot_path}")


def make_gamma_mesh(gamma_start: float, gamma_stop: float, num_points: int) -> np.ndarray:
    if num_points < 2:
        raise ValueError("num_points must be >= 2")
    if gamma_start >= 0 or gamma_stop >= 0:
        raise ValueError("gamma_start and gamma_stop must be negative (seed chamber)")
    if gamma_stop >= gamma_start:
        raise ValueError("gamma_stop must be more negative than gamma_start (decreasing mesh)")
    return np.linspace(gamma_start, gamma_stop, num_points)


def unwrap_im_branch(im_raw: list[float]) -> tuple[list[float], bool]:
    """Post-hoc unwrap of principal Im(Phi); diagnostic only.

    Prefer ``Phi_eff_im_lifted`` from the certified ratio-lift along the mesh
    (``continue_negative_gamma_branch``) for Stokes / branch-wise analysis.
    """
    if not im_raw:
        return [], True
    out = [float(im_raw[0])]
    continuous = True
    for k in range(1, len(im_raw)):
        delta = float(im_raw[k]) - float(im_raw[k - 1])
        if abs(delta) > np.pi * 0.95:
            continuous = False
        while delta > np.pi:
            delta -= 2.0 * np.pi
        while delta < -np.pi:
            delta += 2.0 * np.pi
        out.append(out[-1] + delta)
    return out, continuous


def leading_re_phi_eff(r: float, gamma: float) -> float:
    """Memo Eq. (5.9): r/4 * (3 sin^2(gamma/4) - sin(gamma/2))."""
    return float((r / 4.0) * (3.0 * np.sin(gamma / 4.0) ** 2 - np.sin(gamma / 2.0)))


def unwrap_phase_diff(im_seed: float, im_other: float) -> float:
    """Principal-branch difference Im(seed) - Im(other), wrapped to (-pi, pi]."""
    delta = float(im_seed) - float(im_other)
    while delta > np.pi:
        delta -= 2.0 * np.pi
    while delta <= -np.pi:
        delta += 2.0 * np.pi
    return delta


def cluster_numerical_roots(
    roots: list[np.ndarray],
    cluster_tol: float = 1e-5,
    w_ref: Optional[np.ndarray] = None,
) -> list[np.ndarray]:
    """Merge roots within L-infinity tolerance (scaled by |w_ref| when given)."""
    scale = max(float(np.max(np.abs(w_ref))), 1e-3) if w_ref is not None else 1.0
    tol_eff = float(cluster_tol) * scale
    clusters: list[np.ndarray] = []
    for w in roots:
        w = np.asarray(w, dtype=complex)
        if not any(np.linalg.norm(w - c, ord=np.inf) < tol_eff for c in clusters):
            clusters.append(w)
    return clusters


def dedupe_certified_by_action(
    rows: list[dict[str, Any]],
    re_tol: float = 1e-9,
    im_tol: float = 1e-9,
) -> list[dict[str, Any]]:
    """Drop certified roots with duplicate (Re Phi, Im Phi) after clustering."""
    unique: list[dict[str, Any]] = []
    for row in rows:
        if any(
            abs(row["Phi_eff_real"] - u["Phi_eff_real"]) < re_tol
            and abs(row["Phi_eff_imag"] - u["Phi_eff_imag"]) < im_tol
            for u in unique
        ):
            continue
        unique.append(row)
    return unique


def _root_record(
    sys: WSaddleSystem,
    w: np.ndarray,
    *,
    certified: bool,
    info: dict[str, Any],
    residual: float,
) -> dict[str, Any]:
    phi = sys.Phi_eff(w)
    return {
        "w_real": w.real.tolist(),
        "w_imag": w.imag.tolist(),
        "residual": float(residual),
        "Phi_eff_real": float(phi.real),
        "Phi_eff_imag": float(phi.imag),
        "delta_lower": float(info.get("delta_lower", 0.0)) if certified else None,
        "contraction_bound": float(info.get("contraction_bound", np.inf)),
        "krawczyk_certified": bool(certified),
        "krawczyk_info": info,
    }


def sweep_certified_competitors_at_gamma(
    sys: WSaddleSystem,
    w_seed: np.ndarray,
    *,
    num_starts: int,
    seed: int,
    dps: int,
    max_inflate_iters: int,
    escalate: bool,
    cluster_tol: float = 1e-5,
    newton_tol: float = 1e-12,
    seed_ident_tol: float = 1e-4,
) -> dict[str, Any]:
    """Random-start discovery, cluster, Krawczyk-certify, and gap vs seed branch."""
    w_seed = np.asarray(w_seed, dtype=complex)
    search_scale = max(float(np.max(np.abs(w_seed))) * 8.0, 0.08)
    numerical = discover_w_roots(
        sys,
        num_starts=num_starts,
        seed=seed,
        w_init=w_seed,
        tol=newton_tol,
        search_scale=search_scale,
    )
    clustered = cluster_numerical_roots(
        numerical, cluster_tol=cluster_tol, w_ref=w_seed
    )

    certified_rows: list[dict[str, Any]] = []
    for w in clustered:
        res = float(np.linalg.norm(sys.F_complex(w), ord=np.inf))
        if not np.isfinite(res) or res > 1e-7:
            continue
        ok, info = certify_point(
            sys,
            w,
            dps=dps,
            max_inflate_iters=max_inflate_iters,
            escalate=escalate,
        )
        if not ok:
            continue
        certified_rows.append(
            _root_record(sys, w, certified=True, info=info, residual=res)
        )

    certified_rows = dedupe_certified_by_action(certified_rows)

    if not certified_rows:
        return {
            "num_numerical_roots": len(numerical),
            "num_clustered_roots": len(clustered),
            "num_certified_roots": 0,
            "certified_roots": [],
            "seed_index": None,
            "min_certified_gap_re": None,
            "closest_competitor": None,
            "competitors": [],
        }

    dists: list[float] = []
    for row in certified_rows:
        w = np.array(row["w_real"]) + 1j * np.array(row["w_imag"])
        dists.append(float(np.linalg.norm(w - w_seed, ord=np.inf)))

    seed_index = int(np.argmin(dists))
    for i, row in enumerate(certified_rows):
        row["distance_to_seed"] = dists[i]
        row["is_seed_branch"] = bool(i == seed_index and dists[i] < seed_ident_tol)

    phi_seed_re = certified_rows[seed_index]["Phi_eff_real"]
    phi_seed_im = certified_rows[seed_index]["Phi_eff_imag"]

    competitors: list[dict[str, Any]] = []
    for i, row in enumerate(certified_rows):
        if i == seed_index:
            continue
        gap_re = float(phi_seed_re - row["Phi_eff_real"])
        competitors.append(
            {
                **row,
                "gap_re": gap_re,
                "phase_diff": unwrap_phase_diff(phi_seed_im, row["Phi_eff_imag"]),
            }
        )

    closest: Optional[dict[str, Any]] = None
    min_gap: Optional[float] = None
    min_positive_gap: Optional[float] = None
    best_competitor_re: Optional[float] = None
    if competitors:
        closest = min(competitors, key=lambda c: c["gap_re"])
        min_gap = float(closest["gap_re"])
        best_competitor_re = float(max(c["Phi_eff_real"] for c in competitors))
        pos_gaps = [c["gap_re"] for c in competitors if c["gap_re"] > 0]
        min_positive_gap = float(min(pos_gaps)) if pos_gaps else None

    return {
        "num_numerical_roots": len(numerical),
        "num_clustered_roots": len(clustered),
        "num_certified_roots": len(certified_rows),
        "num_certified_competitors": len(competitors),
        "certified_roots": certified_rows,
        "seed_index": seed_index,
        "min_certified_gap_re": min_gap,
        "min_positive_gap_re": min_positive_gap,
        "best_competitor_re_phi": best_competitor_re,
        "seed_dominates_all": bool(competitors and min_gap is not None and min_gap > 0),
        "closest_competitor": closest,
        "competitors": competitors,
        "search_scale": search_scale,
    }


def certify_point(
    sys: WSaddleSystem,
    w: np.ndarray,
    *,
    dps: int,
    max_inflate_iters: int,
    escalate: bool,
    enclose_action: bool = False,
    log_branch_method: str = "continuation_ratio_lift",
) -> tuple[bool, dict[str, Any]]:
    if escalate:
        return krawczyk_certify_with_escalation(
            sys,
            w,
            dps_start=dps,
            max_inflate_iters=max_inflate_iters,
            enclose_action=enclose_action,
            log_branch_method=log_branch_method,
        )
    return krawczyk_certify_w_root(
        sys,
        w,
        dps=dps,
        max_inflate_iters=max_inflate_iters,
        enclose_action=enclose_action,
        log_branch_method=log_branch_method,
    )


def continue_negative_gamma_branch(
    r: float,
    gammas: np.ndarray,
    *,
    w_init: Optional[np.ndarray] = None,
    dps: int = 60,
    max_inflate_iters: int = 6,
    escalate: bool = True,
    delta_lower_threshold: float = 1e-6,
    contraction_max: float = 1.0,
    branch_jump_tol: float = np.pi * 0.95,
    newton_tol: float = 1e-12,
    certify_competitors: bool = False,
    stop_on_competitor: bool = False,
    competitor_starts: int = 500,
    competitor_re_tol: float = 1e-4,
    cluster_tol: float = 1e-5,
    seed: int = 0,
    enclose_action: bool = False,
    gamma_convention: Optional[str] = None,
) -> dict[str, Any]:
    """Run gamma continuation; stop on first failed proof predicate."""
    points: list[dict[str, Any]] = []
    stop: Optional[StopReason] = None
    w_prev: Optional[np.ndarray] = w_init
    log_delta_lifted: Optional[complex] = None
    sys_prev: Optional[WSaddleSystem] = None
    info_prev: Optional[dict[str, Any]] = None

    for j, gamma in enumerate(gammas):
        sys = WSaddleSystem(r=float(r), gamma=float(gamma))

        if w_prev is None:
            w_prev = sys.leading_seed()
        w_star, newton_ok, res_norm = solve_w_from_init(sys, w_prev, tol=newton_tol)
        if not newton_ok:
            stop = StopReason(
                "newton_failed",
                f"Newton did not converge from warm start (residual={res_norm:.3e})",
                j,
                float(gamma),
            )
            break

        certified, info = certify_point(
            sys,
            w_star,
            dps=dps,
            max_inflate_iters=max_inflate_iters,
            escalate=escalate,
            enclose_action=enclose_action,
            log_branch_method="continuation_ratio_lift",
        )

        phi = sys.Phi_eff(w_star)

        if j == 0:
            log_delta_lifted = initial_log_delta_lift(w_star, beta=sys.beta, p=sys.p)
            log_step_ok = True
            log_step_info: dict[str, Any] = {
                "log_ratio_real": 0.0,
                "log_ratio_imag": 0.0,
                "log_branch_step_certified": True,
                "log_branch_method": "continuation_ratio_lift",
            }
        else:
            assert sys_prev is not None and info_prev is not None and w_prev is not None
            log_step_ok, log_step_info = certify_log_delta_ratio_step(
                sys_prev, w_prev, info_prev, sys, w_star, info
            )
            log_delta_lifted = log_delta_lifted + complex(
                log_step_info["log_ratio_real"], log_step_info["log_ratio_imag"]
            )

        assert log_delta_lifted is not None
        phi_lifted = phi_eff_with_lifted_log(sys, w_star, log_delta_lifted)

        contraction = float(info.get("contraction_bound", np.inf))
        delta_lower = float(info.get("delta_lower", 0.0))

        phi_leading = leading_re_phi_eff(float(r), float(gamma))
        row: dict[str, Any] = {
            "index": int(j),
            "gamma": float(gamma),
            "w_star_real": w_star.real.tolist(),
            "w_star_imag": w_star.imag.tolist(),
            "Phi_eff_real": float(phi.real),
            "Phi_eff_imag": float(phi.imag),
            "Phi_eff_re_lifted": float(phi_lifted.real),
            "Phi_eff_im_lifted": float(phi_lifted.imag),
            "Phi_eff_im_unwrapped": float(phi_lifted.imag),
            "log_Delta_lifted_real": float(log_delta_lifted.real),
            "log_Delta_lifted_imag": float(log_delta_lifted.imag),
            "log_ratio_real": float(log_step_info.get("log_ratio_real", 0.0)),
            "log_ratio_imag": float(log_step_info.get("log_ratio_imag", 0.0)),
            "log_branch_step_certified": bool(log_step_info.get("log_branch_step_certified", False)),
            "log_branch_method": log_step_info.get("log_branch_method", "continuation_ratio_lift"),
            "Phi_eff_re_leading": phi_leading,
            "residual": float(res_norm),
            "residual_F_norm": float(res_norm),
            "delta_lower": delta_lower,
            "contraction_bound": contraction,
            "krawczyk_contraction_bound": contraction,
            "krawczyk_certified": bool(certified),
            "krawczyk_info": info,
            "newton_ok": bool(newton_ok),
        }
        _attach_krawczyk_fields(row, info)

        if certify_competitors:
            sweep = sweep_certified_competitors_at_gamma(
                sys,
                w_star,
                num_starts=competitor_starts,
                seed=seed + j + 1,
                dps=dps,
                max_inflate_iters=max_inflate_iters,
                escalate=escalate,
                cluster_tol=cluster_tol,
                newton_tol=newton_tol,
            )
            row["competitor_sweep"] = sweep
            row["num_numerical_roots"] = sweep["num_numerical_roots"]
            row["num_certified_roots"] = sweep["num_certified_roots"]
            row["min_certified_gap_re"] = sweep["min_certified_gap_re"]
            row["closest_competitor"] = sweep["closest_competitor"]
            if (
                stop_on_competitor
                and sweep["min_certified_gap_re"] is not None
                and float(sweep["min_certified_gap_re"]) <= competitor_re_tol
            ):
                stop = StopReason(
                    "competing_root",
                    f"Certified competitor within Re(Phi) gap {competitor_re_tol} of seed",
                    j,
                    float(gamma),
                )
                points.append(row)
                break

        points.append(row)

        if not log_step_ok:
            stop = StopReason(
                "log_branch_cut_crossing",
                "Delta ratio on certified boxes may cross negative real axis",
                j,
                float(gamma),
            )
            break

        if not certified:
            stop = StopReason(
                "krawczyk_failed",
                str(info.get("reason", "Krawczyk certification failed")),
                j,
                float(gamma),
            )
            break
        if contraction >= contraction_max:
            stop = StopReason(
                "contraction_bound",
                f"|I-YJ|={contraction:.6g} >= {contraction_max}",
                j,
                float(gamma),
            )
            break
        if delta_lower <= delta_lower_threshold:
            stop = StopReason(
                "divisor_approach",
                f"delta_lower={delta_lower:.6g} <= {delta_lower_threshold}",
                j,
                float(gamma),
            )
            break

        sys_prev = sys
        info_prev = info
        w_prev = w_star

    im_raw = [float(p["Phi_eff_imag"]) for p in points]
    im_principal_unwrap, unwrap_ok = unwrap_im_branch(im_raw)
    for row, im_pu in zip(points, im_principal_unwrap):
        row["Phi_eff_im_principal_unwrap"] = float(im_pu)

    lift_ok = all(bool(p.get("log_branch_step_certified", False)) for p in points)
    if not lift_ok and stop is None and points:
        stop = StopReason(
            "log_branch_cut_crossing",
            "log Delta ratio-lift step failed certification on mesh",
            len(points) - 1,
            float(points[-1]["gamma"]),
        )

    payload: dict[str, Any] = {
        "r": float(r),
        "gamma_mesh": gammas.tolist(),
        "num_points_requested": int(len(gammas)),
        "num_points_completed": int(len(points)),
        "completed_full_mesh": stop is None and len(points) == len(gammas),
        "certificate_type": CERTIFICATE_TYPE_DISCRETE,
        "stop": None if stop is None else {
            "code": stop.code,
            "message": stop.message,
            "index": stop.index,
            "gamma": stop.gamma,
        },
        "points": points,
        "log_branch_lift_certified": bool(lift_ok and stop is None),
        "log_branch_method": "continuation_ratio_lift",
        "unwrap_continuous": bool(unwrap_ok),
        "Phi_eff_im_method": "continuation_ratio_lift",
        "table": build_compact_table(points),
    }
    return payload


def build_compact_table(points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flat continuation table for plotting / analysis."""
    rows: list[dict[str, Any]] = []
    for p in points:
        rows.append(
            {
                "gamma": p["gamma"],
                "Re_Phi": p["Phi_eff_real"],
                "Im_Phi": p["Phi_eff_imag"],
                "Im_Phi_lifted": p.get("Phi_eff_im_lifted", p.get("Phi_eff_im_unwrapped", p["Phi_eff_imag"])),
                "Im_Phi_unwrapped": p.get("Phi_eff_im_unwrapped", p["Phi_eff_imag"]),
                "Im_Phi_principal_unwrap": p.get("Phi_eff_im_principal_unwrap", p["Phi_eff_imag"]),
                "log_Delta_lifted_imag": p.get("log_Delta_lifted_imag"),
                "Re_Phi_leading": p.get("Phi_eff_re_leading"),
                "delta_lower": p["delta_lower"],
                "contraction_bound": p["contraction_bound"],
                "residual": p["residual"],
                "krawczyk_certified": p["krawczyk_certified"],
                "w_star_real": p["w_star_real"],
                "w_star_imag": p["w_star_imag"],
                "num_numerical_roots": p.get("num_numerical_roots"),
                "num_certified_roots": p.get("num_certified_roots"),
                "min_certified_gap_re": p.get("min_certified_gap_re"),
                "min_positive_gap_re": p.get("min_positive_gap_re"),
                "num_certified_competitors": p.get("competitor_sweep", {}).get(
                    "num_certified_competitors"
                ),
                "seed_dominates_all": p.get("competitor_sweep", {}).get("seed_dominates_all"),
            }
        )
    return rows


def interpolate_w_seed_from_payload(payload: dict[str, Any], gamma: float) -> np.ndarray:
    """Linear interpolation of w_star between nearest mesh points in gamma."""
    pts = sorted(payload["points"], key=lambda p: float(p["gamma"]), reverse=True)
    gamma = float(gamma)
    for i in range(len(pts) - 1):
        g_hi = float(pts[i]["gamma"])
        g_lo = float(pts[i + 1]["gamma"])
        if g_hi >= gamma >= g_lo:
            t = (gamma - g_hi) / (g_lo - g_hi) if g_lo != g_hi else 0.0
            w_hi = np.array(pts[i]["w_star_real"]) + 1j * np.array(pts[i]["w_star_imag"])
            w_lo = np.array(pts[i + 1]["w_star_real"]) + 1j * np.array(pts[i + 1]["w_star_imag"])
            return (1.0 - t) * w_hi + t * w_lo
    if gamma >= float(pts[0]["gamma"]):
        p = pts[0]
    else:
        p = pts[-1]
    return np.array(p["w_star_real"]) + 1j * np.array(p["w_star_imag"])


def _sweep_row_from_gamma(
    payload: dict[str, Any],
    gamma: float,
    *,
    dps: int,
    max_inflate_iters: int,
    escalate: bool,
    competitor_starts: int,
    cluster_tol: float,
    newton_tol: float,
    seed: int,
) -> dict[str, Any]:
    """One continuation row + competitor sweep at gamma."""
    r = float(payload["r"])
    sys = WSaddleSystem(r=r, gamma=float(gamma))
    w_guess = interpolate_w_seed_from_payload(payload, gamma)
    w_star, newton_ok, res_norm = solve_w_from_init(sys, w_guess, tol=newton_tol)
    if not newton_ok:
        w_star = w_guess
    certified, info = certify_point(
        sys, w_star, dps=dps, max_inflate_iters=max_inflate_iters, escalate=escalate
    )
    phi = sys.Phi_eff(w_star)
    sweep = sweep_certified_competitors_at_gamma(
        sys,
        w_star,
        num_starts=competitor_starts,
        seed=seed,
        dps=dps,
        max_inflate_iters=max_inflate_iters,
        escalate=escalate,
        cluster_tol=cluster_tol,
        newton_tol=newton_tol,
    )
    return {
        "gamma": float(gamma),
        "w_star_real": w_star.real.tolist(),
        "w_star_imag": w_star.imag.tolist(),
        "Phi_eff_real": float(phi.real),
        "Phi_eff_imag": float(phi.imag),
        "Phi_eff_re_leading": leading_re_phi_eff(r, gamma),
        "residual": float(res_norm),
        "residual_F_norm": float(res_norm),
        "delta_lower": float(info.get("delta_lower", 0.0)) if certified else None,
        "contraction_bound": float(info.get("contraction_bound", np.inf)),
        "krawczyk_contraction_bound": float(info.get("contraction_bound", np.inf)),
        "krawczyk_certified": bool(certified),
        "krawczyk_info": info,
        "newton_ok": bool(newton_ok),
        "competitor_sweep": sweep,
        "num_numerical_roots": sweep["num_numerical_roots"],
        "num_certified_roots": sweep["num_certified_roots"],
        "num_certified_competitors": sweep["num_certified_competitors"],
        "min_certified_gap_re": sweep["min_certified_gap_re"],
        "min_positive_gap_re": sweep["min_positive_gap_re"],
        "best_competitor_re_phi": sweep["best_competitor_re_phi"],
        "seed_dominates_all": sweep["seed_dominates_all"],
        "closest_competitor": sweep["closest_competitor"],
    }


def _mesh_gamma_step(pts: list[dict[str, Any]]) -> float:
    gammas = sorted(float(p["gamma"]) for p in pts)
    if len(gammas) < 2:
        return 0.01
    diffs = [abs(gammas[i + 1] - gammas[i]) for i in range(len(gammas) - 1)]
    return float(np.median(diffs))


def _row_refine_signals(row: dict[str, Any], *, gap_threshold: float, spike_threshold: float) -> list[str]:
    """Reasons this mesh point should be in a dense resweep window."""
    reasons: list[str] = []
    phi_seed = float(row.get("Phi_eff_real", 0.0))
    min_gap = row.get("min_certified_gap_re")
    min_pos = row.get("min_positive_gap_re")
    n_comp = int(row.get("num_certified_competitors") or 0)
    best = row.get("best_competitor_re_phi")

    if min_gap is not None and float(min_gap) <= 0.0:
        reasons.append("competitor_beats_seed")
    if min_gap is not None and float(min_gap) < gap_threshold:
        reasons.append("small_certified_gap")
    if min_pos is not None and float(min_pos) < gap_threshold:
        reasons.append("small_positive_gap")
    if n_comp >= 2:
        reasons.append("multiple_certified_competitors")
    if best is not None and abs(float(best) - phi_seed) > spike_threshold:
        reasons.append("diagnostic_re_spike")
    return reasons


def _contiguous_gamma_runs(
    gammas: list[float], *, max_gap: float
) -> list[tuple[float, float]]:
    """Merge sorted gammas into [lo, hi] intervals separated by gaps > max_gap."""
    if not gammas:
        return []
    runs: list[list[float]] = [[gammas[0]]]
    for g in gammas[1:]:
        if abs(g - runs[-1][-1]) <= max_gap:
            runs[-1].append(g)
        else:
            runs.append([g])
    return [(min(r), max(r)) for r in runs]


def detect_refine_gamma_window(
    payload: dict[str, Any],
    *,
    gap_threshold: float = 1.0,
    spike_threshold: float = 2.0,
    margin: float = 0.05,
    min_span: float = 0.08,
    min_flagged_points: int = 3,
    merge_run_gap: float = 0.12,
) -> Optional[tuple[float, float, dict[str, Any]]]:
    """Pick [gamma_lo, gamma_hi] for dense resweep from a coarse competitor sweep.

    Uses only mesh diagnostics (gaps, competitor counts, diagnostic Re spikes)—
    no hard-coded wall location. Returns None if nothing interesting is found.
    """
    pts = list(payload.get("points") or [])
    if len(pts) < 4:
        return None

    mesh_min = min(float(p["gamma"]) for p in pts)
    mesh_max = max(float(p["gamma"]) for p in pts)
    step = _mesh_gamma_step(pts)
    max_gap = max(1.5 * step, 0.015)

    flagged: list[tuple[float, list[str]]] = []
    for row in pts:
        reasons = _row_refine_signals(
            row, gap_threshold=gap_threshold, spike_threshold=spike_threshold
        )
        if reasons:
            flagged.append((float(row["gamma"]), reasons))

    # Do not use diagnostic_re_spike for the window (coarse per-gamma max-Re artifact).
    high_priority = [
        (g, rs)
        for g, rs in flagged
        if "competitor_beats_seed" in rs
        or "small_certified_gap" in rs
        or "small_positive_gap" in rs
    ]
    if len(high_priority) >= min_flagged_points:
        flagged = high_priority
    elif len(flagged) < min_flagged_points:
        fallback: list[float] = []
        for row in pts:
            if int(row.get("num_certified_competitors") or 0) >= 1:
                fallback.append(float(row["gamma"]))
        if len(fallback) < min_flagged_points:
            return None
        flagged = [(g, ["fallback_has_competitor"]) for g in fallback]

    runs = _contiguous_gamma_runs(sorted(g for g, _ in flagged), max_gap=max_gap)
    if not runs:
        return None

    def run_score(iv: tuple[float, float]) -> tuple[int, int, float]:
        lo, hi = iv
        in_run = [(g, rs) for g, rs in flagged if lo - 1e-12 <= g <= hi + 1e-12]
        n_threat = sum(1 for _, rs in in_run if "competitor_beats_seed" in rs)
        n_spike = sum(1 for _, rs in in_run if "diagnostic_re_spike" in rs)
        worst_gap = min(
            (float(p.get("min_certified_gap_re")) for p in pts if lo <= float(p["gamma"]) <= hi and p.get("min_certified_gap_re") is not None),
            default= 0.0,
        )
        return (n_threat + 2 * n_spike, len(in_run), worst_gap)

    runs_sorted = sorted(runs, key=run_score, reverse=True)
    merged_lo, merged_hi = runs_sorted[0]
    for lo, hi in runs_sorted[1:]:
        if lo <= merged_hi + merge_run_gap:
            merged_lo = min(merged_lo, lo)
            merged_hi = max(merged_hi, hi)

    gamma_lo = max(mesh_min, merged_lo - margin)
    gamma_hi = min(mesh_max, merged_hi + margin)
    if gamma_hi - gamma_lo < min_span:
        mid = 0.5 * (gamma_lo + gamma_hi)
        half = 0.5 * min_span
        gamma_lo = max(mesh_min, mid - half)
        gamma_hi = min(mesh_max, mid + half)

    meta = {
        "method": "auto_from_coarse_sweep",
        "gap_threshold": gap_threshold,
        "spike_threshold": spike_threshold,
        "margin": margin,
        "num_flagged_points": len(flagged),
        "flagged_gammas": [g for g, _ in flagged],
        "flag_reasons": {f"{g:.6g}": rs for g, rs in flagged},
        "contiguous_runs_before_merge": [[lo, hi] for lo, hi in runs],
        "selected_run": [merged_lo, merged_hi],
        "mesh_gamma_range": [mesh_min, mesh_max],
    }
    return float(gamma_lo), float(gamma_hi), meta


def merge_refined_gamma_window(
    payload: dict[str, Any],
    gamma_lo: float,
    gamma_hi: float,
    dense_gammas: np.ndarray,
    *,
    dps: int,
    max_inflate_iters: int,
    escalate: bool,
    competitor_starts: int,
    cluster_tol: float,
    newton_tol: float,
    seed: int,
) -> dict[str, Any]:
    """Replace points in [gamma_lo, gamma_hi] with a denser re-swept window."""
    gamma_lo, gamma_hi = float(gamma_lo), float(gamma_hi)
    if gamma_lo > gamma_hi:
        gamma_lo, gamma_hi = gamma_hi, gamma_lo
    kept = [
        p
        for p in payload["points"]
        if float(p["gamma"]) < gamma_lo or float(p["gamma"]) > gamma_hi
    ]
    refined: list[dict[str, Any]] = []
    for j, gamma in enumerate(dense_gammas):
        refined.append(
            _sweep_row_from_gamma(
                payload,
                float(gamma),
                dps=dps,
                max_inflate_iters=max_inflate_iters,
                escalate=escalate,
                competitor_starts=competitor_starts,
                cluster_tol=cluster_tol,
                newton_tol=newton_tol,
                seed=seed + 10000 + j,
            )
        )
    merged = sorted(kept + refined, key=lambda p: float(p["gamma"]), reverse=True)
    out = dict(payload)
    out["points"] = merged
    out["gamma_mesh"] = [float(p["gamma"]) for p in merged]
    out["num_points_completed"] = len(merged)
    out["num_points_requested"] = len(merged)
    rw = dict(payload.get("refined_window") or {})
    rw.update(
        {
            "gamma_lo": gamma_lo,
            "gamma_hi": gamma_hi,
            "num_dense_points": len(dense_gammas),
        }
    )
    out["refined_window"] = rw
    out["table"] = build_compact_table(merged)
    return out


def competitor_sweep_on_payload(
    payload: dict[str, Any],
    *,
    dps: int,
    max_inflate_iters: int,
    escalate: bool,
    competitor_starts: int,
    cluster_tol: float,
    newton_tol: float,
    seed: int = 0,
) -> dict[str, Any]:
    """Run certified competitor sweep on each point of an existing continuation JSON."""
    r = float(payload["r"])
    for j, row in enumerate(payload.get("points", [])):
        gamma = float(row["gamma"])
        sys = WSaddleSystem(r=r, gamma=gamma)
        w_seed = np.array(row["w_star_real"]) + 1j * np.array(row["w_star_imag"])
        sweep = sweep_certified_competitors_at_gamma(
            sys,
            w_seed,
            num_starts=competitor_starts,
            seed=seed + j + 1,
            dps=dps,
            max_inflate_iters=max_inflate_iters,
            escalate=escalate,
            cluster_tol=cluster_tol,
            newton_tol=newton_tol,
        )
        row["competitor_sweep"] = sweep
        row["num_numerical_roots"] = sweep["num_numerical_roots"]
        row["num_certified_roots"] = sweep["num_certified_roots"]
        row["min_certified_gap_re"] = sweep["min_certified_gap_re"]
        row["min_positive_gap_re"] = sweep["min_positive_gap_re"]
        row["num_certified_competitors"] = sweep["num_certified_competitors"]
        row["best_competitor_re_phi"] = sweep["best_competitor_re_phi"]
        row["seed_dominates_all"] = sweep["seed_dominates_all"]
        row["closest_competitor"] = sweep["closest_competitor"]
    payload["table"] = build_compact_table(payload["points"])
    payload["competitor_sweep_completed"] = True
    return payload


def nearest_sweep_point(
    payload: dict[str, Any], gamma: float, *, tol: float = 0.025
) -> dict[str, Any]:
    """Return sweep row whose gamma is closest to the requested anchor."""
    pts = payload.get("points") or []
    if not pts:
        raise ValueError("sweep payload has no points")
    row = min(pts, key=lambda p: abs(float(p["gamma"]) - float(gamma)))
    if abs(float(row["gamma"]) - float(gamma)) > tol:
        raise ValueError(
            f"no sweep point within {tol} of anchor gamma={gamma} "
            f"(nearest {row['gamma']})"
        )
    return row


def pick_tracked_competitor_at_row(row: dict[str, Any]) -> dict[str, Any]:
    """Select the Re(Phi)-dominant certified competitor that beats the seed."""
    sweep = row.get("competitor_sweep") or {}
    competitors: list[dict[str, Any]] = list(sweep.get("competitors") or [])
    if not competitors:
        raise ValueError(f"no certified competitors at gamma={row['gamma']}")
    beating = [c for c in competitors if float(c["gap_re"]) < 0.0]
    pool = beating if beating else competitors
    return max(pool, key=lambda c: float(c["Phi_eff_real"]))


def w_from_root_record(rec: dict[str, Any]) -> np.ndarray:
    if "w_star_real" in rec:
        return np.array(rec["w_star_real"]) + 1j * np.array(rec["w_star_imag"])
    return np.array(rec["w_real"]) + 1j * np.array(rec["w_imag"])


def make_gamma_arms(
    anchor: float,
    step: float,
    gamma_hi: float,
    gamma_lo: float,
) -> tuple[list[float], list[float]]:
    """Mesh arms from anchor: toward less negative (hi) and toward more negative (lo)."""
    if step <= 0:
        raise ValueError("track gamma step must be positive")
    toward_hi: list[float] = []
    g = float(anchor) + float(step)
    while g <= float(gamma_hi) + 1e-12:
        toward_hi.append(float(g))
        g += float(step)
    toward_lo: list[float] = []
    g = float(anchor) - float(step)
    while g >= float(gamma_lo) - 1e-12:
        toward_lo.append(float(g))
        g -= float(step)
    return toward_hi, toward_lo


def _tracked_root_row(
    sys: WSaddleSystem,
    w: np.ndarray,
    *,
    gamma: float,
    index: int,
    certified: bool,
    info: dict[str, Any],
    residual: float,
    newton_ok: bool,
) -> dict[str, Any]:
    phi = sys.Phi_eff(w)
    return {
        "index": int(index),
        "gamma": float(gamma),
        "w_star_real": w.real.tolist(),
        "w_star_imag": w.imag.tolist(),
        "Phi_eff_real": float(phi.real),
        "Phi_eff_imag": float(phi.imag),
        "Phi_eff_re_leading": leading_re_phi_eff(float(sys.r), float(gamma)),
        "residual": float(residual),
        "delta_lower": float(info.get("delta_lower", 0.0)) if certified else None,
        "contraction_bound": float(info.get("contraction_bound", np.inf)),
        "krawczyk_certified": bool(certified),
        "krawczyk_info": info,
        "newton_ok": bool(newton_ok),
    }


def certify_tracked_root_at_gamma(
    r: float,
    gamma: float,
    w_init: np.ndarray,
    *,
    index: int,
    dps: int,
    max_inflate_iters: int,
    escalate: bool,
    newton_tol: float,
    delta_lower_threshold: float,
    contraction_max: float,
    enclose_action: bool = False,
) -> tuple[Optional[dict[str, Any]], Optional[StopReason]]:
    sys = WSaddleSystem(r=float(r), gamma=float(gamma))
    w_star, newton_ok, res_norm = solve_w_from_init(sys, w_init, tol=newton_tol)
    if not newton_ok:
        return None, StopReason(
            "newton_failed",
            f"Newton did not converge (residual={res_norm:.3e})",
            index,
            float(gamma),
        )
    certified, info = certify_point(
        sys,
        w_star,
        dps=dps,
        max_inflate_iters=max_inflate_iters,
        escalate=escalate,
        enclose_action=enclose_action,
        log_branch_method="tracked_branch_continuation",
    )
    row = _tracked_root_row(
        sys,
        w_star,
        gamma=gamma,
        index=index,
        certified=certified,
        info=info,
        residual=res_norm,
        newton_ok=newton_ok,
    )
    _attach_krawczyk_fields(row, info)
    if not certified:
        return row, StopReason(
            "krawczyk_failed",
            str(info.get("reason", "Krawczyk certification failed")),
            index,
            float(gamma),
        )
    contraction = float(info.get("contraction_bound", np.inf))
    delta_lower = float(info.get("delta_lower", 0.0))
    if contraction >= contraction_max:
        return row, StopReason(
            "contraction_bound",
            f"|I-YJ|={contraction:.6g} >= {contraction_max}",
            index,
            float(gamma),
        )
    if delta_lower <= delta_lower_threshold:
        return row, StopReason(
            "motion_divisor_approach",
            f"delta_lower={delta_lower:.6g} <= {delta_lower_threshold}",
            index,
            float(gamma),
        )
    return row, None


def continue_tracked_root_arm(
    r: float,
    gammas: list[float],
    w_start: np.ndarray,
    *,
    arm_name: str,
    dps: int,
    max_inflate_iters: int,
    escalate: bool,
    newton_tol: float,
    delta_lower_threshold: float,
    contraction_max: float,
    enclose_action: bool = False,
) -> tuple[list[np.ndarray], list[dict[str, Any]], Optional[StopReason]]:
    """Warm-start continuation along one gamma arm; returns w chain and certified rows."""
    w_prev = np.asarray(w_start, dtype=complex)
    rows: list[dict[str, Any]] = []
    for j, gamma in enumerate(gammas):
        row, stop = certify_tracked_root_at_gamma(
            r,
            gamma,
            w_prev,
            index=j,
            dps=dps,
            max_inflate_iters=max_inflate_iters,
            escalate=escalate,
            newton_tol=newton_tol,
            delta_lower_threshold=delta_lower_threshold,
            contraction_max=contraction_max,
            enclose_action=enclose_action,
        )
        if row is None:
            return [w_prev], rows, stop
        rows.append(row)
        if stop is not None:
            return [w_prev, w_from_root_record(row)], rows, stop
        w_prev = w_from_root_record(row)
    return [w_prev], rows, None


def seed_snapshot_at_gamma(
    seed_payload: dict[str, Any],
    r: float,
    gamma: float,
    *,
    dps: int,
    max_inflate_iters: int,
    escalate: bool,
    newton_tol: float,
    delta_lower_threshold: float,
    contraction_max: float,
    enclose_action: bool = False,
) -> tuple[Optional[dict[str, Any]], Optional[StopReason]]:
    w_guess = interpolate_w_seed_from_payload(seed_payload, gamma)
    row, stop = certify_tracked_root_at_gamma(
        r,
        gamma,
        w_guess,
        index=0,
        dps=dps,
        max_inflate_iters=max_inflate_iters,
        escalate=escalate,
        newton_tol=newton_tol,
        delta_lower_threshold=delta_lower_threshold,
        contraction_max=contraction_max,
        enclose_action=enclose_action,
    )
    if row is None:
        return None, stop
    seed = {
        "gamma": float(gamma),
        "w_star_real": row["w_star_real"],
        "w_star_imag": row["w_star_imag"],
        "Phi_eff_real": row["Phi_eff_real"],
        "Phi_eff_imag": row["Phi_eff_imag"],
        "delta_lower": row["delta_lower"],
        "krawczyk_certified": row["krawczyk_certified"],
        "newton_ok": row["newton_ok"],
        "residual": row["residual"],
    }
    return seed, stop


def track_competitor_branch_from_anchor(
    sweep_payload: dict[str, Any],
    seed_payload: dict[str, Any],
    *,
    anchor_gamma: float,
    gamma_step: float,
    gamma_hi: float,
    gamma_lo: float,
    dps: int,
    max_inflate_iters: int,
    escalate: bool,
    newton_tol: float,
    delta_lower_threshold: float,
    contraction_max: float,
    enclose_action: bool = False,
) -> dict[str, Any]:
    """Continue one competitor sheet from a sweep anchor in both gamma directions."""
    r = float(sweep_payload.get("r", seed_payload.get("r", 1.0)))
    anchor_row = nearest_sweep_point(sweep_payload, anchor_gamma)
    anchor_gamma_eff = float(anchor_row["gamma"])
    anchor_comp = pick_tracked_competitor_at_row(anchor_row)
    w_anchor = w_from_root_record(anchor_comp)

    toward_hi, toward_lo = make_gamma_arms(
        anchor_gamma_eff, gamma_step, gamma_hi, gamma_lo
    )

    _, comp_hi, stop_hi = continue_tracked_root_arm(
        r,
        toward_hi,
        w_anchor,
        arm_name="toward_less_negative",
        dps=dps,
        max_inflate_iters=max_inflate_iters,
        escalate=escalate,
        newton_tol=newton_tol,
        delta_lower_threshold=delta_lower_threshold,
        contraction_max=contraction_max,
        enclose_action=enclose_action,
    )
    _, comp_lo, stop_lo = continue_tracked_root_arm(
        r,
        toward_lo,
        w_anchor,
        arm_name="toward_more_negative",
        dps=dps,
        max_inflate_iters=max_inflate_iters,
        escalate=escalate,
        newton_tol=newton_tol,
        delta_lower_threshold=delta_lower_threshold,
        contraction_max=contraction_max,
        enclose_action=enclose_action,
    )

    comp_by_gamma: dict[float, dict[str, Any]] = {}
    for row in comp_hi + comp_lo:
        comp_by_gamma[float(row["gamma"])] = row

    anchor_comp_row, _ = certify_tracked_root_at_gamma(
        r,
        anchor_gamma_eff,
        w_anchor,
        index=0,
        dps=dps,
        max_inflate_iters=max_inflate_iters,
        escalate=escalate,
        newton_tol=newton_tol,
        delta_lower_threshold=delta_lower_threshold,
        contraction_max=contraction_max,
        enclose_action=enclose_action,
    )
    if anchor_comp_row is not None:
        comp_by_gamma[anchor_gamma_eff] = anchor_comp_row

    gammas_all = sorted(comp_by_gamma.keys(), reverse=True)
    points: list[dict[str, Any]] = []
    for gamma in gammas_all:
        comp = comp_by_gamma[gamma]
        seed, seed_stop = seed_snapshot_at_gamma(
            seed_payload,
            r,
            gamma,
            dps=dps,
            max_inflate_iters=max_inflate_iters,
            escalate=escalate,
            newton_tol=newton_tol,
            delta_lower_threshold=delta_lower_threshold,
            contraction_max=contraction_max,
            enclose_action=enclose_action,
        )
        if seed is None:
            break
        gap_re = float(seed["Phi_eff_real"]) - float(comp["Phi_eff_real"])
        points.append(
            {
                "gamma": float(gamma),
                "gap_re_seed_minus_competitor": gap_re,
                "seed": seed,
                "competitor": comp,
                "anchor_gamma": anchor_gamma_eff,
            }
        )
        if seed_stop is not None:
            break

    label = (
        f"anchor={anchor_gamma_eff:.3f} "
        f"Re={float(anchor_comp['Phi_eff_real']):.3f}"
    )
    return {
        "branch_label": label,
        "r": r,
        "anchor_gamma": anchor_gamma_eff,
        "gamma_step": float(gamma_step),
        "gamma_lo": float(gamma_lo),
        "gamma_hi": float(gamma_hi),
        "anchor_competitor": anchor_comp,
        "arm_stops": [
            {"arm": "toward_less_negative", "stop": _stop_dict(stop_hi)},
            {"arm": "toward_more_negative", "stop": _stop_dict(stop_lo)},
        ],
        "points": points,
        "certificate_type": CERTIFICATE_TYPE_TRACKED_COMPETITOR,
        "sweep_source": sweep_payload.get("metadata"),
    }


def _stop_dict(stop: Optional[StopReason]) -> Optional[dict[str, Any]]:
    if stop is None:
        return None
    return {
        "code": stop.code,
        "message": stop.message,
        "index": stop.index,
        "gamma": stop.gamma,
    }


def track_competitor_branches(
    sweep_payload: dict[str, Any],
    seed_payload: dict[str, Any],
    anchor_gammas: list[float],
    *,
    gamma_step: float,
    gamma_hi: float,
    gamma_lo: float,
    dps: int,
    max_inflate_iters: int,
    escalate: bool,
    newton_tol: float,
    delta_lower_threshold: float,
    contraction_max: float,
    enclose_action: bool = False,
    gamma_convention: Optional[str] = None,
) -> dict[str, Any]:
    branches = [
        track_competitor_branch_from_anchor(
            sweep_payload,
            seed_payload,
            anchor_gamma=g,
            gamma_step=gamma_step,
            gamma_hi=gamma_hi,
            gamma_lo=gamma_lo,
            dps=dps,
            max_inflate_iters=max_inflate_iters,
            escalate=escalate,
            newton_tol=newton_tol,
            delta_lower_threshold=delta_lower_threshold,
            contraction_max=contraction_max,
            enclose_action=enclose_action,
        )
        for g in anchor_gammas
    ]
    return {
        "tracked_competitor_branches": branches,
        "certificate_type": CERTIFICATE_TYPE_TRACKED_COMPETITOR,
        "gamma_convention": gamma_convention,
    }


def plot_tracked_competitor_branches(
    payload: dict[str, Any], plot_path: Path, *, allow_overwrite: bool = False
) -> Path:
    """Re(Phi_seed) - Re(Phi_competitor) along each tracked competitor sheet."""
    plot_path = resolve_plot_output_path(plot_path, allow_overwrite=allow_overwrite)
    import matplotlib.pyplot as plt

    branches = payload.get("tracked_competitor_branches") or []
    n = len(branches)
    if n == 0:
        raise ValueError("no tracked_competitor_branches in payload")

    fig, axes = plt.subplots(n, 1, figsize=(9, 3.2 * n), squeeze=False)
    for ax, br in zip(axes.ravel(), branches):
        pts = sorted(br["points"], key=lambda p: float(p["gamma"]), reverse=True)
        gamma = np.array([float(p["gamma"]) for p in pts])
        gap = np.array([float(p["gap_re_seed_minus_competitor"]) for p in pts])
        colors = np.where(gap >= 0, "C0", "C3")
        ax.scatter(gamma, gap, c=colors, s=22, zorder=3)
        ax.plot(gamma, gap, "-", lw=0.9, alpha=0.55, color="k")
        ax.axhline(0.0, color="gray", lw=0.8, ls="--")
        ax.axvline(float(br["anchor_gamma"]), color="C2", lw=0.8, ls=":", label="anchor")
        ax.set_xlabel(r"$\gamma$")
        ax.set_ylabel(r"$\mathrm{Re}\,\Phi_{\mathrm{seed}} - \mathrm{Re}\,\Phi_{\mathrm{comp}}$")
        ax.set_title(
            f"Anchor γ={float(br['anchor_gamma']):.4g}  "
            f"({len(pts)} certified gap points)",
            fontsize=9,
        )
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.25)
    fig.suptitle(
        "Track step: Re Φ_seed − Re Φ_comp along one certified competitor sheet per anchor\n"
        "(green = seed wins Re Φ; red = competitor wins; dotted vertical = anchor γ)",
        fontsize=10,
    )
    fig.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    return plot_path


# ---------------------------------------------------------------------------
# Plot styling (shared colors + segmented lines)
# ---------------------------------------------------------------------------

SEED_PLOT_COLOR = "#1f77b4"
WIN_COLOR = "#2ca02c"
LOSS_COLOR = "#d62728"
NONE_COLOR = "#7f7f7f"

EQ_GAP_MIN = (
    r"$\Delta_{\min}(\gamma)=\min_{j\in\mathcal{C}_\gamma\setminus\{s\}}"
    r"(\mathrm{Re}\,\Phi_{\mathrm{eff}}(w_s)-\mathrm{Re}\,\Phi_{\mathrm{eff}}(w_j))$"
)
EQ_GAP_POS = (
    r"$\Delta_{+}(\gamma)=\min_{j:\,\Delta_j>0}\Delta_j$"
    r", $\Delta_j=\mathrm{Re}\,\Phi_{\mathrm{eff}}(w_s)-\mathrm{Re}\,\Phi_{\mathrm{eff}}(w_j)$"
)
EQ_PHI_SEED = r"$\mathrm{Re}\,\Phi_{\mathrm{eff}}(w^\star(\gamma))$ (seed branch)"
EQ_PHI_ALL_SHEETS = (
    r"$\mathrm{Re}\,\Phi_{\mathrm{eff}}$ on each tracked sheet "
    r"(blue dashed = seed)"
)
EQ_PHI_BEST = (
    r"$\max_{j\in\mathcal{C}_\gamma\setminus\{s\}}\mathrm{Re}\,\Phi_{\mathrm{eff}}(w_j)$"
    r" (per-$\gamma$ diagnostic, not one sheet)"
)
EQ_DELTA_BRANCH = (
    r"$\Delta\mathrm{Re}(\gamma)=\mathrm{Re}\,\Phi_{\mathrm{eff}}(w_s)"
    r"-\mathrm{Re}\,\Phi_{\mathrm{eff}}(w_{\mathrm{branch}})$"
)
EQ_IM_COMP_SHEETS = (
    r"$\mathrm{Im}\,\Phi_{\mathrm{eff}}$ on competitor sheets (unwrapped per sheet)"
)
TITLE_CROSSING_BRACKETS = (
    r"Certified $\Delta\mathrm{Re}=0$ brackets (Krawczyk bisection on resolve mesh)"
)


def _competitor_fingerprint(comp: Optional[dict[str, Any]]) -> Optional[tuple[float, ...]]:
    if not comp:
        return None
    return tuple(
        round(float(x), 5)
        for x in list(comp.get("w_real", [])) + list(comp.get("w_imag", []))
    )


def _best_competitor_record(pt: dict[str, Any]) -> Optional[dict[str, Any]]:
    sweep = pt.get("competitor_sweep") or {}
    comps = sweep.get("competitors") or []
    if comps:
        return max(comps, key=lambda c: float(c["Phi_eff_real"]))
    return pt.get("closest_competitor")


def _competitor_color_palette() -> list[str]:
    import matplotlib.pyplot as plt

    return [plt.cm.tab10(i % 10) for i in range(10)]


def _color_for_competitor_fingerprints(
    fingerprints: list[Optional[tuple[float, ...]]],
) -> dict[Optional[tuple[float, ...]], Any]:
    palette = _competitor_color_palette()
    mapping: dict[Optional[tuple[float, ...]], Any] = {None: NONE_COLOR}
    idx = 0
    for fp in fingerprints:
        if fp is None or fp in mapping:
            continue
        mapping[fp] = palette[idx % len(palette)]
        idx += 1
    return mapping


def _branch_color_map(branches: list[dict[str, Any]]) -> dict[int, Any]:
    import matplotlib.pyplot as plt

    palette = _competitor_color_palette()
    colors: dict[int, Any] = {}
    comp_i = 0
    for br in branches:
        bid = int(br["branch_id"])
        if br.get("is_seed_sheet"):
            colors[bid] = SEED_PLOT_COLOR
        else:
            colors[bid] = palette[comp_i % len(palette)]
            comp_i += 1
    return colors


def _plot_branch_curves_on_ax(
    ax: Any,
    branches: list[dict[str, Any]],
    br_colors: dict[int, Any],
    *,
    y_key: str,
    skip_seed: bool = False,
    skip_seed_duplicate: bool = True,
    legend: bool = True,
    label_max: int = 28,
) -> None:
    """One continuous curve per tracked branch_id (resolve step), same color everywhere."""
    for br in branches:
        if skip_seed and br.get("is_seed_sheet"):
            continue
        bid = int(br["branch_id"])
        color = br_colors[bid]
        pts = sorted(br["points"], key=lambda p: float(p["gamma"]), reverse=True)
        g, y = [], []
        for p in pts:
            if skip_seed_duplicate and p.get("status") == STATUS_SEED_DUPLICATE:
                continue
            g.append(float(p["gamma"]))
            if y_key == "DeltaRe_vs_seed":
                y.append(float(p["DeltaRe_vs_seed"]))
            elif y_key == "Phi_eff_real":
                y.append(float(p["Phi_eff_real"]))
            else:
                y.append(float(p.get("Phi_eff_imag_unwrapped", p["Phi_eff_imag"])))
        if not g:
            continue
        ls = "--" if br.get("is_seed_sheet") else "-"
        lw = 1.4 if br.get("is_seed_sheet") else 1.2
        ax.plot(
            g,
            y,
            ls,
            color=color,
            marker="o",
            ms=2,
            lw=lw,
            label=br["label"][:label_max] if legend else None,
        )
    if legend:
        ax.legend(fontsize=5, ncol=1)


def _segment_breaks(
    fps: list[Optional[tuple[float, ...]]],
    gaps: np.ndarray,
) -> np.ndarray:
    """True between point i-1 and i when identity or win/loss class changes."""
    n = len(fps)
    brk = np.zeros(n, dtype=bool)
    for i in range(1, n):
        fp_break = fps[i] != fps[i - 1]
        g0, g1 = gaps[i - 1], gaps[i]
        if not (np.isfinite(g0) and np.isfinite(g1)):
            sign_break = np.isfinite(g0) != np.isfinite(g1)
        else:
            sign_break = (g0 > 0) != (g1 > 0) or (g0 <= 0) != (g1 <= 0)
        brk[i] = bool(fp_break or sign_break)
    return brk


def _plot_xy_segments(
    ax: Any,
    x: np.ndarray,
    y: np.ndarray,
    breaks: np.ndarray,
    *,
    color: Any,
    lw: float = 1.0,
    alpha: float = 0.85,
    marker: Optional[str] = None,
    ms: float = 3.0,
) -> None:
    """Plot connected segments only where breaks[i] is False between i-1 and i."""
    if len(x) == 0:
        return
    start = 0
    for i in range(1, len(x) + 1):
        if i == len(x) or breaks[i]:
            if i - start >= 1:
                ax.plot(
                    x[start:i],
                    y[start:i],
                    "-",
                    color=color,
                    lw=lw,
                    alpha=alpha,
                    marker=marker,
                    ms=ms,
                )
            start = i


def _per_point_competitor_series(
    pts: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[Optional[tuple[float, ...]]]]:
    pts = sorted(pts, key=lambda p: float(p["gamma"]), reverse=True)
    gamma = np.array([float(p["gamma"]) for p in pts])
    re_seed = np.array([float(p["Phi_eff_real"]) for p in pts])
    min_gap = np.array(
        [
            float(p["min_certified_gap_re"])
            if p.get("min_certified_gap_re") is not None
            else np.nan
            for p in pts
        ]
    )
    fps = [_competitor_fingerprint(p.get("closest_competitor")) for p in pts]
    return gamma, re_seed, min_gap, fps


def _refined_window_gamma_mask(
    gamma: np.ndarray, payload: dict[str, Any], *, fallback_hi: float = -0.7
) -> tuple[np.ndarray, str]:
    """Boolean mask for sweep-only wall zoom; prefer refined_window from sweep JSON."""
    ref = payload.get("refined_window") or {}
    lo, hi = ref.get("gamma_lo"), ref.get("gamma_hi")
    if lo is not None and hi is not None:
        g_lo, g_hi = float(lo), float(hi)
        if g_lo > g_hi:
            g_lo, g_hi = g_hi, g_lo
        mask = (gamma >= g_lo) & (gamma <= g_hi)
        title = rf"Dense-window zoom: $\gamma\in[{g_lo:.3g},\,{g_hi:.3g}]$ (per-$\gamma$ $\Delta_{{\min}}$)"
        return mask, title
    mask = gamma <= fallback_hi
    return mask, rf"Wall zoom: $\gamma\le {fallback_hi:.2g}$ (per-$\gamma$ $\Delta_{{\min}}$)"


def _plot_crossing_brackets_on_ax(
    ax: Any,
    resolved: dict[str, Any],
    br_colors: dict[int, Any],
) -> int:
    """Shaded γ intervals where certified ΔRe brackets straddle zero."""
    n = 0
    for iv in resolved.get("crossing_certificates") or []:
        if iv.get("suppressed_duplicate_same_sheet"):
            continue
        g0, g1 = iv["gamma_interval"]
        bid = int(iv["crossing_branch_id"])
        color = br_colors.get(bid, LOSS_COLOR)
        ax.axvspan(
            g1,
            g0,
            alpha=0.25,
            color=color,
            label=rf"branch {bid} crossing",
        )
        n += 1
    ax.axhline(0.0, color="gray", ls="--", lw=0.6, alpha=0.5)
    ax.set_xlabel(r"$\gamma$")
    ax.set_ylabel("(interval marker)")
    ax.set_title(TITLE_CROSSING_BRACKETS, fontsize=9)
    if n:
        ax.legend(fontsize=6)
    ax.grid(True, alpha=0.25)
    return n


def _plot_sweep_gap_scatter_on_ax(
    ax: Any,
    gamma: np.ndarray,
    min_gap: np.ndarray,
    *,
    wall_only: bool = False,
    payload: Optional[dict[str, Any]] = None,
) -> None:
    g, gaps = gamma, min_gap
    if wall_only and payload is not None:
        mask, wall_title = _refined_window_gamma_mask(gamma, payload)
        g, gaps = gamma[mask], min_gap[mask]
        panel_title = wall_title
    else:
        panel_title = r"Sweep: min certified gap (full mesh)"
    for gv, gap in zip(g, gaps):
        if not np.isfinite(gap):
            ax.scatter(gv, gap, c=NONE_COLOR, s=20, zorder=3)
            continue
        ax.scatter(gv, gap, c=WIN_COLOR if gap > 0 else LOSS_COLOR, s=22, zorder=4)
    ax.axhline(0.0, color=LOSS_COLOR, ls="--", lw=1, label=r"$\Delta_{\min}=0$")
    ax.set_xlabel(r"$\gamma$")
    ax.set_ylabel(r"$\Delta_{\min}(\gamma)$")
    ax.set_title(panel_title + f"\n{EQ_GAP_MIN}", fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)


def _plot_sweep_counts_on_ax(
    ax: Any,
    gamma: np.ndarray,
    n_comp: np.ndarray,
    n_cert: np.ndarray,
) -> None:
    ax.scatter(gamma, n_comp, c=NONE_COLOR, s=18, label="certified competitors")
    ax.plot(gamma, n_cert, "s--", ms=3, lw=0.8, alpha=0.6, color="#9467bd", label="all certified roots")
    ax.set_xlabel(r"$\gamma$")
    ax.set_ylabel("count")
    ax.set_title(r"Sweep: $|\mathcal{C}_\gamma|$ at each mesh $\gamma$")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)


def _plot_sweep_summary_bar_on_ax(ax: Any, pts: list[dict[str, Any]]) -> None:
    dom = sum(1 for p in pts if p.get("seed_dominates_all"))
    threat = sum(
        1
        for p in pts
        if p.get("min_certified_gap_re") is not None and float(p["min_certified_gap_re"]) <= 0
    )
    no_comp = sum(1 for p in pts if int(p.get("num_certified_competitors", 0) or 0) == 0)
    ax.bar(
        ["seed wins", "competitor wins", "no competitor"],
        [dom, threat, no_comp],
        color=[WIN_COLOR, LOSS_COLOR, NONE_COLOR],
    )
    ax.set_ylabel("mesh points")
    ax.set_title(r"Sweep: mesh points by outcome", fontsize=9)
    for label in ax.get_xticklabels():
        label.set_rotation(12)
        label.set_ha("right")


def plot_competitor_analysis(
    payload: dict[str, Any],
    plot_path: Path,
    *,
    resolved_payload: Optional[dict[str, Any]] = None,
    branches_only: bool = False,
    allow_overwrite: bool = False,
) -> Path:
    """Unified sweep (+ optional resolve) dashboard. One PNG replaces old sweep + resolved pair."""
    if branches_only and resolved_payload is None:
        raise ValueError("branches_only requires resolved_payload from the resolve step")
    plot_path = resolve_plot_output_path(plot_path, allow_overwrite=allow_overwrite)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pts = sorted(payload.get("points") or [], key=lambda p: float(p["gamma"]), reverse=True)
    if not pts:
        raise ValueError("no points to plot")
    gamma, re_seed, min_gap, _fps = _per_point_competitor_series(pts)
    resolved = resolved_payload
    use_branches = branches_only or resolved is not None
    branches = (resolved or {}).get("branches") or []
    br_colors = _branch_color_map(branches) if use_branches else {}
    n_comp_sheets = sum(1 for b in branches if not b.get("is_seed_sheet"))
    n_cross = len((resolved or {}).get("crossing_certificates") or [])

    n_comp = np.array([int(p.get("num_certified_competitors", 0) or 0) for p in pts])
    n_cert = np.array([int(p.get("num_certified_roots", 0) or 0) for p in pts])

    if use_branches:
        fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), constrained_layout=True)
        ax = axes[0, 0]
        _plot_branch_curves_on_ax(
            ax, branches, br_colors, y_key="DeltaRe_vs_seed", skip_seed=True, legend=True
        )
        ax.axhline(0.0, color="gray", ls="--", lw=0.8)
        ax.set_xlabel(r"$\gamma$")
        ax.set_ylabel(r"$\Delta\mathrm{Re}(\gamma)$")
        ax.set_title(f"Resolve: ΔRe per sheet\n{EQ_DELTA_BRANCH}", fontsize=9)
        ax.grid(True, alpha=0.25)

        ax = axes[0, 1]
        _plot_branch_curves_on_ax(
            ax, branches, br_colors, y_key="Phi_eff_real", skip_seed=False, legend=True
        )
        ax.set_xlabel(r"$\gamma$")
        ax.set_ylabel(r"$\mathrm{Re}\,\Phi_{\mathrm{eff}}$")
        ax.set_title(EQ_PHI_ALL_SHEETS, fontsize=9)
        ax.grid(True, alpha=0.25)

        ax = axes[0, 2]
        _plot_crossing_brackets_on_ax(ax, resolved, br_colors)

        ax = axes[1, 0]
        _plot_sweep_gap_scatter_on_ax(ax, gamma, min_gap, wall_only=False)

        ax = axes[1, 1]
        _plot_sweep_counts_on_ax(ax, gamma, n_comp, n_cert)

        ax = axes[1, 2]
        _plot_sweep_summary_bar_on_ax(ax, pts)

        r = payload.get("r", 1.0)
        ref = payload.get("refined_window")
        title = f"Combined dashboard  r={r}  sweep N={len(pts)}"
        if ref:
            title += f"  dense [{ref.get('gamma_lo')}, {ref.get('gamma_hi')}]"
        subtitle = (
            f"Top row: resolve ({n_comp_sheets} competitor sheets, {n_cross} crossing(s)). "
            f"Bottom row: sweep mesh discovery."
        )
    else:
        fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), constrained_layout=True)
        ax = axes[0, 0]
        _plot_sweep_gap_scatter_on_ax(ax, gamma, min_gap, wall_only=False)

        ax = axes[0, 1]
        _plot_sweep_gap_scatter_on_ax(ax, gamma, min_gap, wall_only=True, payload=payload)

        ax = axes[1, 0]
        _plot_sweep_counts_on_ax(ax, gamma, n_comp, n_cert)

        ax = axes[1, 1]
        _plot_sweep_summary_bar_on_ax(ax, pts)

        r = payload.get("r", 1.0)
        ref = payload.get("refined_window")
        title = f"Sweep dashboard  r={r}  N={len(pts)} mesh points"
        if ref:
            title += f"  dense [{ref.get('gamma_lo')}, {ref.get('gamma_hi')}]"
        subtitle = (
            r"Run resolve with --plot pointing here to upgrade this file "
            r"to the combined 2×3 dashboard (no separate resolved_*.png needed)."
        )

    fig.suptitle(title + "\n" + subtitle, fontsize=10)
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)
    return plot_path


def _plot_resolve_only_dashboard(
    resolved: dict[str, Any],
    plot_path: Path,
    *,
    allow_overwrite: bool = False,
) -> Path:
    """Compact resolve-only figure when no sweep JSON is available."""
    plot_path = resolve_plot_output_path(plot_path, allow_overwrite=allow_overwrite)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    branches = resolved.get("branches") or []
    br_colors = _branch_color_map(branches)
    n_comp = sum(1 for b in branches if not b["is_seed_sheet"])
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)

    ax = axes[0]
    _plot_branch_curves_on_ax(ax, branches, br_colors, y_key="DeltaRe_vs_seed", skip_seed=True)
    ax.axhline(0.0, color="gray", ls="--", lw=0.8)
    ax.set_xlabel(r"$\gamma$")
    ax.set_ylabel(r"$\Delta\mathrm{Re}(\gamma)$")
    ax.set_title(f"ΔRe per sheet\n{EQ_DELTA_BRANCH}", fontsize=9)
    ax.grid(True, alpha=0.25)

    ax = axes[1]
    _plot_branch_curves_on_ax(
        ax, branches, br_colors, y_key="Phi_eff_real", skip_seed=False, label_max=20
    )
    ax.set_xlabel(r"$\gamma$")
    ax.set_ylabel(r"$\mathrm{Re}\,\Phi_{\mathrm{eff}}$")
    ax.set_title(EQ_PHI_ALL_SHEETS, fontsize=9)
    ax.grid(True, alpha=0.25)

    ax = axes[2]
    n_cross = _plot_crossing_brackets_on_ax(ax, resolved, br_colors)
    fig.suptitle(
        f"Resolve-only dashboard  r={resolved.get('r')}  "
        f"{n_comp} competitor sheet(s)  {n_cross} crossing(s)",
        fontsize=10,
    )
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    return plot_path


def plot_branch(
    payload: dict[str, Any], plot_path: Path, *, allow_overwrite: bool = False
) -> Path:
    plot_path = resolve_plot_output_path(plot_path, allow_overwrite=allow_overwrite)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pts = payload["points"]
    if not pts:
        raise ValueError("no points to plot")
    gamma = np.array([p["gamma"] for p in pts])
    re_phi = np.array([p["Phi_eff_real"] for p in pts])
    im_phi = np.array([p["Phi_eff_imag"] for p in pts])
    im_lifted = np.array(
        [p.get("Phi_eff_im_lifted", p.get("Phi_eff_im_unwrapped", p["Phi_eff_imag"])) for p in pts]
    )
    im_pu = np.array([p.get("Phi_eff_im_principal_unwrap", p["Phi_eff_imag"]) for p in pts])
    delta = np.array([p["delta_lower"] for p in pts])
    kappa = np.array([p["krawczyk_contraction_bound"] for p in pts])
    has_gaps = any(p.get("min_certified_gap_re") is not None for p in pts)
    nrows = 3 if has_gaps else 2

    fig, axes = plt.subplots(nrows, 2, figsize=(10, 4 * nrows), constrained_layout=True)
    if nrows == 2:
        axes = np.asarray(axes)
    ax = axes[0, 0]
    ax.plot(gamma, re_phi, "o-", ms=3, lw=1, label=r"$\mathrm{Re}\,\Phi_{\mathrm{eff}}$")
    if "Phi_eff_re_leading" in pts[0]:
        re_lead = np.array([p["Phi_eff_re_leading"] for p in pts])
        ax.plot(gamma, re_lead, "--", lw=1, alpha=0.7, label="leading Eq. (5.9)")
    ax.set_xlabel(r"$\gamma$")
    ax.set_ylabel(r"$\mathrm{Re}\,\Phi_{\mathrm{eff}}$")
    ax.set_title(r"Re $\Phi_{\mathrm{eff}}$ on certified seed branch (+ leading Eq. 5.9)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(gamma, im_phi, ".", ms=3, alpha=0.5, label="principal Im")
    ax.plot(gamma, im_pu, "--", ms=3, lw=1, alpha=0.6, label="principal unwrap (diag.)")
    ax.plot(gamma, im_lifted, "o-", ms=3, lw=1, label="ratio-lift Im")
    ax.set_xlabel(r"$\gamma$")
    ax.set_ylabel(r"$\mathrm{Im}\,\Phi_{\mathrm{eff}}$")
    ax.set_title(
        r"Im $\Phi_{\mathrm{eff}}$: certified $\log\Delta_\star$ ratio-lift vs principal branch"
    )
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.semilogy(gamma, np.maximum(delta, 1e-300), "o-", ms=3, lw=1)
    ax.set_xlabel(r"$\gamma$")
    ax.set_ylabel(r"$\delta_{\mathrm{lower}} = \inf|\Delta_\star|$")
    ax.set_title(r"Krawczyk: $\inf|\Delta_\star|$ lower bound on seed box")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.semilogy(gamma, np.maximum(kappa, 1e-300), "o-", ms=3, lw=1)
    ax.axhline(1.0, color="r", ls="--", lw=0.8, label="|I-YJ| = 1")
    ax.set_xlabel(r"$\gamma$")
    ax.set_ylabel(r"$|I - YJ|$")
    ax.set_title(r"Krawczyk contraction bound $|I-YJ|$ on seed box")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    if has_gaps:
        min_gaps = np.array(
            [
                p["min_certified_gap_re"] if p.get("min_certified_gap_re") is not None else np.nan
                for p in pts
            ]
        )
        ax = axes[2, 0]
        ax.plot(gamma, min_gaps, "o-", ms=3, lw=1, color="C2")
        ax.set_xlabel(r"$\gamma$")
        ax.set_ylabel(r"$\min_j \Delta_{\mathrm{gap}}^{(j)}$")
        ax.set_title(r"Min certified $\mathrm{Re}\,\Phi_{\rm seed}-\mathrm{Re}\,\Phi_j$")
        ax.grid(True, alpha=0.3)
        ax2 = axes[2, 1]
        n_cert = np.array([p.get("num_certified_roots", 0) for p in pts])
        n_num = np.array([p.get("num_numerical_roots", 0) for p in pts])
        ax2.plot(gamma, n_num, "o-", ms=3, lw=1, label="numerical")
        ax2.plot(gamma, n_cert, "s-", ms=3, lw=1, label="certified")
        ax2.set_xlabel(r"$\gamma$")
        ax2.set_ylabel("root count")
        ax2.set_title("Discovery yield")
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)

    stop = payload.get("stop")
    g0, g1 = float(payload["gamma_mesh"][0]), float(payload["gamma_mesh"][-1])
    title = f"Continue step: seed branch  r={payload['r']}  γ∈[{g0:.3g}, {g1:.3g}]  N={len(pts)}"
    if stop:
        title += f"  STOP @ γ={stop['gamma']:.4g} ({stop['code']})"
    fig.suptitle(title, fontsize=11)
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    return plot_path



# =============================================================================
# Branch-resolved competitor tracking (formerly competitor_branch_tracking)
# =============================================================================

STATUS_SEED_CERTIFIED_LOCAL = "seed_certified_local"
STATUS_COMPETITOR_CERTIFIED_LOCAL = "competitor_certified_local"
STATUS_COMPETITOR_PL_ACTIVE = "competitor_pl_active_diagnostic"
STATUS_SEED_DUPLICATE = "seed_duplicate"

SEED_DUPLICATE_W_FACTOR = 10.0

PL_ACTIVE_NOTE = (
    "competitor_pl_active_diagnostic: Re(Phi_competitor) > Re(Phi_seed) on this "
    "mesh point; does NOT follow from Krawczyk alone (thimble/PL dominance not certified)."
)

SEED_DUPLICATE_NOTE = (
    "seed_duplicate: |w_comp-w_seed| < 10(rho_seed+rho_comp) or Krawczyk boxes overlap; "
    "not counted as an independent competitor sheet."
)


# ---------------------------------------------------------------------------
# Box geometry and action intervals
# ---------------------------------------------------------------------------


def w_from_node(node: dict[str, Any]) -> np.ndarray:
    return np.array(node["w_real"]) + 1j * np.array(node["w_imag"])


def rho_from_node(node: dict[str, Any]) -> float:
    return float(node.get("box_radius") or 0.0)


def krawczyk_box_record(w: np.ndarray, rho: float) -> dict[str, Any]:
    return {
        "w_center_real": w.real.tolist(),
        "w_center_imag": w.imag.tolist(),
        "box_radius": float(rho),
    }


def component_boxes_disjoint(z1: complex, r1: float, z2: complex, r2: float) -> bool:
    """Per-component (Re, Im) hyperrectangle: disjoint if Re or Im intervals separate."""
    re_sep = abs(z1.real - z2.real) > r1 + r2
    im_sep = abs(z1.imag - z2.imag) > r1 + r2
    return bool(re_sep or im_sep)


def krawczyk_boxes_disjoint(
    w1: np.ndarray,
    rho1: float,
    w2: np.ndarray,
    rho2: float,
) -> bool:
    """True iff the certified hyperrectangles are separated in at least one component."""
    w1 = np.asarray(w1, dtype=complex)
    w2 = np.asarray(w2, dtype=complex)
    return any(component_boxes_disjoint(w1[i], rho1, w2[i], rho2) for i in range(4))


def action_interval_from_info(info: dict[str, Any]) -> dict[str, Any]:
    """Normalize action enclosure fields from certify_point / krawczyk_info."""
    re_iv = info.get("action_interval_real")
    im_iv = info.get("action_interval_imag")
    if re_iv is None:
        re_c = float(info.get("Phi_eff_real", info.get("action_center_real", 0.0)))
        re_iv = [re_c, re_c]
    if im_iv is None:
        im_c = float(info.get("Phi_eff_imag", info.get("action_center_imag", 0.0)))
        im_iv = [im_c, im_c]
    re_lo, re_hi = float(re_iv[0]), float(re_iv[1])
    im_lo, im_hi = float(im_iv[0]), float(im_iv[1])
    return {
        "RePhi_interval": [re_lo, re_hi],
        "ImPhi_interval": [im_lo, im_hi],
        "action_width_real": float(re_hi - re_lo),
        "action_width_imag": float(im_hi - im_lo),
        "log_branch_method": info.get("log_branch_method", "unknown"),
    }


def delta_re_interval_from_action(
    seed_action: dict[str, Any],
    comp_action: dict[str, Any],
) -> list[float]:
    """Conservative enclosure: Re seed_lo - Re comp_hi <= DeltaRe <= Re seed_hi - Re comp_lo."""
    sr = seed_action["RePhi_interval"]
    cr = comp_action["RePhi_interval"]
    return [float(sr[0] - cr[1]), float(sr[1] - cr[0])]


def interval_straddles_zero(interval: list[float]) -> bool:
    lo, hi = float(interval[0]), float(interval[1])
    return lo <= 0.0 <= hi


def intervals_certified_opposite_sign(a: list[float], b: list[float]) -> bool:
    """True if enclosures are disjoint and on opposite sides of zero."""
    a_lo, a_hi = float(a[0]), float(a[1])
    b_lo, b_hi = float(b[0]), float(b[1])
    return (a_hi < 0 < b_lo) or (b_hi < 0 < a_lo)


def classify_vs_seed(
    node: dict[str, Any],
    w_seed: np.ndarray,
    rho_seed: float,
) -> dict[str, Any]:
    w = w_from_node(node)
    rho = rho_from_node(node)
    w_dist = float(np.linalg.norm(w - w_seed, ord=np.inf))
    w_thresh = SEED_DUPLICATE_W_FACTOR * (rho_seed + rho)
    w_near = w_dist < w_thresh
    overlap = not krawczyk_boxes_disjoint(w_seed, rho_seed, w, rho)
    is_seed = bool(node.get("is_seed_sheet"))
    is_dup = (w_near or overlap) and not is_seed
    return {
        "w_distance_inf": w_dist,
        "w_duplicate_threshold": w_thresh,
        "w_near_seed_duplicate": w_near,
        "box_overlaps_seed": overlap,
        "box_disjoint_from_seed": not overlap,
        "is_seed_duplicate": is_dup,
    }


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class BranchPoint:
    branch_id: int
    gamma: float
    w_real: list[float]
    w_imag: list[float]
    Phi_eff_real: float
    Phi_eff_imag: float
    Phi_eff_imag_unwrapped: float
    box_radius: float
    delta_lower: float
    status: str
    distance_to_seed_w: float
    DeltaRe_vs_seed: float
    DeltaIm_vs_seed: float
    krawczyk_certified: bool = True
    is_seed_duplicate: bool = False
    box_disjoint_from_seed: bool = True
    RePhi_interval: Optional[list[float]] = None
    ImPhi_interval: Optional[list[float]] = None
    action_width_real: Optional[float] = None
    log_branch_method: Optional[str] = None
    log_Delta_lifted_real: Optional[float] = None
    log_Delta_lifted_imag: Optional[float] = None
    log_branch_step_certified: bool = True
    DeltaRe_interval_vs_seed: Optional[list[float]] = None


@dataclass
class CompetitorBranch:
    branch_id: int
    points: list[BranchPoint] = field(default_factory=list)
    is_seed_sheet: bool = False
    label: str = ""
    possible_same_sheet_as_branch_id: Optional[int] = None

    def gammas(self) -> np.ndarray:
        return np.array([p.gamma for p in self.points])

    def delta_re(self) -> np.ndarray:
        return np.array([p.DeltaRe_vs_seed for p in self.points])


# ---------------------------------------------------------------------------
# Gamma slices and tracking
# ---------------------------------------------------------------------------


def _w_norm_scale(w: np.ndarray) -> float:
    return max(float(np.max(np.abs(w))), 1e-3)


def _certified_node_from_record(
    rec: dict[str, Any],
    gamma: float,
    *,
    distance_to_seed_w: float,
    is_seed_sheet: bool,
    w_seed: np.ndarray,
    rho_seed: float,
) -> dict[str, Any]:
    info = rec.get("krawczyk_info") or {}
    rho = float(info.get("box_radius", rec.get("box_radius", 0.0)))
    node = {
        "gamma": float(gamma),
        "w_real": rec["w_real"],
        "w_imag": rec["w_imag"],
        "Phi_eff_real": float(rec["Phi_eff_real"]),
        "Phi_eff_imag": float(rec["Phi_eff_imag"]),
        "box_radius": rho,
        "delta_lower": float(rec.get("delta_lower") or info.get("delta_lower") or 0.0),
        "krawczyk_certified": bool(rec.get("krawczyk_certified", True)),
        "distance_to_seed_w": float(distance_to_seed_w),
        "is_seed_sheet": bool(is_seed_sheet),
        "krawczyk_info": info,
    }
    node.update(classify_vs_seed(node, w_seed, rho_seed))
    re_iv = info.get("action_interval_real")
    if re_iv is not None:
        node["RePhi_interval"] = [float(re_iv[0]), float(re_iv[1])]
        node["ImPhi_interval"] = [
            float(info["action_interval_imag"][0]),
            float(info["action_interval_imag"][1]),
        ]
    return node


def extract_gamma_nodes(
    sweep_payload: dict[str, Any],
    seed_payload: dict[str, Any],
    *,
    seed_ident_tol: float = 1e-3,
) -> list[dict[str, Any]]:
    """Per-gamma certified roots from sweep + continuation seed reference."""
    r = float(sweep_payload.get("r", seed_payload.get("r", 1.0)))
    rows = sorted(sweep_payload.get("points", []), key=lambda p: float(p["gamma"]), reverse=True)
    out: list[dict[str, Any]] = []
    for row in rows:
        gamma = float(row["gamma"])
        w_seed = np.array(row["w_star_real"]) + 1j * np.array(row["w_star_imag"])
        rho_seed = float(row.get("box_radius") or 1e-16)
        sweep = row.get("competitor_sweep") or {}
        roots = list(sweep.get("certified_roots") or [])
        if not roots:
            continue
        seed_re = float(row.get("Phi_eff_real", 0.0))
        seed_im = float(
            row.get(
                "Phi_eff_im_lifted",
                row.get("Phi_eff_im_unwrapped", row.get("Phi_eff_imag", 0.0)),
            )
        )
        seed_re_iv = [seed_re, seed_re]
        if row.get("action_interval_real"):
            seed_re_iv = list(row["action_interval_real"])
        nodes = []
        for rec in roots:
            w = np.array(rec["w_real"]) + 1j * np.array(rec["w_imag"])
            dist = float(np.linalg.norm(w - w_seed, ord=np.inf))
            is_seed = bool(rec.get("is_seed_branch")) or dist < seed_ident_tol
            nodes.append(
                _certified_node_from_record(
                    rec,
                    gamma,
                    distance_to_seed_w=dist,
                    is_seed_sheet=is_seed,
                    w_seed=w_seed,
                    rho_seed=rho_seed,
                )
            )
        out.append(
            {
                "gamma": gamma,
                "r": r,
                "w_seed": w_seed,
                "rho_seed": rho_seed,
                "seed_Phi_real": seed_re,
                "seed_Phi_imag": seed_im,
                "seed_RePhi_interval": seed_re_iv,
                "nodes": nodes,
            }
        )
    return out


def _krawczyk_info_from_branch_point(pt: BranchPoint) -> dict[str, Any]:
    info: dict[str, Any] = {"box_radius": float(pt.box_radius)}
    if pt.log_branch_method:
        info["log_branch_method"] = pt.log_branch_method
    return info


def _lifted_im_phi_for_branch_point(
    br: CompetitorBranch,
    node: dict[str, Any],
    r: float,
) -> tuple[float, complex, bool, str]:
    """Ratio-lift Im Phi on a tracked competitor sheet."""
    w_n = w_from_node(node)
    sys_curr = WSaddleSystem(r=float(r), gamma=float(node["gamma"]))
    info_curr = dict(node.get("krawczyk_info") or {})
    info_curr.setdefault("box_radius", float(node["box_radius"]))
    info_curr.setdefault("dps_used", int(info_curr.get("dps_used", 60)))

    if not br.points:
        log_lift = initial_log_delta_lift(w_n, beta=sys_curr.beta, p=sys_curr.p)
        phi_lifted = phi_eff_with_lifted_log(sys_curr, w_n, log_lift)
        return float(phi_lifted.imag), log_lift, True, "continuation_ratio_lift"

    prev = br.points[-1]
    w_prev = np.array(prev.w_real) + 1j * np.array(prev.w_imag)
    sys_prev = WSaddleSystem(r=float(r), gamma=float(prev.gamma))
    info_prev = _krawczyk_info_from_branch_point(prev)
    info_prev.setdefault("dps_used", int(info_curr.get("dps_used", 60)))

    if prev.log_Delta_lifted_real is not None and prev.log_Delta_lifted_imag is not None:
        log_prev = complex(prev.log_Delta_lifted_real, prev.log_Delta_lifted_imag)
    else:
        log_prev = initial_log_delta_lift(w_prev, beta=sys_prev.beta, p=sys_prev.p)

    step_ok, step_info = certify_log_delta_ratio_step(
        sys_prev, w_prev, info_prev, sys_curr, w_n, info_curr
    )
    log_lift = log_prev + complex(step_info["log_ratio_real"], step_info["log_ratio_imag"])
    phi_lifted = phi_eff_with_lifted_log(sys_curr, w_n, log_lift)
    method = str(step_info.get("log_branch_method", "continuation_ratio_lift"))
    return float(phi_lifted.imag), log_lift, bool(step_ok), method


def _unwrap_im_on_branch(im_values: list[float]) -> list[float]:
    if not im_values:
        return []
    out = [float(im_values[0])]
    for v in im_values[1:]:
        out.append(float(unwrap_phase_diff(out[-1], v)))
    return out


def _match_cost(
    br: CompetitorBranch,
    node: dict[str, Any],
    sl: dict[str, Any],
    *,
    branch_step_tol: float,
    use_tangent: bool,
    re_weight: float,
    im_weight: float,
) -> float:
    w_end = np.array(br.points[-1].w_real) + 1j * np.array(br.points[-1].w_imag)
    w_n = w_from_node(node)
    if use_tangent and len(br.points) >= 2:
        p1, p2 = br.points[-2], br.points[-1]
        w1 = np.array(p1.w_real) + 1j * np.array(p1.w_imag)
        w2 = w_end
        dg = float(p1.gamma - p2.gamma)
        if abs(dg) > 1e-14:
            t = (float(sl["gamma"]) - float(p2.gamma)) / dg
            w_pred = w2 + t * (w2 - w1)
        else:
            w_pred = w_end
    else:
        w_pred = w_end

    scale = _w_norm_scale(w_end) * max(branch_step_tol, 0.05)
    d_w = float(np.linalg.norm(w_n - w_pred, ord=np.inf)) / scale
    d_re = abs(float(node["Phi_eff_real"]) - br.points[-1].Phi_eff_real) / max(abs(br.points[-1].Phi_eff_real), 1.0)
    d_im = abs(
        float(unwrap_phase_diff(br.points[-1].Phi_eff_imag_unwrapped, float(node["Phi_eff_imag"])))
    ) / (2.0 * np.pi)
    return d_w + re_weight * d_re + im_weight * d_im


def track_branches_from_nodes(
    gamma_slices: list[dict[str, Any]],
    *,
    branch_step_tol: float = 0.35,
    seed_mean_dist_max: float = 0.05,
    use_tangent: bool = True,
    re_weight: float = 0.15,
    im_weight: float = 0.10,
) -> list[CompetitorBranch]:
    """Match certified roots along gamma (w + Re/Im continuity + optional tangent)."""
    if not gamma_slices:
        return []

    branches: list[CompetitorBranch] = []
    next_id = 0

    def start_branch(node: dict[str, Any]) -> CompetitorBranch:
        nonlocal next_id
        br = CompetitorBranch(branch_id=next_id, is_seed_sheet=False)
        next_id += 1
        return br

    def append_point(br: CompetitorBranch, sl: dict[str, Any], node: dict[str, Any]) -> BranchPoint:
        delta_re = float(sl["seed_Phi_real"]) - float(node["Phi_eff_real"])
        delta_im = float(unwrap_phase_diff(sl["seed_Phi_imag"], node["Phi_eff_imag"]))
        im_lifted, log_lift, step_ok, lift_method = _lifted_im_phi_for_branch_point(
            br, node, float(sl["r"])
        )

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
        comp_re_iv = node.get("RePhi_interval") or [node["Phi_eff_real"], node["Phi_eff_real"]]
        delta_re_iv = [
            float(seed_re_iv[0] - comp_re_iv[1]),
            float(seed_re_iv[1] - comp_re_iv[0]),
        ]

        return BranchPoint(
            branch_id=br.branch_id,
            gamma=float(node["gamma"]),
            w_real=list(node["w_real"]),
            w_imag=list(node["w_imag"]),
            Phi_eff_real=float(node["Phi_eff_real"]),
            Phi_eff_imag=float(node["Phi_eff_imag"]),
            Phi_eff_imag_unwrapped=im_lifted,
            box_radius=float(node["box_radius"]),
            delta_lower=float(node["delta_lower"]),
            status=status,
            distance_to_seed_w=float(node["distance_to_seed_w"]),
            DeltaRe_vs_seed=delta_re,
            DeltaIm_vs_seed=delta_im,
            krawczyk_certified=bool(node["krawczyk_certified"]),
            is_seed_duplicate=bool(node.get("is_seed_duplicate")),
            box_disjoint_from_seed=bool(node.get("box_disjoint_from_seed", True)),
            RePhi_interval=list(comp_re_iv),
            ImPhi_interval=node.get("ImPhi_interval"),
            action_width_real=(
                float(comp_re_iv[1] - comp_re_iv[0]) if comp_re_iv else None
            ),
            log_branch_method=lift_method,
            log_Delta_lifted_real=float(log_lift.real),
            log_Delta_lifted_imag=float(log_lift.imag),
            log_branch_step_certified=bool(step_ok),
            DeltaRe_interval_vs_seed=delta_re_iv,
        )

    def tracking_nodes(sl: dict[str, Any]) -> list[dict[str, Any]]:
        return [n for n in sl["nodes"] if not n.get("is_seed_duplicate")]

    first = gamma_slices[0]
    for node in tracking_nodes(first):
        br = start_branch(node)
        br.points.append(append_point(br, first, node))
        branches.append(br)

    for sl in gamma_slices[1:]:
        # Seed sheet assigned after tracking; extend all non-duplicate competitor branches.
        open_branches = [b for b in branches if b.points]
        endpoints = [
            (b, np.array(b.points[-1].w_real) + 1j * np.array(b.points[-1].w_imag))
            for b in open_branches
        ]
        nodes = tracking_nodes(sl)
        unused = list(range(len(nodes)))
        matches: list[tuple[float, int, int]] = []
        for bi, (br, _w_end) in enumerate(endpoints):
            scale = _w_norm_scale(_w_end) * max(branch_step_tol, 0.05)
            for ni in unused:
                w_n = w_from_node(nodes[ni])
                d = float(np.linalg.norm(w_n - _w_end, ord=np.inf))
                if d <= scale:
                    cost = _match_cost(
                        br,
                        nodes[ni],
                        sl,
                        branch_step_tol=branch_step_tol,
                        use_tangent=use_tangent,
                        re_weight=re_weight,
                        im_weight=im_weight,
                    )
                    matches.append((cost, bi, ni))
        matches.sort(key=lambda t: t[0])
        used_b: set[int] = set()
        used_n: set[int] = set()
        for _cost, bi, ni in matches:
            if bi in used_b or ni in used_n:
                continue
            used_b.add(bi)
            used_n.add(ni)
            br = open_branches[bi]
            br.points.append(append_point(br, sl, nodes[ni]))
        for ni in unused:
            if ni in used_n:
                continue
            node = nodes[ni]
            br = start_branch(node)
            br.points.append(append_point(br, sl, node))
            branches.append(br)

    _assign_seed_sheet_branch(branches, seed_mean_dist_max=seed_mean_dist_max)
    for br in branches:
        br.label = _branch_label(br)
    return branches


def _assign_seed_sheet_branch(
    branches: list[CompetitorBranch],
    *,
    seed_mean_dist_max: float = 0.05,
) -> None:
    """Longest low-distance branch = seed sheet."""
    candidates = [br for br in branches if br.points]
    if not candidates:
        return

    scored: list[tuple[CompetitorBranch, int, float]] = []
    for br in candidates:
        mean_d = float(np.mean([p.distance_to_seed_w for p in br.points]))
        scored.append((br, len(br.points), mean_d))

    low_dist = [t for t in scored if t[2] <= seed_mean_dist_max]
    pool = low_dist if low_dist else scored
    seed_br = max(pool, key=lambda t: (t[1], -t[2]))[0]

    for br in branches:
        br.is_seed_sheet = br.branch_id == seed_br.branch_id
        for p in br.points:
            if br.is_seed_sheet:
                p.status = STATUS_SEED_CERTIFIED_LOCAL
            elif p.is_seed_duplicate or not p.box_disjoint_from_seed:
                p.status = STATUS_SEED_DUPLICATE
            elif p.DeltaRe_vs_seed < 0:
                p.status = STATUS_COMPETITOR_PL_ACTIVE
            else:
                p.status = STATUS_COMPETITOR_CERTIFIED_LOCAL


def _branch_label(br: CompetitorBranch) -> str:
    if br.is_seed_sheet:
        return f"branch {br.branch_id} (seed sheet)"
    if br.possible_same_sheet_as_branch_id is not None:
        return (
            f"branch {br.branch_id} Re~{br.points[0].Phi_eff_real:.3f} "
            f"(~branch {br.possible_same_sheet_as_branch_id})"
        )
    if not br.points:
        return f"branch {br.branch_id}"
    re0 = br.points[0].Phi_eff_real
    return f"branch {br.branch_id} Re~{re0:.3f}"


def _gamma_range_overlap(br_a: CompetitorBranch, br_b: CompetitorBranch) -> float:
    ga = br_a.gammas()
    gb = br_b.gammas()
    if ga.size == 0 or gb.size == 0:
        return 0.0
    lo = max(float(ga.min()), float(gb.min()))
    hi = min(float(ga.max()), float(gb.max()))
    if lo > hi:
        return 0.0
    span = max(float(ga.max() - ga.min()), 1e-9)
    return float((hi - lo) / span)


def diagnose_same_sheet_pairs(
    branches: list[CompetitorBranch],
    *,
    w_match_frac: float = 0.35,
    re_tol: float = 0.35,
    min_overlap: int = 2,
    min_range_overlap: float = 0.55,
) -> list[dict[str, Any]]:
    """Flag branch pairs that likely split one sheet (Im± duplicate or matcher break)."""
    comps = [b for b in branches if not b.is_seed_sheet and len(b.points) >= 2]
    diagnoses: list[dict[str, Any]] = []
    for i, a in enumerate(comps):
        for b in comps[i + 1 :]:
            gam_a = {p.gamma: p for p in a.points}
            gam_b = {p.gamma: p for p in b.points}
            shared = sorted(set(gam_a) & set(gam_b), reverse=True)
            range_ov = _gamma_range_overlap(a, b)
            if len(shared) < min_overlap and range_ov < min_range_overlap:
                continue
            w_hits = 0
            re_hits = 0
            im_pair_hits = 0
            for g in shared:
                pa, pb = gam_a[g], gam_b[g]
                wa = np.array(pa.w_real) + 1j * np.array(pa.w_imag)
                wb = np.array(pb.w_real) + 1j * np.array(pb.w_imag)
                scale = _w_norm_scale(wa) * w_match_frac
                if float(np.linalg.norm(wa - wb, ord=np.inf)) <= max(scale, 0.05):
                    w_hits += 1
                if abs(pa.Phi_eff_real - pb.Phi_eff_real) <= re_tol:
                    re_hits += 1
                if abs(pa.Phi_eff_imag + pb.Phi_eff_imag) <= 0.5:
                    im_pair_hits += 1
            n = max(len(shared), 1)
            w_frac = w_hits / n
            re_frac = re_hits / n if shared else 0.0
            im_frac = im_pair_hits / n if shared else 0.0
            # Same sheet: (w+Re) continuity on shared mesh, OR Re-matched Im± pair with overlapping range.
            w_re_sheet = shared and w_frac >= 0.5 and re_frac >= 0.5
            re_im_sheet = (
                shared
                and re_frac >= 0.5
                and (im_frac >= 0.4 or range_ov >= min_range_overlap)
                and range_ov >= min_range_overlap
            )
            if w_re_sheet or re_im_sheet:
                diagnoses.append(
                    {
                        "branch_a": a.branch_id,
                        "branch_b": b.branch_id,
                        "shared_gamma_count": len(shared),
                        "gamma_range_overlap": range_ov,
                        "w_close_fraction": w_frac,
                        "re_close_fraction": re_frac,
                        "im_opposite_fraction": im_frac,
                        "likely_same_sheet": True,
                        "diagnosis": "w_re_continuous" if w_re_sheet else "re_im_pm_split",
                    }
                )
                a.possible_same_sheet_as_branch_id = b.branch_id
                b.possible_same_sheet_as_branch_id = a.branch_id
    return diagnoses


def merge_same_sheet_branches(
    branches: list[CompetitorBranch],
    diagnoses: list[dict[str, Any]],
) -> list[CompetitorBranch]:
    """Merge diagnosed duplicate sheets into the longer branch."""
    if not diagnoses:
        return branches
    by_id = {b.branch_id: b for b in branches}
    merged_ids: set[int] = set()

    for d in diagnoses:
        if not d.get("likely_same_sheet"):
            continue
        id_a, id_b = int(d["branch_a"]), int(d["branch_b"])
        if id_a in merged_ids or id_b in merged_ids:
            continue
        ba, bb = by_id.get(id_a), by_id.get(id_b)
        if not ba or not bb or ba.is_seed_sheet or bb.is_seed_sheet:
            continue
        keep, drop = (ba, bb) if len(ba.points) >= len(bb.points) else (bb, ba)
        gam_keep = {p.gamma for p in keep.points}
        for p in sorted(drop.points, key=lambda x: x.gamma, reverse=True):
            if p.gamma not in gam_keep:
                p.branch_id = keep.branch_id
                keep.points.append(p)
        keep.points.sort(key=lambda x: x.gamma, reverse=True)
        keep.possible_same_sheet_as_branch_id = None
        merged_ids.add(drop.branch_id)
        drop.points.clear()

    out = [b for b in branches if b.points]
    for br in out:
        br.label = _branch_label(br)
    return out


def find_delta_re_crossing_intervals(
    br: CompetitorBranch,
    *,
    require_disjoint_boxes: bool = True,
) -> list[dict[str, Any]]:
    """Mesh intervals where DeltaRe vs seed changes sign (center or interval enclosure)."""
    if br.is_seed_sheet:
        return []
    if len(br.points) < 2:
        return []
    intervals: list[dict[str, Any]] = []
    pts = sorted(br.points, key=lambda p: p.gamma, reverse=True)
    for i in range(len(pts) - 1):
        a, b = pts[i], pts[i + 1]
        if require_disjoint_boxes and (not a.box_disjoint_from_seed or not b.box_disjoint_from_seed):
            continue
        center_cross = a.DeltaRe_vs_seed * b.DeltaRe_vs_seed < 0
        iv_a = a.DeltaRe_interval_vs_seed
        iv_b = b.DeltaRe_interval_vs_seed
        interval_cross = False
        if iv_a and iv_b:
            interval_cross = interval_straddles_zero(iv_a) and interval_straddles_zero(iv_b)
            if not interval_cross:
                interval_cross = intervals_certified_opposite_sign(iv_a, iv_b)
        if not (center_cross or interval_cross):
            continue
        intervals.append(
            {
                "branch_id": br.branch_id,
                "gamma_hi": float(max(a.gamma, b.gamma)),
                "gamma_lo": float(min(a.gamma, b.gamma)),
                "DeltaRe_hi": float(a.DeltaRe_vs_seed),
                "DeltaRe_lo": float(b.DeltaRe_vs_seed),
                "DeltaRe_interval_hi": iv_a,
                "DeltaRe_interval_lo": iv_b,
                "DeltaIm_hi": float(a.DeltaIm_vs_seed),
                "DeltaIm_lo": float(b.DeltaIm_vs_seed),
                "sign_change_from_center": center_cross,
                "sign_change_from_interval": interval_cross,
            }
        )
    return intervals


def _certify_pair_at_gamma(
    r: float,
    gamma: float,
    w_seed_init: np.ndarray,
    w_comp_init: np.ndarray,
    *,
    dps: int,
    max_inflate_iters: int,
    escalate: bool,
    enclose_action: bool = True,
) -> dict[str, Any]:
    sys = WSaddleSystem(r=r, gamma=gamma)
    out: dict[str, Any] = {"gamma": float(gamma)}
    for label, w_init in (("seed", w_seed_init), ("competitor", w_comp_init)):
        w_star, ok, res = solve_w_from_init(sys, w_init, tol=1e-12)
        cert, info = certify_point(
            sys,
            w_star,
            dps=dps,
            max_inflate_iters=max_inflate_iters,
            escalate=escalate,
            enclose_action=enclose_action,
            log_branch_method="crossing_refinement",
        )
        phi = sys.Phi_eff(w_star)
        rho = float(info.get("box_radius", 0.0)) if cert else 0.0
        act = action_interval_from_info(info) if cert else {}
        out[label] = {
            "w_star_real": w_star.real.tolist(),
            "w_star_imag": w_star.imag.tolist(),
            "Phi_eff_real": float(phi.real),
            "Phi_eff_imag": float(phi.imag),
            "krawczyk_certified": bool(cert),
            "delta_lower": float(info.get("delta_lower", 0.0)) if cert else None,
            "box_radius": rho,
            "box": krawczyk_box_record(w_star, rho),
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
    w_s = np.array(out["seed"]["w_star_real"]) + 1j * np.array(out["seed"]["w_star_imag"])
    w_c = np.array(out["competitor"]["w_star_real"]) + 1j * np.array(out["competitor"]["w_star_imag"])
    out["box_disjoint"] = (
        out["both_certified"]
        and krawczyk_boxes_disjoint(w_s, out["seed"]["box_radius"], w_c, out["competitor"]["box_radius"])
    )
    return out


def refine_crossing_interval(
    seed_payload: dict[str, Any],
    interval: dict[str, Any],
    w_comp_end_hi: np.ndarray,
    w_comp_end_lo: np.ndarray,
    *,
    dps: int,
    max_inflate_iters: int,
    escalate: bool,
    max_depth: int = 16,
    target_width: float = 1e-4,
) -> dict[str, Any]:
    """Bisect gamma; Krawczyk + interval enclosures for DeltaRe at each node."""
    r = float(seed_payload.get("r", 1.0))
    g_hi = float(interval["gamma_hi"])
    g_lo = float(interval["gamma_lo"])

    w_seed_hi = interpolate_w_seed_from_payload(seed_payload, g_hi)
    w_seed_lo = interpolate_w_seed_from_payload(seed_payload, g_lo)

    hi_pt = _certify_pair_at_gamma(
        r, g_hi, w_seed_hi, w_comp_end_hi, dps=dps, max_inflate_iters=max_inflate_iters, escalate=escalate
    )
    lo_pt = _certify_pair_at_gamma(
        r, g_lo, w_seed_lo, w_comp_end_lo, dps=dps, max_inflate_iters=max_inflate_iters, escalate=escalate
    )

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
        mid_pt = _certify_pair_at_gamma(
            r,
            g_mid,
            w_s_mid,
            w_c_mid,
            dps=dps,
            max_inflate_iters=max_inflate_iters,
            escalate=escalate,
        )
        history.append(mid_pt)
        d_mid_iv = mid_pt["DeltaRe_interval"]
        if not mid_pt.get("both_certified"):
            break
        left_cross = interval_straddles_zero(delta_left_iv) and interval_straddles_zero(d_mid_iv)
        right_cross = interval_straddles_zero(d_mid_iv) and interval_straddles_zero(delta_right_iv)
        if left_cross or (
            not right_cross
            and left_pt["DeltaRe"] * mid_pt["DeltaRe"] <= 0
        ):
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


def _dedupe_crossings_by_same_sheet(
    branches: list[CompetitorBranch],
    crossing_certs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Drop duplicate crossing certs for branches flagged as same sheet."""
    by_branch = {b.branch_id: b for b in branches}
    seen_groups: set[frozenset[int]] = set()
    out: list[dict[str, Any]] = []
    for cert in sorted(crossing_certs, key=lambda c: c.get("gamma_width", 1.0)):
        bid = int(cert["crossing_branch_id"])
        br = by_branch.get(bid)
        partner = br.possible_same_sheet_as_branch_id if br else None
        if partner is not None:
            key = frozenset({bid, int(partner)})
            if key in seen_groups:
                cert["suppressed_duplicate_same_sheet"] = True
                continue
            seen_groups.add(key)
        out.append(cert)
    return out


def resolve_competitor_branches(
    sweep_payload: dict[str, Any],
    seed_payload: dict[str, Any],
    *,
    branch_step_tol: float = 0.35,
    seed_mean_dist_max: float = 0.05,
    use_tangent: bool = True,
    merge_same_sheet: bool = True,
    crossing_target_width: float = 1e-4,
    refine_crossing_near: Optional[float] = None,
    refine_window: float = 0.08,
    refine_all_crossings: bool = True,
    dps: int = 60,
    max_inflate_iters: int = 6,
    escalate: bool = True,
) -> dict[str, Any]:
    slices = extract_gamma_nodes(sweep_payload, seed_payload)
    branches = track_branches_from_nodes(
        slices,
        branch_step_tol=branch_step_tol,
        seed_mean_dist_max=seed_mean_dist_max,
        use_tangent=use_tangent,
    )

    same_sheet_diagnoses = diagnose_same_sheet_pairs(branches)
    if merge_same_sheet:
        branches = merge_same_sheet_branches(branches, same_sheet_diagnoses)

    all_intervals: list[dict[str, Any]] = []
    for br in branches:
        all_intervals.extend(find_delta_re_crossing_intervals(br))

    crossing_certs: list[dict[str, Any]] = []
    for iv in all_intervals:
        mid = 0.5 * (iv["gamma_hi"] + iv["gamma_lo"])
        if not refine_all_crossings:
            continue
        if refine_crossing_near is not None and abs(mid - refine_crossing_near) > refine_window:
            continue
        br = next(b for b in branches if b.branch_id == iv["branch_id"])
        if br.is_seed_sheet:
            continue
        pt_hi = min(br.points, key=lambda p: abs(p.gamma - iv["gamma_hi"]))
        pt_lo = min(br.points, key=lambda p: abs(p.gamma - iv["gamma_lo"]))
        if not (pt_hi.box_disjoint_from_seed and pt_lo.box_disjoint_from_seed):
            continue
        w_hi = np.array(pt_hi.w_real) + 1j * np.array(pt_hi.w_imag)
        w_lo = np.array(pt_lo.w_real) + 1j * np.array(pt_lo.w_imag)
        crossing_certs.append(
            refine_crossing_interval(
                seed_payload,
                iv,
                w_hi,
                w_lo,
                dps=dps,
                max_inflate_iters=max_inflate_iters,
                escalate=escalate,
                target_width=crossing_target_width,
            )
        )

    crossing_certs = _dedupe_crossings_by_same_sheet(branches, crossing_certs)

    n_dup = sum(
        1
        for sl in slices
        for n in sl["nodes"]
        if n.get("is_seed_duplicate")
    )

    def branch_to_dict(br: CompetitorBranch) -> dict[str, Any]:
        return {
            "branch_id": br.branch_id,
            "label": br.label,
            "is_seed_sheet": br.is_seed_sheet,
            "possible_same_sheet_as_branch_id": br.possible_same_sheet_as_branch_id,
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
                    "delta_lower": p.delta_lower,
                    "status": p.status,
                    "distance_to_seed_w": p.distance_to_seed_w,
                    "DeltaRe_vs_seed": p.DeltaRe_vs_seed,
                    "DeltaIm_vs_seed": p.DeltaIm_vs_seed,
                    "DeltaRe_interval_vs_seed": p.DeltaRe_interval_vs_seed,
                    "RePhi_interval": p.RePhi_interval,
                    "ImPhi_interval": p.ImPhi_interval,
                    "action_width_real": p.action_width_real,
                    "log_branch_method": p.log_branch_method,
                    "is_seed_duplicate": p.is_seed_duplicate,
                    "box_disjoint_from_seed": p.box_disjoint_from_seed,
                    "krawczyk_certified": p.krawczyk_certified,
                    "pl_active_note": PL_ACTIVE_NOTE if p.status == STATUS_COMPETITOR_PL_ACTIVE else None,
                    "seed_duplicate_note": SEED_DUPLICATE_NOTE if p.status == STATUS_SEED_DUPLICATE else None,
                }
                for p in sorted(br.points, key=lambda x: x.gamma, reverse=True)
            ],
        }

    n_pl = sum(
        1
        for br in branches
        for p in br.points
        if p.status == STATUS_COMPETITOR_PL_ACTIVE
    )
    n_comp = sum(
        1
        for br in branches
        for p in br.points
        if p.status in (STATUS_COMPETITOR_CERTIFIED_LOCAL, STATUS_COMPETITOR_PL_ACTIVE)
        and p.box_disjoint_from_seed
    )

    payload: dict[str, Any] = {
        "r": float(sweep_payload.get("r", 1.0)),
        "certificate_type": CERTIFICATE_TYPE_BRANCH_RESOLVED,
        "branch_step_tol": float(branch_step_tol),
        "use_tangent_matching": bool(use_tangent),
        "merge_same_sheet_branches": bool(merge_same_sheet),
        "seed_duplicate_w_factor": SEED_DUPLICATE_W_FACTOR,
        "num_branches": len(branches),
        "num_seed_duplicate_nodes": n_dup,
        "num_competitor_points_disjoint": n_comp,
        "num_pl_active_diagnostic_points": n_pl,
        "same_sheet_diagnoses": same_sheet_diagnoses,
        "status_legend": {
            STATUS_SEED_CERTIFIED_LOCAL: "Krawczyk: unique root in seed box (local).",
            STATUS_COMPETITOR_CERTIFIED_LOCAL: "Krawczyk: unique root in competitor box (local); disjoint from seed.",
            STATUS_COMPETITOR_PL_ACTIVE: PL_ACTIVE_NOTE,
            STATUS_SEED_DUPLICATE: SEED_DUPLICATE_NOTE,
        },
        "theorem_hierarchy_note": (
            "Does NOT claim global uniqueness or seed dominance. "
            "Multiple certified sheets may coexist. "
            "Crossings require disjoint Krawczyk boxes. "
            f"{COMPETITOR_DISCLAIMER}"
        ),
        "branches": [branch_to_dict(br) for br in branches],
        "delta_re_crossing_intervals_mesh": all_intervals,
        "crossing_refinement_near_gamma": refine_crossing_near,
        "crossing_certificates": crossing_certs,
        "proof_note": lean_proof_note(CERTIFICATE_TYPE_BRANCH_RESOLVED),
    }
    return payload


def plot_branch_resolved_analysis(
    resolved: dict[str, Any],
    plot_path: Path,
    *,
    show_diagnostic_best_competitor: bool = False,
    sweep_payload: Optional[dict[str, Any]] = None,
    allow_overwrite: bool = False,
) -> Path:
    """Write the unified dashboard (same file as sweep analysis when sweep JSON is passed)."""
    del show_diagnostic_best_competitor  # retained for CLI compatibility
    if sweep_payload is not None:
        return plot_competitor_analysis(
            sweep_payload,
            plot_path,
            resolved_payload=resolved,
            allow_overwrite=allow_overwrite,
        )
    return _plot_resolve_only_dashboard(
        resolved, plot_path, allow_overwrite=allow_overwrite
    )

