"""
Certified negative-gamma continuation for the corrected w-saddle slice.

Loops a gamma mesh, warm-starts each point from the previous certified root,
and records proof-relevant quantities (Step 18): w*, Re/Im Phi_eff, delta_lower,
and Krawczyk contraction |I - YJ|.

Example::

    python -m phasecraft.continue_w_saddle_branch \\
      --r 1.0 --gamma-start -0.01 --gamma-stop -1.0 --num-points 100 \\
      --dps 80 --escalate \\
      --out phasecraft/w_branch_r1_neg_gamma.json \\
      --plot phasecraft/w_branch_r1_neg_gamma.png
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from phasecraft.krawczyk_w_saddle import (
    WSaddleSystem,
    discover_w_roots,
    krawczyk_certify_with_escalation,
    krawczyk_certify_w_root,
    solve_w_from_init,
)


@dataclass
class StopReason:
    code: str
    message: str
    index: int
    gamma: float


def make_gamma_mesh(gamma_start: float, gamma_stop: float, num_points: int) -> np.ndarray:
    if num_points < 2:
        raise ValueError("num_points must be >= 2")
    if gamma_start >= 0 or gamma_stop >= 0:
        raise ValueError("gamma_start and gamma_stop must be negative (seed chamber)")
    if gamma_stop >= gamma_start:
        raise ValueError("gamma_stop must be more negative than gamma_start (decreasing mesh)")
    return np.linspace(gamma_start, gamma_stop, num_points)


def unwrap_im_branch(im_raw: list[float]) -> tuple[list[float], bool]:
    """Unwrap Im(Phi) along the branch; return (unwrapped, continuous_ok)."""
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
) -> tuple[bool, dict[str, Any]]:
    if escalate:
        return krawczyk_certify_with_escalation(
            sys, w, dps_start=dps, max_inflate_iters=max_inflate_iters
        )
    return krawczyk_certify_w_root(sys, w, dps=dps, max_inflate_iters=max_inflate_iters)


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
) -> dict[str, Any]:
    """Run gamma continuation; stop on first failed proof predicate."""
    points: list[dict[str, Any]] = []
    stop: Optional[StopReason] = None
    w_prev: Optional[np.ndarray] = w_init
    im_raw: list[float] = []

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
        )

        phi = sys.Phi_eff(w_star)
        im_raw.append(float(phi.imag))

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

        if j > 0:
            im_jump = abs(im_raw[j] - im_raw[j - 1])
            if im_jump > branch_jump_tol:
                stop = StopReason(
                    "action_branch_discontinuity",
                    f"|Im Phi jump|={im_jump:.6g} > {branch_jump_tol}",
                    j,
                    float(gamma),
                )
                break

        w_prev = w_star

    im_unwrapped, unwrap_ok = unwrap_im_branch(im_raw)
    for row, im_u in zip(points, im_unwrapped):
        row["Phi_eff_im_unwrapped"] = float(im_u)

    if not unwrap_ok and stop is None and points:
        stop = StopReason(
            "action_branch_discontinuity",
            "Im(Phi) branch could not be unwrapped continuously along mesh",
            len(points) - 1,
            float(points[-1]["gamma"]),
        )

    payload = {
        "r": float(r),
        "gamma_mesh": gammas.tolist(),
        "num_points_requested": int(len(gammas)),
        "num_points_completed": int(len(points)),
        "completed_full_mesh": stop is None and len(points) == len(gammas),
        "certificate_type": "discrete_gamma_points",
        "proof_note": (
            "Each mesh point has a local Krawczyk box (typically radius ~1e-16); "
            "this is NOT an interval-in-gamma certificate. Boxes do not overlap "
            "between adjacent gamma values."
        ),
        "stop": None if stop is None else {
            "code": stop.code,
            "message": stop.message,
            "index": stop.index,
            "gamma": stop.gamma,
        },
        "points": points,
        "unwrap_continuous": bool(unwrap_ok),
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
                "Im_Phi_unwrapped": p.get("Phi_eff_im_unwrapped", p["Phi_eff_imag"]),
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
    out["refined_window"] = {
        "gamma_lo": gamma_lo,
        "gamma_hi": gamma_hi,
        "num_dense_points": len(dense_gammas),
    }
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
        "certificate_type": "discrete_gamma_points_tracked_competitor",
        "proof_note": (
            "Competitor w is warm-started along one sheet from anchor; "
            "Re(Phi) gap is vs certified/interpolated seed at each gamma."
        ),
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
        )
        for g in anchor_gammas
    ]
    return {"tracked_competitor_branches": branches}


def plot_tracked_competitor_branches(payload: dict[str, Any], plot_path: Path) -> None:
    """Re(Phi_seed) - Re(Phi_competitor) along each tracked competitor sheet."""
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
        ax.set_title(br.get("branch_label", "tracked competitor"))
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.25)
    fig.suptitle(
        "Tracked competitor branch: certified gap vs seed (one sheet per panel)",
        fontsize=11,
    )
    fig.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)


def plot_competitor_analysis(payload: dict[str, Any], plot_path: Path) -> None:
    """Dedicated competitor-gap diagnostics (more informative than branch-only plot)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pts = sorted(payload["points"], key=lambda p: float(p["gamma"]), reverse=True)
    if not pts:
        raise ValueError("no points to plot")
    gamma = np.array([float(p["gamma"]) for p in pts])
    re_seed = np.array([float(p["Phi_eff_real"]) for p in pts])
    min_gap = np.array(
        [p["min_certified_gap_re"] if p.get("min_certified_gap_re") is not None else np.nan for p in pts]
    )
    min_pos = np.array(
        [p.get("min_positive_gap_re") if p.get("min_positive_gap_re") is not None else np.nan for p in pts]
    )
    best_comp = np.array(
        [p.get("best_competitor_re_phi") if p.get("best_competitor_re_phi") is not None else np.nan for p in pts]
    )
    n_comp = np.array([int(p.get("num_certified_competitors", 0) or 0) for p in pts])
    n_cert = np.array([int(p.get("num_certified_roots", 0) or 0) for p in pts])

    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)

    ax = axes[0, 0]
    colors = np.where(min_gap > 0, "C2", np.where(np.isnan(min_gap), "gray", "C3"))
    ax.scatter(gamma, min_gap, c=colors, s=18, zorder=3)
    ax.plot(gamma, min_gap, "-", lw=0.8, alpha=0.5, color="k")
    ax.axhline(0.0, color="red", ls="--", lw=1, label="Re gap = 0")
    ax.set_xlabel(r"$\gamma$")
    ax.set_ylabel(r"$\min_j(\mathrm{Re}\,\Phi_{\rm seed}-\mathrm{Re}\,\Phi_j)$")
    ax.set_title("Certified action gap (green=seed wins)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)

    ax = axes[0, 1]
    ax.plot(gamma, re_seed, "o-", ms=3, lw=1, label=r"seed $\mathrm{Re}\,\Phi$")
    ax.plot(gamma, best_comp, "s--", ms=3, lw=1, alpha=0.8, label=r"best competitor $\mathrm{Re}\,\Phi$")
    ax.set_xlabel(r"$\gamma$")
    ax.set_ylabel(r"$\mathrm{Re}\,\Phi_{\mathrm{eff}}$")
    ax.set_title("Seed vs strongest certified competitor")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 2]
    valid = np.isfinite(min_pos)
    if np.any(valid):
        ax.semilogy(gamma[valid], np.maximum(min_pos[valid], 1e-16), "o-", ms=3, lw=1)
    ax.set_xlabel(r"$\gamma$")
    ax.set_ylabel(r"$\min_{j:\,\Delta_j>0} \Delta_{\rm gap}$")
    ax.set_title("Filtered gap (seed-dominating competitors only)")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(gamma, n_comp, "o-", ms=3, lw=1, label="certified competitors")
    ax.plot(gamma, n_cert, "s--", ms=3, lw=1, alpha=0.7, label="all certified roots")
    ax.set_xlabel(r"$\gamma$")
    ax.set_ylabel("count")
    ax.set_title("Discovery / certification yield")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Zoom on potential wall region: gamma in [-1, -0.7]
    ax = axes[1, 1]
    mask = gamma <= -0.7
    if np.any(mask):
        g_z = gamma[mask]
        ax.scatter(g_z, min_gap[mask], c=np.where(min_gap[mask] > 0, "C2", "C3"), s=25)
        ax.axhline(0.0, color="red", ls="--", lw=1)
        ax.set_xlabel(r"$\gamma$")
        ax.set_ylabel(r"min certified $\Delta_{\rm gap}$")
        ax.set_title(r"Zoom: $\gamma\in[-1,-0.7]$")
        ax.grid(True, alpha=0.3)

    # Summary bar: fraction of mesh where seed dominates
    ax = axes[1, 2]
    dom = sum(1 for p in pts if p.get("seed_dominates_all"))
    threat = sum(
        1
        for p in pts
        if p.get("min_certified_gap_re") is not None and float(p["min_certified_gap_re"]) <= 0
    )
    no_comp = sum(1 for p in pts if int(p.get("num_certified_competitors", 0) or 0) == 0)
    labels = ["seed dominates", "competitor wins", "no competitor"]
    vals = [dom, threat, no_comp]
    ax.bar(labels, vals, color=["C2", "C3", "C0"])
    ax.set_ylabel("mesh points")
    ax.set_title("Summary counts")
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=15, ha="right")

    r = payload.get("r", 1.0)
    ref = payload.get("refined_window")
    title = f"Competitor analysis  r={r}  N={len(pts)}"
    if ref:
        title += f"  refined [{ref['gamma_lo']}, {ref['gamma_hi']}]"
    fig.suptitle(title, fontsize=11)
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)


def plot_branch(payload: dict[str, Any], plot_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pts = payload["points"]
    if not pts:
        raise ValueError("no points to plot")
    gamma = np.array([p["gamma"] for p in pts])
    re_phi = np.array([p["Phi_eff_real"] for p in pts])
    im_phi = np.array([p["Phi_eff_imag"] for p in pts])
    im_u = np.array([p.get("Phi_eff_im_unwrapped", p["Phi_eff_imag"]) for p in pts])
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
    ax.set_title("Action (real)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(gamma, im_phi, ".", ms=3, alpha=0.5, label="raw Im")
    ax.plot(gamma, im_u, "o-", ms=3, lw=1, label="unwrapped")
    ax.set_xlabel(r"$\gamma$")
    ax.set_ylabel(r"$\mathrm{Im}\,\Phi_{\mathrm{eff}}$")
    ax.set_title("Action (imag)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.semilogy(gamma, np.maximum(delta, 1e-300), "o-", ms=3, lw=1)
    ax.set_xlabel(r"$\gamma$")
    ax.set_ylabel(r"$\delta_{\mathrm{lower}} = \inf|\Delta_\star|$")
    ax.set_title("Divisor separation")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.semilogy(gamma, np.maximum(kappa, 1e-300), "o-", ms=3, lw=1)
    ax.axhline(1.0, color="r", ls="--", lw=0.8, label="|I-YJ| = 1")
    ax.set_xlabel(r"$\gamma$")
    ax.set_ylabel(r"$|I - YJ|$")
    ax.set_title("Krawczyk contraction")
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
    title = f"w-saddle branch  r={payload['r']}"
    if stop:
        title += f"  STOP @ γ={stop['gamma']:.4g} ({stop['code']})"
    fig.suptitle(title, fontsize=11)
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Certified negative-gamma continuation (w-saddle slice)."
    )
    parser.add_argument("--r", type=float, default=1.0)
    parser.add_argument("--gamma-start", type=float, default=-0.01)
    parser.add_argument("--gamma-stop", type=float, default=-1.0)
    parser.add_argument("--num-points", type=int, default=100)
    parser.add_argument("--dps", type=int, default=80)
    parser.add_argument("--max-inflate-iters", type=int, default=6)
    parser.add_argument(
        "--no-escalate",
        action="store_true",
        help="Disable automatic dps escalation on Krawczyk failure (default: escalate).",
    )
    parser.add_argument("--delta-lower-threshold", type=float, default=1e-6)
    parser.add_argument("--contraction-max", type=float, default=1.0)
    parser.add_argument("--branch-jump-tol", type=float, default=np.pi * 0.95)
    parser.add_argument(
        "--certify-competitors",
        action="store_true",
        help="At each gamma: random starts, cluster, Krawczyk-certify all roots, gap vs seed.",
    )
    parser.add_argument(
        "--competitor-sweep-only",
        action="store_true",
        help="Load --in JSON and run competitor sweep only (no continuation re-run).",
    )
    parser.add_argument(
        "--stop-on-competitor",
        action="store_true",
        help="Stop if min certified Re(Phi) gap <= --competitor-re-tol.",
    )
    parser.add_argument("--competitor-starts", type=int, default=500)
    parser.add_argument("--competitor-re-tol", type=float, default=1e-4)
    parser.add_argument("--cluster-tol", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--in", dest="in_path", type=str, default="", help="Input continuation JSON.")
    parser.add_argument("--w-init-json", type=str, default="", help="Optional JSON with prior w_star.")
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--plot", type=str, default="", help="Branch continuation PNG.")
    parser.add_argument(
        "--plot-analysis",
        type=str,
        default="",
        help="Competitor-gap analysis PNG (recommended after sweep).",
    )
    parser.add_argument(
        "--refine-gamma-lo",
        type=float,
        default=None,
        help="With sweep: replace [lo, hi] window by denser mesh (e.g. -0.98).",
    )
    parser.add_argument("--refine-gamma-hi", type=float, default=None)
    parser.add_argument("--refine-num-points", type=int, default=40)
    parser.add_argument(
        "--track-competitor-branches",
        action="store_true",
        help="Continue dominant competitor sheets from sweep anchors (no random search).",
    )
    parser.add_argument(
        "--seed-branch-in",
        type=str,
        default="",
        help="Seed continuation JSON for gap comparison (with --track-competitor-branches).",
    )
    parser.add_argument(
        "--track-anchors",
        type=str,
        default="-0.62,-0.91",
        help="Comma-separated anchor gammas in competitor sweep JSON.",
    )
    parser.add_argument("--track-gamma-step", type=float, default=0.01)
    parser.add_argument("--track-gamma-hi", type=float, default=-0.50)
    parser.add_argument("--track-gamma-lo", type=float, default=-0.98)
    parser.add_argument(
        "--track-plot",
        type=str,
        default="",
        help="PNG: Re(Phi_seed) - Re(Phi_competitor) along each tracked sheet.",
    )
    args = parser.parse_args()

    if args.track_competitor_branches:
        sweep_path = args.in_path or args.w_init_json
        if not sweep_path:
            raise SystemExit("--track-competitor-branches requires --in (competitor sweep JSON)")
        if not args.seed_branch_in:
            raise SystemExit("--track-competitor-branches requires --seed-branch-in")
        with open(sweep_path, encoding="utf-8") as f:
            sweep_payload = json.load(f)
        with open(args.seed_branch_in, encoding="utf-8") as f:
            seed_payload = json.load(f)
        anchors = [float(x.strip()) for x in args.track_anchors.split(",") if x.strip()]
        payload = track_competitor_branches(
            sweep_payload,
            seed_payload,
            anchors,
            gamma_step=args.track_gamma_step,
            gamma_hi=args.track_gamma_hi,
            gamma_lo=args.track_gamma_lo,
            dps=args.dps,
            max_inflate_iters=args.max_inflate_iters,
            escalate=not args.no_escalate,
            newton_tol=1e-12,
            delta_lower_threshold=args.delta_lower_threshold,
            contraction_max=args.contraction_max,
        )
        payload["metadata"] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "command": " ".join(sys.argv),
            "sweep_in": sweep_path,
            "seed_branch_in": args.seed_branch_in,
            "anchors": anchors,
        }
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(
            json.dumps(
                {
                    "out": str(out_path),
                    "branches": len(payload["tracked_competitor_branches"]),
                    "points": [len(b["points"]) for b in payload["tracked_competitor_branches"]],
                },
                indent=2,
            )
        )
        if args.track_plot:
            plot_tracked_competitor_branches(payload, Path(args.track_plot))
        return

    if args.competitor_sweep_only:
        in_path = args.in_path or args.w_init_json
        if not in_path:
            raise SystemExit("--competitor-sweep-only requires --in PATH")
        with open(in_path, encoding="utf-8") as f:
            payload = json.load(f)
        payload = competitor_sweep_on_payload(
            payload,
            dps=args.dps,
            max_inflate_iters=args.max_inflate_iters,
            escalate=not args.no_escalate,
            competitor_starts=args.competitor_starts,
            cluster_tol=args.cluster_tol,
            newton_tol=1e-12,
            seed=args.seed,
        )
        if args.refine_gamma_lo is not None and args.refine_gamma_hi is not None:
            dense = np.linspace(args.refine_gamma_lo, args.refine_gamma_hi, args.refine_num_points)
            payload = merge_refined_gamma_window(
                payload,
                args.refine_gamma_lo,
                args.refine_gamma_hi,
                dense,
                dps=args.dps,
                max_inflate_iters=args.max_inflate_iters,
                escalate=not args.no_escalate,
                competitor_starts=args.competitor_starts,
                cluster_tol=args.cluster_tol,
                newton_tol=1e-12,
                seed=args.seed + 50000,
            )
    else:
        gammas = make_gamma_mesh(args.gamma_start, args.gamma_stop, args.num_points)
        w_init = None
        init_path = args.in_path or args.w_init_json
        if init_path:
            with open(init_path, encoding="utf-8") as f:
                prior = json.load(f)
            if "points" in prior and prior["points"]:
                last = prior["points"][-1]
                w_init = np.array(last["w_star_real"]) + 1j * np.array(last["w_star_imag"])
            elif "roots" in prior and prior["roots"]:
                last = prior["roots"][0]
                w_init = np.array(last["w_real"]) + 1j * np.array(last["w_imag"])

        payload = continue_negative_gamma_branch(
            args.r,
            gammas,
            w_init=w_init,
            dps=args.dps,
            max_inflate_iters=args.max_inflate_iters,
            escalate=not args.no_escalate,
            delta_lower_threshold=args.delta_lower_threshold,
            contraction_max=args.contraction_max,
            branch_jump_tol=args.branch_jump_tol,
            certify_competitors=args.certify_competitors,
            stop_on_competitor=args.stop_on_competitor,
            competitor_starts=args.competitor_starts,
            competitor_re_tol=args.competitor_re_tol,
            cluster_tol=args.cluster_tol,
            seed=args.seed,
        )
    payload["metadata"] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "gamma_start": payload.get("gamma_mesh", [args.gamma_start])[0]
        if payload.get("gamma_mesh")
        else args.gamma_start,
        "gamma_stop": payload.get("gamma_mesh", [args.gamma_stop])[-1]
        if payload.get("gamma_mesh")
        else args.gamma_stop,
        "num_points": payload.get("num_points_requested", args.num_points),
        "dps": args.dps,
        "escalate": not args.no_escalate,
        "certify_competitors": args.certify_competitors or args.competitor_sweep_only,
        "competitor_starts": args.competitor_starts,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    summary = {
        "out": str(out_path),
        "completed": payload["num_points_completed"],
        "requested": payload["num_points_requested"],
        "full_mesh": payload["completed_full_mesh"],
        "stop": payload["stop"],
    }
    print(json.dumps(summary, indent=2))

    if args.plot:
        plot_branch(payload, Path(args.plot))
    if args.plot_analysis:
        plot_competitor_analysis(payload, Path(args.plot_analysis))


if __name__ == "__main__":
    main()
