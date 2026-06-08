"""
CLI for the w-saddle Krawczyk workflow (``python -m phasecraft.w_saddle``).

Artifacts default to ``phasecraft/w_saddle/runs/`` via the ``pipeline`` preset.
See ``phasecraft/w_saddle/README.md``.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from phasecraft.lib.certificates.certificate_language import ALLOWED_GAMMA_CONVENTIONS, finalize_payload_certificate
from phasecraft.w_saddle.core import (
    SLICE_BETA,
    WSaddleSystem,
    discover_w_roots,
    krawczyk_certify_w_root,
    krawczyk_certify_with_escalation,
)
from phasecraft.w_saddle.workflow import (
    continue_negative_gamma_branch,
    competitor_sweep_on_payload,
    detect_refine_gamma_window,
    make_gamma_mesh,
    merge_refined_gamma_window,
    plot_branch,
    plot_branch_resolved_analysis,
    plot_competitor_analysis,
    plot_tracked_competitor_branches,
    resolve_competitor_branches,
    track_competitor_branches,
)

PACKAGE_DIR = Path(__file__).resolve().parent
RUNS_DIR = PACKAGE_DIR / "runs"


def _run_dir_ids(runs_dir: Path = RUNS_DIR) -> list[int]:
    if not runs_dir.is_dir():
        return []
    ids: list[int] = []
    for p in runs_dir.iterdir():
        if p.is_dir() and p.name.startswith("run") and p.name[3:].isdigit():
            ids.append(int(p.name[3:]))
    return sorted(ids)


def allocate_run_dir(
    runs_dir: Path = RUNS_DIR,
    *,
    run_id: Optional[int] = None,
) -> Path:
    """Create ``runs/runN`` (next N, or a fixed id)."""
    runs_dir.mkdir(parents=True, exist_ok=True)
    if run_id is not None:
        out = runs_dir / f"run{int(run_id)}"
        out.mkdir(parents=True, exist_ok=True)
        return out
    next_id = max(_run_dir_ids(runs_dir), default=0) + 1
    out = runs_dir / f"run{next_id}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def artifact_paths(run_dir: Path, *, robust: bool) -> dict[str, str]:
    """Standard filenames inside one pipeline run folder."""
    d = Path(run_dir)
    if robust:
        return {
            "seed_json": str(d / "seed_r1_neg_gamma.json"),
            "sweep_json": str(d / "competitors_robust.json"),
            "resolved_json": str(d / "resolved_robust.json"),
            "seed_plot": str(d / "seed_branch.png"),
            "sweep_plot": str(d / "competitors_robust_analysis.png"),
            "resolved_plot": str(d / "resolved_robust.png"),
        }
    return {
        "seed_json": str(d / "seed_r1_neg_gamma.json"),
        "sweep_json": str(d / "competitors.json"),
        "resolved_json": str(d / "resolved.json"),
        "seed_plot": str(d / "seed_branch.png"),
        "sweep_plot": str(d / "competitors_analysis.png"),
        "resolved_plot": str(d / "resolved.png"),
    }


def default_runs_paths() -> dict[str, str]:
    """Legacy flat paths under ``runs/`` (prefer ``allocate_run_dir`` for new work)."""
    return artifact_paths(RUNS_DIR, robust=False)


def default_runs_paths_robust() -> dict[str, str]:
    return artifact_paths(RUNS_DIR, robust=True)


PRESETS: dict[str, dict[str, str]] = {
    "r1-neg-gamma": default_runs_paths(),
    "r1-neg-gamma-robust": default_runs_paths_robust(),
}


def _add_output_flags(parser: argparse.ArgumentParser) -> None:
    g = parser.add_mutually_exclusive_group()
    g.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print full JSON summaries (default: one line per step).",
    )


def _emit_step(args: argparse.Namespace, step: str, fields: dict[str, Any]) -> None:
    """Compact terminal output; full detail stays in written JSON files."""
    if getattr(args, "verbose", False):
        print(json.dumps({step: fields}, indent=2))
        return
    parts = [f"w_saddle/{step}:"]
    for key, val in fields.items():
        if val is None:
            continue
        if isinstance(val, bool):
            parts.append(f"{key}={'ok' if val else 'no'}")
        elif isinstance(val, (int, np.integer)):
            parts.append(f"{key}={int(val)}")
        elif isinstance(val, float):
            parts.append(f"{key}={val:.6g}")
        else:
            parts.append(f"{key}={val}")
    print(" ".join(parts))


def _refined_window_fields(rw: Optional[dict[str, Any]]) -> dict[str, Any]:
    if not rw:
        return {}
    if rw.get("method") == "auto_skipped":
        return {"wall": "skipped", "reason": rw.get("reason", "")}
    lo, hi = float(rw["gamma_lo"]), float(rw["gamma_hi"])
    fields: dict[str, Any] = {
        "wall": f"[{lo:.4g},{hi:.4g}]",
        "dense": rw.get("num_dense_points"),
    }
    ad = rw.get("auto_detection") or {}
    if ad.get("selected_run"):
        sr = ad["selected_run"]
        fields["core"] = f"[{sr[0]:.4g},{sr[1]:.4g}]"
        fields["flagged"] = ad.get("num_flagged_points")
    elif rw.get("method") == "manual_cli":
        fields["wall_mode"] = "manual"
    return fields


def _add_proof_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--gamma-convention",
        type=str,
        default="",
        choices=[""] + sorted(ALLOWED_GAMMA_CONVENTIONS),
    )
    parser.add_argument("--enclose-action", action="store_true")
    parser.add_argument("--proof-metadata", action="store_true")
    parser.add_argument("--theorem-levels", type=str, default="A,B")
    parser.add_argument("--allow-theorem-level-c", action="store_true")
    parser.add_argument("--plot-overwrite", action="store_true")


def _save_payload(
    payload: dict[str, Any],
    out_path: Path,
    *,
    gamma_convention: Optional[str],
    enclose_action: bool,
    proof_metadata: bool,
    theorem_levels: list[str],
    allow_level_c: bool,
) -> dict[str, Any]:
    finalize_payload_certificate(
        payload,
        gamma_convention=gamma_convention or None,
        enclose_action=enclose_action,
        proof_metadata=proof_metadata,
        theorem_levels=theorem_levels,
        allow_level_c=allow_level_c,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return payload


def _theorem_levels_from_args(args: argparse.Namespace) -> list[str]:
    return [x.strip().upper() for x in args.theorem_levels.split(",") if x.strip()]


def parse_track_anchors(values: Sequence[str]) -> list[float]:
    """Parse anchor gammas from CLI tokens (comma and/or space separated)."""
    anchors: list[float] = []
    for token in values:
        for part in str(token).split(","):
            part = part.strip()
            if part:
                anchors.append(float(part))
    if not anchors:
        raise ValueError("track-anchors must list at least one gamma value")
    return anchors


def cmd_certify(args: argparse.Namespace) -> None:
    beta = float(getattr(args, "beta", SLICE_BETA))
    sys = WSaddleSystem(r=args.r, gamma=args.gamma, beta=beta)
    roots_w = discover_w_roots(
        sys, num_starts=args.num_starts, seed=args.seed, tol=args.tol
    )
    payload: dict[str, Any] = {
        "r": args.r,
        "gamma": args.gamma,
        "beta": beta,
        "p": sys.p,
        "num_roots": len(roots_w),
        "couplings_real": [float(c.real) for c in sys.c],
        "couplings_imag": [float(c.imag) for c in sys.c],
        "leading_seed_real": [float(w.real) for w in sys.leading_seed()],
        "leading_seed_imag": [float(w.imag) for w in sys.leading_seed()],
        "roots": [],
    }
    for i, w in enumerate(roots_w):
        f_norm = float(np.linalg.norm(sys.F_complex(w), ord=np.inf))
        if args.escalate:
            cert, info = krawczyk_certify_with_escalation(
                sys,
                w,
                dps_start=args.dps,
                max_inflate_iters=args.max_inflate_iters,
                enclose_action=args.enclose_action,
            )
        else:
            cert, info = krawczyk_certify_w_root(
                sys,
                w,
                dps=args.dps,
                max_inflate_iters=args.max_inflate_iters,
                enclose_action=args.enclose_action,
            )
        phi = sys.Phi_eff(w)
        payload["roots"].append(
            {
                "idx": i,
                "w_real": w.real.tolist(),
                "w_imag": w.imag.tolist(),
                "residual_F_norm": f_norm,
                "Phi_eff_real": float(phi.real),
                "Phi_eff_imag": float(phi.imag),
                "krawczyk_certified": bool(cert),
                "krawczyk_info": info,
            }
        )
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    _emit_step(
        args,
        "certify",
        {
            "out": args.out or "(stdout)",
            "gamma": args.gamma,
            "roots": payload["num_roots"],
            "certified": sum(1 for r in payload["roots"] if r["krawczyk_certified"]),
        },
    )


def _load_w_init(path: str) -> Optional[np.ndarray]:
    if not path:
        return None
    with open(path, encoding="utf-8") as f:
        prior = json.load(f)
    if "points" in prior and prior["points"]:
        last = prior["points"][-1]
        return np.array(last["w_star_real"]) + 1j * np.array(last["w_star_imag"])
    if "roots" in prior and prior["roots"]:
        last = prior["roots"][0]
        return np.array(last["w_real"]) + 1j * np.array(last["w_imag"])
    return None


def cmd_continue(args: argparse.Namespace) -> None:
    gammas = make_gamma_mesh(args.gamma_start, args.gamma_stop, args.num_points)
    init_path = args.in_path or args.w_init_json
    payload = continue_negative_gamma_branch(
        args.r,
        gammas,
        w_init=_load_w_init(init_path) if init_path else None,
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
        enclose_action=args.enclose_action,
        gamma_convention=args.gamma_convention or None,
    )
    payload["metadata"] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "subcommand": "continue",
        "dps": args.dps,
        "escalate": not args.no_escalate,
        "certify_competitors": args.certify_competitors,
    }
    out_path = Path(args.out)
    levels = _theorem_levels_from_args(args)
    payload = _save_payload(
        payload,
        out_path,
        gamma_convention=args.gamma_convention or None,
        enclose_action=args.enclose_action,
        proof_metadata=args.proof_metadata,
        theorem_levels=levels,
        allow_level_c=args.allow_theorem_level_c,
    )
    stop = payload.get("stop")
    _emit_step(
        args,
        "continue",
        {
            "out": out_path,
            "points": f"{payload['num_points_completed']}/{payload['num_points_requested']}",
            "full_mesh": payload["completed_full_mesh"],
            "stop": stop["code"] if stop else "none",
        },
    )
    if args.plot:
        plot_branch(payload, Path(args.plot), allow_overwrite=args.plot_overwrite)
    if args.plot_analysis:
        plot_competitor_analysis(
            payload, Path(args.plot_analysis), allow_overwrite=args.plot_overwrite
        )


def _apply_wall_refinement(payload: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """Dense resweep in a gamma window (manual bounds or auto-detected from coarse sweep)."""
    refine_meta: dict[str, Any]
    if args.refine_gamma_lo is not None and args.refine_gamma_hi is not None:
        gamma_lo, gamma_hi = float(args.refine_gamma_lo), float(args.refine_gamma_hi)
        refine_meta = {"method": "manual_cli"}
    elif not getattr(args, "no_auto_refine_wall", False):
        detected = detect_refine_gamma_window(
            payload,
            gap_threshold=args.refine_gap_threshold,
            spike_threshold=args.refine_spike_threshold,
            margin=args.refine_margin,
        )
        if detected is None:
            payload["refined_window"] = {"method": "auto_skipped", "reason": "no_competitor_activity"}
            return payload
        gamma_lo, gamma_hi, refine_meta = detected
    else:
        return payload

    dense = np.linspace(gamma_lo, gamma_hi, int(args.refine_num_points))
    payload = merge_refined_gamma_window(
        payload,
        gamma_lo,
        gamma_hi,
        dense,
        dps=args.dps,
        max_inflate_iters=args.max_inflate_iters,
        escalate=not args.no_escalate,
        competitor_starts=args.competitor_starts,
        cluster_tol=args.cluster_tol,
        newton_tol=1e-12,
        seed=args.seed + 50000,
    )
    payload["refined_window"]["auto_detection"] = refine_meta
    return payload


def cmd_sweep(args: argparse.Namespace) -> None:
    with open(args.in_path, encoding="utf-8") as f:
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
    payload = _apply_wall_refinement(payload, args)
    out_path = Path(args.out)
    levels = _theorem_levels_from_args(args)
    payload = _save_payload(
        payload,
        out_path,
        gamma_convention=args.gamma_convention or None,
        enclose_action=args.enclose_action,
        proof_metadata=args.proof_metadata,
        theorem_levels=levels,
        allow_level_c=args.allow_theorem_level_c,
    )
    rw = payload.get("refined_window") or {}
    fields: dict[str, Any] = {
        "out": out_path,
        "points": payload.get("num_points_completed"),
        **_refined_window_fields(rw),
    }
    n = int(payload.get("num_points_completed") or 0)
    if n < 90 and rw.get("gamma_lo") is not None:
        fields["warn"] = "mesh_shrunk_check_JSON_refined_window"
    _emit_step(args, "sweep", fields)
    if args.plot_analysis:
        plot_competitor_analysis(
            payload,
            Path(args.plot_analysis),
            allow_overwrite=args.plot_overwrite,
        )


def cmd_resolve(args: argparse.Namespace) -> None:
    with open(args.sweep_in, encoding="utf-8") as f:
        sweep_payload = json.load(f)
    with open(args.seed_branch_in, encoding="utf-8") as f:
        seed_payload = json.load(f)
    payload = resolve_competitor_branches(
        sweep_payload,
        seed_payload,
        branch_step_tol=args.branch_step_tol,
        seed_mean_dist_max=args.seed_mean_dist_max,
        crossing_target_width=args.crossing_target_width,
        refine_crossing_near=args.refine_crossing_near,
        refine_window=args.refine_window,
        refine_all_crossings=not args.no_crossing_refinement,
        dps=args.dps,
        max_inflate_iters=args.max_inflate_iters,
        escalate=not args.no_escalate,
    )
    payload["crossing_refinement_mode"] = (
        "none"
        if args.no_crossing_refinement
        else (
            "all_mesh_brackets"
            if args.refine_crossing_near is None
            else f"near_{args.refine_crossing_near}"
        )
    )
    out_path = Path(args.out)
    levels = _theorem_levels_from_args(args)
    payload = _save_payload(
        payload,
        out_path,
        gamma_convention=args.gamma_convention or None,
        enclose_action=args.enclose_action,
        proof_metadata=args.proof_metadata,
        theorem_levels=levels,
        allow_level_c=args.allow_theorem_level_c,
    )
    plot_path = args.plot or args.plot_analysis
    if plot_path:
        plot_competitor_analysis(
            sweep_payload,
            Path(plot_path),
            resolved_payload=payload,
            allow_overwrite=args.plot_overwrite,
        )
    _emit_step(
        args,
        "resolve",
        {
            "out": out_path,
            "branches": payload["num_branches"],
            "crossings": len(payload.get("crossing_certificates", [])),
        },
    )


def cmd_track(args: argparse.Namespace) -> None:
    with open(args.sweep_in, encoding="utf-8") as f:
        sweep_payload = json.load(f)
    with open(args.seed_branch_in, encoding="utf-8") as f:
        seed_payload = json.load(f)
    anchors = parse_track_anchors(args.track_anchors)
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
        enclose_action=args.enclose_action,
        gamma_convention=args.gamma_convention or None,
    )
    payload["metadata"] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "subcommand": "track",
        "anchors": anchors,
    }
    out_path = Path(args.out)
    levels = _theorem_levels_from_args(args)
    _save_payload(
        payload,
        out_path,
        gamma_convention=args.gamma_convention or None,
        enclose_action=args.enclose_action,
        proof_metadata=args.proof_metadata,
        theorem_levels=levels,
        allow_level_c=args.allow_theorem_level_c,
    )
    _emit_step(
        args,
        "track",
        {"out": out_path, "branches": len(payload["tracked_competitor_branches"])},
    )
    if args.plot:
        plot_tracked_competitor_branches(
            payload, Path(args.plot), allow_overwrite=args.plot_overwrite
        )


def cmd_pipeline(args: argparse.Namespace) -> None:
    preset_name = args.preset
    if args.robust_sweep and preset_name == "r1-neg-gamma":
        preset_name = "r1-neg-gamma-robust"
    robust = preset_name.endswith("robust")

    run_dir: Optional[Path] = None
    if args.run_dir:
        run_dir = Path(args.run_dir).resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
    elif args.run is not None:
        run_dir = allocate_run_dir(run_id=int(args.run))
    elif not args.output_dir:
        run_dir = allocate_run_dir()

    if run_dir is not None:
        preset = artifact_paths(run_dir, robust=robust)
    elif args.output_dir:
        preset = dict(PRESETS[preset_name])
        base = Path(args.output_dir)
        for key in ("seed_json", "sweep_json", "resolved_json", "seed_plot", "sweep_plot", "resolved_plot"):
            preset[key] = str(base / Path(preset[key]).name)
    else:
        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        preset = dict(PRESETS[preset_name])

    seed_json = Path(preset["seed_json"])
    sweep_json = Path(preset["sweep_json"])
    resolved_json = Path(preset["resolved_json"])

    if not args.skip_continue:
        cont = argparse.Namespace(
            r=args.r,
            gamma_start=args.gamma_start,
            gamma_stop=args.gamma_stop,
            num_points=args.num_points,
            dps=args.dps,
            max_inflate_iters=args.max_inflate_iters,
            no_escalate=args.no_escalate,
            delta_lower_threshold=1e-6,
            contraction_max=1.0,
            branch_jump_tol=np.pi * 0.95,
            certify_competitors=False,
            stop_on_competitor=False,
            competitor_starts=500,
            competitor_re_tol=1e-4,
            cluster_tol=1e-5,
            seed=args.seed,
            enclose_action=args.enclose_action or args.robust_sweep,
            gamma_convention=args.gamma_convention,
            proof_metadata=args.proof_metadata,
            theorem_levels=args.theorem_levels,
            allow_theorem_level_c=args.allow_theorem_level_c,
            plot_overwrite=args.plot_overwrite,
            verbose=args.verbose,
            in_path="",
            w_init_json="",
            out=str(seed_json),
            plot=str(Path(preset["seed_plot"])) if args.plots else "",
            plot_analysis="",
        )
        cmd_continue(cont)

    if not args.skip_sweep:
        sw = argparse.Namespace(
            in_path=str(seed_json),
            out=str(sweep_json),
            dps=args.dps,
            max_inflate_iters=args.max_inflate_iters,
            no_escalate=args.no_escalate,
            competitor_starts=args.competitor_starts,
            cluster_tol=1e-5,
            seed=args.seed,
            refine_gamma_lo=args.refine_gamma_lo,
            refine_gamma_hi=args.refine_gamma_hi,
            refine_num_points=args.refine_num_points,
            no_auto_refine_wall=args.no_auto_refine_wall,
            refine_gap_threshold=args.refine_gap_threshold,
            refine_spike_threshold=args.refine_spike_threshold,
            refine_margin=args.refine_margin,
            enclose_action=args.enclose_action or args.robust_sweep,
            gamma_convention=args.gamma_convention,
            proof_metadata=args.proof_metadata,
            theorem_levels=args.theorem_levels,
            allow_theorem_level_c=args.allow_theorem_level_c,
            plot_overwrite=args.plot_overwrite,
            verbose=args.verbose,
            plot_analysis=str(Path(preset["sweep_plot"])) if args.plots else "",
        )
        cmd_sweep(sw)

    if not args.skip_resolve:
        res = argparse.Namespace(
            sweep_in=str(sweep_json),
            seed_branch_in=str(seed_json),
            out=str(resolved_json),
            branch_step_tol=0.35,
            seed_mean_dist_max=0.05,
            crossing_target_width=0.008,
            refine_crossing_near=None,
            refine_window=0.12,
            refine_all_crossings=True,
            dps=args.dps,
            max_inflate_iters=args.max_inflate_iters,
            no_escalate=args.no_escalate,
            enclose_action=args.enclose_action or args.robust_sweep,
            no_crossing_refinement=False,
            gamma_convention=args.gamma_convention,
            proof_metadata=args.proof_metadata,
            theorem_levels=args.theorem_levels,
            allow_theorem_level_c=args.allow_theorem_level_c,
            plot_overwrite=args.plot_overwrite,
            verbose=args.verbose,
            plot=str(Path(preset["sweep_plot"])) if args.plots else "",
            plot_analysis="",
        )
        cmd_resolve(res)

    fields: dict[str, Any] = {
        "preset": preset_name,
        "seed": seed_json,
        "sweep": sweep_json,
        "resolved": resolved_json,
    }
    if run_dir is not None:
        fields["run"] = run_dir.name
    _emit_step(args, "pipeline", fields)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="W-saddle Krawczyk workflow (certify, continue, sweep, resolve, track, pipeline)."
    )
    sub = parser.add_subparsers(dest="subcommand", required=True)

    p_cert = sub.add_parser("certify", help="Single-gamma root discovery + Krawczyk boxes.")
    p_cert.add_argument("--r", type=float, required=True)
    p_cert.add_argument("--gamma", type=float, required=True)
    p_cert.add_argument("--beta", type=float, default=SLICE_BETA,
                        help="Mixer angle beta (default -pi/2).")
    p_cert.add_argument("--num-starts", type=int, default=200)
    p_cert.add_argument("--seed", type=int, default=0)
    p_cert.add_argument("--tol", type=float, default=1e-12)
    p_cert.add_argument("--dps", type=int, default=60)
    p_cert.add_argument("--max-inflate-iters", type=int, default=6)
    p_cert.add_argument("--escalate", action="store_true")
    p_cert.add_argument("--out", type=str, default="")
    p_cert.add_argument("--enclose-action", action="store_true")
    _add_output_flags(p_cert)
    p_cert.set_defaults(func=cmd_certify)

    p_cont = sub.add_parser("continue", help="Certified negative-gamma seed branch.")
    p_cont.add_argument("--r", type=float, default=1.0)
    p_cont.add_argument("--gamma-start", type=float, default=-0.01)
    p_cont.add_argument("--gamma-stop", type=float, default=-1.0)
    p_cont.add_argument("--num-points", type=int, default=100)
    p_cont.add_argument("--dps", type=int, default=80)
    p_cont.add_argument("--max-inflate-iters", type=int, default=6)
    p_cont.add_argument("--no-escalate", action="store_true")
    p_cont.add_argument("--delta-lower-threshold", type=float, default=1e-6)
    p_cont.add_argument("--contraction-max", type=float, default=1.0)
    p_cont.add_argument("--branch-jump-tol", type=float, default=np.pi * 0.95)
    p_cont.add_argument("--certify-competitors", action="store_true")
    p_cont.add_argument("--stop-on-competitor", action="store_true")
    p_cont.add_argument("--competitor-starts", type=int, default=500)
    p_cont.add_argument("--competitor-re-tol", type=float, default=1e-4)
    p_cont.add_argument("--cluster-tol", type=float, default=1e-5)
    p_cont.add_argument("--seed", type=int, default=0)
    p_cont.add_argument("--in", dest="in_path", type=str, default="")
    p_cont.add_argument("--w-init-json", type=str, default="")
    p_cont.add_argument("--out", type=str, required=True)
    p_cont.add_argument("--plot", type=str, default="")
    p_cont.add_argument("--plot-analysis", type=str, default="")
    _add_proof_flags(p_cont)
    _add_output_flags(p_cont)
    p_cont.set_defaults(func=cmd_continue)

    p_sweep = sub.add_parser("sweep", help="Competitor discovery on existing continuation JSON.")
    p_sweep.add_argument("--in", dest="in_path", type=str, required=True)
    p_sweep.add_argument("--out", type=str, required=True)
    p_sweep.add_argument("--dps", type=int, default=80)
    p_sweep.add_argument("--max-inflate-iters", type=int, default=6)
    p_sweep.add_argument("--no-escalate", action="store_true")
    p_sweep.add_argument("--competitor-starts", type=int, default=500)
    p_sweep.add_argument("--cluster-tol", type=float, default=1e-5)
    p_sweep.add_argument("--seed", type=int, default=0)
    p_sweep.add_argument(
        "--refine-gamma-lo",
        type=float,
        default=None,
        help="Manual dense window low gamma (overrides --auto-refine-wall).",
    )
    p_sweep.add_argument("--refine-gamma-hi", type=float, default=None)
    p_sweep.add_argument("--refine-num-points", type=int, default=40)
    p_sweep.add_argument(
        "--no-auto-refine-wall",
        action="store_true",
        help="Skip automatic wall detection after coarse sweep (default: auto-detect and resweep).",
    )
    p_sweep.add_argument("--refine-gap-threshold", type=float, default=1.0)
    p_sweep.add_argument("--refine-spike-threshold", type=float, default=2.0)
    p_sweep.add_argument(
        "--refine-margin",
        type=float,
        default=0.05,
        help="Expand auto-detected window by this much in gamma on each side.",
    )
    p_sweep.add_argument("--plot-analysis", type=str, default="")
    _add_proof_flags(p_sweep)
    _add_output_flags(p_sweep)
    p_sweep.set_defaults(func=cmd_sweep)

    p_res = sub.add_parser("resolve", help="Branch-resolve competitor sheets from sweep JSON.")
    p_res.add_argument("--sweep-in", type=str, required=True)
    p_res.add_argument("--seed-branch-in", type=str, required=True)
    p_res.add_argument("--out", type=str, required=True)
    p_res.add_argument("--plot", type=str, default="")
    p_res.add_argument("--plot-analysis", type=str, default="")
    p_res.add_argument("--branch-step-tol", type=float, default=0.35)
    p_res.add_argument("--seed-mean-dist-max", type=float, default=0.05)
    p_res.add_argument(
        "--refine-crossing-near",
        type=float,
        default=None,
        help="If set, only bisect crossings whose midpoint is within --refine-window of this gamma. "
        "Default: refine every mesh bracket where DeltaRe changes sign on a disjoint competitor sheet.",
    )
    p_res.add_argument("--refine-window", type=float, default=0.12)
    p_res.add_argument(
        "--no-crossing-refinement",
        action="store_true",
        help="Skip Krawczyk bisection at sign-change brackets (branch tracking only).",
    )
    p_res.add_argument("--crossing-target-width", type=float, default=0.008)
    p_res.add_argument("--dps", type=int, default=60)
    p_res.add_argument("--max-inflate-iters", type=int, default=6)
    p_res.add_argument("--no-escalate", action="store_true")
    _add_proof_flags(p_res)
    _add_output_flags(p_res)
    p_res.set_defaults(func=cmd_resolve)

    p_track = sub.add_parser("track", help="Continue one competitor sheet from sweep anchors.")
    p_track.add_argument("--sweep-in", type=str, required=True)
    p_track.add_argument("--seed-branch-in", type=str, required=True)
    p_track.add_argument("--out", type=str, required=True)
    p_track.add_argument(
        "--track-anchors",
        nargs="+",
        default=["-0.62", "-0.91"],
        metavar="GAMMA",
        help=(
            "Anchor gamma values (space-separated negatives work, e.g. "
            "--track-anchors -0.62 -0.66 -0.91). Comma form also works: "
            "--track-anchors=-0.62,-0.66,-0.91"
        ),
    )
    p_track.add_argument("--track-gamma-step", type=float, default=0.01)
    p_track.add_argument("--track-gamma-hi", type=float, default=-0.50)
    p_track.add_argument("--track-gamma-lo", type=float, default=-0.98)
    p_track.add_argument("--dps", type=int, default=80)
    p_track.add_argument("--max-inflate-iters", type=int, default=6)
    p_track.add_argument("--no-escalate", action="store_true")
    p_track.add_argument("--delta-lower-threshold", type=float, default=1e-6)
    p_track.add_argument("--contraction-max", type=float, default=1.0)
    p_track.add_argument("--plot", type=str, default="")
    _add_proof_flags(p_track)
    _add_output_flags(p_track)
    p_track.set_defaults(func=cmd_track)

    p_pipe = sub.add_parser(
        "pipeline",
        help="Run continue → sweep → resolve using a named preset.",
    )
    p_pipe.add_argument("--preset", type=str, default="r1-neg-gamma", choices=sorted(PRESETS))
    p_pipe.add_argument(
        "--run-dir",
        type=str,
        default="",
        help="Write all artifacts under this directory (default: new runs/runN).",
    )
    p_pipe.add_argument(
        "--run",
        type=int,
        default=None,
        metavar="N",
        help="Use runs/runN (create if missing). Ignored if --run-dir is set.",
    )
    p_pipe.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Flat output directory (legacy; disables auto runs/runN).",
    )
    p_pipe.add_argument("--r", type=float, default=1.0)
    p_pipe.add_argument("--gamma-start", type=float, default=-0.01)
    p_pipe.add_argument("--gamma-stop", type=float, default=-1.0)
    p_pipe.add_argument("--num-points", type=int, default=100)
    p_pipe.add_argument("--dps", type=int, default=80)
    p_pipe.add_argument("--max-inflate-iters", type=int, default=6)
    p_pipe.add_argument("--no-escalate", action="store_true")
    p_pipe.add_argument("--competitor-starts", type=int, default=500)
    p_pipe.add_argument("--seed", type=int, default=0)
    p_pipe.add_argument(
        "--refine-gamma-lo",
        type=float,
        default=None,
        help="Optional manual dense window (overrides auto wall detection on sweep).",
    )
    p_pipe.add_argument("--refine-gamma-hi", type=float, default=None)
    p_pipe.add_argument("--refine-num-points", type=int, default=40)
    p_pipe.add_argument(
        "--no-auto-refine-wall",
        action="store_true",
        help="Sweep: do not auto-detect competitor wall; uniform mesh only.",
    )
    p_pipe.add_argument("--refine-gap-threshold", type=float, default=1.0)
    p_pipe.add_argument("--refine-spike-threshold", type=float, default=2.0)
    p_pipe.add_argument("--refine-margin", type=float, default=0.05)
    p_pipe.add_argument(
        "--robust-sweep",
        action="store_true",
        help="Use robust artifact names; sweep auto-detects wall gamma window (no fixed -0.98/-0.82).",
    )
    p_pipe.add_argument("--plots", action="store_true", help="Write PNGs from preset paths.")
    p_pipe.add_argument("--skip-continue", action="store_true")
    p_pipe.add_argument("--skip-sweep", action="store_true")
    p_pipe.add_argument("--skip-resolve", action="store_true")
    _add_proof_flags(p_pipe)
    _add_output_flags(p_pipe)
    p_pipe.set_defaults(func=cmd_pipeline)

    return parser


def legacy_continue_argv(argv: Sequence[str]) -> list[str]:
    """Map ``continue_w_saddle_branch`` flags to ``run_w_saddle`` subcommands."""
    args = list(argv)
    if "--resolve-competitor-branches" in args:
        out: list[str] = ["resolve"]
        i = 0
        while i < len(args):
            a = args[i]
            if a == "--resolve-competitor-branches":
                i += 1
                continue
            if a in ("--in",) and i + 1 < len(args):
                out.extend(["--sweep-in", args[i + 1]])
                i += 2
                continue
            if a == "--resolved-plot" and i + 1 < len(args):
                out.extend(["--plot", args[i + 1]])
                i += 2
                continue
            out.append(a)
            i += 1
        return out
    if "--competitor-sweep-only" in args:
        out = ["sweep"]
        skip = {"--competitor-sweep-only"}
        i = 0
        while i < len(args):
            a = args[i]
            if a in skip:
                i += 1
                continue
            out.append(a)
            i += 1
        return out
    if "--track-competitor-branches" in args:
        out = ["track"]
        i = 0
        while i < len(args):
            a = args[i]
            if a == "--track-competitor-branches":
                i += 1
                continue
            if a in ("--in",) and i + 1 < len(args):
                out.extend(["--sweep-in", args[i + 1]])
                i += 2
                continue
            if a == "--track-plot" and i + 1 < len(args):
                out.extend(["--plot", args[i + 1]])
                i += 2
                continue
            out.append(a)
            i += 1
        return out
    return ["continue", *args]


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    args.func(args)


if __name__ == "__main__":
    main()
