#!/usr/bin/env python3
"""BM24 annealed-vs-typical exponent gap audit.

This module backs the three public entrypoints:

* ``train.py`` freezes LR-QAOA angles to JSON.
* ``evaluate.py`` reads frozen angles and writes exact per-instance rows.
* ``analyze.py`` turns those rows into red/green/blue exponent and gap plots.

The row-level contract is intentionally simple:
``seed,n,p,objective,train_n,X,p_succ`` where
``X = -(1/n) log2(p_succ)``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "phasecraft-mpl"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_PHASECRAFT = Path(__file__).resolve().parents[2]
_REPO = _PHASECRAFT.parent
_LR = Path(__file__).resolve().parent
for _p in (_REPO, _PHASECRAFT, _LR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from phasecraft.experiments.lr_scaling.train_lr_fixed_n import (  # noqa: E402
    MODE_RNG_OFFSET,
    generate_training_instances,
    train_angles_fixed_n,
)
from phasecraft.lib.paths import bm24_runs_dir  # noqa: E402
from phasecraft.lib.sim.bm24_qaoa_sim import (  # noqa: E402
    build_h_diagonal,
    generate_random_clause,
    make_lr_angles,
    per_instance_success_probability,
    run_qaoa,
)
from phasecraft.lib.sim.bm24_run_io import (  # noqa: E402
    make_run_stem,
    resolve_bm24_run_output_paths,
)

try:  # Optional CUDA/CuPy backend; CPU path remains exact.
    from phasecraft.experiments.lr_scaling.gpu.backend import (  # type: ignore
        detect_device,
        success_probs_for_instances,
    )
except Exception:  # pragma: no cover - optional dependency
    detect_device = None  # type: ignore[assignment]
    success_probs_for_instances = None  # type: ignore[assignment]

LN2 = float(np.log(2.0))
OBJECTIVE_TO_MODE = {
    "mean_p": "bm24_mean_p_fixed_n",
    "median_rt": "median_runtime_fixed_n",
}
MODE_TO_OBJECTIVE = {v: k for k, v in OBJECTIVE_TO_MODE.items()}
REQUIRED_ROW_COLUMNS = ["seed", "n", "p", "objective", "train_n", "X", "p_succ"]


def _parse_ints(spec: str) -> List[int]:
    return [int(x) for x in str(spec).split(",") if x.strip()]


def _parse_objectives(spec: str) -> List[str]:
    out = [x.strip() for x in str(spec).split(",") if x.strip()]
    bad = [x for x in out if x not in OBJECTIVE_TO_MODE]
    if bad:
        raise ValueError(f"unknown objective(s) {bad}; use {sorted(OBJECTIVE_TO_MODE)}")
    return out


def _trace_blocks(payload: Mapping[str, Any]) -> Dict[str, List[dict]]:
    if payload.get("traces_by_mode"):
        return {str(k): list(v) for k, v in payload["traces_by_mode"].items()}
    out: Dict[str, List[dict]] = {}
    for mode in OBJECTIVE_TO_MODE.values():
        key = f"trace_{mode}"
        if payload.get(key):
            out[mode] = list(payload[key])
    return out


def _angle_id(objective: str, train_n: int, depth: int) -> str:
    return f"{objective}|train_n={int(train_n)}|p={int(depth)}"


def _load_existing_angles(paths: Sequence[Path]) -> Dict[Tuple[str, int, int], dict]:
    frozen: Dict[Tuple[str, int, int], dict] = {}
    for path in paths:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        cfg = payload.get("config", {})
        train_n = int(cfg.get("train_n", payload.get("train_n", -1)))
        beta_schedule = str(cfg.get("lr_beta_schedule", "decreasing"))
        for mode, rows in _trace_blocks(payload).items():
            objective = MODE_TO_OBJECTIVE.get(mode)
            if objective is None:
                continue
            for row in rows:
                if "delta_gamma" not in row or "delta_beta" not in row:
                    continue
                depth = int(row["depth"])
                dg = float(row["delta_gamma"])
                db = float(row["delta_beta"])
                betas, gammas = make_lr_angles(
                    dg,
                    db,
                    depth,
                    beta_schedule=beta_schedule,
                    angle_convention="bm24",
                )
                frozen[(objective, train_n, depth)] = {
                    "angle_id": _angle_id(objective, train_n, depth),
                    "objective": objective,
                    "training_mode": mode,
                    "train_n": train_n,
                    "p": depth,
                    "delta_gamma": dg,
                    "delta_beta": db,
                    "betas": betas.tolist(),
                    "gammas": gammas.tolist(),
                    "beta_schedule": beta_schedule,
                    "source": str(Path(path).resolve()),
                    "source_kind": "run_json",
                }
    return frozen


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, allow_nan=True), encoding="utf-8")
    tmp.replace(path)


def main_train(argv: Optional[Sequence[str]] = None) -> None:
    p = argparse.ArgumentParser(description="Freeze BM24 LR-QAOA audit angles.")
    p.add_argument("--objectives", default="mean_p,median_rt")
    p.add_argument("--train-ns", default="12,16")
    p.add_argument("--depths", default="1,2,5,10,20,50")
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--r", type=float, default=176.54)
    p.add_argument("--seed", type=int, default=27, help="training seed")
    p.add_argument("--train-size", type=int, default=100)
    p.add_argument("--cobyla-maxiter", type=int, default=160)
    p.add_argument("--cobyla-restarts", type=int, default=8)
    p.add_argument("--cobyla-perturb-scale", type=float, default=0.2)
    p.add_argument("--grid-top-k", type=int, default=5)
    p.add_argument("--skip-grid", action="store_true")
    p.add_argument("--lr-beta-schedule", default="decreasing")
    p.add_argument("--from-run-json", type=Path, action="append", default=[])
    p.add_argument(
        "--skip-train-missing",
        action="store_true",
        help="Only freeze angles found in --from-run-json; do not optimize missing cells.",
    )
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--output-stem", default=None)
    args = p.parse_args(argv)

    objectives = _parse_objectives(args.objectives)
    train_ns = _parse_ints(args.train_ns)
    depths = _parse_ints(args.depths)

    frozen = _load_existing_angles(args.from_run_json)
    angle_rows: List[dict] = []
    missing: List[Tuple[str, int, int]] = []
    for objective in objectives:
        for train_n in train_ns:
            for depth in depths:
                key = (objective, int(train_n), int(depth))
                if key in frozen:
                    angle_rows.append(frozen[key])
                else:
                    missing.append(key)

    if missing and args.skip_train_missing:
        print(f"Skipping {len(missing)} missing angle cells (requested).")
    elif missing:
        by_train_n: Dict[int, List[np.ndarray]] = {}
        warm_by_arm: Dict[Tuple[str, int], Tuple[float, float]] = {}
        for objective, train_n, depth in missing:
            mode = OBJECTIVE_TO_MODE[objective]
            if train_n not in by_train_n:
                print(f"Building training set train_n={train_n}, size={args.train_size}...")
                by_train_n[train_n] = generate_training_instances(
                    train_n=train_n,
                    k=args.k,
                    r=args.r,
                    train_size=args.train_size,
                    base_seed=args.seed,
                )
            warm = warm_by_arm.get((objective, train_n))
            rng = np.random.default_rng(
                int(args.seed)
                + 1000
                + int(depth)
                + int(MODE_RNG_OFFSET.get(mode, 0))
                + 1_000_000 * int(train_n)
            )
            print(f"Training objective={objective} train_n={train_n} p={depth}...")
            _, diag = train_angles_fixed_n(
                training_mode=mode,
                train_n=train_n,
                depth=depth,
                instances=by_train_n[train_n],
                initial_angles=warm,
                beta_schedule=args.lr_beta_schedule,
                skip_grid=bool(args.skip_grid),
                skip_grid_if_warm_start=True,
                cobyla_maxiter=args.cobyla_maxiter,
                cobyla_restarts=args.cobyla_restarts,
                cobyla_perturb_scale=args.cobyla_perturb_scale,
                grid_top_k=args.grid_top_k,
                rng=rng,
            )
            dg, db = map(float, diag["best_deltas"])
            warm_by_arm[(objective, train_n)] = (dg, db)
            betas, gammas = make_lr_angles(
                dg,
                db,
                depth,
                beta_schedule=args.lr_beta_schedule,
                angle_convention="bm24",
            )
            angle_rows.append(
                {
                    "angle_id": _angle_id(objective, train_n, depth),
                    "objective": objective,
                    "training_mode": mode,
                    "train_n": int(train_n),
                    "p": int(depth),
                    "delta_gamma": dg,
                    "delta_beta": db,
                    "betas": betas.tolist(),
                    "gammas": gammas.tolist(),
                    "beta_schedule": args.lr_beta_schedule,
                    "source": "train.py",
                    "source_kind": "trained",
                    "diagnostics": diag,
                }
            )

    angle_rows = sorted(
        angle_rows,
        key=lambda r: (int(r["train_n"]), str(r["objective"]), int(r["p"])),
    )
    out_path: Path
    if args.output is not None:
        out_path = Path(args.output).expanduser().resolve()
        run_dir = out_path.parent
    else:
        out_dir = args.output_dir or (bm24_runs_dir() / "bm24_gap_audit")
        stem = args.output_stem or make_run_stem("bm24-gap-angles")
        paths = resolve_bm24_run_output_paths(out_dir, stem)
        run_dir = paths["run_dir"]
        out_path = run_dir / f"{stem}-angles.json"

    payload = {
        "schema_version": 1,
        "kind": "bm24_gap_frozen_angles",
        "created_at_unix": time.time(),
        "config": {
            "k": int(args.k),
            "r": float(args.r),
            "train_seed": int(args.seed),
            "train_size": int(args.train_size),
            "objectives": objectives,
            "train_ns": train_ns,
            "depths": depths,
            "lr_beta_schedule": args.lr_beta_schedule,
        },
        "run_dir": str(run_dir),
        "angles": angle_rows,
        "missing": [
            {"objective": o, "train_n": int(n), "p": int(d)}
            for o, n, d in missing
            if args.skip_train_missing
        ],
    }
    _write_json_atomic(out_path, payload)
    print(f"Wrote frozen angles: {out_path}")
    print(f"Angle cells: {len(angle_rows)}")


@dataclass
class EvalInstance:
    seed: int
    index: int
    trial: int
    m: int
    sat: bool
    h_diag: np.ndarray


def _sample_m(rng: np.random.Generator, n: int, r: float, mode: str) -> int:
    m = int(rng.poisson(float(r) * int(n)))
    if mode == "notebook":
        return max(1, m)
    if mode == "bm24":
        return m
    raise ValueError("--m-sampling must be notebook or bm24")


def build_eval_instances(
    *,
    n: int,
    k: int,
    r: float,
    count: int,
    base_seed: int,
    require_sat: bool,
    m_sampling: str,
    max_trials: int,
) -> List[EvalInstance]:
    out: List[EvalInstance] = []
    accepted = 0
    trial = 0
    while accepted < int(count):
        if trial >= int(max_trials):
            raise RuntimeError(f"too many rejected eval formulas at n={n}")
        ss = np.random.SeedSequence([int(base_seed), int(n), int(accepted), int(trial)])
        seed = int(ss.generate_state(1, dtype=np.uint32)[0])
        rng = np.random.default_rng(seed)
        m = _sample_m(rng, n, r, m_sampling)
        clauses = [generate_random_clause(n, k, rng) for _ in range(m)]
        h_diag = build_h_diagonal(clauses, n)
        sat = bool(np.any(h_diag == 0))
        if require_sat and not sat:
            trial += 1
            continue
        out.append(EvalInstance(seed=seed, index=accepted, trial=trial, m=int(m), sat=sat, h_diag=h_diag))
        accepted += 1
        trial += 1
    return out


def _seed_id(parts: Sequence[int]) -> int:
    ss = np.random.SeedSequence([int(x) for x in parts])
    return int(ss.generate_state(1, dtype=np.uint32)[0])


def reconstruct_train_seed_ids(
    *,
    train_ns: Sequence[int],
    train_size: int,
    train_seed: int,
    k: int,
    r: float,
    m_sampling: str,
) -> Dict[int, List[int]]:
    """Reconstruct accepted training RNG ids used by generate_training_instances."""
    out: Dict[int, List[int]] = {}
    for train_n in train_ns:
        ids: List[int] = []
        accepted = 0
        trial = 0
        while accepted < int(train_size):
            seed_id = _seed_id([int(train_seed), int(train_n), int(accepted), int(trial)])
            rng = np.random.default_rng(np.random.SeedSequence([int(train_seed), int(train_n), int(accepted), int(trial)]))
            m = _sample_m(rng, int(train_n), float(r), m_sampling)
            clauses = [generate_random_clause(int(train_n), int(k), rng) for _ in range(m)]
            h_diag = build_h_diagonal(clauses, int(train_n))
            if np.any(h_diag == 0):
                ids.append(seed_id)
                accepted += 1
            trial += 1
            if trial > 1_000_000:
                raise RuntimeError(f"too many train reconstruction trials at n={train_n}")
        out[int(train_n)] = ids
    return out


def build_eval_instances_with_audit(
    *,
    n: int,
    k: int,
    r: float,
    count: int,
    base_seed: int,
    require_sat: bool,
    m_sampling: str,
    max_trials: int,
) -> Tuple[List[EvalInstance], dict]:
    """Build paired eval instances and retain candidate-pool diagnostics."""
    out: List[EvalInstance] = []
    candidate_ms: List[int] = []
    candidate_sat: List[bool] = []
    accepted = 0
    trial = 0
    while accepted < int(count):
        if trial >= int(max_trials):
            raise RuntimeError(f"too many rejected eval formulas at n={n}")
        seed_id = _seed_id([int(base_seed), int(n), int(accepted), int(trial)])
        rng = np.random.default_rng(np.random.SeedSequence([int(base_seed), int(n), int(accepted), int(trial)]))
        m = _sample_m(rng, n, r, m_sampling)
        clauses = [generate_random_clause(n, k, rng) for _ in range(m)]
        h_diag = build_h_diagonal(clauses, n)
        sat = bool(np.any(h_diag == 0))
        candidate_ms.append(int(m))
        candidate_sat.append(sat)
        if require_sat and not sat:
            trial += 1
            continue
        out.append(EvalInstance(seed=seed_id, index=accepted, trial=trial, m=int(m), sat=sat, h_diag=h_diag))
        accepted += 1
        trial += 1
    ms = np.asarray(candidate_ms, dtype=float)
    sat_arr = np.asarray(candidate_sat, dtype=bool)
    accepted_ms = np.asarray([inst.m for inst in out], dtype=float)
    audit = {
        "n": int(n),
        "requested_N": int(count),
        "accepted_N": int(len(out)),
        "candidate_count": int(len(candidate_ms)),
        "eval_seed": int(base_seed),
        "mean_alpha_candidates": float(np.mean(ms / float(n))) if len(ms) else float("nan"),
        "std_alpha_candidates": float(np.std(ms / float(n), ddof=1)) if len(ms) > 1 else float("nan"),
        "mean_alpha_accepted": float(np.mean(accepted_ms / float(n))) if len(accepted_ms) else float("nan"),
        "sat_fraction_candidates": float(np.mean(sat_arr)) if len(sat_arr) else float("nan"),
        "sat_fraction_accepted": float(np.mean([inst.sat for inst in out])) if out else float("nan"),
        "seed_min": int(min(inst.seed for inst in out)) if out else None,
        "seed_max": int(max(inst.seed for inst in out)) if out else None,
    }
    return out, audit


def seed_audit_pass(
    audits: Sequence[Mapping[str, Any]],
    *,
    r: float,
    train_seed_overlap: int,
    entropy_roots_disjoint: bool,
    alpha_z_tol: float,
    sat_z_tol: float,
) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    if not entropy_roots_disjoint:
        reasons.append("train/eval SeedSequence entropy roots are not disjoint")
    if int(train_seed_overlap) != 0:
        reasons.append(f"train/eval seed overlap={train_seed_overlap}")
    by_n = {int(a["n"]): a for a in audits}
    if 15 in by_n:
        a15 = by_n[15]
        alpha = float(a15["mean_alpha_candidates"])
        n15 = int(a15["n"])
        alpha_se = math.sqrt(float(r) / float(n15) / max(1, int(a15["candidate_count"])))
        alpha_z = abs(alpha - float(r)) / max(alpha_se, 1e-12)
        a15["alpha_z_vs_r"] = float(alpha_z)
        if alpha_z > float(alpha_z_tol):
            reasons.append(f"n=15 alpha z={alpha_z:.2f} > {alpha_z_tol:g}")
        others = [a for n, a in by_n.items() if n != 15]
        if others:
            p15 = float(a15["sat_fraction_candidates"])
            pbar = float(np.mean([float(a["sat_fraction_candidates"]) for a in others]))
            se = math.sqrt(max(pbar * (1.0 - pbar), 1e-12) / max(1, int(a15["candidate_count"])))
            sat_z = abs(p15 - pbar) / max(se, 1e-12)
            a15["sat_fraction_z_vs_other_n"] = float(sat_z)
            if sat_z > float(sat_z_tol):
                reasons.append(f"n=15 SAT-fraction z={sat_z:.2f} > {sat_z_tol:g}")
    else:
        reasons.append("n=15 missing from audit")
    return (len(reasons) == 0), reasons


def _select_angles(
    angle_rows: Sequence[Mapping[str, Any]],
    objectives: Optional[Sequence[str]],
    train_ns: Optional[Sequence[int]],
    depths: Optional[Sequence[int]],
) -> List[Mapping[str, Any]]:
    out = []
    obj_set = set(objectives or [])
    n_set = {int(x) for x in (train_ns or [])}
    d_set = {int(x) for x in (depths or [])}
    for row in angle_rows:
        if obj_set and str(row["objective"]) not in obj_set:
            continue
        if n_set and int(row["train_n"]) not in n_set:
            continue
        if d_set and int(row["p"]) not in d_set:
            continue
        out.append(row)
    return sorted(out, key=lambda r: (int(r["train_n"]), str(r["objective"]), int(r["p"])))


def main_evaluate(argv: Optional[Sequence[str]] = None) -> None:
    p = argparse.ArgumentParser(description="Evaluate frozen BM24 audit angles.")
    p.add_argument("--angles", type=Path, required=True)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--n-values", default=None)
    p.add_argument("--n-min", type=int, default=12)
    p.add_argument("--n-max", type=int, default=16)
    p.add_argument("--N", "--num-instances", dest="num_instances", type=int, default=10_000)
    p.add_argument("--eval-seed", type=int, default=100_027)
    p.add_argument("--objectives", default=None)
    p.add_argument("--train-ns", default=None)
    p.add_argument("--depths", default=None)
    p.add_argument("--sat-filter", dest="sat_filter", action="store_true", default=True)
    p.add_argument("--unconditional", dest="sat_filter", action="store_false")
    p.add_argument("--m-sampling", choices=["notebook", "bm24"], default="notebook")
    p.add_argument("--eps", type=float, default=1e-300)
    p.add_argument("--max-trials-per-n", type=int, default=1_000_000)
    p.add_argument("--audit-only", action="store_true", help="Run seed/alpha/SAT audit and stop before QAOA simulation.")
    p.add_argument("--audit-output", type=Path, default=None)
    p.add_argument("--train-seed", type=int, default=None)
    p.add_argument("--train-size", type=int, default=None)
    p.add_argument("--alpha-z-tol", type=float, default=5.0)
    p.add_argument("--sat-z-tol", type=float, default=5.0)
    p.add_argument("--gpu-batch-size", type=int, default=8)
    p.add_argument("--force-cpu", action="store_true")
    p.add_argument("--require-gpu", action="store_true", help="Fail instead of falling back to CPU.")
    p.add_argument("--batch-on-cpu", action="store_true", help="Use the batched backend even when CUDA is unavailable.")
    args = p.parse_args(argv)

    angle_payload = json.loads(Path(args.angles).read_text(encoding="utf-8"))
    cfg = angle_payload.get("config", {})
    k = int(cfg.get("k", 8))
    r = float(cfg.get("r", 176.54))
    n_values = _parse_ints(args.n_values) if args.n_values else list(range(args.n_min, args.n_max + 1))
    objectives = _parse_objectives(args.objectives) if args.objectives else None
    train_ns = _parse_ints(args.train_ns) if args.train_ns else None
    depths = _parse_ints(args.depths) if args.depths else None
    angles = _select_angles(angle_payload["angles"], objectives, train_ns, depths)
    if not angles:
        raise SystemExit("No frozen angle rows matched the requested filters.")
    train_ns_for_audit = sorted({int(a["train_n"]) for a in angles})
    train_seed = int(args.train_seed if args.train_seed is not None else cfg.get("train_seed", cfg.get("seed", 27)))
    train_size = int(args.train_size if args.train_size is not None else cfg.get("train_size", 100))

    out_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else Path(args.angles).with_name(f"{Path(args.angles).stem}-eval_rows.csv")
    )
    meta_path = out_path.with_suffix(".meta.json")
    audit_path = (
        Path(args.audit_output).expanduser().resolve()
        if args.audit_output
        else out_path.with_suffix(".seed_audit.json")
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.parent.mkdir(parents=True, exist_ok=True)

    device = None
    device_summary = "CPU serial"
    use_batched = False
    if detect_device is not None and success_probs_for_instances is not None:
        device = detect_device(prefer="numpy" if args.force_cpu else None)
        device_summary = device.summary()
        use_batched = bool(device.cuda_available or args.batch_on_cpu)
    if args.require_gpu and not (device is not None and getattr(device, "cuda_available", False)):
        raise SystemExit(f"--require-gpu was set, but backend is {device_summary}")
    print(f"Backend: {device_summary}; batched={use_batched}; batch_size={args.gpu_batch_size}")

    entropy_roots_disjoint = int(train_seed) != int(args.eval_seed)

    fieldnames = REQUIRED_ROW_COLUMNS + [
        "angle_id",
        "eval_index",
        "trial",
        "delta_gamma",
        "delta_beta",
    ]
    total = 0
    audits: List[dict] = []
    eval_seed_ids: set[int] = set()
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for n in n_values:
            print(f"Building paired eval set n={n}, N={args.num_instances}...")
            instances, audit = build_eval_instances_with_audit(
                n=n,
                k=k,
                r=r,
                count=args.num_instances,
                base_seed=args.eval_seed,
                require_sat=bool(args.sat_filter),
                m_sampling=args.m_sampling,
                max_trials=args.max_trials_per_n,
            )
            eval_seed_ids.update(inst.seed for inst in instances)
            audits.append(audit)
            print(
                f"  audit n={n}: alpha={audit['mean_alpha_candidates']:.4f}, "
                f"SAT={audit['sat_fraction_candidates']:.4f}, candidates={audit['candidate_count']}"
            )
            overlap = 0 if entropy_roots_disjoint else len(eval_seed_ids)
            passed, reasons = seed_audit_pass(
                audits,
                r=r,
                train_seed_overlap=overlap,
                entropy_roots_disjoint=entropy_roots_disjoint,
                alpha_z_tol=args.alpha_z_tol,
                sat_z_tol=args.sat_z_tol,
            )
            audit_payload = {
                "schema_version": 1,
                "kind": "bm24_gap_seed_audit",
                "angles": str(Path(args.angles).resolve()),
                "eval_seed": int(args.eval_seed),
                "train_seed": int(train_seed),
                "train_size": int(train_size),
                "train_ns": train_ns_for_audit,
                "N": int(args.num_instances),
                "sat_filter": bool(args.sat_filter),
                "m_sampling": args.m_sampling,
                "train_eval_entropy_roots_disjoint": bool(entropy_roots_disjoint),
                "train_eval_seed_overlap": int(overlap),
                "passed": bool(passed),
                "fail_reasons": reasons,
                "per_n": audits,
            }
            _write_json_atomic(audit_path, audit_payload)
            if not passed and 15 in [int(a["n"]) for a in audits]:
                raise SystemExit(f"Seed audit failed before simulation: {reasons}")
            if args.audit_only:
                continue
            for angle in angles:
                betas = np.asarray(angle["betas"], dtype=float)
                gammas = np.asarray(angle["gammas"], dtype=float)
                depth = int(angle["p"])
                objective = str(angle["objective"])
                train_n = int(angle["train_n"])
                print(f"  eval objective={objective} train_n={train_n} p={depth}")
                if use_batched and success_probs_for_instances is not None:
                    probs = success_probs_for_instances(
                        [inst.h_diag for inst in instances],
                        int(n),
                        betas,
                        gammas,
                        device=device,
                        batch_size=int(args.gpu_batch_size),
                    )
                else:
                    probs = []
                    for inst in instances:
                        psi = run_qaoa(inst.h_diag, betas, gammas, int(n))
                        probs.append(float(per_instance_success_probability(psi, inst.h_diag)))
                    probs = np.asarray(probs, dtype=float)
                for inst, p_succ_raw in zip(instances, probs):
                    p_succ = float(p_succ_raw)
                    clipped = max(p_succ, float(args.eps))
                    x_val = -math.log2(clipped) / float(n)
                    writer.writerow(
                        {
                            "seed": inst.seed,
                            "n": int(n),
                            "p": depth,
                            "objective": objective,
                            "train_n": train_n,
                            "X": f"{x_val:.17g}",
                            "p_succ": f"{p_succ:.17g}",
                            "angle_id": str(angle.get("angle_id", _angle_id(objective, train_n, depth))),
                            "eval_index": inst.index,
                            "trial": inst.trial,
                            "delta_gamma": f"{float(angle['delta_gamma']):.17g}",
                            "delta_beta": f"{float(angle['delta_beta']):.17g}",
                        }
                    )
                    total += 1
    final_overlap = 0 if entropy_roots_disjoint else len(eval_seed_ids)
    passed, reasons = seed_audit_pass(
        audits,
        r=r,
        train_seed_overlap=final_overlap,
        entropy_roots_disjoint=entropy_roots_disjoint,
        alpha_z_tol=args.alpha_z_tol,
        sat_z_tol=args.sat_z_tol,
    )
    audit_payload = {
        "schema_version": 1,
        "kind": "bm24_gap_seed_audit",
        "angles": str(Path(args.angles).resolve()),
        "eval_seed": int(args.eval_seed),
        "train_seed": int(train_seed),
        "train_size": int(train_size),
        "train_ns": train_ns_for_audit,
        "N": int(args.num_instances),
        "sat_filter": bool(args.sat_filter),
        "m_sampling": args.m_sampling,
        "train_eval_entropy_roots_disjoint": bool(entropy_roots_disjoint),
        "train_eval_seed_overlap": int(final_overlap),
        "passed": bool(passed),
        "fail_reasons": reasons,
        "per_n": audits,
    }
    _write_json_atomic(audit_path, audit_payload)
    if args.audit_only:
        print(f"Wrote seed audit: {audit_path}")
        if not passed:
            raise SystemExit(f"Seed audit failed: {reasons}")
        print("Seed audit passed; stopping before QAOA simulation (--audit-only).")
        return
    _write_json_atomic(
        meta_path,
        {
            "schema_version": 1,
            "kind": "bm24_gap_eval_rows",
            "angles": str(Path(args.angles).resolve()),
            "output": str(out_path),
            "k": k,
            "r": r,
            "n_values": n_values,
            "N": int(args.num_instances),
            "eval_seed": int(args.eval_seed),
            "sat_filter": bool(args.sat_filter),
            "m_sampling": args.m_sampling,
            "angle_count": len(angles),
            "row_count": total,
            "required_columns": REQUIRED_ROW_COLUMNS,
            "seed_audit": str(audit_path),
            "backend": device_summary,
            "batched": bool(use_batched),
            "gpu_batch_size": int(args.gpu_batch_size),
        },
    )
    print(f"Wrote rows: {out_path}")
    print(f"Wrote metadata: {meta_path}")
    print(f"Wrote seed audit: {audit_path}")


def _read_rows(paths: Sequence[Path]) -> List[dict]:
    rows: List[dict] = []
    for path in paths:
        with Path(path).open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            missing = [c for c in REQUIRED_ROW_COLUMNS if c not in (reader.fieldnames or [])]
            if missing:
                raise ValueError(f"{path} is missing required columns: {missing}")
            for r in reader:
                p_succ = float(r["p_succ"])
                x_val = float(r["X"])
                if not (np.isfinite(x_val) and np.isfinite(p_succ)):
                    continue
                rows.append(
                    {
                        "seed": int(float(r["seed"])),
                        "n": int(r["n"]),
                        "p": int(r["p"]),
                        "objective": str(r["objective"]),
                        "train_n": int(r["train_n"]),
                        "X": x_val,
                        "p_succ": p_succ,
                    }
                )
    return rows


def _slope_log2(ns: Sequence[int], ys: Sequence[float]) -> float:
    n_arr = np.asarray(ns, dtype=float)
    y_arr = np.asarray(ys, dtype=float)
    mask = np.isfinite(y_arr) & (y_arr > 0)
    if int(mask.sum()) < 2:
        return float("nan")
    x = n_arr[mask]
    z = np.log2(y_arr[mask])
    return float(np.polyfit(x, z, 1)[0])


def _variance(x: np.ndarray) -> float:
    return float(np.var(x, ddof=1)) if len(x) >= 2 else float("nan")


def _trimmed(x: np.ndarray, trim: float) -> np.ndarray:
    if len(x) == 0 or trim <= 0:
        return x
    lo, hi = np.quantile(x, [trim, 1.0 - trim])
    return x[(x >= lo) & (x <= hi)]


def _skewness(x: np.ndarray) -> float:
    if len(x) < 3:
        return float("nan")
    mu = float(np.mean(x))
    sd = float(np.std(x, ddof=1))
    if sd <= 0:
        return 0.0
    return float(np.mean(((x - mu) / sd) ** 3))


def _third_central_moment(x: np.ndarray) -> float:
    if len(x) < 2:
        return float("nan")
    mu = float(np.mean(x))
    return float(np.mean((x - mu) ** 3))


def _richardson(ns: Sequence[int], vals: Sequence[float]) -> Tuple[float, float]:
    n_arr = np.asarray(ns, dtype=float)
    y_arr = np.asarray(vals, dtype=float)
    mask = np.isfinite(y_arr) & np.isfinite(n_arr) & (n_arr > 0)
    if int(mask.sum()) < 2:
        return float("nan"), float("nan")
    design = np.column_stack([np.ones(int(mask.sum())), 1.0 / n_arr[mask]])
    coeff, *_ = np.linalg.lstsq(design, y_arr[mask], rcond=None)
    return float(coeff[0]), float(coeff[1])


def _band_pair(a: float, b: float) -> Tuple[float, float]:
    vals = [x for x in (a, b) if np.isfinite(x)]
    if not vals:
        return float("nan"), float("nan")
    return float(min(vals)), float(max(vals))


def _group_rows(rows: Sequence[dict]) -> Dict[Tuple[str, int, int], Dict[int, List[dict]]]:
    grouped: Dict[Tuple[str, int, int], Dict[int, List[dict]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[(row["objective"], int(row["train_n"]), int(row["p"]))][int(row["n"])].append(row)
    return grouped


def analyze_rows(rows: Sequence[dict], *, c_cl: float, plateau_rel_tol: float) -> dict:
    out_groups: List[dict] = []
    for (objective, train_n, depth), by_n in sorted(_group_rows(rows).items()):
        per_n = []
        for n, rs in sorted(by_n.items()):
            ps = np.asarray([float(r["p_succ"]) for r in rs], dtype=float)
            xs = np.asarray([float(r["X"]) for r in rs], dtype=float)
            mean_p = float(np.mean(ps))
            median_rt = float(np.median(1.0 / np.maximum(ps, 1e-300)))
            var_x = _variance(xs)
            kappa3_x = _third_central_moment(xs)
            n_var = float(n * var_x) if np.isfinite(var_x) else float("nan")
            rel = math.sqrt(2.0 / max(1, len(xs)))
            saturated = bool(mean_p > 0.1)
            per_n.append(
                {
                    "n": int(n),
                    "N": int(len(xs)),
                    "mean_p_succ": mean_p,
                    "median_p_succ": float(np.median(ps)),
                    "median_runtime": median_rt,
                    "mean_X": float(np.mean(xs)),
                    "median_X": float(np.median(xs)),
                    "var_X": var_x,
                    "kappa3_X": kappa3_x,
                    "n_var_X": n_var,
                    "n_var_X_se": float(abs(n_var) * rel),
                    "n2_kappa3_X": float((n ** 2) * kappa3_x) if np.isfinite(kappa3_x) else float("nan"),
                    "n_var_X_trim05": float(n * _variance(_trimmed(xs, 0.05))),
                    "n_var_X_trim10": float(n * _variance(_trimmed(xs, 0.10))),
                    "skew_X": _skewness(xs),
                    "skew_X_se": float(math.sqrt(6.0 / max(1, len(xs)))),
                    "saturated_mean_p_gt_0p1": saturated,
                    "c_ann_point": float(-math.log2(max(mean_p, 1e-300)) / n),
                    "c_inv_prob_point": float(math.log2(1.0 / max(mean_p, 1e-300)) / n),
                    "c_med_point": float(math.log2(max(median_rt, 1e-300)) / n),
                }
            )
        fit_per_n = [r for r in per_n if not r["saturated_mean_p_gt_0p1"]]
        ns = [r["n"] for r in fit_per_n]
        mean_ps = [r["mean_p_succ"] for r in fit_per_n]
        inv_mean = [1.0 / max(r["mean_p_succ"], 1e-300) for r in fit_per_n]
        med_rt = [r["median_runtime"] for r in fit_per_n]
        nvars = [r["n_var_X"] for r in fit_per_n]
        nvar_bars = [r["n_var_X_se"] for r in fit_per_n if np.isfinite(r["n_var_X_se"])]
        n2_kappa3 = [r["n2_kappa3_X"] for r in fit_per_n]
        skews = [r["skew_X"] for r in fit_per_n]
        c_ann_fit = -_slope_log2(ns, mean_ps)
        c_inv_fit = _slope_log2(ns, inv_mean)
        c_med_fit = _slope_log2(ns, med_rt)
        v_hat, a_hat = _richardson(ns, nvars)
        w_hat, b_hat = _richardson(ns, n2_kappa3)
        skew_lim, skew_a = _richardson(ns, skews)
        skew_se = float(np.mean([r["skew_X_se"] for r in fit_per_n])) if fit_per_n else float("nan")
        c_med_gap = (
            c_ann_fit + 0.5 * LN2 * v_hat
            if np.isfinite(c_ann_fit) and np.isfinite(v_hat)
            else float("nan")
        )
        third_term = -((LN2 ** 2) / 6.0) * w_hat if np.isfinite(w_hat) else float("nan")
        c_med_third = (
            c_med_gap + third_term
            if np.isfinite(c_med_gap) and np.isfinite(third_term)
            else float("nan")
        )
        tail = abs(a_hat / max(ns)) if ns and np.isfinite(a_hat) else float("nan")
        plateau = bool(
            np.isfinite(v_hat)
            and np.isfinite(tail)
            and tail <= float(plateau_rel_tol) * max(abs(v_hat), 1e-12)
        )
        band_low, band_high = _band_pair(c_med_gap, c_med_third)
        out_groups.append(
            {
                "objective": objective,
                "train_n": int(train_n),
                "p": int(depth),
                "per_n": per_n,
                "fits": {
                    "c_ann": c_ann_fit,
                    "c_inv_prob": c_inv_fit,
                    "c_med_fit": c_med_fit,
                    "v_richardson": v_hat,
                    "v_bar": float(np.mean(nvar_bars)) if nvar_bars else float("nan"),
                    "richardson_a": a_hat,
                    "w_n2_kappa3_richardson": w_hat,
                    "richardson_b_kappa3": b_hat,
                    "skew_limit_richardson": skew_lim,
                    "skew_limit_bar": skew_se,
                    "skew_decays_to_zero": bool(
                        np.isfinite(skew_lim)
                        and np.isfinite(skew_se)
                        and abs(skew_lim) <= 2.0 * max(skew_se, 1e-12)
                    ),
                    "gap_from_v": 0.5 * LN2 * v_hat if np.isfinite(v_hat) else float("nan"),
                    "third_cumulant_term": third_term,
                    "c_med_from_gap": c_med_gap,
                    "c_med_with_third": c_med_third,
                    "c_med_band_low": band_low,
                    "c_med_band_high": band_high,
                    "c_med_band_width": abs(c_med_third - c_med_gap) if np.isfinite(c_med_third) and np.isfinite(c_med_gap) else float("nan"),
                    "c_med_fit_minus_c_ann": (
                        c_med_fit - c_ann_fit
                        if np.isfinite(c_med_fit) and np.isfinite(c_ann_fit)
                        else float("nan")
                    ),
                    "c_med_from_gap_minus_c_cl": (
                        c_med_gap - c_cl if np.isfinite(c_med_gap) else float("nan")
                    ),
                    "c_med_with_third_minus_c_cl": (
                        c_med_third - c_cl if np.isfinite(c_med_third) else float("nan")
                    ),
                    "below_classical_gaussian": bool(np.isfinite(c_med_gap) and c_med_gap < c_cl),
                    "below_classical_with_third": bool(np.isfinite(c_med_third) and c_med_third < c_cl),
                    "band_straddles_classical": bool(
                        np.isfinite(c_med_gap)
                        and np.isfinite(c_med_third)
                        and band_low <= c_cl <= band_high
                    ),
                    "n_var_plateau_go": plateau,
                    "plateau_tail_at_nmax": tail,
                    "fit_n_values": ns,
                    "excluded_saturated_n_values": [
                        int(r["n"]) for r in per_n if r["saturated_mean_p_gt_0p1"]
                    ],
                },
            }
        )
    return {"groups": out_groups, "c_cl": float(c_cl)}


def _plot_nvar(summary: Mapping[str, Any], out: Path) -> None:
    groups = summary["groups"]
    keys = sorted({(g["objective"], g["train_n"]) for g in groups})
    if not keys:
        return
    fig, axes = plt.subplots(len(keys), 1, figsize=(8, max(3.5, 3.0 * len(keys))), squeeze=False)
    for ax, (objective, train_n) in zip(axes[:, 0], keys):
        for g in groups:
            if (g["objective"], g["train_n"]) != (objective, train_n):
                continue
            ns = [r["n"] for r in g["per_n"]]
            ys = [r["n_var_X"] for r in g["per_n"]]
            es = [r["n_var_X_se"] for r in g["per_n"]]
            ax.errorbar(ns, ys, yerr=es, marker="o", linewidth=1.5, label=f"p={g['p']}")
        ax.set_title(f"n Var(X_n): {objective}, train_n={train_n}")
        ax.set_xlabel("n")
        ax.set_ylabel("n Var(X_n)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, ncol=3)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def _plot_exponents(summary: Mapping[str, Any], out: Path) -> None:
    groups = summary["groups"]
    keys = sorted({(g["objective"], g["train_n"]) for g in groups})
    if not keys:
        return
    fig, axes = plt.subplots(1, len(keys), figsize=(6.0 * len(keys), 4.5), squeeze=False)
    for ax, (objective, train_n) in zip(axes[0], keys):
        gs = [g for g in groups if (g["objective"], g["train_n"]) == (objective, train_n)]
        ps = [g["p"] for g in gs]
        c_ann = [g["fits"]["c_ann"] for g in gs]
        c_inv = [g["fits"]["c_inv_prob"] for g in gs]
        overlap = np.allclose(c_ann, c_inv, rtol=1e-10, atol=1e-12, equal_nan=True)
        ax.plot(
            ps,
            c_inv,
            "o-",
            color="green",
            linewidth=3.0,
            alpha=0.55,
            label="inverse probability",
            zorder=2,
        )
        ax.plot(
            ps,
            c_ann,
            "o--" if overlap else "o-",
            color="red",
            markerfacecolor="white",
            markeredgewidth=1.4,
            linewidth=1.8,
            label="mean p_succ exponent" + (" (overlaps green)" if overlap else ""),
            zorder=4,
        )
        ax.plot(ps, [g["fits"]["c_med_fit"] for g in gs], "o-", color="blue", label="median runtime fit")
        ax.plot(ps, [g["fits"]["c_med_from_gap"] for g in gs], "--", color="blue", label="c_ann+(ln2/2)v")
        ax.plot(ps, [g["fits"]["c_med_with_third"] for g in gs], ":", color="purple", linewidth=2, label="+ third cumulant")
        ax.axhline(float(summary["c_cl"]), color="black", linestyle="--", linewidth=1.2, label=f"c_cl={summary['c_cl']:.3f}")
        ax.set_title(f"{objective}, train_n={train_n}")
        ax.set_xlabel("QAOA depth p")
        ax.set_ylabel("exponent")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def _plot_verdict(summary: Mapping[str, Any], out: Path) -> None:
    groups = summary["groups"]
    keys = sorted({(g["objective"], g["train_n"]) for g in groups})
    if not keys:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    for objective, train_n in keys:
        gs = [g for g in groups if (g["objective"], g["train_n"]) == (objective, train_n)]
        ps = np.asarray([g["p"] for g in gs], dtype=float)
        lo = np.asarray([g["fits"]["c_med_band_low"] for g in gs], dtype=float)
        hi = np.asarray([g["fits"]["c_med_band_high"] for g in gs], dtype=float)
        mid = np.asarray([g["fits"]["c_med_with_third"] for g in gs], dtype=float)
        mask = np.isfinite(lo) & np.isfinite(hi)
        if np.any(mask):
            ax.fill_between(ps[mask], lo[mask], hi[mask], alpha=0.16)
        ax.plot(
            ps,
            mid,
            marker="o",
            linewidth=1.8,
            label=f"{objective}, train_n={train_n}",
        )
    ax.axhline(float(summary["c_cl"]), color="black", linestyle="--", linewidth=1.2, label=f"c_cl={summary['c_cl']:.3f}")
    ax.set_xlabel("QAOA depth p")
    ax.set_ylabel(r"$c_{\mathrm{typ}}$ with skew band")
    ax.set_title("Typical-exponent verdict: Gaussian vs third-cumulant band")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def main_analyze(argv: Optional[Sequence[str]] = None) -> None:
    p = argparse.ArgumentParser(description="Analyze BM24 red-blue exponent gap rows.")
    p.add_argument("rows", type=Path, nargs="+")
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--c-cl", type=float, default=0.325)
    p.add_argument("--plateau-rel-tol", type=float, default=0.10)
    args = p.parse_args(argv)

    rows = _read_rows(args.rows)
    if not rows:
        raise SystemExit("No finite rows loaded.")
    out_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else Path(args.rows[0]).resolve().parent / "analysis"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = analyze_rows(rows, c_cl=args.c_cl, plateau_rel_tol=args.plateau_rel_tol)
    summary_path = out_dir / "bm24_gap_summary.json"
    _write_json_atomic(summary_path, summary)
    _plot_nvar(summary, out_dir / "nvar_vs_n.png")
    _plot_exponents(summary, out_dir / "exponent_vs_p.png")
    _plot_verdict(summary, out_dir / "typical_verdict.png")

    print(f"Wrote summary: {summary_path}")
    print(f"Wrote plots: {out_dir}")
    print("\nVerdict table (Gaussian and third-cumulant typical exponents):")
    print(
        f"{'objective':>10} {'train_n':>7} {'p':>4} {'c_ann':>9} {'v':>9} "
        f"{'c_gauss':>9} {'c_3rd':>9} {'band':>13} {'plateau':>8}"
    )
    for g in summary["groups"]:
        fit = g["fits"]
        print(
            f"{g['objective']:>10} {g['train_n']:7d} {g['p']:4d} "
            f"{fit['c_ann']:9.4f} {fit['v_richardson']:9.4f} "
            f"{fit['c_med_from_gap']:9.4f} {fit['c_med_with_third']:9.4f} "
            f"[{fit['c_med_band_low']:.4f},{fit['c_med_band_high']:.4f}] "
            f"{str(fit['n_var_plateau_go']):>8}"
        )
