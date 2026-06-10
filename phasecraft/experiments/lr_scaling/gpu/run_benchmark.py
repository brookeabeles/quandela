#!/usr/bin/env python3
"""
Terminal entrypoint for the LR_QAOA_benchmark.ipynb pipeline (GPU-accelerated QAOA).

Outputs JSON/PNG checkpoints under ``phasecraft/results/bm24_runs/MM-DD/runN/``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
from scipy.stats import linregress
from tqdm import tqdm

from ._bootstrap import ensure_paths, results_bm24_dir
from .backend import detect_device, success_probs_for_instances
from .classical import clauses_to_numba_arrays, evaluate_classical_once
from .train_lr_fixed_n import (
    DEFAULT_EVAL_RUNTIME_REGRESSION_FACTOR,
    DEFAULT_EVAL_TRAIN_RETRIES,
    configure_gpu,
    generate_training_instances,
    run_train_eval_with_retries,
    train_angles_fixed_n,
)

ensure_paths()

from phasecraft.lib.sim.bm24_qaoa_sim import (  # noqa: E402
    build_h_diagonal,
    generate_random_clause,
    make_lr_angles,
)
from phasecraft.lib.sim.bm24_run_io import (  # noqa: E402
    normalize_eval_aggregation,
    normalize_eval_axis,
    plot_scaling_vs_depth,
    scaling_plot_headline,
)
from notebook_run_checkpoint import (  # noqa: E402
    build_payload,
    load_or_start_run,
    save_setup_checkpoint,
    write_checkpoint,
)

LN2 = float(np.log(2.0))


def _parse_int_list(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def _parse_float_pair(text: str) -> tuple[float, float]:
    parts = [p.strip() for p in str(text).split(",") if p.strip()]
    if len(parts) != 2:
        raise ValueError(f"expected lo,hi pair, got {text!r}")
    return float(parts[0]), float(parts[1])


def apply_exploratory_preset(args: argparse.Namespace) -> None:
    """Robust per-depth angle search: full grid, no protocol shortcuts."""
    args.skip_grid = False
    args.skip_grid_if_warm_start = False
    args.depth_warm_start = False
    args.anti_regression = False
    args.collapse_guard = False
    args.eval_train_retries = 0
    args.check_eval_regression = False
    args.fallback_to_previous_angles = False
    if args.dg_bounds is None:
        args.dg_bounds = "-2,2"
    if args.db_bounds is None:
        args.db_bounds = "0.1,4"
    if args.cobyla_restarts < 8:
        args.cobyla_restarts = 8
    if args.grid_top_k < 11:
        args.grid_top_k = 11


def generate_benchmark_dataset_cached(
    n_values: list[int],
    k: int,
    r: float,
    test_size: int,
    base_seed: int,
) -> Dict[int, List[dict]]:
    dataset: Dict[int, List[dict]] = {}
    for n in n_values:
        n = int(n)
        accepted = 0
        trial = 0
        instances = []
        pbar = tqdm(total=test_size, desc=f"dataset n={n}", leave=False)
        while accepted < test_size:
            if trial > 200_000:
                raise RuntimeError(f"too many rejections at n={n}")
            ss = np.random.SeedSequence([int(base_seed), n, accepted, trial])
            rng = np.random.default_rng(ss)
            m_clauses = max(1, int(rng.poisson(float(r) * n)))
            clauses = [generate_random_clause(n, k, rng) for _ in range(m_clauses)]
            h_diag = build_h_diagonal(clauses, n)
            if not np.any(h_diag == 0):
                trial += 1
                continue
            c_vars, c_signs = clauses_to_numba_arrays(clauses, k)
            instances.append({
                "clauses": clauses,
                "h_diag": h_diag,
                "c_vars": c_vars,
                "c_signs": c_signs,
            })
            accepted += 1
            trial += 1
            pbar.update(1)
        pbar.close()
        dataset[n] = instances
    return dataset


def fit_log2_slope(n_values, y_values) -> float:
    n_arr = np.asarray(n_values, dtype=float)
    y_arr = np.asarray(y_values, dtype=float)
    mask = np.isfinite(y_arr) & (y_arr > 0)
    if mask.sum() < 2:
        return float("nan")
    slope_nat = linregress(n_arr[mask], np.log(y_arr[mask])).slope
    return float(slope_nat / LN2)


def evaluate_lr_qaoa_depth(
    dataset,
    n_values,
    betas,
    gammas,
    *,
    eps: float = 1e-300,
    device=None,
    batch_size: int | None = None,
) -> dict:
    med_rt = {}
    for n in n_values:
        insts = [inst["h_diag"] for inst in dataset[int(n)]]
        probs = success_probs_for_instances(
            insts, int(n), betas, gammas, device=device, batch_size=batch_size,
        )
        costs = 1.0 / np.maximum(probs, eps)
        med_rt[int(n)] = float(np.median(costs))
    return med_rt


def build_cfg_from_args(args: argparse.Namespace) -> dict:
    dg_bounds = (
        _parse_float_pair(args.dg_bounds)
        if args.dg_bounds is not None
        else None
    )
    db_bounds = (
        _parse_float_pair(args.db_bounds)
        if args.db_bounds is not None
        else None
    )
    return {
        "k": args.k,
        "r": args.r,
        "seed": args.seed,
        "depths": _parse_int_list(args.depths),
        "lr_beta_schedule": args.lr_beta_schedule,
        "training_mode": args.training_mode,
        "train_n": args.train_n,
        "train_size": args.train_size,
        "n_min": args.n_min,
        "n_max": args.n_max,
        "test_size": args.test_size,
        "skip_grid": args.skip_grid,
        "skip_grid_if_warm_start": args.skip_grid_if_warm_start,
        "depth_warm_start": bool(args.depth_warm_start),
        "dg_bounds": dg_bounds,
        "db_bounds": db_bounds,
        "anti_regression": bool(args.anti_regression),
        "collapse_guard": bool(args.collapse_guard),
        "check_eval_regression": bool(args.check_eval_regression),
        "fallback_to_previous_angles": bool(args.fallback_to_previous_angles),
        "exploratory": bool(getattr(args, "exploratory", False)),
        "cobyla_maxiter": args.cobyla_maxiter,
        "cobyla_restarts": args.cobyla_restarts,
        "cobyla_perturb_scale": args.cobyla_perturb_scale,
        "grid_top_k": args.grid_top_k,
        "walksat_p_noise": args.walksat_p_noise,
        "walksatlm_p_noise": args.walksatlm_p_noise,
        "walksatlm_w1": args.walksatlm_w1,
        "walksatlm_w2": args.walksatlm_w2,
        "max_flips": args.max_flips,
        "eval_axis": normalize_eval_axis(args.eval_axis),
        "eval_aggregation": normalize_eval_aggregation(args.eval_aggregation),
        "annotate_first_win": args.annotate_first_win,
        "eval_train_retries": args.eval_train_retries,
        "eval_runtime_regression_factor": args.eval_runtime_regression_factor,
        "output_dir": Path(args.output_dir) if args.output_dir else results_bm24_dir(),
        "run_stem": args.run_stem,
        "auto_resume": args.auto_resume,
        "recover_from": Path(args.recover_from) if args.recover_from else None,
        "gpu_batch_size": args.gpu_batch_size,
        "force_cpu": args.force_cpu,
    }


def run_pipeline(cfg: dict) -> Path:
    device = detect_device(prefer="numpy" if cfg.get("force_cpu") else None)
    configure_gpu(device=device, batch_size=cfg.get("gpu_batch_size"))
    print(f"Device: {device.summary()}")

    cfg["eval_axis"] = normalize_eval_axis(cfg.get("eval_axis", "runtime"))
    cfg["eval_aggregation"] = normalize_eval_aggregation(cfg.get("eval_aggregation", "median"))
    cfg["output_dir"] = Path(cfg["output_dir"])
    cfg["output_dir"].mkdir(parents=True, exist_ok=True)

    run_stem, run_paths, trace, saved_setup = load_or_start_run(cfg)
    print(f"Run id: {run_stem}")
    print(f"Checkpoint dir: {run_paths['run_dir']}")

    k = int(cfg["k"])
    n_values = list(range(int(cfg["n_min"]), int(cfg["n_max"]) + 1))
    batch_size = cfg.get("gpu_batch_size")

    t0 = time.time()
    print("Building SAT benchmark dataset (cached H_diag)...")
    dataset = generate_benchmark_dataset_cached(
        n_values, k=k, r=float(cfg["r"]), test_size=int(cfg["test_size"]), base_seed=int(cfg["seed"]),
    )

    if saved_setup and "classical" in saved_setup:
        classical = {
            algo: {int(n): float(v) for n, v in per_n.items()}
            for algo, per_n in saved_setup["classical"].items()
        }
        ws_slope = float(saved_setup["ws_slope"])
        lm_slope = float(saved_setup["lm_slope"])
        print("Restored classical baselines from checkpoint")
    else:
        print("Classical solvers (once)...")
        classical = evaluate_classical_once(dataset, cfg)
        ws_slope = fit_log2_slope(n_values, [classical["walksat"][n] for n in n_values])
        lm_slope = fit_log2_slope(n_values, [classical["walksatlm"][n] for n in n_values])

    print(f"  WalkSAT   log2 slope = {ws_slope:.4f}")
    print(f"  WalkSATlm log2 slope = {lm_slope:.4f}")

    print(f"Training instances (mode={cfg['training_mode']!r}, n={cfg['train_n']})...")
    training_instances = generate_training_instances(
        train_n=int(cfg["train_n"]),
        k=k,
        r=float(cfg["r"]),
        train_size=int(cfg["train_size"]),
        base_seed=int(cfg["seed"]),
    )
    saved_setup = {
        "classical": {
            algo: {str(n): float(v) for n, v in per_n.items()}
            for algo, per_n in classical.items()
        },
        "ws_slope": float(ws_slope),
        "lm_slope": float(lm_slope),
        "device": device.summary(),
    }
    setup_json = save_setup_checkpoint(
        run_paths,
        run_stem=run_stem,
        cfg=cfg,
        trace=trace,
        classical=classical,
        ws_slope=ws_slope,
        lm_slope=lm_slope,
    )
    print(f"Setup checkpoint -> {setup_json.name} ({time.time() - t0:.1f}s)")

    depths_to_run = [int(p) for p in cfg["depths"]]
    completed = {int(r["depth"]) for r in trace}
    remaining = [d for d in depths_to_run if d not in completed]
    print(f"Depths: {depths_to_run[0]} … {depths_to_run[-1]} ({len(remaining)} remaining)")

    prev_deltas = None
    prev_med_rt = None
    prev_accepted_deltas = None
    if trace:
        last = trace[-1]
        prev_deltas = (float(last["delta_gamma"]), float(last["delta_beta"]))
        prev_accepted_deltas = prev_deltas
        med = last.get("median_runtime_per_n")
        if med:
            prev_med_rt = {int(kk): float(v) for kk, v in med.items()}

    eval_rt_factor = float(
        cfg.get("eval_runtime_regression_factor", DEFAULT_EVAL_RUNTIME_REGRESSION_FACTOR)
    )
    train_kwargs: dict = {}
    if cfg.get("dg_bounds") is not None:
        train_kwargs["dg_bounds"] = tuple(cfg["dg_bounds"])
    if cfg.get("db_bounds") is not None:
        train_kwargs["db_bounds"] = tuple(cfg["db_bounds"])
    train_kwargs["anti_regression"] = bool(cfg.get("anti_regression", True))
    train_kwargs["collapse_guard"] = bool(cfg.get("collapse_guard", True))
    depth_warm_start = bool(cfg.get("depth_warm_start", True))

    for depth in remaining:
        depth = int(depth)
        t_depth = time.time()
        print(f"\n{'=' * 60}\nDepth p = {depth}\n{'=' * 60}")

        def _train_at_depth(retry_index: int):
            warm = None
            if depth_warm_start:
                warm = (
                    tuple(prev_accepted_deltas)
                    if retry_index > 0 and prev_accepted_deltas is not None
                    else (tuple(prev_deltas) if prev_deltas is not None else None)
                )
            perturb = float(cfg["cobyla_perturb_scale"]) * (1.25 ** int(retry_index))
            train_rng = np.random.default_rng(
                int(cfg["seed"]) + 1000 + depth + 10_000 * int(retry_index)
            )
            _, diag_local = train_angles_fixed_n(
                training_mode=str(cfg["training_mode"]),
                train_n=int(cfg["train_n"]),
                depth=depth,
                instances=training_instances,
                initial_angles=warm,
                beta_schedule=str(cfg["lr_beta_schedule"]),
                skip_grid=bool(cfg["skip_grid"]) and retry_index == 0,
                skip_grid_if_warm_start=bool(cfg.get("skip_grid_if_warm_start", True)) and retry_index == 0,
                cobyla_maxiter=int(cfg["cobyla_maxiter"]),
                cobyla_restarts=int(cfg["cobyla_restarts"]),
                cobyla_perturb_scale=perturb,
                grid_top_k=int(cfg["grid_top_k"]),
                rng=train_rng,
                **train_kwargs,
            )
            dg_l = float(diag_local["best_deltas"][0])
            db_l = float(diag_local["best_deltas"][1])
            print(
                f"  trained dg={dg_l:.6f} db={db_l:.6f} "
                f"mean_p@train_n={diag_local['best_avg_train_p_succ']:.4e}"
            )
            return dg_l, db_l, diag_local

        def _evaluate_at_angles(dg_eval: float, db_eval: float):
            betas_e, gammas_e = make_lr_angles(
                dg_eval, db_eval, depth,
                beta_schedule=str(cfg["lr_beta_schedule"]),
                angle_convention="bm24",
            )
            return evaluate_lr_qaoa_depth(
                dataset, n_values, betas_e, gammas_e,
                device=device, batch_size=batch_size,
            )

        te_result = run_train_eval_with_retries(
            train_at_depth=_train_at_depth,
            evaluate_at_angles=_evaluate_at_angles,
            prev_med_rt=prev_med_rt,
            prev_accepted_deltas=prev_accepted_deltas,
            n_min=int(cfg["n_min"]),
            n_max=int(cfg["n_max"]),
            max_retries=int(cfg.get("eval_train_retries", DEFAULT_EVAL_TRAIN_RETRIES)),
            regression_factor=eval_rt_factor,
            check_eval_regression=bool(cfg.get("check_eval_regression", True)),
            fallback_to_previous_angles=bool(cfg.get("fallback_to_previous_angles", True)),
        )
        dg, db = te_result["dg"], te_result["db"]
        diag = te_result["diag"]
        med_rt = te_result["med_rt"]
        eval_rejected = bool(te_result["eval_rejected"])
        reject_reason = te_result["eval_reject_reason"]

        trained_dg, trained_db = (
            float(diag.get("trained_deltas", diag["best_deltas"])[0]),
            float(diag.get("trained_deltas", diag["best_deltas"])[1]),
        )
        if not eval_rejected:
            prev_accepted_deltas = (dg, db)
            prev_med_rt = dict(med_rt)

        prev_deltas = prev_accepted_deltas
        lr_slope = fit_log2_slope(n_values, [med_rt[n] for n in n_values])
        beats_ws = bool(np.isfinite(lr_slope) and np.isfinite(ws_slope) and lr_slope < ws_slope)
        beats_lm = bool(np.isfinite(lr_slope) and np.isfinite(lm_slope) and lr_slope < lm_slope)

        row = {
            "depth": depth,
            "delta_gamma": float(dg),
            "delta_beta": float(db),
            "trained_delta_gamma": float(trained_dg),
            "trained_delta_beta": float(trained_db),
            "training_mode": str(cfg["training_mode"]),
            "training_objective": diag.get("objective"),
            "train_rejected": bool(diag.get("train_rejected", False)),
            "train_reject_reason": diag.get("train_reject_reason"),
            "collapse_guard_applied": bool(diag.get("collapse_guard_applied", False)),
            "eval_rejected": bool(eval_rejected),
            "eval_reject_reason": reject_reason if eval_rejected else None,
            "eval_train_attempts": int(te_result.get("eval_train_attempts", 1)),
            "used_previous_angles": bool(te_result.get("used_previous_angles", False)),
            "training_failed": bool(
                te_result.get("used_previous_angles") or te_result.get("eval_rejected")
            ),
            "best_train_objective": diag.get("best_train_objective"),
            "best_avg_train_p_succ": diag.get("best_avg_train_p_succ"),
            "lr_log2_slope": lr_slope,
            "walksat_log2_slope": ws_slope,
            "walksatlm_log2_slope": lm_slope,
            "median_runtime_per_n": {str(n): med_rt[n] for n in n_values},
            "lr_beats_walksat_scaling": beats_ws,
            "lr_beats_walksatlm_scaling": beats_lm,
            "elapsed_s": time.time() - t_depth,
            "device": device.summary(),
        }
        trace.append(row)
        print(f"  LR log2 slope={lr_slope:.4f}  elapsed={row['elapsed_s']:.1f}s")
        out_json = write_checkpoint(
            run_paths,
            build_payload(run_stem=run_stem, cfg=cfg, trace=trace, setup=saved_setup),
        )
        print(f"  checkpoint -> {out_json.name}")

    out_json = write_checkpoint(
        run_paths,
        build_payload(run_stem=run_stem, cfg=cfg, trace=trace, setup=saved_setup),
    )

    plot_path = plot_scaling_vs_depth(
        trace,
        run_paths["png"],
        settings={**cfg, "k": k},
        headline=scaling_plot_headline(
            "GPU LR sweep", cfg["eval_axis"], cfg["eval_aggregation"]
        ),
        annotate_first_win=bool(cfg.get("annotate_first_win", False)),
    )
    print(f"\nWrote {out_json}")
    print(f"Plot -> {plot_path}")
    return out_json


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="LR-QAOA benchmark (GPU QAOA, CPU fallback). "
        "Mirrors experiments/lr_scaling/notebooks/LR_QAOA_benchmark.ipynb.",
    )
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--r", type=float, default=176.54)
    p.add_argument("--seed", type=int, default=27)
    p.add_argument("--depths", default="2,5,8,10,15,20,30,40,50,60,100")
    p.add_argument("--lr-beta-schedule", default="decreasing", dest="lr_beta_schedule")
    p.add_argument(
        "--training-mode",
        default="bm24_mean_p_fixed_n",
        choices=[
            "bm24_mean_p_fixed_n",
            "median_runtime_fixed_n",
            "mean_log_runtime_fixed_n",
        ],
    )
    p.add_argument("--train-n", type=int, default=14)
    p.add_argument("--train-size", type=int, default=100)
    p.add_argument("--n-min", type=int, default=12)
    p.add_argument("--n-max", type=int, default=20)
    p.add_argument("--test-size", type=int, default=200)
    p.add_argument("--skip-grid", action="store_true")
    p.add_argument("--skip-grid-if-warm-start", action="store_true", default=False)
    p.add_argument(
        "--exploratory",
        action="store_true",
        help="Robust angle search: full grid every depth, dg in [-2,2], no warm-start "
        "chain, no anti-regression/collapse/eval fallbacks (see apply_exploratory_preset).",
    )
    p.add_argument(
        "--depth-warm-start",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Warm-start each depth from previous depth angles (default: on).",
    )
    p.add_argument(
        "--dg-bounds",
        default=None,
        help="Grid/COBYLA box for delta_gamma as lo,hi (default: config.py, or -2,2 with --exploratory).",
    )
    p.add_argument(
        "--db-bounds",
        default=None,
        help="Grid/COBYLA box for delta_beta as lo,hi (default: config.py, or 0.1,4 with --exploratory).",
    )
    p.add_argument(
        "--anti-regression",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Revert to warm-start if COBYLA is worse than warm-start (default: on).",
    )
    p.add_argument(
        "--collapse-guard",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Revert to warm-start on dg>0 or tiny mean_p (default: on).",
    )
    p.add_argument(
        "--check-eval-regression",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reject angles when eval median runtime regresses vs previous depth.",
    )
    p.add_argument(
        "--fallback-to-previous-angles",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="After eval retries, keep previous depth angles instead of last train.",
    )
    p.add_argument("--cobyla-maxiter", type=int, default=160)
    p.add_argument("--cobyla-restarts", type=int, default=8)
    p.add_argument("--cobyla-perturb-scale", type=float, default=0.2)
    p.add_argument("--grid-top-k", type=int, default=5)
    p.add_argument("--walksat-p-noise", type=float, default=0.5)
    p.add_argument("--walksatlm-p-noise", type=float, default=0.15)
    p.add_argument("--walksatlm-w1", type=int, default=6)
    p.add_argument("--walksatlm-w2", type=int, default=5)
    p.add_argument("--max-flips", type=int, default=100_000)
    p.add_argument("--eval-axis", default="runtime")
    p.add_argument("--eval-aggregation", default="median")
    p.add_argument("--annotate-first-win", action="store_true")
    p.add_argument("--eval-train-retries", type=int, default=DEFAULT_EVAL_TRAIN_RETRIES)
    p.add_argument(
        "--eval-runtime-regression-factor",
        type=float,
        default=DEFAULT_EVAL_RUNTIME_REGRESSION_FACTOR,
    )
    p.add_argument(
        "--output-dir",
        default=None,
        help="Defaults to phasecraft/results/bm24_runs",
    )
    p.add_argument("--run-stem", default=None, help="Checkpoint stem; auto-generated if omitted")
    p.add_argument("--auto-resume", action="store_true", help="Resume from latest checkpoint")
    p.add_argument("--recover-from", default=None, help="Path to .partial.json or .json checkpoint")
    p.add_argument(
        "--gpu-batch-size",
        type=int,
        default=32,
        help="Instances per GPU batch (lower if OOM at large n)",
    )
    p.add_argument("--force-cpu", action="store_true", help="Disable CuPy even if CUDA is available")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.exploratory:
        apply_exploratory_preset(args)
    if args.run_stem is None:
        from phasecraft.lib.sim.bm24_run_io import make_run_stem

        tag = "gpu-explore" if args.exploratory else "gpu-scaling"
        args.run_stem = make_run_stem(tag)
    cfg = build_cfg_from_args(args)
    if cfg.get("exploratory"):
        print(
            "Exploratory preset: grid every depth, dg=[-2,2], no depth warm-start, "
            "no trainer/eval fallbacks"
        )
    try:
        run_pipeline(cfg)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
