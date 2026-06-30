#!/usr/bin/env python3
"""
Run regular QAOA on BM24 random k-SAT and compare scaling with LR-QAOA.

The regular-QAOA branch follows the PRX Quantum numerical protocol: train fixed
angles on a small fixed-n SAT-filtered sample by maximizing empirical mean
success probability, then evaluate median runtime 1/p_succ over held-out sizes.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "phasecraft-mpl"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress
from tqdm import tqdm

_THIS_DIR = Path(__file__).resolve().parent
_PHASECRAFT_ROOT = _THIS_DIR.parent
_REPO_ROOT = _PHASECRAFT_ROOT.parent
for _p in (_THIS_DIR, _REPO_ROOT, _PHASECRAFT_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from train_qaoa_fixed_n import (  # noqa: E402
    DEFAULT_BETA_BOUNDS,
    DEFAULT_GAMMA_BOUNDS,
    MODE_RNG_OFFSET,
    SUPPORTED_TRAINING_MODES,
    generate_training_instances,
    resize_angles,
    train_regular_qaoa_fixed_n,
)
from phasecraft.experiments.lr_scaling.train_lr_fixed_n import (  # noqa: E402
    MODE_RNG_OFFSET as LR_MODE_RNG_OFFSET,
    train_angles_fixed_n as train_lr_angles_fixed_n,
)
from phasecraft.lib.sim.bm24_qaoa_sim import (  # noqa: E402
    build_h_diagonal,
    generate_random_clause,
    make_lr_angles,
    per_instance_success_probability,
    run_qaoa,
)

try:
    from phasecraft.experiments.lr_scaling.gpu.backend import (  # type: ignore
        detect_device,
        success_probs_for_instances,
    )
except Exception:  # pragma: no cover
    detect_device = None  # type: ignore[assignment]
    success_probs_for_instances = None  # type: ignore[assignment]


LN2 = float(np.log(2.0))
DEFAULT_RESULTS_DIR = _THIS_DIR / "results"


def parse_int_list(text: str) -> List[int]:
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def parse_float_pair(text: Optional[str], default: tuple[float, float]) -> tuple[float, float]:
    if text is None:
        return default
    parts = [p.strip() for p in str(text).split(",") if p.strip()]
    if len(parts) != 2:
        raise ValueError(f"expected lo,hi pair, got {text!r}")
    return float(parts[0]), float(parts[1])


def make_run_stem(prefix: str) -> str:
    return f"{prefix}-{time.strftime('%Y%m%d-%H%M%S')}"


def atomic_write_json(path: Path, payload: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, allow_nan=True, default=str), encoding="utf-8")
    tmp.replace(path)


def sample_m(rng: np.random.Generator, n: int, r: float, mode: str) -> int:
    m = int(rng.poisson(float(r) * int(n)))
    if mode == "notebook":
        return max(1, m)
    if mode == "bm24":
        return m
    raise ValueError("m_sampling must be notebook or bm24")


def generate_benchmark_dataset(
    n_values: Sequence[int],
    *,
    k: int,
    r: float,
    test_size: int,
    base_seed: int,
    require_sat: bool,
    m_sampling: str,
    max_trials_per_n: int,
) -> Dict[int, List[np.ndarray]]:
    dataset: Dict[int, List[np.ndarray]] = {}
    for n in n_values:
        accepted = 0
        trial = 0
        insts: List[np.ndarray] = []
        pbar = tqdm(total=int(test_size), desc=f"dataset n={n}", leave=False)
        while accepted < int(test_size):
            if trial >= int(max_trials_per_n):
                raise RuntimeError(f"too many rejected eval formulas at n={n}")
            ss = np.random.SeedSequence([int(base_seed), int(n), int(accepted), int(trial)])
            rng = np.random.default_rng(ss)
            m = sample_m(rng, int(n), float(r), m_sampling)
            clauses = [generate_random_clause(int(n), int(k), rng) for _ in range(m)]
            h_diag = build_h_diagonal(clauses, int(n))
            if require_sat and not np.any(h_diag == 0):
                trial += 1
                continue
            insts.append(h_diag)
            accepted += 1
            trial += 1
            pbar.update(1)
        pbar.close()
        dataset[int(n)] = insts
    return dataset


def fit_log2_slope(n_values: Sequence[int], y_values: Sequence[float]) -> float:
    n_arr = np.asarray(n_values, dtype=float)
    y_arr = np.asarray(y_values, dtype=float)
    mask = np.isfinite(y_arr) & (y_arr > 0)
    if int(mask.sum()) < 2:
        return float("nan")
    res = linregress(n_arr[mask], np.log(y_arr[mask]))
    return float(res.slope / LN2)


def success_probs(
    instances: Sequence[np.ndarray],
    n: int,
    betas: np.ndarray,
    gammas: np.ndarray,
    *,
    device=None,
    batch_size: Optional[int] = None,
) -> np.ndarray:
    if success_probs_for_instances is not None and device is not None:
        return success_probs_for_instances(
            instances,
            int(n),
            betas,
            gammas,
            device=device,
            batch_size=batch_size,
        )
    out = []
    for h_diag in instances:
        psi = run_qaoa(h_diag, betas, gammas, int(n))
        out.append(float(per_instance_success_probability(psi, h_diag)))
    return np.asarray(out, dtype=float)


def evaluate_angles(
    dataset: Mapping[int, Sequence[np.ndarray]],
    n_values: Sequence[int],
    betas: np.ndarray,
    gammas: np.ndarray,
    *,
    eps: float,
    device=None,
    batch_size: Optional[int] = None,
) -> dict:
    per_n = {}
    for n in n_values:
        probs = success_probs(
            dataset[int(n)],
            int(n),
            betas,
            gammas,
            device=device,
            batch_size=batch_size,
        )
        costs = 1.0 / np.maximum(probs, float(eps))
        per_n[int(n)] = {
            "N": int(len(probs)),
            "mean_p_succ": float(np.mean(probs)) if len(probs) else float("nan"),
            "median_p_succ": float(np.median(probs)) if len(probs) else float("nan"),
            "median_runtime": float(np.median(costs)) if len(costs) else float("nan"),
            "fraction_zero_success": float(np.mean(probs <= 0.0)) if len(probs) else float("nan"),
        }
    med_slope = fit_log2_slope(n_values, [per_n[int(n)]["median_runtime"] for n in n_values])
    mean_slope = -fit_log2_slope(n_values, [per_n[int(n)]["mean_p_succ"] for n in n_values])
    return {
        "per_n": {str(k): v for k, v in per_n.items()},
        "median_runtime_log2_slope": float(med_slope),
        "mean_success_exponent_log2": float(mean_slope),
    }


def load_lr_reference(path: Optional[Path]) -> Dict[int, float]:
    if path is None:
        return {}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = payload.get("trace", payload.get("rows", []))
    out: Dict[int, float] = {}
    for row in rows:
        if "depth" in row and "lr_log2_slope" in row:
            out[int(row["depth"])] = float(row["lr_log2_slope"])
        elif "p" in row and "lr_log2_slope" in row:
            out[int(row["p"])] = float(row["lr_log2_slope"])
    return out


def plot_trace(trace: Sequence[Mapping], output: Path) -> None:
    if not trace:
        return
    depths = [int(r["depth"]) for r in trace]
    qaoa = [float(r["qaoa_log2_slope"]) for r in trace]
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    ax.plot(depths, qaoa, "o-", label="regular QAOA")
    if any(r.get("lr_log2_slope") is not None for r in trace):
        lr_depths = [int(r["depth"]) for r in trace if r.get("lr_log2_slope") is not None]
        lr_vals = [float(r["lr_log2_slope"]) for r in trace if r.get("lr_log2_slope") is not None]
        ax.plot(lr_depths, lr_vals, "s--", label="LR-QAOA")
    ax.set_xlabel("QAOA depth p")
    ax.set_ylabel("log2 slope of median(1/p_succ) vs n")
    ax.set_title("Regular QAOA vs LR-QAOA scaling")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def run_pipeline(cfg: Mapping) -> Path:
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    run_stem = str(cfg["run_stem"] or make_run_stem("qaoa-reg"))
    out_json = output_dir / f"{run_stem}.json"
    out_png = output_dir / f"{run_stem}.png"

    n_values = list(range(int(cfg["n_min"]), int(cfg["n_max"]) + 1))
    device = None
    device_summary = "CPU serial"
    if detect_device is not None and success_probs_for_instances is not None:
        detected = detect_device(prefer="numpy" if cfg.get("force_cpu") else None)
        device_summary = detected.summary()
        if bool(cfg.get("require_gpu")) and not getattr(detected, "cuda_available", False):
            raise SystemExit(f"--require-gpu was set, but backend is {device_summary}")
        if getattr(detected, "cuda_available", False) or cfg.get("batch_on_cpu"):
            device = detected
        else:
            device = None
    active_backend = device.summary() if device is not None and hasattr(device, "summary") else "CPU serial"
    print(f"Backend: {active_backend}; batch_size={cfg['gpu_batch_size'] if device is not None else 'n/a'}")

    lr_train_fn = train_lr_angles_fixed_n
    lr_mode_offsets = LR_MODE_RNG_OFFSET
    if device is not None:
        try:
            from phasecraft.experiments.lr_scaling.gpu import train_lr_fixed_n as gpu_lr_train  # type: ignore

            gpu_lr_train.configure_gpu(device=device, batch_size=int(cfg["gpu_batch_size"]))
            lr_train_fn = gpu_lr_train.train_angles_fixed_n
            lr_mode_offsets = gpu_lr_train.MODE_RNG_OFFSET
            print("LR-QAOA training: batched backend enabled")
        except Exception as exc:  # pragma: no cover - optional backend
            if bool(cfg.get("require_gpu")):
                raise
            print(f"LR-QAOA training: CPU serial (GPU patch unavailable: {exc})")
    elif bool(cfg.get("compare_lr")):
        print("LR-QAOA training: CPU serial")

    print("Building shared SAT-filtered benchmark dataset...")
    dataset = generate_benchmark_dataset(
        n_values,
        k=int(cfg["k"]),
        r=float(cfg["r"]),
        test_size=int(cfg["test_size"]),
        base_seed=int(cfg["eval_seed"]),
        require_sat=bool(cfg["sat_filter"]),
        m_sampling=str(cfg["m_sampling"]),
        max_trials_per_n=int(cfg["max_trials_per_n"]),
    )

    print(f"Building training set n={cfg['train_n']}, size={cfg['train_size']}...")
    training_instances = generate_training_instances(
        train_n=int(cfg["train_n"]),
        k=int(cfg["k"]),
        r=float(cfg["r"]),
        train_size=int(cfg["train_size"]),
        base_seed=int(cfg["seed"]),
    )

    lr_reference = load_lr_reference(Path(cfg["lr_checkpoint"]) if cfg.get("lr_checkpoint") else None)
    trace: List[dict] = []
    prev_qaoa: Optional[tuple[np.ndarray, np.ndarray]] = None
    prev_lr_deltas: Optional[tuple[float, float]] = None

    for depth in [int(x) for x in cfg["depths"]]:
        t_depth = time.time()
        print(f"\n{'=' * 60}\nDepth p = {depth}\n{'=' * 60}")

        regular_warm = None
        if bool(cfg["depth_warm_start"]) and prev_qaoa is not None:
            regular_warm = resize_angles(prev_qaoa[0], prev_qaoa[1], depth)

        qaoa_rng = np.random.default_rng(
            int(cfg["seed"])
            + 1000
            + depth
            + int(MODE_RNG_OFFSET.get(str(cfg["training_mode"]), 0))
        )
        _, qaoa_diag = train_regular_qaoa_fixed_n(
            training_mode=str(cfg["training_mode"]),
            train_n=int(cfg["train_n"]),
            depth=depth,
            instances=training_instances,
            initial_angles=regular_warm,
            gamma_bounds=tuple(cfg["gamma_bounds"]),
            beta_bounds=tuple(cfg["beta_bounds"]),
            optimizer=str(cfg["optimizer"]),
            maxiter=int(cfg["maxiter"]),
            tol=float(cfg["tol"]),
            restarts=int(cfg["restarts"]),
            perturb_scale=float(cfg["perturb_scale"]),
            anti_regression=bool(cfg["anti_regression"]),
            collapse_guard=bool(cfg["collapse_guard"]),
            eps=float(cfg["eps"]),
            device=device,
            batch_size=int(cfg["gpu_batch_size"]) if device is not None else None,
            rng=qaoa_rng,
        )
        qaoa_betas = np.asarray(qaoa_diag["best_betas"], dtype=float)
        qaoa_gammas = np.asarray(qaoa_diag["best_gammas"], dtype=float)
        prev_qaoa = (qaoa_betas, qaoa_gammas)
        qaoa_eval = evaluate_angles(
            dataset,
            n_values,
            qaoa_betas,
            qaoa_gammas,
            eps=float(cfg["eps"]),
            device=device,
            batch_size=int(cfg["gpu_batch_size"]),
        )

        lr_eval = None
        lr_diag = None
        lr_slope = lr_reference.get(depth)
        if bool(cfg["compare_lr"]):
            lr_rng = np.random.default_rng(
                int(cfg["seed"])
                + 2000
                + depth
                + int(lr_mode_offsets.get(str(cfg["training_mode"]), 0))
            )
            _, lr_diag = lr_train_fn(
                training_mode=str(cfg["training_mode"]),
                train_n=int(cfg["train_n"]),
                depth=depth,
                instances=training_instances,
                initial_angles=prev_lr_deltas if bool(cfg["depth_warm_start"]) else None,
                beta_schedule=str(cfg["lr_beta_schedule"]),
                skip_grid=bool(cfg["lr_skip_grid"]),
                skip_grid_if_warm_start=True,
                cobyla_maxiter=int(cfg["lr_cobyla_maxiter"]),
                cobyla_restarts=int(cfg["lr_cobyla_restarts"]),
                cobyla_perturb_scale=float(cfg["lr_cobyla_perturb_scale"]),
                grid_top_k=int(cfg["lr_grid_top_k"]),
                rng=lr_rng,
                verbose=True,
            )
            dg, db = map(float, lr_diag["best_deltas"])
            prev_lr_deltas = (dg, db)
            lr_betas, lr_gammas = make_lr_angles(
                dg,
                db,
                depth,
                beta_schedule=str(cfg["lr_beta_schedule"]),
                angle_convention="bm24",
            )
            lr_eval = evaluate_angles(
                dataset,
                n_values,
                lr_betas,
                lr_gammas,
                eps=float(cfg["eps"]),
                device=device,
                batch_size=int(cfg["gpu_batch_size"]),
            )
            lr_slope = float(lr_eval["median_runtime_log2_slope"])

        row = {
            "depth": int(depth),
            "training_mode": str(cfg["training_mode"]),
            "train_n": int(cfg["train_n"]),
            "qaoa_log2_slope": float(qaoa_eval["median_runtime_log2_slope"]),
            "qaoa_mean_success_exponent_log2": float(qaoa_eval["mean_success_exponent_log2"]),
            "qaoa_median_runtime_per_n": {
                str(n): qaoa_eval["per_n"][str(n)]["median_runtime"] for n in n_values
            },
            "qaoa_mean_p_succ_per_n": {
                str(n): qaoa_eval["per_n"][str(n)]["mean_p_succ"] for n in n_values
            },
            "qaoa_betas": qaoa_betas.tolist(),
            "qaoa_gammas": qaoa_gammas.tolist(),
            "qaoa_diagnostics": qaoa_diag,
            "lr_log2_slope": lr_slope,
            "lr_reference_source": str(cfg["lr_checkpoint"]) if cfg.get("lr_checkpoint") and not cfg["compare_lr"] else None,
            "lr_diagnostics": lr_diag,
            "lr_eval": lr_eval,
            "elapsed_s": float(time.time() - t_depth),
        }
        trace.append(row)
        payload = {
            "schema_version": 1,
            "kind": "qaoa_reg_scaling_benchmark",
            "run_stem": run_stem,
            "created_at_unix": time.time(),
            "config": dict(cfg),
            "trace": trace,
        }
        atomic_write_json(out_json, payload)
        plot_trace(trace, out_png)
        msg = f"  regular QAOA log2 slope={row['qaoa_log2_slope']:.4f}"
        if lr_slope is not None:
            msg += f"  LR-QAOA log2 slope={float(lr_slope):.4f}"
        msg += f"  elapsed={row['elapsed_s']:.1f}s"
        print(msg)
        print(f"  checkpoint -> {out_json}")

    print(f"\nWrote {out_json}")
    print(f"Plot -> {out_png}")
    return out_json


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Regular QAOA BM24 scaling benchmark.")
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--r", type=float, default=176.54)
    p.add_argument("--seed", type=int, default=27, help="training seed")
    p.add_argument("--eval-seed", type=int, default=100_027)
    p.add_argument("--depths", default="2,5,8,10,15,20")
    p.add_argument("--training-mode", choices=sorted(SUPPORTED_TRAINING_MODES), default="bm24_mean_p_fixed_n")
    p.add_argument("--train-n", type=int, default=12)
    p.add_argument("--train-size", type=int, default=100)
    p.add_argument("--n-min", type=int, default=12)
    p.add_argument("--n-max", type=int, default=20)
    p.add_argument("--test-size", type=int, default=200)
    p.add_argument("--m-sampling", choices=["notebook", "bm24"], default="notebook")
    p.add_argument("--sat-filter", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--max-trials-per-n", type=int, default=1_000_000)
    p.add_argument("--gamma-bounds", default=None, help="Regular-QAOA gamma box as lo,hi. Default: -2,0.")
    p.add_argument("--beta-bounds", default=None, help="Regular-QAOA beta box as lo,hi. Default: 0,4.")
    p.add_argument("--optimizer", choices=["COBYLA", "L-BFGS-B", "Powell"], default="COBYLA")
    p.add_argument("--maxiter", type=int, default=240)
    p.add_argument("--tol", type=float, default=1e-3)
    p.add_argument("--restarts", type=int, default=4)
    p.add_argument("--perturb-scale", type=float, default=0.1)
    p.add_argument("--depth-warm-start", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--anti-regression", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--collapse-guard", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--eps", type=float, default=1e-300)
    p.add_argument("--compare-lr", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--lr-checkpoint", default=None, help="Existing LR checkpoint JSON to overlay when --no-compare-lr is used.")
    p.add_argument("--lr-beta-schedule", default="decreasing")
    p.add_argument("--lr-skip-grid", action="store_true")
    p.add_argument("--lr-cobyla-maxiter", type=int, default=160)
    p.add_argument("--lr-cobyla-restarts", type=int, default=8)
    p.add_argument("--lr-cobyla-perturb-scale", type=float, default=0.2)
    p.add_argument("--lr-grid-top-k", type=int, default=5)
    p.add_argument("--gpu-batch-size", type=int, default=32)
    p.add_argument("--force-cpu", action="store_true")
    p.add_argument("--batch-on-cpu", action="store_true")
    p.add_argument("--require-gpu", action="store_true", help="Fail fast unless CUDA/CuPy is available.")
    p.add_argument("--output-dir", default=str(DEFAULT_RESULTS_DIR))
    p.add_argument("--run-stem", default=None)
    return p.parse_args(argv)


def build_cfg(args: argparse.Namespace) -> dict:
    return {
        "k": int(args.k),
        "r": float(args.r),
        "seed": int(args.seed),
        "eval_seed": int(args.eval_seed),
        "depths": parse_int_list(args.depths),
        "training_mode": str(args.training_mode),
        "train_n": int(args.train_n),
        "train_size": int(args.train_size),
        "n_min": int(args.n_min),
        "n_max": int(args.n_max),
        "test_size": int(args.test_size),
        "m_sampling": str(args.m_sampling),
        "sat_filter": bool(args.sat_filter),
        "max_trials_per_n": int(args.max_trials_per_n),
        "gamma_bounds": parse_float_pair(args.gamma_bounds, DEFAULT_GAMMA_BOUNDS),
        "beta_bounds": parse_float_pair(args.beta_bounds, DEFAULT_BETA_BOUNDS),
        "optimizer": str(args.optimizer),
        "maxiter": int(args.maxiter),
        "tol": float(args.tol),
        "restarts": int(args.restarts),
        "perturb_scale": float(args.perturb_scale),
        "depth_warm_start": bool(args.depth_warm_start),
        "anti_regression": bool(args.anti_regression),
        "collapse_guard": bool(args.collapse_guard),
        "eps": float(args.eps),
        "compare_lr": bool(args.compare_lr),
        "lr_checkpoint": args.lr_checkpoint,
        "lr_beta_schedule": str(args.lr_beta_schedule),
        "lr_skip_grid": bool(args.lr_skip_grid),
        "lr_cobyla_maxiter": int(args.lr_cobyla_maxiter),
        "lr_cobyla_restarts": int(args.lr_cobyla_restarts),
        "lr_cobyla_perturb_scale": float(args.lr_cobyla_perturb_scale),
        "lr_grid_top_k": int(args.lr_grid_top_k),
        "gpu_batch_size": int(args.gpu_batch_size),
        "force_cpu": bool(args.force_cpu),
        "batch_on_cpu": bool(args.batch_on_cpu),
        "require_gpu": bool(args.require_gpu),
        "output_dir": str(Path(args.output_dir).expanduser().resolve()),
        "run_stem": args.run_stem,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    cfg = build_cfg(args)
    run_pipeline(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
