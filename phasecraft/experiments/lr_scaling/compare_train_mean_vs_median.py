#!/usr/bin/env python3
"""
Quick A/B: train LR on mean(ln 1/p) vs median(1/p) slope; same eval (median cost vs n).

Example (from repo root, ~15–40 min depending on machine)::

    python phasecraft/experiments/lr_scaling/compare_train_mean_vs_median.py \\
      --depths 2,5,10 --seed 27
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

_PHASECRAFT = Path(__file__).resolve().parents[2]
_REPO = _PHASECRAFT.parent
for _p in (_REPO, _PHASECRAFT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from phasecraft.lib.paths import bm24_runs_dir  # noqa: E402
from phasecraft.lib.sim.bm24_qaoa_sim import (  # noqa: E402
    run_algorithm_benchmark,
    summarize_benchmark_scaling,
)
from phasecraft.lib.sim.bm24_run_io import (  # noqa: E402
    make_run_stem,
    resolve_bm24_run_output_paths,
)

from train_lr_notebook_protocol import (  # noqa: E402
    DEFAULT_EVAL_TRAIN_RETRIES,
    generate_training_h_diagonals,
    generate_training_h_diagonals_multi_n,
    proxy_n_values_for_training,
    run_train_eval_with_retries,
    train_lr_grid_search_bm24,
)


def _med_rt_from_bres(bres: dict) -> Dict[int, float]:
    lr_pn = bres.get("results", {}).get("lr_qaoa", {}).get("per_n", {})
    return {int(n): float(d["median_runtime"]) for n, d in lr_pn.items()}


def _run_arm(
    *,
    label: str,
    train_on_median: bool,
    depths: List[int],
    args: argparse.Namespace,
    training_h: list,
    proxy_h: dict,
    proxy_ns: List[int],
    walksat_slopes: Tuple[float, float],
) -> List[dict]:
    trace: List[dict] = []
    prev_deltas: Tuple[float, float] | None = None
    prev_med_rt: Dict[int, float] | None = None
    prev_accepted: Tuple[float, float] | None = None

    print(f"\n{'#' * 72}\nARM: {label} (train_on_median={train_on_median})\n{'#' * 72}")

    for depth in depths:
        t0 = time.time()
        print(f"\n--- depth p={depth} ({label}) ---")
        bench_holder: list = []

        def _train_at_depth(retry_index: int):
            warm = (
                tuple(prev_accepted)
                if retry_index > 0 and prev_accepted is not None
                else prev_deltas
            )
            skip_grid = bool(args.skip_grid)
            if retry_index > 0:
                skip_grid = False
            rng = np.random.default_rng(
                int(args.seed) + 2000 + depth + 10_000 * int(retry_index)
                + (5000 if train_on_median else 0)
            )
            _, diag = train_lr_grid_search_bm24(
                training_h,
                train_n=args.train_n,
                depth=depth,
                skip_grid=skip_grid,
                initial_deltas=warm,
                skip_grid_if_warm_start=not args.no_warm_skip_grid
                and retry_index == 0,
                beta_schedule=args.lr_beta_schedule,
                cobyla_maxiter=args.cobyla_maxiter,
                cobyla_restarts=args.cobyla_restarts,
                cobyla_perturb_scale=float(args.cobyla_perturb_scale)
                * (1.25 ** int(retry_index)),
                grid_top_k=args.grid_top_k,
                proxy_h_by_n=proxy_h,
                proxy_n_values=proxy_ns,
                train_on_median=train_on_median,
                rng=rng,
            )
            return (
                float(diag["best_deltas"][0]),
                float(diag["best_deltas"][1]),
                diag,
            )

        def _evaluate_at_angles(dg: float, db: float) -> Dict[int, float]:
            bres = run_algorithm_benchmark(
                n_min=args.n_min,
                n_max=args.n_max,
                k=args.k,
                r=args.r,
                test_size=args.test_size,
                base_seed=args.seed,
                algorithms=["lr_qaoa"],
                depth=depth,
                lr_delta_gamma=dg,
                lr_delta_beta=db,
                lr_beta_schedule=args.lr_beta_schedule,
                lr_angle_convention="bm24",
                require_sat=True,
            )
            bench_holder.append(bres)
            return _med_rt_from_bres(bres)

        te = run_train_eval_with_retries(
            train_at_depth=_train_at_depth,
            evaluate_at_angles=_evaluate_at_angles,
            prev_med_rt=prev_med_rt,
            prev_accepted_deltas=prev_accepted,
            n_min=args.n_min,
            n_max=args.n_max,
            max_retries=args.eval_train_retries,
            verbose=True,
        )
        dg, db = te["dg"], te["db"]
        diag = te["diag"]
        med_rt = te["med_rt"]
        sc = summarize_benchmark_scaling(bench_holder[-1])

        if not te["eval_rejected"] and not te["used_previous_angles"]:
            prev_accepted = (dg, db)
            prev_med_rt = dict(med_rt)
        prev_deltas = (dg, db)

        row = {
            "depth": depth,
            "train_arm": label,
            "train_on_median": train_on_median,
            "training_objective": diag.get("objective_version"),
            "delta_gamma": dg,
            "delta_beta": db,
            "best_train_slope_log2": diag.get("best_train_slope_log2"),
            "lr_log2_slope": sc["lr_qaoa"]["median_runtime_slope_log2"],
            "eval_rejected": te["eval_rejected"],
            "used_previous_angles": te["used_previous_angles"],
            "eval_train_attempts": te["eval_train_attempts"],
            "train_rejected": diag.get("train_rejected"),
            "elapsed_s": time.time() - t0,
        }
        trace.append(row)
        print(
            f"  => eval median-cost log2 slope = {row['lr_log2_slope']:.4f}  "
            f"(train slope log2 = {row['best_train_slope_log2']})  "
            f"prev_angles={row['used_previous_angles']}"
        )

    ws_l2, lm_l2 = walksat_slopes
    for row in trace:
        row["walksat_log2_slope"] = ws_l2
        row["walksatlm_log2_slope"] = lm_l2
    return trace


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--depths", type=str, default="2,5,10")
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--r", type=float, default=176.54)
    p.add_argument("--seed", type=int, default=27)
    p.add_argument("--train-n", type=int, default=12)
    p.add_argument("--train-size", type=int, default=50)
    p.add_argument("--n-min", type=int, default=12)
    p.add_argument("--n-max", type=int, default=20)
    p.add_argument("--test-size", type=int, default=100)
    p.add_argument("--proxy-size-per-n", type=int, default=25)
    p.add_argument("--proxy-n-span", type=int, default=4)
    p.add_argument("--proxy-n-step", type=int, default=2)
    p.add_argument("--cobyla-maxiter", type=int, default=100)
    p.add_argument("--cobyla-restarts", type=int, default=4)
    p.add_argument("--cobyla-perturb-scale", type=float, default=0.2)
    p.add_argument("--grid-top-k", type=int, default=5)
    p.add_argument("--skip-grid", action="store_true")
    p.add_argument("--no-warm-skip-grid", action="store_true")
    p.add_argument(
        "--eval-train-retries",
        type=int,
        default=1,
        help="Re-trains after eval regression (default 1 for speed).",
    )
    p.add_argument("--lr-beta-schedule", type=str, default="decreasing")
    p.add_argument("--output-dir", type=Path, default=None)
    args = p.parse_args()
    depths = [int(x) for x in args.depths.split(",") if x.strip()]

    out_dir = args.output_dir or bm24_runs_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    run_stem = make_run_stem("mean-vs-median-train-diag")
    run_paths = resolve_bm24_run_output_paths(out_dir, run_stem)

    proxy_ns = proxy_n_values_for_training(
        args.train_n,
        proxy_n_span=int(args.proxy_n_span),
        n_max_cap=int(args.n_max),
        step=int(args.proxy_n_step),
    )
    print(f"Proxy n for training: {proxy_ns}")
    proxy_h = generate_training_h_diagonals_multi_n(
        proxy_ns,
        k=args.k,
        r=args.r,
        train_size_per_n=int(args.proxy_size_per_n),
        base_seed=int(args.seed) + 77,
        m_sampling="notebook",
    )
    training_h = generate_training_h_diagonals(
        train_n=args.train_n,
        k=args.k,
        r=args.r,
        train_size=args.train_size,
        base_seed=args.seed,
        m_sampling="notebook",
    )

    print("Classical baselines (once)...")
    cl_bres = run_algorithm_benchmark(
        n_min=args.n_min,
        n_max=args.n_max,
        k=args.k,
        r=args.r,
        test_size=args.test_size,
        base_seed=args.seed,
        algorithms=["walksat", "walksatlm"],
        require_sat=True,
    )
    cl_sc = summarize_benchmark_scaling(cl_bres)
    ws_l2 = float(cl_sc["walksat"]["median_flips_slope_log2"])
    lm_l2 = float(cl_sc["walksatlm"]["median_flips_slope_log2"])
    print(f"  WalkSAT log2 slope = {ws_l2:.4f}, WalkSATlm = {lm_l2:.4f}")

    t0 = time.time()
    trace_mean = _run_arm(
        label="mean_train",
        train_on_median=False,
        depths=depths,
        args=args,
        training_h=training_h,
        proxy_h=proxy_h,
        proxy_ns=proxy_ns,
        walksat_slopes=(ws_l2, lm_l2),
    )
    trace_median = _run_arm(
        label="median_train",
        train_on_median=True,
        depths=depths,
        args=args,
        training_h=training_h,
        proxy_h=proxy_h,
        proxy_ns=proxy_ns,
        walksat_slopes=(ws_l2, lm_l2),
    )

    payload = {
        "run_stem": run_stem,
        "diagnostic": "train_mean_vs_median",
        "settings": {
            k: (str(v) if isinstance(v, Path) else v)
            for k, v in vars(args).items()
        },
        "depths": depths,
        "walksat_log2_slope": ws_l2,
        "walksatlm_log2_slope": lm_l2,
        "trace_mean_train": trace_mean,
        "trace_median_train": trace_median,
        "elapsed_s": time.time() - t0,
    }
    json_path = run_paths["json"]
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote {json_path}")

    fig, ax = plt.subplots(figsize=(8, 5))
    for trace, style, name in [
        (trace_mean, "o-", "train: mean(ln 1/p)"),
        (trace_median, "s--", "train: median(1/p)"),
    ]:
        ps = [r["depth"] for r in trace]
        ys = [r["lr_log2_slope"] for r in trace]
        ax.plot(ps, ys, style, label=name, linewidth=2, markersize=8)
        for r in trace:
            if r.get("used_previous_angles"):
                ax.plot(
                    r["depth"],
                    r["lr_log2_slope"],
                    marker="x",
                    color=ax.lines[-1].get_color(),
                    markersize=12,
                    markeredgewidth=2,
                    linestyle="none",
                )
    ax.axhline(ws_l2, color="C2", linestyle=":", label=f"WalkSAT ({ws_l2:.3f})")
    ax.axhline(lm_l2, color="C3", linestyle=":", label=f"WalkSATlm ({lm_l2:.3f})")
    ax.set_xlabel("QAOA depth p")
    ax.set_ylabel(r"Eval: $\log_2$ slope of median(1/p) vs $n$")
    ax.set_title(
        f"Train objective A/B · k={args.k}, seed={args.seed} · "
        f"n∈[{args.n_min},{args.n_max}] · test={args.test_size} · "
        f"proxy={args.proxy_size_per_n}/n · × = reused prior angles"
    )
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    png_path = run_paths["png"]
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {png_path}")

    print("\n--- Summary (eval median-cost log2 slope) ---")
    print(f"{'p':>4}  {'mean_train':>12}  {'median_train':>12}  {'Δ(med-mean)':>12}")
    for d in depths:
        m = next(r for r in trace_mean if r["depth"] == d)
        med = next(r for r in trace_median if r["depth"] == d)
        dm = float(m["lr_log2_slope"]) - float(med["lr_log2_slope"])
        print(
            f"{d:4d}  {m['lr_log2_slope']:12.4f}  {med['lr_log2_slope']:12.4f}  "
            f"{dm:+12.4f}  (neg=mean better)"
        )


if __name__ == "__main__":
    main()
