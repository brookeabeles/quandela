#!/usr/bin/env python3
"""
compare_objectives_fixed_n.py
==============================
A/B comparison of the two training modes in train_lr_fixed_n.py:

  "bm24_mean_p_fixed_n"    — maximize mean(p_succ)            [BM24 baseline]
  "median_runtime_fixed_n" — minimize median(1/(p_succ+eps))  [robust runtime]

Both modes train at a single fixed train_n; evaluation is the log2 slope of
median(1/p_succ) vs n across n_min..n_max (same metric as the notebook).

Usage (from repo root or lr_scaling dir):
    python experiments/lr_scaling/compare_objectives_fixed_n.py
    python experiments/lr_scaling/compare_objectives_fixed_n.py \\
        --train-n 14 --train-size 30 --test-size 50 --n-min 14 --n-max 17 \\
        --depths 2,5,10,15

Outputs (default): ``bm24_runs/MM-DD/runN/{stem}.json`` and ``.png`` where
``stem`` is like ``06-09_1723-train14-tr30-te50-n14-17`` (time + key params).
Replot: ``--from-json path/to/run.json``
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress
from tqdm.auto import tqdm

_PHASECRAFT = Path(__file__).resolve().parents[2]
_REPO = _PHASECRAFT.parent
_LR_SCALING = Path(__file__).resolve().parent
for _p in (_REPO, _PHASECRAFT, _LR_SCALING):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from phasecraft.lib.paths import bm24_runs_dir  # noqa: E402
from phasecraft.lib.sim.bm24_run_io import (  # noqa: E402
    apply_matplotlib_title,
    format_benchmark_title,
    make_run_stem,
    resolve_bm24_run_output_paths,
)
from phasecraft.lib.sim.bm24_qaoa_sim import (  # noqa: E402
    build_h_diagonal,
    generate_random_clause,
    make_lr_angles,
    per_instance_success_probability,
    run_qaoa,
)
from train_lr_fixed_n import (  # noqa: E402
    DEFAULT_EVAL_RUNTIME_REGRESSION_FACTOR,
    DEFAULT_EVAL_TRAIN_RETRIES,
    eval_median_runtime_reject,
    generate_training_instances,
    run_train_eval_with_retries,
    train_angles_fixed_n,
)

LN2 = float(np.log(2.0))


def make_objective_comparison_kind(cfg: dict) -> str:
    """Kind tag for run stem: encodes train/eval sizes and n range."""
    return (
        f"train{int(cfg['train_n'])}-tr{int(cfg['train_size'])}"
        f"-te{int(cfg['test_size'])}-n{int(cfg['n_min'])}-{int(cfg['n_max'])}"
    )


# --------------------------------------------------------------------------- #
# Dataset helpers (mirrors notebook generate_benchmark_dataset_cached)         #
# --------------------------------------------------------------------------- #

def _clauses_to_arrays(clauses, k: int):
    m = len(clauses)
    c_vars = np.zeros((m, k), dtype=np.int32)
    c_signs = np.zeros((m, k), dtype=np.int32)
    for i, clause in enumerate(clauses):
        for j, (var, is_negated) in enumerate(clause):
            c_vars[i, j] = int(var)
            c_signs[i, j] = 1 - int(bool(is_negated))
    return c_vars, c_signs


def generate_eval_dataset(
    n_values: List[int],
    k: int,
    r: float,
    test_size: int,
    base_seed: int,
) -> Dict[int, List[dict]]:
    dataset: Dict[int, List[dict]] = {}
    for n in n_values:
        accepted, trial = 0, 0
        instances = []
        pbar = tqdm(total=test_size, desc=f"dataset n={n}", leave=False)
        while accepted < test_size:
            if trial > 200_000:
                raise RuntimeError(f"Too many rejections at n={n}")
            ss = np.random.SeedSequence([int(base_seed), n, accepted, trial])
            rng = np.random.default_rng(ss)
            m = max(1, int(rng.poisson(float(r) * n)))
            clauses = [generate_random_clause(n, k, rng) for _ in range(m)]
            h_diag = build_h_diagonal(clauses, n)
            if np.any(h_diag == 0):
                instances.append({"h_diag": h_diag})
                accepted += 1
                pbar.update(1)
            trial += 1
        pbar.close()
        dataset[n] = instances
    return dataset


def evaluate_qaoa(
    dataset: Dict[int, List[dict]],
    n_values: List[int],
    betas: np.ndarray,
    gammas: np.ndarray,
    eps: float = 1e-300,
) -> Dict[int, float]:
    med_rt: Dict[int, float] = {}
    for n in n_values:
        costs = []
        for inst in dataset[int(n)]:
            psi = run_qaoa(inst["h_diag"], betas, gammas, int(n))
            p = per_instance_success_probability(psi, inst["h_diag"])
            costs.append(1.0 / max(float(p), eps))
        med_rt[int(n)] = float(np.median(costs))
    return med_rt


def fit_log2_slope(n_values: List[int], y_values: List[float]) -> float:
    n_arr = np.asarray(n_values, dtype=float)
    y_arr = np.asarray(y_values, dtype=float)
    mask = np.isfinite(y_arr) & (y_arr > 0)
    if mask.sum() < 2:
        return float("nan")
    slope_nat = linregress(n_arr[mask], np.log(y_arr[mask])).slope
    return float(slope_nat / LN2)


def plot_objective_comparison(
    *,
    trace_mean: List[dict],
    trace_median: List[dict],
    n_values: List[int],
    depths: List[int],
    cfg: dict,
    walksat_log2_slope: float,
    output_path: Path,
) -> None:
    """Single slope-vs-depth panel (no secondary delta or scaling gauge)."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for trace, label, style in [
        (trace_mean, "bm24_mean_p_fixed_n", "o-"),
        (trace_median, "median_runtime_fixed_n", "s--"),
    ]:
        ps = [r["depth"] for r in trace]
        ys = [r["lr_log2_slope"] for r in trace]
        ax.plot(ps, ys, style, label=label, linewidth=2, markersize=7)
        for r in trace:
            if r.get("used_previous_angles") or r.get("eval_rejected"):
                ax.plot(
                    r["depth"],
                    r["lr_log2_slope"],
                    "x",
                    color=ax.lines[-1].get_color(),
                    markersize=11,
                    markeredgewidth=2,
                    linestyle="none",
                )
    if np.isfinite(walksat_log2_slope):
        ax.axhline(
            walksat_log2_slope,
            color="C2",
            linestyle=":",
            linewidth=1.5,
            label=f"WalkSAT ({walksat_log2_slope:.3f})",
        )
    ax.set_xlabel("QAOA depth p")
    ax.set_ylabel(r"Eval: $\log_2$ slope of median(1/p_succ) vs n")
    title = format_benchmark_title(
        cfg,
        headline="LR objective A/B · mean_p vs median_rt @ train_n",
        depths=depths,
    )
    apply_matplotlib_title(ax, f"{title}\n× = prior angles used", fig=fig)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(depths)

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def replot_from_json(json_path: Path, output_path: Path | None = None) -> Path:
    with open(json_path, encoding="utf-8") as f:
        payload = json.load(f)
    cfg = payload["config"]
    n_values = [int(n) for n in payload["n_values"]]
    depths = [int(d) for d in payload["depths"]]
    out = output_path or json_path.with_suffix(".png")
    plot_objective_comparison(
        trace_mean=payload["trace_bm24_mean_p_fixed_n"],
        trace_median=payload["trace_median_runtime_fixed_n"],
        n_values=n_values,
        depths=depths,
        cfg=cfg,
        walksat_log2_slope=float(payload.get("walksat_log2_slope", float("nan"))),
        output_path=out,
    )
    return out


# --------------------------------------------------------------------------- #
# Classical baselines (WalkSAT via numba — inline for self-containedness)      #
# --------------------------------------------------------------------------- #

try:
    from numba import njit

    @njit(cache=True)
    def _walksat_kernel(n, c_vars, c_signs, max_flips, p_noise):
        assignment = np.random.randint(0, 2, n)
        m, k_sat = c_vars.shape
        unsat_buf = np.empty(m, dtype=np.int32)
        for flip in range(max_flips):
            unsat_n = 0
            for i in range(m):
                ok = False
                for kk in range(k_sat):
                    if assignment[c_vars[i, kk]] == c_signs[i, kk]:
                        ok = True
                        break
                if not ok:
                    unsat_buf[unsat_n] = i
                    unsat_n += 1
            if unsat_n == 0:
                return flip + 1
            ci = unsat_buf[np.random.randint(0, unsat_n)]
            if np.random.random() < p_noise:
                vf = c_vars[ci, np.random.randint(0, k_sat)]
            else:
                best_var, min_b = -1, 999999
                for kk in range(k_sat):
                    cv = c_vars[ci, kk]
                    assignment[cv] = 1 - assignment[cv]
                    brk = 0
                    for i_s in range(m):
                        sat = False
                        for kk2 in range(k_sat):
                            if assignment[c_vars[i_s, kk2]] == c_signs[i_s, kk2]:
                                sat = True
                                break
                        if not sat:
                            brk += 1
                    if brk < min_b:
                        min_b = brk
                        best_var = cv
                    assignment[cv] = 1 - assignment[cv]
                vf = best_var
            assignment[vf] = 1 - assignment[vf]
        return max_flips

    _HAVE_NUMBA = True
except Exception:
    _HAVE_NUMBA = False


def _run_walksat_classical(
    dataset: Dict[int, List[dict]],
    k: int,
    p_noise: float,
    max_flips: int,
    base_seed: int,
) -> Dict[int, float]:
    if not _HAVE_NUMBA:
        print("  (numba not available — skipping classical WalkSAT baseline)")
        return {}
    result: Dict[int, float] = {}
    for n in sorted(dataset):
        flips = []
        for idx, inst in enumerate(tqdm(dataset[n], desc=f"WalkSAT n={n}", leave=False)):
            clauses = inst.get("clauses")
            if clauses is None:
                continue
            c_vars, c_signs = _clauses_to_arrays(clauses, k)
            seed = int(np.random.SeedSequence([base_seed, n, idx]).generate_state(1)[0])
            np.random.seed(seed)
            flips.append(int(_walksat_kernel(n, c_vars, c_signs, max_flips, p_noise)))
        if flips:
            result[n] = float(np.median(flips))
    return result


# --------------------------------------------------------------------------- #
# Per-arm training + eval loop                                                 #
# --------------------------------------------------------------------------- #

def run_arm(
    *,
    training_mode: str,
    depths: List[int],
    training_instances: List[np.ndarray],
    dataset: Dict[int, List[dict]],
    n_values: List[int],
    cfg: dict,
) -> List[dict]:
    print(f"\n{'#'*72}\nARM: {training_mode}\n{'#'*72}")

    prev_deltas: Optional[Tuple[float, float]] = None
    prev_accepted_deltas: Optional[Tuple[float, float]] = None
    prev_med_rt: Optional[Dict[int, float]] = None
    trace: List[dict] = []

    for depth in depths:
        depth = int(depth)
        t0 = time.time()
        print(f"\n{'='*60}\nDepth p={depth} | mode={training_mode}\n{'='*60}")

        def _train_at_depth(retry_index: int):
            warm = (
                tuple(prev_accepted_deltas)
                if retry_index > 0 and prev_accepted_deltas is not None
                else (tuple(prev_deltas) if prev_deltas is not None else None)
            )
            perturb = float(cfg["cobyla_perturb_scale"]) * (1.25 ** int(retry_index))
            rng = np.random.default_rng(
                int(cfg["seed"]) + 1000 + depth + 10_000 * int(retry_index)
                + (99999 if training_mode == "median_runtime_fixed_n" else 0)
            )
            _, diag = train_angles_fixed_n(
                training_mode=training_mode,
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
                rng=rng,
            )
            return float(diag["best_deltas"][0]), float(diag["best_deltas"][1]), diag

        def _evaluate_at_angles(dg: float, db: float) -> Dict[int, float]:
            betas_e, gammas_e = make_lr_angles(
                dg, db, depth,
                beta_schedule=str(cfg["lr_beta_schedule"]),
                angle_convention="bm24",
            )
            return evaluate_qaoa(dataset, n_values, betas_e, gammas_e)

        te = run_train_eval_with_retries(
            train_at_depth=_train_at_depth,
            evaluate_at_angles=_evaluate_at_angles,
            prev_med_rt=prev_med_rt,
            prev_accepted_deltas=prev_accepted_deltas,
            n_min=int(cfg["n_min"]),
            n_max=int(cfg["n_max"]),
            max_retries=int(cfg.get("eval_train_retries", DEFAULT_EVAL_TRAIN_RETRIES)),
            regression_factor=float(cfg.get("eval_runtime_regression_factor",
                                            DEFAULT_EVAL_RUNTIME_REGRESSION_FACTOR)),
        )

        dg, db = te["dg"], te["db"]
        diag = te["diag"]
        med_rt = te["med_rt"]
        lr_slope = fit_log2_slope(n_values, [med_rt[n] for n in n_values])

        if not te["eval_rejected"]:
            prev_accepted_deltas = (dg, db)
            prev_med_rt = dict(med_rt)
        prev_deltas = prev_accepted_deltas

        row = {
            "depth": depth,
            "training_mode": training_mode,
            "delta_gamma": float(dg),
            "delta_beta": float(db),
            "lr_log2_slope": lr_slope,
            "best_avg_train_p_succ": diag.get("best_avg_train_p_succ"),
            "best_train_objective": diag.get("best_train_objective"),
            "train_rejected": bool(diag.get("train_rejected", False)),
            "collapse_guard_applied": bool(diag.get("collapse_guard_applied", False)),
            "eval_rejected": bool(te["eval_rejected"]),
            "eval_train_attempts": int(te.get("eval_train_attempts", 1)),
            "used_previous_angles": bool(te.get("used_previous_angles", False)),
            "median_runtime_per_n": {str(n): med_rt[n] for n in n_values},
            "elapsed_s": time.time() - t0,
        }
        trace.append(row)
        print(f"  LR log2 slope = {lr_slope:.4f}  dg={dg:.5f}  db={db:.5f}"
              f"  mean_p@n={diag.get('best_avg_train_p_succ', float('nan')):.4e}"
              f"  elapsed={row['elapsed_s']:.1f}s")

    return trace


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--depths", default="2,5,8,10,15,20,30,40,50")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--r", type=float, default=176.54)
    ap.add_argument("--seed", type=int, default=27)
    ap.add_argument("--train-n", type=int, default=12)
    ap.add_argument("--train-size", type=int, default=50)
    ap.add_argument("--n-min", type=int, default=12)
    ap.add_argument("--n-max", type=int, default=18)
    ap.add_argument("--test-size", type=int, default=100)
    ap.add_argument("--cobyla-maxiter", type=int, default=160)
    ap.add_argument("--cobyla-restarts", type=int, default=8)
    ap.add_argument("--cobyla-perturb-scale", type=float, default=0.2)
    ap.add_argument("--grid-top-k", type=int, default=5)
    ap.add_argument("--skip-grid", action="store_true")
    ap.add_argument("--lr-beta-schedule", default="decreasing")
    ap.add_argument("--eval-train-retries", type=int, default=3)
    ap.add_argument("--output-dir", type=Path, default=None)
    ap.add_argument("--output-stem", default=None)
    ap.add_argument(
        "--from-json",
        type=Path,
        default=None,
        help="Replot PNG from a saved JSON (no re-run).",
    )
    args = ap.parse_args()

    if args.from_json is not None:
        json_path = Path(args.from_json).expanduser().resolve()
        if args.output_dir is not None:
            out_dir = Path(args.output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            stem = args.output_stem or json_path.stem
            png_path = out_dir / f"{stem}.png"
        else:
            png_path = (
                json_path.parent / f"{args.output_stem}.png"
                if args.output_stem
                else json_path.with_suffix(".png")
            )
        replot_from_json(json_path, png_path)
        print(f"Wrote {png_path}")
        return

    depths = [int(x) for x in args.depths.split(",") if x.strip()]
    n_values = list(range(args.n_min, args.n_max + 1))

    cfg = {
        "k": args.k,
        "r": args.r,
        "seed": args.seed,
        "train_n": args.train_n,
        "train_size": args.train_size,
        "n_min": args.n_min,
        "n_max": args.n_max,
        "test_size": args.test_size,
        "cobyla_maxiter": args.cobyla_maxiter,
        "cobyla_restarts": args.cobyla_restarts,
        "cobyla_perturb_scale": args.cobyla_perturb_scale,
        "grid_top_k": args.grid_top_k,
        "skip_grid": args.skip_grid,
        "skip_grid_if_warm_start": True,
        "lr_beta_schedule": args.lr_beta_schedule,
        "eval_train_retries": args.eval_train_retries,
        "walksat_p_noise": 0.5,
        "max_flips": 100_000,
    }

    out_dir = args.output_dir or bm24_runs_dir()
    run_stem = args.output_stem or make_run_stem(make_objective_comparison_kind(cfg))
    run_paths = resolve_bm24_run_output_paths(out_dir, run_stem)
    json_path = run_paths["json"]
    png_path = run_paths["png"]
    run_dir = run_paths["run_dir"]

    print(f"Output: {run_dir}")
    print(f"Stem:   {run_stem}")
    print(f"Config: k={args.k} r={args.r} seed={args.seed} train_n={args.train_n} "
          f"train_size={args.train_size} n=[{args.n_min},{args.n_max}] "
          f"test_size={args.test_size}")
    print(f"Depths: {depths}")

    # --- Build shared evaluation dataset once ---
    print(f"\nBuilding eval dataset (n={args.n_min}..{args.n_max}, test_size={args.test_size})...")
    t_ds = time.time()
    dataset = generate_eval_dataset(n_values, args.k, args.r, args.test_size, args.seed)
    print(f"  Done in {time.time()-t_ds:.1f}s")

    # --- Build shared training instances once ---
    print(f"\nBuilding training instances (n={args.train_n}, train_size={args.train_size})...")
    t_tr = time.time()
    training_instances = generate_training_instances(
        train_n=args.train_n,
        k=args.k,
        r=args.r,
        train_size=args.train_size,
        base_seed=args.seed,
    )
    print(f"  {len(training_instances)} instances at n={args.train_n} in {time.time()-t_tr:.1f}s")

    # --- Run both arms ---
    t_total = time.time()
    trace_mean = run_arm(
        training_mode="bm24_mean_p_fixed_n",
        depths=depths,
        training_instances=training_instances,
        dataset=dataset,
        n_values=n_values,
        cfg=cfg,
    )
    trace_median = run_arm(
        training_mode="median_runtime_fixed_n",
        depths=depths,
        training_instances=training_instances,
        dataset=dataset,
        n_values=n_values,
        cfg=cfg,
    )
    total_elapsed = time.time() - t_total

    # --- WalkSAT log2 slope (cheap estimate from the BM24 formula for these n) ---
    # Fit from a short walksat run; if numba unavailable, leave as NaN.
    ws_slope = float("nan")

    # --- Save JSON (re-mkdir: output tree may disappear on long runs) ---
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "stem": run_stem,
        "run_dir": str(run_dir),
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in cfg.items()},
        "depths": depths,
        "n_values": n_values,
        "walksat_log2_slope": ws_slope,
        "trace_bm24_mean_p_fixed_n": trace_mean,
        "trace_median_runtime_fixed_n": trace_median,
        "total_elapsed_s": total_elapsed,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote {json_path}")

    mean_by_d = {r["depth"]: r["lr_log2_slope"] for r in trace_mean}
    med_by_d = {r["depth"]: r["lr_log2_slope"] for r in trace_median}

    plot_objective_comparison(
        trace_mean=trace_mean,
        trace_median=trace_median,
        n_values=n_values,
        depths=depths,
        cfg=cfg,
        walksat_log2_slope=ws_slope,
        output_path=png_path,
    )
    print(f"Wrote {png_path}")

    # --- Text summary ---
    print(f"\n{'='*72}")
    print(f"SUMMARY — eval log2 slope of median(1/p_succ) vs n")
    print(f"  k={args.k}  r={args.r}  seed={args.seed}  train_n={args.train_n}")
    print(f"  n∈[{args.n_min},{args.n_max}]  train_size={args.train_size}  test_size={args.test_size}")
    print(f"{'='*72}")
    print(f"{'p':>4}  {'mean_p_fixed_n':>16}  {'median_rt_fixed_n':>18}  {'Δ(med-mean)':>13}  winner")
    for d in depths:
        ms = mean_by_d.get(d, float("nan"))
        mds = med_by_d.get(d, float("nan"))
        diff = mds - ms
        winner = "mean" if diff > 0 else ("median" if diff < 0 else "tie")
        print(f"{d:4d}  {ms:16.4f}  {mds:18.4f}  {diff:+13.4f}  {winner}")
    print(f"\nTotal elapsed: {total_elapsed:.0f}s")
    print(f"Wrote: {json_path}")
    print(f"Plot:  {png_path}")


if __name__ == "__main__":
    main()
