"""
train_lr_notebook_protocol.py
==============================

Re-implements the **LR-QAOA training protocol** from
``Final  LR QAOA vs QAOA vs walksat (1).ipynb`` (grid over ``(dg, db)`` +
COBYLA refinement on **median** ``p_succ`` at ``train_n``), but evaluates
circuits with ``phasecraft/bm24_qaoa_sim.py`` — i.e. ``run_qaoa`` and
``make_lr_angles`` in **BM24 convention** (half-angle cost and mixer).

The notebook file itself is not modified.  Optimised ``(dg, db)`` are in
**BM24 units**: pass them to the benchmark CLI as::

    --lr-dgamma ... --lr-dbeta ... --lr-angle-convention bm24

Training-set clause count matches the notebook generator when
``--m-sampling notebook`` (``m = max(1, Poisson(r n))``).  Use
``--m-sampling bm24`` for strict ``m ~ Poisson(r n)`` (may include ``m=0``).

Example (defaults aligned with typical notebook cells: n=12, 50 SAT
instances, k=8, r=176.54)::

    python phasecraft/train_lr_notebook_protocol.py --depth 5 --seed 0

Then paste the printed ``bm24_qaoa_sim.py --benchmark`` line.  Compare scaling
exponents in the report (log2 slope of ``ln(median 1/p)`` vs ``n``) to WalkSAT.

Each successful run **appends** one record to a text log (default:
``phasecraft/bm24_runs/lr_train_optimal_angles.txt``) with timestamp, deltas,
expanded betas/gammas, the suggested benchmark command, and (unless
``--skip-exponent-fit``) a fit of ``ln(mean p_succ)`` vs ``n`` over
``n in [train_n, n_max_benchmark]`` with the slope reported in **natural log**
and in **log2** (slope / ln 2).  Disable logging with ``--no-angle-log`` or set
``--angle-log PATH``.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Literal, Tuple

import numpy as np
from scipy.optimize import minimize
from scipy.stats import linregress

_SCRIPT_DIR = Path(__file__).resolve().parent
_DEFAULT_ANGLE_LOG = _SCRIPT_DIR / "bm24_runs" / "lr_train_optimal_angles_legacy.txt"
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from bm24_qaoa_sim import (  # noqa: E402
    build_h_diagonal,
    generate_benchmark_dataset,
    generate_random_clause,
    make_lr_angles,
    per_instance_success_probability,
    run_qaoa,
)

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(it, **kwargs):
        return it


def _sample_m(rng: np.random.Generator, n: int, r: float, mode: str) -> int:
    lam = float(r) * int(n)
    m = int(rng.poisson(lam))
    if mode == "notebook":
        return max(1, m)
    if mode == "bm24":
        return m
    raise ValueError("m_sampling must be 'notebook' or 'bm24'")


def generate_training_h_diagonals(
    train_n: int,
    k: int,
    r: float,
    train_size: int,
    base_seed: int,
    m_sampling: Literal["notebook", "bm24"] = "notebook",
) -> List[np.ndarray]:
    """
    SAT-filtered random k-SAT at fixed n, same *logic* as the notebook's
    ``generate_training_set`` (keep instances with >=1 solution).
    Returns precomputed ``H_diag`` arrays for speed.
    """
    h_list: List[np.ndarray] = []
    trial = 0
    while len(h_list) < int(train_size):
        ss = np.random.SeedSequence([int(base_seed), int(train_n), len(h_list), trial])
        rng = np.random.default_rng(ss)
        m = _sample_m(rng, train_n, r, m_sampling)
        clauses = [generate_random_clause(train_n, k, rng) for _ in range(m)]
        h_diag = build_h_diagonal(clauses, train_n)
        if np.any(h_diag == 0):
            h_list.append(h_diag)
        trial += 1
        if trial > 1_000_000:
            raise RuntimeError("Exceeded trial limit building training set")
    return h_list


def train_lr_grid_search_bm24(
    training_h: List[np.ndarray],
    train_n: int,
    depth: int,
    *,
    skip_grid: bool = False,
    initial_deltas: Tuple[float, float] | None = None,
    beta_schedule: str = "decreasing",
    cobyla_maxiter: int = 200,
    cobyla_tol: float = 1e-3,
) -> Tuple[np.ndarray, dict]:
    """
    Same control flow as ``train_lr_grid_search`` in the notebook:
      1) optional 11x11 grid on dg in [-2,2], db in [0.1, 4]
      2) COBYLA refine starting from best grid point (or default [-0.8, 0.49])

    Returns ``(params_concat, diagnostics)`` where ``params_concat`` matches
    the notebook layout ``np.concatenate([gammas, betas])`` with angles in
    **BM24** units (from ``make_lr_angles``).
    """
    dg_vals = np.linspace(-2.0, 2.0, 11)
    db_vals = np.linspace(0.1, 4.0, 11)
    best_p = -1.0
    best_deltas = list(initial_deltas) if initial_deltas is not None else [-0.8, 0.49]

    def per_instance_p_succ(dg: float, db: float) -> np.ndarray:
        betas, gammas = make_lr_angles(
            float(dg), float(db), int(depth),
            beta_schedule=str(beta_schedule),
            angle_convention="bm24",
        )
        ps = np.empty(len(training_h), dtype=np.float64)
        for i, h_diag in enumerate(training_h):
            psi = run_qaoa(h_diag, betas, gammas, int(train_n))
            ps[i] = float(per_instance_success_probability(psi, h_diag))
        return ps

    def get_median_p_succ(dg: float, db: float) -> float:
        ps = per_instance_p_succ(dg, db)
        return float(np.median(ps)) if len(ps) else 0.0

    def get_mean_p_succ(dg: float, db: float) -> float:
        ps = per_instance_p_succ(dg, db)
        return float(np.mean(ps)) if len(ps) else 0.0

    if not skip_grid:
        for dg in tqdm(dg_vals, desc="Grid Scanning (dg)"):
            for db in db_vals:
                p_val = get_median_p_succ(float(dg), float(db))
                if p_val > best_p:
                    best_p = p_val
                    best_deltas = [float(dg), float(db)]
    else:
        best_p = get_median_p_succ(best_deltas[0], best_deltas[1])

    print(
        f"  > Best grid start: dg={best_deltas[0]:.4f}, db={best_deltas[1]:.4f} "
        f"(median train p_succ={best_p:.6f})"
    )

    def objective(d: np.ndarray) -> float:
        return -1.0 * get_median_p_succ(float(d[0]), float(d[1]))

    res = minimize(
        objective,
        np.asarray(best_deltas, dtype=float),
        method="COBYLA",
        tol=float(cobyla_tol),
        options={"maxiter": int(cobyla_maxiter)},
    )
    dg_opt, db_opt = float(res.x[0]), float(res.x[1])
    best_median = float(-res.fun)
    best_mean = get_mean_p_succ(dg_opt, db_opt)
    print(f"  > LR QAOA train median p_succ after COBYLA: {best_median:.6f}")
    print(f"  > LR QAOA train mean p_succ (diagnostic):   {best_mean:.6f}")

    betas, gammas = make_lr_angles(
        dg_opt, db_opt, int(depth),
        beta_schedule=str(beta_schedule),
        angle_convention="bm24",
    )
    params_concat = np.concatenate([gammas, betas])
    diag = {
        "best_deltas": [dg_opt, db_opt],
        "best_median_train_p_succ": best_median,
        "best_avg_train_p_succ": best_mean,
        "training_objective": "legacy-median-p-train-n",
        "cobyla_success": bool(res.success),
        "cobyla_message": str(res.message),
        "nfev": int(getattr(res, "nfev", -1)),
    }
    return params_concat, diag


def compute_mean_success_exponent_vs_n(
    n_min: int,
    n_max: int,
    k: int,
    r: float,
    test_size: int,
    base_seed: int,
    betas: np.ndarray,
    gammas: np.ndarray,
) -> dict:
    """
    For each n in [n_min, n_max], mean p_succ on a fresh SAT-filtered dataset
    (same generator as ``bm24_qaoa_sim.generate_benchmark_dataset``), then
    least-squares fit::

        ln(mean p_succ(n)) = slope_nat * n + intercept

    Returns ``slope_nat`` and ``slope_log2 = slope_nat / ln(2)`` (bits / var
    for the log-base-2 mean success exponent).
    """
    n_min, n_max = int(n_min), int(n_max)
    if n_max < n_min:
        return {"ok": False, "reason": "n_max < n_min"}
    n_values = list(range(n_min, n_max + 1))
    if len(n_values) < 2:
        return {"ok": False, "reason": "need at least two n values for a slope"}

    dataset = generate_benchmark_dataset(
        n_values=n_values,
        k=int(k),
        r=float(r),
        test_size=int(test_size),
        base_seed=int(base_seed),
        require_sat=True,
    )
    means_list: List[float] = []
    for n in n_values:
        tot = 0.0
        insts = dataset[int(n)]
        for inst in insts:
            H = build_h_diagonal(inst["clauses"], int(n))
            psi = run_qaoa(H, betas, gammas, int(n))
            tot += float(per_instance_success_probability(psi, H))
        means_list.append(tot / max(len(insts), 1))

    means = np.asarray(means_list, dtype=float)
    ns_arr = np.asarray(n_values, dtype=float)
    mask = means > 0.0
    if int(np.sum(mask)) < 2:
        return {
            "ok": False,
            "reason": "fewer than 2 n with positive mean p_succ",
            "mean_p_succ_per_n": {int(n_values[i]): float(means[i]) for i in range(len(n_values))},
        }
    res = linregress(ns_arr[mask], np.log(means[mask]))
    slope_nat = float(res.slope)
    return {
        "ok": True,
        "mean_p_succ_per_n": {int(n_values[i]): float(means[i]) for i in range(len(n_values))},
        "slope_ln_mean_p_succ_vs_n_natural_log": slope_nat,
        "slope_ln_mean_p_succ_vs_n_log2": float(slope_nat / np.log(2.0)),
        "intercept_ln_mean_p_succ_natural_log": float(res.intercept),
        "stderr_slope_ln_nat": float(res.stderr) if res.stderr is not None else float("nan"),
        "eval_n_min": n_min,
        "eval_n_max": n_max,
        "eval_test_size": int(test_size),
        "eval_base_seed": int(base_seed),
    }


def append_optimal_angles_log(
    path: Path,
    *,
    timestamp: str,
    settings: dict,
    dg: float,
    db: float,
    betas: np.ndarray,
    gammas: np.ndarray,
    best_median_train_p_succ: float,
    best_avg_train_p_succ: float,
    benchmark_cmd: str,
    exponent_fit: dict | None = None,
) -> None:
    """Append one run record; creates parent directories if needed."""
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    sep = "-" * 78
    lines = [
        sep,
        f"timestamp: {timestamp}",
        f"settings: {json.dumps(settings, sort_keys=True)}",
        f"delta_gamma (BM24, --lr-dgamma): {dg:+.16g}",
        f"delta_beta  (BM24, --lr-dbeta):  {db:+.16g}",
        f"betas_bm24:  {np.round(betas, 12).tolist()}",
        f"gammas_bm24: {np.round(gammas, 12).tolist()}",
        f"best_median_train_p_succ: {best_median_train_p_succ:.12g}",
        f"best_avg_train_p_succ: {best_avg_train_p_succ:.12g}",
        f"suggested_benchmark_command: {benchmark_cmd}",
    ]
    if exponent_fit is not None:
        lines.append(f"exponent_fit: {json.dumps(exponent_fit, sort_keys=True)}")
        if exponent_fit.get("ok"):
            lines.append(
                "mean_success_exponent_slope_natural_log (d/dn ln E[p_succ]): "
                f"{exponent_fit['slope_ln_mean_p_succ_vs_n_natural_log']:+.12g}"
            )
            lines.append(
                "mean_success_exponent_slope_log2 ( / ln 2 ): "
                f"{exponent_fit['slope_ln_mean_p_succ_vs_n_log2']:+.12g}"
            )
        else:
            lines.append(f"mean_success_exponent: unavailable ({exponent_fit.get('reason')})")
    lines.append("")
    with open(path, "a", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _benchmark_cli_snippet(
    train_n: int,
    n_max: int,
    k: int,
    r: float,
    dg: float,
    db: float,
    depth: int,
    test_size: int,
    seed: int,
    beta_schedule: str,
) -> str:
    return (
        f'{sys.executable} "{Path(__file__).resolve().parent / "bm24_qaoa_sim.py"}" '
        f"--benchmark "
        f"--n-min {train_n} --n-max {n_max} --k {k} --r {r} "
        f"--test-size {test_size} --seed {seed} "
        f"--algorithms lr_qaoa walksat walksatlm "
        f"--depth {depth} "
        f"--lr-dgamma {dg} --lr-dbeta {db} "
        f"--lr-beta-schedule {beta_schedule} "
        f"--lr-angle-convention bm24"
    )


def main() -> None:
    p = argparse.ArgumentParser(
        description="Notebook LR training protocol + bm24_qaoa_sim.run_qaoa evaluation.",
    )
    p.add_argument("--train-n", type=int, default=12, help="Fixed n for training (notebook default 12).")
    p.add_argument("--train-size", type=int, default=50, help="Number of SAT training instances.")
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--r", type=float, default=176.54)
    p.add_argument("--depth", type=int, required=True, help="QAOA depth p (same as notebook P_DEPTH).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--skip-grid", action="store_true", help="Skip 11x11 grid; start COBYLA from --initial-dgamma/--initial-dbeta or [-0.8, 0.49].")
    p.add_argument("--initial-dgamma", type=float, default=None)
    p.add_argument("--initial-dbeta", type=float, default=None)
    p.add_argument(
        "--m-sampling",
        type=str,
        default="notebook",
        choices=["notebook", "bm24"],
        help="notebook: m=max(1,Poisson(rn)) like the notebook training cell; bm24: strict Poisson.",
    )
    p.add_argument(
        "--lr-beta-schedule",
        type=str,
        default="decreasing",
        choices=["decreasing", "increasing"],
    )
    p.add_argument("--cobyla-maxiter", type=int, default=200)
    p.add_argument("--cobyla-tol", type=float, default=1e-3)
    p.add_argument("--n-max-benchmark", type=int, default=21,
                   help="Upper n for suggested benchmark (notebook TEST_N_RANGE ends at 21).")
    p.add_argument("--test-size-benchmark", type=int, default=100)
    p.add_argument("--save-json", type=str, default=None)
    p.add_argument(
        "--angle-log",
        type=str,
        default=str(_DEFAULT_ANGLE_LOG),
        help="Append optimal angles to this .txt file after each run (default: bm24_runs/lr_train_optimal_angles.txt).",
    )
    p.add_argument(
        "--no-angle-log",
        action="store_true",
        help="Do not append to the angle log file.",
    )
    p.add_argument(
        "--eval-exponent-test-size",
        type=int,
        default=30,
        help="Instances per n when fitting ln(mean p_succ) vs n for the log2 exponent.",
    )
    p.add_argument(
        "--eval-exponent-seed",
        type=int,
        default=None,
        help="Seed for exponent-fit dataset (default: --seed + 1_000_003).",
    )
    p.add_argument(
        "--skip-exponent-fit",
        action="store_true",
        help="Skip ln(mean p_succ) vs n fit (faster).",
    )
    args = p.parse_args()

    print(f"Building training set: n={args.train_n}, size={args.train_size}, k={args.k}, r={args.r}, "
          f"m_sampling={args.m_sampling}, seed={args.seed}")
    training_h = generate_training_h_diagonals(
        train_n=args.train_n,
        k=args.k,
        r=args.r,
        train_size=args.train_size,
        base_seed=args.seed,
        m_sampling=args.m_sampling,
    )
    print(f"  collected {len(training_h)} SAT-filtered instances.")

    initial_deltas = None
    if args.initial_dgamma is not None and args.initial_dbeta is not None:
        initial_deltas = (float(args.initial_dgamma), float(args.initial_dbeta))
    elif args.initial_dgamma is not None or args.initial_dbeta is not None:
        raise SystemExit("Provide both --initial-dgamma and --initial-dbeta, or neither.")

    print("\n--- Training LR QAOA (notebook protocol, BM24 simulator) ---")
    params, diag = train_lr_grid_search_bm24(
        training_h,
        train_n=args.train_n,
        depth=args.depth,
        skip_grid=args.skip_grid,
        initial_deltas=initial_deltas,
        beta_schedule=args.lr_beta_schedule,
        cobyla_maxiter=args.cobyla_maxiter,
        cobyla_tol=args.cobyla_tol,
    )
    depth = int(args.depth)
    gammas = params[:depth]
    betas = params[depth:]
    dg, db = diag["best_deltas"]

    print("\n" + "=" * 72)
    print("BEST (BM24 convention — use with --lr-angle-convention bm24)")
    print("=" * 72)
    print(f"  delta_gamma (for bm24_qaoa_sim --lr-dgamma): {dg:+.10f}")
    print(f"  delta_beta  (for bm24_qaoa_sim --lr-dbeta):  {db:+.10f}")
    print(f"  betas  (length {depth}): {np.round(betas, 6).tolist()}")
    print(f"  gammas (length {depth}): {np.round(gammas, 6).tolist()}")

    cmd = _benchmark_cli_snippet(
        train_n=args.train_n,
        n_max=args.n_max_benchmark,
        k=args.k,
        r=args.r,
        dg=dg,
        db=db,
        depth=depth,
        test_size=args.test_size_benchmark,
        seed=args.seed,
        beta_schedule=args.lr_beta_schedule,
    )
    print("\nSuggested full benchmark (copy-paste):")
    print(cmd)

    eval_seed = (
        int(args.eval_exponent_seed)
        if args.eval_exponent_seed is not None
        else int(args.seed) + 1_000_003
    )
    exponent_fit: dict | None = None
    if not args.skip_exponent_fit and int(args.n_max_benchmark) >= int(args.train_n):
        print("\n--- Mean success exponent (fit ln E[p_succ] vs n) ---")
        exponent_fit = compute_mean_success_exponent_vs_n(
            n_min=args.train_n,
            n_max=args.n_max_benchmark,
            k=args.k,
            r=args.r,
            test_size=args.eval_exponent_test_size,
            base_seed=eval_seed,
            betas=betas,
            gammas=gammas,
        )
        if exponent_fit.get("ok"):
            print(f"  mean_p_succ per n: {exponent_fit['mean_p_succ_per_n']}")
            print(
                "  slope d/dn ln(mean p_succ) [natural log] = "
                f"{exponent_fit['slope_ln_mean_p_succ_vs_n_natural_log']:+.8f}"
            )
            print(
                "  same slope in log2 units (divide by ln 2)     = "
                f"{exponent_fit['slope_ln_mean_p_succ_vs_n_log2']:+.8f}"
            )
        else:
            print(f"  (skipped or failed: {exponent_fit.get('reason')})")
            if "mean_p_succ_per_n" in exponent_fit:
                print(f"  mean_p_succ per n: {exponent_fit['mean_p_succ_per_n']}")

    out = {
        "settings": {
            "train_n": args.train_n,
            "train_size": args.train_size,
            "k": args.k,
            "r": args.r,
            "depth": depth,
            "seed": args.seed,
            "skip_grid": bool(args.skip_grid),
            "m_sampling": args.m_sampling,
            "lr_beta_schedule": args.lr_beta_schedule,
            "cobyla_maxiter": args.cobyla_maxiter,
            "cobyla_tol": args.cobyla_tol,
            "angle_convention": "bm24",
            "objective": "legacy-median-p-train-n",
            "initial_dgamma": args.initial_dgamma,
            "initial_dbeta": args.initial_dbeta,
        },
        "best_deltas": {"delta_gamma": dg, "delta_beta": db},
        "betas_bm24": [float(x) for x in betas],
        "gammas_bm24": [float(x) for x in gammas],
        "params_concat_gammas_then_betas": [float(x) for x in params],
        "diagnostics": diag,
        "suggested_benchmark_command": cmd,
        "mean_success_exponent_fit": exponent_fit,
    }
    if args.save_json:
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"\nWrote JSON to {args.save_json}")

    if not args.no_angle_log:
        log_path = Path(args.angle_log)
        settings_compact = {
            "train_n": args.train_n,
            "train_size": args.train_size,
            "k": args.k,
            "r": args.r,
            "depth": depth,
            "seed": args.seed,
            "skip_grid": bool(args.skip_grid),
            "m_sampling": args.m_sampling,
            "lr_beta_schedule": args.lr_beta_schedule,
            "cobyla_maxiter": args.cobyla_maxiter,
            "cobyla_tol": args.cobyla_tol,
            "objective": "legacy-median-p-train-n",
            "initial_dgamma": args.initial_dgamma,
            "initial_dbeta": args.initial_dbeta,
        }
        append_optimal_angles_log(
            log_path,
            timestamp=datetime.now().isoformat(timespec="seconds"),
            settings=settings_compact,
            dg=dg,
            db=db,
            betas=betas,
            gammas=gammas,
            best_median_train_p_succ=float(diag["best_median_train_p_succ"]),
            best_avg_train_p_succ=float(diag["best_avg_train_p_succ"]),
            benchmark_cmd=cmd,
            exponent_fit=exponent_fit,
        )
        print(f"\nAppended optimal angles to: {log_path.resolve()}")


if __name__ == "__main__":
    main()
