"""
evaluate_variance.py — per-instance exact p_succ at frozen LR-QAOA angles.

Reads angles_frozen.json (from train_variance.py) and the benchmark clause
caches (seed=27). For each (objective, train_n, p, n, inst_idx) writes one
JSONL row to variance_rows.jsonl:

    {"inst_idx": int, "n": int, "p": int, "objective": str, "train_n": int,
     "p_succ": float, "X": float, "unsat": bool}

X = -(1/n) log2(max(p_succ, EPS)).  Never re-optimizes angles.

Usage:
    python -m phasecraft.experiments.variance_scaling.evaluate_variance \\
        [--angles PATH] [--out-dir DIR] [--n-max INT] [--depths INT [INT ...]]
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[3]
for _p in (_REPO_ROOT, _REPO_ROOT / "phasecraft"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from phasecraft.lib.paths import bm24_runs_dir
from phasecraft.lib.sim.bm24_qaoa_sim import (
    build_h_diagonal,
    make_lr_angles,
    per_instance_success_probability,
    run_qaoa,
)

EPS = 1e-15  # p_succ floor for log computation

K = 8
R = 176.54
BETA_SCHEDULE = "decreasing"
ANGLE_CONVENTION = "bm24"
BASE_SEED = 27  # eval pool seed, must match cache files

_RUN1_DIR = bm24_runs_dir() / "06-09" / "run1"


# ---------------------------------------------------------------------------
# Clause cache loading (no dependency on lr_scaling.benchmark_dataset_cache)
# ---------------------------------------------------------------------------

def _load_clauses_gz(path: Path) -> Dict[int, List[List[List[Tuple[int, bool]]]]]:
    """
    Returns {n: [formula_clauses_0, formula_clauses_1, ...]} where each
    formula_clauses is a list of clauses (each a list of (var, neg) tuples).
    """
    with gzip.open(path, "rt", encoding="utf-8") as f:
        payload = json.load(f)
    clauses_by_n = payload["clauses_by_n"]
    result: Dict[int, List[List[List[Tuple[int, bool]]]]] = {}
    for n_key, instances in clauses_by_n.items():
        n = int(n_key)
        result[n] = [
            [
                [(int(var), bool(neg)) for var, neg in clause]
                for clause in formula_clauses
            ]
            for formula_clauses in instances
        ]
    return result


def _find_cache(n_values: Sequence[int], test_size: int) -> Optional[Path]:
    """Find the appropriate .json.gz cache file for the given n range and size."""
    n_lo, n_hi = min(n_values), max(n_values)
    name = (
        f"benchmark_clauses_seed{BASE_SEED}_k{K}"
        f"_r{R:g}_te{test_size}_n{n_lo}-{n_hi}.json.gz"
    )
    p = _RUN1_DIR / name
    return p if p.is_file() else None


def load_eval_clauses(
    n_values: Sequence[int],
    test_size: int,
) -> Dict[int, List[List[List[Tuple[int, bool]]]]]:
    """
    Load clause data for the given n values from the benchmark cache.
    Tries to find a single cache covering all n_values; falls back to
    loading individual per-n caches.
    """
    n_values = sorted(set(int(n) for n in n_values))
    cache = _find_cache(n_values, test_size)
    if cache is not None:
        data = _load_clauses_gz(cache)
        # Keep only requested n values
        return {n: data[n] for n in n_values if n in data}

    # Fall back: load each n individually from available caches
    result: Dict[int, List[List[List[Tuple[int, bool]]]]] = {}
    for n in n_values:
        cache_n = _find_cache([n], test_size)
        if cache_n is None:
            # Try the main n12-18 cache
            cache_main = _find_cache(list(range(12, 19)), 200)
            if cache_main is not None:
                data = _load_clauses_gz(cache_main)
                if n in data:
                    result[n] = data[n][:test_size]
                    continue
            raise FileNotFoundError(
                f"No clause cache found for n={n}, test_size={test_size}. "
                f"Looked in {_RUN1_DIR}"
            )
        data = _load_clauses_gz(cache_n)
        if n in data:
            result[n] = data[n][:test_size]
    return result


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def eval_instance(
    clauses: List[List[Tuple[int, bool]]],
    n: int,
    betas: np.ndarray,
    gammas: np.ndarray,
) -> Tuple[float, float, bool]:
    """
    Returns (p_succ, X, unsat) for one instance.
    X = -(1/n) log2(max(p_succ, EPS)).
    unsat=True iff no satisfying assignment exists.
    """
    h_diag = build_h_diagonal(clauses, n)
    psi = run_qaoa(h_diag, betas, gammas, n)
    p = per_instance_success_probability(psi, h_diag)
    unsat = bool(not (h_diag == 0).any())
    x = -(1.0 / n) * math.log2(max(p, EPS))
    return float(p), float(x), unsat


def _row(
    inst_idx: int,
    n: int,
    p: int,
    objective: str,
    train_n: int,
    p_succ: float,
    X: float,
    unsat: bool,
) -> str:
    return json.dumps({
        "inst_idx": inst_idx,
        "n": n,
        "p": p,
        "objective": objective,
        "train_n": train_n,
        "p_succ": p_succ,
        "X": X,
        "unsat": unsat,
    })


def run_evaluate(
    frozen: dict,
    out_path: Path,
    *,
    n_values: Sequence[int],
    test_size: int,
    depth_filter: Optional[Sequence[int]] = None,
    append: bool = False,
    verbose: bool = True,
) -> None:
    """
    Evaluate all (objective, train_n, depth/p, n, inst_idx) combinations
    and write JSONL rows to out_path.
    """
    cfg = frozen["config"]
    objectives: List[str] = cfg["objectives"]
    train_ns: List[int] = cfg["train_ns"]
    beta_schedule: str = cfg.get("beta_schedule", BETA_SCHEDULE)
    angle_convention: str = cfg.get("angle_convention", ANGLE_CONVENTION)

    angles_d = frozen["angles"]
    target_depths: List[int] = sorted(
        int(p) for p in next(iter(next(iter(angles_d.values())).values())).keys()
    )
    if depth_filter is not None:
        target_depths = [p for p in target_depths if p in set(depth_filter)]

    n_values = sorted(set(int(n) for n in n_values))

    # Load eval instances once (shared across all angle sources)
    if verbose:
        print(f"Loading eval clauses: n={n_values}, test_size={test_size}")
    clauses_by_n = load_eval_clauses(n_values, test_size)
    actual_ns = sorted(clauses_by_n.keys())
    if verbose:
        for n in actual_ns:
            print(f"  n={n}: {len(clauses_by_n[n])} instances")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    total_rows = 0
    t0 = time.time()

    with open(out_path, mode, encoding="utf-8") as fout:
        for obj in objectives:
            for tn in train_ns:
                for p_depth in target_depths:
                    angle_entry = angles_d[obj][str(tn)][str(p_depth)]
                    dg = float(angle_entry["dg"])
                    db = float(angle_entry["db"])
                    betas, gammas = make_lr_angles(
                        dg, db, p_depth,
                        beta_schedule=beta_schedule,
                        angle_convention=angle_convention,
                    )
                    if verbose:
                        print(
                            f"  [{obj}, train_n={tn}, p={p_depth}] "
                            f"dg={dg:+.5f} db={db:+.5f}"
                        )
                    for n in actual_ns:
                        inst_list = clauses_by_n[n]
                        for idx, formula_clauses in enumerate(inst_list):
                            p_succ, X, unsat = eval_instance(
                                formula_clauses, n, betas, gammas
                            )
                            fout.write(
                                _row(idx, n, p_depth, obj, tn, p_succ, X, unsat)
                                + "\n"
                            )
                            total_rows += 1

    elapsed = time.time() - t0
    if verbose:
        print(
            f"\nWrote {total_rows} rows to {out_path} "
            f"({elapsed:.1f}s, {total_rows / elapsed:.0f} rows/s)"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    default_angles = bm24_runs_dir() / "variance_scaling" / "angles_frozen.json"
    default_out = bm24_runs_dir() / "variance_scaling" / "variance_rows.jsonl"

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--angles", type=Path, default=default_angles)
    parser.add_argument("--out", type=Path, default=default_out)
    parser.add_argument(
        "--n-values", type=int, nargs="+", default=list(range(12, 17)),
        help="n values to evaluate (default 12-16)"
    )
    parser.add_argument(
        "--test-size", type=int, default=200,
        help="Instances per n (200 for main grid, 50 for hero)"
    )
    parser.add_argument(
        "--depths", type=int, nargs="+", default=None,
        help="Restrict to these depths (default: all in angles_frozen.json)"
    )
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    with open(args.angles) as f:
        frozen = json.load(f)

    run_evaluate(
        frozen,
        out_path=args.out,
        n_values=args.n_values,
        test_size=args.test_size,
        depth_filter=args.depths,
        append=args.append,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
