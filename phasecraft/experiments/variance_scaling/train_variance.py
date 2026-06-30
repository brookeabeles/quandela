"""
train_variance.py — freeze LR-QAOA angles for variance estimation.

Extracts trained (dg, db) from existing run JSONs for p in {2,5,10,20,50},
then trains fresh p=1 angles for all 4 (objective × train_n) combinations.
Writes angles_frozen.json to --out-dir.

Usage:
    python -m phasecraft.experiments.variance_scaling.train_variance [--out-dir DIR]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[3]
for _p in (_REPO_ROOT, _REPO_ROOT / "phasecraft"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from phasecraft.lib.paths import bm24_runs_dir
from phasecraft.experiments.lr_scaling.train_lr_notebook_protocol import (
    generate_training_h_diagonals_multi_n,
    generate_training_h_diagonals,
    proxy_n_values_for_training,
    train_lr_grid_search_bm24,
)

# ---------------------------------------------------------------------------
# Paths to existing run JSONs
# ---------------------------------------------------------------------------

_RUN_DIR = bm24_runs_dir() / "06-09"

_RUN1_JSON = _RUN_DIR / "run1" / "train12-tr100-te200-n12-18.json"
_RUN4_JSON = _RUN_DIR / "run4" / "train16-tr100-te200-n12-18.json"

# Objective name -> JSON trace key
_TRACE_KEYS = {
    "mean_p":    "trace_bm24_mean_p_fixed_n",
    "median_rt": "trace_median_runtime_fixed_n",
}

# Target depth grid (p=1 trained fresh, rest extracted from run JSONs)
TARGET_DEPTHS = [1, 2, 5, 10, 20, 50]

K = 8
R = 176.54
BETA_SCHEDULE = "decreasing"
ANGLE_CONVENTION = "bm24"

# Training settings for p=1
_TRAIN_SIZE = 100
_PROXY_SIZE_PER_N = 50
_PROXY_N_SPAN = 4
_PROXY_N_STEP = 2
_BASE_SEED = 42


# ---------------------------------------------------------------------------
# Extract angles from existing run JSONs
# ---------------------------------------------------------------------------

def _load_run_angles(
    json_path: Path,
    objective: str,
) -> Dict[int, Tuple[float, float]]:
    """Return {depth: (dg, db)} from a run JSON for the given objective."""
    trace_key = _TRACE_KEYS[objective]
    with open(json_path) as f:
        data = json.load(f)
    trace = data.get(trace_key, [])
    out: Dict[int, Tuple[float, float]] = {}
    for entry in trace:
        depth = int(entry["depth"])
        dg = float(entry["delta_gamma"])
        db = float(entry["delta_beta"])
        out[depth] = (dg, db)
    return out


def extract_existing_angles(
    objectives: list[str],
    train_ns: list[int],
    target_depths: list[int],
) -> Dict[str, Dict[int, Dict[int, Optional[Tuple[float, float]]]]]:
    """
    Return nested dict: objective -> train_n -> depth -> (dg, db) or None.
    p=1 is always None here; filled in by train_p1.
    """
    run_map = {12: _RUN1_JSON, 16: _RUN4_JSON}
    result: Dict[str, Dict[int, Dict[int, Optional[Tuple[float, float]]]]] = {}
    for obj in objectives:
        result[obj] = {}
        for tn in train_ns:
            angles_from_json = _load_run_angles(run_map[tn], obj)
            result[obj][tn] = {}
            for p in target_depths:
                result[obj][tn][p] = angles_from_json.get(p)
    return result


# ---------------------------------------------------------------------------
# Train fresh p=1 angles
# ---------------------------------------------------------------------------

def train_p1(
    objective: str,
    train_n: int,
    *,
    base_seed: int = _BASE_SEED,
    train_size: int = _TRAIN_SIZE,
    proxy_size_per_n: int = _PROXY_SIZE_PER_N,
    verbose: bool = True,
) -> Tuple[float, float]:
    """Train (dg, db) at depth=1 for the given objective and train_n."""
    depth = 1
    use_median = (objective == "median_rt")

    proxy_ns = proxy_n_values_for_training(
        train_n,
        proxy_n_span=_PROXY_N_SPAN,
        n_max_cap=20,
        step=_PROXY_N_STEP,
    )
    if verbose:
        print(f"  Training p=1 [{objective}, train_n={train_n}]: proxy_ns={proxy_ns}")

    proxy_h = generate_training_h_diagonals_multi_n(
        proxy_ns, k=K, r=R,
        train_size_per_n=proxy_size_per_n,
        base_seed=base_seed + 10_000 * train_n + 77,
    )
    training_h = generate_training_h_diagonals(
        train_n=train_n, k=K, r=R,
        train_size=train_size,
        base_seed=base_seed + 10_000 * train_n,
    )

    rng = np.random.default_rng(base_seed + train_n * 31 + 999)
    _, diag = train_lr_grid_search_bm24(
        training_h,
        train_n=train_n,
        depth=depth,
        proxy_h_by_n=proxy_h,
        proxy_n_values=proxy_ns,
        train_on_median=use_median,
        rng=rng,
        verbose=verbose,
    )
    dg, db = diag["best_deltas"]
    if verbose:
        print(f"  -> dg={dg:+.6f}, db={db:+.6f}")
    return float(dg), float(db)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_frozen_angles(
    objectives: list[str],
    train_ns: list[int],
    target_depths: list[int],
    *,
    base_seed: int = _BASE_SEED,
    verbose: bool = True,
) -> dict:
    """Build the full frozen angles dict, training p=1 fresh."""
    angles = extract_existing_angles(objectives, train_ns, target_depths)

    missing = []
    for obj in objectives:
        for tn in train_ns:
            for p in target_depths:
                if angles[obj][tn][p] is None:
                    missing.append((obj, tn, p))

    if missing and verbose:
        print(f"\nNeed to train {len(missing)} angle(s): {missing}")

    provenance: Dict[str, str] = {}
    run_map = {12: str(_RUN1_JSON), 16: str(_RUN4_JSON)}

    for obj in objectives:
        for tn in train_ns:
            for p in target_depths:
                key = f"{obj}|{tn}|{p}"
                if angles[obj][tn][p] is not None:
                    provenance[key] = run_map[tn]
                else:
                    if verbose:
                        print(f"\n--- Training p={p} [{obj}, train_n={tn}] ---")
                    dg, db = train_p1(obj, tn, base_seed=base_seed, verbose=verbose)
                    angles[obj][tn][p] = (dg, db)
                    provenance[key] = "trained_fresh"

    # Serialise: str keys for JSON
    angles_serial: dict = {}
    for obj in objectives:
        angles_serial[obj] = {}
        for tn in train_ns:
            angles_serial[obj][str(tn)] = {}
            for p in target_depths:
                dg, db = angles[obj][tn][p]  # type: ignore[misc]
                angles_serial[obj][str(tn)][str(p)] = {"dg": dg, "db": db}

    return {
        "config": {
            "k": K,
            "r": R,
            "beta_schedule": BETA_SCHEDULE,
            "angle_convention": ANGLE_CONVENTION,
            "objectives": objectives,
            "train_ns": train_ns,
            "target_depths": target_depths,
        },
        "angles": angles_serial,
        "provenance": provenance,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=bm24_runs_dir() / "variance_scaling",
        help="Output directory for angles_frozen.json",
    )
    parser.add_argument("--seed", type=int, default=_BASE_SEED)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    objectives = ["mean_p", "median_rt"]
    train_ns = [12, 16]
    target_depths = TARGET_DEPTHS

    verbose = not args.quiet
    if verbose:
        print("=== Variance angle freeze ===")
        print(f"Objectives: {objectives}")
        print(f"train_ns:   {train_ns}")
        print(f"Depths:     {target_depths}")

    frozen = build_frozen_angles(
        objectives, train_ns, target_depths,
        base_seed=args.seed, verbose=verbose,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "angles_frozen.json"
    with open(out_path, "w") as f:
        json.dump(frozen, f, indent=2)
    print(f"\nWrote {out_path}")

    # Quick summary
    print("\nAngles frozen (dg, db):")
    for obj in objectives:
        for tn in train_ns:
            vals = []
            for p in target_depths:
                entry = frozen["angles"][obj][str(tn)][str(p)]
                vals.append(f"p={p}:({entry['dg']:+.4f},{entry['db']:+.4f})")
            print(f"  [{obj}, n_tr={tn}]: {', '.join(vals)}")


if __name__ == "__main__":
    main()
