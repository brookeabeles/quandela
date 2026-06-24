#!/usr/bin/env python3
"""
Multi-seed A/B: mean_p vs median_rt training (ctyp_vs_cann robustness check).

Runs compare_objectives_fixed_n.py for each seed with matched protocol to
train12-tr100-te200-n12-18 (train_n=12, tr100, te200, n=12..18).

Example::

    python experiments/lr_scaling/run_multi_seed_objective_ab.py \\
        --seeds 0,42,99 --depths 5,10,20

    python experiments/lr_scaling/run_multi_seed_objective_ab.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import List

_PHASECRAFT = Path(__file__).resolve().parents[2]
_LR = Path(__file__).resolve().parent
for _p in (_PHASECRAFT.parent, _PHASECRAFT, _LR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
from phasecraft.lib.paths import bm24_runs_dir  # noqa: E402

_COMPARE = _LR / "compare_objectives_fixed_n.py"
_DEFAULT_OUT = bm24_runs_dir() / "multi_seed_ctyp_cann"


def _run_one(
    *,
    seed: int,
    depths: str,
    out_dir: Path,
    train_size: int,
    test_size: int,
    dry_run: bool,
) -> dict:
    stem = f"train12-tr100-te200-n12-18-seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(_COMPARE),
        "--seed", str(seed),
        "--train-n", "12",
        "--train-size", str(train_size),
        "--test-size", str(test_size),
        "--n-min", "12",
        "--n-max", "18",
        "--depths", depths,
        "--output-dir", str(out_dir),
        "--output-stem", stem,
        "--cobyla-maxiter", "160",
        "--cobyla-restarts", "8",
    ]
    print(f"\n{'='*60}\nseed={seed}  stem={stem}\n  {' '.join(cmd)}\n", flush=True)
    if dry_run:
        return {"seed": seed, "stem": stem, "status": "dry_run", "json": None}

    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(_PHASECRAFT))
    elapsed = time.time() - t0
    json_globs = list(out_dir.rglob(f"{stem}.json"))
    json_path = json_globs[0] if json_globs else None
    status = "ok" if proc.returncode == 0 and json_path else "failed"
    return {
        "seed": seed,
        "stem": stem,
        "status": status,
        "returncode": proc.returncode,
        "elapsed_s": elapsed,
        "json": str(json_path) if json_path else None,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", default="0,42,99", help="Comma-separated seeds (27 is baseline run1)")
    p.add_argument("--depths", default="5,10,20")
    p.add_argument("--train-size", type=int, default=100)
    p.add_argument("--test-size", type=int, default=200)
    p.add_argument("--output-dir", type=Path, default=_DEFAULT_OUT)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--include-baseline-seed27", action="store_true",
                   help="Also register existing run1 JSON in manifest (no re-run).")
    args = p.parse_args()

    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    out_dir = args.output_dir.resolve()
    manifest_path = out_dir / "manifest.json"

    records: List[dict] = []
    for seed in seeds:
        records.append(_run_one(
            seed=seed,
            depths=args.depths,
            out_dir=out_dir,
            train_size=int(args.train_size),
            test_size=int(args.test_size),
            dry_run=bool(args.dry_run),
        ))

    if args.include_baseline_seed27:
        baseline = _PHASECRAFT / "bm24_runs/06-09/run1/train12-tr100-te200-n12-18.json"
        if baseline.is_file():
            records.append({
                "seed": 27,
                "stem": baseline.stem,
                "status": "baseline",
                "json": str(baseline.resolve()),
            })

    manifest = {
        "protocol": {
            "train_n": 12,
            "train_size": int(args.train_size),
            "test_size": int(args.test_size),
            "n_min": 12,
            "n_max": 18,
            "depths": [int(x) for x in args.depths.split(",") if x.strip()],
            "modes": ["bm24_mean_p_fixed_n", "median_runtime_fixed_n"],
        },
        "runs": records,
    }
    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"\nWrote {manifest_path}")

    failed = [r for r in records if r.get("status") == "failed"]
    if failed:
        print(f"\n{len(failed)} run(s) failed.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
