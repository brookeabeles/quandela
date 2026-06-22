#!/usr/bin/env python3
"""
Recompute WalkSAT and WalkSATlm baselines in an existing scaling-tn14 checkpoint
using the standard bm24_qaoa_sim implementations (classic break-count WalkSAT).

Usage:
    cd /Users/b/Quandela
    python3 phasecraft/experiments/lr_scaling/fix_walksat_baseline.py \
        phasecraft/results/bm24_runs/06-10/run1/scaling-tn14.json

Writes:  <original>  (overwrites in-place, backs up to <original>.pre-fix.json)
         <dir>/scaling-tn14.png  (updated plot)
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

_PHASECRAFT = Path(__file__).resolve().parents[2]
_REPO = _PHASECRAFT.parent
for _p in (_REPO, _PHASECRAFT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from phasecraft.lib.sim.bm24_qaoa_sim import (  # noqa: E402
    evaluate_walksat_on_dataset,
    generate_benchmark_dataset,
)
from phasecraft.lib.sim.bm24_run_io import plot_scaling_vs_depth, scaling_plot_headline  # noqa: E402


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: fix_walksat_baseline.py <path/to/scaling-*.json>")
        sys.exit(1)

    json_path = Path(sys.argv[1]).resolve()
    if not json_path.exists():
        print(f"Not found: {json_path}")
        sys.exit(1)

    data = json.loads(json_path.read_text(encoding="utf-8"))
    cfg = data["config"]
    trace = data["trace"]

    k       = int(cfg["k"])
    r       = float(cfg["r"])
    seed    = int(cfg["seed"])
    n_min   = int(cfg["n_min"])
    n_max   = int(cfg["n_max"])
    ts      = int(cfg["test_size"])
    mf      = int(cfg["max_flips"])
    # BM24 plain WalkSAT = pure random flip (p_noise=1.0), no greedy component.
    # Ignore cfg["walksat_p_noise"] (which defaults to 0.5 Selman greedy).
    ws_p    = 1.0
    lm_p    = float(cfg["walksatlm_p_noise"])
    lm_w1   = int(cfg.get("walksatlm_w1", 6))
    lm_w2   = int(cfg.get("walksatlm_w2", 5))

    n_values = list(range(n_min, n_max + 1))
    print(f"Regenerating dataset: k={k} r={r} seed={seed} n={n_values} test_size={ts}")
    # generate_benchmark_dataset uses identical seeding to generate_benchmark_dataset_cached
    dataset = generate_benchmark_dataset(n_values, k, r, ts, seed, require_sat=True)
    print("  done.")

    print("Running BM24-faithful WalkSAT (pure random p_noise=1.0)...")
    ws_res = evaluate_walksat_on_dataset(
        dataset, k=k, max_flips=mf, p_noise=ws_p, seed=seed, solver="walksat"
    )
    lm_res = evaluate_walksat_on_dataset(
        dataset, k=k, max_flips=mf, p_noise=lm_p, seed=seed, solver="walksatlm",
        walksatlm_w1=lm_w1, walksatlm_w2=lm_w2,
    )

    ws_slope = float(ws_res["fitted_exponents_log2"]["median_runtime_or_flips_log_slope"])
    lm_slope = float(lm_res["fitted_exponents_log2"]["median_runtime_or_flips_log_slope"])
    print(f"  WalkSAT   slope = {ws_slope:.4f}  (was {data['setup']['ws_slope']:.4f})")
    print(f"  WalkSATlm slope = {lm_slope:.4f}  (was {data['setup']['lm_slope']:.4f})")

    # Update setup section
    data["setup"]["ws_slope"] = ws_slope
    data["setup"]["lm_slope"] = lm_slope
    data["setup"]["classical"]["walksat"]   = {str(n): v["median_flips"] for n, v in ws_res["per_n"].items()}
    data["setup"]["classical"]["walksatlm"] = {str(n): v["median_flips"] for n, v in lm_res["per_n"].items()}

    # Update every trace row
    for row in trace:
        lr = float(row["lr_log2_slope"])
        row["walksat_log2_slope"]        = ws_slope
        row["walksatlm_log2_slope"]      = lm_slope
        row["lr_beats_walksat_scaling"]  = bool(lr < ws_slope)
        row["lr_beats_walksatlm_scaling"] = bool(lr < lm_slope)

    # Back up and overwrite
    backup = json_path.with_suffix(".pre-fix.json")
    shutil.copy2(json_path, backup)
    print(f"Backed up original -> {backup.name}")

    json_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"Wrote corrected    -> {json_path.name}")

    # Re-plot
    png_path = json_path.with_suffix(".png")
    plot_scaling_vs_depth(
        trace,
        png_path,
        settings={**cfg, "k": k},
        headline=scaling_plot_headline("Efficient LR sweep", cfg.get("eval_axis", "runtime"), cfg.get("eval_aggregation", "median")),
        annotate_first_win=bool(cfg.get("annotate_first_win", False)),
    )
    print(f"Saved plot         -> {png_path.name}")

    # Print updated win status
    wins = [row["depth"] for row in trace if row.get("lr_beats_walksat_scaling")]
    print(f"\nDepths beating WalkSAT ({ws_slope:.4f}): {wins if wins else 'none yet'}")
    lm_wins = [row["depth"] for row in trace if row.get("lr_beats_walksatlm_scaling")]
    print(f"Depths beating WalkSATlm ({lm_slope:.4f}): {lm_wins if lm_wins else 'none yet'}")


if __name__ == "__main__":
    main()
