#!/usr/bin/env python3
"""
Re-benchmark saved run angles with full per_n stats (mean + median success, median runtime).

Use this to compute median(p)/mean(p) and median_rt/(1/mean) for notebook run JSONs
that only stored median_runtime_per_n.

Example (run4, depth 20, full test_size — slow at large depth)::

    python experiments/lr_scaling/benchmark_run_per_n_ratios.py \\
        results/bm24_runs/06-09/run4/06-09_1912-train16-tr100-te200-n14-18.json \\
        --depth 20 --mode both

Quick smoke (test_size=50)::

    python experiments/lr_scaling/benchmark_run_per_n_ratios.py \\
        results/bm24_runs/06-09/run4/06-09_1912-train16-tr100-te200-n14-18.json \\
        --depth 10 --test-size 50 --mode both
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
from scipy.stats import linregress

_PHASECRAFT = Path(__file__).resolve().parents[2]
_REPO = _PHASECRAFT.parent
for _p in (_REPO, _PHASECRAFT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from phasecraft.lib.sim.bm24_qaoa_sim import (  # noqa: E402
    evaluate_qaoa_runtime_on_dataset,
    generate_benchmark_dataset,
    make_lr_angles,
)

LN2 = float(np.log(2))


def _traces(payload: dict) -> Dict[str, List[dict]]:
    if "traces_by_mode" in payload:
        return dict(payload["traces_by_mode"])
    out: Dict[str, List[dict]] = {}
    for key in ("trace_bm24_mean_p_fixed_n", "trace_median_runtime_fixed_n"):
        if key in payload:
            out[key.replace("trace_", "")] = list(payload[key])
    return out


def fit_log2_slope(ns: List[int], ys: List[float]) -> float:
    n_arr = np.asarray(ns, dtype=float)
    y_arr = np.asarray(ys, dtype=float)
    mask = np.isfinite(y_arr) & (y_arr > 0)
    if int(mask.sum()) < 2:
        return float("nan")
    return float(linregress(n_arr[mask], np.log(y_arr[mask])).slope / LN2)


def per_n_rows(per_n: dict) -> List[dict]:
    rows = []
    for n in sorted(int(k) for k in per_n):
        d = per_n[str(n)] if str(n) in per_n else per_n[n]
        ms = float(d["mean_success"])
        medp = float(d["median_success"])
        mrt = float(d["median_runtime"])
        invm = float(d["inverse_mean_success"])
        rows.append({
            "n": n,
            "mean_p": ms,
            "median_p": medp,
            "median_rt": mrt,
            "inverse_mean": invm,
            "spread_ratio": medp / ms if ms > 0 else float("nan"),
            "rt_over_inv_mean": mrt / invm if invm > 0 else float("nan"),
        })
    return rows


def benchmark_one(
    *,
    cfg: dict,
    depth: int,
    delta_gamma: float,
    delta_beta: float,
    dataset: dict,
) -> dict:
    betas, gammas = make_lr_angles(
        delta_gamma=float(delta_gamma),
        delta_beta=float(delta_beta),
        depth=int(depth),
        beta_schedule=cfg.get("lr_beta_schedule", "decreasing"),
        angle_convention="bm24",
    )
    ev = evaluate_qaoa_runtime_on_dataset(dataset, betas, gammas)
    rows = per_n_rows(ev["per_n"])
    ns = [r["n"] for r in rows]
    mean_p = [r["mean_p"] for r in rows]
    med_rt = [r["median_rt"] for r in rows]
    s_mean = fit_log2_slope(ns, mean_p)
    s_rt = fit_log2_slope(ns, med_rt)
    return {
        "depth": int(depth),
        "delta_gamma": float(delta_gamma),
        "delta_beta": float(delta_beta),
        "per_n": rows,
        "slope_mean_p_log2": s_mean,
        "slope_median_rt_log2": s_rt,
        "gap_log2": float(s_rt + s_mean),
        "fitted_exponents_log2": ev["fitted_exponents_log2"],
    }


def print_report(label: str, res: dict, stored_slope: float | None) -> None:
    print(f"\n{'=' * 60}\n{label}\n{'=' * 60}")
    print(
        f"  |mean_p| slope = {-res['slope_mean_p_log2']:.4f}   "
        f"median_rt slope = {res['slope_median_rt_log2']:.4f}   "
        f"gap = {res['gap_log2']:.4f}"
    )
    if stored_slope is not None and np.isfinite(stored_slope):
        print(f"  stored eval lr_log2_slope = {stored_slope:.4f}")
    print(f"\n  {'n':>3} {'mean_p':>10} {'med_p':>10} {'med/mean':>8} {'med_rt':>8} {'rt/(1/mean)':>11}")
    for r in res["per_n"]:
        print(
            f"  {r['n']:3d} {r['mean_p']:10.4f} {r['median_p']:10.4f} "
            f"{r['spread_ratio']:8.3f} {r['median_rt']:8.1f} {r['rt_over_inv_mean']:11.3f}"
        )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_json", type=Path)
    p.add_argument("--depth", type=int, required=True)
    p.add_argument(
        "--mode",
        choices=("mean_p", "median_rt", "both"),
        default="both",
        help="Which saved trace to benchmark",
    )
    p.add_argument("--test-size", type=int, default=None, help="Override config test_size")
    p.add_argument("--n-min", type=int, default=None)
    p.add_argument("--n-max", type=int, default=None)
    p.add_argument("--output", type=Path, default=None)
    args = p.parse_args()

    payload = json.loads(args.run_json.read_text(encoding="utf-8"))
    cfg = payload["config"]
    test_size = int(args.test_size if args.test_size is not None else cfg["test_size"])
    n_min = int(args.n_min if args.n_min is not None else cfg["n_min"])
    n_max = int(args.n_max if args.n_max is not None else cfg["n_max"])
    traces = _traces(payload)

    print(
        f"Dataset: n={n_min}..{n_max}, test_size={test_size}, "
        f"k={cfg['k']}, r={cfg['r']}, seed={cfg['seed']}"
    )
    dataset = generate_benchmark_dataset(
        list(range(n_min, n_max + 1)),
        int(cfg["k"]),
        float(cfg["r"]),
        test_size,
        int(cfg["seed"]),
        require_sat=True,
    )

    mode_map = {
        "mean_p": ("bm24_mean_p_fixed_n", "mean-p training angles"),
        "median_rt": ("median_runtime_fixed_n", "median-rt training angles"),
    }
    modes = list(mode_map) if args.mode == "both" else [args.mode]

    out: dict = {
        "source_run": args.run_json.stem,
        "source_path": str(args.run_json.resolve()),
        "test_size_used": test_size,
        "config": cfg,
        "results": {},
    }

    for m in modes:
        trace_key, label = mode_map[m]
        tr = traces.get(trace_key, [])
        row = next((r for r in tr if int(r["depth"]) == args.depth), None)
        if row is None:
            raise SystemExit(f"No depth={args.depth} in {trace_key}")
        res = benchmark_one(
            cfg=cfg,
            depth=args.depth,
            delta_gamma=float(row["delta_gamma"]),
            delta_beta=float(row["delta_beta"]),
            dataset=dataset,
        )
        stored = float(row.get("lr_log2_slope", float("nan")))
        res["stored_lr_log2_slope"] = stored
        out["results"][m] = res
        print_report(f"depth={args.depth} — {label}", res, stored)

    out_path = args.output or (
        args.run_json.parent / f"{args.run_json.stem}-per_n-d{args.depth}-te{test_size}.json"
    )
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
