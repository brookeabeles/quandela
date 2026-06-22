#!/usr/bin/env python3
"""
Test whether the median_p / mean_p ratio converges as n grows.

Evaluates saved trained angles over an extended n range with large test_size,
computing mean_p, median_p, and ln(median_p/mean_p) per n. Fits a linear trend
to ln(ratio) vs n to test whether the gap between runtime and success exponents
is still drifting or has stabilised.

Usage::

    python experiments/lr_scaling/test_ratio_convergence.py \
        --depth 17 --n-max 26 --test-size 500

    # Multiple depths in one run:
    python experiments/lr_scaling/test_ratio_convergence.py \
        --depth 17 20 40 --n-max 26 --test-size 500
"""

from __future__ import annotations

import argparse
import json
import sys
import time
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
    build_h_diagonal,
    generate_benchmark_dataset,
    make_lr_angles,
    per_instance_success_probability,
    run_qaoa,
)

LN2 = float(np.log(2))

# Best trained angles indexed by depth, from run4 (train_n=16, large dataset)
# Source: results/bm24_runs/06-09/run4 (bm24_mean_p_fixed_n trace, n=12-18)
_ANGLES = {
    2:  (-1.7445, 1.2829),
    5:  (-1.6152, 1.0143),
    8:  (-1.5626, 1.0248),
    10: (-1.5490, 1.0035),
    17: (-1.5826, 1.0026),   # from depth_sweep_until_win (legacy mean-p, train_n=12)
    20: (-1.4700, 0.9137),
    40: (-1.3123, 0.9165),
    50: (-1.2764, 0.8973),
}


def fit_log2(ns: List[int], ys: List[float]) -> tuple[float, float, float]:
    """Return (slope_log2, intercept, rvalue) for log-linear fit."""
    n_arr = np.asarray(ns, dtype=float)
    y_arr = np.asarray(ys, dtype=float)
    mask = np.isfinite(y_arr) & (y_arr > 0)
    if int(mask.sum()) < 3:
        return float("nan"), float("nan"), float("nan")
    lr = linregress(n_arr[mask], np.log(y_arr[mask]))
    return float(lr.slope / LN2), float(lr.intercept), float(lr.rvalue)


def bootstrap_ratio_ci(
    per_instance_p: Dict[int, np.ndarray], n_boot: int = 500, ci: float = 0.95
) -> Dict[int, tuple[float, float]]:
    """Return {n: (lo, hi)} 95% CI on median/mean ratio by bootstrap."""
    rng = np.random.default_rng(42)
    out: Dict[int, tuple[float, float]] = {}
    for n, ps in per_instance_p.items():
        ratios = []
        for _ in range(n_boot):
            s = rng.choice(ps, size=len(ps), replace=True)
            m = float(s.mean())
            if m > 0:
                ratios.append(float(np.median(s) / m))
        if ratios:
            lo = float(np.percentile(ratios, 100 * (1 - ci) / 2))
            hi = float(np.percentile(ratios, 100 * (1 + ci) / 2))
            out[n] = (lo, hi)
    return out


def eval_per_instance(
    betas: np.ndarray,
    gammas: np.ndarray,
    dataset_n: list,
    n: int,
) -> np.ndarray:
    """Run QAOA on each instance and return array of exact p_succ values."""
    ps = []
    for inst in dataset_n:
        H = build_h_diagonal(inst["clauses"], n)
        psi = run_qaoa(H, betas, gammas, n)
        ps.append(float(per_instance_success_probability(psi, H)))
    return np.asarray(ps, dtype=float)


def run_one_depth(
    depth: int,
    delta_gamma: float,
    delta_beta: float,
    ns: List[int],
    test_sizes: Dict[int, int],
    k: int,
    r: float,
    seed: int,
) -> dict:
    betas, gammas = make_lr_angles(
        delta_gamma=delta_gamma,
        delta_beta=delta_beta,
        depth=depth,
        beta_schedule="decreasing",
        angle_convention="bm24",
    )
    betas = np.asarray(betas, dtype=float)
    gammas = np.asarray(gammas, dtype=float)

    rows: List[dict] = []
    per_instance_p: Dict[int, np.ndarray] = {}

    print(f"\n  depth={depth}  dg={delta_gamma:+.4f}  db={delta_beta:+.4f}")
    print(f"  {'n':>3} {'mean_p':>9} {'med_p':>9} {'ratio':>7} {'ln_ratio':>9} {'CI_lo':>7} {'CI_hi':>7} {'med_rt':>8} {'time':>6}")

    for n in ns:
        test_size = test_sizes[n]
        t0 = time.time()
        ds = generate_benchmark_dataset([n], k, r, test_size, seed, require_sat=True)
        inst_p = eval_per_instance(betas, gammas, ds[n], n)
        dt = time.time() - t0

        mean_p = float(inst_p.mean())
        med_p = float(np.median(inst_p))
        eps = 1e-300
        med_rt = float(np.median(1.0 / np.maximum(inst_p, eps)))

        per_instance_p[n] = inst_p

        ratio = med_p / mean_p if mean_p > 0 else float("nan")
        ln_ratio = float(np.log(ratio)) if ratio > 0 else float("nan")

        # Bootstrap CI on ratio
        cis = bootstrap_ratio_ci({n: inst_p}, n_boot=500)
        ci_lo, ci_hi = cis.get(n, (float("nan"), float("nan")))

        rows.append({
            "n": n,
            "mean_p": mean_p,
            "median_p": med_p,
            "median_rt": med_rt,
            "ratio": ratio,
            "ln_ratio": ln_ratio,
            "ci_lo": ci_lo,
            "ci_hi": ci_hi,
        })
        print(f"  {n:3d} {mean_p:9.5f} {med_p:9.5f} {ratio:7.3f} {ln_ratio:9.4f} {ci_lo:7.3f} {ci_hi:7.3f} {med_rt:8.1f} {dt:5.1f}s", flush=True)

    # Fit trends
    ns_arr = [r["n"] for r in rows]
    ln_ratios = [r["ln_ratio"] for r in rows]
    mean_ps = [r["mean_p"] for r in rows]
    med_rts = [r["median_rt"] for r in rows]

    # Trend in ln(ratio): slope ≠ 0 → gap persists; slope ≈ 0 → gap closes
    mask = [np.isfinite(x) for x in ln_ratios]
    ns_masked = [n for n, m in zip(ns_arr, mask) if m]
    lr_masked = [x for x, m in zip(ln_ratios, mask) if m]
    if len(ns_masked) >= 3:
        trend = linregress(ns_masked, lr_masked)
        trend_slope = float(trend.slope)
        trend_rval = float(trend.rvalue)
        trend_pval = float(trend.pvalue)
    else:
        trend_slope = trend_rval = trend_pval = float("nan")

    s_mean_p, _, _ = fit_log2(ns_arr, mean_ps)
    s_med_rt, _, _ = fit_log2(ns_arr, med_rts)
    exponent_gap = float(s_med_rt + s_mean_p)   # both positive since s_mean_p is negative

    # Separate fits: small-n (n<=18) vs large-n (n>18)
    small_ns = [r for r in rows if r["n"] <= 18]
    large_ns = [r for r in rows if r["n"] > 18]

    def _gap(subset):
        if len(subset) < 3:
            return float("nan"), float("nan"), float("nan")
        _ns = [r["n"] for r in subset]
        s_m, _, _ = fit_log2(_ns, [r["mean_p"] for r in subset])
        s_r, _, _ = fit_log2(_ns, [r["median_rt"] for r in subset])
        s_lr = linregress(_ns, [r["ln_ratio"] for r in subset if np.isfinite(r["ln_ratio"])])
        return float(s_r + s_m), float(s_lr.slope), float(s_lr.pvalue)

    gap_small, lr_slope_small, lr_pval_small = _gap(small_ns)
    gap_large, lr_slope_large, lr_pval_large = _gap(large_ns)

    print(f"\n  === depth={depth} summary ===")
    print(f"  mean_p slope (log2):   {-s_mean_p:+.4f}")
    print(f"  median_rt slope (log2): {s_med_rt:+.4f}")
    print(f"  exponent gap (all n):  {exponent_gap:+.4f}")
    print(f"  ln(ratio) trend slope: {trend_slope:+.5f}/n  r={trend_rval:.3f}  p={trend_pval:.4f}")
    print(f"  Interpretation: {'DRIFTING (gap persists)' if trend_pval < 0.05 else 'FLAT (gap likely closes)'}")
    if large_ns:
        print(f"  Gap (n≤18): {gap_small:+.4f}   Gap (n>{18}): {gap_large:+.4f}")
        print(f"  ln(ratio) slope large-n: {lr_slope_large:+.5f}/n  p={lr_pval_large:.4f}")

    return {
        "depth": depth,
        "delta_gamma": delta_gamma,
        "delta_beta": delta_beta,
        "rows": rows,
        "slope_mean_p_log2": s_mean_p,
        "slope_median_rt_log2": s_med_rt,
        "exponent_gap_log2": exponent_gap,
        "ln_ratio_trend_slope_per_n": trend_slope,
        "ln_ratio_trend_rvalue": trend_rval,
        "ln_ratio_trend_pvalue": trend_pval,
        "gap_small_n": gap_small,
        "gap_large_n": gap_large,
        "ln_ratio_slope_large_n": lr_slope_large,
        "ln_ratio_pvalue_large_n": lr_pval_large,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--depth", type=int, nargs="+", default=[17],
                    help="Circuit depths to test (default: 17)")
    ap.add_argument("--n-min", type=int, default=12)
    ap.add_argument("--n-max", type=int, default=24)
    ap.add_argument("--test-size", type=int, default=500,
                    help="Instances per n (can be overridden per-n with --test-size-large-n)")
    ap.add_argument("--test-size-large-n", type=int, default=None,
                    help="Use smaller test-size for n > --large-n-threshold")
    ap.add_argument("--large-n-threshold", type=int, default=18,
                    help="n above which --test-size-large-n applies (default 18)")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--r", type=float, default=176.54)
    ap.add_argument("--seed", type=int, default=27)
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    ns = list(range(args.n_min, args.n_max + 1))
    ts_small = args.test_size
    ts_large = args.test_size_large_n if args.test_size_large_n is not None else args.test_size
    thresh = args.large_n_threshold
    test_sizes = {n: ts_small if n <= thresh else ts_large for n in ns}
    print(f"Convergence test: k={args.k} r={args.r} seed={args.seed}", flush=True)
    print(f"n range: {ns}", flush=True)
    print(f"test_size: {ts_small} for n<={thresh},  {ts_large} for n>{thresh}", flush=True)

    results = []
    for depth in args.depth:
        if depth not in _ANGLES:
            print(f"WARNING: no stored angles for depth={depth}, skipping")
            continue
        dg, db = _ANGLES[depth]
        res = run_one_depth(
            depth=depth,
            delta_gamma=dg,
            delta_beta=db,
            ns=ns,
            test_sizes=test_sizes,
            k=args.k,
            r=args.r,
            seed=args.seed,
        )
        results.append(res)

    out_path = args.output or (
        Path(__file__).parent.parent.parent
        / "results" / "bm24_runs"
        / f"ratio_convergence_d{'_'.join(str(d) for d in args.depth)}_n{args.n_min}-{args.n_max}_te{args.test_size}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "k": args.k, "r": args.r, "seed": args.seed,
        "n_range": ns, "test_size": args.test_size,
        "depths_tested": args.depth,
        "results": results,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}")

    # Final verdict
    print("\n=== VERDICT ===")
    for res in results:
        slope = res["ln_ratio_trend_slope_per_n"]
        pval = res["ln_ratio_trend_pvalue"]
        gap = res["exponent_gap_log2"]
        gap_large = res["gap_large_n"]
        print(f"depth={res['depth']}: gap={gap:+.4f} log2  "
              f"ln(ratio) trend={slope:+.5f}/n  p={pval:.4f}  "
              f"→ {'PERSISTENT gap' if pval < 0.05 else 'CLOSING gap'}")
        if not np.isnan(gap_large):
            print(f"         large-n gap={gap_large:+.4f}  "
                  f"(smaller = gap shrinking with n)")


if __name__ == "__main__":
    main()
