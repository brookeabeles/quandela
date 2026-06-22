#!/usr/bin/env python3
"""
Visual proof that p_succ(σ) is broad in log-space and mean(p) is inflated by easy instances.

Produces a multi-panel figure under --output-dir from:
  1. Per-instance p_succ on SAT-filtered training instances (fast).
  2. Aggregate mean vs median vs 1/median_rt from saved per_n re-benchmark JSON.

Example::

    python experiments/lr_scaling/plot_p_succ_spread_proof.py
    python experiments/lr_scaling/plot_p_succ_spread_proof.py --depth 20 --train-n 14
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

_PHASECRAFT = Path(__file__).resolve().parents[2]
_REPO = _PHASECRAFT.parent
_LR = Path(__file__).resolve().parent
for _p in (_REPO, _PHASECRAFT, _LR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from train_lr_fixed_n import generate_training_instances  # noqa: E402
from phasecraft.lib.sim.bm24_qaoa_sim import (  # noqa: E402
    make_lr_angles,
    per_instance_success_probability,
    run_qaoa,
)

LN2 = float(np.log(2))
EPS = 1e-300


def _load_angles(run_json: Path, depth: int) -> Tuple[float, float]:
    payload = json.loads(run_json.read_text(encoding="utf-8"))
    trace = payload.get("trace") or payload.get("trace_bm24_mean_p_fixed_n") or []
    row = next((r for r in trace if int(r["depth"]) == depth), None)
    if row is None:
        raise SystemExit(f"depth={depth} not found in {run_json}")
    return float(row["delta_gamma"]), float(row["delta_beta"])


def eval_training_ps(
    *,
    train_n: int,
    k: int,
    r: float,
    train_size: int,
    seed: int,
    depth: int,
    delta_gamma: float,
    delta_beta: float,
) -> np.ndarray:
    instances = generate_training_instances(train_n, k, r, train_size, seed)
    betas, gammas = make_lr_angles(
        delta_gamma=delta_gamma,
        delta_beta=delta_beta,
        depth=depth,
        beta_schedule="decreasing",
        angle_convention="bm24",
    )
    return np.array(
        [
            per_instance_success_probability(run_qaoa(H, betas, gammas, train_n), H)
            for H in instances
        ],
        dtype=float,
    )


def _agg_stats(ps: np.ndarray) -> Dict[str, float]:
    log_p = np.log(np.maximum(ps, EPS))
    mean_p = float(ps.mean())
    med_p = float(np.median(ps))
    gmean = float(np.exp(log_p.mean()))
    return {
        "mean_p": mean_p,
        "median_p": med_p,
        "gmean_p": gmean,
        "med_over_mean": med_p / mean_p if mean_p > 0 else float("nan"),
        "gmean_over_mean": gmean / mean_p if mean_p > 0 else float("nan"),
        "mean_log": float(log_p.mean()),
        "med_log": float(np.median(log_p)),
        "std_log": float(log_p.std()),
        "jensen_gap": float(np.log(max(mean_p, EPS)) - log_p.mean()),
        "top10_share": float(np.sort(ps)[::-1][: max(1, len(ps) // 10)].sum() / max(ps.sum(), EPS)),
    }


def load_per_n_benchmark(path: Path) -> List[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    block = data.get("results", {}).get("mean_p") or data
    return list(block["per_n"])


def plot_proof(
    *,
    ps: np.ndarray,
    depth: int,
    train_n: int,
    per_n_rows: List[dict],
    out_dir: Path,
    depths_for_bar: List[int] | None = None,
    bar_stats: Dict[int, Dict[str, float]] | None = None,
) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: List[Path] = []

    log_p = np.log(np.maximum(ps, EPS))
    stats = _agg_stats(ps)

    # --- Figure 1: four-panel proof on training instances ---
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))

    ax = axes[0, 0]
    ax.hist(log_p, bins=20, color="#4C72B0", alpha=0.85, edgecolor="white")
    ax.axvline(stats["mean_log"], color="#C44E52", lw=2, ls="--", label=f"mean(log p) = {stats['mean_log']:.2f}")
    ax.axvline(stats["med_log"], color="#55A868", lw=2, label=f"median(log p) = {stats['med_log']:.2f}")
    ax.set_xlabel("log p_succ")
    ax.set_ylabel("count")
    ax.set_title(f"Per-instance spread (n={train_n}, depth={depth}, N={len(ps)})")
    ax.legend(fontsize=9)

    ax = axes[0, 1]
  # ECDF in p-space (log x)
    ps_sorted = np.sort(ps)
    y = np.arange(1, len(ps) + 1) / len(ps)
    ax.semilogx(ps_sorted, y, color="#4C72B0", lw=2)
    ax.axvline(stats["mean_p"], color="#C44E52", ls="--", lw=2, label=f"mean = {stats['mean_p']:.4f}")
    ax.axvline(stats["median_p"], color="#55A868", lw=2, label=f"median = {stats['median_p']:.4f}")
    ax.axvline(stats["gmean_p"], color="#DD8452", lw=2, label=f"geomean = {stats['gmean_p']:.4f}")
    ax.set_xlabel("p_succ")
    ax.set_ylabel("ECDF")
    ax.set_title("Right skew: mean > median > geomean")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    labels = ["median(p)", "geomean(p)", "mean(p)"]
    vals = [stats["median_p"], stats["gmean_p"], stats["mean_p"]]
    colors = ["#55A868", "#DD8452", "#C44E52"]
    bars = ax.bar(labels, vals, color=colors)
    ax.set_ylabel("p_succ")
    ax.set_title(
        f"Typical vs BM24 objective\n"
        f"mean is {(stats['mean_p']/stats['median_p']-1)*100:.0f}% above median"
    )
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.4f}", ha="center", va="bottom", fontsize=9)

    ax = axes[1, 1]
    cum = np.cumsum(np.sort(ps)[::-1]) / ps.sum()
    ranks = np.arange(1, len(ps) + 1)
    ax.plot(ranks, cum, color="#4C72B0", lw=2)
    ax.axhline(0.9, color="k", ls=":", alpha=0.5)
    top_k = max(1, len(ps) // 10)
    ax.axvline(top_k, color="#C44E52", ls="--", alpha=0.8)
    ax.scatter([top_k], [cum[top_k - 1]], color="#C44E52", zorder=5, s=40)
    ax.set_xlabel("instance rank (easiest first)")
    ax.set_ylabel("cumulative share of Σ p_succ")
    ax.set_title(f"Top 10% carry {stats['top10_share']*100:.0f}% of total success mass")
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"p_succ distribution is broad; easy tail inflates mean(p)\n"
        f"k=8, r=176.54, seed=27 · std(log p)={stats['std_log']:.2f} · Jensen gap={stats['jensen_gap']:.2f}",
        fontsize=11,
    )
    fig.tight_layout()
    p1 = out_dir / f"p_succ_spread_proof_train_n{train_n}_d{depth}.png"
    fig.savefig(p1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved.append(p1)

    # --- Figure 2: median/mean across eval n (from saved per_n benchmark) ---
    if per_n_rows:
        ns = [int(r["n"]) for r in per_n_rows]
        spread = [float(r["median_p"]) / float(r["mean_p"]) for r in per_n_rows]
        mean_p = [float(r["mean_p"]) for r in per_n_rows]
        med_p = [float(r["median_p"]) for r in per_n_rows]
        rt_gap = [float(r["rt_over_inv_mean"]) for r in per_n_rows]

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        ax = axes[0]
        ax.plot(ns, spread, "o-", color="#4C72B0", lw=2)
        ax.axhline(1.0, color="k", ls="--", alpha=0.4)
        ax.set_xlabel("n")
        ax.set_ylabel("median(p) / mean(p)")
        ax.set_title(f"< 1 ⟹ mean(p) > median(p) (easy tail)\ndepth={depth}, test_size=200")
        ax.set_ylim(0.5, 1.05)
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        ax.semilogy(ns, mean_p, "o-", color="#C44E52", label="mean(p)")
        ax.semilogy(ns, med_p, "s-", color="#55A868", label="median(p)")
        ax.set_xlabel("n")
        ax.set_ylabel("p_succ")
        ax.set_title("Gap widens with n")
        ax.legend()
        ax.grid(True, alpha=0.3)

        fig.suptitle("Aggregate proof from held-out eval (run4, depth=20)", fontsize=11)
        fig.tight_layout()
        p2 = out_dir / "p_succ_mean_vs_median_vs_n.png"
        fig.savefig(p2, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved.append(p2)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(ns, rt_gap, "o-", color="#8172B2", lw=2)
        ax.axhline(1.0, color="k", ls="--", alpha=0.4)
        ax.set_xlabel("n")
        ax.set_ylabel("median(1/p) / (1/mean(p))")
        ax.set_title("Runtime proxy gap from mean-vs-median spread")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        p3 = out_dir / "p_succ_runtime_gap_vs_n.png"
        fig.savefig(p3, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved.append(p3)

    # --- Figure 3: mean inflation vs depth on training set ---
    if bar_stats:
        depths = sorted(bar_stats)
        infl = [(bar_stats[d]["mean_p"] / bar_stats[d]["median_p"] - 1) * 100 for d in depths]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar([str(d) for d in depths], infl, color="#DD8452")
        ax.set_xlabel("QAOA depth p")
        ax.set_ylabel("mean(p) above median(p) [%]")
        ax.set_title(f"Easy-tail inflation at train_n={train_n}")
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        p4 = out_dir / f"p_succ_mean_inflation_vs_depth_n{train_n}.png"
        fig.savefig(p4, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved.append(p4)

    summary = {
        "train_n": train_n,
        "depth_main": depth,
        "n_instances": len(ps),
        "per_instance": stats,
        "per_n_benchmark": per_n_rows,
        "bar_stats_by_depth": bar_stats,
    }
    p_json = out_dir / "p_succ_spread_proof_summary.json"
    p_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    saved.append(p_json)

    return saved


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--run-json",
        type=Path,
        default=Path("results/bm24_runs/06-10/run1/scaling-tn14.json"),
    )
    p.add_argument(
        "--per-n-json",
        type=Path,
        default=Path("results/bm24_runs/06-09/run4/train16-tr100-te200-n14-18-per_n-d20-te200.json"),
    )
    p.add_argument("--train-n", type=int, default=14)
    p.add_argument("--depth", type=int, default=20)
    p.add_argument("--train-size", type=int, default=100)
    p.add_argument("--depths-bar", type=int, nargs="*", default=[2, 5, 10, 20, 30])
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/bm24_runs/analysis/p_succ_spread_proof"),
    )
    args = p.parse_args()

    run_json = (_PHASECRAFT / args.run_json).resolve() if not args.run_json.is_absolute() else args.run_json
    per_n_json = (
        (_PHASECRAFT / args.per_n_json).resolve() if not args.per_n_json.is_absolute() else args.per_n_json
    )
    out_dir = (_PHASECRAFT / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir

    cfg = json.loads(run_json.read_text(encoding="utf-8"))["config"]
    k, r, seed = int(cfg["k"]), float(cfg["r"]), int(cfg["seed"])

    dg, db = _load_angles(run_json, args.depth)
    print(f"Evaluating {args.train_size} training instances at n={args.train_n}, depth={args.depth}...")
    ps = eval_training_ps(
        train_n=args.train_n,
        k=k,
        r=r,
        train_size=args.train_size,
        seed=seed,
        depth=args.depth,
        delta_gamma=dg,
        delta_beta=db,
    )
    s = _agg_stats(ps)
    print(
        f"  mean={s['mean_p']:.5f}  median={s['median_p']:.5f}  geomean={s['gmean_p']:.5f}  "
        f"med/mean={s['med_over_mean']:.3f}  top10%={s['top10_share']*100:.1f}%"
    )

    bar_stats: Dict[int, Dict[str, float]] = {}
    for d in args.depths_bar:
        try:
            dg_d, db_d = _load_angles(run_json, d)
        except SystemExit:
            continue
        ps_d = eval_training_ps(
            train_n=args.train_n,
            k=k,
            r=r,
            train_size=args.train_size,
            seed=seed,
            depth=d,
            delta_gamma=dg_d,
            delta_beta=db_d,
        )
        bar_stats[d] = _agg_stats(ps_d)
        print(f"  depth={d}: mean/median - 1 = {(bar_stats[d]['mean_p']/bar_stats[d]['median_p']-1)*100:.1f}%")

    per_n_rows = load_per_n_benchmark(per_n_json) if per_n_json.is_file() else []

    paths = plot_proof(
        ps=ps,
        depth=args.depth,
        train_n=args.train_n,
        per_n_rows=per_n_rows,
        out_dir=out_dir,
        bar_stats=bar_stats,
    )
    mirror = _PHASECRAFT / "bm24_runs" / "analysis" / "p_succ_spread_proof"
    if mirror.resolve() != out_dir.resolve():
        plot_proof(
            ps=ps,
            depth=args.depth,
            train_n=args.train_n,
            per_n_rows=per_n_rows,
            out_dir=mirror,
            bar_stats=bar_stats,
        )

    print("\nWrote:")
    for path in paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()
