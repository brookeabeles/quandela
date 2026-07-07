#!/usr/bin/env python3
"""
Estimate p_eff: largest LR-QAOA depth before performance is indistinguishable
from a random sampler (Montañez-Barrera et al. 2025, Sec. A.3).

Adapted to BM24 random k-SAT:
  - Performance metric r(p): mean p_succ over held-out instances (higher = better).
  - Random baseline: uniform random LR angles (delta_gamma, delta_beta) in the
    training box; threshold = mean + 3*std over ``num_random`` samples (99.73%).
  - Paper-style mixed-state baseline: uniform random bitstrings (optional).

p_eff = max depth p where trained r(p) > random threshold.

In noise-free simulation trained angles typically beat random at all depths;
use ``--depolarizing-eps > 0`` to emulate hardware decoherence.

Example::

    python experiments/lr_scaling/find_p_eff.py \\
        --run-json results/bm24_runs/06-09/run1/train12-tr100-te200-n12-18.json \\
        --num-random 100 --eval-n 12
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

_PHASECRAFT = Path(__file__).resolve().parents[2]
for _p in (_PHASECRAFT.parent, _PHASECRAFT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from phasecraft.experiments.lr_scaling.config import (  # noqa: E402
    DEFAULT_DB_BOUNDS,
    DEFAULT_DG_BOUNDS,
)
from phasecraft.experiments.lr_scaling.train_lr_fixed_n import (  # noqa: E402
    generate_training_instances,
)
from phasecraft.lib.sim.bm24_qaoa_sim import (  # noqa: E402
    build_h_diagonal,
    generate_benchmark_dataset,
    make_lr_angles,
    per_instance_success_probability,
    run_qaoa,
)


def _n_qubits(h_diag: np.ndarray) -> int:
    return int(round(np.log2(int(h_diag.size))))


def _mean_p_succ_h_list(
    h_list: Sequence[np.ndarray],
    betas: np.ndarray,
    gammas: np.ndarray,
    *,
    depolarizing_eps: float = 0.0,
) -> float:
    """Mean success probability over instances at fixed n."""
    vals: List[float] = []
    for h_diag in h_list:
        n = _n_qubits(h_diag)
        psi = run_qaoa(h_diag, betas, gammas, n)
        p = per_instance_success_probability(psi, h_diag)
        if depolarizing_eps > 0.0:
            # Simple depth-agnostic depolarizing proxy on success probability.
            p = (1.0 - depolarizing_eps) * p + depolarizing_eps * _uniform_random_p_succ(h_diag)
        vals.append(float(p))
    return float(np.mean(vals)) if vals else float("nan")


def _uniform_random_p_succ(h_diag: np.ndarray) -> float:
    """Fully-mixed measurement baseline: fraction of satisfying assignments."""
    return float(np.sum(h_diag == 0.0)) / float(h_diag.size)


def _eval_angles(
    dg: float,
    db: float,
    depth: int,
    h_list: Sequence[np.ndarray],
    *,
    beta_schedule: str = "decreasing",
    depolarizing_eps: float = 0.0,
) -> float:
    betas, gammas = make_lr_angles(
        dg, db, depth,
        beta_schedule=beta_schedule,
        angle_convention="bm24",
    )
    return _mean_p_succ_h_list(h_list, betas, gammas, depolarizing_eps=depolarizing_eps)


def random_angle_baseline(
    depth: int,
    h_list: Sequence[np.ndarray],
    *,
    num_random: int,
    seed: int,
    dg_bounds: Tuple[float, float],
    db_bounds: Tuple[float, float],
    beta_schedule: str,
    depolarizing_eps: float,
) -> dict:
    rng = np.random.default_rng(int(seed) + 10_000 * int(depth))
    dg_lo, dg_hi = dg_bounds
    db_lo, db_hi = db_bounds
    scores = np.array([
        _eval_angles(
            float(rng.uniform(dg_lo, dg_hi)),
            float(rng.uniform(db_lo, db_hi)),
            depth,
            h_list,
            beta_schedule=beta_schedule,
            depolarizing_eps=depolarizing_eps,
        )
        for _ in range(int(num_random))
    ], dtype=float)
    mu = float(np.mean(scores))
    sigma = float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0
    return {
        "scores": scores,
        "mean": mu,
        "std": sigma,
        "threshold_3sigma": mu + 3.0 * sigma,
        "max_random": float(np.max(scores)),
    }


def mixed_state_baseline(h_list: Sequence[np.ndarray]) -> float:
    return float(np.mean([_uniform_random_p_succ(h) for h in h_list]))


def load_trained_trace(run_json: Path) -> Tuple[dict, List[dict]]:
    payload = json.loads(run_json.read_text(encoding="utf-8"))
    trace = payload.get("trace_bm24_mean_p_fixed_n") or payload.get("trace") or []
    if not trace:
        raise ValueError(f"No trained trace in {run_json}")
    rows = sorted(trace, key=lambda r: int(r["depth"]))
    # Deduplicate depths (run1 has duplicate p=30).
    seen: set[int] = set()
    deduped: List[dict] = []
    for row in rows:
        p = int(row["depth"])
        if p in seen:
            continue
        seen.add(p)
        deduped.append(row)
    return payload.get("config", {}), deduped


def find_p_eff(
    depths: Sequence[int],
    trained_scores: Dict[int, float],
    thresholds: Dict[int, float],
) -> Tuple[int | None, Dict[int, bool]]:
    """Return (p_eff, pass_map). p_eff is largest depth still above threshold."""
    pass_map = {
        int(p): bool(np.isfinite(trained_scores[p]) and np.isfinite(thresholds[p])
                     and trained_scores[p] > thresholds[p])
        for p in depths
    }
    passing = [int(p) for p in depths if pass_map.get(int(p), False)]
    return (max(passing) if passing else None), pass_map


def plot_p_eff(
    *,
    depths: List[int],
    trained: List[float],
    rand_mean: List[float],
    rand_thresh: List[float],
    mixed: float,
    p_eff: int | None,
    out_path: Path,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    d = np.array(depths, dtype=float)
    ax.plot(d, trained, "o-", color="#2c7bb6", lw=2, ms=6, label="trained mean $p_{\\mathrm{succ}}$")
    ax.plot(d, rand_mean, "s--", color="#888888", lw=1.5, ms=5, alpha=0.9,
            label="random angles (mean)")
    rand_lo = [2.0 * m - t for m, t in zip(rand_mean, rand_thresh)]  # μ - 3σ
    ax.fill_between(
        d,
        rand_lo,
        rand_thresh,
        color="#d7191c",
        alpha=0.15,
        label="random 99.73% band ($\\mu \\pm 3\\sigma$)",
    )
    ax.axhline(mixed, color="#fdae61", ls=":", lw=1.5,
               label=f"uniform bitstrings ({mixed:.4g})")
    if p_eff is not None:
        ax.axvline(p_eff, color="black", ls="--", lw=1.2, alpha=0.7,
                   label=f"$p_{{\\mathrm{{eff}}}}={p_eff}$")
    ax.set_xlabel("LR-QAOA depth $p$")
    ax.set_ylabel("mean $p_{\\mathrm{succ}}$  (eval set)")
    ax.set_title(title)
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--run-json",
        type=Path,
        default=Path("results/bm24_runs/06-09/run1/train12-tr100-te200-n12-18.json"),
    )
    ap.add_argument("--eval-n", type=int, default=12, help="Fixed n for p_eff metric")
    ap.add_argument("--test-size", type=int, default=100, help="Held-out instances at eval-n")
    ap.add_argument("--num-random", type=int, default=100, help="Random angle samples per depth")
    ap.add_argument("--random-seed", type=int, default=4242)
    ap.add_argument("--depolarizing-eps", type=float, default=0.0,
                    help="Optional decoherence proxy on p_succ")
    ap.add_argument("--depth-max", type=int, default=None,
                    help="Only evaluate depths <= this value")
    ap.add_argument("--output-dir", type=Path, default=None)
    args = ap.parse_args()

    run_json = (_PHASECRAFT / args.run_json).resolve() if not args.run_json.is_absolute() else args.run_json
    cfg, trace = load_trained_trace(run_json)
    k = int(cfg.get("k", 8))
    r = float(cfg.get("r", 176.54))
    seed = int(cfg.get("seed", 27))
    train_n = int(cfg.get("train_n", 12))
    beta_schedule = str(cfg.get("lr_beta_schedule", "decreasing"))
    eval_n = int(args.eval_n)

    out_dir = args.output_dir or (_PHASECRAFT / "results/bm24_runs/analysis/p_eff")
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Building eval set: n={eval_n}, size={args.test_size}, seed={seed}...")
    dataset = generate_benchmark_dataset(
        [eval_n], k=k, r=r,
        test_size=int(args.test_size), base_seed=seed + 999_001,
        require_sat=True,
    )
    h_list = [build_h_diagonal(inst["clauses"], eval_n) for inst in dataset[eval_n]]
    mixed = mixed_state_baseline(h_list)
    print(f"  uniform bitstring baseline (fully mixed): {mixed:.6g}")

    depths = [int(row["depth"]) for row in trace]
    if args.depth_max is not None:
        depths = [p for p in depths if p <= int(args.depth_max)]
        trace = [row for row in trace if int(row["depth"]) in set(depths)]
    trained_scores: Dict[int, float] = {}
    rand_stats: Dict[int, dict] = {}

    dep = float(args.depolarizing_eps)
    for row in trace:
        p = int(row["depth"])
        dg = float(row["delta_gamma"])
        db = float(row["delta_beta"])
        trained_scores[p] = _eval_angles(
            dg, db, p, h_list,
            beta_schedule=beta_schedule,
            depolarizing_eps=dep,
        )
        rb = random_angle_baseline(
            p, h_list,
            num_random=int(args.num_random),
            seed=int(args.random_seed),
            dg_bounds=DEFAULT_DG_BOUNDS,
            db_bounds=DEFAULT_DB_BOUNDS,
            beta_schedule=beta_schedule,
            depolarizing_eps=dep,
        )
        rand_stats[p] = rb
        ok = trained_scores[p] > rb["threshold_3sigma"]
        print(
            f"  p={p:3d}  trained={trained_scores[p]:.5f}  "
            f"rand μ={rb['mean']:.5f}  +3σ={rb['threshold_3sigma']:.5f}  "
            f"{'PASS' if ok else 'FAIL'}"
        )

    p_eff, pass_map = find_p_eff(
        depths,
        trained_scores,
        {p: rand_stats[p]["threshold_3sigma"] for p in depths},
    )

    # r_eff at best passing depth (paper Eq. 8 analog on [0,1] scale)
    r_eff = float("nan")
    if p_eff is not None:
        r_max = trained_scores[p_eff]
        r_rand = rand_stats[p_eff]["threshold_3sigma"]
        denom = 1.0 - r_rand
        r_eff = (r_max - r_rand) / denom if denom > 0 else float("nan")

    summary = {
        "run_json": str(run_json),
        "eval_n": eval_n,
        "test_size": int(args.test_size),
        "num_random": int(args.num_random),
        "depolarizing_eps": dep,
        "mixed_state_baseline": mixed,
        "p_eff": p_eff,
        "r_eff_at_p_eff": r_eff,
        "depths": [
            {
                "depth": p,
                "trained_mean_p_succ": trained_scores[p],
                "random_mean": rand_stats[p]["mean"],
                "random_std": rand_stats[p]["std"],
                "random_threshold_3sigma": rand_stats[p]["threshold_3sigma"],
                "passes": pass_map[p],
                "delta_gamma": float(next(r["delta_gamma"] for r in trace if int(r["depth"]) == p)),
                "delta_beta": float(next(r["delta_beta"] for r in trace if int(r["depth"]) == p)),
            }
            for p in depths
        ],
    }

    stem = run_json.stem[:40]
    tag = "step1" if dep == 0.0 else f"step1_eps{dep:g}"
    json_path = out_dir / f"{tag}.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    plot_path = out_dir / f"{tag}.png"
    plot_p_eff(
        depths=depths,
        trained=[trained_scores[p] for p in depths],
        rand_mean=[rand_stats[p]["mean"] for p in depths],
        rand_thresh=[rand_stats[p]["threshold_3sigma"] for p in depths],
        mixed=mixed,
        p_eff=p_eff,
        out_path=plot_path,
        title=(
            f"$p_{{\\mathrm{{eff}}}}$ diagnostic  "
            f"(BM24 k-SAT, $n={eval_n}$, $k={k}$, $r={r:g}$"
            + (f", $\\varepsilon={dep:g}$" if dep > 0 else "")
            + ")"
        ),
    )

    print(f"\n{'='*60}")
    if p_eff is None:
        print("p_eff: NONE (trained never exceeds random + 3σ)")
    else:
        print(f"p_eff = {p_eff}  (largest depth with trained > random + 3σ)")
        print(f"r_eff at p_eff = {r_eff:.4f}  (paper Eq. 8 analog)")
    print(f"Wrote {json_path}")
    print(f"Wrote {plot_path}")


if __name__ == "__main__":
    main()
