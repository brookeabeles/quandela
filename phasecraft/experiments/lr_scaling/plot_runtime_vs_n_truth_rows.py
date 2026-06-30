#!/usr/bin/env python3
"""
Median runtime vs n, coloured by # satisfying assignments (true rows in truth table).

Uses BM24 gap-audit eval CSV rows (per-instance p_succ at frozen angles).  For each
instance we count ``num_solutions = sum(H_diag == 0)`` — the number of true rows in
the energy / truth table — and plot runtime proxy ``1 / p_succ`` vs problem size ``n``.

Two plot styles (``--style``):
  * ``instances`` — one trajectory per eval instance (matches the reference figure).
  * ``bucket_median`` — one curve per true-row count, y = median runtime in that bucket.

Example::

    python experiments/lr_scaling/plot_runtime_vs_n_truth_rows.py
    python experiments/lr_scaling/plot_runtime_vs_n_truth_rows.py \\
        --csv results/bm24_runs/bm24_gap_audit/prelim_n12-16_N100.csv \\
        --depth 20 --train-n 12 --style instances
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

_PHASECRAFT = Path(__file__).resolve().parents[2]
_REPO = _PHASECRAFT.parent
_LR = Path(__file__).resolve().parent
for _p in (_REPO, _PHASECRAFT, _LR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from phasecraft.experiments.lr_scaling.bm24_gap_audit import (  # noqa: E402
    build_eval_instances_with_audit,
)

EPS = 1e-300


def _load_meta(csv_path: Path) -> dict:
    meta_path = csv_path.with_suffix(".meta.json")
    if meta_path.is_file():
        return json.loads(meta_path.read_text(encoding="utf-8"))
    return {}


def _true_rows_by_eval_index(
    *,
    n_values: Iterable[int],
    k: int,
    r: float,
    count: int,
    base_seed: int,
    m_sampling: str,
) -> Dict[Tuple[int, int], int]:
    """Bulk-build eval instances per n; return {(n, eval_index): num_true_rows}."""
    out: Dict[Tuple[int, int], int] = {}
    for n in sorted({int(v) for v in n_values}):
        print(f"  counting true rows for n={n}, N={count}...", flush=True)
        instances, _ = build_eval_instances_with_audit(
            n=n,
            k=k,
            r=r,
            count=count,
            base_seed=base_seed,
            require_sat=True,
            m_sampling=m_sampling,
            max_trials=1_000_000,
        )
        for inst in instances:
            out[(int(n), int(inst.index))] = int(np.sum(inst.h_diag == 0))
    return out


def load_instance_rows(
    csv_path: Path,
    *,
    depth: int,
    objective: str,
    train_n: int,
) -> List[dict]:
    rows: List[dict] = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if (
                int(row["p"]) == int(depth)
                and row["objective"] == objective
                and int(row["train_n"]) == int(train_n)
            ):
                rows.append(row)
    if not rows:
        raise SystemExit(
            f"No rows in {csv_path} for p={depth}, objective={objective}, train_n={train_n}"
        )
    return rows


def enrich_with_true_rows(
    rows: Iterable[dict],
    meta: dict,
    *,
    true_rows: Dict[Tuple[int, int], int] | None = None,
) -> List[dict]:
    rows = list(rows)
    k = int(meta.get("k", 8))
    r = float(meta.get("r", 176.54))
    base_seed = int(meta.get("eval_seed", 100_027))
    m_sampling = str(meta.get("m_sampling", "notebook"))
    count = int(meta.get("N", max(int(r["eval_index"]) for r in rows) + 1))
    if true_rows is None:
        n_values = sorted({int(r["n"]) for r in rows})
        true_rows = _true_rows_by_eval_index(
            n_values=n_values,
            k=k,
            r=r,
            count=count,
            base_seed=base_seed,
            m_sampling=m_sampling,
        )
    out: List[dict] = []
    for row in rows:
        n = int(row["n"])
        eval_index = int(row["eval_index"])
        trial = int(row["trial"])
        p_succ = float(row["p_succ"])
        key = (n, eval_index)
        if key not in true_rows:
            raise KeyError(f"missing true-row count for n={n}, eval_index={eval_index}")
        out.append(
            {
                "n": n,
                "eval_index": eval_index,
                "trial": trial,
                "p_succ": p_succ,
                "runtime": float(1.0 / max(p_succ, EPS)),
                "num_true_rows": int(true_rows[key]),
            }
        )
    return out


def bucket_median_curves(rows: List[dict]) -> Dict[int, Dict[int, float]]:
    """Return {num_true_rows: {n: median_runtime}}."""
    buckets: Dict[int, Dict[int, List[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        buckets[int(row["num_true_rows"])][int(row["n"])].append(float(row["runtime"]))
    return {
        k: {n: float(np.median(vs)) for n, vs in sorted(by_n.items())}
        for k, by_n in sorted(buckets.items())
    }


def instance_trajectories(rows: List[dict]) -> Dict[int, List[dict]]:
    by_idx: Dict[int, List[dict]] = defaultdict(list)
    for row in rows:
        by_idx[int(row["eval_index"])].append(row)
    for idx in by_idx:
        by_idx[idx].sort(key=lambda r: int(r["n"]))
    return dict(by_idx)


def plot_truth_rows_runtime(
    rows: List[dict],
    *,
    out_path: Path,
    style: str,
    color_max: int,
    title: str | None = None,
) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmap = plt.cm.viridis
    norm = plt.Normalize(vmin=0, vmax=max(1, int(color_max)))

    fig, ax = plt.subplots(figsize=(8, 6))

    if style == "bucket_median":
        curves = bucket_median_curves(rows)
        for k, by_n in curves.items():
            ns = sorted(by_n)
            rts = [by_n[n] for n in ns]
            color = cmap(norm(k))
            ax.semilogy(ns, rts, "o-", color=color, lw=1.8, ms=5, label=f"k={k}")
    elif style == "instances":
        trajectories = instance_trajectories(rows)
        for _idx, pts in trajectories.items():
            ns = [int(p["n"]) for p in pts]
            rts = [float(p["runtime"]) for p in pts]
            # Colour each instance line by its true-row count at the smallest n.
            k = int(pts[0]["num_true_rows"])
            color = cmap(norm(k))
            ax.semilogy(ns, rts, "o-", color=color, lw=1.2, ms=3, alpha=0.85)
    else:
        raise ValueError(f"unknown style={style!r}")

    ax.set_xlabel(r"$n$", fontsize=16)
    ax.set_ylabel("Median runtime", fontsize=16)
    if title:
        ax.set_title(title, fontsize=13)
    ns_all = sorted({int(r["n"]) for r in rows})
    ax.set_xticks(ns_all)
    ax.tick_params(labelsize=12)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label("# of true rows in truth table", fontsize=14, rotation=270, labelpad=22)
    cbar.ax.tick_params(labelsize=11)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main(argv: List[str] | None = None) -> Path:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--csv",
        type=Path,
        default=Path("results/bm24_runs/bm24_gap_audit/prelim_n12-16_N100.csv"),
        help="Gap-audit eval CSV with per-instance p_succ rows.",
    )
    p.add_argument("--depth", type=int, default=20)
    p.add_argument("--objective", default="mean_p")
    p.add_argument("--train-n", type=int, default=12)
    p.add_argument(
        "--style",
        choices=("instances", "bucket_median"),
        default="instances",
    )
    p.add_argument(
        "--color-max",
        type=int,
        default=7,
        help="Colorbar upper bound (reference figure uses 0–7).",
    )
    p.add_argument("--title", default=None, help="Optional plot title.")
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PNG path (default: alongside CSV).",
    )
    args = p.parse_args(argv)

    csv_path = args.csv.expanduser().resolve()
    meta = _load_meta(csv_path)
    raw = load_instance_rows(
        csv_path,
        depth=args.depth,
        objective=args.objective,
        train_n=args.train_n,
    )
    print(f"Loaded {len(raw)} rows from {csv_path}")
    enriched = enrich_with_true_rows(raw, meta)
    ns = sorted({int(r["n"]) for r in enriched})
    kvals = sorted({int(r["num_true_rows"]) for r in enriched})
    print(f"n values: {ns}")
    print(f"true-row counts present: {kvals}")

    out_path = (
        args.output.expanduser().resolve()
        if args.output
        else csv_path.with_name(
            f"runtime_vs_n_truth_rows_p{args.depth}_tn{args.train_n}_{args.style}.png"
        )
    )
    title = args.title
    if title is None and args.style == "bucket_median":
        title = (
            f"LR-QAOA depth p={args.depth}, train n={args.train_n}, "
            f"objective={args.objective}"
        )
    saved = plot_truth_rows_runtime(
        enriched,
        out_path=out_path,
        style=args.style,
        color_max=args.color_max,
        title=title,
    )
    print(f"Wrote {saved}")
    return saved


if __name__ == "__main__":
    main()
