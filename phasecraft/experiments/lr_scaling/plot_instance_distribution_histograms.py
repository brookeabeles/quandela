#!/usr/bin/env python3
"""
Histograms of per-instance p_succ and runtime (1/p_succ) as n and/or p increase.

Uses BM24 gap-audit CSV rows (instance-level p_succ at frozen LR angles).
Marks aggregate mean(p_succ) and median(1/p_succ) on each panel — the quantities
that enter the annealed (c_ann) and typical-runtime (c_typ) exponents.

Example::

    python experiments/lr_scaling/plot_instance_distribution_histograms.py
    python experiments/lr_scaling/plot_instance_distribution_histograms.py \\
        --depth 20 --train-n 12 --fixed-n 16
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "phasecraft-mpl"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_PHASECRAFT = Path(__file__).resolve().parents[2]
_REPO = _PHASECRAFT.parent
_LR = Path(__file__).resolve().parent
for _p in (_REPO, _PHASECRAFT, _LR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

EPS = 1e-300
LN2 = float(np.log(2))


def load_rows(paths: Sequence[Path]) -> List[dict]:
    rows: List[dict] = []
    for path in paths:
        with Path(path).open(newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                p_succ = float(r["p_succ"])
                if not (math.isfinite(p_succ) and p_succ > 0):
                    continue
                rows.append(
                    {
                        "n": int(r["n"]),
                        "p": int(r["p"]),
                        "objective": str(r["objective"]),
                        "train_n": int(r["train_n"]),
                        "p_succ": p_succ,
                    }
                )
    return rows


def filter_rows(
    rows: Sequence[dict],
    *,
    objective: str,
    train_n: int,
    depth: int | None = None,
    fixed_n: int | None = None,
) -> List[dict]:
    out: List[dict] = []
    for r in rows:
        if r["objective"] != objective or r["train_n"] != train_n:
            continue
        if depth is not None and r["p"] != depth:
            continue
        if fixed_n is not None and r["n"] != fixed_n:
            continue
        out.append(r)
    return out


def group_by(rows: Sequence[dict], key: str) -> Dict[int, np.ndarray]:
    buckets: Dict[int, list] = defaultdict(list)
    for r in rows:
        buckets[int(r[key])].append(r["p_succ"])
    return {k: np.array(v, dtype=float) for k, v in sorted(buckets.items())}


def group_by_np(rows: Sequence[dict]) -> Dict[Tuple[int, int], np.ndarray]:
    buckets: Dict[Tuple[int, int], list] = defaultdict(list)
    for r in rows:
        buckets[(int(r["n"]), int(r["p"]))].append(r["p_succ"])
    return {k: np.array(v, dtype=float) for k, v in sorted(buckets.items())}


def _shared_log_bins(arrays: Iterable[np.ndarray], *, n_bins: int) -> np.ndarray:
    vals = np.concatenate([np.log(np.maximum(a, EPS)) for a in arrays if len(a)])
    lo, hi = float(np.min(vals)), float(np.max(vals))
    pad = 0.05 * (hi - lo + 1e-12)
    return np.linspace(lo - pad, hi + pad, n_bins + 1)


def _agg_lines(ps: np.ndarray) -> Tuple[float, float, float, float]:
    mean_p = float(np.mean(ps))
    med_rt = float(np.median(1.0 / np.maximum(ps, EPS)))
    return mean_p, med_rt, float(np.log(max(mean_p, EPS))), float(np.log(max(med_rt, EPS)))


def plot_vs_key(
    groups: Dict[int, np.ndarray],
    *,
    key_label: str,
    title: str,
    out_path: Path,
    n_bins: int,
) -> dict:
    """Faceted histograms: top row p_succ, bottom row runtime = 1/p_succ."""
    keys = sorted(groups)
    if not keys:
        raise SystemExit(f"No groups for {out_path.name}")

    all_ps = [groups[k] for k in keys]
    bins_p = _shared_log_bins(all_ps, n_bins=n_bins)
    all_rt = [1.0 / np.maximum(groups[k], EPS) for k in keys]
    bins_rt = _shared_log_bins(all_rt, n_bins=n_bins)

    ncols = len(keys)
    fig, axes = plt.subplots(2, ncols, figsize=(2.2 * ncols + 1.5, 5.5), squeeze=False)

    summary: Dict[str, dict] = {}
    cmap = plt.cm.viridis(np.linspace(0.15, 0.85, ncols))

    for j, k in enumerate(keys):
        ps = groups[k]
        rt = 1.0 / np.maximum(ps, EPS)
        mean_p, med_rt, log_mean_p, log_med_rt = _agg_lines(ps)
        summary[str(k)] = {
            "N": int(len(ps)),
            "mean_p_succ": mean_p,
            "median_runtime": med_rt,
            "log_mean_p_succ": log_mean_p,
            "log_median_runtime": log_med_rt,
        }

        ax = axes[0, j]
        ax.hist(np.log(np.maximum(ps, EPS)), bins=bins_p, color=cmap[j], alpha=0.85, edgecolor="white")
        ax.axvline(log_mean_p, color="#C44E52", lw=2, ls="--")
        ax.set_title(f"{key_label}={k}", fontsize=10)
        if j == 0:
            ax.set_ylabel("count")
        ax.tick_params(labelbottom=False)

        ax = axes[1, j]
        ax.hist(np.log(np.maximum(rt, EPS)), bins=bins_rt, color=cmap[j], alpha=0.85, edgecolor="white")
        ax.axvline(log_med_rt, color="#55A868", lw=2)
        ax.set_xlabel("log runtime")
        if j == 0:
            ax.set_ylabel("count")

    axes[0, 0].annotate(
        "mean(p_succ)",
        xy=(0.02, 0.96),
        xycoords="axes fraction",
        fontsize=8,
        color="#C44E52",
        va="top",
    )
    axes[1, 0].annotate(
        "median(1/p_succ)",
        xy=(0.02, 0.96),
        xycoords="axes fraction",
        fontsize=8,
        color="#55A868",
        va="top",
    )
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return summary


def plot_overlay_vs_key(
    groups: Dict[int, np.ndarray],
    *,
    key_label: str,
    title: str,
    out_path: Path,
    n_bins: int,
) -> None:
    keys = sorted(groups)
    all_ps = [groups[k] for k in keys]
    bins_p = _shared_log_bins(all_ps, n_bins=n_bins)
    all_rt = [1.0 / np.maximum(groups[k], EPS) for k in keys]
    bins_rt = _shared_log_bins(all_rt, n_bins=n_bins)
    cmap = plt.cm.viridis(np.linspace(0.15, 0.85, len(keys)))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for j, k in enumerate(keys):
        ps = groups[k]
        rt = 1.0 / np.maximum(ps, EPS)
        mean_p, med_rt, log_mean_p, log_med_rt = _agg_lines(ps)
        label = f"{key_label}={k}"
        axes[0].hist(
            np.log(np.maximum(ps, EPS)),
            bins=bins_p,
            histtype="step",
            lw=1.8,
            color=cmap[j],
            label=label,
        )
        axes[0].axvline(log_mean_p, color=cmap[j], lw=1.2, ls="--", alpha=0.9)
        axes[1].hist(
            np.log(np.maximum(rt, EPS)),
            bins=bins_rt,
            histtype="step",
            lw=1.8,
            color=cmap[j],
            label=label,
        )
        axes[1].axvline(log_med_rt, color=cmap[j], lw=1.2, ls=":", alpha=0.9)

    axes[0].set_xlabel("log p_succ")
    axes[0].set_ylabel("count")
    axes[0].set_title("Success probability (dashed = mean)")
    axes[1].set_xlabel("log runtime (= log 1/p_succ)")
    axes[1].set_ylabel("count")
    axes[1].set_title("Runtime proxy (dotted = median)")
    axes[0].legend(fontsize=8, loc="upper left")
    axes[1].legend(fontsize=8, loc="upper left")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_vs_np_sum(
    by_np: Dict[Tuple[int, int], np.ndarray],
    *,
    title: str,
    out_path: Path,
    n_bins: int,
) -> dict:
    """One panel per (n,p) pair, ordered by increasing n+p then n."""
    pairs = sorted(by_np, key=lambda t: (t[0] + t[1], t[0], t[1]))
    labels = {n + p: f"n={n}, p={p}" for n, p in pairs}

    all_ps = [by_np[p] for p in pairs]
    bins_p = _shared_log_bins(all_ps, n_bins=n_bins)
    all_rt = [1.0 / np.maximum(by_np[p], EPS) for p in pairs]
    bins_rt = _shared_log_bins(all_rt, n_bins=n_bins)

    ncols = len(pairs)
    fig, axes = plt.subplots(2, ncols, figsize=(2.0 * ncols + 1.5, 5.5), squeeze=False)
    summary: Dict[str, dict] = {}
    cmap = plt.cm.plasma(np.linspace(0.1, 0.9, ncols))

    for j, (n, p) in enumerate(pairs):
        ps = by_np[(n, p)]
        rt = 1.0 / np.maximum(ps, EPS)
        mean_p, med_rt, log_mean_p, log_med_rt = _agg_lines(ps)
        key = f"n={n},p={p}"
        summary[key] = {
            "n": n,
            "p": p,
            "n_plus_p": n + p,
            "N": int(len(ps)),
            "mean_p_succ": mean_p,
            "median_runtime": med_rt,
        }

        ax = axes[0, j]
        ax.hist(np.log(np.maximum(ps, EPS)), bins=bins_p, color=cmap[j], alpha=0.85, edgecolor="white")
        ax.axvline(log_mean_p, color="#C44E52", lw=2, ls="--")
        ax.set_title(f"{labels[n+p]}\n(n+p={n+p})", fontsize=8)
        if j == 0:
            ax.set_ylabel("count")
        ax.tick_params(labelbottom=False)

        ax = axes[1, j]
        ax.hist(np.log(np.maximum(rt, EPS)), bins=bins_rt, color=cmap[j], alpha=0.85, edgecolor="white")
        ax.axvline(log_med_rt, color="#55A868", lw=2)
        ax.set_xlabel("log runtime")
        if j == 0:
            ax.set_ylabel("count")

    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        type=Path,
        nargs="+",
        default=[
            Path("results/bm24_runs/bm24_gap_audit/N500_n12-20_all.csv"),
        ],
        help="Gap-audit CSV file(s) with instance-level p_succ",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/bm24_runs/analysis/instance_distributions"))
    parser.add_argument("--objective", default="mean_p")
    parser.add_argument("--train-n", type=int, default=12)
    parser.add_argument("--depth", type=int, default=20, help="Fixed QAOA depth p for vs-n sweep")
    parser.add_argument("--fixed-n", type=int, default=16, help="Fixed n for vs-p sweep")
    parser.add_argument("--bins", type=int, default=25)
    args = parser.parse_args()

    csv_paths = [p if p.is_absolute() else _PHASECRAFT / p for p in args.csv]
    for p in csv_paths:
        if not p.is_file():
            raise SystemExit(f"CSV not found: {p}")

    rows = load_rows(csv_paths)
    out_dir = args.output_dir if args.output_dir.is_absolute() else _PHASECRAFT / args.output_dir

    vs_n_rows = filter_rows(
        rows, objective=args.objective, train_n=args.train_n, depth=args.depth
    )
    vs_p_rows = filter_rows(
        rows, objective=args.objective, train_n=args.train_n, fixed_n=args.fixed_n
    )
    all_np_rows = filter_rows(rows, objective=args.objective, train_n=args.train_n)

    summary: dict = {
        "csv": [str(p) for p in csv_paths],
        "objective": args.objective,
        "train_n": args.train_n,
    }

    by_n = group_by(vs_n_rows, "n")
    tag = f"train{args.train_n}_obj{args.objective}_p{args.depth}"
    summary["vs_n"] = plot_vs_key(
        by_n,
        key_label="n",
        title=f"Instance distributions vs n (fixed p={args.depth}, train_n={args.train_n})",
        out_path=out_dir / f"hist_vs_n_{tag}.png",
        n_bins=args.bins,
    )
    plot_overlay_vs_key(
        by_n,
        key_label="n",
        title=f"Overlaid instance distributions vs n (p={args.depth})",
        out_path=out_dir / f"hist_vs_n_overlay_{tag}.png",
        n_bins=args.bins,
    )

    by_p = group_by(vs_p_rows, "p")
    tag_p = f"train{args.train_n}_obj{args.objective}_n{args.fixed_n}"
    summary["vs_p"] = plot_vs_key(
        by_p,
        key_label="p",
        title=f"Instance distributions vs p (fixed n={args.fixed_n}, train_n={args.train_n})",
        out_path=out_dir / f"hist_vs_p_{tag_p}.png",
        n_bins=args.bins,
    )
    plot_overlay_vs_key(
        by_p,
        key_label="p",
        title=f"Overlaid instance distributions vs p (n={args.fixed_n})",
        out_path=out_dir / f"hist_vs_p_overlay_{tag_p}.png",
        n_bins=args.bins,
    )

    by_np = group_by_np(all_np_rows)
    summary["vs_np"] = plot_vs_np_sum(
        by_np,
        title=f"Instance distributions vs (n, p) ordered by n+p (train_n={args.train_n})",
        out_path=out_dir / f"hist_vs_np_train{args.train_n}.png",
        n_bins=args.bins,
    )

    summary_path = out_dir / "instance_distribution_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote figures to {out_dir}")
    for name in sorted(out_dir.glob("*.png")):
        print(f"  {name.name}")
    print(f"  {summary_path.name}")


if __name__ == "__main__":
    main()
