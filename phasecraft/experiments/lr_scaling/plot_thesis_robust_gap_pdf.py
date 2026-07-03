#!/usr/bin/env python3
"""
One-PDF robustness figure for the typical-vs-mean success exponent gap.

Output: bm24_runs/analysis/thesis_gap_candidates/robust_typical_mean_gap.pdf
No PNG, JSON, or table sidecars are written.
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
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "phasecraft-mpl"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_PHASECRAFT = Path(__file__).resolve().parents[2]


MODE_LABEL = {
    "bm24_mean_p_fixed_n": r"mean-$p$ train",
    "median_runtime_fixed_n": r"median-runtime train",
}
MODE_COLOR = {
    "bm24_mean_p_fixed_n": "#4C72B0",
    "median_runtime_fixed_n": "#C44E52",
}


def _trace_rows(payload: dict, mode: str) -> List[dict]:
    traces_by_mode = payload.get("traces_by_mode") or {}
    if mode in traces_by_mode:
        return list(traces_by_mode[mode])
    key = f"trace_{mode}"
    return list(payload.get(key) or [])


def load_multiseed_gap_rows(manifest_path: Path) -> List[dict]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows: List[dict] = []
    for run in manifest["runs"]:
        run_json = Path(run["json"])
        payload = json.loads(run_json.read_text(encoding="utf-8"))
        cfg = payload["config"]
        cache_path = run_json.parent / f"{run_json.stem}-ctyp-cann-cache.json"
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
        rebench = cache.get("rebenchmark") or {}
        seed = int(cfg["seed"])
        for mode in ("bm24_mean_p_fixed_n", "median_runtime_fixed_n"):
            for tr in _trace_rows(payload, mode):
                depth = int(tr["depth"])
                rb = rebench.get(f"{mode}|{depth}")
                if not rb:
                    continue
                c_typ = float(tr["lr_log2_slope"])
                c_mean = float(rb["c_inv_mean_rt"])
                rows.append(
                    {
                        "seed": seed,
                        "mode": mode,
                        "depth": depth,
                        "c_typ": c_typ,
                        "c_mean": c_mean,
                        "gap": c_typ - c_mean,
                        "train_size": int(cfg["train_size"]),
                        "test_size": int(cfg["test_size"]),
                        "n_min": int(cfg["n_min"]),
                        "n_max": int(cfg["n_max"]),
                    }
                )
    return rows


def common_depths(rows: Sequence[dict]) -> List[int]:
    by_key = {(r["seed"], r["mode"]): set() for r in rows}
    for r in rows:
        by_key[(r["seed"], r["mode"])].add(int(r["depth"]))
    common = None
    for depths in by_key.values():
        common = set(depths) if common is None else common & set(depths)
    return sorted(common or [])


def slope_log2(xs: Sequence[int], ys: Sequence[float]) -> float:
    return float(np.polyfit(np.asarray(xs, dtype=float), np.log2(np.asarray(ys, dtype=float)), 1)[0])


def load_audit_depth_gaps(csv_path: Path, *, objective: str, train_n: int) -> List[dict]:
    buckets: Dict[Tuple[int, int], List[float]] = defaultdict(list)
    with csv_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["objective"] != objective or int(row["train_n"]) != train_n:
                continue
            p_succ = float(row["p_succ"])
            if math.isfinite(p_succ) and p_succ > 0:
                buckets[(int(row["p"]), int(row["n"]))].append(p_succ)
    rows: List[dict] = []
    for depth in sorted({p for p, _ in buckets}):
        ns = sorted(n for p, n in buckets if p == depth)
        means = [float(np.mean(buckets[(depth, n)])) for n in ns]
        med_rt = [float(np.median(1.0 / np.asarray(buckets[(depth, n)], dtype=float))) for n in ns]
        c_mean = -slope_log2(ns, means)
        c_typ = slope_log2(ns, med_rt)
        rows.append(
            {
                "depth": depth,
                "c_mean": c_mean,
                "c_typ": c_typ,
                "gap": c_typ - c_mean,
                "n_min": min(ns),
                "n_max": max(ns),
                "N": len(buckets[(depth, ns[0])]),
            }
        )
    return rows


def bootstrap_audit_gap_ci(
    csv_path: Path,
    *,
    objective: str,
    train_n: int,
    n_boot: int,
    seed: int,
) -> Dict[int, Tuple[float, float]]:
    rng = np.random.default_rng(seed)
    buckets: Dict[Tuple[int, int], np.ndarray] = defaultdict(list)
    with csv_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["objective"] != objective or int(row["train_n"]) != train_n:
                continue
            p_succ = float(row["p_succ"])
            if math.isfinite(p_succ) and p_succ > 0:
                buckets[(int(row["p"]), int(row["n"]))].append(p_succ)
    buckets = {k: np.asarray(v, dtype=float) for k, v in buckets.items()}

    ci: Dict[int, Tuple[float, float]] = {}
    for depth in sorted({p for p, _ in buckets}):
        ns = sorted(n for p, n in buckets if p == depth)
        vals = np.empty(n_boot, dtype=float)
        for b in range(n_boot):
            means = []
            med_rt = []
            for n in ns:
                arr = buckets[(depth, n)]
                sample = arr[rng.integers(0, len(arr), len(arr))]
                means.append(float(np.mean(sample)))
                med_rt.append(float(np.median(1.0 / sample)))
            vals[b] = slope_log2(ns, med_rt) - (-slope_log2(ns, means))
        lo, hi = np.quantile(vals, [0.025, 0.975])
        ci[depth] = (float(lo), float(hi))
    return ci


def mean_sem(vals: Sequence[float]) -> Tuple[float, float]:
    arr = np.asarray(vals, dtype=float)
    mean = float(np.mean(arr))
    sem = float(np.std(arr, ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0
    return mean, sem


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9.5,
            "axes.labelsize": 9.5,
            "axes.titlesize": 10,
            "legend.fontsize": 8,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.23,
            "grid.linewidth": 0.7,
        }
    )


def plot_pdf(
    *,
    multiseed_rows: Sequence[dict],
    audit_rows: Sequence[dict],
    audit_ci: Mapping[int, Tuple[float, float]],
    out_pdf: Path,
) -> None:
    depths = [d for d in common_depths(multiseed_rows) if d in (10, 20, 40)]
    if not depths:
        raise SystemExit("No common depths found across all seeds and modes.")
    seeds = sorted({int(r["seed"]) for r in multiseed_rows})
    seed_offsets = {seed: off for seed, off in zip(seeds, np.linspace(-0.18, 0.18, len(seeds)))}

    fig, axes = plt.subplots(1, 3, figsize=(8.6, 2.85), gridspec_kw={"width_ratios": [1.15, 1.0, 1.05]})

    # A: multi-seed typical-vs-mean gap.
    ax = axes[0]
    for mode in ("bm24_mean_p_fixed_n", "median_runtime_fixed_n"):
        color = MODE_COLOR[mode]
        mode_rows = [r for r in multiseed_rows if r["mode"] == mode and r["depth"] in depths]
        x_shift = -0.09 if mode == "bm24_mean_p_fixed_n" else 0.09
        means = []
        sems = []
        for d in depths:
            vals = [r["gap"] for r in mode_rows if r["depth"] == d]
            mean, sem = mean_sem(vals)
            means.append(mean)
            sems.append(sem)
            for r in [rr for rr in mode_rows if rr["depth"] == d]:
                ax.scatter(
                    d + x_shift + seed_offsets[int(r["seed"])],
                    r["gap"],
                    s=18,
                    color=color,
                    alpha=0.55,
                    edgecolor="none",
                )
        ax.errorbar(
            np.asarray(depths) + x_shift,
            means,
            yerr=sems,
            fmt="o-" if mode == "bm24_mean_p_fixed_n" else "s--",
            color=color,
            capsize=2.5,
            lw=1.35,
            ms=4.2,
            label=MODE_LABEL[mode],
        )
    ax.axhline(0.0, color="0.25", lw=0.8)
    ax.set_title(r"(a) Three seeds, $N_\mathrm{test}=200$")
    ax.set_xlabel(r"QAOA depth $p$")
    ax.set_ylabel(r"$c_\mathrm{typ}-c_\mathrm{mean}$")
    ax.set_xticks(depths)
    ax.set_xlim(min(depths) - 3.0, max(depths) + 3.0)
    ax.legend(loc="upper left", frameon=True, framealpha=0.92)
    ax.text(0.03, 0.05, r"$n=12,\ldots,18$; train size 100", transform=ax.transAxes, fontsize=7.7)

    # B: training objective sensitivity.
    ax = axes[1]
    for d in depths:
        for seed in seeds:
            m = [
                r for r in multiseed_rows
                if r["seed"] == seed and r["depth"] == d and r["mode"] == "bm24_mean_p_fixed_n"
            ]
            q = [
                r for r in multiseed_rows
                if r["seed"] == seed and r["depth"] == d and r["mode"] == "median_runtime_fixed_n"
            ]
            if not m or not q:
                continue
            ax.scatter(d + seed_offsets[seed], q[0]["c_typ"] - m[0]["c_typ"], s=20, color="#6C757D", alpha=0.65)
        vals = []
        for seed in seeds:
            m = [r for r in multiseed_rows if r["seed"] == seed and r["depth"] == d and r["mode"] == "bm24_mean_p_fixed_n"]
            q = [r for r in multiseed_rows if r["seed"] == seed and r["depth"] == d and r["mode"] == "median_runtime_fixed_n"]
            if m and q:
                vals.append(q[0]["c_typ"] - m[0]["c_typ"])
        mean, sem = mean_sem(vals)
        ax.errorbar(d, mean, yerr=sem, fmt="D", color="#2A9D8F", capsize=2.5, ms=4.4)
    ax.axhline(0.0, color="0.25", lw=0.8)
    ax.set_title(r"(b) Training objective")
    ax.set_xlabel(r"QAOA depth $p$")
    ax.set_ylabel(r"$c_\mathrm{typ}^{\mathrm{median\,train}}-c_\mathrm{typ}^{\mathrm{mean\,train}}$")
    ax.set_xticks(depths)
    ax.set_xlim(min(depths) - 3.0, max(depths) + 3.0)

    # C: N=500 audit over the wider n-window.
    ax = axes[2]
    audit_depths = [r["depth"] for r in audit_rows]
    gaps = [r["gap"] for r in audit_rows]
    yerr = np.asarray(
        [
            [r["gap"] - audit_ci[r["depth"]][0] for r in audit_rows],
            [audit_ci[r["depth"]][1] - r["gap"] for r in audit_rows],
        ]
    )
    ax.errorbar(audit_depths, gaps, yerr=yerr, fmt="o-", color="#7B68A6", capsize=2.5, lw=1.35, ms=4.2)
    ax.axhline(0.0, color="0.25", lw=0.8)
    ax.set_title(r"(c) Wider audit, $N=500$")
    ax.set_xlabel(r"QAOA depth $p$")
    ax.set_ylabel(r"$c_\mathrm{typ}-c_\mathrm{mean}$")
    ax.set_xscale("log")
    ax.set_xticks(audit_depths)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.text(0.05, 0.05, r"$n=12,\ldots,20$; 95% bootstrap CI", transform=ax.transAxes, fontsize=7.7)

    fig.suptitle("Robustness of the typical-runtime vs mean-success exponent gap", y=1.04, fontsize=11)
    fig.tight_layout()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("results/bm24_runs/multi_seed_ctyp_cann/manifest.json"),
    )
    parser.add_argument(
        "--audit-csv",
        type=Path,
        default=Path("results/bm24_runs/bm24_gap_audit/N500_n12-20_all.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("bm24_runs/analysis/thesis_gap_candidates/robust_typical_mean_gap.pdf"),
    )
    parser.add_argument("--bootstrap", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args()

    setup_style()
    manifest = args.manifest if args.manifest.is_absolute() else _PHASECRAFT / args.manifest
    audit_csv = args.audit_csv if args.audit_csv.is_absolute() else _PHASECRAFT / args.audit_csv
    out_pdf = args.output if args.output.is_absolute() else _PHASECRAFT / args.output

    multiseed_rows = load_multiseed_gap_rows(manifest)
    audit_rows = load_audit_depth_gaps(audit_csv, objective="mean_p", train_n=12)
    audit_ci = bootstrap_audit_gap_ci(
        audit_csv,
        objective="mean_p",
        train_n=12,
        n_boot=int(args.bootstrap),
        seed=int(args.seed),
    )
    plot_pdf(multiseed_rows=multiseed_rows, audit_rows=audit_rows, audit_ci=audit_ci, out_pdf=out_pdf)
    print(out_pdf)


if __name__ == "__main__":
    main()
