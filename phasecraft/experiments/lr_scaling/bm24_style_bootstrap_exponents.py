#!/usr/bin/env python3
"""BM24-style exponent fits with half-sample resampling errors.

This script uses raw per-instance success probabilities.  For each depth and
problem size, the central exponent is fit on the full sample.  The error bar is
the standard deviation of refitted exponents after repeatedly choosing half of
the instances at each n, matching the resampling protocol described in BM24.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

_PHASECRAFT = Path(__file__).resolve().parents[2]
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(os.environ.get("TMPDIR", "/tmp")) / "phasecraft-matplotlib-cache"),
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_DEFAULT_CSV = (
    _PHASECRAFT
    / "results/bm24_runs/bm24_gap_audit/N500_n12-20_all.csv"
)
_DEFAULT_WALKSAT = (
    _PHASECRAFT
    / "results/bm24_runs/06-09/run1/walksat_walksatlm_n12-20_te200.json"
)
_DEFAULT_OUT = (
    _PHASECRAFT / "results/bm24_runs/analysis/bm24_style_bootstrap"
)
_DEFAULT_THESIS = _PHASECRAFT / "THESIS-FINAL GRAPHS"
_EPS = 1e-300


def _parse_ints(text: str) -> List[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def _fit_log2_slope(xs: Sequence[int], ys: Sequence[float]) -> Tuple[float, float]:
    x = np.asarray(xs, dtype=float)
    y = np.log2(np.maximum(np.asarray(ys, dtype=float), _EPS))
    design = np.vstack([x, np.ones_like(x)]).T
    slope, intercept = np.linalg.lstsq(design, y, rcond=None)[0]
    return float(slope), float(intercept)


def _fit_power_law(depths: Sequence[int], cs: Sequence[float]) -> Dict[str, float]:
    x = np.log(np.asarray(depths, dtype=float))
    y = np.log(np.maximum(np.asarray(cs, dtype=float), _EPS))
    design = np.vstack([x, np.ones_like(x)]).T
    slope, intercept = np.linalg.lstsq(design, y, rcond=None)[0]
    pred = design @ np.asarray([slope, intercept])
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
    return {
        "factor": float(math.exp(intercept)),
        "alpha": float(-slope),
        "r2_log": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
    }


def _load_success_rows(
    path: Path,
    *,
    objective: str,
    train_n: int,
    depths: Sequence[int],
    n_values: Sequence[int],
) -> Dict[Tuple[int, int], np.ndarray]:
    wanted_depths = set(int(p) for p in depths)
    wanted_ns = set(int(n) for n in n_values)
    rows: Dict[Tuple[int, int], List[float]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["objective"] != objective:
                continue
            if int(row["train_n"]) != int(train_n):
                continue
            p = int(row["p"])
            n = int(row["n"])
            if p not in wanted_depths or n not in wanted_ns:
                continue
            rows[(p, n)].append(float(row["p_succ"]))

    out = {key: np.asarray(vals, dtype=float) for key, vals in rows.items()}
    missing = [
        (p, n)
        for p in depths
        for n in n_values
        if (int(p), int(n)) not in out
    ]
    if missing:
        raise ValueError(f"missing raw p_succ rows for {missing[:8]}")
    return out


def _load_walksat_baselines(path: Path, range_key: str) -> Dict[str, float]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    slopes = (payload.get("slopes_log2") or {}).get(range_key) or {}
    return {
        "walksat": float(slopes.get("walksat", float("nan"))),
        "walksatlm": float(slopes.get("walksatlm", float("nan"))),
    }


def _summarize_depth(
    raw: Dict[Tuple[int, int], np.ndarray],
    *,
    depth: int,
    n_values: Sequence[int],
    rng: np.random.Generator,
    bootstrap_reps: int,
) -> Tuple[dict, np.ndarray]:
    mean_p_per_n: List[float] = []
    median_rt_per_n: List[float] = []
    instance_counts: List[int] = []

    for n in n_values:
        ps = raw[(int(depth), int(n))]
        instance_counts.append(int(len(ps)))
        mean_p_per_n.append(float(np.mean(ps)))
        median_rt_per_n.append(float(np.median(1.0 / np.maximum(ps, _EPS))))

    mean_slope, mean_intercept = _fit_log2_slope(n_values, mean_p_per_n)
    rt_slope, rt_intercept = _fit_log2_slope(n_values, median_rt_per_n)
    boot = np.empty((int(bootstrap_reps), 2), dtype=float)

    for b in range(int(bootstrap_reps)):
        mean_p_sample: List[float] = []
        median_rt_sample: List[float] = []
        for n in n_values:
            ps = raw[(int(depth), int(n))]
            half = max(1, len(ps) // 2)
            idx = rng.choice(len(ps), size=half, replace=False)
            sub = ps[idx]
            mean_p_sample.append(float(np.mean(sub)))
            median_rt_sample.append(float(np.median(1.0 / np.maximum(sub, _EPS))))
        mean_slope_b, _ = _fit_log2_slope(n_values, mean_p_sample)
        rt_slope_b, _ = _fit_log2_slope(n_values, median_rt_sample)
        boot[b, 0] = -float(mean_slope_b)
        boot[b, 1] = float(rt_slope_b)

    summary = {
        "p": int(depth),
        "n_values": [int(n) for n in n_values],
        "instances_per_n": instance_counts,
        "mean_p_per_n": mean_p_per_n,
        "median_runtime_per_n": median_rt_per_n,
        "c_sp": -float(mean_slope),
        "c_sp_error_bm24_half_sample": float(np.std(boot[:, 0], ddof=1)),
        "c_sp_log2_intercept": float(mean_intercept),
        "c_rt": float(rt_slope),
        "c_rt_error_bm24_half_sample": float(np.std(boot[:, 1], ddof=1)),
        "c_rt_log2_intercept": float(rt_intercept),
    }
    return summary, boot


def _power_law_summaries(
    depth_rows: Sequence[dict],
    boots_by_depth: Dict[int, np.ndarray],
    *,
    bootstrap_reps: int,
) -> Dict[str, dict]:
    by_depth = {int(r["p"]): r for r in depth_rows}
    depths_all = [int(r["p"]) for r in depth_rows]
    subsets = {
        "all_p": depths_all,
        "p_le_10": [p for p in depths_all if p <= 10],
        "p_ge_15": [p for p in depths_all if p >= 15],
        "p_ge_20": [p for p in depths_all if p >= 20],
    }
    out: Dict[str, dict] = {}
    for subset_name, subset in subsets.items():
        if len(subset) < 2:
            continue
        for metric, boot_col in (("c_sp", 0), ("c_rt", 1)):
            central = _fit_power_law(subset, [by_depth[p][metric] for p in subset])
            boot_factors = []
            boot_alphas = []
            for b in range(int(bootstrap_reps)):
                fit = _fit_power_law(
                    subset,
                    [float(boots_by_depth[p][b, boot_col]) for p in subset],
                )
                boot_factors.append(float(fit["factor"]))
                boot_alphas.append(float(fit["alpha"]))
            central["factor_error_bm24_half_sample"] = float(
                np.std(boot_factors, ddof=1)
            )
            central["alpha_error_bm24_half_sample"] = float(
                np.std(boot_alphas, ddof=1)
            )
            out[f"{metric}_{subset_name}"] = central
    return out


def _write_csv(path: Path, rows: Sequence[dict]) -> None:
    fieldnames = [
        "p",
        "c_sp",
        "c_sp_error_bm24_half_sample",
        "c_rt",
        "c_rt_error_bm24_half_sample",
        "instances_per_n_min",
        "instances_per_n_max",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            counts = row["instances_per_n"]
            writer.writerow({
                "p": row["p"],
                "c_sp": row["c_sp"],
                "c_sp_error_bm24_half_sample": row[
                    "c_sp_error_bm24_half_sample"
                ],
                "c_rt": row["c_rt"],
                "c_rt_error_bm24_half_sample": row[
                    "c_rt_error_bm24_half_sample"
                ],
                "instances_per_n_min": min(counts),
                "instances_per_n_max": max(counts),
            })


def _plot_summary(
    path: Path,
    *,
    rows: Sequence[dict],
    power_law: Dict[str, dict],
    baselines: Dict[str, float],
    title: str,
    xscale: str,
) -> None:
    depths = np.asarray([int(r["p"]) for r in rows], dtype=float)
    c_sp = np.asarray([float(r["c_sp"]) for r in rows], dtype=float)
    e_sp = np.asarray(
        [float(r["c_sp_error_bm24_half_sample"]) for r in rows], dtype=float
    )
    c_rt = np.asarray([float(r["c_rt"]) for r in rows], dtype=float)
    e_rt = np.asarray(
        [float(r["c_rt_error_bm24_half_sample"]) for r in rows], dtype=float
    )

    plt.rcParams.update({
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.titlesize": 13,
        "legend.fontsize": 10.5,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
    })
    fig, ax = plt.subplots(figsize=(7.8, 4.7))

    ax.errorbar(
        depths,
        c_rt,
        yerr=e_rt,
        color="black",
        ls="-",
        marker="D",
        ms=7,
        lw=2.4,
        capsize=3,
        label="Median runtime",
    )
    ax.errorbar(
        depths,
        c_sp,
        yerr=e_sp,
        color="black",
        ls="--",
        marker="o",
        mfc="white",
        mec="black",
        ms=7,
        lw=2.2,
        capsize=3,
        label="Mean success",
    )

    xx = np.linspace(float(depths.min()), float(depths.max()), 250)
    for key, color, ls in (
        ("c_rt_all_p", "#4C72B0", "-"),
        ("c_sp_all_p", "#C44E52", "--"),
    ):
        fit = power_law.get(key)
        if not fit:
            continue
        yy = float(fit["factor"]) * xx ** (-float(fit["alpha"]))
        ax.plot(xx, yy, color=color, ls=ls, lw=1.4, alpha=0.75, label="_nolegend_")

    baseline_styles = [
        ("walksatlm", "0.35", "--", "WalkSATlm"),
        ("walksat", "0.55", ":", "WalkSAT"),
    ]
    for key, color, ls, label in baseline_styles:
        val = float(baselines.get(key, float("nan")))
        if not math.isfinite(val):
            continue
        ax.axhline(val, color=color, ls=ls, lw=1.2, zorder=0)
        ax.text(
            0.985,
            val,
            f"{label} ({val:.2f})",
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="bottom",
            color=color,
            fontsize=10.5,
        )

    ax.set_xscale(xscale)
    ax.set_xticks(depths)
    ax.set_xticklabels([str(int(p)) for p in depths])
    if xscale == "linear":
        ax.set_xlim(max(0.0, float(depths.min()) - 2.0), float(depths.max()) + 2.0)
    ax.set_xlabel(r"Depth ($p$)")
    ax.set_ylabel("Exponent")
    ax.set_title(title)
    ax.grid(True, alpha=0.18, linewidth=0.7)
    ax.legend(frameon=False, loc="upper right", handlelength=2.6)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, facecolor="white")
    plt.close(fig)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=_DEFAULT_CSV)
    parser.add_argument("--walksat-json", type=Path, default=_DEFAULT_WALKSAT)
    parser.add_argument("--out-dir", type=Path, default=_DEFAULT_OUT)
    parser.add_argument("--thesis-dir", type=Path, default=_DEFAULT_THESIS)
    parser.add_argument("--objective", default="mean_p")
    parser.add_argument("--train-n", type=int, default=12)
    parser.add_argument("--n-lo", type=int, default=12)
    parser.add_argument("--n-hi", type=int, default=20)
    parser.add_argument("--depths", default="2,5,10,20,50")
    parser.add_argument("--bootstrap-reps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=22080609)
    parser.add_argument("--xscale", choices=("linear", "log"), default="linear")
    parser.add_argument(
        "--output-tag",
        default="",
        help="Optional suffix for output filenames, e.g. all_available_raw.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    depths = _parse_ints(args.depths)
    n_values = list(range(int(args.n_lo), int(args.n_hi) + 1))
    raw = _load_success_rows(
        args.csv,
        objective=args.objective,
        train_n=int(args.train_n),
        depths=depths,
        n_values=n_values,
    )
    rng = np.random.default_rng(int(args.seed))
    rows = []
    boots_by_depth = {}
    for depth in depths:
        row, boot = _summarize_depth(
            raw,
            depth=int(depth),
            n_values=n_values,
            rng=rng,
            bootstrap_reps=int(args.bootstrap_reps),
        )
        rows.append(row)
        boots_by_depth[int(depth)] = boot

    power_law = _power_law_summaries(
        rows,
        boots_by_depth,
        bootstrap_reps=int(args.bootstrap_reps),
    )
    range_key = f"{args.n_lo}-{args.n_hi}"
    baselines = _load_walksat_baselines(args.walksat_json, range_key)

    summary = {
        "method": (
            "Central c values are least-squares exponential fits in log2 space "
            "over n. Errors are BM24-style half-sample resampling standard "
            "deviations: for each bootstrap repeat, choose half the instances "
            "uniformly without replacement at each n, recompute per-n "
            "aggregates, and refit."
        ),
        "source_csv": str(args.csv),
        "objective": args.objective,
        "train_n": int(args.train_n),
        "n_values": n_values,
        "depths": depths,
        "bootstrap_reps": int(args.bootstrap_reps),
        "seed": int(args.seed),
        "depth_rows": rows,
        "power_law_fits": power_law,
        "walksat_baselines_log2": baselines,
    }

    stem = (
        f"bm24_style_exponents_train{args.train_n}_{args.objective}"
        f"_n{args.n_lo}-{args.n_hi}"
    )
    if args.output_tag:
        stem = f"{stem}_{args.output_tag}"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / f"{stem}.json"
    csv_path = args.out_dir / f"{stem}.csv"
    png_path = args.out_dir / f"{stem}.png"
    pdf_path = args.out_dir / f"{stem}.pdf"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_csv(csv_path, rows)
    objective_label = args.objective.replace("_", "-")
    min_count = min(min(r["instances_per_n"]) for r in rows)
    max_count = max(max(r["instances_per_n"]) for r in rows)
    count_label = f"N={min_count} per n" if min_count == max_count else f"N={min_count}-{max_count} per n"
    title = (
        f"BM24-style exponents: train n={args.train_n}, "
        f"{objective_label}, {count_label}"
    )
    _plot_summary(
        png_path,
        rows=rows,
        power_law=power_law,
        baselines=baselines,
        title=title,
        xscale=args.xscale,
    )
    _plot_summary(
        pdf_path,
        rows=rows,
        power_law=power_law,
        baselines=baselines,
        title=title,
        xscale=args.xscale,
    )

    if args.thesis_dir:
        args.thesis_dir.mkdir(parents=True, exist_ok=True)
        thesis_png = args.thesis_dir / f"{stem}.png"
        thesis_pdf = args.thesis_dir / f"{stem}.pdf"
        thesis_png.write_bytes(png_path.read_bytes())
        thesis_pdf.write_bytes(pdf_path.read_bytes())

    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {png_path}")
    print(f"Wrote {pdf_path}")
    if args.thesis_dir:
        print(f"Copied figure to {args.thesis_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
