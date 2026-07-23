#!/usr/bin/env python3
"""Most complete aggregate p=100 LR-QAOA exponent figure."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np

_PHASECRAFT = Path(__file__).resolve().parents[2]
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(os.environ.get("TMPDIR", "/tmp")) / "phasecraft-matplotlib-cache"),
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


_OUT_DIR = _PHASECRAFT / "THESIS-FINAL GRAPHS"
_ANALYSIS_DIR = _PHASECRAFT / "results/bm24_runs/analysis/p100_most_complete"
_CLASSICAL_JSON = _PHASECRAFT / "results/bm24_runs/06-09/run1/walksat_walksatlm_n12-20_te200.json"
_RUNS = [
    _PHASECRAFT / "results/bm24_runs/06-09/run1/train12-tr100-te200-n12-18.json",
    _PHASECRAFT / "results/bm24_runs/multi_seed_ctyp_cann/06-24/run1/train12-tr100-te200-n12-18-seed0.json",
    _PHASECRAFT / "results/bm24_runs/multi_seed_ctyp_cann/06-24/run2/train12-tr100-te200-n12-18-seed42.json",
]

_MEAN_COLOR = "#4169E1"
_MEDIAN_COLOR = "#D62728"


def _fit_power_law(depths: list[int], values: list[float]) -> dict[str, float]:
    x = np.log(np.asarray(depths, dtype=float))
    y = np.log(np.asarray(values, dtype=float))
    design = np.vstack([x, np.ones_like(x)]).T
    slope, intercept = np.linalg.lstsq(design, y, rcond=None)[0]
    pred = design @ np.asarray([slope, intercept])
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
    return {
        "factor": float(math.exp(intercept)),
        "alpha": float(-slope),
        "r2_log": float(1.0 - ss_res / ss_tot),
    }


def _collect() -> dict:
    rows: dict[str, dict[int, list[dict[str, float]]]] = {
        "mean_success": {},
        "median_runtime": {},
    }
    configs = []
    for path in _RUNS:
        data = json.loads(path.read_text(encoding="utf-8"))
        cfg = data["config"]
        configs.append(
            {
                "path": str(path),
                "seed": int(cfg["seed"]),
                "train_n": int(cfg["train_n"]),
                "train_size": int(cfg["train_size"]),
                "n_min": int(cfg["n_min"]),
                "n_max": int(cfg["n_max"]),
                "test_size": int(cfg["test_size"]),
                "depths": list(data["depths"]),
            }
        )
        for mode_key, trace_key in [
            ("mean_success", "trace_bm24_mean_p_fixed_n"),
            ("median_runtime", "trace_median_runtime_fixed_n"),
        ]:
            for entry in data[trace_key]:
                depth = int(entry["depth"])
                rows[mode_key].setdefault(depth, []).append(
                    {
                        "seed": int(cfg["seed"]),
                        "exponent": float(entry["lr_log2_slope"]),
                    }
                )

    summary = {"configs": configs, "series": {}}
    for mode_key, by_depth in rows.items():
        depths = sorted(by_depth)
        means = []
        lows = []
        highs = []
        n_seeds = []
        per_depth = []
        for depth in depths:
            values = [x["exponent"] for x in by_depth[depth]]
            mean = float(np.mean(values))
            low = float(np.min(values))
            high = float(np.max(values))
            means.append(mean)
            lows.append(low)
            highs.append(high)
            n_seeds.append(len(values))
            per_depth.append(
                {
                    "depth": depth,
                    "mean": mean,
                    "min": low,
                    "max": high,
                    "n_seeds": len(values),
                    "by_seed": by_depth[depth],
                }
            )
        summary["series"][mode_key] = {
            "depths": depths,
            "mean": means,
            "min": lows,
            "max": highs,
            "n_seeds": n_seeds,
            "per_depth": per_depth,
            "fit": _fit_power_law(depths, means),
        }
    return summary


def _style_x_axis(ax: plt.Axes) -> None:
    major = np.arange(0, 101, 10)
    minor = np.arange(5, 101, 10)
    ax.set_xlim(0, 103)
    ax.set_xticks(major)
    ax.set_xticks(minor, minor=True)
    ax.tick_params(axis="x", which="major", length=7, width=1.25, labelsize=10)
    ax.tick_params(axis="x", which="minor", length=4, width=1.0, labelbottom=False)
    ax.grid(True, axis="x", which="major", color="#DADADA", linewidth=0.85, alpha=0.75)
    ax.grid(True, axis="x", which="minor", color="#ECECEC", linewidth=0.45, alpha=0.65)


def _plot(summary: dict, path: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.labelsize": 16,
            "legend.fontsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 12,
            "axes.linewidth": 1.0,
        }
    )
    fig, ax = plt.subplots(figsize=(7.8, 4.35))
    x_fit = np.linspace(2.0, 100.0, 500)
    for mode_key, label, color, marker, linestyle in [
        ("mean_success", "Mean success", _MEAN_COLOR, "o", "--"),
        ("median_runtime", "Median runtime", _MEDIAN_COLOR, "D", "-"),
    ]:
        series = summary["series"][mode_key]
        depths = np.asarray(series["depths"], dtype=float)
        means = np.asarray(series["mean"], dtype=float)
        lows = np.asarray(series["min"], dtype=float)
        highs = np.asarray(series["max"], dtype=float)
        yerr = np.vstack([means - lows, highs - means])
        fit = series["fit"]
        ax.plot(
            x_fit,
            fit["factor"] * x_fit ** (-fit["alpha"]),
            color=color,
            lw=1.25,
            ls=linestyle,
            alpha=0.30,
            zorder=1,
        )
        ax.errorbar(
            depths,
            means,
            yerr=yerr,
            color=color,
            ls=linestyle,
            marker=marker,
            mfc="white" if mode_key == "mean_success" else color,
            mec=color,
            mew=1.2,
            lw=2.15,
            ms=5.6,
            capsize=2.5,
            elinewidth=1.0,
            label=label,
            zorder=3 if mode_key == "mean_success" else 4,
        )
    _style_x_axis(ax)
    ax.set_ylim(0.28, 0.72)
    ax.set_xlabel(r"Depth ($p$)")
    ax.set_ylabel("Exponent")
    ax.grid(True, axis="y", color="#E3E3E3", linewidth=0.75, alpha=0.75)
    ax.legend(frameon=False, loc="upper right", handlelength=2.8)
    fig.tight_layout(pad=0.6)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, facecolor="white")
    plt.close(fig)


def _load_classical_baselines() -> dict[str, float]:
    data = json.loads(_CLASSICAL_JSON.read_text(encoding="utf-8"))
    slopes = data["slopes_log2"]["12-18"]
    return {
        "walksat": float(slopes["walksat"]),
        "walksatlm": float(slopes["walksatlm"]),
    }


def _plot_gap(summary: dict, path: Path) -> None:
    baselines = _load_classical_baselines()
    reference = baselines["walksatlm"]
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.labelsize": 16,
            "legend.fontsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 12,
            "axes.linewidth": 1.0,
        }
    )
    fig, ax = plt.subplots(figsize=(7.8, 4.35))
    for mode_key, label, color, marker, linestyle in [
        ("mean_success", "Mean success", _MEAN_COLOR, "o", "--"),
        ("median_runtime", "Median runtime", _MEDIAN_COLOR, "D", "-"),
    ]:
        series = summary["series"][mode_key]
        depths = np.asarray(series["depths"], dtype=float)
        means = reference - np.asarray(series["mean"], dtype=float)
        lows = reference - np.asarray(series["max"], dtype=float)
        highs = reference - np.asarray(series["min"], dtype=float)
        yerr = np.vstack([means - lows, highs - means])
        ax.errorbar(
            depths,
            means,
            yerr=yerr,
            color=color,
            ls=linestyle,
            marker=marker,
            mfc="white" if mode_key == "mean_success" else color,
            mec=color,
            mew=1.2,
            lw=2.15,
            ms=5.6,
            capsize=2.5,
            elinewidth=1.0,
            label=label,
            zorder=3 if mode_key == "mean_success" else 4,
        )
    ax.axhline(0.0, color="#555555", lw=1.15, ls="--", alpha=0.85, zorder=0)
    walksat_gap = reference - baselines["walksat"]
    ax.axhline(walksat_gap, color="#D62728", lw=1.0, ls=":", alpha=0.65, zorder=0)
    ax.text(1.5, 0.006, "WalkSATlm", color="#4A4A4A", ha="left", va="bottom", fontsize=11)
    ax.text(
        1.5,
        walksat_gap - 0.006,
        "WalkSAT",
        color="#4A4A4A",
        ha="left",
        va="top",
        fontsize=11,
    )
    _style_x_axis(ax)
    ax.set_ylim(-0.05, 0.12)
    ax.set_xlabel(r"Depth ($p$)")
    ax.set_ylabel("Exponent gap")
    ax.grid(True, axis="y", color="#E3E3E3", linewidth=0.75, alpha=0.75)
    ax.legend(frameon=False, loc="upper right", handlelength=2.8)
    fig.tight_layout(pad=0.6)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, facecolor="white")
    plt.close(fig)


def _plot_lr_exponent_gap(summary: dict, path: Path) -> None:
    mean_series = summary["series"]["mean_success"]
    runtime_series = summary["series"]["median_runtime"]
    common_depths = sorted(set(mean_series["depths"]) & set(runtime_series["depths"]))
    mean_by_depth = {x["depth"]: x for x in mean_series["per_depth"]}
    runtime_by_depth = {x["depth"]: x for x in runtime_series["per_depth"]}
    depths = []
    gaps = []
    lows = []
    highs = []
    for depth in common_depths:
        mean_seeds = {x["seed"]: x["exponent"] for x in mean_by_depth[depth]["by_seed"]}
        runtime_seeds = {x["seed"]: x["exponent"] for x in runtime_by_depth[depth]["by_seed"]}
        seed_gaps = [
            runtime_seeds[seed] - mean_seeds[seed]
            for seed in sorted(set(mean_seeds) & set(runtime_seeds))
        ]
        if not seed_gaps:
            continue
        depths.append(depth)
        gaps.append(float(np.mean(seed_gaps)))
        lows.append(float(np.min(seed_gaps)))
        highs.append(float(np.max(seed_gaps)))

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.labelsize": 16,
            "legend.fontsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 12,
            "axes.linewidth": 1.0,
        }
    )
    fig, ax = plt.subplots(figsize=(7.8, 4.35))
    depths_arr = np.asarray(depths, dtype=float)
    gaps_arr = np.asarray(gaps, dtype=float)
    lows_arr = np.asarray(lows, dtype=float)
    highs_arr = np.asarray(highs, dtype=float)
    yerr = np.vstack([gaps_arr - lows_arr, highs_arr - gaps_arr])
    ax.errorbar(
        depths_arr,
        gaps_arr,
        yerr=yerr,
        color="#333333",
        ls="-",
        marker="o",
        mfc="white",
        mec="#333333",
        mew=1.2,
        lw=2.0,
        ms=5.6,
        capsize=2.5,
        elinewidth=1.0,
        label="Median runtime - mean success",
        zorder=3,
    )
    ax.axhline(0.0, color="#777777", lw=1.15, ls="--", alpha=0.85, zorder=0)
    _style_x_axis(ax)
    ax.set_ylim(-0.035, 0.035)
    ax.set_xlabel(r"Depth ($p$)")
    ax.set_ylabel("Exponent gap")
    ax.grid(True, axis="y", color="#E3E3E3", linewidth=0.75, alpha=0.75)
    ax.legend(frameon=False, loc="upper right", handlelength=2.8)
    fig.tight_layout(pad=0.6)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, facecolor="white")
    plt.close(fig)


def _caption(summary: dict) -> str:
    mean_fit = summary["series"]["mean_success"]["fit"]
    median_fit = summary["series"]["median_runtime"]["fit"]
    p100_mean = next(
        x for x in summary["series"]["mean_success"]["per_depth"] if x["depth"] == 100
    )
    p100_median = next(
        x for x in summary["series"]["median_runtime"]["per_depth"] if x["depth"] == 100
    )
    return (
        "Most complete aggregate p=100 LR-QAOA scaling evidence. Parameters were "
        "trained at train_n=12 on 100 random instances for k=8, r=176.54 using the "
        "BM24 linear-ramp angle convention and decreasing beta schedule, then "
        "evaluated on 200 test instances per n over n=12..18. The figure combines "
        "three available training/evaluation seeds, 0, 27, and 42. Blue dashed "
        "circles show the exponent from the BM24 mean-success training objective; "
        "red diamonds show the exponent from median-runtime training. Points are "
        "the mean exponent across available seeds at that depth, and vertical bars "
        "show the min-to-max seed range, not BM24 half-sample bootstrap error. "
        "Seed 27 provides the densest depth grid p=2,5,8,10,15,20,30,40,50,80,100; "
        "seeds 0 and 42 provide the high-depth confirmation grid p=2,10,20,40,50,60,80,100. "
        f"Power-law fits to the plotted seed-mean exponents give c_mean = "
        f"{mean_fit['factor']:.2f} p^-{mean_fit['alpha']:.2f} and c_runtime = "
        f"{median_fit['factor']:.2f} p^-{median_fit['alpha']:.2f}. At p=100, "
        f"the mean-success exponent is {p100_mean['mean']:.2f} "
        f"(seed range {p100_mean['min']:.2f}-{p100_mean['max']:.2f}) and the "
        f"median-runtime exponent is {p100_median['mean']:.2f} "
        f"(seed range {p100_median['min']:.2f}-{p100_median['max']:.2f}); the "
        "lowest individual median-runtime exponent is 0.3115 from seed 42 at p=100. "
        "Because raw per-instance rows are not available for this full p=100 grid, "
        "this is an aggregate multi-seed diagnostic rather than a BM24-bootstrap "
        "n=12..20 final benchmark."
    )


def _gap_caption(summary: dict) -> str:
    baselines = _load_classical_baselines()
    p100_mean = next(
        x for x in summary["series"]["mean_success"]["per_depth"] if x["depth"] == 100
    )
    p100_median = next(
        x for x in summary["series"]["median_runtime"]["per_depth"] if x["depth"] == 100
    )
    mean_gap = baselines["walksatlm"] - p100_mean["mean"]
    median_gap = baselines["walksatlm"] - p100_median["mean"]
    return (
        "Improvement-gap version of the most complete aggregate p=100 LR-QAOA scaling "
        "evidence. The y-axis plots c_WalkSATlm - c_LR using the same n=12..18 "
        f"classical reference as the aggregate p=100 data: c_WalkSATlm = "
        f"{baselines['walksatlm']:.4f}; the dotted WalkSAT guide is at "
        f"c_WalkSATlm - c_WalkSAT = {baselines['walksatlm'] - baselines['walksat']:.4f}. "
        "Positive values therefore mean the LR-QAOA exponent is below WalkSATlm on "
        "this n=12..18 aggregate fit. LR-QAOA parameters were trained at train_n=12 "
        "on 100 instances and evaluated on 200 test instances per n for k=8, r=176.54. "
        "The plot combines seeds 0, 27, and 42; points are seed-mean exponents and "
        "vertical bars are min-to-max seed ranges, not BM24 half-sample bootstrap "
        f"errors. At p=100, the mean-success gap is {mean_gap:.4f} and the "
        f"median-runtime gap is {median_gap:.4f}. This is a p=100 aggregate "
        "diagnostic over n=12..18, not the BM24-bootstrap n=12..20 final benchmark."
    )


def _lr_gap_caption(summary: dict) -> str:
    p100_mean = next(
        x for x in summary["series"]["mean_success"]["per_depth"] if x["depth"] == 100
    )
    p100_median = next(
        x for x in summary["series"]["median_runtime"]["per_depth"] if x["depth"] == 100
    )
    return (
        "Literal LR-QAOA exponent-gap diagnostic for the p=100 aggregate data. "
        "The y-axis plots c_runtime - c_mean, comparing the median-runtime exponent "
        "against the mean-success exponent at each depth. Points are averages over "
        "available paired seeds and vertical bars are min-to-max seed ranges. This "
        "quantity does not show a monotone growing gap with p; at p=100 the gap is "
        f"{p100_median['mean'] - p100_mean['mean']:.4f}, so the two LR-QAOA "
        "exponents are essentially overlapping at the highest depth."
    )


def main() -> int:
    summary = _collect()
    for out_dir in (_OUT_DIR, _ANALYSIS_DIR):
        _plot(summary, out_dir / "p100_most_complete_train12_n12-18.png")
        _plot(summary, out_dir / "p100_most_complete_train12_n12-18.pdf")
        _plot_gap(summary, out_dir / "p100_most_complete_train12_n12-18_gap_to_walksatlm.png")
        _plot_gap(summary, out_dir / "p100_most_complete_train12_n12-18_gap_to_walksatlm.pdf")
        _plot_lr_exponent_gap(summary, out_dir / "p100_most_complete_train12_n12-18_lr_exponent_gap.png")
        _plot_lr_exponent_gap(summary, out_dir / "p100_most_complete_train12_n12-18_lr_exponent_gap.pdf")
        (out_dir / "p100_most_complete_train12_n12-18_caption.txt").write_text(
            _caption(summary),
            encoding="utf-8",
        )
        (out_dir / "p100_most_complete_train12_n12-18_gap_caption.txt").write_text(
            _gap_caption(summary),
            encoding="utf-8",
        )
        (out_dir / "p100_most_complete_train12_n12-18_lr_exponent_gap_caption.txt").write_text(
            _lr_gap_caption(summary),
            encoding="utf-8",
        )
    _ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    (_ANALYSIS_DIR / "p100_most_complete_train12_n12-18_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote p=100 figure to {_OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
