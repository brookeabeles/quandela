#!/usr/bin/env python3
"""Final paired exponent figures for thesis use."""

from __future__ import annotations

import json
import math
import os
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

_OUT_DIR = _PHASECRAFT / "THESIS-FINAL GRAPHS"
_ANALYSIS_DIR = _PHASECRAFT / "results/bm24_runs/analysis/final_exponent_pair"
_RAW_JSON = (
    _PHASECRAFT
    / "results/bm24_runs/analysis/bm24_style_bootstrap/"
    / "bm24_style_exponents_train12_mean_p_n12-20_all_available_raw.json"
)
_OLD_RT_JSON = (
    _PHASECRAFT
    / "results/bm24_runs/analysis/subwindow_slopes/"
    / "mean_p_window_stability_n9-16_n12-15_n12-20.json"
)
_OLD_SP_JSON = (
    _PHASECRAFT
    / "results/bm24_runs/analysis/subwindow_slopes/"
    / "mean_success_window_stability_n9-16_n12-15_n12-20.json"
)

_MEDIAN_COLOR = "#D62728"
_MEAN_COLOR = "#4169E1"
_BASELINE_COLOR = "#D62728"
_BASELINE_TEXT = "#4A4A4A"


def _fit_power_law(depths: Sequence[int], values: Sequence[float]) -> Dict[str, float]:
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
        "r2_log": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
    }


def _range_entry(row: dict, n_lo: int = 12, n_hi: int = 20) -> dict:
    for entry in row["range_slopes"]:
        if int(entry["n_lo"]) == n_lo and int(entry["n_hi"]) == n_hi:
            return entry
    raise KeyError(f"missing slope range {n_lo}-{n_hi} for depth {row['depth']}")


def _load_old_aggregate() -> dict:
    rt = json.loads(_OLD_RT_JSON.read_text(encoding="utf-8"))
    sp = json.loads(_OLD_SP_JSON.read_text(encoding="utf-8"))
    rt_rows = {int(r["depth"]): _range_entry(r) for r in rt["mean_p"]}
    sp_rows = {int(r["depth"]): _range_entry(r) for r in sp["mean_p"]}
    depths = sorted(set(rt_rows) & set(sp_rows))
    c_rt = [float(rt_rows[p]["slope"]) for p in depths]
    c_sp = [float(sp_rows[p]["slope"]) for p in depths]
    e_rt = [float(rt_rows[p].get("stderr", float("nan"))) for p in depths]
    e_sp = [float(sp_rows[p].get("stderr", float("nan"))) for p in depths]
    return {
        "depths": depths,
        "c_rt": c_rt,
        "c_sp": c_sp,
        "e_rt": e_rt,
        "e_sp": e_sp,
        "walksat": 0.4046448015622218,
        "walksat_im": 0.346309706545203,
        "fit_rt": _fit_power_law(depths, c_rt),
        "fit_sp": _fit_power_law(depths, c_sp),
    }


def _load_raw_more_instances() -> dict:
    data = json.loads(_RAW_JSON.read_text(encoding="utf-8"))
    rows = data["depth_rows"]
    depths = [int(r["p"]) for r in rows]
    return {
        "depths": depths,
        "c_rt": [float(r["c_rt"]) for r in rows],
        "c_sp": [float(r["c_sp"]) for r in rows],
        "e_rt": [float(r["c_rt_error_bm24_half_sample"]) for r in rows],
        "e_sp": [float(r["c_sp_error_bm24_half_sample"]) for r in rows],
        "walksat": float(data["walksat_baselines_log2"]["walksat"]),
        "walksat_im": float(data["walksat_baselines_log2"]["walksatlm"]),
        "fit_rt": data["power_law_fits"]["c_rt_all_p"],
        "fit_sp": data["power_law_fits"]["c_sp_all_p"],
    }


def _style_x_axis(ax: plt.Axes) -> None:
    major = np.arange(0, 51, 10)
    minor = np.arange(5, 51, 10)
    ax.set_xlim(0, 52)
    ax.set_xticks(major)
    ax.set_xticks(minor, minor=True)
    ax.tick_params(axis="x", which="major", length=7, width=1.25, labelsize=12)
    ax.tick_params(axis="x", which="minor", length=4, width=1.0, labelbottom=False)
    ax.grid(True, axis="x", which="major", color="#DADADA", linewidth=0.9, alpha=0.75)
    ax.grid(True, axis="x", which="minor", color="#ECECEC", linewidth=0.55, alpha=0.7)


def _plot_one(path: Path, data: dict) -> None:
    depths = np.asarray(data["depths"], dtype=float)
    c_rt = np.asarray(data["c_rt"], dtype=float)
    c_sp = np.asarray(data["c_sp"], dtype=float)
    e_rt = np.asarray(data["e_rt"], dtype=float)
    e_sp = np.asarray(data["e_sp"], dtype=float)

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.labelsize": 16,
        "legend.fontsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 12,
        "axes.linewidth": 1.0,
    })
    fig, ax = plt.subplots(figsize=(7.2, 4.35))

    x_fit = np.linspace(2.0, 50.0, 400)
    rt_fit = data["fit_rt"]
    sp_fit = data["fit_sp"]
    ax.plot(
        x_fit,
        float(rt_fit["factor"]) * x_fit ** (-float(rt_fit["alpha"])),
        color=_MEDIAN_COLOR,
        lw=1.35,
        alpha=0.38,
        zorder=1,
    )
    ax.plot(
        x_fit,
        float(sp_fit["factor"]) * x_fit ** (-float(sp_fit["alpha"])),
        color=_MEAN_COLOR,
        ls="--",
        lw=1.35,
        alpha=0.42,
        zorder=1,
    )

    ax.errorbar(
        depths,
        c_sp,
        yerr=e_sp,
        color=_MEAN_COLOR,
        ls="--",
        marker="o",
        mfc="white",
        mec=_MEAN_COLOR,
        mew=1.2,
        lw=2.2,
        ms=6.2,
        capsize=2.5,
        elinewidth=1.0,
        label="Mean success",
        zorder=3,
    )
    ax.errorbar(
        depths,
        c_rt,
        yerr=e_rt,
        color=_MEDIAN_COLOR,
        ls="-",
        marker="D",
        mfc=_MEDIAN_COLOR,
        mec=_MEDIAN_COLOR,
        mew=1.0,
        lw=2.4,
        ms=6.0,
        capsize=2.5,
        elinewidth=1.0,
        label="Median runtime",
        zorder=4,
    )

    walksat = float(data["walksat"])
    walksat_im = float(data["walksat_im"])
    ax.axhline(walksat, color=_BASELINE_COLOR, ls=":", lw=1.2, alpha=0.7, zorder=0)
    ax.axhline(walksat_im, color=_BASELINE_COLOR, ls="--", lw=1.2, alpha=0.7, zorder=0)
    ax.text(
        1.0,
        walksat + 0.010,
        "WalkSAT",
        color=_BASELINE_TEXT,
        ha="left",
        va="bottom",
        fontsize=11,
    )
    ax.text(
        1.0,
        walksat_im - 0.012,
        "WalkSATlm",
        color=_BASELINE_TEXT,
        ha="left",
        va="top",
        fontsize=11,
    )

    _style_x_axis(ax)
    ax.set_ylim(0.25, 0.72)
    ax.set_xlabel(r"Depth ($p$)")
    ax.set_ylabel("Exponent")
    ax.grid(True, axis="y", color="#E3E3E3", linewidth=0.75, alpha=0.75)
    ax.legend(frameon=False, loc="upper right", handlelength=2.8)
    fig.tight_layout(pad=0.6)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, facecolor="white")
    plt.close(fig)


def _caption_text(raw: dict, old: dict) -> str:
    return (
        "Figure A: Fewer-depth, more-instance BM24-style exponent comparison. "
        "LR-QAOA parameters are trained at n=12 on the mean-success objective and "
        "evaluated over n=12..20 using raw per-instance success probabilities. "
        "Error bars use BM24-style half-sample resampling. The all-p power-law "
        f"fits are c_rt = {raw['fit_rt']['factor']:.2f} p^-{raw['fit_rt']['alpha']:.2f} "
        f"and c_sp = {raw['fit_sp']['factor']:.2f} p^-{raw['fit_sp']['alpha']:.2f}. "
        "Horizontal references show WalkSAT and WalkSATlm median-runtime "
        "exponents over the same n window.\n\n"
        "Figure B: More-depth, fewer-instance aggregate exponent comparison. "
        "The same train-n=12 mean-success LR-QAOA protocol is evaluated over "
        "n=12..20 at depths p=2,5,8,10,15,20,30,40,50 using the aggregate "
        "N=200 summaries from the earlier run. Error bars are regression "
        "standard errors over n, not BM24 half-sample errors, because the raw "
        "per-instance rows for p=8,15,30,40 are not present. The all-p "
        f"power-law fits are c_rt = {old['fit_rt']['factor']:.2f} "
        f"p^-{old['fit_rt']['alpha']:.2f} and c_sp = "
        f"{old['fit_sp']['factor']:.2f} p^-{old['fit_sp']['alpha']:.2f}."
    )


def main(_: Iterable[str] | None = None) -> int:
    raw = _load_raw_more_instances()
    old = _load_old_aggregate()
    pairs = [
        ("final_less_depth_more_instances_exponents", raw),
        ("final_more_depth_less_instances_exponents", old),
    ]
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    _ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    for stem, data in pairs:
        for out_dir in (_OUT_DIR, _ANALYSIS_DIR):
            _plot_one(out_dir / f"{stem}.png", data)
            _plot_one(out_dir / f"{stem}.pdf", data)
    captions = _caption_text(raw, old)
    (_OUT_DIR / "final_exponent_pair_captions.txt").write_text(captions, encoding="utf-8")
    (_ANALYSIS_DIR / "final_exponent_pair_captions.txt").write_text(captions, encoding="utf-8")
    summary = {
        "less_depth_more_instances": raw,
        "more_depth_less_instances": old,
        "captions": captions,
    }
    (_ANALYSIS_DIR / "final_exponent_pair_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote figures to {_OUT_DIR}")
    print(f"Wrote analysis copies to {_ANALYSIS_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
