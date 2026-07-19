#!/usr/bin/env python3
"""Plot finite-n window stability for mean-p-trained angles."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

_PHASECRAFT = Path(__file__).resolve().parents[2]
_LR = Path(__file__).resolve().parent
for _p in (_PHASECRAFT.parent, _PHASECRAFT, _LR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from eval_subwindow_slopes import (  # noqa: E402
    _DEFAULT_CACHE,
    _DEFAULT_EXTRA_CACHE,
    analyze_ranges,
    fit_range_slope,
    load_per_n_cache,
    parse_ranges,
)

_RUN1 = _PHASECRAFT / "results/bm24_runs/06-09/run1"
_DEFAULT_N9_11_CACHE = _RUN1 / "train12-tr100-te200-n12-18-per_n-n9-11.json"
_DEFAULT_CTYP_CACHE = _RUN1 / "train12-tr100-te200-n12-18-ctyp-cann-cache.json"
_DEFAULT_WIDER_N9_11 = _RUN1 / "train12-tr100-te200-n12-18-wider-n9-11-te200.json"
_DEFAULT_WIDER_N19_20 = _RUN1 / "train12-tr100-te200-n12-18-wider-n19-20-te200.json"
_DEFAULT_WIDER_N16_20 = _RUN1 / "train12-tr100-te200-n12-18-wider-n16-20-te50.json"
_DEFAULT_WALKSAT_JSON = _RUN1 / "walksat_walksatlm_n12-20_te200.json"
_TARGET_DEPTHS = (2, 5, 8, 10, 15, 20, 30, 40, 50)

_SLIDING_WINDOWS = ((12, 15),)
_HIGHLIGHT_WINDOWS = ((9, 16), (12, 20))

_SERIES_STYLES = [
    {"range": (9, 16), "color": "#D55E00", "ls": "-", "marker": "o", "lw": 3.0, "zorder": 4},
    {"range": (12, 15), "color": "#0072B2", "ls": "-", "marker": "s", "lw": 3.0, "zorder": 4},
    {"range": (12, 20), "color": "#000000", "ls": "-", "marker": "D", "lw": 3.2, "zorder": 5},
]

_METRIC_CFG = {
    "median_runtime": {
        "per_n_key": "median_runtime_per_n",
        "slope_sign": 1.0,
        "ylabel": "Median-runtime exponent",
        "stem": "mean_p_window_stability",
    },
    "mean_success": {
        "per_n_key": "mean_success_per_n",
        "slope_sign": -1.0,
        "ylabel": "Mean-success exponent",
        "stem": "mean_success_window_stability",
    },
}


def _merge_extra_caches(
    modes_data: Dict[str, List[dict]],
    extra_paths: Sequence[Path],
    *,
    per_n_key: str,
) -> Dict[str, List[dict]]:
    for extra_path in extra_paths:
        if not extra_path.is_file():
            continue
        extra = load_per_n_cache(extra_path, None)
        for mode, rows in extra.items():
            extra_by_depth = {int(r["depth"]): r for r in rows}
            for row in modes_data.get(mode, []):
                depth = int(row["depth"])
                if depth not in extra_by_depth:
                    continue
                per_n = dict(row.get(per_n_key, {}))
                per_n.update(extra_by_depth[depth].get(per_n_key, {}))
                row[per_n_key] = per_n
    return modes_data


def _load_wider_mean_success(wider_path: Path, n_tag: str) -> Dict[int, Dict[str, float]]:
    if not wider_path.is_file():
        return {}
    payload = json.loads(wider_path.read_text(encoding="utf-8"))
    out: Dict[int, Dict[str, float]] = {}
    for key, entry in payload.items():
        if not key.startswith("bm24_mean_p_fixed_n|"):
            continue
        if f"|{n_tag}|" not in key:
            continue
        depth = int(entry["depth"])
        out[depth] = {str(k): float(v) for k, v in entry["mean_p_per_n"].items()}
    return out


def _wider_mean_success_scale_factors(wider_path: Path) -> Tuple[float, float]:
    """mean(p_succ) ratios n=19/18 and n=20/18 from a wider-n re-eval cache."""
    payload = json.loads(wider_path.read_text(encoding="utf-8"))
    r19: List[float] = []
    r20: List[float] = []
    for row in payload.values():
        per_n = row.get("mean_p_per_n", {})
        if "18" not in per_n:
            continue
        m18 = float(per_n["18"])
        if m18 <= 0:
            continue
        if "19" in per_n:
            r19.append(float(per_n["19"]) / m18)
        if "20" in per_n:
            r20.append(float(per_n["20"]) / m18)
    if not r19 or not r20:
        raise ValueError(f"No n=19,20 mean-success wider-n data in {wider_path}")
    return float(np.mean(r19)), float(np.mean(r20))


def _augment_mean_success_per_n(per_n: Dict[str, float], *, r19: float, r20: float) -> Dict[str, float]:
    merged = dict(per_n)
    if "18" not in merged:
        raise ValueError("per_n missing n=18 anchor for wider-n extrapolation")
    m18 = float(merged["18"])
    merged["19"] = m18 * r19
    merged["20"] = m18 * r20
    return merged


def load_mean_success_modes_data(
    *,
    ctyp_cache: Path,
    wider_n9_11: Path,
    wider_n19_20: Path,
    wider_n16_20: Path,
) -> Dict[str, List[dict]]:
    """Per-depth mean(p_succ) from ctyp rebenchmark + wider n extensions."""
    rebench = json.loads(ctyp_cache.read_text(encoding="utf-8")).get("rebenchmark", {})
    extra_9_11 = _load_wider_mean_success(wider_n9_11, "n9-11")
    extra_19_20 = _load_wider_mean_success(wider_n19_20, "n19-20")
    ratio_19 = ratio_20 = float("nan")
    if wider_n16_20.is_file():
        try:
            ratio_19, ratio_20 = _wider_mean_success_scale_factors(wider_n16_20)
        except ValueError:
            pass

    rows: List[dict] = []
    for depth in _TARGET_DEPTHS:
        per_n: Dict[str, float] = {}
        reb = rebench.get(f"bm24_mean_p_fixed_n|{depth}")
        if reb:
            for n, mp in zip(reb["ns"], reb["mean_p"]):
                per_n[str(int(n))] = float(mp)
        if depth in extra_9_11:
            per_n.update(extra_9_11[depth])
        if depth in extra_19_20:
            per_n.update(extra_19_20[depth])
        elif (
            per_n
            and "18" in per_n
            and ("19" not in per_n or "20" not in per_n)
            and np.isfinite(ratio_19)
            and np.isfinite(ratio_20)
        ):
            per_n = _augment_mean_success_per_n(per_n, r19=ratio_19, r20=ratio_20)
        needed = set(range(9, 21))
        if not needed.issubset(int(k) for k in per_n):
            continue
        if per_n:
            rows.append({"depth": depth, "mean_success_per_n": per_n})

    return {"mean_p": rows}


def analyze_ranges_metric(
    modes_data: Dict[str, List[dict]],
    ranges: List[Tuple[int, int]],
    *,
    per_n_key: str,
    slope_sign: float,
) -> Dict[str, List[dict]]:
    out: Dict[str, List[dict]] = {}
    for mode_label, rows in modes_data.items():
        out[mode_label] = []
        for row in rows:
            per_n = row.get(per_n_key, {})
            if not per_n:
                continue
            all_ns = sorted(int(k) for k in per_n)
            range_slopes = [
                {
                    "n_lo": lo,
                    "n_hi": hi,
                    "slope": slope_sign * fit_range_slope(per_n, lo, hi),
                }
                for lo, hi in ranges
            ]
            out[mode_label].append({
                "depth": int(row["depth"]),
                "all_ns": all_ns,
                "range_slopes": range_slopes,
            })
    return out


def _slopes_for_range(rows: List[dict], n_lo: int, n_hi: int) -> List[float]:
    return [
        next(
            (s["slope"] for s in row["range_slopes"]
             if s["n_lo"] == n_lo and s["n_hi"] == n_hi),
            float("nan"),
        )
        for row in rows
    ]


_PAPER_LABEL_FS = 16
_PAPER_TICK_FS = 13
_PAPER_LEGEND_FS = 15
_PAPER_PANEL_FS = 16


def _apply_paper_rcparams() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "axes.unicode_minus": False,
        "font.size": 14,
        "axes.labelsize": _PAPER_LABEL_FS,
        "xtick.labelsize": _PAPER_TICK_FS,
        "ytick.labelsize": _PAPER_TICK_FS,
        "legend.fontsize": _PAPER_LEGEND_FS,
        "xtick.direction": "in",
        "ytick.direction": "in",
    })


def _style_axis_labels(ax: plt.Axes, *, xlabel: str, ylabel: str) -> None:
    ax.set_xlabel(xlabel, fontsize=_PAPER_LABEL_FS)
    ax.set_ylabel(ylabel, fontsize=_PAPER_LABEL_FS, labelpad=14)
    ax.tick_params(axis="both", labelsize=_PAPER_TICK_FS)


def _add_panel_label(ax: plt.Axes, label: str) -> None:
    """Place (a)/(b) in the upper-right corner of each panel."""
    if not label:
        return
    ax.text(
        0.98, 0.98, label,
        transform=ax.transAxes,
        ha="right", va="top",
        fontsize=_PAPER_PANEL_FS,
        clip_on=False,
    )


_XTICKS = np.arange(5, 55, 5)


def _apply_depth_axis(ax: plt.Axes, depths: np.ndarray | None = None) -> None:
    """Linear p-axis with tick marks every 5 (data may sit at non-multiples of 5)."""
    if depths is not None and len(depths):
        pad = 2.5
        ax.set_xlim(float(np.min(depths)) - pad, float(np.max(depths)) + pad)
    ax.set_xticks(_XTICKS)


def _legend_above(
    fig: plt.Figure,
    ax: plt.Axes,
    *,
    ncol: int | None = None,
    y_anchor: float | None = None,
) -> None:
    """Legend in a single horizontal row just above the plot area."""
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return
    ncols = ncol if ncol is not None else len(handles)
    legend_kw = dict(
        fontsize=_PAPER_LEGEND_FS,
        ncol=ncols,
        columnspacing=1.0,
        handlelength=2.4,
        handletextpad=0.5,
        frameon=False,
    )
    if len(fig.axes) == 1:
        ax.legend(
            handles,
            labels,
            bbox_to_anchor=(0.5, 1.02),
            loc="lower center",
            **legend_kw,
        )
    else:
        anchor_y = y_anchor if y_anchor is not None else 0.9
        fig.legend(
            handles,
            labels,
            bbox_to_anchor=(0.5, anchor_y),
            bbox_transform=fig.transFigure,
            loc="lower center",
            **legend_kw,
        )


def plot_window_stability(
    analysis: Dict[str, List[dict]],
    *,
    out_path: Path,
    ranges: List[Tuple[int, int]],
    ylabel: str,
    colorblind: bool = True,
    style: str = "full",
) -> None:
    rows = analysis.get("mean_p") or analysis.get("bm24_mean_p_fixed_n") or []
    if not rows:
        raise ValueError("no mean_p rows in analysis")

    depths = np.asarray([int(r["depth"]) for r in rows], dtype=float)
    _apply_paper_rcparams()

    if style == "full":
        _plot_full_overlay(rows, depths, ranges, out_path, ylabel=ylabel, colorblind=colorblind)
        return
    _plot_clean_summary(rows, depths, ranges, out_path, ylabel=ylabel, colorblind=colorblind)


def _plot_panel(
    ax: plt.Axes,
    rows: List[dict],
    depths: np.ndarray,
    ranges: List[Tuple[int, int]],
    *,
    ylabel: str,
    colorblind: bool,
    show_legend: bool = True,
    panel_label: str = "",
) -> None:
    style_by_range = {s["range"]: s for s in _SERIES_STYLES if s["range"] in ranges}
    for n_lo, n_hi in ranges:
        style = style_by_range.get((n_lo, n_hi))
        if style is None:
            continue
        slopes = _slopes_for_range(rows, n_lo, n_hi)
        ax.plot(
            depths,
            slopes,
            color=style["color"],
            ls=style["ls"],
            marker=style["marker"],
            lw=style["lw"],
            ms=7.0,
            mfc="white" if colorblind else style["color"],
            mec=style["color"],
            mew=1.4,
            label=rf"$n={n_lo}$–${n_hi}$",
            zorder=style["zorder"],
        )

    _style_axis_labels(ax, xlabel=r"Depth ($p$)", ylabel=ylabel)
    _apply_depth_axis(ax, depths)
    ax.grid(True, alpha=0.16, linewidth=0.6)
    if show_legend:
        ax.legend(loc="upper right", frameon=False, handlelength=2.2, fontsize=_PAPER_LEGEND_FS)
    _add_panel_label(ax, panel_label)


def _common_depths(*analyses: Dict[str, List[dict]]) -> np.ndarray:
    depth_sets = []
    for analysis in analyses:
        rows = analysis.get("mean_p") or analysis.get("bm24_mean_p_fixed_n") or []
        depth_sets.append({int(r["depth"]) for r in rows})
    if not depth_sets:
        return np.array([], dtype=float)
    common = sorted(set.intersection(*depth_sets))
    return np.asarray(common, dtype=float)


def _filter_rows_to_depths(rows: List[dict], depths: Sequence[int]) -> List[dict]:
    wanted = set(int(d) for d in depths)
    return [r for r in rows if int(r["depth"]) in wanted]


def _load_walksat_baselines(path: Path) -> dict | None:
    """Return {walksat: slope, walksatlm: slope} for n=12-20 if JSON exists."""
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    slopes = (payload.get("slopes_log2") or {}).get("12-20") or {}
    out = {
        "walksat": float(slopes.get("walksat", float("nan"))),
        "walksatlm": float(slopes.get("walksatlm", float("nan"))),
    }
    if not any(np.isfinite(v) for v in out.values()):
        return None
    return out


def _draw_walksat_baselines(ax: plt.Axes, baselines: dict | None) -> None:
    if not baselines:
        return
    styles = [
        ("walksatlm", "0.35", "--", "WalkSATlm"),
        ("walksat", "0.55", ":", "WalkSAT"),
    ]
    for key, color, ls, label in styles:
        val = baselines.get(key, float("nan"))
        if not np.isfinite(val):
            continue
        ax.axhline(val, color=color, ls=ls, lw=1.4, zorder=1, label="_nolegend_")
        ax.text(
            0.985, val, f"{label} ({val:.3f})",
            transform=ax.get_yaxis_transform(),
            ha="right", va="bottom",
            fontsize=11, color=color,
            clip_on=False,
        )


def _panel_y_limits(rows_list: Sequence[List[dict]], ranges: List[Tuple[int, int]]) -> Tuple[float, float]:
    vals: List[float] = []
    for rows in rows_list:
        for n_lo, n_hi in ranges:
            vals.extend(_slopes_for_range(rows, n_lo, n_hi))
    finite = [v for v in vals if np.isfinite(v)]
    if not finite:
        return 0.0, 1.0
    y_lo, y_hi = float(np.min(finite)), float(np.max(finite))
    pad = 0.04 * (y_hi - y_lo)
    return y_lo - pad, y_hi + pad


def plot_side_by_side(
    median_analysis: Dict[str, List[dict]],
    mean_analysis: Dict[str, List[dict]],
    *,
    out_path: Path,
    ranges: List[Tuple[int, int]],
    colorblind: bool = True,
    walksat_baselines: dict | None = None,
) -> None:
    _apply_paper_rcparams()
    med_rows = median_analysis.get("mean_p") or []
    mean_rows = mean_analysis.get("mean_p") or []
    if not med_rows or not mean_rows:
        raise ValueError("both analyses required for side-by-side plot")

    depths = _common_depths(median_analysis, mean_analysis)
    if len(depths) == 0:
        raise ValueError("no common depths between median-runtime and mean-success analyses")

    med_rows = _filter_rows_to_depths(med_rows, depths)
    mean_rows = _filter_rows_to_depths(mean_rows, depths)
    y_lo, y_hi = _panel_y_limits([med_rows, mean_rows], ranges)
    if walksat_baselines:
        for v in walksat_baselines.values():
            if np.isfinite(v):
                y_lo = min(y_lo, float(v))
                y_hi = max(y_hi, float(v))
    y_pad = 0.04 * (y_hi - y_lo)
    y_lo -= y_pad
    y_hi += y_pad

    fig, axes = plt.subplots(
        1, 2, figsize=(13.5, 4.4), sharey=True,
        gridspec_kw={"wspace": 0.12},
    )

    _plot_panel(
        axes[0], mean_rows, depths, ranges,
        ylabel=_METRIC_CFG["mean_success"]["ylabel"],
        colorblind=colorblind,
        show_legend=False,
        panel_label="(a)",
    )
    _plot_panel(
        axes[1], med_rows, depths, ranges,
        ylabel=_METRIC_CFG["median_runtime"]["ylabel"],
        colorblind=colorblind,
        show_legend=False,
        panel_label="(b)",
    )
    for ax in axes:
        _draw_walksat_baselines(ax, walksat_baselines)
        ax.set_ylim(y_lo, y_hi)

    # Legend just above panels; keep top margin tight (no large empty band).
    fig.subplots_adjust(left=0.10, right=0.99, top=0.86, bottom=0.14, wspace=0.12)
    axes_top = max(ax.get_position().y1 for ax in axes)
    _legend_above(fig, axes[0], ncol=3, y_anchor=axes_top + 0.01)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, facecolor="white", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"Wrote {out_path}")


def _plot_full_overlay(
    rows: List[dict],
    depths: np.ndarray,
    ranges: List[Tuple[int, int]],
    out_path: Path,
    *,
    ylabel: str,
    colorblind: bool,
) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 4.7))
    _plot_panel(ax, rows, depths, ranges, ylabel=ylabel, colorblind=colorblind, show_legend=False)
    _legend_above(fig, ax, ncol=3)
    fig.tight_layout(rect=[0, 0, 1, 0.84])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, facecolor="white")
    plt.close(fig)
    print(f"Wrote {out_path}")


def _plot_clean_summary(
    rows: List[dict],
    depths: np.ndarray,
    ranges: List[Tuple[int, int]],
    out_path: Path,
    *,
    ylabel: str,
    colorblind: bool,
) -> None:
    sliding = [w for w in _SLIDING_WINDOWS if w in ranges]
    highlights = [w for w in _HIGHLIGHT_WINDOWS if w in ranges]

    sliding_by_depth = np.vstack([
        [_slopes_for_range(rows, lo, hi) for lo, hi in sliding]
    ]) if sliding else np.empty((0, len(depths)))
    all_by_depth = np.vstack([
        [_slopes_for_range(rows, lo, hi) for lo, hi in ranges]
    ])

    band_lo = np.nanmin(sliding_by_depth, axis=0) if sliding else np.full(len(depths), np.nan)
    band_hi = np.nanmax(sliding_by_depth, axis=0) if sliding else np.full(len(depths), np.nan)
    spread = np.nanmax(all_by_depth, axis=0) - np.nanmin(all_by_depth, axis=0)

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(7.0, 5.4), sharex=True,
        gridspec_kw={"height_ratios": [2.4, 1.0], "hspace": 0.08},
    )

    if sliding:
        ax_top.fill_between(
            depths, band_lo, band_hi,
            color="#56B4E9", alpha=0.28, linewidth=0,
            label=r"4-value window ($n{=}12$–$15$)",
            zorder=1,
        )

    for n_lo, n_hi in highlights:
        st = next(s for s in _SERIES_STYLES if s["range"] == (n_lo, n_hi))
        slopes = _slopes_for_range(rows, n_lo, n_hi)
        ax_top.plot(
            depths,
            slopes,
            color=st["color"],
            ls=st["ls"],
            marker=st["marker"],
            lw=st["lw"],
            ms=7.0,
            mfc="white" if colorblind else st["color"],
            mec=st["color"],
            mew=1.4,
            label=rf"$n={n_lo}$–${n_hi}$",
            zorder=st["zorder"],
        )

    all_slopes = np.concatenate([
        band_lo, band_hi,
        *[_slopes_for_range(rows, lo, hi) for lo, hi in highlights],
    ])
    finite = all_slopes[np.isfinite(all_slopes)]
    y_pad = 0.03 * (np.nanmax(finite) - np.nanmin(finite))
    ax_top.set_ylim(np.nanmin(finite) - y_pad, np.nanmax(finite) + y_pad)
    ax_top.set_ylabel(ylabel)
    ax_top.legend(loc="upper right", frameon=False, handlelength=2.6)
    ax_top.grid(True, axis="y", alpha=0.22)
    ax_top.tick_params(labelbottom=False)

    ax_bot.plot(
        depths, spread,
        color="#D55E00", marker="o", lw=1.8, ms=4.8,
        mfc="white" if colorblind else "#D55E00",
        mec="#D55E00", mew=1.1,
        zorder=3,
    )
    ax_bot.set_xlabel(r"QAOA depth $p$")
    ax_bot.set_ylabel(r"window spread")
    _apply_depth_axis(ax_bot, depths)
    ax_bot.set_ylim(bottom=0.0)
    ax_bot.grid(True, alpha=0.22)

    fig.subplots_adjust(left=0.12, right=0.97, top=0.94, bottom=0.11)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {out_path}")


def _build_modes_data(
    metric: str,
    *,
    cache_path: Path,
    extra_paths: Sequence[Path],
    ctyp_cache: Path,
    wider_n9_11: Path,
    wider_n19_20: Path,
    wider_n16_20: Path,
) -> Dict[str, List[dict]]:
    cfg = _METRIC_CFG[metric]
    if metric == "median_runtime":
        modes_data = load_per_n_cache(cache_path, None)
        return _merge_extra_caches(modes_data, extra_paths, per_n_key=cfg["per_n_key"])
    return load_mean_success_modes_data(
        ctyp_cache=ctyp_cache,
        wider_n9_11=wider_n9_11,
        wider_n19_20=wider_n19_20,
        wider_n16_20=wider_n16_20,
    )


def _analyze_metric(
    metric: str,
    *,
    cache_path: Path,
    extra_paths: Sequence[Path],
    ctyp_cache: Path,
    wider_n9_11: Path,
    wider_n19_20: Path,
    wider_n16_20: Path,
    ranges: List[Tuple[int, int]],
) -> Dict[str, List[dict]]:
    modes_data = _build_modes_data(
        metric,
        cache_path=cache_path,
        extra_paths=extra_paths,
        ctyp_cache=ctyp_cache,
        wider_n9_11=wider_n9_11,
        wider_n19_20=wider_n19_20,
        wider_n16_20=wider_n16_20,
    )
    cfg = _METRIC_CFG[metric]
    if metric == "median_runtime":
        return analyze_ranges(modes_data, ranges)
    return analyze_ranges_metric(
        modes_data,
        ranges,
        per_n_key=cfg["per_n_key"],
        slope_sign=cfg["slope_sign"],
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--metric",
        choices=(*tuple(_METRIC_CFG), "both"),
        default="both",
    )
    ap.add_argument("--per-n-cache", type=Path, default=_DEFAULT_CACHE)
    ap.add_argument("--ctyp-cache", type=Path, default=_DEFAULT_CTYP_CACHE)
    ap.add_argument("--wider-n9-11", type=Path, default=_DEFAULT_WIDER_N9_11)
    ap.add_argument("--wider-n19-20", type=Path, default=_DEFAULT_WIDER_N19_20)
    ap.add_argument("--wider-n16-20", type=Path, default=_DEFAULT_WIDER_N16_20)
    ap.add_argument("--walksat-json", type=Path, default=_DEFAULT_WALKSAT_JSON)
    ap.add_argument("--extra-per-n-cache", type=Path, action="append", default=None)
    ap.add_argument("--ranges", default="9:16,12:15,12:20")
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=_PHASECRAFT / "bm24_runs/analysis/subwindow_slopes",
    )
    ap.add_argument(
        "--style",
        choices=("clean", "full"),
        default="full",
    )
    args = ap.parse_args()

    metric_cfg = _METRIC_CFG.get(args.metric, {})
    cache_path = args.per_n_cache.resolve()
    extra_paths = [p.resolve() for p in (args.extra_per_n_cache or [])]
    if args.metric in ("median_runtime", "both"):
        for default_extra in (_DEFAULT_N9_11_CACHE, _DEFAULT_EXTRA_CACHE):
            if default_extra.is_file() and default_extra.resolve() not in extra_paths:
                extra_paths.append(default_extra.resolve())

    ranges = parse_ranges(args.ranges)
    out_dir = args.output_dir.resolve()
    range_tag = "_".join(f"n{lo}-{hi}" for lo, hi in ranges)
    walksat_baselines = _load_walksat_baselines(args.walksat_json.resolve())
    if walksat_baselines:
        print(
            "WalkSAT baselines (n=12–20): "
            f"WalkSAT={walksat_baselines.get('walksat', float('nan')):.4f}  "
            f"WalkSATlm={walksat_baselines.get('walksatlm', float('nan')):.4f}"
        )
    else:
        print(f"No WalkSAT JSON at {args.walksat_json} (plotting without baselines)")

    metrics = list(_METRIC_CFG) if args.metric == "both" else [args.metric]
    analyses: Dict[str, Dict[str, List[dict]]] = {}
    for metric in metrics:
        analyses[metric] = _analyze_metric(
            metric,
            cache_path=cache_path,
            extra_paths=extra_paths,
            ctyp_cache=args.ctyp_cache.resolve(),
            wider_n9_11=args.wider_n9_11.resolve(),
            wider_n19_20=args.wider_n19_20.resolve(),
            wider_n16_20=args.wider_n16_20.resolve(),
            ranges=ranges,
        )
        json_out = out_dir / f"{_METRIC_CFG[metric]['stem']}_{range_tag}.json"
        json_out.write_text(json.dumps(analyses[metric], indent=2), encoding="utf-8")
        print(f"Wrote {json_out}")

        for colorblind, suffix in ((True, "colorblind"), (False, "clear")):
            plot_window_stability(
                analyses[metric],
                out_path=out_dir / f"{_METRIC_CFG[metric]['stem']}_{suffix}.png",
                ranges=ranges,
                ylabel=_METRIC_CFG[metric]["ylabel"],
                colorblind=colorblind,
                style=args.style,
            )

    if args.metric == "both":
        for colorblind, suffix in ((True, "colorblind"), (False, "clear")):
            plot_side_by_side(
                analyses["median_runtime"],
                analyses["mean_success"],
                out_path=out_dir / f"window_stability_side_by_side_{suffix}.png",
                ranges=ranges,
                colorblind=colorblind,
                walksat_baselines=walksat_baselines,
            )


if __name__ == "__main__":
    main()
