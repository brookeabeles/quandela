#!/usr/bin/env python3
"""
Aggregate multi-seed mean_p vs median_rt A/B runs.

Reads manifest from run_multi_seed_objective_ab.py (plus optional seed-27 baseline).

Example::

    python experiments/lr_scaling/plot_multi_seed_objective_ab.py
    python experiments/lr_scaling/plot_multi_seed_objective_ab.py --with-ctyp-cann
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

_PHASECRAFT = Path(__file__).resolve().parents[2]
_LR = Path(__file__).resolve().parent
for _p in (_PHASECRAFT.parent, _PHASECRAFT, _LR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
from phasecraft.lib.paths import bm24_runs_dir  # noqa: E402
from plot_compare_run1_run4 import (  # noqa: E402
    _draw_baseline_refs,
    classical_slopes,
    load_classical_baselines,
)

_DEFAULT_MANIFEST = bm24_runs_dir() / "multi_seed_ctyp_cann" / "manifest.json"
_BASELINE = bm24_runs_dir() / "06-09/run1/train12-tr100-te200-n12-18.json"
FAIR_WINDOW: Tuple[int, int] = (14, 18)
# BM24 random k-SAT median-runtime exponent scaling (k=8): c ≈ A p^{-B}
_K8_ANALYTIC_AMP = 0.69
_K8_ANALYTIC_EXP = 0.32


def _traces(payload: dict) -> Dict[str, List[dict]]:
    if "traces_by_mode" in payload:
        return {k: list(v) for k, v in payload["traces_by_mode"].items()}
    out: Dict[str, List[dict]] = {}
    for key in ("trace_bm24_mean_p_fixed_n", "trace_median_runtime_fixed_n"):
        if key in payload and payload[key]:
            out[key.replace("trace_", "")] = list(payload[key])
    return out


def load_runs(manifest_path: Path, include_baseline: bool) -> List[dict]:
    runs: List[dict] = []
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for rec in manifest.get("runs", []):
            jp = rec.get("json")
            if jp and Path(jp).is_file() and rec.get("status") in ("ok", "baseline", None):
                runs.append({"seed": rec["seed"], "path": Path(jp)})
    if include_baseline and _BASELINE.is_file():
        if not any(r["seed"] == 27 for r in runs):
            runs.append({"seed": 27, "path": _BASELINE})
    runs.sort(key=lambda r: int(r["seed"]))
    return runs


def extract_deltas(run: dict) -> List[dict]:
    payload = json.loads(run["path"].read_text(encoding="utf-8"))
    traces = _traces(payload)
    mean_tr = {int(r["depth"]): r for r in traces.get("bm24_mean_p_fixed_n", [])}
    med_tr = {int(r["depth"]): r for r in traces.get("median_runtime_fixed_n", [])}
    rows: List[dict] = []
    for depth in sorted(set(mean_tr) & set(med_tr)):
        c_mean = float(mean_tr[depth]["lr_log2_slope"])
        c_med = float(med_tr[depth]["lr_log2_slope"])
        rows.append({
            "seed": int(run["seed"]),
            "depth": depth,
            "c_typ_mean_p": c_mean,
            "c_typ_median_rt": c_med,
            "delta_ctyp": c_med - c_mean,
        })
    return rows


def _filter_rows(all_rows: List[dict], depths: Optional[List[int]]) -> Tuple[List[dict], List[int]]:
    if depths:
        depth_set = set(depths)
        rows = [r for r in all_rows if int(r["depth"]) in depth_set]
        use_depths = [d for d in depths if any(int(r["depth"]) == d for r in rows)]
    else:
        rows = list(all_rows)
        use_depths = sorted({int(r["depth"]) for r in rows})
    return rows, use_depths


from scipy.stats import linregress

LN2 = float(np.log(2.0))


def fit_log2_slope(ns: Sequence[int], ys: Sequence[float], *, signed: bool = False) -> float:
    n_arr = np.asarray(list(ns), dtype=float)
    y_arr = np.asarray(list(ys), dtype=float)
    mask = np.isfinite(y_arr)
    if not signed:
        mask &= y_arr > 0
    if int(mask.sum()) < 2:
        return float("nan")
    return float(linregress(n_arr[mask], np.log(y_arr[mask])).slope / LN2)


def fit_log2_slope_window(
    per_n: dict,
    n_lo: int,
    n_hi: int,
    *,
    signed: bool = False,
) -> float:
    """log2 slope from per-n dict keyed by str(n), restricted to [n_lo, n_hi]."""
    ns = [n for n in range(n_lo, n_hi + 1) if str(n) in per_n or n in per_n]
    ys = [float(per_n.get(str(n), per_n.get(n))) for n in ns]
    return fit_log2_slope(ns, ys, signed=signed)


def _parse_window(text: str) -> Tuple[int, int]:
    parts = [int(x.strip()) for x in text.split(",") if x.strip()]
    if len(parts) != 2:
        raise ValueError(f"expected lo,hi window, got {text!r}")
    return parts[0], parts[1]


def _train_n_from_runs(runs: List[dict]) -> int:
    if not runs:
        return 12
    payload = json.loads(runs[0]["path"].read_text(encoding="utf-8"))
    cfg = payload.get("config", {})
    return int(cfg.get("train_n", payload.get("train_n", 12)))


def _ctyp_summary_path(run_path: Path) -> Path:
    stem = run_path.stem
    for root in (
        _PHASECRAFT / "bm24_runs/analysis/ctyp_vs_cann",
        _PHASECRAFT / "results/bm24_runs/analysis/ctyp_vs_cann",
    ):
        candidate = root / stem / "ctyp_vs_cann_summary.json"
        if candidate.is_file():
            return candidate
    return root / stem / "ctyp_vs_cann_summary.json"


def _ctyp_cache_path(run_path: Path) -> Path:
    return run_path.parent / f"{run_path.stem}-ctyp-cann-cache.json"


def _merge_mean_p_rows_from_cache(
    rows: List[dict],
    run: dict,
    *,
    mode: str = "bm24_mean_p_fixed_n",
    focus_depths: Optional[List[int]] = None,
) -> List[dict]:
    """Fill in partial rebenchmark rows from incremental ctyp-cann cache."""
    cache_path = _ctyp_cache_path(run["path"])
    if not cache_path.is_file():
        return rows
    cache = json.loads(cache_path.read_text(encoding="utf-8"))
    rb = cache.get("rebenchmark") or {}
    depth_set = set(focus_depths) if focus_depths else None
    by_depth = {int(r["depth"]): r for r in rows if int(r["seed"]) == int(run["seed"])}
    for key, entry in rb.items():
        if "|" not in key:
            continue
        mode_part, depth_s = key.rsplit("|", 1)
        if mode_part != mode:
            continue
        depth = int(depth_s)
        if depth_set is not None and depth not in depth_set:
            continue
        c_inv = entry.get("c_inv_mean_rt")
        if c_inv is None or not np.isfinite(float(c_inv)):
            continue
        c_inv = float(c_inv)
        row = by_depth.get(depth)
        if row is None:
            row = {
                "seed": int(run["seed"]),
                "depth": depth,
                "c_inv_mean": c_inv,
                "c_emp_mean": float(entry.get("c_emp_mean", float("nan"))),
                "has_mean_rebenchmark": True,
            }
            rows.append(row)
            by_depth[depth] = row
        elif not row.get("has_mean_rebenchmark") or not np.isfinite(
            float(row.get("c_inv_mean", float("nan")))
        ):
            row["c_inv_mean"] = c_inv
            if entry.get("c_emp_mean") is not None:
                row["c_emp_mean"] = float(entry["c_emp_mean"])
            row["has_mean_rebenchmark"] = True
    return rows


def extract_exponent_equivalence_rows(
    runs: List[dict],
    *,
    focus_depths: Optional[List[int]] = None,
) -> List[dict]:
    """Same mean_p training: c_typ (median 1/p) vs inverted mean-p exponent."""
    rows: List[dict] = []
    depth_set = set(focus_depths) if focus_depths else None

    for run in runs:
        payload = json.loads(run["path"].read_text(encoding="utf-8"))
        traces = _traces(payload)
        mean_tr = {int(r["depth"]): r for r in traces.get("bm24_mean_p_fixed_n", [])}
        cfg = payload.get("config", {})
        ns = payload.get("n_values") or list(
            range(int(cfg["n_min"]), int(cfg["n_max"]) + 1)
        )

        summary_by_depth: Dict[int, dict] = {}
        summary_path = _ctyp_summary_path(run["path"])
        if summary_path.is_file():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            for row in summary.get("rows", []):
                if row.get("mode") != "bm24_mean_p_fixed_n":
                    continue
                summary_by_depth[int(row["depth"])] = row

        for depth, tr in sorted(mean_tr.items()):
            if depth_set is not None and depth not in depth_set:
                continue
            c_typ = float(tr.get("lr_log2_slope", float("nan")))
            med_per_n = tr.get("median_runtime_per_n") or {}
            if med_per_n:
                med_ns = sorted(int(n) for n in med_per_n)
                med_ys = [float(med_per_n[str(n)]) for n in med_ns]
                c_typ_fit = fit_log2_slope(med_ns, med_ys)
            else:
                c_typ_fit = float("nan")

            summ = summary_by_depth.get(depth, {})
            c_inv_mean = summ.get("c_inv_mean_rt")
            c_emp_mean = summ.get("c_emp_mean")
            if c_inv_mean is None and c_emp_mean is not None:
                c_inv_mean = -float(c_emp_mean)
            c_inv_mean = float(c_inv_mean) if c_inv_mean is not None else float("nan")

            rows.append({
                "seed": int(run["seed"]),
                "depth": depth,
                "c_typ": c_typ,
                "c_typ_fit": c_typ_fit,
                "c_inv_mean": c_inv_mean,
                "gap_typ_minus_invmean": (
                    c_typ - c_inv_mean if np.isfinite(c_typ) and np.isfinite(c_inv_mean) else float("nan")
                ),
                "has_rebenchmark": np.isfinite(c_inv_mean),
            })
    return rows


def plot_exponent_equivalence(
    equiv_rows: List[dict],
    out_dir: Path,
    *,
    focus_depths: Optional[List[int]] = None,
) -> Path:
    """
  Same training (mean_p): overlay runtime exponent c_typ with inverted mean-p exponent.

  c_typ = log₂ slope of median(1/p_succ) vs n  (equivalently median runtime)
  c_inv_mean = log₂ slope of mean(1/p_succ) vs n = −(log₂ slope of mean p_succ)
  """
    out_dir.mkdir(parents=True, exist_ok=True)
    depths = focus_depths or sorted({int(r["depth"]) for r in equiv_rows})
    seeds = sorted({int(r["seed"]) for r in equiv_rows})

    cmap = plt.cm.tab10
    seed_colors = {seed: cmap(i % 10) for i, seed in enumerate(seeds)}
    dash = (8, 4)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), gridspec_kw={"width_ratios": [2.1, 1]})
    ax = axes[0]
    all_vals: List[float] = []

    for seed in seeds:
        sub = [r for r in equiv_rows if int(r["seed"]) == seed]
        c_typ_map = {int(r["depth"]): r["c_typ"] for r in sub}
        c_inv_map = {int(r["depth"]): r["c_inv_mean"] for r in sub}
        color = seed_colors[seed]
        ys_typ = [c_typ_map.get(d, float("nan")) for d in depths]
        ys_inv = [c_inv_map.get(d, float("nan")) for d in depths]
        for y in ys_typ + ys_inv:
            if np.isfinite(y):
                all_vals.append(y)
        ax.plot(
            depths, ys_typ, "o-", color=color, lw=2.3, ms=8, alpha=0.95, zorder=3,
        )
        ax.plot(
            depths, ys_inv, linestyle="--", dashes=dash, color=color, lw=2.0, ms=8,
            marker="s", markerfacecolor="white", markeredgecolor=color,
            markeredgewidth=1.6, alpha=0.9, zorder=2,
        )

    y_min = float(np.nanmin(all_vals))
    y_max = float(np.nanmax(all_vals))
    ax.set_ylim(y_min - 0.02, y_max + 0.05)
    ax.set_xlabel("depth p", fontsize=11)
    ax.set_ylabel(r"$\log_2$ scaling exponent vs $n$", fontsize=11)
    ax.set_title("Same training (mean_p): runtime vs inverted mean-p exponent", fontsize=11)
    ax.set_xticks(depths)
    ax.set_xticklabels([str(d) for d in depths])
    ax.grid(True, alpha=0.3)

    style_handles = [
        Line2D([0], [0], color="#333", linestyle="-", lw=2.5, marker="o", ms=8,
               label=r"$c_{\mathrm{typ}}$ = slope of median($1/p_{\mathrm{succ}}$)"),
        Line2D([0], [0], color="#333", linestyle="--", dashes=dash, lw=2.5,
               marker="s", markerfacecolor="white", markeredgecolor="#333", ms=8,
               label=r"$c_{1/\langle p\rangle}$ = slope of mean($1/p_{\mathrm{succ}}$)"),
    ]
    seed_handles = [
        Line2D([0], [0], color=seed_colors[s], linestyle="-", lw=2.3, marker="o", ms=8,
               label=f"seed {s}")
        for s in seeds
    ]
    leg1 = ax.legend(handles=style_handles, fontsize=9, loc="upper right", title="Exponent", title_fontsize=9)
    ax.add_artist(leg1)
    ax.legend(handles=seed_handles, fontsize=9, loc="upper center", ncol=len(seeds),
              title="Colour = seed", title_fontsize=9, framealpha=0.95)

    ax2 = axes[1]
    for seed in seeds:
        sub = [r for r in equiv_rows if int(r["seed"]) == seed and r["has_rebenchmark"]]
        if not sub:
            continue
        dmap = {int(r["depth"]): r["gap_typ_minus_invmean"] for r in sub}
        ax2.plot(
            [d for d in depths if d in dmap],
            [dmap[d] for d in depths if d in dmap],
            "o-", color=seed_colors[seed], lw=2, ms=7, label=f"seed {seed}",
        )
    ax2.axhline(0, color="k", lw=0.8, alpha=0.35)
    ax2.set_xlabel("depth p", fontsize=11)
    ax2.set_ylabel(r"$c_{\mathrm{typ}} - c_{1/\langle p\rangle}$", fontsize=11)
    ax2.set_title("Equivalence gap\n(→ 0 if median ≈ mean)", fontsize=11)
    ax2.set_xticks(depths)
    ax2.set_xticklabels([str(d) for d in depths])
    ax2.grid(True, alpha=0.3)
    if any(r["has_rebenchmark"] for r in equiv_rows):
        ax2.legend(fontsize=8)

    fig.suptitle(
        "Identical mean_p training — solid: median(1/p), dashed: mean(1/p)",
        fontsize=12, y=1.02,
    )
    fig.tight_layout()
    png = out_dir / "multi_seed_exponent_equivalence.png"
    fig.savefig(png, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {png}")
    return png


def extract_ctyp_by_seed(
    runs: List[dict],
    *,
    mode: str = "bm24_mean_p_fixed_n",
    eval_window: Optional[Tuple[int, int]] = None,
) -> List[dict]:
    """Per-seed c_typ: refit log2 slope of median(1/p) vs n on eval_window."""
    rows: List[dict] = []
    for run in runs:
        payload = json.loads(run["path"].read_text(encoding="utf-8"))
        traces = _traces(payload)
        cfg = payload.get("config", {})
        n_lo, n_hi = eval_window or (
            int(payload.get("n_min", cfg.get("n_min", 12))),
            int(payload.get("n_max", cfg.get("n_max", 18))),
        )
        for row in traces.get(mode, []):
            med_per_n = row.get("median_runtime_per_n") or {}
            if med_per_n:
                c_typ = fit_log2_slope_window(med_per_n, n_lo, n_hi)
            else:
                c_typ = float(row.get("lr_log2_slope", float("nan")))
            rows.append({
                "seed": int(run["seed"]),
                "depth": int(row["depth"]),
                "c_typ": c_typ,
            })
    return rows


def extract_mean_p_exponent_rows(
    runs: List[dict],
    *,
    mode: str = "bm24_mean_p_fixed_n",
    eval_window: Optional[Tuple[int, int]] = None,
    focus_depths: Optional[List[int]] = None,
) -> List[dict]:
    """
    Per-seed mean-success exponent from rebenchmark summaries when available.

    Returns c_inv_mean = log2 slope of mean(1/p_succ) vs n (same units as c_typ).
    Requires ctyp_vs_cann_summary.json (from plot_ctyp_vs_cann.py rebenchmark).
    """
    depth_set = set(focus_depths) if focus_depths else None
    rows: List[dict] = []
    for run in runs:
        payload = json.loads(run["path"].read_text(encoding="utf-8"))
        traces = _traces(payload)
        cfg = payload.get("config", {})
        n_lo, n_hi = eval_window or (
            int(payload.get("n_min", cfg.get("n_min", 12))),
            int(payload.get("n_max", cfg.get("n_max", 18))),
        )
        summary_by_depth: Dict[int, dict] = {}
        summary_path = _ctyp_summary_path(run["path"])
        if summary_path.is_file():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            for srow in summary.get("rows", []):
                if srow.get("mode") != mode:
                    continue
                summary_by_depth[int(srow["depth"])] = srow

        for row in traces.get(mode, []):
            depth = int(row["depth"])
            if depth_set is not None and depth not in depth_set:
                continue
            c_inv_mean = float("nan")
            c_emp_mean = float("nan")
            summ = summary_by_depth.get(depth, {})
            per_n = summ.get("per_n") or {}
            inv = per_n.get("inverse_mean")
            mean_ps = per_n.get("mean_p")
            ns_raw = per_n.get("ns")
            if inv and ns_raw:
                pairs = [
                    (int(n), float(v))
                    for n, v in zip(ns_raw, inv)
                    if n_lo <= int(n) <= n_hi
                ]
                if len(pairs) >= 2:
                    c_inv_mean = fit_log2_slope([p[0] for p in pairs], [p[1] for p in pairs])
            elif summ.get("c_inv_mean_rt") is not None:
                c_inv_mean = float(summ["c_inv_mean_rt"])
            if mean_ps and ns_raw:
                pairs_m = [
                    (int(n), float(v))
                    for n, v in zip(ns_raw, mean_ps)
                    if n_lo <= int(n) <= n_hi
                ]
                if len(pairs_m) >= 2:
                    c_emp_mean = fit_log2_slope(
                        [p[0] for p in pairs_m], [p[1] for p in pairs_m], signed=True
                    )
            elif summ.get("c_emp_mean") is not None:
                c_emp_mean = float(summ["c_emp_mean"])
            rows.append({
                "seed": int(run["seed"]),
                "depth": depth,
                "c_inv_mean": c_inv_mean,
                "c_emp_mean": c_emp_mean,
                "has_mean_rebenchmark": np.isfinite(c_inv_mean),
            })
        rows = _merge_mean_p_rows_from_cache(
            rows,
            run,
            mode=mode,
            focus_depths=focus_depths,
        )
    return rows


def _eval_window_from_runs(runs: List[dict]) -> Tuple[int, int]:
    if not runs:
        return 12, 18
    payload = json.loads(runs[0]["path"].read_text(encoding="utf-8"))
    cfg = payload.get("config", {})
    n_lo = int(payload.get("n_min", cfg.get("n_min", 12)))
    n_hi = int(payload.get("n_max", cfg.get("n_max", 18)))
    return n_lo, n_hi


def plot_seed_convergence(
    ctyp_rows: List[dict],
    out_dir: Path,
    *,
    ctyp_rows_median_rt: Optional[List[dict]] = None,
    focus_depths: Optional[List[int]] = None,
    train_n: int = 12,
    classical: Optional[dict] = None,
    eval_window: Optional[Tuple[int, int]] = None,
) -> Path:
    """Same training protocol across seeds: mean c_typ with seed-range error bars."""
    out_dir.mkdir(parents=True, exist_ok=True)
    depth_set = {int(r["depth"]) for r in ctyp_rows}
    if ctyp_rows_median_rt:
        depth_set |= {int(r["depth"]) for r in ctyp_rows_median_rt}
    depths = focus_depths or sorted(depth_set)
    seeds = sorted({int(r["seed"]) for r in ctyp_rows})
    if ctyp_rows_median_rt:
        seeds = sorted(set(seeds) | {int(r["seed"]) for r in ctyp_rows_median_rt})

    mean_mean, mean_lo, mean_hi, _ = _aggregate_seed_series(
        ctyp_rows, "c_typ", depths, seeds
    )
    yerr_mean_lo = [m - lo for m, lo in zip(mean_mean, mean_lo)]
    yerr_mean_hi = [hi - m for m, hi in zip(mean_mean, mean_hi)]

    ws_slope = lm_slope = float("nan")
    if classical and eval_window:
        ws_slope, lm_slope = classical_slopes(classical, eval_window[0], eval_window[1])

    all_vals = [v for v in mean_mean + mean_lo + mean_hi if np.isfinite(v)]
    if ctyp_rows_median_rt:
        med_mean, med_lo, med_hi, _ = _aggregate_seed_series(
            ctyp_rows_median_rt, "c_typ", depths, seeds
        )
        all_vals.extend(v for v in med_mean + med_lo + med_hi if np.isfinite(v))
    for v in (ws_slope, lm_slope):
        if np.isfinite(v):
            all_vals.append(v)

    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    ax.errorbar(
        depths,
        mean_mean,
        yerr=[yerr_mean_lo, yerr_mean_hi],
        fmt="o-",
        color="#4C72B0",
        mfc="white",
        mec="#4C72B0",
        mew=1.2,
        lw=2.0,
        ms=7,
        elinewidth=1.0,
        capsize=3.0,
        capthick=1.0,
        zorder=4,
        label=rf"mean$_p$ training ({len(seeds)} seeds)",
    )
    if ctyp_rows_median_rt:
        yerr_med_lo = [m - lo for m, lo in zip(med_mean, med_lo)]
        yerr_med_hi = [hi - m for m, hi in zip(med_mean, med_hi)]
        ax.errorbar(
            depths,
            med_mean,
            yerr=[yerr_med_lo, yerr_med_hi],
            fmt="s--",
            color="#DD8452",
            mfc="white",
            mec="#DD8452",
            mew=1.2,
            lw=2.0,
            ms=6,
            elinewidth=1.0,
            capsize=3.0,
            capthick=1.0,
            zorder=3,
            label=rf"median$_{{rt}}$ training ({len(seeds)} seeds)",
        )
    _draw_baseline_refs(ax, ws_slope, lm_slope, label_right=True)

    y_min = float(np.nanmin(all_vals))
    y_max = float(np.nanmax(all_vals))
    span = max(y_max - y_min, 0.05)
    pad = max(0.012, 0.08 * span)
    ax.set_ylim(y_min - pad, y_max + pad)
    ax.set_xlabel(r"QAOA depth $p$", fontsize=11)
    ax.set_ylabel(r"Median-runtime exponent $c_{\mathrm{typ}}$", fontsize=11)
    n_lo, n_hi = eval_window or (12, 18)
    obj_label = (
        r"mean$_p$ vs median$_{rt}$ training"
        if ctyp_rows_median_rt
        else r"mean$_p$ training"
    )
    ax.set_title(
        rf"Multi-seed $c_{{\mathrm{{typ}}}}$ convergence "
        rf"({len(seeds)} seeds, train $n={train_n}$, {obj_label}, "
        rf"$n \in [{n_lo},{n_hi}]$)",
        fontsize=11,
    )
    ax.set_xticks(depths)
    ax.set_xticklabels([str(d) for d in depths])
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="upper right", framealpha=0.95)
    fig.subplots_adjust(right=0.88)
    fig.tight_layout()

    png = out_dir / "multi_seed_convergence.png"
    fig.savefig(png, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {png}")
    return png


def _aggregate_seed_series(
    rows: List[dict],
    value_key: str,
    depths: List[int],
    seeds: List[int],
) -> Tuple[List[float], List[float], List[float], int]:
    """Mean and min–max across seeds for one exponent key."""
    by_seed: Dict[int, Dict[int, float]] = {s: {} for s in seeds}
    for r in rows:
        d = int(r["depth"])
        v = float(r[value_key])
        if d in depths and np.isfinite(v):
            by_seed[int(r["seed"])][d] = v
    seed_mean: List[float] = []
    seed_lo: List[float] = []
    seed_hi: List[float] = []
    n_used = 0
    for d in depths:
        vals = [by_seed[s][d] for s in seeds if d in by_seed[s]]
        n_used = max(n_used, len(vals))
        seed_mean.append(float(np.mean(vals)) if vals else float("nan"))
        seed_lo.append(float(np.min(vals)) if vals else float("nan"))
        seed_hi.append(float(np.max(vals)) if vals else float("nan"))
    return seed_mean, seed_lo, seed_hi, n_used


def _aggregate_seed_mean_sem(
    rows: List[dict],
    value_key: str,
    depths: List[int],
    seeds: List[int],
) -> Tuple[List[float], List[float], List[int]]:
    """Mean and standard error across available seeds for one exponent key."""
    by_seed: Dict[int, Dict[int, float]] = {s: {} for s in seeds}
    for r in rows:
        d = int(r["depth"])
        v = float(r[value_key])
        if d in depths and np.isfinite(v):
            by_seed[int(r["seed"])][d] = v

    means: List[float] = []
    sems: List[float] = []
    counts: List[int] = []
    for d in depths:
        vals = np.asarray([by_seed[s][d] for s in seeds if d in by_seed[s]], dtype=float)
        counts.append(int(vals.size))
        if vals.size:
            means.append(float(np.mean(vals)))
            sem = float(np.std(vals, ddof=1) / np.sqrt(vals.size)) if vals.size > 1 else 0.0
            sems.append(sem)
        else:
            means.append(float("nan"))
            sems.append(float("nan"))
    return means, sems, counts


def _seed_count_label(counts: Sequence[int]) -> str:
    finite = [int(c) for c in counts if int(c) > 0]
    if not finite:
        return "N=0"
    lo, hi = min(finite), max(finite)
    return f"N={hi}" if lo == hi else f"N={lo}-{hi}"


def _fit_offset_power(ps: np.ndarray, ys: np.ndarray) -> Optional[dict]:
    """Fit c(p) = c_inf + A p^(-alpha)."""
    mask = np.isfinite(ps) & np.isfinite(ys) & (ps > 0)
    if int(mask.sum()) < 3:
        return None
    ps_f = ps[mask]
    ys_f = ys[mask]

    def fn(p, c_inf, amp, alpha):
        return c_inf + amp * np.asarray(p, dtype=float) ** (-alpha)

    guesses = (
        [0.20, 0.8, 0.40],
        [0.25, 0.5, 0.30],
        [0.15, 1.0, 0.50],
    )
    best = None
    for p0 in guesses:
        try:
            popt, pcov = curve_fit(
                fn,
                ps_f,
                ys_f,
                p0=p0,
                bounds=([0.0, 0.0, 0.01], [1.0, 10.0, 5.0]),
                maxfev=20000,
            )
        except Exception:
            continue
        pred = fn(ps_f, *popt)
        ssr = float(np.sum((ys_f - pred) ** 2))
        if best is None or ssr < best["ssr"]:
            best = {
                "fn": fn,
                "params": popt,
                "param_err": np.sqrt(np.diag(pcov)),
                "ssr": ssr,
            }
    return best


def _fit_pure_power(ps: np.ndarray, ys: np.ndarray) -> Optional[dict]:
    """Fit c(p) = A p^(-alpha)."""
    mask = np.isfinite(ps) & np.isfinite(ys) & (ps > 0)
    if int(mask.sum()) < 3:
        return None
    ps_f = ps[mask]
    ys_f = ys[mask]

    def fn(p, amp, alpha):
        return amp * np.asarray(p, dtype=float) ** (-alpha)

    try:
        popt, pcov = curve_fit(
            fn,
            ps_f,
            ys_f,
            p0=[1.0, 0.3],
            bounds=([0.0, 0.0], [10.0, 5.0]),
            maxfev=20000,
        )
    except Exception:
        return None
    pred = fn(ps_f, *popt)
    return {
        "fn": fn,
        "params": popt,
        "param_err": np.sqrt(np.diag(pcov)),
        "ssr": float(np.sum((ys_f - pred) ** 2)),
    }


def plot_dual_exponent_convergence(
    ctyp_rows: List[dict],
    mean_rows: List[dict],
    out_dir: Path,
    *,
    focus_depths: Optional[List[int]] = None,
    train_n: int = 12,
    eval_window: Tuple[int, int] = (12, 18),
    classical: Optional[dict] = None,
    png_name: str = "multi_seed_exponent_convergence.png",
    thesis_style: bool = False,
    paper_style: bool = False,
) -> Path:
    """c_typ and inverse-mean-success exponents on one true-depth axis."""
    if paper_style:
        thesis_style = True
    out_dir.mkdir(parents=True, exist_ok=True)
    depths = focus_depths or sorted(
        {int(r["depth"]) for r in ctyp_rows} | {int(r["depth"]) for r in mean_rows}
    )
    seeds = sorted({int(r["seed"]) for r in ctyp_rows})
    n_lo, n_hi = eval_window

    typ_mean, typ_sem, typ_counts = _aggregate_seed_mean_sem(
        ctyp_rows, "c_typ", depths, seeds
    )
    mean_seeds = sorted({int(r["seed"]) for r in mean_rows if r.get("has_mean_rebenchmark")})
    inv_mean, inv_sem, inv_counts = _aggregate_seed_mean_sem(
        [r for r in mean_rows if r.get("has_mean_rebenchmark")],
        "c_inv_mean",
        depths,
        mean_seeds,
    )
    has_mean_line = any(np.isfinite(v) for v in inv_mean)

    ws_slope = lm_slope = float("nan")
    if classical:
        ws_slope, lm_slope = classical_slopes(classical, n_lo, n_hi)

    p_max = float(max(depths))
    all_vals = [
        v
        for v in (
            typ_mean
            + [m - e for m, e in zip(typ_mean, typ_sem)]
            + [m + e for m, e in zip(typ_mean, typ_sem)]
            + inv_mean
            + [m - e for m, e in zip(inv_mean, inv_sem)]
            + [m + e for m, e in zip(inv_mean, inv_sem)]
        )
        if np.isfinite(v)
    ]
    for v in (lm_slope,) if paper_style else (ws_slope, lm_slope):
        if np.isfinite(v):
            all_vals.append(v)

    color_typ = "#DD8452" if thesis_style else "#4C72B0"
    color_inv = "#2C7BB6" if thesis_style else "#DD8452"
    analytic_label = (
        "BM24 analytic mean success probability"
        if thesis_style
        else rf"$k{{=}}8$ analytic (${_K8_ANALYTIC_AMP}\,p^{{-{_K8_ANALYTIC_EXP}}}$)"
    )
    analytic_legend_key = "BM24 analytic" if thesis_style else "k=8 analytic"
    if paper_style:
        png_name = "multi_seed_exponent_convergence_thesis_paper.png"
    elif thesis_style and png_name == "multi_seed_exponent_convergence.png":
        png_name = "multi_seed_exponent_convergence_thesis.png"

    label_fs = 16 if paper_style else 11
    tick_fs = 13 if paper_style else None
    legend_fs = 12 if paper_style else 8.2
    baseline_fs = 12 if paper_style else 8.0
    xlabel = r"Depth ($p$)" if paper_style else r"QAOA depth $p$"
    ylabel = "Exponent" if paper_style else r"Scaling exponent $c$"
    paper_rc = (
        {
            "font.family": "serif",
            "mathtext.fontset": "cm",
            "axes.unicode_minus": False,
            "font.size": 14,
            "axes.labelsize": label_fs,
            "xtick.labelsize": tick_fs,
            "ytick.labelsize": tick_fs,
            "legend.fontsize": legend_fs,
        }
        if paper_style
        else {}
    )

    if paper_style:
        plt.rcParams.update(paper_rc)

    fig, ax = plt.subplots(figsize=(7.6, 4.7))
    ax.errorbar(
        depths,
        typ_mean,
        yerr=typ_sem,
        fmt="o-",
        color=color_typ,
        mfc="white",
        mec=color_typ,
        mew=1.2,
        lw=2.0,
        ms=7,
        elinewidth=1.0,
        capsize=3.0,
        capthick=1.0,
        zorder=4,
        label="Median runtime",
    )
    ps_typ = np.asarray(depths, dtype=float)
    ys_typ = np.asarray(typ_mean, dtype=float)
    mask_typ = np.isfinite(ys_typ)
    if mask_typ.sum() >= 3:
        ps_f, ys_f = ps_typ[mask_typ], ys_typ[mask_typ]
        power_fit = _fit_offset_power(ps_f, ys_f)
        if power_fit is not None:
            fn = power_fit["fn"]
            p_line = np.linspace(float(np.min(ps_f)), p_max, 400)
            y_fit = fn(p_line, *power_fit["params"])
            ax.plot(
                p_line,
                y_fit,
                color=color_typ,
                lw=1.6,
                ls=":",
                alpha=0.85,
                zorder=2,
                label="Power-law fit",
            )
            all_vals.extend(y_fit.tolist())
    if has_mean_line:
        mean_label = (
            "Inverse mean success"
        )
        ax.errorbar(
            depths,
            inv_mean,
            yerr=inv_sem,
            fmt="s-",
            color=color_inv,
            mfc="white",
            mec=color_inv,
            mew=1.2,
            lw=2.0,
            ms=6,
            elinewidth=1.0,
            capsize=3.0,
            capthick=1.0,
            zorder=3,
            label=mean_label,
        )
        ps_inv = np.asarray(depths, dtype=float)
        ys_inv = np.asarray(inv_mean, dtype=float)
        mask_inv = np.isfinite(ys_inv)
        if mask_inv.sum() >= 3:
            ps_f, ys_f = ps_inv[mask_inv], ys_inv[mask_inv]
            power_fit = _fit_pure_power(ps_f, ys_f)
            if power_fit is not None:
                fn = power_fit["fn"]
                p_line = np.linspace(float(np.min(ps_f)), p_max, 400)
                y_fit = fn(p_line, *power_fit["params"])
                ax.plot(
                    p_line,
                    y_fit,
                    color=color_inv,
                    lw=1.6,
                    ls=":",
                    alpha=0.95,
                    zorder=2,
                    label="_nolegend_",
                )
                all_vals.extend(y_fit.tolist())
    p_k8 = np.linspace(1.0, p_max, 400)
    y_k8 = _K8_ANALYTIC_AMP * p_k8 ** (-_K8_ANALYTIC_EXP)
    ax.plot(
        p_k8,
        y_k8,
        color="#2C7BB6",
        lw=2.0,
        ls="--",
        alpha=0.9,
        zorder=2,
        label=analytic_label,
    )
    all_vals.extend(y_k8.tolist())
    baseline_label_offsets = (
        ((0.012, "bottom"), (-0.012, "top")) if thesis_style else None
    )
    _draw_baseline_refs(
        ax,
        float("nan") if paper_style else ws_slope,
        lm_slope,
        label_in_axes=True,
        label_offsets=baseline_label_offsets,
        label_fontsize=baseline_fs,
    )

    y_min = float(np.nanmin(all_vals))
    y_max = float(np.nanmax(all_vals))
    span = max(y_max - y_min, 0.05)
    pad = max(0.018, 0.12 * span)
    ax.set_ylim(y_min - pad, y_max + pad)
    ax.set_xlabel(xlabel, fontsize=label_fs)
    ax.set_ylabel(ylabel, fontsize=label_fs)
    if tick_fs is not None:
        ax.tick_params(axis="both", labelsize=tick_fs)
    x_ticks = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([str(d) for d in x_ticks])
    ax.grid(True, alpha=0.16, linewidth=0.6)
    handles, labels = ax.get_legend_handles_labels()
    legend_order = [
        "Median runtime",
        "Inverse mean success",
        "Power-law fit",
        analytic_legend_key,
    ]
    ordered: List[int] = []
    for key in legend_order:
        ordered.extend(i for i, label in enumerate(labels) if key in label and i not in ordered)
    ordered.extend(i for i in range(len(labels)) if i not in ordered)
    if paper_style:
        ax.legend(
            [handles[i] for i in ordered],
            [labels[i] for i in ordered],
            fontsize=legend_fs,
            loc="lower center",
            ncol=2,
            bbox_to_anchor=(0.5, 1.02),
            columnspacing=1.4,
            handlelength=2.2,
            handletextpad=0.5,
            frameon=False,
        )
        fig.subplots_adjust(right=0.88)
        fig.tight_layout(rect=[0, 0, 1, 0.84])
    else:
        ax.legend(
            [handles[i] for i in ordered],
            [labels[i] for i in ordered],
            fontsize=legend_fs,
            loc="upper center",
            ncol=2,
            bbox_to_anchor=(0.5, 0.99),
            columnspacing=1.2,
            handlelength=2.2,
            handletextpad=0.5,
            framealpha=0.95,
        )
        if not has_mean_line:
            ax.text(
                0.02,
                0.02,
                "Mean $1/p$ line: run plot_ctyp_vs_cann.py (rebenchmark) per seed",
                transform=ax.transAxes,
                fontsize=7.5,
                color="#555555",
                va="bottom",
            )
        fig.subplots_adjust(right=0.88)
        fig.tight_layout()

    png = out_dir / png_name
    fig.savefig(png, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {png}")
    return png


def plot_split_exponent_convergence_paper(
    ctyp_rows: List[dict],
    mean_rows: List[dict],
    out_dir: Path,
    *,
    focus_depths: Optional[List[int]] = None,
    eval_window: Tuple[int, int] = (12, 18),
    classical: Optional[dict] = None,
) -> List[Path]:
    """Three paper-style panels split from the combined exponent-convergence plot."""
    out_dir.mkdir(parents=True, exist_ok=True)
    depths = focus_depths or sorted(
        {int(r["depth"]) for r in ctyp_rows} | {int(r["depth"]) for r in mean_rows}
    )
    seeds = sorted({int(r["seed"]) for r in ctyp_rows})
    typ_mean, typ_sem, _ = _aggregate_seed_mean_sem(ctyp_rows, "c_typ", depths, seeds)

    mean_seeds = sorted({int(r["seed"]) for r in mean_rows if r.get("has_mean_rebenchmark")})
    inv_mean, inv_sem, _ = _aggregate_seed_mean_sem(
        [r for r in mean_rows if r.get("has_mean_rebenchmark")],
        "c_inv_mean",
        depths,
        mean_seeds,
    )

    _, lm_slope = classical_slopes(classical, eval_window[0], eval_window[1]) if classical else (float("nan"), float("nan"))

    plt.rcParams.update(
        {
            "font.family": "serif",
            "mathtext.fontset": "cm",
            "axes.unicode_minus": False,
            "font.size": 10,
            "axes.labelsize": 11,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 9,
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "grid.alpha": 0.18,
            "grid.linewidth": 0.45,
        }
    )

    color_runtime = "#0072B2"
    color_mean = "#CC79A7"
    color_analytic = "#009E73"
    color_walksat = "0.35"
    p_max = float(max(depths))
    p_k8 = np.linspace(1.0, p_max, 400)
    y_k8 = _K8_ANALYTIC_AMP * p_k8 ** (-_K8_ANALYTIC_EXP)

    def finish(fig, ax, stem: str, values: Sequence[float]) -> Path:
        vals = [float(v) for v in values if np.isfinite(v)]
        if vals:
            y_min, y_max = min(vals), max(vals)
            pad = max(0.018, 0.12 * max(y_max - y_min, 0.05))
            ax.set_ylim(y_min - pad, y_max + pad)
        ax.set_xlim(0, p_max + 3)
        ax.set_xticks([d for d in range(0, int(p_max) + 1, 10)])
        ax.set_xlabel(r"Depth ($p$)")
        ax.set_ylabel("Exponent")
        ax.grid(True, alpha=0.18, linewidth=0.45)
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, 1.20),
            ncol=2,
            frameon=False,
            columnspacing=1.0,
            handlelength=1.6,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.90))
        png = out_dir / f"{stem}.png"
        pdf = out_dir / f"{stem}.pdf"
        fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
        fig.savefig(pdf, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"Wrote {png}")
        print(f"Wrote {pdf}")
        return png

    written: List[Path] = []

    fig, ax = plt.subplots(figsize=(5.2, 3.05))
    ax.errorbar(
        depths,
        typ_mean,
        yerr=typ_sem,
        fmt="o-",
        color=color_runtime,
        mfc="white",
        mec=color_runtime,
        mew=0.9,
        lw=1.45,
        ms=4.8,
        elinewidth=0.75,
        capsize=2.0,
        capthick=0.75,
        label="LR-QAOA median runtime",
    )
    if np.isfinite(lm_slope):
        ax.axhline(lm_slope, color=color_walksat, lw=1.2, ls="--", label="WalkSATlm")
    written.append(
        finish(
            fig,
            ax,
            "multi_seed_split_median_runtime_vs_walksat",
            list(typ_mean) + [lm_slope],
        )
    )

    fig, ax = plt.subplots(figsize=(5.2, 3.05))
    ax.errorbar(
        depths,
        typ_mean,
        yerr=typ_sem,
        fmt="o-",
        color=color_runtime,
        mfc="white",
        mec=color_runtime,
        mew=0.9,
        lw=1.45,
        ms=4.8,
        elinewidth=0.75,
        capsize=2.0,
        capthick=0.75,
        label="LR-QAOA median runtime",
    )
    ax.errorbar(
        depths,
        inv_mean,
        yerr=inv_sem,
        fmt="s-",
        color=color_mean,
        mfc=color_mean,
        mec=color_mean,
        mew=0.7,
        lw=1.45,
        ms=4.4,
        elinewidth=0.75,
        capsize=2.0,
        capthick=0.75,
        label="LR-QAOA mean success probability",
    )
    written.append(
        finish(
            fig,
            ax,
            "multi_seed_split_median_runtime_vs_lr_qaoa_mean_success",
            list(typ_mean) + list(inv_mean),
        )
    )

    fig, ax = plt.subplots(figsize=(5.2, 3.05))
    ax.errorbar(
        depths,
        inv_mean,
        yerr=inv_sem,
        fmt="s-",
        color=color_mean,
        mfc=color_mean,
        mec=color_mean,
        mew=0.7,
        lw=1.45,
        ms=4.4,
        elinewidth=0.75,
        capsize=2.0,
        capthick=0.75,
        label="LR-QAOA mean success probability",
    )
    ax.plot(
        p_k8,
        y_k8,
        color=color_analytic,
        lw=1.45,
        ls="--",
        label="QAOA (BM24 analytic)",
    )
    written.append(
        finish(
            fig,
            ax,
            "multi_seed_split_mean_success_vs_qaoa_bm24_analytic",
            list(inv_mean) + y_k8.tolist(),
        )
    )

    return written


def plot_seed_overlay(
    all_rows: List[dict],
    out_dir: Path,
    *,
    focus_depths: Optional[List[int]] = None,
) -> Path:
    """Single-panel overlay: all seeds × both training objectives on one axis."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, depths = _filter_rows(all_rows, focus_depths)
    seeds = sorted({int(r["seed"]) for r in rows})

    cmap = plt.cm.tab10
    seed_colors = {seed: cmap(i % 10) for i, seed in enumerate(seeds)}
    mean_dashes = (8, 4)

    fig, ax = plt.subplots(figsize=(8.5, 5.2))

    mean_by_depth: Dict[int, List[float]] = {d: [] for d in depths}
    all_vals: List[float] = []

    for seed in seeds:
        sub = [r for r in rows if int(r["seed"]) == seed]
        mean_map = {int(r["depth"]): r["c_typ_mean_p"] for r in sub}
        med_map = {int(r["depth"]): r["c_typ_median_rt"] for r in sub}
        color = seed_colors[seed]
        mean_ys = [mean_map.get(d, float("nan")) for d in depths]
        med_ys = [med_map.get(d, float("nan")) for d in depths]
        for d, ym, ymed in zip(depths, mean_ys, med_ys):
            if not np.isnan(ym):
                mean_by_depth[d].append(ym)
                all_vals.append(ym)
            if not np.isnan(ymed):
                all_vals.append(ymed)
        ax.plot(
            depths,
            mean_ys,
            linestyle="--",
            dashes=mean_dashes,
            color=color,
            lw=2.0,
            ms=8,
            marker="o",
            markerfacecolor="white",
            markeredgecolor=color,
            markeredgewidth=1.8,
            alpha=0.9,
            zorder=2,
        )
        ax.plot(
            depths,
            med_ys,
            linestyle="-",
            color=color,
            lw=2.4,
            ms=8,
            marker="o",
            markerfacecolor=color,
            markeredgecolor=color,
            markeredgewidth=1.2,
            alpha=0.95,
            zorder=3,
        )

    mean_lo = [min(mean_by_depth[d]) if mean_by_depth[d] else float("nan") for d in depths]
    mean_hi = [max(mean_by_depth[d]) if mean_by_depth[d] else float("nan") for d in depths]
    ax.fill_between(depths, mean_lo, mean_hi, color="#888888", alpha=0.15, zorder=0)

    y_min = float(np.nanmin(all_vals))
    y_max = float(np.nanmax(all_vals))
    y_pad_lo = 0.02
    y_pad_hi = 0.05
    ax.set_ylim(y_min - y_pad_lo, y_max + y_pad_hi)

    ax.set_xlabel("depth p", fontsize=11)
    ax.set_ylabel(r"$c_{\mathrm{typ}}$ (eval log₂ slope)", fontsize=11)
    ax.set_title(f"Multi-seed overlay ({len(seeds)} seeds)", fontsize=11)
    ax.set_xticks(depths)
    ax.set_xticklabels([str(d) for d in depths])
    ax.grid(True, alpha=0.3)

    style_handles = [
        Line2D(
            [0], [0],
            color="#333333",
            linestyle="--",
            dashes=mean_dashes,
            lw=2.5,
            marker="o",
            markerfacecolor="white",
            markeredgecolor="#333333",
            markeredgewidth=1.8,
            ms=8,
            label="mean_p training (dashed, open markers)",
        ),
        Line2D(
            [0], [0],
            color="#333333",
            linestyle="-",
            lw=2.5,
            marker="o",
            markerfacecolor="#333333",
            markeredgecolor="#333333",
            ms=8,
            label="median_rt training (solid, filled markers)",
        ),
    ]
    seed_handles = [
        Line2D(
            [0], [0],
            color=seed_colors[seed],
            linestyle="-",
            lw=2.4,
            marker="o",
            markerfacecolor=seed_colors[seed],
            ms=8,
            label=f"seed {seed}",
        )
        for seed in seeds
    ]
    leg_style = ax.legend(
        handles=style_handles,
        fontsize=9,
        loc="upper right",
        framealpha=0.95,
        title="Line style = training objective",
        title_fontsize=9,
    )
    ax.add_artist(leg_style)
    ax.legend(
        handles=seed_handles,
        fontsize=9,
        loc="upper center",
        framealpha=0.95,
        title="Colour = random seed",
        title_fontsize=9,
        ncol=len(seeds),
    )
    fig.tight_layout()

    png = out_dir / "multi_seed_ctyp_overlay.png"
    fig.savefig(png, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {png}")
    return png


def plot_summary(
    all_rows: List[dict],
    out_dir: Path,
    *,
    focus_depths: Optional[List[int]] = None,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, depths = _filter_rows(all_rows, focus_depths)
    seeds = sorted({int(r["seed"]) for r in rows})

    cmap = plt.cm.tab10
    seed_colors = {seed: cmap(i % 10) for i, seed in enumerate(seeds)}

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # Left: c_typ eval slopes — both training objectives
    ax = axes[0]
    for seed in seeds:
        sub = [r for r in rows if int(r["seed"]) == seed]
        mean_map = {int(r["depth"]): r["c_typ_mean_p"] for r in sub}
        med_map = {int(r["depth"]): r["c_typ_median_rt"] for r in sub}
        color = seed_colors[seed]
        xs = depths
        ax.plot(
            xs,
            [mean_map.get(d, float("nan")) for d in xs],
            "o--",
            color=color,
            alpha=0.55,
            ms=5,
            label=f"seed={seed} mean_p",
        )
        ax.plot(
            xs,
            [med_map.get(d, float("nan")) for d in xs],
            "o-",
            color=color,
            alpha=0.95,
            ms=5,
            label=f"seed={seed} median_rt",
        )
    ax.set_xlabel("depth p")
    ax.set_ylabel("c_typ (eval log₂ slope)")
    ax.set_title("Eval scaling by training objective")
    ax.set_xticks(depths)
    ax.set_xticklabels([str(d) for d in depths])
    ax.legend(fontsize=6.5, ncol=2, loc="upper right")
    ax.grid(True, alpha=0.3)

    # Middle: Δ c_typ per seed
    ax = axes[1]
    for seed in seeds:
        sub = [r for r in rows if int(r["seed"]) == seed]
        dmap = {int(r["depth"]): r["delta_ctyp"] for r in sub}
        ys = [dmap.get(d, float("nan")) for d in depths]
        ax.plot(depths, ys, "o-", label=f"seed={seed}", color=seed_colors[seed], alpha=0.9)
    ax.axhline(0, color="k", lw=0.8, alpha=0.35)
    ax.set_xlabel("depth p")
    ax.set_ylabel("Δ c_typ (median_rt − mean_p)")
    ax.set_title("Training objective effect")
    ax.set_xticks(depths)
    ax.set_xticklabels([str(d) for d in depths])
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Right: mean |Δ| per depth with per-seed scatter
    ax = axes[2]
    mean_abs = []
    std_delta = []
    n_at_depth = []
    for d in depths:
        deltas = [r["delta_ctyp"] for r in rows if int(r["depth"]) == d]
        mean_abs.append(float(np.mean(np.abs(deltas))))
        std_delta.append(float(np.std(deltas)) if len(deltas) > 1 else 0.0)
        n_at_depth.append(len(deltas))
        jitter = np.linspace(-0.12, 0.12, max(len(deltas), 1))
        for j, delta in enumerate(deltas):
            ax.scatter(d + jitter[j], abs(delta), s=28, color="#555555", alpha=0.65, zorder=3)
    x = np.arange(len(depths))
    bars = ax.bar(x, mean_abs, yerr=std_delta, capsize=4, color="#4C72B0", alpha=0.75, zorder=2)
    for i, (bar, n) in enumerate(zip(bars, n_at_depth)):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + std_delta[i] + 0.001,
            f"n={n}",
            ha="center",
            va="bottom",
            fontsize=7,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([str(d) for d in depths])
    ax.set_xlabel("depth p")
    ax.set_ylabel("mean |Δ c_typ| across seeds")
    ax.set_title(f"{len(seeds)} seeds")
    ax.grid(True, alpha=0.3, axis="y")

    depth_label = ",".join(str(d) for d in depths)
    fig.suptitle(
        f"Multi-seed robustness (p ∈ {{{depth_label}}}): median vs mean training",
        fontsize=11,
    )
    fig.tight_layout()
    png = out_dir / "multi_seed_delta_ctyp.png"
    fig.savefig(png, dpi=150, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "n_seeds": len(seeds),
        "seeds": seeds,
        "depths": depths,
        "rows": rows,
        "per_depth": {
            str(d): {
                "n_seeds": len([r for r in rows if int(r["depth"]) == d]),
                "mean_delta": float(np.mean([r["delta_ctyp"] for r in rows if int(r["depth"]) == d])),
                "mean_abs_delta": float(np.mean(np.abs([r["delta_ctyp"] for r in rows if int(r["depth"]) == d]))),
                "std_delta": float(np.std([r["delta_ctyp"] for r in rows if int(r["depth"]) == d])),
            }
            for d in depths
        },
    }
    json_path = out_dir / "multi_seed_summary.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {png}")
    print(f"Wrote {json_path}")
    return png


def run_ctyp_cann_rebenchmark(json_path: Path, *, focus_depths: str) -> None:
    script = _LR / "plot_ctyp_vs_cann.py"
    out_name = json_path.stem
    for root in (
        _PHASECRAFT / "bm24_runs/analysis/ctyp_vs_cann",
        _PHASECRAFT / "results/bm24_runs/analysis/ctyp_vs_cann",
    ):
        out_dir = root / out_name
        cmd = [
            sys.executable,
            str(script),
            str(json_path),
            "--skip-theory",
            "--depths",
            focus_depths,
            "--output-dir",
            str(out_dir),
        ]
        print(f"  ctyp_vs_cann rebenchmark: {json_path.name} -> {out_dir}", flush=True)
        subprocess.run(cmd, cwd=str(_PHASECRAFT), check=False)


def run_ctyp_cann(json_path: Path) -> None:
    script = _LR / "plot_ctyp_vs_cann.py"
    out_name = json_path.stem
    out_dir = _PHASECRAFT / "bm24_runs/analysis/ctyp_vs_cann" / out_name
    cmd = [
        sys.executable, str(script), str(json_path),
        "--theory-max-depth", "10",
        "--no-rebenchmark",
        "--output-dir", str(out_dir),
    ]
    print(f"  ctyp_vs_cann: {json_path.name}", flush=True)
    subprocess.run(cmd, cwd=str(_PHASECRAFT), check=False)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, default=_DEFAULT_MANIFEST)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=_PHASECRAFT / "bm24_runs/analysis/multi_seed_ctyp_cann",
    )
    p.add_argument("--include-baseline-seed27", action="store_true", default=True)
    p.add_argument(
        "--depths",
        default="2,10,20,40,50,60,80,100",
        help="Comma-separated depths to plot (default: 2,10,20,40,50,60,80,100).",
    )
    p.add_argument(
        "--eval-window",
        default="12,18",
        help="n range for exponent fits and WalkSAT baselines (default: 12,18). "
        "Use 14,18 for paper fair window.",
    )
    p.add_argument(
        "--with-ctyp-cann",
        action="store_true",
        help="Also run plot_ctyp_vs_cann per JSON (theory p≤10, no rebenchmark).",
    )
    p.add_argument(
        "--with-mean-p-rebenchmark",
        action="store_true",
        help="Run plot_ctyp_vs_cann with rebenchmark (slow) to populate mean 1/p exponents.",
    )
    args = p.parse_args()

    runs = load_runs(args.manifest.resolve(), include_baseline=bool(args.include_baseline_seed27))
    if not runs:
        print(f"No runs found (manifest: {args.manifest})", file=sys.stderr)
        sys.exit(1)

    all_rows: List[dict] = []
    for run in runs:
        all_rows.extend(extract_deltas(run))
        if args.with_ctyp_cann:
            run_ctyp_cann(run["path"])
        if args.with_mean_p_rebenchmark:
            run_ctyp_cann_rebenchmark(run["path"], focus_depths=args.depths)

    focus_depths = [int(x) for x in args.depths.split(",") if x.strip()]
    out_dir = args.output_dir.resolve()
    eval_window = _parse_window(args.eval_window)
    train_n = _train_n_from_runs(runs)
    classical = load_classical_baselines(None)
    equiv_rows = extract_exponent_equivalence_rows(runs, focus_depths=focus_depths)
    plot_exponent_equivalence(equiv_rows, out_dir, focus_depths=focus_depths)
    ctyp_rows = extract_ctyp_by_seed(runs, eval_window=eval_window)
    ctyp_rows_median_rt = extract_ctyp_by_seed(
        runs, mode="median_runtime_fixed_n", eval_window=eval_window
    )
    mean_rows = extract_mean_p_exponent_rows(
        runs, eval_window=eval_window, focus_depths=focus_depths
    )
    plot_seed_convergence(
        ctyp_rows,
        out_dir,
        ctyp_rows_median_rt=ctyp_rows_median_rt,
        focus_depths=focus_depths,
        train_n=train_n,
        classical=classical,
        eval_window=eval_window,
    )
    plot_dual_exponent_convergence(
        ctyp_rows,
        mean_rows,
        out_dir,
        focus_depths=focus_depths,
        train_n=train_n,
        eval_window=eval_window,
        classical=classical,
    )
    plot_dual_exponent_convergence(
        ctyp_rows,
        mean_rows,
        out_dir,
        focus_depths=focus_depths,
        train_n=train_n,
        eval_window=eval_window,
        classical=classical,
        paper_style=True,
    )
    plot_seed_overlay(all_rows, out_dir, focus_depths=focus_depths)
    plot_summary(all_rows, out_dir, focus_depths=focus_depths)
    mirror = _PHASECRAFT / "results/bm24_runs/analysis/multi_seed_ctyp_cann"
    if mirror.resolve() != out_dir.resolve():
        plot_exponent_equivalence(equiv_rows, mirror, focus_depths=focus_depths)
        plot_seed_convergence(
            ctyp_rows,
            mirror,
            ctyp_rows_median_rt=ctyp_rows_median_rt,
            focus_depths=focus_depths,
            train_n=train_n,
            classical=classical,
            eval_window=eval_window,
        )
        plot_dual_exponent_convergence(
            ctyp_rows,
            mean_rows,
            mirror,
            focus_depths=focus_depths,
            train_n=train_n,
            eval_window=eval_window,
            classical=classical,
        )
        plot_dual_exponent_convergence(
            ctyp_rows,
            mean_rows,
            mirror,
            focus_depths=focus_depths,
            train_n=train_n,
            eval_window=eval_window,
            classical=classical,
            paper_style=True,
        )
        plot_seed_overlay(all_rows, mirror, focus_depths=focus_depths)
        plot_summary(all_rows, mirror, focus_depths=focus_depths)

    _, depths = _filter_rows(all_rows, focus_depths)
    print("\nPer-depth seed spread (max−min, mean_p training):")
    by_seed = extract_ctyp_by_seed(runs, eval_window=eval_window)
    for d in depths:
        vals = [r["c_typ"] for r in by_seed if int(r["depth"]) == d]
        if vals:
            print(f"  p={d:2d}: spread={max(vals)-min(vals):.4f}  seeds={len(vals)}")
    print("\nPer-depth |Δ c_typ| (mean_p vs median_rt, different training):")
    for d in depths:
        deltas = [abs(r["delta_ctyp"]) for r in all_rows if int(r["depth"]) == d]
        print(f"  p={d:2d}: {np.mean(deltas):.4f}  (seeds={len(deltas)})")


if __name__ == "__main__":
    main()
