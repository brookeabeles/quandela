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

_PHASECRAFT = Path(__file__).resolve().parents[2]
_LR = Path(__file__).resolve().parent
for _p in (_PHASECRAFT.parent, _PHASECRAFT, _LR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
from phasecraft.lib.paths import bm24_runs_dir  # noqa: E402
from fit_ctyp_convergence import _MODELS_DEF, _fit_exp_only  # noqa: E402
from plot_compare_run1_run4 import (  # noqa: E402
    _draw_baseline_refs,
    classical_slopes,
    load_classical_baselines,
)

_DEFAULT_MANIFEST = bm24_runs_dir() / "multi_seed_ctyp_cann" / "manifest.json"
_BASELINE = bm24_runs_dir() / "06-09/run1/train12-tr100-te200-n12-18.json"
FAIR_WINDOW: Tuple[int, int] = (14, 18)


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
) -> Path:
    """c_typ (median runtime) and mean(1/p) exponent on one panel."""
    out_dir.mkdir(parents=True, exist_ok=True)
    depths = focus_depths or sorted(
        {int(r["depth"]) for r in ctyp_rows} | {int(r["depth"]) for r in mean_rows}
    )
    seeds = sorted({int(r["seed"]) for r in ctyp_rows})
    n_lo, n_hi = eval_window

    typ_mean, typ_lo, typ_hi, _ = _aggregate_seed_series(ctyp_rows, "c_typ", depths, seeds)
    mean_seeds = sorted({int(r["seed"]) for r in mean_rows if r.get("has_mean_rebenchmark")})
    inv_mean, inv_lo, inv_hi, n_mean_seeds = _aggregate_seed_series(
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
    depths_arr = np.asarray(depths, dtype=float)
    x_eq = np.arange(len(depths), dtype=float)

    def _p_to_x(ps: np.ndarray) -> np.ndarray:
        return np.interp(ps, depths_arr, x_eq)

    all_vals = [
        v
        for v in typ_mean + typ_lo + typ_hi + inv_mean + inv_lo + inv_hi
        if np.isfinite(v)
    ]
    for v in (ws_slope, lm_slope):
        if np.isfinite(v):
            all_vals.append(v)

    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    yerr_typ_lo = [m - lo for m, lo in zip(typ_mean, typ_lo)]
    yerr_typ_hi = [hi - m for m, hi in zip(typ_mean, typ_hi)]
    ax.errorbar(
        x_eq,
        typ_mean,
        yerr=[yerr_typ_lo, yerr_typ_hi],
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
        label=rf"$c_{{\mathrm{{typ}}}}$ median $1/p$ ({len(seeds)} seeds)",
    )
    ps_typ = np.asarray(depths, dtype=float)
    ys_typ = np.asarray(typ_mean, dtype=float)
    mask_typ = np.isfinite(ys_typ)
    if mask_typ.sum() >= 3:
        ps_f, ys_f = ps_typ[mask_typ], ys_typ[mask_typ]
        exp_fit = _fit_exp_only(ps_f, ys_f)
        if exp_fit is not None:
            fn = _MODELS_DEF["exp"][0]
            p_line = np.linspace(float(np.min(ps_f)), p_max * 1.2, 400)
            x_line = _p_to_x(p_line)
            y_fit = fn(p_line, *exp_fit["params"])
            c_inf, c_err = exp_fit["c_inf"], exp_fit["c_inf_err"]
            beta = float(exp_fit["params"][2])
            beta_err = float(exp_fit["param_err"][2])
            ax.plot(
                x_line,
                y_fit,
                color="#4C72B0",
                lw=1.6,
                ls="--",
                alpha=0.85,
                zorder=2,
                label=(
                    rf"$2^{{-cn}}$ fit: $c_\infty={c_inf:.3f}\!\pm\!{c_err:.3f}$, "
                    rf"$\beta={beta:.3f}\!\pm\!{beta_err:.3f}$"
                ),
            )
            all_vals.extend(y_fit.tolist())
    if has_mean_line:
        yerr_inv_lo = [m - lo for m, lo in zip(inv_mean, inv_lo)]
        yerr_inv_hi = [hi - m for m, hi in zip(inv_mean, inv_hi)]
        mean_label = (
            rf"$c_{{1/\langle p\rangle}}$ mean $1/p$ ({n_mean_seeds} seed"
            + ("s" if n_mean_seeds != 1 else "")
            + ", rebenchmark)"
        )
        ax.errorbar(
            x_eq,
            inv_mean,
            yerr=[yerr_inv_lo, yerr_inv_hi],
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
            label=mean_label,
        )
    _draw_baseline_refs(ax, ws_slope, lm_slope, label_right=True)

    y_min = float(np.nanmin(all_vals))
    y_max = float(np.nanmax(all_vals))
    span = max(y_max - y_min, 0.05)
    pad = max(0.012, 0.08 * span)
    ax.set_ylim(y_min - pad, y_max + pad)
    ax.set_xlabel(r"QAOA depth $p$", fontsize=11)
    ax.set_ylabel(
        r"Scaling exponent $c$ ($p_{\mathrm{succ}} \propto 2^{-cn}$)",
        fontsize=11,
    )
    ax.set_title(
        rf"Multi-seed exponent convergence (train $n={train_n}$, "
        rf"mean$_p$ training, $n \in [{n_lo},{n_hi}]$)",
        fontsize=11,
    )
    ax.set_xticks(x_eq)
    ax.set_xticklabels([str(int(d)) for d in depths])
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8.5, loc="upper right", framealpha=0.95)
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
        default="10,20,40,50,60,80,100",
        help="Comma-separated depths to plot (default: 10,20,40,50,60,80,100).",
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
