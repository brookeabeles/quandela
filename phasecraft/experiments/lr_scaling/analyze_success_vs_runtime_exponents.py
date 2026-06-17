#!/usr/bin/env python3
"""
Analyze divergence between mean-success and median-runtime scaling exponents.

Reads all ``bm24_runs/**/*.json`` (and ``results/bm24_runs``), produces a
multi-panel report under ``--output-dir``.

Example (plots only, fast)::

    python experiments/lr_scaling/analyze_success_vs_runtime_exponents.py

Add held-out re-benchmarks at saved angles (slow, fills per-instance spread)::

    python experiments/lr_scaling/analyze_success_vs_runtime_exponents.py \\
        --rebenchmark results/bm24_runs/06-09/run1/06-09_1649-train12-tr100-te200-n12-18.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress

_PHASECRAFT = Path(__file__).resolve().parents[2]
_REPO = _PHASECRAFT.parent
_LR = Path(__file__).resolve().parent
for _p in (_REPO, _PHASECRAFT, _LR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from phasecraft.lib.paths import bm24_runs_dir  # noqa: E402
from phasecraft.lib.sim.bm24_qaoa_sim import (  # noqa: E402
    evaluate_qaoa_runtime_on_dataset,
    generate_benchmark_dataset,
    make_lr_angles,
)

LN2 = float(np.log(2.0))


def fit_log2_slope(ns: Iterable[int], ys: Iterable[float], *, positive_only: bool = True) -> float:
    n_arr = np.asarray(list(ns), dtype=float)
    y_arr = np.asarray(list(ys), dtype=float)
    mask = np.isfinite(y_arr)
    if positive_only:
        mask &= y_arr > 0
    if int(mask.sum()) < 2:
        return float("nan")
    return float(linregress(n_arr[mask], np.log(y_arr[mask])).slope / LN2)


def _traces_from_payload(payload: dict) -> Dict[str, List[dict]]:
    if "traces_by_mode" in payload:
        return {k: list(v) for k, v in payload["traces_by_mode"].items()}
    out: Dict[str, List[dict]] = {}
    for key in ("trace_bm24_mean_p_fixed_n", "trace_median_runtime_fixed_n", "trace_mean_log_runtime_fixed_n", "trace"):
        if key in payload and payload[key]:
            mode = key.replace("trace_", "").replace("trace", "legacy_single")
            out[mode] = list(payload[key])
    return out


def _iter_json_runs(roots: List[Path]) -> List[Path]:
    seen: set[Path] = set()
    paths: List[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for p in sorted(root.rglob("*.json")):
            if ".partial." in p.name:
                continue
            rp = p.resolve()
            if rp not in seen:
                seen.add(rp)
                paths.append(rp)
    return paths


def collect_training_mode_pairs(json_paths: List[Path]) -> List[dict]:
    """Pair mean-p vs median-rt eval slopes at the same depth within one run."""
    rows: List[dict] = []
    for jp in json_paths:
        try:
            payload = json.loads(jp.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(payload, dict):
            continue
        traces = _traces_from_payload(payload)
        mean_tr = traces.get("bm24_mean_p_fixed_n")
        med_tr = traces.get("median_runtime_fixed_n")
        if not mean_tr or not med_tr:
            continue
        med_by_depth = {int(r["depth"]): r for r in med_tr}
        cfg = payload.get("config", {})
        for mr in mean_tr:
            d = int(mr["depth"])
            if d not in med_by_depth:
                continue
            mm = med_by_depth[d]
            sm = float(mr.get("lr_log2_slope", float("nan")))
            sd = float(mm.get("lr_log2_slope", float("nan")))
            rows.append({
                "run": jp.stem,
                "path": str(jp),
                "train_n": cfg.get("train_n"),
                "depth": d,
                "mean_p_slope": sm,
                "median_rt_slope": sd,
                "delta_median_minus_mean": sd - sm,
                "mean_dgamma": mr.get("delta_gamma"),
                "mean_dbeta": mr.get("delta_beta"),
                "med_dgamma": mm.get("delta_gamma"),
                "med_dbeta": mm.get("delta_beta"),
            })
    return rows


def collect_bench_per_n(json_paths: List[Path]) -> List[dict]:
    rows: List[dict] = []
    for jp in json_paths:
        try:
            payload = json.loads(jp.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(payload, dict):
            continue
        per_n = payload.get("results", {}).get("lr_qaoa", {}).get("per_n")
        if not per_n:
            continue
        ns = sorted(int(k) for k in per_n)
        mean_p = [float(per_n[str(n)]["mean_success"]) for n in ns]
        med_p = [float(per_n[str(n)]["median_success"]) for n in ns]
        med_rt = [float(per_n[str(n)]["median_runtime"]) for n in ns]
        inv_mean = [float(per_n[str(n)]["inverse_mean_success"]) for n in ns]
        rows.append({
            "source": jp.stem,
            "path": str(jp),
            "ns": ns,
            "mean_p": mean_p,
            "median_p": med_p,
            "median_rt": med_rt,
            "inv_mean": inv_mean,
            "slope_mean_p": fit_log2_slope(ns, mean_p),
            "slope_median_p": fit_log2_slope(ns, med_p),
            "slope_median_rt": fit_log2_slope(ns, med_rt),
            "slope_inv_mean": fit_log2_slope(ns, inv_mean),
            "gap_rt_vs_neg_mean": fit_log2_slope(ns, med_rt) + fit_log2_slope(ns, mean_p),
            "settings": payload.get("settings", payload.get("config", {})),
        })
    # Prefer widest n window for primary plots
    rows.sort(key=lambda r: len(r["ns"]), reverse=True)
    return rows


def parse_angle_log_exponents(log_paths: List[Path]) -> List[dict]:
    """Parse lr_train_optimal_angles*.txt for mean-p exponent fits."""
    rows: List[dict] = []
    pat = re.compile(
        r"slope_ln_mean_p_succ_vs_n_log2[\"']?\s*[:=]\s*([-\d.eE+]+)"
    )
    depth_pat = re.compile(r"depth\s*=\s*(\d+)", re.I)
    for lp in log_paths:
        try:
            text = lp.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        blocks = re.split(r"(?=depth\s*=)", text)
        for block in blocks:
            dm = depth_pat.search(block)
            em = pat.search(block)
            if dm and em:
                rows.append({
                    "source": lp.name,
                    "depth": int(dm.group(1)),
                    "mean_p_log2_slope": float(em.group(1)),
                })
    return rows


def rebenchmark_angles(
    payload: dict,
    *,
    depths: Optional[List[int]] = None,
    modes: Optional[List[str]] = None,
) -> List[dict]:
    cfg = payload["config"]
    traces = _traces_from_payload(payload)
    modes = modes or ["bm24_mean_p_fixed_n", "median_runtime_fixed_n"]
    mode_to_trace = {
        "bm24_mean_p_fixed_n": traces.get("bm24_mean_p_fixed_n"),
        "median_runtime_fixed_n": traces.get("median_runtime_fixed_n"),
    }
    n_min, n_max = int(cfg["n_min"]), int(cfg["n_max"])
    dataset = generate_benchmark_dataset(
        list(range(n_min, n_max + 1)),
        int(cfg["k"]),
        float(cfg["r"]),
        int(cfg["test_size"]),
        int(cfg["seed"]),
        require_sat=True,
    )
    rows: List[dict] = []
    for mode in modes:
        tr = mode_to_trace.get(mode)
        if not tr:
            continue
        for row in tr:
            d = int(row["depth"])
            if depths is not None and d not in depths:
                continue
            dg, db = float(row["delta_gamma"]), float(row["delta_beta"])
            betas, gammas = make_lr_angles(
                depth=d,
                delta_gamma=dg,
                delta_beta=db,
                beta_schedule=cfg.get("lr_beta_schedule", "decreasing"),
                angle_convention="bm24",
            )
            ev = evaluate_qaoa_runtime_on_dataset(dataset, betas, gammas)
            per_n = ev["per_n"]
            ns = sorted(int(n) for n in per_n)
            mean_p = [per_n[n]["mean_success"] for n in ns]
            med_p = [per_n[n]["median_success"] for n in ns]
            med_rt = [per_n[n]["median_runtime"] for n in ns]
            inv_mean = [per_n[n]["inverse_mean_success"] for n in ns]
            # instance spread: median_p / mean_p  (tight ~ 1), IQR proxy via ratio
            spread_ratio = [mp / ms if ms > 0 else float("nan") for mp, ms in zip(med_p, mean_p)]
            rt_over_inv_mean = [rt / im if im > 0 else float("nan") for rt, im in zip(med_rt, inv_mean)]
            rows.append({
                "mode": mode,
                "depth": d,
                "delta_gamma": dg,
                "delta_beta": db,
                "ns": ns,
                "mean_p": mean_p,
                "median_p": med_p,
                "median_rt": med_rt,
                "spread_ratio_med_over_mean": spread_ratio,
                "rt_over_inv_mean": rt_over_inv_mean,
                "slope_mean_p": fit_log2_slope(ns, mean_p),
                "slope_median_rt": fit_log2_slope(ns, med_rt),
                "gap_rt_plus_mean_p": fit_log2_slope(ns, med_rt) + fit_log2_slope(ns, mean_p),
                "stored_lr_log2_slope": float(row.get("lr_log2_slope", float("nan"))),
            })
    return rows


def _save(fig: plt.Figure, out_dir: Path, name: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_training_mode_pairs(pairs: List[dict], out_dir: Path) -> List[Path]:
    saved: List[Path] = []
    if not pairs:
        return saved

    by_run: Dict[str, List[dict]] = {}
    for r in pairs:
        by_run.setdefault(r["run"], []).append(r)

    # 1) Per-run depth curves
    for run, rows in by_run.items():
        rows = sorted(rows, key=lambda x: x["depth"])
        depths = [r["depth"] for r in rows]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(depths, [r["mean_p_slope"] for r in rows], "o-", label="eval slope (trained on mean p)")
        ax.plot(depths, [r["median_rt_slope"] for r in rows], "s--", label="eval slope (trained on median 1/p)")
        ax.set_xlabel("QAOA depth p")
        ax.set_ylabel("eval log₂ slope of median(1/p_succ) vs n")
        ax.set_title(f"Training objective changes eval scaling\n{run}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        saved.append(_save(fig, out_dir, f"01_train_mode_curves_{run[:40]}.png"))

    # 2) Delta vs depth (all runs)
    fig, ax = plt.subplots(figsize=(9, 5))
    for run, rows in by_run.items():
        rows = sorted(rows, key=lambda x: x["depth"])
        ax.plot(
            [r["depth"] for r in rows],
            [r["delta_median_minus_mean"] for r in rows],
            "o-",
            alpha=0.8,
            label=f"{run} (train_n={rows[0].get('train_n')})",
        )
    ax.axhline(0, color="k", lw=0.8, alpha=0.5)
    ax.set_xlabel("depth p")
    ax.set_ylabel("Δ eval slope = median-rt-train − mean-p-train")
    ax.set_title("Does optimizing median(1/p) change eval scaling exponent?")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    saved.append(_save(fig, out_dir, "02_delta_eval_slope_by_depth.png"))

    # 3) Scatter paired slopes
    fig, ax = plt.subplots(figsize=(6, 6))
    xs = [r["mean_p_slope"] for r in pairs]
    ys = [r["median_rt_slope"] for r in pairs]
    ax.scatter(xs, ys, c=[r["depth"] for r in pairs], cmap="viridis", s=60, alpha=0.85)
    lo = min(xs + ys)
    hi = max(xs + ys)
    ax.plot([lo, hi], [lo, hi], "k--", alpha=0.4, label="y = x")
    for r in pairs:
        if r["depth"] in (15, 40):
            ax.annotate(
                f"p={r['depth']}\nΔ={r['delta_median_minus_mean']:+.3f}",
                (r["mean_p_slope"], r["median_rt_slope"]),
                textcoords="offset points",
                xytext=(6, 6),
                fontsize=8,
            )
    ax.set_xlabel("eval slope (mean-p training)")
    ax.set_ylabel("eval slope (median-rt training)")
    ax.set_title("Paired eval slopes at same depth")
    ax.grid(True, alpha=0.3)
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap="viridis"), ax=ax)
    cb.set_label("depth p")
    saved.append(_save(fig, out_dir, "03_paired_eval_slopes_scatter.png"))

    return saved


def plot_bench_divergence(bench_rows: List[dict], out_dir: Path) -> List[Path]:
    saved: List[Path] = []
    if not bench_rows:
        return saved

    br = bench_rows[0]
    ns, mean_p, med_p, med_rt, inv_mean = (
        br["ns"], br["mean_p"], br["median_p"], br["median_rt"], br["inv_mean"]
    )

    # 4) Per-n curves (log scale)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax = axes[0]
    ax.semilogy(ns, mean_p, "o-", label="mean p_succ")
    ax.semilogy(ns, med_p, "s-", label="median p_succ")
    ax.set_xlabel("n")
    ax.set_ylabel("success probability")
    ax.set_title("Success probability vs n\n(tight spread ⟺ curves close)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.semilogy(ns, med_rt, "o-", label="median(1/p)")
    ax.semilogy(ns, inv_mean, "s--", label="1/mean(p)")
    ax.semilogy(ns, [1.0 / m if m > 0 else np.nan for m in med_p], "^:", label="1/median(p)")
    ax.set_xlabel("n")
    ax.set_ylabel("runtime proxy")
    ax.set_title("Runtime proxies vs n")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.suptitle(f"Benchmark: {br['source']}", y=1.02)
    saved.append(_save(fig, out_dir, "04_per_n_success_and_runtime.png"))

    # 5) Spread metrics vs n
    spread = [mp / ms if ms > 0 else np.nan for mp, ms in zip(med_p, mean_p)]
    rt_gap = [rt / im if im > 0 else np.nan for rt, im in zip(med_rt, inv_mean)]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(ns, spread, "o-")
    axes[0].axhline(1.0, color="k", ls="--", alpha=0.4)
    axes[0].set_xlabel("n")
    axes[0].set_ylabel("median(p) / mean(p)")
    axes[0].set_title("Instance spread proxy\n(→1 means tight; lower = heavy tail of hard instances)")

    axes[1].plot(ns, rt_gap, "o-")
    axes[1].axhline(1.0, color="k", ls="--", alpha=0.4)
    axes[1].set_xlabel("n")
    axes[1].set_ylabel("median(1/p) / (1/mean(p))")
    axes[1].set_title("Runtime gap from using mean vs median\n(→1 means exponents would agree)")
    saved.append(_save(fig, out_dir, "05_spread_and_runtime_gap_vs_n.png"))

    # 6) Exponent bar comparison
    labels = ["ln(mean p)", "ln(median p)", "ln(1/mean p)", "ln(median 1/p)"]
    slopes = [br["slope_mean_p"], br["slope_median_p"], br["slope_inv_mean"], br["slope_median_rt"]]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"]
    ax.bar(labels, slopes, color=colors)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("log₂ slope vs n")
    ax.set_title(
        f"Different exponents from same benchmark\n"
        f"gap median_rt − |mean_p| = {br['gap_rt_vs_neg_mean']:.4f} log₂"
    )
    saved.append(_save(fig, out_dir, "06_exponent_bar_same_benchmark.png"))

    return saved


def plot_all_bench_exponents(bench_rows: List[dict], out_dir: Path) -> List[Path]:
    saved: List[Path] = []
    if len(bench_rows) < 1:
        return saved
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(bench_rows))
    w = 0.2
    for i, (key, lab, col) in enumerate([
        ("slope_mean_p", "|mean p| (negated)", "#1f77b4"),
        ("slope_median_rt", "median 1/p", "#d62728"),
        ("gap_rt_vs_neg_mean", "gap", "#9467bd"),
    ]):
        vals = [(-b[key] if key == "slope_mean_p" else b[key]) for b in bench_rows]
        ax.bar(x + (i - 1) * w, vals, width=w, label=lab, color=col, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{b['source']}\nn={b['ns'][0]}..{b['ns'][-1]}" for b in bench_rows], fontsize=8)
    ax.set_ylabel("log₂ slope vs n")
    ax.set_title("Exponent gap across saved full benchmarks")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    saved.append(_save(fig, out_dir, "11_all_bench_exponents.png"))
    return saved


def plot_rebenchmark(rb_rows: List[dict], out_dir: Path) -> List[Path]:
    saved: List[Path] = []
    if not rb_rows:
        return saved

    # 7) |mean_p| vs median_rt exponent across depth × mode
    fig, ax = plt.subplots(figsize=(8, 6))
    for mode, marker in [("bm24_mean_p_fixed_n", "o"), ("median_runtime_fixed_n", "s")]:
        sub = [r for r in rb_rows if r["mode"] == mode]
        ax.scatter(
            [-r["slope_mean_p"] for r in sub],
            [r["slope_median_rt"] for r in sub],
            marker=marker,
            s=70,
            label=f"trained: {mode.split('_')[0]}…",
        )
        for r in sub:
            if r["depth"] in (15, 40):
                ax.annotate(
                    f"p={r['depth']}",
                    (-r["slope_mean_p"], r["slope_median_rt"]),
                    fontsize=8,
                    xytext=(5, 5),
                    textcoords="offset points",
                )
    lims = [
        min([-r["slope_mean_p"] for r in rb_rows] + [r["slope_median_rt"] for r in rb_rows]),
        max([-r["slope_mean_p"] for r in rb_rows] + [r["slope_median_rt"] for r in rb_rows]),
    ]
    ax.plot(lims, lims, "k--", alpha=0.35, label="perfect agreement")
    ax.set_xlabel("|log₂ slope of mean p_succ|")
    ax.set_ylabel("log₂ slope of median(1/p)")
    ax.set_title("Mean-success vs median-runtime exponent\n(re-benchmark at saved angles)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    saved.append(_save(fig, out_dir, "07_rebench_mean_vs_median_rt_exponent.png"))

    # 8) Gap vs depth, colored by mode
    fig, ax = plt.subplots(figsize=(8, 5))
    for mode, ls in [("bm24_mean_p_fixed_n", "-"), ("median_runtime_fixed_n", "--")]:
        sub = sorted([r for r in rb_rows if r["mode"] == mode], key=lambda x: x["depth"])
        ax.plot(
            [r["depth"] for r in sub],
            [r["gap_rt_plus_mean_p"] for r in sub],
            f"o{ls}",
            label=mode,
        )
    ax.axhline(0, color="k", lw=0.8, alpha=0.4)
    ax.set_xlabel("depth p")
    ax.set_ylabel("slope(median 1/p) + slope(mean p)  [log₂, →0 if agree]")
    ax.set_title("Exponent gap vs depth (asymptotic agreement → 0?)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    saved.append(_save(fig, out_dir, "08_gap_vs_depth_rebench.png"))

    # 9) Spread ratio vs n for selected depths
    pick = [15, 40]
    fig, ax = plt.subplots(figsize=(8, 5))
    for r in rb_rows:
        if r["depth"] not in pick:
            continue
        lab = f"p={r['depth']} {r['mode'][:6]}"
        ax.plot(r["ns"], r["spread_ratio_med_over_mean"], "o-", label=lab)
    ax.axhline(1.0, color="k", ls="--", alpha=0.4)
    ax.set_xlabel("n")
    ax.set_ylabel("median(p) / mean(p)")
    ax.set_title("When spread ratio drifts below 1, mean-p exponent underestimates runtime")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    saved.append(_save(fig, out_dir, "09_spread_ratio_selected_depths.png"))

    # 10) Depth asymptotic: eval slope vs depth for both modes
    fig, ax = plt.subplots(figsize=(8, 5))
    for mode, ls in [("bm24_mean_p_fixed_n", "-"), ("median_runtime_fixed_n", "--")]:
        sub = sorted([r for r in rb_rows if r["mode"] == mode], key=lambda x: x["depth"])
        ax.plot(
            [r["depth"] for r in sub],
            [r["slope_median_rt"] for r in sub],
            f"o{ls}",
            label=f"median-rt slope ({mode[:10]}…)",
        )
    ax.set_xlabel("depth p")
    ax.set_ylabel("log₂ slope median(1/p) vs n")
    ax.set_title("Eval scaling exponent vs circuit depth\n(do mean-p and median-rt training converge as p→∞?)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    saved.append(_save(fig, out_dir, "10_eval_slope_vs_depth_rebench.png"))

    return saved


def write_summary(
    out_dir: Path,
    pairs: List[dict],
    bench_rows: List[dict],
    rb_rows: List[dict],
) -> Path:
    lines = [
        "# Mean success vs median runtime exponent analysis",
        "",
        "## What “tight instance-to-instance spread” means",
        "",
        "At fixed n, each random k-SAT instance gives a success probability pᵢ.",
        "**Tight spread** = all pᵢ are close (low variance / IQR). Then mean(p) ≈ median(p)",
        "and median(1/p) ≈ 1/mean(p), so success and runtime exponents agree.",
        "",
        "**Loose spread** = a tail of hard instances (small pᵢ). mean(p) is pulled down less",
        "than median(p), but median(1/p) is dominated by typical hardness → runtime exponent",
        "is steeper (larger) than |mean-p exponent|.",
        "",
        "Proxy used here: `median(p) / mean(p)` — near 1 is tight, falling below ~0.8 is loose.",
        "",
        "## Where to see mean-p vs median-rt training (06-09 run1)",
        "",
        "| depth | eval slope (mean-p train) | eval slope (median-rt train) | Δ |",
        "|------:|--------------------------:|-----------------------------:|--:|",
    ]
    run1 = [r for r in pairs if "1649-train12" in r["run"]]
    for r in sorted(run1, key=lambda x: x["depth"]):
        lines.append(
            f"| {r['depth']} | {r['mean_p_slope']:.4f} | {r['median_rt_slope']:.4f} | "
            f"{r['delta_median_minus_mean']:+.4f} |"
        )
    lines += [
        "",
        "File: `results/bm24_runs/06-09/run1/06-09_1649-train12-tr100-te200-n12-18.json`",
        "Keys: `trace_bm24_mean_p_fixed_n` vs `trace_median_runtime_fixed_n`, field `lr_log2_slope`.",
        "",
        "## Benchmark exponent gap (05-29 depth-15)",
        "",
    ]
    if bench_rows:
        b = bench_rows[0]
        lines += [
            f"- slope ln(mean p): {b['slope_mean_p']:+.4f}",
            f"- slope ln(median 1/p): {b['slope_median_rt']:+.4f}",
            f"- gap (median_rt + mean_p): {b['gap_rt_vs_neg_mean']:+.4f} log₂",
            f"- source: `{b['source']}`",
            "",
        ]
        if len(bench_rows) > 1:
            lines.append("### All benchmarks with per_n data")
            lines.append("| source | n range | |mean_p| | median_rt | gap |")
            lines.append("|--------|---------|--------:|----------:|----:|")
            for b in bench_rows:
                lines.append(
                    f"| {b['source']} | {b['ns'][0]}–{b['ns'][-1]} | "
                    f"{-b['slope_mean_p']:.4f} | {b['slope_median_rt']:.4f} | "
                    f"{b['gap_rt_vs_neg_mean']:+.4f} |"
                )
            lines.append("")
    if rb_rows:
        lines += ["## Re-benchmark exponent gaps by depth", ""]
        for mode in ("bm24_mean_p_fixed_n", "median_runtime_fixed_n"):
            lines.append(f"### {mode}")
            lines.append("| depth | |mean_p| slope | median_rt slope | gap |")
            lines.append("|------:|----------------:|----------------:|----:|")
            sub = sorted([r for r in rb_rows if r["mode"] == mode], key=lambda x: x["depth"])
            for r in sub:
                lines.append(
                    f"| {r['depth']} | {-r['slope_mean_p']:.4f} | {r['slope_median_rt']:.4f} | "
                    f"{r['gap_rt_plus_mean_p']:+.4f} |"
                )
            lines.append("")
    path = out_dir / "README.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: bm24_runs/exponent-divergence-analysis/",
    )
    p.add_argument(
        "--rebenchmark",
        type=Path,
        nargs="*",
        default=[],
        help="JSON run(s) to re-benchmark at saved angles for per-n spread",
    )
    p.add_argument(
        "--rebenchmark-depths",
        type=str,
        default=None,
        help="Comma depths for rebenchmark (default: all in file)",
    )
    args = p.parse_args()

    out_dir = args.output_dir or (bm24_runs_dir() / "exponent-divergence-analysis")
    roots = [bm24_runs_dir(), _PHASECRAFT / "results" / "bm24_runs"]
    json_paths = _iter_json_runs(roots)

    pairs = collect_training_mode_pairs(json_paths)
    bench_rows = collect_bench_per_n(json_paths)
    log_rows = parse_angle_log_exponents(
        list(bm24_runs_dir().glob("lr_train_optimal_angles*.txt"))
        + list((_PHASECRAFT / "results" / "bm24_runs").glob("lr_train_optimal_angles*.txt"))
    )

    rb_rows: List[dict] = []
    depths = None
    if args.rebenchmark_depths:
        depths = [int(x) for x in args.rebenchmark_depths.split(",") if x.strip()]
    for jp in args.rebenchmark:
        payload = json.loads(Path(jp).read_text(encoding="utf-8"))
        print(f"Re-benchmarking {jp} ...")
        rb_rows.extend(rebenchmark_angles(payload, depths=depths))

    saved: List[Path] = []
    saved += plot_training_mode_pairs(pairs, out_dir)
    saved += plot_bench_divergence(bench_rows, out_dir)
    saved += plot_all_bench_exponents(bench_rows, out_dir)
    saved += plot_rebenchmark(rb_rows, out_dir)

    summary = {
        "n_json_scanned": len(json_paths),
        "n_training_mode_pairs": len(pairs),
        "n_bench_with_per_n": len(bench_rows),
        "n_angle_log_entries": len(log_rows),
        "n_rebenchmark_rows": len(rb_rows),
        "training_mode_pairs": pairs,
        "bench_rows": [{k: v for k, v in b.items() if k != "settings"} for b in bench_rows],
        "rebenchmark": rb_rows,
    }
    json_out = out_dir / "summary.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    readme = write_summary(out_dir, pairs, bench_rows, rb_rows)

    print(f"\nWrote {len(saved)} plots + {json_out.name} + {readme.name} -> {out_dir}")
    for s in saved:
        print(f"  {s.name}")


if __name__ == "__main__":
    main()
