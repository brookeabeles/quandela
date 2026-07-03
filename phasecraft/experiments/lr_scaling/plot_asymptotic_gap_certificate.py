#!/usr/bin/env python3
"""Finite-window diagnostics for the BM24 annealed-vs-typical gap.

This script intentionally avoids claiming asymptotic convergence. It turns the
gap question into three finite-window checks:

1. Exact empirical identity:
       c_typ - c_ann = slope_n log2(mean p_succ / median p_succ).
2. Decomposition:
       median(X) - c_ann = [median(X) - mean(X)] + [mean(X) - c_ann].
   The second bracket is the Jensen/annealed term.
3. Cumulant truncation diagnostic:
       mean(X) - c_ann = (ln 2 / 2)n kappa_2
                         - (ln 2)^2/6 n^2 kappa_3
                         + (ln 2)^3/24 n^3 kappa_4 - ...

The variance-only/log-normal prediction is therefore a local, second-order
description of the Jensen term, not a proof that the gap persists as n -> inf.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "phasecraft-mpl"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LN2 = float(np.log(2.0))
PHASECRAFT = Path(__file__).resolve().parents[2]

CellKey = Tuple[int, int]  # (train_n, depth)


def slope(xs: Sequence[float], ys: Sequence[float]) -> float:
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 2:
        return float("nan")
    return float(np.polyfit(x[mask], y[mask], 1)[0])


def central_moments(xs: np.ndarray) -> Tuple[float, float, float, float]:
    mean_x = float(np.mean(xs))
    centered = xs - mean_x
    k2 = float(np.mean(centered**2))
    k3 = float(np.mean(centered**3))
    k4 = float(np.mean(centered**4) - 3.0 * k2 * k2)
    return mean_x, k2, k3, k4


def load_cells(csv_path: Path, objective: str) -> Dict[CellKey, Dict[int, dict]]:
    buckets: Dict[Tuple[int, int, int], Dict[str, List[float]]] = defaultdict(
        lambda: {"p_succ": [], "X": []}
    )
    with csv_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["objective"] != objective:
                continue
            p_succ = float(row["p_succ"])
            x_val = float(row["X"])
            if not (math.isfinite(p_succ) and p_succ > 0 and math.isfinite(x_val)):
                continue
            key = (int(row["train_n"]), int(row["p"]), int(row["n"]))
            buckets[key]["p_succ"].append(p_succ)
            buckets[key]["X"].append(x_val)

    cells: Dict[CellKey, Dict[int, dict]] = defaultdict(dict)
    for (train_n, depth, n), vals in sorted(buckets.items()):
        ps = np.asarray(vals["p_succ"], dtype=float)
        xs = np.asarray(vals["X"], dtype=float)
        cells[(train_n, depth)][n] = per_n_stats(ps, xs, int(n))
    return {k: dict(v) for k, v in cells.items()}


def per_n_stats(ps: np.ndarray, xs: np.ndarray, n: int) -> dict:
    mean_p = float(np.mean(ps))
    median_p = float(np.median(ps))
    median_x = float(np.median(xs))
    mean_x, k2, k3, k4 = central_moments(xs)
    c_ann_point = -float(np.log2(mean_p)) / float(n)
    point_gap = median_x - c_ann_point
    skew_gap = median_x - mean_x
    jensen_gap = mean_x - c_ann_point

    desc = np.sort(ps)[::-1]
    top1_count = max(1, int(math.ceil(0.01 * len(desc))))
    top5_count = max(1, int(math.ceil(0.05 * len(desc))))
    total_p = float(np.sum(desc))
    top1_share = float(np.sum(desc[:top1_count]) / total_p)
    top5_share = float(np.sum(desc[:top5_count]) / total_p)

    loo_shifts = []
    for idx in np.argsort(ps)[-max(1, min(5, len(ps))) :]:
        ps2 = np.delete(ps, idx)
        c_ann2 = -float(np.log2(np.mean(ps2))) / float(n)
        loo_shifts.append((median_x - c_ann2) - point_gap)
    loo_easiest_shift = float(min(loo_shifts, key=abs)) if loo_shifts else float("nan")
    loo_max_abs_shift = float(max(abs(v) for v in loo_shifts)) if loo_shifts else float("nan")

    return {
        "N": int(len(ps)),
        "p_succ": ps,
        "X": xs,
        "mean_p": mean_p,
        "median_p": median_p,
        "mean_X": mean_x,
        "median_X": median_x,
        "c_ann_point": c_ann_point,
        "point_gap": point_gap,
        "log2_ratio_consistent": float(n * point_gap),
        "skew_gap": skew_gap,
        "jensen_gap": jensen_gap,
        "kappa2_X": k2,
        "kappa3_X": k3,
        "kappa4_X": k4,
        "n_kappa2_X": float(n * k2),
        "n2_kappa3_X": float((n**2) * k3),
        "n3_kappa4_X": float((n**3) * k4),
        "jensen_second_order": float(0.5 * LN2 * n * k2),
        "jensen_third_order_term": float(-(LN2**2) * (n**2) * k3 / 6.0),
        "jensen_fourth_order_term": float((LN2**3) * (n**3) * k4 / 24.0),
        "top1_mass_share": top1_share,
        "top5_mass_share": top5_share,
        "loo_easiest_shift": loo_easiest_shift,
        "loo_max_abs_shift": loo_max_abs_shift,
    }


def summarize_cell(by_n: Mapping[int, dict], tail_n_min: int) -> dict:
    ns = sorted(by_n)
    mean_ps = [float(by_n[n]["mean_p"]) for n in ns]
    median_ps = [float(by_n[n]["median_p"]) for n in ns]
    log2_ratio = [float(by_n[n]["log2_ratio_consistent"]) for n in ns]
    skew_y = [float(n * by_n[n]["skew_gap"]) for n in ns]
    jensen_y = [float(n * by_n[n]["jensen_gap"]) for n in ns]
    second_y = [float(n * by_n[n]["jensen_second_order"]) for n in ns]

    tail_ns = [n for n in ns if n >= tail_n_min] or ns
    summary = {
        "ns": ns,
        "tail_ns": tail_ns,
        "c_ann": -slope(ns, np.log2(mean_ps)),
        "c_typ": -slope(ns, np.log2(median_ps)),
        "measured_gap_slope": slope(ns, log2_ratio),
        "skew_slope": slope(ns, skew_y),
        "jensen_slope": slope(ns, jensen_y),
        "second_order_slope": slope(ns, second_y),
        "tail_n_kappa2_X": float(np.mean([by_n[n]["n_kappa2_X"] for n in tail_ns])),
        "tail_n2_kappa3_X": float(np.mean([by_n[n]["n2_kappa3_X"] for n in tail_ns])),
        "tail_n3_kappa4_X": float(np.mean([by_n[n]["n3_kappa4_X"] for n in tail_ns])),
        "tail_second_order": float(np.mean([by_n[n]["jensen_second_order"] for n in tail_ns])),
        "tail_third_order_term": float(np.mean([by_n[n]["jensen_third_order_term"] for n in tail_ns])),
        "tail_fourth_order_term": float(np.mean([by_n[n]["jensen_fourth_order_term"] for n in tail_ns])),
        "tail_point_gap": float(np.mean([by_n[n]["point_gap"] for n in tail_ns])),
        "tail_jensen_gap": float(np.mean([by_n[n]["jensen_gap"] for n in tail_ns])),
        "tail_skew_gap": float(np.mean([by_n[n]["skew_gap"] for n in tail_ns])),
        "tail_top1_mass_share": float(np.mean([by_n[n]["top1_mass_share"] for n in tail_ns])),
        "tail_top5_mass_share": float(np.mean([by_n[n]["top5_mass_share"] for n in tail_ns])),
        "tail_loo_max_abs_shift": float(np.mean([by_n[n]["loo_max_abs_shift"] for n in tail_ns])),
    }
    summary["decomposition_residual"] = (
        summary["measured_gap_slope"] - summary["skew_slope"] - summary["jensen_slope"]
    )
    return summary


def bootstrap_slope_cis(
    by_n: Mapping[int, dict],
    *,
    n_boot: int,
    rng: np.random.Generator,
) -> Dict[str, Tuple[float, float]]:
    ns = sorted(by_n)
    vals = {name: np.empty(int(n_boot), dtype=float) for name in ("gap", "jensen", "skew")}
    for b in range(int(n_boot)):
        gap_y = []
        jensen_y = []
        skew_y = []
        for n in ns:
            ps = np.asarray(by_n[n]["p_succ"], dtype=float)
            xs = np.asarray(by_n[n]["X"], dtype=float)
            idx = rng.integers(0, len(ps), len(ps))
            st = per_n_stats(ps[idx], xs[idx], int(n))
            gap_y.append(float(n * st["point_gap"]))
            jensen_y.append(float(n * st["jensen_gap"]))
            skew_y.append(float(n * st["skew_gap"]))
        vals["gap"][b] = slope(ns, gap_y)
        vals["jensen"][b] = slope(ns, jensen_y)
        vals["skew"][b] = slope(ns, skew_y)
    return {
        name: tuple(float(x) for x in np.quantile(arr, [0.025, 0.975]))
        for name, arr in vals.items()
    }


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 10.5,
            "legend.fontsize": 8.2,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.24,
            "grid.linewidth": 0.7,
        }
    )


def plot_mechanism(
    *,
    cells: Mapping[CellKey, Mapping[int, dict]],
    summaries: Mapping[CellKey, dict],
    cis: Mapping[CellKey, Mapping[str, Tuple[float, float]]],
    train_ns: Sequence[int],
    focus_depths: Sequence[int],
    out_png: Path,
    out_pdf: Path | None,
    tail_n_min: int,
) -> None:
    setup_style()
    colors = {20: "#C44E52", 50: "#8172B2"}
    markers = {12: "o", 16: "s"}
    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.25))

    # A: the observable whose slope is exactly c_typ - c_ann.
    ax = axes[0]
    for train_n in train_ns:
        for depth in focus_depths:
            key = (int(train_n), int(depth))
            if key not in cells:
                continue
            ns = sorted(cells[key])
            ys = [cells[key][n]["log2_ratio_consistent"] for n in ns]
            ls = "-" if int(train_n) == int(train_ns[0]) else "--"
            color = colors.get(int(depth), "0.35")
            ax.plot(
                ns,
                ys,
                marker=markers.get(int(train_n), "o"),
                color=color,
                ls=ls,
                lw=1.7,
                ms=5,
                label=rf"$p={depth}$, train $n={train_n}$",
            )
            fit = np.polyfit(np.asarray(ns, dtype=float), np.asarray(ys, dtype=float), 1)
            xx = np.linspace(min(ns), max(ns), 80)
            ax.plot(xx, np.polyval(fit, xx), color=color, ls=":", lw=1.0, alpha=0.8)
    ax.axvspan(tail_n_min, max(max(cells[k]) for k in cells), color="0.92", zorder=-5)
    ax.axhline(0.0, color="0.2", lw=0.8)
    ax.set_xlabel("system size n")
    ax.set_ylabel(r"$\log_2(\mathrm{mean}\,p_{\rm succ}/\mathrm{median}\,p_{\rm succ})$")
    ax.set_title(r"(a) Exact finite-window observable")
    ax.legend(loc="upper left", frameon=True, framealpha=0.92)

    # B: slope-level decomposition into skew and Jensen pieces.
    ax = axes[1]
    keys = [(tn, d) for tn in train_ns for d in focus_depths if (tn, d) in summaries]
    x = np.arange(len(keys), dtype=float)
    skew = np.array([summaries[k]["skew_slope"] for k in keys], dtype=float)
    jensen = np.array([summaries[k]["jensen_slope"] for k in keys], dtype=float)
    total = skew + jensen
    ax.bar(x, jensen, width=0.62, color="#4C72B0", alpha=0.78, label="Jensen term")
    ax.bar(
        x,
        skew,
        bottom=jensen,
        width=0.62,
        color="#DD8452",
        alpha=0.82,
        label="median(X)-mean(X) term",
    )
    ax.scatter(x, total, color="0.15", s=22, zorder=4, label="sum")
    ax.axhline(0.0, color="0.2", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"tn={tn}\np={d}" for tn, d in keys])
    ax.set_ylabel(r"slope contribution to $c_{\rm typ}-c_{\rm ann}$")
    ax.set_title("(b) Gap decomposition")
    ax.legend(loc="upper left", frameon=True, framealpha=0.92)

    # C: second-order prediction for the Jensen term only.
    ax = axes[2]
    all_depths = sorted({d for _, d in summaries})
    cmap = plt.cm.viridis
    depth_min, depth_max = min(all_depths), max(all_depths)
    for key, summary in sorted(summaries.items()):
        train_n, depth = key
        x_pred = float(summary["second_order_slope"])
        y_meas = float(summary["jensen_slope"])
        lo, hi = cis[key]["jensen"]
        color = cmap((depth - depth_min) / max(depth_max - depth_min, 1))
        ax.errorbar(
            x_pred,
            y_meas,
            yerr=[[y_meas - lo], [hi - y_meas]],
            fmt=markers.get(int(train_n), "o"),
            color=color,
            ecolor=color,
            capsize=2.5,
            ms=6,
            alpha=0.9,
        )
        if depth in focus_depths and train_n == int(train_ns[0]):
            ax.text(x_pred + 0.002, y_meas + 0.002, f"p={depth}", fontsize=8, color="0.25")
    vals = [
        *[float(s["second_order_slope"]) for s in summaries.values()],
        *[float(s["jensen_slope"]) for s in summaries.values()],
    ]
    lo = min(0.0, min(vals) - 0.01)
    hi = max(vals) + 0.015
    ax.plot([lo, hi], [lo, hi], color="0.25", lw=1.0, ls="--", label="y = x")
    ax.axhline(0.0, color="0.2", lw=0.8)
    ax.axvline(0.0, color="0.2", lw=0.8)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel(r"second-order slope $(\ln2/2)\,n\kappa_2(X)$")
    ax.set_ylabel(r"measured Jensen slope")
    ax.set_title("(c) Variance-only truncation")
    ax.legend(loc="upper left", frameon=False)

    fig.suptitle("Finite-window mechanism diagnostic for the typical-vs-annealed gap", y=1.03, fontsize=12)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    if out_pdf is not None:
        out_pdf.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def plot_cumulant_tail_diagnostics(
    *,
    cells: Mapping[CellKey, Mapping[int, dict]],
    train_ns: Sequence[int],
    focus_depths: Sequence[int],
    out_png: Path,
    out_pdf: Path | None,
) -> None:
    setup_style()
    colors = {20: "#C44E52", 50: "#8172B2"}
    markers = {12: "o", 16: "s"}
    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.1))

    # A: cumulant contributions to the Jensen term.
    ax = axes[0]
    line_defs = [
        ("jensen_second_order", "-", r"$+(\ln2/2)n\kappa_2$"),
        ("jensen_third_order_term", "--", r"$-(\ln2)^2 n^2\kappa_3/6$"),
        ("jensen_fourth_order_term", ":", r"$+(\ln2)^3 n^3\kappa_4/24$"),
    ]
    for train_n in train_ns:
        for depth in focus_depths:
            key = (int(train_n), int(depth))
            if key not in cells:
                continue
            ns = sorted(cells[key])
            for metric, ls, label in line_defs:
                ax.plot(
                    ns,
                    [cells[key][n][metric] for n in ns],
                    color=colors.get(int(depth), "0.35"),
                    marker=markers.get(int(train_n), "o") if metric == "jensen_second_order" else None,
                    ls=ls,
                    lw=1.45,
                    ms=4.5,
                    alpha=0.86,
                    label=label if key == (int(train_ns[0]), int(focus_depths[0])) else None,
                )
    ax.axhline(0.0, color="0.2", lw=0.8)
    ax.set_xlabel("system size n")
    ax.set_ylabel("contribution to pointwise Jensen gap")
    ax.set_title("(a) Higher cumulants are visible")
    ax.legend(loc="upper right", frameon=True, framealpha=0.9)

    # B: how concentrated the empirical mean success is in the easiest formulas.
    ax = axes[1]
    for train_n in train_ns:
        for depth in focus_depths:
            key = (int(train_n), int(depth))
            if key not in cells:
                continue
            ns = sorted(cells[key])
            color = colors.get(int(depth), "0.35")
            ls = "-" if int(train_n) == int(train_ns[0]) else "--"
            ax.plot(
                ns,
                [cells[key][n]["top1_mass_share"] for n in ns],
                color=color,
                ls=ls,
                marker=markers.get(int(train_n), "o"),
                lw=1.55,
                ms=4.5,
                label=rf"top 1%, $p={depth}$, tn={train_n}",
            )
            ax.plot(
                ns,
                [cells[key][n]["top5_mass_share"] for n in ns],
                color=color,
                ls=":",
                lw=1.45,
                alpha=0.88,
            )
    ax.set_xlabel("system size n")
    ax.set_ylabel(r"share of $\sum_i p_i$")
    ax.set_title("(b) Tail mass in empirical mean")
    ax.text(0.05, 0.92, "solid/dashed: top 1%; dotted: top 5%", transform=ax.transAxes, fontsize=8)
    ax.legend(loc="lower right", frameon=True, framealpha=0.9, fontsize=7.1)

    # C: leave-one-out sensitivity to the easiest observed instance.
    ax = axes[2]
    for train_n in train_ns:
        for depth in focus_depths:
            key = (int(train_n), int(depth))
            if key not in cells:
                continue
            ns = sorted(cells[key])
            color = colors.get(int(depth), "0.35")
            ls = "-" if int(train_n) == int(train_ns[0]) else "--"
            ax.plot(
                ns,
                [cells[key][n]["loo_max_abs_shift"] for n in ns],
                color=color,
                ls=ls,
                marker=markers.get(int(train_n), "o"),
                lw=1.6,
                ms=4.5,
                label=rf"$p={depth}$, train $n={train_n}$",
            )
    ax.set_xlabel("system size n")
    ax.set_ylabel(r"max $|\Delta_n^{(-i)}-\Delta_n|$")
    ax.set_title("(c) Leave-one-out easiest-instance shift")
    ax.legend(loc="upper left", frameon=True, framealpha=0.9)

    fig.suptitle("Cumulant and tail-robustness diagnostics", y=1.03, fontsize=12)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    if out_pdf is not None:
        out_pdf.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def write_theory_note(path: Path, summary_payload: Mapping[str, object]) -> None:
    rows = summary_payload["rows"]  # type: ignore[index]
    focus = [r for r in rows if r["depth"] in (20, 50)]  # type: ignore[index]
    lines = [
        "# Finite-Window Annealed-vs-Typical Gap Diagnostic",
        "",
        "This note is generated with `plot_asymptotic_gap_certificate.py`.",
        "",
        "## What Is Proven Exactly",
        "",
        "For each finite evaluation window, define",
        "",
        "```text",
        "X_n = -(1/n) log2 p_succ(instance).",
        "```",
        "",
        "The empirical slope gap is the slope of",
        "",
        "```text",
        "log2(mean p_succ / median p_succ).",
        "```",
        "",
        "Equivalently, at point level,",
        "",
        "```text",
        "median(X_n) - c_ann(n)",
        "  = [median(X_n) - mean(X_n)] + [mean(X_n) - c_ann(n)].",
        "```",
        "",
        "The second bracket is nonnegative by Jensen and is the piece targeted by",
        "the second-order/log-normal approximation.",
        "",
        "## What Is Only A Local Approximation",
        "",
        "The cumulant expansion gives",
        "",
        "```text",
        "mean(X_n) - c_ann(n)",
        "  = (ln2/2)n kappa_2(X_n)",
        "    - (ln2)^2 n^2 kappa_3(X_n)/6",
        "    + (ln2)^3 n^3 kappa_4(X_n)/24 - ...",
        "```",
        "",
        "If kappa_j(X_n) scales like n^{-(j-1)}, every displayed scaled cumulant",
        "can contribute O(1). Therefore the variance-only formula is a finite-window",
        "diagnostic, not an asymptotic theorem.",
        "",
        "## Current N=500 Window",
        "",
        "| train_n | depth | gap slope | Jensen slope | skew slope | second-order slope | top1 mass | top5 mass | LOO shift |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in focus:
        lines.append(
            "| {train_n} | {depth} | {measured_gap_slope:.4f} | {jensen_slope:.4f} | "
            "{skew_slope:.4f} | {second_order_slope:.4f} | "
            "{tail_top1_mass_share:.3f} | {tail_top5_mass_share:.3f} | "
            "{tail_loo_max_abs_shift:.4f} |".format(**r)
        )
    lines.extend(
        [
            "",
            "## Defensible Claim",
            "",
            "At p=20 and p=50 over n=12..20, the mean-vs-median exponent gap is",
            "well resolved and is mostly a Jensen/annealed effect. The variance-only",
            "term gives a useful local description of that Jensen component, but higher",
            "scaled cumulants are visible and must be tracked before making any n -> inf",
            "claim.",
            "",
            "## Next Measurements",
            "",
            "1. Repeat this diagnostic across independent training/evaluation seeds.",
            "2. Extend n substantially beyond 20.",
            "3. Track scaled cumulants n kappa2, n^2 kappa3, n^3 kappa4, ... directly.",
            "4. Use tail-aware estimates of E[p_succ] if top-tail concentration grows.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("results/bm24_runs/bm24_gap_audit/N500_n12-20_all.csv"),
    )
    parser.add_argument("--objective", default="mean_p")
    parser.add_argument("--train-ns", default="12,16")
    parser.add_argument("--focus-depths", default="20,50")
    parser.add_argument("--tail-n-min", type=int, default=16)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--output-png",
        type=Path,
        default=Path(
            "results/bm24_runs/bm24_gap_audit/CLOSE_gap_analysis/"
            "04_asymptotic_gap_certificate.png"
        ),
    )
    parser.add_argument(
        "--output-pdf",
        type=Path,
        default=Path(
            "results/bm24_runs/bm24_gap_audit/CLOSE_gap_analysis/"
            "04_asymptotic_gap_certificate.pdf"
        ),
    )
    parser.add_argument(
        "--diagnostics-png",
        type=Path,
        default=Path(
            "results/bm24_runs/bm24_gap_audit/CLOSE_gap_analysis/"
            "05_cumulant_tail_diagnostics.png"
        ),
    )
    parser.add_argument(
        "--diagnostics-pdf",
        type=Path,
        default=Path(
            "results/bm24_runs/bm24_gap_audit/CLOSE_gap_analysis/"
            "05_cumulant_tail_diagnostics.pdf"
        ),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path(
            "results/bm24_runs/bm24_gap_audit/CLOSE_gap_analysis/"
            "04_asymptotic_gap_certificate_summary.json"
        ),
    )
    parser.add_argument(
        "--theory-note",
        type=Path,
        default=Path(
            "results/bm24_runs/bm24_gap_audit/CLOSE_gap_analysis/"
            "04_asymptotic_gap_theory.md"
        ),
    )
    args = parser.parse_args()

    csv_path = args.csv if args.csv.is_absolute() else PHASECRAFT / args.csv
    out_png = args.output_png if args.output_png.is_absolute() else PHASECRAFT / args.output_png
    out_pdf = args.output_pdf if args.output_pdf.is_absolute() else PHASECRAFT / args.output_pdf
    diagnostics_png = (
        args.diagnostics_png if args.diagnostics_png.is_absolute() else PHASECRAFT / args.diagnostics_png
    )
    diagnostics_pdf = (
        args.diagnostics_pdf if args.diagnostics_pdf.is_absolute() else PHASECRAFT / args.diagnostics_pdf
    )
    summary_path = args.summary if args.summary.is_absolute() else PHASECRAFT / args.summary
    theory_note = args.theory_note if args.theory_note.is_absolute() else PHASECRAFT / args.theory_note

    train_ns = [int(x) for x in str(args.train_ns).split(",") if x.strip()]
    focus_depths = [int(x) for x in str(args.focus_depths).split(",") if x.strip()]

    cells = load_cells(csv_path, args.objective)
    cells = {k: v for k, v in cells.items() if k[0] in train_ns}
    summaries = {
        key: summarize_cell(by_n, int(args.tail_n_min))
        for key, by_n in cells.items()
    }
    rng = np.random.default_rng(int(args.seed))
    cis = {
        key: bootstrap_slope_cis(cells[key], n_boot=int(args.bootstrap), rng=rng)
        for key in summaries
    }

    plot_mechanism(
        cells=cells,
        summaries=summaries,
        cis=cis,
        train_ns=train_ns,
        focus_depths=focus_depths,
        out_png=out_png,
        out_pdf=out_pdf,
        tail_n_min=int(args.tail_n_min),
    )
    plot_cumulant_tail_diagnostics(
        cells=cells,
        train_ns=train_ns,
        focus_depths=focus_depths,
        out_png=diagnostics_png,
        out_pdf=diagnostics_pdf,
    )

    summary_rows = []
    for key, summary in sorted(summaries.items()):
        train_n, depth = key
        row = {
            "train_n": train_n,
            "depth": depth,
            **summary,
            "bootstrap_ci": cis[key],
        }
        summary_rows.append(row)
    summary_payload = {
        "csv": str(csv_path),
        "objective": args.objective,
        "tail_n_min": int(args.tail_n_min),
        "bootstrap": int(args.bootstrap),
        "rows": summary_rows,
        "outputs": {
            "mechanism_png": str(out_png),
            "mechanism_pdf": str(out_pdf),
            "diagnostics_png": str(diagnostics_png),
            "diagnostics_pdf": str(diagnostics_pdf),
            "theory_note": str(theory_note),
        },
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    write_theory_note(theory_note, summary_payload)

    print(out_png)
    print(out_pdf)
    print(diagnostics_png)
    print(diagnostics_pdf)
    print(summary_path)
    print(theory_note)


if __name__ == "__main__":
    main()
