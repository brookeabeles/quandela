#!/usr/bin/env python3
"""Close the annealed-vs-typical exponent gap argument.

Exact algebraic identity (no approximation):
    c_typ - c_ann = slope_n [ log2( mean_p(n) / median_p(n) ) ]

Proof:
    median(1/p) = 1/median(p)          [1/x is monotone decreasing]
    c_typ = slope_n log2(1/median_p)   [definition of median-runtime exponent]
    c_ann = slope_n log2(1/mean_p)     [definition of annealed exponent]
    c_typ - c_ann = slope_n log2(mean_p/median_p)   [subtract, exact]

So the gap is nonzero iff log2(mean_p/median_p) grows with n, which holds iff the
distribution of p_succ is right-skewed (mean > median) and the skewness grows with n.

Usage::

    python experiments/lr_scaling/close_gap_argument.py \\
        results/bm24_runs/bm24_gap_audit/prelim_n12-16_N100.csv

    python experiments/lr_scaling/close_gap_argument.py \\
        results/bm24_runs/bm24_gap_audit/N500_n12-16.csv \\
        results/bm24_runs/bm24_gap_audit/prelim_n12-16_N100.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "phasecraft-mpl"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LN2 = float(np.log(2.0))
REQUIRED_COLS = ["seed", "n", "p", "objective", "train_n", "X", "p_succ"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_rows(paths: Sequence[Path]) -> List[dict]:
    rows: List[dict] = []
    for path in paths:
        with Path(path).open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                p_succ = float(r["p_succ"])
                x_val = float(r["X"])
                if not (math.isfinite(x_val) and math.isfinite(p_succ) and p_succ > 0):
                    continue
                rows.append({
                    "n": int(r["n"]),
                    "p": int(r["p"]),
                    "objective": str(r["objective"]),
                    "train_n": int(r["train_n"]),
                    "p_succ": p_succ,
                })
    return rows


def load_rows_merged(paths: Sequence[Path]) -> List[dict]:
    """Load CSVs; when cells overlap, keep the file with the most instances per cell."""
    cell_counts: Dict[Tuple[str, int, int, int], int] = {}
    cell_rows: Dict[Tuple[str, int, int, int], List[dict]] = {}
    for path in paths:
        buckets: Dict[Tuple[str, int, int, int], List[dict]] = defaultdict(list)
        for row in load_rows([path]):
            key = (row["objective"], int(row["train_n"]), int(row["p"]), int(row["n"]))
            buckets[key].append(row)
        for key, rs in buckets.items():
            if len(rs) > cell_counts.get(key, 0):
                cell_counts[key] = len(rs)
                cell_rows[key] = rs
    out: List[dict] = []
    for rs in cell_rows.values():
        out.extend(rs)
    return out


def group_rows(
    rows: Sequence[dict], objective: str = "mean_p"
) -> Dict[Tuple[int, int], Dict[int, np.ndarray]]:
    """Returns {(train_n, depth) -> {n -> array of p_succ values}}."""
    grouped: Dict[Tuple[int, int], Dict[int, list]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["objective"] != objective:
            continue
        grouped[(r["train_n"], r["p"])][r["n"]].append(r["p_succ"])
    return {k: {n: np.array(v) for n, v in by_n.items()} for k, by_n in grouped.items()}


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def slope_log2(ns: np.ndarray, ys: np.ndarray) -> float:
    mask = np.isfinite(ys) & (ys > 0)
    if mask.sum() < 2:
        return float("nan")
    return float(np.polyfit(ns[mask], np.log2(ys[mask]), 1)[0])


def slope_plain(ns: np.ndarray, ys: np.ndarray) -> float:
    mask = np.isfinite(ys)
    if mask.sum() < 2:
        return float("nan")
    return float(np.polyfit(ns[mask], ys[mask], 1)[0])


def bootstrap_log2_ratio(ps: np.ndarray, *, B: int = 1000, rng: np.random.Generator) -> Tuple[float, float]:
    """Bootstrap SE of log2(mean/median)."""
    estimates = []
    n = len(ps)
    for _ in range(B):
        s = rng.choice(ps, size=n, replace=True)
        mp, med = float(np.mean(s)), float(np.median(s))
        if med > 0 and mp > 0:
            estimates.append(math.log2(mp / med))
    if len(estimates) < 10:
        return float("nan"), float("nan")
    arr = np.array(estimates)
    return float(np.mean(arr)), float(np.std(arr))


# ---------------------------------------------------------------------------
# Identity verification and gap decomposition
# ---------------------------------------------------------------------------

def compute_gap_decomposition(
    grouped: Dict[Tuple[int, int], Dict[int, np.ndarray]]
) -> List[dict]:
    results = []
    rng = np.random.default_rng(42)
    for (train_n, depth), by_n in sorted(grouped.items()):
        ns_sorted = sorted(by_n)
        if len(ns_sorted) < 2:
            continue
        ns = np.array(ns_sorted, dtype=float)
        mean_ps = np.array([float(np.mean(by_n[n])) for n in ns_sorted])
        med_ps = np.array([float(np.median(by_n[n])) for n in ns_sorted])
        Ns = np.array([len(by_n[n]) for n in ns_sorted])

        # Exact identity: these two must agree to machine precision
        c_ann = slope_log2(ns, mean_ps)   # slope of log2(1/mean_p) = -slope of log2(mean_p)
        # Note: slope_log2 computes slope of log2(y), so for 1/mean_p:
        c_ann_direct = -float(np.polyfit(ns, np.log2(mean_ps), 1)[0])
        c_typ_direct = -float(np.polyfit(ns, np.log2(med_ps + 1e-300), 1)[0])
        direct_gap = c_typ_direct - c_ann_direct

        log2_ratio = np.array([
            math.log2(mp / med) if mp > 0 and med > 0 else float("nan")
            for mp, med in zip(mean_ps, med_ps)
        ])
        spread_slope = slope_plain(ns, log2_ratio)

        # Bootstrap SE per n
        bootstrap_mean, bootstrap_se = [], []
        for n in ns_sorted:
            bm, bs = bootstrap_log2_ratio(by_n[n], B=500, rng=rng)
            bootstrap_mean.append(bm)
            bootstrap_se.append(bs)

        results.append({
            "train_n": train_n,
            "depth": depth,
            "ns": ns_sorted,
            "Ns": Ns.tolist(),
            "mean_ps": mean_ps.tolist(),
            "med_ps": med_ps.tolist(),
            "log2_ratio": log2_ratio.tolist(),
            "bootstrap_mean": bootstrap_mean,
            "bootstrap_se": bootstrap_se,
            "c_ann": c_ann_direct,
            "c_typ": c_typ_direct,
            "direct_gap": direct_gap,
            "spread_slope": spread_slope,
            "identity_residual": direct_gap - spread_slope,
            "spread_ratio_per_n": (med_ps / mean_ps).tolist(),
        })
    return results


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

def print_identity_table(results: List[dict]) -> None:
    print("\n=== Exact identity: c_typ - c_ann = slope_n log2(mean_p/median_p) ===\n")
    print(f"{'train_n':>7} {'depth':>5}  {'c_ann':>7}  {'c_typ':>7}  {'direct_gap':>10}  {'spread_slope':>12}  {'residual':>10}  {'N/n':>5}")
    print("-" * 80)
    for r in results:
        avg_N = int(np.mean(r["Ns"]))
        print(
            f"{r['train_n']:>7} {r['depth']:>5}  {r['c_ann']:>7.4f}  {r['c_typ']:>7.4f}"
            f"  {r['direct_gap']:>10.4f}  {r['spread_slope']:>12.6f}"
            f"  {r['identity_residual']:>10.2e}  {avg_N:>5d}"
        )

    print("\nResidual should be ~0 to machine precision (algebraic identity).")
    print("Positive direct_gap → positive spread_slope → mean_p/median_p grows with n.\n")


def print_spread_table(results: List[dict]) -> None:
    print("=== Per-n spread: log2(mean_p/median_p) with bootstrap SE ===\n")
    for r in results:
        print(f"  depth p={r['depth']:2d}, train_n={r['train_n']}:")
        for n, lr, bm, bs in zip(r["ns"], r["log2_ratio"], r["bootstrap_mean"], r["bootstrap_se"]):
            ok = "✓" if lr > 0 else "✗"
            se_str = f"±{bs:.3f}" if math.isfinite(bs) else "  n/a  "
            print(f"    n={n:2d}: log2(mean/med)={lr:+.3f} ({bm:+.3f} {se_str}) {ok}")
        print(f"    → slope = {r['spread_slope']:.4f}  (= exponent gap {r['direct_gap']:.4f})\n")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

DEPTH_COLORS = {
    2: "#0072B2",
    5: "#009E73",
    10: "#E69F00",
    20: "#D55E00",
    50: "#CC79A7",
}
PAPER_PT2_DEPTHS = (5, 10, 20, 50)  # drop p=2: gap ≈0.03, adds clutter without new physics

# Stable output names (no ".pt2.png" — editors/OS misparsed that as a broken extension).
OUT_SPREAD = "01_spread_vs_n.png"
OUT_SPREAD_PAPER = "01_spread_vs_n_paper.png"
OUT_GAP_VS_DEPTH = "02_gap_vs_depth.png"
OUT_SKEW_RATIO = "03_skew_ratio_vs_n.png"
DEFAULT_OUT_DIR = "CLOSE_gap_analysis"


def _apply_paper_rcparams() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "mathtext.fontset": "dejavusans",
            "font.size": 8.5,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "legend.fontsize": 7.5,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.linewidth": 0.8,
            "axes.edgecolor": "0.2",
            "axes.labelcolor": "0.1",
            "xtick.color": "0.15",
            "ytick.color": "0.15",
            "lines.linewidth": 1.5,
            "lines.markersize": 4.5,
            "savefig.dpi": 300,
            "axes.grid": True,
            "grid.alpha": 0.18,
            "grid.linewidth": 0.45,
        }
    )


def _reset_rcparams() -> None:
    import matplotlib as mpl

    mpl.rcdefaults()


def plot_log2_ratio_vs_n(results: List[dict], out: Path) -> None:
    """Key proof figure: log2(mean_p/median_p) vs n, one panel per train_n group."""
    _apply_paper_rcparams()
    try:
        train_ns = sorted(set(r["train_n"] for r in results))
        fig, axes = plt.subplots(
            1,
            len(train_ns),
            figsize=(3.35 * len(train_ns), 2.75),
            sharey=True,
            squeeze=False,
        )
        legend_handles, legend_labels = [], []

        for ax, tn in zip(axes[0], train_ns):
            sub = [r for r in results if r["train_n"] == tn]
            n_min = min(min(r["ns"]) for r in sub)
            n_max = max(max(r["ns"]) for r in sub)

            for r in sorted(sub, key=lambda x: x["depth"]):
                ns = np.array(r["ns"], dtype=float)
                lr = np.array(r["log2_ratio"])
                bs = np.array(r["bootstrap_se"])
                col = DEPTH_COLORS.get(r["depth"], "0.35")

                finite = np.isfinite(lr) & np.isfinite(bs)
                if finite.sum() < 2:
                    continue

                eb = ax.errorbar(
                    ns[finite],
                    lr[finite],
                    yerr=1.96 * bs[finite],
                    fmt="o",
                    color=col,
                    ecolor=col,
                    capsize=2.0,
                    ms=3.7,
                    mew=0.7,
                    elinewidth=0.9,
                    alpha=0.95,
                    zorder=3,
                )
                fit = np.polyfit(ns[finite], lr[finite], 1)
                x_range = np.linspace(n_min - 0.35, n_max + 0.35, 80)
                ax.plot(
                    x_range,
                    np.polyval(fit, x_range),
                    "-",
                    color=col,
                    linewidth=1.15,
                    alpha=0.74,
                    zorder=2,
                )
                label = rf"$p={r['depth']}$"
                if label not in legend_labels:
                    legend_handles.append(eb.lines[0])
                    legend_labels.append(label)

            ax.axhline(0, color="0.25", linewidth=0.7, alpha=0.38, zorder=1)
            ax.set_xlim(n_min - 0.55, n_max + 0.55)
            ax.set_xticks(sorted({n for r in sub for n in r["ns"] if n % 2 == 0}))
            ax.set_xlabel("System size $n$")
            ax.text(
                0.035,
                0.95,
                rf"trained at $n={tn}$",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=8,
                color="0.25",
            )
            ax.grid(True, axis="y")
            ax.grid(False, axis="x")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        axes[0, 0].set_ylabel(
            r"$\log_2(\mathrm{mean}\,p_{\mathrm{succ}} / \mathrm{median}\,p_{\mathrm{succ}})$"
        )
        fig.legend(
            legend_handles,
            legend_labels,
            loc="upper center",
            ncol=len(legend_labels),
            frameon=False,
            bbox_to_anchor=(0.5, 1.03),
            handlelength=1.2,
            columnspacing=1.0,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.94), w_pad=1.3)
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
    finally:
        _reset_rcparams()


def plot_log2_ratio_vs_n_pt2(
    results: List[dict],
    out: Path,
    *,
    train_n: int = 12,
    depths: Sequence[int] = PAPER_PT2_DEPTHS,
    ci_sigma: float = 1.96,
) -> None:
    """Paper-ready single-panel figure (filename suffix ``.pt2``)."""
    _apply_paper_rcparams()
    try:
        sub = sorted(
            [r for r in results if r["train_n"] == train_n and r["depth"] in depths],
            key=lambda x: x["depth"],
        )
        if len(sub) < 2:
            raise ValueError(f"need ≥2 depth curves for train_n={train_n}, depths={depths}")

        fig, ax = plt.subplots(figsize=(4.9, 3.15))
        n_min = min(min(r["ns"]) for r in sub)
        n_max = max(max(r["ns"]) for r in sub)

        for r in sub:
            ns = np.array(r["ns"], dtype=float)
            lr = np.array(r["log2_ratio"])
            bs = np.array(r["bootstrap_se"])
            col = DEPTH_COLORS.get(r["depth"], "gray")
            finite = np.isfinite(lr) & np.isfinite(bs)
            if finite.sum() < 2:
                continue

            n_inst = int(round(np.mean(r["Ns"])))
            delta_c = r["spread_slope"]
            ax.errorbar(
                ns[finite],
                lr[finite],
                yerr=ci_sigma * bs[finite],
                fmt="o",
                color=col,
                ecolor=col,
                capsize=2.0,
                ms=4.0,
                mew=0.8,
                elinewidth=0.9,
                zorder=3,
            )
            fit = np.polyfit(ns[finite], lr[finite], 1)
            x_range = np.linspace(n_min - 0.35, n_max + 0.55, 80)
            ax.plot(
                x_range,
                np.polyval(fit, x_range),
                "-",
                color=col,
                linewidth=1.2,
                alpha=0.74,
                zorder=2,
            )
            label_y = float(np.polyval(fit, n_max + 0.1))
            if r["depth"] == 20:
                label_y += 0.025
            elif r["depth"] == 50:
                label_y += 0.055
            elif r["depth"] == 10:
                label_y -= 0.01
            ax.text(
                n_max + 0.32,
                label_y,
                rf"$p={r['depth']}$, $\Delta c={delta_c:.3f}$",
                color=col,
                fontsize=7.7,
                va="center",
                ha="left",
                clip_on=False,
            )

        ax.axhline(0, color="0.25", linewidth=0.7, linestyle="-", alpha=0.45, zorder=1)
        ax.set_xlim(n_min - 0.6, n_max + 1.9)
        ax.set_xticks(sorted({n for r in sub for n in r["ns"]}))
        ax.set_xlabel("System size $n$")
        ax.set_ylabel(
            r"$\log_2(\mathrm{mean}\,p_{\mathrm{succ}} / \mathrm{median}\,p_{\mathrm{succ}})$"
        )
        ax.text(
            0.03,
            0.96,
            rf"trained at $n={train_n}$; $N={n_inst}$ per $n$",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=7.7,
            color="0.3",
        )
        ax.grid(True, axis="y")
        ax.grid(False, axis="x")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
    finally:
        _reset_rcparams()


def plot_gap_vs_depth(results: List[dict], out: Path) -> None:
    """Show that both direct_gap and spread_slope grow with p (and agree exactly)."""
    train_ns = sorted(set(r["train_n"] for r in results))
    fig, ax = plt.subplots(figsize=(8, 5))

    markers = ["o", "s", "^", "D"]
    for i, tn in enumerate(train_ns):
        sub = sorted([r for r in results if r["train_n"] == tn], key=lambda x: x["depth"])
        depths = [r["depth"] for r in sub]
        gaps = [r["direct_gap"] for r in sub]
        spreads = [r["spread_slope"] for r in sub]
        m = markers[i % len(markers)]
        ax.plot(depths, gaps, f"{m}-", linewidth=2.0, label=f"direct gap, trained n={tn}", markersize=7)
        ax.plot(depths, spreads, f"{m}--", linewidth=1.0, alpha=0.6, label=f"spread slope, trained n={tn}", markersize=5)

    ax.axhline(0, color="black", linewidth=0.8, alpha=0.4)
    ax.set_xlabel("QAOA depth p", fontsize=12)
    ax.set_ylabel("Typical minus annealed exponent gap", fontsize=12)
    ax.set_title("Exponent gap increases with circuit depth", fontsize=11)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_spread_ratio_vs_n(results: List[dict], out: Path) -> None:
    """Show median/mean < 1 at all n and decreasing — the raw evidence for right skew."""
    train_ns = sorted(set(r["train_n"] for r in results))
    fig, axes = plt.subplots(1, len(train_ns), figsize=(6.5 * len(train_ns), 4.5), squeeze=False)

    for ax, tn in zip(axes[0], train_ns):
        sub = [r for r in results if r["train_n"] == tn]
        for r in sorted(sub, key=lambda x: x["depth"]):
            ns = np.array(r["ns"], dtype=float)
            sr = np.array(r["spread_ratio_per_n"])
            col = DEPTH_COLORS.get(r["depth"], "gray")
            ax.plot(ns, sr, "o-", color=col, linewidth=1.8, markersize=5, label=f"p={r['depth']}")

        ax.axhline(1.0, color="black", linewidth=1.0, linestyle="--", alpha=0.5, label="mean = median")
        ax.set_xlabel("System size $n$", fontsize=12)
        ax.set_ylabel(r"$\mathrm{median}\,p_{\mathrm{succ}} / \mathrm{mean}\,p_{\mathrm{succ}}$", fontsize=11)
        ax.set_title(f"Angles trained at $n={tn}$", fontsize=11)
        ax.set_ylim(0.4, 1.15)
        ax.legend(fontsize=9, title="QAOA depth")
        ax.grid(True, alpha=0.3)

    fig.suptitle("Median success rate below mean (right-skewed distribution)", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csvs", type=Path, nargs="+", help="Gap-audit eval CSV files")
    p.add_argument("--objective", default="mean_p", choices=["mean_p", "median_rt"])
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument(
        "--no-paper",
        "--no-pt2",
        action="store_true",
        dest="no_paper",
        help=f"Skip paper-ready {OUT_SPREAD_PAPER} (default: write it).",
    )
    p.add_argument("--pt2-train-n", type=int, default=12)
    args = p.parse_args(argv)

    rows = load_rows_merged(args.csvs) if len(args.csvs) > 1 else load_rows(args.csvs)
    if not rows:
        raise SystemExit("No valid rows loaded.")

    grouped = group_rows(rows, objective=args.objective)
    results = compute_gap_decomposition(grouped)
    if not results:
        raise SystemExit("No depth groups found.")

    print_identity_table(results)
    print_spread_table(results)

    total_N = sum(sum(r["Ns"]) for r in results)
    print(f"Total instances across all (depth, n) cells: {total_N:,}")
    if total_N / len(results) < 500:
        print(
            "\n⚠  Average N per cell is low — consider running evaluate with --N 1000 or more\n"
            "   for tighter bootstrap confidence intervals.\n"
            "   Command:\n"
            "     python experiments/lr_scaling/bm24_gap_audit.py evaluate \\\n"
            "       --angles results/bm24_runs/bm24_gap_audit/angles_grid_existing.json \\\n"
            "       --N 1000 --n-min 12 --n-max 18 \\\n"
            "       --output results/bm24_runs/bm24_gap_audit/N1000_n12-18.csv"
        )

    out_dir = args.output_dir or (
        Path(args.csvs[0]).resolve().parent / DEFAULT_OUT_DIR
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_log2_ratio_vs_n(results, out_dir / OUT_SPREAD)
    plot_gap_vs_depth(results, out_dir / OUT_GAP_VS_DEPTH)
    plot_spread_ratio_vs_n(results, out_dir / OUT_SKEW_RATIO)
    if not args.no_paper:
        plot_log2_ratio_vs_n_pt2(
            results,
            out_dir / OUT_SPREAD_PAPER,
            train_n=int(args.pt2_train_n),
        )

    print(f"\nWrote figures to: {out_dir}")
    print(f"  {OUT_SPREAD}        — spread vs n (all train_n panels)")
    if not args.no_paper:
        print(f"  {OUT_SPREAD_PAPER} — paper panel (p=5,10,20,50; train_n={args.pt2_train_n})")
    print(f"  {OUT_GAP_VS_DEPTH}     — exponent gap vs depth")
    print(f"  {OUT_SKEW_RATIO}   — median/mean ratio vs n")

    ns_avail = sorted({n for r in results for n in r["ns"]})
    if max(ns_avail) <= 16:
        print(
            f"\nNote: per-instance gap-audit rows span n={ns_avail[0]}–{ns_avail[-1]} only.\n"
            "For larger n, run evaluate then re-plot:\n"
            "  python experiments/lr_scaling/bm24_gap_audit.py evaluate \\\n"
            "    --angles results/bm24_runs/bm24_gap_audit/angles_grid_existing.json \\\n"
            "    --N 100 --n-min 12 --n-max 18 \\\n"
            "    --output results/bm24_runs/bm24_gap_audit/prelim_n12-18_N100.csv\n"
            "  python experiments/lr_scaling/close_gap_argument.py \\\n"
            "    results/bm24_runs/bm24_gap_audit/prelim_n12-18_N100.csv"
        )


if __name__ == "__main__":
    main()
