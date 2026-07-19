#!/usr/bin/env python3
"""Plot the BM24 training-objective objection test.

BM24-style objection:
    Perhaps the annealed-vs-typical exponent gap appears only because the
    angles were trained on mean success probability. If the angles were trained
    on median runtime instead, c_typ and c_ann might agree.

This plot compares the measured gap c_typ - c_ann under both training
objectives on the same held-out N=500 audit rows.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "phasecraft-mpl"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from plot_asymptotic_gap_certificate import (  # noqa: E402
    bootstrap_slope_cis,
    load_cells,
    setup_style,
    summarize_cell,
)


def parse_ints(spec: str) -> list[int]:
    return [int(x) for x in str(spec).split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("results/bm24_runs/bm24_gap_audit/N500_n12-20_all.csv"),
    )
    parser.add_argument("--train-ns", default="12,16")
    parser.add_argument("--depths", default="2,5,10,20,50")
    parser.add_argument("--tail-n-min", type=int, default=16)
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=1000,
        help="Bootstrap resamples per point for 95%% confidence intervals.",
    )
    parser.add_argument(
        "--paper-style",
        action="store_true",
        help="Use thesis-paper styling and omit the explanatory annotation.",
    )
    parser.add_argument(
        "--output-png",
        type=Path,
        default=Path(
            "results/bm24_runs/bm24_gap_audit/CLOSE_gap_analysis/"
            "08_bm24_training_objective_objection.png"
        ),
    )
    parser.add_argument(
        "--output-pdf",
        type=Path,
        default=Path(
            "results/bm24_runs/bm24_gap_audit/CLOSE_gap_analysis/"
            "08_bm24_training_objective_objection.pdf"
        ),
    )
    args = parser.parse_args()

    train_ns = parse_ints(args.train_ns)
    depths = parse_ints(args.depths)
    objective_labels = (
        {
            "mean_p": "Mean-success training",
            "median_rt": "Median-runtime training",
        }
        if args.paper_style
        else {
            "mean_p": "trained on mean success",
            "median_rt": "trained on median runtime",
        }
    )
    colors = (
        {"mean_p": "#2C7BB6", "median_rt": "#DD8452"}
        if args.paper_style
        else {"mean_p": "#4C72B0", "median_rt": "#C44E52"}
    )
    markers = {"mean_p": "o", "median_rt": "s"}

    summaries = {}
    cis = {}
    rng = np.random.default_rng(20260708)
    for objective in ("mean_p", "median_rt"):
        cells = load_cells(args.csv, objective)
        selected_cells = {
            key: by_n
            for key, by_n in cells.items()
            if int(key[0]) in set(train_ns) and int(key[1]) in set(depths)
        }
        summaries[objective] = {
            key: summarize_cell(by_n, int(args.tail_n_min))
            for key, by_n in selected_cells.items()
        }
        cis[objective] = {
            key: bootstrap_slope_cis(by_n, n_boot=int(args.bootstrap_samples), rng=rng)["gap"]
            for key, by_n in selected_cells.items()
        }

    if args.paper_style:
        plt.rcParams.update(
            {
                "font.family": "serif",
                "mathtext.fontset": "cm",
                "axes.unicode_minus": False,
                "font.size": 10,
                "axes.labelsize": 11,
                "axes.titlesize": 9,
                "legend.fontsize": 9,
                "xtick.labelsize": 9.5,
                "ytick.labelsize": 9.5,
                "axes.linewidth": 0.8,
                "axes.spines.top": True,
                "axes.spines.right": True,
                "axes.grid": True,
                "grid.alpha": 0.20,
                "grid.linewidth": 0.7,
                "figure.facecolor": "white",
                "axes.facecolor": "white",
            }
        )
    else:
        setup_style()

    figsize = (4.05, 2.75) if args.paper_style and len(train_ns) == 1 else (10.8, 4.15)
    fig, axes = plt.subplots(1, len(train_ns), figsize=figsize, sharey=True)
    if len(train_ns) == 1:
        axes = [axes]

    for ax, train_n in zip(axes, train_ns):
        for objective in ("mean_p", "median_rt"):
            xs = []
            ys = []
            yerr_lo = []
            yerr_hi = []
            for depth in depths:
                key = (int(train_n), int(depth))
                row = summaries[objective].get(key)
                if row is None:
                    continue
                y = float(row["measured_gap_slope"])
                lo, hi = cis[objective].get(key, (float("nan"), float("nan")))
                xs.append(int(depth))
                ys.append(y)
                yerr_lo.append(max(0.0, y - float(lo)) if np.isfinite(lo) else 0.0)
                yerr_hi.append(max(0.0, float(hi) - y) if np.isfinite(hi) else 0.0)
            ax.errorbar(
                xs,
                ys,
                yerr=[yerr_lo, yerr_hi],
                fmt="-",
                marker=markers[objective],
                color=colors[objective],
                markerfacecolor="white" if args.paper_style else colors[objective],
                markeredgecolor=colors[objective],
                markeredgewidth=0.9 if args.paper_style else 1.0,
                linewidth=1.35 if args.paper_style else 2.0,
                markersize=4.8 if args.paper_style else 6.5,
                elinewidth=0.8 if args.paper_style else 0.9,
                capsize=2.0 if args.paper_style else 3.0,
                capthick=0.8,
                label=objective_labels[objective],
            )
        if not args.paper_style:
            ax.axhline(0.0, color="0.2", linewidth=1.0)
        if not args.paper_style:
            ax.set_xscale("log")
        if args.paper_style:
            ax.set_xlim(0, max(depths) + 2)
            ax.set_xticks(np.arange(0, max(depths) + 1, 10))
            ax.set_ylim(-0.024, 0.103)
            ax.set_yticks([0.00, 0.05, 0.10])
        else:
            ax.set_xticks(depths)
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.set_xlabel(r"Depth ($p$)" if args.paper_style else "QAOA depth p")
        if args.paper_style:
            pass
        else:
            ax.set_title(f"angles trained at n={train_n}")
            ax.text(
                0.03,
                0.95,
                "BM24 objection would predict\nmedian-runtime curve near 0",
                transform=ax.transAxes,
                va="top",
                fontsize=8.5,
                bbox={"facecolor": "white", "edgecolor": "0.85", "alpha": 0.9, "pad": 4},
            )
    axes[0].set_ylabel(
        r"Exponent gap ($\mathrm{c}_{\mathrm{rt}}-\mathrm{c}_{\mathrm{sp}}$)"
        if args.paper_style
        else r"Exponent gap $c_{\rm typ}-c_{\rm ann}$"
    )
    if args.paper_style:
        handles, labels = axes[-1].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.52, 0.98),
            ncol=2,
            frameon=False,
            handlelength=2.0,
            columnspacing=1.2,
        )
    else:
        axes[-1].legend(loc="lower right", frameon=True, framealpha=0.95)
        fig.suptitle(
            "Retraining on median runtime does not close the annealed-vs-typical exponent gap",
            y=1.02,
            fontsize=12,
        )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.88) if args.paper_style else None)

    args.output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_png, dpi=300, bbox_inches="tight")
    fig.savefig(args.output_pdf, bbox_inches="tight")
    print(args.output_png)
    print(args.output_pdf)


if __name__ == "__main__":
    main()
