"""Paper-ready summary figure for BM24 transition + PL diagnostics.

This combines:
1. refined transition boundary across beta,
2. PL homotopy Stokes/anti-Stokes diagnostics at beta_opt,
3. exact finite-n exponent gap switch along the same continued competitor branch.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-phasecraft")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
OUT = RESULTS / "paper_pl_summary"
BETA_OPT = 0.5433996420760803

BLUE = "#2b6cb0"
ORANGE = "#dd6b20"
PURPLE = "#6b46c1"
RED = "#c53030"
GRAY = "#4a5568"


def load_json(path: Path):
    with path.open() as f:
        return json.load(f)


def robust_transition_points(transitions: list[dict]):
    bx, by, be = [], [], []
    amb_x, amb_y = [], []
    for t in transitions:
        trs = t.get("transitions", [])
        if (
            len(trs) == 1
            and trs[0]["from"] == "seed_or_decoy"
            and trs[0]["to"] == "competitor"
        ):
            bx.append(float(t["beta"]))
            by.append(float(trs[0]["Gamma_mid"]))
            be.append(float(trs[0]["uncertainty"]))
        elif trs:
            amb_x.append(float(t["beta"]))
            amb_y.append(float(trs[0]["Gamma_mid"]))
    return np.array(bx), np.array(by), np.array(be), np.array(amb_x), np.array(amb_y)


def make_figure() -> list[str]:
    OUT.mkdir(parents=True, exist_ok=True)
    refined = load_json(RESULTS / "refined_transition_scan" / "refined_transition_scan_summary.json")
    pl = load_json(RESULTS / "pl_homotopy_tracker" / "pl_homotopy_tracker.json")
    bx, by, be, amb_x, amb_y = robust_transition_points(refined["transitions"])
    rows = [r for r in pl["rows"] if not r.get("competitor_failed")]
    G = np.array([float(r["Gamma"]) for r in rows])
    dRe = np.array([float(r["delta_re_phi"]) for r in rows])
    dIm = np.array([float(r["delta_im_mod_2pi_abs"]) for r in rows])
    seed_gap = np.array([float(r["seed"]["gap_abs"]) for r in rows])
    comp_gap = np.array([float(r["competitor"]["gap_abs"]) for r in rows])

    fig = plt.figure(figsize=(12.6, 8.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.05], wspace=0.28, hspace=0.34)
    ax0 = fig.add_subplot(gs[0, :])
    ax1 = fig.add_subplot(gs[1, 0])
    ax2 = fig.add_subplot(gs[1, 1])

    # A: global refined boundary.
    ax0.errorbar(bx, by, yerr=be, fmt="o-", color=PURPLE, ecolor="#718096", capsize=3, lw=1.8, ms=5)
    if len(amb_x):
        ax0.scatter(amb_x, amb_y, marker="x", s=70, color=RED, linewidths=2.0, label="nonmonotone slices")
    ax0.axvline(BETA_OPT, color=GRAY, ls="--", lw=1.2)
    for lo, hi in pl["crossings"]["physical_switch_intervals"]:
        ax0.axhspan(lo, hi, xmin=0.0, xmax=1.0, color=ORANGE, alpha=0.16)
    ax0.scatter([BETA_OPT], [1.8375], s=95, facecolor="white", edgecolor=ORANGE, linewidth=2.2, zorder=5)
    ax0.annotate(
        "PL-tested window",
        xy=(BETA_OPT, 1.8375),
        xytext=(0.575, 1.80),
        arrowprops=dict(arrowstyle="->", color=GRAY, lw=1.0),
        fontsize=10,
    )
    ax0.set_ylabel(r"transition $\Gamma=-\gamma$")
    ax0.set_xlabel(r"$\beta$")
    ax0.set_title("A. Physical seed-to-competitor transition band")
    ax0.grid(True, alpha=0.25)
    ax0.set_xlim(0.33, 0.67)
    ax0.set_ylim(1.64, 1.885)

    # B: PL necessary conditions.
    ax1.axhline(0.0, color=GRAY, lw=1.0)
    ax1.plot(G, dRe, "o-", color=PURPLE, label=r"$\Delta Re\,\Phi$")
    for lo, hi in pl["crossings"]["anti_stokes_intervals"]:
        ax1.axvspan(lo, hi, color=ORANGE, alpha=0.20, label="anti-Stokes / switch bracket")
    ax1b = ax1.twinx()
    ax1b.plot(G, dIm, "s--", color=BLUE, ms=4, label=r"$|\Delta Im\,\Phi|$ mod $2\pi$")
    ax1b.axhline(pl["stokes_tol"], color=BLUE, ls=":", lw=1.0, alpha=0.7)
    ax1.set_xlabel(r"$\Gamma=-\gamma$ at $\beta_\mathrm{opt}$")
    ax1.set_ylabel(r"$\Delta Re\,\Phi$", color=PURPLE)
    ax1b.set_ylabel("phase distance", color=BLUE)
    ax1.set_title("B. Stokes and anti-Stokes diagnostics align")
    ax1.grid(True, alpha=0.25)
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1b.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="upper left", fontsize=8)

    # C: exact finite-n gap switch on same branch.
    ax2.plot(G, seed_gap, "o-", color=BLUE, label="seed")
    ax2.plot(G, comp_gap, "o-", color=ORANGE, label="continued competitor")
    for lo, hi in pl["crossings"]["physical_switch_intervals"]:
        ax2.axvspan(lo, hi, color=ORANGE, alpha=0.20)
    ax2.set_yscale("log")
    ax2.set_xlabel(r"$\Gamma=-\gamma$ at $\beta_\mathrm{opt}$")
    ax2.set_ylabel(r"$|E-\lambda_n|$")
    ax2.set_title("C. Exact exponent selects the same switch")
    ax2.grid(True, alpha=0.25)
    ax2.legend(loc="lower left", fontsize=9)

    fig.suptitle("BM24 saddle transition: physical switch and PL necessary conditions", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    png = OUT / "paper_pl_transition_summary.png"
    pdf = OUT / "paper_pl_transition_summary.pdf"
    fig.savefig(png, dpi=220, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    report = OUT / "paper_pl_transition_summary_report.md"
    report.write_text(
        "\n".join(
            [
                "# Paper PL transition summary figure",
                "",
                "Recommended lead figure for the BM24 saddle-transition mechanism.",
                "",
                "Panel A shows the refined physical transition boundary across beta.",
                "Panel B shows that the continued beta-opt competitor satisfies the Stokes phase condition while crossing anti-Stokes.",
                "Panel C shows that exact finite-n exponent matching switches in the same Gamma bracket.",
                "",
                f"- PNG: `{png.name}`",
                f"- PDF: `{pdf.name}`",
                "",
            ]
        )
    )
    return [str(png), str(pdf), str(report)]


def main() -> None:
    for path in make_figure():
        print(path)


if __name__ == "__main__":
    main()
