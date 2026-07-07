"""Generate improved v2 copies of selected PAPER-RESULTS figures.

Does not overwrite originals — writes *_v2.png alongside them.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]  # phasecraft/
OUT = REPO / "results" / "bm24_saddle_audit_p1" / "PAPER-RESULTS"
JSON_DIR = OUT / "json"

PHI_PREF = -0.689609375
GAMMA_ANTI_STOKES = -1.830842334140663

BLUE = "#2b6cb0"
ORANGE = "#dd6b20"
GREEN = "#2f855a"
RED = "#c53030"
GRAY = "#718096"
LIGHT = "#e2e8f0"


def load_json(name: str) -> dict:
    with (JSON_DIR / name).open() as f:
        return json.load(f)


def plot_prefactor_fig4_v2() -> Path:
    data = load_json("prefactor_convergence.json")
    rows = data["results"]

    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 1, height_ratios=[2.0, 1.0], hspace=0.28)
    ax_top = fig.add_subplot(gs[0])
    ax_bot = fig.add_subplot(gs[1], sharex=ax_top)

    n_corr = np.linspace(18, 40, 200)
    for row in rows:
        beta, gamma = float(row["beta"]), float(row["gamma"])
        c = row["color"]
        lb = row["label"]
        c0, c1 = float(row["fit_c0"]), float(row["fit_c1"])
        L0 = float(row["full_L0"])
        ns = np.array(row["n_values"], dtype=float)
        exact = np.array(row["lam_exact_n"], dtype=float)
        corrected = L0 - c1 / n_corr

        if abs(beta - 0.5434) < 1e-4 and abs(gamma - (-1.83)) < 0.01:
            lw, alpha = 2.4, 1.0
            zorder = 5
        elif abs(beta - 0.5434) < 1e-4:
            lw, alpha = 1.6, 0.9
            zorder = 3
        else:
            lw, alpha = 1.2, 0.55
            zorder = 2

        ax_top.axhline(L0, color=c, lw=0.7, ls="--", alpha=0.35 * (alpha / 0.9))
        ax_top.plot(n_corr, corrected, "-", color=c, lw=lw, alpha=alpha, zorder=zorder, label=lb)
        ax_top.plot(ns, exact, "o", color=c, ms=3.5, alpha=alpha, zorder=zorder + 1)

        resid = np.abs(exact - (L0 - c1 / ns))
        ax_bot.plot(ns, resid, "o-", color=c, lw=lw * 0.8, ms=3.5, alpha=alpha, label=lb)

    # Two-saddle overlay at crossing (from two_saddle_crossing.json).
    two = load_json("two_saddle_crossing.json")
    cross = next(r for r in two["results"] if abs(float(r["gamma"]) - (-1.83)) < 0.005)
    ns_two = np.array([int(t["n"]) for t in cross["two_saddle"]], dtype=float)
    lam_two = np.array([float(t["lam_two"]) for t in cross["two_saddle"]])
    ax_top.plot(
        ns_two,
        lam_two,
        "-",
        color=GREEN,
        lw=2.6,
        label=r"Two-saddle at $\gamma=-1.83$",
        zorder=6,
    )
    ax_bot.plot(
        ns_two,
        np.abs(lam_two - np.array([float(cross["lam_exact"][str(int(n))]) for n in ns_two])),
        "s-",
        color=GREEN,
        lw=2.2,
        ms=4,
        zorder=6,
    )

    ax_top.set_ylabel(r"$\log|P_n|/n$")
    ax_top.set_title(
        r"Single-saddle $+\,c_1/n$ correction (v2)"
        "\n"
        r"Dashed $=$ leading order $L_0$; solid $=$ $L_0 - c_1/n$ fit; dots $=$ exact"
        "\n"
        r"At anti-Stokes ($\gamma=-1.83$): red curve misses exact even after $c_1/n$; green two-saddle tracks better"
    )
    ax_top.legend(fontsize=7, ncol=2, loc="lower right")
    ax_top.grid(True, alpha=0.3)

    ax_bot.set_xlabel(r"$n$")
    ax_bot.set_ylabel(r"$|$exact $-$ corrected$|$")
    ax_bot.set_title(r"Residual after single-saddle $c_1/n$ fit (lower is better)")
    ax_bot.grid(True, alpha=0.3)

    out = OUT / "prefactor_fig4_corrected_estimate_v2.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_two_saddle_fig3_v2() -> Path:
    two = load_json("two_saddle_crossing.json")
    cross = next(r for r in two["results"] if abs(float(r["gamma"]) - (-1.83)) < 0.005)

    ns = np.array([int(t["n"]) for t in cross["two_saddle"]], dtype=float)
    exact_n = np.array([float(cross["lam_exact"][str(int(n))]) for n in ns])
    seed_n = np.full_like(ns, float(cross["full_seed"]))
    comp_n = np.full_like(ns, float(cross["full_comp"]))
    two_n = np.array([float(t["lam_two"]) for t in cross["two_saddle"]])
    log2_n = PHI_PREF + float(cross["re_phi_seed"]) + np.log(2.0) / ns

    single_lo = np.minimum(seed_n, comp_n)
    single_hi = np.maximum(seed_n, comp_n)

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))

    ax = axes[0]
    ax.fill_between(ns, single_lo, two_n, color=LIGHT, alpha=0.9, label="Bracket: [best single, two-saddle]")
    ax.plot(ns, exact_n, "kD-", lw=2.0, ms=5, label=r"Exact $\lambda_{\rm abs}(n)$", zorder=5)
    ax.plot(ns, seed_n, "--", color=BLUE, lw=1.8, label="Seed only (single saddle)")
    ax.plot(ns, comp_n, "--", color=ORANGE, lw=1.8, label="Branch 43 only (single saddle)")
    ax.plot(ns, two_n, "-", color=GREEN, lw=2.6, label="Two-saddle sum", zorder=4)
    ax.plot(ns, log2_n, ":", color=RED, lw=1.8, label=r"Seed $+\,\log 2/n$ (equal-weight limit)")
    ax.set_xlabel(r"$n$")
    ax.set_ylabel(r"$\log|P_n|/n$")
    ax.set_title(
        rf"Anti-Stokes $\gamma=-1.83$, $\beta=0.5434$"
        "\n"
        r"Exact lies between single-saddle levels and the two-saddle sum"
    )
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    seed_gap = seed_n - exact_n
    comp_gap = comp_n - exact_n
    two_gap = two_n - exact_n
    log2_gap = log2_n - exact_n
    bracket_lo = np.minimum(seed_gap, comp_gap)
    bracket_hi = np.maximum(seed_gap, two_gap)

    ax.fill_between(ns, bracket_lo, bracket_hi, color=LIGHT, alpha=0.9)
    ax.axhline(0, color="k", lw=0.9)
    ax.plot(ns, seed_gap, "--", color=BLUE, lw=1.8, label="Seed gap")
    ax.plot(ns, comp_gap, "--", color=ORANGE, lw=1.8, label="Branch-43 gap")
    ax.plot(ns, two_gap, "-", color=GREEN, lw=2.6, label="Two-saddle gap")
    ax.plot(ns, log2_gap, ":", color=RED, lw=1.8, label=r"Seed $+\,\log 2/n$ gap")
    ax.set_xlabel(r"$n$")
    ax.set_ylabel(r"approximation $-\,\lambda_{\rm abs}(n)$")
    ax.set_title(r"Gaps: bracket width $\propto 1/n$ near crossing (both singles wrong-signed)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        rf"Why one saddle fails at $\gamma_{{\rm AS}}\approx{GAMMA_ANTI_STOKES:.4f}$"
        "\n"
        r"$\lambda_{\rm two}(n)=\phi_{\rm pref}+\frac{1}{n}\log(e^{n\,\mathrm{Re}\Phi_{\rm seed}}+e^{n\,\mathrm{Re}\Phi_{43}})$"
        "\n"
        r"(equal Gaussian prefactors; see paper_two_saddle.py)",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))

    out = OUT / "two_saddle_fig3_crossing_n_convergence_v2.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    p1 = plot_prefactor_fig4_v2()
    p2 = plot_two_saddle_fig3_v2()
    print(f"Wrote {p1}")
    print(f"Wrote {p2}")


if __name__ == "__main__":
    main()
