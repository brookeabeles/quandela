"""Four-panel workflow figure for the BM24 saddle competitor story.

This figure reuses existing saved results. Panels 1 and 2 are data-backed
schematics because the repo retains search budgets and certified/clustered root
counts, but not the raw Newton start trajectories or rejected candidate boxes.
Panels 3 and 4 are fully data-driven from the saved certified-root outputs.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-phasecraft")

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RESULTS = REPO / "results" / "bm24_saddle_audit_p1"
OUT = RESULTS / "PAPER-RESULTS"

PHI_PREF = -0.6896093750000001
PRIMARY_GAMMA = -1.830842334140663
GAMMA_PAIR_TOL = 0.011

BLUE = "#2b6cb0"
ORANGE = "#dd6b20"
GREEN = "#2f855a"
RED = "#c53030"
PURPLE = "#6b46c1"
GRAY = "#4a5568"
LIGHT_GRAY = "#e2e8f0"
MID_GRAY = "#718096"
BLACK = "#1a202c"


def load_json(path: Path) -> Any:
    with path.open() as f:
        return json.load(f)


def savefig(fig: plt.Figure, stem: str) -> dict[str, str]:
    OUT.mkdir(parents=True, exist_ok=True)
    png = OUT / f"{stem}.png"
    pdf = OUT / f"{stem}.pdf"
    fig.savefig(png, dpi=220, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return {"png": str(png), "pdf": str(pdf)}


def interp(gamma: float, xs: np.ndarray, ys: np.ndarray) -> float:
    order = np.argsort(xs)
    return float(np.interp(gamma, xs[order], ys[order]))


def im_mod_zero(x: float) -> float:
    return abs((x + math.pi) % (2.0 * math.pi) - math.pi)


def load_seed() -> dict[str, np.ndarray]:
    data = load_json(RESULTS / "run_seed_branch_g-2pi" / "seed_branch_continuation.json")
    rows = [r for r in data["continuation"] if r.get("certified") and not r.get("failed")]
    return {
        "gamma": np.array([float(r["gamma"]) for r in rows]),
        "seed_re": np.array([float(r["re_phi_m"]) for r in rows]),
        "seed_exp": np.array([float(r["full_conv2_exponent"]) for r in rows]),
        "exact": np.array([float(r["lambda_abs_n_max"]) for r in rows]),
    }


def load_pair_series() -> dict[str, np.ndarray]:
    resolved = load_json(
        RESULTS
        / "run_seed_branch_g-2pi"
        / "archive"
        / "competitor_dominance"
        / "z_robust_full"
        / "resolved_robust_z.json"
    )
    seed = load_seed()
    tables: list[dict[float, dict[str, float]]] = []
    for bid in (43, 46):
        br = next(b for b in resolved["branches"] if int(b["branch_id"]) == bid)
        tab = {}
        for p in br["points"]:
            if not p.get("krawczyk_certified", True) or not p.get("box_disjoint_from_seed", True):
                continue
            g = round(float(p["gamma"]), 6)
            tab[g] = {
                "re": float(p["Phi_eff_real"]),
                "im": float(p["Phi_eff_imag"]),
            }
        tables.append(tab)

    all_g = sorted(set(tables[0]) | set(tables[1]), reverse=True)
    gammas, pair_re, pair_im_mod, seed_exp, seed_re, exact = [], [], [], [], [], []
    for g in all_g:
        vals, imvals = [], []
        for tab in tables:
            if g in tab:
                vals.append(tab[g]["re"])
                imvals.append(tab[g]["im"])
                continue
            near = [(abs(g - g2), row) for g2, row in tab.items() if abs(g - g2) <= GAMMA_PAIR_TOL]
            if near:
                row = min(near, key=lambda t: t[0])[1]
                vals.append(row["re"])
                imvals.append(row["im"])
        if not vals:
            continue
        gammas.append(g)
        pair_re.append(float(np.mean(vals)))
        pair_im_mod.append(float(np.mean([im_mod_zero(v) for v in imvals])))
        seed_exp.append(interp(g, seed["gamma"], seed["seed_exp"]))
        seed_re.append(interp(g, seed["gamma"], seed["seed_re"]))
        exact.append(interp(g, seed["gamma"], seed["exact"]))

    pair_re_arr = np.array(pair_re)
    return {
        "gamma": np.array(gammas),
        "pair_re": pair_re_arr,
        "pair_exp": PHI_PREF + pair_re_arr,
        "pair_im_mod": np.array(pair_im_mod),
        "seed_exp": np.array(seed_exp),
        "seed_re": np.array(seed_re),
        "exact": np.array(exact),
        "delta_re": pair_re_arr - np.array(seed_re),
    }


def load_all_candidate_points(seed: dict[str, np.ndarray]) -> np.ndarray:
    comp = load_json(
        RESULTS
        / "run_seed_branch_g-2pi"
        / "archive"
        / "competitor_dominance"
        / "competitor_dominance_summary.json"
    )
    rows = []
    for pg in comp["per_gamma"]:
        g = float(pg["gamma"])
        lam = interp(g, seed["gamma"], seed["exact"])
        for c in pg.get("competitors", []):
            re_phi = float(c["re_phi_m"])
            full = PHI_PREF + re_phi
            rows.append(
                (
                    g,
                    float(c["signed_re_phi_gap_vs_seed"]),
                    abs(full - lam),
                    re_phi,
                )
            )
    return np.asarray(rows, dtype=float)


def choose_validation_row() -> dict[str, Any]:
    robust = load_json(
        RESULTS
        / "run_seed_branch_g-2pi"
        / "archive"
        / "competitor_dominance"
        / "z_robust_full"
        / "competitors_robust_z.json"
    )
    points = robust["points"]
    target = min(points, key=lambda p: abs(float(p["gamma"]) - PRIMARY_GAMMA))
    return target


def add_panel_label(ax: plt.Axes, text: str) -> None:
    ax.text(
        0.0,
        1.03,
        text,
        transform=ax.transAxes,
        fontsize=13,
        fontweight="bold",
        ha="left",
        va="bottom",
        color=BLACK,
    )


def panel_explore(ax: plt.Axes, dominance: dict[str, Any]) -> None:
    add_panel_label(ax, "1. Explore the saddle landscape")
    rng = np.random.default_rng(123)
    rows = dominance["per_gamma"]
    x_positions = np.arange(len(rows), dtype=float)

    for x, row in zip(x_positions, rows, strict=True):
        n_starts = int(dominance["metadata"]["num_random_starts"])
        n_comp = int(row["num_certified_competitors"])
        n_show = 140
        starts_y = rng.uniform(0.05, 0.95, size=n_show)
        root_levels = np.linspace(0.18, 0.82, num=min(max(n_comp // 18, 3), 7))
        root_sizes = np.linspace(120, 360, num=len(root_levels))

        for y in starts_y:
            root_y = float(rng.choice(root_levels))
            jitter = rng.normal(0.0, 0.013)
            ax.plot([x - 0.22, x + 0.12], [y, root_y + jitter], color="#cbd5e0", lw=0.35, alpha=0.18, zorder=1)
            ax.scatter(x - 0.22, y, s=3, color="#a0aec0", alpha=0.25, zorder=2)

        for root_y, size in zip(root_levels, root_sizes, strict=True):
            ax.scatter(
                x + 0.16,
                root_y,
                s=size,
                facecolor=LIGHT_GRAY,
                edgecolor=MID_GRAY,
                lw=0.8,
                alpha=0.95,
                zorder=3,
            )

        ax.text(x - 0.22, 1.015, "random\nstarts", ha="center", va="bottom", fontsize=7.5, color=MID_GRAY)
        ax.text(
            x + 0.16,
            -0.06,
            f"{n_starts} starts\n{n_comp} certified\ncompetitors",
            ha="center",
            va="top",
            fontsize=7.5,
            color=GRAY,
        )

    ax.set_xlim(-0.6, len(rows) - 0.4)
    ax.set_ylim(-0.12, 1.12)
    ax.set_xticks(x_positions)
    ax.set_xticklabels([fr"$\gamma={float(r['gamma']):.2f}$" for r in rows], fontsize=8)
    ax.set_yticks([])
    ax.grid(False)
    for side in ("left", "right", "top"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#cbd5e0")
    ax.set_title("Many Newton starts are needed because competitors have no known starting branch", fontsize=10, pad=8)
    ax.text(
        0.01,
        0.03,
        "Unlike the seed, competitors have no known starting branch.",
        transform=ax.transAxes,
        fontsize=8.5,
        color=GRAY,
        ha="left",
        va="bottom",
    )
    ax.text(
        0.99,
        0.98,
        "Data-backed schematic:\ncounts from saved random-start searches",
        transform=ax.transAxes,
        fontsize=7.5,
        color=MID_GRAY,
        ha="right",
        va="top",
    )


def panel_validate(ax: plt.Axes, robust_row: dict[str, Any]) -> None:
    add_panel_label(ax, "2. Validate candidate roots")
    sweep = robust_row["competitor_sweep"]
    n_num = int(sweep["num_numerical_roots"])
    n_cluster = int(sweep["num_clustered_roots"])
    n_cert = int(sweep["num_certified_roots"])
    n_reject = max(n_cluster - n_cert, 0)

    cols = [0.15, 0.5, 0.84]
    labels = [
        f"Newton candidates\n{n_num}",
        f"Distinct clusters\n{n_cluster}",
        f"Krawczyk certified\n{n_cert}",
    ]
    colors = [LIGHT_GRAY, "#edf2f7", "#f0fff4"]
    edges = [MID_GRAY, BLACK, GREEN]

    rng = np.random.default_rng(7)
    for x, lab, fc, ec in zip(cols, labels, colors, edges, strict=True):
        box = mpatches.FancyBboxPatch(
            (x - 0.11, 0.18),
            0.22,
            0.64,
            boxstyle="round,pad=0.015,rounding_size=0.02",
            fc=fc,
            ec=ec,
            lw=1.4,
        )
        ax.add_patch(box)
        ax.text(x, 0.86, lab, ha="center", va="bottom", fontsize=9, color=BLACK)

    y_num = np.linspace(0.24, 0.76, num=min(n_num, 12))
    for y in y_num:
        ax.scatter(cols[0] + rng.normal(0.0, 0.018), y + rng.normal(0.0, 0.015), s=28, color="#a0aec0", zorder=3)

    y_cluster = np.linspace(0.24, 0.76, num=min(max(n_cluster, 4), 10))
    for y in y_cluster:
        ax.scatter(cols[1] + rng.normal(0.0, 0.013), y, s=34, color="white", edgecolor=BLACK, lw=0.9, zorder=3)

    y_cert = np.linspace(0.24, 0.76, num=min(max(n_cert, 4), 10))
    for y in y_cert:
        ax.scatter(cols[2], y, marker="s", s=72, facecolor="#c6f6d5", edgecolor=GREEN, lw=1.2, zorder=4)

    for y in y_cluster[-min(n_reject, len(y_cluster)) :] if n_reject else []:
        ax.scatter(cols[1] + 0.065, y, marker="x", s=70, color=RED, lw=1.8, zorder=5)

    for x0, x1 in zip(cols[:-1], cols[1:], strict=True):
        ax.annotate(
            "",
            xy=(x1 - 0.13, 0.5),
            xytext=(x0 + 0.13, 0.5),
            arrowprops=dict(arrowstyle="->", color=MID_GRAY, lw=1.4),
        )

    ax.text(
        0.03,
        0.04,
        "Newton discovers candidates; Krawczyk proves they exist.",
        transform=ax.transAxes,
        fontsize=8.7,
        color=GRAY,
        ha="left",
        va="bottom",
    )
    ax.text(
        0.98,
        0.97,
        fr"Representative saved slice: $\gamma={float(robust_row['gamma']):.3f}$",
        transform=ax.transAxes,
        fontsize=7.8,
        color=MID_GRAY,
        ha="right",
        va="top",
    )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.08, 0.98)
    ax.axis("off")


def panel_identify(ax: plt.Axes, seed: dict[str, np.ndarray], pair: dict[str, np.ndarray]) -> None:
    add_panel_label(ax, "3. Identify relevant competitors")
    all_pts = load_all_candidate_points(seed)
    delta = all_pts[:, 1]
    gap = all_pts[:, 2]

    pair_good = (pair["delta_re"] >= -5e-3) & (np.abs(pair["pair_exp"] - pair["exact"]) <= 0.02)
    other_good = (delta >= -5e-3) & (gap <= 0.02)
    decoy = (delta > 0.25) & (gap > 0.1)
    other = (~other_good) & (~decoy)

    ax.scatter(delta[other], gap[other], s=18, color="#cbd5e0", alpha=0.6, label="other certified candidates")
    ax.scatter(delta[decoy], gap[decoy], s=26, color=RED, alpha=0.75, label="larger-action decoys")
    ax.scatter(delta[other_good], gap[other_good], s=30, facecolor="#d9f99d", edgecolor=GREEN, lw=0.5, alpha=0.9)
    ax.scatter(
        pair["delta_re"][pair_good],
        np.abs(pair["pair_exp"][pair_good] - pair["exact"][pair_good]),
        s=55,
        facecolor=ORANGE,
        edgecolor=BLACK,
        lw=0.7,
        label="selected 43/46 pair",
        zorder=5,
    )
    ax.axvline(0.0, color=BLACK, lw=1.0)
    ax.axhline(0.02, color=GREEN, lw=1.2, ls="--")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\mathrm{Re}\,\Phi_{\mathrm{comp}} - \mathrm{Re}\,\Phi_{\mathrm{seed}}$")
    ax.set_ylabel(r"$|\lambda_{\mathrm{comp}} - \lambda_{\mathrm{exact}}(n)|$")
    ax.set_title("Larger real action is not enough; relevance also needs exact-benchmark agreement", fontsize=10, pad=8)
    ax.grid(True, alpha=0.18)
    ax.legend(loc="upper right", fontsize=7.8, frameon=False)

    ax.text(
        0.03,
        0.965,
        "Larger real action and agreement with the exact finite-n benchmark",
        transform=ax.transAxes,
        fontsize=8.8,
        color=BLACK,
        ha="left",
        va="top",
        bbox=dict(fc="white", ec="#cbd5e0", alpha=0.92),
    )
    ax.text(
        0.98,
        0.06,
        "This filters candidates; it does not prove contour contribution.",
        transform=ax.transAxes,
        fontsize=7.8,
        color=GRAY,
        ha="right",
        va="bottom",
    )


def panel_continue(ax: plt.Axes) -> None:
    add_panel_label(ax, "4. Continue the selected branch")
    resolved = load_json(
        RESULTS
        / "run_seed_branch_g-2pi"
        / "archive"
        / "competitor_dominance"
        / "z_robust_full"
        / "resolved_robust_z.json"
    )

    branch_data: dict[int, dict[str, np.ndarray]] = {}
    for bid in (43, 46):
        br = next(b for b in resolved["branches"] if int(b["branch_id"]) == bid)
        rows = [
            p for p in br["points"] if p.get("krawczyk_certified", True) and p.get("box_disjoint_from_seed", True)
        ]
        g = np.array([float(p["gamma"]) for p in rows])
        re = np.array([float(p["Phi_eff_real"]) for p in rows])
        order = np.argsort(g)
        branch_data[bid] = {"gamma": g[order], "re": re[order]}

    ax.plot(branch_data[43]["gamma"], branch_data[43]["re"], color=ORANGE, lw=2.2, label="branch 43")
    ax.plot(branch_data[46]["gamma"], branch_data[46]["re"], color=PURPLE, lw=2.0, ls="--", label="branch 46")
    ax.scatter(branch_data[43]["gamma"], branch_data[43]["re"], color=ORANGE, s=10)
    ax.scatter(branch_data[46]["gamma"], branch_data[46]["re"], color=PURPLE, s=10)

    for bid, color in ((43, ORANGE), (46, PURPLE)):
        g = branch_data[bid]["gamma"]
        re = branch_data[bid]["re"]
        for idx in (35, 80, 125):
            if idx + 1 < len(g):
                ax.annotate(
                    "",
                    xy=(g[idx + 1], re[idx + 1]),
                    xytext=(g[idx], re[idx]),
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.2, alpha=0.9),
                )

    ax.axvline(PRIMARY_GAMMA, color=GRAY, lw=1.2, ls=":")
    ax.set_xlabel(r"$\gamma$")
    ax.set_ylabel(r"$\mathrm{Re}\,\Phi$")
    ax.set_title("Use each certified root to initialize and certify the next $\\gamma$-value", fontsize=10, pad=8)
    ax.grid(True, alpha=0.18)
    ax.legend(loc="lower left", fontsize=8, frameon=False)

    ax.text(
        0.04,
        0.93,
        r"$z_{\mathrm{comp}}(\gamma_i)\rightarrow z_{\mathrm{comp}}(\gamma_{i+1})\rightarrow\cdots$",
        transform=ax.transAxes,
        fontsize=9,
        color=BLACK,
        ha="left",
        va="top",
    )
    ax.text(
        0.04,
        0.05,
        r"Outcome: relevant competitor = conjugate saddle pair $z_c,\overline{z_c}$"
        "\n"
        r"with $\mathrm{Re}\,\Phi(z_c)=\mathrm{Re}\,\Phi(\overline{z_c})$.",
        transform=ax.transAxes,
        fontsize=8.8,
        color=BLACK,
        ha="left",
        va="bottom",
        bbox=dict(fc="white", ec="#cbd5e0", alpha=0.92),
    )


def make_figure() -> dict[str, str]:
    dominance = load_json(
        RESULTS
        / "run_seed_branch_g-2pi"
        / "archive"
        / "competitor_dominance"
        / "competitor_dominance_summary.json"
    )
    seed = load_seed()
    pair = load_pair_series()
    robust_row = choose_validation_row()

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 10,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8,
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(14.5, 10.8))
    panel_explore(axes[0, 0], dominance)
    panel_validate(axes[0, 1], robust_row)
    panel_identify(axes[1, 0], seed, pair)
    panel_continue(axes[1, 1])

    fig.suptitle("From competitor search to certified conjugate saddle pair", fontsize=16, fontweight="bold", y=0.995)
    footer = (
        "Panels 1 and 2 use data-backed schematics from saved random-start budgets and saved candidate/certification "
        "counts because the raw Newton trajectories and rejected candidate coordinates were not retained. Panels 3 and 4 "
        "use saved certified-root and branch-resolved continuation data."
    )
    fig.text(0.01, 0.005, footer, ha="left", va="bottom", fontsize=8, color=GRAY, wrap=True)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    return savefig(fig, "four_stage_competitor_workflow")


def main() -> None:
    paths = make_figure()
    summary = {
        "figure": paths,
        "sources": {
            "competitor_dominance_summary": str(
                RESULTS / "run_seed_branch_g-2pi" / "archive" / "competitor_dominance" / "competitor_dominance_summary.json"
            ),
            "competitors_robust_z": str(
                RESULTS / "run_seed_branch_g-2pi" / "archive" / "competitor_dominance" / "z_robust_full" / "competitors_robust_z.json"
            ),
            "resolved_robust_z": str(
                RESULTS / "run_seed_branch_g-2pi" / "archive" / "competitor_dominance" / "z_robust_full" / "resolved_robust_z.json"
            ),
            "seed_branch_continuation": str(RESULTS / "run_seed_branch_g-2pi" / "seed_branch_continuation.json"),
        },
    }
    out = OUT / "four_stage_competitor_workflow_summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
