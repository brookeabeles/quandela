"""Overlay PL homotopy diagnostics on the refined transition figures."""

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
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
REFINED = RESULTS / "refined_transition_scan"
PL_ALL = RESULTS / "pl_homotopy_all_slices"
OUT = RESULTS / "pl_overlay_refined"

BETA_OPT = 0.5433996420760803
BLUE = "#2b6cb0"
ORANGE = "#dd6b20"
GREEN = "#2f855a"
RED = "#c53030"
PURPLE = "#6b46c1"
GRAY = "#4a5568"


def load_json(path: Path) -> Any:
    with path.open() as f:
        return json.load(f)


def pcolor_edges(vals: np.ndarray) -> np.ndarray:
    if len(vals) == 1:
        return np.array([vals[0] - 0.5, vals[0] + 0.5])
    mids = 0.5 * (vals[:-1] + vals[1:])
    return np.concatenate([[vals[0] - 0.5 * (vals[1] - vals[0])], mids, [vals[-1] + 0.5 * (vals[-1] - vals[-2])]])


def classify(row: dict[str, Any], decoy_margin: float = 0.02) -> str:
    if row.get("status") != "ok":
        return "failed"
    if not bool(row["seed_is_best_match_to_exact"]):
        return "competitor"
    if float(row.get("max_competitor_delta_re", 0.0)) > decoy_margin:
        return "decoy"
    return "seed"


def refined_grid() -> dict[str, np.ndarray]:
    data = load_json(REFINED / "refined_transition_scan.json")
    rows = [r for r in data["rows"] if r.get("status") == "ok"]
    betas = np.array(sorted({float(r["beta"]) for r in rows}))
    Gammas = np.array(sorted({-float(r["gamma"]) for r in rows}))
    cls = np.full((len(Gammas), len(betas)), np.nan)
    code = {"seed": 0.0, "decoy": 1.0, "competitor": 2.0, "failed": np.nan}
    for r in rows:
        i = int(np.where(np.isclose(Gammas, -float(r["gamma"])))[0][0])
        j = int(np.where(np.isclose(betas, float(r["beta"])))[0][0])
        cls[i, j] = code[classify(r)]
    return {"betas": betas, "Gammas": Gammas, "class": cls}


def robust_points(summaries: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    bx, by, be, ax, ay = [], [], [], [], []
    for s in summaries:
        trs = s.get("transitions", [])
        if (
            len(trs) == 1
            and trs[0]["from"] == "seed_or_decoy"
            and trs[0]["to"] == "competitor"
        ):
            bx.append(float(s["beta"]))
            by.append(float(trs[0]["Gamma_mid"]))
            be.append(float(trs[0]["uncertainty"]))
        elif trs:
            ax.append(float(s["beta"]))
            ay.append(float(trs[0]["Gamma_mid"]))
    return np.array(bx), np.array(by), np.array(be), np.array(ax), np.array(ay)


def pl_summary() -> list[dict[str, Any]]:
    return load_json(PL_ALL / "pl_homotopy_all_slices.json")["summaries"]


def plot_map() -> str:
    OUT.mkdir(parents=True, exist_ok=True)
    grid = refined_grid()
    refined_summary = load_json(REFINED / "refined_transition_scan_summary.json")
    bx, by, be, ax, ay = robust_points(refined_summary["transitions"])
    pl = pl_summary()

    fig, ax0 = plt.subplots(figsize=(11.0, 6.3))
    cmap = mcolors.ListedColormap(["#cfe8ff", "#d8f0d2", "#ffd1a6"])
    norm = mcolors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)
    mesh = ax0.pcolormesh(
        pcolor_edges(grid["betas"]),
        pcolor_edges(grid["Gammas"]),
        grid["class"],
        cmap=cmap,
        norm=norm,
        shading="flat",
    )
    cbar = fig.colorbar(mesh, ax=ax0, ticks=[0, 1, 2])
    cbar.ax.set_yticklabels(["seed controls", "high-Re decoy", "competitor controls"])

    if len(bx):
        ax0.errorbar(bx, by, yerr=be, fmt="o-", color=PURPLE, ecolor="#718096", capsize=3, lw=1.6, ms=4, label="exact-switch boundary")
    if len(ax):
        ax0.scatter(ax, ay, marker="x", color=RED, s=65, linewidths=2.0, label="nonmonotone exact slice")

    for s in pl:
        beta = float(s["beta"])
        if s["status"] == "ok":
            for lo, hi in s.get("anti_stokes_intervals", []):
                ax0.plot([beta, beta], [lo, hi], color=ORANGE, lw=4.5, alpha=0.75, solid_capstyle="round")
            y = np.mean(s["physical_switch_intervals"][0]) if s.get("physical_switch_intervals") else float(s.get("anchor_Gamma", math.nan))
            marker = "o" if s.get("pl_aligned") else "x"
            color = GREEN if s.get("pl_aligned") else RED
            ax0.scatter(beta, y, marker=marker, color=color, s=80, edgecolor="white" if marker == "o" else color, linewidth=1.4, zorder=5)
        else:
            ax0.scatter(beta, float(s.get("anchor_Gamma", math.nan)), marker="x", color=RED, s=90, linewidths=2.4, zorder=5)

    ax0.plot([], [], color=ORANGE, lw=4.5, alpha=0.75, label="PL anti-Stokes bracket")
    ax0.scatter([], [], color=GREEN, s=80, edgecolor="white", label="PL aligned")
    ax0.axvline(BETA_OPT, color=GRAY, ls="--", lw=1.0, label=r"$\beta_\mathrm{opt}$")
    ax0.set_xlabel(r"$\beta$")
    ax0.set_ylabel(r"$\Gamma=-\gamma$")
    ax0.set_title("Refined physical saddle-control map with PL homotopy overlay")
    ax0.set_xlim(0.33, 0.67)
    ax0.set_ylim(1.43, 2.12)
    ax0.legend(loc="upper right", fontsize=9, frameon=True)
    fig.tight_layout()
    path = OUT / "refined_transition_map_with_pl.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def plot_boundary() -> str:
    OUT.mkdir(parents=True, exist_ok=True)
    refined_summary = load_json(REFINED / "refined_transition_scan_summary.json")
    bx, by, be, ax, ay = robust_points(refined_summary["transitions"])
    pl = pl_summary()

    fig, ax0 = plt.subplots(figsize=(9.4, 5.0))
    if len(bx):
        ax0.errorbar(bx, by, yerr=be, fmt="o", color=PURPLE, ecolor="#718096", capsize=3, ms=5, label="exact-switch bracket")
        ax0.plot(bx, by, color=PURPLE, lw=1.2, alpha=0.55)
    if len(ax):
        ax0.scatter(ax, ay, marker="x", color=RED, s=70, linewidths=2.0, label="nonmonotone exact slice")
    for s in pl:
        beta = float(s["beta"])
        if s["status"] != "ok":
            ax0.scatter(beta, float(s.get("anchor_Gamma", math.nan)), marker="x", color=RED, s=85, linewidths=2.2)
            continue
        for lo, hi in s.get("anti_stokes_intervals", []):
            ax0.plot([beta, beta], [lo, hi], color=ORANGE, lw=4.5, alpha=0.8, solid_capstyle="round")
        if s.get("pl_aligned"):
            y = np.mean(s["physical_switch_intervals"][0]) if s.get("physical_switch_intervals") else float(s["anchor_Gamma"])
            ax0.scatter(beta, y, color=GREEN, s=65, edgecolor="white", linewidth=1.1, zorder=5)
    ax0.plot([], [], color=ORANGE, lw=4.5, alpha=0.8, label="PL anti-Stokes bracket")
    ax0.scatter([], [], color=GREEN, s=65, edgecolor="white", label="PL aligned")
    ax0.axvline(BETA_OPT, color=GRAY, ls="--", lw=1.0)
    ax0.set_xlabel(r"$\beta$")
    ax0.set_ylabel(r"transition $\Gamma=-\gamma$")
    ax0.set_title("Transition boundary with same-branch PL evidence")
    ax0.grid(True, alpha=0.25)
    ax0.legend(loc="best", fontsize=9, frameon=True)
    fig.tight_layout()
    path = OUT / "refined_boundary_curve_with_pl.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def main() -> None:
    paths = [plot_map(), plot_boundary()]
    report = OUT / "pl_overlay_refined_report.md"
    report.write_text(
        "# PL overlays on refined transition figures\n\n"
        "Use the map as the main explanatory figure and the boundary curve as a compact quantitative summary.\n\n"
        + "\n".join(f"- `{Path(p).name}`" for p in paths)
        + "\n"
    )
    paths.append(str(report))
    for p in paths:
        print(p)


if __name__ == "__main__":
    main()
