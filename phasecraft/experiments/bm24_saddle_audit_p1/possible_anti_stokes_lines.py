"""Find candidate anti-Stokes lines after the physically possible exponent filter.

This script uses the refined audit that excludes positive standalone exponents
(`E > 0`).  It finds anti-Stokes candidates for the physically feasible
competitor envelope:

    Delta E(beta, Gamma) = E_max_possible_competitor - E_seed.

Zeros of this envelope are necessary algebraic anti-Stokes candidates.  They
are not, by themselves, physical dominance proofs; exact exponent matching and
PL/Stokes evidence are still needed to decide which candidates matter.
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
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
POSSIBLE = RESULTS / "physically_possible_refined"
PL_ALL = RESULTS / "pl_homotopy_all_slices"
OUT = RESULTS / "possible_anti_stokes"

BETA_OPT = 0.5433996420760803
PURPLE = "#6b46c1"
ORANGE = "#dd6b20"
GREEN = "#2f855a"
RED = "#c53030"
GRAY = "#4a5568"


def load_json(path: Path) -> Any:
    with path.open() as f:
        return json.load(f)


def pcolor_edges(vals: np.ndarray) -> np.ndarray:
    if len(vals) == 1:
        return np.array([vals[0] - 0.5, vals[0] + 0.5])
    mids = 0.5 * (vals[:-1] + vals[1:])
    return np.concatenate(
        [
            [vals[0] - 0.5 * (vals[1] - vals[0])],
            mids,
            [vals[-1] + 0.5 * (vals[-1] - vals[-2])],
        ]
    )


def load_grid() -> dict[str, Any]:
    data = load_json(POSSIBLE / "physically_possible_refined.json")
    rows = [r for r in data["rows"] if r.get("status") == "ok"]
    betas = np.array(sorted({float(r["beta"]) for r in rows}))
    Gammas = np.array(sorted({-float(r["gamma"]) for r in rows}))
    delta = np.full((len(Gammas), len(betas)), np.nan)
    best_gap = np.full_like(delta, np.nan)
    n_possible = np.full_like(delta, np.nan)
    n_impossible = np.full_like(delta, np.nan)
    possible_winner = np.full_like(delta, np.nan)
    for r in rows:
        i = int(np.where(np.isclose(Gammas, -float(r["gamma"])))[0][0])
        j = int(np.where(np.isclose(betas, float(r["beta"])))[0][0])
        delta[i, j] = float(r.get("max_possible_competitor_delta", np.nan))
        if bool(r.get("seed_is_best_match_possible")) or math.isnan(float(r.get("best_possible_match_abs_gap", np.nan))):
            best_gap[i, j] = abs(float(r.get("seed_gap", np.nan)))
            possible_winner[i, j] = 0.0
        else:
            best_gap[i, j] = float(r["best_possible_match_abs_gap"])
            possible_winner[i, j] = 1.0
        n_possible[i, j] = float(r.get("n_possible_competitors", 0))
        n_impossible[i, j] = float(r.get("n_impossible_competitors", 0))
    return {
        "metadata": data["metadata"],
        "rows": rows,
        "betas": betas,
        "Gammas": Gammas,
        "delta": delta,
        "best_gap": best_gap,
        "n_possible": n_possible,
        "n_impossible": n_impossible,
        "possible_winner": possible_winner,
    }


def zero_crossings_1d(xs: np.ndarray, ys: np.ndarray) -> list[dict[str, float]]:
    out: list[dict[str, float]] = []
    for idx, (x0, x1, y0, y1) in enumerate(zip(xs[:-1], xs[1:], ys[:-1], ys[1:])):
        if not (math.isfinite(float(y0)) and math.isfinite(float(y1))):
            continue
        if y0 == 0.0:
            out.append({"index_low": idx, "x_low": float(x0), "x_high": float(x0), "x_mid": float(x0), "y_low": float(y0), "y_high": float(y0)})
        if y0 * y1 < 0.0:
            t = abs(y0) / (abs(y0) + abs(y1))
            xz = float(x0 + t * (x1 - x0))
            out.append({"index_low": idx, "x_low": float(x0), "x_high": float(x1), "x_mid": xz, "y_low": float(y0), "y_high": float(y1)})
        elif y1 == 0.0 and idx == len(xs) - 2:
            out.append({"index_low": idx, "x_low": float(x1), "x_high": float(x1), "x_mid": float(x1), "y_low": float(y1), "y_high": float(y1)})
    return out


def find_vertical_lines(grid: dict[str, Any]) -> list[dict[str, Any]]:
    betas = grid["betas"]
    Gammas = grid["Gammas"]
    delta = grid["delta"]
    out: list[dict[str, Any]] = []
    for j, beta in enumerate(betas):
        for z in zero_crossings_1d(Gammas, delta[:, j]):
            out.append(
                {
                    "orientation": "fixed_beta",
                    "beta": float(beta),
                    "Gamma_low": z["x_low"],
                    "Gamma_high": z["x_high"],
                    "Gamma_est": z["x_mid"],
                    "delta_low": z["y_low"],
                    "delta_high": z["y_high"],
                }
            )
    return out


def find_horizontal_lines(grid: dict[str, Any]) -> list[dict[str, Any]]:
    betas = grid["betas"]
    Gammas = grid["Gammas"]
    delta = grid["delta"]
    out: list[dict[str, Any]] = []
    for i, Gamma in enumerate(Gammas):
        for z in zero_crossings_1d(betas, delta[i, :]):
            out.append(
                {
                    "orientation": "fixed_Gamma",
                    "Gamma": float(Gamma),
                    "beta_low": z["x_low"],
                    "beta_high": z["x_high"],
                    "beta_est": z["x_mid"],
                    "delta_low": z["y_low"],
                    "delta_high": z["y_high"],
                }
            )
    return out


def contour_segments(grid: dict[str, Any]) -> list[dict[str, Any]]:
    fig, ax = plt.subplots()
    contour = ax.contour(grid["betas"], grid["Gammas"], grid["delta"], levels=[0.0])
    segments: list[dict[str, Any]] = []
    for level_segments in contour.allsegs:
        for seg in level_segments:
            if len(seg) < 2:
                continue
            segments.append(
                {
                    "n_vertices": int(len(seg)),
                    "beta_min": float(np.min(seg[:, 0])),
                    "beta_max": float(np.max(seg[:, 0])),
                    "Gamma_min": float(np.min(seg[:, 1])),
                    "Gamma_max": float(np.max(seg[:, 1])),
                    "vertices": [[float(x), float(y)] for x, y in seg],
                }
            )
    plt.close(fig)
    return segments


def possible_transition_points() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    summary = load_json(POSSIBLE / "physically_possible_refined_summary.json")
    bx, by, be, ax, ay = [], [], [], [], []
    for t in summary["transitions"]:
        trs = t.get("transitions", [])
        if len(trs) == 1 and trs[0]["from"] == "seed_or_decoy" and trs[0]["to"] == "possible_competitor":
            bx.append(float(t["beta"]))
            by.append(float(trs[0]["Gamma_mid"]))
            be.append(float(trs[0]["uncertainty"]))
        elif trs:
            ax.append(float(t["beta"]))
            ay.append(float(trs[0]["Gamma_mid"]))
    return np.array(bx), np.array(by), np.array(be), np.array(ax), np.array(ay)


def pl_summary() -> list[dict[str, Any]]:
    path = PL_ALL / "pl_homotopy_all_slices.json"
    if path.exists():
        return load_json(path)["summaries"]
    return []


def plot_envelope(grid: dict[str, Any], segments: list[dict[str, Any]]) -> str:
    OUT.mkdir(parents=True, exist_ok=True)
    bx, by, be, axx, axy = possible_transition_points()
    fig, ax = plt.subplots(figsize=(10.8, 6.2))
    vmax = float(np.nanpercentile(np.abs(grid["delta"]), 92))
    vmax = max(vmax, 0.1)
    mesh = ax.pcolormesh(
        pcolor_edges(grid["betas"]),
        pcolor_edges(grid["Gammas"]),
        grid["delta"],
        cmap="coolwarm",
        vmin=-vmax,
        vmax=vmax,
        shading="flat",
    )
    fig.colorbar(mesh, ax=ax, label=r"$E_{\max,\mathrm{possible}}-E_\mathrm{seed}$")
    ax.contour(grid["betas"], grid["Gammas"], grid["delta"], levels=[0.0], colors="black", linewidths=2.0)
    ax.plot([], [], color="black", lw=2.0, label="possible anti-Stokes envelope")
    if len(bx):
        ax.errorbar(bx, by, yerr=be, fmt="o-", color=PURPLE, lw=1.4, ms=4, capsize=3, label="exact-match switch")
    if len(axx):
        ax.scatter(axx, axy, marker="x", color=RED, s=65, linewidths=2.0, label="nonmonotone exact slice")
    for s in pl_summary():
        if s.get("status") != "ok":
            continue
        beta = float(s["beta"])
        for lo, hi in s.get("anti_stokes_intervals", []):
            ax.plot([beta, beta], [lo, hi], color=ORANGE, lw=4.2, alpha=0.75, solid_capstyle="round")
    ax.plot([], [], color=ORANGE, lw=4.2, alpha=0.75, label="same-branch PL anti-Stokes")
    ax.axvline(BETA_OPT, color=GRAY, ls="--", lw=1.0, label=r"$\beta_\mathrm{opt}$")
    ax.set_title("All candidate anti-Stokes lines from physically possible competitors")
    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel(r"$\Gamma=-\gamma$")
    ax.set_xlim(float(grid["betas"].min()) - 0.02, float(grid["betas"].max()) + 0.02)
    ax.set_ylim(float(grid["Gammas"].min()) - 0.02, float(grid["Gammas"].max()) + 0.02)
    ax.legend(loc="best", fontsize=8.5, frameon=True)
    fig.tight_layout()
    path = OUT / "possible_anti_stokes_envelope.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def plot_segments(grid: dict[str, Any], vertical: list[dict[str, Any]], horizontal: list[dict[str, Any]]) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.0), sharey=True)
    for ax in axes:
        ax.axvline(BETA_OPT, color=GRAY, ls="--", lw=1.0)
        ax.set_xlabel(r"$\beta$")
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel(r"$\Gamma=-\gamma$")
    for z in vertical:
        axes[0].plot([z["beta"], z["beta"]], [z["Gamma_low"], z["Gamma_high"]], color="black", lw=2.0)
        axes[0].scatter([z["beta"]], [z["Gamma_est"]], color=PURPLE, s=26)
    axes[0].set_title("Fixed-beta anti-Stokes brackets")
    for z in horizontal:
        axes[1].plot([z["beta_low"], z["beta_high"]], [z["Gamma"], z["Gamma"]], color="black", lw=1.6)
        axes[1].scatter([z["beta_est"]], [z["Gamma"]], color=PURPLE, s=20)
    axes[1].set_title("Fixed-Gamma anti-Stokes brackets")
    fig.tight_layout()
    path = OUT / "possible_anti_stokes_brackets.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def write_report(vertical: list[dict[str, Any]], horizontal: list[dict[str, Any]], segments: list[dict[str, Any]], figures: list[str]) -> str:
    lines = [
        "# Possible anti-Stokes line audit",
        "",
        "This uses only certified competitors with nonpositive standalone exponent `E <= 0`.",
        "The zero condition is the physically feasible competitor envelope `E_max_possible - E_seed = 0`.",
        "These are candidate anti-Stokes loci, not physical dominance proofs by themselves.",
        "",
        "## Counts",
        "",
        f"- fixed-beta zero brackets: {len(vertical)}",
        f"- fixed-Gamma zero brackets: {len(horizontal)}",
        f"- interpolated 2D contour segments: {len(segments)}",
        "",
        "## Fixed-beta candidates",
        "",
        "| beta | Gamma bracket | Gamma linear estimate | delta endpoints |",
        "|---:|---:|---:|---:|",
    ]
    for z in vertical:
        lines.append(
            f"| {z['beta']:.6f} | {z['Gamma_low']:.4f}-{z['Gamma_high']:.4f} | "
            f"{z['Gamma_est']:.6f} | {z['delta_low']:+.3g}, {z['delta_high']:+.3g} |"
        )
    lines += [
        "",
        "## Fixed-Gamma candidates",
        "",
        "| Gamma | beta bracket | beta linear estimate | delta endpoints |",
        "|---:|---:|---:|---:|",
    ]
    for z in horizontal:
        lines.append(
            f"| {z['Gamma']:.4f} | {z['beta_low']:.6f}-{z['beta_high']:.6f} | "
            f"{z['beta_est']:.6f} | {z['delta_low']:+.3g}, {z['delta_high']:+.3g} |"
        )
    lines += ["", "## Contour segments", ""]
    for i, s in enumerate(segments, 1):
        lines.append(
            f"- segment {i}: beta {s['beta_min']:.6f}-{s['beta_max']:.6f}, "
            f"Gamma {s['Gamma_min']:.6f}-{s['Gamma_max']:.6f}, vertices={s['n_vertices']}"
        )
    lines += ["", "## Figures", ""]
    lines += [f"- `{Path(fig).name}`" for fig in figures]
    lines.append("")
    path = OUT / "possible_anti_stokes_report.md"
    path.write_text("\n".join(lines))
    return str(path)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    grid = load_grid()
    vertical = find_vertical_lines(grid)
    horizontal = find_horizontal_lines(grid)
    segments = contour_segments(grid)
    figures = [plot_envelope(grid, segments), plot_segments(grid, vertical, horizontal)]
    report = write_report(vertical, horizontal, segments, figures)
    out = {
        "metadata": {
            "source": str((POSSIBLE / "physically_possible_refined.json").resolve()),
            "criterion": "Delta E = E_max_possible_competitor - E_seed = 0, with E_comp <= 0",
            "caveat": "Envelope anti-Stokes candidates, not branch intersection proofs.",
        },
        "fixed_beta_candidates": vertical,
        "fixed_Gamma_candidates": horizontal,
        "contour_segments": segments,
        "figures": figures,
        "report": report,
    }
    out_path = OUT / "possible_anti_stokes_lines.json"
    with out_path.open("w") as f:
        json.dump(out, f, indent=2)
    print(out_path)
    print(report)
    for fig in figures:
        print(fig)


if __name__ == "__main__":
    main()
