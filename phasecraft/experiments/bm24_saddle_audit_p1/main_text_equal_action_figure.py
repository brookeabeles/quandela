"""Create a simplified main-text equal-action boundary figure.

This figure intentionally leaves the detailed residual heatmap and branch
diagnostics in the appendix-style two-panel plot.  It shows the resolved
equal-action boundary, the beta=0.60 partial branch-resolved candidate, and
all numerical phase-alignment candidates.
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-phasecraft")

import matplotlib

matplotlib.use("Agg")
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
REPO_PARENT = HERE.parents[2]
if str(REPO_PARENT) not in sys.path:
    sys.path.insert(0, str(REPO_PARENT))

from phasecraft.bm24_saddle_audit_p1.refined_transition_scan import BETA_OPT


OUT = HERE / "results" / "physically_possible_refined"
PL_ALL = HERE / "results" / "pl_homotopy_all_slices"

PURPLE = "#6b46c1"
PHASE_GRAY = "#c2cad5"
GRID = "#d8dee8"
TEXT = "#1a202c"


def load_json(path: Path) -> Any:
    with path.open() as f:
        return json.load(f)


def zero_root_candidates(xs: np.ndarray, ys: np.ndarray) -> list[dict[str, float]]:
    roots: list[dict[str, float]] = []
    finite = np.isfinite(xs) & np.isfinite(ys)
    xs = xs[finite]
    ys = ys[finite]
    if len(xs) < 2:
        return roots
    order = np.argsort(xs)
    xs = xs[order]
    ys = ys[order]
    for x0, x1, y0, y1 in zip(xs[:-1], xs[1:], ys[:-1], ys[1:]):
        if y0 == 0.0:
            roots.append({"root": float(x0), "lo": float(x0), "hi": float(x0)})
        if y0 == 0.0 or y1 == 0.0 or y0 * y1 > 0.0:
            continue
        root = float(x0 - y0 * (x1 - x0) / (y1 - y0))
        roots.append({"root": root, "lo": float(min(x0, x1)), "hi": float(max(x0, x1))})
    if ys[-1] == 0.0:
        roots.append({"root": float(xs[-1]), "lo": float(xs[-1]), "hi": float(xs[-1])})
    dedup: dict[float, dict[str, float]] = {}
    for root in roots:
        dedup.setdefault(round(float(root["root"]), 10), root)
    return [dedup[key] for key in sorted(dedup)]


def blockwise_unwrap(raw_values: np.ndarray) -> np.ndarray:
    raw = np.asarray(raw_values, dtype=float)
    unwrapped = np.full_like(raw, np.nan)
    valid = np.flatnonzero(np.isfinite(raw))
    blocks = np.split(valid, np.where(np.diff(valid) > 1)[0] + 1)
    for block in blocks:
        if len(block):
            unwrapped[block] = np.unwrap(raw[block])
    return unwrapped


def phase_alignment_candidates() -> tuple[np.ndarray, np.ndarray]:
    path = PL_ALL / "pl_homotopy_all_slices.json"
    if not path.exists():
        return np.array([]), np.array([])
    data = load_json(path)
    betas: list[float] = []
    gammas: list[float] = []
    for result in data.get("results", []):
        beta = float(result.get("beta", math.nan))
        rows = [
            row for row in result.get("rows", [])
            if not row.get("competitor_failed")
        ]
        if not math.isfinite(beta) or not rows:
            continue
        xs = np.array([float(row["Gamma"]) for row in rows], dtype=float)
        raw_delta_im = np.array([float(row.get("delta_im_phi_raw", math.nan)) for row in rows], dtype=float)
        delta_im = blockwise_unwrap(raw_delta_im)
        for root in zero_root_candidates(xs, delta_im):
            betas.append(beta)
            gammas.append(float(root["root"]))
    return np.asarray(betas), np.asarray(gammas)


def resolved_equal_action_points() -> tuple[np.ndarray, np.ndarray]:
    data = load_json(OUT / "equal_real_action_refined_brackets.json")
    rows = [
        row for row in data.get("rows", [])
        if row.get("status") == "ok"
        and row.get("Gamma_lo") is not None
        and row.get("Gamma_hi") is not None
        and not math.isclose(float(row.get("beta", math.nan)), 0.60, abs_tol=1e-9)
    ]
    rows.sort(key=lambda row: float(row["beta"]))
    return (
        np.asarray([float(row["beta"]) for row in rows], dtype=float),
        np.asarray([float(row["midpoint"]) for row in rows], dtype=float),
    )


def beta060_partial_midpoint() -> float:
    data = load_json(OUT / "beta060_equal_action_roots.json")
    roots = data.get("roots", [])
    lows = [float(root["Gamma_lo"]) for root in roots if root.get("Gamma_lo") is not None]
    highs = [float(root["Gamma_hi"]) for root in roots if root.get("Gamma_hi") is not None]
    if not lows or not highs:
        return float("nan")
    overlap_lo = max(lows)
    overlap_hi = min(highs)
    if overlap_lo <= overlap_hi:
        return 0.5 * (overlap_lo + overlap_hi)
    mids = [float(root["midpoint"]) for root in roots if root.get("midpoint") is not None]
    return float(np.mean(mids)) if mids else float("nan")


def write_caption() -> Path:
    path = OUT / "main_text_equal_action_boundary_caption.txt"
    path.write_text(
        "Equal-action boundary and phase-alignment candidates. Filled purple circles mark "
        "resolved equal-action brackets, with thin solid guides connecting only resolved "
        "neighboring slices. The open purple diamond at beta=0.60 is a partial branch-resolved "
        "candidate placed at the midpoint of the overlapping partial brackets near Gamma=1.859; "
        "dotted guides indicate that it follows the surrounding trend but has weaker numerical "
        "status because the same competitor branch is not certified through the slice. Pale gray "
        "squares show numerical phase-alignment candidates, which are supporting PL necessary-condition "
        "diagnostics only.\n"
    )
    return path


def plot() -> tuple[Path, Path, Path]:
    beta, gamma = resolved_equal_action_points()
    phase_beta, phase_gamma = phase_alignment_candidates()
    partial_gamma = beta060_partial_midpoint()

    fig, ax = plt.subplots(figsize=(7.4, 4.8))

    if len(phase_beta):
        ax.scatter(
            phase_beta,
            phase_gamma,
            marker="s",
            s=12,
            color=PHASE_GRAY,
            alpha=0.48,
            label="phase-alignment candidates",
            zorder=1,
        )

    left = beta < 0.60
    right = beta > 0.60
    for mask in [left, right]:
        if int(np.sum(mask)) >= 2:
            order = np.argsort(beta[mask])
            ax.plot(beta[mask][order], gamma[mask][order], color=PURPLE, lw=1.25, zorder=3)
    ax.scatter(
        beta,
        gamma,
        marker="o",
        s=42,
        color=PURPLE,
        edgecolor="white",
        linewidth=0.7,
        label="equal-action boundary",
        zorder=4,
    )

    if math.isfinite(partial_gamma):
        y_left = gamma[np.where(np.isclose(beta, 0.575))[0][0]]
        y_right = gamma[np.where(np.isclose(beta, 0.625))[0][0]]
        ax.plot([0.575, 0.60], [y_left, partial_gamma], color=PURPLE, lw=1.15, ls=":", zorder=2)
        ax.plot([0.60, 0.625], [partial_gamma, y_right], color=PURPLE, lw=1.15, ls=":", zorder=2)
        ax.scatter(
            [0.60],
            [partial_gamma],
            marker="D",
            s=58,
            facecolor="white",
            edgecolor=PURPLE,
            linewidth=1.5,
            label="partial branch-resolved candidate",
            zorder=5,
        )

    ax.axvline(BETA_OPT, color="#4a5568", ls="--", lw=1.0, zorder=0)
    ax.text(
        BETA_OPT + 0.004,
        2.082,
        r"$\beta_{\mathrm{opt}}$",
        color="#4a5568",
        fontsize=10,
        va="top",
    )

    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel(r"$\Gamma \equiv -\gamma$")
    ax.set_xlim(0.33, 0.67)
    ax.set_ylim(1.68, 2.10)
    ax.grid(True, color=GRID, linewidth=0.7, alpha=0.65)

    handles = [
        mlines.Line2D([], [], color=PURPLE, marker="o", lw=1.25, markersize=5.5, label="equal-action boundary"),
        mlines.Line2D([], [], color=PURPLE, marker="D", markerfacecolor="white", lw=0, markersize=6.0, label="partial candidate"),
        mlines.Line2D([], [], color=PHASE_GRAY, marker="s", lw=0, markersize=4.5, alpha=0.55, label="phase-alignment candidates"),
    ]
    ax.legend(handles=handles, loc="lower right", frameon=True, fontsize=9)

    fig.tight_layout()
    pdf = OUT / "main_text_equal_action_boundary.pdf"
    png = OUT / "main_text_equal_action_boundary.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    caption = write_caption()
    return pdf, png, caption


def main() -> None:
    pdf, png, caption = plot()
    print(pdf)
    print(png)
    print(caption)


if __name__ == "__main__":
    main()
