"""High-resolution anti-Stokes and finite-size drift diagnostics.

This is the narrow "make it memorable" layer:

* locate the seed vs branch 43/46 anti-Stokes crossing from Delta Re Phi
* compute exact finite-n rates on a dense gamma window around that crossing
* estimate the finite-n apparent crossover gamma_cross(n)
* show how gamma_cross(n) drifts toward the anti-Stokes boundary

The script reuses certified branch/seed artifacts and does not run new root
searches.  Exact finite-n evaluations are cached because n=80 is nontrivial.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-phasecraft")

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
PACKAGE_ROOT = HERE.parents[2]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from phasecraft.bm24_saddle_audit_p1 import push_groundbreaking_plots as push
from phasecraft.bm24_saddle_audit_p1.audit import finite_n_exponent_grid


RESULTS = HERE / "results"
OUT = RESULTS / "anti_stokes_finite_size_push"
RAW = OUT / "dense_finite_n_window.json"

Q = 3
K_CLAUSE = 8
R = 176.54
BETA = 0.5433996420760803
PHI_PREF = -0.6896093750000001

BLUE = "#2b6cb0"
ORANGE = "#dd6b20"
GREEN = "#2f855a"
RED = "#c53030"
PURPLE = "#6b46c1"
GRAY = "#4a5568"

# Sparse exact-n curves shown on the rate-overlay figure (same color, distinct linestyles).
RATE_OVERLAY_EXACT_NS = [20, 100]
FULL_GAMMA_LO = -2.25
CROSSOVER_GAMMA_LO = -1.95
CROSSOVER_GAMMA_HI = -1.70


def crossover_gammas() -> list[float]:
    vals = (
        frange(CROSSOVER_GAMMA_LO, CROSSOVER_GAMMA_HI, 0.01)
        + frange(-1.875, -1.795, 0.0025)
        + [-1.830842334140663]
    )
    return sorted(set(round(float(v), 10) for v in vals))


def default_gammas() -> list[float]:
    vals = frange(FULL_GAMMA_LO, -1.96, 0.03) + crossover_gammas() + frange(-1.68, -0.05, 0.03)
    return sorted(set(round(float(v), 10) for v in vals), reverse=True)


def frange(start: float, stop: float, step: float) -> list[float]:
    vals = []
    x = start
    while x <= stop + 0.5 * step:
        vals.append(round(x, 10))
        x += step
    return vals


def default_n_values() -> list[int]:
    return [12, 16, 20, 24, 30, 36, 40]


def load_json(path: Path) -> Any:
    with path.open() as f:
        return json.load(f)


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2)


def gamma_key(gamma: float) -> str:
    return f"{gamma:.10f}"


def load_cache() -> dict[str, dict[str, Any]]:
    if not RAW.exists():
        return {}
    data = load_json(RAW)
    return {gamma_key(float(r["gamma"])): r for r in data.get("rows", [])}


def eval_exact_grid(gamma: float, n_values: list[int]) -> dict[str, Any]:
    fn = finite_n_exponent_grid(K_CLAUSE, Q, R, BETA, gamma, n_values)
    return {
        "gamma": float(gamma),
        "lambda_abs": {str(n): float(fn["lambda_abs"][str(n)]) for n in n_values},
    }


def anti_stokes_crossing(merged: dict[str, np.ndarray]) -> tuple[float, list[float]]:
    mask = (merged["gamma"] <= -1.70) & (merged["gamma"] >= -1.95)
    g = np.array(merged["gamma"][mask], dtype=float)
    d = np.array(merged["delta_re"][mask], dtype=float)
    order = np.argsort(g)
    g = g[order]
    d = d[order]
    crossings = []
    for i in range(len(g) - 1):
        if d[i] == 0.0:
            crossings.append(float(g[i]))
        elif d[i] * d[i + 1] < 0.0:
            t = -d[i] / (d[i + 1] - d[i])
            crossings.append(float(g[i] + t * (g[i + 1] - g[i])))
    primary = min(crossings, key=lambda x: abs(x + 1.83)) if crossings else float("nan")
    return primary, crossings


def build_dense_rows(gammas: list[float], n_values: list[int], resume: bool) -> list[dict[str, Any]]:
    cache = load_cache() if resume else {}
    rows = []
    for i, gamma in enumerate(gammas, start=1):
        key = gamma_key(gamma)
        cached = cache.get(key)
        if cached is not None and all(str(n) in cached.get("lambda_abs", {}) for n in n_values):
            rows.append(cached)
            print(f"{i:3d}/{len(gammas)} gamma={gamma:+.6f}: cached", flush=True)
            continue
        print(f"{i:3d}/{len(gammas)} gamma={gamma:+.6f}: exact finite-n", flush=True)
        row = eval_exact_grid(gamma, n_values)
        rows.append(row)
        merged = {gamma_key(float(r["gamma"])): r for r in rows}
        save_json(
            RAW,
            {
                "metadata": metadata(gammas, n_values),
                "rows": sorted(merged.values(), key=lambda r: float(r["gamma"])),
            },
        )
    rows_by_key = dict(cache) if resume else {}
    for r in rows:
        rows_by_key[gamma_key(float(r["gamma"]))] = r
    out = [rows_by_key[gamma_key(g)] for g in gammas]
    save_json(RAW, {"metadata": metadata(gammas, n_values), "rows": out})
    return out


def metadata(gammas: list[float], n_values: list[int]) -> dict[str, Any]:
    return {
        "q": Q,
        "K_clause": K_CLAUSE,
        "r": R,
        "beta": BETA,
        "phi_pref_conv2": PHI_PREF,
        "gamma_grid": gammas,
        "n_values": n_values,
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def interpolate_series(gamma: float, xs: np.ndarray, ys: np.ndarray) -> float:
    order = np.argsort(xs)
    return float(np.interp(gamma, xs[order], ys[order]))


def assemble(rows: list[dict[str, Any]], n_values: list[int]) -> dict[str, Any]:
    seed = push.load_seed_continuation()
    merged = push.load_merged_43_46_series(PHI_PREF)
    anti_gamma, anti_all = anti_stokes_crossing(merged)
    gammas = np.array([float(r["gamma"]) for r in rows], dtype=float)
    seed_exp = np.array([interpolate_series(g, seed["gamma"], seed["seed_exp"]) for g in gammas])
    pair_exp = np.array([interpolate_series(g, merged["gamma"], merged["merged_exp"]) for g in gammas])
    delta_re = np.array([interpolate_series(g, merged["gamma"], merged["delta_re"]) for g in gammas])
    im_mod = np.array([interpolate_series(g, merged["gamma"], merged["merged_im_mod_abs"]) for g in gammas])
    lambdas = {
        n: np.array([float(r["lambda_abs"][str(n)]) for r in rows], dtype=float)
        for n in n_values
    }
    return {
        "gamma": gammas,
        "seed_exp": seed_exp,
        "pair_exp": pair_exp,
        "delta_re": delta_re,
        "pair_im_mod_abs": im_mod,
        "lambda_abs": lambdas,
        "anti_stokes_gamma": anti_gamma,
        "anti_stokes_all_crossings": anti_all,
    }


def crossing_from_gapdiff(gamma: np.ndarray, diff: np.ndarray, anti_gamma: float) -> dict[str, Any]:
    """Crossing where seed gap minus pair gap changes from <0 to >0."""
    order = np.argsort(gamma)
    g = gamma[order]
    d = diff[order]
    candidates = []
    for i in range(len(g) - 1):
        if not (np.isfinite(d[i]) and np.isfinite(d[i + 1])):
            continue
        if d[i] == 0.0:
            candidates.append((float(g[i]), i))
        elif d[i] * d[i + 1] < 0.0:
            t = -d[i] / (d[i + 1] - d[i])
            candidates.append((float(g[i] + t * (g[i + 1] - g[i])), i))
    if not candidates:
        return {"gamma_cross": float("nan"), "bracket": None, "num_crossings": 0}
    gamma_cross, idx = min(candidates, key=lambda item: abs(item[0] - anti_gamma))
    return {
        "gamma_cross": gamma_cross,
        "bracket": [float(g[idx]), float(g[idx + 1])],
        "num_crossings": len(candidates),
        "all_crossings": [c[0] for c in candidates],
    }


def finite_size_crossings(data: dict[str, Any], n_values: list[int]) -> list[dict[str, Any]]:
    out = []
    gamma = data["gamma"]
    anti = float(data["anti_stokes_gamma"])
    for n in n_values:
        lam = data["lambda_abs"][n]
        seed_gap = np.abs(lam - data["seed_exp"])
        pair_gap = np.abs(lam - data["pair_exp"])
        diff = seed_gap - pair_gap
        cross = crossing_from_gapdiff(gamma, diff, anti)
        out.append(
            {
                "n": int(n),
                "gamma_cross": float(cross["gamma_cross"]),
                "bracket": cross["bracket"],
                "num_crossings": int(cross["num_crossings"]),
                "all_crossings": cross.get("all_crossings", []),
                "drift_from_anti_stokes": float(cross["gamma_cross"] - anti)
                if math.isfinite(float(cross["gamma_cross"]))
                else float("nan"),
                "min_seed_gap": float(np.min(seed_gap)),
                "min_pair_gap": float(np.min(pair_gap)),
            }
        )
    return out


def fit_drift(crossings: list[dict[str, Any]], anti_gamma: float) -> dict[str, Any]:
    usable = [c for c in crossings if math.isfinite(float(c["gamma_cross"]))]
    x = np.array([1.0 / float(c["n"]) for c in usable])
    y = np.array([float(c["gamma_cross"]) for c in usable])
    if len(usable) < 3:
        return {
            "linear_gamma_infinity": float("nan"),
            "quadratic_gamma_infinity": float("nan"),
            "anti_stokes_gamma": float(anti_gamma),
            "linear_offset_from_anti_stokes": float("nan"),
            "quadratic_offset_from_anti_stokes": float("nan"),
            "linear_slope": float("nan"),
            "quadratic_coefficients": [float("nan"), float("nan"), float("nan")],
        }
    lin = np.polyfit(x, y, 1)
    quad = np.polyfit(x, y, 2) if len(usable) >= 4 else [float("nan"), float("nan"), float("nan")]
    return {
        "linear_gamma_infinity": float(lin[1]),
        "linear_slope": float(lin[0]),
        "quadratic_gamma_infinity": float(quad[2]),
        "quadratic_coefficients": [float(v) for v in quad],
        "anti_stokes_gamma": float(anti_gamma),
        "linear_offset_from_anti_stokes": float(lin[1] - anti_gamma),
        "quadratic_offset_from_anti_stokes": float(quad[2] - anti_gamma),
    }


def plot_anti_stokes_window(data: dict[str, Any], n_values: list[int], crossings: list[dict[str, Any]]) -> str:
    gamma = data["gamma"]
    order = np.argsort(gamma)
    g = gamma[order]
    anti = float(data["anti_stokes_gamma"])

    fig, axes = plt.subplots(4, 1, figsize=(11, 12), sharex=True)

    axes[0].plot(g, data["delta_re"][order], color=RED, lw=2.0)
    axes[0].axhline(0, color="black", lw=0.8)
    axes[0].axvline(anti, color=PURPLE, ls="--", lw=1.5, label=f"anti-Stokes {anti:.6f}")
    axes[0].set_ylabel(r"$Re\Phi_{43/46}-Re\Phi_{seed}$")
    axes[0].set_title("Anti-Stokes transition: seed versus branch 43/46")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(g, data["pair_im_mod_abs"][order], color=PURPLE, lw=1.8)
    axes[1].axhline(0, color="black", lw=0.8)
    axes[1].set_ylabel(r"$|Im\Phi_{43/46}|$ mod $2\pi$")
    axes[1].grid(True, alpha=0.25)

    selected = [n for n in (12, 24, 40, 80) if n in n_values]
    for n in selected:
        lam = data["lambda_abs"][n]
        seed_gap = np.abs(lam - data["seed_exp"])
        pair_gap = np.abs(lam - data["pair_exp"])
        axes[2].plot(g, (seed_gap - pair_gap)[order], lw=1.5, label=f"n={n}")
    axes[2].axhline(0, color="black", lw=0.8)
    axes[2].axvline(anti, color=PURPLE, ls="--", lw=1.2)
    axes[2].set_ylabel("seed gap - pair gap")
    axes[2].legend(fontsize=8, ncol=2)
    axes[2].grid(True, alpha=0.25)

    winner = np.full((len(n_values), len(gamma)), np.nan)
    for i, n in enumerate(n_values):
        lam = data["lambda_abs"][n]
        winner[i, :] = np.where(np.abs(lam - data["seed_exp"]) <= np.abs(lam - data["pair_exp"]), 0.0, 1.0)
    cmap = mcolors.ListedColormap(["#cfe8ff", "#ffd1a6"])
    axes[3].imshow(
        winner[:, order],
        aspect="auto",
        interpolation="nearest",
        origin="lower",
        extent=[float(g[0]), float(g[-1]), min(n_values), max(n_values)],
        cmap=cmap,
        vmin=0,
        vmax=1,
    )
    for c in crossings:
        if math.isfinite(float(c["gamma_cross"])):
            axes[3].plot(float(c["gamma_cross"]), int(c["n"]), "o", color=PURPLE, ms=4)
    axes[3].axvline(anti, color=PURPLE, ls="--", lw=1.2)
    axes[3].set_ylabel("n")
    axes[3].set_xlabel(r"$\gamma$")
    axes[3].set_title("Finite-n winner: blue seed, orange 43/46 pair")

    fig.tight_layout()
    path = OUT / "anti_stokes_dense_window.png"
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def plot_finite_size_drift(crossings: list[dict[str, Any]], fit: dict[str, Any]) -> str:
    usable = [c for c in crossings if math.isfinite(float(c["gamma_cross"]))]
    n = np.array([int(c["n"]) for c in usable])
    y = np.array([float(c["drift_from_anti_stokes"]) for c in usable])
    anti = float(fit["anti_stokes_gamma"])

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.plot(n, y, "o-", color=ORANGE, lw=1.8, label="nearest-saddle crossover")
    ax.axhline(0.0, color=PURPLE, ls=":", lw=2.0, label=f"anti-Stokes {anti:.6f}")
    ax.set_xlabel("n")
    ax.set_ylabel(r"$\gamma_{cross}(n)-\gamma_{anti-Stokes}$")
    ax.set_title("Winner boundary is pinned to the anti-Stokes crossing")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = OUT / "finite_size_drift_to_anti_stokes.png"
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def plot_gamma_cross_vs_inverse_n(crossings: list[dict[str, Any]], fit: dict[str, Any]) -> str:
    usable = [c for c in crossings if math.isfinite(float(c["gamma_cross"]))]
    n = np.array([int(c["n"]) for c in usable], dtype=float)
    x = 1.0 / n
    y = np.array([float(c["gamma_cross"]) for c in usable], dtype=float)
    anti = float(fit["anti_stokes_gamma"])
    legacy = load_json(RESULTS / "push_further_analysis.json")["crossover_analysis"]
    legacy_points = [
        (12, float(legacy["n12_22_crossover_gamma"]), "legacy coarse n=12/22"),
        (22, float(legacy["n12_22_crossover_gamma"]), "legacy coarse n=12/22"),
        (40, float(legacy["n40_crossover_gamma"]), "legacy coarse n=40"),
    ]

    fig, ax = plt.subplots(figsize=(9.2, 5.8))
    ax.scatter(x, y, s=58, color=ORANGE, edgecolor="black", linewidth=0.5, zorder=4, label="certified 43/46 dense window")
    if len(x) >= 2:
        ax.plot(x, y, color=ORANGE, lw=1.4, alpha=0.85)
    for xi, yi, ni in zip(x, y, n):
        ax.text(xi, yi, f" {int(ni)}", fontsize=8, va="center")

    lx = np.array([1.0 / p[0] for p in legacy_points])
    ly = np.array([p[1] for p in legacy_points])
    ax.scatter(lx, ly, marker="x", s=90, color=RED, linewidths=2.0, label="legacy sparse apparent crossover")

    xx = np.linspace(0.0, max(x) * 1.05, 200)
    if math.isfinite(float(fit["linear_gamma_infinity"])):
        ax.plot(xx, fit["linear_slope"] * xx + fit["linear_gamma_infinity"], color=GREEN, lw=1.6, label="dense linear guide")
    ax.axhline(anti, color=PURPLE, ls=":", lw=2.0, label=f"anti-Stokes {anti:.6f}")
    ax.axvline(0.0, color=GRAY, lw=0.8)
    ax.set_xlabel(r"$1/n$")
    ax.set_ylabel(r"$\gamma_{cross}(n)$")
    ax.set_title("Finite-n saddle misidentification drifts to the asymptotic anti-Stokes crossing")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = OUT / "gamma_cross_vs_inverse_n_misidentification.png"
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def _auto_ylim(
    *series,
    pad_lo: float = 0.06,
    pad_hi: float = 0.05,
) -> tuple[float, float]:
    ys = np.concatenate([np.asarray(s, dtype=float).ravel() for s in series])
    ys = ys[np.isfinite(ys)]
    lo, hi = float(np.min(ys)), float(np.max(ys))
    span = max(hi - lo, 1e-3)
    return lo - pad_lo * span, hi + pad_hi * span


def _plot_rate_overlay(
    data: dict[str, Any],
    n_values: list[int],
    *,
    gamma_lo: float,
    gamma_hi: float,
    xlim: tuple[float, float],
    ylim: tuple[float, float] | None,
    filename_stem: str,
    figsize: tuple[float, float] = (7.0, 5.25),
    x_abs_gamma: bool = False,
) -> list[str]:
    plt.rcParams.update(
        {
            "font.size": 14,
            "axes.labelsize": 16,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 11,
        }
    )
    lw = 2.5
    gamma = np.asarray(data["gamma"], dtype=float)
    mask = (gamma >= gamma_lo) & (gamma <= gamma_hi)
    if x_abs_gamma:
        order = np.argsort(np.abs(gamma[mask]))
    else:
        order = np.argsort(gamma[mask])
    g = gamma[mask][order]
    x = np.abs(g) if x_abs_gamma else g
    seed_y = np.asarray(data["seed_exp"])[mask][order]
    pair_y = np.asarray(data["pair_exp"])[mask][order]
    exact_ns = [n for n in RATE_OVERLAY_EXACT_NS if n in n_values]
    exact_ys = [np.asarray(data["lambda_abs"][n])[mask][order] for n in exact_ns]

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(x, seed_y, color=BLUE, lw=lw, ls="--", label="seed exponent")
    ax.plot(x, pair_y, color=ORANGE, lw=lw, ls="-", label="merged 43/46 exponent")
    exact_linestyles = ["-", "--"]
    exact_lw = lw * 0.55 if x_abs_gamma else lw
    for i, (n, y) in enumerate(zip(exact_ns, exact_ys)):
        ls = exact_linestyles[i % len(exact_linestyles)]
        ax.plot(x, y, lw=exact_lw, color=GRAY, ls=ls, alpha=0.95, label=f"exact $n={n}$")

    anti = float(data["anti_stokes_gamma"])
    if gamma_lo <= anti <= gamma_hi:
        anti_x = abs(anti) if x_abs_gamma else anti
        ax.axvline(
            anti_x,
            color=RED,
            ls=":",
            lw=lw * 0.95,
            alpha=0.9,
            zorder=1,
            label="anti-Stokes crossing",
        )
        y_seed = interpolate_series(anti, gamma, data["seed_exp"])
        y_pair = interpolate_series(anti, gamma, data["pair_exp"])
        y_star = 0.5 * (y_seed + y_pair)
        ax.plot(
            anti_x,
            y_star,
            marker="*",
            markersize=16,
            color="gold",
            markeredgecolor="black",
            markeredgewidth=0.6,
            zorder=8,
            linestyle="none",
        )

    if ylim is None:
        ylim = _auto_ylim(seed_y, pair_y, *exact_ys)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel(r"$|\gamma|$" if x_abs_gamma else r"$\gamma$")
    ax.set_ylabel("Exponent" if x_abs_gamma else r"Rate exponent")
    ax.grid(True, alpha=0.2)
    if x_abs_gamma:
        ax.legend(
            loc="lower left",
            bbox_to_anchor=(0.0, 1.01, 1.0, 0.18),
            mode="expand",
            ncol=2,
            frameon=False,
            columnspacing=1.0,
            handletextpad=0.45,
            borderaxespad=0.0,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.86])
    else:
        ax.legend(
            loc="lower center",
            bbox_to_anchor=(0.5, 1.01),
            ncol=3,
            frameon=False,
            columnspacing=1.1,
            handletextpad=0.5,
            borderaxespad=0.0,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.90])

    paths: list[str] = []
    for suffix in (".png", ".pdf"):
        path = OUT / f"{filename_stem}{suffix}"
        fig.savefig(path, dpi=190 if suffix == ".png" else None, bbox_inches="tight")
        paths.append(str(path))
    plt.close(fig)
    return paths


def plot_rate_overlay_crossover(data: dict[str, Any], n_values: list[int]) -> list[str]:
    """Original crossover zoom: gamma in [-1.95, -1.70]."""
    return _plot_rate_overlay(
        data,
        n_values,
        gamma_lo=CROSSOVER_GAMMA_LO,
        gamma_hi=CROSSOVER_GAMMA_HI,
        xlim=(CROSSOVER_GAMMA_LO - 0.02, CROSSOVER_GAMMA_HI + 0.02),
        ylim=None,
        filename_stem="rate_overlay_seed_pair_exact_n",
        figsize=(6.8, 5.0),
    )


def plot_rate_overlay_full_gamma(data: dict[str, Any], n_values: list[int]) -> list[str]:
    """Extended view: |gamma| from 0 to the data edge (gamma from -2.25 up to 0)."""
    gamma = np.asarray(data["gamma"], dtype=float)
    mask = (gamma >= FULL_GAMMA_LO) & (gamma <= float(np.max(gamma)))
    x_hi = float(np.max(np.abs(gamma[mask]))) + 0.02
    return _plot_rate_overlay(
        data,
        n_values,
        gamma_lo=FULL_GAMMA_LO,
        gamma_hi=float(np.max(gamma)),
        xlim=(0.0, x_hi),
        ylim=None,
        filename_stem="rate_overlay_seed_pair_exact_n_full_gamma",
        figsize=(7.0, 5.25),
        x_abs_gamma=True,
    )


def publish_paper_results(paths: list[str]) -> list[str]:
    paper_dir = RESULTS / "PAPER-RESULTS"
    paper_dir.mkdir(parents=True, exist_ok=True)
    published: list[str] = []
    for src in paths:
        dst = paper_dir / Path(src).name
        shutil.copy2(src, dst)
        published.append(str(dst))
    return published


def residual_convergence(data: dict[str, Any], n_values: list[int]) -> list[dict[str, Any]]:
    probes = [
        ("seed_side", -1.8000000000),
        ("anti_stokes", float(data["anti_stokes_gamma"])),
        ("pair_side", -1.8500000000),
        ("deeper_pair_side", -1.9000000000),
    ]
    out = []
    for label, gamma in probes:
        seed_exp = interpolate_series(gamma, data["gamma"], data["seed_exp"])
        pair_exp = interpolate_series(gamma, data["gamma"], data["pair_exp"])
        envelope = max(seed_exp, pair_exp)
        controller = "seed" if seed_exp >= pair_exp else "43/46"
        residuals = []
        for n in n_values:
            lam = interpolate_series(gamma, data["gamma"], data["lambda_abs"][n])
            residuals.append(abs(lam - envelope))
        x = np.log(np.array(n_values, dtype=float))
        y = np.log(np.clip(np.array(residuals, dtype=float), 1e-15, None))
        slope, intercept = np.polyfit(x, y, 1)
        out.append(
            {
                "label": label,
                "gamma": float(gamma),
                "controller": controller,
                "seed_exp": float(seed_exp),
                "pair_exp": float(pair_exp),
                "envelope_exp": float(envelope),
                "residual_by_n": {str(n): float(r) for n, r in zip(n_values, residuals)},
                "power_law_slope_loglog": float(slope),
                "residual_ratio_nmax_to_nmin": float(residuals[-1] / residuals[0]),
            }
        )
    return out


def plot_residual_convergence(residuals: list[dict[str, Any]], n_values: list[int]) -> str:
    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    colors = [BLUE, PURPLE, ORANGE, RED]
    for row, color in zip(residuals, colors):
        vals = [float(row["residual_by_n"][str(n)]) for n in n_values]
        ax.loglog(n_values, vals, "o-", color=color, lw=1.8, label=f"{row['label']} ({row['controller']})")
    ax.set_xlabel("n")
    ax.set_ylabel(r"$|\lambda_n - max(E_{seed}, E_{43/46})|$")
    ax.set_title("Finite-size convergence to the controlling saddle envelope")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = OUT / "finite_size_residual_convergence.png"
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def write_report(
    rows: list[dict[str, Any]],
    data: dict[str, Any],
    crossings: list[dict[str, Any]],
    fit: dict[str, Any],
    residuals: list[dict[str, Any]],
    figures: list[str],
) -> str:
    pair = load_json(RESULTS / "push_groundbreaking" / "groundbreaking_push_summary.json")["branch_43_46_pair"]
    refined = load_json(RESULTS / "refined_transition_scan" / "refined_transition_scan_summary.json")
    report = OUT / "anti_stokes_finite_size_report.md"
    lines = [
        "# Anti-Stokes plus finite-size drift push",
        "",
        "## Headline",
        "",
        (
            "BM24's small-gamma seed saddle is not globally controlling. "
            "At beta=0.5433996421, the seed and the branch 43/46 conjugate pair meet at a sharply located "
            "anti-Stokes boundary where their real actions cross; finite-n exact rates drift toward this "
            "boundary as n grows."
        ),
        "",
        "## Anti-Stokes boundary",
        "",
        f"- Delta Re crossing selected near QAOA window: gamma = {float(data['anti_stokes_gamma']):.9f}",
        f"- all Delta Re zeroes in dense window: {[round(float(x), 9) for x in data['anti_stokes_all_crossings']]}",
        f"- branch 43/46 shared gamma count: {pair['shared_gamma_count']}",
        f"- median |Re43-Re46|: {pair['median_abs_re_difference']:.3g}",
        f"- median |Im43+Im46|: {pair['median_abs_im_sum']:.3g}",
        "",
        "## Finite-n crossing drift",
        "",
        (
            "The nearest-saddle winner boundary is essentially pinned to the anti-Stokes equality for all tested n; "
            "the finite-size effect is mainly the residual collapse of exact lambda_n toward the seed/43-46 envelope."
        ),
        (
            "The gamma_cross-vs-1/n figure overlays the older sparse apparent crossover points; those are useful "
            "as a cautionary example of how low-resolution finite-n evidence can suggest a premature transition "
            "near gamma=-1.5."
        ),
        "",
        "| n | gamma_cross(n) | bracket | drift from anti-Stokes | crossings in window |",
        "|---:|---:|---:|---:|---:|",
    ]
    for c in crossings:
        bracket = "-"
        if c["bracket"] is not None:
            bracket = f"{c['bracket'][0]:.6f} to {c['bracket'][1]:.6f}"
        lines.append(
            f"| {c['n']} | {c['gamma_cross']:.9f} | {bracket} | {c['drift_from_anti_stokes']:+.6f} | {c['num_crossings']} |"
        )
    lines += [
        "",
        f"- linear 1/n extrapolated gamma_infinity: {fit['linear_gamma_infinity']:.9f}",
        f"- linear extrapolation offset from anti-Stokes: {fit['linear_offset_from_anti_stokes']:+.6f}",
        f"- quadratic-guide gamma_infinity: {fit['quadratic_gamma_infinity']:.9f}",
        f"- quadratic-guide offset from anti-Stokes: {fit['quadratic_offset_from_anti_stokes']:+.6f}",
        "",
        "## Finite-size residual convergence",
        "",
        "| probe | gamma | controller | residual n_min | residual n_max | n_max/n_min ratio | log-log slope |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    n_keys = sorted(int(k) for k in residuals[0]["residual_by_n"].keys()) if residuals else []
    for row in residuals:
        n_min = n_keys[0]
        n_max = n_keys[-1]
        lines.append(
            f"| {row['label']} | {row['gamma']:.9f} | {row['controller']} | "
            f"{row['residual_by_n'][str(n_min)]:.6g} | {row['residual_by_n'][str(n_max)]:.6g} | "
            f"{row['residual_ratio_nmax_to_nmin']:.3f} | {row['power_law_slope_loglog']:.3f} |"
        )
    lines += [
        "",
        "## Decoy filter",
        "",
        (
            "The refined transition scan separates algebraic high-Re competitors from physical control: "
            f"{refined['counts']['decoy_challenge']} refined points have high-Re decoy pressure while the seed still "
            f"matches exact finite-n better; {refined['counts']['competitor_best']} points are true competitor-control."
        ),
        "",
        "## Files",
        "",
        f"- dense exact cache: `{RAW}`",
    ]
    lines += [f"- `{Path(fig).name}`" for fig in figures]
    lines.append("")
    report.write_text("\n".join(lines))
    return str(report)


def run(args: argparse.Namespace) -> None:
    n_values = args.n_values or default_n_values()
    gammas = args.gammas or default_gammas()
    n_values = sorted(set(int(n) for n in n_values))
    gammas = sorted(set(round(float(g), 10) for g in gammas))

    print(f"Dense finite-n drift: {len(gammas)} gamma values x {len(n_values)} n values")
    print(f"n values: {n_values}")
    rows = build_dense_rows(gammas, n_values, resume=args.resume)
    data = assemble(rows, n_values)
    crossings = finite_size_crossings(data, n_values)
    fit = fit_drift(crossings, float(data["anti_stokes_gamma"]))

    OUT.mkdir(parents=True, exist_ok=True)
    rate_crossover = plot_rate_overlay_crossover(data, n_values)
    rate_full = plot_rate_overlay_full_gamma(data, n_values)
    paper_copies = publish_paper_results([*rate_crossover, *rate_full])
    figures = [
        plot_anti_stokes_window(data, n_values, crossings),
        plot_finite_size_drift(crossings, fit),
        plot_gamma_cross_vs_inverse_n(crossings, fit),
        *rate_crossover,
        *rate_full,
        *paper_copies,
    ]
    residuals = residual_convergence(data, n_values)
    figures.append(plot_residual_convergence(residuals, n_values))
    summary = {
        "metadata": metadata(gammas, n_values),
        "anti_stokes_gamma": float(data["anti_stokes_gamma"]),
        "anti_stokes_all_crossings": [float(x) for x in data["anti_stokes_all_crossings"]],
        "finite_size_crossings": crossings,
        "drift_fit": fit,
        "residual_convergence": residuals,
        "figures": figures,
    }
    summary["report"] = write_report(rows, data, crossings, fit, residuals, figures)
    save_json(OUT / "anti_stokes_finite_size_summary.json", summary)
    print(json.dumps(summary, indent=2))


def parse_csv_floats(value: str) -> list[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def parse_csv_ints(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-resume", dest="resume", action="store_false", default=True)
    parser.add_argument("--gammas", type=parse_csv_floats, default=None, help="comma-separated gamma grid override")
    parser.add_argument("--n-values", type=parse_csv_ints, default=None, help="comma-separated finite n values")
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
