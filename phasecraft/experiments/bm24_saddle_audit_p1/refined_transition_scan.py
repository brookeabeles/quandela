"""Refined BM24 saddle dominance scan near the gamma transition band.

This script runs additional saddle searches on a denser local grid than the
coarse full-domain sweep.  It separates three notions that are easy to blur:

* seed best: the BM24 seed saddle is the closest exponent match to exact finite-n
* competitor best: a discovered competitor is the closest exponent match
* decoy challenge: a certified competitor has larger Re Phi but fails the
  finite-n exponent test, so it is not treated as physically dominant

The purpose is not to prove Picard-Lefschetz intersection numbers; it is to
make the empirical/certified evidence sharper and reproducible.
"""

from __future__ import annotations

import argparse
import json
import math
import os
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

from phasecraft.bm24_saddle_audit_p1 import sweep_beta_gamma as sweep
from phasecraft.bm24_saddle_audit_p1.audit import _x_to_z
from phasecraft.bm24_saddle_audit_p1.continue_seed_branch_certified import (
    certify_z,
    conv2_full_exponent,
    conv2_pref,
    lambda_abs_n_max,
    polish_to_residual,
)
from phasecraft.lib.saddles.krawczyk_p1_roots import SaddleSystem
from phasecraft.lib.saddles.picard_lefschetz import compute_phi


RESULTS = HERE / "results"
OUT = RESULTS / "refined_transition_scan"
RAW = OUT / "refined_transition_scan.json"

Q = sweep.Q
K_CLAUSE = sweep.K_CLAUSE
R = sweep.R
BETA_OPT = 0.5433996420760803

BLUE = "#2b6cb0"
ORANGE = "#dd6b20"
GREEN = "#2f855a"
RED = "#c53030"
PURPLE = "#6b46c1"
GRAY = "#4a5568"


def frange(start: float, stop: float, step: float) -> list[float]:
    vals = []
    x = start
    while x <= stop + 0.5 * step:
        vals.append(round(x, 10))
        x += step
    return vals


def default_betas(profile: str) -> list[float]:
    if profile == "quick":
        vals = [0.35, 0.40, 0.45, 0.50, BETA_OPT, 0.575, 0.60, 0.625, 0.65]
    elif profile == "dense":
        vals = (
            frange(0.25, 0.50, 0.025)
            + [BETA_OPT]
            + frange(0.55, 0.75, 0.025)
        )
    else:
        vals = (
            frange(0.30, 0.50, 0.025)
            + [BETA_OPT]
            + frange(0.55, 0.70, 0.025)
        )
    return sorted(set(round(float(v), 10) for v in vals))


def default_gammas(profile: str) -> list[float]:
    """Return positive Gamma values; the BM24 scripts use gamma = -Gamma."""
    if profile == "quick":
        vals = (
            frange(1.45, 1.75, 0.05)
            + [1.80, 1.825, 1.85, 1.875]
            + frange(1.90, 2.10, 0.05)
        )
    elif profile == "dense":
        vals = frange(1.30, 2.25, 0.025)
    else:
        vals = (
            frange(1.35, 1.70, 0.05)
            + frange(1.725, 1.95, 0.025)
            + frange(2.00, 2.20, 0.05)
        )
    return sorted(set(round(float(v), 10) for v in vals))


def row_key(beta: float, gamma: float) -> str:
    return f"{beta:.10f}|{gamma:.10f}"


def load_cached_rows() -> dict[str, dict[str, Any]]:
    if not RAW.exists():
        return {}
    with RAW.open() as f:
        data = json.load(f)
    return {
        row_key(float(r["beta"]), float(r["gamma"])): r
        for r in data.get("rows", [])
    }


def save_rows(rows: list[dict[str, Any]], metadata: dict[str, Any]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with RAW.open("w") as f:
        json.dump({"metadata": metadata, "rows": rows}, f, indent=2)


def seed_from_previous(
    z_prev: np.ndarray,
    beta: float,
    gamma: float,
) -> tuple[np.ndarray | None, bool]:
    """Fast local seed continuation without recomputing finite-n exponents."""
    try:
        sys_ = SaddleSystem.build(
            q=Q,
            r=R,
            betas=np.array([beta]),
            gammas=np.array([gamma]),
        )
        x0 = np.concatenate([z_prev.real, z_prev.imag])
        x_pol, res = polish_to_residual(
            sys_,
            x0,
            min_residual=sweep.MIN_RESIDUAL,
            dps=sweep.DPS,
        )
        if res > 1e-6:
            return None, False
        z_pol = _x_to_z(x_pol, sys_.nvars)
        ok, _ = certify_z(sys_, z_pol, dps=sweep.DPS)
        return z_pol, bool(ok)
    except Exception:
        return None, False


def warm_seed_to(beta: float, target_gamma: float) -> tuple[np.ndarray | None, bool]:
    """Continue the BM24 seed from small Gamma to the first refined grid point."""
    target_G = -float(target_gamma)
    if target_G <= 0.05:
        z, _, ok = sweep._init_seed(beta, target_gamma)
        return z, ok

    warm_G = [0.05, 0.30, 0.60, 1.00]
    if target_G > 1.00:
        warm_G += frange(1.10, target_G, 0.10)
    warm_G.append(target_G)
    warm_G = sorted(set(round(g, 10) for g in warm_G if g <= target_G + 1e-10))

    z_current: np.ndarray | None = None
    ok_current = False
    for i, G in enumerate(warm_G):
        gamma = -G
        if i == 0 or z_current is None:
            z_current, _, ok_current = sweep._init_seed(beta, gamma)
        else:
            z_next, ok_next = seed_from_previous(z_current, beta, gamma)
            if z_next is None:
                z_next, _, ok_next = sweep._step_seed(z_current, beta, gamma)
            if z_next is None:
                z_next, _, ok_next = sweep._init_seed(beta, gamma)
            z_current, ok_current = z_next, ok_next
        if z_current is None:
            return None, False
    return z_current, ok_current


def full_seed_only_point(beta: float, gamma: float, seed_z: np.ndarray) -> dict[str, Any]:
    """Compute the exact finite-n check for the seed when competitor search fails."""
    sys_ = SaddleSystem.build(q=Q, r=R, betas=np.array([beta]), gammas=np.array([gamma]))
    phi = compute_phi(seed_z, q=Q, r=R, betas=sys_.betas, gammas=sys_.gammas)
    full_seed = conv2_full_exponent(float(phi.real), K_CLAUSE, R)
    lam = lambda_abs_n_max(Q, K_CLAUSE, R, beta, gamma, sweep.N_VALUES)
    return {
        "beta": beta,
        "gamma": gamma,
        "re_phi_seed": float(phi.real),
        "im_phi_seed": float(phi.imag),
        "full_conv2_seed": full_seed,
        "lambda_abs": lam,
        "seed_gap": full_seed - lam,
        "n_certified_competitors": 0,
        "max_competitor_delta_re": float("nan"),
        "seed_algebraically_dominates": None,
        "seed_is_best_match_to_exact": True,
        "best_match_abs_gap": float("nan"),
        "competitor_search_failed": True,
    }


def classify(row: dict[str, Any], decoy_margin: float) -> str:
    if row.get("status") != "ok":
        return "failed"
    if not bool(row.get("seed_is_best_match_to_exact")):
        return "competitor_best"
    delta = float(row.get("max_competitor_delta_re", 0.0))
    if math.isfinite(delta) and delta > decoy_margin:
        return "decoy_challenge"
    return "seed_best"


def effective_gap(row: dict[str, Any]) -> float:
    seed_gap = abs(float(row.get("seed_gap", float("nan"))))
    best_gap = float(row.get("best_match_abs_gap", float("nan")))
    if bool(row.get("seed_is_best_match_to_exact")) or math.isnan(best_gap):
        return seed_gap
    return best_gap


def transition_table(rows: list[dict[str, Any]], decoy_margin: float) -> list[dict[str, Any]]:
    out = []
    for beta in sorted({float(r["beta"]) for r in rows if r.get("status") == "ok"}):
        rows_b = sorted(
            [r for r in rows if r.get("status") == "ok" and math.isclose(float(r["beta"]), beta)],
            key=lambda r: -float(r["gamma"]),
        )
        flags = [classify(r, decoy_margin) != "competitor_best" for r in rows_b]
        gamma_pos = [-float(r["gamma"]) for r in rows_b]
        transitions = []
        for a, b, Ga, Gb in zip(flags[:-1], flags[1:], gamma_pos[:-1], gamma_pos[1:]):
            if a != b:
                transitions.append(
                    {
                        "Gamma_low": Ga,
                        "Gamma_high": Gb,
                        "Gamma_mid": 0.5 * (Ga + Gb),
                        "uncertainty": 0.5 * abs(Gb - Ga),
                        "from": "seed_or_decoy" if a else "competitor",
                        "to": "seed_or_decoy" if b else "competitor",
                    }
                )
        classes = [classify(r, decoy_margin) for r in rows_b]
        out.append(
            {
                "beta": beta,
                "n_points": len(rows_b),
                "n_seed_best": classes.count("seed_best"),
                "n_decoy_challenge": classes.count("decoy_challenge"),
                "n_competitor_best": classes.count("competitor_best"),
                "transitions": transitions,
            }
        )
    return out


def grid_from_rows(rows: list[dict[str, Any]], decoy_margin: float) -> dict[str, np.ndarray]:
    ok = [r for r in rows if r.get("status") == "ok"]
    betas = np.array(sorted({float(r["beta"]) for r in ok}))
    Gammas = np.array(sorted({-float(r["gamma"]) for r in ok}))
    cls = np.full((len(Gammas), len(betas)), np.nan)
    delta = np.full_like(cls, np.nan, dtype=float)
    gap = np.full_like(cls, np.nan, dtype=float)
    ncomp = np.full_like(cls, np.nan, dtype=float)

    code = {"seed_best": 0.0, "decoy_challenge": 1.0, "competitor_best": 2.0, "failed": np.nan}
    for r in ok:
        b = float(r["beta"])
        G = -float(r["gamma"])
        i = int(np.where(np.isclose(Gammas, G))[0][0])
        j = int(np.where(np.isclose(betas, b))[0][0])
        cls[i, j] = code[classify(r, decoy_margin)]
        delta[i, j] = float(r.get("max_competitor_delta_re", float("nan")))
        gap[i, j] = effective_gap(r)
        ncomp[i, j] = float(r.get("n_certified_competitors", float("nan")))
    return {"betas": betas, "Gammas": Gammas, "class": cls, "delta": delta, "gap": gap, "ncomp": ncomp}


def pcolor_edges(vals: np.ndarray) -> np.ndarray:
    if len(vals) == 1:
        return np.array([vals[0] - 0.5, vals[0] + 0.5])
    mids = 0.5 * (vals[:-1] + vals[1:])
    first = vals[0] - 0.5 * (vals[1] - vals[0])
    last = vals[-1] + 0.5 * (vals[-1] - vals[-2])
    return np.concatenate([[first], mids, [last]])


def plot_class_map(grid: dict[str, np.ndarray], transitions: list[dict[str, Any]]) -> str:
    OUT.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    cmap = mcolors.ListedColormap(["#cfe8ff", "#d8f0d2", "#ffd1a6"])
    norm = mcolors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)
    mesh = ax.pcolormesh(
        pcolor_edges(grid["betas"]),
        pcolor_edges(grid["Gammas"]),
        grid["class"],
        cmap=cmap,
        norm=norm,
        shading="flat",
    )
    cbar = fig.colorbar(mesh, ax=ax, ticks=[0, 1, 2])
    cbar.ax.set_yticklabels(["seed controls", "high-Re decoy", "competitor controls"])

    bx, by, be = [], [], []
    axx, axy = [], []
    for t in transitions:
        is_robust = (
            len(t["transitions"]) == 1
            and t["transitions"][0]["from"] == "seed_or_decoy"
            and t["transitions"][0]["to"] == "competitor"
        )
        if is_robust:
            primary = t["transitions"][0]
            bx.append(float(t["beta"]))
            by.append(float(primary["Gamma_mid"]))
            be.append(float(primary["uncertainty"]))
        elif t["transitions"]:
            first = t["transitions"][0]
            axx.append(float(t["beta"]))
            axy.append(float(first["Gamma_mid"]))
    if bx:
        ax.errorbar(
            bx,
            by,
            yerr=be,
            fmt="o-",
            color=PURPLE,
            lw=1.5,
            ms=4,
            label="robust single switch",
        )
    if axx:
        ax.scatter(
            axx,
            axy,
            marker="x",
            s=70,
            color=RED,
            linewidths=2.0,
            label="ambiguous/nonmonotone slice",
            zorder=4,
        )
    ax.axvline(BETA_OPT, color=GRAY, ls="--", lw=1.1, label=r"$\beta_\mathrm{opt}$")
    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel(r"$\Gamma=-\gamma$")
    ax.set_title("Refined saddle-control map near the transition band")
    ax.legend(loc="upper right", frameon=True, fontsize=9)
    fig.tight_layout()
    path = OUT / "refined_transition_map.png"
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def plot_decoy_pressure(grid: dict[str, np.ndarray]) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), sharey=True)
    X = pcolor_edges(grid["betas"])
    Y = pcolor_edges(grid["Gammas"])

    vmax = np.nanpercentile(np.abs(grid["delta"]), 95)
    vmax = max(float(vmax), 0.05)
    im0 = axes[0].pcolormesh(
        X,
        Y,
        grid["delta"],
        cmap="coolwarm",
        vmin=-vmax,
        vmax=vmax,
        shading="flat",
    )
    axes[0].contour(grid["betas"], grid["Gammas"], grid["class"], levels=[0.5, 1.5], colors=[GREEN, ORANGE], linewidths=1.2)
    axes[0].set_title(r"Algebraic pressure: max competitor $\Delta$ exponent")
    axes[0].set_xlabel(r"$\beta$")
    axes[0].set_ylabel(r"$\Gamma=-\gamma$")
    fig.colorbar(im0, ax=axes[0], label=r"$E_\mathrm{max\,comp}-E_\mathrm{seed}$")

    log_gap = np.log10(np.clip(grid["gap"], 1e-8, None))
    im1 = axes[1].pcolormesh(X, Y, log_gap, cmap="viridis_r", shading="flat")
    axes[1].contour(grid["betas"], grid["Gammas"], grid["class"], levels=[0.5, 1.5], colors=[GREEN, ORANGE], linewidths=1.2)
    axes[1].set_title("Exact finite-n exponent match")
    axes[1].set_xlabel(r"$\beta$")
    fig.colorbar(im1, ax=axes[1], label=r"$\log_{10}|E_\mathrm{best}-\lambda_n|$")

    for ax in axes:
        ax.axvline(BETA_OPT, color=GRAY, ls="--", lw=1.0)
    fig.tight_layout()
    path = OUT / "refined_decoy_pressure_and_exact_gap.png"
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def plot_boundary(transitions: list[dict[str, Any]]) -> str:
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    bx, by, be = [], [], []
    axx, axy = [], []
    for t in transitions:
        if not t["transitions"]:
            continue
        is_robust = (
            len(t["transitions"]) == 1
            and t["transitions"][0]["from"] == "seed_or_decoy"
            and t["transitions"][0]["to"] == "competitor"
        )
        if is_robust:
            primary = t["transitions"][0]
            bx.append(float(t["beta"]))
            by.append(float(primary["Gamma_mid"]))
            be.append(float(primary["uncertainty"]))
        else:
            axx.append(float(t["beta"]))
            axy.append(float(t["transitions"][0]["Gamma_mid"]))
    if bx:
        ax.errorbar(bx, by, yerr=be, fmt="none", ecolor=GRAY, alpha=0.65, capsize=3)
        ax.scatter(bx, by, c=PURPLE, s=38, zorder=3, label="robust single switch")
        ax.plot(bx, by, color=PURPLE, alpha=0.5, lw=1.0)
    if axx:
        ax.scatter(
            axx,
            axy,
            marker="x",
            s=70,
            color=RED,
            linewidths=2.0,
            label="ambiguous/nonmonotone",
            zorder=4,
        )
    ax.axvline(BETA_OPT, color=GRAY, ls="--", lw=1.0)
    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel(r"transition $\Gamma=-\gamma$")
    ax.set_title("Refined transition estimates from extra saddle searches")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", frameon=True, fontsize=9)
    fig.tight_layout()
    path = OUT / "refined_boundary_curve.png"
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def write_report(
    rows: list[dict[str, Any]],
    metadata: dict[str, Any],
    transitions: list[dict[str, Any]],
    figures: list[str],
    decoy_margin: float,
) -> str:
    ok = [r for r in rows if r.get("status") == "ok"]
    classes = [classify(r, decoy_margin) for r in ok]
    decoys = [
        r for r in ok
        if classify(r, decoy_margin) == "decoy_challenge"
    ]
    comp = [r for r in ok if classify(r, decoy_margin) == "competitor_best"]

    lines = [
        "# Refined transition scan",
        "",
        "## Method",
        "",
        "A certified saddle is not declared physically dominant merely because it has larger `Re Phi`. "
        "For each grid point the script compares saddle exponents against the exact finite-n exponent "
        "`lambda_abs = log|P_n|/n` over the configured `n` range.  A large-Re saddle is marked as a "
        "decoy challenge when it beats the seed algebraically but the seed remains the closest match to "
        "the exact exponent.  This is an empirical/certified filter, not a Picard-Lefschetz intersection proof.",
        "",
        "## Scan",
        "",
        f"- beta points: {len(metadata['beta_grid'])}",
        f"- Gamma points: {len(metadata['Gamma_grid'])}",
        f"- successful grid points: {len(ok)} / {len(rows)}",
        f"- competitor random starts per point: {metadata['n_competitor_starts']}",
        f"- finite-n values: {metadata['n_values']}",
        f"- decoy margin: `{decoy_margin}` in conv2 exponent units",
        "",
        "## Classification counts",
        "",
        f"- seed controls cleanly: {classes.count('seed_best')}",
        f"- seed controls despite high-Re decoys: {classes.count('decoy_challenge')}",
        f"- competitor controls by exact-exponent match: {classes.count('competitor_best')}",
        "",
        "## Transition estimates",
        "",
        "| beta | transition interval in Gamma | midpoint | notes |",
        "|---:|---:|---:|---|",
    ]
    for t in transitions:
        if not t["transitions"]:
            note = "no transition on refined grid"
            lines.append(f"| {t['beta']:.6f} | - | - | {note} |")
            continue
        parts = []
        for tr in t["transitions"]:
            parts.append(f"{tr['Gamma_low']:.4f}-{tr['Gamma_high']:.4f}")
        first = t["transitions"][0]
        note = f"{t['n_decoy_challenge']} decoy-challenge points, {t['n_competitor_best']} competitor-best points"
        lines.append(f"| {t['beta']:.6f} | {', '.join(parts)} | {first['Gamma_mid']:.4f} | {note} |")

    if decoys:
        max_decoy = max(decoys, key=lambda r: float(r.get("max_competitor_delta_re", 0.0)))
        lines += [
            "",
            "## Strongest decoy challenge",
            "",
            f"- beta={float(max_decoy['beta']):.6f}, Gamma={-float(max_decoy['gamma']):.6f}",
            f"- max competitor advantage over seed: {float(max_decoy['max_competitor_delta_re']):+.6f}",
            f"- seed exact gap: {abs(float(max_decoy['seed_gap'])):.6g}",
            f"- best competitor exact gap: {float(max_decoy.get('best_match_abs_gap', float('nan'))):.6g}",
        ]
    if comp:
        first_comp = min(comp, key=lambda r: (abs(float(r["beta"]) - BETA_OPT), -float(r["gamma"])))
        lines += [
            "",
            "## First competitor-control evidence near beta_opt",
            "",
            f"- beta={float(first_comp['beta']):.6f}, Gamma={-float(first_comp['gamma']):.6f}",
            f"- seed exact gap: {abs(float(first_comp['seed_gap'])):.6g}",
            f"- best competitor exact gap: {float(first_comp.get('best_match_abs_gap', float('nan'))):.6g}",
            f"- max competitor advantage over seed: {float(first_comp['max_competitor_delta_re']):+.6f}",
        ]

    lines += [
        "",
        "## Figures",
        "",
    ]
    lines += [f"- `{Path(fig).name}`" for fig in figures]
    lines.append("")

    path = OUT / "refined_transition_scan_report.md"
    path.write_text("\n".join(lines))
    return str(path)


def run_scan(args: argparse.Namespace) -> None:
    betas = args.betas if args.betas else default_betas(args.profile)
    Gammas = args.gammas if args.gammas else default_gammas(args.profile)
    betas = sorted(set(round(float(v), 10) for v in betas))
    Gammas = sorted(set(round(float(v), 10) for v in Gammas))
    gammas = [-G for G in Gammas]

    sweep.N_COMPETITOR_STARTS = int(args.starts)

    metadata = {
        "q": Q,
        "K_clause": K_CLAUSE,
        "r": R,
        "phi_pref_conv2": conv2_pref(K_CLAUSE, R),
        "beta_grid": betas,
        "Gamma_grid": Gammas,
        "gamma_grid": gammas,
        "n_values": sweep.N_VALUES,
        "n_competitor_starts": sweep.N_COMPETITOR_STARTS,
        "profile": args.profile,
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    cached = load_cached_rows() if args.resume else {}
    all_rows: list[dict[str, Any]] = []
    print(f"Refined scan: {len(betas)} beta x {len(gammas)} Gamma = {len(betas) * len(gammas)} points")
    print(f"Competitor starts per point: {sweep.N_COMPETITOR_STARTS}")
    print(f"Output: {RAW}")

    for ib, beta in enumerate(betas, start=1):
        print(f"\n=== beta {ib}/{len(betas)} = {beta:.6f} ===", flush=True)
        beta_cached = all(row_key(beta, gamma) in cached for gamma in gammas)
        if beta_cached:
            z_current = None
            ok_current = False
            print("  full beta slice cached", flush=True)
        else:
            z_current, ok_current = warm_seed_to(beta, gammas[0])
            if z_current is None:
                print(f"  warm start to Gamma={-gammas[0]:.4f} failed; trying grid cold starts", flush=True)
            else:
                print(f"  warm start to Gamma={-gammas[0]:.4f}: ok", flush=True)

        for ig, gamma in enumerate(gammas):
            key = row_key(beta, gamma)
            if key in cached:
                row = cached[key]
                all_rows.append(row)
                print(f"  Gamma={-gamma:.4f}: cached {classify(row, args.decoy_margin)}", flush=True)
                z_current = None
                continue

            print(f"  Gamma={-gamma:.4f}: ", end="", flush=True)
            if ig == 0 and z_current is not None:
                z = z_current
                ok = ok_current
            elif z_current is not None:
                z, ok = seed_from_previous(z_current, beta, gamma)
                if z is None:
                    z, _, ok = sweep._init_seed(beta, gamma)
            else:
                z, _, ok = sweep._init_seed(beta, gamma)
            if z is None:
                print("retrying warm path; ", end="", flush=True)
                z, ok = warm_seed_to(beta, gamma)
            if z is None:
                row = {
                    "beta": beta,
                    "gamma": gamma,
                    "status": "seed_failed",
                }
                all_rows.append(row)
                z_current = None
                print("seed failed", flush=True)
                continue

            z_current = z
            try:
                row = sweep.sweep_point(beta, gamma, z)
            except Exception as exc:
                print(f"competitor search failed ({exc}); seed-only fallback", end="; ", flush=True)
                row = full_seed_only_point(beta, gamma, z)
            row["certified_seed"] = ok
            row["status"] = "ok"
            all_rows.append(row)

            cls = classify(row, args.decoy_margin)
            print(
                f"{cls}; seed_gap={float(row['seed_gap']):+.4g}; "
                f"max_delta={float(row.get('max_competitor_delta_re', float('nan'))):+.4g}; "
                f"n_comp={int(row.get('n_certified_competitors', 0))}",
                flush=True,
            )

            if args.checkpoint_every and len(all_rows) % args.checkpoint_every == 0:
                merged = merge_rows(cached, all_rows)
                save_rows(merged, metadata)

    rows = merge_rows(cached, all_rows)
    save_rows(rows, metadata)
    transitions = transition_table(rows, args.decoy_margin)
    grid = grid_from_rows(rows, args.decoy_margin)
    figures = [
        plot_class_map(grid, transitions),
        plot_decoy_pressure(grid),
        plot_boundary(transitions),
    ]
    report = write_report(rows, metadata, transitions, figures, args.decoy_margin)
    summary = {
        "metadata": metadata,
        "counts": {
            "rows": len(rows),
            "ok": sum(1 for r in rows if r.get("status") == "ok"),
            "seed_best": sum(1 for r in rows if classify(r, args.decoy_margin) == "seed_best"),
            "decoy_challenge": sum(1 for r in rows if classify(r, args.decoy_margin) == "decoy_challenge"),
            "competitor_best": sum(1 for r in rows if classify(r, args.decoy_margin) == "competitor_best"),
        },
        "transitions": transitions,
        "figures": figures,
        "report": report,
    }
    with (OUT / "refined_transition_scan_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print("\nWrote:")
    print(f"  {RAW}")
    print(f"  {OUT / 'refined_transition_scan_summary.json'}")
    print(f"  {report}")
    for fig in figures:
        print(f"  {fig}")


def merge_rows(cached: dict[str, dict[str, Any]], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged = dict(cached)
    for row in rows:
        if "beta" not in row or "gamma" not in row:
            continue
        merged[row_key(float(row["beta"]), float(row["gamma"]))] = row
    return sorted(merged.values(), key=lambda r: (float(r["beta"]), -float(r["gamma"])))


def parse_csv_floats(value: str) -> list[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=["quick", "standard", "dense"], default="standard")
    parser.add_argument("--starts", type=int, default=500, help="random competitor starts per grid point")
    parser.add_argument("--decoy-margin", type=float, default=0.02)
    parser.add_argument("--checkpoint-every", type=int, default=20)
    parser.add_argument("--no-resume", dest="resume", action="store_false", default=True)
    parser.add_argument("--betas", type=parse_csv_floats, default=None, help="comma-separated beta grid override")
    parser.add_argument("--gammas", type=parse_csv_floats, default=None, help="comma-separated positive Gamma grid override")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_scan(args)


if __name__ == "__main__":
    main()
