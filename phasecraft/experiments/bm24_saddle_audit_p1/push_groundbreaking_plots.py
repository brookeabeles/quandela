"""Generate focused "push further" BM24 saddle diagnostics.

This script consumes existing audit artifacts and creates a compact set of
figures aimed at the physical saddle-transition story:

1. Physical/explanatory saddle dominance phase diagram.
2. Anti-Stokes crossing diagnostic near gamma ~= -1.83.
3. Finite-n crossing drift.
4. Algebraic-vs-physical saddle scatter.
5. Large-gamma plateau diagnostic.
6. Branch 43/46 pair diagnostic.

It does not rerun expensive root searches.
"""

from __future__ import annotations

import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-phasecraft")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
OUT = RESULTS / "push_groundbreaking"
REPO_PARENT = HERE.parents[2]
if str(REPO_PARENT) not in sys.path:
    sys.path.insert(0, str(REPO_PARENT))

PHI_PREF_DEFAULT = -0.689609375
GAMMA_PAIR_TOL = 0.011


def load_json(path: Path) -> Any:
    with path.open() as f:
        return json.load(f)


def conv2_full(re_phi: float, phi_pref: float) -> float:
    return phi_pref + re_phi


def im_mod_distance_to_zero(im_phi: float) -> float:
    """Distance of a phase to 0 modulo 2*pi."""
    return abs((im_phi + math.pi) % (2.0 * math.pi) - math.pi)


def interp(gamma: float, xs: np.ndarray, ys: np.ndarray) -> float:
    order = np.argsort(xs)
    return float(np.interp(gamma, xs[order], ys[order]))


def style_gamma(ax: plt.Axes) -> None:
    ax.set_xlabel(r"$\gamma$")
    ax.grid(True, alpha=0.25)


def savefig(fig: plt.Figure, name: str, dpi: int = 170) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def load_seed_continuation() -> dict[str, np.ndarray]:
    data = load_json(RESULTS / "run_seed_branch_g-2pi" / "seed_branch_continuation.json")
    rows = [
        r
        for r in data["continuation"]
        if r.get("certified") and not r.get("failed")
    ]
    return {
        "gamma": np.array([float(r["gamma"]) for r in rows]),
        "seed_exp": np.array([float(r["full_conv2_exponent"]) for r in rows]),
        "seed_re": np.array([float(r["re_phi_m"]) for r in rows]),
        "seed_im": np.array([float(r["im_phi_m"]) for r in rows]),
        "exact": np.array([float(r["lambda_abs_n_max"]) for r in rows]),
    }


def load_merged_43_46_series(phi_pref: float) -> dict[str, np.ndarray]:
    resolved = load_json(
        RESULTS
        / "run_seed_branch_g-2pi"
        / "archive"
        / "competitor_dominance"
        / "z_robust_full"
        / "resolved_robust_z.json"
    )
    seed = load_seed_continuation()
    seed_g = seed["gamma"]
    exact = seed["exact"]
    seed_exp = seed["seed_exp"]
    seed_re = seed["seed_re"]

    branch_tables: list[dict[float, dict[str, float]]] = []
    for bid in (43, 46):
        br = next(b for b in resolved["branches"] if int(b["branch_id"]) == bid)
        table: dict[float, dict[str, float]] = {}
        for p in br["points"]:
            if not p.get("box_disjoint_from_seed", True):
                continue
            g = round(float(p["gamma"]), 6)
            table[g] = {
                "re": float(p["Phi_eff_real"]),
                "im": float(p["Phi_eff_imag"]),
            }
        branch_tables.append(table)

    all_g = sorted(set(branch_tables[0]) | set(branch_tables[1]), reverse=True)
    gammas: list[float] = []
    merged_re: list[float] = []
    merged_im_abs: list[float] = []
    exact_i: list[float] = []
    seed_i: list[float] = []
    seed_re_i: list[float] = []

    for g in all_g:
        vals = []
        ims = []
        for table in branch_tables:
            if g in table:
                vals.append(table[g]["re"])
                ims.append(table[g]["im"])
                continue
            for g2, row in table.items():
                if abs(g2 - g) <= GAMMA_PAIR_TOL:
                    vals.append(row["re"])
                    ims.append(row["im"])
                    break
        if not vals:
            continue
        gammas.append(g)
        merged_re.append(float(np.mean(vals)))
        merged_im_abs.append(float(np.mean([im_mod_distance_to_zero(v) for v in ims])))
        exact_i.append(interp(g, seed_g, exact))
        seed_i.append(interp(g, seed_g, seed_exp))
        seed_re_i.append(interp(g, seed_g, seed_re))

    return {
        "gamma": np.array(gammas),
        "merged_re": np.array(merged_re),
        "merged_exp": np.array([conv2_full(v, phi_pref) for v in merged_re]),
        "merged_im_mod_abs": np.array(merged_im_abs),
        "exact": np.array(exact_i),
        "seed_exp": np.array(seed_i),
        "seed_re": np.array(seed_re_i),
        "delta_re": np.array(merged_re) - np.array(seed_re_i),
    }


def plot_physical_dominance_phase_diagram(phi_pref: float) -> dict[str, Any]:
    sweep = load_json(RESULTS / "sweep_beta_gamma_full" / "sweep_beta_gamma_full.json")
    rows = [r for r in sweep["rows"] if r.get("status") == "ok"]
    betas = np.array(sweep["metadata"]["beta_grid"], dtype=float)
    gammas = np.array(sweep["metadata"]["gamma_grid"], dtype=float)
    B, G = len(betas), len(gammas)
    dom = np.full((B, G), np.nan)
    err = np.full((B, G), np.nan)
    algebraic_warning = np.zeros((B, G), dtype=float)

    b_index = {round(float(b), 10): i for i, b in enumerate(betas)}
    g_index = {round(float(g), 10): i for i, g in enumerate(gammas)}
    for r in rows:
        bi = b_index[round(float(r["beta"]), 10)]
        gi = g_index[round(float(r["gamma"]), 10)]
        seed_best = bool(r["seed_is_best_match_to_exact"])
        dom[bi, gi] = 0.0 if seed_best else 1.0
        bm = float(r.get("best_match_abs_gap", float("nan")))
        sg = abs(float(r["seed_gap"]))
        err[bi, gi] = sg if seed_best or math.isnan(bm) else bm
        algebraic_warning[bi, gi] = 1.0 if float(r.get("max_competitor_delta_re", 0.0)) > 0.25 else 0.0

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), sharey=True)
    ax = axes[0]
    cmap = matplotlib.colors.ListedColormap(["#276fbf", "#d95f02"])
    im = ax.pcolormesh(betas, gammas, dom.T, cmap=cmap, vmin=0, vmax=1, shading="nearest")
    ax.scatter([0.5434], [-0.749], marker="*", s=180, c="gold", edgecolor="k", zorder=5)
    warn_y, warn_x = np.where(algebraic_warning.T > 0)
    if len(warn_x):
        ax.scatter(
            betas[warn_x],
            gammas[warn_y],
            s=11,
            facecolors="none",
            edgecolors="k",
            linewidths=0.4,
            alpha=0.45,
            label="high-Re certified roots present",
        )
    ax.set_title("Finite-n explanatory dominance")
    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel(r"$\gamma$")
    ax.legend(loc="lower right", fontsize=7)
    cbar = fig.colorbar(im, ax=ax, ticks=[0.25, 0.75])
    cbar.ax.set_yticklabels(["seed", "competitor"])

    ax = axes[1]
    err_clip = np.clip(err, 1e-4, 0.2)
    im2 = ax.pcolormesh(
        betas,
        gammas,
        err_clip.T,
        norm=matplotlib.colors.LogNorm(vmin=1e-4, vmax=0.2),
        cmap="viridis_r",
        shading="nearest",
    )
    cs = ax.contour(betas, gammas, err.T, levels=[0.002, 0.005, 0.01, 0.05], colors="white", linewidths=0.8)
    ax.clabel(cs, fontsize=7, fmt="%.3f")
    ax.scatter([0.5434], [-0.749], marker="*", s=180, c="gold", edgecolor="k", zorder=5)
    ax.set_title("Best explanatory-saddle residual")
    ax.set_xlabel(r"$\beta$")
    fig.colorbar(im2, ax=ax, label=r"$|\lambda_{\rm exact}-E_{\rm best}|$")
    fig.suptitle("Physical/explanatory phase diagram; not raw max Re Phi")
    path = savefig(fig, "physical_dominance_phase_diagram_beta_gamma.png")

    beta_idx = int(np.argmin(np.abs(betas - 0.5434)))
    beta_rows = [(gammas[j], dom[beta_idx, j], err[beta_idx, j]) for j in range(G)]
    return {
        "path": str(path),
        "beta_opt_rows": beta_rows,
        "note": "Uses existing sweep_beta_gamma_full finite-n explanatory labels.",
        "phi_pref": phi_pref,
    }


def plot_anti_stokes_crossing(phi_pref: float) -> dict[str, Any]:
    merged = load_merged_43_46_series(phi_pref)
    mask = (merged["gamma"] <= -1.2) & (merged["gamma"] >= -2.25)
    gammas = merged["gamma"][mask]
    delta_re = merged["delta_re"][mask]
    delta_im_mod = merged["merged_im_mod_abs"][mask]
    exact = merged["exact"][mask]
    seed_exp = merged["seed_exp"][mask]
    comp_exp = merged["merged_exp"][mask]

    order = np.argsort(gammas)
    g = gammas[order]
    d = delta_re[order]
    crosses = []
    for i in range(len(g) - 1):
        if d[i] == 0 or d[i] * d[i + 1] < 0:
            t = -d[i] / (d[i + 1] - d[i])
            crosses.append(float(g[i] + t * (g[i + 1] - g[i])))

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    axes[0].plot(gammas, delta_re, "o-", color="C3")
    axes[0].axhline(0, color="k", lw=0.8)
    for c in crosses:
        axes[0].axvline(c, color="0.3", ls=":", lw=1)
    axes[0].set_ylabel(r"$\Delta Re\Phi$")
    axes[0].set_title("Anti-Stokes diagnostic: branch 43/46 versus seed")

    axes[1].plot(gammas, delta_im_mod, "s-", color="C4")
    axes[1].axhline(0, color="k", lw=0.8)
    axes[1].set_ylabel(r"$|\Delta Im\Phi|$ mod $2\pi$")

    axes[2].plot(gammas, exact - seed_exp, "o-", label="exact n_max - seed")
    axes[2].plot(gammas, exact - comp_exp, "s-", label="exact n_max - merged 43/46")
    axes[2].axhline(0, color="k", lw=0.8)
    axes[2].set_ylabel("exponent residual")
    axes[2].legend(fontsize=8)
    style_gamma(axes[2])
    path = savefig(fig, "anti_stokes_crossing_gamma.png")
    primary = min(crosses, key=lambda x: abs(x + 1.83)) if crosses else float("nan")
    return {
        "path": str(path),
        "crossings_from_delta_re": crosses,
        "primary_crossing_near_qaoa_window": primary,
        "secondary_crossing_note": "Extra crossings near branch onset should be treated as local wiggles unless confirmed by branch/intersection analysis.",
        "min_delta_im_mod": float(np.min(delta_im_mod)),
        "max_delta_im_mod": float(np.max(delta_im_mod)),
    }


def plot_finite_n_crossing_drift() -> dict[str, Any]:
    push = load_json(RESULTS / "push_further_analysis.json")
    conv = push["push_4_n_max_convergence"]["results"]
    crossover = push["crossover_analysis"]

    crossing_rows = [
        {"n": 12, "gamma_cross_best_match": float(crossover["n12_22_crossover_gamma"]), "source": "push_further_analysis"},
        {"n": 22, "gamma_cross_best_match": float(crossover["n12_22_crossover_gamma"]), "source": "push_further_analysis"},
        {"n": 40, "gamma_cross_best_match": float(crossover["n40_crossover_gamma"]), "source": "push_further_analysis"},
    ]
    x = np.array([1.0 / r["n"] for r in crossing_rows])
    y = np.array([r["gamma_cross_best_match"] for r in crossing_rows])
    fit = np.polyfit(x, y, 1)
    extrap = float(fit[1])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    ax.plot(x, y, "o", label="reported apparent crossover")
    xx = np.linspace(0, max(x) * 1.05, 100)
    ax.plot(xx, np.polyval(fit, xx), "-", label=rf"linear guide, $n\to\infty$: {extrap:.3f}")
    ax.axhline(float(crossover["n40_crossover_gamma"]), color="0.35", ls=":", label="n=40 estimate")
    ax.set_xlabel(r"$1/n$")
    ax.set_ylabel(r"$\gamma_{\rm cross}(n)$")
    ax.set_title("Finite-size drift of apparent crossover")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)

    ax = axes[1]
    for row in conv:
        g = row["gamma"]
        seed_gap = [row["lambda_n12"] - row["seed_exp"], row["lambda_n22"] - row["seed_exp"], row["lambda_n40"] - row["seed_exp"]]
        b43_gap = [row["lambda_n12"] - row["b43_exp"], row["lambda_n22"] - row["b43_exp"], row["lambda_n40"] - row["b43_exp"]]
        ax.plot([12, 22, 40], np.abs(seed_gap), "o-", alpha=0.8, label=rf"$\gamma={g}$ seed")
        ax.plot([12, 22, 40], np.abs(b43_gap), "s--", alpha=0.8, label=rf"$\gamma={g}$ b43")
    ax.set_yscale("log")
    ax.set_xlabel(r"$n$")
    ax.set_ylabel("absolute residual to exact lambda")
    ax.set_title("Convergence chooses seed before crossing, 43/46 after")
    ax.legend(fontsize=6, ncol=2)
    ax.grid(True, alpha=0.25)
    path = savefig(fig, "finite_n_crossing_drift.png")
    return {
        "path": str(path),
        "rows": crossing_rows,
        "linear_fit_gamma_cross_at_infinite_n": extrap,
        "linear_fit_slope": float(fit[0]),
        "warning": "Uses already-computed push_further_analysis values; treat the line fit as a visual guide only.",
    }


def plot_algebraic_vs_physical_scatter(phi_pref: float) -> dict[str, Any]:
    comp = load_json(
        RESULTS
        / "run_seed_branch_g-2pi"
        / "archive"
        / "competitor_dominance"
        / "competitor_dominance_summary.json"
    )
    seed = load_seed_continuation()
    rows = []
    for pg in comp["per_gamma"]:
        g = float(pg["gamma"])
        lam = interp(g, seed["gamma"], seed["exact"])
        for c in pg.get("competitors", []):
            re_phi = float(c["re_phi_m"])
            im_phi = float(c["im_phi_m"])
            full = phi_pref + re_phi
            rows.append(
                {
                    "gamma": g,
                    "delta_re": float(c["signed_re_phi_gap_vs_seed"]),
                    "abs_gap_to_exact": abs(full - lam),
                    "im_mod_abs": im_mod_distance_to_zero(im_phi),
                    "z_distance": float(c.get("z_distance_from_seed", float("nan"))),
                }
            )

    delta = np.array([r["delta_re"] for r in rows])
    gap = np.array([r["abs_gap_to_exact"] for r in rows])
    immod = np.array([r["im_mod_abs"] for r in rows])
    gamma = np.array([r["gamma"] for r in rows])

    fig, ax = plt.subplots(figsize=(9, 6))
    sc = ax.scatter(delta, gap, c=immod, s=18, cmap="magma_r", alpha=0.68, edgecolors="none")
    ax.axvline(0, color="k", lw=0.8)
    ax.axhline(0.02, color="C2", ls="--", lw=1, label="0.02 exact-exponent gap")
    ax.set_yscale("log")
    ax.set_xlabel(r"$Re\Phi_{\rm comp}-Re\Phi_{\rm seed}$")
    ax.set_ylabel(r"$|(\phi_{\rm pref}+Re\Phi_{\rm comp})-\lambda_{\rm exact}|$")
    ax.set_title("Certified roots: algebraic dominance versus physical explanation")
    ax.legend(fontsize=8)
    fig.colorbar(sc, ax=ax, label=r"$|Im\Phi|$ mod $2\pi$")
    path = savefig(fig, "algebraic_vs_physical_saddles_scatter.png")

    dangerous = [
        r
        for r in rows
        if r["delta_re"] > 0.25 and r["abs_gap_to_exact"] > 0.1
    ]
    explanatory = [
        r
        for r in rows
        if r["abs_gap_to_exact"] < 0.02 and r["im_mod_abs"] < 0.05
    ]
    return {
        "path": str(path),
        "num_competitor_points": len(rows),
        "num_high_re_bad_exact_match": len(dangerous),
        "num_low_gap_near_real_phase": len(explanatory),
        "gamma_values": sorted(set(float(x) for x in gamma), reverse=True),
    }


def plot_large_gamma_plateau(phi_pref: float) -> dict[str, Any]:
    seed = load_seed_continuation()
    merged = load_merged_43_46_series(phi_pref)
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(seed["gamma"], seed["exact"], color="C0", lw=1.8, label="exact finite-n exponent")
    axes[0].plot(seed["gamma"], seed["seed_exp"], color="k", lw=1.6, label="seed saddle")
    axes[0].plot(merged["gamma"], merged["merged_exp"], color="C2", lw=2.2, label="merged 43/46")
    axes[0].axhline(phi_pref, color="C3", ls="--", lw=1.2, label=rf"$\phi_{{pref}}={phi_pref:.6f}$")
    axes[0].set_ylabel("exponent")
    axes[0].set_title("Large-gamma plateau: physical exponent follows branch 43/46, not seed")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(merged["gamma"], merged["exact"] - merged["merged_exp"], color="C2", lw=1.5, label="exact - merged 43/46")
    axes[1].plot(merged["gamma"], merged["exact"] - merged["seed_exp"], color="k", lw=1.2, label="exact - seed")
    axes[1].axhline(0, color="0.4", lw=0.8)
    axes[1].set_ylabel("residual")
    axes[1].legend(fontsize=8)
    style_gamma(axes[1])
    path = savefig(fig, "large_gamma_plateau.png")

    mask_large = merged["gamma"] < -2.5
    return {
        "path": str(path),
        "merged_rmse_large_gamma": float(np.sqrt(np.mean((merged["exact"][mask_large] - merged["merged_exp"][mask_large]) ** 2))),
        "seed_rmse_large_gamma": float(np.sqrt(np.mean((merged["exact"][mask_large] - merged["seed_exp"][mask_large]) ** 2))),
        "phi_pref": phi_pref,
    }


def plot_branch_43_46_pair(phi_pref: float) -> dict[str, Any]:
    resolved = load_json(
        RESULTS
        / "run_seed_branch_g-2pi"
        / "archive"
        / "competitor_dominance"
        / "z_robust_full"
        / "resolved_robust_z.json"
    )
    tables: dict[int, dict[float, dict[str, float]]] = {}
    for bid in (43, 46):
        br = next(b for b in resolved["branches"] if int(b["branch_id"]) == bid)
        table = {}
        for p in br["points"]:
            g = round(float(p["gamma"]), 6)
            table[g] = {
                "re": float(p["Phi_eff_real"]),
                "im": float(p["Phi_eff_imag"]),
                "dist": float(p.get("distance_to_seed_w", float("nan"))),
            }
        tables[bid] = table

    pairs = []
    used46: set[float] = set()
    for g43, row43 in sorted(tables[43].items(), reverse=True):
        candidates = [
            (abs(g43 - g46), g46, row46)
            for g46, row46 in tables[46].items()
            if g46 not in used46 and abs(g43 - g46) <= GAMMA_PAIR_TOL
        ]
        if not candidates:
            continue
        _, g46, row46 = min(candidates, key=lambda t: t[0])
        used46.add(g46)
        pairs.append((0.5 * (g43 + g46), row43, row46))

    g = np.array([p[0] for p in pairs])
    re43 = np.array([p[1]["re"] for p in pairs])
    re46 = np.array([p[2]["re"] for p in pairs])
    im43 = np.array([p[1]["im"] for p in pairs])
    im46 = np.array([p[2]["im"] for p in pairs])

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    axes[0].plot(g, phi_pref + re43, label="branch 43", color="C2")
    axes[0].plot(g, phi_pref + re46, label="branch 46", color="C1", ls="--")
    axes[0].set_ylabel("full exponent")
    axes[0].legend(fontsize=8)
    axes[0].set_title("Branches 43 and 46 are a paired family, not independent evidence twice")

    axes[1].plot(g, re43 - re46, color="C3")
    axes[1].axhline(0, color="0.4", lw=0.8)
    axes[1].set_ylabel(r"$Re\Phi_{43}-Re\Phi_{46}$")

    axes[2].plot(g, im43 + im46, color="C4", label=r"$Im\Phi_{43}+Im\Phi_{46}$")
    axes[2].plot(g, [im_mod_distance_to_zero(a - b) for a, b in zip(im43, -im46)], color="C0", alpha=0.7, label="conjugacy phase residual")
    axes[2].axhline(0, color="0.4", lw=0.8)
    axes[2].set_ylabel("phase pairing")
    axes[2].legend(fontsize=8)
    style_gamma(axes[2])
    path = savefig(fig, "branch_43_46_pair_diagnostic.png")
    return {
        "path": str(path),
        "shared_gamma_count": len(pairs),
        "max_abs_re_difference": float(np.max(np.abs(re43 - re46))) if len(pairs) else float("nan"),
        "median_abs_re_difference": float(np.median(np.abs(re43 - re46))) if len(pairs) else float("nan"),
        "median_abs_im_sum": float(np.median(np.abs(im43 + im46))) if len(pairs) else float("nan"),
        "interpretation": "Two z-distinct conjugate/symmetric saddles with nearly identical Re action over their overlap; merge for exponent envelopes, keep both for amplitude/interference questions.",
    }


def write_report(summary: dict[str, Any]) -> Path:
    report = OUT / "groundbreaking_push_report.md"
    lines = [
        "# BM24 saddle push-further diagnostics",
        "",
        "Generated from existing audit artifacts; no new root searches were run.",
        "",
        "## Headline",
        "",
        (
            "The strongest empirical story is an explanatory saddle transition: "
            "the BM24 seed saddle controls in the QAOA-relevant window, while "
            "the branch 43/46 conjugate pair controls the large-|gamma| plateau."
        ),
        "",
        "## Key numerical outputs",
        "",
    ]
    anti = summary["anti_stokes"]
    drift = summary["finite_n_drift"]
    scatter = summary["algebraic_vs_physical"]
    plateau = summary["large_gamma_plateau"]
    pair = summary["branch_43_46_pair"]
    lines.extend(
        [
            f"- Primary Anti-Stokes DeltaRe crossing near QAOA window: {anti['primary_crossing_near_qaoa_window']:.6f}",
            f"- Other DeltaRe zeroes seen in the onset window: {anti['crossings_from_delta_re']}",
            f"- DeltaIm mod 2pi range on crossing plot: {anti['min_delta_im_mod']:.3g} to {anti['max_delta_im_mod']:.3g}",
            f"- Finite-n drift linear extrapolation from sparse grid: gamma_cross(infty) ~= {drift['linear_fit_gamma_cross_at_infinite_n']:.4f}",
            f"- Scatter points: {scatter['num_competitor_points']} certified competitors; {scatter['num_high_re_bad_exact_match']} high-Re roots with bad exact-exponent match.",
            f"- Large-gamma RMSE, merged 43/46: {plateau['merged_rmse_large_gamma']:.5f}; seed: {plateau['seed_rmse_large_gamma']:.5f}.",
            f"- Branch 43/46 shared gamma count: {pair['shared_gamma_count']}; median |Re43-Re46|: {pair['median_abs_re_difference']:.3g}.",
        ]
    )
    lines.extend(
        [
            "",
            "## Interpretation of 43/46",
            "",
            (
                "The dominance is not challenged by two unrelated saddles. "
                "Branches 43 and 46 are best treated as a conjugate/symmetric pair: "
                "they are distinct z-roots, but they have nearly the same real action "
                "over the overlap. For the exponential rate, their shared Re Phi is "
                "what matters; for prefactors and oscillatory interference, the pair "
                "should remain distinct."
            ),
            "",
            "## Generated figures",
            "",
        ]
    )
    for key, val in summary.items():
        if isinstance(val, dict) and "path" in val:
            lines.append(f"- `{key}`: `{val['path']}`")
    lines.append("")
    report.write_text("\n".join(lines), encoding="utf-8")
    return report


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    phi_pref = PHI_PREF_DEFAULT
    push_path = RESULTS / "push_further_analysis.json"
    if push_path.exists():
        phi_pref = float(load_json(push_path)["params"]["phi_pref_conv2"])

    summary = {
        "physical_phase_diagram": plot_physical_dominance_phase_diagram(phi_pref),
        "anti_stokes": plot_anti_stokes_crossing(phi_pref),
        "finite_n_drift": plot_finite_n_crossing_drift(),
        "algebraic_vs_physical": plot_algebraic_vs_physical_scatter(phi_pref),
        "large_gamma_plateau": plot_large_gamma_plateau(phi_pref),
        "branch_43_46_pair": plot_branch_43_46_pair(phi_pref),
    }
    summary["report_path"] = str(write_report(summary))
    with (OUT / "groundbreaking_push_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
