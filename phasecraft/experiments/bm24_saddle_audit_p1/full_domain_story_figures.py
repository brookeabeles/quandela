"""Full-domain BM24 saddle story figures.

The existing sweep uses negative QAOA gamma values.  These figures present the
same data on the nonnegative domain

    beta in [0, 2*pi],   Gamma = -gamma in [0, 2*pi].

That makes the plots read from small Gamma upward, across the full beta range.
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
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
OUT = RESULTS / "full_domain_story"

TWOPI = 2.0 * math.pi
PHI_PREF = -0.6896093750000001
BETA_OPT = 0.5434
GAMMA_CROSS_LOCAL = 1.830842334140663  # plotted Gamma = -gamma
GAMMA_PAIR_TOL = 0.011

BLUE = "#2b6cb0"
ORANGE = "#dd6b20"
GREEN = "#2f855a"
RED = "#c53030"
PURPLE = "#6b46c1"
GRAY = "#4a5568"
LIGHT_BLUE = "#cfe8ff"
LIGHT_ORANGE = "#ffd9b3"


def load_json(path: Path) -> Any:
    with path.open() as f:
        return json.load(f)


def savefig(fig: plt.Figure, name: str) -> str:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def im_mod_zero(x: float) -> float:
    return abs((x + math.pi) % (2.0 * math.pi) - math.pi)


def interp(gamma: float, xs: np.ndarray, ys: np.ndarray) -> float:
    order = np.argsort(xs)
    return float(np.interp(gamma, xs[order], ys[order]))


def sweep_rows() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    data = load_json(RESULTS / "sweep_beta_gamma_full" / "sweep_beta_gamma_full.json")
    return data["metadata"], [r for r in data["rows"] if r.get("status") == "ok"]


def eff_gap(row: dict[str, Any]) -> float:
    seed_gap = abs(float(row["seed_gap"]))
    best_gap = float(row.get("best_match_abs_gap", float("nan")))
    if bool(row["seed_is_best_match_to_exact"]) or math.isnan(best_gap):
        return seed_gap
    return best_gap


def comp_gap(row: dict[str, Any]) -> float:
    return float(row.get("best_match_abs_gap", float("nan")))


def build_sweep_grids() -> dict[str, np.ndarray]:
    _, rows = sweep_rows()
    betas = np.array(sorted(set(float(r["beta"]) for r in rows)))
    gammas_abs = np.array(sorted(set(-float(r["gamma"]) for r in rows)))
    dom = np.full((len(gammas_abs), len(betas)), np.nan)
    gap = np.full_like(dom, np.nan)
    decoy = np.zeros_like(dom)
    ncomp = np.full_like(dom, np.nan)

    for r in rows:
        b = float(r["beta"])
        G = -float(r["gamma"])
        i = int(np.where(np.isclose(gammas_abs, G))[0][0])
        j = int(np.where(np.isclose(betas, b))[0][0])
        dom[i, j] = 0.0 if bool(r["seed_is_best_match_to_exact"]) else 1.0
        gap[i, j] = eff_gap(r)
        decoy[i, j] = 1.0 if bool(r["seed_is_best_match_to_exact"]) and float(r.get("max_competitor_delta_re", 0.0)) > 0.25 else 0.0
        ncomp[i, j] = float(r.get("n_certified_competitors", 0.0))
    return {
        "betas": betas,
        "Gamma": gammas_abs,
        "dominance": dom,
        "gap": gap,
        "decoy": decoy,
        "ncomp": ncomp,
    }


def load_seed() -> dict[str, np.ndarray]:
    data = load_json(RESULTS / "run_seed_branch_g-2pi" / "seed_branch_continuation.json")
    rows = [r for r in data["continuation"] if r.get("certified") and not r.get("failed")]
    return {
        "gamma": np.array([float(r["gamma"]) for r in rows]),
        "Gamma": np.array([-float(r["gamma"]) for r in rows]),
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

    pair_re = np.array(pair_re)
    gamma = np.array(gammas)
    return {
        "gamma": gamma,
        "Gamma": -gamma,
        "pair_re": pair_re,
        "pair_exp": PHI_PREF + pair_re,
        "pair_im_mod": np.array(pair_im_mod),
        "seed_exp": np.array(seed_exp),
        "seed_re": np.array(seed_re),
        "exact": np.array(exact),
        "delta_re": pair_re - np.array(seed_re),
    }


def full_domain_phase_map() -> str:
    grids = build_sweep_grids()
    summaries = load_json(RESULTS / "global_beta_robustness" / "global_beta_robustness_summary.json")["summaries"]
    beta = grids["betas"]
    Gamma = grids["Gamma"]
    dom = grids["dominance"]

    fig, ax = plt.subplots(figsize=(10.8, 7.2))
    cmap = mcolors.ListedColormap([LIGHT_BLUE, LIGHT_ORANGE])
    ax.pcolormesh(beta, Gamma, dom, cmap=cmap, vmin=0, vmax=1, shading="nearest")

    robust = [s for s in summaries if s["regime"] == "robust_seed_to_competitor"]
    weak = [s for s in summaries if s["regime"] == "weak_seed_to_competitor"]
    ambiguous = [s for s in summaries if s["regime"] == "nonmonotone_or_ambiguous"]
    for s in weak:
        tr = s.get("primary_transition")
        if tr:
            ax.errorbar(s["beta"], -tr["gamma_mid"], yerr=tr["uncertainty"], fmt="o", color="#b7791f", capsize=3, ms=6, alpha=0.85)
    for s in ambiguous:
        tr = s.get("primary_transition")
        if tr:
            ax.errorbar(s["beta"], -tr["gamma_mid"], yerr=tr["uncertainty"], fmt="o", color=RED, capsize=3, ms=6, alpha=0.75)
    for s in robust:
        tr = s["primary_transition"]
        ax.errorbar(s["beta"], -tr["gamma_mid"], yerr=tr["uncertainty"], fmt="o", color=GREEN, capsize=4, ms=8)
    rb = np.array([s["beta"] for s in robust])
    rG = np.array([-s["primary_transition"]["gamma_mid"] for s in robust])
    order = np.argsort(rb)
    ax.plot(rb[order], rG[order], color=GREEN, lw=2.5, label="clean transition band")

    ax.axvline(BETA_OPT, color=PURPLE, ls=":", lw=2.0, label=r"$\beta_{opt}$")
    ax.axhline(GAMMA_CROSS_LOCAL, color=GRAY, ls="--", lw=1.4, label=r"$\Gamma_{cross}\approx1.83$")
    ax.scatter([BETA_OPT], [0.749], marker="*", s=230, color="gold", edgecolor="k", zorder=8, label="QAOA angle")
    ax.text(0.18, 0.28, "small gamma:\nseed region", color=BLUE, fontsize=11, va="bottom")
    ax.text(0.18, 5.9, "large gamma:\ncompetitor-pair/ambiguous regions", color=ORANGE, fontsize=11, va="top")

    ax.set_xlim(0, TWOPI)
    ax.set_ylim(0, TWOPI)
    ax.set_xlabel(r"$\beta \in [0,2\pi]$")
    ax.set_ylabel(r"$\Gamma=-\gamma \in [0,2\pi]$")
    ax.set_title("Full-domain saddle story: seed vs physical competitor")
    ax.set_xticks([0, math.pi / 2, math.pi, 3 * math.pi / 2, TWOPI])
    ax.set_xticklabels([r"$0$", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"])
    ax.set_yticks([0, math.pi / 2, math.pi, 3 * math.pi / 2, TWOPI])
    ax.set_yticklabels([r"$0$", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"])
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.16)
    return savefig(fig, "full_01_phase_map_beta_gamma_0_2pi.png")


def full_domain_residual_map() -> str:
    grids = build_sweep_grids()
    beta = grids["betas"]
    Gamma = grids["Gamma"]
    gap = grids["gap"]
    decoy = grids["decoy"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8), sharey=True)
    ax = axes[0]
    im = ax.pcolormesh(
        beta,
        Gamma,
        np.clip(gap, 1e-4, 0.2),
        norm=mcolors.LogNorm(vmin=1e-4, vmax=0.2),
        cmap="viridis_r",
        shading="nearest",
    )
    ax.contour(beta, Gamma, gap, levels=[0.002, 0.005, 0.01, 0.025, 0.05], colors="white", linewidths=0.8)
    ax.scatter([BETA_OPT], [0.749], marker="*", s=170, color="gold", edgecolor="k", zorder=5)
    ax.set_title("How well the best explanatory saddle matches exact finite-n")
    fig.colorbar(im, ax=ax, label="absolute exponent residual")

    ax = axes[1]
    ax.pcolormesh(beta, Gamma, decoy, cmap=mcolors.ListedColormap(["white", "#c4b5fd"]), vmin=0, vmax=1, shading="nearest")
    ax.scatter([BETA_OPT], [0.749], marker="*", s=170, color="gold", edgecolor="k", zorder=5)
    ax.set_title("Where high-Re certified roots are decoys while seed still wins")

    for ax in axes:
        ax.set_xlim(0, TWOPI)
        ax.set_ylim(0, TWOPI)
        ax.set_xlabel(r"$\beta$")
        ax.set_xticks([0, math.pi / 2, math.pi, 3 * math.pi / 2, TWOPI])
        ax.set_xticklabels([r"$0$", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"])
        ax.grid(True, alpha=0.16)
    axes[0].set_ylabel(r"$\Gamma=-\gamma$")
    axes[0].set_yticks([0, math.pi / 2, math.pi, 3 * math.pi / 2, TWOPI])
    axes[0].set_yticklabels([r"$0$", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"])
    fig.suptitle("Full-domain reliability checks")
    return savefig(fig, "full_02_residual_and_decoy_maps.png")


def beta_opt_full_gamma_story() -> str:
    pair = load_pair_series()
    order = np.argsort(pair["Gamma"])
    G = pair["Gamma"][order]
    exact = pair["exact"][order]
    seed = pair["seed_exp"][order]
    comp = pair["pair_exp"][order]
    delta = pair["delta_re"][order]

    fig, axes = plt.subplots(3, 1, figsize=(10.5, 9.5), sharex=True, gridspec_kw={"height_ratios": [1.5, 0.9, 0.9]})
    ax = axes[0]
    ax.plot(G, exact, color="black", lw=2.6, label="exact finite-n exponent")
    ax.plot(G, seed, color=BLUE, lw=2.0, label="BM24 seed")
    ax.plot(G, comp, color=ORANGE, lw=2.0, label="43/46 pair")
    ax.axvline(GAMMA_CROSS_LOCAL, color=GRAY, ls="--", lw=1.4)
    ax.axhline(PHI_PREF, color="0.4", ls=":", lw=1.0, label=r"$\phi_{pref}$ plateau")
    ax.set_ylabel("exponent")
    ax.set_title(r"Full $\Gamma\in[0,2\pi]$ mechanism at $\beta\approx0.5434$")
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(True, alpha=0.2)

    ax = axes[1]
    ax.plot(G, delta, color=RED, lw=2.0)
    ax.axhline(0, color="black", lw=0.8)
    ax.axvline(GAMMA_CROSS_LOCAL, color=GRAY, ls="--", lw=1.4)
    ax.set_ylabel(r"$Re\Phi_{43/46}-Re\Phi_{seed}$")
    ax.grid(True, alpha=0.2)

    ax = axes[2]
    ax.semilogy(G, np.abs(exact - seed), color=BLUE, lw=2, label="|exact - seed|")
    ax.semilogy(G, np.abs(exact - comp), color=ORANGE, lw=2, label="|exact - 43/46|")
    ax.axvline(GAMMA_CROSS_LOCAL, color=GRAY, ls="--", lw=1.4)
    ax.set_ylabel("residual")
    ax.set_xlabel(r"$\Gamma=-\gamma$")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)
    ax.set_xlim(0, TWOPI)
    ax.set_xticks([0, math.pi / 2, math.pi, 3 * math.pi / 2, TWOPI])
    ax.set_xticklabels([r"$0$", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"])
    return savefig(fig, "full_03_beta_opt_gamma_0_2pi_mechanism.png")


def full_domain_beta_cuts() -> str:
    _, rows = sweep_rows()
    targets = [0.4, BETA_OPT, 0.8, math.pi / 2, math.pi, 5.0]
    betas = sorted(set(float(r["beta"]) for r in rows))
    fig, axes = plt.subplots(len(targets), 1, figsize=(10.8, 2.25 * len(targets)), sharex=True)
    for ax, target in zip(axes, targets):
        b = min(betas, key=lambda x: abs(x - target))
        sub = sorted([r for r in rows if abs(float(r["beta"]) - b) < 1e-9], key=lambda r: -float(r["gamma"]))
        G = np.array([-float(r["gamma"]) for r in sub])
        seed_gap = np.array([abs(float(r["seed_gap"])) for r in sub])
        best_gap = np.array([comp_gap(r) for r in sub])
        ax.semilogy(G, seed_gap, "o-", color=BLUE, lw=1.6, ms=4, label="seed residual")
        ax.semilogy(G, best_gap, "s--", color=ORANGE, lw=1.4, ms=4, label="best competitor residual")
        ax.axvline(GAMMA_CROSS_LOCAL, color=GRAY, ls=":", lw=1.0)
        ax.set_ylabel(rf"$\beta={b:.4g}$")
        ax.grid(True, alpha=0.2)
        ax.legend(fontsize=7, loc="upper right")
    axes[-1].set_xlabel(r"$\Gamma=-\gamma$")
    axes[-1].set_xlim(0, TWOPI)
    axes[-1].set_xticks([0, math.pi / 2, math.pi, 3 * math.pi / 2, TWOPI])
    axes[-1].set_xticklabels([r"$0$", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"])
    fig.suptitle("Full-domain beta cuts: residual race from small Gamma to 2pi")
    return savefig(fig, "full_04_beta_cuts_gamma_0_2pi.png")


def full_domain_regime_strip() -> str:
    summaries = load_json(RESULTS / "global_beta_robustness" / "global_beta_robustness_summary.json")["summaries"]
    color = {
        "robust_seed_to_competitor": GREEN,
        "weak_seed_to_competitor": "#d69e2e",
        "nonmonotone_or_ambiguous": RED,
        "always_seed_on_grid": BLUE,
        "always_competitor_on_grid": ORANGE,
    }
    label = {
        "robust_seed_to_competitor": "clean transition",
        "weak_seed_to_competitor": "coarse transition",
        "nonmonotone_or_ambiguous": "ambiguous",
        "always_seed_on_grid": "seed throughout grid",
        "always_competitor_on_grid": "competitor throughout grid",
    }
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.6), sharex=True, gridspec_kw={"height_ratios": [0.75, 1.35]})
    ax = axes[0]
    for s in summaries:
        ax.scatter(s["beta"], 0, s=120, color=color[s["regime"]], edgecolor="white", linewidth=0.8)
    present = {s["regime"] for s in summaries}
    handles = [plt.Line2D([0], [0], marker="o", lw=0, color=c, label=label[k], markersize=8) for k, c in color.items() if k in present]
    ax.legend(handles=handles, ncol=3, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, 1.45))
    ax.axvline(BETA_OPT, color=PURPLE, ls=":", lw=1.6)
    ax.set_yticks([])
    ax.set_title("Beta-regime summary on the full [0, 2pi] domain")
    ax.grid(True, axis="x", alpha=0.18)

    ax = axes[1]
    for s in summaries:
        tr = s.get("primary_transition")
        if not tr:
            continue
        ax.errorbar(s["beta"], -tr["gamma_mid"], yerr=tr["uncertainty"], fmt="o", color=color[s["regime"]], capsize=3, ms=7)
    ax.axvline(BETA_OPT, color=PURPLE, ls=":", lw=1.6)
    ax.axhline(GAMMA_CROSS_LOCAL, color=GRAY, ls="--", lw=1.2)
    ax.set_xlim(0, TWOPI)
    ax.set_ylim(0, TWOPI)
    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel(r"primary transition $\Gamma=-\gamma$")
    ax.set_xticks([0, math.pi / 2, math.pi, 3 * math.pi / 2, TWOPI])
    ax.set_xticklabels([r"$0$", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"])
    ax.set_yticks([0, math.pi / 2, math.pi, 3 * math.pi / 2, TWOPI])
    ax.set_yticklabels([r"$0$", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"])
    ax.grid(True, alpha=0.2)
    return savefig(fig, "full_05_beta_regime_strip_0_2pi.png")


def write_report(paths: dict[str, str]) -> str:
    OUT.mkdir(parents=True, exist_ok=True)
    report = OUT / "full_domain_story_report.md"
    lines = [
        "# Full-Domain BM24 Saddle Story",
        "",
        "All plots use beta in [0, 2pi] and Gamma = -gamma in [0, 2pi].",
        "The underlying sweep starts at Gamma = 0.05 rather than exactly zero; the axes are shown on the full domain.",
        "",
        "## Main Message",
        "",
        "The clean QAOA-band transition is visible inside the full beta/gamma domain rather than only in a zoomed plot.",
        "At beta approximately 0.5434, the mechanism remains the same: the seed controls at small Gamma, then the 43/46 pair controls after Gamma about 1.83.",
        "Away from that beta band, the full-domain view separates clean transitions from coarse, ambiguous, and symmetry-protected regimes.",
        "",
        "## Figures",
        "",
    ]
    for name, path in paths.items():
        lines.append(f"- `{name}`: `{path}`")
    report.write_text("\n".join(lines), encoding="utf-8")
    return str(report)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    paths = {
        "full_01_phase_map": full_domain_phase_map(),
        "full_02_residual_and_decoy_maps": full_domain_residual_map(),
        "full_03_beta_opt_mechanism": beta_opt_full_gamma_story(),
        "full_04_beta_cuts": full_domain_beta_cuts(),
        "full_05_beta_regime_strip": full_domain_regime_strip(),
    }
    report = write_report(paths)
    summary = {"figures": paths, "report": report}
    (OUT / "full_domain_story_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
