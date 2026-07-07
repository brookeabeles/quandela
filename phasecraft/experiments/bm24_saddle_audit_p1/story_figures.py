"""Clear story figures for the BM24 p=1 saddle audit.

These plots intentionally hide most diagnostic clutter.  Each figure answers
one question:

1. Where is the clean beta-band transition?
2. What happens at beta ~= 0.5434?
3. Are branches 43/46 two independent dominants or one conjugate rate family?
4. Why is max Re Phi the wrong criterion?
5. What is the global beta-regime picture?
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
OUT = RESULTS / "story_figures"

PHI_PREF = -0.6896093750000001
PRIMARY_GAMMA = -1.830842334140663
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
    return {
        "gamma": np.array(gammas),
        "pair_re": pair_re,
        "pair_exp": PHI_PREF + pair_re,
        "pair_im_mod": np.array(pair_im_mod),
        "seed_exp": np.array(seed_exp),
        "seed_re": np.array(seed_re),
        "exact": np.array(exact),
        "delta_re": pair_re - np.array(seed_re),
    }


def load_branch_pair_overlap() -> dict[str, np.ndarray]:
    resolved = load_json(
        RESULTS
        / "run_seed_branch_g-2pi"
        / "archive"
        / "competitor_dominance"
        / "z_robust_full"
        / "resolved_robust_z.json"
    )
    tabs: dict[int, dict[float, dict[str, float]]] = {}
    for bid in (43, 46):
        br = next(b for b in resolved["branches"] if int(b["branch_id"]) == bid)
        tabs[bid] = {
            round(float(p["gamma"]), 6): {
                "re": float(p["Phi_eff_real"]),
                "im": float(p["Phi_eff_imag"]),
            }
            for p in br["points"]
        }
    pairs = []
    used46 = set()
    for g43, r43 in sorted(tabs[43].items(), reverse=True):
        candidates = [(abs(g43 - g46), g46, r46) for g46, r46 in tabs[46].items() if g46 not in used46 and abs(g43 - g46) <= GAMMA_PAIR_TOL]
        if not candidates:
            continue
        _, g46, r46 = min(candidates, key=lambda t: t[0])
        used46.add(g46)
        pairs.append((0.5 * (g43 + g46), r43, r46))
    return {
        "gamma": np.array([p[0] for p in pairs]),
        "re43": np.array([p[1]["re"] for p in pairs]),
        "re46": np.array([p[2]["re"] for p in pairs]),
        "im43": np.array([p[1]["im"] for p in pairs]),
        "im46": np.array([p[2]["im"] for p in pairs]),
    }


def figure_1_transition_band() -> str:
    sweep = load_json(RESULTS / "sweep_beta_gamma_full" / "sweep_beta_gamma_full.json")
    global_summary = load_json(RESULTS / "global_beta_robustness" / "global_beta_robustness_summary.json")
    rows = [r for r in sweep["rows"] if r.get("status") == "ok"]
    betas = np.array(sorted(set(float(r["beta"]) for r in rows)))
    gammas = np.array(sorted(set(float(r["gamma"]) for r in rows), reverse=True))
    dom = np.full((len(betas), len(gammas)), np.nan)
    for r in rows:
        i = int(np.where(np.isclose(betas, float(r["beta"])))[0][0])
        j = int(np.where(np.isclose(gammas, float(r["gamma"])))[0][0])
        dom[i, j] = 0.0 if bool(r["seed_is_best_match_to_exact"]) else 1.0

    fig, ax = plt.subplots(figsize=(9.5, 6.2))
    cmap = mcolors.ListedColormap([LIGHT_BLUE, LIGHT_ORANGE])
    ax.pcolormesh(betas, gammas, dom.T, cmap=cmap, vmin=0, vmax=1, shading="nearest")

    robust = [s for s in global_summary["summaries"] if s["regime"] == "robust_seed_to_competitor"]
    weak_near = [s for s in global_summary["summaries"] if s["beta"] in (0.2, 0.6, 0.7)]
    for s in weak_near:
        tr = s.get("primary_transition")
        if tr:
            ax.errorbar(s["beta"], tr["gamma_mid"], yerr=tr["uncertainty"], fmt="o", color="#b7791f", capsize=4, ms=7)
    for s in robust:
        tr = s["primary_transition"]
        ax.errorbar(s["beta"], tr["gamma_mid"], yerr=tr["uncertainty"], fmt="o", color=GREEN, capsize=5, ms=8)
        ax.text(s["beta"], tr["gamma_mid"] - 0.09, f"{s['beta']:.3g}", color=GREEN, ha="center", fontsize=8)

    rb = np.array([s["beta"] for s in robust])
    rg = np.array([s["primary_transition"]["gamma_mid"] for s in robust])
    order = np.argsort(rb)
    ax.plot(rb[order], rg[order], color=GREEN, lw=2.2, label="clean transition band")
    ax.axvline(0.5434, color=PURPLE, ls=":", lw=2, label=r"$\beta_{opt}$")
    ax.axhline(PRIMARY_GAMMA, color=GRAY, ls="--", lw=1.4, label=r"local 43/46 crossing")

    ax.set_xlim(0.15, 0.78)
    ax.set_ylim(-2.15, -1.15)
    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel(r"$\gamma$")
    ax.set_title("Main story: the seed loses dominance along a clean QAOA-band transition")
    ax.text(0.18, -1.23, "blue: seed saddle\nexplains exact exponent", color=BLUE, fontsize=11, va="top")
    ax.text(0.18, -2.05, "orange: competitor pair\nexplains exact exponent", color=ORANGE, fontsize=11, va="bottom")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.18)
    return savefig(fig, "story_01_qaoa_band_transition.png")


def figure_2_beta_opt_mechanism() -> str:
    pair = load_pair_series()
    mask = (pair["gamma"] >= -2.12) & (pair["gamma"] <= -1.55)
    g = pair["gamma"][mask]
    order = np.argsort(g)
    g = g[order]
    exact = pair["exact"][mask][order]
    seed = pair["seed_exp"][mask][order]
    comp = pair["pair_exp"][mask][order]
    delta = pair["delta_re"][mask][order]

    fig, axes = plt.subplots(3, 1, figsize=(9.5, 9), sharex=True, gridspec_kw={"height_ratios": [1.4, 0.9, 0.9]})
    ax = axes[0]
    ax.plot(g, exact, color="black", lw=2.6, label="exact finite-n exponent")
    ax.plot(g, seed, color=BLUE, lw=2, label="BM24 seed")
    ax.plot(g, comp, color=ORANGE, lw=2, label="43/46 pair")
    ax.axvline(PRIMARY_GAMMA, color=GRAY, ls="--", lw=1.4)
    ax.set_ylabel("exponent")
    ax.set_title(r"At $\beta\approx0.5434$: exact exponent switches from seed to 43/46")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.22)

    ax = axes[1]
    ax.plot(g, delta, color=RED, lw=2.2)
    ax.axhline(0, color="black", lw=0.9)
    ax.axvline(PRIMARY_GAMMA, color=GRAY, ls="--", lw=1.4)
    ax.set_ylabel(r"$Re\Phi_{43/46}-Re\Phi_{seed}$")
    ax.text(-2.1, 0.07, "43/46 has larger rate", color=ORANGE, fontsize=9)
    ax.text(-1.74, -0.08, "seed has larger rate", color=BLUE, fontsize=9)
    ax.grid(True, alpha=0.22)

    ax = axes[2]
    ax.semilogy(g, np.abs(exact - seed), color=BLUE, lw=2, label="|exact - seed|")
    ax.semilogy(g, np.abs(exact - comp), color=ORANGE, lw=2, label="|exact - 43/46|")
    ax.axvline(PRIMARY_GAMMA, color=GRAY, ls="--", lw=1.4)
    ax.set_ylabel("residual")
    ax.set_xlabel(r"$\gamma$")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.22)
    return savefig(fig, "story_02_beta_opt_mechanism.png")


def figure_3_pair_identity() -> str:
    pair = load_branch_pair_overlap()
    g = pair["gamma"]
    order = np.argsort(g)
    g = g[order]
    re43 = pair["re43"][order]
    re46 = pair["re46"][order]
    imsum = pair["im43"][order] + pair["im46"][order]

    fig, axes = plt.subplots(2, 1, figsize=(9.5, 6.8), sharex=True)
    ax = axes[0]
    ax.plot(g, PHI_PREF + re43, color=GREEN, lw=2.4, label="branch 43")
    ax.plot(g, PHI_PREF + re46, color=ORANGE, lw=2.0, ls="--", label="branch 46")
    ax.set_ylabel("full exponent")
    ax.set_title("Branches 43 and 46 are two roots but one exponential rate")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.22)

    ax = axes[1]
    eps = 1e-16
    ax.semilogy(g, np.maximum(np.abs(re43 - re46), eps), color=RED, lw=2, label=r"$|Re\Phi_{43}-Re\Phi_{46}|$")
    ax.semilogy(g, np.maximum(np.abs(imsum), eps), color=PURPLE, lw=2, label=r"$|Im\Phi_{43}+Im\Phi_{46}|$")
    ax.set_ylabel("pairing error")
    ax.set_xlabel(r"$\gamma$")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.22)
    ax.text(
        0.02,
        0.05,
        "Conclusion: keep both for amplitude/phases;\nmerge them for the leading exponent.",
        transform=ax.transAxes,
        fontsize=10,
        bbox=dict(fc="white", ec="0.8", alpha=0.9),
    )
    return savefig(fig, "story_03_43_46_pair_identity.png")


def figure_4_decoy_filter() -> str:
    paper_style = {
        "font.size": 14,
        "axes.labelsize": 16,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 12,
    }
    lw = 2.5
    comp = load_json(
        RESULTS
        / "run_seed_branch_g-2pi"
        / "archive"
        / "competitor_dominance"
        / "competitor_dominance_summary.json"
    )
    seed = load_seed()
    rows = []
    for pg in comp["per_gamma"]:
        g = float(pg["gamma"])
        lam = interp(g, seed["gamma"], seed["exact"])
        for c in pg.get("competitors", []):
            re_phi = float(c["re_phi_m"])
            full = PHI_PREF + re_phi
            rows.append(
                (
                    float(c["signed_re_phi_gap_vs_seed"]),
                    abs(full - lam),
                    im_mod_zero(float(c["im_phi_m"])),
                )
            )
    arr = np.array(rows)
    delta, gap, immod = arr[:, 0], arr[:, 1], arr[:, 2]
    decoy = (delta > 0.25) & (gap > 0.1)
    good = gap < 0.02
    other = (~decoy) & (~good)
    # Conjugate pairs share the same Re Phi, so they project to one point here.
    good_unique = np.array(
        sorted({(round(d, 8), round(g, 12)) for d, g in zip(delta[good], gap[good])})
    )

    with plt.rc_context(paper_style):
        fig, ax = plt.subplots(figsize=(7.0, 4.5))
        ax.scatter(
            delta[other],
            gap[other],
            s=14,
            color="0.72",
            alpha=0.6,
            label=f"other certified roots ({int(other.sum())})",
        )
        ax.scatter(
            delta[decoy],
            gap[decoy],
            s=28,
            color=RED,
            alpha=0.8,
            label=f"high-Re decoys ({int(decoy.sum())})",
        )
        ax.scatter(
            good_unique[:, 0],
            good_unique[:, 1],
            s=55,
            color=GREEN,
            edgecolor="k",
            linewidth=0.4,
            label=f"exact-explanatory roots ({len(good_unique)})",
        )
        ax.axvline(0, color="black", lw=lw)
        ax.axhline(0.02, color=GREEN, ls="--", lw=lw)
        ax.set_yscale("log")
        x_lo, x_hi = float(delta.min()), float(delta.max())
        x_span = x_hi - x_lo
        x_pad_right = 0.35 * x_span
        xlim_left = x_lo - 0.04 * x_span
        xlim_right = x_hi + x_pad_right
        ax.set_xlim(xlim_left, xlim_right)
        ax.set_xlabel(r"$Re\,\phi_{\mathrm{competitor}} - Re\,\Phi_{\mathrm{seed}}$")
        ax.set_ylabel(r"$\lambda_{\mathrm{competitor}} - \lambda_{\mathrm{exact}}$")
        ax.text(
            xlim_right - 0.04 * x_span,
            0.13,
            "certified roots with larger Re\nbut wrong exponent",
            ha="right",
            va="center",
            color=RED,
            fontsize=10.5,
        )
        ax.text(
            xlim_left + 0.03 * x_span,
            0.022,
            "good physical match\nrequires small exact gap",
            ha="left",
            va="bottom",
            color=GREEN,
            fontsize=10.5,
        )
        handles, labels = ax.get_legend_handles_labels()
        ax.grid(True, alpha=0.22)
        ax.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.005),
            ncol=3,
            frameon=False,
            columnspacing=0.9,
            handletextpad=0.35,
            prop={"size": 10},
        )
        fig.tight_layout(rect=[0, 0, 1, 0.98])
    return savefig(fig, "story_04_decoy_filter.png")


def figure_5_global_regime_strip() -> str:
    data = load_json(RESULTS / "global_beta_robustness" / "global_beta_robustness_summary.json")
    summaries = data["summaries"]
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
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.8), sharex=True, gridspec_kw={"height_ratios": [0.8, 1.4]})
    ax = axes[0]
    for s in summaries:
        ax.scatter(s["beta"], 0, s=130, color=color[s["regime"]], edgecolor="white", linewidth=0.8)
    present = [s["regime"] for s in summaries]
    handles = [
        plt.Line2D([0], [0], marker="o", lw=0, color=c, label=label[k], markersize=8)
        for k, c in color.items()
        if k in present
    ]
    ax.legend(handles=handles, ncol=3, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, 1.45))
    ax.set_yticks([])
    ax.set_title("Global beta picture: clean band plus symmetry/ambiguous regions")
    ax.axvline(0.5434, color=PURPLE, ls=":", lw=1.5)
    ax.text(0.5434, -0.28, r"$\beta_{opt}$", color=PURPLE, ha="center", fontsize=9)
    ax.set_ylim(-0.4, 0.4)
    ax.grid(True, axis="x", alpha=0.18)

    ax = axes[1]
    for s in summaries:
        tr = s.get("primary_transition")
        if tr is None:
            continue
        ax.errorbar(
            s["beta"],
            tr["gamma_mid"],
            yerr=tr["uncertainty"],
            fmt="o",
            color=color[s["regime"]],
            capsize=3,
            ms=7,
        )
    ax.axvline(0.5434, color=PURPLE, ls=":", lw=1.5)
    ax.axhline(PRIMARY_GAMMA, color=GRAY, ls="--", lw=1.2)
    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel(r"primary transition $\gamma$")
    ax.grid(True, alpha=0.22)
    ax.text(0.42, -2.12, "clean QAOA-band\ntransition cluster", color=GREEN, fontsize=10)
    ax.set_ylim(-2.65, 0.05)
    return savefig(fig, "story_05_global_regime_strip.png")


def write_report(paths: dict[str, str]) -> str:
    out = OUT / "story_figures_report.md"
    lines = [
        "# Clear BM24 Saddle Story Figures",
        "",
        "These figures are simplified presentation layers over the existing diagnostics.",
        "",
        "## Story",
        "",
        "1. The cleanest result is a QAOA-band seed-to-competitor transition for beta around 0.4 to 0.5434.",
        "2. At beta approximately 0.5434, the local mechanism is a crossing near gamma = -1.830842.",
        "3. The post-crossing rate is carried by a 43/46 conjugate pair: two roots, one leading exponential rate.",
        "4. Larger Re Phi alone is not physical dominance; many certified high-Re roots fail the exact-exponent test.",
        "5. Globally in beta, the clean band is surrounded by coarse, ambiguous, and symmetry-protected regimes.",
        "",
        "## Figures",
        "",
    ]
    for k, p in paths.items():
        lines.append(f"- `{k}`: `{p}`")
    lines.append("")
    out.write_text("\n".join(lines), encoding="utf-8")
    return str(out)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    paths = {
        "story_01_qaoa_band_transition": figure_1_transition_band(),
        "story_02_beta_opt_mechanism": figure_2_beta_opt_mechanism(),
        "story_03_43_46_pair_identity": figure_3_pair_identity(),
        "story_04_decoy_filter": figure_4_decoy_filter(),
        "story_05_global_regime_strip": figure_5_global_regime_strip(),
    }
    report = write_report(paths)
    summary = {"figures": paths, "report": report}
    (OUT / "story_figures_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
