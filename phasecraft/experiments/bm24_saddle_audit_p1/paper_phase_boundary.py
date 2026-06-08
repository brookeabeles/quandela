"""
C: Certified beta-gamma phase boundary.

Extracts gamma_cross(beta) from the existing 600-point sweep, interpolates
to get the certified anti-Stokes boundary curve, and generates a publication-
quality phase diagram.

Results: phasecraft/results/bm24_saddle_audit_p1/paper_results/phase_boundary_*.{json,png}
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from scipy.interpolate import make_interp_spline

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SWEEP_JSON = (
    REPO_ROOT
    / "phasecraft" / "results" / "bm24_saddle_audit_p1"
    / "sweep_beta_gamma_full" / "sweep_beta_gamma_full.json"
)
OUT_DIR = REPO_ROOT / "phasecraft" / "results" / "bm24_saddle_audit_p1" / "paper_results"

PHI_PREF = -0.689609375


def load_sweep():
    with open(SWEEP_JSON) as f:
        data = json.load(f)
    return data["metadata"], data["rows"]


def extract_boundary(rows, beta_grid):
    """
    For each beta, find gamma_cross as the midpoint of the last seed-is-best gamma
    and the first competitor-is-best gamma.  Returns list of (beta, gamma_cross, uncertainty).
    """
    boundary = []
    for beta in beta_grid:
        rows_b = [
            r for r in rows
            if abs(r["beta"] - beta) < 1e-6 and r.get("status") == "ok"
        ]
        # Iterate from least-negative to most-negative gamma (seed→crossing→competitor)
        rows_b.sort(key=lambda r: r["gamma"], reverse=True)

        last_seed = None
        first_comp = None
        for r in rows_b:
            sib = r.get("seed_is_best_match_to_exact")
            if sib is None:
                continue
            if sib:
                last_seed = r["gamma"]
            elif last_seed is not None and first_comp is None:
                first_comp = r["gamma"]

        if last_seed is None or first_comp is None:
            # Either always seed or always competitor
            if last_seed is None:
                boundary.append({"beta": beta, "gamma_cross": 0.0, "uncertainty": None, "type": "always_competitor"})
            else:
                boundary.append({"beta": beta, "gamma_cross": 2 * math.pi, "uncertainty": None, "type": "always_seed"})
        else:
            gc = (last_seed + first_comp) / 2.0
            unc = abs(first_comp - last_seed) / 2.0
            boundary.append({"beta": beta, "gamma_cross": gc, "uncertainty": unc, "type": "crossed"})

    return boundary


def effective_gap_grid(rows, beta_grid, gamma_grid):
    """Build B x G matrix of effective gap (min(|seed_gap|, best_match_gap)) for heatmap."""
    B = len(beta_grid)
    G = len(gamma_grid)
    eff = np.full((B, G), np.nan)
    seed_best = np.zeros((B, G), dtype=bool)

    for r in rows:
        if r.get("status") != "ok":
            continue
        b_matches = [i for i, b in enumerate(beta_grid) if abs(b - r["beta"]) < 1e-6]
        g_matches = [j for j, g in enumerate(gamma_grid) if abs(g - r["gamma"]) < 1e-6]
        if not b_matches or not g_matches:
            continue
        bi, gi = b_matches[0], g_matches[0]
        sg = r.get("seed_gap", float("nan"))
        bm = r.get("best_match_abs_gap", float("nan"))
        sib = r.get("seed_is_best_match_to_exact", True)
        if sib:
            eff[bi, gi] = abs(sg)
        elif not math.isnan(bm):
            eff[bi, gi] = bm
        else:
            eff[bi, gi] = abs(sg)
        seed_best[bi, gi] = bool(sib)

    return eff, seed_best


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    meta, rows = load_sweep()
    beta_grid = meta["beta_grid"]
    gamma_grid = meta["gamma_grid"]

    print(f"Loaded {len(rows)} rows. Betas: {len(beta_grid)}, Gammas: {len(gamma_grid)}")

    # Extract phase boundary
    boundary = extract_boundary(rows, beta_grid)
    crossed = [b for b in boundary if b["type"] == "crossed"]
    print(f"Boundary: {len(crossed)} crossing betas, "
          f"{sum(1 for b in boundary if b['type']=='always_seed')} always-seed, "
          f"{sum(1 for b in boundary if b['type']=='always_competitor')} always-competitor")

    for b in crossed:
        print(f"  beta={b['beta']:.4f}  gamma_cross={b['gamma_cross']:.3f}  +/-{b['uncertainty']:.3f}")

    out_json = OUT_DIR / "phase_boundary.json"
    with open(out_json, "w") as f:
        json.dump({"boundary": boundary, "metadata": meta}, f, indent=2)
    print(f"Wrote {out_json}")

    # Effective gap grid
    eff_gap, seed_best = effective_gap_grid(rows, beta_grid, gamma_grid)
    G_arr = np.array(gamma_grid)
    B_arr = np.array(beta_grid)

    # -----------------------------------------------------------------------
    # Figure 1: Phase diagram with boundary overlay
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 7))

    # Categorical: seed (blue) vs competitor (orange) dominant
    dom_img = np.where(seed_best, 0.0, 1.0)
    extent = [G_arr[0], G_arr[-1], B_arr[0], B_arr[-1]]
    cmap_cat = mcolors.ListedColormap(["#4878D0", "#EE854A"])
    im = ax.imshow(
        dom_img, origin="lower", aspect="auto",
        extent=extent,
        cmap=cmap_cat, vmin=0, vmax=1, alpha=0.6,
    )

    # Effective gap contours on top
    eff_clip = np.clip(eff_gap, 1e-4, 0.1)
    from matplotlib.colors import LogNorm
    X, Y = np.meshgrid(G_arr, B_arr)
    valid = ~np.isnan(eff_clip)
    if valid.any():
        cs = ax.contour(
            X, Y, eff_clip,
            levels=[0.002, 0.005, 0.01, 0.05],
            colors=["#222222"],
            linewidths=[0.8, 1.0, 1.2, 1.5],
            alpha=0.7,
        )
        ax.clabel(cs, inline=True, fontsize=7, fmt="%.3f")

    # Phase boundary curve
    betas_cross = [b["beta"] for b in crossed]
    gammas_cross = [b["gamma_cross"] for b in crossed]
    uncertainties = [b["uncertainty"] for b in crossed]

    if len(betas_cross) >= 4:
        # Smooth spline through the boundary
        order = min(3, len(betas_cross) - 1)
        try:
            spl = make_interp_spline(betas_cross, gammas_cross, k=order)
            beta_fine = np.linspace(min(betas_cross), max(betas_cross), 200)
            gamma_fine = spl(beta_fine)
            ax.plot(gamma_fine, beta_fine, "k-", lw=2.5, label="Anti-Stokes boundary")
        except Exception:
            ax.plot(gammas_cross, betas_cross, "k-", lw=2.5, label="Anti-Stokes boundary")
    ax.errorbar(
        gammas_cross, betas_cross,
        xerr=uncertainties,
        fmt="ko", ms=4, capsize=3, label="Certified crossing points",
        zorder=5,
    )

    # Annotations
    ax.axhline(math.pi / 4, color="red", lw=1.5, ls="--", label=r"$\beta=\pi/4$ (death zone)")
    ax.axhline(0.5434, color="purple", lw=1.0, ls=":", label=r"$\beta_{\rm opt}=0.5434$")

    # Legend patches
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#4878D0", alpha=0.6, label="Seed dominant"),
        Patch(facecolor="#EE854A", alpha=0.6, label="Competitor dominant"),
        plt.Line2D([0], [0], color="k", lw=2.5, label="Anti-Stokes boundary"),
        plt.Line2D([0], [0], color="red", lw=1.5, ls="--", label=r"$\beta=\pi/4$"),
        plt.Line2D([0], [0], color="purple", lw=1.0, ls=":", label=r"$\beta_{\rm opt}$"),
    ]
    ax.legend(handles=legend_elements, fontsize=8, loc="upper right")

    ax.set_xlabel(r"$\gamma$")
    ax.set_ylabel(r"$\beta$")
    ax.set_title(
        r"Phase diagram: saddle dominance in $(\beta,\gamma)$ space"
        "\n" + r"$q=3,\,k=8,\,r=176.54$; contours = effective gap"
    )
    ax.set_xlim(G_arr[0], G_arr[-1])
    ax.set_ylim(B_arr[0], B_arr[-1])

    # Add pi tick labels on y-axis
    y_ticks = [0, math.pi/4, math.pi/2, math.pi, 3*math.pi/2, 2*math.pi]
    y_labels = [r"$0$", r"$\pi/4$", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"]
    ax.set_yticks([t for t in y_ticks if B_arr[0] <= t <= B_arr[-1]])
    ax.set_yticklabels([l for t, l in zip(y_ticks, y_labels) if B_arr[0] <= t <= B_arr[-1]])

    fig.tight_layout()
    fig.savefig(OUT_DIR / "phase_boundary_fig1_diagram.png", dpi=150)
    plt.close(fig)
    print("Saved phase_boundary_fig1_diagram.png")

    # -----------------------------------------------------------------------
    # Figure 2: gamma_cross vs beta with polynomial fit
    # -----------------------------------------------------------------------
    if len(crossed) >= 4:
        bc = np.array(betas_cross)
        gc = np.array(gammas_cross)
        unc = np.array(uncertainties)

        # Fit a polynomial in beta to gamma_cross
        # Use only the "crossed" region: typically beta < pi/4 and some range above
        deg = min(4, len(bc) - 1)
        poly = np.polyfit(bc, gc, deg)
        beta_fit = np.linspace(bc.min(), bc.max(), 200)
        gamma_fit = np.polyval(poly, beta_fit)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.errorbar(bc, gc, yerr=unc, fmt="ko", ms=5, capsize=4,
                    label="Certified $\\gamma_{\\rm cross}(\\beta)$")
        ax.plot(beta_fit, gamma_fit, "C0-", lw=2,
                label=f"Polynomial fit (deg {deg})")
        ax.axhline(-math.pi / 2, color="gray", lw=0.8, ls="--", label=r"$-\pi/2$")
        ax.axhline(-math.pi, color="gray", lw=0.8, ls="-.", label=r"$-\pi$")
        ax.axvline(math.pi / 4, color="red", lw=1.0, ls=":", label=r"$\pi/4$")
        ax.axvline(0.5434, color="purple", lw=1.0, ls=":", label=r"$\beta_{\rm opt}$")
        ax.set_xlabel(r"$\beta$")
        ax.set_ylabel(r"$\gamma_{\rm cross}(\beta)$")
        ax.set_title(
            r"Anti-Stokes boundary: $\gamma_{\rm cross}(\beta)$ with interpolated uncertainty"
        )
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUT_DIR / "phase_boundary_fig2_curve.png", dpi=150)
        plt.close(fig)
        print("Saved phase_boundary_fig2_curve.png")

        # Print polynomial
        print(f"\nPolynomial fit (degree {deg}) for gamma_cross(beta):")
        print("  " + " + ".join(f"{c:.4f}*beta^{deg-i}" for i, c in enumerate(poly)))

    # -----------------------------------------------------------------------
    # Figure 3: Effective gap along the boundary
    # -----------------------------------------------------------------------
    # What is the effective gap at the crossing points? (= transition zone width)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(bc, unc * 2, width=0.05, color="C3", alpha=0.7, label="Transition zone width (2 * uncertainty)")
    ax.axvline(math.pi / 4, color="red", lw=1.5, ls="--", label=r"$\pi/4$")
    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel(r"$\Delta\gamma_{\rm cross}$ (coarse-grid resolution)")
    ax.set_title(r"Width of the anti-Stokes transition zone vs $\beta$")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "phase_boundary_fig3_width.png", dpi=150)
    plt.close(fig)
    print("Saved phase_boundary_fig3_width.png")

    # -----------------------------------------------------------------------
    # Summary statistics
    # -----------------------------------------------------------------------
    print("\n=== Phase Boundary Summary ===")
    print(f"{'beta':>7} | {'gamma_cross':>12} | {'uncertainty':>11} | type")
    print("-" * 45)
    for b in boundary:
        gc_str = f"{b['gamma_cross']:+.3f}" if b["type"] == "crossed" else b["type"]
        unc_str = f"{b['uncertainty']:.3f}" if b["uncertainty"] is not None else "N/A"
        print(f"{b['beta']:7.4f} | {gc_str:>12} | {unc_str:>11} | {b['type']}")


if __name__ == "__main__":
    main()
