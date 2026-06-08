"""
Phase diagram plots for the (beta, gamma) saddle dominance sweep.

Answers the question: is there always a single saddle that closely tracks
lambda_abs (the multinomial sum), or does the approximation break down?

Figures:
  fig1_phase_diagram.png     — 4-panel: predictor identity, effective gap,
                               Re Phi landscape, n_competitors
  fig2_line_cuts.png         — seed_gap vs gamma for selected beta values,
                               alongside best-competitor gap
  fig3_exponent_landscape.png — Re Phi of seed vs lambda_abs for 4 beta cuts,
                                colored bands showing who controls
  fig4_effective_gap.png     — log-scale effective gap heatmap + crossover contours

Reads: phasecraft/results/bm24_saddle_audit_p1/sweep_beta_gamma_full/
       sweep_beta_gamma_full.json
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_FILE = (
    REPO_ROOT
    / "phasecraft" / "results" / "bm24_saddle_audit_p1"
    / "sweep_beta_gamma_full" / "sweep_beta_gamma_full.json"
)
OUT_DIR = DATA_FILE.parent

# ── load and pivot ─────────────────────────────────────────────────────────────

with open(DATA_FILE) as f:
    raw = json.load(f)

phi_pref = raw["metadata"]["phi_pref_conv2"]   # -0.6896
rows = [r for r in raw["rows"] if r.get("status") == "ok"]

betas_all  = sorted(set(r["beta"]  for r in rows))
gammas_all = sorted(set(r["gamma"] for r in rows))   # ascending (most-negative last)

B = len(betas_all)
G = len(gammas_all)
bi = {b: i for i, b in enumerate(betas_all)}
gi = {g: i for i, g in enumerate(gammas_all)}

nan = float("nan")

def arr(): return np.full((B, G), nan)

re_phi    = arr()   # Re Phi_M of seed
lam       = arr()   # lambda_abs (exact)
seed_gap_ = arr()   # full_conv2_seed - lambda_abs
full_seed = arr()   # full_conv2_seed
eff_gap   = arr()   # min(|seed_gap|, best_match_abs_gap)  — best predictor gap
bm_gap    = arr()   # best_match_abs_gap (competitor only)
seed_best_= arr()   # 1=seed best, 0=competitor best
n_comp    = arr()

for r in rows:
    b_i = bi[r["beta"]]
    g_i = gi[r["gamma"]]
    re_phi[b_i, g_i]     = r["re_phi_seed"]
    lam[b_i, g_i]        = r["lambda_abs"]
    seed_gap_[b_i, g_i]  = r["seed_gap"]
    full_seed[b_i, g_i]  = r["full_conv2_seed"]
    n_comp[b_i, g_i]     = r["n_certified_competitors"]

    sg = r["seed_gap"]
    bm = r.get("best_match_abs_gap", nan)
    sb = bool(r["seed_is_best_match_to_exact"])
    seed_best_[b_i, g_i] = 1.0 if sb else 0.0

    if sb:
        eff_gap[b_i, g_i] = abs(sg)
    elif bm == bm:   # not NaN
        eff_gap[b_i, g_i] = bm
    else:
        eff_gap[b_i, g_i] = abs(sg)

    bm_gap[b_i, g_i] = bm if bm == bm else nan

beta_arr  = np.array(betas_all)
gamma_arr = np.array(gammas_all)

# Reference lines at special beta values
BREF = [
    (math.pi / 4,     r"$\pi/4$"),
    (math.pi / 2,     r"$\pi/2$"),
    (math.pi,         r"$\pi$"),
    (3 * math.pi / 2, r"$3\pi/2$"),
    (2 * math.pi,     r"$2\pi$"),
]


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — 4-panel heatmaps
# ══════════════════════════════════════════════════════════════════════════════

fig1, axes = plt.subplots(2, 2, figsize=(17, 12))
fig1.suptitle(
    r"BM24 saddle dominance phase diagram   [$q=3,\;k=8,\;r=176.54$]"
    "\n30 β × 20 γ grid  |  250 competitor starts per point",
    fontsize=12, fontweight="bold",
)

def _heatmap(ax, Z, *, cmap, vmin=None, vmax=None, logscale=False,
             title="", cblabel=""):
    masked = np.ma.masked_invalid(Z)
    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad("0.82")
    norm = (mcolors.LogNorm(vmin=vmin, vmax=vmax) if logscale
            else mcolors.Normalize(vmin=vmin, vmax=vmax))
    im = ax.pcolormesh(beta_arr, gamma_arr, masked.T,
                       cmap=cmap_obj, norm=norm, shading="nearest")
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(cblabel, fontsize=8)
    for ref_b, lbl in BREF:
        ax.axvline(ref_b, color="white", lw=0.8, ls="--", alpha=0.55)
        ax.text(ref_b + 0.05, gamma_arr.max() + 0.05, lbl,
                color="white", fontsize=6.5, va="bottom")
    ax.set_xlabel(r"$\beta$  (rad)", fontsize=9)
    ax.set_ylabel(r"$\gamma$  (rad)", fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold")
    return im

# Panel A — predictor identity
_heatmap(axes[0, 0], seed_best_,
         cmap="RdBu_r", vmin=0, vmax=1,
         title=r"A — Who best predicts $\lambda_{abs}$?",
         cblabel="0 = competitor   /   1 = seed")
axes[0, 0].plot(0.5434, -0.749, "*", ms=14, color="gold", zorder=10,
                label="p=1 optimal (β,γ)")
axes[0, 0].legend(fontsize=7)

# Panel B — effective gap (log)
_heatmap(axes[0, 1], np.clip(eff_gap, 5e-5, 1.0),
         cmap="plasma_r", vmin=5e-5, vmax=0.5, logscale=True,
         title=r"B — $|\text{best predictor} - \lambda_{abs}|$  (log scale)",
         cblabel="Prediction gap   [bright = small = valid approx]")
# Overlay 5% accuracy contour
try:
    cs = axes[0, 1].contour(beta_arr, gamma_arr, eff_gap.T,
                            levels=[0.05], colors=["lime"], linewidths=[2])
    axes[0, 1].clabel(cs, fmt={0.05: "5% error"}, fontsize=7)
except Exception:
    pass
axes[0, 1].plot(0.5434, -0.749, "*", ms=14, color="gold", zorder=10)

# Panel C — Re Phi of seed
_heatmap(axes[1, 0], re_phi,
         cmap="RdYlGn", vmin=-1.5, vmax=0.25,
         title=r"C — Re $\Phi_M$ of BM24 seed saddle",
         cblabel=r"Re $\Phi_M$   [green = contributing saddle]")
axes[1, 0].plot(0.5434, -0.749, "*", ms=14, color="gold", zorder=10)

# Panel D — n certified competitors
_heatmap(axes[1, 1], np.clip(n_comp, 0, 110),
         cmap="viridis",
         title="D — Number of certified competing saddles found",
         cblabel="Count  (out of 250 starts)")
axes[1, 1].plot(0.5434, -0.749, "*", ms=14, color="gold", zorder=10)

fig1.tight_layout(rect=[0, 0, 1, 0.95])
fig1.savefig(OUT_DIR / "fig1_phase_diagram.png", dpi=160, bbox_inches="tight")
print(f"Saved fig1_phase_diagram.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — line cuts: who tracks lambda_abs as gamma varies
# Shows (a) seed gap, (b) best competitor gap, (c) effective (minimum) gap
# ══════════════════════════════════════════════════════════════════════════════

target_betas = [0.2, 0.3, 0.4, 0.5, 0.5434, 0.6, 0.7, 0.8, 0.9, 1.0]
colors = plt.cm.RdYlGn(np.linspace(0.05, 0.95, len(target_betas)))

fig2, axes2 = plt.subplots(3, 1, figsize=(13, 14), sharex=True)
fig2.suptitle(
    r"Saddle prediction of $\lambda_{abs}$ vs $\gamma$ — selected $\beta$ values",
    fontsize=12, fontweight="bold",
)

for color, tb in zip(colors, target_betas):
    b_idx = min(range(B), key=lambda i: abs(betas_all[i] - tb))
    b_val = betas_all[b_idx]
    label = rf"$\beta={b_val:.4f}$"
    sg = seed_gap_[b_idx, :]
    eg = eff_gap[b_idx, :]
    bm = bm_gap[b_idx, :]

    axes2[0].plot(gamma_arr, sg, "-o", ms=3.5, lw=1.8, color=color, label=label)
    axes2[1].plot(gamma_arr, np.abs(bm), "--s", ms=3, lw=1.3, color=color, label=label)
    axes2[2].plot(gamma_arr, np.abs(eg), "-^", ms=3.5, lw=1.8, color=color, label=label)

for ax in axes2:
    ax.axhline(0, color="k", lw=0.7, ls="--", alpha=0.35)
    ax.set_xlim(gamma_arr[0] - 0.15, gamma_arr[-1] + 0.15)

axes2[0].set_ylim(-1.4, 0.35)
axes2[0].set_ylabel(r"Seed gap  $= E_\mathrm{seed} - \lambda_{abs}$", fontsize=10)
axes2[0].set_title("Seed prediction error  (0 = seed perfectly tracks)", fontsize=10)
axes2[0].legend(loc="lower left", fontsize=7, ncol=2)

axes2[1].set_yscale("symlog", linthresh=0.01)
axes2[1].set_ylim(-0.001, 2.0)
axes2[1].set_ylabel(r"$|E_\mathrm{best\,competitor} - \lambda_{abs}|$", fontsize=10)
axes2[1].set_title("Best competitor prediction error  (tracks when small)", fontsize=10)
axes2[1].legend(loc="upper right", fontsize=7, ncol=2)

axes2[2].set_yscale("symlog", linthresh=0.005)
axes2[2].set_ylim(-0.0005, 1.5)
axes2[2].set_ylabel(r"$\min(|\text{seed gap}|,\;|\text{comp gap}|)$", fontsize=10)
axes2[2].set_title(
    r"Effective saddle approximation error — best of {seed, competitors}"
    "\n→ if this is small, SOME single saddle always controls",
    fontsize=10,
)
axes2[2].set_xlabel(r"$\gamma$  (rad)", fontsize=10)
axes2[2].legend(loc="upper right", fontsize=7, ncol=2)

# Shade the 5% error band
for ax in axes2[2:]:
    ax.axhspan(0, 0.05, alpha=0.08, color="green", label="<5% error")

fig2.tight_layout(rect=[0, 0, 1, 0.96])
fig2.savefig(OUT_DIR / "fig2_line_cuts.png", dpi=160, bbox_inches="tight")
print(f"Saved fig2_line_cuts.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — exponent trajectories: seed vs lambda_abs for 6 beta cuts
# Colored bands: blue where seed ≈ lambda (gap < 0.05), red where competitor ≈ lambda
# ══════════════════════════════════════════════════════════════════════════════

target_betas_3 = [0.2, 0.4, 0.5434, 0.7, 0.8, 1.2]
fig3, axs3 = plt.subplots(2, 3, figsize=(17, 10))
fig3.suptitle(
    r"Saddle exponent trajectories: seed vs $\lambda_{abs}$  — 6 $\beta$ cuts"
    "\n"
    r"Blue band: seed tracks $\lambda_{abs}$  |  "
    r"Red band: a competitor tracks $\lambda_{abs}$  |  "
    r"Gray: no saddle closely predicts",
    fontsize=11, fontweight="bold",
)

for ax, tb in zip(axs3.flat, target_betas_3):
    b_idx = min(range(B), key=lambda i: abs(betas_all[i] - tb))
    b_val = betas_all[b_idx]

    g_  = gamma_arr
    fcs = full_seed[b_idx, :]
    lam_= lam[b_idx, :]
    sg  = seed_gap_[b_idx, :]
    sb  = seed_best_[b_idx, :]
    eg  = eff_gap[b_idx, :]

    ax.plot(g_, lam_, "k-",  lw=2.8, label=r"$\lambda_{abs}$ (exact)", zorder=5)
    ax.plot(g_, fcs,  "b--", lw=1.8, alpha=0.85, label=r"$E_\mathrm{seed}$", zorder=4)
    ax.axhline(phi_pref, color="gray", lw=0.7, ls=":", alpha=0.55)
    ax.text(g_[0] + 0.15, phi_pref + 0.02, r"$\phi_\mathrm{pref}$",
            color="gray", fontsize=7)

    # Shade regions by who controls
    for gi_idx in range(G - 1):
        xL, xR = g_[gi_idx], g_[gi_idx + 1]
        xspan = (xL, xR)
        eff = eg[gi_idx] if eg[gi_idx] == eg[gi_idx] else 1.0
        if sb[gi_idx] > 0.5 and eff < 0.05:
            ax.axvspan(xL, xR, alpha=0.15, color="steelblue", lw=0)
        elif sb[gi_idx] < 0.5 and eff < 0.05:
            ax.axvspan(xL, xR, alpha=0.15, color="tomato", lw=0)
        else:
            ax.axvspan(xL, xR, alpha=0.10, color="gray", lw=0)

    ax.set_xlabel(r"$\gamma$ (rad)", fontsize=9)
    ax.set_ylabel("Conv2 exponent", fontsize=9)
    ax.set_title(rf"$\beta = {b_val:.4f}$", fontsize=11, fontweight="bold")
    ax.legend(fontsize=7, loc="lower right")
    ax.set_xlim(g_[0] - 0.1, g_[-1] + 0.1)

    ylo = min(np.nanmin(fcs), np.nanmin(lam_), phi_pref) - 0.15
    yhi = max(np.nanmax(fcs), np.nanmax(lam_)) + 0.15
    ax.set_ylim(ylo, yhi)

fig3.tight_layout(rect=[0, 0, 1, 0.94])
fig3.savefig(OUT_DIR / "fig3_exponent_landscape.png", dpi=160, bbox_inches="tight")
print(f"Saved fig3_exponent_landscape.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — 2-panel: effective gap heatmap (log) + dominant predictor + 5% contour
# This is the "does the approximation always hold" figure
# ══════════════════════════════════════════════════════════════════════════════

fig4, axes4 = plt.subplots(1, 2, figsize=(17, 7))
fig4.suptitle(
    r"Does a single saddle always control $\lambda_{abs}$?"
    "\n"
    r"Effective prediction gap  $|\text{best predictor} - \lambda_{abs}|$",
    fontsize=12, fontweight="bold",
)

# Left — log-scale effective gap heatmap
ax = axes4[0]
masked_eg = np.ma.masked_invalid(eff_gap)
cmap_eg   = plt.get_cmap("inferno_r").copy()
cmap_eg.set_bad("0.75")
norm_eg = mcolors.LogNorm(vmin=5e-5, vmax=0.5)
im4 = ax.pcolormesh(beta_arr, gamma_arr, masked_eg.T,
                    cmap=cmap_eg, norm=norm_eg, shading="nearest")
cb4 = plt.colorbar(im4, ax=ax)
cb4.set_label(r"$|\text{best predictor} - \lambda_{abs}|$  (log scale)", fontsize=9)
try:
    c1 = ax.contour(beta_arr, gamma_arr, eff_gap.T, levels=[0.01],
                    colors=["lime"], linewidths=[2.0])
    ax.clabel(c1, fmt={0.01: "1% error"}, fontsize=7.5)
    c2 = ax.contour(beta_arr, gamma_arr, eff_gap.T, levels=[0.05],
                    colors=["yellow"], linewidths=[1.5], linestyles=["--"])
    ax.clabel(c2, fmt={0.05: "5% error"}, fontsize=7.5)
except Exception:
    pass
for ref_b, lbl in BREF:
    ax.axvline(ref_b, color="cyan", lw=0.9, ls="--", alpha=0.65)
    ax.text(ref_b + 0.04, gamma_arr[-1] + 0.1, lbl, color="cyan", fontsize=7)
ax.plot(0.5434, -0.749, "*", ms=16, color="gold", zorder=10,
        label="p=1 optimal")
ax.set_xlabel(r"$\beta$ (rad)", fontsize=10)
ax.set_ylabel(r"$\gamma$ (rad)", fontsize=10)
ax.set_title("Bright = small gap = saddle approximation valid\n"
             "Dark = large gap = approximation breaks", fontsize=10)
ax.legend(fontsize=8)

# Right — categorical: seed / competitor / trivial, with 5% contour
ax2 = axes4[1]
# Encode: 2=seed best + small gap, 1=competitor best + small gap,
#         0=no good predictor (eff_gap > 0.05 or trivial β)
cat = np.full((B, G), nan)
for b_i in range(B):
    for g_i in range(G):
        eg_val = eff_gap[b_i, g_i]
        sb_val = seed_best_[b_i, g_i]
        if eg_val != eg_val:
            cat[b_i, g_i] = nan
        elif eg_val < 0.05:
            cat[b_i, g_i] = 2.0 if sb_val > 0.5 else 1.0
        else:
            cat[b_i, g_i] = 0.0

masked_cat = np.ma.masked_invalid(cat)
cmap_cat = mcolors.ListedColormap(["#cc3333", "#4488cc", "#2ca02c"])
bounds_cat = [-0.5, 0.5, 1.5, 2.5]
norm_cat = mcolors.BoundaryNorm(bounds_cat, cmap_cat.N)
ax2.pcolormesh(beta_arr, gamma_arr, masked_cat.T,
               cmap=cmap_cat, norm=norm_cat, shading="nearest")

# Legend patches
from matplotlib.patches import Patch
legend_els = [
    Patch(facecolor="#2ca02c", label="Seed controls  (<5% error)"),
    Patch(facecolor="#4488cc", label="Competitor controls  (<5% error)"),
    Patch(facecolor="#cc3333", label="No saddle controls  (>5% error)"),
    Patch(facecolor="0.75",    label="Failed / trivial"),
]
ax2.legend(handles=legend_els, loc="lower right", fontsize=8)

# 5% error boundary from eff_gap
try:
    cs_cat = ax2.contour(beta_arr, gamma_arr, eff_gap.T, levels=[0.05],
                         colors=["k"], linewidths=[2.0], linestyles=["--"])
    ax2.clabel(cs_cat, fmt={0.05: "5% boundary"}, fontsize=8)
except Exception:
    pass

for ref_b, lbl in BREF:
    ax2.axvline(ref_b, color="k", lw=0.9, ls="--", alpha=0.4)
    ax2.text(ref_b + 0.04, gamma_arr[-1] + 0.1, lbl, color="k", fontsize=7)
ax2.plot(0.5434, -0.749, "*", ms=16, color="gold", zorder=10, label="p=1 optimal")
ax2.set_xlabel(r"$\beta$ (rad)", fontsize=10)
ax2.set_ylabel(r"$\gamma$ (rad)", fontsize=10)
ax2.set_title("Which single saddle controls?", fontsize=10)

fig4.tight_layout(rect=[0, 0, 1, 0.93])
fig4.savefig(OUT_DIR / "fig4_effective_gap.png", dpi=160, bbox_inches="tight")
print(f"Saved fig4_effective_gap.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 5 — "relay race" plot: who controls lambda_abs as gamma sweeps from 0
# to -2pi, for 4 representative beta values
# Shows: actual lambda_abs (black), seed exponent (blue dashed),
#        and the best-competitor exponent (reconstructed as lambda +/- bm_gap)
# ══════════════════════════════════════════════════════════════════════════════

relay_betas = [0.3, 0.5434, 0.8, 1.5708]
relay_labels = [r"$\beta=0.3$  (sub-optimal)", r"$\beta=0.5434$  (p=1 optimal)",
                r"$\beta=0.8$  (past $\pi/4$, death zone)",
                r"$\beta=\pi/2$  (trivial/degenerate)"]
relay_colors = ["steelblue", "green", "tomato", "purple"]

fig5, axs5 = plt.subplots(2, 2, figsize=(16, 11), sharey=False)
fig5.suptitle(
    r"The saddle relay: who carries $\lambda_{abs}$ as $\gamma$ varies?"
    "\n"
    r"Solid black = $\lambda_{abs}$ (exact).  Blue dashes = BM24 seed.  "
    r"Colored dots = best competitor when it controls.",
    fontsize=11, fontweight="bold",
)

for ax, tb, label, color in zip(axs5.flat, relay_betas, relay_labels, relay_colors):
    b_idx = min(range(B), key=lambda i: abs(betas_all[i] - tb))
    b_val = betas_all[b_idx]

    g_   = gamma_arr
    lam_ = lam[b_idx, :]
    fcs  = full_seed[b_idx, :]
    eg   = eff_gap[b_idx, :]
    bm   = bm_gap[b_idx, :]
    sb   = seed_best_[b_idx, :]

    # Plot exact lambda_abs
    ax.plot(g_, lam_, "k-", lw=3.0, label=r"$\lambda_{abs}$ (exact)", zorder=6)

    # Plot seed exponent
    ax.plot(g_, fcs, "b--", lw=1.8, alpha=0.75, label=r"$E_\mathrm{seed}$", zorder=4)

    # Reconstruct best competitor exponent ≈ lambda_abs (up to bm_gap)
    # When competitor is best, mark lambda_abs with colored dots
    g_comp, lam_comp = [], []
    g_seed, lam_seed = [], []
    for gi_i in range(G):
        if sb[gi_i] < 0.5 and eg[gi_i] < 0.05:
            g_comp.append(g_[gi_i])
            lam_comp.append(lam_[gi_i])
        elif sb[gi_i] > 0.5 and eg[gi_i] < 0.05:
            g_seed.append(g_[gi_i])
            lam_seed.append(lam_[gi_i])

    if g_seed:
        ax.plot(g_seed, lam_seed, "o", ms=7, color="royalblue", zorder=7,
                label="Seed controls  (gap < 5%)", alpha=0.85)
    if g_comp:
        ax.plot(g_comp, lam_comp, "D", ms=7, color=color, zorder=8,
                label="Competitor controls  (gap < 5%)", alpha=0.85)

    ax.axhline(phi_pref, color="0.6", lw=0.8, ls=":", alpha=0.7)
    ax.text(g_[0] + 0.2, phi_pref + 0.02, r"$\phi_\mathrm{pref}$",
            color="0.5", fontsize=7.5)

    # Show the effective gap as a shaded band around lambda
    for gi_i in range(G - 1):
        xL, xR = g_[gi_i], g_[gi_i + 1]
        gap_here = eg[gi_i] if eg[gi_i] == eg[gi_i] else 0.5
        # shade gray if gap > 5% (bad approximation)
        if gap_here > 0.05:
            ax.axvspan(xL, xR, alpha=0.18, color="gray", lw=0)

    ax.set_xlabel(r"$\gamma$ (rad)", fontsize=9)
    ax.set_ylabel("Conv2 exponent", fontsize=9)
    ax.set_title(label, fontsize=10, fontweight="bold")
    ax.legend(fontsize=7.5, loc="lower right")
    ax.set_xlim(g_[0] - 0.1, g_[-1] + 0.1)

    ylo = min(np.nanmin(fcs), np.nanmin(lam_), phi_pref) - 0.15
    yhi = max(np.nanmax(fcs), np.nanmax(lam_)) + 0.15
    ax.set_ylim(ylo, yhi)
    ax.text(0.02, 0.97,
            r"Gray band: no saddle closely predicts ($>5\%$ error)",
            transform=ax.transAxes, va="top", fontsize=7, color="0.45")

fig5.tight_layout(rect=[0, 0, 1, 0.93])
fig5.savefig(OUT_DIR / "fig5_relay_race.png", dpi=160, bbox_inches="tight")
print(f"Saved fig5_relay_race.png")

print(f"\nAll figures written to {OUT_DIR}")
