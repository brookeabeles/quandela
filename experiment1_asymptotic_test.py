"""
Experiment 1 — Asymptotic test

- γ̃ up to 500π
- Sparse sampling (few points, fast run)
- Dominant saddle only (highest Re Φ at each γ̃)
- Plot d/L vs 1/γ̃ for multiple β (positive and negative) → plateau means limit c(β) exists
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from qaoa_core import all_saddles, nearest_singularity, lattice_spacing

plt.rcParams.update({'font.size': 11, 'figure.dpi': 150})

# β values: positive and negative for each
betas = [
    np.pi / 4, -np.pi / 4,
    np.pi / 2, -np.pi / 2,
    np.pi / 3, -np.pi / 3,
    2 * np.pi / 3, -2 * np.pi / 3,
    3 * np.pi / 4, -3 * np.pi / 4,
]

# Sparse sampling: uniform in 1/γ̃ so the plot x-axis is well covered
# γ̃ from ~5π to 500π → 1/γ̃ from 1/(500π) to 1/(5π)
inv_gt_min = 1.0 / (500 * np.pi)
inv_gt_max = 1.0 / (5 * np.pi)
n_points = 55  # sparse
inv_gts = np.linspace(inv_gt_min, inv_gt_max, n_points)
gts = 1.0 / inv_gts  # descending from 500π to 5π

grid_size = 40

def beta_legend(b):
    s = b / np.pi
    if s >= 0:
        return rf'β = {s:.2g}π'
    return rf'β = {s:.2g}π'

print("Experiment 1: asymptotic test (γ̃ up to 500π, sparse, dominant saddle only)")
print(f"  β values: {[round(b/np.pi, 2) for b in betas]} π")
print(f"  Sampling: {n_points} points, γ̃/π ∈ [{gts.min()/np.pi:.1f}, {gts.max()/np.pi:.1f}]")

# Colors: distinct for 10 curves (use tab10 or a list)
colors = plt.cm.tab10(np.linspace(0, 1, 10))
# Alternate style: positive = solid, negative = dashed
styles = ['-', '--', '-', '--', '-', '--', '-', '--', '-', '--']

results = []  # list of (beta, inv_gts_sane, dL_sane, plateau_mean, plateau_std)

for idx, beta in enumerate(betas):
    d_over_L = []
    for i, gt in enumerate(gts):
        hw = max(15, 3 * np.sqrt(gt))
        saddles = all_saddles(gt, beta, grid_size=grid_size, half_width=hw, n_saddles_max=5)
        if not saddles:
            d_over_L.append(np.nan)
            continue
        th = saddles[0][0]
        d = nearest_singularity(th, gt, kmax=80, beta=beta)[0]
        L = lattice_spacing(gt)
        d_over_L.append(d / (L + 1e-300))
    d_over_L = np.array(d_over_L)
    valid = np.isfinite(d_over_L)
    sane = valid & (d_over_L > 0) & (d_over_L <= 1.2)
    dL_sane = d_over_L[sane]
    inv_sane = inv_gts[sane]
    if len(dL_sane) > 0:
        first_third = dL_sane[: max(1, len(dL_sane) // 3)]
        plateau_mean = np.mean(first_third)
        plateau_std = np.std(first_third)
    else:
        plateau_mean = plateau_std = np.nan
    results.append((beta, inv_gts[sane], d_over_L[sane], plateau_mean, plateau_std))
    print(f"  β = {beta/np.pi:+.2f}π  →  plateau d/L = {plateau_mean:.4f} ± {plateau_std:.4f}  (n_sane = {np.sum(sane)})")

fig, ax = plt.subplots(figsize=(8, 5))
for idx, (beta, inv_sane, dL_sane, pm, ps) in enumerate(results):
    if len(inv_sane) == 0:
        continue
    lab = beta_legend(beta)
    ax.plot(inv_sane, dL_sane, color=colors[idx], ls=styles[idx], lw=1.2, ms=3, marker='.', label=lab)

ax.set_xlabel(r'$1/\tilde\gamma$')
ax.set_ylabel(r'$d/L$')
ax.set_ylim(0, 1.05)
ax.set_title(r'Experiment 1: $d/L$ vs $1/\tilde\gamma$ (γ̃ up to 500π, sparse, dominant saddle)')
ax.legend(loc='best', ncol=2, fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('experiment1_asymptotic_dL_vs_inv_gamma.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved experiment1_asymptotic_dL_vs_inv_gamma.png")
