"""
Saddle–singularity separation: analytic bound + numerical verification.

Produces one figure with 2 panels:
  (a) d(γ̃) measured (blue), ρ*(γ̃) Rouché bound (green dots), horizontal line ρ* (uniform);
  (b) Landscape at γ̃ = 2π with capped exclusion circles, dominant saddle, singularities
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from qaoa_core import (
    w_of_gamma, A, Phi, dPhi, d2Phi, singularities, lattice_spacing,
    dominant_saddle_path, nearest_singularity, DEFAULT_BETA,
)

OUT_DIR = Path(__file__).resolve().parent
beta = DEFAULT_BETA

plt.rcParams.update({
    'font.size': 11,
    'text.usetex': False,
    'figure.dpi': 150,
})


def exact_analytic_bound(theta_star, gt, beta_=beta):
    """
    Exact bound: ρ_k = 2/|w - θ_k|. For the nearest singularity: d_min(γ̃) = 2/|w - θ_{k_nearest}|.
    """
    _, theta_k = nearest_singularity(theta_star, gt, kmax=50, beta=beta_)
    ww = w_of_gamma(gt)
    denom = np.abs(ww - theta_k)
    if denom < 1e-15:
        return np.inf
    return 2.0 / denom


def rouche_exclusion_radius_fixed(gt, kmax=5, beta_=beta, n_angles=200):
    """Largest ρ where min|f| > max|g| on |θ - θ_k| = ρ; f = 1/(θ-θ_k), g = Φ' - f."""
    sings = singularities(gt, kmax, beta_)
    angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
    best_rho = 0.0
    for theta_k in sings:
        rhos = np.logspace(-6, 0, 300)
        rho_star_k = 0.0
        for rho in rhos:
            circle = theta_k + rho * np.exp(1j * angles)
            f_vals = 1.0 / (circle - theta_k)
            min_f = np.min(np.abs(f_vals))
            g_vals = np.array([dPhi(th, gt, beta_) for th in circle]) - f_vals
            max_g = np.max(np.abs(g_vals))
            if min_f > max_g:
                rho_star_k = rho
            else:
                break
        if rho_star_k > best_rho:
            best_rho = rho_star_k
    return best_rho


def main():
    # Restrict scan to γ̃ up to 2π for testing.
    gt_max = 2 * np.pi
    n_gt = 400
    gts_fine, thetas_fine = dominant_saddle_path(gt_max, n_steps=n_gt)

    d_measured = np.zeros(n_gt)
    L_vals = np.zeros(n_gt)
    for i in range(n_gt):
        d_meas_i, _ = nearest_singularity(thetas_fine[i], gts_fine[i], kmax=50, beta=beta)
        d_measured[i] = d_meas_i
        L_vals[i] = lattice_spacing(gts_fine[i])

    n_rouche = 20
    gt_rouche = np.linspace(0.3 * np.pi, 2.0 * np.pi, n_rouche)
    rho_stars = np.array([rouche_exclusion_radius_fixed(gt) for gt in gt_rouche])
    rho_min_uniform = max(0.5, np.min(rho_stars) * 0.98)

    gt_show = 2 * np.pi
    _, thetas_show = dominant_saddle_path(gt_show, n_steps=300)
    theta_star = thetas_show[-1]

    # Main panel: separation curves
    fig, ax = plt.subplots(1, 1, figsize=(8, 4.5))
    mask_plot = (gts_fine / np.pi >= 0.3) & (gts_fine / np.pi <= 4.0)
    ax.semilogy(gts_fine[mask_plot] / np.pi, d_measured[mask_plot], 'b-', lw=1.5, label=r'$d(\tilde\gamma)$ measured')
    ax.semilogy(gt_rouche / np.pi, rho_stars, 'go', ms=6, label=r'Rouché $\rho^*(\tilde\gamma)$ (rigorous bound)')
    ax.axhline(rho_min_uniform, color='black', linestyle='--', lw=1.2, label=rf'$\rho^* \geq {rho_min_uniform:.2f}$ (uniform)')
    ax.set_xlabel(r'$\tilde\gamma$ (units of $\pi$)')
    ax.set_ylabel('Distance (log scale)')
    ax.set_title(r'Saddle–singularity separation')
    ax.legend(fontsize=9)
    ax.set_xlim(0.3, 2.0)
    ax.set_ylim(bottom=1e-3)
    ax.grid(True, alpha=0.3, which='both')

    # Inset panel (b): zoomed landscape with the two largest Rouché radii,
    # placed at the bottom-right of panel (a).
    ax_inset = ax.inset_axes([0.58, 0.12, 0.38, 0.55])
    nx, ny = 200, 200
    x = np.linspace(-6, 6, nx)
    y = np.linspace(-6, 6, ny)
    X, Y = np.meshgrid(x, y)
    Theta = X + 1j * Y
    Z = np.real(Phi(Theta, gt_show, beta))
    Z = np.clip(Z, -10, 20)
    cf = ax_inset.contourf(X, Y, Z, levels=50, cmap='viridis')
    ax_inset.set_xlabel(r'Re $\theta$', fontsize=8)
    ax_inset.set_ylabel(r'Im $\theta$', fontsize=8)
    ax_inset.set_title(r'(b) Landscape $\tilde\gamma = 2\pi$', fontsize=9)

    ww_show = w_of_gamma(gt_show)
    sings = singularities(gt_show, kmax=10, beta=beta)
    sings_visible = [s for s in sings if -6 < s.real < 6 and -6 < s.imag < 6]
    centers = []
    radii = []
    for sk in sings_visible:
        ax_inset.plot(sk.real, sk.imag, 'kx', ms=6, mew=1.5)
        rho_from_bound = 2.0 / (np.abs(ww_show - sk) + 1e-300)
        other_dists = [np.abs(sk - s) for s in sings if s != sk]
        half_nn = (min(other_dists) / 2.0) if other_dists else rho_from_bound
        rho_k = min(rho_from_bound, half_nn)
        rho_k = min(rho_k, 1.5)
        centers.append(sk)
        radii.append(rho_k)
        circ = Circle((sk.real, sk.imag), rho_k, fill=False, edgecolor='white', linestyle='--', lw=1.2)
        ax_inset.add_patch(circ)

    # Focus the inset on the two largest exclusion radii.
    if radii:
        idx_sorted = np.argsort(radii)
        top_idx = idx_sorted[-2:] if len(radii) >= 2 else idx_sorted[-1:]
        xs = [centers[i].real for i in top_idx]
        ys = [centers[i].imag for i in top_idx]
        rs = [radii[i] for i in top_idx]
        padding = 2.0
        x_min = min(x - padding * r for x, r in zip(xs, rs))
        x_max = max(x + padding * r for x, r in zip(xs, rs))
        y_min = min(y - padding * r for y, r in zip(ys, rs))
        y_max = max(y + padding * r for y, r in zip(ys, rs))
        ax_inset.set_xlim(x_min, x_max)
        ax_inset.set_ylim(y_min, y_max)
    else:
        ax_inset.set_xlim(-6, 6)
        ax_inset.set_ylim(-6, 6)

    ax_inset.plot(theta_star.real, theta_star.imag, 'o', color='lime', ms=8, mec='black', mew=1.0, zorder=5)
    ax_inset.set_aspect('equal')

    plt.tight_layout()
    out_path = OUT_DIR / 'saddle_singularity_separation_combined.png'
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print("Saved", out_path)


if __name__ == '__main__':
    main()
