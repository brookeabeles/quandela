"""
Explain R > 1 and N_eff < 1 via phase structure of the lattice sum.

Part A: Phase structure — table and Argand diagram of c_k = 1/(θ* - θ_k)².
Part B: Angle between (θ* - θ_k_nearest) and lattice direction 2πi/w.
Part C: Corrected R_magnitude and N_eff_mag (magnitude-only, no phase confusion).
Output: 2-panel figure.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from qaoa_core import (
    d2Phi, dominant_saddle_path, singularities, nearest_singularity,
    w_of_gamma, DEFAULT_BETA,
)

beta = DEFAULT_BETA
plt.rcParams.update({'font.size': 10, 'figure.dpi': 150})

# ── Compute lattice contributions c_k = 1/(θ* - θ_k)² ─────────────────────

def get_contributions(theta_star, gt, kmax=50, n_nearest=10, beta_=beta):
    """For the n_nearest singularities to θ*, return c_k, |c_k|, arg(c_k), and cumulative sum."""
    sings = singularities(gt, kmax, beta_)
    dists = np.abs(theta_star - sings)
    order = np.argsort(dists)
    c_list = []
    for idx in order[:n_nearest]:
        sk = sings[idx]
        ck = 1.0 / (theta_star - sk)**2
        c_list.append((ck, np.abs(ck), np.angle(ck) / np.pi))
    c_list = np.array(c_list, dtype=object)
    cumsum = np.cumsum([x[0] for x in c_list])
    return c_list, cumsum


def angle_saddle_to_lattice(theta_star, gt, beta_=beta):
    """Angle (in radians) between (θ* - θ_k_nearest) and lattice direction 2πi/w."""
    d, theta_k = nearest_singularity(theta_star, gt, kmax=50, beta=beta_)
    ww = w_of_gamma(gt)
    lattice_dir = 2j * np.pi / ww  # direction to next singularity
    diff = theta_star - theta_k
    if np.abs(diff) < 1e-14:
        return np.nan
    cos_angle = np.real(diff * np.conj(lattice_dir)) / (np.abs(diff) * np.abs(lattice_dir) + 1e-20)
    cos_angle = np.clip(cos_angle, -1, 1)
    return np.arccos(cos_angle)


# Path for γ̃ ∈ [0.1π, 20π]
print("Hessian cancellation: path and lattice sums...")
gts, thetas = dominant_saddle_path(20 * np.pi, n_steps=200)

# Part A: Table and Argand for γ̃ ∈ {2π, 6π, 12π, 20π}
gt_sample = [2*np.pi, 6*np.pi, 12*np.pi, 20*np.pi]
for gt in [6*np.pi, 20*np.pi]:  # print table for two
    idx = np.argmin(np.abs(gts - gt))
    th = thetas[idx]
    c_list, cumsum = get_contributions(th, gt, n_nearest=10)
    phi2 = d2Phi(th, gt, beta)
    total = phi2 + 0.5  # Σ c_k = Φ'' + 1/2
    print(f"\nγ̃ = {gt/np.pi:.0f}π: Σ c_k = {total:.4f}; |Σ c_k| = {np.abs(total):.4f}")
    print("  k  |c_k|     arg(c_k)/π   cumsum")
    for k, (ck, abs_ck, arg_ck) in enumerate(c_list):
        cs = cumsum[k]
        print(f"  {k:2d}  {abs_ck:.5f}   {arg_ck:.4f}       {np.abs(cs):.5f}")

# Part B: Angle vs γ̃
angles = []
for th, gt in zip(thetas, gts):
    angles.append(angle_saddle_to_lattice(th, gt))
angles = np.array(angles)

# Part C: R_complex, R_magnitude, N_eff_mag
kmax_c = 80
R_complex_vals = []
R_magnitude_vals = []
N_eff_mag_vals = []
for th, gt in zip(thetas, gts):
    sings = singularities(gt, kmax=kmax_c)
    contribs = np.array([1.0 / (th - sk)**2 for sk in sings])
    total_complex = np.sum(contribs)
    d, theta_k = nearest_singularity(th, gt, kmax=kmax_c)
    c_nearest = 1.0 / (th - theta_k)**2
    R_complex_vals.append(np.abs(c_nearest) / (np.abs(total_complex) + 1e-20))
    abs2 = np.abs(contribs)**2
    sum_abs2 = np.sum(abs2)
    R_magnitude_vals.append(np.abs(c_nearest)**2 / (sum_abs2 + 1e-20))
    abs4 = np.abs(contribs)**4
    N_eff_mag_vals.append((sum_abs2**2) / (np.sum(abs4) + 1e-30))
R_complex_vals = np.array(R_complex_vals)
R_magnitude_vals = np.array(R_magnitude_vals)
N_eff_mag_vals = np.array(N_eff_mag_vals)

# ── Figure: 2 panels (a) = two Argand subpanels, (b) = R_mag, N_eff_mag ─────

fig, axes = plt.subplots(1, 3, figsize=(14, 5))

# (a) Argand at γ̃ = 6π and 20π
for i_plot, gt in enumerate([6*np.pi, 20*np.pi]):
    ax = axes[i_plot]
    idx = np.argmin(np.abs(gts - gt))
    th = thetas[idx]
    c_list, cumsum = get_contributions(th, gt, n_nearest=10)
    vecs = np.array([x[0] for x in c_list])
    x, y = 0.0, 0.0
    scale = 1.0 / (np.max(np.abs(vecs)) + 1e-10)
    for v in vecs:
        vv = v * scale
        ax.arrow(x, y, np.real(vv), np.imag(vv), head_width=0.03, head_length=0.02, fc='blue', ec='blue', alpha=0.7)
        x, y = x + np.real(vv), y + np.imag(vv)
    ax.plot([0, x], [0, y], 'k-', lw=2)
    ax.set_xlabel(r'Re $c_k$')
    ax.set_ylabel(r'Im $c_k$')
    ax.set_title(rf'(a) Argand $\tilde\gamma={gt/np.pi:.0f}\pi$')
    ax.axis('equal')
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='gray', ls='--')
    ax.axvline(0, color='gray', ls='--')

# (b) R_magnitude and N_eff_mag vs γ̃
ax2 = axes[2]
ax2.plot(gts / np.pi, R_magnitude_vals, 'b-', lw=1.5, label=r'$R_{\mathrm{mag}}$')
ax2.plot(gts / np.pi, N_eff_mag_vals, 'r-', lw=1.5, label=r'$N_{\mathrm{eff,mag}}$')
ax2.axhline(1.0, color='gray', ls='--', alpha=0.5)
ax2.set_xlabel(r'$\tilde{\gamma}$ (units of $\pi$)')
ax2.set_ylabel('Ratio / count')
ax2.set_title(r'(b) $R_{\mathrm{mag}}$, $N_{\mathrm{eff,mag}}$')
ax2.set_xlim(0, 20)
ax2.set_ylim(bottom=0)
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('hessian_cancellation.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved hessian_cancellation.png")

# Angle near π/2?
idx_6 = np.argmin(np.abs(gts - 6*np.pi))
print(f"  Angle (θ*-θ_k) vs 2πi/w at 6π: {np.degrees(angles[idx_6]):.1f}° (π/2 = 90°)")
