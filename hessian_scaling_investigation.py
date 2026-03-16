"""
Fix 4: Investigate |Φ''(θ*)| scaling vs d and decompose Hessian.

Validation found log-log slope of |Φ''| vs 1/d² was 0.19, not 1.0.
This script:
1. Plots |Φ''| vs d, 1/d, 1/d², log(d) to find correct scaling
2. Decomposes Φ'' = -1/2 + (singularity contribution), plots each
3. Plots |Φ''| vs w(γ̃) to check relationship through lattice
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from qaoa_core import (
    dominant_saddle_path,
    nearest_singularity,
    d2Phi,
    w_of_gamma,
    A,
    DEFAULT_BETA,
)

beta = DEFAULT_BETA
plt.rcParams.update({'font.size': 10, 'figure.dpi': 150})

# Path continuation to get θ*(γ̃) and compute d, |Φ''|
gt_max = 20 * np.pi
n_steps = 300
print("Hessian scaling: path continuation to 20π...")
gts, thetas = dominant_saddle_path(gt_max, n_steps=n_steps, beta=beta)

d_vals = np.array([nearest_singularity(th, gt, kmax=80, beta=beta)[0] for th, gt in zip(thetas, gts)])
phi2_vals = np.array([d2Phi(th, gt, beta) for th, gt in zip(thetas, gts)])
phi2_abs = np.abs(phi2_vals)

# Decompose Φ'' = term1 + term2 + term3 (from qaoa_core.d2Phi)
# term1 = -0.5, term2 = -1j*ww^2*s*e/aa, term3 = -(ww*s*e/aa)^2
def d2Phi_decomposed(theta, gt, beta_=beta):
    ww = w_of_gamma(gt)
    s = np.sin(beta_ / 2)
    e = np.exp(theta * ww)
    aa = A(theta, gt, beta_) + 1e-300
    term1 = -0.5
    term2 = -1j * ww**2 * s * e / aa
    term3 = -(ww * s * e / aa) ** 2
    return term1, term2, term3

term1_vals = []
term2_vals = []
term3_vals = []
for th, gt in zip(thetas, gts):
    t1, t2, t3 = d2Phi_decomposed(th, gt)
    term1_vals.append(t1)
    term2_vals.append(t2)
    term3_vals.append(t3)
term1_vals = np.array(term1_vals)
term2_vals = np.array(term2_vals)
term3_vals = np.array(term3_vals)
sing_part = term2_vals + term3_vals
sing_abs = np.abs(sing_part)
w_vals = np.array([np.abs(w_of_gamma(gt)) for gt in gts])

# Mask valid (d > 0, finite)
valid = np.isfinite(phi2_abs) & (d_vals > 1e-10) & np.isfinite(d_vals)
gts_v = gts[valid]
d_v = d_vals[valid]
phi2_v = phi2_abs[valid]
inv_d = 1.0 / d_v
inv_d2 = 1.0 / (d_v ** 2)
log_d = np.log(d_v + 1e-30)
sing_v = sing_abs[valid]
w_v = w_vals[valid]

# Linear fits in log-log to get slope
def log_fit_slope(x, y):
    mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    if np.sum(mask) < 5:
        return np.nan
    logx = np.log(x[mask])
    logy = np.log(y[mask])
    slope = np.polyfit(logx, logy, 1)[0]
    return slope

slope_d = log_fit_slope(d_v, phi2_v)
slope_inv_d = log_fit_slope(inv_d, phi2_v)
slope_inv_d2 = log_fit_slope(inv_d2, phi2_v)
slope_w = log_fit_slope(w_v, phi2_v)
print(f"  Log-log slope |Φ''| vs d:       {slope_d:.3f}")
print(f"  Log-log slope |Φ''| vs 1/d:     {slope_inv_d:.3f}")
print(f"  Log-log slope |Φ''| vs 1/d²:    {slope_inv_d2:.3f}")
print(f"  Log-log slope |Φ''| vs |w(γ̃)|:  {slope_w:.3f}")

# Figure 1: |Φ''| vs d, 1/d, 1/d², log(d)
fig1, axes = plt.subplots(2, 2, figsize=(9, 8))
ax = axes[0, 0]
ax.loglog(d_v, phi2_v, 'b.', ms=2, alpha=0.7)
ax.set_xlabel(r'$d$ (distance to nearest singularity)')
ax.set_ylabel(r'$|\Phi''(\theta^*)|$')
ax.set_title(rf'$|\Phi''|$ vs $d$ (slope ≈ {slope_d:.2f})')
ax.grid(True, alpha=0.3)

ax = axes[0, 1]
ax.loglog(inv_d, phi2_v, 'b.', ms=2, alpha=0.7)
ax.set_xlabel(r'$1/d$')
ax.set_ylabel(r'$|\Phi''(\theta^*)|$')
ax.set_title(rf'$|\Phi''|$ vs $1/d$ (slope ≈ {slope_inv_d:.2f})')
ax.grid(True, alpha=0.3)

ax = axes[1, 0]
ax.loglog(inv_d2, phi2_v, 'b.', ms=2, alpha=0.7)
ax.set_xlabel(r'$1/d^2$')
ax.set_ylabel(r'$|\Phi''(\theta^*)|$')
ax.set_title(rf'$|\Phi''|$ vs $1/d^2$ (slope ≈ {slope_inv_d2:.2f})')
ax.grid(True, alpha=0.3)

ax = axes[1, 1]
ax.semilogy(log_d, phi2_v, 'b.', ms=2, alpha=0.7)
ax.set_xlabel(r'$\log d$')
ax.set_ylabel(r'$|\Phi''(\theta^*)|$')
ax.set_title(r'$|\Phi''|$ vs $\\log d$')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('hessian_scaling_vs_d.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved hessian_scaling_vs_d.png")

# Figure 2: Decomposition Φ'' = -1/2 + (singularity contribution)
fig2, axes = plt.subplots(2, 1, figsize=(7, 7))
ax = axes[0]
ax.semilogy(gts / np.pi, phi2_abs, 'k-', lw=1.2, label=r'$|\Phi''|$')
ax.semilogy(gts / np.pi, np.abs(term1_vals) + 1e-30, 'g-', label=r'$|-1/2|$ (Gaussian)')
ax.semilogy(gts / np.pi, sing_abs + 1e-30, 'b-', lw=1, label=r'$|$singularity part$|$')
ax.set_xlabel(r'$\tilde\gamma / \pi$')
ax.set_ylabel(r'magnitude')
ax.set_title(r'Decomposition: $\Phi'' = -1/2 +$ (singularity contribution)')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.plot(gts / np.pi, phi2_abs, 'k-', lw=1.2, label=r'$|\Phi''|$')
ax.plot(gts / np.pi, w_vals, 'r-', lw=1, label=r'$|w(\tilde\gamma)|$')
ax.set_xlabel(r'$\tilde\gamma / \pi$')
ax.set_ylabel(r'magnitude')
ax.set_title(r'$|\Phi''|$ vs $|w(\\tilde\gamma)|$ (lattice scale)')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('hessian_decomposition.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved hessian_decomposition.png")
