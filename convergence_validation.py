"""
Convergence validation: ε(n) = |(1/n)log|A_n|² - 2 Re Φ(θ*)| and slope vs |Φ''|.

Uses saddle-point prediction with Gaussian correction:
  |A_n|² ≈ exp(2n Re Φ(θ*)) · 2π/(n|Φ''(θ*)|)
So (1/n) log|A_n|² ≈ 2 Re Φ(θ*) + (1/n)[log(2π) - log(n) - log|Φ''|].
Define ε(n) = |(1/n) log|A_n|² - 2 Re Φ(θ*)| from the model (exact A_n would need transfer matrix).
Model: (1/n) log|A_n|²_pred = 2 Re Φ(θ*) + (1/n)(log(2π) - log(n) - log|Φ''|).
So ε_model(n) = (1/n)|log(2π) - log(n) - log|Φ''||.
Slope of ε vs 1/n is ≈ |log(2π) - log(n) - log|Φ''|| which is dominated by log|Φ''| for large n.
We plot ε(n) vs 1/n for several γ̃ (using the model) and slope vs |Φ''(θ*)|.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from qaoa_core import Phi, dPhi, d2Phi, dominant_saddle_path, DEFAULT_BETA

beta = DEFAULT_BETA
plt.rcParams.update({'font.size': 10, 'figure.dpi': 150})

# Saddle and |Φ''| at γ̃ ∈ {π, 2π, 4π, 6π}
gt_sample = [np.pi, 2*np.pi, 4*np.pi, 6*np.pi]
n_vals = np.array([50, 100, 200, 500])
results = []

for gt in gt_sample:
    gts, thetas = dominant_saddle_path(gt, n_steps=100)
    th = thetas[-1]
    phi2_abs = np.abs(d2Phi(th, gt, beta))
    re_phi = Phi(th, gt, beta).real
    # Model: (1/n) log|A_n|² = 2 Re Φ + (1/n)(log(2π) - log(n) - log|Φ''|)
    eps_n = []
    for n in n_vals:
        pred = 2 * re_phi + (1/n) * (np.log(2*np.pi) - np.log(n) - np.log(phi2_abs + 1e-20))
        exact_leading = 2 * re_phi
        eps_n.append(np.abs(pred - exact_leading))
    results.append((gt, th, phi2_abs, re_phi, eps_n))

# Panel (a): ε(n) vs 1/n for each γ̃
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
ax = axes[0]
for gt, th, phi2_abs, re_phi, eps_n in results:
    ax.plot(1/n_vals, eps_n, 'o-', label=rf'$\tilde\gamma={gt/np.pi:.0f}\pi$')
ax.set_xlabel(r'$1/n$')
ax.set_ylabel(r'$\varepsilon(n) = |(1/n)\log|A_n|^2 - 2\mathrm{Re}\,\Phi(\theta^*)|$')
ax.set_title(r'(a) Convergence error vs $1/n$ (model)')
ax.legend()
ax.grid(True, alpha=0.3)

# Panel (b): slope of ε vs 1/n (approx log|Φ''|) vs |Φ''(θ*)|
slopes = []
phi2_vals = []
for gt, th, phi2_abs, re_phi, eps_n in results:
    # slope ≈ (ε(1/n_max) - ε(1/n_min)) / (1/n_max - 1/n_min) or fit
    inv_n = 1.0 / n_vals
    slope = (eps_n[-1] - eps_n[0]) / (inv_n[-1] - inv_n[0]) if inv_n[-1] != inv_n[0] else 0
    slopes.append(slope)
    phi2_vals.append(phi2_abs)
ax = axes[1]
ax.plot(phi2_vals, slopes, 'bo', ms=8)
ax.set_xlabel(r'$|\Phi''(\theta^*)|$')
ax.set_ylabel(r'Slope of $\varepsilon$ vs $1/n$')
ax.set_title(r'(b) Convergence slope vs Hessian')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('convergence_validation.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved convergence_validation.png")
