"""
Cell fraction c(β): test universality of c = lim d/L.

FIX 1: Use TRUE dominant saddle (highest Re Φ) at each γ̃ via all_saddles,
not path continuation from θ*=0, to capture Stokes transitions.

FIX 2: Extend to γ̃ = 200π for convergence check; plot d/L vs 1/γ̃;
fit c + a/γ̃^α or growth models; report whether c(β) limit exists.

Outputs:
- cell_fraction_beta.png: c(β) vs β (from [10π,20π]) + Re Φ panel for Stokes
- cell_fraction_convergence.png: d/L vs 1/γ̃ for β=-π/2 out to 200π + fits
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from qaoa_core import (
    all_saddles,
    nearest_singularity,
    lattice_spacing,
    Phi,
)

plt.rcParams.update({'font.size': 10, 'figure.dpi': 150})

# Beta values for main c(β) plot
betas = [-np.pi/4, -np.pi/3, -np.pi/2, -2*np.pi/3, -3*np.pi/4]
gt_max_main = 20 * np.pi
# Use QUICK_RUN=True for a fast check (fewer steps); False for full validation
QUICK_RUN = True
n_steps_main = 50 if QUICK_RUN else 200
mask_fit = lambda gts: (gts >= 10*np.pi) & (gts <= 20*np.pi)


def get_dominant_saddle_path(gts, beta, grid_size=40, n_saddles_track=3):
    """
    At each γ̃ in gts, find all saddles and take the one with highest Re Φ.
    Return (thetas_dom, re_phi_per_saddle) where re_phi_per_saddle is (n_steps, n_saddles_track).
    """
    thetas_dom = []
    re_phi_list = []  # list of length n_steps, each entry list of top Re Φ
    for gt in gts:
        hw = max(15, 3 * np.sqrt(gt))
        saddles = all_saddles(gt, beta, grid_size=grid_size, half_width=hw, n_saddles_max=10)
        if not saddles:
            thetas_dom.append(np.nan + 1j * np.nan)
            re_phi_list.append([np.nan] * n_saddles_track)
            continue
        theta_dom = saddles[0][0]
        re_phis = [s[1].real for s in saddles[:n_saddles_track]]
        while len(re_phis) < n_saddles_track:
            re_phis.append(np.nan)
        thetas_dom.append(theta_dom)
        re_phi_list.append(re_phis)
    return np.array(thetas_dom), np.array(re_phi_list)


# ─── Main: c(β) using dominant saddle, and Re Φ panel ─────────────────────
c_vals = []
c_stds = []
all_gts = np.linspace(1e-6, gt_max_main, n_steps_main)
# Store for Stokes panel: one representative beta (e.g. -π/2)
stokes_gts = None
stokes_re_phi = None
stokes_beta = -np.pi/2

print("Cell fraction c(β) using TRUE dominant saddle (highest Re Φ) at each γ̃:")
for beta in betas:
    gts = all_gts
    thetas_dom, re_phi_per_saddle = get_dominant_saddle_path(gts, beta, grid_size=40, n_saddles_track=3)
    valid = ~np.isnan(thetas_dom.real)
    d_vals = np.full_like(thetas_dom, np.nan, dtype=float)
    L_vals = np.array([lattice_spacing(gt) for gt in gts])
    for i in np.where(valid)[0]:
        d_vals[i] = nearest_singularity(thetas_dom[i], gts[i], kmax=80, beta=beta)[0]
    dL = np.where(valid, d_vals / (L_vals + 1e-300), np.nan)
    gts_fit = gts[mask_fit(gts)]
    dL_fit = dL[mask_fit(gts)]
    dL_fit = dL_fit[~np.isnan(dL_fit)]
    if len(dL_fit) == 0:
        c_mean, c_std = np.nan, np.nan
    else:
        c_mean = np.mean(dL_fit)
        c_std = np.std(dL_fit)
    c_vals.append(c_mean)
    c_stds.append(c_std)
    print(f"  β = {beta/np.pi:.3f}π  =>  c = {c_mean:.4f} ± {c_std:.4f}")

    if abs(beta - stokes_beta) < 1e-6:
        stokes_gts = gts
        stokes_re_phi = re_phi_per_saddle

# Figure 1: c(β) vs β + Stokes panel (Re Φ for top 2-3 saddles vs γ̃)
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

ax = axes[0]
ax.errorbar([b/np.pi for b in betas], c_vals, yerr=c_stds, fmt='bo', capsize=4)
ax.set_xlabel(r'$\beta$ (units of $\pi$)')
ax.set_ylabel(r'$c(\beta) = \lim_{\tilde\gamma\to\infty} d/L$ (dominant saddle)')
ax.set_title(r'Cell fraction $c(\beta)$ (γ̃/π ∈ [10,20])')
ax.grid(True, alpha=0.3)

ax2 = axes[1]
if stokes_gts is not None and stokes_re_phi is not None:
    for j in range(stokes_re_phi.shape[1]):
        ax2.plot(stokes_gts / np.pi, stokes_re_phi[:, j], '-', label=f'Re Φ (saddle {j+1})')
ax2.set_xlabel(r'$\tilde\gamma / \pi$')
ax2.set_ylabel(r'Re $\Phi$')
ax2.set_title(r'Stokes: Re Φ of top saddles vs γ̃ (β = −π/2)')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('cell_fraction_beta.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved cell_fraction_beta.png")

# ─── Fix 2: Convergence check — extend to 200π for β = -π/2 ─────────────────
gt_max_ext = 200 * np.pi
n_steps_ext = 500 if QUICK_RUN else 2000
beta_conv = -np.pi/2
print(f"\nConvergence check: β = {beta_conv/np.pi:.2f}π, γ̃ ∈ [0, {gt_max_ext/np.pi:.0f}π], n_steps = {n_steps_ext}")
gts_ext = np.linspace(1e-6, gt_max_ext, n_steps_ext)
thetas_ext, _ = get_dominant_saddle_path(gts_ext, beta_conv, grid_size=40, n_saddles_track=1)
L_ext = np.array([lattice_spacing(gt) for gt in gts_ext])
d_ext = np.full(len(gts_ext), np.nan)
for i in range(len(gts_ext)):
    if not np.isnan(thetas_ext[i].real):
        d_ext[i] = nearest_singularity(thetas_ext[i], gts_ext[i], kmax=80, beta=beta_conv)[0]
dL_ext = np.where(~np.isnan(d_ext), d_ext / (L_ext + 1e-300), np.nan)
inv_gt = 1.0 / (gts_ext + 1e-300)

# Fit: d/L = c + a/γ̃^α on [10π, 200π]
mask_fit_ext = (gts_ext >= 10*np.pi) & (gts_ext <= 200*np.pi)
gts_f_full = gts_ext[mask_fit_ext]
dL_f_full = dL_ext[mask_fit_ext]
valid_f = ~np.isnan(dL_f_full)
gts_f = gts_f_full[valid_f]
dL_f = dL_f_full[valid_f]

def model_c_plus_a_gt_alpha(gt, c, a, alpha):
    return c + a * (gt + 1e-30) ** (-alpha)

convergence_report = []
c_simple = np.nan
c_fit = np.nan
a_fit = np.nan
alpha_fit = 0.0  # so "if alpha_fit > 0" is safe
try:
    from scipy.optimize import curve_fit
    c_simple = np.mean(dL_f) if len(dL_f) else np.nan
    convergence_report.append(f"  Mean d/L on [10π, 200π]: {c_simple:.4f}")

    if len(dL_f) > 10:
        try:
            popt, _ = curve_fit(
                model_c_plus_a_gt_alpha, gts_f, dL_f,
                p0=[0.55, 1.0, 0.5], bounds=([0, -np.inf, 0.01], [2, np.inf, 2])
            )
            c_fit, a_fit, alpha_fit = popt[0], popt[1], popt[2]
            convergence_report.append(f"  Fit d/L = c + a/γ̃^α: c = {c_fit:.4f}, a = {a_fit:.4f}, α = {alpha_fit:.4f}")
            if alpha_fit > 0 and c_fit < 2:
                convergence_report.append(f"  → Limit c(β) ≈ {c_fit:.4f} as γ̃→∞ (α > 0 suggests convergence)")
            else:
                convergence_report.append(f"  → Limit unclear or d/L may grow (α ≤ 0 or c large)")
        except Exception as e:
            convergence_report.append(f"  Fit c + a/γ̃^α failed: {e}")
except ImportError:
    convergence_report.append("  scipy not available for fit")

# Growth model: d/L = a*log(γ̃) + b (if still growing)
def model_log(gt, a, b):
    return a * np.log(gt + 1) + b
try:
    from scipy.optimize import curve_fit
    if len(dL_f) > 10:
        popt_log, _ = curve_fit(model_log, gts_f, dL_f, p0=[0.01, 0.5])
        convergence_report.append(f"  Fit d/L = a*log(γ̃)+b: a = {popt_log[0]:.4f}, b = {popt_log[1]:.4f}")
except Exception:
    pass

print("Convergence report (β = -π/2):")
for line in convergence_report:
    print(line)

# Plot: d/L vs 1/γ̃
fig2, ax = plt.subplots(figsize=(6, 4))
valid = ~np.isnan(dL_ext)
ax.plot(inv_gt[valid], dL_ext[valid], 'b.', ms=1, alpha=0.7, label=r'$d/L$ (dominant saddle)')
ax.set_xlabel(r'$1/\tilde\gamma$')
ax.set_ylabel(r'$d/L$')
ax.set_title(r'Convergence check: $d/L$ vs $1/\tilde\gamma$ (β = −π/2, γ̃ up to 200π)')
if not np.isnan(c_fit) and alpha_fit > 0:
    gt_plot = np.linspace(10*np.pi, 200*np.pi, 200)
    ax.plot(1/gt_plot, model_c_plus_a_gt_alpha(gt_plot, c_fit, a_fit, alpha_fit), 'r-', lw=2, label=rf'fit $c + a/\tilde\gamma^{{\alpha}}$')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('cell_fraction_convergence.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved cell_fraction_convergence.png")
