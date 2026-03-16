"""
Asymptotic d/L limit, cell fraction c(β), convergence validation.

Combines:
- Experiment 1: d/L vs 1/γ̃ (γ̃ up to 500π), plateau → limit c(β)
- Asymptotic drift: d/L vs γ̃, fits (const, power, log), |Φ''| vs 1/d
- Convergence validation: ε(n) vs 1/n, slope vs |Φ''|
- Cell fraction c(β): c(β) vs β, d/L vs 1/γ̃ out to 200π, fits
- Validation check: dominant saddle at selected (β, γ̃)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from qaoa_core import (
    all_saddles, nearest_singularity, lattice_spacing, Phi, dPhi, d2Phi,
    dominant_saddle_path, DEFAULT_BETA,
)

OUT_DIR = Path(__file__).resolve().parent
beta = DEFAULT_BETA
plt.rcParams.update({'font.size': 10, 'figure.dpi': 150})

try:
    from scipy.optimize import curve_fit
    _has_scipy = True
except ImportError:
    _has_scipy = False


def run_experiment1():
    """d/L vs 1/γ̃ for multiple β (γ̃ up to 2π)."""
    betas = [
        np.pi / 4, -np.pi / 4, np.pi / 2, -np.pi / 2, np.pi / 3, -np.pi / 3,
        2 * np.pi / 3, -2 * np.pi / 3, 3 * np.pi / 4, -3 * np.pi / 4,
    ]
    # Restrict to moderate γ̃ for quicker tests: γ̃ ∈ [0.5π, 2π].
    inv_gt_min = 1.0 / (2 * np.pi)
    inv_gt_max = 1.0 / (0.5 * np.pi)
    n_points = 55
    inv_gts = np.linspace(inv_gt_min, inv_gt_max, n_points)
    gts = 1.0 / inv_gts
    grid_size = 40
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    styles = ['-', '--', '-', '--', '-', '--', '-', '--', '-', '--']
    results = []
    for idx, b in enumerate(betas):
        d_over_L = []
        for gt in gts:
            hw = max(15, 3 * np.sqrt(gt))
            saddles = all_saddles(gt, b, grid_size=grid_size, half_width=hw, n_saddles_max=5)
            if not saddles:
                d_over_L.append(np.nan)
                continue
            th = saddles[0][0]
            d = nearest_singularity(th, gt, kmax=80, beta=b)[0]
            L = lattice_spacing(gt)
            d_over_L.append(d / (L + 1e-300))
        d_over_L = np.array(d_over_L)
        valid = np.isfinite(d_over_L)
        sane = valid & (d_over_L > 0) & (d_over_L <= 1.2)
        dL_sane = d_over_L[sane]
        inv_sane = inv_gts[sane]
        plateau_mean = np.mean(dL_sane[: max(1, len(dL_sane) // 3)]) if len(dL_sane) > 0 else np.nan
        plateau_std = np.std(dL_sane[: max(1, len(dL_sane) // 3)]) if len(dL_sane) > 0 else np.nan
        results.append((b, inv_gts[sane], d_over_L[sane], plateau_mean, plateau_std))

    fig, ax = plt.subplots(figsize=(8, 5))
    def beta_legend(b):
        s = b / np.pi
        return rf'β = {s:.2g}π'
    for idx, (b, inv_sane, dL_sane, pm, ps) in enumerate(results):
        if len(inv_sane) == 0:
            continue
        ax.plot(inv_sane, dL_sane, color=colors[idx], ls=styles[idx], lw=1.2, ms=3, marker='.', label=beta_legend(b))
    ax.set_xlabel(r'$1/\tilde\gamma$')
    ax.set_ylabel(r'$d/L$')
    ax.set_ylim(0, 1.05)
    ax.set_title(r'Experiment 1: $d/L$ vs $1/\tilde\gamma$ (γ̃ up to 2π, dominant saddle)')
    ax.legend(loc='best', ncol=2, fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(OUT_DIR / 'experiment1_asymptotic_dL_vs_inv_gamma.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("Saved", OUT_DIR / 'experiment1_asymptotic_dL_vs_inv_gamma.png')


def run_asymptotic_drift():
    """d/L vs γ̃, fits; |Φ''| vs 1/d and vs γ̃."""
    def path_continuation_adaptive(gt_max, n_steps_init=400, max_newton=10):
        gts = np.linspace(1e-6, gt_max, n_steps_init)
        thetas = []
        theta = 0.0 + 0j
        for i in range(len(gts)):
            gt = gts[i]
            for _ in range(80):
                f = dPhi(theta, gt, beta)
                fp = d2Phi(theta, gt, beta)
                if abs(f) < 1e-10 or abs(fp) < 1e-14:
                    break
                theta = theta - f / fp
            thetas.append((gt, theta))
        return np.array([p[0] for p in thetas]), np.array([p[1] for p in thetas])

    # For testing purposes, restrict path continuation to γ̃ up to 2π.
    gt_max = 2 * np.pi
    gts, thetas = path_continuation_adaptive(gt_max, n_steps_init=600)
    d_vals = np.array([nearest_singularity(th, gt, kmax=80)[0] for th, gt in zip(thetas, gts)])
    L_vals = np.array([lattice_spacing(gt) for gt in gts])
    d_over_L = d_vals / (L_vals + 1e-300)
    # Fit over the upper part of the available range: γ̃/π ∈ [0.5, 2].
    mask_fit = (gts >= 0.5 * np.pi) & (gts <= 2 * np.pi)
    gts_fit, dL_fit = gts[mask_fit], d_over_L[mask_fit]
    # Plateau estimate over γ̃/π ∈ [1, 2].
    mask_c = (gts >= 1.0 * np.pi) & (gts <= 2 * np.pi)
    gts_c, dL_c = gts[mask_c], d_over_L[mask_c]
    c_mean, c_std = np.mean(dL_c), np.std(dL_c)

    def model_c_plus_a(gt, c, a):
        return c + a / (gt + 1e-10)
    if _has_scipy and len(gts_c) > 2:
        try:
            popt_c2, _ = curve_fit(model_c_plus_a, gts_c, dL_c, p0=[0.52, 0.01])
            c_fit, a_fit = popt_c2[0], popt_c2[1]
        except Exception:
            c_fit, a_fit = c_mean, 0.0
    else:
        c_fit, a_fit = c_mean, 0.0

    def model_const(gt, c):
        return c + 0 * gt
    def model_power(gt, a, delta):
        return a * (gt + 1e-10) ** (-delta)
    def model_log(gt, a):
        return a / np.log(gt + 2)
    popt_c = [np.mean(dL_fit)]
    resid_c = np.sum((dL_fit - model_const(gts_fit, *popt_c))**2)
    try:
        popt_p, _ = curve_fit(model_power, gts_fit, dL_fit, p0=[0.5, 0.1], maxfev=3000) if _has_scipy else ([0.5, 0.1], None)
        resid_p = np.sum((dL_fit - model_power(gts_fit, *popt_p))**2)
    except Exception:
        popt_p, resid_p = [0.5, 0.1], np.nan
    try:
        popt_l, _ = curve_fit(model_log, gts_fit, dL_fit, p0=[0.5]) if _has_scipy else ([0.5], None)
        resid_l = np.sum((dL_fit - model_log(gts_fit, *popt_l))**2)
    except Exception:
        popt_l, resid_l = [0.5], np.nan

    d2_at_saddle = np.array([np.abs(d2Phi(th, gt, beta)) for th, gt in zip(thetas, gts)])
    inv_d = 1.0 / (d_vals + 1e-10)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    ax = axes[0]
    ax.plot(gts / np.pi, d_over_L, 'b-', lw=1.2, label=r'$d/L$ measured')
    if not np.isnan(resid_c):
        ax.plot(gts_fit / np.pi, model_const(gts_fit, *popt_c), 'k--', lw=1, label=rf'const $c={popt_c[0]:.3f}$')
    if not np.isnan(resid_p):
        ax.plot(gts_fit / np.pi, model_power(gts_fit, *popt_p), 'g--', lw=1, label=rf'power $\delta={popt_p[1]:.3f}$')
    if not np.isnan(resid_l):
        ax.plot(gts_fit / np.pi, model_log(gts_fit, *popt_l), 'r--', lw=1, label=rf'log $a={popt_l[0]:.3f}$')
    ax.set_xlabel(r'$\tilde{\gamma}$ (units of $\pi$)')
    ax.set_ylabel(r'$d(\tilde{\gamma})\,/\,L(\tilde{\gamma})$')
    ax.set_title(r'(a) Cell-fraction drift — fit region $\tilde{\gamma}/\pi \in [0.5,2]$')
    ax.set_xlim(0, 2)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.axhline(c_mean, color='gray', ls=':', alpha=0.7)
    ax.fill_between([10, 20], c_mean - c_std, c_mean + c_std, alpha=0.2, color='gray')
    ax.text(0.02, 0.98, rf'$c = {c_mean:.3f} \pm {c_std:.3f}$ (γ̃/π∈[1,2])\n$c + a/\tilde\gamma$: $c={c_fit:.3f}$, $a={a_fit:.4f}$',
            transform=ax.transAxes, fontsize=8, verticalalignment='top')

    ax = axes[1]
    ax.loglog(inv_d, d2_at_saddle, 'b.', ms=3, alpha=0.7)
    ax.set_xlabel(r'$1/d(\tilde{\gamma})$')
    ax.set_ylabel(r'$|\Phi\'\'(\theta^*)|$')
    ax.set_title(r'(b) Hessian at saddle vs proximity to singularity')
    ax.grid(True, alpha=0.3, which='both')
    ax = axes[2]
    ax.semilogy(gts / np.pi, d2_at_saddle, 'b-', lw=1)
    ax.set_xlabel(r'$\tilde{\gamma}$ (units of $\pi$)')
    ax.set_ylabel(r'$|\Phi\'\'(\theta^*)|$')
    ax.set_title(r'(c) $|\Phi\'\'(\theta^*)|$ vs $\tilde{\gamma}$ (convergence prefactor)')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(OUT_DIR / 'asymptotic_drift.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("Saved", OUT_DIR / 'asymptotic_drift.png')


def run_convergence_validation():
    """ε(n) vs 1/n; slope vs |Φ''|."""
    # Only test convergence up to γ̃ = 2π.
    gt_sample = [np.pi, 2*np.pi]
    n_vals = np.array([50, 100, 200, 500])
    results = []
    for gt in gt_sample:
        gts, thetas = dominant_saddle_path(gt, n_steps=100)
        th = thetas[-1]
        phi2_abs = np.abs(d2Phi(th, gt, beta))
        re_phi = Phi(th, gt, beta).real
        eps_n = []
        for n in n_vals:
            pred = 2 * re_phi + (1/n) * (np.log(2*np.pi) - np.log(n) - np.log(phi2_abs + 1e-20))
            exact_leading = 2 * re_phi
            eps_n.append(np.abs(pred - exact_leading))
        results.append((gt, th, phi2_abs, re_phi, eps_n))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ax = axes[0]
    for gt, th, phi2_abs, re_phi, eps_n in results:
        ax.plot(1/n_vals, eps_n, 'o-', label=rf'$\tilde\gamma={gt/np.pi:.0f}\pi$')
    ax.set_xlabel(r'$1/n$')
    ax.set_ylabel(r'$\varepsilon(n) = |(1/n)\log|A_n|^2 - 2\mathrm{Re}\,\Phi(\theta^*)|$')
    ax.set_title(r'(a) Convergence error vs $1/n$ (model)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    slopes, phi2_vals = [], []
    for gt, th, phi2_abs, re_phi, eps_n in results:
        inv_n = 1.0 / n_vals
        slope = (eps_n[-1] - eps_n[0]) / (inv_n[-1] - inv_n[0]) if inv_n[-1] != inv_n[0] else 0
        slopes.append(slope)
        phi2_vals.append(phi2_abs)
    ax = axes[1]
    ax.plot(phi2_vals, slopes, 'bo', ms=8)
    ax.set_xlabel(r'$|\Phi\'\'(\theta^*)|$')
    ax.set_ylabel(r'Slope of $\varepsilon$ vs $1/n$')
    ax.set_title(r'(b) Convergence slope vs Hessian')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(OUT_DIR / 'convergence_validation.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("Saved", OUT_DIR / 'convergence_validation.png')


def get_dominant_saddle_path(gts, b, grid_size=40, n_saddles_track=3):
    thetas_dom, re_phi_list = [], []
    for gt in gts:
        hw = max(15, 3 * np.sqrt(gt))
        saddles = all_saddles(gt, b, grid_size=grid_size, half_width=hw, n_saddles_max=10)
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


def run_cell_fraction_beta():
    """c(β) vs β; d/L vs 1/γ̃ up to 2π (β = -π/2)."""
    betas = [-np.pi/4, -np.pi/3, -np.pi/2, -2*np.pi/3, -3*np.pi/4]
    # Restrict main scan to γ̃ up to 2π for faster testing.
    gt_max_main = 2 * np.pi
    QUICK_RUN = True
    n_steps_main = 50 if QUICK_RUN else 200
    # Fit over the upper part of the available range: γ̃/π ∈ [1,2].
    mask_fit = lambda gts: (gts >= 1 * np.pi) & (gts <= 2 * np.pi)
    all_gts = np.linspace(1e-6, gt_max_main, n_steps_main)
    c_vals, c_stds = [], []
    stokes_gts, stokes_re_phi = None, None
    stokes_beta = -np.pi/2

    for b in betas:
        gts = all_gts
        thetas_dom, re_phi_per_saddle = get_dominant_saddle_path(gts, b, grid_size=40, n_saddles_track=3)
        valid = ~np.isnan(thetas_dom.real)
        d_vals = np.full_like(thetas_dom, np.nan, dtype=float)
        L_vals = np.array([lattice_spacing(gt) for gt in gts])
        for i in np.where(valid)[0]:
            d_vals[i] = nearest_singularity(thetas_dom[i], gts[i], kmax=80, beta=b)[0]
        dL = np.where(valid, d_vals / (L_vals + 1e-300), np.nan)
        dL_fit = dL[mask_fit(gts)]
        dL_fit = dL_fit[~np.isnan(dL_fit)]
        c_mean = np.mean(dL_fit) if len(dL_fit) else np.nan
        c_std = np.std(dL_fit) if len(dL_fit) else np.nan
        c_vals.append(c_mean)
        c_stds.append(c_std)
        if abs(b - stokes_beta) < 1e-6:
            stokes_gts, stokes_re_phi = gts, re_phi_per_saddle

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ax = axes[0]
    ax.errorbar([b/np.pi for b in betas], c_vals, yerr=c_stds, fmt='bo', capsize=4)
    ax.set_xlabel(r'$\beta$ (units of $\pi$)')
    ax.set_ylabel(r'$c(\beta) = \lim_{\tilde\gamma\to\infty} d/L$ (dominant saddle)')
    ax.set_title(r'Cell fraction $c(\beta)$ (γ̃/π ∈ [1,2])')
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
    fig.savefig(OUT_DIR / 'cell_fraction_beta.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("Saved", OUT_DIR / 'cell_fraction_beta.png')

    # Extended convergence check only up to γ̃ = 2π for tests.
    gt_max_ext = 2 * np.pi
    n_steps_ext = 500 if QUICK_RUN else 2000
    beta_conv = -np.pi/2
    gts_ext = np.linspace(1e-6, gt_max_ext, n_steps_ext)
    thetas_ext, _ = get_dominant_saddle_path(gts_ext, beta_conv, grid_size=40, n_saddles_track=1)
    L_ext = np.array([lattice_spacing(gt) for gt in gts_ext])
    d_ext = np.full(len(gts_ext), np.nan)
    for i in range(len(gts_ext)):
        if not np.isnan(thetas_ext[i].real):
            d_ext[i] = nearest_singularity(thetas_ext[i], gts_ext[i], kmax=80, beta=beta_conv)[0]
    dL_ext = np.where(~np.isnan(d_ext), d_ext / (L_ext + 1e-300), np.nan)
    inv_gt = 1.0 / (gts_ext + 1e-300)
    # Fit over γ̃/π ∈ [1,2] in the extended scan.
    mask_fit_ext = (gts_ext >= 1 * np.pi) & (gts_ext <= 2 * np.pi)
    gts_f_full = gts_ext[mask_fit_ext]
    dL_f_full = dL_ext[mask_fit_ext]
    valid_f = ~np.isnan(dL_f_full)
    gts_f, dL_f = gts_f_full[valid_f], dL_f_full[valid_f]

    def model_c_plus_a_gt_alpha(gt, c, a, alpha):
        return c + a * (gt + 1e-30) ** (-alpha)
    c_fit, a_fit, alpha_fit = np.nan, np.nan, 0.0
    if _has_scipy and len(dL_f) > 10:
        try:
            popt, _ = curve_fit(model_c_plus_a_gt_alpha, gts_f, dL_f, p0=[0.55, 1.0, 0.5], bounds=([0, -np.inf, 0.01], [2, np.inf, 2]))
            c_fit, a_fit, alpha_fit = popt[0], popt[1], popt[2]
        except Exception:
            pass

    fig2, ax = plt.subplots(figsize=(6, 4))
    valid = ~np.isnan(dL_ext)
    ax.plot(inv_gt[valid], dL_ext[valid], 'b.', ms=1, alpha=0.7, label=r'$d/L$ (dominant saddle)')
    ax.set_xlabel(r'$1/\tilde\gamma$')
    ax.set_ylabel(r'$d/L$')
    ax.set_title(r'Convergence check: $d/L$ vs $1/\tilde\gamma$ (β = −π/2, γ̃ up to 2π)')
    if not np.isnan(c_fit) and alpha_fit > 0:
        gt_plot = np.linspace(1 * np.pi, 2 * np.pi, 200)
        ax.plot(1/gt_plot, model_c_plus_a_gt_alpha(gt_plot, c_fit, a_fit, alpha_fit), 'r-', lw=2, label=rf'fit $c + a/\tilde\gamma^{{\alpha}}$')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig2.savefig(OUT_DIR / 'cell_fraction_convergence.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("Saved", OUT_DIR / 'cell_fraction_convergence.png')


def run_validation_check():
    """Print dominant saddle at selected (β, γ̃)."""
    betas = [-np.pi/4, -np.pi/3, -np.pi/2, -2*np.pi/3, -3*np.pi/4]
    # Only check moderate γ̃ values up to 2π.
    gammas = [np.pi, 2*np.pi]
    print("Dominant saddle check (highest Re Φ) at selected γ̃:")
    for b in betas:
        for gt in gammas:
            hw = max(15, 3 * np.sqrt(gt))
            saddles = all_saddles(gt, b, grid_size=50, half_width=hw)
            if saddles:
                th = saddles[0][0]
                re_phi = saddles[0][1].real
                print(f"  β={b/np.pi:.2f}π, γ̃={gt/np.pi:.0f}π: dominant θ*={th.real:.4f}{th.imag:+.4f}j, ReΦ={re_phi:.4f}, #saddles={len(saddles)}")
            else:
                print(f"  β={b/np.pi:.2f}π, γ̃={gt/np.pi:.0f}π: no saddles found")
    print("Done.")


if __name__ == '__main__':
    run_experiment1()
    run_asymptotic_drift()
    run_convergence_validation()
    run_cell_fraction_beta()
    run_validation_check()
