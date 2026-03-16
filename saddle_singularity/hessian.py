"""
Hessian at saddle: scaling vs d, spectrum, decomposition, and cancellation (phase structure).

Combines:
- |Φ''(θ*)| vs d, 1/d, 1/d² and decomposition Φ'' = -1/2 + (singularity contribution)
- Higher derivatives and lattice sum S_K = Σ 1/(θ* - θ_k)²
- Phase structure: c_k = 1/(θ* - θ_k)², R_mag, N_eff_mag
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from qaoa_core import (
    dominant_saddle_path, nearest_singularity, d2Phi, d3Phi, w_of_gamma, A,
    singularities, DEFAULT_BETA,
)

OUT_DIR = Path(__file__).resolve().parent
beta = DEFAULT_BETA
plt.rcParams.update({'font.size': 10, 'figure.dpi': 150})


def _d2Phi_decomposed(theta, gt, beta_=beta):
    ww = w_of_gamma(gt)
    s = np.sin(beta_ / 2)
    e = np.exp(theta * ww)
    aa = A(theta, gt, beta_) + 1e-300
    term1 = -0.5
    term2 = -1j * ww**2 * s * e / aa
    term3 = -(ww * s * e / aa) ** 2
    return term1, term2, term3


def _d4Phi_num(theta, gt, beta_=beta, h=1e-6):
    return (d3Phi(theta + h, gt, beta_) - d3Phi(theta - h, gt, beta_)) / (2 * h)


def _d5Phi_num(theta, gt, beta_=beta, h=1e-6):
    return (_d4Phi_num(theta + h, gt, beta_) - _d4Phi_num(theta - h, gt, beta_)) / (2 * h)


def _get_contributions(theta_star, gt, kmax=50, n_nearest=10, beta_=beta):
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


def _angle_saddle_to_lattice(theta_star, gt, beta_=beta):
    d, theta_k = nearest_singularity(theta_star, gt, kmax=50, beta=beta_)
    ww = w_of_gamma(gt)
    lattice_dir = 2j * np.pi / ww
    diff = theta_star - theta_k
    if np.abs(diff) < 1e-14:
        return np.nan
    cos_angle = np.real(diff * np.conj(lattice_dir)) / (np.abs(diff) * np.abs(lattice_dir) + 1e-20)
    cos_angle = np.clip(cos_angle, -1, 1)
    return np.arccos(cos_angle)


def _log_fit_slope(x, y):
    mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    if np.sum(mask) < 5:
        return np.nan
    return np.polyfit(np.log(x[mask]), np.log(y[mask]), 1)[0]


def run_scaling_and_decomposition():
    """|Φ''| vs d, 1/d, 1/d²; decomposition Φ'' = -1/2 + (singularity part)."""
    # Restrict γ̃ scan to 2π for faster tests.
    gt_max = 2 * np.pi
    n_steps = 300
    gts, thetas = dominant_saddle_path(gt_max, n_steps=n_steps, beta=beta)
    d_vals = np.array([nearest_singularity(th, gt, kmax=80, beta=beta)[0] for th, gt in zip(thetas, gts)])
    phi2_vals = np.array([d2Phi(th, gt, beta) for th, gt in zip(thetas, gts)])
    phi2_abs = np.abs(phi2_vals)
    term1_vals, term2_vals, term3_vals = [], [], []
    for th, gt in zip(thetas, gts):
        t1, t2, t3 = _d2Phi_decomposed(th, gt)
        term1_vals.append(t1)
        term2_vals.append(t2)
        term3_vals.append(t3)
    term1_vals = np.array(term1_vals)
    term2_vals = np.array(term2_vals)
    term3_vals = np.array(term3_vals)
    sing_part = term2_vals + term3_vals
    sing_abs = np.abs(sing_part)
    w_vals = np.array([np.abs(w_of_gamma(gt)) for gt in gts])
    valid = np.isfinite(phi2_abs) & (d_vals > 1e-10) & np.isfinite(d_vals)
    gts_v, d_v = gts[valid], d_vals[valid]
    phi2_v = phi2_abs[valid]
    inv_d = 1.0 / d_v
    inv_d2 = 1.0 / (d_v ** 2)
    log_d = np.log(d_v + 1e-30)
    sing_v = sing_abs[valid]
    w_v = w_vals[valid]
    slope_d = _log_fit_slope(d_v, phi2_v)
    slope_inv_d = _log_fit_slope(inv_d, phi2_v)
    slope_inv_d2 = _log_fit_slope(inv_d2, phi2_v)
    slope_w = _log_fit_slope(w_v, phi2_v)

    fig1, axes = plt.subplots(2, 2, figsize=(9, 8))
    ax = axes[0, 0]
    ax.loglog(d_v, phi2_v, 'b.', ms=2, alpha=0.7)
    ax.set_xlabel(r'$d$ (distance to nearest singularity)')
    ax.set_ylabel(r'$|\Phi\'\'(\theta^*)|$')
    ax.set_title(rf'$|\Phi\'\'|$ vs $d$ (slope ≈ {slope_d:.2f})')
    ax.grid(True, alpha=0.3)
    ax = axes[0, 1]
    ax.loglog(inv_d, phi2_v, 'b.', ms=2, alpha=0.7)
    ax.set_xlabel(r'$1/d$')
    ax.set_ylabel(r'$|\Phi\'\'(\theta^*)|$')
    ax.set_title(rf'$|\Phi\'\'|$ vs $1/d$ (slope ≈ {slope_inv_d:.2f})')
    ax.grid(True, alpha=0.3)
    ax = axes[1, 0]
    ax.loglog(inv_d2, phi2_v, 'b.', ms=2, alpha=0.7)
    ax.set_xlabel(r'$1/d^2$')
    ax.set_ylabel(r'$|\Phi\'\'(\theta^*)|$')
    ax.set_title(rf'$|\Phi\'\'|$ vs $1/d^2$ (slope ≈ {slope_inv_d2:.2f})')
    ax.grid(True, alpha=0.3)
    ax = axes[1, 1]
    ax.semilogy(log_d, phi2_v, 'b.', ms=2, alpha=0.7)
    ax.set_xlabel(r'$\log d$')
    ax.set_ylabel(r'$|\Phi\'\'(\theta^*)|$')
    ax.set_title(r'$|\Phi\'\'|$ vs $\log d$')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig1.savefig(OUT_DIR / 'hessian_scaling_vs_d.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("Saved", OUT_DIR / 'hessian_scaling_vs_d.png')

    fig2, axes = plt.subplots(2, 1, figsize=(7, 7))
    ax = axes[0]
    ax.semilogy(gts / np.pi, phi2_abs, 'k-', lw=1.2, label=r'$|\Phi\'\'|$')
    ax.semilogy(gts / np.pi, np.abs(term1_vals) + 1e-30, 'g-', label=r'$|-1/2|$ (Gaussian)')
    ax.semilogy(gts / np.pi, sing_abs + 1e-30, 'b-', lw=1, label=r'$|$singularity part$|$')
    ax.set_xlabel(r'$\tilde\gamma / \pi$')
    ax.set_ylabel(r'magnitude')
    ax.set_title(r'Decomposition: $\Phi\'\' = -1/2 +$ (singularity contribution)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax = axes[1]
    ax.plot(gts / np.pi, phi2_abs, 'k-', lw=1.2, label=r'$|\Phi\'\'|$')
    ax.plot(gts / np.pi, w_vals, 'r-', lw=1, label=r'$|w(\tilde\gamma)|$')
    ax.set_xlabel(r'$\tilde\gamma / \pi$')
    ax.set_ylabel(r'magnitude')
    ax.set_title(r'$|\Phi\'\'|$ vs $|w(\tilde\gamma)|$ (lattice scale)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig2.savefig(OUT_DIR / 'hessian_decomposition.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("Saved", OUT_DIR / 'hessian_decomposition.png')


def run_spectrum():
    """Φ^(k)(θ*) for k=2,3,4,5; lattice sum S_K vs K."""
    # Restrict γ̃ scan to 2π for faster tests.
    gt_max = 2 * np.pi
    gts, thetas = dominant_saddle_path(gt_max, n_steps=250)
    d2_vals = np.array([d2Phi(th, gt, beta) for th, gt in zip(thetas, gts)])
    d3_vals = np.array([d3Phi(th, gt, beta) for th, gt in zip(thetas, gts)])
    d4_vals = np.array([_d4Phi_num(th, gt, beta) for th, gt in zip(thetas, gts)])
    d5_vals = np.array([_d5Phi_num(th, gt, beta) for th, gt in zip(thetas, gts)])
    d2_abs = np.abs(d2_vals)
    # Sample only up to γ̃ = 2π.
    gt_sample = [np.pi, 2*np.pi]
    kmax_sings = 60

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ax = axes[0]
    ax.semilogy(gts / np.pi, d2_abs, 'b-', lw=1.2, label=r'$|\Phi\'\'|$')
    ax.semilogy(gts / np.pi, np.abs(d3_vals), 'g-', lw=1, label=r'$|\Phi^{(3)}|$')
    ax.semilogy(gts / np.pi, np.abs(d4_vals), 'r-', lw=1, label=r'$|\Phi^{(4)}|$')
    ax.semilogy(gts / np.pi, np.abs(d5_vals), 'm-', lw=1, label=r'$|\Phi^{(5)}|$')
    ax.set_xlabel(r'$\tilde{\gamma}$ (units of $\pi$)')
    ax.set_ylabel(r'$|\Phi^{(k)}(\theta^*)|$')
    ax.set_title(r'(a) Higher derivatives at dominant saddle')
    ax.set_xlim(0.1, 20)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, which='both')

    ax = axes[1]
    for gt in gt_sample:
        idx = np.argmin(np.abs(gts - gt))
        th = thetas[idx]
        phi2_exact = d2Phi(th, gt, beta)
        sings = singularities(gt, kmax=kmax_sings)
        K_list = list(range(1, min(51, kmax_sings + 1)))
        S_list = [sum(1.0 / (th - sk)**2 for sk in sings[kmax_sings - K : kmax_sings + K + 1]) for K in K_list]
        ax.plot(K_list, np.abs(S_list), '-', lw=1, label=rf'$\tilde\gamma={gt/np.pi:.1f}\pi$')
        ax.axhline(np.abs(phi2_exact + 0.5), ls='--', alpha=0.5)
    ax.set_xlabel(r'$K$ (partial sum $|k| \leq K$)')
    ax.set_ylabel(r'$|S_K|$')
    ax.set_title(r'(b) Lattice sum $S_K = \sum_{|k|\leq K} 1/(\theta^*-\theta_k)^2$')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(OUT_DIR / 'hessian_spectrum.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("Saved", OUT_DIR / 'hessian_spectrum.png')


def run_cancellation():
    """Argand c_k = 1/(θ* - θ_k)²; R_mag, N_eff_mag vs γ̃."""
    # Restrict γ̃ scan to 2π for faster tests.
    gts, thetas = dominant_saddle_path(2 * np.pi, n_steps=200)
    kmax_c = 80
    R_complex_vals, R_magnitude_vals, N_eff_mag_vals = [], [], []
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
    R_magnitude_vals = np.array(R_magnitude_vals)
    N_eff_mag_vals = np.array(N_eff_mag_vals)

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for i_plot, gt in enumerate([np.pi, 2*np.pi]):
        ax = axes[i_plot]
        idx = np.argmin(np.abs(gts - gt))
        th = thetas[idx]
        c_list, cumsum = _get_contributions(th, gt, n_nearest=10)
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
    ax2 = axes[2]
    ax2.plot(gts / np.pi, R_magnitude_vals, 'b-', lw=1.5, label=r'$R_{\mathrm{mag}}$')
    ax2.plot(gts / np.pi, N_eff_mag_vals, 'r-', lw=1.5, label=r'$N_{\mathrm{eff,mag}}$')
    ax2.axhline(1.0, color='gray', ls='--', alpha=0.5)
    ax2.set_xlabel(r'$\tilde{\gamma}$ (units of $\pi$)')
    ax2.set_ylabel('Ratio / count')
    ax2.set_title(r'(b) $R_{\mathrm{mag}}$, $N_{\mathrm{eff,mag}}$')
    ax2.set_xlim(0, 2)
    ax2.set_ylim(bottom=0)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(OUT_DIR / 'hessian_cancellation.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("Saved", OUT_DIR / 'hessian_cancellation.png')


if __name__ == '__main__':
    run_scaling_and_decomposition()
    run_spectrum()
    run_cancellation()
