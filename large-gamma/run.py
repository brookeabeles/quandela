"""
Diagrams and graphs as proof of the Large Gamma Proof (Large Gamma Proof (2).pdf).

Generates:
  1. Saddle analysis: dominant (green, small Im) vs subdominant (red, larger Im); complex conjugate pairs
  2. Saddle existence: |Φ'(θ)| along rays → linear growth ⇒ Φ' must have zeros
  3. Dominant saddle path: θ*(γ̃) in the complex plane as γ̃ runs 0 → 2π
  4. Singularities and saddles: lattice of branch points and valley structure
  (Ridge/term competition figure is in term_competition.)
"""

from pathlib import Path
OUT_DIR = Path(__file__).resolve().parent

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    from test_proof import Phi, dPhi, find_saddles
except ImportError:
    def Phi(theta, beta, gt):
        w = np.sqrt(-1j * gt / 2 + 0j)
        arg = np.cos(beta/2) - 1j*np.sin(beta/2)*np.exp(theta*w)
        return -np.log(2)/2 - 1j*beta/2 - theta**2/4 + np.log(arg + 1e-300)
    def dPhi(theta, beta, gt):
        w = np.sqrt(-1j * gt / 2 + 0j)
        ez = np.exp(theta * w)
        return (-theta/2 + (-1j*np.sin(beta/2)*w*ez) / (np.cos(beta/2) - 1j*np.sin(beta/2)*ez + 1e-300))
    def find_saddles(beta, gt, half_width=6, grid=50):
        found = []
        h = 1e-6
        for re0 in np.linspace(-half_width, half_width, grid):
            for im0 in np.linspace(-half_width, half_width, grid):
                th = complex(re0, im0)
                for _ in range(500):
                    f = dPhi(th, beta, gt)
                    f2 = (dPhi(th+h, beta, gt) - dPhi(th-h, beta, gt))/(2*h)
                    if abs(f2) < 1e-13:
                        break
                    step = f/f2
                    if abs(step) > 0.5:
                        step *= 0.5/abs(step)
                    th -= step
                    if abs(f) < 1e-10:
                        break
                if abs(dPhi(th, beta, gt)) < 1e-8:
                    if all(abs(th - s[0]) > 5e-4 for s in found):
                        found.append((th, Phi(th, beta, gt)))
        found.sort(key=lambda x: -x[1].real)
        return found

try:
    from proof_demo import find_singularities
except ImportError:
    def find_singularities(beta, gt, half_width=8, k_max=20):
        w = np.sqrt(-1j * gt / 2 + 0j)
        cot_half = np.cos(beta/2) / (np.sin(beta/2) + 1e-300)
        z0 = -1j * cot_half
        r0, phi0 = np.abs(z0), np.angle(z0)
        out = []
        for k in range(-k_max, k_max + 1):
            th = (np.log(r0) + 1j * (phi0 + 2*np.pi*k)) / w
            if abs(th.real) <= half_width and abs(th.imag) <= half_width:
                out.append(th)
        return out

def _newton_refine(theta, beta, gt, tol=1e-11, max_iter=50, h=1e-8):
    for _ in range(max_iter):
        f = dPhi(theta, beta, gt)
        if abs(f) < tol:
            return theta
        f2 = (dPhi(theta + h, beta, gt) - dPhi(theta - h, beta, gt)) / (2 * h)
        if abs(f2) < 1e-14:
            return theta
        step = f / f2
        if abs(step) > 1.0:
            step *= 1.0 / abs(step)
        theta = theta - step
    return theta

BETA = -np.pi/2


# ----- 1. SADDLE ANALYSIS: dominant (max Re[Φ]) vs subdominant -----
def plot_saddle_analysis(beta=BETA, gt=2*np.pi, half=5, n_grid=80):
    """
    PDF: One dominant saddle (largest Re[Φ(θ*)]) = green; subdominant = red.
    Dominant is defined by max Re[Φ], not by smallest |Im θ| (some subdominants can have smaller |Im θ|).
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    re_theta = np.linspace(-half, half, n_grid)
    im_theta = np.linspace(-half, half, n_grid)
    ReT, ImT = np.meshgrid(re_theta, im_theta)
    Z = np.zeros_like(ReT, dtype=float)
    for i in range(ReT.shape[0]):
        for j in range(ReT.shape[1]):
            Z[i, j] = Phi(complex(ReT[i, j], ImT[i, j]), beta, gt).real
    vmin, vmax = np.nanpercentile(Z, [2, 98])
    cf = axes[0].contourf(ReT, ImT, Z, levels=40, cmap='viridis', vmin=vmin, vmax=vmax)
    saddles = find_saddles(beta, gt, half_width=half, grid=40)
    if saddles:
        dom = saddles[0][0]
        axes[0].scatter(dom.real, dom.imag, s=180, c='lime', edgecolors='darkgreen', linewidths=2.5,
                        label=r'Dominant (max Re$\Phi$)', zorder=5)
        axes[0].scatter([s[0].real for s in saddles[1:]], [s[0].imag for s in saddles[1:]],
                        s=100, c='red', edgecolors='darkred', linewidths=1.5, label=r'Subdominant', zorder=5)
    axes[0].set_xlabel(r'Re $\theta$')
    axes[0].set_ylabel(r'Im $\theta$')
    axes[0].set_title(r'Saddles at $\tilde\gamma=2\pi$, $\beta=-\pi/2$: dominant (green) vs subdominant (red)')
    axes[0].legend(loc='upper right', fontsize=8)
    axes[0].set_aspect('equal')
    plt.colorbar(cf, ax=axes[0], shrink=0.7, label=r'Re$\Phi(\theta)$')

    if saddles:
        # Middle: Re[Φ(θ*)] — dominant = index 0 = highest bar
        re_phi_vals = [s[1].real for s in saddles]
        colors = ['green'] + ['red'] * (len(saddles) - 1)
        axes[1].bar(range(len(saddles)), re_phi_vals, color=colors, edgecolor='black')
        axes[1].set_xlabel('Saddle index (0 = dominant)')
        axes[1].set_ylabel(r'Re$\Phi(\theta^*)$')
        axes[1].set_title(r'Dominant = saddle with largest Re$\Phi(\theta^*)$')
        axes[1].axhline(0, color='gray', ls='--', alpha=0.5)

        # Right: |Im θ*| (dominant need not have smallest |Im θ|)
        im_vals = [abs(s[0].imag) for s in saddles]
        axes[2].bar(range(len(saddles)), im_vals, color=colors, edgecolor='black')
        axes[2].set_xlabel('Saddle index (0 = dominant)')
        axes[2].set_ylabel(r'|Im $\theta^*$|')
        axes[2].set_title(r'|Im $\theta^*$| (by index; dominant $\neq$ smallest |Im$\theta$|)')
    plt.tight_layout()
    return fig


# ----- 2b. SADDLE QUANTITIES VS β (fixed γ̃) -----
def plot_saddles_vs_beta(gt=2*np.pi, beta_min=-2, beta_max=2, n_beta=41, half_width=6, grid=35):
    """
    Vary β from beta_min to beta_max; plot number of saddles and dominant saddle
    prediction 2*Re[Φ(θ*)] (the asymptotic (1/n)log|A_n|² limit).
    """
    betas = np.linspace(beta_min, beta_max, n_beta)
    n_saddles = []
    pred = []  # 2 * Re[Φ(θ*)] for dominant saddle
    for b in betas:
        saddles = find_saddles(b, gt, half_width=half_width, grid=grid)
        n_saddles.append(len(saddles))
        if saddles:
            pred.append(2 * saddles[0][1].real)
        else:
            pred.append(np.nan)
    pred = np.array(pred)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    axes[0].plot(betas, n_saddles, 'b-o', markersize=4)
    axes[0].axvline(-np.pi/2, color='green', ls='--', alpha=0.8, label=r'$\beta=-\pi/2$ (QAOA optimal)')
    axes[0].set_xlabel(r'$\beta$')
    axes[0].set_ylabel('Number of saddles')
    axes[0].set_title(r'Number of saddles vs $\beta$ (fixed $\tilde{\gamma}=2\pi$)')
    axes[0].legend(loc='upper right', fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(betas, pred, 'b-o', markersize=4)
    axes[1].axvline(-np.pi/2, color='green', ls='--', alpha=0.8, label=r'$\beta=-\pi/2$')
    axes[1].axhline(0, color='gray', ls=':', alpha=0.5)
    axes[1].set_xlabel(r'$\beta$')
    axes[1].set_ylabel(r'$2\,\mathrm{Re}\,\Phi(\theta^*)$ (saddle prediction)')
    axes[1].set_title(r'Dominant saddle prediction vs $\beta$ (fixed $\tilde{\gamma}=2\pi$)')
    axes[1].legend(loc='upper right', fontsize=8)
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(r'Saddle structure vs $\beta$ ($\tilde{\gamma}=2\pi$)', fontsize=12)
    plt.tight_layout()
    return fig


# ----- 3. SADDLE EXISTENCE: |Φ'(θ)| grows linearly along rays -----
def plot_phiprime_growth(beta=BETA, gt=2*np.pi, R_max=15, n_R=150):
    """
    PDF: Φ'(θ) = -θ/2 + O(e^(aθ)); along rays with Re(aθ)≠0, |Φ'| ~ |θ|/2 → ∞.
    Entire function with no zeros would be e^g(θ) (exponential growth); linear growth
    ⇒ Φ' must have zeros ⇒ saddles exist.
    Angles in 0.5π multiples (6 rays) plus two rays near singularities at 0.75π, 1.75π.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    R = np.linspace(0.5, R_max, n_R)
    # 6 rays in 0.5π multiples: φ/π = 0, 0.5, 1, 1.5, 2, 2.5
    angles_05 = [k * 0.5 * np.pi for k in range(6)]
    # 2 rays near singularities
    phi_sing1 = 0.75 * np.pi
    phi_sing2 = 1.75 * np.pi

    # Plot 6 rays (0.5 multiples) in orange, one legend entry
    for i, phi in enumerate(angles_05):
        theta_ray = R * np.exp(1j * phi)
        abs_phip = [abs(dPhi(t, beta, gt)) for t in theta_ray]
        label = r"6 rays: $|\Phi'|\sim|\theta|/2$ or small" if i == 0 else None
        ax.plot(R, abs_phip, '-', alpha=0.8, color='orange', label=label)
    # Plot rays near singularities
    for th_ray, ph, color, lbl in [
        (R * np.exp(1j * phi_sing1), phi_sing1, 'darkblue', r'Ray near singularity ($\varphi=0.75\pi$)'),
        (R * np.exp(1j * phi_sing2), phi_sing2, 'teal', r'Ray near singularity ($\varphi=1.75\pi$)'),
    ]:
        abs_phip = [abs(dPhi(t, beta, gt)) for t in th_ray]
        ax.plot(R, abs_phip, '-', alpha=0.8, color=color, label=lbl)
    # Linear reference
    ax.plot(R, R/2, 'k--', linewidth=2, marker='o', markersize=3, markevery=5, label=r'$|\theta|/2$ (linear)')

    ax.set_xlabel(r'$R = |\theta|$')
    ax.set_ylabel(r"$|\Phi'(\theta)|$")
    ax.set_title(r"Proof of saddle existence: $|\Phi'|\sim|\theta|/2\to\infty$ along rays $\Rightarrow$ $\Phi'$ has zeros (fixed $\tilde{\gamma}=2\pi$)")
    ax.legend(loc='upper right', fontsize=7, ncol=1)
    ax.set_yscale('linear')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


# ----- 4. DOMINANT SADDLE PATH θ*(γ̃) in the complex plane -----
def plot_dominant_saddle_path(beta=BETA, gt_max=2*np.pi, n_steps=60):
    """
    PDF: Dominant saddle is the branch θ*(γ̃) with θ*(0)=0 (path continuation).
    March γ̃ from 0 to gt_max, at each step Newton-refine from previous θ*.
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    gt_vals = np.linspace(1e-5, gt_max, n_steps)
    path_re, path_im = [], []
    theta = 0.0 + 0.0j
    for gt in gt_vals:
        theta = _newton_refine(theta, beta, gt)
        path_re.append(theta.real)
        path_im.append(theta.imag)
    ax.plot(path_re, path_im, 'b-', linewidth=2, label=r'$\theta^*(\tilde\gamma)$')
    ax.scatter([path_re[0]], [path_im[0]], s=120, c='green', edgecolors='darkgreen', linewidths=2, label=r'$\tilde\gamma\to 0$')
    ax.scatter([path_re[-1]], [path_im[-1]], s=120, c='red', edgecolors='darkred', linewidths=2, label=r'$\tilde\gamma=2\pi$')
    ax.axhline(0, color='gray', ls='--', alpha=0.5)
    ax.axvline(0, color='gray', ls='--', alpha=0.5)
    ax.set_xlabel(r'Re $\theta^*$')
    ax.set_ylabel(r'Im $\theta^*$')
    ax.set_title(r'Dominant saddle path: $\theta^*(0)=0$, continuation to $\tilde\gamma=2\pi$')
    ax.legend(loc='best', fontsize=9)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


# ----- 4a. VERIFICATION: singularities ≠ saddles (see singularities_vs_saddles_proof.md) -----
def verify_singularities_disjoint_from_saddles(beta=BETA, gt=2*np.pi, half=8, tol=1e-4):
    """
    Numerical check: min distance between any saddle and any singularity is > tol.
    Proof: at a singularity L(θ)=0 so Φ'(θ) = -θ/2 + L'/L is undefined; saddles require Φ'(θ*)=0 hence defined.
    """
    saddles = find_saddles(beta, gt, half_width=half, grid=40)
    sing = find_singularities(beta, gt, half_width=half, k_max=20)
    if not saddles or not sing:
        return True, float('inf')
    min_dist = min(abs(s[0] - sigma) for s in saddles for sigma in sing)
    return min_dist > tol, min_dist


def plot_singularities_saddles_proof_diagram(beta=BETA, gt_vals=None, half=8, out_path='singularities_saddles_proof.png'):
    """
    Proof as diagram: (1) Schematic logic — singularities vs saddles are disjoint;
    (2) Numerical evidence — min saddle–singularity distance > 0 for several γ̃.
    """
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ----- Panel 1: Schematic proof -----
    ax1 = axes[0]
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.axis('off')
    # Boxes and text
    box_kw = dict(boxstyle='round,pad=0.4', facecolor='lightyellow', edgecolor='black', linewidth=1.2)
    ax1.text(5, 8.5, 'Proof: singularities and saddles never coincide', ha='center', fontsize=12, fontweight='bold')
    ax1.text(2.5, 6.8, r'Singularity: $L(\theta)=0$ (arg of log)', ha='center', fontsize=10, bbox=box_kw)
    ax1.text(2.5, 4.2, r"$\Phi' = -\theta/2 + L'/L$" + '\n⇒ division by 0\n⇒ ' + r"$\Phi'$ undefined", ha='center', fontsize=9, bbox=dict(boxstyle='round,pad=0.3', facecolor='#ffcccc', edgecolor='black'))
    ax1.annotate('', xy=(2.5, 4.6), xytext=(2.5, 6.2), arrowprops=dict(arrowstyle='->', lw=1.5))
    ax1.text(7.5, 6.8, r"Saddle: $\Phi'(\theta^*)=0$", ha='center', fontsize=10, bbox=box_kw)
    ax1.text(7.5, 4.2, r"requires $\Phi'$ defined at $\theta^*$", ha='center', fontsize=9, bbox=dict(boxstyle='round,pad=0.3', facecolor='#ccffcc', edgecolor='black'))
    ax1.annotate('', xy=(7.5, 4.6), xytext=(7.5, 6.2), arrowprops=dict(arrowstyle='->', lw=1.5))
    ax1.text(5, 2.2, '⇒ No $\\theta$ can be both.\nSingularities and saddles are disjoint.', ha='center', fontsize=11, bbox=dict(boxstyle='round,pad=0.4', facecolor='#e6e6ff', edgecolor='blue', linewidth=1.5))
    ax1.annotate('', xy=(5, 2.8), xytext=(2.5, 3.7), arrowprops=dict(arrowstyle='->', lw=1.2))
    ax1.annotate('', xy=(5, 2.8), xytext=(7.5, 3.7), arrowprops=dict(arrowstyle='->', lw=1.2))
    ax1.set_title('Logical proof', fontsize=11)

    # ----- Panel 2: Min distance (saddle–singularity) vs γ̃ -----
    ax2 = axes[1]
    if gt_vals is None:
        gt_vals = np.linspace(0.3*np.pi, 3*np.pi, 12)
    distances = []
    for gt in gt_vals:
        _, d = verify_singularities_disjoint_from_saddles(beta=beta, gt=gt, half=half)
        distances.append(d)
    ax2.plot(gt_vals/np.pi, distances, 'b-o', markersize=6, linewidth=2)
    ax2.axhline(0, color='gray', ls='--', alpha=0.6)
    ax2.fill_between(gt_vals/np.pi, 0, distances, alpha=0.2, color='blue')
    ax2.set_xlabel(r'$\tilde{\gamma} / \pi$', fontsize=11)
    ax2.set_ylabel('Min distance (saddle ↔ singularity)', fontsize=10)
    ax2.set_title('Numerical evidence: distance always > 0', fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, max(distances) * 1.15)

    fig.suptitle(r'Singularities $\cap$ Saddles = $\emptyset$: proof and verification ($\beta=-\pi/2$)', fontsize=12)
    plt.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        print('Saved', out_path)
    plt.close(fig)
    return fig


# ----- 4b. SADDLE-POINT LANDSCAPE (standalone, publication-style) -----
def plot_saddle_landscape_nice(beta=BETA, gt=2*np.pi, half=8, n_grid=120, out_path='saddle-pt-landscape.png'):
    """
    Saddle-point landscape Re[Φ(θ)] with clear saddle and singularity markers.
    Singularities = log branch points (arg=0); saddles = critical points Φ'(θ)=0.
    """
    fig, ax = plt.subplots(figsize=(9, 7.5))
    re_theta = np.linspace(-half, half, n_grid)
    im_theta = np.linspace(-half, half, n_grid)
    ReT, ImT = np.meshgrid(re_theta, im_theta)
    Z = np.zeros_like(ReT, dtype=float)
    for i in range(ReT.shape[0]):
        for j in range(ReT.shape[1]):
            Z[i, j] = Phi(complex(ReT[i, j], ImT[i, j]), beta, gt).real
    vmin, vmax = np.nanpercentile(Z, [2, 98])
    cf = ax.contourf(ReT, ImT, Z, levels=50, cmap='viridis', vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(cf, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label(r'Re$\Phi(\theta)$', fontsize=11)

    saddles = find_saddles(beta, gt, half_width=half, grid=50)
    sing = find_singularities(beta, gt, half_width=half, k_max=20)

    # Singularities: bold black X with white outline so they show on any background
    if sing:
        for s in sing:
            ax.scatter(s.real, s.imag, s=140, c='white', marker='x', linewidths=4, zorder=6)
        ax.scatter([s.real for s in sing], [s.imag for s in sing], s=100, c='black', marker='x',
                   linewidths=2.5, label='Singularities (log arg=0)', zorder=7)
    # Saddles: dominant = green, others = red
    if saddles:
        ax.scatter(saddles[0][0].real, saddles[0][0].imag, s=200, c='lime', edgecolors='darkgreen',
                   linewidths=2.5, label='Dominant saddle', zorder=8)
        ax.scatter([s[0].real for s in saddles[1:]], [s[0].imag for s in saddles[1:]],
                   s=120, c='red', edgecolors='darkred', linewidths=2, label='Saddles', zorder=8)

    ax.set_xlabel(r'Re($\theta$)', fontsize=11)
    ax.set_ylabel(r'Im($\theta$)', fontsize=11)
    ax.set_title(r'Saddle-point landscape: Re[$\Phi(\theta)$]  ($\beta={:.3f}$, $\tilde{{\gamma}}={:.2f}\pi$)'.format(beta, gt/np.pi), fontsize=12)
    ax.legend(loc='upper right', fontsize=9, framealpha=0.95)
    ax.set_aspect('equal')
    ax.set_xlim(-half, half)
    ax.set_ylim(-half, half)
    plt.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        print('Saved', out_path)
    plt.close(fig)
    return fig


# ----- 5. SINGULARITIES (lattice) AND SADDLES (valleys) -----
def plot_singularities_and_saddles(beta=BETA, gt=2*np.pi, half=5, n_grid=100):
    """
    PDF: Singularities = branch points of log(arg), lattice θ_k = (log(-i cot(β/2))+2πik)/w.
    Saddles sit in valleys of Re[Φ], away from singularities.
    """
    fig, ax = plt.subplots(figsize=(8, 7))
    re_theta = np.linspace(-half, half, n_grid)
    im_theta = np.linspace(-half, half, n_grid)
    ReT, ImT = np.meshgrid(re_theta, im_theta)
    Z = np.zeros_like(ReT, dtype=float)
    for i in range(ReT.shape[0]):
        for j in range(ReT.shape[1]):
            Z[i, j] = Phi(complex(ReT[i, j], ImT[i, j]), beta, gt).real
    vmin, vmax = np.nanpercentile(Z, [2, 98])
    ax.contourf(ReT, ImT, Z, levels=40, cmap='viridis', vmin=vmin, vmax=vmax)
    saddles = find_saddles(beta, gt, half_width=half, grid=40)
    sing = find_singularities(beta, gt, half_width=half, k_max=15)
    if saddles:
        ax.scatter(saddles[0][0].real, saddles[0][0].imag, s=150, c='lime', edgecolors='darkgreen', linewidths=2,
                   label='Dominant saddle', zorder=5)
        ax.scatter([s[0].real for s in saddles[1:]], [s[0].imag for s in saddles[1:]], s=80, c='red',
                   edgecolors='darkred', linewidths=1.5, label='Subdominant saddles', zorder=5)
    if sing:
        ax.scatter([s.real for s in sing], [s.imag for s in sing], s=60, c='black', marker='x', linewidths=2,
                   label='Singularities (log arg=0)', zorder=6)
    ax.set_xlabel(r'Re $\theta$')
    ax.set_ylabel(r'Im $\theta$')
    ax.set_title(r'Singularities (lattice) and saddles (valleys): $\tilde\gamma=2\pi$')
    ax.legend(loc='upper right', fontsize=9)
    ax.set_aspect('equal')
    plt.tight_layout()
    return fig


# ----- Winding-number proof: good arcs vs bad arcs on C_R -----
def plot_winding_good_bad_arcs(gt=2*np.pi, R=8, eps=0.3, n_phi=400, out_path='winding_good_bad_arcs.png'):
    """
    Diagram for the argument-principle proof: circle C_R in the θ-plane with
    good arcs A_ε (where Re(e^{iφ}w) ≤ -ε ⇒ |e^{θw}| ≤ e^{-εR}) and bad arcs B_ε
    (complement). Good arcs: Φ' ≈ -θ/2 and contribute a definite winding; bad arcs
    are the directions where θ points so that θw has positive real part (e^{θw} large).
    """
    # Bug 1: w = sqrt(-i γ̃/2) principal branch. At γ̃=2π: -i*2π/2 = -iπ => w = sqrt(π)*e^{-iπ/4}
    w = np.sqrt(-1j * gt / 2 + 0j)
    abs_w = np.abs(w)
    alpha = np.angle(w)
    expected_abs_at_2pi = np.sqrt(np.pi)
    expected_angle = -np.pi / 4
    print('Bug1 check: w = sqrt(-i γ̃/2); at γ̃=2π expect w = sqrt(π)*e^{-iπ/4}')
    print('  w =', w, '  |w| =', abs_w, '  np.angle(w) =', alpha)
    print('  expected |w| = sqrt(π) ≈', expected_abs_at_2pi, '  expected arg(w) = -π/4 ≈', expected_angle)
    if np.isclose(gt, 2*np.pi):
        assert np.isclose(abs_w, expected_abs_at_2pi), '|w| should be sqrt(π) at γ̃=2π'
        assert np.isclose(alpha, expected_angle), 'arg(w) should be -π/4 at γ̃=2π'

    phi_all = np.linspace(0, 2*np.pi, n_phi, endpoint=False)
    # Bug 2: Good set = Re(e^{iφ}w) ≤ -ε. Use SIGNED real part only (never np.abs(re_ew)),
    # so we get exactly one contiguous good arc and one contiguous bad arc (not alternating bands).
    re_ew = np.real(np.exp(1j * phi_all) * w)
    is_good = (re_ew <= -eps)

    # Bug 3: Good arc has angular width π + O(ε). Center = where Re(e^{iφ}w) is minimal:
    #   Re(e^{iφ}w) = |w| cos(φ + α) => minimum at φ + α = π => φ_center = π - α (e.g. 5π/4 for α = -π/4).
    #   arg(-w) = α + π = 3π/4 when α = -π/4 (direction of -w); arc center in [0,2π] is π - α.
    good_where = np.where(is_good)[0]
    dphi = 2*np.pi / n_phi
    run_starts = np.array([], dtype=int)
    run_ends = np.array([], dtype=int)
    if len(good_where) > 0:
        # Contiguous runs: where diff(good_where) > 1 we have a boundary between runs
        d = np.diff(good_where)
        gap_after = np.where(d > 1)[0]
        run_ends_idx = np.concatenate([gap_after, [len(good_where) - 1]])
        run_starts_idx = np.concatenate([[0], gap_after + 1])
        run_starts = good_where[run_starts_idx]
        run_ends = good_where[run_ends_idx]
        lengths = (run_ends - run_starts + 1) * dphi
        good_arc_length = np.sum(lengths)
        expected_width = 2*np.pi - 2*np.arccos(np.clip(-eps/abs_w, -1, 1))
        print('Bug3 check: good arc total length =', good_arc_length, '(expect π+O(ε) ≈', np.pi, '; formula 2π−2arccos(−ε/|w|) ≈', expected_width, ')')
        print('  number of contiguous good arcs:', len(lengths), '(expect 1 for single good arc)')
        if len(lengths) >= 1:
            phi_center = (phi_all[run_starts[0]] + phi_all[run_ends[0]]) / 2
            center_formula = np.pi - alpha  # where Re(e^{iφ}w) is minimal
            arg_minus_w = alpha + np.pi     # 3π/4 when α = -π/4
            print('  first good arc center φ ≈', phi_center, '(formula π−α ≈', center_formula, '; arg(-w)=', arg_minus_w, ')')
    else:
        good_arc_length = 0.0
        print('Bug3: no good arc found (check ε and w)')

    # For arrows: sample φ from the main good arc
    if len(run_starts) > 0:
        phi_good_start = phi_all[run_starts[0]]
        phi_good_end = phi_all[run_ends[0]]
    else:
        phi_good_start = 0.0
        phi_good_end = np.pi

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Left: θ-plane, circle C_R with good (green) and bad (red) arcs
    ax = axes[0]
    th_circle = R * np.exp(1j * phi_all)
    x_c, y_c = th_circle.real, th_circle.imag
    for i in range(n_phi):
        j = (i + 1) % n_phi
        color = 'green' if is_good[i] else 'red'
        ax.plot([x_c[i], x_c[j]], [y_c[i], y_c[j]], color=color, linewidth=3, solid_capstyle='round')
    ax.plot(R * np.cos(phi_all), R * np.sin(phi_all), 'k-', linewidth=0.8, alpha=0.5)
    # Direction -θ/2: arrows on the good arc only
    n_arrows = 5
    sample_phi = np.linspace(phi_good_start, phi_good_end, n_arrows)
    for phi in sample_phi:
        theta = R * np.exp(1j * phi)
        neg_half = -theta / 2
        scale = R * 0.35 / max(abs(neg_half), 1e-6)
        ax.annotate('', xy=(theta.real + scale*neg_half.real, theta.imag + scale*neg_half.imag),
                    xytext=(theta.real, theta.imag),
                    arrowprops=dict(arrowstyle='->', color='darkgreen', lw=2))
    ax.set_xlabel(r'Re $\theta$')
    ax.set_ylabel(r'Im $\theta$')
    ax.set_title(r'Contour $C_R$: good arcs (green), bad arcs (red); $\Phi^{\prime} \approx -\theta/2$ on good arcs')
    ax.set_aspect('equal')
    ax.axhline(0, color='gray', ls=':', alpha=0.5)
    ax.axvline(0, color='gray', ls=':', alpha=0.5)
    ax.scatter([0], [0], s=80, c='black', zorder=5)
    ax.set_xlim(-R*1.25, R*1.25)
    ax.set_ylim(-R*1.25, R*1.25)
    ax.legend([plt.Line2D([0],[0], color='green', lw=4), plt.Line2D([0],[0], color='red', lw=4)],
              [r'Good $A_\varepsilon$: $\mathrm{Re}(e^{i\varphi}w)\leq -\varepsilon$',
               r'Bad $B_\varepsilon$: $\mathrm{Re}(e^{i\varphi}w) > -\varepsilon$'],
              loc='upper right', fontsize=9)

    # Right: φ-axis [0,2π] with A_ε and B_ε marked
    ax2 = axes[1]
    ax2.set_xlim(0, 2*np.pi)
    ax2.set_ylim(-0.1, 1.1)
    for i in range(n_phi):
        phi = phi_all[i]
        ax2.axvspan(phi, phi + 2*np.pi/n_phi, color='green' if is_good[i] else 'red', alpha=0.7)
    ax2.set_xlabel(r'$\varphi \in [0, 2\pi)$')
    ax2.set_ylabel('')
    ax2.set_yticks([])
    ax2.set_title(r'Angular decomposition: $A_\varepsilon$ (green), $B_\varepsilon$ (red); '
                 r'bad arcs = directions where $\theta w$ has positive real part')
    ax2.text(np.pi, 0.5, r'$|A_\varepsilon| = 2\pi - 2\arccos(-\varepsilon/|w|)$' + '\n'
             + r'$|B_\varepsilon| = 2\arccos(-\varepsilon/|w|)$', fontsize=10,
             ha='center', va='center', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    plt.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        print('Saved', out_path)
    return fig


def plot_num_saddles_vs_gamma(beta=BETA, gt_max=3*np.pi, n_pts=50, out_path='saddle-pt-#s.png'):
    """Number of saddle points vs γ̃ (fixed β). Saves as saddle-pt-#s.png."""
    fig, ax = plt.subplots(figsize=(7, 4))
    gammas = np.linspace(0.01, gt_max, n_pts)
    counts = [len(find_saddles(beta, gt, half_width=6, grid=35)) for gt in gammas]
    ax.plot(gammas / np.pi, counts, 'b-', linewidth=2, label='Number of saddles')
    ax.fill_between([0, 0.5], 0, max(counts) + 0.5, alpha=0.2, color='green', label='[BM24] unique regime')
    ax.set_xlabel(r'$\tilde{\gamma} / \pi$')
    ax.set_ylabel('Number of saddles')
    ax.set_title(r'Number of saddle points vs $\tilde{\gamma}$ ($\beta=-\pi/2$)')
    ax.set_ylim(0, max(counts) + 0.5)
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        print('Saved', out_path)
    plt.close(fig)
    return fig


# ----- Dominant saddle location vs (β, γ̃): where to expect θ* -----
def plot_dominant_saddle_location_map(beta_min=-1.5, beta_max=1.5, gt_min=0.2*np.pi, gt_max=2.5*np.pi,
                                      n_beta=24, n_gamma=24, half_width=6, grid=30, out_path='dominant_saddle_location.png'):
    """
    For a grid of (β, γ̃), find the dominant saddle (max Re Φ) and plot Re θ* and Im θ*
    as 2D heatmaps. Shows where we can expect the dominant saddle in the θ-plane as
    parameters vary.
    """
    betas = np.linspace(beta_min, beta_max, n_beta)
    gammas = np.linspace(gt_min, gt_max, n_gamma)
    Re_theta = np.full((n_gamma, n_beta), np.nan)
    Im_theta = np.full((n_gamma, n_beta), np.nan)
    Re_Phi_at = np.full((n_gamma, n_beta), np.nan)

    for i, gt in enumerate(gammas):
        for j, b in enumerate(betas):
            saddles = find_saddles(b, gt, half_width=half_width, grid=grid)
            if saddles:
                theta_star = saddles[0][0]
                Re_theta[i, j] = theta_star.real
                Im_theta[i, j] = theta_star.imag
                Re_Phi_at[i, j] = saddles[0][1].real

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    extent = [beta_min, beta_max, gt_min/np.pi, gt_max/np.pi]
    # Axis order for pcolormesh: first dim = rows = y = γ̃, second = cols = x = β
    ax = axes[0]
    valid = np.isfinite(Re_theta)
    vmin_re, vmax_re = np.nanmin(Re_theta), np.nanmax(Re_theta)
    pc0 = ax.pcolormesh(betas, gammas/np.pi, Re_theta, shading='auto', cmap='RdBu_r',
                        vmin=vmin_re, vmax=vmax_re)
    ax.set_xlabel(r'$\beta$')
    ax.set_ylabel(r'$\tilde{\gamma} / \pi$')
    ax.set_title(r'Re $\theta^*$ (dominant saddle)')
    plt.colorbar(pc0, ax=ax, label=r'Re $\theta^*$')

    ax = axes[1]
    vmin_im, vmax_im = np.nanmin(Im_theta), np.nanmax(Im_theta)
    pc1 = ax.pcolormesh(betas, gammas/np.pi, Im_theta, shading='auto', cmap='RdBu_r',
                        vmin=vmin_im, vmax=vmax_im)
    ax.set_xlabel(r'$\beta$')
    ax.set_ylabel(r'$\tilde{\gamma} / \pi$')
    ax.set_title(r'Im $\theta^*$ (dominant saddle)')
    plt.colorbar(pc1, ax=ax, label=r'Im $\theta^*$')

    ax = axes[2]
    pc2 = ax.pcolormesh(betas, gammas/np.pi, Re_Phi_at, shading='auto', cmap='viridis')
    ax.set_xlabel(r'$\beta$')
    ax.set_ylabel(r'$\tilde{\gamma} / \pi$')
    ax.set_title(r'Re $\Phi(\theta^*)$ (contribution strength)')
    plt.colorbar(pc2, ax=ax, label=r'Re $\Phi(\theta^*)$')

    fig.suptitle('Where to expect the dominant saddle: location vs $\\beta$ and $\\tilde{\\gamma}$', fontsize=12)
    plt.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        print('Saved', out_path)
    return fig


def main():
    out_dir = str(OUT_DIR) + '/'
    # Consistent names: large_gamma_*.png, saddle-pt-*.png, proof_*.png (ridge → term_competition)
    fig1 = plot_saddle_analysis()
    fig1.savefig(out_dir + 'large_gamma_saddles.png', dpi=150, bbox_inches='tight')
    print('Saved large_gamma_saddles.png')
    plt.close(fig1)

    fig3 = plot_phiprime_growth()
    fig3.savefig(out_dir + 'large_gamma_existence.png', dpi=150, bbox_inches='tight')
    print('Saved large_gamma_existence.png')
    plt.close(fig3)

    fig4 = plot_dominant_saddle_path()
    fig4.savefig(out_dir + 'large_gamma_path.png', dpi=150, bbox_inches='tight')
    print('Saved large_gamma_path.png')
    plt.close(fig4)

    fig5 = plot_singularities_and_saddles()
    fig5.savefig(out_dir + 'proof_singularities_saddles.png', dpi=150, bbox_inches='tight')
    print('Saved proof_singularities_saddles.png')
    plt.close(fig5)

    fig6 = plot_saddles_vs_beta(beta_min=-2, beta_max=2, n_beta=41)
    fig6.savefig(out_dir + 'beta-vary_saddles.png', dpi=150, bbox_inches='tight')
    print('Saved beta-vary_saddles.png')
    plt.close(fig6)

    plot_num_saddles_vs_gamma(out_path=out_dir + 'saddle-pt-#s.png')
    plot_saddle_landscape_nice(out_path=out_dir + 'saddle-pt-landscape.png')

    fig_dom = plot_dominant_saddle_location_map(out_path=out_dir + 'dominant_saddle_location.png')
    plt.close(fig_dom)

    # Verify singularities and saddles are disjoint (see singularities_vs_saddles_proof.md)
    ok, d = verify_singularities_disjoint_from_saddles()
    print('Singularities vs saddles: min distance = {:.6f} (disjoint: {})'.format(d, ok))

    plot_singularities_saddles_proof_diagram(out_path=out_dir + 'singularities_saddles_proof.png')

    plot_winding_good_bad_arcs(out_path=out_dir + 'winding_good_bad_arcs.png')

    print('Done. All proof diagrams saved.')


if __name__ == '__main__':
    main()
