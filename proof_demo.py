"""
Numerical proof figures for the large-γ̃ extension of Boulebnane–Montanaro [BM24].
Based on: final_large_gamma_proof.pdf (Quandela Device Theory Team, Feb 2026).

Generates the six-panel figure (Figure 1 of the proof):
  Top-left:    Re[Φ(θ)] saddle landscape at γ̃ = 2π
  Top-middle:  Number of saddles vs γ̃
  Top-right:   Convergence (1/n)log|A_n|² → saddle prediction
  Bottom-left: Saddle prediction vs extrapolated limit (key proof)
  Bottom-middle: n×residual bounded ⇒ O(1/n) error
  Bottom-right: Residuals vs 1/n (Gaussian correction)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from math import comb

# Import core formulas from test suite (or define here if run standalone)
try:
    from test_proof import exact_amplitude, Phi, dPhi, find_saddles
except ImportError:
    def exact_amplitude(n, beta, gt):
        """Eq. A14 of [BM24]"""
        gamma = gt / n
        c = np.cos(beta/2) / np.sqrt(2.0)
        s = -1j * np.sin(beta/2) / np.sqrt(2.0)
        return sum(comb(n, t) * c**(n-t) * s**t * np.exp(-1j*gamma*t**2/2)
                   for t in range(n+1))

    def Phi(theta, beta, gt):
        """Eq. A16 of [BM24]"""
        w = np.sqrt(-1j * gt / 2 + 0j)
        arg = np.cos(beta/2) - 1j*np.sin(beta/2)*np.exp(theta*w)
        return -np.log(2)/2 - 1j*beta/2 - theta**2/4 + np.log(arg + 1e-300)

    def dPhi(theta, beta, gt):
        w = np.sqrt(-1j * gt / 2 + 0j)
        ez = np.exp(theta * w)
        return (-theta/2 +
                (-1j*np.sin(beta/2)*w*ez) / (np.cos(beta/2) - 1j*np.sin(beta/2)*ez + 1e-300))

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


BETA = -np.pi/2  # optimal β (Corollary 1 of [BM24])


# ---------------------------------------------------------------------------
# Singularities of Φ: where the log argument vanishes (Eq. A16)
# Φ(θ) = ... + log( cos(β/2) - i sin(β/2) exp(θw) ). Singularities when:
#   cos(β/2) - i sin(β/2) exp(θw) = 0  =>  exp(θw) = cos(β/2)/(i sin(β/2)) = -i cot(β/2)
# So θw = log(-i cot(β/2)) + 2πik  =>  θ_k = (log(-i cot(β/2)) + 2πik) / w  (k ∈ Z)
# For β = -π/2: -i cot(β/2) = i, so θ_k = (iπ/2 + 2πik) / w — a lattice in θ-plane.
# ---------------------------------------------------------------------------
def find_singularities(beta, gt, half_width=8, k_max=20):
    """Return list of complex θ where the argument of log in Φ vanishes (branch points)."""
    w = np.sqrt(-1j * gt / 2 + 0j)
    # exp(θw) = -i cot(β/2); avoid cot at β=0
    cot_half = np.cos(beta/2) / (np.sin(beta/2) + 1e-300)
    z0 = -1j * cot_half  # complex target for exp(θw)
    r0 = np.abs(z0)
    if r0 < 1e-12:
        return []
    phi0 = np.angle(z0)  # in (-π, π]
    # θw = log(z0) + 2πik => θ_k = (log(r0) + i(phi0 + 2πk)) / w
    out = []
    for k in range(-k_max, k_max + 1):
        theta_w = np.log(r0) + 1j * (phi0 + 2*np.pi*k)
        th = theta_w / w
        if abs(th.real) <= half_width and abs(th.imag) <= half_width:
            out.append(th)
    return out


def landscape_re_phi(ax, beta=BETA, gt=2*np.pi, half=5, n_grid=80, plot_singularities=True):
    """Top-left: Re[Φ(θ)] landscape at γ̃ = 2π; mark dominant (green) and other saddles (red)."""
    re_theta = np.linspace(-half, half, n_grid)
    im_theta = np.linspace(-half, half, n_grid)
    ReT, ImT = np.meshgrid(re_theta, im_theta)
    Z = np.zeros_like(ReT, dtype=float)
    for i in range(ReT.shape[0]):
        for j in range(ReT.shape[1]):
            th = complex(ReT[i, j], ImT[i, j])
            Z[i, j] = Phi(th, beta, gt).real
    # Clip for cleaner contour
    vmin, vmax = np.nanpercentile(Z, [2, 98])
    cf = ax.contourf(ReT, ImT, Z, levels=40, cmap='viridis', vmin=vmin, vmax=vmax)
    ax.set_xlabel(r'Re($\theta$)')
    ax.set_ylabel(r'Im($\theta$)')
    ax.set_title(r'Saddle landscape Re[$\Phi(\theta)$] at $\tilde\gamma = 2\pi$')

    saddles = find_saddles(beta, gt, half_width=half, grid=40)
    if saddles:
        dom = saddles[0][0]
        ax.scatter(dom.real, dom.imag, s=120, c='lime', edgecolors='darkgreen', linewidths=2,
                  label='Dominant saddle (max Re[Φ])', zorder=5)
        for th, _ in saddles[1:]:
            ax.scatter(th.real, th.imag, s=80, c='red', edgecolors='darkred', linewidths=1.5,
                      label='Other saddles' if th == saddles[1][0] else None, zorder=5)
    if plot_singularities:
        sing = find_singularities(beta, gt, half_width=half, k_max=15)
        if sing:
            ax.scatter([s.real for s in sing], [s.imag for s in sing], s=60, c='black', marker='x',
                      linewidths=1.5, label='Singularities (log arg=0)', zorder=6)
    ax.legend(loc='upper right', fontsize=7)
    ax.set_aspect('equal')
    ax.set_rasterization_zorder(0)
    return cf  # for colorbar in main()


def num_saddles_vs_gamma(ax, beta=BETA, gt_max=3*np.pi, n_pts=50):
    """Top-middle: Number of saddle points vs γ̃. Optionally shade small-γ̃ uniqueness region."""
    gammas = np.linspace(0.01, gt_max, n_pts)
    counts = []
    for gt in gammas:
        counts.append(len(find_saddles(beta, gt, half_width=6, grid=35)))
    ax.plot(gammas / np.pi, counts, 'b-', linewidth=2, label='Number of saddles')
    ax.fill_between([0, 0.5], 0, max(counts) + 0.5, alpha=0.2, color='green', label='[BM24] unique regime')
    ax.set_xlabel(r'$\tilde\gamma / \pi$')
    ax.set_ylabel('Number of saddles')
    ax.set_title('Number of saddles vs $\\tilde\\gamma$')
    ax.set_ylim(0, max(counts) + 0.5)
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)


def convergence_exact_to_saddle(ax, beta=BETA, gammas_plot=None, n_vals=None):
    """Top-right: (1/n)log|A_n|² vs n for several γ̃; dashed = saddle prediction."""
    if gammas_plot is None:
        gammas_plot = [0.5*np.pi, np.pi, 2*np.pi]
    if n_vals is None:
        n_vals = np.arange(30, 151, 10)

    for gt in gammas_plot:
        pred = 2 * find_saddles(beta, gt)[0][1].real
        exact_vals = [np.log(abs(exact_amplitude(n, beta, gt))**2) / n for n in n_vals]
        label = r'$\tilde\gamma = ' + f'{gt/np.pi:.1f}' + r'\pi$'
        ax.plot(n_vals, exact_vals, 'o-', label=label, markersize=4)
        ax.axhline(pred, color=ax.get_lines()[-1].get_color(), ls='--', alpha=0.8)
    ax.set_xlabel('$n$')
    ax.set_ylabel(r'$(1/n)\log|A_n|^2$')
    ax.set_title('Convergence: exact $(n)$ → saddle prediction (dashed = pred.)')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)


def saddle_vs_extrapolated_limit(ax, beta=BETA, gammas=None, n_vals=None):
    """Bottom-left: Key proof — saddle prediction P vs extrapolated limit L (should coincide)."""
    if gammas is None:
        gammas = np.linspace(0.2*np.pi, 3*np.pi, 20)
    if n_vals is None:
        n_vals = [40, 60, 80, 100, 120, 150]

    predictions = []
    extrapolated = []
    for gt in gammas:
        saddles = find_saddles(beta, gt)
        P = 2 * saddles[0][1].real
        predictions.append(P)
        vals = [np.log(abs(exact_amplitude(n, beta, gt))**2) / n for n in n_vals]
        inv_n = np.array([1/n for n in n_vals])
        c0 = np.polyfit(inv_n, vals, 1)[1]
        extrapolated.append(c0)
    ax.plot(gammas / np.pi, predictions, 'g-', linewidth=2, label='Saddle prediction $P$')
    ax.plot(gammas / np.pi, extrapolated, 'r--', linewidth=1.5, label='Extrapolated limit $L$')
    ax.set_xlabel(r'$\tilde\gamma / \pi$')
    ax.set_ylabel(r'$(1/n)\log|A_n|^2$ limit')
    ax.set_title('Key proof: $P$ and $L$ coincide')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)


def n_times_residual(ax, beta=BETA, gt=2*np.pi, n_vals=None):
    """Bottom-middle: n × (exact - prediction) vs n ⇒ bounded O(1/n) error."""
    if n_vals is None:
        n_vals = np.array([50, 80, 100, 120, 150, 200])
    pred = 2 * find_saddles(beta, gt)[0][1].real
    exact_vals = np.array([np.log(abs(exact_amplitude(n, beta, gt))**2) / n for n in n_vals])
    residuals = exact_vals - pred
    n_times_r = n_vals * residuals
    ax.plot(n_vals, n_times_r, 'bo-', markersize=8)
    ax.axhline(0, color='gray', ls='--', alpha=0.7)
    ax.set_xlabel('$n$')
    ax.set_ylabel(r'$n \cdot [(1/n)\log|A_n|^2 - P]$')
    ax.set_title(r'$n\times$residual bounded $\Rightarrow$ O(1/n) error')
    ax.grid(True, alpha=0.3)


def residuals_vs_inv_n(ax, beta=BETA, gt=2*np.pi, n_vals=None):
    """Bottom-right: Residual (exact - prediction) vs 1/n (linear ⇒ Gaussian correction)."""
    if n_vals is None:
        n_vals = np.array([40, 60, 80, 100, 120, 150, 200])
    pred = 2 * find_saddles(beta, gt)[0][1].real
    exact_vals = np.array([np.log(abs(exact_amplitude(n, beta, gt))**2) / n for n in n_vals])
    residuals = exact_vals - pred
    inv_n = 1 / n_vals
    ax.plot(inv_n, residuals, 'bo-', markersize=8, label='Residual')
    # Linear fit: residual ≈ c1/n
    coefs = np.polyfit(inv_n, residuals, 1)
    fit = np.poly1d(coefs)(inv_n)
    ax.plot(inv_n, fit, 'r--', label=r'Fit $c_1/n$')
    ax.set_xlabel('$1/n$')
    ax.set_ylabel(r'$(1/n)\log|A_n|^2 - P$')
    ax.set_title('Residuals match Gaussian correction (Prop. 6 [BM24])')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)


def log_arg(theta, beta, gt):
    """Argument of the log in Φ(θ): cos(β/2) - i sin(β/2) exp(θw). Zero at singularities."""
    w = np.sqrt(-1j * gt / 2 + 0j)
    return np.cos(beta/2) - 1j*np.sin(beta/2)*np.exp(theta*w)


def extended_singularity_saddle_analysis(beta=BETA, gt=2*np.pi, half=5, n_grid=100):
    """
    Extended analysis: relationship between singularities and saddle points.

    Relationship (Picard–Lefschetz / steepest descent):
    - Singularities: Φ has log(arg) with arg = cos(β/2) - i sin(β/2) exp(θw). Where arg=0,
      log blows up — these are branch points and form a lattice θ_k = (log(-i cot(β/2)) + 2πik)/w.
    - Saddles: critical points with dΦ/dθ=0; they are the endpoints of steepest descent contours.
    - Saddles typically sit in "valleys" of Re[Φ], away from singularities (so the integrand
      enΦ is analytic along the thimble). The number of saddles grows with γ̃; the singularities
      (in a fixed window) also increase with γ̃. The dominant saddle is the one with largest
      Re[Φ], governing the asymptotic (1/n)log|A_n|².
    """
    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    w = np.sqrt(-1j * gt / 2 + 0j)
    re_theta = np.linspace(-half, half, n_grid)
    im_theta = np.linspace(-half, half, n_grid)
    ReT, ImT = np.meshgrid(re_theta, im_theta)

    # Panel A: Re[Φ] with saddles and singularities (same as top-left but with annotations)
    Z = np.zeros_like(ReT, dtype=float)
    for i in range(ReT.shape[0]):
        for j in range(ReT.shape[1]):
            th = complex(ReT[i, j], ImT[i, j])
            Z[i, j] = Phi(th, beta, gt).real
    vmin, vmax = np.nanpercentile(Z, [2, 98])
    cf_a = axes[0, 0].contourf(ReT, ImT, Z, levels=40, cmap='viridis', vmin=vmin, vmax=vmax)
    saddles = find_saddles(beta, gt, half_width=half, grid=40)
    sing = find_singularities(beta, gt, half_width=half, k_max=15)
    if saddles:
        axes[0, 0].scatter(saddles[0][0].real, saddles[0][0].imag, s=120, c='lime', edgecolors='darkgreen',
                          linewidths=2, label='Dominant saddle', zorder=5)
        axes[0, 0].scatter([s[0].real for s in saddles[1:]], [s[0].imag for s in saddles[1:]],
                          s=80, c='red', edgecolors='darkred', linewidths=1.5, label='Other saddles', zorder=5)
    if sing:
        axes[0, 0].scatter([s.real for s in sing], [s.imag for s in sing], s=70, c='black', marker='x',
                          linewidths=2, label='Singularities (log arg=0)', zorder=6)
    axes[0, 0].set_xlabel(r'Re($\theta$)')
    axes[0, 0].set_ylabel(r'Im($\theta$)')
    axes[0, 0].set_title(r'Re[$\Phi(\theta)$]: saddles vs singularities ($\tilde\gamma=2\pi$)')
    axes[0, 0].legend(loc='upper right', fontsize=8)
    axes[0, 0].set_aspect('equal')
    cbar_a = fig.colorbar(cf_a, ax=axes[0, 0], shrink=0.7)
    cbar_a.set_label(r'Re[$\Phi(\theta)$]', fontsize=9)

    # Panel B: |log arg| — distance to singularity (in a sense); saddles sit where |arg| is not tiny
    Z_arg = np.zeros_like(ReT, dtype=float)
    for i in range(ReT.shape[0]):
        for j in range(ReT.shape[1]):
            th = complex(ReT[i, j], ImT[i, j])
            a = log_arg(th, beta, gt)
            Z_arg[i, j] = np.log10(np.abs(a) + 1e-20)
    axb = axes[0, 1]
    cf_b = axb.contourf(ReT, ImT, Z_arg, levels=30, cmap='plasma')
    if saddles:
        axb.scatter(saddles[0][0].real, saddles[0][0].imag, s=100, c='lime', edgecolors='darkgreen', linewidths=2,
                   label='Dominant saddle', zorder=5)
        axb.scatter([s[0].real for s in saddles[1:]], [s[0].imag for s in saddles[1:]],
                    s=60, c='red', edgecolors='darkred', linewidths=1.5, label='Other saddles', zorder=5)
    if sing:
        axb.scatter([s.real for s in sing], [s.imag for s in sing], s=50, c='black', marker='x', linewidths=1.5,
                   label='Singularities', zorder=6)
    axb.set_xlabel(r'Re($\theta$)')
    axb.set_ylabel(r'Im($\theta$)')
    axb.set_title(r'$\log_{10}|\mathrm{arg}(\log)|$: saddles avoid singularities')
    axb.legend(loc='upper right', fontsize=8)
    axb.set_aspect('equal')
    cbar_b = fig.colorbar(cf_b, ax=axes[0, 1], shrink=0.7)
    cbar_b.set_label(r'$\log_{10}|\mathrm{arg}|$', fontsize=9)

    # Panel C: Distance from each saddle to nearest singularity (saddle index vs distance)
    if saddles and sing:
        sing_arr = np.array(sing)
        distances = []
        for th, _ in saddles:
            d = np.min(np.abs(sing_arr - th))
            distances.append(d)
        bars = axes[1, 0].bar(range(len(saddles)), distances, color=['green'] + ['red']*(len(saddles)-1), edgecolor='black')
        axes[1, 0].set_xlabel('Saddle index (0 = dominant)')
        axes[1, 0].set_ylabel('Distance to nearest singularity')
        axes[1, 0].set_title('Saddle–singularity separation')
        mean_line = axes[1, 0].axhline(np.mean(distances), color='blue', ls='--', alpha=0.7)
        # Key for bar colors
        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D
        leg_handles = [Patch(facecolor='green', edgecolor='black'), Patch(facecolor='red', edgecolor='black'),
                       Line2D([0], [0], color='blue', ls='--', linewidth=2)]
        axes[1, 0].legend(leg_handles, ['Dominant saddle', 'Other saddles', 'Mean distance'], loc='upper right', fontsize=8)
    else:
        axes[1, 0].text(0.5, 0.5, 'No saddles/singularities in window', ha='center', va='center', transform=axes[1, 0].transAxes)

    # Panel D: As γ̃ increases, number of saddles vs number of singularities in window
    gammas = np.linspace(0.2*np.pi, 3*np.pi, 25)
    n_saddles = [len(find_saddles(beta, g, half_width=6, grid=35)) for g in gammas]
    n_sing = [len(find_singularities(beta, g, half_width=6, k_max=10)) for g in gammas]
    axes[1, 1].plot(gammas/np.pi, n_saddles, 'b-o', label='# saddles', markersize=5)
    axes[1, 1].plot(gammas/np.pi, n_sing, 'k--s', label='# singularities in window', markersize=4)
    axes[1, 1].set_xlabel(r'$\tilde\gamma / \pi$')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_title(r'Saddles vs singularities vs $\tilde\gamma$')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle('Extended analysis: singularities (log arg=0) and saddle points', fontsize=11)
    plt.tight_layout()
    return fig


def main():
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    fig.suptitle('Large-$\\tilde\\gamma$ proof: numerical verification (Boulebnane–Montanaro extension)',
                 fontsize=12)

    cf = landscape_re_phi(axes[0, 0], plot_singularities=True)
    cbar = fig.colorbar(cf, ax=axes[0, 0], shrink=0.7)
    cbar.set_label(r'Re[$\Phi(\theta)$] (color scale)', fontsize=9)
    num_saddles_vs_gamma(axes[0, 1])
    convergence_exact_to_saddle(axes[0, 2])
    saddle_vs_extrapolated_limit(axes[1, 0])
    n_times_residual(axes[1, 1])
    residuals_vs_inv_n(axes[1, 2])

    plt.tight_layout()
    out = 'proof_figure.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f'Saved {out}')
    plt.close()

    # Extended analysis: singularities vs saddles
    fig2 = extended_singularity_saddle_analysis()
    out2 = 'proof_singularities_saddles.png'
    fig2.savefig(out2, dpi=150, bbox_inches='tight')
    print(f'Saved {out2}')
    plt.close(fig2)


if __name__ == '__main__':
    main()
