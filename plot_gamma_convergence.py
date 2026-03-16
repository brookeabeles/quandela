"""
Generate gamma-convergence.png: (1/n)log|A_n|² → saddle prediction for several γ̃.

β = −π/2 is the optimal value for the QAOA toy model (Boulebnane–Montanaro [BM24],
Corollary 1), so it is the standard choice for these plots.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Professional LaTeX-style rendering
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13

try:
    from test_proof import exact_amplitude, find_saddles
except ImportError:
    from math import comb
    def exact_amplitude(n, beta, gt):
        gamma = gt / n
        c = np.cos(beta/2) / np.sqrt(2.0)
        s = -1j * np.sin(beta/2) / np.sqrt(2.0)
        return sum(comb(n, t) * c**(n-t) * s**t * np.exp(-1j*gamma*t**2/2)
                   for t in range(n+1))
    def Phi(th, beta, gt):
        w = np.sqrt(-1j * gt / 2 + 0j)
        arg = np.cos(beta/2) - 1j*np.sin(beta/2)*np.exp(th*w)
        return -np.log(2)/2 - 1j*beta/2 - th**2/4 + np.log(arg + 1e-300)
    def dPhi(th, beta, gt):
        w = np.sqrt(-1j * gt / 2 + 0j)
        ez = np.exp(th * w)
        return (-th/2 + (-1j*np.sin(beta/2)*w*ez) / (np.cos(beta/2) - 1j*np.sin(beta/2)*ez + 1e-300))
    def find_saddles(beta, gt, half_width=6, grid=50):
        found = []
        h = 1e-6
        for re0 in np.linspace(-half_width, half_width, grid):
            for im0 in np.linspace(-half_width, half_width, grid):
                th = complex(re0, im0)
                for _ in range(300):
                    f = dPhi(th, beta, gt)
                    f2 = (dPhi(th+h, beta, gt) - dPhi(th-h, beta, gt))/(2*h)
                    if abs(f2) < 1e-13:
                        break
                    th -= f/f2
                    if abs(f) < 1e-8:
                        break
                if abs(dPhi(th, beta, gt)) < 1e-6 and all(abs(th - s[0]) > 1e-3 for s in found):
                    found.append((th, Phi(th, beta, gt)))
        found.sort(key=lambda x: -x[1].real)
        return found


# [BM24] Corollary 1: optimal β for QAOA toy model
BETA = -np.pi / 2


def plot_convergence(
    beta=BETA,
    gammas_pi=(0.5, 1.0, 1.5, 2.0, 2.5),
    n_vals=None,
    out_path='gamma-convergence.png',
):
    if n_vals is None:
        n_vals = np.arange(50, 201, 5)

    fig, ax = plt.subplots(figsize=(9, 6))

    for gpi in gammas_pi:
        gt = float(gpi * np.pi)
        saddles = find_saddles(beta, gt)
        P = 2 * saddles[0][1].real
        ys = [np.log(abs(exact_amplitude(n, beta, gt))**2) / n for n in n_vals]
        label = r'$\tilde{\gamma} = ' + f'{gpi:.1f}' + r'\pi$'
        ax.plot(n_vals, ys, 'o-', label=label, markersize=4, markeredgewidth=0.5)
        ax.axhline(P, color=ax.get_lines()[-1].get_color(), ls='--', alpha=0.85, linewidth=1.2)

    ax.set_xlabel(r'$n$', fontsize=12)
    ax.set_ylabel(r'$\frac{1}{n}\log|A_n|^2$', fontsize=13)
    ax.set_title(
        r'Convergence: $\frac{1}{n}\log|A_n|^2 \to 2\mathrm{Re}\,\Phi(\theta^*)$'
        r' $(\beta = -\pi/2,\ \tilde{\gamma} \in \{0.5\pi, \ldots, 2\pi\})$',
        fontsize=12,
    )
    from matplotlib.lines import Line2D
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([0], [0], color='gray', ls='--', linewidth=1.5, label='saddle prediction'))
    labels.append('saddle prediction')
    ax.legend(
        handles, labels,
        loc='upper right',
        frameon=True,
        framealpha=0.95,
        fontsize=10,
        edgecolor='0.8',
        fancybox=True,
    )
    ax.grid(True, alpha=0.35)
    ax.set_xlim(n_vals.min(), n_vals.max())
    ax.autoscale(axis='y')  # scale y to remaining series (no 3π)
    fig.tight_layout()

    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f'Saved {out_path}')
    plt.close(fig)
    return fig


if __name__ == '__main__':
    plot_convergence(out_path='gamma-convergence.png')
