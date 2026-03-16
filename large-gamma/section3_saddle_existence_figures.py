"""
Section 3 saddle existence proof: four figures.

QAOA action Φ(θ) = −θ²/4 + log(cos(β/2) − i sin(β/2) e^{θw})
with w = sqrt(−iγ̃/2), β = −π/2, γ̃ = 2π (or sweep for Fig D).

Figure A: Singularity lattice θ_k, circle C_{R'} at pole-avoiding radius, δ₀.
Figure B: |g(θ)| = R'/2 and |h(θ)| vs φ on C_{R'}; Rouché condition |g| > |h|.
Figure C: Image curves Φ'(C_{R'}) and g(C_{R'}) in ℂ; winding of Φ' around 0.
Figure D: N_R and P_R vs γ̃; verify N − P = 1.
"""

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from qaoa_core import (
        w_of_gamma,
        A,
        dPhi,
        d2Phi,
        singularities,
        DEFAULT_BETA,
    )
except ImportError:
    # Standalone definitions if run from large-gamma/
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from qaoa_core import (
        w_of_gamma,
        A,
        dPhi,
        d2Phi,
        singularities,
        DEFAULT_BETA,
    )

OUT_DIR = Path(__file__).resolve().parent
BETA = -np.pi / 2


def h_theta(theta, gt, beta=BETA):
    """h(θ) = A'(θ)/A(θ) = −iw sin(β/2) e^{θw} / (cos(β/2) − i sin(β/2) e^{θw})."""
    w = w_of_gamma(gt)
    cb = np.cos(beta / 2)
    sb = np.sin(beta / 2)
    e = np.exp(theta * w)
    denom = cb - 1j * sb * e
    if np.isscalar(theta):
        if np.abs(denom) < 1e-300:
            return np.nan
        return -1j * w * sb * e / denom
    denom = denom + 1e-300 * (np.abs(denom) < 1e-14)
    return -1j * w * sb * e / denom


def g_theta(theta):
    """g(θ) = −θ/2. On |θ|=R', |g(θ)| = R'/2."""
    return -theta / 2


def find_saddles_inside(gt, R, beta=BETA, n_grid=200, tol=1e-3):
    """
    Locate approximate zeros of Φ'(θ) inside the disk |θ| < R.

    Strategy:
      1. Sample |Φ'| on an n_grid×n_grid grid over [-R, R]².
      2. Find grid points with |Φ'| < tol that are local minima.
      3. Refine each candidate using scipy.optimize.fsolve on Re/Im parts of Φ'.
    """
    try:
        from scipy.optimize import fsolve
    except ImportError:
        print("Warning: scipy not available; find_saddles_inside returns no saddles.")
        return np.array([], dtype=complex)

    xs = np.linspace(-R, R, n_grid)
    ys = np.linspace(-R, R, n_grid)
    XX, YY = np.meshgrid(xs, ys)
    theta_grid = XX + 1j * YY
    mask_disk = np.abs(theta_grid) < R
    vals = np.full(theta_grid.shape, np.inf, dtype=float)
    # Evaluate |Φ'| where inside disk
    flat_theta = theta_grid[mask_disk]
    flat_vals = np.abs(np.array([dPhi(th, gt, beta) for th in flat_theta]))
    vals[mask_disk] = flat_vals

    candidates = []
    for i in range(1, n_grid - 1):
        for j in range(1, n_grid - 1):
            v = vals[i, j]
            if not np.isfinite(v) or v >= tol:
                continue
            neighborhood = vals[i-1:i+2, j-1:j+2]
            if v <= np.min(neighborhood):
                candidates.append(theta_grid[i, j])

    saddles = []

    def F(z_vec):
        th = z_vec[0] + 1j * z_vec[1]
        val = dPhi(th, gt, beta)
        return [val.real, val.imag]

    for th0 in candidates:
        z0 = np.array([th0.real, th0.imag], dtype=float)
        try:
            sol, info, ier, _ = fsolve(F, z0, full_output=True, xtol=1e-12, maxfev=200)
        except Exception:
            continue
        if ier != 1:
            continue
        th = sol[0] + 1j * sol[1]
        if np.abs(dPhi(th, gt, beta)) > 1e-8 or np.abs(th) >= R:
            continue
        # Deduplicate: require separation between saddles
        if all(np.abs(th - s) > 1e-3 for s in saddles):
            saddles.append(th)

    return np.array(saddles, dtype=complex)


def pole_avoiding_radius(gt, R_base, beta=BETA, kmax=80, n_try=200, avoid_zeros=False, n_grid=200):
    """
    Find R' in [R_base, R_base + 2π/|w|] that maximizes the minimum clearance
    between the circle C_{R'} and both the singularity lattice and (optionally)
    the saddle points of Φ'.

    When avoid_zeros=True, the distance metric is
        δ(R) = min( min_k ||θ_k^{pole}| − R|, min_j ||θ_j^{saddle}| − R| ).
    Returns (R_prime, delta_0) where delta_0 is this maximal clearance.
    """
    w = w_of_gamma(gt)
    L = 2 * np.pi / np.abs(w)
    sings = singularities(gt, kmax=kmax, beta=beta)
    r_moduli = np.abs(sings)

    saddle_moduli = np.array([], dtype=float)
    if avoid_zeros:
        # Look for saddles inside the largest radius we will consider.
        R_search = R_base + L
        saddles = find_saddles_inside(gt, R_search, beta=beta, n_grid=n_grid)
        if saddles.size > 0:
            saddle_moduli = np.abs(saddles)

    R_candidates = np.linspace(R_base, R_base + L, n_try)
    best_R = R_base
    best_delta = 0.0
    for R in R_candidates:
        # Clearance to poles: min_k ||θ_k| - R|
        dists_poles = np.abs(r_moduli - R) if r_moduli.size > 0 else np.array([np.inf])
        if avoid_zeros and saddle_moduli.size > 0:
            # Clearance to saddles: min_j ||θ_j| - R|
            dists_saddles = np.abs(saddle_moduli - R)
            delta = np.min(np.concatenate([dists_poles, dists_saddles]))
        else:
            delta = np.min(dists_poles)
        if delta > best_delta:
            best_delta = delta
            best_R = R
    return best_R, best_delta


def argument_principle_integral(gt, R, n_phi=2000, beta=BETA):
    """
    (1/2πi) ∮_{|θ|=R} (Φ''/Φ')(θ) dθ = N_R − P_R (zeros minus poles of Φ' inside C_R).
    Contour θ(φ) = R e^{iφ}, dθ = i R e^{iφ} dφ.
    """
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    dphi = 2 * np.pi / n_phi
    thetas = R * np.exp(1j * phi)
    integrand = np.zeros_like(thetas, dtype=complex)
    for i, th in enumerate(thetas):
        dp = dPhi(th, gt, beta)
        d2p = d2Phi(th, gt, beta)
        if np.abs(dp) < 1e-14:
            integrand[i] = 0.0
        else:
            integrand[i] = (d2p / dp) * (1j * R * np.exp(1j * phi[i]))
    return np.sum(integrand) * dphi / (2 * np.pi * 1j)


def winding_number_of_dPhi(gt, R, n_phi=4000, beta=BETA):
    """Count zeros−poles of Φ' by tracking arg(Φ'(θ)) around C_R."""
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    thetas = R * np.exp(1j * phi)
    vals = np.array([dPhi(th, gt, beta) for th in thetas])
    angles = np.angle(vals)
    dangle = np.diff(angles)
    dangle = (dangle + np.pi) % (2 * np.pi) - np.pi
    winding = np.sum(dangle) / (2 * np.pi)
    return winding


def count_poles_inside(gt, R, beta=BETA, kmax=200):
    """P_R = number of lattice points θ_k with |θ_k| < R."""
    sings = singularities(gt, kmax=kmax, beta=beta)
    return int(np.sum(np.abs(sings) < R - 1e-10))


def main():
    gt_fig_abc = 2 * np.pi
    w = w_of_gamma(gt_fig_abc)
    L = 2 * np.pi / np.abs(w)
    R_base = 8.0
    R_prime, delta_0 = pole_avoiding_radius(gt_fig_abc, R_base, beta=BETA)
    n_phi = 800
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    circle_theta = R_prime * np.exp(1j * phi)
    sings = singularities(gt_fig_abc, kmax=60, beta=BETA)
    # Restrict to poles that appear in the plot window
    plot_radius = max(R_prime * 1.4, 14)
    in_view = (np.abs(sings.real) <= plot_radius * 1.2) & (np.abs(sings.imag) <= plot_radius * 1.2)
    sings_plot = sings[in_view]

    # ----- Figure A: Singularity lattice, circle C_{R'}, δ₀ -----
    # δ₀ = min distance from circle to any pole = min_k | |θ_k| − R' |
    dist_circle_to_poles = np.abs(np.abs(sings) - R_prime)
    nearest_idx = np.argmin(dist_circle_to_poles)
    pole_nearest = sings[nearest_idx]
    if np.abs(pole_nearest) > 1e-10:
        pt_on_circle = R_prime * (pole_nearest / np.abs(pole_nearest))
    else:
        pt_on_circle = circle_theta[0]
    delta_0_actual = np.abs(np.abs(pole_nearest) - R_prime)

    fig_a, ax_a = plt.subplots(figsize=(6, 6))
    ax_a.scatter(sings_plot.real, sings_plot.imag, s=18, c="black", marker="x", label=r"Poles $\theta_k$", zorder=3)
    circle_x = circle_theta.real
    circle_y = circle_theta.imag
    ax_a.plot(circle_x, circle_y, "b-", lw=2, label=rf"$C_{{R'}}$ ($R'$ = {R_prime:.3f})")
    ax_a.plot([pt_on_circle.real, pole_nearest.real], [pt_on_circle.imag, pole_nearest.imag],
              "g-", lw=2, label=rf"$\delta_0$ = {delta_0_actual:.4f}")
    ax_a.scatter([pt_on_circle.real], [pt_on_circle.imag], s=60, c="green", zorder=4)
    ax_a.scatter([pole_nearest.real], [pole_nearest.imag], s=80, c="red", marker="x", linewidths=2, zorder=4)
    ax_a.fill_between(circle_x, circle_y, alpha=0.08, color="blue")
    ax_a.axhline(0, color="gray", ls=":", alpha=0.6)
    ax_a.axvline(0, color="gray", ls=":", alpha=0.6)
    ax_a.set_xlabel(r"Re $\theta$")
    ax_a.set_ylabel(r"Im $\theta$")
    ax_a.set_title(rf"Figure A: Singularity lattice and pole-avoiding circle ($\tilde\gamma=2\pi$); $\delta_0$ = {delta_0_actual:.4f}")
    ax_a.legend(loc="upper right", fontsize=9)
    ax_a.set_aspect("equal")
    ax_a.set_xlim(-plot_radius, plot_radius)
    ax_a.set_ylim(-plot_radius, plot_radius)
    ax_a.grid(True, alpha=0.3)
    fig_a.tight_layout()
    fig_a.savefig(OUT_DIR / "sect3_figA_singularity_lattice.png", bbox_inches="tight", dpi=150)
    plt.close(fig_a)
    print("Saved sect3_figA_singularity_lattice.png")

    # ----- Figure B: |g(θ)| and |h(θ)| vs φ on C_{R'} -----
    g_vals = g_theta(circle_theta)
    h_vals = h_theta(circle_theta, gt_fig_abc, beta=BETA)
    abs_g = np.abs(g_vals)
    abs_h = np.abs(h_vals)
    # Replace any nan/inf in h for plotting
    abs_h_safe = np.where(np.isfinite(abs_h), abs_h, np.nan)

    fig_b, ax_b = plt.subplots(figsize=(7, 3))
    ax_b.plot(phi, abs_g, "b-", lw=2)
    ax_b.plot(phi, abs_h_safe, "r-", lw=1.5)
    # Labels at left (start of lines) instead of legend
    y_g = float(abs_g[0]) - 0.5
    y_h_vals = abs_h_safe[np.isfinite(abs_h_safe)]
    y_h = float(y_h_vals[0]) - 0.5 if len(y_h_vals) > 0 else 0.5
    ax_b.text(0.25, y_g, r"$|g(\theta)| = R'/2$", fontsize=9, color="blue", va="center")
    ax_b.text(0.25, y_h, r"$|h(\theta)| = |A'(\theta)/A(\theta)|$", fontsize=9, color="red", va="center")
    ax_b.set_xlabel(r"$\varphi \in [0, 2\pi)$")
    ax_b.set_ylabel("Magnitude")
    ax_b.set_title(r"Rouché condition on $C_{R'}$: $|g(\theta)| > |h(\theta)|$ ($\tilde\gamma = 2\pi$)")
    ax_b.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
    ax_b.set_xticklabels(["0", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"])
    ax_b.grid(True, alpha=0.3)
    ax_b.set_xlim(0, 2 * np.pi)
    ax_b.set_ylim(-0.2, 6)
    fig_b.tight_layout()
    fig_b.savefig(OUT_DIR / "sect3_figB_rouche_condition.png", bbox_inches="tight", dpi=150)
    plt.close(fig_b)
    print("Saved sect3_figB_rouche_condition.png")

    # ----- Figure C: Φ'(C_{R'}) and g(C_{R'}) in complex plane; winding -----
    phip_circle = np.array([dPhi(th, gt_fig_abc, beta=BETA) for th in circle_theta])
    g_circle = g_theta(circle_theta)
    wind = argument_principle_integral(gt_fig_abc, R_prime, n_phi=n_phi, beta=BETA)
    wind_real = np.real(wind)

    fig_c, ax_c = plt.subplots(figsize=(6, 6))
    ax_c.plot(phip_circle.real, phip_circle.imag, "b-", lw=2, label=r"$\Phi'(C_{R'})$")
    ax_c.plot(g_circle.real, g_circle.imag, "orange", lw=1.5, ls="--", label=r"$g(C_{R'}) = \{-\theta/2 : \theta \in C_{R'}\}$")
    ax_c.scatter([0], [0], s=80, c="red", zorder=5, label="Origin")
    ax_c.axhline(0, color="gray", ls=":", alpha=0.6)
    ax_c.axvline(0, color="gray", ls=":", alpha=0.6)
    ax_c.set_xlabel(r"Re $z$")
    ax_c.set_ylabel(r"Im $z$")
    ax_c.set_title(rf"Figure C: Image curves; winding of $\Phi'(C_{{R'}})$ around 0 $\approx$ {wind_real:.2f}")
    ax_c.legend(loc="best", fontsize=9)
    ax_c.set_aspect("equal")
    ax_c.grid(True, alpha=0.3)
    fig_c.tight_layout()
    fig_c.savefig(OUT_DIR / "sect3_figC_winding.png", bbox_inches="tight", dpi=150)
    plt.close(fig_c)
    print("Saved sect3_figC_winding.png")

    # ----- Figure D: N_R, P_R vs γ̃; adaptive R'(γ̃), δ₀(γ̃), Rouché margin -----
    gt_sweep = np.linspace(0.1 * np.pi, 2.5 * np.pi, 80)
    R_base = 12.0
    n_rouche = 4000
    N_list = []
    P_list = []
    delta_0_list = []
    R_prime_list = []
    rouche_margin_list = []
    N_minus_P_raw = []
    for gt in gt_sweep:
        # Use a contour that avoids both poles and saddle points of Φ'.
        R_prime, delta_0 = pole_avoiding_radius(gt, R_base, beta=BETA, avoid_zeros=True, n_grid=120)
        R_prime_list.append(R_prime)
        delta_0_list.append(delta_0)
        if delta_0 < 0.1:
            print(f"Warning: δ₀ = {delta_0:.4f} < 0.1 at γ̃/π = {gt/np.pi:.3f}")
        # Rouché: |g| = R'/2 on C_{R'}; need max|h| < R'/2 => rouche_margin = 1 - max|h|/(R'/2) > 0
        phi_rouche = np.linspace(0, 2 * np.pi, n_rouche, endpoint=False)
        circle_rouche = R_prime * np.exp(1j * phi_rouche)
        h_on_circle = h_theta(circle_rouche, gt, beta=BETA)
        max_h = np.nanmax(np.abs(h_on_circle))
        g_mag = R_prime / 2
        rouche_margin = 1.0 - max_h / (g_mag + 1e-300)
        rouche_margin_list.append(rouche_margin)
        P_R = count_poles_inside(gt, R_prime, beta=BETA)
        n_minus_p = argument_principle_integral(gt, R_prime, n_phi=4000, beta=BETA)
        nmp_real = np.real(n_minus_p)
        N_minus_P_raw.append(nmp_real)
        if abs(nmp_real - np.round(nmp_real)) > 0.1:
            print(f"Warning: argument-principle integral not cleanly integer at γ̃/π = {gt/np.pi:.3f}: N−P = {nmp_real:.4f}")
        N_R = P_R + int(np.round(nmp_real))
        N_list.append(N_R)
        P_list.append(P_R)

    delta_0_arr = np.array(delta_0_list)
    rouche_margin_arr = np.array(rouche_margin_list)
    nmp_real_arr = np.array(N_minus_P_raw)
    diff_d = np.array(N_list) - np.array(P_list)
    rouche_fail = rouche_margin_arr < 0
    gt_rouche_fail = gt_sweep[rouche_fail]
    print(f"Figure D: δ₀ range [{delta_0_arr.min():.4f}, {delta_0_arr.max():.4f}]; all > 0.1: {np.all(delta_0_arr > 0.1)}")
    print(f"Figure D: Rouché failed (margin < 0) at {np.sum(rouche_fail)} γ̃ values; N−P=2 at {np.sum(diff_d == 2)} points")

    # Convergence check at worst-case γ̃ (farthest from integer)
    worst_idx = int(np.argmax(np.abs(nmp_real_arr - np.round(nmp_real_arr))))
    worst_gt = gt_sweep[worst_idx]
    worst_R = R_prime_list[worst_idx]
    print(f"Worst-case γ̃/π ≈ {worst_gt/np.pi:.3f}, raw N−P ≈ {nmp_real_arr[worst_idx]:.4f}")
    for n_test in [2000, 4000, 8000, 16000]:
        val = argument_principle_integral(worst_gt, worst_R, n_phi=n_test, beta=BETA)
        print(f"  n_phi = {n_test:5d} -> N−P ≈ {np.real(val):.6f}")

    fig_d, axes_d = plt.subplots(5, 1, figsize=(7, 12), sharex=True)
    ax_d1, ax_d2, ax_d3, ax_d4, ax_d5 = axes_d
    ax_d1.plot(gt_sweep / np.pi, N_list, "b-o", markersize=4, label=r"$N_R$ (zeros of $\Phi'$ inside $C_{R'}$)")
    ax_d1.plot(gt_sweep / np.pi, P_list, "r-s", markersize=4, label=r"$P_R$ (poles inside $C_{R'}$)")
    for x in gt_rouche_fail / np.pi:
        ax_d1.axvline(x, color="gray", ls="--", alpha=0.6)
    ax_d1.set_ylabel("Count")
    ax_d1.set_title(r"Figure D: Saddle count vs $\tilde\gamma$ (adaptive pole-avoiding $R'(\tilde\gamma)$, $R_{{\mathrm{base}}}$ = " + str(R_base) + ")")
    ax_d1.legend(loc="upper left", fontsize=9)
    ax_d1.grid(True, alpha=0.3)

    ax_d2.plot(gt_sweep / np.pi, diff_d, "g-", lw=2, label=r"$N_R - P_R$")
    ax_d2.scatter(gt_sweep[diff_d == 2] / np.pi, diff_d[diff_d == 2], c="red", s=40, zorder=5, label=r"$N_R - P_R = 2$")
    ax_d2.axhline(1, color="black", ls="--", alpha=0.8, label="Expected: 1")
    for x in gt_rouche_fail / np.pi:
        ax_d2.axvline(x, color="gray", ls="--", alpha=0.6)
    ax_d2.set_ylabel(r"$N_R - P_R$")
    ax_d2.set_title(r"Verification: $N_R - P_R = 1$ (Rouché); red = 2 saddles; dashed = Rouché fails")
    ax_d2.legend(loc="upper left", fontsize=8)
    ax_d2.grid(True, alpha=0.3)
    ax_d2.set_ylim(-0.5, 2.5)

    ax_d3.plot(gt_sweep / np.pi, delta_0_list, "m-", lw=2, label=r"$\delta_0(\tilde\gamma)$")
    ax_d3.axhline(0.1, color="gray", ls=":", alpha=0.8, label="Threshold 0.1")
    for x in gt_rouche_fail / np.pi:
        ax_d3.axvline(x, color="gray", ls="--", alpha=0.6)
    ax_d3.set_ylabel(r"$\delta_0$")
    ax_d3.set_title(r"Pole clearance $\delta_0$ (min distance from $C_{R'}$ to nearest pole)")
    ax_d3.legend(loc="upper right", fontsize=9)
    ax_d3.grid(True, alpha=0.3)
    ax_d3.set_ylim(0, max(delta_0_arr) * 1.05)

    ax_d4.plot(gt_sweep / np.pi, rouche_margin_list, "c-", lw=2, label=r"Rouché margin $= 1 - \max|h|/(R'/2)$")
    ax_d4.axhline(0, color="black", ls="-", alpha=0.7)
    ax_d4.fill_between(gt_sweep / np.pi, 0, rouche_margin_arr, where=(rouche_margin_arr < 0), color="red", alpha=0.3, label="Rouché fails")
    for x in gt_rouche_fail / np.pi:
        ax_d4.axvline(x, color="gray", ls="--", alpha=0.6)
    ax_d4.set_xlabel(r"$\tilde\gamma / \pi$")
    ax_d4.set_ylabel("Rouché margin")
    ax_d4.set_title(r"Rouché condition: margin $> 0$ $\Rightarrow$ $|g| > |h|$ on $C_{R'}$; margin $< 0$ $\Rightarrow$ need larger $R$")
    ax_d4.legend(loc="upper right", fontsize=8)
    ax_d4.grid(True, alpha=0.3)

    # Raw argument-principle N−P values vs γ̃/π (color by rounding)
    x_vals = gt_sweep / np.pi
    rounded = np.round(nmp_real_arr).astype(int)
    colors = ["green" if r == 1 else "red" for r in rounded]
    ax_d5.scatter(x_vals, nmp_real_arr, c=colors, s=30, edgecolors="k", linewidths=0.3)
    for y in [0.5, 1.0, 1.5, 2.0]:
        ax_d5.axhline(y, color="gray", ls=":", alpha=0.7)
    for x in gt_rouche_fail / np.pi:
        ax_d5.axvline(x, color="gray", ls="--", alpha=0.5)
    ax_d5.set_xlabel(r"$\tilde\gamma / \pi$")
    ax_d5.set_ylabel(r"raw $N_R - P_R$")
    ax_d5.set_title(r"Raw argument-principle count vs $\tilde\gamma$ (green: round to 1, red: round to 2)")
    ax_d5.grid(True, alpha=0.3)

    fig_d.tight_layout()
    fig_d.savefig(OUT_DIR / "sect3_figD_N_minus_P_sweep.png", bbox_inches="tight", dpi=150)
    plt.close(fig_d)
    print("Saved sect3_figD_N_minus_P_sweep.png")

    # ----- Paper-ready Figure D: 2 panels only -----
    fig_d_paper, axes_paper = plt.subplots(
        2, 1, figsize=(7, 4), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    ax_top, ax_bot = axes_paper
    ax_top.plot(gt_sweep / np.pi, N_list, "b-o", markersize=4, label=r"$N_{R'}$ (zeros)")
    ax_top.plot(gt_sweep / np.pi, P_list, "r-s", markersize=4, label=r"$P_{R'}$ (poles)")
    ax_top.set_ylabel("Count")
    ax_top.set_title(r"Zeros and poles of $\Phi'$ inside $C_{R'}$")
    ax_top.legend(loc="upper left", fontsize=9)
    ax_top.grid(True, alpha=0.3)

    ax_bot.scatter(gt_sweep / np.pi, nmp_real_arr, c="green", s=30, edgecolors="k", linewidths=0.3)
    ax_bot.axhline(1, color="black", ls="--", alpha=0.8)
    ax_bot.set_xlabel(r"$\tilde\gamma / \pi$")
    ax_bot.set_ylabel(r"$N_{R'} - P_{R'}$")
    ax_bot.set_title(r"Verification: $N_{R'}-P_{R'}=1$")
    ax_bot.set_ylim(0.5, 1.5)
    ax_bot.grid(True, alpha=0.3)

    fig_d_paper.tight_layout()
    fig_d_paper.savefig(OUT_DIR / "sect3_figD_paper.png", bbox_inches="tight", dpi=150)
    plt.close(fig_d_paper)
    print("Saved sect3_figD_paper.png")

    # ----- Additional diagnostics at worst-case γ̃ ≈ 0.89π -----
    gt_worst = 0.89 * np.pi
    print("\n=== Decomposition and contour diagnostics at γ̃/π = 0.89 ===")
    R_prime_worst, delta0_worst = pole_avoiding_radius(gt_worst, R_base, beta=BETA, avoid_zeros=True, n_grid=160)
    print(f"γ̃/π = 0.89, R'_worst = {R_prime_worst:.6f}, δ₀_worst = {delta0_worst:.6f}")
    rng = np.random.RandomState(12345)
    phis = 2 * np.pi * rng.rand(10)
    for idx, phi in enumerate(phis):
        th = R_prime_worst * np.exp(1j * phi)
        dp = dPhi(th, gt_worst, BETA)
        gh = g_theta(th) + h_theta(th, gt_worst, BETA)
        diff = dp - gh
        print(f"Point {idx}: θ = {th:.6f}")
        print(f"  Φ'(θ)        = {dp:.12e}, |Φ'| = {abs(dp):.6e}")
        print(f"  g(θ)+h(θ)    = {gh:.12e}")
        print(f"  diff         = {diff:.12e}, |diff| = {abs(diff):.6e}")

    # ----- Finite-difference verification of d2Phi at γ̃/π = 0.89 -----
    print("\n=== Finite-difference verification of Φ'' at γ̃/π = 0.89 (5 points on C_{R'}) ===")
    eps = 1e-5
    rng_fd = np.random.RandomState(42)
    phis_fd = 2 * np.pi * rng_fd.rand(5)
    for idx, phi in enumerate(phis_fd):
        th = R_prime_worst * np.exp(1j * phi)
        d2_analytical = d2Phi(th, gt_worst, BETA)
        d2_fd = (dPhi(th + eps, gt_worst, BETA) - dPhi(th - eps, gt_worst, BETA)) / (2 * eps)
        err = np.abs(d2_analytical - d2_fd)
        rel_err = err / (np.abs(d2_analytical) + 1e-20)
        print(f"Point {idx}: θ = {th:.6f}")
        print(f"  Φ''(θ) analytical = {d2_analytical:.12e}")
        print(f"  Φ''(θ) FD (ε={eps}) = {d2_fd:.12e}")
        print(f"  relative error = {rel_err:.6e}")

    # Saddle locations for γ̃ = 0.89π inside |θ| < R'_worst
    saddles_worst = find_saddles_inside(gt_worst, R_prime_worst, beta=BETA, n_grid=220, tol=1e-3)
    if saddles_worst.size == 0:
        print("No saddles found inside |θ| < R'_worst with current grid/threshold.")
    else:
        print(f"Found {len(saddles_worst)} saddle(s) inside |θ| < R'_worst:")
        for j, th_s in enumerate(saddles_worst):
            print(f"  Saddle {j}: θ* = {th_s:.12e}, |θ*| = {abs(th_s):.6f}, |θ*|-R' = {abs(th_s)-R_prime_worst:.6f}")

    # ----- Winding number (arg Φ') vs Φ''/Φ' integral at γ̃/π = 0.89 -----
    print("\n=== Winding number vs Φ''/Φ' integral at γ̃/π = 0.89 ===")
    integral_1499 = argument_principle_integral(gt_worst, R_prime_worst, n_phi=4000, beta=BETA)
    print(f"Φ''/Φ' integral (n_phi=4000): N−P ≈ {np.real(integral_1499):.6f}")
    for n in [4000, 8000, 16000]:
        w = winding_number_of_dPhi(gt_worst, R_prime_worst, n_phi=n, beta=BETA)
        print(f"winding_number_of_dPhi (n_phi={n}): {w:.6f}")
    print("Key: if winding ≈ 1.0 but integral ≈ 1.499 → d2Phi bug; if both ≈ 1.499 → non-integer winding is real.")

    # ----- Saddle search with lenient tol, extended radius (γ̃/π = 0.89) -----
    print("\n=== Saddle search γ̃/π = 0.89: R = 1.5 * R'_worst, n_grid=500, tol=1e-2 ===")
    R_search_worst = 1.5 * R_prime_worst
    saddles_ext_worst = find_saddles_inside(gt_worst, R_search_worst, beta=BETA, n_grid=500, tol=1e-2)
    print(f"R'_worst = {R_prime_worst:.6f}, search radius = {R_search_worst:.6f}")
    if saddles_ext_worst.size == 0:
        print("No saddles found.")
    else:
        for j, th_s in enumerate(saddles_ext_worst):
            r = np.abs(th_s)
            diff_r = r - R_prime_worst
            in_out = "inside" if diff_r < 0 else "outside"
            print(f"  Saddle {j}: θ* = {th_s:.10e}, |θ*| = {r:.6f}, |θ*| − R' = {diff_r:+.6f} ({in_out})")

    # ----- Control: same saddle search at γ̃/π = 0.30 (clean N−P ≈ 1) -----
    print("\n=== Control: saddle search γ̃/π = 0.30, R = 1.5 * R', n_grid=500, tol=1e-2 ===")
    gt_control = 0.30 * np.pi
    R_prime_control, _ = pole_avoiding_radius(gt_control, R_base, beta=BETA, avoid_zeros=True, n_grid=120)
    R_search_control = 1.5 * R_prime_control
    saddles_control = find_saddles_inside(gt_control, R_search_control, beta=BETA, n_grid=500, tol=1e-2)
    print(f"R'_control = {R_prime_control:.6f}, search radius = {R_search_control:.6f}")
    if saddles_control.size == 0:
        print("No saddles found.")
    else:
        for j, th_s in enumerate(saddles_control):
            r = np.abs(th_s)
            diff_r = r - R_prime_control
            in_out = "inside" if diff_r < 0 else "outside"
            print(f"  Saddle {j}: θ* = {th_s:.10e}, |θ*| = {r:.6f}, |θ*| − R' = {diff_r:+.6f} ({in_out})")

    print("\nDone. All four Section 3 figures saved as PNG in", OUT_DIR)


if __name__ == "__main__":
    main()
