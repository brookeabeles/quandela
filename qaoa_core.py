"""
Shared QAOA action utilities for saddle–singularity analysis.

Provides: w_of_gamma, A, Phi, dPhi, d2Phi, d3Phi, singularities, dominant_saddle_path,
all_saddles, nearest_singularity, rouche_exclusion_radius, lattice_spacing, validate_saddle.
"""

import numpy as np

# Default β (e.g. -π/2)
DEFAULT_BETA = -np.pi / 2


def w_of_gamma(gt):
    return np.sqrt(-1j * gt / 2 + 0j)


def A(theta, gt, beta=DEFAULT_BETA):
    ww = w_of_gamma(gt)
    return np.cos(beta / 2) - 1j * np.sin(beta / 2) * np.exp(theta * ww)


def Phi(theta, gt, beta=DEFAULT_BETA):
    aa = A(theta, gt, beta)
    return -theta**2 / 4 + np.log(aa + 1e-300)


def dPhi(theta, gt, beta=DEFAULT_BETA):
    """Φ'(θ). With overflow protection for large Re(θ·w)."""
    ww = w_of_gamma(gt)
    exponent = theta * ww
    # If Re(exponent) > 500, A'/A ≈ w, so Φ' ≈ -θ/2 + w (avoids overflow)
    if np.isscalar(theta):
        if np.real(exponent) > 500:
            return -theta / 2 + ww
        aa = A(theta, gt, beta)
        if np.abs(aa) < 1e-14:
            return np.inf
        return -theta / 2 - 1j * ww * np.sin(beta / 2) * np.exp(exponent) / (aa + 1e-300)
    aa = A(theta, gt, beta)
    return -theta / 2 - 1j * ww * np.sin(beta / 2) * np.exp(exponent) / (aa + 1e-300)


def validate_saddle(theta_star, gt, beta=DEFAULT_BETA, tol=1e-8):
    """Verify θ* is actually a saddle: |Φ'(θ*)| < tol and Φ''(θ*) ≠ 0."""
    residual = np.abs(dPhi(theta_star, gt, beta))
    if np.isinf(residual):
        return False
    hessian = np.abs(d2Phi(theta_star, gt, beta))
    return residual < tol and hessian > tol


def d2Phi(theta, gt, beta=DEFAULT_BETA):
    """
    Φ''(θ) = −1/2 + (A''A − (A')²)/A².
    With A' = −i sin(β/2) w e^{θw}, A'' = −i sin(β/2) w² e^{θw}, we have
    (A')² = (−i s w e)² = −(s w e)², so (A''A − (A')²)/A² = A''/A + (s w e)²/A².
    """
    ww = w_of_gamma(gt)
    s = np.sin(beta / 2)
    e = np.exp(theta * ww)
    aa = A(theta, gt, beta) + 1e-300
    term1 = -0.5
    term2 = -1j * ww**2 * s * e / aa
    term3 = (ww * s * e / aa) ** 2
    return term1 + term2 + term3


def d3Phi(theta, gt, beta=DEFAULT_BETA):
    """Third derivative Φ'''(θ). Φ'' = -1/2 + (A'/A)' so Φ''' = (A'/A)''."""
    ww = w_of_gamma(gt)
    s = np.sin(beta / 2)
    e = np.exp(theta * ww)
    aa = A(theta, gt, beta) + 1e-300
    ap = -1j * ww * s * e
    app = -1j * ww**2 * s * e
    appp = -1j * ww**3 * s * e
    # (A'/A)' = A''/A - (A'/A)^2
    # (A'/A)'' = A'''/A - 2 A''A'/A^2 + 2(A')^3/A^3 - A'A''/A^2 = A'''/A - 3 A''A'/A^2 + 2(A')^3/A^3
    ratio_pp = appp / aa - 3 * (app * ap) / (aa**2) + 2 * (ap**3) / (aa**3)
    return ratio_pp


def singularities(gt, kmax=50, beta=DEFAULT_BETA):
    """Return singularity lattice {θ_k} for |k| ≤ kmax."""
    ww = w_of_gamma(gt)
    log_arg = -1j / np.tan(beta / 2)
    base = np.log(log_arg)
    ks = np.arange(-kmax, kmax + 1)
    return (base + 2j * np.pi * ks) / ww


def lattice_spacing(gt):
    """L(γ̃) = 2π/|w(γ̃)|."""
    return 2 * np.pi / np.abs(w_of_gamma(gt))


def dominant_saddle_path(gt_target, n_steps=500, beta=DEFAULT_BETA):
    """Path continuation from γ̃→0 to gt_target. Returns (gts, thetas)."""
    import warnings
    gts = np.linspace(1e-6, gt_target, n_steps)
    thetas = np.zeros(n_steps, dtype=complex)
    thetas[0] = 0.0 + 0j
    for i in range(1, n_steps):
        gt = gts[i]
        th = thetas[i - 1]
        for _ in range(200):
            f = dPhi(th, gt, beta)
            fp = d2Phi(th, gt, beta)
            if abs(fp) < 1e-15:
                break
            dth = -f / fp
            th = th + dth
            if abs(dth) < 1e-14:
                break
        # Polishing: 5 extra iterations with no step-size damping
        for _ in range(5):
            f = dPhi(th, gt, beta)
            fp = d2Phi(th, gt, beta)
            if abs(fp) < 1e-15:
                break
            th = th - f / fp
        res = abs(dPhi(th, gt, beta))
        if res > 1e-10:
            try:
                from scipy.optimize import root
                sol = root(lambda z: np.array([dPhi(z[0] + 1j * z[1], gt, beta).real, dPhi(z[0] + 1j * z[1], gt, beta).imag]),
                           [th.real, th.imag], method='hybr')
                if sol.success:
                    th = sol.x[0] + 1j * sol.x[1]
                    res = abs(dPhi(th, gt, beta))
            except Exception:
                pass
        if res > 1e-8:
            warnings.warn(f"dominant_saddle_path: |Φ'| = {res:.2e} at γ̃/π = {gt/np.pi:.4f}", UserWarning)
        thetas[i] = th
    return gts, thetas


def _newton_saddle(theta_init, gt, beta, tol=1e-10, max_iter=200):
    """Newton to solve Φ'(θ)=0. Returns refined θ or None."""
    import warnings
    th = theta_init + 0.0
    for _ in range(max_iter):
        f = dPhi(th, gt, beta)
        if abs(f) < tol:
            break
        fp = d2Phi(th, gt, beta)
        if abs(fp) < 1e-14:
            return None
        step = f / fp
        if abs(step) > 1.0:
            step = step / abs(step)
        th = th - step
    # Polishing: 5 iterations with no damping
    for _ in range(5):
        f = dPhi(th, gt, beta)
        fp = d2Phi(th, gt, beta)
        if abs(fp) < 1e-14:
            break
        th = th - f / fp
    res = abs(dPhi(th, gt, beta))
    if res > 1e-10:
        try:
            from scipy.optimize import root
            sol = root(lambda z: np.array([dPhi(z[0] + 1j * z[1], gt, beta).real, dPhi(z[0] + 1j * z[1], gt, beta).imag]),
                       [th.real, th.imag], method='hybr')
            if sol.success:
                th = sol.x[0] + 1j * sol.x[1]
                res = abs(dPhi(th, gt, beta))
        except Exception:
            pass
    # Warn only when we return a saddle with marginal residual (accepted but > 1e-8)
    if res < 1e-6 and res > 1e-8:
        warnings.warn(f"_newton_saddle: |Φ'| = {res:.2e} at θ≈{th}", UserWarning)
    return th if res < 1e-6 else None


def all_saddles(gt, beta=DEFAULT_BETA, grid_size=20, half_width=8, tol_dedup=1e-8, n_saddles_max=20):
    """
    Find ALL saddles by grid search + Newton, deduplicate, sort by Re Φ descending.
    Returns list of (theta, Phi_value) with dominant first.
    """
    re_pts = np.linspace(-half_width, half_width, grid_size)
    im_pts = np.linspace(-half_width, half_width, grid_size)
    found = []
    for re0 in re_pts:
        for im0 in im_pts:
            th0 = re0 + 1j * im0
            th = _newton_saddle(th0, gt, beta)
            if th is None:
                continue
            if abs(dPhi(th, gt, beta)) > 1e-6:
                continue
            # Deduplicate
            if any(abs(th - s[0]) < tol_dedup for s in found):
                continue
            if not validate_saddle(th, gt, beta, tol=1e-6):
                import warnings
                warnings.warn(f"Saddle validation failed at θ={th}, γ̃={gt}", UserWarning)
            found.append((th, Phi(th, gt, beta)))
    found.sort(key=lambda x: -x[1].real)
    return found[:n_saddles_max]


def nearest_singularity(theta, gt, kmax=50, beta=DEFAULT_BETA):
    """Distance from theta to nearest singularity; returns (dist, theta_k)."""
    sings = singularities(gt, kmax, beta)
    dists = np.abs(theta - sings)
    imin = np.argmin(dists)
    return dists[imin], sings[imin]


def rouche_exclusion_radius(gt, theta_k, beta=DEFAULT_BETA, n_angles=200):
    """Largest ρ where min|f| > max|g| on |θ - θ_k| = ρ; f = 1/(θ-θ_k), g = Φ' - f."""
    angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
    rhos = np.logspace(-6, 0, 300)
    rho_star = 0.0
    for rho in rhos:
        circle = theta_k + rho * np.exp(1j * angles)
        f_vals = 1.0 / (circle - theta_k)
        min_f = np.min(np.abs(f_vals))
        g_vals = np.array([dPhi(th, gt, beta) for th in circle]) - f_vals
        max_g = np.max(np.abs(g_vals))
        if min_f > max_g:
            rho_star = rho
        else:
            break
    return rho_star


def rouche_exclusion_radius_max(gt, kmax=5, beta=DEFAULT_BETA):
    """Max of rouche_exclusion_radius over all singularities with |k| ≤ kmax."""
    sings = singularities(gt, kmax, beta)
    return max(rouche_exclusion_radius(gt, sk, beta) for sk in sings) if sings else 0.0
