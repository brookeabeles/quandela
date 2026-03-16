"""
Stokes transitions v3: detect from step function, bisect, forward continuation.

Part A: N(γ̃) on fine grid (0.01π), detect transitions where N increases, bisect to γ̃_c.
Part B: At each γ̃_c find θ_c via set difference of saddles at γ̃_c±ε; track branches forward.
Part C: |θ_birth - θ_k_nearest| / L(γ̃_c); optional inset: θ_c in complex plane.
Output: 3-panel figure (+ optional inset).
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from qaoa_core import (
    Phi, dPhi, d2Phi, singularities, nearest_singularity, lattice_spacing,
    validate_saddle, all_saddles, DEFAULT_BETA,
)

beta = DEFAULT_BETA
plt.rcParams.update({'font.size': 10, 'figure.dpi': 150})

R, N_QUAD = 7.93, 600
try:
    from scipy.integrate import quad
    _has_scipy = True
except ImportError:
    _has_scipy = False


def integrand_ratio(theta, gt, beta_=beta):
    fp = dPhi(theta, gt, beta_)
    fpp = d2Phi(theta, gt, beta_)
    if np.abs(fp) < 1e-14:
        return 0.0
    return fpp / fp


def argument_principle_integral(gt, R_=R, n_quad=N_QUAD, beta_=beta):
    total = 0.0 + 0j
    def side_bottom(t):
        th = (-R_ + 2*R_*t) - 1j*R_
        return integrand_ratio(th, gt, beta_) * (2*R_)
    def side_right(t):
        th = R_ + 1j*(-R_ + 2*R_*t)
        return integrand_ratio(th, gt, beta_) * (2*R_*1j)
    def side_top(t):
        th = (R_ - 2*R_*t) + 1j*R_
        return integrand_ratio(th, gt, beta_) * (-2*R_)
    def side_left(t):
        th = -R_ + 1j*(R_ - 2*R_*t)
        return integrand_ratio(th, gt, beta_) * (-2*R_*1j)
    if _has_scipy:
        for func in [side_bottom, side_right, side_top, side_left]:
            re_val, _ = quad(lambda t: np.real(func(t)), 0, 1, limit=100)
            im_val, _ = quad(lambda t: np.imag(func(t)), 0, 1, limit=100)
            total += re_val + 1j*im_val
    else:
        t = np.linspace(0.5/n_quad, 1 - 0.5/n_quad, n_quad)
        dt = 1.0 / n_quad
        for func in [side_bottom, side_right, side_top, side_left]:
            vals = np.array([func(ti) for ti in t])
            total += np.sum(vals) * dt
    return int(np.round(np.real(total / (2*np.pi*1j))))


def count_poles_inside(gt, R_=R, kmax=100, beta_=beta):
    sings = singularities(gt, kmax, beta_)
    return sum(1 for s in sings if np.abs(s.real) < R_ and np.abs(s.imag) < R_)


def saddle_count(gt, beta_=beta):
    w = argument_principle_integral(gt, beta_=beta_)
    return w + count_poles_inside(gt, beta_=beta_)


def saddles_at(gt, beta_=beta, grid_size=50, half_width=10):
    """All saddles at given γ̃ via qaoa_core.all_saddles (robust Newton + validation)."""
    lst = all_saddles(gt, beta=beta_, grid_size=grid_size, half_width=half_width, tol_dedup=1e-6, n_saddles_max=80)
    return [th for th, _ in lst]


def bisect_transition(gt_lo, gt_hi, beta_=beta, tol_gt=0.001*np.pi):
    """Find γ̃_c in (gt_lo, gt_hi] where N jumps; return (γ̃_c, N_before, N_after)."""
    n_lo = saddle_count(gt_lo, beta_)
    n_hi = saddle_count(gt_hi, beta_)
    if n_hi <= n_lo:
        return None
    while gt_hi - gt_lo > tol_gt:
        gt_mid = (gt_lo + gt_hi) / 2
        n_mid = saddle_count(gt_mid, beta_)
        if n_mid > n_lo:
            gt_hi = gt_mid
            n_hi = n_mid
        else:
            gt_lo = gt_mid
            n_lo = n_mid
    return (gt_hi, n_lo, n_hi)


def find_birth_locations(gt_c, n_before, n_after, eps=0.05*np.pi, beta_=beta):
    """
    Find the new saddles born at γ̃_c.

    Strategy: find all saddles at γ̃_c + eps and at γ̃_c - eps.
    For each saddle in the 'after' set, compute its minimum distance
    to any saddle in the 'before' set. The new saddles are those with
    the largest minimum distances.

    Return the individual new saddles (not midpoints).
    Uses larger grid (60×60, half_width=12) to catch births far from origin.
    """
    after = saddles_at(gt_c + eps, beta_=beta_, grid_size=60, half_width=12)
    before = saddles_at(gt_c - eps, beta_=beta_, grid_size=60, half_width=12)
    n_new = n_after - n_before
    if n_new <= 0 or len(after) < n_new:
        return []

    def min_dist_to_set(th, ref_set):
        if not ref_set:
            return np.inf
        return min(np.abs(th - s) for s in ref_set)

    # Score each 'after' saddle by distance from nearest 'before' saddle
    scored = [(th, min_dist_to_set(th, before)) for th in after]
    scored.sort(key=lambda x: -x[1])

    # The top n_new are the new ones
    return [th for th, _ in scored[:n_new]]


def track_forward(theta_start, gt_start, gt_end, step, beta_=beta, max_step=0.15):
    """Track saddle forward from gt_start to gt_end. Newton with step clipping. Returns list of (gt, theta)."""
    path = [(gt_start, theta_start + 0.0)]  # record starting saddle first (before any Newton)
    th = theta_start + 0.0
    gt = gt_start + step
    while gt <= gt_end - step * 0.5:
        for _ in range(80):
            f = dPhi(th, gt, beta_)
            fp = d2Phi(th, gt, beta_)
            if np.abs(f) < 1e-10:
                break
            if np.abs(fp) < 1e-14:
                break
            dth = -f / fp
            if np.abs(dth) > max_step:
                dth = max_step * dth / np.abs(dth)
            th = th + dth
        if np.abs(dPhi(th, gt, beta_)) > 1e-4:
            break  # stop tracking if we lost the saddle
        path.append((gt, th + 0.0))
        gt += step
    return path


# ── Part A: Fine grid N(γ̃), detect transitions ───────────────────────────

print("Stokes v3: fine grid N(γ̃)...")
step_fine = 0.01 * np.pi
gts_fine = np.arange(0.15*np.pi, 6*np.pi + step_fine/2, step_fine)
N_fine = np.array([saddle_count(gt) for gt in gts_fine])

# Detect jumps: N[i] > N[i-1] => transition in (gts_fine[i-1], gts_fine[i]]
# Fix 5: from each conjugate pair track only one (Im θ < 0; else Im θ > 0)
EPS_BIRTH = 0.05 * np.pi
transitions = []
for i in range(1, len(N_fine)):
    if N_fine[i] > N_fine[i-1]:
        res = bisect_transition(gts_fine[i-1], gts_fine[i])
        if res:
            gt_c, n_before, n_after = res
            new_saddles = find_birth_locations(gt_c, n_before, n_after)
            new_saddles = [th for th in new_saddles if th.imag < 0]
            if not new_saddles:
                new_saddles = find_birth_locations(gt_c, n_before, n_after)
                new_saddles = [th for th in new_saddles if th.imag > 0]
            for theta_c in new_saddles:
                d, _ = nearest_singularity(theta_c, gt_c, kmax=40)
                L_c = lattice_spacing(gt_c)
                transitions.append((gt_c, theta_c, d, L_c))

print(f"  Found {len(transitions)} transitions (birth locations)")

# Part B: Forward continuation from each birth
# Fix 2: validate (and recover via Newton) starting point before tracking; use eps=0.05π
from qaoa_core import dominant_saddle_path
branch_curves = []
gt_max = 6 * np.pi
step_fwd = 0.01 * np.pi
gts_dom, thetas_dom = dominant_saddle_path(gt_max, n_steps=300)
dom_curve = [(gt, Phi(th, gt).real) for gt, th in zip(gts_dom, thetas_dom)]
branch_curves.append(dom_curve)

for gt_c, theta_c, d, L_c in transitions:
    gt_start = gt_c + EPS_BIRTH
    th = theta_c + 0.0
    if not validate_saddle(theta_c, gt_start, beta):
        for _ in range(100):
            f = dPhi(th, gt_start, beta)
            fp = d2Phi(th, gt_start, beta)
            if abs(fp) < 1e-14:
                break
            dth = -f / fp
            if abs(dth) > 1.0:
                dth = 1.0 * dth / abs(dth)
            th = th + dth
            if abs(f) < 1e-12:
                break
        if not validate_saddle(th, gt_start, beta):
            continue
        theta_c = th
    path = track_forward(theta_c, gt_start, gt_max, step_fwd)
    if path:
        branch_curves.append([(gt, Phi(th, gt).real) for gt, th in path])

# Fix 4: keep only branches that are subdominant at their first point (strictly below dominant)
gts_dom_arr = np.array([p[0] for p in dom_curve])
rephis_dom_arr = np.array([p[1] for p in dom_curve])
filtered = [dom_curve]
for curve in branch_curves[1:]:
    if not curve:
        continue
    gt_first, rephi_first = curve[0][0], curve[0][1]
    dom_at_first = np.interp(gt_first, gts_dom_arr, rephis_dom_arr)
    # Keep if curve starts below dominant (with small tolerance for numerical noise)
    if rephi_first < dom_at_first - 1e-10:
        filtered.append(curve)
branch_curves = filtered
print(f"  Panel (b): {len(branch_curves) - 1} subdominant branches (after filter)")

# ── Figure: 3 panels ────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# (a) N(γ̃) step function
ax = axes[0]
ax.step(gts_fine / np.pi, N_fine, where='post', color='black', lw=1.5)
ax.set_xlabel(r'$\tilde{\gamma}$ (units of $\pi$)')
ax.set_ylabel(r'$N(\tilde{\gamma})$')
ax.set_title(r'(a) Saddle count (argument principle, fine grid)')
ax.set_xlim(0.15, 6)
ax.set_ylim(bottom=0)
ax.grid(True, alpha=0.3)

# (b) Re Φ branches (dominant + from births)
ax = axes[1]
colors = ['green'] + ['red'] * 20
for j, curve in enumerate(branch_curves):
    if not curve:
        continue
    gts_b, rephis = zip(*curve)
    ax.plot(np.array(gts_b)/np.pi, rephis, '-', color=colors[min(j, len(colors)-1)], lw=1.0 if j > 0 else 1.5)
ax.set_xlabel(r'$\tilde{\gamma}$ (units of $\pi$)')
ax.set_ylabel(r'Re $\Phi(\theta^*)$')
ax.set_title(r'(b) Branches (dominant + forward from births)')
ax.set_xlim(0.15, 6)
ax.grid(True, alpha=0.3)

# (c) |θ_birth - θ_k| / L(γ̃_c)
ax = axes[2]
if transitions:
    gt_c_vals = [t[0] for t in transitions]
    dist_vals = [t[2] for t in transitions]
    L_vals = [t[3] for t in transitions]
    ratio_vals = [d / (L + 1e-20) for d, L in zip(dist_vals, L_vals)]
    ax.semilogy(np.array(gt_c_vals)/np.pi, dist_vals, 'ko', ms=6, label=r'$|\theta_c - \theta_k|$')
    ax2 = ax.twinx()
    ax2.plot(np.array(gt_c_vals)/np.pi, ratio_vals, 'bs', ms=5, label=r'$d/L$')
    ax2.set_ylabel(r'$|\theta_c - \theta_k|/L$')
    ax2.legend(loc='upper right')
ax.set_xlabel(r'$\tilde{\gamma}_c$ (units of $\pi$)')
ax.set_ylabel(r'$|\theta_{\mathrm{birth}} - \theta_{k}|$')
ax.set_title(r'(c) Birth–singularity distance (and $d/L$ at birth)')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('stokes_transitions_v3.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved stokes_transitions_v3.png")
