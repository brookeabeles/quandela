"""Consolidated BM24 workflow core (toy, k-SAT p=1, general-p, optimization, validation)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import numpy as np

try:
    from scipy.optimize import differential_evolution, minimize, root
    from scipy.special import gammaln
except Exception:  # pragma: no cover
    differential_evolution = None
    minimize = None
    root = None
    gammaln = None


THRESHOLDS = {2: 1.0, 4: 9.93, 8: 176.54, 10: 708.92, 16: 45425.2}
TOY_BETA_RANGE = (-np.pi, -1e-2)
TOY_GAMMA_TILDE_RANGE = (1e-2, 3.0 * np.pi)


def _require_scipy() -> None:
    if minimize is None or gammaln is None:
        raise RuntimeError("SciPy is required.")


def sum_over_alpha(v: np.ndarray, dim: int, apply_half: bool = True) -> np.ndarray:
    size = v.size
    z1 = v.copy()
    for i in range(dim):
        shape = (1 << (dim - 1 - i), 2, 1 << i)
        zr = z1.reshape(shape)
        zr[:, 1, :] += zr[:, 0, :]
    idx = np.arange(size, dtype=np.uint64)
    rev = idx ^ (size - 1)
    z0 = v[rev].copy()
    for i in range(dim):
        shape = (1 << (dim - 1 - i), 2, 1 << i)
        zr = z0.reshape(shape)
        zr[:, 0, :] += zr[:, 1, :]
    z0 = z0[rev]
    out = z0 + z1 - v[0]
    return out / 2.0 if apply_half else out


def sum_over_s(v: np.ndarray, dim: int, apply_half: bool = True) -> np.ndarray:
    size = v.size
    idx = np.arange(size, dtype=np.uint64)
    rev = idx ^ (size - 1)
    z0 = v[rev].copy()
    z1 = v.copy()
    for i in range(dim):
        shape = (1 << (dim - 1 - i), 2, 1 << i)
        zr0 = z0.reshape(shape)
        zr0[:, 0, :] += zr0[:, 1, :]
        zr1 = z1.reshape(shape)
        zr1[:, 1, :] += zr1[:, 0, :]
    out = z0 + z1 - v[0]
    return out / 2.0 if apply_half else out


def toy_amplitude_exact(n: int, beta: float, gamma_tilde: float) -> np.complex128:
    _require_scipy()
    gamma = gamma_tilde / float(n)
    amp = np.complex128(0.0 + 0.0j)
    c = np.cos(beta / 2.0)
    s = -1j * np.sin(beta / 2.0)
    log_half = 0.5 * n * np.log(2.0)
    for k in range(n + 1):
        log_binom = gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)
        term = np.exp(log_binom + (n - k) * np.log(c + 0.0j) + k * np.log(s + 0.0j) - log_half - 1j * gamma * (k**2) / 2.0)
        amp += term
    return np.complex128(amp)


def Phi_toy(theta: complex, beta: float, gamma_tilde: float) -> np.complex128:
    w = np.sqrt(-1j * gamma_tilde / 2.0)
    A = np.cos(beta / 2.0) - 1j * np.sin(beta / 2.0) * np.exp(theta * w)
    return -np.log(2.0) / 2.0 - 1j * beta / 2.0 - theta**2 / 4.0 + np.log(np.exp(1j * beta / 2.0) * A)


def dPhi_toy(theta: complex, beta: float, gamma_tilde: float) -> np.complex128:
    w = np.sqrt(-1j * gamma_tilde / 2.0)
    exp_tw = np.exp(theta * w)
    A = np.cos(beta / 2.0) - 1j * np.sin(beta / 2.0) * exp_tw
    dA = -1j * w * np.sin(beta / 2.0) * exp_tw
    return -theta / 2.0 + dA / A


def d2Phi_toy(theta: complex, beta: float, gamma_tilde: float) -> np.complex128:
    w = np.sqrt(-1j * gamma_tilde / 2.0)
    exp_tw = np.exp(theta * w)
    A = np.cos(beta / 2.0) - 1j * np.sin(beta / 2.0) * exp_tw
    dA = -1j * w * np.sin(beta / 2.0) * exp_tw
    d2A = -1j * (w**2) * np.sin(beta / 2.0) * exp_tw
    return -0.5 + (d2A * A - dA**2) / (A**2)


@dataclass
class ContinuationConfig:
    n_steps: int = 200
    newton_max_iter: int = 50
    newton_tol: float = 1e-14
    jac_tol: float = 1e-15


def find_saddle_continuation(beta: float, gamma_tilde_target: float, cfg: Optional[ContinuationConfig] = None, theta_init: complex = 0.0 + 0.0j) -> np.complex128:
    cfg = cfg or ContinuationConfig()
    theta_star = np.complex128(theta_init)
    for gt in np.linspace(0.0, gamma_tilde_target, cfg.n_steps + 1)[1:]:
        for _ in range(cfg.newton_max_iter):
            f = dPhi_toy(theta_star, beta, float(gt))
            if abs(f) < cfg.newton_tol:
                break
            df = d2Phi_toy(theta_star, beta, float(gt))
            if abs(df) < cfg.jac_tol:
                break
            theta_star = theta_star - f / df
    return theta_star


def scaling_exponent_toy(beta: float, gamma_tilde: float, cfg: Optional[ContinuationConfig] = None) -> float:
    theta_star = find_saddle_continuation(beta, gamma_tilde, cfg=cfg)
    return float(2.0 * Phi_toy(theta_star, beta, gamma_tilde).real)


def optimize_toy_model(beta_bounds: tuple[float, float], gamma_tilde_bounds: tuple[float, float], seed: int = 42) -> dict:
    _require_scipy()
    if differential_evolution is None:
        raise RuntimeError("SciPy differential_evolution is required.")

    def neg_exponent(params: np.ndarray) -> float:
        beta, gamma_tilde = params
        try:
            val = scaling_exponent_toy(float(beta), float(gamma_tilde))
            return -min(val, 0.0)
        except Exception:
            return 1e10

    de = differential_evolution(neg_exponent, bounds=[beta_bounds, gamma_tilde_bounds], seed=seed, maxiter=120, tol=1e-8, polish=False)
    nm = minimize(neg_exponent, x0=np.array(de.x, dtype=float), method="Nelder-Mead", options={"xatol": 1e-10, "fatol": 1e-12, "maxiter": 2000})
    return {"beta": float(nm.x[0]), "gamma_tilde": float(nm.x[1]), "exponent": float(-nm.fun), "success": bool(nm.success), "message": str(nm.message)}


def validate_toy(beta: float, gamma_tilde: float, n_values: Optional[Iterable[int]] = None) -> dict:
    _require_scipy()
    n_values = n_values or range(100, 5001, 100)
    n_arr = np.array(list(n_values), dtype=float)
    logs = [np.log(np.abs(toy_amplitude_exact(int(n), beta, gamma_tilde)) ** 2) for n in n_arr.astype(int)]
    slope, intercept = np.polyfit(n_arr, np.array(logs, dtype=float), 1)
    pred = scaling_exponent_toy(beta, gamma_tilde)
    return {"empirical_exponent": float(slope), "predicted_exponent": float(pred), "intercept": float(intercept), "abs_difference": float(abs(slope - pred))}


def ksat_p1_psucc_vectorized(n: int, k: int, r: float, beta: float, gamma: float) -> float:
    _require_scipy()
    prefactor = np.exp(-(2.0 ** (-k)) * r * n * (1.0 + 4.0 * np.sin(gamma / 4.0) ** 2))
    log_fact = gammaln(np.arange(n + 1) + 1)
    total = 0.0 + 0.0j
    s2 = np.sin(gamma / 4.0) ** 2
    c01 = 1.0 - np.exp(-1j * gamma / 2.0)
    c12 = 1.0 - np.exp(1j * gamma / 2.0)
    c02 = 4.0 * s2
    cos2 = np.cos(beta / 2.0) ** 2
    sin2 = np.sin(beta / 2.0) ** 2
    log_cos2 = np.log(cos2) if cos2 > 0 else -np.inf
    log_sin2 = np.log(sin2) if sin2 > 0 else -np.inf
    log_isin = np.log(1j * np.sin(beta / 2.0))
    log_misin = np.log(-1j * np.sin(beta / 2.0))
    for na in range(n + 1):
        for nb in range(n - na + 1):
            nc_max = n - na - nb
            nc_arr = np.arange(nc_max + 1)
            nd_arr = nc_max - nc_arr
            log_multi = log_fact[n] - log_fact[na] - log_fact[nb] - log_fact[nc_arr] - log_fact[nd_arr]
            log_base = na * log_cos2 + nb * log_sin2 + nc_arr * log_isin + nd_arr * log_misin
            x_a = na / (2.0 * n)
            x_ab = (na + nb) / (2.0 * n)
            x_ac = (na + nc_arr) / (2.0 * n)
            x_ad = (na + nd_arr) / (2.0 * n)
            exp_arg = r * n * (c02 * (x_ab**k - x_a**k) + c01 * x_ac**k + c12 * x_ad**k)
            total += np.sum(np.exp(log_multi + log_base + exp_arg))
    return float((prefactor * total).real)


def scan_ksat_p1_landscape(k: int, r: float, n_eval: int = 20, n_beta: int = 100, n_gamma: int = 100) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if k == 2:
        beta_range, gamma_range = (-np.pi, np.pi), (-2 * np.pi, 2 * np.pi)
    elif k in (4, 8):
        beta_range, gamma_range = (-np.pi / 2, np.pi / 2), (-np.pi, np.pi)
    elif k == 16:
        beta_range, gamma_range = (-np.pi / 4, np.pi / 4), (-np.pi / 2, np.pi / 2)
    else:
        beta_range, gamma_range = (-np.pi, np.pi), (-2 * np.pi, 2 * np.pi)
    betas = np.linspace(*beta_range, n_beta)
    gammas = np.linspace(*gamma_range, n_gamma)
    landscape = np.zeros((n_beta, n_gamma), dtype=float)
    for i, b in enumerate(betas):
        for j, g in enumerate(gammas):
            ps = max(ksat_p1_psucc_vectorized(n_eval, k, r, float(b), float(g)), 1e-300)
            landscape[i, j] = np.log(ps) / n_eval
    return betas, gammas, landscape


def _bit_matrix(size: int, dim: int) -> np.ndarray:
    idx = np.arange(size, dtype=np.uint64)[:, None]
    shifts = np.arange(dim, dtype=np.uint64)[None, :]
    return ((idx >> shifts) & 1).astype(np.uint8)


def compute_bs(beta_vec: np.ndarray) -> np.ndarray:
    p = int(beta_vec.size)
    dim = 2 * p + 1
    bits = _bit_matrix(1 << dim, dim)
    sign = (-1.0) ** (bits[:, 0] ^ bits[:, p])
    prod_cos = np.ones(1 << dim, dtype=np.complex128)
    prod_sin = np.ones(1 << dim, dtype=np.complex128)
    for j in range(p):
        c = np.cos(beta_vec[j] / 2.0)
        s = 1j * np.sin(beta_vec[j] / 2.0)
        eq = (1 - (bits[:, j] ^ bits[:, j + 1])) + (1 - (bits[:, 2 * p - j] ^ bits[:, 2 * p - j - 1]))
        neq = (bits[:, j] ^ bits[:, j + 1]) + (bits[:, 2 * p - j] ^ bits[:, 2 * p - j - 1])
        prod_cos *= c**eq
        prod_sin *= s**neq
    return (sign * prod_cos * prod_sin) / 2.0


def compute_c_alpha(gamma_vec: np.ndarray, r: float, include_small_subsets: bool = True) -> np.ndarray:
    p = int(gamma_vec.size)
    dim = 2 * p + 1
    bits = _bit_matrix(1 << dim, dim)
    # PRX Eq. (18) with r absorbed: c_alpha = r * c_phase. Linear coeff in F uses (-c_phase)^{1/2^q} (see compute_linear_coeff_prx).
    c = r * ((-1.0) ** bits[:, p]).astype(np.complex128)
    for j in range(dim):
        mask = bits[:, j] == 1
        if j < p:
            c[mask] *= np.exp(-1j * gamma_vec[j] / 2.0) - 1.0
        elif j > p:
            c[mask] *= np.exp(1j * gamma_vec[2 * p - j] / 2.0) - 1.0
    if not include_small_subsets:
        c[bits.sum(axis=1) < 2] = 0.0
    return c


def compute_linear_coeff_prx(c_alpha: np.ndarray, r: float, q: int) -> np.ndarray:
    """
    PRX Eq. (16): exponent uses (-c_α)^{1/2^q} z_α with (r/2) in front of the inner sum over α.

    In this module `compute_c_alpha(...)` already includes the clause-ratio factor in c_α.
    To stay consistent with the notebook implementation and avoid double-counting r, we use
    coeff_α = (-c_α)^{1/2^q}.
    """
    if q < 1:
        raise ValueError("q must be >= 1")
    k_clause = int(2**q)
    _ = r  # kept for API compatibility
    return np.power(-np.asarray(c_alpha, dtype=np.complex128) + 0.0j, 1.0 / float(k_clause))


# Backwards-compatible alias (older name)
def compute_linear_coeff_bm24(c_alpha: np.ndarray, r: float, q: int) -> np.ndarray:
    return compute_linear_coeff_prx(c_alpha, r, q)


def _active_indices_subset_ge2(d: int) -> np.ndarray:
    return np.array([a for a in range(d) if bin(a).count("1") >= 2], dtype=np.int64)


def _all_alpha_indices(d: int) -> np.ndarray:
    """PRX Eqs. (15),(16),(20) are defined over all alpha in [2p+1]."""
    return np.arange(d, dtype=np.int64)


def _compute_weights_from_u(
    u_active: np.ndarray,
    active_idx: np.ndarray,
    A: np.ndarray,
    log_b: np.ndarray,
    d: int,
) -> np.ndarray:
    """Softmax weights from u-space: w_s ∝ b_s exp(∑_α u_α A_{αs})."""
    u_full = np.zeros(d, dtype=np.complex128)
    u_full[active_idx] = u_active
    linear = u_full @ A
    log_w = log_b + linear
    log_w = np.where(np.isfinite(log_w), log_w, -700.0)
    shift = np.max(log_w.real)
    w = np.exp(log_w - shift)
    w = w / np.sum(w)
    if (not np.all(np.isfinite(w))) or (np.abs(np.sum(w) - 1.0) > 1e-6):
        raise FloatingPointError("Non-finite/unnormalized softmax weights in PRX Newton solve.")
    return w


def newton_solve_u_prx(
    u_init: np.ndarray,
    coeff_k_active: np.ndarray,
    active_idx: np.ndarray,
    A: np.ndarray,
    log_b: np.ndarray,
    d: int,
    k_clause: int,
    tol: float = 1e-10,
    max_iter: int = 200,
) -> Tuple[np.ndarray, bool, float, int]:
    """
    Newton for g(u) = u + k · (coeff^k) · E^{k-1} = 0 with k = 2^q (PRX Eq. (20) in u-form).

    ``coeff_k_active`` must equal ``coeff[active_idx] ** k_clause`` (coeff from ``compute_linear_coeff_prx``).
    This matches PRX powers 2^q, not the separate literal ``2·q`` used in some auxiliary notes.
    """
    exponent = k_clause - 1
    prefactor = float(k_clause)
    deriv_factor = float(k_clause * exponent)

    A_act = A[active_idx]
    n_act = int(len(active_idx))

    u = np.asarray(u_init, dtype=np.complex128).copy()
    if u.shape != (n_act,):
        raise ValueError("u_init has wrong shape")

    best_u = u.copy()
    best_res = float("inf")
    stall_count = 0

    for it in range(max_iter):
        try:
            w = _compute_weights_from_u(u, active_idx, A, log_b, d)
        except FloatingPointError:
            return best_u, False, best_res, it
        E_A = w @ A_act.T

        g = u + prefactor * coeff_k_active * np.power(E_A, exponent)
        res = float(np.max(np.abs(g)))

        if res < best_res:
            best_res = res
            best_u = u.copy()
            stall_count = 0
        else:
            stall_count += 1

        if res < tol:
            return u, True, res, it
        if stall_count > 30:
            return best_u, best_res < tol, best_res, it

        E_AA = A_act @ (w[:, None] * A_act.T)
        Cov = E_AA - np.outer(E_A, E_A)
        D = deriv_factor * coeff_k_active * np.power(E_A, exponent - 1)
        J = np.eye(n_act, dtype=np.complex128) + D[:, None] * Cov

        try:
            cond = float(np.linalg.cond(J))
        except Exception:
            cond = float("inf")
        if cond > 1e13:
            J = J + (1e-8 * np.eye(n_act, dtype=np.complex128))

        try:
            delta = np.linalg.solve(J, -g)
        except np.linalg.LinAlgError:
            return best_u, best_res < tol, best_res, it

        step = 1.0
        for _ in range(50):
            u_trial = u + step * delta
            try:
                w_t = _compute_weights_from_u(u_trial, active_idx, A, log_b, d)
            except FloatingPointError:
                step *= 0.5
                continue
            E_t = w_t @ A_act.T
            g_t = u_trial + prefactor * coeff_k_active * np.power(E_t, exponent)
            res_t = float(np.max(np.abs(g_t)))
            if np.isfinite(res_t) and res_t < res:
                break
            step *= 0.5

        u = u + step * delta

    return best_u, best_res < tol, best_res, max_iter


def newton_solve_u_artificial_homotopy(
    u_init: np.ndarray,
    u_anchor: np.ndarray,
    coeff_k_active: np.ndarray,
    active_idx: np.ndarray,
    A: np.ndarray,
    log_b: np.ndarray,
    d: int,
    k_clause: int,
    t_blend: float,
    tol: float = 1e-10,
    max_iter: int = 200,
) -> Tuple[np.ndarray, bool, float, int]:
    """
    Newton for artificial homotopy residual:
        R_t(u) = u - (1-t) u_anchor + t * k * coeff^k * E(u)^(k-1) = 0.
    At t=0, u=u_anchor is an exact root by construction.
    At t=1, this recovers the physical PRX residual g(u)=0.
    """
    exponent = k_clause - 1
    prefactor = float(k_clause)
    deriv_factor = float(k_clause * exponent)
    t_blend = float(t_blend)
    if t_blend < 0.0 or t_blend > 1.0:
        raise ValueError("t_blend must be in [0, 1]")

    A_act = A[active_idx]
    n_act = int(len(active_idx))
    u = np.asarray(u_init, dtype=np.complex128).copy()
    if u.shape != (n_act,):
        raise ValueError("u_init has wrong shape")
    u_anchor = np.asarray(u_anchor, dtype=np.complex128)
    if u_anchor.shape != (n_act,):
        raise ValueError("u_anchor has wrong shape")

    best_u = u.copy()
    best_res = float("inf")
    stall_count = 0

    for it in range(max_iter):
        try:
            w = _compute_weights_from_u(u, active_idx, A, log_b, d)
        except FloatingPointError:
            return best_u, False, best_res, it
        E_A = w @ A_act.T

        R = u - (1.0 - t_blend) * u_anchor + t_blend * prefactor * coeff_k_active * np.power(E_A, exponent)
        res = float(np.max(np.abs(R)))
        if res < best_res:
            best_res = res
            best_u = u.copy()
            stall_count = 0
        else:
            stall_count += 1
        if res < tol:
            return u, True, res, it
        if stall_count > 30:
            return best_u, best_res < tol, best_res, it

        E_AA = A_act @ (w[:, None] * A_act.T)
        Cov = E_AA - np.outer(E_A, E_A)
        D = t_blend * deriv_factor * coeff_k_active * np.power(E_A, exponent - 1)
        J = np.eye(n_act, dtype=np.complex128) + D[:, None] * Cov

        try:
            cond = float(np.linalg.cond(J))
        except Exception:
            cond = float("inf")
        if cond > 1e13:
            J = J + (1e-8 * np.eye(n_act, dtype=np.complex128))

        try:
            delta = np.linalg.solve(J, -R)
        except np.linalg.LinAlgError:
            return best_u, best_res < tol, best_res, it

        step = 1.0
        for _ in range(50):
            u_trial = u + step * delta
            try:
                w_t = _compute_weights_from_u(u_trial, active_idx, A, log_b, d)
            except FloatingPointError:
                step *= 0.5
                continue
            E_t = w_t @ A_act.T
            R_t = u_trial - (1.0 - t_blend) * u_anchor + t_blend * prefactor * coeff_k_active * np.power(E_t, exponent)
            res_t = float(np.max(np.abs(R_t)))
            if np.isfinite(res_t) and res_t < res:
                break
            step *= 0.5
        u = u + step * delta

    return best_u, best_res < tol, best_res, max_iter


def solve_bm24_saddle_t_homotopy(
    p: int,
    betas: np.ndarray,
    gammas_target: np.ndarray,
    k: int,
    r: float,
    include_small_subsets: bool,
    t_start: float = 1e-6,
    t_factor: float = 1.1,
    t_max_steps: int = 800,
    newton_tol: float = 1e-12,
    newton_max_iter: int = 800,
    # PRX Newton with k=2^q is stiffer than the old 2·q auxiliary map; looser tol helps t→1 complete.
    target_residual: float = 5e-3,
    max_newton_calls: int = 5000,
    use_artificial_anchor_homotopy: bool = True,
    anchor_magnitude: float = 2.0,
    enforce_physical_branch_guard: bool = True,
    max_positive_exponent: float = 1e-10,
    max_exponent_increase: float = 0.2,
) -> Tuple[np.ndarray, bool, float]:
    """
    BM24 Eq. (20) via Newton in u-space (u_α = coeff_α y_α) + scalar t-homotopy on γ.

    Uses the same b_s as `compute_bs` (half-angle QAOA convention), matching
    `ksat_p1_psucc_vectorized` / finite-n crosschecks.  Symmetry-reduction's
    `compute_b_s` uses a different β convention and must not be mixed here.
    Solves over all alpha indices, consistent with PRX Eqs. (15),(16),(20).
    Homotopy uses adaptive step control: failed Newton attempt => halve t-step and retry.
    If `use_artificial_anchor_homotopy=True`, gamma is fixed to target and we blend from a
    large symmetry-broken anchor at t=0 to the physical PRX equations at t=1.
    """
    from symmetry_reduction.core import build_structure_matrix

    q = int(np.log2(k))
    if 2**q != k:
        raise ValueError(f"k must be a power of 2, got {k}")
    k_clause = int(k)

    p = int(p)
    betas = np.asarray(betas, dtype=float)
    gammas_target = np.asarray(gammas_target, dtype=float)
    if betas.shape != (p,) or gammas_target.shape != (p,):
        raise ValueError("betas and gammas_target must have shape (p,)")

    A = build_structure_matrix(p, q=q)
    bs = compute_bs(betas)
    d = int(A.shape[0])
    active_idx = _all_alpha_indices(d)
    n_act = int(len(active_idx))
    log_b = np.log(bs + 1e-300 * (1 - np.sign(np.abs(bs))))
    log_b = np.where(np.isfinite(log_b), log_b, -700.0)

    c_target = compute_c_alpha(gammas_target, r=r, include_small_subsets=include_small_subsets)
    coeff_final = compute_linear_coeff_prx(c_target, r, q)
    if use_artificial_anchor_homotopy:
        # Large, symmetry-broken anchor as requested: slight magnitude/phase offsets.
        z_anchor = np.zeros(n_act, dtype=np.complex128)
        for i in range(n_act):
            mag = float(anchor_magnitude) * (1.0 + 0.1 * (i % 10))
            phase = 0.05 * i
            z_anchor[i] = mag * np.exp(1j * phase)
        coeff_act_final = coeff_final[active_idx]
        u = coeff_act_final * z_anchor
    else:
        u = np.zeros(n_act, dtype=np.complex128)
    t_cur = 0.0
    t_step = float(t_start)
    if t_step <= 0.0 or t_step > 1.0:
        raise ValueError("t_start must be in (0, 1] and is interpreted as initial step size")
    final_res = 0.0
    steps = 0
    newton_calls = 0
    prev_accepted_full_exponent: Optional[float] = None

    while t_cur < 1.0 - 1e-14 and steps < int(t_max_steps):
        if newton_calls >= int(max_newton_calls):
            break

        t_try = min(1.0, t_cur + t_step)
        if use_artificial_anchor_homotopy:
            coeff_k_active = np.power(coeff_final[active_idx], k_clause)
            u_new, conv, res, _nit = newton_solve_u_artificial_homotopy(
                u_init=u,
                u_anchor=(coeff_final[active_idx] * z_anchor),
                coeff_k_active=coeff_k_active,
                active_idx=active_idx,
                A=A,
                log_b=log_b,
                d=d,
                k_clause=k_clause,
                t_blend=t_try,
                tol=newton_tol,
                max_iter=newton_max_iter,
            )
        else:
            gammas_step = t_try * gammas_target
            c_alpha = compute_c_alpha(gammas_step, r=r, include_small_subsets=include_small_subsets)
            coeff_step = compute_linear_coeff_prx(c_alpha, r, q)
            coeff_k_active = np.power(coeff_step[active_idx], k_clause)
            u_new, conv, res, _nit = newton_solve_u_prx(
                u_init=u,
                coeff_k_active=coeff_k_active,
                active_idx=active_idx,
                A=A,
                log_b=log_b,
                d=d,
                k_clause=k_clause,
                tol=newton_tol,
                max_iter=newton_max_iter,
            )
        newton_calls += 1
        steps += 1
        final_res = float(res)

        if bool(conv) or (np.isfinite(final_res) and final_res <= float(target_residual)):
            branch_ok = True
            if enforce_physical_branch_guard:
                coeff_eval = coeff_final if use_artificial_anchor_homotopy else coeff_step
                c_eval = c_target if use_artificial_anchor_homotopy else c_alpha
                gamma_eval = gammas_target if use_artificial_anchor_homotopy else gammas_step
                z_trial = np.zeros(d, dtype=np.complex128)
                coeff_act_eval = coeff_eval[active_idx]
                mask = np.abs(coeff_act_eval) > 1e-14
                z_trial[active_idx[mask]] = u_new[mask] / coeff_act_eval[mask]
                try:
                    trial_sp = scaling_exponent_ksat(
                        z_trial,
                        bs,
                        c_eval,
                        k=k_clause,
                        p=p,
                        r=r,
                        gamma_vec=gamma_eval,
                    )
                    trial_full = float(trial_sp["full_exponent_real"])
                except Exception:
                    trial_full = float("inf")
                if not np.isfinite(trial_full):
                    branch_ok = False
                elif trial_full > float(max_positive_exponent):
                    branch_ok = False
                elif prev_accepted_full_exponent is not None and trial_full > prev_accepted_full_exponent + float(max_exponent_increase):
                    branch_ok = False
                if branch_ok:
                    prev_accepted_full_exponent = trial_full

            if branch_ok:
                # Accept and grow next step.
                u = u_new
                t_cur = t_try
                remaining = max(1.0 - t_cur, 0.0)
                if remaining > 0.0:
                    t_step = min(max(t_step * float(t_factor), 1e-12), remaining)
            else:
                # Reject branch-jumping step and retry with smaller step.
                t_step *= 0.5
                if t_step < 1e-12:
                    break
        else:
            # Reject and retry from same t_cur with smaller step.
            t_step *= 0.5
            if t_step < 1e-12:
                break

        if newton_calls >= int(max_newton_calls):
            break

    c_final = compute_c_alpha(gammas_target, r=r, include_small_subsets=include_small_subsets)
    coeff_final = compute_linear_coeff_prx(c_final, r, q)
    y_star = np.zeros(d, dtype=np.complex128)
    coeff_act = coeff_final[active_idx]
    for i in range(n_act):
        if abs(coeff_act[i]) > 1e-14:
            y_star[active_idx[i]] = u[i] / coeff_act[i]

    converged = bool(t_cur >= 1.0 - 1e-14)
    return y_star, converged, final_res


def solve_bm24_saddle_exact_ansatz_p1(
    beta_vec: np.ndarray,
    gamma_vec_target: np.ndarray,
    k: int,
    r: float,
    include_small_subsets: bool,
    seed: int = 1234,
) -> Tuple[np.ndarray, bool, float]:
    """
    Direct p=1 LM solve from a symmetry-broken ansatz.

    This targets the non-paramagnetic basin directly and avoids continuation from gamma=0.
    Returns (z_star, converged, residual_inf).
    """
    if root is None:
        raise RuntimeError("SciPy root solver is required for exact-ansatz p=1 solve.")
    p = int(len(gamma_vec_target))
    if p != 1:
        raise ValueError("solve_bm24_saddle_exact_ansatz_p1 requires p=1")
    q = int(np.log2(k))
    if 2**q != k:
        raise ValueError(f"k must be a power of 2, got {k}")

    from symmetry_reduction.core import build_structure_matrix

    A = build_structure_matrix(p, q=q)
    d = int(A.shape[0])
    idx = _all_alpha_indices(d)
    bs = compute_bs(np.asarray(beta_vec, dtype=float))
    log_b = np.log(bs + 1e-300 * (1 - np.sign(np.abs(bs))))
    log_b = np.where(np.isfinite(log_b), log_b, -700.0)
    c_target = compute_c_alpha(np.asarray(gamma_vec_target, dtype=float), r=r, include_small_subsets=include_small_subsets)
    coeff = compute_linear_coeff_prx(c_target, r, q)
    coeff_k = np.power(coeff[idx], int(k))

    rng = np.random.default_rng(seed)

    def _residual_real(x: np.ndarray, violating_index: int) -> np.ndarray:
        u = x[:d] + 1j * x[d:]
        try:
            w = _compute_weights_from_u(u, idx, A, log_b, d)
            E = w @ A[idx].T
            u_next = -float(k) * coeff_k * np.power(E, int(k) - 1)
            diff = u - u_next
        except Exception:
            diff = np.full(d, 1e6 + 0.0j, dtype=np.complex128)
        return np.concatenate([np.real(diff), np.imag(diff)])

    best_z: Optional[np.ndarray] = None
    best_res = float("inf")
    best_ratio = 0.0
    best_full_exp = float("inf")
    # Try both common violating-index conventions.
    for violating_index in (0, d - 1):
        z0 = np.full(d, 1.5 + 0.0j, dtype=np.complex128)
        z0[violating_index] = 15.0 * np.exp(1j * 0.8)
        noise = rng.uniform(-0.01, 0.01, d) + 1j * rng.uniform(-0.01, 0.01, d)
        z0 = z0 + noise
        u0 = coeff * z0
        x0 = np.concatenate([np.real(u0), np.imag(u0)])

        sol = root(lambda x: _residual_real(x, violating_index), x0, method="lm", options={"ftol": 1e-12, "maxiter": 5000})
        if not sol.success:
            continue
        u_fin = sol.x[:d] + 1j * sol.x[d:]
        try:
            w = _compute_weights_from_u(u_fin, idx, A, log_b, d)
            E = w @ A[idx].T
            u_next = -float(k) * coeff_k * np.power(E, int(k) - 1)
            res = float(np.max(np.abs(u_fin - u_next)))
        except Exception:
            continue
        z_fin = np.zeros(d, dtype=np.complex128)
        mask = np.abs(coeff) > 1e-14
        z_fin[mask] = u_fin[mask] / coeff[mask]
        mags = np.abs(u_fin)
        ratio = float(np.max(mags) / (np.min(mags) + 1e-12))
        # Physical screening: prefer negative full exponent roots.
        try:
            sp = scaling_exponent_ksat(
                z_fin,
                bs,
                c_target,
                k=k,
                p=1,
                r=r,
                gamma_vec=np.asarray(gamma_vec_target, dtype=float),
            )
            full_exp = float(sp["full_exponent_real"])
        except Exception:
            full_exp = float("inf")
        score = (0 if (np.isfinite(full_exp) and full_exp <= 1e-10) else 1, abs(full_exp), res, -ratio)
        best_score = (0 if (np.isfinite(best_full_exp) and best_full_exp <= 1e-10) else 1, abs(best_full_exp), best_res, -best_ratio)
        if score < best_score:
            best_res = res
            best_ratio = ratio
            best_full_exp = full_exp
            best_z = z_fin

    if best_z is None:
        return np.zeros(d, dtype=np.complex128), False, float("inf")
    # Reject collapse to near-uniform paramagnetic root.
    if best_ratio < 1.1:
        return best_z, False, best_res
    # Reject roots that remain on an unphysical positive-exponent branch.
    if not np.isfinite(best_full_exp) or best_full_exp > 1e-10:
        return best_z, False, best_res
    return best_z, bool(best_res <= 1e-8), best_res


@dataclass
class FixedPointConfig:
    k: int
    p: int
    r: float = 176.54
    max_iter: int = 800
    tol: float = 1e-10
    damping: float = 0.35


@dataclass
class PRXMode:
    exp_sign: int = +1
    A_half_in_sum: bool = True
    include_small_subsets: bool = True


def find_fixed_point(
    bs: np.ndarray,
    c_alpha: np.ndarray,
    cfg: FixedPointConfig,
    z_init: Optional[np.ndarray] = None,
    mode: Optional[PRXMode] = None,
) -> np.ndarray:
    mode = mode or PRXMode()
    dim = 2 * cfg.p + 1
    q = int(np.log2(cfg.k))
    if 2**q != cfg.k:
        raise ValueError(f"k must be a power of 2, got {cfg.k}.")
    r = float(cfg.r)
    z = np.zeros(1 << dim, dtype=np.complex128) if z_init is None else z_init.astype(np.complex128).copy()
    # PRX Eq. (20): z_α = -2^q (∂_α F)^{2^q - 1}; linear coeff r·(-c_phase)^{1/2^q} (see compute_linear_coeff_prx).
    coeff = compute_linear_coeff_prx(c_alpha, r, q)
    k_clause = int(cfg.k)
    for _ in range(cfg.max_iter):
        x = sum_over_alpha(coeff * z, dim, apply_half=mode.A_half_in_sum)
        sigma = float(mode.exp_sign)
        shift = np.max((sigma * x).real)
        weighted = bs * np.exp(sigma * x - shift)
        denom = np.sum(weighted)
        if abs(denom) < 1e-15:
            break
        grad = sigma * coeff * (sum_over_s(weighted, dim, apply_half=mode.A_half_in_sum) / denom)
        z_new = -float(k_clause) * (grad ** (k_clause - 1))
        z_next = cfg.damping * z + (1.0 - cfg.damping) * z_new
        if np.max(np.abs(z_next - z)) < cfg.tol:
            z = z_next
            break
        z = z_next
    return z


def find_fixed_point_continuation(
    beta_vec: np.ndarray,
    gamma_vec_target: np.ndarray,
    r: float,
    cfg: FixedPointConfig,
    n_steps: int = 80,
    mode: Optional[PRXMode] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    BM24 saddle: Newton + t-homotopy on γ (same pattern as symmetry_reduction/newton_saddle_q).

    Damped fixed-point iteration in z alone is not reliable at q>1 and large r (spurious small-z
    fixed points or blow-up); u-space Newton matches the PRX saddle used for certification.
    """
    mode = mode or PRXMode()
    bs = compute_bs(beta_vec)
    q = int(np.log2(cfg.k))
    if 2**q != cfg.k:
        raise ValueError(f"k must be a power of 2, got {cfg.k}.")
    # For p=1, use direct symmetry-broken LM solve first (closest to BM24 ansatz workflow).
    if int(cfg.p) == 1:
        z_exact, ok_exact, _res_exact = solve_bm24_saddle_exact_ansatz_p1(
            beta_vec,
            gamma_vec_target,
            cfg.k,
            r,
            include_small_subsets=mode.include_small_subsets,
        )
        if ok_exact:
            c_final = compute_c_alpha(gamma_vec_target, r=r, include_small_subsets=mode.include_small_subsets)
            return z_exact, bs, c_final
    # Small t_start avoids jumping to γ where Newton(u; k=2^q) rejects every step (returns z=0).
    t_start = 1e-6
    z, converged, _res = solve_bm24_saddle_t_homotopy(
        cfg.p,
        beta_vec,
        gamma_vec_target,
        cfg.k,
        r,
        include_small_subsets=mode.include_small_subsets,
        t_start=t_start,
        t_factor=1.1,
        t_max_steps=max(800, n_steps * 12),
    )
    if not converged:
        raise RuntimeError("Physical branch lost before t=1; refusing toxic fallback.")
    c_final = compute_c_alpha(gamma_vec_target, r=r, include_small_subsets=mode.include_small_subsets)
    return z, bs, c_final


def scaling_exponent_ksat(
    z_star: np.ndarray,
    bs: np.ndarray,
    c_alpha: np.ndarray,
    k: int,
    p: int,
    r: float,
    gamma_vec: Optional[np.ndarray] = None,
    mode: Optional[PRXMode] = None,
) -> Dict[str, float]:
    """
    PRX Eq. (15): F(z*) + (2^q - 1) sum_α (∂_α F)^{2^q} with k = 2^q clause arity.

    Use ``include_small_subsets=True`` unless you know the masked-c formulation matches
    the correction sum; with ``False``, fewer α participate in complex cancellation and
    the correction term can blow up.
    """
    mode = mode or PRXMode()
    dim = 2 * p + 1
    q = int(np.log2(k))
    if 2**q != k:
        raise ValueError(f"k must be a power of 2, got {k}.")
    k_clause = int(k)
    coeff = compute_linear_coeff_prx(c_alpha, float(r), q)
    sigma = float(mode.exp_sign)
    x = sum_over_alpha(coeff * z_star, dim, apply_half=mode.A_half_in_sum)
    shift = np.max((sigma * x).real)
    weighted = bs * np.exp(sigma * x - shift)
    denom = np.sum(weighted)
    if abs(denom) < 1e-15:
        return {"raw": -np.inf, "excess": -np.inf}
    grad = sigma * coeff * (sum_over_s(weighted, dim, apply_half=mode.A_half_in_sum) / denom)
    F_term = np.log(denom) + shift
    # Eq. (15): (2^q - 1) Σ (∂F)^{2^q}, not (2·q - 1) Σ (∂F)^{2·q}.
    correction_term = (k_clause - 1) * np.sum(grad**k_clause)
    raw = float(np.real(F_term + correction_term))
    baseline = -(2.0 ** (-k)) * r
    gamma_vec = np.zeros(p, dtype=float) if gamma_vec is None else np.asarray(gamma_vec, dtype=float)
    prefactor_real = float(-(2.0 ** (-k)) * r * (1.0 + 4.0 * np.sum(np.sin(gamma_vec / 4.0) ** 2)))
    full_exponent_real = float(prefactor_real + np.real(F_term) + np.real(correction_term))
    root_arg = np.angle(coeff)
    return {
        "raw": raw,
        "excess": float(raw - baseline),
        "full_exponent_real": full_exponent_real,
        "prefactor_real": prefactor_real,
        "F_term_real": float(np.real(F_term)),
        "correction_real": float(np.real(correction_term)),
        "max_grad_abs": float(np.max(np.abs(grad))),
        "max_root_arg_abs": float(np.max(np.abs(root_arg))),
        "num_root_not_finite": int(np.sum(~np.isfinite(coeff))),
        "q_log2_k": float(q),
        "k_clause": float(k_clause),
    }


def optimize_toy() -> Dict[str, float]:
    return optimize_toy_model(TOY_BETA_RANGE, TOY_GAMMA_TILDE_RANGE)


def optimize_ksat_p1(k: int, r: float, n_eval: int = 30, seed: int = 42) -> Dict[str, float]:
    _require_scipy()
    if differential_evolution is None:
        raise RuntimeError("SciPy optimize is required.")
    if k == 8:
        beta_bounds, gamma_bounds = (-np.pi, 0.0), (0.0, 2.0 * np.pi)
    elif k == 2:
        beta_bounds, gamma_bounds = (-np.pi, np.pi), (-2 * np.pi, 2 * np.pi)
    elif k == 4:
        beta_bounds, gamma_bounds = (-np.pi / 2, np.pi / 2), (-np.pi, np.pi)
    else:
        beta_bounds, gamma_bounds = (-np.pi, np.pi), (-2 * np.pi, 2 * np.pi)

    def neg_obj(x: np.ndarray) -> float:
        return -np.log(max(ksat_p1_psucc_vectorized(n_eval, k, r, float(x[0]), float(x[1])), 1e-300)) / n_eval

    de = differential_evolution(neg_obj, bounds=[beta_bounds, gamma_bounds], seed=seed, maxiter=80, tol=1e-6, polish=False)
    nm = minimize(neg_obj, de.x, method="Nelder-Mead", options={"xatol": 1e-8, "fatol": 1e-10, "maxiter": 1000})
    return {"beta": float(nm.x[0]), "gamma": float(nm.x[1]), "exponent": float(-nm.fun), "success": bool(nm.success), "message": str(nm.message)}


def optimize_ksat_general(k: int, p: int, r: float | None = None, seed: int = 42) -> Dict[str, object]:
    mode = PRXMode()
    _require_scipy()
    if differential_evolution is None:
        raise RuntimeError("SciPy optimize is required.")
    r = THRESHOLDS[k] if r is None else r
    cfg = FixedPointConfig(k=k, p=p, r=r, max_iter=700, tol=1e-10, damping=0.35)

    def neg_raw(params: np.ndarray) -> float:
        try:
            beta = np.asarray(params[:p], dtype=float)
            gamma = np.asarray(params[p:], dtype=float)
            # More continuation steps for larger gamma targets improves robustness.
            gamma_scale = float(np.max(np.abs(gamma))) if gamma.size else 0.0
            n_steps = int(np.clip(80 + 80 * gamma_scale / np.pi, 80, 260))
            z, bs, c = find_fixed_point_continuation(beta, gamma, r=r, cfg=cfg, n_steps=n_steps, mode=mode)
            raw = scaling_exponent_ksat(z, bs, c, k=k, p=p, r=r, gamma_vec=gamma, mode=mode)["full_exponent_real"]
            if not np.isfinite(raw):
                return 1e10
            # Numerical sanity guard: reject clearly non-physical/exploded saddles.
            if raw > 0.1 or raw < -5.0:
                return 1e9 + abs(raw)
            return -raw
        except Exception:
            return 1e10

    bounds = [(-np.pi / 2, np.pi / 2)] * p + [(-np.pi, np.pi)] * p

    de = differential_evolution(
        neg_raw,
        bounds=bounds,
        seed=seed,
        maxiter=60,
        popsize=10,
        tol=1e-6,
        polish=False,
    )
    res = minimize(
        neg_raw,
        x0=np.asarray(de.x, dtype=float),
        method="Nelder-Mead",
        options={"maxiter": 2500, "xatol": 1e-8, "fatol": 1e-10},
    )
    return {
        "beta": res.x[:p].tolist(),
        "gamma": res.x[p:].tolist(),
        "raw_exponent": float(-res.fun),
        "success": bool(res.success),
        "message": str(res.message),
    }


def diagnostic_p1_crosscheck(k: int = 8, r: float = 176.54, n_eval: int = 30) -> Dict[str, float]:
    """
    Compare p=1 exact finite-n exponent to saddle-point prediction.
    """
    mode = PRXMode()
    p1 = optimize_ksat_p1(k, r, n_eval=n_eval)
    beta_opt = float(p1["beta"])
    gamma_opt = float(p1["gamma"])
    exact_exp = float(p1["exponent"])

    beta_vec = np.array([beta_opt], dtype=float)
    gamma_vec = np.array([gamma_opt], dtype=float)
    cfg = FixedPointConfig(k=k, p=1, r=r, max_iter=700, tol=1e-10, damping=0.35)
    try:
        z, bs, c = find_fixed_point_continuation(beta_vec, gamma_vec, r=r, cfg=cfg, n_steps=100, mode=mode)
        sp = scaling_exponent_ksat(z, bs, c, k=k, p=1, r=r, gamma_vec=gamma_vec, mode=mode)
    except RuntimeError:
        return {
            "beta": beta_opt,
            "gamma": gamma_opt,
            "exact_exponent_finite_n": exact_exp,
            "saddle_exponent_n_inf": float("nan"),
            "abs_difference": float("inf"),
            "relative_difference": float("inf"),
            "saddle_prefactor_real": float("nan"),
            "saddle_F_term_real": float("nan"),
            "saddle_correction_real": float("nan"),
            "saddle_max_grad_abs": float("nan"),
            "saddle_max_root_arg_abs": float("nan"),
            "saddle_num_root_not_finite": float("nan"),
        }
    return {
        "beta": beta_opt,
        "gamma": gamma_opt,
        "exact_exponent_finite_n": exact_exp,
        "saddle_exponent_n_inf": float(sp["full_exponent_real"]),
        "abs_difference": float(abs(exact_exp - sp["full_exponent_real"])),
        "relative_difference": float(abs(exact_exp - sp["full_exponent_real"]) / abs(exact_exp)),
        "saddle_prefactor_real": float(sp["prefactor_real"]),
        "saddle_F_term_real": float(sp["F_term_real"]),
        "saddle_correction_real": float(sp["correction_real"]),
        "saddle_max_grad_abs": float(sp["max_grad_abs"]),
        "saddle_max_root_arg_abs": float(sp["max_root_arg_abs"]),
        "saddle_num_root_not_finite": float(sp["num_root_not_finite"]),
    }


def diagnostic_p1_prx_mode_sweep(k: int = 8, r: float = 176.54, n_eval: int = 30) -> dict:
    p1 = optimize_ksat_p1(k, r, n_eval=n_eval)
    beta_opt = float(p1["beta"])
    gamma_opt = float(p1["gamma"])
    exact_exp = float(p1["exponent"])
    beta_vec = np.array([beta_opt], dtype=float)
    gamma_vec = np.array([gamma_opt], dtype=float)
    cfg = FixedPointConfig(k=k, p=1, r=r, max_iter=700, tol=1e-10, damping=0.35)
    modes = {
        "A": PRXMode(exp_sign=+1, A_half_in_sum=True, include_small_subsets=True),
        "B": PRXMode(exp_sign=+1, A_half_in_sum=True, include_small_subsets=False),
        "C": PRXMode(exp_sign=+1, A_half_in_sum=False, include_small_subsets=True),
        "D": PRXMode(exp_sign=-1, A_half_in_sum=True, include_small_subsets=True),
    }
    out = {"beta": beta_opt, "gamma": gamma_opt, "exact_exponent_finite_n": exact_exp, "modes": {}}
    for name, mode in modes.items():
        try:
            z, bs, c = find_fixed_point_continuation(beta_vec, gamma_vec, r=r, cfg=cfg, n_steps=100, mode=mode)
            sp = scaling_exponent_ksat(z, bs, c, k=k, p=1, r=r, gamma_vec=gamma_vec, mode=mode)
            out["modes"][name] = {
                "mode": {
                    "exp_sign": mode.exp_sign,
                    "A_half_in_sum": mode.A_half_in_sum,
                    "include_small_subsets": mode.include_small_subsets,
                },
                "full_exponent_real": float(sp["full_exponent_real"]),
                "prefactor_real": float(sp["prefactor_real"]),
                "F_term_real": float(sp["F_term_real"]),
                "correction_real": float(sp["correction_real"]),
                "abs_difference": float(abs(exact_exp - sp["full_exponent_real"])),
                "relative_difference": float(abs(exact_exp - sp["full_exponent_real"]) / abs(exact_exp)),
                "max_grad_abs": float(sp["max_grad_abs"]),
                "max_root_arg_abs": float(sp["max_root_arg_abs"]),
                "num_root_not_finite": int(sp["num_root_not_finite"]),
            }
        except RuntimeError:
            out["modes"][name] = {
                "mode": {
                    "exp_sign": mode.exp_sign,
                    "A_half_in_sum": mode.A_half_in_sum,
                    "include_small_subsets": mode.include_small_subsets,
                },
                "full_exponent_real": float("nan"),
                "prefactor_real": float("nan"),
                "F_term_real": float("nan"),
                "correction_real": float("nan"),
                "abs_difference": float("inf"),
                "relative_difference": float("inf"),
                "max_grad_abs": float("nan"),
                "max_root_arg_abs": float("nan"),
                "num_root_not_finite": int(0),
            }
    return out


def validate_toy_point(beta: float, gamma_tilde: float) -> dict:
    return validate_toy(beta, gamma_tilde)


def validate_ksat_p1_scaling(
    k: int,
    r: float,
    beta: float,
    gamma: float,
    n_values: Iterable[int] = (20, 30, 40, 50, 60, 70),
) -> dict:
    n_arr = np.array(list(n_values), dtype=float)
    y = np.array([np.log(max(ksat_p1_psucc_vectorized(int(n), k, r, beta, gamma), 1e-300)) for n in n_arr.astype(int)], dtype=float)
    slope, intercept = np.polyfit(n_arr, y, 1)
    return {"empirical_exponent": float(slope), "intercept": float(intercept), "n_values": n_arr.tolist(), "log_psucc": y.tolist()}
