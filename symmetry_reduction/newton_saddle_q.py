"""
Newton solver for the q>1 BM24 saddle-point equation (Eq. 20).

For q=3 and large r (e.g. r=176.54 for 8-SAT at threshold), the fixed-point map
    y <- -k (∂F/∂y)^{k-1},  k = 2^q,
has Lipschitz constant ~ r^{k-1} and is non-contractive in our parameter sweep.

We instead solve a root-finding problem in u-space, with u_α = coeff_α * y_α,
restricted to the non-phantom subset indices |α| >= 2.

Weights are defined by:
    w_s ∝ |b_s| * exp(∑_α u_α A_{αs}),
and the residual is (PRX Eq. (20) in u-form, k = 2^q):
    g_α(u) = u_α + k * coeff_α^{k} * (E_w[A_α])^{k-1}.

We solve via γ-homotopy: start at very small γ where the system is nearly linear,
then ramp γ geometrically to the target, warmstarting Newton at each step.
"""

from __future__ import annotations

import numpy as np

from .core import (
    alpha_linear_coefficients,
    bm24_clause_arity,
    build_structure_matrix,
    compute_b_s,
    popcount,
)


def _get_active_indices(d: int) -> np.ndarray:
    """Return indices α with |α| >= 2 (non-phantom subsets)."""
    sizes = np.fromiter((popcount(a) for a in range(d)), dtype=np.int64, count=d)
    return np.where(sizes >= 2)[0]


def active_indices_for_p(p: int) -> np.ndarray:
    """Active subset indices for depth p (|α| >= 2)."""
    d = 1 << (2 * int(p) + 1)
    return _get_active_indices(d)


def _compute_weights_from_u(
    u_active: np.ndarray,
    active_idx: np.ndarray,
    A: np.ndarray,
    log_b: np.ndarray,
    d: int,
) -> np.ndarray:
    """Softmax weights from u-space: w_s ∝ b_s exp(∑_α u_α A_{αs})."""
    u_full = np.zeros(d, dtype=complex)
    u_full[active_idx] = u_active
    linear = u_full @ A  # (d,)
    log_w = log_b + linear
    log_w = np.where(np.isfinite(log_w), log_w, -700.0)
    shift = np.max(log_w.real)
    w = np.exp(log_w - shift)
    w = w / np.sum(w)
    # For stability, ensure weights are finite and sum to 1
    if not np.all(np.isfinite(w)) or np.abs(np.sum(w) - 1.0) > 1e-6:
        w = np.ones(d, dtype=complex) / float(d)
    return w


def newton_solve_q(
    u_init: np.ndarray,
    c_2q_active: np.ndarray,
    active_idx: np.ndarray,
    A: np.ndarray,
    log_b: np.ndarray,
    d: int,
    q: int = 3,
    tol: float = 1e-10,
    max_iter: int = 200,
) -> tuple[np.ndarray, bool, float, int]:
    """
    Newton's method for g(u) = u + k · coeff^k · E_w[A]^{k-1} = 0 on active indices,
    with k = 2^q (BM24 Eq. (20) u-form).

    ``c_2q_active`` must equal ``coeff[active]^k`` (legacy parameter name).

    Returns (u_solution, converged, residual, n_iterations).
    """
    k = bm24_clause_arity(q)
    exponent = k - 1
    prefactor = float(k)
    deriv_factor = float(k * exponent)

    A_act = A[active_idx]  # (n_act, d)
    n_act = int(len(active_idx))

    u = np.asarray(u_init, dtype=complex).copy()
    if u.shape != (n_act,):
        raise ValueError("u_init has wrong shape")

    best_u = u.copy()
    best_res = float("inf")
    stall_count = 0

    for it in range(max_iter):
        w = _compute_weights_from_u(u, active_idx, A, log_b, d)
        E_A = w @ A_act.T  # (n_act,)

        g = u + prefactor * c_2q_active * np.power(E_A, exponent)
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

        # Jacobian: J = I + k(k-1) * diag(coeff^k * E^{k-2}) * Cov_w(A)
        E_AA = A_act @ (w[:, None] * A_act.T)  # (n_act, n_act)
        Cov = E_AA - np.outer(E_A, E_A)
        D = deriv_factor * c_2q_active * np.power(E_A, exponent - 1)  # (n_act,)
        J = np.eye(n_act, dtype=complex) + D[:, None] * Cov

        # Mild regularization if extremely ill-conditioned
        try:
            cond = float(np.linalg.cond(J))
        except Exception:
            cond = float("inf")
        if cond > 1e13:
            J = J + (1e-8 * np.eye(n_act, dtype=complex))

        try:
            delta = np.linalg.solve(J, -g)
        except np.linalg.LinAlgError:
            return best_u, best_res < tol, best_res, it

        # Backtracking line search
        step = 1.0
        for _ in range(50):
            u_trial = u + step * delta
            w_t = _compute_weights_from_u(u_trial, active_idx, A, log_b, d)
            E_t = w_t @ A_act.T
            g_t = u_trial + prefactor * c_2q_active * np.power(E_t, exponent)
            res_t = float(np.max(np.abs(g_t)))
            if np.isfinite(res_t) and res_t < res:
                break
            step *= 0.5

        u = u + step * delta

    return best_u, best_res < tol, best_res, max_iter


def solve_8sat_saddle(
    p: int,
    gamma_target: float,
    betas: np.ndarray,
    q: int = 3,
    r: float = 176.54,
    gamma_homotopy_factor: float = 1.1,
    gamma_start: float = 1e-12,
    gamma_max_steps: int = 400,
    target_residual: float = 1e-3,
    newton_tol: float = 1e-10,
    newton_max_iter: int = 200,
    u_init_active: np.ndarray | None = None,
    gamma_start_override: float | None = None,
    max_newton_calls: int = 5000,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray, bool, float]:
    """
    Solve the BM24 Eq.(20) saddle for q>1 using Newton + γ-homotopy in u-space.

    Returns (y_star, coeff_alpha, converged, residual).
    - y_star: full d-dimensional saddle point (|α|<=1 components set to 0).
    - coeff_alpha: coeff_α at γ=gamma_target (for Hessian evaluation).
    - residual: max|g(u)| at final step.
    """
    A = build_structure_matrix(p, q=q)
    b_s = compute_b_s(p, betas)
    d = int(A.shape[0])

    active_idx = _get_active_indices(d)
    n_act = int(len(active_idx))
    # Match the existing softmax convention used elsewhere in the codebase.
    log_b = np.log(b_s + 1e-300 * (1 - np.sign(np.abs(b_s))))
    log_b = np.where(np.isfinite(log_b), log_b, -700.0)

    # Adaptive γ-homotopy (geometric with backtracking on failures).
    gt = float(gamma_target)
    gs = float(gamma_start)
    if gt <= 0:
        raise ValueError("gamma_target must be positive for this homotopy")
    if gs <= 0:
        raise ValueError("gamma_start must be positive")

    fac = float(gamma_homotopy_factor)
    if fac <= 1.0:
        raise ValueError("gamma_homotopy_factor must be > 1")

    if u_init_active is not None:
        u = np.asarray(u_init_active, dtype=complex).copy()
        if u.shape != (n_act,):
            raise ValueError("u_init_active has wrong shape")
    else:
        u = np.zeros(n_act, dtype=complex)
    final_res = float("inf")
    if gamma_start_override is not None:
        g_cur = float(gamma_start_override)
        if g_cur <= 0:
            raise ValueError("gamma_start_override must be positive")
        if g_cur > gt:
            g_cur = gt
    else:
        g_cur = gs
    steps_used = 0
    newton_calls = 0

    while g_cur < gt - 1e-20 and steps_used < int(gamma_max_steps):
        g_hi = min(gt, g_cur * fac)
        g_next = g_hi

        accepted = False
        nit = 0
        for _bt in range(60):
            if newton_calls >= int(max_newton_calls):
                break
            gammas = np.full(p, g_next, dtype=float)
            coeff = alpha_linear_coefficients(p, gammas, q=q, r=r)
            c_2q_active = np.power(coeff[active_idx], bm24_clause_arity(q))

            u_new, _conv, res, nit = newton_solve_q(
                u_init=u,
                c_2q_active=c_2q_active,
                active_idx=active_idx,
                A=A,
                log_b=log_b,
                d=d,
                q=q,
                tol=newton_tol,
                max_iter=newton_max_iter,
            )
            newton_calls += 1

            final_res = float(res)
            if np.isfinite(final_res) and final_res <= float(target_residual):
                u = u_new
                g_cur = g_next
                accepted = True
                break

            # Backtrack: shrink γ step (geometric bisection)
            g_next = float(np.sqrt(g_cur * g_next))
            if (g_next - g_cur) / max(g_cur, 1e-30) < 1e-6:
                break

        steps_used += 1
        if verbose:
            status = "ACCEPT" if accepted else "FAIL"
            print(
                f"  γ-step {steps_used}: γ={g_cur:.3e} ({status}), "
                f"res={final_res:.3e}, nit={nit}"
            )

        if not accepted:
            break
        if newton_calls >= int(max_newton_calls):
            break

    # Convert u back to y at the final gamma_target coeffs
    gammas_final = np.full(p, gt, dtype=float)
    coeff_final = alpha_linear_coefficients(p, gammas_final, q=q, r=r)
    y_star = np.zeros(d, dtype=complex)
    y_star[active_idx] = u / coeff_final[active_idx]

    converged = bool((g_cur >= gt - 1e-20) and np.isfinite(final_res) and (final_res <= float(target_residual)))
    return y_star, coeff_final, converged, final_res


def solve_8sat_saddle_general(
    p: int,
    betas: np.ndarray,
    gammas_target: np.ndarray,
    q: int = 3,
    r: float = 176.54,
    t_start: float = 1e-6,
    t_factor: float = 1.1,
    t_max_steps: int = 800,
    newton_tol: float = 1e-12,
    newton_max_iter: int = 800,
    target_residual: float = 1e-4,
    max_newton_calls: int = 5000,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray, bool, float]:
    """
    General p solver via scalar t-homotopy on a target gamma vector:
        gamma(t) = t * gammas_target,  t in [t_start, 1].

    Returns (y_star_full, coeff_final_full, converged, residual).
    """
    p = int(p)
    betas = np.asarray(betas, dtype=float)
    gammas_target = np.asarray(gammas_target, dtype=float)
    if betas.shape != (p,):
        raise ValueError("betas must have shape (p,)")
    if gammas_target.shape != (p,):
        raise ValueError("gammas_target must have shape (p,)")
    if t_start <= 0 or t_start > 1:
        raise ValueError("t_start must be in (0, 1]")
    if t_factor <= 1.0:
        raise ValueError("t_factor must be > 1")

    A = build_structure_matrix(p, q=q)
    b_s = compute_b_s(p, betas)
    d = int(A.shape[0])
    active_idx = _get_active_indices(d)
    n_act = int(len(active_idx))
    log_b = np.log(b_s + 1e-300 * (1 - np.sign(np.abs(b_s))))
    log_b = np.where(np.isfinite(log_b), log_b, -700.0)

    u = np.zeros(n_act, dtype=complex)
    t_cur = float(t_start)
    final_res = float("inf")
    steps = 0
    newton_calls = 0

    while t_cur < 1.0 - 1e-14 and steps < int(t_max_steps):
        t_hi = min(1.0, t_cur * float(t_factor))
        t_try = float(t_hi)
        accepted = False
        nit = 0

        for _ in range(80):
            if newton_calls >= int(max_newton_calls):
                break
            gammas_step = t_try * gammas_target
            coeff_step = alpha_linear_coefficients(p, gammas_step, q=q, r=r)
            c_2q_step = np.power(coeff_step[active_idx], bm24_clause_arity(q))
            u_new, conv, res, nit = newton_solve_q(
                u_init=u,
                c_2q_active=c_2q_step,
                active_idx=active_idx,
                A=A,
                log_b=log_b,
                d=d,
                q=q,
                tol=newton_tol,
                max_iter=newton_max_iter,
            )
            newton_calls += 1
            final_res = float(res)
            if bool(conv) or (np.isfinite(final_res) and final_res <= float(target_residual)):
                u = u_new
                t_cur = t_try
                accepted = True
                break
            # backtrack in homotopy parameter
            t_try = float(np.sqrt(t_cur * t_try))
            if (t_try - t_cur) / max(t_cur, 1e-30) < 1e-8:
                break

        steps += 1
        if verbose:
            status = "ACCEPT" if accepted else "FAIL"
            print(f"  t-step {steps}: t={t_cur:.6f} ({status}), res={final_res:.3e}, nit={nit}")
        if not accepted:
            break
        if newton_calls >= int(max_newton_calls):
            break

    coeff_final = alpha_linear_coefficients(p, gammas_target, q=q, r=r)
    y_star = np.zeros(d, dtype=complex)
    coeff_act = coeff_final[active_idx]
    for i in range(n_act):
        if abs(coeff_act[i]) > 1e-14:
            y_star[active_idx[i]] = u[i] / coeff_act[i]

    converged = bool(t_cur >= 1.0 - 1e-14)
    return y_star, coeff_final, converged, final_res

