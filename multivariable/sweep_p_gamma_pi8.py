"""
Sweep p in {1,2,3}, gamma in pi/8 increments, beta fixed at -pi/2.

For each (p, gamma), attempt to find distinct roots (via homotopy seed + multistart),
then record:
  1) root components
  2) residual ||g(y)||_inf
  3) Re[Phi(y)] = Re[F(y) + 7 * sum_alpha (dF_alpha(y))^8]
  4) spectral radius rho(J_g(y))
  5) near-zero vs large component counts
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from symmetry_reduction.core import (
    alpha_linear_coefficients,
    bm24_clause_arity,
    build_structure_matrix,
    compute_b_s,
)
from symmetry_reduction.newton_saddle_q import (
    _get_active_indices,
    _compute_weights_from_u,
    newton_solve_q,
    solve_8sat_saddle_general,
)


def _log_b_from_bs(b_s: np.ndarray) -> np.ndarray:
    log_b = np.log(b_s + 1e-300 * (1 - np.sign(np.abs(b_s))))
    return np.where(np.isfinite(log_b), log_b, -700.0)


def _y_active_to_full(y_active: np.ndarray, active_idx: np.ndarray, d: int) -> np.ndarray:
    y_full = np.zeros(d, dtype=complex)
    y_full[active_idx] = y_active
    return y_full


def _weights_from_y(y_active: np.ndarray, active_idx: np.ndarray, coeff: np.ndarray, A: np.ndarray, log_b: np.ndarray, d: int) -> np.ndarray:
    y_full = _y_active_to_full(y_active, active_idx, d)
    linear = (coeff * y_full) @ A
    log_w = log_b + linear
    log_w = np.where(np.isfinite(log_w), log_w, -700.0)
    shift = np.max(np.real(log_w))
    w = np.exp(log_w - shift)
    return w / np.sum(w)


def _g_and_jacobian_active(y_active: np.ndarray, active_idx: np.ndarray, coeff: np.ndarray, A: np.ndarray, log_b: np.ndarray, q: int) -> tuple[np.ndarray, np.ndarray]:
    d = A.shape[0]
    A_act = A[active_idx]
    coeff_act = coeff[active_idx]
    k = bm24_clause_arity(q)
    w = _weights_from_y(y_active, active_idx, coeff, A, log_b, d)
    E = w @ A_act.T
    g = y_active + float(k) * np.power(coeff_act * E, k - 1)

    E_AA = A_act @ (w[:, None] * A_act.T)
    cov = E_AA - np.outer(E, E)
    dgrad = np.diag(coeff_act) @ cov @ np.diag(coeff_act)
    dvec = float(k * (k - 1)) * np.power(coeff_act * E, k - 2)
    J = np.eye(len(active_idx), dtype=complex) + dvec[:, None] * dgrad
    return g, J


def _F_and_grad_full(y_active: np.ndarray, active_idx: np.ndarray, coeff: np.ndarray, A: np.ndarray, log_b: np.ndarray) -> tuple[complex, np.ndarray]:
    d = A.shape[0]
    y_full = _y_active_to_full(y_active, active_idx, d)
    linear = (coeff * y_full) @ A
    log_terms = log_b + linear
    log_terms = np.where(np.isfinite(log_terms), log_terms, -700.0)
    shift = np.max(np.real(log_terms))
    exp_terms = np.exp(log_terms - shift)
    Z_shift = np.sum(exp_terms)
    w = exp_terms / Z_shift
    F = shift + np.log(Z_shift)
    E_full = w @ A.T
    grad = coeff * E_full
    return F, grad


def _canonicalize_root(y_active: np.ndarray, tol: float = 1e-6) -> tuple[float, ...]:
    arr = np.concatenate([np.real(y_active), np.imag(y_active)])
    return tuple(np.round(arr / tol) * tol)


def find_distinct_roots_for_point(
    p: int,
    gamma: float,
    beta: float = -np.pi / 2,
    q: int = 3,
    r: float = 176.54,
    n_random_starts: int = 3,
) -> list[np.ndarray]:
    A = build_structure_matrix(p, q=q)
    d = A.shape[0]
    active_idx = _get_active_indices(d)
    n_act = len(active_idx)
    betas = np.full(p, beta, dtype=float)
    gammas_target = np.full(p, gamma, dtype=float)
    b_s = compute_b_s(p, betas)
    log_b = _log_b_from_bs(b_s)
    coeff = alpha_linear_coefficients(p, gammas_target, q=q, r=r)
    c_2q = np.power(coeff[active_idx], bm24_clause_arity(q))

    roots: list[np.ndarray] = []
    seen: set[tuple[float, ...]] = set()

    # First seed: robust homotopy solver.
    tfac = 1.12 if p <= 2 else 1.25
    tmax = 1200 if p <= 2 else 120
    newton_iter = 1200 if p <= 2 else 400
    y_full_seed, _coeff, converged, _res = solve_8sat_saddle_general(
        p=p,
        betas=betas,
        gammas_target=gammas_target,
        q=q,
        r=r,
        t_start=1e-6,
        t_factor=tfac,
        t_max_steps=tmax,
        newton_tol=1e-12,
        newton_max_iter=newton_iter,
        target_residual=1e-6,
        max_newton_calls=(2000 if p <= 2 else 500),
        verbose=False,
    )
    if converged:
        y_act = y_full_seed[active_idx]
        key = _canonicalize_root(y_act, tol=1e-6)
        seen.add(key)
        roots.append(y_act)

    # Additional random multistarts in u-space, converted to y.
    rng = np.random.default_rng(12345 + p * 1000 + int(round(gamma * 1000)))
    if p >= 3:
        return roots

    for scale in (1e-2, 1.0):
        for _ in range(n_random_starts):
            u0 = scale * (rng.standard_normal(n_act) + 1j * rng.standard_normal(n_act))
            u_sol, conv, res, _it = newton_solve_q(
                u_init=u0,
                c_2q_active=c_2q,
                active_idx=active_idx,
                A=A,
                log_b=log_b,
                d=d,
                q=q,
                tol=1e-11,
                max_iter=1200,
            )
            if not (conv or (np.isfinite(res) and float(res) < 1e-6)):
                continue
            y_act = np.zeros(n_act, dtype=complex)
            coeff_act = coeff[active_idx]
            good = True
            for i in range(n_act):
                if abs(coeff_act[i]) < 1e-14:
                    if abs(u_sol[i]) > 1e-7:
                        good = False
                        break
                    y_act[i] = 0.0 + 0.0j
                else:
                    y_act[i] = u_sol[i] / coeff_act[i]
            if not good:
                continue
            key = _canonicalize_root(y_act, tol=1e-6)
            if key in seen:
                continue
            seen.add(key)
            roots.append(y_act)

    return roots


def analyze_root(
    y_active: np.ndarray,
    p: int,
    gamma: float,
    beta: float = -np.pi / 2,
    q: int = 3,
    r: float = 176.54,
) -> dict:
    A = build_structure_matrix(p, q=q)
    d = A.shape[0]
    active_idx = _get_active_indices(d)
    betas = np.full(p, beta, dtype=float)
    gammas_target = np.full(p, gamma, dtype=float)
    b_s = compute_b_s(p, betas)
    log_b = _log_b_from_bs(b_s)
    coeff = alpha_linear_coefficients(p, gammas_target, q=q, r=r)

    g, J = _g_and_jacobian_active(y_active, active_idx, coeff, A, log_b, q=q)
    residual_inf = float(np.max(np.abs(g)))
    rho = float(np.max(np.abs(np.linalg.eigvals(J))))
    F, grad_full = _F_and_grad_full(y_active, active_idx, coeff, A, log_b)
    re_phi = float(np.real(F + 7.0 * np.sum(np.power(grad_full, 8))))

    y_full = _y_active_to_full(y_active, active_idx, d)
    abs_y = np.abs(y_full)
    near_zero_thresh = 1e-8
    large_thresh = 1.0
    n_near_zero = int(np.sum(abs_y < near_zero_thresh))
    n_large = int(np.sum(abs_y > large_thresh))

    return {
        "active_idx": [int(a) for a in active_idx],
        "y_active_real": [float(np.real(v)) for v in y_active],
        "y_active_imag": [float(np.imag(v)) for v in y_active],
        "y_full_real": [float(np.real(v)) for v in y_full],
        "y_full_imag": [float(np.imag(v)) for v in y_full],
        "residual_inf": residual_inf,
        "re_phi": re_phi,
        "rho_Jg": rho,
        "n_near_zero_components_full": n_near_zero,
        "n_large_components_full": n_large,
        "near_zero_threshold": near_zero_thresh,
        "large_threshold": large_thresh,
    }


def main() -> None:
    out = Path("multivariable/results/p123_gamma_pi8_sweep.json")
    q = 3
    r = 176.54
    beta = -np.pi / 2
    gammas = [k * np.pi / 8 for k in range(17)]  # 0 to 2pi inclusive

    if out.exists():
        with out.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    else:
        payload = {
            "q": q,
            "r": r,
            "beta_fixed": float(beta),
            "gammas": [float(g) for g in gammas],
            "results": [],
        }
    done = {(int(rw["p"]), round(float(rw["gamma"]), 12)) for rw in payload["results"]}

    for p in (1, 2, 3):
        for gamma in gammas:
            key = (int(p), round(float(gamma), 12))
            if key in done:
                print(f"p={p}, gamma={gamma:.6f}: skip (already computed)", flush=True)
                continue
            roots = find_distinct_roots_for_point(p=p, gamma=gamma, beta=beta, q=q, r=r)
            analyzed = [
                analyze_root(y_active=root, p=p, gamma=gamma, beta=beta, q=q, r=r)
                for root in roots
            ]
            payload["results"].append(
                {
                    "p": int(p),
                    "gamma": float(gamma),
                    "n_roots_found": len(analyzed),
                    "roots": analyzed,
                }
            )
            print(f"p={p}, gamma={gamma:.6f}: found {len(analyzed)} roots", flush=True)
            # Checkpoint after each (p, gamma) in case of long runs.
            out.parent.mkdir(parents=True, exist_ok=True)
            with out.open("w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
