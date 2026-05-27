"""
Full Krawczyk workflow for 8-SAT at p=1, q=3.

This module now includes:
  1) exact root system and Jacobian in u-space,
  2) real lift H(z)=0 in R^8,
  3) interval arithmetic enclosure (via mpmath.iv),
  4) Krawczyk operator construction and strict inclusion check.
"""

from __future__ import annotations

from dataclasses import dataclass
import argparse
import json
from pathlib import Path
import subprocess
import sys

from mpmath import iv
import numpy as np

# Ensure repository root is importable when running as a script.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from symmetry_reduction.core import (
    alpha_linear_coefficients,
    bm24_clause_arity,
    build_structure_matrix,
    compute_b_s,
    popcount,
)
from symmetry_reduction.newton_saddle_q import newton_solve_q


@dataclass(frozen=True)
class P1Q3Data:
    p: int
    q: int
    r: float
    beta: float
    gamma: float
    d: int
    active_idx: np.ndarray
    A: np.ndarray
    A_act: np.ndarray
    log_b: np.ndarray
    coeff: np.ndarray
    c_2q_active: np.ndarray


@dataclass(frozen=True)
class KrawczykResult:
    beta: float
    gamma: float
    radius: float
    success: bool
    residual_inf: float
    rho_JT: float
    note: str


def build_p1_q3_data(beta: float, gamma: float, r: float = 176.54) -> P1Q3Data:
    p = 1
    q = 3
    d = 1 << (2 * p + 1)
    active = np.array([a for a in range(d) if popcount(a) >= 2], dtype=np.int64)
    A = build_structure_matrix(p, q=q)
    A_act = A[active]
    b_s = compute_b_s(p, np.array([beta], dtype=float))
    log_b = np.log(b_s + 1e-300 * (1 - np.sign(np.abs(b_s))))
    log_b = np.where(np.isfinite(log_b), log_b, -700.0)
    coeff = alpha_linear_coefficients(p, np.array([gamma], dtype=float), q=q, r=r)
    k = bm24_clause_arity(q)
    c_2q_active = np.power(coeff[active], k)
    return P1Q3Data(
        p=p,
        q=q,
        r=float(r),
        beta=float(beta),
        gamma=float(gamma),
        d=d,
        active_idx=active,
        A=A,
        A_act=A_act,
        log_b=log_b,
        coeff=coeff,
        c_2q_active=c_2q_active,
    )


def _weights_from_y(y_active: np.ndarray, data: P1Q3Data) -> np.ndarray:
    y_full = np.zeros(data.d, dtype=complex)
    y_full[data.active_idx] = y_active
    coeff_y = data.coeff * y_full
    log_w = data.log_b + (coeff_y @ data.A)
    log_w = np.where(np.isfinite(log_w), log_w, -700.0)
    shift = np.max(log_w.real)
    w = np.exp(log_w - shift)
    return w / np.sum(w)


def g_y(y_active: np.ndarray, data: P1Q3Data) -> np.ndarray:
    """
    Complex root system in y-space (BM24 Eq. (20), k = 2^q):
        g(y) = y + k * (coeff * E_w[A])^{k-1} = 0.
    For q=3 (k=8): g(y) = y + 8 * (coeff * E)^7.
    """
    k = bm24_clause_arity(data.q)
    coeff_act = data.coeff[data.active_idx]
    w = _weights_from_y(y_active, data)
    E = w @ data.A_act.T
    return y_active + float(k) * np.power(coeff_act * E, k - 1)


def jacobian_g_y(y_active: np.ndarray, data: P1Q3Data) -> np.ndarray:
    """
    Complex Jacobian:
        J_g = I + k(k-1) diag((coeff*E)^{k-2}) * d(coeff*E)/dy.
    """
    k = bm24_clause_arity(data.q)
    n = len(data.active_idx)
    coeff_act = data.coeff[data.active_idx]
    w = _weights_from_y(y_active, data)
    E = w @ data.A_act.T
    E_AA = data.A_act @ (w[:, None] * data.A_act.T)
    cov = E_AA - np.outer(E, E)
    dgrad = np.diag(coeff_act) @ cov @ np.diag(coeff_act)
    dvec = float(k * (k - 1)) * np.power(coeff_act * E, k - 2)
    return np.eye(n, dtype=complex) + dvec[:, None] * dgrad


def fixed_point_jacobian_y(y_active: np.ndarray, data: P1Q3Data) -> np.ndarray:
    """
    Jacobian of the fixed-point map T(y) = -k (coeff*E)^{k-1}, k = 2^q.
    """
    return jacobian_g_y(y_active, data) - np.eye(len(data.active_idx), dtype=complex)


def real_lift_vector(v: np.ndarray) -> np.ndarray:
    return np.concatenate([np.real(v), np.imag(v)])


def real_lift_matrix(m: np.ndarray) -> np.ndarray:
    a = np.real(m)
    b = np.imag(m)
    return np.block([[a, -b], [b, a]])


def h_z(z: np.ndarray, data: P1Q3Data) -> np.ndarray:
    n = len(data.active_idx)
    y = z[:n] + 1j * z[n:]
    return real_lift_vector(g_y(y, data))


def jacobian_h_z(z: np.ndarray, data: P1Q3Data) -> np.ndarray:
    n = len(data.active_idx)
    y = z[:n] + 1j * z[n:]
    return real_lift_matrix(jacobian_g_y(y, data))


def approximate_root(data: P1Q3Data) -> np.ndarray:
    """
    Compute a floating-point anchor y0 on active indices.

    Robust strategy:
    - gamma ~= 0: return exact y=0 anchor.
    - otherwise: signed-gamma continuation in u-space with Newton warm starts,
      then safe conversion u->y.
    """
    n = len(data.active_idx)
    if abs(float(data.gamma)) < 1e-15:
        return np.zeros(n, dtype=complex)

    g_target = float(data.gamma)
    g_start = 1e-6 if g_target > 0 else -1e-6
    direction = 1.0 if g_target > g_start else -1.0
    step = max(1e-3, min(0.02, abs(g_target - g_start) / 20.0))
    min_step = 1e-7
    max_accepts = 5000

    u = np.zeros(n, dtype=complex)
    g_cur = g_start
    accepts = 0
    while direction * (g_target - g_cur) > 1e-14:
        if accepts > max_accepts:
            raise RuntimeError("Signed-gamma continuation exceeded max accepted steps.")
        g_try = g_cur + direction * step
        if direction * (g_try - g_target) > 0:
            g_try = g_target

        coeff_step = alpha_linear_coefficients(
            data.p,
            np.array([float(g_try)], dtype=float),
            q=data.q,
            r=data.r,
        )
        c2q_step = np.power(coeff_step[data.active_idx], bm24_clause_arity(data.q))

        u_try, conv, res, _it = newton_solve_q(
            u_init=u,
            c_2q_active=c2q_step,
            active_idx=data.active_idx,
            A=data.A,
            log_b=data.log_b,
            d=data.d,
            q=data.q,
            tol=1e-11,
            max_iter=600,
        )
        ok = bool(conv) or (np.isfinite(res) and float(res) <= 1e-4)
        if ok:
            u = u_try
            g_cur = g_try
            accepts += 1
            # Expand step modestly after success.
            step = min(step * 1.25, 0.05)
            continue

        # Retry with looser tolerance.
        u_try2, conv2, res2, _it2 = newton_solve_q(
            u_init=u,
            c_2q_active=c2q_step,
            active_idx=data.active_idx,
            A=data.A,
            log_b=data.log_b,
            d=data.d,
            q=data.q,
            tol=1e-8,
            max_iter=1000,
        )
        ok2 = bool(conv2) or (np.isfinite(res2) and float(res2) <= 5e-4)
        if ok2:
            u = u_try2
            g_cur = g_try
            accepts += 1
            step = min(step * 1.1, 0.05)
            continue

        # Multi-start perturbation around current iterate to escape branch lock.
        rng = np.random.default_rng(0)
        rescued = False
        best_u = u
        best_res = float("inf")
        for scale in (1e-4, 1e-3, 1e-2):
            for _ in range(6):
                perturb = scale * (
                    rng.standard_normal(u.shape) + 1j * rng.standard_normal(u.shape)
                )
                u_init = u + perturb
                u_m, conv_m, res_m, _it_m = newton_solve_q(
                    u_init=u_init,
                    c_2q_active=c2q_step,
                    active_idx=data.active_idx,
                    A=data.A,
                    log_b=data.log_b,
                    d=data.d,
                    q=data.q,
                    tol=1e-8,
                    max_iter=1200,
                )
                if np.isfinite(res_m) and float(res_m) < best_res:
                    best_res = float(res_m)
                    best_u = u_m
                if conv_m or (np.isfinite(res_m) and float(res_m) <= 5e-4):
                    u = u_m
                    g_cur = g_try
                    accepts += 1
                    step = min(step * 1.05, 0.03)
                    rescued = True
                    break
            if rescued:
                break
        if rescued:
            continue

        # Backtrack step.
        step *= 0.5
        if step < min_step:
            raise RuntimeError(
                f"Signed-gamma continuation stalled near gamma={g_cur:.6g}; "
                f"next try={g_try:.6g}, residual={res2}, best_multistart_res={best_res}."
            )

    # Final polish at target coefficients.
    u_polish, _conv2, _res2, _it2 = newton_solve_q(
        u_init=u,
        c_2q_active=data.c_2q_active,
        active_idx=data.active_idx,
        A=data.A,
        log_b=data.log_b,
        d=data.d,
        q=data.q,
        tol=1e-15,
        max_iter=800,
    )
    u_final = u_polish

    # Safe u -> y conversion (guarding coeff ~ 0).
    coeff_act = data.coeff[data.active_idx]
    y = np.zeros(n, dtype=complex)
    for i in range(n):
        c = coeff_act[i]
        if abs(c) < 1e-14:
            if abs(u_final[i]) > 1e-8:
                raise RuntimeError(
                    "Near-zero coeff with non-negligible u prevents stable y recovery."
                )
            y[i] = 0.0 + 0.0j
        else:
            y[i] = u_final[i] / c
    return y


def point_metrics(beta: float, gamma: float, r: float = 176.54) -> dict:
    data = build_p1_q3_data(beta=beta, gamma=gamma, r=r)
    y0 = approximate_root(data)
    g0 = g_y(y0, data)
    j_t = fixed_point_jacobian_y(y0, data)
    rho = float(np.max(np.abs(np.linalg.eigvals(j_t))))
    return {
        "beta": float(beta),
        "gamma": float(gamma),
        "residual_inf": float(np.max(np.abs(g0))),
        "rho_JT": rho,
        "y0": y0,
    }


def pick_best_first_point(candidates: list[tuple[float, float]], r: float = 176.54) -> dict:
    """
    Pick easiest candidate: prioritize tiny residual, then smaller rho(J_T).
    """
    scored: list[tuple[tuple[float, float], dict]] = []
    for beta, gamma in candidates:
        try:
            m = point_metrics(beta, gamma, r=r)
            scored.append(((m["residual_inf"], m["rho_JT"]), m))
        except Exception:
            continue
    if not scored:
        raise RuntimeError("No candidate produced a converged floating-point anchor.")
    scored.sort(key=lambda x: x[0])
    return scored[0][1]


def demo_best_point() -> dict:
    """
    Small default set in the currently stable p=1 regime.
    """
    candidates = [
        (-np.pi / 2, 0.1),
        (-np.pi / 2, 0.1361946795412982),
        (-np.pi / 2, 0.14),
    ]
    return pick_best_first_point(candidates)


def _iv_const(x: float):
    return iv.mpf([float(x), float(x)])


def _iv_box(center: float, radius: float):
    return iv.mpf([float(center - radius), float(center + radius)])


def _to_float(x) -> float:
    return float(x)


def _iv_midpoint(x) -> float:
    return 0.5 * (float(x.a) + float(x.b))


def _iv_strict_subset(inner, outer) -> bool:
    return (_to_float(inner.a) > _to_float(outer.a)) and (_to_float(inner.b) < _to_float(outer.b))


def _complex_iv_add(a, b):
    return (a[0] + b[0], a[1] + b[1])


def _complex_iv_sub(a, b):
    return (a[0] - b[0], a[1] - b[1])


def _complex_iv_mul(a, b):
    ar, ai = a
    br, bi = b
    return (ar * br - ai * bi, ar * bi + ai * br)


def _complex_iv_scale(c: float, z):
    return (_iv_const(c) * z[0], _iv_const(c) * z[1])


def _complex_iv_exp(z):
    zr, zi = z
    ex = iv.exp(zr)
    return (ex * iv.cos(zi), ex * iv.sin(zi))


def _complex_iv_pow_int(z, n: int):
    if n < 0:
        raise ValueError("n must be nonnegative")
    out = (_iv_const(1.0), _iv_const(0.0))
    base = z
    k = n
    while k > 0:
        if k & 1:
            out = _complex_iv_mul(out, base)
        base = _complex_iv_mul(base, base)
        k >>= 1
    return out


def _complex_iv_div(a, b):
    br, bi = b
    denom = br * br + bi * bi
    # If 0 ∈ denom, reciprocal interval is undefined for proof purposes.
    if _to_float(denom.a) <= 0.0 <= _to_float(denom.b):
        raise RuntimeError("Interval division encountered denominator containing zero.")
    num = _complex_iv_mul(a, (br, -bi))
    return (num[0] / denom, num[1] / denom)


def _complex_point(re: float, im: float):
    return (_iv_const(re), _iv_const(im))


def _complex_from_complex_scalar(z: complex):
    return (_iv_const(float(np.real(z))), _iv_const(float(np.imag(z))))


def _iv_softmax_and_E_from_box(z_box: list, data: P1Q3Data):
    n = len(data.active_idx)
    y_intervals: list[tuple] = []
    for i in range(n):
        y_intervals.append((z_box[i], z_box[i + n]))

    y_full = [(_iv_const(0.0), _iv_const(0.0)) for _ in range(data.d)]
    for k, a in enumerate(data.active_idx):
        y_full[a] = y_intervals[k]

    log_b = data.log_b
    z_terms: list[tuple] = []
    real_midpoints: list[float] = []
    for s in range(data.d):
        linear = (_iv_const(0.0), _iv_const(0.0))
        for a in range(data.d):
            coeff = float(data.A[a, s])
            if coeff == 0.0:
                continue
            coeff_a = _complex_from_complex_scalar(data.coeff[a])
            term = _complex_iv_mul(coeff_a, y_full[a])
            linear = _complex_iv_add(linear, _complex_iv_scale(coeff, term))
        z = _complex_iv_add(_complex_point(float(np.real(log_b[s])), float(np.imag(log_b[s]))), linear)
        z_terms.append(z)
        real_midpoints.append(_iv_midpoint(z[0]))

    # Center exponentials around midpoint dominant real part to reduce interval blow-up.
    shift = max(real_midpoints) if real_midpoints else 0.0
    exp_terms: list[tuple] = []
    for z in z_terms:
        z_shifted = _complex_iv_sub(z, (_iv_const(shift), _iv_const(0.0)))
        exp_terms.append(_complex_iv_exp(z_shifted))

    zsum = (_iv_const(0.0), _iv_const(0.0))
    for t in exp_terms:
        zsum = _complex_iv_add(zsum, t)

    weights = []
    for t in exp_terms:
        weights.append(_complex_iv_div(t, zsum))

    E = []
    for i in range(n):
        acc = (_iv_const(0.0), _iv_const(0.0))
        for s in range(data.d):
            a_is = float(data.A_act[i, s])
            if a_is == 0.0:
                continue
            acc = _complex_iv_add(acc, _complex_iv_scale(a_is, weights[s]))
        E.append(acc)
    return weights, E


def _interval_g_and_jacobian(z_box: list, data: P1Q3Data):
    k = bm24_clause_arity(data.q)
    n = len(data.active_idx)
    weights, E = _iv_softmax_and_E_from_box(z_box, data)

    g_iv = []
    for i in range(n):
        yi = (z_box[i], z_box[i + n])
        coeff_i = _complex_from_complex_scalar(data.coeff[data.active_idx[i]])
        grad = _complex_iv_mul(coeff_i, E[i])
        term = _complex_iv_pow_int(grad, k - 1)
        term = _complex_iv_scale(float(k), term)
        g_iv.append(_complex_iv_add(yi, term))

    # Build covariance matrix on active coordinates.
    E_AA = [[(_iv_const(0.0), _iv_const(0.0)) for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            acc = (_iv_const(0.0), _iv_const(0.0))
            for s in range(data.d):
                c = float(data.A_act[i, s] * data.A_act[j, s])
                if c == 0.0:
                    continue
                acc = _complex_iv_add(acc, _complex_iv_scale(c, weights[s]))
            E_AA[i][j] = acc
    cov = [[None for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            cov[i][j] = _complex_iv_sub(E_AA[i][j], _complex_iv_mul(E[i], E[j]))

    dvec = []
    for i in range(n):
        coeff_i = _complex_from_complex_scalar(data.coeff[data.active_idx[i]])
        grad = _complex_iv_mul(coeff_i, E[i])
        d = _complex_iv_pow_int(grad, k - 2)
        d = _complex_iv_scale(float(k * (k - 1)), d)
        dvec.append(d)

    jg = [[(_iv_const(0.0), _iv_const(0.0)) for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            coeff_i = _complex_from_complex_scalar(data.coeff[data.active_idx[i]])
            coeff_j = _complex_from_complex_scalar(data.coeff[data.active_idx[j]])
            dgrad_ij = _complex_iv_mul(coeff_i, _complex_iv_mul(cov[i][j], coeff_j))
            jg[i][j] = _complex_iv_mul(dvec[i], dgrad_ij)
            if i == j:
                jg[i][j] = _complex_iv_add(jg[i][j], (_iv_const(1.0), _iv_const(0.0)))

    return g_iv, jg


def _real_lift_interval_vector(g_iv: list[tuple]):
    n = len(g_iv)
    out = [None] * (2 * n)
    for i in range(n):
        out[i] = g_iv[i][0]
        out[i + n] = g_iv[i][1]
    return out


def _real_lift_interval_matrix(jg_iv: list[list[tuple]]):
    n = len(jg_iv)
    m = [[_iv_const(0.0) for _ in range(2 * n)] for _ in range(2 * n)]
    for i in range(n):
        for j in range(n):
            re, im = jg_iv[i][j]
            m[i][j] = re
            m[i][j + n] = -im
            m[i + n][j] = im
            m[i + n][j + n] = re
    return m


def _interval_matvec(m, v):
    rows = len(m)
    cols = len(m[0])
    out = [_iv_const(0.0) for _ in range(rows)]
    for i in range(rows):
        acc = _iv_const(0.0)
        for j in range(cols):
            acc = acc + m[i][j] * v[j]
        out[i] = acc
    return out


def _float_times_interval_matrix(r_float: np.ndarray, j_iv):
    rows, cols = r_float.shape
    out = [[_iv_const(0.0) for _ in range(cols)] for _ in range(rows)]
    for i in range(rows):
        for j in range(cols):
            acc = _iv_const(0.0)
            for k in range(cols):
                acc = acc + _iv_const(float(r_float[i, k])) * j_iv[k][j]
            out[i][j] = acc
    return out


def h_e_z(z: np.ndarray, data: P1Q3Data) -> np.ndarray:
    n = len(data.active_idx)
    e = z[:n] + 1j * z[n:]
    k = bm24_clause_arity(data.q)
    u = -float(k) * data.c_2q_active * np.power(e, k - 1)
    u_full = np.zeros(data.d, dtype=complex)
    u_full[data.active_idx] = u
    log_w = data.log_b + (u_full @ data.A)
    log_w = np.where(np.isfinite(log_w), log_w, -700.0)
    shift = np.max(log_w.real)
    w = np.exp(log_w - shift)
    w = w / np.sum(w)
    e_new = w @ data.A_act.T
    return real_lift_vector(e - e_new)


def jacobian_h_e_z(z: np.ndarray, data: P1Q3Data) -> np.ndarray:
    n = len(data.active_idx)
    e = z[:n] + 1j * z[n:]
    k = bm24_clause_arity(data.q)
    u = -float(k) * data.c_2q_active * np.power(e, k - 1)
    u_full = np.zeros(data.d, dtype=complex)
    u_full[data.active_idx] = u
    log_w = data.log_b + (u_full @ data.A)
    log_w = np.where(np.isfinite(log_w), log_w, -700.0)
    shift = np.max(log_w.real)
    w = np.exp(log_w - shift)
    w = w / np.sum(w)
    E = w @ data.A_act.T
    E_AA = data.A_act @ (w[:, None] * data.A_act.T)
    cov = E_AA - np.outer(E, E)
    du = -float(k * (k - 1)) * data.c_2q_active * np.power(e, k - 2)
    j_complex = np.eye(n, dtype=complex) - cov @ np.diag(du)
    return real_lift_matrix(j_complex)


def _interval_he_and_jacobian(z_box: list, data: P1Q3Data):
    k = bm24_clause_arity(data.q)
    n = len(data.active_idx)
    e_intervals = [(z_box[i], z_box[i + n]) for i in range(n)]

    # Build u(E) = -k * coeff^k * E^{k-1}
    u_active = []
    for i in range(n):
        c2q_i = _complex_from_complex_scalar(data.c_2q_active[i])
        epow = _complex_iv_pow_int(e_intervals[i], k - 1)
        u_active.append(_complex_iv_scale(-float(k), _complex_iv_mul(c2q_i, epow)))

    u_full = [(_iv_const(0.0), _iv_const(0.0)) for _ in range(data.d)]
    for k, a in enumerate(data.active_idx):
        u_full[a] = u_active[k]

    z_terms = []
    real_midpoints: list[float] = []
    for s in range(data.d):
        linear = (_iv_const(0.0), _iv_const(0.0))
        for a in range(data.d):
            coeff = float(data.A[a, s])
            if coeff != 0.0:
                linear = _complex_iv_add(linear, _complex_iv_scale(coeff, u_full[a]))
        z = _complex_iv_add(_complex_point(float(np.real(data.log_b[s])), float(np.imag(data.log_b[s]))), linear)
        z_terms.append(z)
        real_midpoints.append(_iv_midpoint(z[0]))

    shift = max(real_midpoints) if real_midpoints else 0.0
    exp_terms = []
    for z in z_terms:
        z_shifted = _complex_iv_sub(z, (_iv_const(shift), _iv_const(0.0)))
        exp_terms.append(_complex_iv_exp(z_shifted))

    zsum = (_iv_const(0.0), _iv_const(0.0))
    for t in exp_terms:
        zsum = _complex_iv_add(zsum, t)
    weights = [_complex_iv_div(t, zsum) for t in exp_terms]

    e_new = []
    for i in range(n):
        acc = (_iv_const(0.0), _iv_const(0.0))
        for s in range(data.d):
            a_is = float(data.A_act[i, s])
            if a_is != 0.0:
                acc = _complex_iv_add(acc, _complex_iv_scale(a_is, weights[s]))
        e_new.append(acc)

    h_iv = [_complex_iv_sub(e_intervals[i], e_new[i]) for i in range(n)]

    # Covariance
    E_AA = [[(_iv_const(0.0), _iv_const(0.0)) for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            acc = (_iv_const(0.0), _iv_const(0.0))
            for s in range(data.d):
                c = float(data.A_act[i, s] * data.A_act[j, s])
                if c != 0.0:
                    acc = _complex_iv_add(acc, _complex_iv_scale(c, weights[s]))
            E_AA[i][j] = acc
    cov = [[_complex_iv_sub(E_AA[i][j], _complex_iv_mul(e_new[i], e_new[j])) for j in range(n)] for i in range(n)]

    du = []
    for j in range(n):
        c2q_j = _complex_from_complex_scalar(data.c_2q_active[j])
        ep = _complex_iv_pow_int(e_intervals[j], k - 2)
        du.append(_complex_iv_scale(-float(k * (k - 1)), _complex_iv_mul(c2q_j, ep)))

    jh = [[(_iv_const(0.0), _iv_const(0.0)) for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            term = _complex_iv_mul(cov[i][j], du[j])
            jh[i][j] = _complex_iv_sub((_iv_const(1.0), _iv_const(0.0)) if i == j else (_iv_const(0.0), _iv_const(0.0)), term)
    return h_iv, jh


def _krawczyk_single_radius(
    z0: np.ndarray,
    h0: np.ndarray,
    r_mat: np.ndarray,
    data: P1Q3Data,
    abs_radius: float,
    rel_radius: float = 0.0,
    interval_space: str = "E",
    relative_only: bool = False,
    rad_vec_override: list[float] | None = None,
) -> tuple[bool, str]:
    n = z0.size
    if rad_vec_override is not None:
        if len(rad_vec_override) != n:
            return False, "rad_vec_override has wrong length"
        rad_vec = [float(max(v, 0.0)) for v in rad_vec_override]
    elif relative_only:
        rad_vec = [rel_radius * max(abs(float(z0[i])), 1e-15) for i in range(n)]
    else:
        rad_vec = [max(abs_radius, rel_radius * abs(float(z0[i]))) for i in range(n)]
    x_box = [_iv_box(float(z0[i]), rad_vec[i]) for i in range(n)]
    x_minus_z0 = [_iv_box(0.0, rad_vec[i]) for i in range(n)]

    try:
        if interval_space.lower() == "y":
            h_iv, jh_c_iv = _interval_g_and_jacobian(x_box, data)
        elif interval_space.lower() == "e":
            h_iv, jh_c_iv = _interval_he_and_jacobian(x_box, data)
        else:
            return False, f"unknown interval_space={interval_space!r}"
        jh_iv = _real_lift_interval_matrix(jh_c_iv)
    except RuntimeError as exc:
        return False, f"interval evaluation failed: {exc}"

    rj = _float_times_interval_matrix(r_mat, jh_iv)
    m = [[_iv_const(0.0) for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            val = -rj[i][j]
            if i == j:
                val = val + _iv_const(1.0)
            m[i][j] = val

    affine_center = z0 - (r_mat @ h0)
    tail = _interval_matvec(m, x_minus_z0)
    k_box = [None] * n
    for i in range(n):
        k_box[i] = _iv_const(float(affine_center[i])) + tail[i]

    ok = True
    for i in range(n):
        if not _iv_strict_subset(k_box[i], x_box[i]):
            ok = False
            break
    if ok:
        return True, "K(X) strictly inside X"
    return False, "inclusion failed"


def run_krawczyk_certification(
    beta: float,
    gamma: float,
    r: float = 176.54,
    space: str = "E",
    path_points: list[tuple[float, float]] | None = None,
) -> KrawczykResult:
    data = build_p1_q3_data(beta=beta, gamma=gamma, r=r)
    if path_points is not None:
        if len(path_points) == 0:
            raise ValueError("path_points must be non-empty when provided.")
        beta_last, gamma_last = path_points[-1]
        if abs(float(beta_last) - float(beta)) > 1e-12 or abs(float(gamma_last) - float(gamma)) > 1e-12:
            raise ValueError("Final path point must match (beta, gamma) arguments.")
        y0 = approximate_root_path(path_points=path_points, r=r)
    else:
        y0 = approximate_root(data)
    space_l = space.lower()
    if space_l == "y":
        # Requested y-space Krawczyk setup:
        # z0 = lift(y0), h0 = lift(g_y(y0)), j0 = lift(J_g(y0)).
        z0 = real_lift_vector(y0)
        h0 = real_lift_vector(g_y(y0, data))
        j0 = real_lift_matrix(jacobian_g_y(y0, data))
        r_mat = np.linalg.inv(j0)
    elif space_l == "e":
        k = bm24_clause_arity(data.q)
        coeff_act = data.coeff[data.active_idx]
        # Convert y-anchor to E-anchor using y = -k * (coeff*E)^{k-1}.
        e0 = np.power((-y0 / float(k)) + 0j, 1.0 / (k - 1)) / coeff_act
        z0 = real_lift_vector(e0)
        h0 = h_e_z(z0, data)
        j0 = jacobian_h_e_z(z0, data)
        r_mat = np.linalg.inv(j0)
    else:
        raise ValueError("space must be 'E' or 'y'")

    jt = fixed_point_jacobian_y(y0, data)
    rho = float(np.max(np.abs(np.linalg.eigvals(jt))))
    residual = float(np.max(np.abs(g_y(y0, data))))

    if space_l == "y":
        # IMPORTANT: in y-space use relative radii only, and anisotropic per-coordinate scaling.
        rel_attempts = [1e-6, 1e-8, 1e-10, 1e-12]
        row_sens = np.sum(np.abs(j0), axis=1)
        row_sens = np.maximum(row_sens, 1e-15)
        row_sens = row_sens / float(np.mean(row_sens))
        base_scale = np.maximum(np.abs(z0), 1e-15)
        rad_vec_attempts: list[list[float]] = []
        for rr in rel_attempts:
            # Tighter radii for more sensitive coordinates.
            rad_vec = (rr * base_scale) / np.sqrt(row_sens)
            rad_vec = np.maximum(rad_vec, rr * 1e-15)
            rad_vec_attempts.append([float(v) for v in rad_vec])
        attempt_pairs = [(0.0, rr) for rr in rel_attempts]
    else:
        attempt_pairs = [
            (1e-6, 0.0),
            (1e-7, 0.0),
            (1e-8, 0.0),
            (1e-9, 0.0),
            (1e-10, 0.0),
            (1e-12, 1e-12),
            (1e-14, 1e-12),
            (1e-16, 1e-12),
            (1e-18, 1e-12),
            (1e-20, 1e-12),
        ]

    for idx, (abs_radius, rel_radius) in enumerate(attempt_pairs):
        rad_vec_override = None
        if space_l == "y":
            rad_vec_override = rad_vec_attempts[idx]
        ok, note = _krawczyk_single_radius(
            z0=z0,
            h0=h0,
            r_mat=r_mat,
            data=data,
            abs_radius=abs_radius,
            rel_radius=rel_radius,
            interval_space=space_l,
            relative_only=(space_l == "y"),
            rad_vec_override=rad_vec_override,
        )
        if ok:
            return KrawczykResult(
                beta=float(beta),
                gamma=float(gamma),
                radius=float(rel_radius if space_l == "y" else max(abs_radius, rel_radius)),
                success=True,
                residual_inf=residual,
                rho_JT=rho,
                note=f"{note} (space={space_l})",
            )

    return KrawczykResult(
        beta=float(beta),
        gamma=float(gamma),
        radius=float("nan"),
        success=False,
        residual_inf=residual,
        rho_JT=rho,
        note=f"No tested radius satisfied strict Krawczyk inclusion (space={space_l}).",
    )


def approximate_root_path(
    path_points: list[tuple[float, float]],
    r: float = 176.54,
) -> np.ndarray:
    """
    Two-parameter warm-start continuation over (beta, gamma) path.

    Returns y on active coordinates at the final path point.
    """
    if not path_points:
        raise ValueError("path_points must be non-empty")

    # Warm-start in u-space across path points.
    u = None
    for beta, gamma in path_points:
        data = build_p1_q3_data(beta=float(beta), gamma=float(gamma), r=r)
        if abs(float(gamma)) < 1e-15:
            u = np.zeros(len(data.active_idx), dtype=complex)
            continue
        if u is None:
            u = np.zeros(len(data.active_idx), dtype=complex)
        u_new, conv, res, _it = newton_solve_q(
            u_init=u,
            c_2q_active=data.c_2q_active,
            active_idx=data.active_idx,
            A=data.A,
            log_b=data.log_b,
            d=data.d,
            q=data.q,
            tol=1e-10,
            max_iter=1000,
        )
        if (not conv) and (not np.isfinite(res) or float(res) > 1e-4):
            raise RuntimeError(
                f"Path continuation failed at beta={beta}, gamma={gamma} with res={res}."
            )
        u = u_new

    beta_f, gamma_f = path_points[-1]
    data_f = build_p1_q3_data(beta=float(beta_f), gamma=float(gamma_f), r=r)
    coeff_act = data_f.coeff[data_f.active_idx]
    y = np.zeros(len(coeff_act), dtype=complex)
    for i, c in enumerate(coeff_act):
        if abs(c) < 1e-14:
            y[i] = 0.0 + 0.0j
        else:
            y[i] = u[i] / c
    return y


def search_certified_point(
    beta: float,
    gammas: list[float],
    r: float = 176.54,
    space: str = "E",
) -> tuple[KrawczykResult | None, list[KrawczykResult]]:
    results: list[KrawczykResult] = []
    for gamma in gammas:
        try:
            cert = run_krawczyk_certification(beta=beta, gamma=gamma, r=r, space=space)
        except Exception as exc:
            cert = KrawczykResult(
                beta=float(beta),
                gamma=float(gamma),
                radius=float("nan"),
                success=False,
                residual_inf=float("nan"),
                rho_JT=float("nan"),
                note=f"exception: {exc}",
            )
        results.append(cert)
        if cert.success:
            return cert, results
    return None, results


def save_report(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True, default=str)


def export_anchor_payload(beta: float, gamma: float, r: float = 176.54) -> dict:
    """
    Build a machine-readable anchor payload for downstream certifiers.
    """
    data = build_p1_q3_data(beta=float(beta), gamma=float(gamma), r=float(r))
    y0 = approximate_root(data)
    g0 = g_y(y0, data)
    return {
        "beta": float(beta),
        "gamma": float(gamma),
        "r": float(r),
        "y0_real": [float(np.real(v)) for v in y0],
        "y0_imag": [float(np.imag(v)) for v in y0],
        "anchor_residual_inf": float(np.max(np.abs(g0))),
        "active_idx": [int(i) for i in data.active_idx],
        "coeff_real": [float(np.real(v)) for v in data.coeff],
        "coeff_imag": [float(np.imag(v)) for v in data.coeff],
        "log_b_real": [float(np.real(v)) for v in data.log_b],
        "log_b_imag": [float(np.imag(v)) for v in data.log_b],
    }


def run_julia_certifier(anchor_path: Path, script_path: Path | None = None) -> tuple[bool, str]:
    """
    Invoke the Julia Arb-based certifier with the exported anchor.
    """
    if script_path is None:
        script_path = REPO_ROOT / "certify_8sat_saddle.jl"
    if not script_path.exists():
        return False, f"julia script not found: {script_path}"
    cmd = ["julia", str(script_path), "--anchor", str(anchor_path)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    out = (proc.stdout or "").strip()
    err = (proc.stderr or "").strip()
    if proc.returncode == 0:
        return True, out or "julia certifier completed successfully"
    msg = out
    if err:
        msg = f"{msg}\n{err}".strip()
    return False, msg or f"julia certifier failed with exit code {proc.returncode}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Krawczyk workflow for p=1, q=3 (8-SAT).")
    parser.add_argument("--backend", choices=["mpmath", "arb"], default="mpmath")
    parser.add_argument("--beta", type=float, default=-float(np.pi / 2))
    parser.add_argument("--gamma", type=float, default=0.14)
    parser.add_argument("--r", type=float, default=176.54)
    parser.add_argument(
        "--anchor-out",
        type=Path,
        default=REPO_ROOT / "multivariable" / "results" / "anchor.json",
        help="Path for exported anchor payload.",
    )
    args = parser.parse_args()

    if args.backend == "arb":
        payload = export_anchor_payload(beta=args.beta, gamma=args.gamma, r=args.r)
        args.anchor_out.parent.mkdir(parents=True, exist_ok=True)
        with args.anchor_out.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
        ok, msg = run_julia_certifier(anchor_path=args.anchor_out)
        print(msg)
        sys.exit(0 if ok else 1)

    best = demo_best_point()
    print("best candidate metrics:", {k: v for k, v in best.items() if k != "y0"})
    beta = args.beta
    gamma_try = [args.gamma, 0.1361946795412982, 0.1, 0.05, 0.01, 1e-6, 1e-8, 1e-10]
    cert, all_results = search_certified_point(beta=beta, gammas=gamma_try, r=args.r, space="E")
    if cert is None:
        print("krawczyk: no strict inclusion found in tested list")
    else:
        print("krawczyk: certified", cert)
    report = {
        "beta": beta,
        "r": args.r,
        "space": "E",
        "gamma_tried": gamma_try,
        "success": bool(cert is not None and cert.success),
        "certified_point": None
        if cert is None
        else {
            "space": "E",
            "gamma": cert.gamma,
            "radius": cert.radius,
            "residual_inf": cert.residual_inf,
            "rho_JT": cert.rho_JT,
            "note": cert.note,
        },
        "attempts": [
            {
                "space": "E",
                "gamma": c.gamma,
                "success": c.success,
                "radius": c.radius,
                "residual_inf": c.residual_inf,
                "rho_JT": c.rho_JT,
                "note": c.note,
            }
            for c in all_results
        ],
    }
    out = REPO_ROOT / "multivariable" / "results" / "krawczyk_p1_q3_report.json"
    save_report(out, report)
    print(f"report written to {out}")
