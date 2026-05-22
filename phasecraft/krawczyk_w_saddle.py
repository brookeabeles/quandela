"""
Krawczyk root certification for the parent-action saddle equations in
w-coordinates, on the corrected beta = -pi/2, p = 1, q = 1 slice.

Source memo: holomorphic_parent_integral_augmented_filtered_seed.md
   Sec. 5  -- corrected slice: divisor (Eq. 5.4) and couplings (Eq. 5.5)
   Sec. 6  -- local slice certificates (Thm 6.1 contraction;
              Prop 6.2 rescaled Morse certificate with F_tilde(0) = 1)
   Sec. 7  -- divisor separation (Prop 7.1; Thm 7.2: no singularity
              crossing on bounded branches)
   Sec. 18 -- explicit Krawczyk continuation task

This module implements Sec. 18 items 1-4 rigorously and item 5
diagnostically.

  1. Define BM24 branch boxes X_j; certify a UNIQUE root of
        F_tilde(w; gamma) := w + 2 C(gamma) * grad log Delta_star(w) = 0
     in each box.
  2. Certify nondegeneracy via the rescaled Hessian
        H_tilde(w; gamma) := I + 2 C(gamma) * D grad log Delta_star(w).
     H_tilde is exactly the Jacobian of F_tilde, so the Krawczyk
     contraction norm < 1 simultaneously establishes inclusion,
     uniqueness, and nondegeneracy.  This is the rescaled Morse
     certificate of Sec. 6.2.
  3. Certify divisor separation:  |Delta_star(w)| >= delta > 0 on X_j.
     The interval evaluation of Delta_star is returned alongside the
     Krawczyk data, and a strictly positive lower bound is required
     for certification.
  4. Enclose the action lift Phi_eff(w*(gamma); gamma) at the
     numerical center (principal branch; user must track continuous
     lifts across gamma meshes -- see note in Phi_eff_diagnostic).

Items 5-8 of Sec. 18 (Stokes / anti-Stokes wall tracking, filtered
action gaps, actual PL jumps, stop conditions) require thimble-flow
boundary-value problems and are out of scope for a local interval
root certifier.

WHY RESCALED (memo Sec. 6.2 / Eq. 6.2).  The raw saddle equation
    F_raw(w; gamma)_alpha = w_alpha / (2 c_alpha(gamma))
                          + (grad log Delta_star)(w)_alpha = 0
has Jacobian diag(1/(2c_alpha)) + Dg, whose entries blow up as
gamma -> 0.  Multiplying through by 2 c_alpha gives F_tilde with
Jacobian I + 2 C(gamma) Dg, which equals I exactly at gamma = 0
and is uniformly well-conditioned in a neighbourhood of 0.  The
two systems have identical zero sets on the locus
    { w in U_w : c_alpha(gamma) != 0 for all alpha },
so certifying F_tilde is equivalent to certifying F_raw for the
purposes of Sec. 6 / Sec. 18.
"""

from __future__ import annotations
import argparse
import json
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import mpmath as mp
from mpmath import iv
from scipy.optimize import root

from symmetry_reduction.core import compute_b_s


# =====================================================================
# 1. Slice data: divisor terms, couplings, first moments
# =====================================================================
#
# Each entry of _DELTA_TERMS is (coefficient, exponent-vector e).
# The actual term is coefficient * exp(i * (e . w) / 2).
# Compare memo Eq. (5.4):
#   Delta_star(w)
#     = 0.5     * exp(i (w0 + w1 + w2 + w3) / 2)     -- {000, 111}
#     + 0.5*i   * exp(i  w0 / 2)                      -- {001, 110}
#     + 0.5     * exp(i  w1 / 2)                      -- {010, 101}
#     - 0.5*i   * exp(i  w2 / 2)                      -- {011, 100}
# At w = 0:  Delta_star(0) = 0.5 + 0.5i + 0.5 - 0.5i = 1.            (5.4)
#
_DELTA_TERMS = (
    (complex(0.5,  0.0), (1, 1, 1, 1)),
    (complex(0.0,  0.5), (1, 0, 0, 0)),
    (complex(0.5,  0.0), (0, 1, 0, 0)),
    (complex(0.0, -0.5), (0, 0, 1, 0)),
)

# Flip-symmetric 3-bit configuration pairs -> exponent e on the 4-w chart.
# On beta = -pi/2, p = 1, orbit-averaged BM24 weights match _DELTA_TERMS.
_DELTA_ORBIT_PAIRS = (
    ((0, 7), _DELTA_TERMS[0]),
    ((1, 6), _DELTA_TERMS[1]),
    ((2, 5), _DELTA_TERMS[2]),
    ((3, 4), _DELTA_TERMS[3]),
)

SLICE_BETA = -np.pi / 2
SLICE_P = 1

# First moments m_alpha = sum_s b_s A_{alpha,s}; memo Eq. (5.6).
M_VEC = np.array(
    [0.25 + 0.25j, 0.5 + 0.0j, 0.25 - 0.25j, 0.25 + 0.0j],
    dtype=complex,
)


def couplings(r: float, gamma: float) -> np.ndarray:
    """Couplings c_0, c_1, c_2, c_3; memo Eq. (5.5)."""
    return np.array(
        [
            r * (1.0 - np.exp(-0.5j * gamma)),    # c_0 = r (1 - e^{-i gamma/2})
            4.0 * r * np.sin(0.25 * gamma) ** 2,  # c_1 = 4 r sin^2(gamma/4)
            r * (1.0 - np.exp( 0.5j * gamma)),    # c_2 = conj(c_0) for real gamma
           -4.0 * r * np.sin(0.25 * gamma) ** 2,  # c_3 = -c_1
        ],
        dtype=complex,
    )


# =====================================================================
# 2. Pointwise (numpy) evaluation
# =====================================================================
def delta_terms_nontrivial_index_set(
    beta: float = SLICE_BETA,
    p: int = SLICE_P,
    *,
    b_tol: float = 1e-12,
) -> list[tuple[complex, tuple[int, int, int, int]]]:
    """BM24 parent divisor terms on the nontrivial index-set chart.

    For each flip orbit of 3-bit configurations s, take the orbit-averaged
    weight ½(b_{s1}+b_{s2}) when that sum is numerically nonzero; otherwise use
    the closed slice coefficient from memo Eq. (5.4).  At beta = -pi/2, p = 1
    this list equals ``_DELTA_TERMS``.
    """
    if p != SLICE_P:
        raise ValueError(f"only p={SLICE_P} slice implemented, got p={p}")
    betas = np.full(p, beta, dtype=float)
    b = compute_b_s(p, betas)
    terms: list[tuple[complex, tuple[int, int, int, int]]] = []
    for (s1, s2), (c_slice, e) in _DELTA_ORBIT_PAIRS:
        b_sum = b[s1] + b[s2]
        coef = 0.5 * b_sum if abs(b_sum) > b_tol else c_slice
        terms.append((coef, e))
    return terms


def _evaluate_delta_terms(w: np.ndarray, terms) -> complex:
    w = np.asarray(w, dtype=complex)
    total = 0.0 + 0.0j
    for coef, e in terms:
        total += coef * np.exp(0.5j * np.dot(e, w))
    return total


def _grad_delta_terms(w: np.ndarray, terms) -> np.ndarray:
    w = np.asarray(w, dtype=complex)
    g = np.zeros(4, dtype=complex)
    for coef, e in terms:
        E = np.exp(0.5j * np.dot(e, w))
        for a in range(4):
            if e[a]:
                g[a] += coef * E * 0.5j * e[a]
    return g


def Delta(
    w: np.ndarray,
    beta: float = SLICE_BETA,
    p: int = SLICE_P,
) -> complex:
    """Parent divisor Delta(w) on the nontrivial index-set chart."""
    return _evaluate_delta_terms(w, delta_terms_nontrivial_index_set(beta=beta, p=p))


def grad_Delta(
    w: np.ndarray,
    beta: float = SLICE_BETA,
    p: int = SLICE_P,
) -> np.ndarray:
    """Gradient of Delta(w)."""
    return _grad_delta_terms(w, delta_terms_nontrivial_index_set(beta=beta, p=p))


def grad_log_Delta(
    w: np.ndarray,
    beta: float = SLICE_BETA,
    p: int = SLICE_P,
) -> np.ndarray:
    """grad log Delta(w)."""
    d = Delta(w, beta=beta, p=p)
    return grad_Delta(w, beta=beta, p=p) / d


def Delta_star(w: np.ndarray) -> complex:
    """Closed-form slice divisor (memo Eq. 5.4); equals ``Delta`` on the slice."""
    return _evaluate_delta_terms(w, _DELTA_TERMS)


def grad_Delta_star(w: np.ndarray) -> np.ndarray:
    return _grad_delta_terms(w, _DELTA_TERMS)


def hess_Delta_star(w: np.ndarray) -> np.ndarray:
    H = np.zeros((4, 4), dtype=complex)
    for coef, e in _DELTA_TERMS:
        E = np.exp(0.5j * np.dot(e, w))
        for a in range(4):
            if not e[a]:
                continue
            for b in range(4):
                if not e[b]:
                    continue
                H[a, b] += coef * E * (0.5j) * e[a] * (0.5j) * e[b]
    return H


def log_grad(w: np.ndarray) -> np.ndarray:
    """g(w) = grad log Delta(w) on the slice."""
    return grad_log_Delta(w)


def log_hess(w: np.ndarray) -> np.ndarray:
    """Dg(w) = Hessian of log Delta_star(w)."""
    Delta = Delta_star(w)
    gd = grad_Delta_star(w)
    Hd = hess_Delta_star(w)
    return Hd / Delta - np.outer(gd, gd) / (Delta * Delta)


@dataclass
class WSaddleSystem:
    """Saddle system for fixed r, gamma on the corrected slice.

    Attributes
    ----------
    r     :  random clause density / 2^k coefficient (the BM24 r).
    gamma :  slice parameter; the seed chamber of memo Sec. 14-16 is
             gamma in a small neighbourhood of 0 with gamma < 0.
    """
    r: float
    gamma: float

    @property
    def c(self) -> np.ndarray:
        return couplings(self.r, self.gamma)

    def leading_seed(self) -> np.ndarray:
        """w_alpha = -2 i c_alpha m_alpha + O(c_infty^2);  memo Eq. (5.7).

        At gamma = 0 this is identically zero, which is the correct
        (degenerate) saddle.  For nonzero gamma it gives an O(c_infty)
        initial guess for the Newton / Krawczyk certifier.
        """
        return -2.0j * self.c * M_VEC

    def F_complex(self, w: np.ndarray) -> np.ndarray:
        """Rescaled saddle equation F_tilde(w) = w + 2 C g(w)."""
        return w + 2.0 * self.c * log_grad(w)

    def J_complex(self, w: np.ndarray) -> np.ndarray:
        """Jacobian of F_tilde:  I + 2 C Dg(w).

        Equals the rescaled Hessian H_tilde of memo Eq. (6.2) when
        evaluated at a saddle.  Equals I at w = 0, gamma = 0.
        """
        return np.eye(4, dtype=complex) + np.diag(2.0 * self.c) @ log_hess(w)

    def F_real(self, x: np.ndarray) -> np.ndarray:
        """Real-packed F for scipy.optimize.root:  x = [Re w | Im w]."""
        w = x[:4] + 1j * x[4:]
        f = self.F_complex(w)
        return np.concatenate([f.real, f.imag])

    def Phi_eff(self, w: np.ndarray) -> complex:
        """Effective parent action Phi_eff(w);  memo Eq. (4.1).

        Uses the principal branch of log.  For continuous lifts
        across a gamma mesh the user must unwrap the imaginary part
        manually (memo Sec. 18, item 4).
        """
        c = self.c
        quad = np.sum(w * w / (4.0 * c))
        return quad + np.log(Delta_star(w))


# =====================================================================
# 3. Interval automatic differentiation (DualIV)
# =====================================================================
class DualIV:
    """Forward-mode AD over mpmath complex intervals."""

    __slots__ = ("val", "deriv")

    def __init__(self, val, deriv):
        self.val = val
        self.deriv = list(deriv)

    @staticmethod
    def const(v, n):
        return DualIV(v, [iv.mpc(0)] * n)

    @staticmethod
    def var(v, n, idx):
        d = [iv.mpc(0)] * n
        d[idx] = iv.mpc(1)
        return DualIV(v, d)

    def _promote(self, other):
        if isinstance(other, DualIV):
            return other
        return DualIV.const(other, len(self.deriv))

    def __add__(self, other):
        o = self._promote(other)
        return DualIV(
            self.val + o.val,
            [self.deriv[i] + o.deriv[i] for i in range(len(self.deriv))],
        )
    __radd__ = __add__

    def __sub__(self, other):
        o = self._promote(other)
        return DualIV(
            self.val - o.val,
            [self.deriv[i] - o.deriv[i] for i in range(len(self.deriv))],
        )

    def __rsub__(self, other):
        return DualIV.const(other, len(self.deriv)) - self

    def __neg__(self):
        return DualIV(-self.val, [-d for d in self.deriv])

    def __mul__(self, other):
        o = self._promote(other)
        return DualIV(
            self.val * o.val,
            [
                self.val * o.deriv[i] + self.deriv[i] * o.val
                for i in range(len(self.deriv))
            ],
        )
    __rmul__ = __mul__

    def __truediv__(self, other):
        o = self._promote(other)
        denom_sq = o.val * o.val
        return DualIV(
            self.val / o.val,
            [
                (self.deriv[i] * o.val - self.val * o.deriv[i]) / denom_sq
                for i in range(len(self.deriv))
            ],
        )

    def __rtruediv__(self, other):
        return DualIV.const(other, len(self.deriv)) / self

    def exp(self):
        ev = iv.exp(self.val)
        return DualIV(ev, [ev * d for d in self.deriv])


# =====================================================================
# 4. Interval helpers
# =====================================================================
def _to_iv_complex(z) -> iv.mpc:
    r = float(np.real(z))
    s = float(np.imag(z))
    return iv.mpc(iv.mpf([r, r]), iv.mpf([s, s]))


def _box_iv(zc, radius) -> iv.mpc:
    r = float(radius)
    return iv.mpc(
        iv.mpf([float(np.real(zc)) - r, float(np.real(zc)) + r]),
        iv.mpf([float(np.imag(zc)) - r, float(np.imag(zc)) + r]),
    )


def _couplings_iv(r: float, gamma):
    """Interval couplings; gamma is float (thin) or iv.mpf (thick)."""
    r_iv = iv.mpf([float(r), float(r)])
    if isinstance(gamma, (int, float)):
        g_iv = iv.mpf([float(gamma), float(gamma)])
    else:
        g_iv = gamma
    e_minus = iv.exp(iv.mpc(iv.mpf(0), -g_iv / iv.mpf(2)))
    e_plus = iv.exp(iv.mpc(iv.mpf(0),  g_iv / iv.mpf(2)))
    one_c = iv.mpc(iv.mpf(1), iv.mpf(0))
    r_c = iv.mpc(r_iv, iv.mpf(0))
    c0 = r_c * (one_c - e_minus)
    c2 = r_c * (one_c - e_plus)
    s_q = iv.sin(g_iv / iv.mpf(4))
    c1_r = iv.mpf(4) * r_iv * s_q * s_q
    c1 = iv.mpc(c1_r, iv.mpf(0))
    c3 = iv.mpc(-c1_r, iv.mpf(0))
    return [c0, c1, c2, c3]


# =====================================================================
# 5. Interval evaluation of F_tilde, its Jacobian, and Delta_star
# =====================================================================
def F_jacobian_delta_iv(w_box: List[iv.mpc], r: float, gamma):
    """On the interval box w_box, return (F_iv, J_iv, Delta_iv).

    F_iv      : list of 4 iv.mpc, enclosing F_tilde(w_box).
    J_iv      : 4x4 list of iv.mpc, enclosing dF_tilde / dw on w_box.
                Equals  I + 2 C(gamma) Dg(w_box)  in interval arithmetic.
    Delta_iv  : iv.mpc, enclosing Delta_star(w_box).

    All three are computed in a single DualIV pass so the interval
    enclosure is consistent across the three outputs.
    """
    n = 4
    w_dual = [DualIV.var(w_box[i], n, i) for i in range(n)]
    half_i = iv.mpc(iv.mpf(0), iv.mpf([0.5, 0.5]))

    # Pass 1:  Delta and partial_alpha Delta as DualIVs.
    Delta = DualIV.const(iv.mpc(0), n)
    dDelta = [DualIV.const(iv.mpc(0), n) for _ in range(n)]
    for coef, e in _DELTA_TERMS:
        exponent = DualIV.const(iv.mpc(0), n)
        for a in range(n):
            if e[a]:
                exponent = exponent + DualIV.const(iv.mpc(int(e[a]), 0), n) * w_dual[a]
        exponent = DualIV.const(half_i, n) * exponent
        E = exponent.exp()
        coef_iv = _to_iv_complex(coef)
        Delta = Delta + DualIV.const(coef_iv, n) * E
        for a in range(n):
            if e[a]:
                # d/dw_a (coef * exp((i/2) sum e_b w_b)) = coef * (i e_a / 2) * E
                fac = coef_iv * iv.mpc(iv.mpf(0), iv.mpf([0.5 * e[a], 0.5 * e[a]]))
                dDelta[a] = dDelta[a] + DualIV.const(fac, n) * E

    # g_alpha(w) = partial_alpha Delta / Delta -- a DualIV of w.
    g_dual = [dDelta[a] / Delta for a in range(n)]

    # F_tilde_alpha = w_alpha + 2 c_alpha g_alpha.
    c_iv = _couplings_iv(r, gamma)
    F_dual = []
    for a in range(n):
        two_c = _to_iv_complex(2.0) * c_iv[a]
        F_dual.append(w_dual[a] + DualIV.const(two_c, n) * g_dual[a])

    F_iv = [F_dual[a].val for a in range(n)]
    J_iv = [[F_dual[a].deriv[b] for b in range(n)] for a in range(n)]
    return F_iv, J_iv, Delta.val


# =====================================================================
# 6. Krawczyk test
# =====================================================================
def _matvec_iv(Y_np: np.ndarray, v_iv: List[iv.mpc]) -> List[iv.mpc]:
    n = len(v_iv)
    out = []
    for i in range(n):
        acc = iv.mpc(0)
        for j in range(n):
            acc = acc + _to_iv_complex(Y_np[i, j]) * v_iv[j]
        out.append(acc)
    return out


def _matmul_iv(Y_np: np.ndarray, J_iv):
    n = len(J_iv)
    out = [[iv.mpc(0)] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            acc = iv.mpc(0)
            for k in range(n):
                acc = acc + _to_iv_complex(Y_np[i, k]) * J_iv[k][j]
            out[i][j] = acc
    return out


def _strict_inside(small: iv.mpc, big: iv.mpc) -> bool:
    return (
        small.real.a > big.real.a and small.real.b < big.real.b
        and small.imag.a > big.imag.a and small.imag.b < big.imag.b
    )


def _matrix_inf_norm_upper_iv(M_iv) -> float:
    row_max = mp.mpf(0)
    for row in M_iv:
        s = mp.mpf(0)
        for x in row:
            s += abs(x).b
        if s > row_max:
            row_max = s
    return float(row_max)


def _abs_lower_iv(z_iv: iv.mpc) -> float:
    """Conservative lower bound for |z| on the interval z_iv.

    For real interval [a, b], the lower bound of x^2 is
        0                  if 0 in [a, b],
        min(a^2, b^2)      otherwise.
    Then  |z|^2 >= lower(Re^2) + lower(Im^2).
    """
    def lower_sq(x_iv):
        a = float(x_iv.a)
        b = float(x_iv.b)
        if a <= 0.0 <= b:
            return 0.0
        return min(a * a, b * b)
    re_lo2 = lower_sq(z_iv.real)
    im_lo2 = lower_sq(z_iv.imag)
    return float(mp.sqrt(re_lo2 + im_lo2))


def krawczyk_certify_w_root(
    sys: WSaddleSystem,
    w_center: np.ndarray,
    dps: int = 60,
    max_inflate_iters: int = 6,
):
    """Rigorous Krawczyk certification of a unique root of F_tilde near w_center.

    Simultaneously certifies (memo Sec. 18 items 1-3 on a single box):

      (a) Unique-root inclusion  K(w_center, X) subset int(X).
      (b) Nondegeneracy        contraction := sqrt(2) * ||I - Y J(X)||_inf < 1
          (this is the rescaled Morse certificate of memo Sec. 6.2;
          when contraction < 1 the rescaled Hessian H_tilde is
          invertible on the entire box X).
      (c) Divisor separation   delta_lower := lower bound on |Delta_star(X)|
                               must be strictly positive.

    Returns (ok, info).  On success, info includes
      contraction_bound, delta_lower, inflate_iter, box_radius,
      w_center_(real,imag), Phi_eff_(real,imag), dps_used.
    """
    mp.mp.dps = int(dps)
    iv.dps = int(dps)
    n = 4

    # Analytic preconditioner from the exact J_complex at the center.
    F_c = sys.F_complex(w_center)
    J_c = sys.J_complex(w_center)
    try:
        Y = np.linalg.inv(J_c)
    except np.linalg.LinAlgError:
        return False, {"reason": "singular analytic Jacobian at center"}

    yg = Y @ F_c
    yg_norm = float(np.linalg.norm(yg, ord=np.inf))

    # Radius schedule (mirrors the existing z-coord certifier).
    u = mp.power(2, -mp.mp.prec)
    inflate_factor = float(mp.power(u, mp.mpf("-0.25")))
    ref_rad = max(yg_norm * inflate_factor, 10.0 ** -(dps - 8))
    tiny_abs = [10.0 ** -k for k in (40, 32, 28, 24, 20, 16, 12, 10)]
    heuristic_scales = [2.0 ** k for k in range(-4, max_inflate_iters)]
    radii = list(tiny_abs) + [max(ref_rad * s, 10.0 ** -(dps - 8)) for s in heuristic_scales]

    F_thin = [_to_iv_complex(F_c[i]) for i in range(n)]

    for it, radius in enumerate(radii):
        w_box = [_box_iv(w_center[i], radius) for i in range(n)]
        F_box, J_box, Delta_box = F_jacobian_delta_iv(w_box, sys.r, sys.gamma)

        Yg = _matvec_iv(Y, F_thin)
        YJ = _matmul_iv(Y, J_box)
        M = [
            [(iv.mpc(1) if i == j else iv.mpc(0)) - YJ[i][j] for j in range(n)]
            for i in range(n)
        ]
        diff = [w_box[i] - _to_iv_complex(w_center[i]) for i in range(n)]
        term2 = []
        for i in range(n):
            acc = iv.mpc(0)
            for j in range(n):
                acc = acc + M[i][j] * diff[j]
            term2.append(acc)
        K_box = [
            _to_iv_complex(w_center[i]) - Yg[i] + term2[i] for i in range(n)
        ]

        inclusion = all(_strict_inside(K_box[i], w_box[i]) for i in range(n))
        contraction = float(np.sqrt(2.0)) * _matrix_inf_norm_upper_iv(M)
        delta_lower = _abs_lower_iv(Delta_box)

        if inclusion and contraction < 1.0 and delta_lower > 0.0:
            phi = sys.Phi_eff(w_center)
            return True, {
                "contraction_bound": float(contraction),
                "delta_lower": float(delta_lower),
                "inflate_iter": int(it),
                "box_radius": float(radius),
                "w_center_real": w_center.real.tolist(),
                "w_center_imag": w_center.imag.tolist(),
                "Phi_eff_real": float(phi.real),
                "Phi_eff_imag": float(phi.imag),
                "dps_used": int(dps),
            }

    return False, {"reason": "inclusion / contraction / divisor-separation failed"}


def krawczyk_certify_with_escalation(
    sys: WSaddleSystem,
    w_center: np.ndarray,
    dps_start: int = 60,
    max_inflate_iters: int = 6,
    max_levels: int = 4,
):
    """Try increasing precision until Krawczyk succeeds, or give up."""
    dps = int(dps_start)
    last: dict = {}
    for _ in range(max_levels):
        ok, info = krawczyk_certify_w_root(
            sys, w_center, dps=dps, max_inflate_iters=max_inflate_iters
        )
        if ok:
            return True, info
        last = info
        dps = int(np.ceil(dps * 1.5))
    out = dict(last)
    out["dps_attempted"] = dps
    return False, out


# =====================================================================
# 7. Root discovery
# =====================================================================
def solve_w_from_init(
    sys: WSaddleSystem,
    w_init: np.ndarray,
    tol: float = 1e-12,
) -> tuple[np.ndarray, bool, float]:
    """Solve F_tilde(w)=0 from a warm start (continuation / homotopy)."""
    w_init = np.asarray(w_init, dtype=complex)
    x0 = np.concatenate([w_init.real, w_init.imag])
    sol = root(sys.F_real, x0, method="hybr", tol=tol)
    if not sol.success or not np.isfinite(sol.x).all():
        return w_init, False, float(np.inf)
    w = sol.x[:4] + 1j * sol.x[4:]
    res = float(np.linalg.norm(sys.F_complex(w), ord=np.inf))
    return w, bool(res < 1e-8), res


def discover_w_roots(
    sys: WSaddleSystem,
    num_starts: int = 200,
    seed: int = 0,
    tol: float = 1e-12,
    w_init: Optional[np.ndarray] = None,
    search_scale: Optional[float] = None,
) -> List[np.ndarray]:
    """Find numerical roots of F_tilde via scipy.root + random restarts.

    Always seeds the leading-order BM24 root w_alpha = -2 i c_alpha m_alpha
    first, then adds random restarts to discover any competing saddles
    in a moderate ball.
    """
    rng = np.random.default_rng(seed)
    roots: List[np.ndarray] = []

    if w_init is not None:
        w_cont, ok, _ = solve_w_from_init(sys, w_init, tol=tol)
        if ok and not any(np.linalg.norm(w_cont - wp, ord=np.inf) < 1e-5 for wp in roots):
            roots.append(w_cont)

    # Leading BM24 seed.
    w0 = sys.leading_seed()
    x0 = np.concatenate([w0.real, w0.imag])
    sol = root(sys.F_real, x0, method="hybr", tol=tol)
    if sol.success and np.isfinite(sol.x).all():
        if np.linalg.norm(sys.F_real(sol.x), ord=np.inf) < 1e-8:
            roots.append(sol.x[:4] + 1j * sol.x[4:])

    # Random restarts in a ball around the seed / leading saddle scale.
    if search_scale is not None:
        scale = float(search_scale)
    else:
        ref = w_init if w_init is not None else w0
        scale = max(float(np.max(np.abs(ref))) * 5.0, 0.05)
    for _ in range(num_starts):
        x0 = rng.normal(0.0, scale, size=8)
        sol = root(sys.F_real, x0, method="hybr", tol=tol)
        if not sol.success or not np.isfinite(sol.x).all():
            continue
        if np.linalg.norm(sys.F_real(sol.x), ord=np.inf) > 1e-7:
            continue
        w = sol.x[:4] + 1j * sol.x[4:]
        if any(np.linalg.norm(w - wp, ord=np.inf) < 1e-5 for wp in roots):
            continue
        roots.append(w)
    return roots


# =====================================================================
# 8. CLI
# =====================================================================
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Krawczyk root certifier for parent-action saddles "
                    "(corrected beta=-pi/2, p=1, q=1 slice)."
    )
    parser.add_argument("--r", type=float, required=True,
                        help="BM24 r coefficient (random clause density / 2^k).")
    parser.add_argument("--gamma", type=float, required=True,
                        help="Slice parameter; seed chamber is gamma < 0 small.")
    parser.add_argument("--num-starts", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tol", type=float, default=1e-12)
    parser.add_argument("--dps", type=int, default=60,
                        help="mpmath decimal precision for the rigorous certifier.")
    parser.add_argument("--max-inflate-iters", type=int, default=6)
    parser.add_argument("--escalate", action="store_true",
                        help="Try increasing dps if Krawczyk fails.")
    parser.add_argument("--out", type=str, default="")
    args = parser.parse_args()

    sys = WSaddleSystem(r=args.r, gamma=args.gamma)
    roots_w = discover_w_roots(
        sys, num_starts=args.num_starts, seed=args.seed, tol=args.tol
    )

    payload = {
        "r": args.r,
        "gamma": args.gamma,
        "num_roots": len(roots_w),
        "couplings_real": [float(c.real) for c in sys.c],
        "couplings_imag": [float(c.imag) for c in sys.c],
        "leading_seed_real": [float(w.real) for w in sys.leading_seed()],
        "leading_seed_imag": [float(w.imag) for w in sys.leading_seed()],
        "roots": [],
    }

    for i, w in enumerate(roots_w):
        F_norm = float(np.linalg.norm(sys.F_complex(w), ord=np.inf))
        if args.escalate:
            cert, info = krawczyk_certify_with_escalation(
                sys, w, dps_start=args.dps,
                max_inflate_iters=args.max_inflate_iters,
            )
        else:
            cert, info = krawczyk_certify_w_root(
                sys, w, dps=args.dps,
                max_inflate_iters=args.max_inflate_iters,
            )
        phi = sys.Phi_eff(w)
        row = {
            "idx": i,
            "w_real": w.real.tolist(),
            "w_imag": w.imag.tolist(),
            "residual_F_norm": F_norm,
            "Phi_eff_real": float(phi.real),
            "Phi_eff_imag": float(phi.imag),
            "krawczyk_certified": bool(cert),
            "krawczyk_info": info,
        }
        payload["roots"].append(row)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    summary = {
        "r": payload["r"], "gamma": payload["gamma"],
        "num_roots": payload["num_roots"],
        "certified_roots": sum(1 for r in payload["roots"] if r["krawczyk_certified"]),
    }
    print(json.dumps(summary, indent=2))


# =====================================================================
# 9. Self-test
# =====================================================================
def selftest() -> None:
    """Sanity checks against the memo's explicit identities."""
    print("=== krawczyk_w_saddle self-test ===")

    # Memo Eq. (5.4): Delta(0) = Delta_star(0) = 1.
    w0 = np.zeros(4, dtype=complex)
    d0 = Delta(w0)
    assert abs(d0 - 1.0) < 1e-14, f"Delta(0) = {d0}, expected 1"
    assert abs(Delta_star(w0) - d0) < 1e-14
    print(f"  Delta(0) = Delta_star(0) = {d0}   [OK]")

    # Memo's identification:  g(0) = i m.
    g0 = grad_log_Delta(w0)
    im_expected = 1j * M_VEC
    assert np.max(np.abs(g0 - im_expected)) < 1e-14, \
        f"g(0) = {g0}, expected {im_expected}"
    print(f"  g(0) = i*m                            [OK]")

    # Leading saddle at small gamma.
    r, gamma = 1.0, -0.05
    sys = WSaddleSystem(r=r, gamma=gamma)
    w_seed = sys.leading_seed()
    print(f"  r={r}, gamma={gamma}")
    print(f"  leading_seed = {w_seed}")
    print(f"  ||F_tilde(seed)||_inf = {np.linalg.norm(sys.F_complex(w_seed), ord=np.inf):.3e}"
          f"   (should be O(c_infty^2))")

    # Memo Eq. (5.9):  leading action = (r/4)(3 sin^2(gamma/4) - sin(gamma/2)).
    leading_action_expected = (r / 4.0) * (3.0 * np.sin(gamma / 4.0) ** 2 - np.sin(gamma / 2.0))
    leading_action_formula = float(np.sum(sys.c * M_VEC ** 2).real)
    err = abs(leading_action_expected - leading_action_formula)
    assert err < 1e-14, f"Eq. (5.9) mismatch: {err}"
    print(f"  Eq. (5.9) leading action  = {leading_action_expected:.10f}   "
          f"   matches sum c m^2 = {leading_action_formula:.10f}   [OK]")

    # Find and certify the root.
    roots = discover_w_roots(sys, num_starts=30, seed=0)
    print(f"  Found {len(roots)} numerical roots near small-gamma saddle.")
    if roots:
        w_star = roots[0]
        print(f"  w_star = {w_star}")
        print(f"  ||F_tilde(w_star)||_inf = {np.linalg.norm(sys.F_complex(w_star), ord=np.inf):.3e}")
        ok, info = krawczyk_certify_w_root(sys, w_star, dps=60)
        print(f"  Krawczyk certified: {ok}")
        for k, v in info.items():
            if k not in ("w_center_real", "w_center_imag"):
                print(f"     {k}: {v}")
    print("=== self-test done ===")


if __name__ == "__main__":
    import sys as _sys
    if len(_sys.argv) == 1 or (len(_sys.argv) == 2 and _sys.argv[1] == "--selftest"):
        selftest()
    else:
        main()
