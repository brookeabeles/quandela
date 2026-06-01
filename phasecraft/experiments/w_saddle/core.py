"""
Krawczyk root certification for the parent-action saddle equations in
w-coordinates on the p = 1, q = 1 nontrivial-index chart (default beta = -pi/2).

Orchestration: ``phasecraft.w_saddle.workflow`` and ``python -m phasecraft.w_saddle``.
See ``phasecraft/w_saddle/README.md``.

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
        F_tilde(w; gamma) := w + 2 C(gamma) * grad log Delta(w; beta) = 0
     in each box.
  2. Certify nondegeneracy: if sup_{J in DF(X)} ||I - Y J||_inf < 1 with Y
     invertible, every J in DF(X) is invertible (Neumann series), so the
     certified root is nondegenerate.  Here DF is the Jacobian of F_tilde_R
     on the realified box X = x_0 + [-rho, rho]^8.
  3. Certify divisor separation:  |Delta(w; beta)| >= delta > 0 on X_j.
     The interval evaluation of Delta is returned alongside the
     Krawczyk data, and a strictly positive lower bound is required
     for certification.
  4. Optional interval enclosure Phi_eff(X; gamma) on the certified box
     (log branch fixed by certified ratio-lift along the mesh).  The Newton
     center is numerical; the root is certified to lie in X, not at the center
     alone.  Im Phi for Stokes analysis uses ell_j from ratio-lifts, not raw
     principal Im Phi_eff.

Items 5-8 of Sec. 18 (Stokes / anti-Stokes wall tracking, filtered
action gaps, actual PL jumps, stop conditions) require thimble-flow
boundary-value problems and are out of scope for a local interval
root certifier.

WHY RESCALED (memo Sec. 6.2 / Eq. 6.2).  The raw saddle equation
    F_raw(w; gamma)_alpha = w_alpha / (2 c_alpha(gamma))
                          + (grad log Delta)(w; beta)_alpha = 0
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
from typing import Any, List, Optional

import numpy as np
import mpmath as mp
from mpmath import iv
from scipy.optimize import root

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
    """BM24 parent divisor terms on the p=1 nontrivial-index w-chart.

    Coefficients are the explicit orbit averages in ``Delta_beta_closed_p1``
    (true zeros stay zero; no fallback to the beta=-pi/2 slice constants).
    At beta = -pi/2 this list equals ``_DELTA_TERMS``.
    """
    if p != SLICE_P:
        raise ValueError(f"only p={SLICE_P} slice implemented, got p={p}")
    cb2 = np.cos(0.5 * beta)
    sb2 = np.sin(0.5 * beta)
    raw: list[tuple[complex, tuple[int, int, int, int]]] = [
        (complex(cb2 * cb2, 0.0), (1, 1, 1, 1)),
        (complex(0.0, -0.5 * np.sin(beta)), (1, 0, 0, 0)),
        (complex(sb2 * sb2, 0.0), (0, 1, 0, 0)),
        (complex(0.0, 0.5 * np.sin(beta)), (0, 0, 1, 0)),
    ]
    terms: list[tuple[complex, tuple[int, int, int, int]]] = []
    for coef, e in raw:
        c = complex(coef)
        if abs(c) <= b_tol:
            c = 0.0 + 0.0j
        terms.append((c, e))
    return terms


def moment_vec(beta: float = SLICE_BETA, p: int = SLICE_P) -> np.ndarray:
    """First moments m_alpha = (1/i) grad log Delta(0; beta) on the chart."""
    w0 = np.zeros(4, dtype=complex)
    return grad_log_Delta(w0, beta=beta, p=p) / (1j)


def Delta_beta_closed_p1(w: np.ndarray, beta: float) -> complex:
    """Closed p=1 divisor with explicit beta (test / cross-check helper)."""
    w = np.asarray(w, dtype=complex)
    cb2 = np.cos(0.5 * beta)
    sb2 = np.sin(0.5 * beta)
    return (
        (cb2 * cb2) * np.exp(0.5j * np.sum(w))
        - 0.5j * np.sin(beta) * np.exp(0.5j * w[0])
        + (sb2 * sb2) * np.exp(0.5j * w[1])
        + 0.5j * np.sin(beta) * np.exp(0.5j * w[2])
    )


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
    return hess_Delta(w, beta=SLICE_BETA, p=SLICE_P)


def hess_Delta(
    w: np.ndarray,
    beta: float = SLICE_BETA,
    p: int = SLICE_P,
) -> np.ndarray:
    """Hessian of Delta(w; beta) on the nontrivial-index chart."""
    w = np.asarray(w, dtype=complex)
    terms = delta_terms_nontrivial_index_set(beta=beta, p=p)
    H = np.zeros((4, 4), dtype=complex)
    for coef, e in terms:
        E = np.exp(0.5j * np.dot(e, w))
        for a in range(4):
            if not e[a]:
                continue
            for b in range(4):
                if not e[b]:
                    continue
                H[a, b] += coef * E * (0.5j) * e[a] * (0.5j) * e[b]
    return H


def log_hess_beta(
    w: np.ndarray,
    beta: float = SLICE_BETA,
    p: int = SLICE_P,
) -> np.ndarray:
    """Dg(w) = Hessian of log Delta(w; beta)."""
    d = Delta(w, beta=beta, p=p)
    gd = grad_Delta(w, beta=beta, p=p)
    Hd = hess_Delta(w, beta=beta, p=p)
    return Hd / d - np.outer(gd, gd) / (d * d)


def log_grad(w: np.ndarray, beta: float = SLICE_BETA, p: int = SLICE_P) -> np.ndarray:
    """g(w) = grad log Delta(w; beta)."""
    return grad_log_Delta(w, beta=beta, p=p)


def log_hess(w: np.ndarray) -> np.ndarray:
    """Legacy name: Hessian of log Delta at beta=-pi/2."""
    return log_hess_beta(w, beta=SLICE_BETA, p=SLICE_P)


@dataclass
class WSaddleSystem:
    """Saddle system for fixed r, gamma on the p=1, q=1 w-chart.

    Attributes
    ----------
    r     :  random clause density / 2^k coefficient (the BM24 r).
    gamma :  slice parameter; the seed chamber of memo Sec. 14-16 is
             gamma in a small neighbourhood of 0 with gamma < 0.
    beta  :  QAOA mixer angle (default -pi/2 reproduces the memo slice).
    p     :  QAOA depth (only p=1 supported).
    """
    r: float
    gamma: float
    beta: float = SLICE_BETA
    p: int = SLICE_P

    def __post_init__(self) -> None:
        if int(self.p) != SLICE_P:
            raise ValueError(f"only p={SLICE_P} implemented, got p={self.p}")

    @property
    def c(self) -> np.ndarray:
        return couplings(self.r, self.gamma)

    def leading_seed(self) -> np.ndarray:
        """w_alpha = -2 i c_alpha m_alpha + O(c_infty^2);  memo Eq. (5.7).

        At gamma = 0 this is identically zero, which is the correct
        (degenerate) saddle.  For nonzero gamma it gives an O(c_infty)
        initial guess for the Newton / Krawczyk certifier.
        """
        m = moment_vec(beta=self.beta, p=self.p)
        return -2.0j * self.c * m

    def F_complex(self, w: np.ndarray) -> np.ndarray:
        """Rescaled saddle equation F_tilde(w) = w + 2 C g(w)."""
        return w + 2.0 * self.c * grad_log_Delta(w, beta=self.beta, p=self.p)

    def J_complex(self, w: np.ndarray) -> np.ndarray:
        """Jacobian of F_tilde:  I + 2 C Dg(w).

        Equals the rescaled Hessian H_tilde of memo Eq. (6.2) when
        evaluated at a saddle.  Equals I at w = 0, gamma = 0.
        """
        return np.eye(4, dtype=complex) + np.diag(2.0 * self.c) @ log_hess_beta(
            w, beta=self.beta, p=self.p
        )

    def F_real(self, x: np.ndarray) -> np.ndarray:
        """Real-packed F for scipy.optimize.root:  x = [Re w | Im w]."""
        w = x[:4] + 1j * x[4:]
        f = self.F_complex(w)
        return np.concatenate([f.real, f.imag])

    def Phi_eff(self, w: np.ndarray, *, log_delta: Optional[complex] = None) -> complex:
        """Effective parent action Phi_eff(w);  memo Eq. (4.1).

        Uses the principal branch of ``log Delta(w)`` when ``log_delta`` is
        omitted.  For Stokes / branch-wise analysis along a gamma mesh,
        pass a lifted ``log_delta`` from ``lift_log_delta_along_mesh`` instead
        of relying on raw principal ``Im Phi_eff``.
        """
        c = self.c
        quad = np.sum(w * w / (4.0 * c))
        if log_delta is None:
            log_delta = np.log(Delta(w, beta=self.beta, p=self.p))
        return quad + log_delta

    def quad_term(self, w: np.ndarray) -> complex:
        """Quadratic part sum_alpha w_alpha^2 / (4 c_alpha)."""
        return np.sum(w * w / (4.0 * self.c))


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
# 5. Interval evaluation of F_tilde, its Jacobian, and Delta(w; beta)
# =====================================================================
def F_jacobian_delta_iv(
    w_box: List[iv.mpc],
    r: float,
    gamma,
    beta: float = SLICE_BETA,
    p: int = SLICE_P,
):
    """On the interval box w_box, return (F_iv, J_iv, Delta_iv).

    F_iv      : list of 4 iv.mpc, enclosing F_tilde(w_box).
    J_iv      : 4x4 list of iv.mpc, enclosing dF_tilde / dw on w_box.
                Equals  I + 2 C(gamma) Dg(w_box)  in interval arithmetic.
    Delta_iv  : iv.mpc, enclosing Delta(w; beta) on w_box.

    All three are computed in a single DualIV pass so the interval
    enclosure is consistent across the three outputs.
    """
    n = 4
    w_dual = [DualIV.var(w_box[i], n, i) for i in range(n)]
    half_i = iv.mpc(iv.mpf(0), iv.mpf([0.5, 0.5]))
    terms = delta_terms_nontrivial_index_set(beta=beta, p=p)

    # Pass 1:  Delta and partial_alpha Delta as DualIVs.
    Delta = DualIV.const(iv.mpc(0), n)
    dDelta = [DualIV.const(iv.mpc(0), n) for _ in range(n)]
    for coef, e in terms:
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


def Phi_eff_iv(
    w_box: List[iv.mpc],
    r: float,
    gamma,
    beta: float = SLICE_BETA,
    p: int = SLICE_P,
) -> iv.mpc:
    """Interval enclosure of Phi_eff(w) = sum w^2/(4c) + log(Delta(w; beta)) on w_box."""
    n = 4
    w_dual = [DualIV.var(w_box[i], n, i) for i in range(n)]
    half_i = iv.mpc(iv.mpf(0), iv.mpf([0.5, 0.5]))
    terms = delta_terms_nontrivial_index_set(beta=beta, p=p)

    Delta = DualIV.const(iv.mpc(0), n)
    for coef, e in terms:
        exponent = DualIV.const(iv.mpf(0), n)
        for a in range(n):
            if e[a]:
                exponent = exponent + DualIV.const(iv.mpc(int(e[a]), 0), n) * w_dual[a]
        exponent = DualIV.const(half_i, n) * exponent
        E = exponent.exp()
        Delta = Delta + DualIV.const(_to_iv_complex(coef), n) * E

    c_iv = _couplings_iv(r, gamma)
    quad = DualIV.const(iv.mpc(0), n)
    for a in range(n):
        quad = quad + (w_dual[a] * w_dual[a]) / (DualIV.const(iv.mpc(4), n) * c_iv[a])

    return quad.val + iv.log(Delta.val)


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


def ratio_iv_avoids_negative_real_axis(ratio_iv: iv.mpc) -> bool:
    """True if the iv enclosure of a ratio cannot intersect (-infty, 0] on R.

    Sufficient for principal ``log(ratio)`` to be analytic on the certified
    step: the ratio does not cross the standard log branch cut.
    """
    re_a = float(ratio_iv.real.a)
    im_a = float(ratio_iv.imag.a)
    im_b = float(ratio_iv.imag.b)
    if im_a <= 0.0 <= im_b and re_a <= 0.0:
        return False
    return True


def delta_iv_on_certified_box(
    sys: WSaddleSystem,
    w_center: np.ndarray,
    box_radius: float,
    *,
    dps: int,
) -> iv.mpc:
    """Interval enclosure of Delta(w; beta) on the Krawczyk hyperrectangle."""
    mp.mp.dps = int(dps)
    iv.dps = int(dps)
    w_box = [_box_iv(w_center[i], box_radius) for i in range(4)]
    _, _, delta_iv = F_jacobian_delta_iv(
        w_box, sys.r, sys.gamma, beta=sys.beta, p=sys.p
    )
    return delta_iv


def principal_log_delta_ratio(delta_curr: complex, delta_prev: complex) -> complex:
    """Principal Log(Delta_curr / Delta_prev) for one mesh increment."""
    return np.log(delta_curr / delta_prev)


def certify_log_delta_ratio_step(
    sys_prev: WSaddleSystem,
    w_prev: np.ndarray,
    info_prev: dict[str, Any],
    sys_curr: WSaddleSystem,
    w_curr: np.ndarray,
    info_curr: dict[str, Any],
) -> tuple[bool, dict[str, Any]]:
    """Certify one continuation step of the lifted log Delta branch.

    Requires certified boxes at consecutive mesh points.  Certifies that the
    interval enclosure of Delta(X_{j+1}) / Delta(X_j) avoids the negative real
    axis, so principal log of the ratio is a valid analytic increment:

        ell_{j+1} = ell_j + Log_principal(Delta(w_{j+1}) / Delta(w_j)).
    """
    dps = int(info_curr.get("dps_used", info_prev.get("dps_used", 60)))
    rho_prev = float(info_prev["box_radius"])
    rho_curr = float(info_curr["box_radius"])

    delta_prev_iv = delta_iv_on_certified_box(sys_prev, w_prev, rho_prev, dps=dps)
    delta_curr_iv = delta_iv_on_certified_box(sys_curr, w_curr, rho_curr, dps=dps)
    ratio_iv = delta_curr_iv / delta_prev_iv
    ok = ratio_iv_avoids_negative_real_axis(ratio_iv)

    d_prev = Delta(w_prev, beta=sys_prev.beta, p=sys_prev.p)
    d_curr = Delta(w_curr, beta=sys_curr.beta, p=sys_curr.p)
    log_ratio = principal_log_delta_ratio(d_curr, d_prev)

    return ok, {
        "log_ratio_real": float(log_ratio.real),
        "log_ratio_imag": float(log_ratio.imag),
        "ratio_re_interval": [float(ratio_iv.real.a), float(ratio_iv.real.b)],
        "ratio_im_interval": [float(ratio_iv.imag.a), float(ratio_iv.imag.b)],
        "log_branch_step_certified": bool(ok),
        "log_branch_method": "continuation_ratio_lift",
    }


def initial_log_delta_lift(w: np.ndarray, *, beta: float = SLICE_BETA, p: int = SLICE_P) -> complex:
    """Seed ell_0 = Log_principal(Delta(w)) on the first mesh point."""
    return np.log(Delta(w, beta=beta, p=p))


def phi_eff_with_lifted_log(
    sys: WSaddleSystem,
    w: np.ndarray,
    log_delta_lifted: complex,
) -> complex:
    """Phi_eff using a branch-lifted log Delta instead of principal log."""
    return sys.quad_term(w) + log_delta_lifted


def _action_enclosure_from_box(
    sys: WSaddleSystem,
    w_center: np.ndarray,
    radius: float,
    *,
    log_branch_method: str = "principal_log_Delta_on_box",
) -> dict[str, Any]:
    """Interval enclosure of Phi_eff on the certified hyperrectangle."""
    w_box = [_box_iv(w_center[i], radius) for i in range(4)]
    phi_iv = Phi_eff_iv(w_box, sys.r, sys.gamma, beta=sys.beta, p=sys.p)
    re_a, re_b = float(phi_iv.real.a), float(phi_iv.real.b)
    im_a, im_b = float(phi_iv.imag.a), float(phi_iv.imag.b)
    phi_c = sys.Phi_eff(w_center)
    return {
        "action_center_real": float(phi_c.real),
        "action_center_imag": float(phi_c.imag),
        "action_interval_real": [re_a, re_b],
        "action_interval_imag": [im_a, im_b],
        "action_width_real": float(re_b - re_a),
        "action_width_imag": float(im_b - im_a),
        "log_branch_method": log_branch_method,
        "Phi_eff_real": float(phi_c.real),
        "Phi_eff_imag": float(phi_c.imag),
    }


def krawczyk_certify_w_root(
    sys: WSaddleSystem,
    w_center: np.ndarray,
    dps: int = 60,
    max_inflate_iters: int = 6,
    enclose_action: bool = False,
    log_branch_method: str = "principal_log_Delta_on_box",
):
    """Rigorous Krawczyk certification of a unique root of F_tilde near w_center.

    Simultaneously certifies (memo Sec. 18 items 1-3 on a single box X):

      (a) Krawczyk inclusion K(X) subset int(X) with
          K(x) = x_0 - Y F(x_0) + (I - Y DF(X))(X - x_0),
          F = F_tilde_R, X = x_0 + [-rho, rho]^8 (realified w in C^4).
      (b) Nondegeneracy: if sup_{J in DF(X)} ||I - Y J||_inf < 1 and Y invertible,
          every J in DF(X) is invertible (Neumann series); the root in X is
          nondegenerate.  contraction_bound upper-bounds that sup norm (scaled).
      (c) Divisor separation: delta_lower > 0 on |Delta(w; beta)(X)|.

    The certified root lies in X; w_center is the numerical Newton center only.

    Returns (ok, info).  On success, info includes contraction_bound,
    delta_lower, inflate_iter, box_radius, w_center, and center Phi_eff values.
    If enclose_action, adds action_interval_* and action_width_* over X.
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
        F_box, J_box, Delta_box = F_jacobian_delta_iv(
            w_box, sys.r, sys.gamma, beta=sys.beta, p=sys.p
        )

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
            info: dict[str, Any] = {
                "contraction_bound": float(contraction),
                "delta_lower": float(delta_lower),
                "inflate_iter": int(it),
                "box_radius": float(radius),
                "w_center_real": w_center.real.tolist(),
                "w_center_imag": w_center.imag.tolist(),
                "Phi_eff_real": float(phi.real),
                "Phi_eff_imag": float(phi.imag),
                "dps_used": int(dps),
                "krawczyk_inclusion": "K(X)_subset_int(X)",
            }
            if enclose_action:
                info.update(
                    _action_enclosure_from_box(
                        sys,
                        w_center,
                        radius,
                        log_branch_method=log_branch_method,
                    )
                )
            return True, info

    return False, {"reason": "inclusion / contraction / divisor-separation failed"}


def krawczyk_certify_with_escalation(
    sys: WSaddleSystem,
    w_center: np.ndarray,
    dps_start: int = 60,
    max_inflate_iters: int = 6,
    max_levels: int = 4,
    enclose_action: bool = False,
    log_branch_method: str = "principal_log_Delta_on_box",
):
    """Try increasing precision until Krawczyk succeeds, or give up."""
    dps = int(dps_start)
    last: dict = {}
    for _ in range(max_levels):
        ok, info = krawczyk_certify_w_root(
            sys,
            w_center,
            dps=dps,
            max_inflate_iters=max_inflate_iters,
            enclose_action=enclose_action,
            log_branch_method=log_branch_method,
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
                    "(p=1, q=1 w-chart; default beta=-pi/2)."
    )
    parser.add_argument("--r", type=float, required=True,
                        help="BM24 r coefficient (random clause density / 2^k).")
    parser.add_argument("--gamma", type=float, required=True,
                        help="Slice parameter; seed chamber is gamma < 0 small.")
    parser.add_argument("--beta", type=float, default=SLICE_BETA,
                        help="Mixer angle beta (default -pi/2).")
    parser.add_argument("--num-starts", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tol", type=float, default=1e-12)
    parser.add_argument("--dps", type=int, default=60,
                        help="mpmath decimal precision for the rigorous certifier.")
    parser.add_argument("--max-inflate-iters", type=int, default=6)
    parser.add_argument("--escalate", action="store_true",
                        help="Try increasing dps if Krawczyk fails.")
    parser.add_argument("--out", type=str, default="")
    parser.add_argument(
        "--enclose-action",
        action="store_true",
        help="Interval-enclose Phi_eff on the certified Krawczyk box.",
    )
    args = parser.parse_args()

    sys = WSaddleSystem(r=args.r, gamma=args.gamma, beta=args.beta)
    roots_w = discover_w_roots(
        sys, num_starts=args.num_starts, seed=args.seed, tol=args.tol
    )

    payload = {
        "r": args.r,
        "gamma": args.gamma,
        "beta": args.beta,
        "p": sys.p,
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
                sys,
                w,
                dps_start=args.dps,
                max_inflate_iters=args.max_inflate_iters,
                enclose_action=args.enclose_action,
            )
        else:
            cert, info = krawczyk_certify_w_root(
                sys,
                w,
                dps=args.dps,
                max_inflate_iters=args.max_inflate_iters,
                enclose_action=args.enclose_action,
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

    w0 = np.zeros(4, dtype=complex)
    for beta in (-np.pi / 2, -1.0, -0.3, 0.0, 0.7):
        d0 = Delta(w0, beta=beta)
        assert abs(d0 - 1.0) < 1e-12, f"Delta(0, beta={beta}) = {d0}, expected 1"
        d_closed = Delta_beta_closed_p1(w0, beta)
        assert abs(d0 - d_closed) < 1e-12
    print("  Delta(0, beta) = 1 and matches Delta_beta_closed_p1   [OK]")

    rng = np.random.default_rng(0)
    for beta in (-np.pi / 2, -1.0, -0.3, 0.0, 0.7):
        w = 0.05 * (rng.normal(size=4) + 1j * rng.normal(size=4))
        d_gen = Delta(w, beta=beta)
        d_closed = Delta_beta_closed_p1(w, beta)
        assert abs(d_gen - d_closed) < 1e-12
    print("  Delta(w,beta) matches closed p=1 formula (sample betas)   [OK]")

    assert np.max(np.abs(moment_vec() - M_VEC)) < 1e-14
    g0 = grad_log_Delta(w0, beta=SLICE_BETA)
    assert np.max(np.abs(g0 - 1j * M_VEC)) < 1e-14
    print("  g(0) = i*m at beta=-pi/2                        [OK]")

    w = rng.normal(size=4) + 1j * rng.normal(size=4)
    assert abs(Delta(w, beta=SLICE_BETA) - Delta_star(w)) < 1e-14
    assert np.max(np.abs(grad_Delta(w, beta=SLICE_BETA) - grad_Delta_star(w))) < 1e-14
    assert np.max(np.abs(hess_Delta(w, beta=SLICE_BETA) - hess_Delta_star(w))) < 1e-14
    print("  beta=-pi/2: Delta/grad/hess match legacy *_star   [OK]")

    r, gamma = 1.0, -0.05
    sys = WSaddleSystem(r=r, gamma=gamma, beta=SLICE_BETA)
    w_seed = sys.leading_seed()
    print(f"  r={r}, gamma={gamma}, beta={sys.beta}")
    print(f"  leading_seed = {w_seed}")
    print(f"  ||F_tilde(seed)||_inf = {np.linalg.norm(sys.F_complex(w_seed), ord=np.inf):.3e}"
          f"   (should be O(c_infty^2))")

    m = moment_vec(beta=sys.beta, p=sys.p)
    leading_action_expected = (r / 4.0) * (3.0 * np.sin(gamma / 4.0) ** 2 - np.sin(gamma / 2.0))
    leading_action_formula = float(np.sum(sys.c * m ** 2).real)
    err = abs(leading_action_expected - leading_action_formula)
    assert err < 1e-14, f"Eq. (5.9) mismatch: {err}"
    print(f"  Eq. (5.9) leading action  = {leading_action_expected:.10f}   "
          f"   matches sum c m^2 = {leading_action_formula:.10f}   [OK]")

    roots = discover_w_roots(sys, num_starts=30, seed=0)
    print(f"  Found {len(roots)} numerical roots (beta=-pi/2).")
    if roots:
        w_star = roots[0]
        print(f"  w_star = {w_star}")
        print(f"  ||F_tilde(w_star)||_inf = {np.linalg.norm(sys.F_complex(w_star), ord=np.inf):.3e}")
        ok, info = krawczyk_certify_w_root(sys, w_star, dps=60)
        print(f"  Krawczyk certified (beta=-pi/2): {ok}")
        for k, v in info.items():
            if k not in ("w_center_real", "w_center_imag"):
                print(f"     {k}: {v}")

        gammas = np.linspace(-0.01, -0.2, 8)
        w_cont = w_star.copy()
        info_cont = dict(info)
        sys_cont = sys
        log_lift = initial_log_delta_lift(w_cont)
        steps_ok = 0
        for g_next in gammas[1:]:
            sys_next = WSaddleSystem(r=r, gamma=float(g_next), beta=SLICE_BETA)
            w_next, ok_n, _ = solve_w_from_init(sys_next, w_cont)
            assert ok_n
            ok_c, info_c = krawczyk_certify_w_root(sys_next, w_next, dps=60)
            assert ok_c
            step_ok, step_info = certify_log_delta_ratio_step(
                sys_cont, w_cont, info_cont, sys_next, w_next, info_c
            )
            assert step_ok, step_info
            log_lift += complex(step_info["log_ratio_real"], step_info["log_ratio_imag"])
            w_cont, info_cont, sys_cont = w_next, info_c, sys_next
            steps_ok += 1
        phi_principal = sys_cont.Phi_eff(w_cont)
        phi_lifted = phi_eff_with_lifted_log(sys_cont, w_cont, log_lift)
        assert abs(phi_lifted - phi_principal) < 1e-8 or abs(
            (phi_lifted - phi_principal) - 2j * np.pi
        ) < 1e-6, (
            f"lifted vs principal mismatch: {phi_lifted} vs {phi_principal}"
        )
        print(f"  log Delta ratio-lift on {steps_ok} steps (gamma mesh)   [OK]")

    sys_b = WSaddleSystem(r=r, gamma=gamma, beta=-1.0)
    roots_b = discover_w_roots(sys_b, num_starts=30, seed=0)
    print(f"  Found {len(roots_b)} numerical roots (beta=-1.0).")
    if roots_b:
        w_b = roots_b[0]
        ok_b, info_b = krawczyk_certify_w_root(sys_b, w_b, dps=60)
        res_b = float(np.linalg.norm(sys_b.F_complex(w_b), ord=np.inf))
        print(f"  beta=-1.0: certified={ok_b}, residual={res_b:.3e}, delta_lower={info_b.get('delta_lower')}")
        assert res_b < 1e-8
        assert ok_b
        assert float(info_b.get("delta_lower", 0)) > 0

    print("=== self-test done ===")


if __name__ == "__main__":
    import sys as _sys

    if len(_sys.argv) == 1 or (len(_sys.argv) == 2 and _sys.argv[1] == "--selftest"):
        selftest()
    else:
        from phasecraft.w_saddle.cli import main as run_main

        run_main(["certify", *_sys.argv[1:]])
