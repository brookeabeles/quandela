r"""
krawczyk_w_saddle_q.py
======================

Rigorous Krawczyk certification of parent-action QAOA saddle points on the
BM24 p = 1 nontrivial-index chart, for **general q >= 1** (q = 3 in particular).

This is the general-q replacement for the q = 1-only ``core.py``
(``phasecraft.w_saddle.core``).  Nothing in the mathematics is new theory: the
only change is *which* of the two saddle equations is used to eliminate the
auxiliary field u.  See the module docstring section "WHY core.py WAS q = 1
ONLY" below.

------------------------------------------------------------------------
The parent action (memo Eq. 14; d := 2q):

    Phi(u, w) = sum_a c_a u_a^{2q} + i sum_a w_a u_a + log Delta(w),

    Delta(w) = sum_s b_s exp(i sum_a w_a A_{a,s})        (q-independent).

Saddle equations:

    (I)   dPhi/du_a = 2q c_a u_a^{2q-1} + i w_a       = 0
    (II)  dPhi/dw_a = i u_a + g_a(w)                   = 0,
          with g_a(w) := d/dw_a log Delta(w).

------------------------------------------------------------------------
WHY core.py WAS q = 1 ONLY
--------------------------
core.py certifies a *reduced* system in w alone.  It built that reduction by
solving (I) for u:

    u_a = ( -i w_a / (2q c_a) )^{1/(2q-1)}                 [FRACTIONAL ROOT]

For q = 1 the exponent is 1/(2q-1) = 1 and the root is trivial, so the reduced
equation w_a + 2 c_a g_a(w) = 0 is rational/holomorphic.  For q >= 2 that
1/(2q-1) power is a genuine branch -- exactly the "fractional 2q-root" pain the
memo set out to avoid -- and the reduction stops being single-valued.

THE FIX (this module).  Eliminate u via the *other* equation (II), which is
LINEAR in u and therefore root-free:

    u_a = i g_a(w).

Substituting into (I) and dividing by i gives a w-only system whose only
w-dependence is through INTEGER powers of g_a:

    F_tilde_a(w) = w_a + kappa_q c_a g_a(w)^{2q-1} = 0,
    kappa_q := 2q (-1)^{q-1}     (kappa_1=2, kappa_2=-4, kappa_3=6, ...).

For q = 1 this is exactly core.py's w + 2 C g.  For q = 3 it is
w + 6 c g^5 -- no fractional root anywhere.  g_a is holomorphic wherever
Delta(w) != 0, certified on the box by the divisor-separation bound
|Delta| >= delta > 0.  This realises the memo's stated advantage verbatim.

Nondegeneracy transfer.  The reduced root w* corresponds bijectively to the
full saddle (u*, w*) = (i g(w*), w*) because (II) is linear in u.  The full
Jacobian [[A, iI],[iI, H]] (A = d_uF^u, H = d_wF^w = Hess log Delta) is
invertible iff the reduced Jacobian I + kappa_q c (2q-1) g^{2q-2} H is
(Schur complement w.r.t. the always-invertible iI block).  So a Krawczyk
certificate of the reduced system certifies existence/uniqueness/nondegeneracy
of the full saddle.  ``FullUWSaddleSystem`` below certifies the full 2|A|
system directly as an independent cross-check.

------------------------------------------------------------------------
SOUNDNESS
---------
Every quantity entering the Krawczyk inclusion/contraction test is evaluated in
mpmath interval arithmetic (``mpmath.iv``) via a forward-mode interval-AD
(``DualIV``):
  * F_tilde and its Jacobian over the fattened box  -> rigorous enclosures;
  * F_tilde at the CENTER is enclosed over the *thin* box (interval, NOT a bare
    float) so rounding in the center evaluation is bounded  -- this closes a
    small rigor gap present in core.py, which used a numpy float for F(center);
  * Delta over the box is enclosed and a strictly positive lower bound on
    |Delta| is required (divisor separation).
The preconditioner Y is an ordinary numerical inverse of the center Jacobian;
Krawczyk only needs Y invertible (checked) -- its accuracy affects only the
*size* of the certified box, never soundness, because the contraction norm
||I - Y DF(X)|| is bounded rigorously in interval arithmetic.

Complex -> real norm factor.  For a complex matrix M acting on the hyperrect
box (each coord a square of half-side rho in C ~ R^2), the realified inf-norm
||M_R||_inf <= sqrt(2) ||M||_{inf,C}, since |Re|+|Im| <= sqrt(2)|.|.  The
contraction bound uses this sqrt(2) factor (same as core.py).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import numpy as np
import mpmath as mp
from mpmath import iv
from scipy.optimize import root


# =====================================================================
# 1. Slice data (p = 1 nontrivial-index chart). q-INDEPENDENT.
#    Copied verbatim in spirit from core.py; the divisor and couplings
#    do not depend on q -- only the saddle-equation exponents do.
# =====================================================================
SLICE_BETA = -np.pi / 2
SLICE_P = 1

# First moments m_a = sum_s b_s A_{a,s} at beta=-pi/2 (memo Eq. 5.6); used only
# for a closed-form selftest cross-check.  General-beta moments are computed
# from grad log Delta(0).
M_VEC_SLICE = np.array([0.25 + 0.25j, 0.5 + 0.0j, 0.25 - 0.25j, 0.25 + 0.0j], dtype=complex)


def delta_terms_nontrivial_index_set(
    beta: float = SLICE_BETA, p: int = SLICE_P, *, b_tol: float = 1e-12
) -> List[Tuple[complex, Tuple[int, int, int, int]]]:
    """BM24 parent divisor terms on the p=1 nontrivial-index w-chart (general beta)."""
    if p != SLICE_P:
        raise ValueError(f"only p={SLICE_P} slice implemented, got p={p}")
    cb2 = np.cos(0.5 * beta)
    sb2 = np.sin(0.5 * beta)
    raw = [
        (complex(cb2 * cb2, 0.0), (1, 1, 1, 1)),
        (complex(0.0, -0.5 * np.sin(beta)), (1, 0, 0, 0)),
        (complex(sb2 * sb2, 0.0), (0, 1, 0, 0)),
        (complex(0.0, 0.5 * np.sin(beta)), (0, 0, 1, 0)),
    ]
    terms = []
    for coef, e in raw:
        c = complex(coef)
        if abs(c) <= b_tol:
            c = 0.0 + 0.0j
        terms.append((c, e))
    return terms


def couplings(r: float, gamma: float) -> np.ndarray:
    """Couplings c_0..c_3 (memo Eq. 5.5). q-INDEPENDENT."""
    return np.array(
        [
            r * (1.0 - np.exp(-0.5j * gamma)),
            4.0 * r * np.sin(0.25 * gamma) ** 2,
            r * (1.0 - np.exp(0.5j * gamma)),
            -4.0 * r * np.sin(0.25 * gamma) ** 2,
        ],
        dtype=complex,
    )


def Delta(w: np.ndarray, beta: float = SLICE_BETA, p: int = SLICE_P) -> complex:
    w = np.asarray(w, dtype=complex)
    total = 0.0 + 0.0j
    for coef, e in delta_terms_nontrivial_index_set(beta=beta, p=p):
        total += coef * np.exp(0.5j * np.dot(e, w))
    return total


def grad_Delta(w: np.ndarray, beta: float = SLICE_BETA, p: int = SLICE_P) -> np.ndarray:
    w = np.asarray(w, dtype=complex)
    g = np.zeros(4, dtype=complex)
    for coef, e in delta_terms_nontrivial_index_set(beta=beta, p=p):
        E = np.exp(0.5j * np.dot(e, w))
        for a in range(4):
            if e[a]:
                g[a] += coef * E * 0.5j * e[a]
    return g


def hess_Delta(w: np.ndarray, beta: float = SLICE_BETA, p: int = SLICE_P) -> np.ndarray:
    w = np.asarray(w, dtype=complex)
    H = np.zeros((4, 4), dtype=complex)
    for coef, e in delta_terms_nontrivial_index_set(beta=beta, p=p):
        E = np.exp(0.5j * np.dot(e, w))
        for a in range(4):
            if not e[a]:
                continue
            for b in range(4):
                if not e[b]:
                    continue
                H[a, b] += coef * E * (0.5j) * e[a] * (0.5j) * e[b]
    return H


def grad_log_Delta(w: np.ndarray, beta: float = SLICE_BETA, p: int = SLICE_P) -> np.ndarray:
    return grad_Delta(w, beta=beta, p=p) / Delta(w, beta=beta, p=p)


def hess_log_Delta(w: np.ndarray, beta: float = SLICE_BETA, p: int = SLICE_P) -> np.ndarray:
    d = Delta(w, beta=beta, p=p)
    gd = grad_Delta(w, beta=beta, p=p)
    Hd = hess_Delta(w, beta=beta, p=p)
    return Hd / d - np.outer(gd, gd) / (d * d)


def moment_vec(beta: float = SLICE_BETA, p: int = SLICE_P) -> np.ndarray:
    """m_a = (1/i) grad log Delta(0)."""
    return grad_log_Delta(np.zeros(4, dtype=complex), beta=beta, p=p) / 1j


def kappa_q(q: int) -> float:
    """kappa_q = 2q (-1)^{q-1}: the reduced-equation coupling prefactor."""
    return 2.0 * q * ((-1.0) ** (q - 1))


# =====================================================================
# 2. Interval automatic differentiation (DualIV) -- adds pow_int over core.py
# =====================================================================
def _iv_pow_int(x, n: int):
    """Integer power of an iv.mpc by binary exponentiation (rigorous)."""
    if n < 0:
        raise ValueError("n must be >= 0")
    out = iv.mpc(1)
    base = x
    k = n
    while k > 0:
        if k & 1:
            out = out * base
        base = base * base
        k >>= 1
    return out


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
        return other if isinstance(other, DualIV) else DualIV.const(other, len(self.deriv))

    def __add__(self, other):
        o = self._promote(other)
        return DualIV(self.val + o.val, [self.deriv[i] + o.deriv[i] for i in range(len(self.deriv))])
    __radd__ = __add__

    def __sub__(self, other):
        o = self._promote(other)
        return DualIV(self.val - o.val, [self.deriv[i] - o.deriv[i] for i in range(len(self.deriv))])

    def __rsub__(self, other):
        return DualIV.const(other, len(self.deriv)) - self

    def __neg__(self):
        return DualIV(-self.val, [-d for d in self.deriv])

    def __mul__(self, other):
        o = self._promote(other)
        return DualIV(
            self.val * o.val,
            [self.val * o.deriv[i] + self.deriv[i] * o.val for i in range(len(self.deriv))],
        )
    __rmul__ = __mul__

    def __truediv__(self, other):
        o = self._promote(other)
        denom_sq = o.val * o.val
        return DualIV(
            self.val / o.val,
            [(self.deriv[i] * o.val - self.val * o.deriv[i]) / denom_sq for i in range(len(self.deriv))],
        )

    def __rtruediv__(self, other):
        return DualIV.const(other, len(self.deriv)) / self

    def exp(self):
        ev = iv.exp(self.val)
        return DualIV(ev, [ev * d for d in self.deriv])

    def pow_int(self, n: int):
        """x^n for integer n >= 0, with rigorous derivative n x^{n-1} x'."""
        if n == 0:
            return DualIV.const(iv.mpc(1), len(self.deriv))
        if n == 1:
            return DualIV(self.val, self.deriv[:])
        v = _iv_pow_int(self.val, n)
        fac = iv.mpc(n) * _iv_pow_int(self.val, n - 1)
        return DualIV(v, [fac * d for d in self.deriv])


# =====================================================================
# 3. Interval helpers
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
    r_iv = iv.mpf([float(r), float(r)])
    g_iv = iv.mpf([float(gamma), float(gamma)]) if isinstance(gamma, (int, float)) else gamma
    e_minus = iv.exp(iv.mpc(iv.mpf(0), -g_iv / iv.mpf(2)))
    e_plus = iv.exp(iv.mpc(iv.mpf(0), g_iv / iv.mpf(2)))
    one_c = iv.mpc(iv.mpf(1), iv.mpf(0))
    r_c = iv.mpc(r_iv, iv.mpf(0))
    c0 = r_c * (one_c - e_minus)
    c2 = r_c * (one_c - e_plus)
    s_q = iv.sin(g_iv / iv.mpf(4))
    c1_r = iv.mpf(4) * r_iv * s_q * s_q
    return [c0, iv.mpc(c1_r, iv.mpf(0)), c2, iv.mpc(-c1_r, iv.mpf(0))]


def _delta_and_grad_duals(w_dual: List[DualIV], terms, n: int):
    """Return (Delta_dual, [dDelta_a_dual]) as DualIVs of the n variables.

    Delta_dual.val   = Delta(w),    Delta_dual.deriv[b] = d Delta / d w_b
    dDelta[a].val    = d Delta/dw_a, dDelta[a].deriv[b] = d^2 Delta/dw_a dw_b
    The w variables are assumed to be w_dual[w_offset + .] handled by caller;
    here w_dual already carries the correct partials.
    """
    Delta_d = DualIV.const(iv.mpc(0), n)
    dDelta = [DualIV.const(iv.mpc(0), n) for _ in range(4)]
    for coef, e in terms:
        exponent = DualIV.const(iv.mpc(0), n)
        for a in range(4):
            if e[a]:
                exponent = exponent + DualIV.const(iv.mpc(int(e[a]), 0), n) * w_dual[a]
        exponent = DualIV.const(iv.mpc(iv.mpf(0), iv.mpf([0.5, 0.5])), n) * exponent
        E = exponent.exp()
        coef_iv = _to_iv_complex(coef)
        Delta_d = Delta_d + DualIV.const(coef_iv, n) * E
        for a in range(4):
            if e[a]:
                fac = coef_iv * iv.mpc(iv.mpf(0), iv.mpf([0.5 * e[a], 0.5 * e[a]]))
                dDelta[a] = dDelta[a] + DualIV.const(fac, n) * E
    return Delta_d, dDelta


# =====================================================================
# 4. REDUCED w-only system, general q
#       F_tilde_a(w) = w_a + kappa_q c_a g_a(w)^{2q-1}
# =====================================================================
@dataclass
class WSaddleSystem:
    """Reduced w-only parent-action saddle on the p=1 chart, general q.

    For q = 1 this reproduces core.py's WSaddleSystem exactly.
    """
    r: float
    gamma: float
    q: int = 1
    beta: float = SLICE_BETA
    p: int = SLICE_P

    def __post_init__(self):
        if int(self.p) != SLICE_P:
            raise ValueError(f"only p={SLICE_P} implemented")
        if int(self.q) < 1:
            raise ValueError("q must be >= 1")
        self.q = int(self.q)

    @property
    def c(self) -> np.ndarray:
        return couplings(self.r, self.gamma)

    @property
    def kappa(self) -> float:
        return kappa_q(self.q)

    def leading_seed(self) -> np.ndarray:
        """w_a = -2q i c_a m_a^{2q-1} + O(c^2)  (general-q leading saddle)."""
        m = moment_vec(beta=self.beta, p=self.p)
        return -2.0 * self.q * 1j * self.c * m ** (2 * self.q - 1)

    def F_complex(self, w: np.ndarray) -> np.ndarray:
        g = grad_log_Delta(w, beta=self.beta, p=self.p)
        return w + self.kappa * self.c * g ** (2 * self.q - 1)

    def J_complex(self, w: np.ndarray) -> np.ndarray:
        g = grad_log_Delta(w, beta=self.beta, p=self.p)
        Dg = hess_log_Delta(w, beta=self.beta, p=self.p)
        pref = self.kappa * self.c * (2 * self.q - 1) * g ** (2 * self.q - 2)  # length-4
        return np.eye(4, dtype=complex) + np.diag(pref) @ Dg

    def F_real(self, x: np.ndarray) -> np.ndarray:
        w = x[:4] + 1j * x[4:]
        f = self.F_complex(w)
        return np.concatenate([f.real, f.imag])

    def Phi_eff(self, w: np.ndarray) -> complex:
        """Phi(u*,w*) with u eliminated; equals sum w^2/(4c)+log Delta only at q=1.

        For general q the on-shell action is
            Phi = sum_a [ c_a u_a^{2q} + i w_a u_a ] + log Delta,  u_a = i g_a(w).
        """
        g = grad_log_Delta(w, beta=self.beta, p=self.p)
        u = 1j * g
        poly = np.sum(self.c * u ** (2 * self.q) + 1j * w * u)
        return poly + np.log(Delta(w, beta=self.beta, p=self.p))


def _F_J_delta_iv_reduced(w_box, sys: WSaddleSystem):
    """Interval enclosure of (F_tilde, J=dF_tilde/dw, Delta) on w_box (n=4)."""
    n = 4
    terms = delta_terms_nontrivial_index_set(beta=sys.beta, p=sys.p)
    w_dual = [DualIV.var(w_box[i], n, i) for i in range(n)]
    Delta_d, dDelta = _delta_and_grad_duals(w_dual, terms, n)
    g_dual = [dDelta[a] / Delta_d for a in range(n)]      # g_a and dg_a/dw_b
    c_iv = _couplings_iv(sys.r, sys.gamma)
    kappa_iv = _to_iv_complex(sys.kappa)
    m = 2 * sys.q - 1
    F_dual = []
    for a in range(n):
        coef = DualIV.const(kappa_iv * c_iv[a], n)
        F_dual.append(w_dual[a] + coef * g_dual[a].pow_int(m))
    F_iv = [F_dual[a].val for a in range(n)]
    J_iv = [[F_dual[a].deriv[b] for b in range(n)] for a in range(n)]
    return F_iv, J_iv, Delta_d.val


# =====================================================================
# 5. FULL (u,w) system, general q  -- independent cross-check
#       F^u_a = 2q c_a u_a^{2q-1} + i w_a
#       F^w_a = i u_a + g_a(w)
#    variables x = (u_0..u_3, w_0..w_3), n = 8.
# =====================================================================
@dataclass
class FullUWSaddleSystem:
    r: float
    gamma: float
    q: int = 1
    beta: float = SLICE_BETA
    p: int = SLICE_P

    def __post_init__(self):
        if int(self.p) != SLICE_P:
            raise ValueError(f"only p={SLICE_P} implemented")
        self.q = int(self.q)

    @property
    def c(self) -> np.ndarray:
        return couplings(self.r, self.gamma)

    def uw_from_w(self, w: np.ndarray) -> np.ndarray:
        """Map a reduced-system w to the full (u,w) vector via u = i g(w)."""
        u = 1j * grad_log_Delta(w, beta=self.beta, p=self.p)
        return np.concatenate([u, w])

    def F_complex(self, x: np.ndarray) -> np.ndarray:
        u = x[:4]
        w = x[4:]
        g = grad_log_Delta(w, beta=self.beta, p=self.p)
        Fu = 2 * self.q * self.c * u ** (2 * self.q - 1) + 1j * w
        Fw = 1j * u + g
        return np.concatenate([Fu, Fw])

    def J_complex(self, x: np.ndarray) -> np.ndarray:
        u = x[:4]
        w = x[4:]
        Dg = hess_log_Delta(w, beta=self.beta, p=self.p)
        J = np.zeros((8, 8), dtype=complex)
        # dF^u/du = 2q(2q-1) c u^{2q-2}  (diagonal)
        diagA = 2 * self.q * (2 * self.q - 1) * self.c * u ** (2 * self.q - 2)
        J[:4, :4] = np.diag(diagA)
        J[:4, 4:] = 1j * np.eye(4)            # dF^u/dw
        J[4:, :4] = 1j * np.eye(4)            # dF^w/du
        J[4:, 4:] = Dg                        # dF^w/dw
        return J

    def F_real(self, xr: np.ndarray) -> np.ndarray:
        x = xr[:8] + 1j * xr[8:]
        f = self.F_complex(x)
        return np.concatenate([f.real, f.imag])

    def Phi(self, x: np.ndarray) -> complex:
        u = x[:4]
        w = x[4:]
        return np.sum(self.c * u ** (2 * self.q) + 1j * w * u) + np.log(
            Delta(w, beta=self.beta, p=self.p)
        )


def _F_J_delta_iv_full(x_box, sys: FullUWSaddleSystem):
    """Interval enclosure of (F, J, Delta) for the full (u,w) system (n=8)."""
    n = 8
    terms = delta_terms_nontrivial_index_set(beta=sys.beta, p=sys.p)
    u_dual = [DualIV.var(x_box[i], n, i) for i in range(4)]
    w_dual = [DualIV.var(x_box[4 + i], n, 4 + i) for i in range(4)]
    Delta_d, dDelta = _delta_and_grad_duals(w_dual, terms, n)
    g_dual = [dDelta[a] / Delta_d for a in range(4)]
    c_iv = _couplings_iv(sys.r, sys.gamma)
    twoq = _to_iv_complex(2 * sys.q)
    i_iv = iv.mpc(iv.mpf(0), iv.mpf([1.0, 1.0]))
    m = 2 * sys.q - 1
    F_dual = []
    for a in range(4):
        coef = DualIV.const(twoq * c_iv[a], n)
        F_dual.append(coef * u_dual[a].pow_int(m) + DualIV.const(i_iv, n) * w_dual[a])
    for a in range(4):
        F_dual.append(DualIV.const(i_iv, n) * u_dual[a] + g_dual[a])
    F_iv = [F_dual[i].val for i in range(n)]
    J_iv = [[F_dual[i].deriv[j] for j in range(n)] for i in range(n)]
    return F_iv, J_iv, Delta_d.val


# =====================================================================
# 6. Krawczyk test (shared by both systems)
# =====================================================================
def _matvec_iv(Y_np, v_iv):
    n = len(v_iv)
    out = []
    for i in range(n):
        acc = iv.mpc(0)
        for j in range(n):
            acc = acc + _to_iv_complex(Y_np[i, j]) * v_iv[j]
        out.append(acc)
    return out


def _matmul_iv(Y_np, J_iv):
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
    def lower_sq(x_iv):
        a = float(x_iv.a)
        b = float(x_iv.b)
        if a <= 0.0 <= b:
            return 0.0
        return min(a * a, b * b)
    return float(mp.sqrt(lower_sq(z_iv.real) + lower_sq(z_iv.imag)))


def _krawczyk_core(eval_iv, F_center_iv, Y, center, n, radii):
    """Run the Krawczyk inclusion/contraction loop given an interval evaluator.

    eval_iv(box) -> (F_iv, J_iv, Delta_iv).  F_center_iv is the rigorous thin-box
    enclosure of F(center).  Returns (ok, contraction, delta_lower, radius, iter).
    """
    for it, radius in enumerate(radii):
        box = [_box_iv(center[i], radius) for i in range(n)]
        _, J_box, Delta_box = eval_iv(box)
        Yg = _matvec_iv(Y, F_center_iv)
        YJ = _matmul_iv(Y, J_box)
        M = [[(iv.mpc(1) if i == j else iv.mpc(0)) - YJ[i][j] for j in range(n)] for i in range(n)]
        diff = [box[i] - _to_iv_complex(center[i]) for i in range(n)]
        term2 = []
        for i in range(n):
            acc = iv.mpc(0)
            for j in range(n):
                acc = acc + M[i][j] * diff[j]
            term2.append(acc)
        K_box = [_to_iv_complex(center[i]) - Yg[i] + term2[i] for i in range(n)]
        inclusion = all(_strict_inside(K_box[i], box[i]) for i in range(n))
        contraction = float(np.sqrt(2.0)) * _matrix_inf_norm_upper_iv(M)
        delta_lower = _abs_lower_iv(Delta_box)
        if inclusion and contraction < 1.0 and delta_lower > 0.0:
            return True, float(contraction), float(delta_lower), float(radius), int(it)
    return False, np.inf, 0.0, 0.0, -1


def _radius_schedule(yg_norm, dps, max_inflate_iters):
    """Descending radius grid. The Krawczyk loop returns the FIRST feasible
    radius; scanning large -> small therefore reports the LARGEST certifiable
    box (strongest isolation of the root). Feasibility is a window [r_min,r_max]
    -- too-large boxes fail by interval blow-up, too-small boxes fail because the
    Newton center-shift ||Y F(x0)|| exceeds the box -- so the first hit from
    above is r_max. The grid still descends to ~machine scale so that
    ultra-accurate Newton roots (tiny ||Y F||) certify on a correspondingly
    tiny box rather than not at all.
    """
    floor = 10.0 ** -(dps - 8)
    decades = [1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 1e-5, 1e-6,
               1e-7, 1e-8, 1e-9, 1e-10, 1e-12, 1e-14, 1e-16]
    return [r for r in decades if r >= floor]


def krawczyk_certify_reduced(sys: WSaddleSystem, w_center, dps=60, max_inflate_iters=6):
    """Certify a unique nondegenerate root of the reduced F_tilde near w_center."""
    mp.mp.dps = int(dps)
    iv.dps = int(dps)
    n = 4
    F_c = sys.F_complex(w_center)
    J_c = sys.J_complex(w_center)
    try:
        Y = np.linalg.inv(J_c)
    except np.linalg.LinAlgError:
        return False, {"reason": "singular analytic Jacobian at center"}
    yg_norm = float(np.linalg.norm(Y @ F_c, ord=np.inf))

    # Rigorous enclosure of F(center) over the THIN box (closes core.py's float gap).
    thin = [_box_iv(w_center[i], 0.0) for i in range(n)]
    F_center_iv, _, _ = _F_J_delta_iv_reduced(thin, sys)

    radii = _radius_schedule(yg_norm, dps, max_inflate_iters)
    ok, contraction, delta_lower, radius, it = _krawczyk_core(
        lambda box: _F_J_delta_iv_reduced(box, sys), F_center_iv, Y, w_center, n, radii
    )
    if not ok:
        return False, {"reason": "inclusion / contraction / divisor-separation failed"}
    phi = sys.Phi_eff(w_center)
    return True, {
        "system": "reduced_w_only",
        "q": sys.q,
        "contraction_bound": contraction,
        "delta_lower": delta_lower,
        "inflate_iter": it,
        "box_radius": radius,
        "w_center_real": w_center.real.tolist(),
        "w_center_imag": w_center.imag.tolist(),
        "Phi_real": float(phi.real),
        "Phi_imag": float(phi.imag),
        "dps_used": int(dps),
        "krawczyk_inclusion": "K(X)_subset_int(X)",
    }


def krawczyk_certify_full(sys: FullUWSaddleSystem, x_center, dps=60, max_inflate_iters=6):
    """Certify a unique nondegenerate root of the full (u,w) system near x_center."""
    mp.mp.dps = int(dps)
    iv.dps = int(dps)
    n = 8
    F_c = sys.F_complex(x_center)
    J_c = sys.J_complex(x_center)
    try:
        Y = np.linalg.inv(J_c)
    except np.linalg.LinAlgError:
        return False, {"reason": "singular analytic Jacobian at center"}
    yg_norm = float(np.linalg.norm(Y @ F_c, ord=np.inf))

    thin = [_box_iv(x_center[i], 0.0) for i in range(n)]
    F_center_iv, _, _ = _F_J_delta_iv_full(thin, sys)

    radii = _radius_schedule(yg_norm, dps, max_inflate_iters)
    ok, contraction, delta_lower, radius, it = _krawczyk_core(
        lambda box: _F_J_delta_iv_full(box, sys), F_center_iv, Y, x_center, n, radii
    )
    if not ok:
        return False, {"reason": "inclusion / contraction / divisor-separation failed"}
    phi = sys.Phi(x_center)
    return True, {
        "system": "full_uw",
        "q": sys.q,
        "contraction_bound": contraction,
        "delta_lower": delta_lower,
        "inflate_iter": it,
        "box_radius": radius,
        "Phi_real": float(phi.real),
        "Phi_imag": float(phi.imag),
        "dps_used": int(dps),
        "krawczyk_inclusion": "K(X)_subset_int(X)",
    }


def krawczyk_certify_reduced_escalate(sys, w_center, dps_start=60, max_levels=4, max_inflate_iters=6):
    dps = int(dps_start)
    last = {}
    for _ in range(max_levels):
        ok, info = krawczyk_certify_reduced(sys, w_center, dps=dps, max_inflate_iters=max_inflate_iters)
        if ok:
            return True, info
        last = info
        dps = int(np.ceil(dps * 1.5))
    out = dict(last)
    out["dps_attempted"] = dps
    return False, out


# =====================================================================
# 7. Root discovery (reduced system) + lift to full
# =====================================================================
def solve_w_from_init(sys: WSaddleSystem, w_init: np.ndarray, tol: float = 1e-12) -> tuple:
    """Newton-polish reduced system F_tilde(w)=0 from warm start w_init.

    Returns (w_star, converged, residual_inf).
    """
    x0 = np.concatenate([np.asarray(w_init, dtype=complex).real, np.asarray(w_init, dtype=complex).imag])
    sol = root(sys.F_real, x0, method="hybr", tol=tol)
    if not sol.success or not np.isfinite(sol.x).all():
        return np.asarray(w_init, dtype=complex), False, float(np.inf)
    w = sol.x[:4] + 1j * sol.x[4:]
    res = float(np.linalg.norm(sys.F_complex(w), ord=np.inf))
    return w, bool(res < 1e-8), res


def discover_w_roots(sys: WSaddleSystem, num_starts=200, seed=0, tol=1e-12):
    rng = np.random.default_rng(seed)
    roots: List[np.ndarray] = []

    def _try(x0):
        sol = root(sys.F_real, x0, method="hybr", tol=tol)
        if not sol.success or not np.isfinite(sol.x).all():
            return
        if np.linalg.norm(sys.F_real(sol.x), ord=np.inf) > 1e-7:
            return
        w = sol.x[:4] + 1j * sol.x[4:]
        if any(np.linalg.norm(w - wp, ord=np.inf) < 1e-5 for wp in roots):
            return
        roots.append(w)

    w0 = sys.leading_seed()
    _try(np.concatenate([w0.real, w0.imag]))
    scale = max(float(np.max(np.abs(w0))) * 5.0, 0.05)
    for _ in range(num_starts):
        _try(rng.normal(0.0, scale, size=8))
    return roots


def solve_uw_from_w(sys_full: FullUWSaddleSystem, w: np.ndarray, tol: float = 1e-13) -> tuple:
    """Lift a reduced-system w root to full (u,w) space and Newton-polish.

    Returns (x_star, converged, residual_inf) where x_star = (u_0..u_3, w_0..w_3).
    """
    x0 = sys_full.uw_from_w(np.asarray(w, dtype=complex))
    xr = np.concatenate([x0.real, x0.imag])
    sol = root(sys_full.F_real, xr, method="hybr", tol=tol)
    if not sol.success or not np.isfinite(sol.x).all():
        return x0, False, float(np.inf)
    x = sol.x[:8] + 1j * sol.x[8:]
    res = float(np.linalg.norm(sys_full.F_complex(x), ord=np.inf))
    return x, bool(res < 1e-8), res


def krawczyk_certify_full_escalate(
    sys: FullUWSaddleSystem,
    x_center: np.ndarray,
    dps_start: int = 60,
    max_levels: int = 4,
    max_inflate_iters: int = 6,
) -> tuple:
    """Escalating-precision full (u,w) system Krawczyk certification."""
    dps = int(dps_start)
    last: dict = {}
    for _ in range(max_levels):
        ok, info = krawczyk_certify_full(sys, x_center, dps=dps, max_inflate_iters=max_inflate_iters)
        if ok:
            return True, info
        last = info
        dps = int(np.ceil(dps * 1.5))
    out = dict(last)
    out["dps_attempted"] = dps
    return False, out


# =====================================================================
# 8. Self-test  (q=1 regression + q=3 + full/reduced cross-check)
# =====================================================================
def selftest() -> None:
    print("=== krawczyk_w_saddle_q self-test ===")

    # --- q = 1 regression against core.py's closed forms -----------------
    w0 = np.zeros(4, dtype=complex)
    for beta in (-np.pi / 2, -1.0, -0.3, 0.0, 0.7):
        assert abs(Delta(w0, beta=beta) - 1.0) < 1e-12
    assert np.max(np.abs(moment_vec() - M_VEC_SLICE)) < 1e-13
    print("  Delta(0)=1, moments match slice closed form (q-independent)   [OK]")

    r, gamma = 1.0, -0.05
    s1 = WSaddleSystem(r=r, gamma=gamma, q=1)
    # core.py form: F = w + 2 c g
    g0 = grad_log_Delta(w0)
    F_manual = w0 + 2.0 * s1.c * g0
    assert np.max(np.abs(s1.F_complex(w0) - F_manual)) < 1e-13
    print("  q=1 F_complex == core.py's w + 2 C g                          [OK]")
    # core.py leading-action closed form  (Eq. 5.9)
    m = moment_vec()
    leading_expected = (r / 4.0) * (3.0 * np.sin(gamma / 4.0) ** 2 - np.sin(gamma / 2.0))
    leading_formula = float(np.sum(s1.c * m ** 2).real)
    assert abs(leading_expected - leading_formula) < 1e-13
    print(f"  q=1 leading action {leading_expected:.10f} == sum c m^2       [OK]")

    # --- q = 1 certification --------------------------------------------
    roots1 = discover_w_roots(s1, num_starts=40, seed=0)
    assert roots1, "no q=1 root found"
    ok1, info1 = krawczyk_certify_reduced(s1, roots1[0], dps=60)
    print(f"  q=1 reduced Krawczyk certified: {ok1}  "
          f"(kappa={info1.get('contraction_bound'):.3e}, rho={info1.get('box_radius'):.2e}, "
          f"|Delta|>={info1.get('delta_lower'):.3e})")
    assert ok1

    # --- q = 3 certification (the point of this module) -----------------
    for (rr, gg) in [(1.0, -0.05), (1.0, -0.15), (50.0, -0.05)]:
        s3 = WSaddleSystem(r=rr, gamma=gg, q=3)
        assert abs(s3.kappa - 6.0) < 1e-15
        roots3 = discover_w_roots(s3, num_starts=60, seed=1)
        assert roots3, f"no q=3 root found for r={rr}, gamma={gg}"
        w3 = roots3[0]
        res3 = float(np.linalg.norm(s3.F_complex(w3), ord=np.inf))
        ok3, info3 = krawczyk_certify_reduced_escalate(s3, w3, dps_start=60)
        print(f"  q=3 r={rr}, gamma={gg}: root residual {res3:.2e}, certified={ok3}, "
              f"kappa={info3.get('contraction_bound'):.3e}, rho={info3.get('box_radius'):.2e}, "
              f"|Delta|>={info3.get('delta_lower'):.3e}")
        assert ok3, info3

        # --- full (u,w) cross-check at the SAME saddle -------------------
        sf = FullUWSaddleSystem(r=rr, gamma=gg, q=3)
        x3 = sf.uw_from_w(w3)
        # (a) full residual is also ~0 at the lifted point
        resf = float(np.linalg.norm(sf.F_complex(x3), ord=np.inf))
        assert resf < 1e-7, f"full residual {resf}"
        # (b) Newton-polish in full coords then certify the full 8-dim system
        solf = root(sf.F_real, np.concatenate([x3.real, x3.imag]), method="hybr", tol=1e-13)
        xf = solf.x[:8] + 1j * solf.x[8:]
        okf, infof = krawczyk_certify_full(sf, xf, dps=80)
        # (c) actions agree between reduced and full formulations
        phi_red = complex(info3["Phi_real"], info3["Phi_imag"])
        phi_full = sf.Phi(xf)
        dphi = abs(phi_red - phi_full)
        print(f"        full (u,w) certified={okf}, kappa={infof.get('contraction_bound', float('nan')):.3e}; "
              f"|Phi_reduced - Phi_full|={dphi:.2e}")
        assert okf, infof
        assert dphi < 1e-6, f"action mismatch {dphi}"

    # --- general-beta off-slice spot check at q=3 -----------------------
    sb = WSaddleSystem(r=1.0, gamma=-0.05, q=3, beta=-1.0)
    rb = discover_w_roots(sb, num_starts=60, seed=2)
    assert rb
    okb, infob = krawczyk_certify_reduced_escalate(sb, rb[0], dps_start=60)
    print(f"  q=3 off-slice beta=-1.0 certified: {okb} "
          f"(|Delta|>={infob.get('delta_lower'):.3e})")
    assert okb

    print("=== self-test PASSED ===")


# =====================================================================
# 9. CLI
# =====================================================================
def main() -> None:
    ap = argparse.ArgumentParser(
        description="General-q Krawczyk certifier for parent-action QAOA saddles (p=1 chart)."
    )
    ap.add_argument("--r", type=float, required=True)
    ap.add_argument("--gamma", type=float, required=True)
    ap.add_argument("--q", type=int, default=3)
    ap.add_argument("--beta", type=float, default=SLICE_BETA)
    ap.add_argument("--system", choices=["reduced", "full", "both"], default="both")
    ap.add_argument("--num-starts", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dps", type=int, default=60)
    ap.add_argument("--escalate", action="store_true")
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()

    s = WSaddleSystem(r=args.r, gamma=args.gamma, q=args.q, beta=args.beta)
    roots = discover_w_roots(s, num_starts=args.num_starts, seed=args.seed)
    payload = {"r": args.r, "gamma": args.gamma, "q": args.q, "beta": args.beta,
               "num_roots": len(roots), "roots": []}
    sf = FullUWSaddleSystem(r=args.r, gamma=args.gamma, q=args.q, beta=args.beta)
    for i, w in enumerate(roots):
        row: dict[str, Any] = {"idx": i, "w_real": w.real.tolist(), "w_imag": w.imag.tolist(),
                               "residual": float(np.linalg.norm(s.F_complex(w), ord=np.inf))}
        if args.system in ("reduced", "both"):
            if args.escalate:
                ok, info = krawczyk_certify_reduced_escalate(s, w, dps_start=args.dps)
            else:
                ok, info = krawczyk_certify_reduced(s, w, dps=args.dps)
            row["reduced_certified"] = bool(ok)
            row["reduced_info"] = info
        if args.system in ("full", "both"):
            x = sf.uw_from_w(w)
            sol = root(sf.F_real, np.concatenate([x.real, x.imag]), method="hybr", tol=1e-13)
            xf = sol.x[:8] + 1j * sol.x[8:]
            okf, infof = krawczyk_certify_full(sf, xf, dps=max(args.dps, 80))
            row["full_certified"] = bool(okf)
            row["full_info"] = infof
        payload["roots"].append(row)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    summary = {"r": args.r, "gamma": args.gamma, "q": args.q, "num_roots": len(roots),
               "reduced_certified": sum(1 for r in payload["roots"] if r.get("reduced_certified")),
               "full_certified": sum(1 for r in payload["roots"] if r.get("full_certified"))}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    import sys as _sys
    if len(_sys.argv) == 1 or (len(_sys.argv) == 2 and _sys.argv[1] == "--selftest"):
        selftest()
    else:
        main()
