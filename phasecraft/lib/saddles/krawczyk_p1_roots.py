"""
Krawczyk root search/certification for Phasecraft / BM24 saddle equations.

Equations match ``generalized_binomial_sum.PATCHED.py`` (BM24 Eq. A34 gamma signs),
the same source used by ``bm24_qaoa_sim.py``.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path
import sys

import mpmath as mp
from mpmath import iv
import numpy as np
from scipy.optimize import root

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_PATCHED_PATH = (
    Path(__file__).resolve().parent.parent
    / "ksat"
    / "variants"
    / "generalized_binomial_sum.PATCHED.py"
)
_KSAT_LIB = _PATCHED_PATH.parent.parent
_PATCHED_SPEC = importlib.util.spec_from_file_location(
    "phasecraft._generalized_binomial_sum_patched",
    _PATCHED_PATH,
    submodule_search_locations=[str(_PATCHED_PATH.parent), str(_KSAT_LIB)],
)
if _PATCHED_SPEC is None or _PATCHED_SPEC.loader is None:
    raise ImportError(f"Cannot load patched module at {_PATCHED_PATH}")
_PATCHED_MOD = importlib.util.module_from_spec(_PATCHED_SPEC)
sys.modules[_PATCHED_SPEC.name] = _PATCHED_MOD
_PATCHED_SPEC.loader.exec_module(_PATCHED_MOD)

B = _PATCHED_MOD.B
parent_function_alpha_sum_sos = _PATCHED_MOD.parent_function_alpha_sum_sos
parent_function_s_sum_sos = _PATCHED_MOD.parent_function_s_sum_sos

from phasecraft.optimal_angles import optimal_angles


@dataclass
class SaddleSystem:
    q: int
    r: float
    p: int
    betas: np.ndarray
    gammas: np.ndarray
    b: np.ndarray
    c_root: np.ndarray

    @classmethod
    def build(cls, q: int, r: float, betas: np.ndarray, gammas: np.ndarray) -> "SaddleSystem":
        p = int(len(betas))
        if len(gammas) != p:
            raise ValueError("betas and gammas must have same length p")
        all_s = np.arange(2 ** (2 * p + 1))
        b = 0.5 * B(np.array(betas, dtype=float), all_s)
        # BM24 Eq. (A34): exp(-i gamma_j/2) for j < p, exp(+i gamma_{2p-j}/2) for j > p.
        prod_elts = np.concatenate(
            (np.exp(-0.5j * np.array(gammas)) - 1.0, [(-1.0)], np.exp(0.5j * np.array(gammas)[::-1]) - 1.0)
        )
        c = r * np.prod([prod_elts[j] * ((all_s >> j) & 1) + 1.0 * ((~all_s >> j) & 1) for j in range(2 * p + 1)], axis=0)
        c_root = (-c) ** (1.0 / (2**q))
        return cls(q=q, r=r, p=p, betas=np.array(betas, dtype=float), gammas=np.array(gammas, dtype=float), b=b, c_root=c_root)

    @property
    def nvars(self) -> int:
        return int(self.b.shape[0])

    def dF(self, z: np.ndarray) -> np.ndarray:
        s_vec = np.exp(parent_function_alpha_sum_sos(0.5 * self.c_root * z))
        log_arg = np.sum(self.b * s_vec)
        return self.c_root * parent_function_s_sum_sos(0.5 * self.b * s_vec) / log_arg

    def G_complex(self, z: np.ndarray) -> np.ndarray:
        return z - (2**self.q) * (-self.dF(z)) ** (2**self.q - 1)

    def G_real(self, x: np.ndarray) -> np.ndarray:
        n = self.nvars
        z = x[:n] + 1j * x[n:]
        g = self.G_complex(z)
        return np.concatenate([g.real, g.imag])


def _to_iv_complex(z: complex):
    return iv.mpc(iv.mpf([float(np.real(z)), float(np.real(z))]), iv.mpf([float(np.imag(z)), float(np.imag(z))]))


def numerical_jacobian(f, x: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    n = x.size
    fx = f(x)
    j = np.zeros((n, n), dtype=float)
    for k in range(n):
        dx = np.zeros(n)
        dx[k] = eps
        j[:, k] = (f(x + dx) - fx) / eps
    return j


def krawczyk_test_sampled(f, x_star: np.ndarray, radius: float = 1e-6, samples: int = 64, seed: int = 0) -> tuple[bool, float]:
    j0 = numerical_jacobian(f, x_star)
    try:
        y = np.linalg.inv(j0)
    except np.linalg.LinAlgError:
        return False, np.inf
    fx = f(x_star)
    center_shift = np.linalg.norm(y @ fx, ord=np.inf)
    rng = np.random.default_rng(seed)
    norm_m = 0.0
    for _ in range(samples):
        x = x_star + rng.uniform(-radius, radius, size=x_star.size)
        jx = numerical_jacobian(f, x)
        m = np.eye(x_star.size) - y @ jx
        norm_m = max(norm_m, np.linalg.norm(m, ord=np.inf))
    bound = center_shift + norm_m * radius
    return (bound < radius), norm_m


def _pow_int(x, n: int):
    if n < 0:
        raise ValueError("n must be nonnegative")
    out = 1
    base = x
    k = n
    while k > 0:
        if k & 1:
            out = out * base
        base = base * base
        k >>= 1
    return out


class DualIV:
    def __init__(self, val, deriv):
        self.val = val
        self.deriv = deriv

    @staticmethod
    def const(v, n):
        return DualIV(v, [iv.mpc(0) for _ in range(n)])

    @staticmethod
    def var(v, n, idx):
        d = [iv.mpc(0) for _ in range(n)]
        d[idx] = iv.mpc(1)
        return DualIV(v, d)

    def __add__(self, other):
        if not isinstance(other, DualIV):
            other = DualIV.const(other, len(self.deriv))
        return DualIV(self.val + other.val, [self.deriv[i] + other.deriv[i] for i in range(len(self.deriv))])

    __radd__ = __add__

    def __sub__(self, other):
        if not isinstance(other, DualIV):
            other = DualIV.const(other, len(self.deriv))
        return DualIV(self.val - other.val, [self.deriv[i] - other.deriv[i] for i in range(len(self.deriv))])

    def __rsub__(self, other):
        return DualIV.const(other, len(self.deriv)) - self

    def __mul__(self, other):
        if not isinstance(other, DualIV):
            other = DualIV.const(other, len(self.deriv))
        return DualIV(
            self.val * other.val,
            [self.val * other.deriv[i] + self.deriv[i] * other.val for i in range(len(self.deriv))],
        )

    __rmul__ = __mul__

    def __truediv__(self, other):
        if not isinstance(other, DualIV):
            other = DualIV.const(other, len(self.deriv))
        return DualIV(
            self.val / other.val,
            [(self.deriv[i] * other.val - self.val * other.deriv[i]) / (other.val * other.val) for i in range(len(self.deriv))],
        )

    def __rtruediv__(self, other):
        return DualIV.const(other, len(self.deriv)) / self

    def __neg__(self):
        return DualIV(-self.val, [-d for d in self.deriv])

    def exp(self):
        ev = iv.exp(self.val)
        return DualIV(ev, [ev * d for d in self.deriv])

    def pow_int(self, n: int):
        if n == 0:
            return DualIV.const(iv.mpc(1), len(self.deriv))
        v = _pow_int(self.val, n)
        if n == 1:
            return DualIV(v, self.deriv[:])
        fac = n * _pow_int(self.val, n - 1)
        return DualIV(v, [fac * d for d in self.deriv])


def _sos_alpha_generic(z):
    n = int(np.log2(len(z)))
    a1 = list(z)
    for i in range(n):
        for mask in range(1 << n):
            if (mask >> i) & 1:
                a1[mask] = a1[mask] + a1[mask ^ (1 << i)]
    a0 = list(reversed(z))
    for i in range(n):
        for mask in range(1 << n):
            if (~mask >> i) & 1:
                a0[mask] = a0[mask] + a0[mask ^ (1 << i)]
    return [a1[s] + a0[s] - z[0] for s in range(1 << n)]


def _sos_s_generic(z):
    n = int(np.log2(len(z)))
    a0 = list(reversed(z))
    for i in range(n):
        for mask in range(1 << n):
            if (~mask >> i) & 1:
                a0[mask] = a0[mask] + a0[mask ^ (1 << i)]
    a1 = list(z)
    for i in range(n):
        for mask in range(1 << n):
            if (~mask >> i) & 1:
                a1[mask] = a1[mask] + a1[mask ^ (1 << i)]
    out = [a0[s] + a1[s] for s in range(1 << n)]
    out[0] = out[0] - a0[0]
    return out


def _build_constants_iv(sys: SaddleSystem):
    b_iv = [_to_iv_complex(v) for v in sys.b]
    c_root_iv = [_to_iv_complex(v) for v in sys.c_root]
    return b_iv, c_root_iv


def G_interval_and_jacobian_box(z_box, sys: SaddleSystem):
    n = len(z_box)
    b_iv, c_root_iv = _build_constants_iv(sys)
    z_dual = [DualIV.var(z_box[i], n, i) for i in range(n)]
    half = DualIV.const(iv.mpc("0.5"), n)
    half_cz = [half * DualIV.const(c_root_iv[i], n) * z_dual[i] for i in range(n)]
    alpha = _sos_alpha_generic(half_cz)
    s_vec = [a.exp() for a in alpha]
    log_arg = DualIV.const(iv.mpc(0), n)
    for i in range(n):
        log_arg = log_arg + DualIV.const(b_iv[i], n) * s_vec[i]
    half_bs = [half * DualIV.const(b_iv[i], n) * s_vec[i] for i in range(n)]
    s_sum = _sos_s_generic(half_bs)
    dF = [DualIV.const(c_root_iv[i], n) * s_sum[i] / log_arg for i in range(n)]
    k = 2 ** sys.q
    G = [z_dual[i] - DualIV.const(iv.mpc(k), n) * (-dF[i]).pow_int(k - 1) for i in range(n)]
    g_iv = [g.val for g in G]
    j_iv = [[G[i].deriv[j] for j in range(n)] for i in range(n)]
    return g_iv, j_iv


def _matvec_const_interval(y_mat: np.ndarray, v_iv):
    n = len(v_iv)
    out = [iv.mpc(0) for _ in range(n)]
    for i in range(n):
        acc = iv.mpc(0)
        for j in range(n):
            acc += _to_iv_complex(y_mat[i, j]) * v_iv[j]
        out[i] = acc
    return out


def _matmul_const_interval(y_mat: np.ndarray, j_iv):
    n = len(j_iv)
    out = [[iv.mpc(0) for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            acc = iv.mpc(0)
            for k in range(n):
                acc += _to_iv_complex(y_mat[i, k]) * j_iv[k][j]
            out[i][j] = acc
    return out


def _strict_subset_complex(a, b) -> bool:
    return (a.real.a > b.real.a and a.real.b < b.real.b and a.imag.a > b.imag.a and a.imag.b < b.imag.b)


def _interval_matrix_inf_norm_upper(m_iv):
    row_max = mp.mpf("0.0")
    for row in m_iv:
        s = mp.mpf("0.0")
        for x in row:
            s += abs(x).b
        if s > row_max:
            row_max = s
    return float(row_max)


def krawczyk_certify_root_rigorous(sys: SaddleSystem, z_center: np.ndarray, dps: int = 60, max_inflate_iters: int = 6):
    mp.mp.dps = int(dps)
    iv.dps = int(dps)

    g_center = sys.G_complex(z_center)
    eps = 1e-8
    n = len(z_center)
    j_num = np.zeros((n, n), dtype=complex)
    for j in range(n):
        dz = np.zeros(n, dtype=complex)
        dz[j] = eps
        j_num[:, j] = (sys.G_complex(z_center + dz) - g_center) / eps
    try:
        y = np.linalg.inv(j_num)
    except np.linalg.LinAlgError:
        return False, {"reason": "singular Jacobian at center"}

    g_thin_iv = [_to_iv_complex(v) for v in g_center]
    yg = y @ g_center
    # u ~ unit roundoff at current precision bits
    u = mp.power(2, -mp.mp.prec)
    inflate_factor = mp.power(u, mp.mpf("-0.25"))
    # Reference radius from epsilon-inflation heuristic
    ref_rad = [max(abs(yg[i]) * inflate_factor, mp.power(10, -(dps - 8))) for i in range(n)]
    # Also test tiny absolute radii first to reduce interval wrapping in nonlinear terms.
    tiny_abs = [mp.power(10, -k) for k in (40, 32, 28, 24, 20, 16, 12, 10)]
    # then around the heuristic scale
    heuristic_scales = [mp.power(2, -4), mp.power(2, -2), mp.power(2, -1)] + [mp.power(2, it) for it in range(max_inflate_iters)]
    radius_candidates = []
    for t in tiny_abs:
        radius_candidates.append([t] * n)
    for s in heuristic_scales:
        radius_candidates.append([max(ref_rad[i] * s, mp.power(10, -(dps - 8))) for i in range(n)])

    for it, rad_vec in enumerate(radius_candidates):
        z_box = []
        for i in range(n):
            r = float(rad_vec[i])
            zc = z_center[i]
            z_box.append(
                iv.mpc(
                    iv.mpf([float(np.real(zc)) - r, float(np.real(zc)) + r]),
                    iv.mpf([float(np.imag(zc)) - r, float(np.imag(zc)) + r]),
                )
            )
        g_box_at_center = g_thin_iv
        _, j_box = G_interval_and_jacobian_box(z_box, sys)
        yg_box = _matvec_const_interval(y, g_box_at_center)
        yj_box = _matmul_const_interval(y, j_box)
        m = [[iv.mpc(0) for _ in range(n)] for _ in range(n)]
        for i in range(n):
            for j in range(n):
                m[i][j] = (iv.mpc(1) if i == j else iv.mpc(0)) - yj_box[i][j]
        diff = [z_box[i] - _to_iv_complex(z_center[i]) for i in range(n)]
        term2 = [iv.mpc(0) for _ in range(n)]
        for i in range(n):
            acc = iv.mpc(0)
            for j in range(n):
                acc += m[i][j] * diff[j]
            term2[i] = acc
        k_box = [_to_iv_complex(z_center[i]) - yg_box[i] + term2[i] for i in range(n)]
        inclusion = all(_strict_subset_complex(k_box[i], z_box[i]) for i in range(n))
        contraction = np.sqrt(2.0) * _interval_matrix_inf_norm_upper(m)
        if inclusion and contraction < 1.0:
            return True, {
                "contraction_bound": float(contraction),
                "inflate_iter": it,
                "box_center_real": [float(np.real(z)) for z in z_center],
                "box_center_imag": [float(np.imag(z)) for z in z_center],
                "box_radius": [float(rad_vec[i]) for i in range(n)],
                "dps_used": int(dps),
            }
    return False, {"reason": "inclusion/contraction failed"}


def _check_krawczyk_at_uniform_radius(sys: SaddleSystem, z_center: np.ndarray, y: np.ndarray, g_center: np.ndarray, radius: float):
    g_thin_iv = [_to_iv_complex(v) for v in g_center]
    z_box = []
    n = len(z_center)
    for i in range(n):
        zc = z_center[i]
        z_box.append(
            iv.mpc(
                iv.mpf([float(np.real(zc)) - radius, float(np.real(zc)) + radius]),
                iv.mpf([float(np.imag(zc)) - radius, float(np.imag(zc)) + radius]),
            )
        )
    _, j_box = G_interval_and_jacobian_box(z_box, sys)
    yg_box = _matvec_const_interval(y, g_thin_iv)
    yj_box = _matmul_const_interval(y, j_box)
    m = [[iv.mpc(0) for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            m[i][j] = (iv.mpc(1) if i == j else iv.mpc(0)) - yj_box[i][j]
    diff = [z_box[i] - _to_iv_complex(z_center[i]) for i in range(n)]
    term2 = [iv.mpc(0) for _ in range(n)]
    for i in range(n):
        acc = iv.mpc(0)
        for j in range(n):
            acc += m[i][j] * diff[j]
        term2[i] = acc
    k_box = [_to_iv_complex(z_center[i]) - yg_box[i] + term2[i] for i in range(n)]
    inclusion = all(_strict_subset_complex(k_box[i], z_box[i]) for i in range(n))
    contraction = np.sqrt(2.0) * _interval_matrix_inf_norm_upper(m)
    return bool(inclusion), float(contraction)


def maximize_certified_uniform_radius(
    sys: SaddleSystem,
    z_center: np.ndarray,
    dps: int,
    max_inflate_iters: int = 6,
    max_radius: float = 1e-3,
    growth_factor: float = 2.0,
    bisect_steps: int = 24,
):
    mp.mp.dps = int(dps)
    iv.dps = int(dps)

    g_center = sys.G_complex(z_center)
    eps = 1e-8
    n = len(z_center)
    j_num = np.zeros((n, n), dtype=complex)
    for j in range(n):
        dz = np.zeros(n, dtype=complex)
        dz[j] = eps
        j_num[:, j] = (sys.G_complex(z_center + dz) - g_center) / eps
    try:
        y = np.linalg.inv(j_num)
    except np.linalg.LinAlgError:
        return False, {"reason": "singular Jacobian at center"}

    # Bootstrap from existing rigorous certifier so this mode never weakens baseline behavior.
    base_ok, base_info = krawczyk_certify_with_escalation(
        sys, z_center, dps_start=dps, max_inflate_iters=max_inflate_iters
    )
    if not base_ok:
        return False, base_info
    base_r = base_info.get("box_radius", [])
    if base_r and len(base_r) == n:
        lo = float(min(base_r))
    else:
        lo = 1e-12
    ok_lo, kappa_lo = _check_krawczyk_at_uniform_radius(sys, z_center, y, g_center, lo)
    if not ok_lo or kappa_lo >= 1.0:
        # Last-resort tiny search if the bootstrap radius is not suitable as a uniform box.
        lo = 1e-14
        ok_lo, kappa_lo = _check_krawczyk_at_uniform_radius(sys, z_center, y, g_center, lo)
        if not ok_lo or kappa_lo >= 1.0:
            return False, {"reason": "failed to initialize radius search", "tested_radius": lo, "contraction_bound": float(kappa_lo)}

    hi = lo
    kappa_hi = kappa_lo
    # Exponential search for failure boundary.
    while hi < max_radius:
        cand = min(hi * growth_factor, max_radius)
        ok, kappa = _check_krawczyk_at_uniform_radius(sys, z_center, y, g_center, cand)
        if ok and kappa < 1.0:
            hi = cand
            kappa_hi = kappa
            if hi >= max_radius:
                return True, {
                    "max_uniform_radius": float(hi),
                    "contraction_bound": float(kappa_hi),
                    "search_hit_cap": True,
                    "dps_used": int(dps),
                }
        else:
            # Binary search between last-good hi/growth_factor and failing cand.
            left = hi
            right = cand
            best_r = hi
            best_kappa = kappa_hi
            for _ in range(bisect_steps):
                mid = 0.5 * (left + right)
                ok_mid, kappa_mid = _check_krawczyk_at_uniform_radius(sys, z_center, y, g_center, mid)
                if ok_mid and kappa_mid < 1.0:
                    best_r = mid
                    best_kappa = kappa_mid
                    left = mid
                else:
                    right = mid
            return True, {
                "max_uniform_radius": float(best_r),
                "contraction_bound": float(best_kappa),
                "search_hit_cap": False,
                "dps_used": int(dps),
            }

    return True, {
        "max_uniform_radius": float(hi),
        "contraction_bound": float(kappa_hi),
        "search_hit_cap": True,
        "dps_used": int(dps),
    }


def krawczyk_certify_with_escalation(sys: SaddleSystem, z_center: np.ndarray, dps_start: int, max_inflate_iters: int, max_levels: int = 4):
    dps = int(dps_start)
    last = {}
    for _ in range(max_levels):
        ok, info = krawczyk_certify_root_rigorous(sys, z_center, dps=dps, max_inflate_iters=max_inflate_iters)
        if ok:
            return True, info
        last = info
        dps = int(np.ceil(dps * 1.5))
    out = dict(last)
    out["dps_used"] = dps
    return False, out


def discover_roots(sys: SaddleSystem, num_starts: int, seed: int, tol: float) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    roots = []
    for _ in range(num_starts):
        x0 = rng.normal(0.0, 1.0, size=2 * sys.nvars)
        sol = root(sys.G_real, x0, method="hybr", tol=tol)
        if not sol.success:
            continue
        x = sol.x
        res = np.linalg.norm(sys.G_real(x), ord=np.inf)
        if not np.isfinite(res) or res > 1e-7:
            continue
        dup = False
        for y in roots:
            if np.linalg.norm(x - y, ord=np.inf) < 1e-5:
                dup = True
                break
        if not dup:
            roots.append(x)
    return roots


def _load_roots_from_json(path: str) -> list[np.ndarray]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    roots = []
    for r in data.get("roots", []):
        zr = np.array(r["z_real"], dtype=float)
        zi = np.array(r["z_imag"], dtype=float)
        roots.append(zr + 1j * zi)
    return roots


def main() -> None:
    parser = argparse.ArgumentParser(description="Phasecraft Krawczyk root finder")
    parser.add_argument("--q", type=int, default=3, help="k=2^q")
    parser.add_argument("--r", type=float, default=176.54)
    parser.add_argument("--p", type=int, default=1, help="QAOA depth p")
    parser.add_argument("--beta", type=float, nargs="*", default=None, help="List of p beta angles")
    parser.add_argument("--gamma", type=float, nargs="*", default=None, help="List of p gamma angles")
    parser.add_argument("--num-starts", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tol", type=float, default=1e-12)
    parser.add_argument("--krawczyk-radius", type=float, default=1e-6)
    parser.add_argument("--krawczyk-samples", type=int, default=128)
    parser.add_argument("--rigorous", action="store_true", help="Use interval-rigorous Krawczyk certification")
    parser.add_argument("--dps", type=int, default=60, help="mpmath decimal precision for rigorous mode")
    parser.add_argument("--max-inflate-iters", type=int, default=6)
    parser.add_argument("--max-roots", type=int, default=0, help="If >0, only process first N roots")
    parser.add_argument("--maximize-radius", action="store_true", help="Search largest certifiable uniform radius per root")
    parser.add_argument("--radius-cap", type=float, default=1e-3, help="Upper cap for --maximize-radius search")
    parser.add_argument("--radius-bisect-steps", type=int, default=24, help="Binary search steps for --maximize-radius")
    parser.add_argument("--in-roots", type=str, default="", help="Optional existing roots JSON to certify")
    parser.add_argument("--out", type=str, default="")
    args = parser.parse_args()

    if args.beta is None or args.gamma is None:
        k = 2**args.q
        try:
            cfg = optimal_angles[(k, args.r)][args.p]
        except KeyError as exc:
            raise ValueError(
                "Missing --beta/--gamma and no matching entry in phasecraft.optimal_angles for (k,r,p)"
            ) from exc
        betas = np.array(cfg["betas"], dtype=float)
        gammas = np.array(cfg["gammas"], dtype=float)
    else:
        betas = np.array(args.beta, dtype=float)
        gammas = np.array(args.gamma, dtype=float)
    if len(betas) != args.p or len(gammas) != args.p:
        raise ValueError(f"Expected exactly p={args.p} betas/gammas")

    sys = SaddleSystem.build(q=args.q, r=args.r, betas=betas, gammas=gammas)
    if args.in_roots:
        roots_z = _load_roots_from_json(args.in_roots)
    else:
        roots_x = discover_roots(sys, num_starts=args.num_starts, seed=args.seed, tol=args.tol)
        n = sys.nvars
        roots_z = [x[:n] + 1j * x[n:] for x in roots_x]
    if args.max_roots and args.max_roots > 0:
        roots_z = roots_z[: args.max_roots]
    payload = {
        "q": args.q,
        "k": 2**args.q,
        "r": args.r,
        "p": args.p,
        "betas": betas.tolist(),
        "gammas": gammas.tolist(),
        "num_roots": len(roots_z),
        "rigorous_mode": bool(args.rigorous),
        "roots": [],
    }
    for i, z in enumerate(roots_z):
        res = float(np.linalg.norm(sys.G_complex(z), ord=np.inf))
        if args.rigorous:
            if args.maximize_radius:
                cert, info = maximize_certified_uniform_radius(
                    sys,
                    z,
                    dps=args.dps,
                    max_inflate_iters=args.max_inflate_iters,
                    max_radius=args.radius_cap,
                    bisect_steps=args.radius_bisect_steps,
                )
            else:
                cert, info = krawczyk_certify_with_escalation(
                    sys, z, dps_start=args.dps, max_inflate_iters=args.max_inflate_iters
                )
            row = {
                "idx": i,
                "residual_norm": res,
                "krawczyk_certified": bool(cert),
                "krawczyk_contraction_bound": float(info.get("contraction_bound", np.inf)),
                "proof": info,
                "z_real": z.real.tolist(),
                "z_imag": z.imag.tolist(),
            }
        else:
            x = np.concatenate([z.real, z.imag])
            cert, kappa = krawczyk_test_sampled(
                sys.G_real,
                x,
                radius=args.krawczyk_radius,
                samples=args.krawczyk_samples,
                seed=args.seed + i + 1,
            )
            row = {
                "idx": i,
                "residual_norm": res,
                "krawczyk_certified": bool(cert),
                "krawczyk_contraction_bound": float(kappa),
                "z_real": z.real.tolist(),
                "z_imag": z.imag.tolist(),
            }
        payload["roots"].append(row)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    print(json.dumps({k: payload[k] for k in ["q", "k", "r", "p", "betas", "gammas", "num_roots"]}, indent=2))
    print("certified_roots:", sum(1 for r in payload["roots"] if r["krawczyk_certified"]))


if __name__ == "__main__":
    main()
