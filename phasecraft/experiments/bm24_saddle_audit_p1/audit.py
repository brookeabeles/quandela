"""
BM24 generalized multinomial saddle audit at p=1, q=3 (K_clause=8), r=176.54.

Compares exact finite-n exponents from BM24 Proposition 4 (Eq. A10) against
certified saddle actions from the z-coordinate Krawczyk pipeline.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import mpmath as mp
import numpy as np
from scipy.optimize import root
from scipy.special import gammaln

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from phasecraft.krawczyk_p1_roots import (
    SaddleSystem,
    _check_krawczyk_at_uniform_radius,
    discover_roots,
    krawczyk_certify_with_escalation,
    krawczyk_test_sampled,
    numerical_jacobian,
)
from phasecraft.optimal_angles import optimal_angles
from phasecraft.picard_lefschetz import compute_phi, detect_stokes_pairs
from phasecraft.saddle_traits import complex_to_real_vector

_PATCHED_PATH = (
    REPO_ROOT
    / "phasecraft"
    / "lib"
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
_PATCHED_MOD = importlib.util.module_from_spec(_PATCHED_SPEC)
assert _PATCHED_SPEC is not None and _PATCHED_SPEC.loader is not None
sys.modules[_PATCHED_SPEC.name] = _PATCHED_MOD
_PATCHED_SPEC.loader.exec_module(_PATCHED_MOD)

generalized_binomial_sum_scaling_exponent_ksat = _PATCHED_MOD.generalized_binomial_sum_scaling_exponent_ksat
bm24_prefactor_exponent_ksat = _PATCHED_MOD.bm24_prefactor_exponent_ksat
bm24_prefactor_exponent_ksat_all_subsets = _PATCHED_MOD.bm24_prefactor_exponent_ksat_all_subsets
generalized_flip_symmetric_expected_success_p1 = _PATCHED_MOD.generalized_flip_symmetric_expected_success_p1

AUDIT_DIR = Path(__file__).resolve().parent
DEFAULT_OUT = AUDIT_DIR / "results"
BM24_ITERATE_CERTIFIED_AS_FIXED_POINT = "BM24_ITERATE_CERTIFIED_AS_FIXED_POINT"
BM24_NONCONVERGED_ITERATE = "BM24_NONCONVERGED_ITERATE"
BM24_NUMERICAL_ITERATE_UNKNOWN_STATUS = "BM24_NUMERICAL_ITERATE_UNKNOWN_STATUS"


# ---------------------------------------------------------------------------
# BM24 Prop 4 exact finite-n (stable complex log-sum-exp)
# ---------------------------------------------------------------------------


def _complex_logsumexp(log_terms: np.ndarray) -> complex:
    """log(sum_k exp(z_k)) for complex z_k via shift by max real part."""
    if log_terms.size == 0:
        return -np.inf + 0j
    shift = float(np.max(np.real(log_terms)))
    scaled = np.exp(log_terms - shift)
    return complex(shift + np.log(np.sum(scaled)))


def log_p_succ_prop4(K_clause: int, r: float, beta: float, gamma: float, n: int) -> complex:
    """ln E[p_succ] at p=1 from BM24 Prop 4 / Eq. (A10) (Convention 1)."""
    if n < 0:
        raise ValueError("n must be nonnegative")
    q = int(round(math.log2(K_clause)))
    if 2**q != K_clause:
        raise ValueError(f"K_clause={K_clause} must be a power of 2")

    log_pref = -(r / (2**K_clause)) * n * (1.0 + 4.0 * math.sin(gamma / 4.0) ** 2)
    cos2 = math.cos(beta / 2.0) ** 2
    sin2 = math.sin(beta / 2.0) ** 2
    sb2 = math.sin(beta) / 2.0
    e_minus = np.exp(-0.5j * gamma)
    e_plus = np.exp(0.5j * gamma)
    s2g4 = math.sin(gamma / 4.0) ** 2

    terms: list[complex] = []
    for na in range(n + 1):
        for nb in range(n + 1 - na):
            for nc in range(n + 1 - na - nb):
                nd = n - na - nb - nc
                log_multi = (
                    gammaln(n + 1)
                    - gammaln(na + 1)
                    - gammaln(nb + 1)
                    - gammaln(nc + 1)
                    - gammaln(nd + 1)
                )
                log_B = (
                    na * math.log(cos2)
                    + nb * math.log(sin2)
                    + nc * (math.log(abs(sb2)) + 0.5j * math.pi if sb2 != 0 else -np.inf)
                    + nd * (math.log(abs(sb2)) - 0.5j * math.pi if sb2 != 0 else -np.inf)
                )
                if nc > 0 and sb2 == 0:
                    continue
                if nd > 0 and sb2 == 0:
                    continue
                exp_arg = r * n * (
                    4.0 * s2g4 * (((nb + na) / (2 * n)) ** K_clause - (na / (2 * n)) ** K_clause)
                    + (1.0 - e_minus) * ((nc + na) / (2 * n)) ** K_clause
                    + (1.0 - e_plus) * ((nd + na) / (2 * n)) ** K_clause
                )
                terms.append(log_multi + log_B + exp_arg)

    log_inner = _complex_logsumexp(np.asarray(terms, dtype=np.complex128))
    return log_pref + log_inner


def p_succ_prop4(K_clause: int, r: float, beta: float, gamma: float, n: int) -> float:
    return float(np.real(np.exp(log_p_succ_prop4(K_clause, r, beta, gamma, n))))


def finite_n_exponent_grid(
    K_clause: int,
    q: int,
    r: float,
    beta: float,
    gamma: float,
    n_values: Iterable[int],
) -> dict:
    """Primary baseline: Conv2 exact finite-n (all-subsets). Prop4 retained as cross-check only."""
    ns = [int(n) for n in n_values]
    log_p_conv2 = {
        n: complex(
            np.log(
                complex(
                    generalized_flip_symmetric_expected_success_p1(
                        q=q,
                        r=r,
                        betas=np.array([beta]),
                        gammas=np.array([gamma]),
                        n=n,
                    )
                )
            )
        )
        for n in ns
    }
    p_conv2 = {n: float(np.real(np.exp(lp))) for n, lp in log_p_conv2.items()}
    lambda_abs_conv2 = {n: float(np.real(log_p_conv2[n]) / n) for n in ns if n > 0}
    lambda_local_conv2 = {}
    for n in ns:
        if n + 1 in log_p_conv2:
            lambda_local_conv2[n] = float(np.real(log_p_conv2[n + 1] - log_p_conv2[n]))

    log_p_prop4 = {n: log_p_succ_prop4(K_clause, r, beta, gamma, n) for n in ns}
    p_prop4 = {n: float(np.real(np.exp(lp))) for n, lp in log_p_prop4.items()}
    lambda_abs_prop4 = {n: float(np.real(log_p_prop4[n]) / n) for n in ns if n > 0}
    lambda_local_prop4 = {}
    for n in ns:
        if n + 1 in log_p_prop4:
            lambda_local_prop4[n] = float(np.real(log_p_prop4[n + 1] - log_p_prop4[n]))

    return {
        "n_values": ns,
        "primary_baseline": "conv2_exact_finite_n",
        "log_p_conv2_exact": {str(n): {"real": float(np.real(v)), "imag": float(np.imag(v))} for n, v in log_p_conv2.items()},
        "p_conv2_exact": {str(n): v for n, v in p_conv2.items()},
        "lambda_abs": {str(n): v for n, v in lambda_abs_conv2.items()},
        "lambda_local": {str(n): v for n, v in lambda_local_conv2.items()},
        "crosscheck_log_p_prop4": {str(n): {"real": float(np.real(v)), "imag": float(np.imag(v))} for n, v in log_p_prop4.items()},
        "crosscheck_p_prop4": {str(n): v for n, v in p_prop4.items()},
        "crosscheck_lambda_abs_prop4": {str(n): v for n, v in lambda_abs_prop4.items()},
        "crosscheck_lambda_local_prop4": {str(n): v for n, v in lambda_local_prop4.items()},
        "phi_pref_prop4": float(-(r / (2**K_clause)) * (1.0 + 4.0 * math.sin(gamma / 4.0) ** 2)),
        "phi_pref_conv2_all_subsets": float(bm24_prefactor_exponent_ksat_all_subsets(k=K_clause, r=r)),
    }


# ---------------------------------------------------------------------------
# Saddle discovery (z chart) + gamma continuation
# ---------------------------------------------------------------------------


def _x_to_z(x: np.ndarray, nvars: int) -> np.ndarray:
    return x[:nvars] + 1j * x[nvars:]


def polish_root_x(sys: SaddleSystem, x: np.ndarray, tol: float = 1e-14) -> Optional[np.ndarray]:
    """Newton polish to G_real(x)=0; returns None if polish fails."""
    methods = ("hybr", "lm", "krylov")
    for method in methods:
        try:
            sol = root(sys.G_real, x, method=method, tol=tol)
        except Exception:
            continue
        if not sol.success:
            continue
        res = float(np.linalg.norm(sys.G_real(sol.x), ord=np.inf))
        if np.isfinite(res) and res <= 1e-7:
            return sol.x
    return None


def _scipy_polish_to_tol(
    sys: SaddleSystem,
    x0: np.ndarray,
    target_inf: float = 1e-25,
    max_tries: int = 5,
    tol: float = 1e-14,
) -> tuple[np.ndarray, float]:
    x = np.array(x0, dtype=float)
    best_res = float(np.linalg.norm(sys.G_real(x), ord=np.inf))
    for _ in range(max_tries):
        improved = False
        for method in ("hybr", "lm", "krylov"):
            try:
                sol = root(sys.G_real, x, method=method, tol=tol)
            except Exception:
                continue
            if not sol.success:
                continue
            res = float(np.linalg.norm(sys.G_real(sol.x), ord=np.inf))
            if np.isfinite(res) and res < best_res:
                x = np.array(sol.x, dtype=float)
                best_res = res
                improved = True
                if best_res <= target_inf:
                    return x, best_res
        if not improved:
            break
    return x, best_res


def _mpmath_newton_polish(
    sys: SaddleSystem,
    x0: np.ndarray,
    target_inf: float = 1e-25,
    dps: int = 120,
    max_iters: int = 10,
) -> tuple[np.ndarray, float]:
    """High-precision Newton refinement on real variables.

    Note: equations are evaluated through sys.G_real (float backend), so this
    is primarily a robust linear-solve refinement path, not a new equation model.
    """
    mp.mp.dps = int(dps)
    x_mp = [mp.mpf(float(v)) for v in x0]
    n = len(x_mp)
    for _ in range(max_iters):
        x_np = np.array([float(v) for v in x_mp], dtype=float)
        g = sys.G_real(x_np)
        res = float(np.linalg.norm(g, ord=np.inf))
        if res <= target_inf:
            return x_np, res
        j = numerical_jacobian(sys.G_real, x_np, eps=1e-9)
        try:
            j_mp = mp.matrix([[mp.mpf(float(j[i, jcol])) for jcol in range(n)] for i in range(n)])
            g_mp = mp.matrix([mp.mpf(float(gi)) for gi in g])
            delta = mp.lu_solve(j_mp, -g_mp)
        except Exception:
            break
        x_mp = [x_mp[i] + delta[i] for i in range(n)]
    x_out = np.array([float(v) for v in x_mp], dtype=float)
    return x_out, float(np.linalg.norm(sys.G_real(x_out), ord=np.inf))


def _adaptive_krawczyk_with_diagnostics(
    sys: SaddleSystem,
    z_center: np.ndarray,
    *,
    dps: int,
) -> tuple[bool, dict]:
    mp.mp.dps = int(dps)
    from mpmath import iv

    iv.dps = int(dps)
    g_center = sys.G_complex(z_center)
    n = len(z_center)
    eps = 1e-8
    j_num = np.zeros((n, n), dtype=complex)
    for j in range(n):
        dz = np.zeros(n, dtype=complex)
        dz[j] = eps
        j_num[:, j] = (sys.G_complex(z_center + dz) - g_center) / eps
    try:
        y = np.linalg.inv(j_num)
    except np.linalg.LinAlgError:
        return False, {
            "reason": "singular Jacobian at center",
            "G_inf": float(np.linalg.norm(g_center, ord=np.inf)),
            "cond_J": float("inf"),
        }

    yg = y @ g_center
    yg_inf = float(np.linalg.norm(yg, ord=np.inf))
    cond_j = float(np.linalg.cond(j_num))
    base = max(yg_inf, 1e-18)
    scales = [1e-6, 1e-4, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3]
    radii = [float(base * s) for s in scales]
    outcomes = []
    for r in radii:
        inclusion, contraction = _check_krawczyk_at_uniform_radius(sys, z_center, y, g_center, r)
        outcomes.append(
            {
                "radius": r,
                "inclusion": bool(inclusion),
                "contraction_bound": float(contraction),
            }
        )
        if inclusion and contraction < 1.0:
            return True, {
                "box_radius": [r] * n,
                "contraction_bound": float(contraction),
                "adaptive": True,
                "G_inf": float(np.linalg.norm(g_center, ord=np.inf)),
                "cond_J": cond_j,
                "Y_G_inf": yg_inf,
                "tested_radii": outcomes,
            }
    return False, {
        "reason": "adaptive radii failed",
        "adaptive": True,
        "G_inf": float(np.linalg.norm(g_center, ord=np.inf)),
        "cond_J": cond_j,
        "Y_G_inf": yg_inf,
        "tested_radii": outcomes,
    }


def dedupe_roots_x(roots_x: list[np.ndarray], tol: float) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    for x in roots_x:
        if not np.all(np.isfinite(x)):
            continue
        if any(np.linalg.norm(x - y, ord=np.inf) < tol for y in out):
            continue
        out.append(x)
    return out


def discover_roots_continuation(
    q: int,
    r: float,
    beta: float,
    gamma_values: np.ndarray,
    num_random_starts: int,
    seed: int,
    tol: float,
    dedupe_tol: float,
) -> dict[float, list[np.ndarray]]:
    """Newton multistart + warm-start continuation in gamma (z coordinates)."""
    rng = np.random.default_rng(seed)
    chain: dict[float, list[np.ndarray]] = {}
    pool: list[np.ndarray] = []

    for gamma in gamma_values:
        sys = SaddleSystem.build(q=q, r=r, betas=np.array([beta]), gammas=np.array([gamma]))
        n2 = 2 * sys.nvars
        candidates = list(pool)
        seed_z, _, _ = bm24_seed_z(q, r, beta, float(gamma))
        seed_x = np.concatenate([seed_z.real, seed_z.imag])
        polished = polish_root_x(sys, seed_x)
        candidates.append(polished if polished is not None else seed_x)
        extra = discover_roots(sys, num_starts=min(200, num_random_starts // 4), seed=seed, tol=tol)
        candidates.extend(extra)
        for _ in range(num_random_starts):
            candidates.append(rng.normal(0.0, 1.0, size=n2))

        found: list[np.ndarray] = []
        for x0 in candidates:
            sol = root(sys.G_real, x0, method="hybr", tol=tol)
            if not sol.success:
                continue
            x = sol.x
            res = np.linalg.norm(sys.G_real(x), ord=np.inf)
            if not np.isfinite(res) or res > 1e-7:
                continue
            found.append(x)

        pool = dedupe_roots_x(found, tol=dedupe_tol)
        chain[float(gamma)] = pool

    return chain


# ---------------------------------------------------------------------------
# Certification + saddle diagnostics
# ---------------------------------------------------------------------------


@dataclass
class SaddleRecord:
    saddle_id: int
    beta: float
    gamma: float
    certified: bool
    box_radius: float
    residual: float
    jacobian_cond: float
    re_phi: float
    im_phi: float
    det_hessian_re_phi: float
    nearest_action_gap: float
    re_phi_plus_pref: float
    re_phi_plus_pref_conv2: float
    matches_lambda_abs: bool
    matches_lambda_local: bool
    matches_lambda_abs_full: bool
    matches_lambda_abs_conv2: bool
    controls_finite_n: bool
    is_bm24_seed_saddle: bool
    z_real: list[float] = field(default_factory=list)
    z_imag: list[float] = field(default_factory=list)
    krawczyk_contraction: float = math.inf
    proof: dict = field(default_factory=dict)

    def to_row(self) -> dict:
        row = asdict(self)
        for key, val in row.items():
            if isinstance(val, (np.bool_, bool)):
                row[key] = bool(val)
            elif isinstance(val, (np.floating, np.integer)):
                row[key] = float(val)
        return row


@dataclass
class ObjectComparisonRow:
    object_type: str
    object_status: str
    beta: float
    gamma: float
    q: int
    certified: bool
    fixed_point: bool
    re_phi_m: float
    im_phi_m: float
    prefactor_used_label: str
    prefactor_used: float
    re_phi_plus_prefactor: float
    conv1_prefactor: float
    conv2_prefactor: float
    re_phi_plus_conv1: float
    re_phi_plus_conv2: float
    residual_inf: float
    residual_l2: float
    scaled_res_inf: float
    componentwise_rel: float
    has_nan_or_inf: bool
    matches_lambda_abs_conv1: bool
    matches_lambda_local_conv1: bool
    matches_lambda_abs_conv2: bool
    matches_lambda_local_conv2: bool
    z_real: list[float] = field(default_factory=list)
    z_imag: list[float] = field(default_factory=list)
    note: str = ""

    def to_row(self) -> dict:
        row = asdict(self)
        for key, val in row.items():
            if isinstance(val, (np.bool_, bool)):
                row[key] = bool(val)
            elif isinstance(val, (np.floating, np.integer)):
                row[key] = float(val)
        return row


def _hessian_det_re_phi(
    z: np.ndarray,
    q: int,
    r: float,
    betas: np.ndarray,
    gammas: np.ndarray,
    eps: float = 1e-5,
) -> float:
    n = z.size

    def re_phi(x: np.ndarray) -> float:
        zz = x[:n] + 1j * x[n:]
        return float(np.real(compute_phi(zz, q=q, r=r, betas=betas, gammas=gammas)))

    x = complex_to_real_vector(z)
    dim = x.size
    h = np.zeros((dim, dim), dtype=float)
    for i in range(dim):
        ei = np.zeros(dim)
        ei[i] = eps
        for j in range(i, dim):
            ej = np.zeros(dim)
            ej[j] = eps
            val = (
                re_phi(x + ei + ej)
                - re_phi(x + ei - ej)
                - re_phi(x - ei + ej)
                + re_phi(x - ei - ej)
            ) / (4.0 * eps * eps)
            h[i, j] = val
            h[j, i] = val
    return float(np.linalg.det(h))


def nearest_action_gap(re_phi: float, others: list[float]) -> float:
    gaps = [re_phi - o for o in others if np.isfinite(o)]
    if not gaps:
        return float("nan")
    return float(min(abs(g) for g in gaps))


def bm24_seed_z(q: int, r: float, beta: float, gamma: float) -> tuple[np.ndarray, complex, bool]:
    """Fixed-point iteration from z=0 (BM24 seed saddle track)."""
    out = generalized_binomial_sum_scaling_exponent_ksat(
        q=q,
        r=r,
        betas=np.array([beta]),
        gammas=np.array([gamma]),
        num_iter=200,
        dz_threshold=1e-10,
        init_z=None,
    )
    if len(out) == 4:
        _, z, _, phi_m = out
        converged = False
    elif len(out) == 5:
        _, z, _, phi_m, converged = out
    else:
        raise ValueError(f"Unexpected return length from generalized_binomial_sum_scaling_exponent_ksat: {len(out)}")
    return z, complex(phi_m), bool(converged)


def diagnose_bm24_iterate(z: np.ndarray, sys: SaddleSystem, eps: float = 1e-30) -> dict:
    dF = sys.dF(z)
    Tz = (2**sys.q) * (-dF) ** (2**sys.q - 1)
    G = z - Tz
    res_inf = float(np.linalg.norm(G, ord=np.inf))
    res_l2 = float(np.linalg.norm(G, ord=2))
    z_inf = float(np.linalg.norm(z, ord=np.inf))
    denom = np.maximum(np.abs(z), eps)
    componentwise_rel = float(np.max(np.abs(G) / denom))
    has_bad = bool(
        np.any(~np.isfinite(z.real))
        or np.any(~np.isfinite(z.imag))
        or np.any(~np.isfinite(G.real))
        or np.any(~np.isfinite(G.imag))
    )
    return {
        "res_inf": res_inf,
        "res_l2": res_l2,
        "scaled_res_inf": float(res_inf / (1.0 + z_inf)),
        "componentwise_rel": componentwise_rel,
        "has_nan_or_inf": has_bad,
        "Tz_real": Tz.real.tolist(),
        "Tz_imag": Tz.imag.tolist(),
    }


def classify_bm24_iterate(diag: dict, tol_inf: float = 1e-8, tol_scaled: float = 1e-10, tol_rel: float = 1e-8) -> tuple[str, bool]:
    if bool(diag.get("has_nan_or_inf", True)):
        return BM24_NUMERICAL_ITERATE_UNKNOWN_STATUS, False
    res_inf = float(diag.get("res_inf", float("inf")))
    scaled = float(diag.get("scaled_res_inf", float("inf")))
    rel = float(diag.get("componentwise_rel", float("inf")))
    if res_inf <= tol_inf and scaled <= tol_scaled and rel <= tol_rel:
        return BM24_ITERATE_CERTIFIED_AS_FIXED_POINT, True
    if np.isfinite(res_inf) and np.isfinite(scaled) and np.isfinite(rel):
        return BM24_NONCONVERGED_ITERATE, False
    return BM24_NUMERICAL_ITERATE_UNKNOWN_STATUS, False


def _match_flags(re_phi: float, lam_abs: Optional[float], lam_loc: Optional[float], pref_conv1: float, pref_conv2: float, tol: float) -> dict:
    conv1 = re_phi + pref_conv1
    conv2 = re_phi + pref_conv2
    return {
        "abs_conv1": bool(lam_abs is not None and abs(lam_abs - conv1) < tol),
        "loc_conv1": bool(lam_loc is not None and abs(lam_loc - conv1) < tol),
        "abs_conv2": bool(lam_abs is not None and abs(lam_abs - conv2) < tol),
        "loc_conv2": bool(lam_loc is not None and abs(lam_loc - conv2) < tol),
    }


def build_object_comparison_rows(
    *,
    q: int,
    r: float,
    beta: float,
    gamma: float,
    finite_n: dict,
    sys: SaddleSystem,
    seed_z: np.ndarray,
    seed_phi: complex,
    seed_diag: dict,
    seed_status: str,
    seed_fixed: bool,
    seed_polished: Optional[np.ndarray],
    saddles_anchor: list[SaddleRecord],
    match_tol: float,
) -> list[ObjectComparisonRow]:
    ns = finite_n["n_values"]
    n_large = max(ns)
    n_prev = max(n for n in ns if n + 1 in ns) if ns else n_large
    lam_abs = finite_n["lambda_abs"].get(str(n_large))
    lam_loc = finite_n["lambda_local"].get(str(n_prev))
    pref_conv1 = float(finite_n["phi_pref_prop4"])
    pref_conv2 = float(bm24_prefactor_exponent_ksat_all_subsets(k=2**q, r=r))
    tol = max(match_tol, 5e-4 * abs(lam_abs)) if lam_abs is not None else match_tol

    rows: list[ObjectComparisonRow] = []

    seed_matches = _match_flags(float(np.real(seed_phi)), lam_abs, lam_loc, pref_conv1, pref_conv2, tol)
    rows.append(
        ObjectComparisonRow(
            object_type="BM24_RAW_ITERATE_OUTPUT",
            object_status=seed_status,
            beta=float(beta),
            gamma=float(gamma),
            q=int(q),
            certified=False,
            fixed_point=bool(seed_fixed),
            re_phi_m=float(np.real(seed_phi)),
            im_phi_m=float(np.imag(seed_phi)),
            prefactor_used_label="CONV2_ALL_SUBSETS_PRIMARY",
            prefactor_used=pref_conv2,
            re_phi_plus_prefactor=float(np.real(seed_phi)) + pref_conv2,
            conv1_prefactor=pref_conv1,
            conv2_prefactor=pref_conv2,
            re_phi_plus_conv1=float(np.real(seed_phi)) + pref_conv1,
            re_phi_plus_conv2=float(np.real(seed_phi)) + pref_conv2,
            residual_inf=float(seed_diag["res_inf"]),
            residual_l2=float(seed_diag["res_l2"]),
            scaled_res_inf=float(seed_diag["scaled_res_inf"]),
            componentwise_rel=float(seed_diag["componentwise_rel"]),
            has_nan_or_inf=bool(seed_diag["has_nan_or_inf"]),
            matches_lambda_abs_conv1=seed_matches["abs_conv1"],
            matches_lambda_local_conv1=seed_matches["loc_conv1"],
            matches_lambda_abs_conv2=seed_matches["abs_conv2"],
            matches_lambda_local_conv2=seed_matches["loc_conv2"],
            z_real=seed_z.real.tolist(),
            z_imag=seed_z.imag.tolist(),
            note="Raw output from generalized_binomial_sum_scaling_exponent_ksat (not assumed saddle); Conv2 baseline is primary.",
        )
    )

    if seed_polished is not None:
        z_pol = _x_to_z(seed_polished, sys.nvars)
        phi_pol = compute_phi(z_pol, q=q, r=r, betas=np.array([beta]), gammas=np.array([gamma]))
        diag_pol = diagnose_bm24_iterate(z_pol, sys)
        status_pol, fixed_pol = classify_bm24_iterate(diag_pol)
        m_pol = _match_flags(float(np.real(phi_pol)), lam_abs, lam_loc, pref_conv1, pref_conv2, tol)
        rows.append(
            ObjectComparisonRow(
                object_type="NEWTON_POLISHED_FROM_BM24_ITERATE",
                object_status=status_pol,
                beta=float(beta),
                gamma=float(gamma),
                q=int(q),
                certified=False,
                fixed_point=bool(fixed_pol),
                re_phi_m=float(np.real(phi_pol)),
                im_phi_m=float(np.imag(phi_pol)),
                prefactor_used_label="CONV2_ALL_SUBSETS_PRIMARY",
                prefactor_used=pref_conv2,
                re_phi_plus_prefactor=float(np.real(phi_pol)) + pref_conv2,
                conv1_prefactor=pref_conv1,
                conv2_prefactor=pref_conv2,
                re_phi_plus_conv1=float(np.real(phi_pol)) + pref_conv1,
                re_phi_plus_conv2=float(np.real(phi_pol)) + pref_conv2,
                residual_inf=float(diag_pol["res_inf"]),
                residual_l2=float(diag_pol["res_l2"]),
                scaled_res_inf=float(diag_pol["scaled_res_inf"]),
                componentwise_rel=float(diag_pol["componentwise_rel"]),
                has_nan_or_inf=bool(diag_pol["has_nan_or_inf"]),
                matches_lambda_abs_conv1=m_pol["abs_conv1"],
                matches_lambda_local_conv1=m_pol["loc_conv1"],
                matches_lambda_abs_conv2=m_pol["abs_conv2"],
                matches_lambda_local_conv2=m_pol["loc_conv2"],
                z_real=z_pol.real.tolist(),
                z_imag=z_pol.imag.tolist(),
                note="Newton-polished root initialized from BM24 iterate.",
            )
        )

    cert_seed = next((s for s in saddles_anchor if s.is_bm24_seed_saddle and s.certified), None)
    if cert_seed is not None:
        zc = np.array(cert_seed.z_real) + 1j * np.array(cert_seed.z_imag)
        diag_c = diagnose_bm24_iterate(zc, sys)
        status_c, fixed_c = classify_bm24_iterate(diag_c)
        rows.append(
            ObjectComparisonRow(
                object_type="KRAWCZYK_CERTIFIED_ROOT_NEAR_BM24_SEED",
                object_status=status_c,
                beta=float(beta),
                gamma=float(gamma),
                q=int(q),
                certified=True,
                fixed_point=bool(fixed_c),
                re_phi_m=float(cert_seed.re_phi),
                im_phi_m=float(cert_seed.im_phi),
                prefactor_used_label="CONV2_ALL_SUBSETS_PRIMARY",
                prefactor_used=pref_conv2,
                re_phi_plus_prefactor=float(cert_seed.re_phi) + pref_conv2,
                conv1_prefactor=pref_conv1,
                conv2_prefactor=pref_conv2,
                re_phi_plus_conv1=float(cert_seed.re_phi) + pref_conv1,
                re_phi_plus_conv2=float(cert_seed.re_phi) + pref_conv2,
                residual_inf=float(diag_c["res_inf"]),
                residual_l2=float(diag_c["res_l2"]),
                scaled_res_inf=float(diag_c["scaled_res_inf"]),
                componentwise_rel=float(diag_c["componentwise_rel"]),
                has_nan_or_inf=bool(diag_c["has_nan_or_inf"]),
                matches_lambda_abs_conv1=bool(cert_seed.matches_lambda_abs_full),
                matches_lambda_local_conv1=bool(cert_seed.matches_lambda_local),
                matches_lambda_abs_conv2=bool(cert_seed.matches_lambda_abs_conv2),
                matches_lambda_local_conv2=bool(
                    lam_loc is not None and abs(lam_loc - (float(cert_seed.re_phi) + pref_conv2)) < tol
                ),
                z_real=cert_seed.z_real,
                z_imag=cert_seed.z_imag,
                note="Krawczyk-certified root selected as nearest certified to BM24 seed action.",
            )
        )
    return rows


def run_small_gamma_sanity_checks(
    *,
    r: float,
    beta: float,
    gammas: list[float],
    dps: int,
    z_tol: float = 1e-4,
    phi_tol: float = 1e-5,
) -> dict:
    q = 1
    checks = []
    all_ok = True
    for g in gammas:
        sys = SaddleSystem.build(q=q, r=r, betas=np.array([beta]), gammas=np.array([g]))
        z_it, phi_it, _ = bm24_seed_z(q, r, beta, g)
        diag_it = diagnose_bm24_iterate(z_it, sys)
        x_it = np.concatenate([z_it.real, z_it.imag])
        x_pol = polish_root_x(sys, x_it)
        if x_pol is None:
            checks.append({"gamma": g, "ok": False, "reason": "newton_polish_failed"})
            all_ok = False
            continue
        z_pol = _x_to_z(x_pol, sys.nvars)
        phi_pol = compute_phi(z_pol, q=q, r=r, betas=np.array([beta]), gammas=np.array([g]))
        z_dist = float(np.linalg.norm(z_it - z_pol, ord=np.inf))
        phi_diff = float(abs(phi_it - phi_pol))
        cert_ok, _info = krawczyk_certify_with_escalation(sys, z_pol, dps_start=dps, max_inflate_iters=6)
        ok = bool(cert_ok and z_dist < z_tol and phi_diff < phi_tol)
        all_ok = all_ok and ok
        checks.append(
            {
                "gamma": float(g),
                "ok": ok,
                "iterate_status": classify_bm24_iterate(diag_it)[0],
                "iterate_res_inf": float(diag_it["res_inf"]),
                "z_inf_distance_iter_vs_polished": z_dist,
                "phi_abs_diff_iter_vs_polished": phi_diff,
                "krawczyk_certified_polished": bool(cert_ok),
            }
        )
    return {"all_passed": all_ok, "checks": checks, "z_tol": z_tol, "phi_tol": phi_tol, "q": q}


def certify_and_analyse_saddles(
    q: int,
    r: float,
    beta: float,
    gamma: float,
    roots_x: list[np.ndarray],
    finite_n: dict,
    *,
    rigorous: bool,
    dps: int,
    krawczyk_radius: float,
    krawczyk_samples: int,
    seed: int,
    match_tol: float,
) -> list[SaddleRecord]:
    sys = SaddleSystem.build(q=q, r=r, betas=np.array([beta]), gammas=np.array([gamma]))
    betas = sys.betas
    gammas = sys.gammas
    nvars = sys.nvars

    seed_z, seed_phi, _ = bm24_seed_z(q, r, beta, gamma)
    seed_x = np.concatenate([seed_z.real, seed_z.imag])

    ns = finite_n["n_values"]
    n_large = max(ns)
    n_prev = max(n for n in ns if n + 1 in ns) if ns else n_large
    lam_abs = finite_n["lambda_abs"].get(str(n_large))
    lam_loc = finite_n["lambda_local"].get(str(n_prev))
    phi_pref = finite_n["phi_pref_prop4"]
    phi_pref_conv2 = float(bm24_prefactor_exponent_ksat_all_subsets(k=2**q, r=r))  # BM24: r/2^K_clause with K_clause=2^q
    match_tol_eff = max(match_tol, 5e-4 * abs(lam_abs)) if lam_abs is not None else match_tol

    records: list[SaddleRecord] = []
    re_phis: list[float] = []

    for sid, x in enumerate(roots_x):
        x_polished, residual = _scipy_polish_to_tol(sys, x, target_inf=1e-25, max_tries=6, tol=1e-14)
        if residual > 1e-25:
            x_polished, residual = _mpmath_newton_polish(sys, x_polished, target_inf=1e-25, dps=max(120, 2 * dps), max_iters=12)
        z = _x_to_z(x_polished, nvars)
        j = numerical_jacobian(sys.G_real, x_polished)
        jcond = float(np.linalg.cond(j)) if j.size else float("inf")

        if rigorous:
            ok, info = _adaptive_krawczyk_with_diagnostics(sys, z, dps=max(dps, 80))
            if not ok:
                ok2, info2 = krawczyk_certify_with_escalation(sys, z, dps_start=max(dps, 80), max_inflate_iters=8)
                if ok2:
                    ok, info = ok2, info2
            radii = info.get("box_radius", [])
            box_r = float(min(radii)) if radii else float("nan")
            kappa = float(info.get("contraction_bound", math.inf))
        else:
            ok, kappa = krawczyk_test_sampled(
                sys.G_real, x_polished, radius=krawczyk_radius, samples=krawczyk_samples, seed=seed + sid
            )
            box_r = krawczyk_radius if ok else float("nan")
            info = {"sampled": True, "contraction_bound": kappa}

        if not ok:
            tested = info.get("tested_radii", [])
            print(
                "[KRAWCZYK_FAIL]"
                f" gamma={gamma:+.6g} sid={sid}"
                f" ||G||_inf={residual:.3e}"
                f" cond(J)={jcond:.3e}"
                f" ||Y G||_inf={float(info.get('Y_G_inf', float('nan'))):.3e}"
                f" tested_box_radii={tested}"
                f" inclusion={any(bool(t.get('inclusion', False)) for t in tested) if tested else False}"
                f" contraction_bound={float(info.get('contraction_bound', float('nan'))):.3e}"
            )

        phi = compute_phi(z, q=q, r=r, betas=betas, gammas=gammas)
        re_phis.append(phi.real)

        records.append(
            SaddleRecord(
                saddle_id=sid,
                beta=float(beta),
                gamma=float(gamma),
                certified=bool(ok),
                box_radius=box_r,
                residual=residual,
                jacobian_cond=jcond,
                re_phi=float(phi.real),
                im_phi=float(phi.imag),
                det_hessian_re_phi=float("nan"),
                nearest_action_gap=float("nan"),
                re_phi_plus_pref=float(phi.real) + phi_pref,
                re_phi_plus_pref_conv2=float(phi.real) + phi_pref_conv2,
                matches_lambda_abs=False,
                matches_lambda_local=False,
                matches_lambda_abs_full=False,
                matches_lambda_abs_conv2=False,
                controls_finite_n=False,
                is_bm24_seed_saddle=False,
                z_real=z.real.tolist(),
                z_imag=z.imag.tolist(),
                krawczyk_contraction=kappa,
                proof=info if rigorous else {},
            )
        )

    certified_indices = [i for i, r in enumerate(records) if r.certified]
    certified_re = [records[i].re_phi for i in certified_indices]
    for rec in records:
        if rec.certified:
            others = [records[j].re_phi for j in certified_indices if j != rec.saddle_id]
            rec.nearest_action_gap = nearest_action_gap(rec.re_phi, others)
            z_arr = np.array(rec.z_real) + 1j * np.array(rec.z_imag)
            rec.det_hessian_re_phi = _hessian_det_re_phi(z_arr, q, r, betas, gammas)
        else:
            others = [r for i, r in enumerate(re_phis) if i != rec.saddle_id]
            rec.nearest_action_gap = nearest_action_gap(rec.re_phi, others)

        if lam_abs is not None:
            rec.matches_lambda_abs = abs(lam_abs - rec.re_phi_plus_pref_conv2) < match_tol_eff
            rec.matches_lambda_abs_full = abs(lam_abs - rec.re_phi_plus_pref) < match_tol_eff
            rec.matches_lambda_abs_conv2 = abs(lam_abs - rec.re_phi_plus_pref_conv2) < match_tol_eff
        if lam_loc is not None:
            rec.matches_lambda_local = abs(lam_loc - rec.re_phi_plus_pref_conv2) < match_tol_eff

    certified_records = [r for r in records if r.certified]
    if certified_records:
        seed_re = float(np.real(seed_phi))
        best = min(certified_records, key=lambda r: abs(r.re_phi - seed_re))
        best.is_bm24_seed_saddle = True

    for rec in records:
        rec.controls_finite_n = rec.matches_lambda_abs_conv2 and rec.certified

    return records


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def plot_exponent_vs_saddles(
    finite_n: dict,
    saddles: list[SaddleRecord],
    object_rows: list[ObjectComparisonRow],
    out_path: Path,
    title: str,
) -> None:
    ns = sorted(n for n in finite_n["n_values"] if n > 0)
    lam_abs = [finite_n["lambda_abs"][str(n)] for n in ns]
    lam_loc_n = [n for n in ns if str(n) in finite_n["lambda_local"]]
    lam_loc = [finite_n["lambda_local"][str(n)] for n in lam_loc_n]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(ns, lam_abs, "o-", label=r"$\lambda_{\mathrm{abs}}(n)=\ln|P_n|/n$")
    if lam_loc:
        ax.plot(
            [n + 0.5 for n in lam_loc_n],
            lam_loc,
            "s--",
            label=r"$\lambda_{\mathrm{local}}(n)=\ln|P_{n+1}/P_n|$",
        )

    phi_pref = float(finite_n.get("phi_pref_prop4", float("nan")))
    phi_pref_conv2 = float(finite_n.get("phi_pref_conv2_all_subsets", float("nan")))
    if np.isfinite(phi_pref):
        ax.axhline(phi_pref, color="k", ls="-", lw=1.2, alpha=0.7, label=r"$\phi_{\mathrm{pref}}^{\mathrm{Conv1}}$")
    if np.isfinite(phi_pref_conv2):
        ax.axhline(
            phi_pref_conv2,
            color="gray",
            ls="-.",
            lw=1.0,
            alpha=0.7,
            label=r"$\phi_{\mathrm{pref}}^{\mathrm{Conv2}}$",
        )

    certified = [s for s in saddles if s.certified]
    for s in certified:
        ax.axhline(s.re_phi, color="C2", alpha=0.2, lw=0.6)
        ax.axhline(s.re_phi_plus_pref, color="C5", alpha=0.15, lw=0.6)
    if certified:
        dom = max(certified, key=lambda s: s.re_phi)
        ax.axhline(
            dom.re_phi,
            color="C3",
            ls="--",
            lw=1.2,
            label=f"max certified Re Φ_M (id={dom.saddle_id})",
        )
        dom_full = max(certified, key=lambda s: s.re_phi_plus_pref)
        ax.axhline(
            dom_full.re_phi_plus_pref,
            color="C6",
            ls="--",
            lw=1.5,
            label=f"max certified Re Φ_M+pref (id={dom_full.saddle_id})",
        )
        seed = next((s for s in certified if s.is_bm24_seed_saddle), None)
        if seed is not None:
            ax.axhline(
                seed.re_phi,
                color="C4",
                ls=":",
                lw=1.2,
                label=f"BM24 seed track Re Φ_M (id={seed.saddle_id})",
            )
            ax.axhline(
                seed.re_phi_plus_pref,
                color="C7",
                ls=":",
                lw=1.2,
                label=f"Certified-near-seed Re Φ_M+Conv1 pref",
            )

    for obj in object_rows:
        if obj.object_type == "BM24_RAW_ITERATE_OUTPUT":
            ax.axhline(obj.re_phi_m, color="tab:red", ls=":", lw=1.2, label="BM24 iterate Re Φ_M")
            ax.axhline(obj.re_phi_plus_conv1, color="tab:red", ls="--", lw=1.2, label="BM24 iterate Re Φ_M+Conv1")
            ax.axhline(obj.re_phi_plus_conv2, color="tab:red", ls="-.", lw=1.2, label="BM24 iterate Re Φ_M+Conv2")
        if obj.object_type == "NEWTON_POLISHED_FROM_BM24_ITERATE":
            ax.axhline(obj.re_phi_m, color="tab:blue", ls=":", lw=1.1, label="Newton-polished Re Φ_M")
            ax.axhline(obj.re_phi_plus_conv1, color="tab:blue", ls="--", lw=1.1, label="Newton-polished Re Φ_M+Conv1")
        if obj.object_type == "KRAWCZYK_CERTIFIED_ROOT_NEAR_BM24_SEED":
            ax.axhline(obj.re_phi_m, color="tab:green", ls=":", lw=1.1, label="Krawczyk-certified Re Φ_M")
            ax.axhline(obj.re_phi_plus_conv1, color="tab:green", ls="--", lw=1.1, label="Krawczyk-certified Re Φ_M+Conv1")

    ax.set_xlabel("n")
    ax.set_ylabel("exponent (natural log)")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_gamma_stokes(scan_rows: list[dict], out_path: Path) -> None:
    if not scan_rows:
        return
    gammas = [row["gamma"] for row in scan_rows]
    n_stokes = [row["near_stokes_count"] for row in scan_rows]
    n_anti = [row["near_anti_stokes_count"] for row in scan_rows]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(gammas, n_stokes, "o-", label="near-Stokes pairs (|ΔIm| small)")
    ax.plot(gammas, n_anti, "s-", label="near-anti-Stokes pairs (|ΔRe| small)")
    ax.set_xlabel(r"$\gamma$")
    ax.set_ylabel("pair count")
    ax.set_title("Stokes / anti-Stokes events along γ scan (principal-log ImΦ; unwrap needed)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Full audit driver
# ---------------------------------------------------------------------------


def run_audit(
    *,
    q: int = 3,
    r: float = 176.54,
    beta: Optional[float] = None,
    gamma: Optional[float] = None,
    n_min: int = 12,
    n_max: int = 24,
    gamma_center: Optional[float] = None,
    gamma_half_width: float = 0.35,
    gamma_steps: int = 9,
    num_starts: int = 1500,
    seed: int = 0,
    rigorous: bool = True,
    dps: int = 60,
    krawczyk_radius: float = 1e-6,
    krawczyk_samples: int = 128,
    stokes_tol: float = 0.1,
    match_tol: float = 0.02,
    max_certify: int = 0,
    sanity_check: bool = True,
    sanity_beta: Optional[float] = None,
    out_dir: Optional[Path] = None,
) -> Path:
    K_clause = 2**q
    if beta is None or gamma is None:
        cfg = optimal_angles[(K_clause, r)][1]
        beta = float(cfg["betas"][0]) if beta is None else beta
        gamma = float(cfg["gammas"][0]) if gamma is None else gamma

    ts = datetime.now(timezone.utc).strftime("run_%m-%d_%H-%M-%SZ")
    run_dir = (out_dir or DEFAULT_OUT) / ts
    run_dir.mkdir(parents=True, exist_ok=True)

    n_values = list(range(n_min, n_max + 1))
    finite_n = finite_n_exponent_grid(K_clause, q, r, beta, gamma, n_values)
    finite_n["convention_labels"] = {
        "conv2_exact_finite_n_rate_primary": "lambda_abs from generalized_flip_symmetric_expected_success_p1",
        "conv1_exact_finite_n_rate_crosscheck": "crosscheck_lambda_abs_prop4 from Prop4 Eq(A10)",
        "conv1_prefactor": "phi_pref_prop4",
        "conv2_prefactor": "phi_pref_conv2_all_subsets",
        "hybrid_note": "Any Re Phi_M + Conv1 comparison is hybrid (Conv2 saddle with Conv1 prefactor).",
    }

    sanity = None
    if sanity_check and q == 3:
        beta_sanity = sanity_beta
        if beta_sanity is None:
            try:
                beta_sanity = float(optimal_angles[(2**1, r)][1]["betas"][0])
            except KeyError:
                beta_sanity = float(beta)
        sanity = run_small_gamma_sanity_checks(
            r=r,
            beta=float(beta_sanity),
            gammas=[1e-3, 1e-2, 5e-2],
            dps=max(40, dps),
        )
        if not sanity["all_passed"]:
            with open(run_dir / "sanity_checks.json", "w", encoding="utf-8") as fh:
                json.dump({"metadata": {"q": 1, "r": r, "beta": float(beta_sanity)}, **sanity}, fh, indent=2)
            raise RuntimeError(
                "Small-gamma sanity checks failed at q=1. "
                "Audit aborted before q=3 run; inspect sanity_checks.json."
            )

    g0 = gamma_center if gamma_center is not None else gamma
    gamma_grid = np.linspace(g0 - gamma_half_width, g0 + gamma_half_width, gamma_steps)

    chain = discover_roots_continuation(
        q=q,
        r=r,
        beta=beta,
        gamma_values=gamma_grid,
        num_random_starts=num_starts,
        seed=seed,
        tol=1e-12,
        dedupe_tol=1e-5,
    )

    anchor_gamma = float(gamma_grid[np.argmin(np.abs(gamma_grid - gamma))])
    anchor_roots = list(chain[anchor_gamma])
    sys_anchor = SaddleSystem.build(q=q, r=r, betas=np.array([beta]), gammas=np.array([gamma]))
    anchor_roots = dedupe_roots_x(
        anchor_roots + discover_roots(sys_anchor, num_starts=num_starts, seed=seed + 1, tol=1e-12),
        tol=1e-5,
    )
    seed_z, seed_phi, seed_converged = bm24_seed_z(q, r, beta, gamma)
    seed_x = np.concatenate([seed_z.real, seed_z.imag])
    seed_polished = polish_root_x(sys_anchor, seed_x)
    for x in (seed_polished, seed_x):
        if x is not None and not any(np.linalg.norm(x - y, ord=np.inf) < 1e-5 for y in anchor_roots):
            anchor_roots.insert(0, x)
    if max_certify > 0:
        anchor_roots = anchor_roots[:max_certify]
    saddles_anchor = certify_and_analyse_saddles(
        q,
        r,
        beta,
        gamma,
        anchor_roots,
        finite_n,
        rigorous=rigorous,
        dps=dps,
        krawczyk_radius=krawczyk_radius,
        krawczyk_samples=krawczyk_samples,
        seed=seed,
        match_tol=match_tol,
    )
    seed_diag = diagnose_bm24_iterate(seed_z, sys_anchor)
    seed_status, seed_fixed = classify_bm24_iterate(seed_diag)
    if not seed_converged:
        seed_status = BM24_NONCONVERGED_ITERATE
        seed_fixed = False
    object_rows = build_object_comparison_rows(
        q=q,
        r=r,
        beta=beta,
        gamma=gamma,
        finite_n=finite_n,
        sys=sys_anchor,
        seed_z=seed_z,
        seed_phi=seed_phi,
        seed_diag=seed_diag,
        seed_status=seed_status,
        seed_fixed=seed_fixed,
        seed_polished=seed_polished,
        saddles_anchor=saddles_anchor,
        match_tol=match_tol,
    )

    gamma_scan: list[dict] = []
    for gval, roots_x in chain.items():
        betas_arr = np.array([beta])
        gammas_arr = np.array([gval])
        sys = SaddleSystem.build(q=q, r=r, betas=betas_arr, gammas=gammas_arr)
        phi_list = []
        for x in roots_x:
            z = _x_to_z(x, sys.nvars)
            if np.linalg.norm(sys.G_complex(z), ord=np.inf) > 1e-6:
                continue
            phi_list.append(compute_phi(z, q=q, r=r, betas=betas_arr, gammas=gammas_arr))

        class _S:
            def __init__(self, i: int, phi: complex):
                self.idx = i
                self.re_phi = phi.real
                self.im_phi = phi.imag

        stub = [_S(i, p) for i, p in enumerate(phi_list)]
        stokes_pairs = detect_stokes_pairs(stub, stokes_tol=stokes_tol)  # type: ignore[arg-type]
        gamma_scan.append(
            {
                "gamma": gval,
                "warning": "Im(Phi) uses principal complex log branch; unwrap along continuous saddle tracks before claiming Stokes events.",
                "num_candidates": len(roots_x),
                "num_residual_ok": len(phi_list),
                "near_stokes_count": sum(1 for sp in stokes_pairs if sp.near_stokes),
                "near_anti_stokes_count": sum(1 for sp in stokes_pairs if sp.near_anti_stokes),
                "stokes_pairs": [
                    {
                        "i": sp.i,
                        "j": sp.j,
                        "delta_re": sp.delta_re,
                        "delta_im": sp.delta_im,
                        "near_stokes": sp.near_stokes,
                        "near_anti_stokes": sp.near_anti_stokes,
                    }
                    for sp in stokes_pairs
                    if sp.near_stokes or sp.near_anti_stokes
                ],
            }
        )

    meta = {
        "q": q,
        "K_clause": K_clause,
        "r": r,
        "p": 1,
        "beta": beta,
        "gamma": gamma,
        "n_grid": n_values,
        "gamma_scan": gamma_grid.tolist(),
        "rigorous_krawczyk": rigorous,
        "timestamp": ts,
    }

    with open(run_dir / "finite_n_exponents.json", "w", encoding="utf-8") as fh:
        json.dump({"metadata": meta, **finite_n}, fh, indent=2)

    with open(run_dir / "gamma_scan.json", "w", encoding="utf-8") as fh:
        json.dump({"metadata": meta, "scan": gamma_scan}, fh, indent=2)
    if sanity is not None:
        with open(run_dir / "sanity_checks.json", "w", encoding="utf-8") as fh:
            json.dump({"metadata": meta, **sanity}, fh, indent=2)

    rows = [s.to_row() for s in saddles_anchor]
    with open(run_dir / "saddle_table.json", "w", encoding="utf-8") as fh:
        json.dump({"metadata": meta, "saddles": rows}, fh, indent=2)

    fieldnames = list(rows[0].keys()) if rows else []
    if fieldnames:
        with open(run_dir / "saddle_table.csv", "w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                flat = dict(row)
                flat["proof"] = json.dumps(flat.get("proof", {}))
                writer.writerow(flat)

    object_rows_dict = [r.to_row() for r in object_rows]
    with open(run_dir / "object_comparison_table.json", "w", encoding="utf-8") as fh:
        json.dump({"metadata": meta, "objects": object_rows_dict}, fh, indent=2)
    if object_rows_dict:
        object_fields = list(object_rows_dict[0].keys())
        with open(run_dir / "object_comparison_table.csv", "w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=object_fields)
            writer.writeheader()
            for row in object_rows_dict:
                writer.writerow(row)

    plot_exponent_vs_saddles(
        finite_n,
        saddles_anchor,
        object_rows,
        run_dir / "exponent_vs_saddles.png",
        title=rf"p=1 audit: $\lambda$ vs Re $\Phi_j$ ($\beta$={beta:.4f}, $\gamma$={gamma:.4f})",
    )
    plot_gamma_stokes(gamma_scan, run_dir / "gamma_stokes_scan.png")

    phi_pref = finite_n["phi_pref_prop4"]
    lam_abs = finite_n["lambda_abs"].get(str(max(n_values)))
    seed_re_phi = float(np.real(seed_phi))
    seed_full_conv1 = seed_re_phi + phi_pref
    seed_full_conv2 = seed_re_phi + float(finite_n["phi_pref_conv2_all_subsets"])
    certified = [s for s in saddles_anchor if s.certified]
    closest_cert = (
        min(certified, key=lambda s: abs(s.re_phi_plus_pref_conv2 - float(lam_abs)))
        if certified and lam_abs is not None
        else None
    )

    summary = {
        "metadata": meta,
        "num_saddle_candidates": len(anchor_roots),
        "num_certified": sum(1 for s in saddles_anchor if s.certified),
        "dominant_certified_re_phi": max(
            (s.re_phi for s in saddles_anchor if s.certified), default=float("nan")
        ),
        "lambda_abs_at_n_max": lam_abs,
        "match_tol_effective": max(match_tol, 5e-4 * abs(float(lam_abs))) if lam_abs is not None else match_tol,
        "controlling_saddle_ids": [s.saddle_id for s in saddles_anchor if s.controls_finite_n],
        "conventions": {
            "conv1_prefactor": float(finite_n["phi_pref_prop4"]),
            "conv2_prefactor": float(finite_n["phi_pref_conv2_all_subsets"]),
            "note": "Primary baseline is Conv2 exact finite-n. Conv1 comparisons are cross-check/hybrid only.",
        },
        "bm24_scaling_iterate": {
            "status": seed_status,
            "fixed_point": bool(seed_fixed),
            "converged_flag_from_iterator": bool(seed_converged),
            "re_phi": seed_re_phi,
            "re_phi_plus_conv1_pref": seed_full_conv1,
            "re_phi_plus_conv2_pref": seed_full_conv2,
            "G_residual_inf": float(seed_diag["res_inf"]),
            "G_residual_l2": float(seed_diag["res_l2"]),
            "scaled_res_inf": float(seed_diag["scaled_res_inf"]),
            "componentwise_rel": float(seed_diag["componentwise_rel"]),
            "matches_lambda_abs_conv2_primary": abs(float(lam_abs) - seed_full_conv2) < max(match_tol, 5e-4 * abs(float(lam_abs)))
            if lam_abs is not None
            else False,
            "matches_lambda_abs_conv1_hybrid": abs(float(lam_abs) - seed_full_conv1) < max(match_tol, 5e-4 * abs(float(lam_abs)))
            if lam_abs is not None
            else False,
            "note": "Iterator output is not labeled as saddle/controller unless converged and fixed-point diagnostics pass.",
        },
        "closest_certified_to_lambda": (
            {
                "saddle_id": closest_cert.saddle_id,
                "re_phi": closest_cert.re_phi,
                "re_phi_plus_pref_conv2": closest_cert.re_phi_plus_pref_conv2,
                "gap_conv2_primary": abs(closest_cert.re_phi_plus_pref_conv2 - float(lam_abs)),
                "re_phi_plus_pref_conv1_hybrid": closest_cert.re_phi_plus_pref,
                "is_bm24_seed_saddle": closest_cert.is_bm24_seed_saddle,
            }
            if closest_cert is not None and lam_abs is not None
            else None
        ),
        "bm24_seed": {
            "re_phi": seed_re_phi,
            "phi_pref_prop4": phi_pref,
            "phi_pref_conv2": float(bm24_prefactor_exponent_ksat_all_subsets(k=K_clause, r=r)),
        },
        "object_table_path": str(run_dir / "object_comparison_table.json"),
    }
    with open(run_dir / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"Results written to {run_dir}")
    return run_dir


def main() -> None:
    p = argparse.ArgumentParser(description="BM24 p=1 saddle audit (z-chart Krawczyk)")
    p.add_argument("--q", type=int, default=3)
    p.add_argument("--r", type=float, default=176.54)
    p.add_argument("--beta", type=float, default=None)
    p.add_argument("--gamma", type=float, default=None)
    p.add_argument("--n-min", type=int, default=12)
    p.add_argument("--n-max", type=int, default=24)
    p.add_argument("--gamma-center", type=float, default=None)
    p.add_argument("--gamma-half-width", type=float, default=0.35)
    p.add_argument("--gamma-steps", type=int, default=9)
    p.add_argument("--num-starts", type=int, default=1500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dps", type=int, default=60)
    p.add_argument("--out-dir", type=str, default="")
    p.add_argument("--match-tol", type=float, default=0.02)
    p.add_argument("--max-certify", type=int, default=0, help="Cap saddles certified at anchor γ (0=all)")
    p.add_argument("--stokes-tol", type=float, default=0.1)
    p.add_argument("--no-sanity-check", dest="sanity_check", action="store_false")
    p.set_defaults(sanity_check=True)
    p.add_argument("--quick", action="store_true", help="Smaller grids / sampled Krawczyk")
    p.add_argument("--rigorous", action="store_true", default=True)
    p.add_argument("--no-rigorous", dest="rigorous", action="store_false")
    args = p.parse_args()

    kwargs = dict(
        q=args.q,
        r=args.r,
        beta=args.beta,
        gamma=args.gamma,
        n_min=args.n_min,
        n_max=args.n_max,
        gamma_center=args.gamma_center,
        gamma_half_width=args.gamma_half_width,
        gamma_steps=args.gamma_steps,
        num_starts=args.num_starts,
        seed=args.seed,
        rigorous=args.rigorous,
        dps=args.dps,
        stokes_tol=args.stokes_tol,
        match_tol=args.match_tol,
        max_certify=args.max_certify,
        sanity_check=args.sanity_check,
        out_dir=Path(args.out_dir) if args.out_dir else None,
    )
    if args.quick:
        kwargs.update(
            n_min=10,
            n_max=14,
            gamma_steps=5,
            gamma_half_width=0.2,
            num_starts=200,
            dps=40,
            rigorous=False,
        )

    run_audit(**kwargs)


if __name__ == "__main__":
    main()
