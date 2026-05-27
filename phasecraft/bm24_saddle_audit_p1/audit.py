"""
BM24 generalized multinomial saddle audit at p=1, q=3 (k=8), r=176.54.

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
import numpy as np
from scipy.optimize import root
from scipy.special import gammaln

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from phasecraft.krawczyk_p1_roots import (
    SaddleSystem,
    discover_roots,
    krawczyk_certify_with_escalation,
    krawczyk_test_sampled,
    numerical_jacobian,
)
from phasecraft.optimal_angles import optimal_angles
from phasecraft.picard_lefschetz import compute_phi, detect_stokes_pairs
from phasecraft.saddle_traits import complex_to_real_vector

_PATCHED_PATH = REPO_ROOT / "phasecraft" / "generalized_binomial_sum.PATCHED.py"
_PATCHED_PARENT = _PATCHED_PATH.parent
_PATCHED_SPEC = importlib.util.spec_from_file_location(
    "phasecraft._generalized_binomial_sum_patched",
    _PATCHED_PATH,
    submodule_search_locations=[str(_PATCHED_PARENT)],
)
_PATCHED_MOD = importlib.util.module_from_spec(_PATCHED_SPEC)
assert _PATCHED_SPEC is not None and _PATCHED_SPEC.loader is not None
sys.modules[_PATCHED_SPEC.name] = _PATCHED_MOD
_PATCHED_SPEC.loader.exec_module(_PATCHED_MOD)

generalized_binomial_sum_scaling_exponent_ksat = _PATCHED_MOD.generalized_binomial_sum_scaling_exponent_ksat
bm24_prefactor_exponent_ksat = _PATCHED_MOD.bm24_prefactor_exponent_ksat
bm24_prefactor_exponent_ksat_all_subsets = _PATCHED_MOD.bm24_prefactor_exponent_ksat_all_subsets

AUDIT_DIR = Path(__file__).resolve().parent
DEFAULT_OUT = AUDIT_DIR / "results"


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


def log_p_succ_prop4(k: int, r: float, beta: float, gamma: float, n: int) -> complex:
    """ln E[p_succ] at p=1 from BM24 Prop 4 / Eq. (A10) (Convention 1)."""
    if n < 0:
        raise ValueError("n must be nonnegative")
    q = int(round(math.log2(k)))
    if 2**q != k:
        raise ValueError(f"k={k} must be a power of 2")

    log_pref = -(r / k) * n * (1.0 + 4.0 * math.sin(gamma / 4.0) ** 2)
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
                    4.0 * s2g4 * (((nb + na) / (2 * n)) ** k - (na / (2 * n)) ** k)
                    + (1.0 - e_minus) * ((nc + na) / (2 * n)) ** k
                    + (1.0 - e_plus) * ((nd + na) / (2 * n)) ** k
                )
                terms.append(log_multi + log_B + exp_arg)

    log_inner = _complex_logsumexp(np.asarray(terms, dtype=np.complex128))
    return log_pref + log_inner


def p_succ_prop4(k: int, r: float, beta: float, gamma: float, n: int) -> float:
    return float(np.real(np.exp(log_p_succ_prop4(k, r, beta, gamma, n))))


def finite_n_exponent_grid(
    k: int,
    r: float,
    beta: float,
    gamma: float,
    n_values: Iterable[int],
) -> dict:
    """lambda_abs(n)=log|P_n|/n and lambda_local(n)=log|P_{n+1}/P_n|."""
    ns = [int(n) for n in n_values]
    log_p = {n: log_p_succ_prop4(k, r, beta, gamma, n) for n in ns}
    abs_p = {n: float(np.real(np.exp(lp))) for n, lp in log_p.items()}

    lambda_abs = {n: float(np.real(log_p[n]) / n) for n in ns if n > 0}
    lambda_local = {}
    for n in ns:
        if n + 1 in log_p:
            lambda_local[n] = float(np.real(log_p[n + 1] - log_p[n]))

    return {
        "n_values": ns,
        "log_p": {str(n): {"real": float(np.real(v)), "imag": float(np.imag(v))} for n, v in log_p.items()},
        "p": {str(n): v for n, v in abs_p.items()},
        "lambda_abs": {str(n): v for n, v in lambda_abs.items()},
        "lambda_local": {str(n): v for n, v in lambda_local.items()},
        "phi_pref_prop4": float(-(r / k) * (1.0 + 4.0 * math.sin(gamma / 4.0) ** 2)),
    }


# ---------------------------------------------------------------------------
# Saddle discovery (z chart) + gamma continuation
# ---------------------------------------------------------------------------


def _x_to_z(x: np.ndarray, nvars: int) -> np.ndarray:
    return x[:nvars] + 1j * x[nvars:]


def polish_root_x(sys: SaddleSystem, x: np.ndarray, tol: float = 1e-14) -> Optional[np.ndarray]:
    """Newton polish to G_real(x)=0; returns None if polish fails."""
    sol = root(sys.G_real, x, method="hybr", tol=tol)
    if not sol.success:
        return None
    res = float(np.linalg.norm(sys.G_real(sol.x), ord=np.inf))
    if not np.isfinite(res) or res > 1e-7:
        return None
    return sol.x


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
        seed_z, _ = bm24_seed_z(q, r, beta, float(gamma))
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


def bm24_seed_z(q: int, r: float, beta: float, gamma: float) -> tuple[np.ndarray, complex]:
    """Fixed-point iteration from z=0 (BM24 seed saddle track)."""
    _, z, _, phi_m = generalized_binomial_sum_scaling_exponent_ksat(
        q=q,
        r=r,
        betas=np.array([beta]),
        gammas=np.array([gamma]),
        num_iter=200,
        dz_threshold=1e-10,
        init_z=None,
    )
    return z, complex(phi_m)


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

    seed_z, seed_phi = bm24_seed_z(q, r, beta, gamma)
    seed_x = np.concatenate([seed_z.real, seed_z.imag])

    ns = finite_n["n_values"]
    n_large = max(ns)
    n_prev = max(n for n in ns if n + 1 in ns) if ns else n_large
    lam_abs = finite_n["lambda_abs"].get(str(n_large))
    lam_loc = finite_n["lambda_local"].get(str(n_prev))
    phi_pref = finite_n["phi_pref_prop4"]
    phi_pref_conv2 = float(bm24_prefactor_exponent_ksat_all_subsets(k=2**q, r=r))  # BM24: r/2^k with k=2^q
    match_tol_eff = max(match_tol, 5e-4 * abs(lam_abs)) if lam_abs is not None else match_tol

    records: list[SaddleRecord] = []
    re_phis: list[float] = []

    for sid, x in enumerate(roots_x):
        z = _x_to_z(x, nvars)
        residual = float(np.linalg.norm(sys.G_complex(z), ord=np.inf))
        j = numerical_jacobian(sys.G_real, x)
        jcond = float(np.linalg.cond(j)) if j.size else float("inf")

        if rigorous:
            ok, info = krawczyk_certify_with_escalation(sys, z, dps_start=dps, max_inflate_iters=6)
            radii = info.get("box_radius", [])
            box_r = float(min(radii)) if radii else float("nan")
            kappa = float(info.get("contraction_bound", math.inf))
        else:
            ok, kappa = krawczyk_test_sampled(
                sys.G_real, x, radius=krawczyk_radius, samples=krawczyk_samples, seed=seed + sid
            )
            box_r = krawczyk_radius if ok else float("nan")
            info = {"sampled": True, "contraction_bound": kappa}

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

        targets = [
            rec.re_phi,
            rec.re_phi + phi_pref,
            rec.re_phi + phi_pref_conv2,
            float(np.real(seed_phi)),
            float(np.real(seed_phi)) + phi_pref,
        ]
        if lam_abs is not None:
            rec.matches_lambda_abs = any(abs(lam_abs - t) < match_tol_eff for t in targets[:2])
            rec.matches_lambda_abs_full = abs(lam_abs - rec.re_phi_plus_pref) < match_tol_eff
            rec.matches_lambda_abs_conv2 = abs(lam_abs - rec.re_phi_plus_pref_conv2) < match_tol_eff
        if lam_loc is not None:
            rec.matches_lambda_local = any(abs(lam_loc - t) < match_tol_eff for t in targets)

    certified_records = [r for r in records if r.certified]
    if certified_records:
        seed_re = float(np.real(seed_phi))
        best = min(certified_records, key=lambda r: abs(r.re_phi - seed_re))
        best.is_bm24_seed_saddle = True

    for rec in records:
        rec.controls_finite_n = rec.matches_lambda_abs_full and rec.certified

    return records


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def plot_exponent_vs_saddles(
    finite_n: dict,
    saddles: list[SaddleRecord],
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

    phi_pref = finite_n.get("phi_pref_prop4")
    if phi_pref is not None:
        ax.axhline(phi_pref, color="k", ls="-", lw=1.2, alpha=0.7, label=r"$\phi_{\mathrm{pref}}$ (Prop4)")

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
                label=f"BM24 seed Re Φ_M+pref",
            )

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
    ax.set_title("Stokes / anti-Stokes events along γ scan")
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
    out_dir: Optional[Path] = None,
) -> Path:
    k = 2**q
    if beta is None or gamma is None:
        cfg = optimal_angles[(k, r)][1]
        beta = float(cfg["betas"][0]) if beta is None else beta
        gamma = float(cfg["gammas"][0]) if gamma is None else gamma

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = (out_dir or DEFAULT_OUT) / ts
    run_dir.mkdir(parents=True, exist_ok=True)

    n_values = list(range(n_min, n_max + 1))
    finite_n = finite_n_exponent_grid(k, r, beta, gamma, n_values)

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
    seed_z, seed_phi = bm24_seed_z(q, r, beta, gamma)
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
        "k": k,
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

    plot_exponent_vs_saddles(
        finite_n,
        saddles_anchor,
        run_dir / "exponent_vs_saddles.png",
        title=rf"p=1 audit: $\lambda$ vs Re $\Phi_j$ ($\beta$={beta:.4f}, $\gamma$={gamma:.4f})",
    )
    plot_gamma_stokes(gamma_scan, run_dir / "gamma_stokes_scan.png")

    phi_pref = finite_n["phi_pref_prop4"]
    lam_abs = finite_n["lambda_abs"].get(str(max(n_values)))
    seed_re_phi = float(np.real(seed_phi))
    seed_full = seed_re_phi + phi_pref
    certified = [s for s in saddles_anchor if s.certified]
    closest_cert = (
        min(certified, key=lambda s: abs(s.re_phi_plus_pref - float(lam_abs)))
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
        "bm24_scaling_iterate": {
            "re_phi": seed_re_phi,
            "re_phi_plus_pref": seed_full,
            "G_residual": float(np.linalg.norm(sys_anchor.G_complex(seed_z), ord=np.inf)),
            "matches_lambda_abs_full": abs(float(lam_abs) - seed_full) < max(match_tol, 5e-4 * abs(float(lam_abs)))
            if lam_abs is not None
            else False,
            "note": "Phi at fixed-point iterate; may not satisfy G(z)=0 until polished.",
        },
        "closest_certified_to_lambda": (
            {
                "saddle_id": closest_cert.saddle_id,
                "re_phi": closest_cert.re_phi,
                "re_phi_plus_pref": closest_cert.re_phi_plus_pref,
                "gap": abs(closest_cert.re_phi_plus_pref - float(lam_abs)),
                "is_bm24_seed_saddle": closest_cert.is_bm24_seed_saddle,
            }
            if closest_cert is not None and lam_abs is not None
            else None
        ),
        "bm24_seed": {
            "re_phi": seed_re_phi,
            "phi_pref_prop4": phi_pref,
            "phi_pref_conv2": float(bm24_prefactor_exponent_ksat_all_subsets(k=k, r=r)),
        },
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
