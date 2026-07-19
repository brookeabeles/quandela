"""
Certified continuation of the BM24 seed saddle branch in gamma.

Tracks one z-branch from small negative gamma (certified, Conv2-matched) toward
more negative gamma. Does not claim contour dominance; reports rigorous
Krawczyk certification and Conv2 action vs exact finite-n baseline.

Convention (primary throughout):
    full_exponent = Re Phi_M + bm24_prefactor_exponent_ksat_all_subsets(k=K_clause, r=r)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from phasecraft.bm24_saddle_audit_p1.audit import (
    AUDIT_DIR,
    bm24_prefactor_exponent_ksat_all_subsets,
    bm24_seed_z,
    classify_bm24_iterate,
    diagnose_bm24_iterate,
    finite_n_exponent_grid,
    _adaptive_krawczyk_with_diagnostics,
    _mpmath_newton_polish,
    _scipy_polish_to_tol,
    _x_to_z,
)
from phasecraft.lib.saddles.krawczyk_p1_roots import (
    SaddleSystem,
    discover_roots,
    krawczyk_certify_with_escalation,
    numerical_jacobian,
)

from phasecraft.lib.saddles.picard_lefschetz import compute_phi

IM_PHI_WARNING = (
    "Im(Phi) uses principal np.log; raw Im Phi can jump by 2*pi along gamma. "
    "Do not infer Stokes events without branch-unwrapped phases."
)

# γ axis for exponent comparison plots: 0 (left) → −2π (right), ticks in π/4 steps.
GAMMA_AXIS_LEFT = 0.0
GAMMA_AXIS_RIGHT = -2.0 * math.pi
GAMMA_PI_TICK_STEP = 0.25 * math.pi

LABEL_SEED_SADDLE_EXPONENT = (
    r"Krawczyk-certified seed-saddle exponent ($\mathrm{Re}\,\Phi_M + \mathrm{BM24}$ A41 prefactor)"
)
LABEL_EXACT_FINITE_N_EXPONENT = r"Exact finite-n exponent at $n_{\max}$"
LABEL_SEED_SADDLE_PAPER = r"Krawczyk-certified seed saddle"
LABEL_EXACT_FINITE_N_PAPER = r"Exact finite-$n$"
TITLE_GAMMA_VS_EXPONENTS = (
    r"Seed-saddle exponent vs exact finite-$n$ exponent (BM24 $q{=}3$, $p{=}1$)"
)

GAMMA_ABS_AXIS_RIGHT = 2.0 * math.pi
ZOOM_INSET_ABS_GAMMA_MAX = 0.75


def _gamma_pi_tick_label(gamma: float) -> str:
    if abs(gamma) < 1e-12:
        return r"$0$"
    k = -gamma / math.pi
    for den in (1, 2, 4):
        num = round(k * den)
        if num == 0 or abs(k - num / den) >= 1e-6:
            continue
        if den == 1:
            return r"$-\pi$" if num == 1 else rf"$-{num}\pi$"
        if num == 1:
            return rf"$-\pi/{den}$"
        return rf"$-{num}\pi/{den}$"
    return rf"${gamma:.3g}$"


def _abs_gamma_pi_tick_label(abs_gamma: float) -> str:
    if abs(abs_gamma) < 1e-12:
        return r"$0$"
    k = abs_gamma / math.pi
    for den in (1, 2, 4):
        num = round(k * den)
        if num == 0 or abs(k - num / den) >= 1e-6:
            continue
        if den == 1:
            return r"$\pi$" if num == 1 else rf"${num}\pi$"
        if num == 1:
            return rf"$\pi/{den}$"
        return rf"${num}\pi/{den}$"
    return rf"${abs_gamma:.3g}$"


def _style_abs_gamma_axis(ax) -> None:
    """X-axis: |γ| = 0 (left) → 2π (right); major ticks at π/4 multiples."""
    ax.set_xlim(0.0, GAMMA_ABS_AXIS_RIGHT)
    ticks = []
    g = 0.0
    while g <= GAMMA_ABS_AXIS_RIGHT + 1e-12:
        ticks.append(g)
        g += GAMMA_PI_TICK_STEP
    ax.set_xticks(ticks)
    ax.set_xticklabels([_abs_gamma_pi_tick_label(t) for t in ticks])
    ax.set_xlabel(r"$|\gamma|$")


def _style_gamma_axis_reading_zero_to_negative(ax) -> None:
    """X-axis: γ = 0 on the left, −2π on the right; major ticks at π/4 multiples."""
    ax.set_xlim(GAMMA_AXIS_LEFT, GAMMA_AXIS_RIGHT)
    ticks = []
    g = GAMMA_AXIS_LEFT
    while g >= GAMMA_AXIS_RIGHT - 1e-12:
        ticks.append(g)
        g -= GAMMA_PI_TICK_STEP
    ax.set_xticks(ticks)
    ax.set_xticklabels([_gamma_pi_tick_label(t) for t in ticks])
    ax.set_xlabel(r"$\gamma$")


def _plot_exponent_curves(
    ax,
    abs_gammas: list[float],
    seed_saddle_exp: list[float],
    exact_finite_n_exp: list[float],
    *,
    lw: float,
    ms: float,
    markevery: int,
    with_labels: bool = False,
    markers: bool = True,
) -> None:
    label_seed = LABEL_SEED_SADDLE_PAPER if with_labels else None
    label_exact = LABEL_EXACT_FINITE_N_PAPER if with_labels else None
    x = np.asarray(abs_gammas, dtype=float)
    seed = np.asarray(seed_saddle_exp, dtype=float)
    exact = np.asarray(exact_finite_n_exp, dtype=float)
    marker_kw = {"ms": ms, "mew": 0.5} if markers else {"marker": ""}
    me = markevery if markers else None
    ax.plot(
        x,
        exact,
        "s-" if markers else "-",
        **marker_kw,
        lw=lw,
        markevery=me,
        color="C1",
        zorder=2,
        label=label_exact,
    )
    ax.plot(
        x,
        seed,
        "o--" if markers else "--",
        **marker_kw,
        lw=lw,
        markevery=me,
        color="C0",
        zorder=3,
        label=label_seed,
    )


def _inset_ylim_for_separation(
    seed: np.ndarray,
    exact: np.ndarray,
) -> tuple[float, float]:
    """Tight y-limits (g0.75-style) so the band between curves reads clearly."""
    y_lo_d = float(min(seed.min(), exact.min()))
    y_hi_d = float(max(seed.max(), exact.max()))
    span = y_hi_d - y_lo_d
    max_gap = float(np.max(np.abs(seed - exact)))
    # Minimal margin: ~2% of span, but at least half the peak gap.
    pad = max(0.004, 0.02 * span, 0.55 * max_gap)
    return y_lo_d - pad, y_hi_d + pad


def _add_small_gamma_inset(
    ax,
    abs_gammas: list[float],
    seed_saddle_exp: list[float],
    exact_finite_n_exp: list[float],
    *,
    lw: float,
    ms: float,
) -> None:
    """Inset zoom like g0.75 crop: |γ|∈[0,0.75], matched line weights."""
    x_hi_lim = ZOOM_INSET_ABS_GAMMA_MAX
    mask = [g <= x_hi_lim + 1e-9 for g in abs_gammas]
    if sum(mask) < 3:
        return
    zoom_abs = [g for g, m in zip(abs_gammas, mask) if m]
    zoom_seed = np.asarray([v for v, m in zip(seed_saddle_exp, mask) if m], dtype=float)
    zoom_exact = np.asarray([v for v, m in zip(exact_finite_n_exp, mask) if m], dtype=float)

    x_lo = 0.0
    y_lo, y_hi = _inset_ylim_for_separation(zoom_seed, zoom_exact)

    axins = ax.inset_axes([0.14, 0.14, 0.34, 0.34])
    inset_markevery = max(1, len(zoom_abs) // 8)
    _plot_exponent_curves(
        axins,
        zoom_abs,
        zoom_seed.tolist(),
        zoom_exact.tolist(),
        lw=1.5,
        ms=2.5,
        markevery=inset_markevery,
        markers=True,
    )
    axins.set_xlim(x_lo, x_hi_lim)
    axins.set_ylim(y_lo, y_hi)
    axins.set_xticks([0.0, 0.25, 0.5, 0.75])
    axins.yaxis.tick_right()
    axins.yaxis.set_label_position("right")
    axins.tick_params(axis="y", left=False, labelleft=False, labelsize=8, pad=1)
    axins.tick_params(axis="x", labelsize=8, pad=1)
    axins.grid(True, alpha=0.2)
    ax.indicate_inset_zoom(
        axins,
        edgecolor="0.35",
        linewidth=1.0,
        alpha=0.95,
    )


def _save_gamma_vs_exponents_plot(
    gammas: list[float],
    seed_saddle_exp: list[float],
    exact_finite_n_exp: list[float],
    out_dir: Path,
    *,
    stem: str = "gamma_vs_exponents",
) -> None:
    paper_style = {
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "axes.unicode_minus": False,
        "font.size": 14,
        "axes.labelsize": 16,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 12,
    }
    lw = 2.5
    ms = 3.0
    abs_gammas = [abs(g) for g in gammas]
    markevery = max(1, len(abs_gammas) // 25) if len(abs_gammas) > 80 else 1
    show_inset = max(abs_gammas) > ZOOM_INSET_ABS_GAMMA_MAX + 0.05
    with plt.rc_context(paper_style):
        fig, ax = plt.subplots(figsize=(7.0, 4.5))
        _plot_exponent_curves(
            ax,
            abs_gammas,
            seed_saddle_exp,
            exact_finite_n_exp,
            lw=lw,
            ms=ms,
            markevery=markevery,
            with_labels=True,
        )
        ax.set_ylabel("Exponent")
        _style_abs_gamma_axis(ax)
        if show_inset:
            _add_small_gamma_inset(
                ax,
                abs_gammas,
                seed_saddle_exp,
                exact_finite_n_exp,
                lw=lw,
                ms=ms,
            )
        ax.legend(
            loc="lower center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=2,
            frameon=False,
        )
        ax.grid(True, alpha=0.2)
        fig.tight_layout(rect=[0, 0, 1, 0.98])
        for suffix, dpi in ((".png", 190), (".pdf", None)):
            fig.savefig(out_dir / f"{stem}{suffix}", dpi=dpi, bbox_inches="tight")
        plt.close(fig)


@dataclass
class ContinuationRow:
    gamma: float
    step_index: int
    certified: bool
    failed: bool
    failure_reason: str
    residual_inf: float
    krawczyk_contraction: float
    box_radius: float
    re_phi_m: float
    im_phi_m: float
    phi_pref_conv2: float
    full_conv2_exponent: float
    lambda_abs_n_max: float
    gap_to_exact: float
    z_distance_from_previous: float
    jacobian_cond: float
    iterate_used: bool
    iterate_converged: bool
    matches_exact_tol: bool
    z_real: list[float] = field(default_factory=list)
    z_imag: list[float] = field(default_factory=list)

    def to_row(self) -> dict:
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, (np.bool_, bool)):
                d[k] = bool(v)
            elif isinstance(v, (np.floating, np.integer)):
                d[k] = float(v)
        return d


def conv2_pref(K_clause: int, r: float) -> float:
    return float(bm24_prefactor_exponent_ksat_all_subsets(k=K_clause, r=r))


def conv2_full_exponent(re_phi_m: float, K_clause: int, r: float) -> float:
    return float(re_phi_m) + conv2_pref(K_clause, r)


def lambda_abs_n_max(
    q: int,
    K_clause: int,
    r: float,
    beta: float,
    gamma: float,
    n_values: list[int],
) -> float:
    fn = finite_n_exponent_grid(K_clause, q, r, beta, gamma, n_values)
    n_large = max(fn["n_values"])
    return float(fn["lambda_abs"][str(n_large)])


def polish_to_residual(
    sys: SaddleSystem,
    x0: np.ndarray,
    *,
    min_residual: float,
    dps: int,
) -> tuple[np.ndarray, float]:
    x, res = _scipy_polish_to_tol(sys, x0, target_inf=min_residual, max_tries=8, tol=1e-14)
    if res > min_residual:
        x, res = _mpmath_newton_polish(sys, x, target_inf=min_residual, dps=max(120, dps), max_iters=16)
    return x, float(res)


def certify_z(
    sys: SaddleSystem,
    z: np.ndarray,
    *,
    dps: int,
) -> tuple[bool, dict]:
    ok, info = _adaptive_krawczyk_with_diagnostics(sys, z, dps=dps)
    if not ok:
        ok2, info2 = krawczyk_certify_with_escalation(sys, z, dps_start=dps, max_inflate_iters=8)
        if ok2:
            return True, info2
    return bool(ok), info


def initialize_seed_at_gamma(
    q: int,
    K_clause: int,
    r: float,
    beta: float,
    gamma: float,
    *,
    n_values: list[int],
    match_tol: float,
    min_residual: float,
    dps: int,
    use_iterator: bool,
) -> tuple[np.ndarray, ContinuationRow, dict]:
    """Return certified z at gamma_start; validate iterator only when use_iterator=True."""
    sys = SaddleSystem.build(q=q, r=r, betas=np.array([beta]), gammas=np.array([gamma]))
    betas = sys.betas
    gammas = sys.gammas
    iterate_used = False
    iterate_converged = False

    if use_iterator:
        z_it, phi_it, conv = bm24_seed_z(q, r, beta, gamma)
        diag_it = diagnose_bm24_iterate(z_it, sys)
        status, fixed = classify_bm24_iterate(diag_it)
        if not conv:
            raise RuntimeError(f"BM24 iterator did not converge at gamma={gamma}")
        if not fixed:
            raise RuntimeError(
                f"BM24 iterator not a fixed point at gamma={gamma}: status={status}, "
                f"||G||_inf={diag_it['res_inf']}"
            )
        x0 = np.concatenate([z_it.real, z_it.imag])
        iterate_used = True
        iterate_converged = True
    else:
        raise ValueError("initialize_seed_at_gamma requires use_iterator=True for cold start")

    x_pol, res = polish_to_residual(sys, x0, min_residual=min_residual, dps=dps)
    z_pol = _x_to_z(x_pol, sys.nvars)
    ok, proof = certify_z(sys, z_pol, dps=dps)
    phi = compute_phi(z_pol, q=q, r=r, betas=betas, gammas=gammas)
    pref = conv2_pref(K_clause, r)
    full = conv2_full_exponent(phi.real, K_clause, r)
    lam = lambda_abs_n_max(q, K_clause, r, beta, gamma, n_values)
    gap = full - lam
    j = numerical_jacobian(sys.G_real, x_pol)
    jcond = float(np.linalg.cond(j)) if j.size else float("inf")
    radii = proof.get("box_radius", [])
    box_r = float(min(radii)) if radii else float("nan")
    match_tol_eff = max(match_tol, 5e-4 * abs(lam))
    row = ContinuationRow(
        gamma=float(gamma),
        step_index=0,
        certified=bool(ok),
        failed=not ok,
        failure_reason="" if ok else str(proof.get("reason", "krawczyk_failed")),
        residual_inf=res,
        krawczyk_contraction=float(proof.get("contraction_bound", math.inf)),
        box_radius=box_r,
        re_phi_m=float(phi.real),
        im_phi_m=float(phi.imag),
        phi_pref_conv2=pref,
        full_conv2_exponent=full,
        lambda_abs_n_max=lam,
        gap_to_exact=gap,
        z_distance_from_previous=0.0,
        jacobian_cond=jcond,
        iterate_used=iterate_used,
        iterate_converged=iterate_converged,
        matches_exact_tol=abs(gap) < match_tol_eff,
        z_real=z_pol.real.tolist(),
        z_imag=z_pol.imag.tolist(),
    )
    if not ok or res > min_residual:
        raise RuntimeError(f"Initialization failed certification/residual at gamma={gamma}")
    return z_pol, row, proof


def step_from_previous_z(
    q: int,
    K_clause: int,
    r: float,
    beta: float,
    gamma: float,
    z_prev: np.ndarray,
    *,
    step_index: int,
    n_values: list[int],
    match_tol: float,
    min_residual: float,
    dps: int,
) -> tuple[Optional[np.ndarray], ContinuationRow]:
    sys = SaddleSystem.build(q=q, r=r, betas=np.array([beta]), gammas=np.array([gamma]))
    x_prev = np.concatenate([z_prev.real, z_prev.imag])
    x_pol, res = polish_to_residual(sys, x_prev, min_residual=min_residual, dps=dps)
    z_pol = _x_to_z(x_pol, sys.nvars)
    z_dist = float(np.linalg.norm(z_pol - z_prev, ord=np.inf))
    ok, proof = certify_z(sys, z_pol, dps=dps)
    phi = compute_phi(z_pol, q=q, r=r, betas=sys.betas, gammas=sys.gammas)
    pref = conv2_pref(K_clause, r)
    full = conv2_full_exponent(phi.real, K_clause, r)
    lam = lambda_abs_n_max(q, K_clause, r, beta, gamma, n_values)
    gap = full - lam
    j = numerical_jacobian(sys.G_real, x_pol)
    jcond = float(np.linalg.cond(j)) if j.size else float("inf")
    radii = proof.get("box_radius", [])
    box_r = float(min(radii)) if radii else float("nan")
    match_tol_eff = max(match_tol, 5e-4 * abs(lam))
    failed = (not ok) or (res > min_residual) or (not np.isfinite(res))
    reason = ""
    if failed:
        if not ok:
            reason = str(proof.get("reason", "krawczyk_failed"))
        elif res > min_residual:
            reason = f"residual_above_min_{min_residual}"
        else:
            reason = "nonfinite"
    row = ContinuationRow(
        gamma=float(gamma),
        step_index=step_index,
        certified=bool(ok) and not failed,
        failed=failed,
        failure_reason=reason,
        residual_inf=res,
        krawczyk_contraction=float(proof.get("contraction_bound", math.inf)),
        box_radius=box_r,
        re_phi_m=float(phi.real),
        im_phi_m=float(phi.imag),
        phi_pref_conv2=pref,
        full_conv2_exponent=full,
        lambda_abs_n_max=lam,
        gap_to_exact=gap,
        z_distance_from_previous=z_dist,
        jacobian_cond=jcond,
        iterate_used=False,
        iterate_converged=False,
        matches_exact_tol=abs(gap) < match_tol_eff,
        z_real=z_pol.real.tolist(),
        z_imag=z_pol.imag.tolist(),
    )
    if failed:
        return None, row
    return z_pol, row


def discover_competitors_at_gamma(
    q: int,
    K_clause: int,
    r: float,
    beta: float,
    gamma: float,
    seed_z: np.ndarray,
    *,
    num_starts: int,
    seed: int,
    branch_tol: float,
    min_residual: float,
    dps: int,
) -> list[dict]:
    sys = SaddleSystem.build(q=q, r=r, betas=np.array([beta]), gammas=np.array([gamma]))
    pref = conv2_pref(K_clause, r)
    seed_full = conv2_full_exponent(
        float(np.real(compute_phi(seed_z, q=q, r=r, betas=sys.betas, gammas=sys.gammas))),
        K_clause,
        r,
    )
    roots_x = discover_roots(sys, num_starts=num_starts, seed=seed, tol=1e-12)
    competitors: list[dict] = []
    for x in roots_x:
        z = _x_to_z(x, sys.nvars)
        z_dist_seed = float(np.linalg.norm(z - seed_z, ord=np.inf))
        if z_dist_seed < branch_tol:
            continue
        x_pol, res = polish_to_residual(sys, x, min_residual=1e-8, dps=dps)
        z_pol = _x_to_z(x_pol, sys.nvars)
        ok, proof = certify_z(sys, z_pol, dps=dps)
        phi = compute_phi(z_pol, q=q, r=r, betas=sys.betas, gammas=sys.gammas)
        full = conv2_full_exponent(phi.real, K_clause, r)
        competitors.append(
            {
                "gamma": float(gamma),
                "certified": bool(ok),
                "residual_inf": float(res),
                "re_phi_m": float(phi.real),
                "im_phi_m": float(phi.imag),
                "full_conv2_exponent": full,
                "re_action_gap_vs_seed_branch": float(full - seed_full),
                "z_distance_from_seed_branch": z_dist_seed,
                "z_real": z_pol.real.tolist(),
                "z_imag": z_pol.imag.tolist(),
                "krawczyk_contraction": float(proof.get("contraction_bound", math.inf)),
                "label": "certified_competitor" if ok else "uncertified_competitor",
                "note": "largest_algebraic_re_phi is not physical dominance without PL data",
            }
        )
    return competitors


def selected_competitor_gammas(gamma_start: float, gamma_end: float, every: float) -> list[float]:
    """Sample gammas along the continuation direction (start -> end)."""
    g, g_end = float(gamma_start), float(gamma_end)
    step = -abs(every) if g > g_end else abs(every)
    out: list[float] = []
    while (step < 0 and g >= g_end - 1e-12) or (step > 0 and g <= g_end + 1e-12):
        out.append(round(g, 10))
        if abs(g - g_end) < 1e-9:
            break
        g += step
    if not out or abs(out[-1] - g_end) > 1e-9:
        out.append(g_end)
    return sorted(set(out))


def replot_gamma_vs_exponents_from_json(
    continuation_json: Path,
    out_dir: Optional[Path] = None,
    *,
    stem: str = "gamma_vs_exponents",
) -> Path:
    """Regenerate gamma_vs_exponents.{png,pdf} from an existing continuation JSON."""
    continuation_json = Path(continuation_json)
    data = json.loads(continuation_json.read_text(encoding="utf-8"))
    ok_rows = [
        r for r in data["continuation"]
        if r.get("certified") and not r.get("failed")
    ]
    if not ok_rows:
        raise ValueError(f"No certified rows in {continuation_json}")
    ok_rows.sort(key=lambda r: abs(float(r["gamma"])))
    gammas = [float(r["gamma"]) for r in ok_rows]
    full = [float(r["full_conv2_exponent"]) for r in ok_rows]
    lam = [float(r["lambda_abs_n_max"]) for r in ok_rows]
    target = Path(out_dir) if out_dir else continuation_json.parent
    target.mkdir(parents=True, exist_ok=True)
    _save_gamma_vs_exponents_plot(gammas, full, lam, target, stem=stem)
    return target


def plot_continuation(rows: list[ContinuationRow], competitors_by_gamma: dict, out_dir: Path) -> None:
    ok_rows = [r for r in rows if r.certified and not r.failed]
    if not ok_rows:
        return
    gammas = [r.gamma for r in ok_rows]
    full = [r.full_conv2_exponent for r in ok_rows]
    lam = [r.lambda_abs_n_max for r in ok_rows]
    gap = [r.gap_to_exact for r in ok_rows]
    res = [r.residual_inf for r in ok_rows]
    kappa = [r.krawczyk_contraction for r in ok_rows]
    nearest_gap = []
    for r in ok_rows:
        comps = competitors_by_gamma.get(str(r.gamma), [])
        cert_re = [
            c["re_action_gap_vs_seed_branch"]
            for c in comps
            if c.get("certified") and c.get("label") == "certified_competitor"
        ]
        nearest_gap.append(min((abs(x) for x in cert_re), default=float("nan")))

    _save_gamma_vs_exponents_plot(gammas, full, lam, out_dir)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(gammas, gap, "o-", color="C2")
    ax.axhline(0.0, color="k", lw=0.8, alpha=0.5)
    ax.set_xlabel(r"$\gamma$")
    ax.set_ylabel("gap_to_exact")
    ax.set_title("Seed branch Conv2 exponent minus exact λ_abs(n_max)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "gamma_vs_gap_to_exact.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogy(gammas, res, "o-", label=r"$||G||_\infty$")
    ax.semilogy(gammas, kappa, "s-", label="Krawczyk contraction")
    ax.set_xlabel(r"$\gamma$")
    ax.set_ylabel("value")
    ax.set_title("Residual and Krawczyk contraction along branch")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "gamma_vs_residual_krawczyk.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(gammas, nearest_gap, "o-", color="C4")
    ax.set_xlabel(r"$\gamma$")
    ax.set_ylabel("|Δ(Re action)| vs nearest certified competitor")
    ax.set_title("Nearest certified competitor Re-action gap (not PL dominance)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "gamma_vs_nearest_competitor_re_gap.png", dpi=150)
    plt.close(fig)


def run_continuation(
    *,
    q: int = 3,
    K_clause: int = 8,
    r: float = 176.54,
    beta: float = 0.5433996420760803,
    gamma_start: float = -0.001,
    gamma_end: float = -0.75,
    gamma_step: float = 0.01,
    min_step: float = 0.0005,
    n_min: int = 12,
    n_max: int = 22,
    match_tol: float = 0.02,
    min_residual: float = 1e-10,
    dps: int = 80,
    competitor_every: float = 0.05,
    competitor_starts: int = 400,
    branch_tol: float = 1e-4,
    seed: int = 0,
    out_dir: Optional[Path] = None,
) -> Path:
    ts = datetime.now(timezone.utc).strftime("run_%m-%d_%H-%M-%SZ")
    run_dir = (out_dir or (AUDIT_DIR / "results" / f"{ts}_seed_branch")) 
    run_dir.mkdir(parents=True, exist_ok=True)

    n_values = list(range(n_min, n_max + 1))
    rows: list[ContinuationRow] = []
    failed_rows: list[ContinuationRow] = []

    z_seed, row0, _ = initialize_seed_at_gamma(
        q,
        K_clause,
        r,
        beta,
        gamma_start,
        n_values=n_values,
        match_tol=match_tol,
        min_residual=min_residual,
        dps=dps,
        use_iterator=True,
    )
    rows.append(row0)
    z_current = z_seed
    step_index = 1

    gamma = float(gamma_start)
    step = float(gamma_step)
    competitor_gammas = set(selected_competitor_gammas(gamma_start, gamma_end, competitor_every))
    competitors_by_gamma: dict[str, list[dict]] = {}

    meta = {
        "q": q,
        "K_clause": K_clause,
        "r": r,
        "beta": beta,
        "gamma_start": gamma_start,
        "gamma_end": gamma_end,
        "gamma_step_initial": gamma_step,
        "min_step": min_step,
        "convention": "full_exponent = Re Phi_M + bm24_prefactor_exponent_ksat_all_subsets",
        "im_phi_warning": IM_PHI_WARNING,
        "branch_continuity_metric": "||z(gamma)-z(previous)||_inf",
        "dominance_note": "Do not interpret largest Re Phi as physical dominance without PL/intersection data.",
        "timestamp": ts,
    }

    while gamma > float(gamma_end) + 1e-12:
        target_gamma = gamma - step
        if target_gamma < float(gamma_end):
            target_gamma = float(gamma_end)
        attempt_step = step
        success = False
        last_fail_row: Optional[ContinuationRow] = None

        while attempt_step >= float(min_step) - 1e-15:
            trial_gamma = gamma - attempt_step
            if trial_gamma < float(gamma_end):
                trial_gamma = float(gamma_end)
            z_next, row = step_from_previous_z(
                q,
                K_clause,
                r,
                beta,
                trial_gamma,
                z_current,
                step_index=step_index,
                n_values=n_values,
                match_tol=match_tol,
                min_residual=min_residual,
                dps=dps,
            )
            if z_next is not None:
                rows.append(row)
                z_current = z_next
                gamma = trial_gamma
                step_index += 1
                success = True
                step = min(float(gamma_step), attempt_step * 1.25)
                break
            last_fail_row = row
            attempt_step *= 0.5

        if not success:
            if last_fail_row is not None:
                last_fail_row.failure_reason = (
                    last_fail_row.failure_reason or "continuation_failed"
                ) + f"; step_size_tried={attempt_step}"
                failed_rows.append(last_fail_row)
            break

        if abs(gamma - float(gamma_end)) < 1e-12:
            break

    z_by_gamma = {r.gamma: np.array(r.z_real) + 1j * np.array(r.z_imag) for r in rows if r.certified}
    for g in sorted(competitor_gammas):
        if g not in z_by_gamma:
            cert_gammas = sorted(z_by_gamma.keys())
            if not cert_gammas:
                continue
            ref_g = min(cert_gammas, key=lambda x: abs(x - g))
            seed_z = z_by_gamma[ref_g]
        else:
            seed_z = z_by_gamma[g]
        comps = discover_competitors_at_gamma(
            q,
            K_clause,
            r,
            beta,
            g,
            seed_z,
            num_starts=competitor_starts,
            seed=seed + int(1000 * abs(g)),
            branch_tol=branch_tol,
            min_residual=min_residual,
            dps=dps,
        )
        largest = max((c["re_phi_m"] for c in comps if c.get("certified")), default=float("nan"))
        for c in comps:
            if c.get("certified") and abs(c["re_phi_m"] - largest) < 1e-12:
                c["largest_algebraic_re_phi_at_gamma"] = True
            else:
                c["largest_algebraic_re_phi_at_gamma"] = False
        competitors_by_gamma[str(g)] = comps

    all_rows = rows + failed_rows
    row_dicts = [r.to_row() for r in all_rows]
    with open(run_dir / "seed_branch_continuation.json", "w", encoding="utf-8") as fh:
        json.dump(
            {
                "metadata": meta,
                "continuation": row_dicts,
                "summary": _build_summary(rows, failed_rows, gamma_end),
            },
            fh,
            indent=2,
        )
    if row_dicts:
        with open(run_dir / "seed_branch_continuation.csv", "w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(row_dicts[0].keys()))
            writer.writeheader()
            writer.writerows(row_dicts)

    with open(run_dir / "competitor_saddles_by_gamma.json", "w", encoding="utf-8") as fh:
        json.dump(
            {
                "metadata": meta,
                "im_phi_warning": IM_PHI_WARNING,
                "by_gamma": competitors_by_gamma,
            },
            fh,
            indent=2,
        )

    plot_continuation(rows, competitors_by_gamma, run_dir)
    print(json.dumps(_build_summary(rows, failed_rows, gamma_end), indent=2))
    print(f"Results written to {run_dir}")
    return run_dir


def _build_summary(rows: list[ContinuationRow], failed_rows: list[ContinuationRow], gamma_end: float) -> dict:
    cert = [r for r in rows if r.certified and not r.failed]
    if not cert:
        return {"certified_steps": 0, "gamma_reached": None, "message": "no certified continuation"}
    g_last = cert[-1].gamma
    gaps = [abs(r.gap_to_exact) for r in cert]
    return {
        "certified_steps": len(cert),
        "failed_steps": len(failed_rows),
        "gamma_start": cert[0].gamma,
        "gamma_last_certified": g_last,
        "reached_gamma_end": bool(g_last <= float(gamma_end) + 1e-9),
        "max_abs_gap_to_exact": max(gaps),
        "mean_abs_gap_to_exact": float(np.mean(gaps)),
        "all_match_exact_within_tol": all(r.matches_exact_tol for r in cert),
        "max_z_step_inf": max((r.z_distance_from_previous for r in cert[1:]), default=0.0),
        "statement": (
            "Numerically rigorous certified continuation of BM24 seed branch in gamma "
            "(Krawczyk), compared to Conv2 exact finite-n; not a claim of contour dominance."
        ),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Certified BM24 seed-branch gamma continuation")
    p.add_argument("--q", type=int, default=3)
    p.add_argument("--K-clause", type=int, default=8)
    p.add_argument("--r", type=float, default=176.54)
    p.add_argument("--beta", type=float, default=0.5433996420760803)
    p.add_argument("--gamma-start", type=float, default=-0.001)
    p.add_argument("--gamma-end", type=float, default=-0.75)
    p.add_argument("--gamma-step", type=float, default=0.01)
    p.add_argument("--min-step", type=float, default=0.0005)
    p.add_argument("--n-min", type=int, default=12)
    p.add_argument("--n-max", type=int, default=22)
    p.add_argument("--match-tol", type=float, default=0.02)
    p.add_argument("--min-residual", type=float, default=1e-10)
    p.add_argument("--dps", type=int, default=80)
    p.add_argument("--competitor-every", type=float, default=0.05)
    p.add_argument("--competitor-starts", type=int, default=400)
    p.add_argument("--branch-tol", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=str, default="")
    p.add_argument("--quick", action="store_true", help="Coarser step/end for smoke test")
    p.add_argument(
        "--plot-only",
        type=str,
        metavar="CONTINUATION_JSON",
        help="Regenerate gamma_vs_exponents.{png,pdf} from existing seed_branch_continuation.json",
    )
    p.add_argument(
        "--plot-stem",
        type=str,
        default="gamma_vs_exponents",
        help="Output filename stem when using --plot-only (default: gamma_vs_exponents)",
    )
    args = p.parse_args()

    if args.plot_only:
        replot_gamma_vs_exponents_from_json(
            Path(args.plot_only),
            Path(args.out_dir) if args.out_dir else None,
            stem=args.plot_stem,
        )
        return

    kwargs = dict(
        q=args.q,
        K_clause=args.K_clause,
        r=args.r,
        beta=args.beta,
        gamma_start=args.gamma_start,
        gamma_end=args.gamma_end,
        gamma_step=args.gamma_step,
        min_step=args.min_step,
        n_min=args.n_min,
        n_max=args.n_max,
        match_tol=args.match_tol,
        min_residual=args.min_residual,
        dps=args.dps,
        competitor_every=args.competitor_every,
        competitor_starts=args.competitor_starts,
        branch_tol=args.branch_tol,
        seed=args.seed,
        out_dir=Path(args.out_dir) if args.out_dir else None,
    )
    if args.quick:
        kwargs.update(gamma_end=-0.05, gamma_step=0.01, competitor_every=0.02, competitor_starts=150)
    run_continuation(**kwargs)


if __name__ == "__main__":
    main()
