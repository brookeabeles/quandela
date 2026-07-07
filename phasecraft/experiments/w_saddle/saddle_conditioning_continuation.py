#!/usr/bin/env python3
"""Jacobian conditioning + refined gamma / pseudo-arclength continuation: w,u vs z.

Every plotted point must satisfy |F| < ROOT_TOL after Newton polish.
Unrefined points are marked rejected and excluded from crossing analysis.

z residual: |G(z)| (BM24 iterator / Krawczyk equation).
wu residual: |F_tilde(w)| (reduced saddle system).

Jacobian condition numbers are chart-dependent — compare only within chart.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import root

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from phasecraft.bm24_saddle_audit_p1.audit import _x_to_z
from phasecraft.bm24_saddle_audit_p1.continue_seed_branch_certified import polish_to_residual
from phasecraft.bm24_saddle_audit_p1.z_robust_workflow import interpolate_z_from_payload
from phasecraft.lib.saddles.krawczyk_p1_roots import SaddleSystem, numerical_jacobian
from phasecraft.lib.saddles.picard_lefschetz import compute_phi
from phasecraft.lib.saddles.saddle_traits import complex_to_real_vector
from phasecraft.w_saddle.krawczyk_w_saddle_q import WSaddleSystem
from phasecraft.w_saddle.workflow import interpolate_w_seed_from_payload

DEFAULT_RUN = REPO_ROOT / "experiments/w_saddle/runs/compare_q3_matched"
R = 176.54
BETA = 0.5433996420760803
Q = 3
ROOT_TOL = 1e-12
MATCH_TOL = 0.02

# Cliff windows for pseudo-arclength probes
WU_ARCLEN_WINDOW = (-0.90, -0.70)
Z_ARCLEN_WINDOW = (-0.15, -0.05)


@dataclass
class RootDiag:
    chart: str
    gamma: float
    label: str
    is_seed: bool
    F_norm: float
    sigma_min: float
    kappa: float
    re_phi: float
    im_phi: float
    vec_real: list[float]
    vec_imag: list[float]
    refined: bool
    newton_iters: Optional[int] = None
    newton_converged: Optional[bool] = None
    matched_label: Optional[str] = None
    delta_x: float = 0.0


@dataclass
class BranchTrack:
    chart: str
    label: str
    is_seed: bool
    gammas: list[float] = field(default_factory=list)
    re_phi: list[float] = field(default_factory=list)
    im_phi: list[float] = field(default_factory=list)
    log10_sigma_min: list[float] = field(default_factory=list)
    F_norm: list[float] = field(default_factory=list)
    kappa: list[float] = field(default_factory=list)
    newton_iters: list[int] = field(default_factory=list)
    delta_x: list[float] = field(default_factory=list)
    refined: list[bool] = field(default_factory=list)
    vec_real: list[list[float]] = field(default_factory=list)
    vec_imag: list[list[float]] = field(default_factory=list)

    def valid_mask(self) -> np.ndarray:
        return np.array(self.refined, dtype=bool)

    def x_real_at(self, i: int) -> np.ndarray:
        return np.concatenate(
            [np.asarray(self.vec_real[i], float), np.asarray(self.vec_imag[i], float)]
        )


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def vec_from_record(rec: dict[str, Any]) -> np.ndarray:
    if "z_real" in rec:
        return np.asarray(rec["z_real"], float) + 1j * np.asarray(rec["z_imag"], float)
    if "w_real" in rec:
        return np.asarray(rec["w_real"], float) + 1j * np.asarray(rec["w_imag"], float)
    return np.asarray(rec["w_star_real"], float) + 1j * np.asarray(rec["w_star_imag"], float)


def vec_to_x(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=complex)
    return np.concatenate([v.real, v.imag])


def x_to_wu(x: np.ndarray) -> np.ndarray:
    return x[:4] + 1j * x[4:]


def nearest_point(payload: dict[str, Any], gamma: float) -> dict[str, Any]:
    pts = payload.get("points") or []
    return min(pts, key=lambda p: abs(float(p["gamma"]) - gamma))


def certified_roots_at_gamma(payload: dict[str, Any], gamma: float) -> list[dict[str, Any]]:
    pt = nearest_point(payload, gamma)
    sweep = pt.get("competitor_sweep") or {}
    return list(sweep.get("certified_roots") or [])


def wu_f_real(sys: WSaddleSystem):
    return sys.F_real


def z_f_real(sys: SaddleSystem):
    def f(x: np.ndarray) -> np.ndarray:
        z = _x_to_z(x, sys.nvars)
        return complex_to_real_vector(sys.G_complex(z))

    return f


def jacobian_stats(f_real: Callable[[np.ndarray], np.ndarray], x: np.ndarray) -> tuple[float, float, float]:
    J = numerical_jacobian(f_real, x)
    if J.size == 0:
        return float("nan"), float("nan"), float("nan")
    svals = np.linalg.svd(J, compute_uv=False)
    smin = float(svals[-1]) if svals.size else float("nan")
    smax = float(svals[0]) if svals.size else float("nan")
    kappa = float(smax / smin) if smin > 0 else float("inf")
    F_norm = float(np.linalg.norm(f_real(x), ord=np.inf))
    return F_norm, smin, kappa


def newton_real_iterations(
    f_real: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    *,
    tol: float = ROOT_TOL,
    max_iters: int = 80,
) -> tuple[np.ndarray, bool, float, int]:
    x = np.array(x0, dtype=float)
    res = float("inf")
    for it in range(max_iters):
        fx = f_real(x)
        res = float(np.linalg.norm(fx, ord=np.inf))
        if res < tol:
            return x, True, res, it
        J = numerical_jacobian(f_real, x)
        try:
            dx = np.linalg.lstsq(J, -fx, rcond=None)[0]
        except np.linalg.LinAlgError:
            break
        if not np.isfinite(dx).all():
            break
        alpha = 1.0
        improved = False
        for _ in range(10):
            x_try = x + alpha * dx
            res_try = float(np.linalg.norm(f_real(x_try), ord=np.inf))
            if res_try < res:
                x = x_try
                res = res_try
                improved = True
                break
            alpha *= 0.5
        if not improved:
            break
    return x, bool(res < tol), res, max_iters


def refine_wu(sys: WSaddleSystem, w_init: np.ndarray) -> tuple[np.ndarray, bool, float, int, float]:
    """Polish w until |F_tilde| < ROOT_TOL."""
    x0 = vec_to_x(w_init)
    best_x, best_res, best_iters = x0, float("inf"), -1
    for method in ("hybr", "lm"):
        try:
            sol = root(sys.F_real, x0, method=method, tol=1e-14)
            if sol.success and np.isfinite(sol.x).all():
                res = float(np.linalg.norm(sys.F_real(sol.x), ord=np.inf))
                if res < best_res:
                    best_x, best_res = sol.x, res
                    best_iters = int(getattr(sol, "nfev", -1))
        except Exception:
            pass
    x_n, ok_n, res_n, it_n = newton_real_iterations(wu_f_real(sys), x0, tol=ROOT_TOL)
    if res_n < best_res:
        best_x, best_res, best_iters = x_n, res_n, it_n
    delta_x = float(np.linalg.norm(best_x - x0))
    w = x_to_wu(best_x)
    ok = best_res < ROOT_TOL
    return w, ok, best_res, best_iters, delta_x


def refine_z(sys: SaddleSystem, z_init: np.ndarray) -> tuple[np.ndarray, bool, float, int, float]:
    x0 = vec_to_x(z_init)
    x_pol, res = polish_to_residual(sys, x0, min_residual=ROOT_TOL, dps=80)
    if res >= ROOT_TOL:
        x_n, ok_n, res_n, it_n = newton_real_iterations(z_f_real(sys), x0, tol=ROOT_TOL)
        if res_n < res:
            x_pol, res = x_n, res_n
    delta_x = float(np.linalg.norm(x_pol - x0))
    z = _x_to_z(x_pol, sys.nvars)
    ok = res < ROOT_TOL
    return z, ok, res, -1, delta_x


def phi_at(chart: str, vec: np.ndarray, gamma: float) -> complex:
    if chart == "wu":
        sys = WSaddleSystem(r=R, gamma=gamma, q=Q, beta=BETA)
        return sys.Phi_eff(vec)
    sys = SaddleSystem.build(q=Q, r=R, betas=np.array([BETA]), gammas=np.array([gamma]))
    return compute_phi(vec, q=Q, r=R, betas=sys.betas, gammas=sys.gammas)


def diagnostics_at(
    chart: str,
    vec: np.ndarray,
    gamma: float,
    *,
    label: str,
    is_seed: bool,
    refined: bool,
    F_norm: float,
    delta_x: float = 0.0,
    newton_iters: Optional[int] = None,
) -> RootDiag:
    if chart == "wu":
        sys = WSaddleSystem(r=R, gamma=gamma, q=Q, beta=BETA)
        x = vec_to_x(vec)
        _, smin, kappa = jacobian_stats(wu_f_real(sys), x)
        phi = sys.Phi_eff(vec)
    else:
        sys = SaddleSystem.build(q=Q, r=R, betas=np.array([BETA]), gammas=np.array([gamma]))
        x = vec_to_x(vec)
        _, smin, kappa = jacobian_stats(z_f_real(sys), x)
        phi = compute_phi(vec, q=Q, r=R, betas=sys.betas, gammas=sys.gammas)
    return RootDiag(
        chart=chart,
        gamma=gamma,
        label=label,
        is_seed=is_seed,
        F_norm=F_norm,
        sigma_min=smin,
        kappa=kappa,
        re_phi=float(phi.real),
        im_phi=float(phi.imag),
        vec_real=vec.real.tolist(),
        vec_imag=vec.imag.tolist(),
        refined=refined,
        newton_iters=newton_iters,
        newton_converged=refined,
        delta_x=delta_x,
    )


def match_vec(
    v: np.ndarray,
    catalog: list[tuple[str, np.ndarray, bool]],
    *,
    tol: float = MATCH_TOL,
) -> Optional[str]:
    scale = max(float(np.max(np.abs(v))), 1e-3)
    for label, u, _ in catalog:
        if float(np.linalg.norm(v - u, ord=np.inf)) <= tol * scale:
            return label
    return None


def catalog_from_sweep(payload: dict[str, Any], gamma: float) -> list[tuple[str, np.ndarray, bool]]:
    out: list[tuple[str, np.ndarray, bool]] = []
    for j, r in enumerate(certified_roots_at_gamma(payload, gamma)):
        is_seed = bool(r.get("is_seed_branch"))
        out.append((f"cert_{j}{'_seed' if is_seed else ''}", vec_from_record(r), is_seed))
    return out


def diagnose_all_roots_refined(
    payload: dict[str, Any],
    gamma: float,
    chart: str,
) -> list[RootDiag]:
    diags: list[RootDiag] = []
    for j, r in enumerate(certified_roots_at_gamma(payload, gamma)):
        v0 = vec_from_record(r)
        is_seed = bool(r.get("is_seed_branch"))
        label = f"cert_{j}{'_seed' if is_seed else ''}"
        if chart == "wu":
            sys = WSaddleSystem(r=R, gamma=gamma, q=Q, beta=BETA)
            v, ok, res, iters, dx = refine_wu(sys, v0)
        else:
            sys = SaddleSystem.build(q=Q, r=R, betas=np.array([BETA]), gammas=np.array([gamma]))
            v, ok, res, iters, dx = refine_z(sys, v0)
        diags.append(
            diagnostics_at(
                chart, v, gamma, label=label, is_seed=is_seed,
                refined=ok, F_norm=res, delta_x=dx, newton_iters=iters,
            )
        )
    return diags


def newton_from_seed(
    gamma: float,
    seed_vec: np.ndarray,
    chart: str,
    catalog: list[tuple[str, np.ndarray, bool]],
) -> RootDiag:
    if chart == "wu":
        sys = WSaddleSystem(r=R, gamma=gamma, q=Q, beta=BETA)
        v, ok, res, iters, dx = refine_wu(sys, seed_vec)
    else:
        sys = SaddleSystem.build(q=Q, r=R, betas=np.array([BETA]), gammas=np.array([gamma]))
        v, ok, res, iters, dx = refine_z(sys, seed_vec)
    diag = diagnostics_at(
        chart, v, gamma, label="newton_from_seed", is_seed=False,
        refined=ok, F_norm=res, delta_x=dx, newton_iters=iters,
    )
    diag.matched_label = match_vec(v, catalog)
    return diag


def pick_competitor_starts(
    catalog: list[tuple[str, np.ndarray, bool]],
    seed_vec: np.ndarray,
    gamma: float,
    chart: str,
    *,
    k: int = 3,
) -> list[tuple[str, np.ndarray]]:
    comps = [(lab, v) for lab, v, is_seed in catalog if not is_seed]
    if not comps:
        return []
    seed_re = float(phi_at(chart, seed_vec, gamma).real)
    comps.sort(key=lambda t: abs(float(phi_at(chart, t[1], gamma).real) - seed_re))
    return comps[:k]


def continue_branch(
    chart: str,
    label: str,
    start_vec: np.ndarray,
    gammas: np.ndarray,
    *,
    is_seed: bool,
) -> BranchTrack:
    track = BranchTrack(chart=chart, label=label, is_seed=is_seed)
    cur = np.asarray(start_vec, dtype=complex).copy()
    for g in gammas:
        g = float(g)
        x_prev = vec_to_x(cur)
        if chart == "wu":
            sys = WSaddleSystem(r=R, gamma=g, q=Q, beta=BETA)
            cur, ok, res, iters, dx = refine_wu(sys, cur)
            x = vec_to_x(cur)
            _, smin, kappa = jacobian_stats(wu_f_real(sys), x)
            phi = sys.Phi_eff(cur)
        else:
            sys = SaddleSystem.build(q=Q, r=R, betas=np.array([BETA]), gammas=np.array([g]))
            cur, ok, res, iters, dx = refine_z(sys, cur)
            x = vec_to_x(cur)
            _, smin, kappa = jacobian_stats(z_f_real(sys), x)
            phi = compute_phi(cur, q=Q, r=R, betas=sys.betas, gammas=sys.gammas)
        track.gammas.append(g)
        track.re_phi.append(float(phi.real))
        track.im_phi.append(float(phi.imag))
        track.log10_sigma_min.append(float(np.log10(max(smin, 1e-30))))
        track.F_norm.append(res)
        track.kappa.append(kappa)
        track.newton_iters.append(int(iters))
        track.delta_x.append(dx if dx > 0 else float(np.linalg.norm(x - x_prev)))
        track.refined.append(bool(ok))
        track.vec_real.append(cur.real.tolist())
        track.vec_imag.append(cur.imag.tolist())
        if not ok:
            # stop continuation on failed refine — later points unreliable
            break
    return track


def find_re_phi_crossings(tracks: list[BranchTrack], *, valid_only: bool = True) -> list[dict[str, Any]]:
    crossings: list[dict[str, Any]] = []
    for i in range(len(tracks)):
        for j in range(i + 1, len(tracks)):
            a, b = tracks[i], tracks[j]
            n = min(len(a.gammas), len(b.gammas))
            for k in range(n - 1):
                if valid_only and (not a.refined[k] or not a.refined[k + 1] or not b.refined[k] or not b.refined[k + 1]):
                    continue
                if a.gammas[k] != b.gammas[k] or a.gammas[k + 1] != b.gammas[k + 1]:
                    continue
                d0 = a.re_phi[k] - b.re_phi[k]
                d1 = a.re_phi[k + 1] - b.re_phi[k + 1]
                if d0 * d1 < 0 and abs(d0 - d1) > 1e-14:
                    t = d0 / (d0 - d1)
                    g_cross = a.gammas[k] + t * (a.gammas[k + 1] - a.gammas[k])
                    crossings.append(
                        {
                            "chart": a.chart,
                            "branch_a": a.label,
                            "branch_b": b.label,
                            "gamma_lo": a.gammas[k + 1],
                            "gamma_hi": a.gammas[k],
                            "gamma_cross_est": g_cross,
                            "delta_re_at_lo": d1,
                            "both_refined": True,
                        }
                    )
    return crossings


def root_separation_series(seed: BranchTrack, comp: BranchTrack) -> list[dict[str, float]]:
    out: list[dict[str, float]] = []
    n = min(len(seed.gammas), len(comp.gammas))
    for i in range(n):
        if not (seed.refined[i] and comp.refined[i]):
            continue
        if seed.gammas[i] != comp.gammas[i]:
            continue
        xs = seed.x_real_at(i)
        xc = comp.x_real_at(i)
        out.append(
            {
                "gamma": seed.gammas[i],
                "separation_inf": float(np.linalg.norm(xs - xc, ord=np.inf)),
                "separation_l2": float(np.linalg.norm(xs - xc)),
            }
        )
    return out


def _residual_x_only(chart: str, gamma: float, x: np.ndarray) -> np.ndarray:
    if chart == "wu":
        sys = WSaddleSystem(r=R, gamma=gamma, q=Q, beta=BETA)
        return wu_f_real(sys)(x)
    sys = SaddleSystem.build(q=Q, r=R, betas=np.array([BETA]), gammas=np.array([gamma]))
    return z_f_real(sys)(x)


def _residual_y(chart: str, y: np.ndarray) -> np.ndarray:
    return _residual_x_only(chart, float(y[0]), y[1:])


def pseudo_arclength_continue_fixed(
    chart: str,
    start_gamma: float,
    start_vec: np.ndarray,
    *,
    gamma_end: float,
    ds: float = 0.015,
    max_steps: int = 40,
) -> list[dict[str, Any]]:
    n_x = len(vec_to_x(start_vec))
    y = np.concatenate([[start_gamma], vec_to_x(start_vec)])
    direction = -1.0 if gamma_end < start_gamma else 1.0
    tangent = np.zeros(1 + n_x)
    tangent[0] = direction
    tangent /= np.linalg.norm(tangent)
    points: list[dict[str, Any]] = []

    for step in range(max_steps):
        gamma = float(y[0])
        x = y[1:]
        F_norm = float(np.linalg.norm(_residual_x_only(chart, gamma, x), ord=np.inf))
        _, smin, kappa = jacobian_stats(
            lambda xx: _residual_x_only(chart, gamma, xx), x
        )
        nvar = n_x // 2
        vec = x_to_wu(x) if chart == "wu" else _x_to_z(x, nvar)
        phi = phi_at(chart, vec, gamma)
        points.append(
            {
                "step": step,
                "gamma": gamma,
                "F_norm": F_norm,
                "sigma_min": smin,
                "kappa": kappa,
                "log10_sigma_min": float(np.log10(max(smin, 1e-30))),
                "re_phi": float(phi.real),
                "refined": F_norm < ROOT_TOL,
            }
        )
        if direction < 0 and gamma <= gamma_end + 1e-10:
            break
        if direction > 0 and gamma >= gamma_end - 1e-10:
            break

        y_pred = y + ds * tangent

        def aug(y_vec: np.ndarray, _yp=y_pred.copy(), _t=tangent.copy()) -> np.ndarray:
            return np.concatenate(
                [_residual_y(chart, y_vec), [float(np.dot(_t, y_vec - _yp))]]
            )

        y_new, ok, _, _ = newton_real_iterations(aug, y_pred, tol=1e-11, max_iters=40)
        if not ok:
            break
        dy = y_new - y
        y = y_new
        if np.linalg.norm(dy) > 1e-14:
            tangent = dy / np.linalg.norm(dy)

    return points


def plot_outputs(
    tracks_wu: list[BranchTrack],
    tracks_z: list[BranchTrack],
    crossings_wu: list[dict[str, Any]],
    crossings_z: list[dict[str, Any]],
    sep_wu: list[dict[str, float]],
    sep_z: list[dict[str, float]],
    arclen_wu: list[dict[str, Any]],
    arclen_z: list[dict[str, Any]],
    out_path: Path,
) -> None:
    fig = plt.figure(figsize=(16, 13))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 0.85, 0.85], hspace=0.35, wspace=0.28)
    fig.suptitle(
        rf"Refined continuation ($|F|<{ROOT_TOL:.0e}$ required; unrefined points excluded)"
        "\nJacobian κ is chart-dependent. Re Φ crossings are candidates only.",
        fontsize=11,
        y=0.98,
    )
    palette = plt.cm.tab10(np.linspace(0, 1, 10))

    def plot_chart(col: int, tracks: list[BranchTrack], title: str, crossings: list[dict], sep: list[dict], arclen: list[dict]):
        ax_traj = fig.add_subplot(gs[0, col])
        ax_re = fig.add_subplot(gs[1, col])
        ax_sig = fig.add_subplot(gs[2, col])
        for i, tr in enumerate(tracks):
            c = palette[i % 10]
            ls = "-" if tr.is_seed else "--"
            lw = 2.4 if tr.is_seed else 1.5
            mask = tr.valid_mask()
            g = np.array(tr.gammas)[mask]
            if g.size == 0:
                continue
            re = np.array(tr.re_phi)[mask]
            sm = np.array(tr.log10_sigma_min)[mask]
            ax_re.plot(g, re, ls, color=c, lw=lw, label=tr.label)
            ax_sig.plot(g, sm, ls, color=c, lw=lw)
            ax_traj.plot(re, sm, "-", color=c, lw=lw, label=tr.label)
            bad = np.where(~mask)[0]
            if bad.size:
                gb = np.array(tr.gammas)[bad]
                ax_re.plot(gb, np.array(tr.re_phi)[bad], "x", color=c, ms=5, alpha=0.5)
        for cr in crossings:
            ax_re.axvline(cr["gamma_cross_est"], color="0.45", ls=":", lw=1.0)
        if arclen:
            ag = [p["gamma"] for p in arclen if p["refined"]]
            asm = [p["log10_sigma_min"] for p in arclen if p["refined"]]
            if ag:
                ax_sig.plot(ag, asm, "o-", color="k", ms=3, lw=1.2, alpha=0.55, label="pseudo-arclength")
        ax_traj.set_title(title + r": $(\mathrm{Re}\,\Phi,\log_{10}\sigma_{\min})$", fontsize=10)
        ax_traj.legend(fontsize=6)
        ax_traj.grid(True, alpha=0.25)
        ax_re.set_ylabel(r"Re $\Phi$")
        ax_re.legend(fontsize=6)
        ax_re.grid(True, alpha=0.25)
        ax_sig.set_xlabel(r"$\gamma$")
        ax_sig.set_ylabel(r"$\log_{10}\sigma_{\min}$")
        ax_sig.legend(fontsize=6)
        ax_sig.grid(True, alpha=0.25)

    plot_chart(0, tracks_wu, "w,u", crossings_wu, sep_wu, arclen_wu)
    plot_chart(1, tracks_z, "z", crossings_z, sep_z, arclen_z)

    # root separation inset figure
    sep_path = out_path.with_name(out_path.stem + "_separation.png")
    fig2, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for ax, sep, title in zip(axes, [sep_wu, sep_z], ["w,u", "z"]):
        if sep:
            g = [s["gamma"] for s in sep]
            ax.semilogy(g, [s["separation_l2"] for s in sep], "o-", lw=1.5)
        ax.set_xlabel(r"$\gamma$")
        ax.set_ylabel(r"$\|x_{\rm seed}-x_{\rm comp}\|_2$")
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
    fig2.suptitle("Root separation (refined points only)")
    fig2.savefig(sep_path, dpi=160)
    plt.close(fig2)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def run_pipeline(
    run_dir: Path,
    *,
    gamma_start: float = -0.01,
    gamma_stop: float = -1.0,
    num_points: int = 50,
    n_competitors: int = 3,
    out_dir: Optional[Path] = None,
    root_tol: float = ROOT_TOL,
) -> dict[str, Path]:
    out_dir = out_dir or run_dir / "conditioning_continuation"
    out_dir.mkdir(parents=True, exist_ok=True)

    gammas = np.linspace(gamma_start, gamma_stop, num_points)
    g0 = float(gamma_start)

    wu_seed_payload = load_json(run_dir / "wu/seed_branch_wu.json")
    z_seed_payload = load_json(run_dir / "z/seed_branch_z.json")
    wu_sweep = load_json(run_dir / "wu/competitors_robust_wu.json")
    z_sweep = load_json(run_dir / "z/competitors_robust_z.json")

    w_seed0, ok_w0, _, _, _ = refine_wu(
        WSaddleSystem(r=R, gamma=g0, q=Q, beta=BETA),
        interpolate_w_seed_from_payload(wu_seed_payload, g0),
    )
    z_seed0, ok_z0, _, _, _ = refine_z(
        SaddleSystem.build(q=Q, r=R, betas=np.array([BETA]), gammas=np.array([g0])),
        interpolate_z_from_payload(z_seed_payload, g0),
    )
    if not ok_w0:
        raise RuntimeError(f"w,u seed failed refinement at gamma={g0}")
    if not ok_z0:
        raise RuntimeError(f"z seed failed refinement at gamma={g0}")

    catalog_wu = catalog_from_sweep(wu_sweep, g0)
    catalog_z = catalog_from_sweep(z_sweep, g0)

    root_diags_wu = diagnose_all_roots_refined(wu_sweep, g0, "wu")
    root_diags_z = diagnose_all_roots_refined(z_sweep, g0, "z")

    newton_wu = newton_from_seed(g0, w_seed0, "wu", catalog_wu)
    newton_z = newton_from_seed(g0, z_seed0, "z", catalog_z)

    comp_starts_wu = pick_competitor_starts(catalog_wu, w_seed0, g0, "wu", k=n_competitors)
    comp_starts_z = pick_competitor_starts(catalog_z, z_seed0, g0, "z", k=n_competitors)

    # refine competitor starts at g0
    comp_refined_wu: list[tuple[str, np.ndarray]] = []
    for lab, v in comp_starts_wu:
        sys = WSaddleSystem(r=R, gamma=g0, q=Q, beta=BETA)
        vr, ok, _, _, _ = refine_wu(sys, v)
        if ok:
            comp_refined_wu.append((lab, vr))

    comp_refined_z: list[tuple[str, np.ndarray]] = []
    for lab, v in comp_starts_z:
        sys = SaddleSystem.build(q=Q, r=R, betas=np.array([BETA]), gammas=np.array([g0]))
        vr, ok, _, _, _ = refine_z(sys, v)
        if ok:
            comp_refined_z.append((lab, vr))

    tracks_wu = [continue_branch("wu", "seed", w_seed0, gammas, is_seed=True)]
    tracks_wu.extend(continue_branch("wu", lab, v, gammas, is_seed=False) for lab, v in comp_refined_wu)

    tracks_z = [continue_branch("z", "seed", z_seed0, gammas, is_seed=True)]
    tracks_z.extend(continue_branch("z", lab, v, gammas, is_seed=False) for lab, v in comp_refined_z)

    crossings_wu = find_re_phi_crossings(tracks_wu, valid_only=True)
    crossings_z = find_re_phi_crossings(tracks_z, valid_only=True)

    seed_wu = tracks_wu[0]
    comp_wu = tracks_wu[1] if len(tracks_wu) > 1 else None
    seed_z = tracks_z[0]
    comp_z = tracks_z[1] if len(tracks_z) > 1 else None
    sep_wu = root_separation_series(seed_wu, comp_wu) if comp_wu else []
    sep_z = root_separation_series(seed_z, comp_z) if comp_z else []

    # pseudo-arclength from last valid point before cliff
    def arclen_seed(chart: str, track: BranchTrack, window: tuple[float, float]) -> list[dict[str, Any]]:
        mask = track.valid_mask()
        if not np.any(mask):
            return []
        g_valid = np.array(track.gammas)[mask]
        i0 = int(np.argmin(np.abs(g_valid - window[1])))
        gi = float(g_valid[i0])
        idx = track.gammas.index(gi)
        if chart == "wu":
            vec = vec_from_record({"w_real": track.vec_real[idx], "w_imag": track.vec_imag[idx]})
        else:
            vec = vec_from_record({"z_real": track.vec_real[idx], "z_imag": track.vec_imag[idx]})
        return pseudo_arclength_continue_fixed(chart, gi, vec, gamma_end=window[0], ds=0.012)

    arclen_wu = arclen_seed("wu", seed_wu, WU_ARCLEN_WINDOW)
    arclen_z = arclen_seed("z", seed_z, Z_ARCLEN_WINDOW)

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "ROOT_TOL": root_tol,
        "parameters": {"r": R, "q": Q, "beta": BETA, "gamma_start": g0, "gamma_stop": gamma_stop, "num_points": num_points},
        "caveat": (
            "Jacobian condition numbers are chart- and normalization-dependent. "
            "Compare κ only within chart. Re Phi crossings are candidate dominance "
            "transitions, not contour/thimble proofs. z uses G(z)=0; w,u uses F_tilde(w)=0."
        ),
        "newton_from_interpolated_seed": {"wu": asdict(newton_wu), "z": asdict(newton_z)},
        "root_diagnostics_at_gamma_start_refined": {
            "wu": [asdict(d) for d in root_diags_wu],
            "z": [asdict(d) for d in root_diags_z],
        },
        "tracks": {"wu": [asdict(t) for t in tracks_wu], "z": [asdict(t) for t in tracks_z]},
        "re_phi_crossings_valid_only": {"wu": crossings_wu, "z": crossings_z},
        "root_separation": {"wu": sep_wu, "z": sep_z},
        "pseudo_arclength": {"wu": arclen_wu, "z": arclen_z},
    }

    json_path = out_dir / "conditioning_continuation.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    plot_path = out_dir / "conditioning_continuation_thesis.png"
    plot_outputs(tracks_wu, tracks_z, crossings_wu, crossings_z, sep_wu, sep_z, arclen_wu, arclen_z, plot_path)

    md_lines = [
        "# Conditioning + continuation (refined roots only)",
        "",
        f"ROOT_TOL = {root_tol:.0e}. Points with |F| ≥ tolerance are **rejected**.",
        "",
        f"γ ∈ [{gamma_stop}, {gamma_start}], {num_points} steps.",
        "",
        "## Newton from interpolated seed at γ₀ (after refine)",
        "",
        "| chart | refined | iters | matched | |F| | σ_min | κ | Re Φ |",
        "|-------|---------|-------|---------|-----|-------|---|------|",
    ]
    for chart, nw in [("wu", newton_wu), ("z", newton_z)]:
        md_lines.append(
            f"| {chart} | {nw.refined} | {nw.newton_iters} | {nw.matched_label} | "
            f"{nw.F_norm:.2e} | {nw.sigma_min:.2e} | {nw.kappa:.2e} | {nw.re_phi:.4f} |"
        )
    md_lines.extend(["", "## Refined root diagnostics at γ₀", ""])
    md_lines.append("| chart | label | seed? | refined | |F| | σ_min | κ | Re Φ |")
    md_lines.append("|-------|-------|-------|---------|-----|-------|---|------|")
    for d in root_diags_wu + root_diags_z:
        md_lines.append(
            f"| {d.chart} | {d.label} | {d.is_seed} | {d.refined} | {d.F_norm:.2e} | "
            f"{d.sigma_min:.2e} | {d.kappa:.2e} | {d.re_phi:.4f} |"
        )
    md_lines.extend(["", "## Re Φ crossings (refined segments only)", ""])
    for chart, crs in [("wu", crossings_wu), ("z", crossings_z)]:
        md_lines.append(f"### {chart}")
        if not crs:
            md_lines.append("(none)")
        for cr in crs:
            md_lines.append(f"- {cr['branch_a']} vs {cr['branch_b']}: γ≈{cr['gamma_cross_est']:.4f}")
        md_lines.append("")
    md_path = out_dir / "conditioning_continuation_summary.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    return {"json": json_path, "plot": plot_path, "summary": md_path}


def main() -> None:
    p = argparse.ArgumentParser(description="Refined Jacobian conditioning + continuation")
    p.add_argument("--run-dir", type=str, default=str(DEFAULT_RUN))
    p.add_argument("--out-dir", type=str, default="")
    p.add_argument("--gamma-start", type=float, default=-0.01)
    p.add_argument("--gamma-stop", type=float, default=-1.0)
    p.add_argument("--num-points", type=int, default=50)
    p.add_argument("--n-competitors", type=int, default=3)
    p.add_argument("--root-tol", type=float, default=ROOT_TOL)
    args = p.parse_args()

    paths = run_pipeline(
        Path(args.run_dir),
        gamma_start=args.gamma_start,
        gamma_stop=args.gamma_stop,
        num_points=args.num_points,
        n_competitors=args.n_competitors,
        out_dir=Path(args.out_dir) if args.out_dir else None,
        root_tol=float(args.root_tol),
    )
    print(json.dumps({k: str(v) for k, v in paths.items()}, indent=2))


if __name__ == "__main__":
    main()
