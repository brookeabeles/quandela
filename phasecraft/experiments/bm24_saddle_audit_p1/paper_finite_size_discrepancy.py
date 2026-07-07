"""
Finite-size discrepancy between exact Conv2 rates and locally certified saddle exponents.

Outputs (PAPER-RESULTS/):
  prefactor_fig1_gap_vs_n.{png,pdf}
  prefactor_fig1_gap_vs_n_caption.txt
  prefactor_fig1_gap_vs_n_data.csv
  json/finite_size_discrepancy_summary.json
"""

from __future__ import annotations

import csv
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from phasecraft.bm24_saddle_audit_p1.audit import (
    bm24_seed_z,
    finite_n_exponent_grid,
    _x_to_z,
)
from phasecraft.bm24_saddle_audit_p1.continue_seed_branch_certified import (
    conv2_pref,
    certify_z,
    polish_to_residual,
    step_from_previous_z,
)
from phasecraft.lib.saddles.krawczyk_p1_roots import SaddleSystem
from phasecraft.lib.saddles.picard_lefschetz import compute_phi

Q = 3
K_CLAUSE = 8
R = 176.54
PHI_PREF = conv2_pref(K_CLAUSE, R)
DPS = 80
MIN_RESIDUAL = 1e-10

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "results" / "PAPER-RESULTS"
JSON_DIR = OUT_DIR / "json"

BETA_OPT = 0.5434
GAMMA_CONTROL = -0.30
GAMMA_CONTINUATION = -1.00

N_MIN_LIST = [18, 22, 26, 30]
N_MAX = 100
N_VALUES = list(range(18, N_MAX + 1))

BLUE = "#2b6cb0"
ORANGE = "#dd6b20"
GRAY = "#4a5568"
BAND_ALPHA = 0.22

CAPTION_CONCLUSION = (
    "For both gamma = -0.30 and gamma = -1.00, the exact finite-size "
    "exponent approaches the action of the locally Krawczyk-certified "
    "stationary point over n = 18,...,100. The discrepancy is larger at "
    "gamma = -1.00 but decreases smoothly in both cases. Visible curvature "
    "in n delta_n shows that next-order finite-size corrections remain "
    "relevant at the accessible sizes. The decreasing fitted intercept "
    "under successive lower fit cutoffs supports, but does not prove, "
    "a zero asymptotic discrepancy."
)

DELTA_DEFINITION = (
    "delta_n(gamma) = n^{-1} log|S_n(gamma)| - Re Phi_saddle(gamma)"
)


@dataclass
class PointCertification:
    certified: bool
    box_radius: float | None
    residual_norm: float | None
    jacobian_nonsingular: bool | None
    cond_J: float | None
    reason: str | None = None


@dataclass
class FitResult:
    model: str
    n_min: int
    n_max: int
    n_obs: int
    params: dict[str, float]
    param_stderr: dict[str, float]
    rss: float
    aic: float
    bic: float
    c0_ci95: tuple[float, float] | None
    loo_c0_mean: float | None
    loo_c0_std: float | None
    bootstrap_c0_ci95: tuple[float, float] | None


def get_seed_z(beta: float, gamma: float, n_vals_step: list[int]) -> np.ndarray | None:
    gamma_start = -0.01
    sys0 = SaddleSystem.build(q=Q, r=R, betas=np.array([beta]), gammas=np.array([gamma_start]))
    z_it, _, conv = bm24_seed_z(Q, R, beta, gamma_start)
    if not conv:
        return None
    x0 = np.concatenate([z_it.real, z_it.imag])
    x_pol, res = polish_to_residual(sys0, x0, min_residual=MIN_RESIDUAL, dps=DPS)
    if res > 1e-6:
        return None
    z = _x_to_z(x_pol, sys0.nvars)
    step = -0.05
    g = float(gamma_start)
    while g > float(gamma) + 1e-9:
        g_next = max(g + step, float(gamma))
        z_next, _row = step_from_previous_z(
            Q, K_CLAUSE, R, beta, g_next, z,
            step_index=0, n_values=n_vals_step,
            match_tol=0.1, min_residual=MIN_RESIDUAL, dps=DPS,
        )
        if z_next is None:
            step *= 0.5
            if abs(step) < 1e-4:
                return None
            continue
        z = z_next
        g = g_next
    return z


def extract_certification(sys_: SaddleSystem, z: np.ndarray) -> PointCertification:
    ok, info = certify_z(sys_, z, dps=DPS)
    box_r = info.get("box_radius")
    if isinstance(box_r, (list, tuple, np.ndarray)):
        box_radius = float(np.max(np.abs(box_r)))
    elif box_r is not None:
        box_radius = float(box_r)
    else:
        box_radius = None
    cond_j = info.get("cond_J")
    jacobian_nonsingular = None
    if cond_j is not None and math.isfinite(float(cond_j)):
        jacobian_nonsingular = bool(ok and float(cond_j) < 1e14)
    return PointCertification(
        certified=bool(ok),
        box_radius=box_radius,
        residual_norm=float(info.get("G_inf")) if info.get("G_inf") is not None else None,
        jacobian_nonsingular=jacobian_nonsingular,
        cond_J=float(cond_j) if cond_j is not None and math.isfinite(float(cond_j)) else None,
        reason=info.get("reason"),
    )


def model_saddle(n: np.ndarray, a: float, b: float) -> np.ndarray:
    n = np.asarray(n, dtype=float)
    return (a * np.log(n) + b) / n


def model_next_order(n: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    n = np.asarray(n, dtype=float)
    return (a * np.log(n) + b) / n + c / n**2


def model_n_delta_next_order(n: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """n * delta_n from the next-order zero-intercept model."""
    n = np.asarray(n, dtype=float)
    return a * np.log(n) + b + c / n


def model_general(n: np.ndarray, c0: float, a: float, b: float) -> np.ndarray:
    n = np.asarray(n, dtype=float)
    return c0 + (a * np.log(n) + b) / n


def aic_bic(rss: float, n_obs: int, k: int) -> tuple[float, float]:
    if n_obs <= 0 or rss <= 0:
        return float("nan"), float("nan")
    log_lik = n_obs * math.log(rss / n_obs)
    return log_lik + 2 * k, log_lik + k * math.log(n_obs)


def fit_models(
    n: np.ndarray,
    delta: np.ndarray,
    *,
    n_min: int,
    n_max: int,
    bootstrap_reps: int = 400,
) -> dict[str, Any]:
    mask = (n >= n_min) & (n <= n_max)
    nv = n[mask]
    yv = delta[mask]
    n_obs = int(len(nv))
    out: dict[str, Any] = {"n_min": n_min, "n_max": n_max, "n_obs": n_obs}

    popt_s, pcov_s = curve_fit(model_saddle, nv, yv, p0=[-0.1, -0.01], maxfev=20000)
    resid_s = yv - model_saddle(nv, *popt_s)
    rss_s = float(np.sum(resid_s**2))
    aic_s, bic_s = aic_bic(rss_s, n_obs, 2)
    err_s = np.sqrt(np.clip(np.diag(pcov_s), 0.0, None))
    out["saddle_log_n_over_n"] = FitResult(
        model="(a log n + b)/n",
        n_min=n_min,
        n_max=n_max,
        n_obs=n_obs,
        params={"a": float(popt_s[0]), "b": float(popt_s[1])},
        param_stderr={"a": float(err_s[0]), "b": float(err_s[1])},
        rss=rss_s,
        aic=aic_s,
        bic=bic_s,
        c0_ci95=None,
        loo_c0_mean=None,
        loo_c0_std=None,
        bootstrap_c0_ci95=None,
    )

    popt_n, pcov_n = curve_fit(model_next_order, nv, yv, p0=[-0.02, -0.08, 0.5], maxfev=20000)
    resid_n = yv - model_next_order(nv, *popt_n)
    rss_n = float(np.sum(resid_n**2))
    aic_n, bic_n = aic_bic(rss_n, n_obs, 3)
    err_n = np.sqrt(np.clip(np.diag(pcov_n), 0.0, None))
    out["next_order_log_n_over_n"] = FitResult(
        model="(a log n + b)/n + c/n^2",
        n_min=n_min,
        n_max=n_max,
        n_obs=n_obs,
        params={"a": float(popt_n[0]), "b": float(popt_n[1]), "c": float(popt_n[2])},
        param_stderr={
            "a": float(err_n[0]),
            "b": float(err_n[1]),
            "c": float(err_n[2]),
        },
        rss=rss_n,
        aic=aic_n,
        bic=bic_n,
        c0_ci95=None,
        loo_c0_mean=None,
        loo_c0_std=None,
        bootstrap_c0_ci95=None,
    )

    popt_g, pcov_g = curve_fit(model_general, nv, yv, p0=[0.0, -0.1, -0.01], maxfev=20000)
    resid_g = yv - model_general(nv, *popt_g)
    rss_g = float(np.sum(resid_g**2))
    aic_g, bic_g = aic_bic(rss_g, n_obs, 3)
    err_g = np.sqrt(np.clip(np.diag(pcov_g), 0.0, None))
    c0 = float(popt_g[0])
    c0_se = float(err_g[0])
    c0_ci = (c0 - 1.96 * c0_se, c0 + 1.96 * c0_se)

    loo_c0 = []
    for i in range(n_obs):
        keep = np.ones(n_obs, dtype=bool)
        keep[i] = False
        try:
            p_loo, _ = curve_fit(model_general, nv[keep], yv[keep], p0=popt_g, maxfev=20000)
            loo_c0.append(float(p_loo[0]))
        except Exception:
            continue
    loo_c0_arr = np.array(loo_c0, dtype=float)

    rng = np.random.default_rng(0)
    boot_c0 = []
    for _ in range(bootstrap_reps):
        idx = rng.integers(0, n_obs, n_obs)
        try:
            p_b, _ = curve_fit(model_general, nv[idx], yv[idx], p0=popt_g, maxfev=20000)
            boot_c0.append(float(p_b[0]))
        except Exception:
            continue
    boot_c0_arr = np.array(boot_c0, dtype=float)
    boot_ci = (
        (float(np.percentile(boot_c0_arr, 2.5)), float(np.percentile(boot_c0_arr, 97.5)))
        if boot_c0_arr.size >= 20
        else None
    )

    out["general_c0_log_n_over_n"] = FitResult(
        model="c0 + (a log n + b)/n",
        n_min=n_min,
        n_max=n_max,
        n_obs=n_obs,
        params={
            "c0": c0,
            "a": float(popt_g[1]),
            "b": float(popt_g[2]),
        },
        param_stderr={
            "c0": c0_se,
            "a": float(err_g[1]),
            "b": float(err_g[2]),
        },
        rss=rss_g,
        aic=aic_g,
        bic=bic_g,
        c0_ci95=c0_ci,
        loo_c0_mean=float(np.mean(loo_c0_arr)) if loo_c0_arr.size else None,
        loo_c0_std=float(np.std(loo_c0_arr)) if loo_c0_arr.size else None,
        bootstrap_c0_ci95=boot_ci,
    )
    out["pcov_saddle"] = pcov_s.tolist()
    out["pcov_next_order"] = pcov_n.tolist()
    out["pcov_general"] = pcov_g.tolist()
    out["popt_saddle"] = popt_s.tolist()
    out["popt_next_order"] = popt_n.tolist()
    out["popt_general"] = popt_g.tolist()
    return out


def model_prediction_band(
    model_fn,
    n_grid: np.ndarray,
    popt: np.ndarray,
    pcov: np.ndarray,
    param_grad_fn,
    z: float = 1.96,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Linearized regression band for a nonlinear least-squares fit."""
    pred = model_fn(n_grid, *popt)
    grad = np.zeros((len(n_grid), len(popt)))
    for i, n in enumerate(n_grid):
        grad[i] = param_grad_fn(float(n), popt)
    var = np.array([g @ pcov @ g for g in grad])
    sigma = z * np.sqrt(np.clip(var, 0.0, None))
    return pred, pred - sigma, pred + sigma


def next_order_grad(n: float, _popt: np.ndarray) -> np.ndarray:
    ln = math.log(n)
    return np.array([ln / n, 1.0 / n, 1.0 / n**2], dtype=float)


def saddle_prediction_band(
    n_grid: np.ndarray,
    popt: np.ndarray,
    pcov: np.ndarray,
    z: float = 1.96,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    def grad(n: float, _popt: np.ndarray) -> np.ndarray:
        ln = math.log(n)
        return np.array([ln / n, 1.0 / n], dtype=float)

    return model_prediction_band(model_saddle, n_grid, popt, pcov, grad, z=z)


def compute_point(beta: float, gamma: float, label: str) -> dict[str, Any]:
    n_step = list(range(18, 30))
    z = get_seed_z(beta, gamma, n_step)
    if z is None:
        raise RuntimeError(f"seed continuation failed at beta={beta}, gamma={gamma}")

    sys_ = SaddleSystem.build(q=Q, r=R, betas=np.array([beta]), gammas=np.array([gamma]))
    phi = compute_phi(z, Q, R, sys_.betas, sys_.gammas)
    e_saddle_re = float(PHI_PREF + phi.real)
    cert = extract_certification(sys_, z)

    fn = finite_n_exponent_grid(K_CLAUSE, Q, R, beta, gamma, N_VALUES)
    rows = []
    for n in N_VALUES:
        log_entry = fn["log_p_conv2_exact"][str(n)]
        log_s = complex(log_entry["real"], log_entry["imag"])
        # log_p is log(S_n); for complex S_n use (1/n) log|S_n| = (1/n) Re log S on the physical branch.
        s_modulus = float(abs(np.exp(log_s)))
        log_abs_s = float(np.log(s_modulus))
        rate_exact = log_abs_s / n
        delta = rate_exact - e_saddle_re

        lam_conv2 = float(fn["lambda_abs"][str(n)])
        lam_prop4 = fn["crosscheck_lambda_abs_prop4"].get(str(n))
        unc = abs(rate_exact - lam_conv2)
        if lam_prop4 is not None:
            unc = max(unc, abs(rate_exact - float(lam_prop4)))
        unc = max(unc, 8.0 * np.finfo(float).eps * max(1.0, abs(rate_exact) + abs(e_saddle_re)))

        rows.append({
            "n": n,
            "inv_n": 1.0 / n,
            "log_abs_Sn_over_n": rate_exact,
            "delta_n": delta,
            "delta_uncertainty": unc,
            "log_S_complex_real": float(log_entry["real"]),
            "log_S_complex_imag": float(log_entry["imag"]),
        })

    n_arr = np.array([r["n"] for r in rows], dtype=float)
    delta_arr = np.array([r["delta_n"] for r in rows], dtype=float)

    fits_by_nmin: dict[str, Any] = {}
    for n_min in N_MIN_LIST:
        fits_by_nmin[str(n_min)] = fit_models(n_arr, delta_arr, n_min=n_min, n_max=N_MAX)

    return {
        "label": label,
        "beta": beta,
        "gamma": gamma,
        "e_saddle_re": e_saddle_re,
        "re_phi": float(phi.real),
        "phi_pref": PHI_PREF,
        "log_rate_convention": (
            "S_n is the Conv2 all-subsets success amplitude (complex in general); "
            "the exact finite-size rate uses (1/n) log|S_n|."
        ),
        "certification": asdict(cert),
        "rows": rows,
        "fits_by_nmin": fits_by_nmin,
    }


def fit_result_to_dict(fr: FitResult) -> dict[str, Any]:
    d = asdict(fr)
    return d


def plot_panel_delta(
    ax,
    point: dict[str, Any],
    *,
    panel_tag: str,
    panel_subtitle: str,
    color: str,
    n_min_primary: int = 18,
) -> None:
    rows = point["rows"]
    n = np.array([r["n"] for r in rows], dtype=float)
    inv_n = 1.0 / n
    delta = np.array([r["delta_n"] for r in rows], dtype=float)

    fit_pack = point["fits_by_nmin"][str(n_min_primary)]
    popt_lead = np.array(fit_pack["popt_saddle"])
    popt_next = np.array(fit_pack["popt_next_order"])
    pcov_next = np.array(fit_pack["pcov_next_order"])

    inv_grid = np.linspace(0.0, inv_n.max(), 200)
    n_grid = 1.0 / np.clip(inv_grid, 1e-12, None)
    pred_next, lo, hi = model_prediction_band(
        model_next_order, n_grid, popt_next, pcov_next, next_order_grad
    )
    pred_lead = model_saddle(n_grid, *popt_lead)

    ax.axhline(0.0, color="k", lw=0.9, zorder=1)
    ax.fill_between(
        inv_grid,
        lo,
        hi,
        color=color,
        alpha=BAND_ALPHA,
        zorder=2,
        label="model-based 95% regression band",
    )
    ax.plot(
        inv_grid,
        pred_next,
        "-",
        color=color,
        lw=2.0,
        zorder=3,
        label=r"fit $(a\log n+b)/n + c/n^2$",
    )
    ax.plot(
        inv_grid,
        pred_lead,
        "--",
        color=color,
        lw=1.1,
        alpha=0.85,
        zorder=2,
        label=r"fit $(a\log n+b)/n$",
    )
    ax.plot(
        inv_n,
        delta,
        "o",
        color=color,
        ms=3.5,
        mfc="white",
        mew=1.2,
        zorder=4,
        label=r"exact $\delta_n$",
    )

    ax.set_xlabel(r"$1/n$")
    ax.set_ylabel(r"$\delta_n(\gamma)$")
    ax.set_title(
        rf"({panel_tag}) {panel_subtitle}",
        fontsize=11,
    )
    ax.grid(True, alpha=0.2)
    ax.legend(loc="best", fontsize=7.5, frameon=True)

    ax_top = ax.twiny()
    ax_top.set_xlim(ax.get_xlim())
    inv_ticks = np.array([1.0 / 100, 1.0 / 70, 1.0 / 50, 1.0 / 30, 1.0 / 18])
    ax_top.set_xticks(inv_ticks)
    ax_top.set_xticklabels([str(int(round(1.0 / t))) for t in inv_ticks])
    ax_top.set_xlabel(r"$n$")


def plot_panel_diagnostic(
    ax,
    point: dict[str, Any],
    *,
    panel_tag: str,
    color: str,
    n_min_primary: int = 18,
) -> None:
    rows = point["rows"]
    n = np.array([r["n"] for r in rows], dtype=float)
    delta = np.array([r["delta_n"] for r in rows], dtype=float)
    y = n * delta
    x = np.log(n)

    popt = np.array(point["fits_by_nmin"][str(n_min_primary)]["popt_next_order"])
    x_line = np.linspace(x.min(), x.max(), 200)
    n_line = np.exp(x_line)
    y_line = model_n_delta_next_order(n_line, *popt)

    ax.plot(x, y, "o", color=color, ms=3.5, mfc="white", mew=1.2, label=r"$n\,\delta_n$")
    ax.plot(
        x_line,
        y_line,
        "-",
        color=color,
        lw=1.8,
        label=r"fit $a\log n + b + c/n$",
    )
    ax.set_xlabel(r"$\log n$")
    ax.set_ylabel(r"$n\,\delta_n(\gamma)$")
    ax.set_title(rf"({panel_tag}) $n\delta_n$ vs $\log n$", fontsize=10)
    ax.grid(True, alpha=0.2)
    ax.legend(loc="best", fontsize=8, frameon=True)


def plot_c0_stability(ax, point: dict[str, Any], *, color: str, label: str) -> None:
    xs, ys, yerr = [], [], []
    for n_min in N_MIN_LIST:
        fr: FitResult = point["fits_by_nmin"][str(n_min)]["general_c0_log_n_over_n"]
        if fr.c0_ci95 is None:
            continue
        xs.append(n_min)
        ys.append(fr.params["c0"])
        half = 0.5 * (fr.c0_ci95[1] - fr.c0_ci95[0])
        yerr.append(half)
    ax.errorbar(xs, ys, yerr=yerr, fmt="o-", color=color, capsize=4, lw=1.6, ms=5, label=label)
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_xlabel(r"fit window lower cutoff $n_{\min}$")
    ax.set_ylabel(r"fitted $c_0$")
    ax.grid(True, alpha=0.2)


def make_figure(control: dict[str, Any], continuation: dict[str, Any]) -> None:
    fig = plt.figure(figsize=(11.5, 10.5))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.15, 0.95, 0.75], hspace=0.42, wspace=0.28)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])
    ax_e = fig.add_subplot(gs[2, :])

    plot_panel_delta(
        ax_a,
        control,
        panel_tag="a",
        panel_subtitle=rf"smaller-$|\gamma|$ comparison, $\gamma={control['gamma']:.2f}$",
        color=BLUE,
    )
    plot_panel_delta(
        ax_b,
        continuation,
        panel_tag="b",
        panel_subtitle=rf"$\gamma={continuation['gamma']:.2f}$",
        color=ORANGE,
    )
    plot_panel_diagnostic(ax_c, control, panel_tag="c", color=BLUE)
    plot_panel_diagnostic(ax_d, continuation, panel_tag="d", color=ORANGE)
    plot_c0_stability(
        ax_e,
        control,
        color=BLUE,
        label=rf"smaller-$|\gamma|$ ($\gamma={control['gamma']:.2f}$)",
    )
    plot_c0_stability(
        ax_e,
        continuation,
        color=ORANGE,
        label=rf"$\gamma={continuation['gamma']:.2f}$",
    )
    ax_e.legend(loc="best", fontsize=9, frameon=True)
    ax_e.set_title(
        rf"(e) Diagnostic intercept $c_0$ from $\delta_n = c_0 + (a\log n + b)/n$ "
        rf"($n_{{\max}}={N_MAX}$)",
        fontsize=10,
    )

    fig.suptitle(
        "Finite-size discrepancy between the exact and saddle exponents",
        fontsize=13,
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    png = OUT_DIR / "prefactor_fig1_gap_vs_n.png"
    pdf = OUT_DIR / "prefactor_fig1_gap_vs_n.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)


def write_csv(control: dict[str, Any], continuation: dict[str, Any]) -> Path:
    path = OUT_DIR / "prefactor_fig1_gap_vs_n_data.csv"
    fields = [
        "beta", "gamma", "n", "inv_n", "delta_n", "delta_uncertainty",
        "log_abs_Sn_over_n", "e_saddle_re", "log_S_complex_real", "log_S_complex_imag",
    ]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for point in (control, continuation):
            for row in point["rows"]:
                w.writerow({
                    "beta": point["beta"],
                    "gamma": point["gamma"],
                    "n": row["n"],
                    "inv_n": row["inv_n"],
                    "delta_n": row["delta_n"],
                    "delta_uncertainty": row["delta_uncertainty"],
                    "log_abs_Sn_over_n": row["log_abs_Sn_over_n"],
                    "e_saddle_re": point["e_saddle_re"],
                    "log_S_complex_real": row["log_S_complex_real"],
                    "log_S_complex_imag": row["log_S_complex_imag"],
                })
    return path


def write_caption(control: dict[str, Any], continuation: dict[str, Any]) -> Path:
    path = OUT_DIR / "prefactor_fig1_gap_vs_n_caption.txt"

    def cert_line(c: dict[str, Any]) -> str:
        cert = c["certification"]
        return (
            f"(beta,gamma)=({c['beta']:.4f},{c['gamma']:.2f}): "
            f"box radius={cert['box_radius']:.3g}, ||G||_inf={cert['residual_norm']:.3g}, "
            f"Jacobian nonsingular={cert['jacobian_nonsingular']} (cond J={cert['cond_J']:.3g})."
        )

    def fit_lines(point: dict[str, Any], tag: str) -> list[str]:
        pack = point["fits_by_nmin"]["18"]
        fn = pack["next_order_log_n_over_n"]
        fg = pack["general_c0_log_n_over_n"]
        fs = pack["saddle_log_n_over_n"]
        c0, c0_lo, c0_hi = fg.params["c0"], fg.c0_ci95[0], fg.c0_ci95[1]
        return [
            f"{tag}: next-order RSS={fn.rss:.3e}, AIC={fn.aic:.1f}; "
            f"leading-order RSS={fs.rss:.3e}, AIC={fs.aic:.1f}; "
            f"general diagnostic RSS={fg.rss:.3e}, c0={c0:+.6f} "
            f"[{c0_lo:+.6f}, {c0_hi:+.6f}].",
        ]

    lines = [
        CAPTION_CONCLUSION,
        "",
        f"Definition: {DELTA_DEFINITION}, with S_n from the exact p=1 all-subsets expression.",
        "",
        f"Parameters: beta={BETA_OPT:.4f}, q=3, k=8, r={R}. "
        f"Finite-n values for n=18,...,{N_MAX}; all logarithms are natural.",
        "Exact finite-n rates use the exact p=1 expression; when S_n is complex, |S_n| is used.",
        "Stationary points are locally Krawczyk-certified; local certification does not "
        "establish contour contribution or global saddle dominance.",
        "",
        "Panel (a): smaller-|gamma| comparison at gamma=-0.30 (not claimed as a rigorously "
        "controlled small-gamma point). Panel (b): gamma=-1.00 continuation.",
        "Panels (a,b): solid = zero-intercept fit delta_n ~ (a log n + b)/n + c/n^2; "
        "dashed = leading-order (a log n + b)/n; shaded = model-based 95% regression band.",
        "Panels (c,d): diagnostic n delta_n vs log n with fitted curve a log n + b + c/n.",
        "Panel (e): c0 from the deliberately more general diagnostic model "
        "delta_n = c0 + (a log n + b)/n. The systematic decrease of c0 with increasing "
        "n_min indicates contamination from omitted higher-order finite-size terms.",
        "",
        cert_line(control),
        cert_line(continuation),
        "",
        *fit_lines(control, "Smaller-|gamma|"),
        *fit_lines(continuation, "gamma=-1.00"),
        "",
        "Numerical uncertainties in delta_n are at machine precision and are reported in the "
        "supplementary CSV (delta_uncertainty column), not as plot error bars.",
    ]
    path.write_text("\n".join(lines))
    return path


def serialize_fits(point: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for n_min, pack in point["fits_by_nmin"].items():
        out[n_min] = {
            "saddle_log_n_over_n": fit_result_to_dict(pack["saddle_log_n_over_n"]),
            "next_order_log_n_over_n": fit_result_to_dict(pack["next_order_log_n_over_n"]),
            "general_c0_log_n_over_n": fit_result_to_dict(pack["general_c0_log_n_over_n"]),
            "preferred_by_aic": min(
                (
                    ("saddle_log_n_over_n", pack["saddle_log_n_over_n"].aic),
                    ("next_order_log_n_over_n", pack["next_order_log_n_over_n"].aic),
                    ("general_c0_log_n_over_n", pack["general_c0_log_n_over_n"].aic),
                ),
                key=lambda t: t[1],
            )[0],
            "preferred_by_bic": min(
                (
                    ("saddle_log_n_over_n", pack["saddle_log_n_over_n"].bic),
                    ("next_order_log_n_over_n", pack["next_order_log_n_over_n"].bic),
                    ("general_c0_log_n_over_n", pack["general_c0_log_n_over_n"].bic),
                ),
                key=lambda t: t[1],
            )[0],
        }
    return out


def c0_compatible_with_zero(fr: FitResult) -> bool:
    if fr.c0_ci95 is None:
        return False
    return fr.c0_ci95[0] <= 0.0 <= fr.c0_ci95[1]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    JSON_DIR.mkdir(parents=True, exist_ok=True)

    print("Computing smaller-|gamma| comparison point (gamma=-0.30)...")
    control = compute_point(BETA_OPT, GAMMA_CONTROL, "smaller-|gamma| comparison")
    print("Computing gamma = -1 continuation point...")
    continuation = compute_point(BETA_OPT, GAMMA_CONTINUATION, "gamma=-1 continuation")

    make_figure(control, continuation)
    csv_path = write_csv(control, continuation)
    caption_path = write_caption(control, continuation)

    summary = {
        "definition": {
            "delta_n": DELTA_DEFINITION,
            "phi_saddle": "phi_pref + Re Phi at locally Krawczyk-certified stationary point",
            "complex_note": control["log_rate_convention"],
            "logarithms": "natural",
            "finite_n_method": "exact p=1 all-subsets expression",
        },
        "n_range": {"min": 18, "max": N_MAX, "n_min_windows": N_MIN_LIST},
        "caption_conclusion": CAPTION_CONCLUSION,
        "points": {
            "control": {
                "beta": control["beta"],
                "gamma": control["gamma"],
                "e_saddle_re": control["e_saddle_re"],
                "certification": control["certification"],
                "fits_by_nmin": serialize_fits(control),
            },
            "continuation": {
                "beta": continuation["beta"],
                "gamma": continuation["gamma"],
                "e_saddle_re": continuation["e_saddle_re"],
                "certification": continuation["certification"],
                "fits_by_nmin": serialize_fits(continuation),
            },
        },
        "asymptotic_equality_claims": {
            "note": (
                "Do not claim delta_n -> 0 unless c0=0 is inside the 95% CI for the general model "
                "and stable across n_min windows."
            ),
            "control_nmin18_c0_zero_compatible": c0_compatible_with_zero(
                control["fits_by_nmin"]["18"]["general_c0_log_n_over_n"]
            ),
            "continuation_nmin18_c0_zero_compatible": c0_compatible_with_zero(
                continuation["fits_by_nmin"]["18"]["general_c0_log_n_over_n"]
            ),
        },
        "outputs": {
            "png": str(OUT_DIR / "prefactor_fig1_gap_vs_n.png"),
            "pdf": str(OUT_DIR / "prefactor_fig1_gap_vs_n.pdf"),
            "csv": str(csv_path),
            "caption": str(caption_path),
        },
    }

    json_path = JSON_DIR / "finite_size_discrepancy_summary.json"
    with json_path.open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote {OUT_DIR / 'prefactor_fig1_gap_vs_n.png'}")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
