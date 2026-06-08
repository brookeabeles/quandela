"""
Model-comparison dominance analysis (z_robust_full).

Δ = Re Φ_comp − Re Φ_seed
  Δ < 0 : competitor below seed
  Δ > 0 : algebraically dominant (not contour dominance without PL/thimble)

Outputs under ``z_robust_full/dominance/`` — no root-spaghetti plots.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from phasecraft.bm24_saddle_audit_p1.continue_seed_branch_certified import (
    GAMMA_AXIS_LEFT,
    GAMMA_AXIS_RIGHT,
    LABEL_EXACT_FINITE_N_EXPONENT,
    LABEL_SEED_SADDLE_EXPONENT,
    _style_gamma_axis_reading_zero_to_negative,
    conv2_full_exponent,
)
from phasecraft.experiments.bm24_saddle_audit_p1.plot_z_robust_re_sheets import (
    DEFAULT_RE_PHYS_HI,
    DEFAULT_RE_PHYS_LO,
    _filter_sweep_records,
    _load,
    _sweep_competitor_records,
)
from phasecraft.w_saddle.workflow import unwrap_phase_diff

DEFAULT_Z_ROBUST = (
    REPO_ROOT
    / "phasecraft/experiments/bm24_saddle_audit_p1/results/run_seed_branch_g-2pi/competitor_dominance/z_robust_full"
)

LANG_ALGEBRAIC = "algebraically dominant (Δ>0; not PL/contour dominance)"
LANG_FINITE_N = "finite-n explanatory (low RMSE vs exact λ_abs n_max)"
LANG_NOT_PL = "No contour/thimble dominance claim without PL evidence."

GAMMA_PAIR_TOL = 0.011
Z_SAME_SHEET_FRAC = 0.35
FINITE_N_RMSE_THRESHOLD = 0.15


def _interp_at_gamma(keys: list[float], values: list[float], gamma: float) -> float:
    if not keys:
        return float("nan")
    pairs = sorted(zip(keys, values), key=lambda t: t[0], reverse=True)
    ks = [p[0] for p in pairs]
    vs = [p[1] for p in pairs]
    if gamma >= ks[0]:
        return vs[0]
    if gamma <= ks[-1]:
        return vs[-1]
    for i in range(len(ks) - 1):
        g0, g1 = ks[i], ks[i + 1]
        if (g0 >= gamma >= g1) or (g1 >= gamma >= g0):
            t = (gamma - g0) / (g1 - g0) if abs(g1 - g0) > 1e-15 else 0.0
            return (1 - t) * vs[i] + t * vs[i + 1]
    j = min(range(len(ks)), key=lambda i: abs(ks[i] - gamma))
    return vs[j]


def _load_seed_continuation(path: Path) -> dict[float, dict[str, float]]:
    data = _load(path)
    out: dict[float, dict[str, float]] = {}
    for row in data.get("continuation", []):
        if not row.get("certified") or row.get("failed"):
            continue
        g = round(float(row["gamma"]), 6)
        out[g] = {
            "re_phi_m": float(row["re_phi_m"]),
            "im_phi_m": float(row["im_phi_m"]),
            "full_conv2": float(row["full_conv2_exponent"]),
            "lambda_abs_n_max": float(row["lambda_abs_n_max"]),
        }
    return out


def _sweep_gamma_rows(sweep: dict[str, Any]) -> list[dict[str, Any]]:
    return sorted(
        [r for r in sweep.get("points", []) if r.get("krawczyk_certified", True)],
        key=lambda r: float(r["gamma"]),
        reverse=True,
    )


def _rmse_mae(pred: np.ndarray, truth: np.ndarray) -> tuple[float, float]:
    err = pred - truth
    return float(np.sqrt(np.mean(err**2))), float(np.mean(np.abs(err)))


def _align_sweep_models(
    sweep: dict[str, Any],
    seed_cont: dict[float, dict[str, float]],
    *,
    re_lo: float,
    re_hi: float,
) -> dict[str, np.ndarray]:
    K = int(sweep.get("K_clause", 8))
    r = float(sweep.get("r", 176.54))
    cont_g = list(seed_cont.keys())
    cont_exact = [seed_cont[k]["lambda_abs_n_max"] for k in cont_g]

    gammas: list[float] = []
    exact: list[float] = []
    seed_exp: list[float] = []
    env_exp: list[float] = []
    max_delta: list[float] = []

    for row in _sweep_gamma_rows(sweep):
        g = float(row["gamma"])
        seed_re = float(row["Phi_eff_real"])
        sw = row.get("competitor_sweep") or {}
        comps, _ = _filter_sweep_records(_sweep_competitor_records(sw), re_lo=re_lo, re_hi=re_hi)
        if not comps:
            continue
        re_comp_max = max(float(c["Phi_eff_real"]) for c in comps)
        re_env = max(seed_re, re_comp_max)
        deltas = [float(c["Phi_eff_real"]) - seed_re for c in comps]
        gammas.append(g)
        exact.append(_interp_at_gamma(cont_g, cont_exact, g))
        seed_exp.append(conv2_full_exponent(seed_re, K, r))
        env_exp.append(conv2_full_exponent(re_env, K, r))
        max_delta.append(float(max(deltas)))

    g = np.array(gammas)
    order = np.argsort(g)[::-1]
    return {
        "gamma": g[order],
        "exact": np.array(exact)[order],
        "seed": np.array(seed_exp)[order],
        "envelope": np.array(env_exp)[order],
        "max_delta": np.array(max_delta)[order],
        "K": K,
        "r": r,
    }


def _bin_gamma_stats(
    gamma: np.ndarray,
    values: np.ndarray,
    *,
    n_bins: int = 45,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-bin center γ, max, 95th percentile (γ sorted ascending for bins)."""
    order = np.argsort(gamma)
    g = gamma[order]
    v = values[order]
    edges = np.linspace(float(g.min()), float(g.max()), n_bins + 1)
    centers: list[float] = []
    vmax: list[float] = []
    vp95: list[float] = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i < n_bins - 1:
            mask = (g >= lo) & (g < hi)
        else:
            mask = (g >= lo) & (g <= hi)
        if not np.any(mask):
            continue
        chunk = v[mask]
        centers.append(0.5 * (lo + hi))
        vmax.append(float(np.max(chunk)))
        vp95.append(float(np.percentile(chunk, 95)))
    c = np.array(centers)
    # plot x-axis reads 0 → −2π (decreasing), so reverse for display
    return c[::-1], np.array(vmax)[::-1], np.array(vp95)[::-1]


def _integrated_positive(gamma: np.ndarray, delta: np.ndarray) -> float:
    """∫ max(Δ,0) d|γ| along ascending γ."""
    if len(gamma) < 2:
        return float(max(0.0, delta[0])) if len(delta) else 0.0
    order = np.argsort(gamma)
    g = gamma[order]
    pos = np.maximum(delta[order], 0.0)
    trapz = getattr(np, "trapezoid", None) or np.trapz
    return float(trapz(pos, g))


def _branch_track_points(
    branch: dict[str, Any],
    *,
    re_lo: float,
    re_hi: float,
) -> list[dict[str, float]]:
    out: list[dict[str, float]] = []
    for p in branch.get("points", []):
        if branch.get("is_seed_sheet"):
            continue
        if not p.get("box_disjoint_from_seed", True):
            continue
        re_c = float(p["Phi_eff_real"])
        if not (re_lo <= re_c <= re_hi):
            continue
        out.append(
            {
                "gamma": float(p["gamma"]),
                "re_comp": re_c,
                "im_comp": float(p["Phi_eff_imag"]),
                "delta_re": -float(p["DeltaRe_vs_seed"]),
            }
        )
    return sorted(out, key=lambda x: x["gamma"], reverse=True)


def _seed_im_at_gamma(
    gamma: float,
    sweep_rows_by_g: dict[float, dict[str, Any]],
    seed_cont: dict[float, dict[str, float]],
) -> float:
    row = sweep_rows_by_g.get(round(gamma, 6))
    if row is not None:
        return float(row.get("Phi_eff_imag", 0.0))
    keys = list(seed_cont.keys())
    return _interp_at_gamma(keys, [seed_cont[k]["im_phi_m"] for k in keys], gamma)


def build_branch_ranking(
    resolved: dict[str, Any],
    seed_cont: dict[float, dict[str, float]],
    sweep: dict[str, Any],
    *,
    re_lo: float,
    re_hi: float,
    min_points: int = 30,
) -> list[dict[str, Any]]:
    K = int(sweep.get("K_clause", 8))
    r = float(sweep.get("r", 176.54))
    cont_g = list(seed_cont.keys())
    cont_exact = [seed_cont[k]["lambda_abs_n_max"] for k in cont_g]
    rows: list[dict[str, Any]] = []

    for br in resolved.get("branches", []):
        if br.get("is_seed_sheet"):
            continue
        pts = _branch_track_points(br, re_lo=re_lo, re_hi=re_hi)
        n = len(pts)
        if n < min_points:
            continue
        g = np.array([p["gamma"] for p in pts])
        d = np.array([p["delta_re"] for p in pts])
        branch_exp = np.array([conv2_full_exponent(p["re_comp"], K, r) for p in pts])
        exact = np.array([_interp_at_gamma(cont_g, cont_exact, p["gamma"]) for p in pts])
        rmse, mae = _rmse_mae(branch_exp, exact)
        rows.append(
            {
                "branch_id": int(br["branch_id"]),
                "label": br.get("label", ""),
                "n_points": n,
                "max_delta": float(np.max(d)),
                "median_delta": float(np.median(d)),
                "frac_delta_positive": float(np.mean(d > 0)),
                "integrated_positive_delta": _integrated_positive(g, d),
                "rmse_vs_exact": rmse,
                "mae_vs_exact": mae,
                "gamma_min": float(g.min()),
                "gamma_max": float(g.max()),
                "finite_n_explanatory": bool(rmse < 0.15),
            }
        )
    rows.sort(
        key=lambda x: (x["integrated_positive_delta"], x["max_delta"], x["n_points"]),
        reverse=True,
    )
    return rows


def plot_model_compare_two_panel(
    models: dict[str, np.ndarray],
    out_path: Path,
    *,
    dpi: int = 160,
) -> dict[str, float]:
    g = models["gamma"]
    ex = models["exact"]
    se = models["seed"]
    ee = models["envelope"]
    res_seed = ex - se
    res_env = ex - ee
    rmse_s, mae_s = _rmse_mae(se, ex)
    rmse_e, mae_e = _rmse_mae(ee, ex)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    ax0, ax1 = axes
    ax0.plot(g, ex, "s--", ms=3, color="C0", label=LABEL_EXACT_FINITE_N_EXPONENT)
    ax0.plot(g, se, "k-", lw=2.2, label=LABEL_SEED_SADDLE_EXPONENT)
    ax0.plot(
        g,
        ee,
        color="C3",
        lw=1.8,
        label=r"Competitor-envelope exponent ($\mathrm{Re}\,\Phi_{\mathrm{env}}$+pref)",
    )
    ax0.set_ylabel(r"$\frac{1}{n}\log|S_n|$ scale ($n_{\max}$)")
    ax0.set_title("Model comparison: seed vs competitor-envelope vs exact finite-n")
    ax0.legend(loc="upper left", fontsize=8)
    ax0.grid(True, alpha=0.25)

    ax1.plot(g, res_seed, "k-", lw=1.2, alpha=0.85, label="exact − seed")
    ax1.plot(g, res_env, color="C3", lw=1.2, alpha=0.85, label="exact − envelope")
    ax1.axhline(0.0, color="gray", lw=0.6)
    ax1.set_ylabel("residual")
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(True, alpha=0.25)
    _style_gamma_axis_reading_zero_to_negative(ax1)

    txt = (
        f"seed: RMSE={rmse_s:.4f}  MAE={mae_s:.4f}  |  "
        f"envelope: RMSE={rmse_e:.4f}  MAE={mae_e:.4f}\n"
        f"{LANG_NOT_PL}"
    )
    fig.text(0.01, 0.01, txt, fontsize=8, color="0.3")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return {"rmse_seed": rmse_s, "mae_seed": mae_s, "rmse_envelope": rmse_e, "mae_envelope": mae_e}


def plot_max_delta_binned(
    models: dict[str, np.ndarray],
    out_path: Path,
    *,
    n_bins: int = 45,
    dpi: int = 160,
) -> dict[str, Any]:
    g = models["gamma"]
    md = models["max_delta"]
    centers, bmax, bp95 = _bin_gamma_stats(g, md, n_bins=n_bins)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(g, md, color="C3", lw=0.6, alpha=0.25, label=r"raw $\max\Delta(\gamma)$")
    ax.plot(centers, bmax, "C3-", lw=2.5, label=r"binned $\max\Delta$")
    ax.plot(centers, bp95, "C1--", lw=1.5, alpha=0.9, label=r"binned 95th pct $\Delta$")
    ax.axhline(0.0, color="k", lw=1.0)

    sustained = bmax > 0
    if len(centers) >= 2 and np.any(sustained):
        # shade contiguous runs where binned max > 0
        i = 0
        while i < len(centers):
            if not sustained[i]:
                i += 1
                continue
            j = i
            while j + 1 < len(centers) and sustained[j + 1]:
                j += 1
            ax.axvspan(min(centers[i], centers[j]), max(centers[i], centers[j]), color="salmon", alpha=0.25)
            i = j + 1
        ax.fill_between([], [], [], color="salmon", alpha=0.25, label=LANG_ALGEBRAIC + " (sustained)")

    ax.set_ylabel(r"$\Delta = \mathrm{Re}\,\Phi_{\mathrm{comp}}-\mathrm{Re}\,\Phi_{\mathrm{seed}}$")
    ax.set_title("Binned competitor dominance (in-band sweep only)")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.25)
    _style_gamma_axis_reading_zero_to_negative(ax)
    fig.text(0.01, 0.01, LANG_NOT_PL, fontsize=8, color="0.35")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return {"n_bins": len(centers), "frac_bins_algebraically_dominant": float(np.mean(sustained))}


def write_branch_ranking_table(rows: list[dict[str, Any]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "branch_ranking.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    fields = [
        "branch_id",
        "label",
        "n_points",
        "max_delta",
        "median_delta",
        "frac_delta_positive",
        "integrated_positive_delta",
        "rmse_vs_exact",
        "mae_vs_exact",
        "gamma_min",
        "gamma_max",
        "finite_n_explanatory",
    ]
    with (out_dir / "branch_ranking.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)


def plot_branch_vs_exact(
    branch: dict[str, Any],
    models: dict[str, np.ndarray],
    seed_cont: dict[float, dict[str, float]],
    out_path: Path,
    *,
    re_lo: float,
    re_hi: float,
    dpi: int = 150,
) -> dict[str, float]:
    K = int(models["K"])
    r = float(models["r"])
    cont_g = list(seed_cont.keys())
    cont_exact = [seed_cont[k]["lambda_abs_n_max"] for k in cont_g]

    pts = _branch_track_points(branch, re_lo=re_lo, re_hi=re_hi)
    if len(pts) < 2:
        return {}
    g = np.array([p["gamma"] for p in pts])
    branch_exp = np.array([conv2_full_exponent(p["re_comp"], K, r) for p in pts])
    exact = np.array([_interp_at_gamma(cont_g, cont_exact, p["gamma"]) for p in pts])

    # interpolate seed/envelope onto branch gammas
    mg = models["gamma"]
    seed_on = np.array([_interp_at_gamma(mg.tolist(), models["seed"].tolist(), gi) for gi in g])
    env_on = np.array([_interp_at_gamma(mg.tolist(), models["envelope"].tolist(), gi) for gi in g])
    ex_on = np.array([_interp_at_gamma(mg.tolist(), models["exact"].tolist(), gi) for gi in g])

    rmse_b, mae_b = _rmse_mae(branch_exp, exact)
    rmse_s, _ = _rmse_mae(seed_on, ex_on)
    rmse_e, _ = _rmse_mae(env_on, ex_on)
    explains = rmse_b < min(rmse_s, rmse_e) * 0.95

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    ax0, ax1 = axes
    ax0.plot(mg, models["exact"], "s--", ms=2, color="C0", alpha=0.5, label="exact (full mesh)")
    ax0.plot(g, ex_on, "o-", ms=3, color="C0", lw=1.2, label="exact @ branch γ")
    ax0.plot(g, branch_exp, "C2-o", ms=4, lw=1.8, label=f"branch {branch['branch_id']} exponent")
    ax0.plot(g, seed_on, "k--", lw=1, alpha=0.6, label="seed model @ branch γ")
    ax0.plot(g, env_on, color="C3", ls="--", lw=1, alpha=0.6, label="envelope @ branch γ")
    tag = LANG_FINITE_N if explains else "not finite-n explanatory vs seed/envelope"
    ax0.set_title(f"Branch {branch['branch_id']}: one sheet vs exact  ({tag})")
    ax0.set_ylabel("exponent")
    ax0.legend(loc="upper left", fontsize=7)
    ax0.grid(True, alpha=0.25)

    ax1.plot(g, exact - branch_exp, "C2-o", ms=3, label="exact − branch")
    ax1.plot(g, exact - seed_on, "k--", lw=1, alpha=0.7, label="exact − seed")
    ax1.plot(g, exact - env_on, color="C3", ls="--", lw=1, alpha=0.7, label="exact − envelope")
    ax1.axhline(0.0, color="gray", lw=0.6)
    ax1.set_ylabel("residual")
    ax1.legend(loc="upper left", fontsize=7)
    ax1.grid(True, alpha=0.25)
    _style_gamma_axis_reading_zero_to_negative(ax1)
    fig.text(
        0.01,
        0.01,
        f"branch RMSE={rmse_b:.4f}  seed RMSE={rmse_s:.4f}  envelope RMSE={rmse_e:.4f}. "
        + LANG_NOT_PL,
        fontsize=8,
        color="0.35",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return {
        "rmse_branch": rmse_b,
        "rmse_seed": rmse_s,
        "rmse_envelope": rmse_e,
        "finite_n_explanatory_vs_mesh": bool(explains),
    }


def _z_vec(p: dict[str, Any]) -> np.ndarray:
    return np.asarray(p["w_real"], dtype=float) + 1j * np.asarray(p["w_imag"], dtype=float)


def _disjoint_branch_points(branch: dict[str, Any], *, re_lo: float, re_hi: float) -> list[dict[str, Any]]:
    return _branch_track_points(branch, re_lo=re_lo, re_hi=re_hi)


def _gamma_step_stats(pts: list[dict[str, Any]]) -> dict[str, float]:
    if len(pts) < 2:
        return {"median": 0.0, "max": 0.0, "p95": 0.0}
    zs = [_z_vec(p) for p in pts]
    steps = [float(np.linalg.norm(zs[i] - zs[i + 1], ord=np.inf)) for i in range(len(zs) - 1)]
    arr = np.array(steps)
    return {"median": float(np.median(arr)), "max": float(np.max(arr)), "p95": float(np.percentile(arr, 95))}


def compare_branches_43_46(
    resolved: dict[str, Any],
    *,
    re_lo: float = DEFAULT_RE_PHYS_LO,
    re_hi: float = DEFAULT_RE_PHYS_HI,
) -> dict[str, Any]:
    """Coordinate / action comparison for branches 43 and 46 at paired γ."""
    b43 = next(b for b in resolved["branches"] if int(b["branch_id"]) == 43)
    b46 = next(b for b in resolved["branches"] if int(b["branch_id"]) == 46)
    p43 = _disjoint_branch_points(b43, re_lo=re_lo, re_hi=re_hi)
    p46 = _disjoint_branch_points(b46, re_lo=re_lo, re_hi=re_hi)
    g46 = np.array([p["gamma"] for p in p46])

    pairs: list[dict[str, Any]] = []
    for p in p43:
        g = float(p["gamma"])
        j = int(np.argmin(np.abs(g46 - g)))
        dg = float(g46[j] - g)
        if abs(dg) > GAMMA_PAIR_TOL:
            continue
        pt43 = next(x for x in b43["points"] if abs(float(x["gamma"]) - g) < 1e-9)
        pt46 = next(x for x in b46["points"] if abs(float(x["gamma"]) - float(g46[j])) < 1e-9)
        z43 = _z_vec(pt43)
        z46 = _z_vec(pt46)
        scale = max(float(np.max(np.abs(z43))), 1e-3) * Z_SAME_SHEET_FRAC
        z_dist = float(np.linalg.norm(z43 - z46, ord=np.inf))
        d_re = float(pt46["Phi_eff_real"]) - float(pt43["Phi_eff_real"])
        d_im = float(pt46["Phi_eff_imag"]) - float(pt43["Phi_eff_imag"])
        im_sum = float(np.abs(pt43["Phi_eff_imag"] + pt46["Phi_eff_imag"]))
        comp_close = [float(abs(z43[i] - z46[i])) for i in range(len(z43))]
        pairs.append(
            {
                "gamma": g,
                "gamma_46": float(g46[j]),
                "z_dist_inf": z_dist,
                "z_close": z_dist <= max(scale, 0.05),
                "d_re": d_re,
                "d_im": d_im,
                "im_opposite_pair": im_sum <= 0.5,
                "max_comp_abs_diff": max(comp_close),
            }
        )

    zd = np.array([x["z_dist_inf"] for x in pairs]) if pairs else np.array([])
    dre = np.array([x["d_re"] for x in pairs]) if pairs else np.array([])
    dim = np.array([x["d_im"] for x in pairs]) if pairs else np.array([])

    w_close_frac = float(np.mean([x["z_close"] for x in pairs])) if pairs else 0.0
    re_close_frac = float(np.mean(np.abs(dre) <= 0.35)) if len(dre) else 0.0
    im_opp_frac = float(np.mean([x["im_opposite_pair"] for x in pairs])) if pairs else 0.0

    same_z_sheet = bool(w_close_frac >= 0.5 and re_close_frac >= 0.5)
    same_re_sheet = bool(re_close_frac >= 0.9 and float(np.max(np.abs(dre))) < 0.5 if len(dre) else False)

    pts43_raw = [
        x
        for x in b43["points"]
        if x.get("box_disjoint_from_seed", True)
        and re_lo <= float(x["Phi_eff_real"]) <= re_hi
    ]
    pts46_raw = [
        x
        for x in b46["points"]
        if x.get("box_disjoint_from_seed", True)
        and re_lo <= float(x["Phi_eff_real"]) <= re_hi
    ]
    step43 = _gamma_step_stats(pts43_raw)
    step46 = _gamma_step_stats(pts46_raw)

    merge_z = same_z_sheet
    merge_re_only = same_re_sheet and not same_z_sheet

    return {
        "branch_ids": [43, 46],
        "n_pairs": len(pairs),
        "n43": len(pts43_raw),
        "n46": len(pts46_raw),
        "z_dist_inf": {
            "median": float(np.median(zd)) if len(zd) else None,
            "p95": float(np.percentile(zd, 95)) if len(zd) else None,
            "max": float(np.max(zd)) if len(zd) else None,
        },
        "delta_re": {
            "max_abs": float(np.max(np.abs(dre))) if len(dre) else None,
            "median": float(np.median(dre)) if len(dre) else None,
        },
        "delta_im": {"max_abs": float(np.max(np.abs(dim))) if len(dim) else None},
        "w_close_fraction": w_close_frac,
        "re_close_fraction": re_close_frac,
        "im_opposite_fraction": im_opp_frac,
        "continuity_z_step": {"branch_43": step43, "branch_46": step46},
        "same_analytic_z_sheet": same_z_sheet,
        "same_re_action_sheet": same_re_sheet,
        "recommend_merge_z_sheet": merge_z,
        "recommend_merge_re_envelope_only": merge_re_only,
        "verdict": (
            "Same z-sheet (mergeable competitor track)"
            if merge_z
            else (
                "Different z-roots, nearly degenerate Re action — merge only for Re envelope / finite-n plot"
                if merge_re_only
                else "Distinct competitor families; do not merge"
            )
        ),
        "pairs_sample": pairs[:5],
    }


def plot_branches_43_46_coordinate_diagnostics(
    comparison: dict[str, Any],
    resolved: dict[str, Any],
    out_path: Path,
    *,
    re_lo: float = DEFAULT_RE_PHYS_LO,
    re_hi: float = DEFAULT_RE_PHYS_HI,
    dpi: int = 160,
) -> None:
    b43 = next(b for b in resolved["branches"] if int(b["branch_id"]) == 43)
    b46 = next(b for b in resolved["branches"] if int(b["branch_id"]) == 46)
    g46 = np.array(
        [
            float(x["gamma"])
            for x in b46["points"]
            if x.get("box_disjoint_from_seed", True)
            and re_lo <= float(x["Phi_eff_real"]) <= re_hi
        ]
    )

    gammas, zd, dre, dim = [], [], [], []
    for x in b43["points"]:
        if not x.get("box_disjoint_from_seed", True):
            continue
        if not (re_lo <= float(x["Phi_eff_real"]) <= re_hi):
            continue
        g = float(x["gamma"])
        j = int(np.argmin(np.abs(g46 - g)))
        if abs(g46[j] - g) > GAMMA_PAIR_TOL:
            continue
        pt46 = next(y for y in b46["points"] if abs(float(y["gamma"]) - float(g46[j])) < 1e-9)
        z43, z46 = _z_vec(x), _z_vec(pt46)
        gammas.append(g)
        zd.append(float(np.linalg.norm(z43 - z46, ord=np.inf)))
        dre.append(float(pt46["Phi_eff_real"]) - float(x["Phi_eff_real"]))
        dim.append(float(pt46["Phi_eff_imag"]) - float(x["Phi_eff_imag"]))

    # continuity steps per branch
    def steps_for(bid: int) -> tuple[np.ndarray, np.ndarray]:
        br = b43 if bid == 43 else b46
        pts = sorted(
            [
                x
                for x in br["points"]
                if x.get("box_disjoint_from_seed", True)
                and re_lo <= float(x["Phi_eff_real"]) <= re_hi
            ],
            key=lambda p: float(p["gamma"]),
            reverse=True,
        )
        gs, st = [], []
        for i in range(len(pts) - 1):
            gs.append(float(pts[i]["gamma"]))
            st.append(float(np.linalg.norm(_z_vec(pts[i]) - _z_vec(pts[i + 1]), ord=np.inf)))
        return np.array(gs), np.array(st)

    g43, s43 = steps_for(43)
    g46s, s46 = steps_for(46)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    ax = axes[0, 0]
    ax.semilogy(gammas, zd, "o", ms=3, alpha=0.7)
    ax.axhline(0.35 * 5.8, color="gray", ls=":", label="typical track tol (~0.35|z|)")
    ax.set_ylabel(r"$\|z_{43}-z_{46}\|_\infty$")
    ax.set_title("Paired γ: z distance")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(gammas, dre, "o", ms=3, alpha=0.7)
    ax.axhline(0.0, color="k", lw=0.6)
    ax.set_ylabel(r"$\Delta\mathrm{Re}\Phi$ (46−43)")
    ax.set_title("Re action nearly coincident")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(gammas, dim, "o", ms=3, alpha=0.7)
    ax.set_ylabel(r"$\Delta\mathrm{Im}\Phi$ (46−43)")
    ax.set_title("Im differences (principal log)")
    _style_gamma_axis_reading_zero_to_negative(ax)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.semilogy(g43, s43, ".-", label="branch 43 step")
    ax.semilogy(g46s, s46, ".-", label="branch 46 step")
    ax.set_ylabel(r"$\|z(\gamma)-z(\gamma')\|_\infty$")
    ax.set_title("Root-path continuity (per branch)")
    ax.legend(fontsize=7)
    _style_gamma_axis_reading_zero_to_negative(ax)
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"Branches 43 vs 46: {comparison['verdict']}",
        fontsize=11,
    )
    fig.text(0.01, 0.01, LANG_NOT_PL, fontsize=8, color="0.35")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def build_merged_43_46_re_series(
    resolved: dict[str, Any],
    seed_cont: dict[str, dict[str, float]],
    sweep: dict[str, Any],
    *,
    re_lo: float,
    re_hi: float,
) -> dict[str, np.ndarray]:
    """Merged Re competitor: union of tracks 43+46 (mean Re at paired γ)."""
    K = int(sweep.get("K_clause", 8))
    r = float(sweep.get("r", 176.54))
    cont_g = list(seed_cont.keys())
    cont_exact = [seed_cont[k]["lambda_abs_n_max"] for k in cont_g]

    def collect(bid: int) -> dict[float, float]:
        br = next(b for b in resolved["branches"] if int(b["branch_id"]) == bid)
        out: dict[float, float] = {}
        for x in br["points"]:
            if not x.get("box_disjoint_from_seed", True):
                continue
            re_c = float(x["Phi_eff_real"])
            if not (re_lo <= re_c <= re_hi):
                continue
            out[round(float(x["gamma"]), 6)] = re_c
        return out

    s43, s46 = collect(43), collect(46)
    all_g = sorted(set(s43) | set(s46), reverse=True)
    gammas: list[float] = []
    merged_re: list[float] = []
    seed_re: list[float] = []
    exact: list[float] = []

    def _re_vals_at(gamma: float) -> list[float]:
        vals: list[float] = []
        for table in (s43, s46):
            if gamma in table:
                vals.append(table[gamma])
            else:
                for gk, v in table.items():
                    if abs(gk - gamma) <= GAMMA_PAIR_TOL:
                        vals.append(v)
        return vals

    for g in all_g:
        vals = _re_vals_at(g)
        if not vals:
            continue
        re_m = float(np.mean(vals))
        row = next((r for r in sweep["points"] if round(float(r["gamma"]), 6) == g), None)
        if row is None:
            row = min(sweep["points"], key=lambda r: abs(float(r["gamma"]) - g))
        gammas.append(g)
        merged_re.append(re_m)
        seed_re.append(float(row["Phi_eff_real"]))
        exact.append(_interp_at_gamma(cont_g, cont_exact, g))

    g = np.array(gammas)
    return {
        "gamma": g,
        "exact": np.array(exact),
        "seed_exp": np.array([conv2_full_exponent(re, K, r) for re in seed_re]),
        "merged_exp": np.array([conv2_full_exponent(re, K, r) for re in merged_re]),
        "seed_re": np.array(seed_re),
        "merged_re": np.array(merged_re),
    }


def plot_main_seed_merged_43_46_vs_exact(
    series: dict[str, np.ndarray],
    comparison: dict[str, Any],
    out_path: Path,
    *,
    dpi: int = 160,
) -> dict[str, float]:
    g = series["gamma"]
    ex = series["exact"]
    se = series["seed_exp"]
    me = series["merged_exp"]
    res_seed = ex - se
    res_m = ex - me
    rmse_s, mae_s = _rmse_mae(se, ex)
    rmse_m, mae_m = _rmse_mae(me, ex)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    ax0, ax1 = axes
    ax0.plot(g, ex, "s--", ms=3, color="C0", label=LABEL_EXACT_FINITE_N_EXPONENT)
    ax0.plot(g, se, "k-", lw=2.2, label=LABEL_SEED_SADDLE_EXPONENT)
    ax0.plot(
        g,
        me,
        color="C2",
        lw=2.0,
        label=r"Merged branches 43+46 (mean $\mathrm{Re}\,\Phi$ at paired $\gamma$)",
    )
    ax0.set_ylabel(r"exponent ($n_{\max}$)")
    ax0.set_title("Finite-n explanatory competitor: merged 43/46 vs seed")
    ax0.legend(loc="upper left", fontsize=8)
    ax0.grid(True, alpha=0.25)

    ax1.plot(g, res_seed, "k-", lw=1.2, label="exact − seed")
    ax1.plot(g, res_m, color="C2", lw=1.2, label="exact − merged 43/46")
    ax1.axhline(0.0, color="gray", lw=0.6)
    ax1.set_ylabel("residual")
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(True, alpha=0.25)
    _style_gamma_axis_reading_zero_to_negative(ax1)

    txt = (
        f"seed: RMSE={rmse_s:.4f} MAE={mae_s:.4f}  |  "
        f"merged 43/46: RMSE={rmse_m:.4f} MAE={mae_m:.4f}\n"
        f"{comparison['verdict']}. {LANG_FINITE_N}. {LANG_NOT_PL}"
    )
    fig.text(0.01, 0.01, txt, fontsize=8, color="0.3")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return {"rmse_seed": rmse_s, "mae_seed": mae_s, "rmse_merged_43_46": rmse_m, "mae_merged_43_46": mae_m}


def plot_branches_43_46_phase(
    resolved: dict[str, Any],
    sweep_rows_by_g: dict[float, dict[str, Any]],
    seed_cont: dict[float, dict[str, float]],
    out_path: Path,
    *,
    re_lo: float,
    re_hi: float,
    dpi: int = 150,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 4))
    for bid, color in [(43, "C0"), (46, "C1")]:
        br = next(b for b in resolved["branches"] if int(b["branch_id"]) == bid)
        pts = _branch_track_points(br, re_lo=re_lo, re_hi=re_hi)
        gs, dim = [], []
        for i, p in enumerate(pts):
            gs.append(p["gamma"])
            im_s = _seed_im_at_gamma(p["gamma"], sweep_rows_by_g, seed_cont)
            dim.append(p["im_comp"] - im_s)
        ax.plot(gs, dim, "-o", ms=2, lw=1.2, color=color, label=f"branch {bid}")
    ax.axhline(0.0, color="k", lw=0.6)
    ax.set_ylabel(r"$\mathrm{Im}\,\Phi_{\mathrm{comp}}-\mathrm{Im}\,\Phi_{\mathrm{seed}}$")
    ax.set_title("Phase offset 43 vs 46 (principal log; large steps = sheet artifact)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    _style_gamma_axis_reading_zero_to_negative(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def run_branches_43_46_study(
    z_robust_dir: Path,
    seed_cont: dict[str, dict[str, float]],
    sweep: dict[str, Any],
    resolved: dict[str, Any],
    *,
    re_lo: float = DEFAULT_RE_PHYS_LO,
    re_hi: float = DEFAULT_RE_PHYS_HI,
    dpi: int = 160,
) -> Path:
    out = Path(z_robust_dir) / "dominance" / "branches_43_46"
    import shutil

    if out.is_dir():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    comparison = compare_branches_43_46(resolved, re_lo=re_lo, re_hi=re_hi)
    (out / "branches_43_46_same_sheet.json").write_text(
        json.dumps(comparison, indent=2), encoding="utf-8"
    )
    plot_branches_43_46_coordinate_diagnostics(
        comparison, resolved, out / "coordinate_diagnostics.png", re_lo=re_lo, re_hi=re_hi, dpi=dpi
    )
    sweep_by_g = {round(float(r["gamma"]), 6): r for r in _sweep_gamma_rows(sweep)}
    plot_branches_43_46_phase(
        resolved,
        sweep_by_g,
        seed_cont,
        out / "phase_43_vs_46.png",
        re_lo=re_lo,
        re_hi=re_hi,
        dpi=dpi,
    )
    series = build_merged_43_46_re_series(resolved, seed_cont, sweep, re_lo=re_lo, re_hi=re_hi)
    stats = plot_main_seed_merged_43_46_vs_exact(
        series, comparison, out / "main_seed_merged_43_46_vs_exact.png", dpi=dpi
    )
    (out / "merged_43_46_stats.json").write_text(
        json.dumps({**comparison, "model_stats": stats}, indent=2), encoding="utf-8"
    )
    return out


def plot_branch_phase_diagnostics(
    branch: dict[str, Any],
    sweep_rows_by_g: dict[float, dict[str, Any]],
    seed_cont: dict[float, dict[str, float]],
    out_path: Path,
    *,
    re_lo: float,
    re_hi: float,
    jump_threshold: float = 1.5,
    dpi: int = 150,
) -> dict[str, Any]:
    pts = _branch_track_points(branch, re_lo=re_lo, re_hi=re_hi)
    if len(pts) < 2:
        return {}
    g = [p["gamma"] for p in pts]
    d_im: list[float] = []
    jumps: list[float] = []
    for i, p in enumerate(pts):
        im_seed = _seed_im_at_gamma(p["gamma"], sweep_rows_by_g, seed_cont)
        d_im.append(p["im_comp"] - im_seed)
        if i > 0:
            jumps.append(abs(unwrap_phase_diff(d_im[i - 1], d_im[i])))

    max_jump = float(max(jumps)) if jumps else 0.0
    flagged = max_jump > jump_threshold

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(g, d_im, "C4-o", ms=3, lw=1.2)
    ax.axhline(0.0, color="k", lw=0.6)
    for i in range(1, len(g)):
        if jumps[i - 1] > jump_threshold:
            ax.axvline(g[i], color="red", alpha=0.15, lw=4)
    ax.set_ylabel(r"$\mathrm{Im}\,\Phi_{\mathrm{comp}}-\mathrm{Im}\,\Phi_{\mathrm{seed}}$ (principal)")
    title = f"Branch {branch['branch_id']} phase diagnostic"
    if flagged:
        title += f"  — large jump ({max_jump:.2f} rad)"
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    _style_gamma_axis_reading_zero_to_negative(ax)
    fig.text(0.01, 0.01, "Red band: |ΔIm step| > π. Large jumps suggest sheet mismatch.", fontsize=8, color="0.35")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return {"max_im_jump_rad": max_jump, "phase_jump_flagged": flagged}


def run_dominance_plots(
    z_robust_dir: Path,
    *,
    seed_continuation: Optional[Path] = None,
    re_lo: float = DEFAULT_RE_PHYS_LO,
    re_hi: float = DEFAULT_RE_PHYS_HI,
    min_branch_points: int = 30,
    top_branches: int = 6,
    dpi: int = 160,
) -> Path:
    z_robust_dir = Path(z_robust_dir)
    out_dir = z_robust_dir / "dominance"
    if out_dir.is_dir():
        for stale in (
            "max_delta_envelope.png",
            "top5_dangerous_delta.png",
            "delta_scatter_by_branch.png",
            "finite_n_vs_envelope.png",
            "dominance_summary.json",
        ):
            (out_dir / stale).unlink(missing_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    import shutil

    vs_exact_dir = out_dir / "branches_vs_exact"
    phase_dir = out_dir / "branches_phase"
    for d in (vs_exact_dir, phase_dir):
        if d.is_dir():
            shutil.rmtree(d)
        d.mkdir(parents=True)

    sweep = _load(z_robust_dir / "competitors_robust_z.json")
    resolved = _load(z_robust_dir / "resolved_robust_z.json")

    seed_path = seed_continuation or (z_robust_dir.parent.parent / "seed_branch_continuation.json")
    if not seed_path.is_file():
        seed_path = (
            REPO_ROOT
            / "phasecraft/experiments/bm24_saddle_audit_p1/results/run_seed_branch_g-2pi/seed_branch_continuation.json"
        )
    seed_cont = _load_seed_continuation(seed_path)
    sweep_by_g = {round(float(r["gamma"]), 6): r for r in _sweep_gamma_rows(sweep)}

    models = _align_sweep_models(sweep, seed_cont, re_lo=re_lo, re_hi=re_hi)
    branches_by_id = {int(b["branch_id"]): b for b in resolved.get("branches", [])}

    summary: dict[str, Any] = {
        "convention": "delta = Re_Phi_comp - Re_Phi_seed",
        "re_band": [re_lo, re_hi],
        "language": {
            "algebraically_dominant": LANG_ALGEBRAIC,
            "finite_n_explanatory": LANG_FINITE_N,
            "disclaimer": LANG_NOT_PL,
        },
    }

    summary["model_compare"] = plot_model_compare_two_panel(
        models, out_dir / "model_compare_seed_env_exact.png", dpi=dpi
    )
    summary["max_delta_binned"] = plot_max_delta_binned(
        models, out_dir / "max_delta_binned.png", dpi=dpi
    )

    ranking = build_branch_ranking(
        resolved, seed_cont, sweep, re_lo=re_lo, re_hi=re_hi, min_points=min_branch_points
    )
    write_branch_ranking_table(ranking, out_dir)
    summary["branch_ranking_top5"] = ranking[:5]

    # Per-branch diagnostics only when finite-n explanatory (not max-Δ alone)
    fn_branches = [
        r
        for r in ranking
        if r["n_points"] >= min_branch_points
        and float(r["rmse_vs_exact"]) < FINITE_N_RMSE_THRESHOLD
    ]
    branch_plots: list[dict[str, Any]] = []
    for row in fn_branches[:top_branches]:
        br = branches_by_id.get(int(row["branch_id"]))
        if not br:
            continue
        bid = int(row["branch_id"])
        if bid in (43, 46):
            continue  # replaced by branches_43_46/ study
        vs_stats = plot_branch_vs_exact(
            br,
            models,
            seed_cont,
            vs_exact_dir / f"branch_{bid}_vs_exact.png",
            re_lo=re_lo,
            re_hi=re_hi,
            dpi=dpi,
        )
        ph_stats = plot_branch_phase_diagnostics(
            br,
            sweep_by_g,
            seed_cont,
            phase_dir / f"branch_{bid}_phase.png",
            re_lo=re_lo,
            re_hi=re_hi,
            dpi=dpi,
        )
        branch_plots.append({"branch_id": bid, **row, "vs_exact": vs_stats, "phase": ph_stats})

    summary["finite_n_explanatory_branches"] = branch_plots
    study_dir = run_branches_43_46_study(
        z_robust_dir, seed_cont, sweep, resolved, re_lo=re_lo, re_hi=re_hi, dpi=dpi
    )
    summary["branches_43_46_study"] = str(study_dir)
    (out_dir / "model_comparison_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return out_dir


def main() -> None:
    p = argparse.ArgumentParser(description="BM24 model-comparison dominance (no spaghetti)")
    p.add_argument("--z-robust-dir", type=str, default=str(DEFAULT_Z_ROBUST))
    p.add_argument("--seed-continuation", type=str, default="")
    p.add_argument("--re-lo", type=float, default=DEFAULT_RE_PHYS_LO)
    p.add_argument("--re-hi", type=float, default=DEFAULT_RE_PHYS_HI)
    p.add_argument("--min-branch-points", type=int, default=30)
    p.add_argument("--top-branches", type=int, default=6)
    p.add_argument("--dpi", type=int, default=160)
    args = p.parse_args()
    seed_path = Path(args.seed_continuation) if args.seed_continuation.strip() else None
    out = run_dominance_plots(
        Path(args.z_robust_dir),
        seed_continuation=seed_path,
        re_lo=args.re_lo,
        re_hi=args.re_hi,
        min_branch_points=args.min_branch_points,
        top_branches=args.top_branches,
        dpi=args.dpi,
    )
    print(f"Wrote model-comparison outputs to {out}")
    for f in sorted(out.rglob("*")):
        if f.is_file() and f.suffix in {".png", ".csv", ".json"}:
            print(f"  {f.relative_to(out)}")


if __name__ == "__main__":
    main()
