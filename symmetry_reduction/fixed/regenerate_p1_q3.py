#!/usr/bin/env python3
"""
Regenerate high-value p=1, q=3 (8-SAT) diagnostic plots using corrected BM24 Eq.(20)
(k = 2^q, not 2q) via symmetry_reduction.newton_saddle_q + hessian_F_at_y.

Outputs go to symmetry_reduction/fixed/figures/ (≈8 plots, not a full sweep grid).
"""

from __future__ import annotations

import json
import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from symmetry_reduction.core import (
    alpha_linear_coefficients,
    build_structure_matrix,
    compute_b_s,
    hessian_F_at_y,
    softmax_weights_with_coeff,
)
from symmetry_reduction.diagnostics import softmax_diagnostics
from symmetry_reduction.newton_saddle_q import solve_8sat_saddle
from symmetry_reduction.mechanisms import energy_by_block, block_norms
from symmetry_reduction.spectral import determinant_ratio_probe, spectral_summary

OUT_DIR = Path(__file__).resolve().parent
FIG_DIR = OUT_DIR / "figures"
DATA_DIR = OUT_DIR / "data"

P = 1
Q = 3
R = 176.54
BETA = -np.pi / 2

# Representative panels (π/2 often fails to converge at r=176.54; use 0.5 instead)
PANEL_GAMMAS = [np.pi / 8, np.pi / 4, 0.14, 0.50]
# Robust γ sweep for k99 / y0-vs-saddle curves
SWEEP_GAMMAS = [
    0.05,
    0.10,
    0.20,
    np.pi / 8,
    np.pi / 4,
    0.50,
    np.pi / 2,
    0.14,
    0.75,
    1.00,
]

_WARMSTART: dict[tuple, tuple[float, np.ndarray]] = {}


def _gamma_tex(g: float) -> str:
    """TeX fragment for γ (no surrounding $), for use in rf\"$\\gamma={...}$\"."""
    g_frac = float(g) / np.pi
    for val, lab in (
        (0.25, r"\pi/4"),
        (0.125, r"\pi/8"),
        (0.5, r"\pi/2"),
        (1.0, r"\pi"),
    ):
        if np.isclose(g_frac, val, rtol=0, atol=1e-3):
            return lab
    return rf"{g:.2g}"


def _gamma_label(g: float) -> str:
    return rf"$\gamma={_gamma_tex(g)}$"


def _popcount_array(d: int, n: int) -> np.ndarray:
    return np.array([bin(a).count("1") for a in range(d)], dtype=int)


def run_block_truncation(H: np.ndarray, p: int, gamma: float) -> dict:
    """Block structure + normal/complement truncation (minimal copy of direction2)."""
    n = 2 * p + 1
    E_mm, f_mm = energy_by_block(H, p)
    frob_mm, op_mm = block_norms(H, p)
    d = H.shape[0]
    sizes = _popcount_array(d, n)
    out = {
        "p": p,
        "gamma": gamma,
        "E_mm": E_mm,
        "f_mm": f_mm,
        "frobenius_mm": frob_mm,
        "operator_mm": op_mm,
        "normal": {},
        "complement": {},
    }
    frob_full_sq = float(np.sum(np.abs(H) ** 2))

    def _trunc_metrics(H_sub: np.ndarray) -> dict:
        frob_sub_sq = float(np.sum(np.abs(H_sub) ** 2))
        k0_axis, err_array = determinant_ratio_probe(H_sub)
        return {
            "energy_captured": frob_sub_sq / (frob_full_sq + 1e-30),
            "determinant_ratio_err": float(err_array[0]) if len(err_array) else np.nan,
        }

    for m in range(n + 1):
        normal_mask = sizes <= m
        n_kept = int(np.sum(normal_mask))
        if 0 < n_kept < d:
            idx = np.where(normal_mask)[0]
            out["normal"][m] = {"n_kept": n_kept, **_trunc_metrics(H[np.ix_(idx, idx)])}
        comp_mask = sizes >= (n - m)
        n_kept_c = int(np.sum(comp_mask))
        if 0 < n_kept_c < d:
            idx_c = np.where(comp_mask)[0]
            out["complement"][m] = {"n_kept": n_kept_c, **_trunc_metrics(H[np.ix_(idx_c, idx_c)])}
    return out


def plot_truncation_comparison(results: dict, out_path: Path) -> None:
    p, gamma = results["p"], results["gamma"]
    n = 2 * p + 1
    m_vals = sorted(set(results["normal"]) | set(results["complement"]))
    norm_energy = [results["normal"].get(m, {}).get("energy_captured", np.nan) for m in m_vals]
    comp_energy = [results["complement"].get(m, {}).get("energy_captured", np.nan) for m in m_vals]
    norm_err = [results["normal"].get(m, {}).get("determinant_ratio_err", np.nan) for m in m_vals]
    comp_err = [results["complement"].get(m, {}).get("determinant_ratio_err", np.nan) for m in m_vals]
    fig, axes = plt.subplots(2, 1, figsize=(6, 6), sharex=True)
    ax0, ax1 = axes
    ax0.plot(m_vals, np.array(norm_energy) * 100, "o-", label=r"Normal $|\alpha|\leq m$")
    ax0.plot(m_vals, np.array(comp_energy) * 100, "s-", label=r"Complement $|\alpha|\geq n-m$")
    ax0.set_ylabel("Frobenius energy captured (%)")
    ax0.legend()
    ax0.grid(True, alpha=0.3)
    ax0.set_title(rf"Block truncation (p={p}, $\gamma={_gamma_tex(gamma)}$)")
    ax1.semilogy(m_vals, np.maximum(np.abs(norm_err), 1e-16), "o-", label="Normal")
    ax1.semilogy(m_vals, np.maximum(np.abs(comp_err), 1e-16), "s-", label="Complement")
    ax1.set_xlabel("m")
    ax1.set_ylabel(r"$|R(k_0{=}0)-1|$")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _p_tag() -> str:
    return f"p{P}"


def _sorted_sweep_gammas() -> list[float]:
    """Monotone γ order so warmstart cache is always valid."""
    return sorted({float(g) for g in SWEEP_GAMMAS})


def _homotopy_kwargs(gamma_target: float, *, retry: bool = False) -> dict:
    """γ-homotopy budget; finer steps only where the saddle is stiff (high γ, large p)."""
    if retry:
        return {
            "gamma_homotopy_factor": 1.02,
            "gamma_max_steps": 1200,
            "target_residual": 5e-3,
            "newton_max_iter": 800,
            "max_newton_calls": 50000,
        }
    if P >= 3:
        if gamma_target >= 0.75:
            return {
                "gamma_homotopy_factor": 1.04,
                "gamma_max_steps": 1000,
                "target_residual": 5e-3,
                "newton_max_iter": 700,
                "max_newton_calls": 50000,
            }
        if gamma_target >= np.pi / 8:
            return {
                "gamma_homotopy_factor": 1.10,
                "gamma_max_steps": 600,
                "target_residual": 5e-3,
                "newton_max_iter": 600,
                "max_newton_calls": 35000,
            }
        return {
            "gamma_homotopy_factor": 1.12,
            "gamma_max_steps": 400,
            "target_residual": 5e-3,
            "newton_max_iter": 500,
            "max_newton_calls": 20000,
        }
    if P >= 2:
        if gamma_target >= np.pi / 4:
            return {
                "gamma_homotopy_factor": 1.06,
                "gamma_max_steps": 600,
                "target_residual": 5e-3,
                "newton_max_iter": 500,
                "max_newton_calls": 20000,
            }
        return {
            "gamma_homotopy_factor": 1.10,
            "gamma_max_steps": 400,
            "target_residual": 5e-3,
            "newton_max_iter": 500,
            "max_newton_calls": 12000,
        }
    return {
        "gamma_homotopy_factor": 1.15,
        "gamma_max_steps": 200,
        "target_residual": 5e-3,
        "newton_max_iter": 400,
        "max_newton_calls": 800,
    }


def solve_saddle(
    gamma: float, *, retry: bool = False
) -> tuple[np.ndarray, np.ndarray, bool, float]:
    """Return (y_star, coeff_alpha, converged, residual) for fixed (p,q,r,beta)."""
    betas = np.full(P, BETA, dtype=float)
    cache_key = (P, Q, R, round(BETA, 12))
    u_init = None
    gamma_start = None
    if cache_key in _WARMSTART:
        g_prev, u_prev = _WARMSTART[cache_key]
        if g_prev < gamma:
            u_init = u_prev
            gamma_start = g_prev

    kw = _homotopy_kwargs(float(gamma), retry=retry)
    y_star, coeff, converged, residual = solve_8sat_saddle(
        p=P,
        gamma_target=float(gamma),
        betas=betas,
        q=Q,
        r=R,
        u_init_active=u_init,
        gamma_start_override=gamma_start,
        verbose=False,
        **kw,
    )
    if converged:
        active = np.where(np.array([bin(a).count("1") for a in range(y_star.size)]) >= 2)[0]
        _WARMSTART[cache_key] = (float(gamma), coeff[active] * y_star[active])
    return y_star, coeff, bool(converged), float(residual)


def solve_saddle_with_retry(gamma: float) -> tuple[np.ndarray, np.ndarray, bool, float]:
    """Solve saddle; on failure retry with ultra-conservative homotopy."""
    y, c, ok, res = solve_saddle(gamma)
    if ok:
        return y, c, ok, res
    y, c, ok, res = solve_saddle(gamma, retry=True)
    return y, c, ok, res


def compute_case(gamma: float) -> dict | None:
    """Hessian + spectral diagnostics at y=0 and saddle for one γ."""
    betas = np.full(P, BETA, dtype=float)
    gammas = np.full(P, float(gamma), dtype=float)
    A = build_structure_matrix(P, q=Q)
    b_s = compute_b_s(P, betas)
    d = A.shape[0]
    coeff = alpha_linear_coefficients(P, gammas, q=Q, r=R)

    y0 = np.zeros(d, dtype=complex)
    H0 = hessian_F_at_y(y0, A, b_s, coeff)
    spec0 = spectral_summary(H0)

    y_star, coeff_s, converged, saddle_res = solve_saddle_with_retry(gamma)
    if not converged:
        return {
            "gamma": float(gamma),
            "converged": False,
            "saddle_residual": saddle_res,
            "k99_y0": int(spec0["k_99"]),
            "stable_rank_y0": float(spec0["stable_rank"]),
        }

    Hs = hessian_F_at_y(y_star, A, b_s, coeff_s)
    spec_s = spectral_summary(Hs)
    k0_axis, err_array = determinant_ratio_probe(Hs)
    w = softmax_weights_with_coeff(y_star, A, b_s, coeff_s)
    diag = softmax_diagnostics(w)

    block = run_block_truncation(Hs, P, float(gamma))

    return {
        "gamma": float(gamma),
        "converged": True,
        "saddle_residual": saddle_res,
        "k99_y0": int(spec0["k_99"]),
        "stable_rank_y0": float(spec0["stable_rank"]),
        "k99_saddle": int(spec_s["k_99"]),
        "stable_rank_saddle": float(spec_s["stable_rank"]),
        "singular_values_saddle": spec_s["singular_values"],
        "probe1": {
            "k0_axis": k0_axis,
            "err_array": err_array,
            "k_99": int(spec_s["k_99"]),
            "stable_rank": float(spec_s["stable_rank"]),
        },
        "softmax": diag,
        "block": block,
    }


def plot_block_panels(cases: list[dict], out_path: Path) -> None:
    """1×4 block fractional-energy heatmaps at selected γ."""
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))
    norm = mcolors.PowerNorm(gamma=0.45, vmin=0.0, vmax=1.0)
    im = None
    for ax, case in zip(axes, cases):
        g = case["gamma"]
        if not case.get("converged") or "block" not in case:
            ax.text(0.5, 0.5, f"{_gamma_label(g)}\n(no converge)", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            continue
        f_mm = case["block"]["f_mm"]
        im = ax.imshow(f_mm, aspect="auto", cmap="plasma", norm=norm)
        ax.set_title(rf"$\gamma={g/np.pi:.3f}\pi$, res={case['saddle_residual']:.1e}")
        ax.set_xlabel(r"$|\alpha'|$")
        ax.set_ylabel(r"$|\alpha|$")
    if im is not None:
        fig.subplots_adjust(right=0.88)
        cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
        fig.colorbar(im, cax=cbar_ax, label=r"$f_{m,m'}$")
    fig.suptitle(rf"Block energy $\nabla^2 F$ at saddle (p={P}, q={Q}, r={R}, corrected Eq.20)", y=1.02)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_k99_sweep(sweep: list[dict], out_path: Path) -> None:
    ok = [c for c in sweep if c.get("converged")]
    if not ok:
        return
    x = np.array([c["gamma"] for c in ok]) / np.pi
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    ax1.plot(x, [c["k99_y0"] for c in ok], "o-", label=r"$k_{99}(y=0)$")
    ax1.plot(x, [c["k99_saddle"] for c in ok], "s-", label=r"$k_{99}$(saddle)")
    ax1.set_ylabel(r"$k_{99}$")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax2.plot(x, [c["stable_rank_y0"] for c in ok], "o-", label=r"$r_s(y=0)$")
    ax2.plot(x, [c["stable_rank_saddle"] for c in ok], "s-", label=r"$r_s$(saddle)")
    ax2.set_ylabel("Stable rank")
    ax2.set_xlabel(r"$\gamma/\pi$")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ticks = [0, 0.25, 0.5, 0.75, 1.0]
    ax2.set_xticks(ticks)
    ax2.set_xticklabels(["0", r"$\pi/4$", r"$\pi/2$", r"$3\pi/4$", r"$\pi$"])
    fig.suptitle(rf"$k_{{99}}$ and stable rank vs $\gamma$ (p={P}, q={Q}, corrected saddle)", y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_sv_decay(cases: list[dict], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for case in cases:
        if not case.get("converged"):
            continue
        sig = np.asarray(case["singular_values_saddle"], dtype=float)
        sig = sig[sig > 1e-15]
        if len(sig) == 0:
            continue
        k = np.arange(1, len(sig) + 1)
        ax.loglog(k, sig / sig[0], "o-", markersize=3, label=_gamma_label(case["gamma"]))
    ax.set_xlabel("k")
    ax.set_ylabel(r"$\sigma_k / \sigma_1$")
    ax.set_title(rf"Singular value decay at saddle (p={P}, q={Q})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_hessian_spectrum(cases: list[dict], out_path: Path) -> None:
    """Sorted |λ| of ∇²F at y=0 vs saddle (probe1 is degenerate at p=1)."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
    betas = np.full(P, BETA)
    A = build_structure_matrix(P, q=Q)
    b_s = compute_b_s(P, betas)
    for ax, g in zip(axes, [np.pi / 4, 0.14]):
        case = next((c for c in cases if np.isclose(c["gamma"], g) and c.get("converged")), None)
        if case is None:
            ax.set_title(_gamma_label(g) + " (no converge)")
            continue
        gammas = np.full(P, g)
        coeff = alpha_linear_coefficients(P, gammas, q=Q, r=R)
        y0 = np.zeros(A.shape[0], dtype=complex)
        H0 = hessian_F_at_y(y0, A, b_s, coeff)
        y_star, coeff_s, _, _ = solve_saddle_with_retry(g)
        Hs = hessian_F_at_y(y_star, A, b_s, coeff_s)
        for H, style, lab in ((H0, "o--", r"$y=0$"), (Hs, "s-", "saddle")):
            sig = np.linalg.svd(H, compute_uv=False)
            sig = np.sort(sig)[::-1]
            sig = np.maximum(sig, 1e-20)
            ax.semilogy(np.arange(1, len(sig) + 1), sig / sig[0], style, markersize=5, label=lab)
        ax.set_title(_gamma_label(g))
        ax.set_xlabel("index (sorted)")
        ax.legend()
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel(r"$\sigma_k / \sigma_1$")
    fig.suptitle(rf"Hessian singular values at $y=0$ vs saddle (p={P}, q={Q})", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_softmax(cases: list[dict], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = plt.cm.viridis(np.linspace(0.15, 0.9, len(cases)))
    d = 1 << (2 * P + 1)
    for case, color in zip(cases, colors):
        if not case.get("converged"):
            continue
        # Recompute sorted weights from block run's saddle (stored in block via y_star path)
        g = case["gamma"]
        betas = np.full(P, BETA)
        gammas = np.full(P, g)
        A = build_structure_matrix(P, q=Q)
        b_s = compute_b_s(P, betas)
        y_star, coeff, conv, _ = solve_saddle_with_retry(g)
        if not conv:
            continue
        w = np.abs(softmax_weights_with_coeff(y_star, A, b_s, coeff))
        w_sorted = np.sort(w)[::-1]
        rank = np.arange(1, len(w_sorted) + 1)
        ax.semilogy(rank, np.maximum(w_sorted, 1e-20), "-", color=color, linewidth=2, label=_gamma_label(g))
    ax.axhline(1.0 / d, color="gray", linestyle="--", alpha=0.7, label=rf"$1/d$, $d={d}$")
    ax.set_xlabel("Rank")
    ax.set_ylabel(r"$|w_s|$ (sorted)")
    ax.set_title(rf"Softmax weight concentration at saddle (p={P}, q={Q})")
    ax.set_xlim(1, d)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_convergence(sweep: list[dict], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    for c in sweep:
        x = c["gamma"] / np.pi
        if c.get("converged"):
            ax.semilogy(x, c["saddle_residual"], "o", color="tab:green")
        else:
            ax.semilogy(x, max(c["saddle_residual"], 1e-16), "x", color="tab:red")
    ax.set_xlabel(r"$\gamma/\pi$")
    ax.set_ylabel("Saddle residual $\\|g\\|_\\infty$")
    ax.set_title(f"Saddle solve quality (p={P}, q={Q}, target residual ≤ 5×10⁻³)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    global P
    parser = argparse.ArgumentParser(description="Regenerate fixed q=3 diagnostics for chosen p.")
    parser.add_argument("--p", type=int, default=1, help="Problem size p (default: 1).")
    args = parser.parse_args()
    if args.p < 1:
        raise SystemExit("--p must be >= 1")
    P = int(args.p)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"=== p={P}, q=3 fixed regeneration (corrected k=2^q saddle) ===")
    print(f"Panel gammas: {[float(g) for g in PANEL_GAMMAS]}")
    sweep_order = _sorted_sweep_gammas()
    print(f"Sweep gammas (sorted): {sweep_order}")
    print(f"Homotopy tiers (p≥3): γ<π/8 factor 1.12; π/8≤γ<0.75 factor 1.10; γ≥0.75 factor 1.04; retry 1.02")

    sweep: list[dict] = []
    for g in sweep_order:
        print(f"  γ={g:.4f} ({_gamma_tex(g)}) ...", end=" ", flush=True)
        case = compute_case(g)
        if case is None:
            print("FAILED")
            continue
        sweep.append(case)
        status = "OK" if case.get("converged") else "NO CONV"
        print(f"{status}  res={case.get('saddle_residual', float('nan')):.3e}")

    panel_cases: list[dict] = []
    for g in PANEL_GAMMAS:
        hit = next((c for c in sweep if np.isclose(c["gamma"], g)), None)
        if hit is None:
            hit = compute_case(g)
            if hit:
                sweep.append(hit)
        if hit:
            panel_cases.append(hit)

    # JSON summary (strip heavy arrays)
    summary = {
        "p": P,
        "q": Q,
        "r": R,
        "beta": BETA,
        "equation": "BM24 Eq.(20) with k=2^q; Hessian is nabla^2 F",
        "panel_gammas": [float(g) for g in PANEL_GAMMAS],
        "sweep_gammas": sweep_order,
        "homotopy_tiers": "gamma-dependent; see _homotopy_kwargs in regenerate_p1_q3.py",
        "runs": [
            {
                "gamma": c["gamma"],
                "converged": c.get("converged", False),
                "saddle_residual": c.get("saddle_residual"),
                "k99_y0": c.get("k99_y0"),
                "k99_saddle": c.get("k99_saddle"),
                "stable_rank_saddle": c.get("stable_rank_saddle"),
            }
            for c in sweep
        ],
    }
    summary_path = DATA_DIR / f"regeneration_{_p_tag()}_q3_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {summary_path}")

    # --- Plots (8 total) ---
    plot_block_panels(panel_cases, FIG_DIR / f"block_heatmap_{_p_tag()}_q3.png")
    plot_k99_sweep(sweep, FIG_DIR / f"k99_vs_gamma_{_p_tag()}_q3.png")
    plot_sv_decay(panel_cases, FIG_DIR / f"sv_decay_{_p_tag()}_q3.png")
    plot_hessian_spectrum(panel_cases, FIG_DIR / f"hessian_spectrum_{_p_tag()}_q3.png")
    plot_softmax(panel_cases, FIG_DIR / f"softmax_weights_{_p_tag()}_q3.png")
    plot_convergence(sweep, FIG_DIR / f"saddle_convergence_{_p_tag()}_q3.png")

    # Truncation at π/4 and γ=0.5 (π/2 does not converge at this r)
    for g, tag in ((np.pi / 4, "pi4"), (0.50, "g0p5")):
        case = next((c for c in panel_cases if np.isclose(c["gamma"], g) and c.get("converged")), None)
        if case and "block" in case:
            plot_truncation_comparison(case["block"], FIG_DIR / f"truncation_{_p_tag()}_q3_{tag}.png")

    print(f"Figures in {FIG_DIR}/")
    for p in sorted(FIG_DIR.glob("*.png")):
        print(f"  {p.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
