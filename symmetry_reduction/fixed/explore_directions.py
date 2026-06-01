#!/usr/bin/env python3
"""
Follow-up insights from corrected BM24 (k=2^q) p=1,q=3 diagnostics.

Generates a small set of *direction* plots in symmetry_reduction/fixed/figures/insight_*.png
plus insight_summary.json. Run after regenerate_p1_q3.py or standalone.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from symmetry_reduction.core import (
    alpha_linear_coefficients,
    bm24_clause_arity,
    build_structure_matrix,
    compute_b_s,
    hessian_F_at_y,
    softmax_weights_with_coeff,
)
from symmetry_reduction.diagnostics import softmax_diagnostics
from symmetry_reduction.mechanisms import energy_by_block
from symmetry_reduction.newton_saddle_q import (
    _compute_weights_from_u,
    _get_active_indices,
    active_indices_for_p,
    newton_solve_q,
    solve_8sat_saddle,
)
from symmetry_reduction.spectral import spectral_summary

OUT_DIR = Path(__file__).resolve().parent
FIG_DIR = OUT_DIR / "figures"
DATA_DIR = OUT_DIR / "data"

Q = 3
R = 176.54
BETA = -np.pi / 2

_WARM: dict[tuple, tuple[float, np.ndarray]] = {}


def _gamma_tex(g: float) -> str:
    g_frac = float(g) / np.pi
    for val, lab in ((0.25, r"\pi/4"), (0.125, r"\pi/8"), (0.5, r"\pi/2"), (1.0, r"\pi")):
        if np.isclose(g_frac, val, rtol=0, atol=1e-3):
            return lab
    return rf"{g:.2g}"


def solve_saddle_warm(p: int, gamma: float, *, verbose: bool = False) -> tuple[np.ndarray, np.ndarray, bool, float]:
    """γ-homotopy with warmstart cache (same pattern as regenerate_p1_q3)."""
    betas = np.full(p, BETA, dtype=float)
    cache_key = (p, Q, R, round(BETA, 12))
    u_init = None
    gamma_start = None
    if cache_key in _WARM:
        g_prev, u_prev = _WARM[cache_key]
        if g_prev < gamma:
            u_init = u_prev
            gamma_start = g_prev

    y_star, coeff, converged, residual = solve_8sat_saddle(
        p=p,
        gamma_target=float(gamma),
        betas=betas,
        q=Q,
        r=R,
        gamma_homotopy_factor=1.15,
        gamma_max_steps=200,
        target_residual=5e-3,
        newton_max_iter=400,
        u_init_active=u_init,
        gamma_start_override=gamma_start,
        max_newton_calls=800,
        verbose=verbose,
    )
    if converged:
        active = active_indices_for_p(p)
        _WARM[cache_key] = (float(gamma), coeff[active] * y_star[active])
    return y_star, coeff, bool(converged), float(residual)


def newton_solve_fixed_k(
    u_init: np.ndarray,
    coeff_active: np.ndarray,
    active_idx: np.ndarray,
    A: np.ndarray,
    log_b: np.ndarray,
    d: int,
    k: int,
    tol: float = 1e-10,
    max_iter: int = 200,
) -> tuple[np.ndarray, bool, float, int]:
    """Newton for g = u + k·coeff^k·E^{k-1} with explicit k (correct k=8 or wrong k=6)."""
    exponent = k - 1
    prefactor = float(k)
    deriv_factor = float(k * exponent)
    c_k = np.power(coeff_active, k)
    A_act = A[active_idx]
    n_act = len(active_idx)
    u = np.asarray(u_init, dtype=complex).copy()
    best_u, best_res, stall = u.copy(), float("inf"), 0
    for it in range(max_iter):
        w = _compute_weights_from_u(u, active_idx, A, log_b, d)
        E_A = w @ A_act.T
        g = u + prefactor * c_k * np.power(E_A, exponent)
        res = float(np.max(np.abs(g)))
        if res < best_res:
            best_res, best_u, stall = res, u.copy(), 0
        else:
            stall += 1
        if res < tol:
            return u, True, res, it
        if stall > 30:
            return best_u, best_res < tol, best_res, it
        E_AA = A_act @ (w[:, None] * A_act.T)
        Cov = E_AA - np.outer(E_A, E_A)
        D = deriv_factor * c_k * np.power(E_A, exponent - 1)
        J = np.eye(n_act, dtype=complex) + D[:, None] * Cov
        try:
            delta = np.linalg.solve(J, -g)
        except np.linalg.LinAlgError:
            return best_u, best_res < tol, best_res, it
        step = 1.0
        for _ in range(50):
            u_trial = u + step * delta
            w_t = _compute_weights_from_u(u_trial, active_idx, A, log_b, d)
            E_t = w_t @ A_act.T
            g_t = u_trial + prefactor * c_k * np.power(E_t, exponent)
            if float(np.max(np.abs(g_t))) < res:
                break
            step *= 0.5
        u = u + step * delta
    return best_u, best_res < tol, best_res, max_iter


def solve_saddle_fixed_k(p: int, gamma: float, k: int) -> tuple[np.ndarray, np.ndarray, bool, float]:
    """Homotopy solve with fixed k (for 2q vs 2^q comparison)."""
    betas = np.full(p, BETA)
    A = build_structure_matrix(p, q=Q)
    b_s = compute_b_s(p, betas)
    d = A.shape[0]
    active_idx = _get_active_indices(d)
    n_act = len(active_idx)
    log_b = np.log(b_s + 1e-300 * (1 - np.sign(np.abs(b_s))))
    log_b = np.where(np.isfinite(log_b), log_b, -700.0)

    gt = float(gamma)
    g_cur = 1e-3
    fac = 1.15
    u = np.zeros(n_act, dtype=complex)
    final_res = float("inf")
    for _ in range(200):
        if g_cur >= gt - 1e-20:
            break
        g_next = min(gt, g_cur * fac)
        gammas = np.full(p, g_next)
        coeff = alpha_linear_coefficients(p, gammas, q=Q, r=R)
        u_new, _, res, _ = newton_solve_fixed_k(
            u, coeff[active_idx], active_idx, A, log_b, d, k=k, max_iter=400
        )
        final_res = float(res)
        if np.isfinite(final_res) and final_res <= 5e-3:
            u = u_new
            g_cur = g_next
        else:
            g_next = float(np.sqrt(g_cur * g_next))
            if (g_next - g_cur) / max(g_cur, 1e-30) < 1e-6:
                break

    coeff_final = alpha_linear_coefficients(p, np.full(p, gt), q=Q, r=R)
    y_star = np.zeros(d, dtype=complex)
    y_star[active_idx] = u / coeff_final[active_idx]
    converged = (g_cur >= gt - 1e-20) and final_res <= 5e-3
    return y_star, coeff_final, converged, final_res


def hessian_case(p: int, gamma: float, at_saddle: bool) -> dict:
    betas = np.full(p, BETA)
    A = build_structure_matrix(p, q=Q)
    b_s = compute_b_s(p, betas)
    coeff = alpha_linear_coefficients(p, np.full(p, gamma), q=Q, r=R)
    y0 = np.zeros(A.shape[0], dtype=complex)
    H0 = hessian_F_at_y(y0, A, b_s, coeff)
    spec0 = spectral_summary(H0)
    out = {
        "p": p,
        "gamma": float(gamma),
        "d": int(A.shape[0]),
        "k99_y0": int(spec0["k_99"]),
        "stable_rank_y0": float(spec0["stable_rank"]),
    }
    if not at_saddle:
        _, f_mm = energy_by_block(H0, p)
        out["f_mm_y0"] = f_mm
        return out
    y, coeff_s, conv, res = solve_saddle_warm(p, gamma)
    out["saddle_converged"] = conv
    out["saddle_residual"] = res
    if conv:
        Hs = hessian_F_at_y(y, A, b_s, coeff_s)
        specs = spectral_summary(Hs)
        w = softmax_weights_with_coeff(y, A, b_s, coeff_s)
        sd = softmax_diagnostics(w)
        _, f_mm = energy_by_block(Hs, p)
        out.update(
            k99_saddle=int(specs["k_99"]),
            stable_rank_saddle=float(specs["stable_rank"]),
            entropy=float(sd["entropy"]),
            kappa=float(sd["kappa"]),
            ipr=float(sd["ipr"]),
            f_mm_saddle=f_mm,
        )
    return out


# --- Plotters ---


def plot_rank_collapse_from_sweep(sweep: list[dict], out: Path) -> None:
    ok = [c for c in sweep if c.get("converged")]
    x = np.array([c["gamma"] for c in ok]) / np.pi
    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    ax0, ax1 = axes
    k0 = np.array([c["k99_y0"] for c in ok], dtype=float)
    ks = np.array([c["k99_saddle"] for c in ok], dtype=float)
    ax0.plot(x, k0, "o-", label=r"$k_{99}(y{=}0)$")
    ax0.plot(x, ks, "s-", label=r"$k_{99}$(saddle)")
    ax0.set_ylabel(r"$k_{99}$")
    ax0.legend()
    ax0.grid(True, alpha=0.3)
    ax0.set_title(rf"Rank collapse at saddle (p=1, q=3): always $k_{{99}}=2\to 1$")
    ratio = np.where(k0 > 0, ks / k0, np.nan)
    ax1.plot(x, ratio, "D-", color="tab:purple", label=r"$k_{99}^{saddle}/k_{99}^{y=0}$")
    ax1.axhline(0.5, color="gray", ls=":", alpha=0.6)
    ax1.set_ylabel("Compression ratio")
    ax1.set_xlabel(r"$\gamma/\pi$")
    ax1.set_ylim(0, 1.05)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    fig.suptitle("Direction A: saddle compresses Hessian rank even at minimal p", y=1.01)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_y0_scaling_p(out: Path) -> dict:
    """k_99 at y=0 grows ~ p^α; independent of saddle cost."""
    gammas = [0.14, np.pi / 4, 1.0]
    p_vals = list(range(1, 6))  # p=6 Hessian SVD is slow; p=5 sufficient for scaling
    results = []
    fig, ax = plt.subplots(figsize=(7, 5))
    for g in gammas:
        k99s = []
        for p in p_vals:
            c = hessian_case(p, g, at_saddle=False)
            k99s.append(c["k99_y0"])
            results.append({"p": p, "gamma": float(g), "k99_y0": c["k99_y0"], "d": c["d"]})
        p_arr = np.array(p_vals, dtype=float)
        k_arr = np.array(k99s, dtype=float)
        ax.loglog(p_arr, k_arr, "o-", label=rf"$\gamma={_gamma_tex(g)}$")
        # power-law fit (skip p=1 if degenerate)
        mask = p_arr >= 2
        if mask.sum() >= 2:
            logp = np.log(p_arr[mask])
            logk = np.log(k_arr[mask])
            alpha = float(np.polyfit(logp, logk, 1)[0])
            C = float(np.exp(np.polyfit(logp, logk, 1)[1]))
            p_fit = np.linspace(2, 6, 50)
            ax.loglog(p_fit, C * p_fit**alpha, "--", alpha=0.5, label=rf"$\alpha\approx{alpha:.2f}$")
    ax.set_xlabel(r"$p$")
    ax.set_ylabel(r"$k_{99}$ at $y=0$")
    ax.set_title(rf"$y=0$ Hessian dimension vs $p$ (q={Q}, $r={R}$)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return {"y0_scaling": results}


def plot_wrong_vs_correct_k(out: Path) -> dict:
    """Quantify the pre-fix 2q vs 2^q bug at p=1."""
    p, g = 1, np.pi / 4
    k_correct = bm24_clause_arity(Q)  # 8
    k_wrong = 2 * Q  # 6 — old bug

    y_c, c_c, conv_c, res_c = solve_saddle_warm(p, g)
    y_w, c_w, conv_w, res_w = solve_saddle_fixed_k(p, g, k_wrong)

    betas = np.full(p, BETA)
    A = build_structure_matrix(p, q=Q)
    b_s = compute_b_s(p, betas)

    def _metrics(y, coeff):
        H = hessian_F_at_y(y, A, b_s, coeff)
        sp = spectral_summary(H)
        w = np.abs(softmax_weights_with_coeff(y, A, b_s, coeff))
        return sp["k_99"], sp["stable_rank"], w

    k99_c, rs_c, w_c = _metrics(y_c, c_c)
    k99_w, rs_w, w_w = _metrics(y_w, c_w)
    dist = float(np.linalg.norm(y_c - y_w))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ax0, ax1 = axes
    rank = np.arange(1, len(w_c) + 1)
    ax0.semilogy(rank, np.maximum(np.sort(w_c)[::-1], 1e-20), "o-", label=rf"correct $k={k_correct}$")
    ax0.semilogy(rank, np.maximum(np.sort(w_w)[::-1], 1e-20), "s-", label=rf"wrong $k={k_wrong}$ ($2q$ bug)")
    ax0.set_xlabel("Rank")
    ax0.set_ylabel(r"$|w_s|$")
    ax0.legend()
    ax0.grid(True, alpha=0.3)
    ax0.set_title(rf"Softmax weights at $\gamma={_gamma_tex(g)}$")

    labels = [rf"correct\n$k={k_correct}$", rf"wrong\n$k={k_wrong}$"]
    k99_vals = [k99_c, k99_w]
    colors = ["tab:green", "tab:red"]
    ax1.bar(labels, k99_vals, color=colors, alpha=0.85)
    ax1.set_ylabel(r"$k_{99}$ at saddle")
    ax1.set_title(rf"$\|y_{{\mathrm{{correct}}}}-y_{{\mathrm{{wrong}}}}\|_2={dist:.2e}$")
    fig.suptitle("Direction B: wrong exponent 2q moves the saddle and Hessian rank", y=1.02)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {
        "p": p,
        "gamma": float(g),
        "k_correct": k_correct,
        "k_wrong": k_wrong,
        "conv_correct": conv_c,
        "conv_wrong": conv_w,
        "res_correct": res_c,
        "res_wrong": res_w,
        "k99_correct": int(k99_c),
        "k99_wrong": int(k99_w),
        "y_distance": dist,
    }


def plot_p12_saddle_collapse(out: Path, sweep_p1: list[dict]) -> list[dict]:
    """Rank compression: p=1 from saved sweep; p=2 on a short warmstarted chain."""
    rows = list(sweep_p1)
    fig, ax = plt.subplots(figsize=(7, 5))

    def _plot_series(p: int, cases: list[dict]) -> None:
        ok = [c for c in cases if c.get("saddle_converged") or c.get("converged")]
        if not ok:
            return
        k0 = np.array([c["k99_y0"] for c in ok], dtype=float)
        ks = np.array([c.get("k99_saddle") for c in ok], dtype=float)
        x = np.array([c["gamma"] for c in ok]) / np.pi
        ax.plot(x, ks / k0, "o-", label=rf"$p={p}$")

    _plot_series(1, sweep_p1)

    # p=2: only 4 γ values, strictly increasing (warmstart)
    p2_cases = []
    for g in [0.05, 0.14, np.pi / 4, 0.5]:
        c = hessian_case(2, g, at_saddle=True)
        p2_cases.append(c)
        rows.append(c)
    _plot_series(2, p2_cases)
    ax.axhline(0.5, color="gray", ls=":", alpha=0.5)
    ax.set_xlabel(r"$\gamma/\pi$")
    ax.set_ylabel(r"$k_{99}^{saddle}/k_{99}^{y=0}$")
    ax.set_title("Saddle rank compression (warmstarted γ sweep)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return rows


def plot_block_y0_vs_p(out: Path) -> None:
    """Block heatmaps at y=0 for p=1,2,3 show when off-diagonal structure appears."""
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))
    g = np.pi / 4
    norm = plt.cm.plasma
    for ax, p in zip(axes, [1, 2, 3]):
        c = hessian_case(p, g, at_saddle=False)
        f = c["f_mm_y0"]
        im = ax.imshow(f, aspect="auto", cmap="plasma", vmin=0, vmax=max(0.3, f.max()))
        ax.set_title(rf"$p={p}$, $d={c['d']}$, $k_{{99}}={c['k99_y0']}$")
        ax.set_xlabel(r"$|\alpha'|$")
        ax.set_ylabel(r"$|\alpha|$")
    fig.suptitle(rf"Block energy of $\nabla^2 F$ at $y=0$, $\gamma={_gamma_tex(g)}$", y=1.02)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_convergence_phase(out: Path, sweep_p1: list[dict]) -> None:
    """Where does γ-homotopy fail? p=1 from sweep; p=2 spot-check at π/2."""
    p_vals = [1, 2]
    gammas = sorted({c["gamma"] for c in sweep_p1})
    mat = np.full((len(p_vals), len(gammas)), np.nan)
    for j, g in enumerate(gammas):
        hit = next((c for c in sweep_p1 if np.isclose(c["gamma"], g)), None)
        if hit:
            mat[0, j] = np.log10(max(hit["saddle_residual"], 1e-16))
    # p=2: only check the fragile point γ=π/2 (others assumed similar to p=1)
    j_pi2 = next((j for j, g in enumerate(gammas) if np.isclose(g, np.pi / 2)), None)
    if j_pi2 is not None:
        _, _, conv, res = solve_saddle_warm(2, np.pi / 2)
        mat[1, j_pi2] = np.log10(max(res, 1e-16))
    # fill p=2 row at γ=0.14 with one warmstarted solve
    j014 = next((j for j, g in enumerate(gammas) if np.isclose(g, 0.14)), None)
    if j014 is not None:
        _, _, _, res = solve_saddle_warm(2, 0.14)
        mat[1, j014] = np.log10(max(res, 1e-16))
    fig, ax = plt.subplots(figsize=(10, 2.8))
    im = ax.imshow(mat, aspect="auto", cmap="RdYlGn_r", vmin=-8, vmax=-1)
    ax.set_yticks(range(len(p_vals)), [f"p={p}" for p in p_vals])
    ax.set_xticks(range(len(gammas)), [f"{g/np.pi:.2f}π" for g in gammas], rotation=45, ha="right")
    ax.set_title(r"$\log_{10}$ saddle residual (green=converged $\leq10^{-2.5}$)")
    fig.colorbar(im, ax=ax, label=r"$\log_{10}|g|_\infty$")
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    summary_path = OUT_DIR / "data" / "regeneration_p1_q3_summary.json"
    if summary_path.exists():
        with summary_path.open(encoding="utf-8") as f:
            sweep = json.load(f).get("runs", [])
    else:
        sweep = []

    print("=== Insight plots (corrected BM24) ===\n", flush=True)

    if sweep:
        plot_rank_collapse_from_sweep(sweep, FIG_DIR / "insight_rank_collapse_gamma.png")
        print("  insight_rank_collapse_gamma.png")

    t0 = time.time()
    y0_data = plot_y0_scaling_p(FIG_DIR / "insight_k99_y0_vs_p.png")
    print(f"  insight_k99_y0_vs_p.png ({time.time()-t0:.1f}s)")

    plot_block_y0_vs_p(FIG_DIR / "insight_block_y0_p123.png")
    print("  insight_block_y0_p123.png")

    t0 = time.time()
    wrong = plot_wrong_vs_correct_k(FIG_DIR / "insight_wrong_vs_correct_k.png")
    print(f"  insight_wrong_vs_correct_k.png ({time.time()-t0:.1f}s)")

    t0 = time.time()
    p12 = plot_p12_saddle_collapse(FIG_DIR / "insight_saddle_collapse_p12.png", sweep)
    print(f"  insight_saddle_collapse_p12.png ({time.time()-t0:.1f}s)", flush=True)

    t0 = time.time()
    plot_convergence_phase(FIG_DIR / "insight_convergence_phase.png", sweep)
    print(f"  insight_convergence_phase.png ({time.time()-t0:.1f}s)", flush=True)

    insight = {
        "directions": [
            "A: universal rank collapse at saddle (even p=1)",
            "B: 2q vs 2^q bug moves saddle and diagnostics",
            "C: y=0 k99 scales ~ p^alpha (power law before saddle)",
            "D: convergence boundary in (p, gamma) — pi/2 fragile",
        ],
        "wrong_vs_correct": wrong,
        "p12_saddle": [
            {k: v for k, v in r.items() if k != "f_mm_saddle" and k != "f_mm_y0"}
            for r in p12
        ],
        **y0_data,
    }
    out_json = DATA_DIR / "insight_summary.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(insight, f, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else o)
    print(f"\nWrote {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
