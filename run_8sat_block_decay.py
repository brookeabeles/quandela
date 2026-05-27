import os
from pathlib import Path

import numpy as np
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

from symmetry_reduction.core import (
    build_structure_matrix,
    compute_b_s,
    compute_c_alpha,
    hessian_F_at_y,
    hessian_log_at_y,
    softmax_weights,
    covariance_from_complex_weights,
)
from symmetry_reduction.mechanisms import energy_by_block
from symmetry_reduction.newton_saddle_q import solve_8sat_saddle
from symmetry_reduction.saddle import saddle_with_adaptive_damping


# ── Parameters ──────────────────────────────────────────────────────────
beta_schedules_8sat = {
    2: np.array([-1.57, -0.78], dtype=float),
    3: np.array([-1.57, -1.04, -0.52], dtype=float),
    4: np.array([-1.57, -1.17, -0.78, -0.39], dtype=float),
}

R_8SAT = 176.54
p_values = (2, 3, 4)  # p=5 is too slow for the q=3 Newton solver

# Row 1: exact q=3 at small γ (Newton u-space)
gamma_row1 = (0.001, 0.003, 0.005, 0.01)

# Rows 2–3: q=1 at large γ (requested γ grid)
gamma_row23 = (np.pi / 4, np.pi, 2 * np.pi)

quick = os.environ.get("QUANDELA_QUICK", "").strip() in {"1", "true", "TRUE", "yes", "YES"}
if quick:
    p_values_row1 = (2, 3)
    gamma_row1 = (0.001, 0.005)
    gamma_row23 = (np.pi / 4, np.pi, 2 * np.pi)
else:
    p_values_row1 = p_values

repo_root = Path(__file__).resolve().parent
out_dir_override = os.environ.get("QUANDELA_8SAT_OUT_DIR", "").strip()
if out_dir_override:
    out_dir = Path(out_dir_override)
    if not out_dir.is_absolute():
        out_dir = repo_root / out_dir
else:
    out_dir = repo_root / "symmetry_reduction" / "figures" / "8sat_block_decay"
out_dir.mkdir(parents=True, exist_ok=True)
summary_path = out_dir / "8sat_universality_summary.png"
summary_path_layers = out_dir / "8sat_universality_summary_layers.png"


def _flag_nonconverged(ax, title: str) -> None:
    for spine in ax.spines.values():
        spine.set_edgecolor("red")
        spine.set_linewidth(3)
    ax.set_title(title + " ⚠", color="red", fontsize=9)


# ── Row 1: exact q=3 Newton saddle at small γ ───────────────────────────
print("=== Row 1: exact q=3 (Newton, r=176.54) at small γ ===")
row1_results: dict[tuple[int, float], dict] = {}
for p in p_values_row1:
    betas = beta_schedules_8sat[p]
    for gv in gamma_row1:
        print(f"  [row1] starting p={p}, γ={float(gv):.3f} ...", flush=True)
        # Fail-fast tuning: some (p,γ) are numerically hard; still plot + flag.
        max_calls = 250
        max_iter = 200
        if quick and float(gv) >= 0.005 and p >= 3:
            max_calls = 10
            max_iter = 60
        y_star, coeff, conv, res = solve_8sat_saddle(
            p=p,
            gamma_target=float(gv),
            betas=betas,
            q=3,
            r=R_8SAT,
            gamma_homotopy_factor=2.0,
            gamma_max_steps=120,
            target_residual=3e-3,
            newton_max_iter=max_iter,
            max_newton_calls=max_calls,
            verbose=False,
        )
        A = build_structure_matrix(p, q=3)
        b_s = compute_b_s(p, betas)
        H = hessian_F_at_y(y_star, A, b_s, coeff)
        _, f_mm = energy_by_block(H, p)
        row1_results[(p, float(gv))] = {"f_mm": f_mm, "converged": bool(conv), "residual": float(res)}
        tag = "OK" if conv else f"res={res:.1e}"
        print(f"  p={p} γ={gv:.3f}: {tag}")


# ── Rows 2–3: q=1 saddle at r=176.54, large γ ──────────────────────────
print("\n=== Rows 2–3: q=1 H_log and Cov_w(A) at r=176.54, large γ ===")
row2_results: dict[tuple[int, float], dict] = {}
row3_results: dict[tuple[int, float], dict] = {}
for p in p_values:
    betas = beta_schedules_8sat[p]
    A = build_structure_matrix(p)
    b_s = compute_b_s(p, betas)
    for gv in gamma_row23:
        gammas = np.full(p, float(gv))
        c_alpha = compute_c_alpha(p, gammas, r=R_8SAT)
        y_star, conv, res = saddle_with_adaptive_damping(A, b_s, c_alpha)

        H_log = hessian_log_at_y(y_star, A, b_s, c_alpha)
        _, f_H = energy_by_block(H_log, p)
        row2_results[(p, float(gv))] = {"f_mm": f_H, "converged": bool(conv), "residual": float(res)}

        sqrt_c = np.sqrt(c_alpha + 0j)
        w = softmax_weights(y_star, A, b_s, sqrt_c)
        Cov = covariance_from_complex_weights(A, w)
        _, f_Cov = energy_by_block(Cov, p)
        row3_results[(p, float(gv))] = {"f_mm": f_Cov, "converged": bool(conv), "residual": float(res)}

        gpi = float(gv / np.pi)
        tag = "OK" if conv else f"res={res:.1e}"
        print(f"  p={p} γ={gpi:.2f}π: {tag}")


# ── Summary figure: 3 rows × N columns, for a single p ─────────────────
p_show = max(p_values)
n_col1 = len(gamma_row1)
n_col23 = len(gamma_row23)
n_cols = max(n_col1, n_col23)

fig, axes = plt.subplots(3, n_cols, figsize=(3.6 * n_cols, 10))
norm = mcolors.PowerNorm(gamma=0.45, vmin=0.0, vmax=1.0)
first_im = None

row_titles = [
    r"Row 1: exact q=3 $\nabla^2 F$ (Newton, small $\gamma$)",
    r"Row 2: q=1 $H_{\log}$ at $r=176.54$ (large $\gamma$)",
    r"Row 3: pure $\mathrm{Cov}_w(A)$ (universal part)",
]

for col in range(n_cols):
    # Row 1
    ax = axes[0, col]
    if col < n_col1:
        gv = float(gamma_row1[col])
        data = row1_results.get((p_show, gv))
        label = rf"$\gamma={gv:.3f}$"
        if data is None:
            ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(label, fontsize=9)
        else:
            im = ax.imshow(data["f_mm"], aspect="auto", cmap="plasma", norm=norm)
            if first_im is None:
                first_im = im
            if not data.get("converged", True):
                _flag_nonconverged(ax, label)
            else:
                ax.set_title(label, fontsize=9)
    else:
        ax.axis("off")
    ax.set_xlabel(r"$|\alpha'|$", fontsize=8)
    ax.set_ylabel(r"$|\alpha|$", fontsize=8)

    # Row 2
    ax = axes[1, col]
    if col < n_col23:
        gv = float(gamma_row23[col])
        data = row2_results.get((p_show, gv))
        label = rf"$\gamma={gv/np.pi:.2f}\pi$"
        if data is None:
            ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(label, fontsize=9)
        else:
            im = ax.imshow(data["f_mm"], aspect="auto", cmap="plasma", norm=norm)
            if first_im is None:
                first_im = im
            if not data.get("converged", True):
                _flag_nonconverged(ax, label)
            else:
                ax.set_title(label, fontsize=9)
    else:
        ax.axis("off")
    ax.set_xlabel(r"$|\alpha'|$", fontsize=8)
    ax.set_ylabel(r"$|\alpha|$", fontsize=8)

    # Row 3
    ax = axes[2, col]
    if col < n_col23:
        gv = float(gamma_row23[col])
        data = row3_results.get((p_show, gv))
        label = rf"$\gamma={gv/np.pi:.2f}\pi$"
        if data is None:
            ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(label, fontsize=9)
        else:
            im = ax.imshow(data["f_mm"], aspect="auto", cmap="plasma", norm=norm)
            if first_im is None:
                first_im = im
            if not data.get("converged", True):
                _flag_nonconverged(ax, label)
            else:
                ax.set_title(label, fontsize=9)
    else:
        ax.axis("off")
    ax.set_xlabel(r"$|\alpha'|$", fontsize=8)
    ax.set_ylabel(r"$|\alpha|$", fontsize=8)

axes[0, 0].set_ylabel(row_titles[0] + "\n" + r"$|\alpha|$", fontsize=8)
axes[1, 0].set_ylabel(row_titles[1] + "\n" + r"$|\alpha|$", fontsize=8)
axes[2, 0].set_ylabel(row_titles[2] + "\n" + r"$|\alpha|$", fontsize=8)

fig.subplots_adjust(right=0.92)
cbar_ax = fig.add_axes([0.94, 0.08, 0.02, 0.82])
if first_im is not None:
    fig.colorbar(first_im, cax=cbar_ax, label=r"Fractional block energy $f_{m,m'}$")

fig.suptitle(
    rf"8-SAT block universality ($p={p_show}$, $r=176.54$): "
    r"exact q=3 (small $\gamma$) vs q=1 (all $\gamma$) vs $\mathrm{Cov}_w(A)$",
    y=1.01,
    fontsize=11,
)
plt.tight_layout(rect=[0, 0, 0.92, 0.98])
fig.savefig(summary_path, dpi=150, bbox_inches="tight")
plt.close(fig)

print(f"\nDone. Universality *layer* summary figure saved to {summary_path_layers}")

# ── New requested plot: p (vertical) × gamma (horizontal), using Row 3 (pure Cov_w(A)) ──
print("Creating p×gamma Cov_w(A) grid (9 panels) ...")

gamma_grid = gamma_row23
norm = mcolors.PowerNorm(gamma=0.45, vmin=0.0, vmax=1.0)
fig2, axes2 = plt.subplots(
    nrows=len(p_values),
    ncols=len(gamma_grid),
    figsize=(3.6 * len(gamma_grid), 3.6 * len(p_values)),
)

first_im = None

def _gamma_label(g: float) -> str:
    if np.isclose(g, np.pi / 4):
        return r"\pi/4"
    if np.isclose(g, np.pi):
        return r"\pi"
    if np.isclose(g, 2 * np.pi):
        return r"2\pi"
    return rf"{(g/np.pi):.2f}\pi"

for i, p in enumerate(p_values):
    for j, gv in enumerate(gamma_grid):
        ax = axes2[i, j]
        data = row3_results.get((p, float(gv)))
        if data is None:
            ax.axis("off")
            continue
        f_mm = data["f_mm"]
        im = ax.imshow(f_mm, aspect="auto", cmap="plasma", norm=norm)
        if first_im is None:
            first_im = im
        title = rf"$p={p}$, $\gamma={_gamma_label(float(gv))}$"
        if not data.get("converged", True):
            for spine in ax.spines.values():
                spine.set_edgecolor("red")
                spine.set_linewidth(3)
            ax.set_title(title + " ⚠", fontsize=9, color="red")
        else:
            ax.set_title(title, fontsize=9)

        # Axis labels only on outer edges to keep the grid readable.
        if i == len(p_values) - 1:
            ax.set_xlabel(r"$|\alpha'|$")
        if j == 0:
            ax.set_ylabel(r"$|\alpha|$")

cbar_ax = fig2.add_axes([0.92, 0.08, 0.02, 0.82])
if first_im is not None:
    fig2.colorbar(first_im, cax=cbar_ax, label=r"Covariance block fractional energy $f_{m,m'}$")

fig2.suptitle(
    rf"8-SAT universality: pure $\mathrm{{Cov}}_w(A)$ (r={R_8SAT})",
    y=1.01,
    fontsize=12,
)
plt.tight_layout(rect=[0, 0, 0.90, 0.98])
fig2.savefig(summary_path, dpi=150, bbox_inches="tight")
plt.close(fig2)

print(f"Done. p×gamma grid saved to {summary_path}")