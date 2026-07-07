#!/usr/bin/env python3
"""3D saddle-landscape comparison: reduced w,u chart vs full z chart.

The full action lives in 4D (w) or 8D (z), so we show:
  1. Local Re Phi slices in the two largest seed coordinates (fixed gamma).
  2. Gamma-continuation ribbons (gamma, Re Phi, Im Phi) for seed + competitors.

Uses matched compare_q3_matched run data by default.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from phasecraft.bm24_saddle_audit_p1.z_robust_workflow import interpolate_z_from_payload
from phasecraft.lib.saddles.picard_lefschetz import compute_phi
from phasecraft.w_saddle.krawczyk_w_saddle_q import WSaddleSystem
from phasecraft.w_saddle.workflow import interpolate_w_seed_from_payload

DEFAULT_RUN = REPO_ROOT / "experiments/w_saddle/runs/compare_q3_matched"
R = 176.54
BETA = 0.5433996420760803
Q = 3


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def nearest_point(payload: dict[str, Any], gamma: float) -> dict[str, Any]:
    pts = payload.get("points") or []
    return min(pts, key=lambda p: abs(float(p["gamma"]) - gamma))


def dominant_index(z: np.ndarray) -> int:
    return int(np.argmax(np.abs(z)))


def vec_from_record(rec: dict[str, Any]) -> np.ndarray:
    if "z_real" in rec:
        return np.asarray(rec["z_real"], float) + 1j * np.asarray(rec["z_imag"], float)
    if "w_real" in rec:
        return np.asarray(rec["w_real"], float) + 1j * np.asarray(rec["w_imag"], float)
    return np.asarray(rec["w_star_real"], float) + 1j * np.asarray(rec["w_star_imag"], float)


def competitors_at_gamma(payload: dict[str, Any], gamma: float, *, tol: float = 0.025) -> list[dict[str, Any]]:
    pt = nearest_point(payload, gamma)
    sweep = pt.get("competitor_sweep") or {}
    roots = sweep.get("certified_roots") or []
    out = []
    for r in roots:
        if r.get("is_seed_branch") or r.get("is_seed_duplicate"):
            continue
        out.append(r)
    return out


def wu_slice_surface(
    gamma: float,
    w_center: np.ndarray,
    idx: int,
    *,
    span_frac: float = 0.45,
    n: int = 55,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sys = WSaddleSystem(r=R, gamma=gamma, q=Q, beta=BETA)
    c = w_center[idx]
    scale = max(span_frac * max(float(np.abs(c)), 0.02), 0.03)
    re_ax = np.linspace(c.real - scale, c.real + scale, n)
    im_ax = np.linspace(c.imag - scale, c.imag + scale, n)
    re_grid, im_grid = np.meshgrid(re_ax, im_ax)
    phi_grid = np.full_like(re_grid, np.nan, dtype=float)
    for i in range(n):
        for j in range(n):
            w = w_center.copy()
            w[idx] = re_grid[i, j] + 1j * im_grid[i, j]
            try:
                phi_grid[i, j] = float(sys.Phi_eff(w).real)
            except (ValueError, FloatingPointError, ZeroDivisionError):
                pass
    return re_grid, im_grid, phi_grid


def z_slice_surface(
    gamma: float,
    z_center: np.ndarray,
    idx: int,
    *,
    span_frac: float = 0.35,
    n: int = 55,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    betas = np.array([BETA])
    gammas = np.array([gamma])
    c = z_center[idx]
    scale = max(span_frac * max(float(np.abs(c)), 0.05), 0.15)
    re_ax = np.linspace(c.real - scale, c.real + scale, n)
    im_ax = np.linspace(c.imag - scale, c.imag + scale, n)
    re_grid, im_grid = np.meshgrid(re_ax, im_ax)
    phi_grid = np.full_like(re_grid, np.nan, dtype=float)
    for i in range(n):
        for j in range(n):
            z = z_center.copy()
            z[idx] = re_grid[i, j] + 1j * im_grid[i, j]
            try:
                phi_grid[i, j] = float(compute_phi(z, q=Q, r=R, betas=betas, gammas=gammas).real)
            except (ValueError, FloatingPointError, ZeroDivisionError):
                pass
    return re_grid, im_grid, phi_grid


def _clip_surface(z: np.ndarray, lo: float, hi: float) -> np.ndarray:
    out = np.array(z, copy=True)
    finite = np.isfinite(out)
    if not np.any(finite):
        return out
    p5, p95 = np.nanpercentile(out[finite], [5, 95])
    cap_lo = max(lo, p5 - 0.35 * (p95 - p5))
    cap_hi = min(hi, p95 + 0.35 * (p95 - p5))
    out = np.clip(out, cap_lo, cap_hi)
    return out


def plot_slice_ax(
    ax,
    re_grid: np.ndarray,
    im_grid: np.ndarray,
    phi_grid: np.ndarray,
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    seed_xy: tuple[float, float],
    competitors: list[tuple[float, float, float]],
) -> None:
    z = _clip_surface(phi_grid, -3.0, 3.0)
    surf = ax.plot_surface(
        re_grid,
        im_grid,
        z,
        cmap=cm.terrain,
        linewidth=0,
        antialiased=True,
        alpha=0.92,
    )
    ax.scatter([seed_xy[0]], [seed_xy[1]], [seed_xy[2]], c="gold", s=60, depthshade=False, label="seed")
    if competitors:
        cx = [c[0] for c in competitors]
        cy = [c[1] for c in competitors]
        cz = [c[2] for c in competitors]
        ax.scatter(cx, cy, cz, c="crimson", s=35, depthshade=False, label="competitors")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_zlabel(r"Re $\Phi$", fontsize=8)
    ax.view_init(elev=28, azim=-58)
    return surf


def branch_ribbon(
    ax,
    branch: dict[str, Any],
    *,
    color: str,
    label: Optional[str] = None,
    max_pts: int = 120,
) -> None:
    pts = sorted(branch.get("points") or [], key=lambda p: float(p["gamma"]), reverse=True)
    if len(pts) > max_pts:
        idx = np.linspace(0, len(pts) - 1, max_pts, dtype=int)
        pts = [pts[i] for i in idx]
    g = np.array([float(p["gamma"]) for p in pts])
    re = np.array([float(p["Phi_eff_real"]) for p in pts])
    im = np.array([float(p.get("Phi_eff_imag_unwrapped", p["Phi_eff_imag"])) for p in pts])
    ax.plot(g, re, im, color=color, lw=1.6, label=label)


def top_competitor_branches(resolved: dict[str, Any], k: int = 4) -> list[dict[str, Any]]:
    comps = [b for b in resolved.get("branches", []) if not b.get("is_seed_sheet")]
    comps.sort(key=lambda b: -int(b.get("num_points", 0)))
    return comps[:k]


def main() -> None:
    p = argparse.ArgumentParser(description="3D w,u vs z saddle landscape comparison")
    p.add_argument("--run-dir", type=str, default=str(DEFAULT_RUN))
    p.add_argument("--gamma", type=float, default=-0.5, help="gamma for local 2D slice")
    p.add_argument("--grid", type=int, default=55)
    p.add_argument("--out", type=str, default="")
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    gamma = float(args.gamma)
    out = Path(args.out) if args.out else run_dir / "saddle_landscape_3d.png"

    wu_seed = load_json(run_dir / "wu/seed_branch_wu.json")
    z_seed = load_json(run_dir / "z/seed_branch_z.json")
    wu_sweep = load_json(run_dir / "wu/competitors_robust_wu.json")
    z_sweep = load_json(run_dir / "z/competitors_robust_z.json")
    wu_res = load_json(run_dir / "wu/resolved_robust_wu.json")
    z_res = load_json(run_dir / "z/resolved_robust_z.json")

    w_center = interpolate_w_seed_from_payload(wu_seed, gamma)
    z_center = interpolate_z_from_payload(z_seed, gamma)
    w_idx = dominant_index(w_center)
    z_idx = dominant_index(z_center)

    wu_re, wu_im, wu_phi = wu_slice_surface(gamma, w_center, w_idx, n=args.grid)
    z_re, z_im, z_phi = z_slice_surface(gamma, z_center, z_idx, n=args.grid)

    sys_w = WSaddleSystem(r=R, gamma=gamma, q=Q, beta=BETA)
    seed_w_phi = float(sys_w.Phi_eff(w_center).real)
    seed_z_phi = float(compute_phi(z_center, q=Q, r=R, betas=[BETA], gammas=[gamma]).real)

    wu_comps: list[tuple[float, float, float]] = []
    for r in competitors_at_gamma(wu_sweep, gamma):
        w = vec_from_record(r)
        wu_comps.append((w[w_idx].real, w[w_idx].imag, float(r["Phi_eff_real"])))

    z_comps: list[tuple[float, float, float]] = []
    for r in competitors_at_gamma(z_sweep, gamma):
        z = vec_from_record(r)
        z_comps.append((z[z_idx].real, z[z_idx].imag, float(r["Phi_eff_real"])))

    fig = plt.figure(figsize=(16, 11))
    fig.suptitle(
        r"BM24 saddle landscapes at $\gamma=%.3f$, $q=3$, $r=%s$, $\beta=%.4f$"
        r"\nTop: local Re $\Phi$ slices (other coordinates fixed at interpolated seed)."
        r"\nBottom: continuation ribbons ($\gamma$, Re $\Phi$, Im $\Phi$); actions not directly comparable across charts."
        % (gamma, R, BETA),
        fontsize=11,
        y=0.98,
    )

    ax1 = fig.add_subplot(2, 2, 1, projection="3d")
    surf1 = plot_slice_ax(
        ax1,
        wu_re,
        wu_im,
        wu_phi,
        title=rf"w,u chart — slice in $(\mathrm{{Re}}\,w_{{{w_idx}}}, \mathrm{{Im}}\,w_{{{w_idx}}})$",
        xlabel=rf"Re $w_{{{w_idx}}}$",
        ylabel=rf"Im $w_{{{w_idx}}}$",
        seed_xy=(w_center[w_idx].real, w_center[w_idx].imag, seed_w_phi),
        competitors=wu_comps,
    )
    ax2 = fig.add_subplot(2, 2, 2, projection="3d")
    surf2 = plot_slice_ax(
        ax2,
        z_re,
        z_im,
        z_phi,
        title=rf"z chart — slice in $(\mathrm{{Re}}\,z_{{{z_idx}}}, \mathrm{{Im}}\,z_{{{z_idx}}})$",
        xlabel=rf"Re $z_{{{z_idx}}}$",
        ylabel=rf"Im $z_{{{z_idx}}}$",
        seed_xy=(z_center[z_idx].real, z_center[z_idx].imag, seed_z_phi),
        competitors=z_comps,
    )

    ax3 = fig.add_subplot(2, 2, 3, projection="3d")
    seed_wu = next(b for b in wu_res["branches"] if b.get("is_seed_sheet"))
    branch_ribbon(ax3, seed_wu, color="gold", label="seed sheet")
    palette = plt.cm.tab10(np.linspace(0, 1, 10))
    for i, br in enumerate(top_competitor_branches(wu_res, k=4)):
        branch_ribbon(ax3, br, color=palette[i], label=f"comp {br['branch_id']}")
    ax3.set_title("w,u continuation ribbons", fontsize=10)
    ax3.set_xlabel(r"$\gamma$")
    ax3.set_ylabel(r"Re $\Phi_{wu}$")
    ax3.set_zlabel(r"Im $\Phi_{wu}$")
    ax3.view_init(elev=22, azim=-62)
    ax3.legend(fontsize=7, loc="upper left")

    ax4 = fig.add_subplot(2, 2, 4, projection="3d")
    seed_z = next(b for b in z_res["branches"] if b.get("is_seed_sheet"))
    branch_ribbon(ax4, seed_z, color="gold", label="seed sheet")
    for i, br in enumerate(top_competitor_branches(z_res, k=4)):
        branch_ribbon(ax4, br, color=palette[i], label=f"comp {br['branch_id']}")
    ax4.set_title("z continuation ribbons", fontsize=10)
    ax4.set_xlabel(r"$\gamma$")
    ax4.set_ylabel(r"Re $\Phi_z$")
    ax4.set_zlabel(r"Im $\Phi_z$")
    ax4.view_init(elev=22, azim=-62)
    ax4.legend(fontsize=7, loc="upper left")

    fig.colorbar(surf1, ax=ax1, shrink=0.55, pad=0.02, label=r"Re $\Phi_{wu}$")
    fig.colorbar(surf2, ax=ax2, shrink=0.55, pad=0.02, label=r"Re $\Phi_z$")
    fig.subplots_adjust(top=0.9, hspace=0.2, wspace=0.15)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
