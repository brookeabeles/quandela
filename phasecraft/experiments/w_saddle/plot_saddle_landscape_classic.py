#!/usr/bin/env python3
"""Classic 2D optimization-landscape views: w,u chart vs z chart.

For fixed gamma, slice the dominant complex coordinate (others held at the
interpolated seed) and plot:

  Row 1 — log10 |F|^2  (saddle-equation residual; zeros = critical points)
  Row 2 — Re Phi       (effective action height)
  Row 3 — 1D cross-sections through the seed along Re / Im axes

Uses matched compare_q3_matched data by default.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from phasecraft.bm24_saddle_audit_p1.z_robust_workflow import interpolate_z_from_payload
from phasecraft.lib.saddles.krawczyk_p1_roots import SaddleSystem
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


def competitors_at_gamma(payload: dict[str, Any], gamma: float) -> list[dict[str, Any]]:
    pt = nearest_point(payload, gamma)
    sweep = pt.get("competitor_sweep") or {}
    return [
        r
        for r in (sweep.get("certified_roots") or [])
        if not r.get("is_seed_branch") and not r.get("is_seed_duplicate")
    ]


def eval_slice(
    gamma: float,
    center: np.ndarray,
    idx: int,
    *,
    chart: str,
    span_frac: float,
    n: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    c = center[idx]
    scale = max(span_frac * max(float(np.abs(c)), 0.02 if chart == "wu" else 0.05), 0.03 if chart == "wu" else 0.15)
    re_ax = np.linspace(c.real - scale, c.real + scale, n)
    im_ax = np.linspace(c.imag - scale, c.imag + scale, n)
    re_grid, im_grid = np.meshgrid(re_ax, im_ax)
    log_res = np.full_like(re_grid, np.nan)
    re_phi = np.full_like(re_grid, np.nan)

    sys_w = WSaddleSystem(r=R, gamma=gamma, q=Q, beta=BETA) if chart == "wu" else None
    betas = np.array([BETA])
    gammas = np.array([gamma])

    for i in range(n):
        for j in range(n):
            v = center.copy()
            v[idx] = re_grid[i, j] + 1j * im_grid[i, j]
            try:
                if chart == "wu":
                    assert sys_w is not None
                    f = sys_w.F_complex(v)
                    log_res[i, j] = np.log10(float(np.sum(np.abs(f) ** 2)) + 1e-30)
                    re_phi[i, j] = float(sys_w.Phi_eff(v).real)
                else:
                    sys_z = SaddleSystem.build(q=Q, r=R, betas=betas, gammas=gammas)
                    g = sys_z.G_complex(v)
                    f = v - g
                    log_res[i, j] = np.log10(float(np.sum(np.abs(f) ** 2)) + 1e-30)
                    re_phi[i, j] = float(compute_phi(v, q=Q, r=R, betas=betas, gammas=gammas).real)
            except (ValueError, FloatingPointError, ZeroDivisionError):
                pass
    return re_grid, im_grid, log_res, re_phi


def _finite_percentile(z: np.ndarray, q: float) -> float:
    vals = z[np.isfinite(z)]
    if vals.size == 0:
        return float("nan")
    return float(np.percentile(vals, q))


def _in_bounds(xy: tuple[float, float], re_grid: np.ndarray, im_grid: np.ndarray) -> bool:
    return (
        float(re_grid.min()) <= xy[0] <= float(re_grid.max())
        and float(im_grid.min()) <= xy[1] <= float(im_grid.max())
    )


def plot_contour_panel(
    ax,
    re_grid: np.ndarray,
    im_grid: np.ndarray,
    z: np.ndarray,
    *,
    title: str,
    cmap: str,
    cbar_label: str,
    vmin: float | None = None,
    vmax: float | None = None,
    seed_xy: tuple[float, float],
    comp_xy: list[tuple[float, float]],
) -> None:
    z_plot = np.array(z, copy=True)
    finite = np.isfinite(z_plot)
    if not np.any(finite):
        ax.set_title(title + " (no data)")
        return
    if vmin is None:
        vmin = _finite_percentile(z_plot, 2)
    if vmax is None:
        vmax = _finite_percentile(z_plot, 98)
    if vmin >= vmax:
        vmax = vmin + 1e-6

    cf = ax.contourf(re_grid, im_grid, z_plot, levels=24, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.contour(re_grid, im_grid, z_plot, levels=12, colors="k", linewidths=0.35, alpha=0.45)
    ax.plot(seed_xy[0], seed_xy[1], "*", color="gold", ms=14, mec="k", mew=0.7, zorder=5, label="seed")
    in_slice = [xy for xy in comp_xy if _in_bounds(xy, re_grid, im_grid)]
    if in_slice:
        cx, cy = zip(*in_slice)
        ax.plot(cx, cy, "x", color="crimson", ms=7, mew=1.2, zorder=5, label=f"competitors ({len(in_slice)})")
    off = len(comp_xy) - len(in_slice)
    if off:
        ax.text(
            0.02,
            0.02,
            f"{off} certified root(s) off-slice",
            transform=ax.transAxes,
            fontsize=7,
            color="0.35",
            va="bottom",
        )
    ax.set_xlim(float(re_grid.min()), float(re_grid.max()))
    ax.set_ylim(float(im_grid.min()), float(im_grid.max()))
    ax.set_title(title, fontsize=10)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(fontsize=7, loc="upper right")
    plt.colorbar(cf, ax=ax, fraction=0.046, pad=0.04, label=cbar_label)


def cross_section(
    gamma: float,
    center: np.ndarray,
    idx: int,
    *,
    chart: str,
    half_width: float,
    n: int,
    axis: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """1D slice through seed: axis is 're' or 'im'."""
    c = center[idx]
    t = np.linspace(-half_width, half_width, n)
    log_res = np.zeros(n)
    re_phi = np.zeros(n)
    for k, dt in enumerate(t):
        v = center.copy()
        if axis == "re":
            v[idx] = (c.real + dt) + 1j * c.imag
            coord = c.real + dt
        else:
            v[idx] = c.real + 1j * (c.imag + dt)
            coord = c.imag + dt
        try:
            if chart == "wu":
                sys = WSaddleSystem(r=R, gamma=gamma, q=Q, beta=BETA)
                f = sys.F_complex(v)
                log_res[k] = np.log10(float(np.sum(np.abs(f) ** 2)) + 1e-30)
                re_phi[k] = float(sys.Phi_eff(v).real)
            else:
                sys_z = SaddleSystem.build(q=Q, r=R, betas=np.array([BETA]), gammas=np.array([gamma]))
                f = v - sys_z.G_complex(v)
                log_res[k] = np.log10(float(np.sum(np.abs(f) ** 2)) + 1e-30)
                re_phi[k] = float(compute_phi(v, q=Q, r=R, betas=[BETA], gammas=[gamma]).real)
        except (ValueError, FloatingPointError, ZeroDivisionError):
            log_res[k] = np.nan
            re_phi[k] = np.nan
        if axis == "re":
            pass
    coords = c.real + t if axis == "re" else c.imag + t
    return coords, log_res, re_phi


def main() -> None:
    p = argparse.ArgumentParser(description="Classic 2D saddle / optimization landscapes")
    p.add_argument("--run-dir", type=str, default=str(DEFAULT_RUN))
    p.add_argument("--gamma", type=float, default=-0.5)
    p.add_argument("--grid", type=int, default=80)
    p.add_argument("--span-frac", type=float, default=0.45)
    p.add_argument("--out", type=str, default="")
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    gamma = float(args.gamma)
    out = Path(args.out) if args.out else run_dir / "saddle_landscape_classic.png"

    wu_seed = load_json(run_dir / "wu/seed_branch_wu.json")
    z_seed = load_json(run_dir / "z/seed_branch_z.json")
    wu_sweep = load_json(run_dir / "wu/competitors_robust_wu.json")
    z_sweep = load_json(run_dir / "z/competitors_robust_z.json")

    w_center = interpolate_w_seed_from_payload(wu_seed, gamma)
    z_center = interpolate_z_from_payload(z_seed, gamma)
    w_idx = dominant_index(w_center)
    z_idx = dominant_index(z_center)

    wu_re, wu_im, wu_logf, wu_phi = eval_slice(
        gamma, w_center, w_idx, chart="wu", span_frac=args.span_frac, n=args.grid
    )
    z_re, z_im, z_logf, z_phi = eval_slice(
        gamma, z_center, z_idx, chart="z", span_frac=args.span_frac, n=args.grid
    )

    wu_comp_xy = [(vec_from_record(r)[w_idx].real, vec_from_record(r)[w_idx].imag) for r in competitors_at_gamma(wu_sweep, gamma)]
    z_comp_xy = [(vec_from_record(r)[z_idx].real, vec_from_record(r)[z_idx].imag) for r in competitors_at_gamma(z_sweep, gamma)]

    seed_wu_xy = (w_center[w_idx].real, w_center[w_idx].imag)
    seed_z_xy = (z_center[z_idx].real, z_center[z_idx].imag)

    half_w = float(wu_re[0, -1] - wu_re[0, 0]) * 0.5
    half_z = float(z_re[0, -1] - z_re[0, 0]) * 0.5

    fig = plt.figure(figsize=(14, 13))
    fig.suptitle(
        r"Classic optimization landscapes at $\gamma=%.3f$  ($q=3$, $r=%.2f$, $\beta=%.4f$)"
        "\nSlice: dominant coordinate; all others fixed at interpolated seed."
        % (gamma, R, BETA),
        fontsize=12,
        y=0.98,
    )
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.0, 0.85], hspace=0.32, wspace=0.22)

    ax = fig.add_subplot(gs[0, 0])
    plot_contour_panel(
        ax,
        wu_re,
        wu_im,
        wu_logf,
        title=rf"w,u — $\log_{{10}}|F(w)|^2$  (slice $w_{{{w_idx}}}$)",
        cmap="magma_r",
        cbar_label=r"$\log_{10}|F|^2$",
        seed_xy=seed_wu_xy,
        comp_xy=wu_comp_xy,
    )
    ax.set_xlabel(rf"Re $w_{{{w_idx}}}$")
    ax.set_ylabel(rf"Im $w_{{{w_idx}}}$")

    ax = fig.add_subplot(gs[0, 1])
    plot_contour_panel(
        ax,
        z_re,
        z_im,
        z_logf,
        title=rf"z — $\log_{{10}}|F(z)|^2$  (slice $z_{{{z_idx}}}$)",
        cmap="magma_r",
        cbar_label=r"$\log_{10}|F|^2$",
        seed_xy=seed_z_xy,
        comp_xy=z_comp_xy,
    )
    ax.set_xlabel(rf"Re $z_{{{z_idx}}}$")
    ax.set_ylabel(rf"Im $z_{{{z_idx}}}$")

    ax = fig.add_subplot(gs[1, 0])
    plot_contour_panel(
        ax,
        wu_re,
        wu_im,
        wu_phi,
        title=rf"w,u — Re $\Phi_{{wu}}$",
        cmap="viridis",
        cbar_label=r"Re $\Phi_{wu}$",
        seed_xy=seed_wu_xy,
        comp_xy=wu_comp_xy,
    )
    ax.set_xlabel(rf"Re $w_{{{w_idx}}}$")
    ax.set_ylabel(rf"Im $w_{{{w_idx}}}$")

    ax = fig.add_subplot(gs[1, 1])
    plot_contour_panel(
        ax,
        z_re,
        z_im,
        z_phi,
        title=rf"z — Re $\Phi_z$",
        cmap="viridis",
        cbar_label=r"Re $\Phi_z$",
        seed_xy=seed_z_xy,
        comp_xy=z_comp_xy,
    )
    ax.set_xlabel(rf"Re $z_{{{z_idx}}}$")
    ax.set_ylabel(rf"Im $z_{{{z_idx}}}$")

    for col, (chart, center, idx, half, coord_tag) in enumerate(
        [
            ("wu", w_center, w_idx, half_w, rf"w_{{{w_idx}}}"),
            ("z", z_center, z_idx, half_z, rf"z_{{{z_idx}}}"),
        ]
    ):
        ax = fig.add_subplot(gs[2, col])
        for axis, ls in (("re", "-"), ("im", "--")):
            coords, logf, _ = cross_section(gamma, center, idx, chart=chart, half_width=half, n=200, axis=axis)
            label = rf"$\mathrm{{Re}}\,{coord_tag}$" if axis == "re" else rf"$\mathrm{{Im}}\,{coord_tag}$"
            ax.plot(coords, logf, ls, lw=1.5, label=label)
        ax.axvline(center[idx].real, color="gold", ls=":", lw=0.9, alpha=0.8)
        ax.axvline(center[idx].imag, color="gray", ls=":", lw=0.9, alpha=0.8)
        ax.set_title(rf"{'w,u' if chart == 'wu' else 'z'} — 1D $\log_{{10}}|F|^2$ through seed", fontsize=10)
        ax.set_xlabel("coordinate value")
        ax.set_ylabel(r"$\log_{10}|F|^2$")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.25)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
