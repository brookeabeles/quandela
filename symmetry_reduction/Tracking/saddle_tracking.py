"""
Saddle path continuation and Stokes-strip visualisation in the θ-plane.

We work with the BM24 QAOA phase Φ(θ; γ̃, β) and its derivatives from `qaoa_core`:
    - Φ(θ, gt, beta)
    - dPhi(θ, gt, beta)
    - d2Phi(θ, gt, beta)
    - all_saddles(gt, beta, ...)

Notation:
    gt   ≡ γ̃ (the scalar phase parameter used in the stokes_transitions_v3 module),
    θ    ∈ ℂ is the saddle coordinate in the complex plane.

This module implements:
    Step A – Path continuation of a chosen saddle branch as gt increases.
    Step B – Identification and backward continuation of a rival saddle branch.
    Step C – 2D complex-plane plots of Re Φ "strips", gradient field (thimbles),
             and the two tracked saddles, at three values of gt around a
             Stokes crossing where Re Φ(primary) ≈ Re Φ(rival).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from qaoa_core import (  # type: ignore
    DEFAULT_BETA,
    Phi,
    all_saddles,
    d2Phi,
    dPhi,
)


Complex = complex


@dataclass
class SaddleBranch:
    """A one-dimensional saddle branch θ(gt) sampled on a grid of gt values."""

    gt: np.ndarray  # shape (N,)
    theta: np.ndarray  # shape (N,), complex

    def rePhi(self, beta: float = DEFAULT_BETA) -> np.ndarray:
        """Re Φ(θ(gt); gt) evaluated pointwise along the branch."""
        return np.array(
            [Phi(th, g, beta).real for th, g in zip(self.theta, self.gt)],
            dtype=float,
        )


def newton_solve_saddle(
    theta0: Complex,
    gt: float,
    beta: float = DEFAULT_BETA,
    max_iter: int = 80,
    tol: float = 1e-12,
    max_step: float = 1.0,
) -> Tuple[Complex, bool]:
    """
    Newton solve for a saddle of Φ at fixed (gt, beta), starting from theta0.

    Solves dPhi(θ, gt, beta) = 0. Returns (theta_star, converged).
    """
    th = complex(theta0)
    for _ in range(max_iter):
        f = dPhi(th, gt, beta)
        if abs(f) < tol:
            return th, True
        fp = d2Phi(th, gt, beta)
        if abs(fp) < 1e-16:
            return th, False
        dth = -f / fp
        if abs(dth) > max_step:
            dth = max_step * dth / abs(dth)
        th = th + dth
    return th, False


def continue_saddle_branch(
    gt_values: Sequence[float],
    theta_start: Complex,
    beta: float = DEFAULT_BETA,
    max_step_newton: float = 1.0,
) -> SaddleBranch:
    """
    Continue a saddle along a prescribed list of gt values.

    Parameters
    ----------
    gt_values:
        Monotone sequence of γ̃ values (increasing or decreasing).
    theta_start:
        Initial saddle guess at gt_values[0]; will be refined by Newton.
    beta:
        Fixed QAOA mixing angle β.

    Returns
    -------
    SaddleBranch
        Branch θ(gt) on the provided grid; entries where Newton fails are NaN.
    """
    gt_arr = np.asarray(gt_values, dtype=float)
    theta_arr = np.full_like(gt_arr, np.nan + 1j * np.nan, dtype=complex)

    # First point: refine starting guess.
    th, ok = newton_solve_saddle(theta_start, float(gt_arr[0]), beta=beta, max_step=max_step_newton)
    if not ok:
        return SaddleBranch(gt_arr, theta_arr)
    theta_arr[0] = th

    # Subsequent points: use previous saddle as initial guess.
    for i in range(1, len(gt_arr)):
        gt = float(gt_arr[i])
        th0 = theta_arr[i - 1]
        if not np.isfinite(th0.real) or not np.isfinite(th0.imag):
            break
        th, ok = newton_solve_saddle(th0, gt, beta=beta, max_step=max_step_newton)
        if not ok:
            break
        theta_arr[i] = th

    return SaddleBranch(gt_arr, theta_arr)


def _dominant_saddle_at(
    gt: float,
    beta: float = DEFAULT_BETA,
    grid_size: int = 50,
    half_width: float = 10.0,
) -> Complex | None:
    """
    Heuristic: choose the saddle with the largest Re Φ(θ; gt) among all_saddles.
    """
    lst = all_saddles(
        gt,
        beta=beta,
        grid_size=grid_size,
        half_width=half_width,
        tol_dedup=1e-6,
        n_saddles_max=40,
    )
    if not lst:
        return None
    saddles = [th for (th, _) in lst]
    rephis = [Phi(th, gt, beta).real for th in saddles]
    idx = int(np.argmax(rephis))
    return saddles[idx]


def _rival_saddle_at(
    gt: float,
    primary_theta: Complex,
    beta: float = DEFAULT_BETA,
    grid_size: int = 60,
    half_width: float = 12.0,
    min_separation: float = 1e-3,
) -> Complex | None:
    """
    Choose a 'rival' saddle at gt that is distinct from primary_theta.

    Strategy: among all saddles with |θ − θ_primary| ≥ min_separation, pick
    the one whose Re Φ is closest to the primary Re Φ (to capture a Stokes rival).
    """
    lst = all_saddles(
        gt,
        beta=beta,
        grid_size=grid_size,
        half_width=half_width,
        tol_dedup=1e-6,
        n_saddles_max=80,
    )
    if not lst:
        return None
    theta_primary = complex(primary_theta)
    cand: List[Tuple[Complex, float]] = []
    re_primary = Phi(theta_primary, gt, beta).real
    for th, _ in lst:
        if abs(th - theta_primary) < min_separation:
            continue
        rephi = Phi(th, gt, beta).real
        cand.append((th, abs(rephi - re_primary)))
    if not cand:
        return None
    cand.sort(key=lambda x: x[1])
    return cand[0][0]


def track_primary_and_rival_branches(
    gt_start: float,
    gt_end: float,
    dgt: float,
    gt_high_for_rival: float | None = None,
    beta: float = DEFAULT_BETA,
) -> Tuple[SaddleBranch, SaddleBranch | None]:
    """
    Step A + B: track a primary saddle forward and a rival saddle backward.

    - Primary branch: start at gt_start using the dominant saddle at this γ̃,
      and continue forward to gt_end in steps of dgt.
    - Rival branch: pick gt_high_for_rival (default: gt_end), find a saddle
      that competes with the primary one at that γ̃, and continue it backward
      to gt_start in steps of -dgt.

    Returns
    -------
    primary : SaddleBranch
    rival   : SaddleBranch | None
        None if a rival saddle could not be identified or continued.
    """
    if dgt <= 0:
        raise ValueError("dgt must be positive.")

    # Step A: primary branch forward.
    theta0 = _dominant_saddle_at(gt_start, beta=beta)
    if theta0 is None:
        raise RuntimeError(f"No saddle found at gt_start={gt_start!r}")
    gt_forward = np.arange(gt_start, gt_end + 0.5 * dgt, dgt, dtype=float)
    primary = continue_saddle_branch(gt_forward, theta0, beta=beta)

    # Step B: rival branch backwards from high γ̃.
    gt_hi = float(gt_high_for_rival if gt_high_for_rival is not None else gt_end)
    # Find primary at gt_hi by interpolation then one Newton refinement.
    th_primary_hi = np.interp(gt_hi, primary.gt.real, primary.theta.real) + 1j * np.interp(
        gt_hi, primary.gt.real, primary.theta.imag
    )
    th_primary_hi, ok = newton_solve_saddle(th_primary_hi, gt_hi, beta=beta)
    if not ok:
        return primary, None

    rival0 = _rival_saddle_at(gt_hi, th_primary_hi, beta=beta)
    if rival0 is None:
        return primary, None

    gt_backward = np.arange(gt_hi, gt_start - 0.5 * dgt, -dgt, dtype=float)
    rival = continue_saddle_branch(gt_backward, rival0, beta=beta)
    return primary, rival


def find_stokes_crossing(
    branch1: SaddleBranch,
    branch2: SaddleBranch,
    beta: float = DEFAULT_BETA,
) -> float | None:
    """
    Locate a γ̃ where Re Φ on two branches cross (Stokes line).

    We interpolate branch2 onto branch1.gt and then look for sign changes in
    Re Φ(branch1) − Re Φ(branch2). Returns an approximate crossing γ̃_c or
    None if no sign change is found.
    """
    gt1 = branch1.gt
    gt2 = branch2.gt
    if len(gt1) == 0 or len(gt2) == 0:
        return None

    re1 = branch1.rePhi(beta=beta)
    # Interpolate rival Re Φ onto primary γ̃ grid
    re2_on_1 = np.interp(gt1, gt2, branch2.rePhi(beta=beta))
    diff = re1 - re2_on_1
    for i in range(1, len(diff)):
        if not np.isfinite(diff[i - 1]) or not np.isfinite(diff[i]):
            continue
        if diff[i - 1] == 0:
            return float(gt1[i - 1])
        if diff[i - 1] * diff[i] < 0:
            # Linear interpolation for zero of diff.
            g_lo, g_hi = gt1[i - 1], gt1[i]
            d_lo, d_hi = diff[i - 1], diff[i]
            t = float(-d_lo / (d_hi - d_lo))
            return float(g_lo + t * (g_hi - g_lo))
    return None


def _grid_around_points(
    points: Iterable[Complex],
    radius: float = 3.0,
    n: int = 300,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a square grid in the θ-plane that comfortably contains all points.
    """
    pts = list(points)
    if not pts:
        raise ValueError("No points provided for grid.")
    xs = [p.real for p in pts]
    ys = [p.imag for p in pts]
    x_center = 0.5 * (min(xs) + max(xs))
    y_center = 0.5 * (min(ys) + max(ys))
    half = radius
    x = np.linspace(x_center - half, x_center + half, n)
    y = np.linspace(y_center - half, y_center + half, n)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j * Y
    return X, Y, Z


def _compute_RePhi_and_grad(
    Z: np.ndarray,
    gt: float,
    beta: float = DEFAULT_BETA,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate Re Φ and its gradient ∇Re Φ on a grid Z.

    For an analytic Φ, if f'(z) = dΦ/dθ, then
        ∇ Re f = (Re f'(z), -Im f'(z)).
    """
    flat = Z.ravel()
    vals = np.array([Phi(z, gt, beta) for z in flat])
    dvals = np.array([dPhi(z, gt, beta) for z in flat])
    rephi = vals.real.reshape(Z.shape)
    gx = dvals.real.reshape(Z.shape)
    gy = -dvals.imag.reshape(Z.shape)
    return rephi, gx, gy


def _plot_single_stokes_slice(
    gt: float,
    theta_primary: Complex,
    theta_rival: Complex,
    out_path: Path,
    beta: float = DEFAULT_BETA,
    n_grid: int = 300,
    radius: float = 3.0,
    levels: int = 50,
) -> None:
    """
    Plot one 2D slice in the θ-plane:
        - filled contours of Re Φ,
        - 'colored strips' where Re Φ(z) < Re Φ(θ_primary),
        - vector field of −∇Re Φ (steepest descent),
        - markers for the two saddle points.
    """
    X, Y, Z = _grid_around_points([theta_primary, theta_rival], radius=radius, n=n_grid)
    rephi, gx, gy = _compute_RePhi_and_grad(Z, gt, beta=beta)

    # Reference action at the primary saddle.
    rephi_primary = float(Phi(theta_primary, gt, beta).real)

    fig, ax = plt.subplots(figsize=(6, 5))

    # Base contour of Re Φ.
    cf = ax.contourf(
        X,
        Y,
        rephi,
        levels=levels,
        cmap="viridis",
    )
    plt.colorbar(cf, ax=ax, label=r"$\mathrm{Re}\,\Phi(\theta)$")

    # Highlight 'colored strip' where Re Φ < Re Φ(θ_primary).
    mask = rephi < rephi_primary
    ax.contourf(
        X,
        Y,
        np.where(mask, rephi, np.nan),
        levels=10,
        cmap="coolwarm",
        alpha=0.5,
    )

    # Vector field of steepest descent: −∇Re Φ.
    step = max(1, n_grid // 30)
    Xs = X[::step, ::step]
    Ys = Y[::step, ::step]
    Gx = -gx[::step, ::step]
    Gy = -gy[::step, ::step]
    norm = np.sqrt(Gx**2 + Gy**2) + 1e-16
    ax.quiver(
        Xs,
        Ys,
        Gx / norm,
        Gy / norm,
        color="k",
        alpha=0.6,
        scale=30,
        width=0.002,
    )

    # Plot the two saddle points.
    ax.scatter(
        [theta_primary.real],
        [theta_primary.imag],
        color="yellow",
        edgecolor="black",
        s=80,
        label="Primary saddle",
        zorder=5,
    )
    ax.scatter(
        [theta_rival.real],
        [theta_rival.imag],
        color="red",
        edgecolor="black",
        s=80,
        label="Rival saddle",
        zorder=5,
    )

    ax.set_xlabel(r"$\mathrm{Re}\,\theta$")
    ax.set_ylabel(r"$\mathrm{Im}\,\theta$")
    ax.set_title(rf"Stokes slice at $\tilde\gamma = {gt:.3f}$")
    ax.legend(loc="upper right")
    ax.set_aspect("equal", "box")
    ax.grid(True, alpha=0.2)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_stokes_slices(
    branch_primary: SaddleBranch,
    branch_rival: SaddleBranch,
    gt_before: float,
    gt_cross: float,
    gt_after: float,
    out_dir: str | Path = "figures/Tracking",
    beta: float = DEFAULT_BETA,
) -> Tuple[Path, Path, Path]:
    """
    Step C: generate three 2D complex-plane plots around a Stokes crossing.

    Parameters
    ----------
    branch_primary, branch_rival:
        The two branches to visualise.
    gt_before, gt_cross, gt_after:
        Three γ̃ values: before, near, and after the crossing.
    out_dir:
        Directory to which PNGs will be written.
    """
    out_dir = Path(out_dir)

    def interp_theta(branch: SaddleBranch, gt: float) -> Complex:
        th_re = np.interp(gt, branch.gt, branch.theta.real)
        th_im = np.interp(gt, branch.gt, branch.theta.imag)
        th, ok = newton_solve_saddle(th_re + 1j * th_im, gt, beta=beta, max_step=0.5)
        return th if ok else th_re + 1j * th_im

    pts = []
    g_list = [gt_before, gt_cross, gt_after]
    out_paths: List[Path] = []
    for label, g in zip(["before", "at", "after"], g_list):
        th1 = interp_theta(branch_primary, g)
        th2 = interp_theta(branch_rival, g)
        pts.extend([th1, th2])
        out_path = out_dir / f"stokes_slice_{label}_gt_{g:.4f}.png"
        _plot_single_stokes_slice(
            g,
            th1,
            th2,
            out_path=out_path,
            beta=beta,
        )
        out_paths.append(out_path)

    return tuple(out_paths)  # type: ignore[return-value]


if __name__ == "__main__":
    """
    Minimal demo: track primary and rival branches and plot three Stokes slices.

    This uses:
        - gt_start = 0.30 (BM24 canonical saddle region),
        - gt_end   = 3.0,
        - dgt      = 0.01.
    It saves the figures under symmetry_reduction/Tracking/figures/Tracking.
    """
    gt_start = 0.30
    gt_end = 3.0
    dgt = 0.01
    primary, rival = track_primary_and_rival_branches(gt_start, gt_end, dgt)
    if rival is None:
        print("Could not identify a rival saddle branch; aborting demo.")
    else:
        gt_c = find_stokes_crossing(primary, rival)
        if gt_c is None:
            print("No Stokes crossing detected along the sampled interval.")
        else:
            # Choose three nearby slices.
            delta = 0.05
            gt_before = gt_c - delta
            gt_after = gt_c + delta
            out_before, out_cross, out_after = plot_stokes_slices(
                primary,
                rival,
                gt_before,
                gt_c,
                gt_after,
                out_dir=Path(__file__).resolve().parent / "figures" / "Tracking",
            )
            print("Saved Stokes slices to:")
            print("  ", out_before)
            print("  ", out_cross)
            print("  ", out_after)

