"""
Utilities for diagnosing saddle traits during parameter scans.

Key diagnostics implemented here follow practical criteria:
- Degeneracy proxy: very small sigma_min(Hessian) or sigma_min(I - DG).
- Coalescence signal: two saddle tracks are close in state and action.
- Stokes / anti-Stokes candidates from sign changes in:
    Delta_I = Im(Phi_sigma - Phi_tau) (with unwrapped phases),
    Delta_R = Re(Phi_sigma - Phi_tau).
- Adaptive refinement suggestions near events.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


ComplexArray = np.ndarray


@dataclass(frozen=True)
class DegeneracySignals:
    sigma_min_hessian: float
    sigma_min_residual_jacobian: float
    residual_norm: float

    def is_near_degenerate(
        self,
        hessian_sigma_tol: float = 1e-4,
        residual_jac_sigma_tol: float = 1e-4,
    ) -> bool:
        h_small = np.isfinite(self.sigma_min_hessian) and self.sigma_min_hessian <= hessian_sigma_tol
        r_small = np.isfinite(self.sigma_min_residual_jacobian) and self.sigma_min_residual_jacobian <= residual_jac_sigma_tol
        return bool(h_small or r_small)


@dataclass(frozen=True)
class PairwiseSaddleSignal:
    i: int
    j: int
    z_distance: float
    action_distance: float
    delta_real: float
    delta_imag: float
    coalescence_candidate: bool
    anti_stokes_candidate: bool
    stokes_candidate: bool


@dataclass(frozen=True)
class ScanRefinementAdvice:
    coarse_step: float
    refined_step: float
    bracket_step: float
    should_refine: bool
    reasons: Tuple[str, ...]


def smallest_singular_value(matrix: np.ndarray) -> float:
    if matrix.size == 0:
        return np.nan
    svals = np.linalg.svd(matrix, compute_uv=False)
    return float(svals[-1]) if svals.size else np.nan


def complex_to_real_vector(z: ComplexArray) -> np.ndarray:
    return np.concatenate([z.real, z.imag])


def real_to_complex_vector(x: np.ndarray) -> ComplexArray:
    n = x.size // 2
    return x[:n] + 1j * x[n:]


def numerical_jacobian_real(
    func: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    eps: float = 1e-7,
) -> np.ndarray:
    n = x.size
    fx = func(x)
    jac = np.zeros((n, n), dtype=float)
    for k in range(n):
        dx = np.zeros(n, dtype=float)
        dx[k] = eps
        jac[:, k] = (func(x + dx) - fx) / eps
    return jac


def estimate_residual_jacobian_smin(
    z: ComplexArray,
    fixed_point_map_complex: Optional[Callable[[ComplexArray], ComplexArray]] = None,
    residual_complex: Optional[Callable[[ComplexArray], ComplexArray]] = None,
    eps: float = 1e-7,
) -> float:
    """
    Estimate sigma_min(J_R) in doubled real coordinates.

    Provide exactly one of:
    - fixed_point_map_complex: G(z) with residual R(z) = z - G(z)
    - residual_complex: R(z) directly
    """
    if (fixed_point_map_complex is None) == (residual_complex is None):
        raise ValueError("Provide exactly one of fixed_point_map_complex or residual_complex")

    def residual_real(x: np.ndarray) -> np.ndarray:
        zc = real_to_complex_vector(x)
        if fixed_point_map_complex is not None:
            rz = zc - fixed_point_map_complex(zc)
        else:
            rz = residual_complex(zc)  # type: ignore[misc]
        return complex_to_real_vector(rz)

    x = complex_to_real_vector(z)
    jac = numerical_jacobian_real(residual_real, x, eps=eps)
    return smallest_singular_value(jac)


def evaluate_degeneracy_signals(
    z: ComplexArray,
    action: complex,
    hessian: np.ndarray,
    residual_norm: float,
    fixed_point_map_complex: Optional[Callable[[ComplexArray], ComplexArray]] = None,
    residual_complex: Optional[Callable[[ComplexArray], ComplexArray]] = None,
    jacobian_eps: float = 1e-7,
) -> DegeneracySignals:
    del action  # action can be logged by callers; not needed in scalar test
    sigma_h = smallest_singular_value(hessian)
    sigma_r = np.nan
    if fixed_point_map_complex is not None or residual_complex is not None:
        sigma_r = estimate_residual_jacobian_smin(
            z=z,
            fixed_point_map_complex=fixed_point_map_complex,
            residual_complex=residual_complex,
            eps=jacobian_eps,
        )
    return DegeneracySignals(
        sigma_min_hessian=float(sigma_h),
        sigma_min_residual_jacobian=float(sigma_r),
        residual_norm=float(residual_norm),
    )


def unwrap_action_imag(actions: Sequence[complex], period: float = 2 * np.pi) -> np.ndarray:
    imag = np.array([np.imag(a) for a in actions], dtype=float)
    discont = period / 2.0
    return np.unwrap(imag, discont=discont, period=period)


def unwrap_action_imag_trackwise(
    actions_by_track_and_step: np.ndarray,
    period: float = 2 * np.pi,
    axis: int = 1,
) -> np.ndarray:
    """
    Unwrap Im(Phi) along scan progression for each saddle track.

    Expected shape is (num_tracks, num_steps) with axis=1 by default.
    """
    imag = np.imag(np.asarray(actions_by_track_and_step))
    discont = period / 2.0
    return np.unwrap(imag, discont=discont, period=period, axis=axis)


def _has_sign_change(prev: float, curr: float, tol: float) -> bool:
    if not (np.isfinite(prev) and np.isfinite(curr)):
        return False
    if abs(prev) <= tol or abs(curr) <= tol:
        return False
    return np.sign(prev) != np.sign(curr)


def pairwise_saddle_signals(
    saddles: Sequence[ComplexArray],
    actions: Sequence[complex],
    actions_unwrapped_imag: Optional[Sequence[float]],
    sigma_mins: Sequence[float],
    z_tol: float = 1e-2,
    action_tol: float = 5e-2,
    sigma_tol: float = 1e-4,
    sign_change_tol: float = 1e-8,
    prev_delta_real: Optional[Dict[Tuple[int, int], float]] = None,
    prev_delta_imag: Optional[Dict[Tuple[int, int], float]] = None,
) -> List[PairwiseSaddleSignal]:
    n = len(saddles)
    if actions_unwrapped_imag is None:
        # Fallback only: if no trackwise-unwrapped input is available, use raw imag.
        unwrapped_imag = np.array([np.imag(a) for a in actions], dtype=float)
    else:
        unwrapped_imag = np.array(actions_unwrapped_imag, dtype=float)
    if unwrapped_imag.shape[0] != n:
        raise ValueError("actions_unwrapped_imag must match number of saddles")
    out: List[PairwiseSaddleSignal] = []
    for i in range(n):
        for j in range(i + 1, n):
            dz = float(np.linalg.norm(saddles[i] - saddles[j], ord=np.inf))
            dphi = actions[i] - actions[j]
            dabs = float(np.abs(dphi))
            dreal = float(np.real(dphi))
            dimag = float(unwrapped_imag[i] - unwrapped_imag[j])
            local_sigma = float(min(sigma_mins[i], sigma_mins[j]))

            coalesce = (dz <= z_tol) and (dabs <= action_tol) and (local_sigma <= sigma_tol)

            key = (i, j)
            anti = False
            stokes = False
            if prev_delta_real is not None and key in prev_delta_real:
                anti = _has_sign_change(prev_delta_real[key], dreal, sign_change_tol)
            if prev_delta_imag is not None and key in prev_delta_imag:
                stokes = _has_sign_change(prev_delta_imag[key], dimag, sign_change_tol)

            out.append(
                PairwiseSaddleSignal(
                    i=i,
                    j=j,
                    z_distance=dz,
                    action_distance=dabs,
                    delta_real=dreal,
                    delta_imag=dimag,
                    coalescence_candidate=bool(coalesce),
                    anti_stokes_candidate=bool(anti),
                    stokes_candidate=bool(stokes),
                )
            )
    return out


def recommend_scan_refinement(
    pair_signals: Sequence[PairwiseSaddleSignal],
    sigma_history: Sequence[float],
    coarse_step: float = 1e-2,
    refined_step: float = 1e-3,
    bracket_step: float = 1e-4,
    delta_trigger: float = 5e-2,
    sigma_trigger: float = 1e-4,
    sigma_drop_factor: float = 5.0,
) -> ScanRefinementAdvice:
    reasons: List[str] = []
    for s in pair_signals:
        if abs(s.delta_real) < delta_trigger:
            reasons.append("small_delta_real")
            break
    for s in pair_signals:
        if abs(s.delta_imag) < delta_trigger:
            reasons.append("small_delta_imag")
            break
    if sigma_history:
        current = float(sigma_history[-1])
        if current < sigma_trigger:
            reasons.append("small_sigma_min")
        if len(sigma_history) >= 2 and sigma_history[-2] > 0:
            ratio = sigma_history[-2] / max(current, 1e-300)
            if ratio >= sigma_drop_factor:
                reasons.append("sharp_sigma_drop")
    if any(s.coalescence_candidate for s in pair_signals):
        reasons.append("coalescence_candidate")
    if any(s.anti_stokes_candidate for s in pair_signals):
        reasons.append("anti_stokes_sign_change")
    if any(s.stokes_candidate for s in pair_signals):
        reasons.append("stokes_sign_change")

    uniq_reasons = tuple(dict.fromkeys(reasons))
    return ScanRefinementAdvice(
        coarse_step=coarse_step,
        refined_step=refined_step,
        bracket_step=bracket_step,
        should_refine=bool(uniq_reasons),
        reasons=uniq_reasons,
    )


def finite_n_tie_scale(n: int) -> float:
    if n <= 0:
        raise ValueError("n must be positive")
    return 1.0 / float(n)


def derivative_safe_step(
    max_abs_derivative: float,
    target_scale: float = 0.2,
) -> float:
    """
    Return h such that h * max|d(Delta)/dt| <= target_scale.
    Smaller target_scale means more conservative stepping.
    """
    if max_abs_derivative <= 0:
        return np.inf
    return float(target_scale / max_abs_derivative)

