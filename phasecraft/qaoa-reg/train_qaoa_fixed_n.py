"""
Regular QAOA fixed-n training for the BM24 / PRX Quantum k-SAT setup.

This mirrors the LR-QAOA fixed-n trainer, but optimizes the full regular-QAOA
angle vector instead of the two LR ramp deltas.  The PRX Quantum convention is
used throughout:

    |psi(beta, gamma)> = prod_l exp(-i beta_l H_B / 2)
                         exp(-i gamma_l H[sigma] / 2) |+>^n

The returned packed parameter vector is ``np.concatenate([gammas, betas])``,
matching the BM24 convention already used in the repository.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.optimize import minimize

_PHASECRAFT_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = _PHASECRAFT_ROOT.parent
for _p in (_REPO_ROOT, _PHASECRAFT_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from phasecraft.experiments.lr_scaling.config import DEFAULT_EPS, MIN_TRAIN_MEAN_P_SUCC  # noqa: E402
from phasecraft.experiments.lr_scaling.train_lr_fixed_n import generate_training_instances  # noqa: E402
from phasecraft.lib.sim.bm24_qaoa_sim import (  # noqa: E402
    per_instance_success_probability,
    run_qaoa,
)

try:
    from phasecraft.experiments.lr_scaling.gpu.backend import (  # type: ignore
        success_probs_for_instances as _batched_success_probs_for_instances,
    )
except Exception:  # pragma: no cover - optional CUDA/CuPy backend
    _batched_success_probs_for_instances = None  # type: ignore[assignment]

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(it, **kwargs):  # type: ignore[misc]
        return it


SUPPORTED_TRAINING_MODES = frozenset({
    "bm24_mean_p_fixed_n",
    "median_runtime_fixed_n",
    "mean_log_runtime_fixed_n",
})

DEFAULT_GAMMA_BOUNDS: Tuple[float, float] = (-2.0, 0.0)
DEFAULT_BETA_BOUNDS: Tuple[float, float] = (0.0, 4.0)
DEFAULT_INITIAL_GAMMA: float = -0.01
DEFAULT_INITIAL_BETA: float = 0.01
DEFAULT_OPTIMIZER: str = "COBYLA"
MODE_RNG_OFFSET: Dict[str, int] = {
    "bm24_mean_p_fixed_n": 0,
    "median_runtime_fixed_n": 99_999,
    "mean_log_runtime_fixed_n": 199_998,
}

_OBJECTIVE_LABELS = {
    "bm24_mean_p_fixed_n": "maximize mean(p_succ) at fixed train_n",
    "median_runtime_fixed_n": "minimize median(1/(p_succ+eps)) at fixed train_n",
    "mean_log_runtime_fixed_n": "minimize mean(ln(1/(p_succ+eps))) at fixed train_n",
}

AngleInput = Union[
    np.ndarray,
    Sequence[float],
    Tuple[Sequence[float], Sequence[float]],
]


def pack_angles(gammas: Sequence[float], betas: Sequence[float]) -> np.ndarray:
    """Pack regular-QAOA angles as [gammas, betas]."""
    gammas_arr = np.asarray(gammas, dtype=float)
    betas_arr = np.asarray(betas, dtype=float)
    if gammas_arr.shape != betas_arr.shape:
        raise ValueError("gammas and betas must have the same shape")
    return np.concatenate([gammas_arr, betas_arr])


def unpack_angles(params: Sequence[float], depth: int) -> Tuple[np.ndarray, np.ndarray]:
    """Unpack [gammas, betas] into (betas, gammas) for run_qaoa."""
    depth = int(depth)
    arr = np.asarray(params, dtype=float)
    if arr.size != 2 * depth:
        raise ValueError(f"expected {2 * depth} packed angles, got {arr.size}")
    gammas = arr[:depth].astype(float, copy=True)
    betas = arr[depth:].astype(float, copy=True)
    return betas, gammas


def constant_initial_angles(
    depth: int,
    *,
    gamma: float = DEFAULT_INITIAL_GAMMA,
    beta: float = DEFAULT_INITIAL_BETA,
) -> Tuple[np.ndarray, np.ndarray]:
    """PRX-style initial point: all beta=+0.01 and all gamma=-0.01."""
    depth = int(depth)
    if depth <= 0:
        raise ValueError("depth must be positive")
    betas = np.full(depth, float(beta), dtype=float)
    gammas = np.full(depth, float(gamma), dtype=float)
    return betas, gammas


def resize_angles(
    betas: Sequence[float],
    gammas: Sequence[float],
    depth: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Interpolate a previous-depth regular-QAOA schedule to a new depth."""
    depth = int(depth)
    old_betas = np.asarray(betas, dtype=float)
    old_gammas = np.asarray(gammas, dtype=float)
    if old_betas.size != old_gammas.size:
        raise ValueError("old betas and gammas must have the same length")
    if old_betas.size == depth:
        return old_betas.copy(), old_gammas.copy()
    if old_betas.size == 0:
        return constant_initial_angles(depth)
    if old_betas.size == 1:
        return (
            np.full(depth, float(old_betas[0]), dtype=float),
            np.full(depth, float(old_gammas[0]), dtype=float),
        )
    old_t = np.linspace(0.0, 1.0, old_betas.size)
    new_t = np.linspace(0.0, 1.0, depth)
    return np.interp(new_t, old_t, old_betas), np.interp(new_t, old_t, old_gammas)


def _coerce_initial_angles(initial_angles: Optional[AngleInput], depth: int) -> Optional[np.ndarray]:
    if initial_angles is None:
        return None
    if isinstance(initial_angles, tuple) and len(initial_angles) == 2:
        betas, gammas = resize_angles(initial_angles[0], initial_angles[1], depth)
        return pack_angles(gammas, betas)
    arr = np.asarray(initial_angles, dtype=float)
    if arr.ndim != 1 or arr.size % 2 != 0:
        raise ValueError("initial_angles must be (betas, gammas) or a flat [gammas, betas] vector")
    old_depth = arr.size // 2
    old_gammas = arr[:old_depth]
    old_betas = arr[old_depth:]
    betas, gammas = resize_angles(old_betas, old_gammas, depth)
    return pack_angles(gammas, betas)


def _clip_params(
    params: Sequence[float],
    depth: int,
    gamma_bounds: Tuple[float, float],
    beta_bounds: Tuple[float, float],
) -> np.ndarray:
    x = np.asarray(params, dtype=float).copy()
    x[:depth] = np.clip(x[:depth], gamma_bounds[0], gamma_bounds[1])
    x[depth:] = np.clip(x[depth:], beta_bounds[0], beta_bounds[1])
    return x


def _bounds(depth: int, gamma_bounds: Tuple[float, float], beta_bounds: Tuple[float, float]) -> List[Tuple[float, float]]:
    return [tuple(gamma_bounds)] * int(depth) + [tuple(beta_bounds)] * int(depth)


def _box_constraints(bounds: Sequence[Tuple[float, float]]) -> list:
    constraints = []
    for idx, (lo, hi) in enumerate(bounds):
        constraints.append({"type": "ineq", "fun": lambda x, idx=idx, lo=lo: x[idx] - lo})
        constraints.append({"type": "ineq", "fun": lambda x, idx=idx, hi=hi: hi - x[idx]})
    return constraints


def success_probabilities(
    instances: Sequence[np.ndarray],
    n: int,
    betas: np.ndarray,
    gammas: np.ndarray,
    *,
    device=None,
    batch_size: Optional[int] = None,
) -> np.ndarray:
    """Per-instance success probabilities, batched on GPU/NumPy when configured."""
    if not instances:
        return np.array([], dtype=np.float64)
    if device is not None and _batched_success_probs_for_instances is not None:
        return _batched_success_probs_for_instances(
            instances,
            int(n),
            np.asarray(betas, dtype=float),
            np.asarray(gammas, dtype=float),
            device=device,
            batch_size=batch_size,
        )
    probs = np.empty(len(instances), dtype=np.float64)
    for i, h_diag in enumerate(instances):
        psi = run_qaoa(h_diag, betas, gammas, int(n))
        probs[i] = float(per_instance_success_probability(psi, h_diag))
    return probs


def compute_objective(
    training_mode: str,
    instances: Sequence[np.ndarray],
    n: int,
    betas: np.ndarray,
    gammas: np.ndarray,
    eps: float = DEFAULT_EPS,
    *,
    device=None,
    batch_size: Optional[int] = None,
) -> float:
    """Scalar objective to minimize for regular QAOA."""
    if training_mode not in SUPPORTED_TRAINING_MODES:
        raise ValueError(f"unknown training_mode {training_mode!r}")
    if not instances:
        return float("nan")
    probs = success_probabilities(
        instances,
        n,
        betas,
        gammas,
        device=device,
        batch_size=batch_size,
    )
    if training_mode == "bm24_mean_p_fixed_n":
        return -float(np.mean(probs))
    if training_mode == "median_runtime_fixed_n":
        return float(np.median(1.0 / np.maximum(probs, float(eps))))
    if training_mode == "mean_log_runtime_fixed_n":
        return float(np.mean(np.log(1.0 / np.maximum(probs, float(eps)))))
    raise AssertionError("unreachable")


def mean_p_succ(
    instances: Sequence[np.ndarray],
    n: int,
    betas: np.ndarray,
    gammas: np.ndarray,
    *,
    device=None,
    batch_size: Optional[int] = None,
) -> float:
    if not instances:
        return 0.0
    probs = success_probabilities(
        instances,
        n,
        betas,
        gammas,
        device=device,
        batch_size=batch_size,
    )
    return float(np.mean(probs)) if probs.size else 0.0


def _adaptive_rhobeg(depth: int, gamma_bounds: Tuple[float, float], beta_bounds: Tuple[float, float]) -> float:
    span = 0.5 * ((gamma_bounds[1] - gamma_bounds[0]) + (beta_bounds[1] - beta_bounds[0]))
    factor = 0.18 / max(1.0, math.log2(max(2, int(depth))) / 2.0)
    return float(max(0.02, min(0.18, factor)) * span)


def _run_minimize(
    objective,
    x0: np.ndarray,
    *,
    depth: int,
    optimizer: str,
    bounds: Sequence[Tuple[float, float]],
    constraints: list,
    maxiter: int,
    tol: float,
    rhobeg: float,
    gamma_bounds: Tuple[float, float],
    beta_bounds: Tuple[float, float],
) -> Tuple[np.ndarray, float]:
    method = str(optimizer)
    if method.upper() == "COBYLA":
        res = minimize(
            objective,
            x0,
            method="COBYLA",
            constraints=constraints,
            tol=tol,
            options={"maxiter": int(maxiter), "rhobeg": float(rhobeg), "catol": 1e-6, "disp": False},
        )
    elif method.upper() in {"L-BFGS-B", "POWELL"}:
        canonical = "L-BFGS-B" if method.upper() == "L-BFGS-B" else "Powell"
        res = minimize(
            objective,
            x0,
            method=canonical,
            bounds=list(bounds),
            tol=tol,
            options={"maxiter": int(maxiter), "disp": False},
        )
    else:
        raise ValueError("optimizer must be one of COBYLA, L-BFGS-B, Powell")
    x_fin = _clip_params(res.x, depth, gamma_bounds, beta_bounds)
    return x_fin, float(objective(x_fin))


def train_regular_qaoa_fixed_n(
    training_mode: str,
    train_n: int,
    depth: int,
    instances: List[np.ndarray],
    *,
    initial_angles: Optional[AngleInput] = None,
    gamma_bounds: Tuple[float, float] = DEFAULT_GAMMA_BOUNDS,
    beta_bounds: Tuple[float, float] = DEFAULT_BETA_BOUNDS,
    optimizer: str = DEFAULT_OPTIMIZER,
    maxiter: int = 240,
    tol: float = 1e-3,
    rhobeg: Optional[float] = None,
    restarts: int = 4,
    perturb_scale: float = 0.1,
    anti_regression: bool = True,
    collapse_guard: bool = True,
    eps: float = DEFAULT_EPS,
    device=None,
    batch_size: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
    verbose: bool = True,
) -> Tuple[np.ndarray, dict]:
    """
    Train full regular-QAOA angles at one fixed n.

    The default objective and initialization follow the PRX Quantum numerical
    protocol: maximize empirical mean success probability over a fixed training
    set, starting from beta=+0.01 and gamma=-0.01 in every layer.
    """
    if training_mode not in SUPPORTED_TRAINING_MODES:
        raise ValueError(f"training_mode={training_mode!r} not in {sorted(SUPPORTED_TRAINING_MODES)}")
    train_n = int(train_n)
    depth = int(depth)
    if depth <= 0:
        raise ValueError("depth must be positive")

    gen = rng if rng is not None else np.random.default_rng()
    bounds = _bounds(depth, gamma_bounds, beta_bounds)
    constraints = _box_constraints(bounds)
    rhobeg_val = (
        float(rhobeg)
        if rhobeg is not None
        else _adaptive_rhobeg(depth, gamma_bounds, beta_bounds)
    )

    default_betas, default_gammas = constant_initial_angles(depth)
    default_x = _clip_params(pack_angles(default_gammas, default_betas), depth, gamma_bounds, beta_bounds)
    warm_x = _coerce_initial_angles(initial_angles, depth)
    if warm_x is not None:
        warm_x = _clip_params(warm_x, depth, gamma_bounds, beta_bounds)

    def obj(x: np.ndarray) -> float:
        betas, gammas = unpack_angles(_clip_params(x, depth, gamma_bounds, beta_bounds), depth)
        return compute_objective(
            training_mode,
            instances,
            train_n,
            betas,
            gammas,
            eps=eps,
            device=device,
            batch_size=batch_size,
        )

    if verbose:
        backend = "CPU serial"
        if device is not None:
            backend = device.summary() if hasattr(device, "summary") else str(device)
        print(
            f"  > train_regular_qaoa_fixed_n: mode={training_mode!r} "
            f"n={train_n} p={depth} instances={len(instances)} "
            f"optimizer={optimizer} backend={backend}"
        )

    starts: List[np.ndarray] = []
    seen: set = set()

    def _add_start(x: np.ndarray) -> None:
        clipped = _clip_params(x, depth, gamma_bounds, beta_bounds)
        key = tuple(np.round(clipped, 10).tolist())
        if key not in seen:
            starts.append(clipped)
            seen.add(key)

    if warm_x is not None:
        _add_start(warm_x)
    _add_start(default_x)
    anchor = starts[0]
    n_random = max(0, int(restarts) - len(starts))
    spans = np.asarray([hi - lo for lo, hi in bounds], dtype=float)
    for _ in range(n_random):
        noise = gen.normal(0.0, float(perturb_scale), size=2 * depth) * spans
        _add_start(anchor + noise)

    warm_obj = obj(warm_x) if warm_x is not None else None
    best_x = starts[0].copy()
    best_obj = obj(best_x)
    restart_records: List[dict] = []
    for i, x0 in enumerate(tqdm(starts, desc="QAOA restarts", disable=not verbose)):
        x_fin, obj_fin = _run_minimize(
            obj,
            x0,
            depth=depth,
            optimizer=optimizer,
            bounds=bounds,
            constraints=constraints,
            maxiter=maxiter,
            tol=tol,
            rhobeg=rhobeg_val,
            gamma_bounds=gamma_bounds,
            beta_bounds=beta_bounds,
        )
        restart_records.append({
            "restart_idx": int(i),
            "x0": x0.tolist(),
            "x_final": x_fin.tolist(),
            "obj": float(obj_fin),
        })
        if obj_fin < best_obj:
            best_obj = float(obj_fin)
            best_x = x_fin.copy()
        if verbose:
            print(f"  > restart {i:2d}: obj={obj_fin:+.6g}")

    anti_regression_applied = False
    if anti_regression and warm_x is not None and warm_obj is not None:
        if best_obj > warm_obj + 1e-12 * max(1.0, abs(warm_obj)):
            anti_regression_applied = True
            best_x = warm_x.copy()
            best_obj = float(warm_obj)
            if verbose:
                print(f"  > Anti-regression: reverted to warm-start (obj={warm_obj:+.6g})")

    trained_x = best_x.copy()
    betas_opt, gammas_opt = unpack_angles(best_x, depth)
    final_mean_p = mean_p_succ(
        instances,
        train_n,
        betas_opt,
        gammas_opt,
        device=device,
        batch_size=batch_size,
    )

    train_rejected = False
    train_reject_reason = ""
    if float(final_mean_p) < float(MIN_TRAIN_MEAN_P_SUCC):
        train_rejected = True
        train_reject_reason = f"mean_p={final_mean_p:.4e} < {MIN_TRAIN_MEAN_P_SUCC:g}"

    collapse_guard_applied = False
    if collapse_guard and train_rejected and warm_x is not None:
        warm_betas, warm_gammas = unpack_angles(warm_x, depth)
        warm_mean_p = mean_p_succ(
            instances,
            train_n,
            warm_betas,
            warm_gammas,
            device=device,
            batch_size=batch_size,
        )
        if warm_mean_p >= float(MIN_TRAIN_MEAN_P_SUCC):
            collapse_guard_applied = True
            best_x = warm_x.copy()
            best_obj = float(obj(best_x))
            betas_opt, gammas_opt = unpack_angles(best_x, depth)
            final_mean_p = float(warm_mean_p)
            if verbose:
                print(f"  > Collapse guard: reverted to warm-start ({train_reject_reason})")

    params_concat = pack_angles(gammas_opt, betas_opt)
    diag = {
        "training_mode": training_mode,
        "train_n": train_n,
        "depth": depth,
        "objective": _OBJECTIVE_LABELS[training_mode],
        "simulator": "bm24",
        "ansatz": "regular_qaoa",
        "uses_multi_n_training": False,
        "eval_slope_only": True,
        "angle_convention": "bm24_half_angle",
        "backend": (
            device.summary()
            if device is not None and hasattr(device, "summary")
            else ("batched" if device is not None else "CPU serial")
        ),
        "batch_size": None if batch_size is None else int(batch_size),
        "best_gammas": gammas_opt.tolist(),
        "best_betas": betas_opt.tolist(),
        "trained_gammas": trained_x[:depth].tolist(),
        "trained_betas": trained_x[depth:].tolist(),
        "best_train_objective": float(best_obj),
        "best_avg_train_p_succ": float(final_mean_p),
        "train_rejected": bool(train_rejected),
        "train_reject_reason": train_reject_reason,
        "collapse_guard_applied": bool(collapse_guard_applied),
        "anti_regression_applied": bool(anti_regression_applied),
        "warm_start_used": bool(initial_angles is not None),
        "optimizer": str(optimizer),
        "optimizer_restarts": int(len(starts)),
        "optimizer_rhobeg_used": float(rhobeg_val),
        "gamma_bounds": list(map(float, gamma_bounds)),
        "beta_bounds": list(map(float, beta_bounds)),
        "restart_records": restart_records,
    }
    if verbose:
        print(
            f"  > Result: obj={best_obj:+.6g} mean_p@train_n={final_mean_p:.4e} "
            f"gamma[0]={gammas_opt[0]:+.4f} beta[0]={betas_opt[0]:+.4f}"
        )
    return params_concat, diag


def verify_objectives() -> None:
    """Small sanity check for packing and objective signs."""
    h_diag = np.array([0, 1], dtype=np.int64)
    betas = np.array([0.01])
    gammas = np.array([-0.01])
    packed = pack_angles(gammas, betas)
    b2, g2 = unpack_angles(packed, 1)
    assert np.allclose(betas, b2)
    assert np.allclose(gammas, g2)
    instances = [h_diag]
    mean_obj = compute_objective("bm24_mean_p_fixed_n", instances, 1, betas, gammas)
    med_obj = compute_objective("median_runtime_fixed_n", instances, 1, betas, gammas)
    assert mean_obj <= 0.0
    assert med_obj >= 1.0


if __name__ == "__main__":
    verify_objectives()
    print("regular QAOA trainer sanity checks passed")
