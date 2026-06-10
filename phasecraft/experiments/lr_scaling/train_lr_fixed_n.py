"""
train_lr_fixed_n.py
===================
Three clean fixed-n LR-QAOA training modes:

  "bm24_mean_p_fixed_n"       — maximize mean(p_succ) at train_n.
                                BM24 baseline / faithful reproduction.
                                Objective: minimize -mean_{sigma}(p_succ(sigma; beta, gamma))

  "median_runtime_fixed_n"    — minimize median(1/(p_succ + eps)) at train_n.
                                Robust runtime objective at a single fixed n.
                                Objective: minimize median_{sigma}(1 / (p_succ(sigma) + eps))

  "mean_log_runtime_fixed_n"  — minimize mean(ln(1/(p_succ + eps))) at train_n.
                                Smoother than median runtime; less spike-sensitive.
                                Objective: minimize mean_{sigma}(ln(1 / (p_succ(sigma) + eps)))

Training rule: ONE fixed n only. Evaluation of scaling slope across n is done
externally (in the notebook / benchmark code) and must run AFTER training.
The n-slope is a diagnostic; it is never used as the training objective here.

Public API
----------
    generate_training_instances(train_n, k, r, train_size, base_seed)
        -> List[np.ndarray]   (SAT-filtered H_diag arrays at train_n)

    train_angles_fixed_n(training_mode, train_n, depth, instances, ...)
        -> (params_concat, diagnostics_dict)
        params_concat = np.concatenate([gammas, betas]) in BM24 convention.
        diagnostics_dict always includes:
            "training_mode", "train_n", "depth", "objective",
            "simulator": "bm24",
            "uses_multi_n_training": False,
            "eval_slope_only": True,
            "best_deltas", "trained_deltas",
            "best_train_objective", "best_avg_train_p_succ",
            "train_rejected", "train_reject_reason",
            "collapse_guard_applied", "anti_regression_applied",
            "warm_start_used", "grid_skipped",
            "cobyla_restarts", "cobyla_rhobeg_used", "restart_records"

    verify_objectives()
        -> None  (raises AssertionError if objective implementations are wrong)
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize

_PHASECRAFT_ROOT = Path(__file__).resolve().parents[2]
_REPO_ROOT = _PHASECRAFT_ROOT.parent
for _p in (_REPO_ROOT, _PHASECRAFT_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from phasecraft.lib.sim.bm24_qaoa_sim import (  # noqa: E402
    build_h_diagonal,
    generate_random_clause,
    make_lr_angles,
    per_instance_success_probability,
    run_qaoa,
)
from phasecraft.experiments.lr_scaling.config import (  # noqa: E402
    DEFAULT_DB_BOUNDS as _DB,
    DEFAULT_DG_BOUNDS as _DG,
    DEFAULT_EPS as _EPS,
    DEFAULT_INITIAL_DELTAS as _INIT,
    MIN_TRAIN_MEAN_P_SUCC as _MIN_P,
)

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(it, **kwargs):  # type: ignore[misc]
        return it


# --------------------------------------------------------------------------- #
# Constants                                                                    #
# --------------------------------------------------------------------------- #

SUPPORTED_TRAINING_MODES = frozenset({
    "bm24_mean_p_fixed_n",
    "median_runtime_fixed_n",
    "mean_log_runtime_fixed_n",
})

DEFAULT_DG_BOUNDS: Tuple[float, float] = _DG
DEFAULT_DB_BOUNDS: Tuple[float, float] = _DB
DEFAULT_INITIAL_DELTAS: Tuple[float, float] = _INIT
DEFAULT_EPS: float = _EPS
MIN_TRAIN_MEAN_P: float = _MIN_P

DEFAULT_EVAL_RUNTIME_REGRESSION_FACTOR: float = 10.0
DEFAULT_EVAL_TRAIN_RETRIES: int = 3

_OBJECTIVE_LABELS = {
    "bm24_mean_p_fixed_n": "maximize mean(p_succ) at fixed train_n",
    "median_runtime_fixed_n": "minimize median(1/(p_succ+eps)) at fixed train_n",
    "mean_log_runtime_fixed_n": "minimize mean(ln(1/(p_succ+eps))) at fixed train_n",
}

# Per-mode COBYLA RNG stream offsets (compare script / retries).
MODE_RNG_OFFSET: Dict[str, int] = {
    "bm24_mean_p_fixed_n": 0,
    "median_runtime_fixed_n": 99_999,
    "mean_log_runtime_fixed_n": 199_998,
}


# --------------------------------------------------------------------------- #
# Instance generation                                                          #
# --------------------------------------------------------------------------- #

def generate_training_instances(
    train_n: int,
    k: int,
    r: float,
    train_size: int,
    base_seed: int,
) -> List[np.ndarray]:
    """
    SAT-filtered random k-SAT H_diag arrays at a single fixed train_n.
    Clause count: m = max(1, Poisson(r * n))  [notebook-style generator].
    Seeds are fully deterministic given (base_seed, train_n, instance index).
    """
    h_list: List[np.ndarray] = []
    trial = 0
    while len(h_list) < int(train_size):
        ss = np.random.SeedSequence([int(base_seed), int(train_n), len(h_list), trial])
        rng = np.random.default_rng(ss)
        m = max(1, int(rng.poisson(float(r) * int(train_n))))
        clauses = [generate_random_clause(train_n, k, rng) for _ in range(m)]
        h_diag = build_h_diagonal(clauses, train_n)
        if np.any(h_diag == 0):
            h_list.append(h_diag)
        trial += 1
        if trial > 1_000_000:
            raise RuntimeError(
                f"Exceeded 1e6 trials building training instances at n={train_n}"
            )
    return h_list


# --------------------------------------------------------------------------- #
# Objectives                                                                   #
# --------------------------------------------------------------------------- #

def _compute_objective(
    training_mode: str,
    instances: Sequence[np.ndarray],
    n: int,
    betas: np.ndarray,
    gammas: np.ndarray,
    eps: float = DEFAULT_EPS,
) -> float:
    """
    Scalar to MINIMIZE for the given training mode.

    "bm24_mean_p_fixed_n":    returns -mean(p_succ) over instances.
                               Maximizing mean p_succ <=> minimizing this.

    "median_runtime_fixed_n":    returns median(1/(p_succ+eps)) over instances.
    "mean_log_runtime_fixed_n":  returns mean(ln(1/(p_succ+eps))) over instances.
    """
    if not instances:
        return float("nan")
    if training_mode == "bm24_mean_p_fixed_n":
        total = 0.0
        for h in instances:
            psi = run_qaoa(h, betas, gammas, int(n))
            total += float(per_instance_success_probability(psi, h))
        return -(total / len(instances))
    if training_mode == "median_runtime_fixed_n":
        costs = np.empty(len(instances), dtype=np.float64)
        for i, h in enumerate(instances):
            psi = run_qaoa(h, betas, gammas, int(n))
            p = float(per_instance_success_probability(psi, h))
            costs[i] = 1.0 / max(p, float(eps))
        return float(np.median(costs))
    if training_mode == "mean_log_runtime_fixed_n":
        total = 0.0
        for h in instances:
            psi = run_qaoa(h, betas, gammas, int(n))
            p = float(per_instance_success_probability(psi, h))
            total += float(np.log(1.0 / max(p, float(eps))))
        return total / len(instances)
    else:
        raise ValueError(
            f"Unknown training_mode {training_mode!r}. "
            f"Supported: {sorted(SUPPORTED_TRAINING_MODES)}"
        )


def _mean_p_succ(
    instances: Sequence[np.ndarray],
    n: int,
    betas: np.ndarray,
    gammas: np.ndarray,
) -> float:
    """Diagnostic: mean p_succ. Used as the collapse-guard metric for both modes."""
    if not instances:
        return 0.0
    total = 0.0
    for h in instances:
        psi = run_qaoa(h, betas, gammas, int(n))
        total += float(per_instance_success_probability(psi, h))
    return total / len(instances)


# --------------------------------------------------------------------------- #
# COBYLA helpers                                                               #
# --------------------------------------------------------------------------- #

def _box_constraints(
    dg_bounds: Tuple[float, float], db_bounds: Tuple[float, float]
) -> list:
    lo_dg, hi_dg = dg_bounds
    lo_db, hi_db = db_bounds
    return [
        {"type": "ineq", "fun": lambda x: x[0] - lo_dg},
        {"type": "ineq", "fun": lambda x: hi_dg - x[0]},
        {"type": "ineq", "fun": lambda x: x[1] - lo_db},
        {"type": "ineq", "fun": lambda x: hi_db - x[1]},
    ]


def _clip(
    x: np.ndarray,
    dg_bounds: Tuple[float, float],
    db_bounds: Tuple[float, float],
) -> np.ndarray:
    out = np.asarray(x, dtype=float).copy()
    out[0] = float(np.clip(out[0], dg_bounds[0], dg_bounds[1]))
    out[1] = float(np.clip(out[1], db_bounds[0], db_bounds[1]))
    return out


def _adaptive_rhobeg(
    depth: int,
    dg_bounds: Tuple[float, float],
    db_bounds: Tuple[float, float],
) -> float:
    """Shrink initial COBYLA trust region as depth grows (angles get smaller)."""
    span = 0.5 * ((dg_bounds[1] - dg_bounds[0]) + (db_bounds[1] - db_bounds[0]))
    factor = 0.20 / max(1.0, math.log2(max(2, int(depth))) / 2.0)
    return float(max(0.03, min(0.20, factor)) * span)


def _run_cobyla(
    objective,
    x0: np.ndarray,
    constraints: list,
    maxiter: int,
    tol: float,
    rhobeg: float,
    dg_bounds: Tuple[float, float],
    db_bounds: Tuple[float, float],
) -> Tuple[np.ndarray, float]:
    res = minimize(
        objective, x0, method="COBYLA",
        constraints=constraints, tol=tol,
        options={"maxiter": maxiter, "rhobeg": rhobeg, "catol": 1e-6, "disp": False},
    )
    return _clip(res.x, dg_bounds, db_bounds), float(res.fun)


# --------------------------------------------------------------------------- #
# Main training function                                                       #
# --------------------------------------------------------------------------- #

def train_angles_fixed_n(
    training_mode: str,
    train_n: int,
    depth: int,
    instances: List[np.ndarray],
    *,
    initial_angles: Optional[Tuple[float, float]] = None,
    beta_schedule: str = "decreasing",
    dg_bounds: Tuple[float, float] = DEFAULT_DG_BOUNDS,
    db_bounds: Tuple[float, float] = DEFAULT_DB_BOUNDS,
    cobyla_maxiter: int = 200,
    cobyla_tol: float = 1e-3,
    cobyla_rhobeg: Optional[float] = None,
    cobyla_restarts: int = 8,
    cobyla_perturb_scale: float = 0.2,
    grid_top_k: int = 5,
    skip_grid: bool = False,
    skip_grid_if_warm_start: bool = True,
    anti_regression: bool = True,
    collapse_guard: bool = True,
    eps: float = DEFAULT_EPS,
    rng: Optional[np.random.Generator] = None,
    verbose: bool = True,
) -> Tuple[np.ndarray, dict]:
    """
    Train LR-QAOA (delta_gamma, delta_beta) at a single fixed train_n.

    training_mode must be one of SUPPORTED_TRAINING_MODES:
      "bm24_mean_p_fixed_n"       — maximize mean p_succ (BM24 baseline).
      "median_runtime_fixed_n"    — minimize median(1/(p_succ+eps)).
      "mean_log_runtime_fixed_n"  — minimize mean(ln(1/(p_succ+eps))).

    Optimizer: optional 11x11 grid scan + multi-restart COBYLA with
    anti-regression and collapse guards.

    Returns
    -------
    params_concat : np.ndarray
        np.concatenate([gammas, betas]) of length 2*depth in BM24 convention.
    diag : dict
        Metadata and diagnostics (see module docstring for full key list).
    """
    if training_mode not in SUPPORTED_TRAINING_MODES:
        raise ValueError(
            f"training_mode={training_mode!r} not in {sorted(SUPPORTED_TRAINING_MODES)}"
        )
    train_n = int(train_n)
    depth = int(depth)
    warm = initial_angles is not None
    gen = rng if rng is not None else np.random.default_rng()
    constraints = _box_constraints(dg_bounds, db_bounds)
    rhobeg = (
        float(cobyla_rhobeg) if cobyla_rhobeg is not None
        else _adaptive_rhobeg(depth, dg_bounds, db_bounds)
    )

    if verbose:
        print(
            f"  > train_angles_fixed_n: mode={training_mode!r} "
            f"n={train_n} depth={depth} instances={len(instances)}"
        )

    def angles(dg: float, db: float) -> Tuple[np.ndarray, np.ndarray]:
        return make_lr_angles(dg, db, depth, beta_schedule=beta_schedule, angle_convention="bm24")

    def obj(x: np.ndarray) -> float:
        betas_, gammas_ = angles(float(x[0]), float(x[1]))
        return _compute_objective(training_mode, instances, train_n, betas_, gammas_, eps=eps)

    # ---- Warm-start point (computed once, reused below) ---- #
    warm_x: Optional[np.ndarray] = None
    warm_obj: Optional[float] = None
    if warm:
        warm_x = _clip(
            np.array([float(initial_angles[0]), float(initial_angles[1])]),
            dg_bounds, db_bounds,
        )
        warm_obj = obj(warm_x)
        if verbose:
            print(
                f"  > Warm-start: dg={warm_x[0]:+.4f} db={warm_x[1]:+.4f} "
                f"obj={warm_obj:+.6f}"
            )

    # ---- Grid scan ---- #
    do_grid = not skip_grid and not (warm and skip_grid_if_warm_start)
    grid_scores: List[Tuple[float, float, float]] = []  # (score, dg, db)
    n_grid_candidates = 0

    if do_grid:
        dg_grid = np.linspace(dg_bounds[0], dg_bounds[1], 11)
        db_grid = np.linspace(db_bounds[0], db_bounds[1], 11)
        for dg_i in tqdm(dg_grid, desc="Grid (dg)", disable=not verbose):
            for db_i in db_grid:
                sc = obj(np.array([float(dg_i), float(db_i)]))
                grid_scores.append((sc, float(dg_i), float(db_i)))
        grid_scores.sort(key=lambda t: t[0])
        n_grid_candidates = max(1, min(int(grid_top_k), len(grid_scores)))
        if verbose:
            print(
                f"  > Best grid: dg={grid_scores[0][1]:+.4f} "
                f"db={grid_scores[0][2]:+.4f} obj={grid_scores[0][0]:+.6f}"
            )
    else:
        label = "warm-start" if warm else "default"
        fallback_x = warm_x if warm_x is not None else _clip(
            np.array(list(DEFAULT_INITIAL_DELTAS)), dg_bounds, db_bounds
        )
        fallback_obj = warm_obj if warm_obj is not None else obj(fallback_x)
        if verbose:
            print(
                f"  > Grid skipped ({label}): dg={fallback_x[0]:+.4f} "
                f"db={fallback_x[1]:+.4f} obj={fallback_obj:+.6f}"
            )

    # ---- Assemble COBYLA restart points ---- #
    starts: List[np.ndarray] = []
    seen: set = set()

    def _add(dg: float, db: float) -> None:
        x = _clip(np.array([dg, db]), dg_bounds, db_bounds)
        key = (round(float(x[0]), 8), round(float(x[1]), 8))
        if key not in seen:
            starts.append(x)
            seen.add(key)

    if warm and warm_x is not None:
        _add(float(warm_x[0]), float(warm_x[1]))
    for sc, dg_i, db_i in grid_scores[:n_grid_candidates]:
        _add(float(dg_i), float(db_i))
    if not starts:
        default_x = _clip(np.array(list(DEFAULT_INITIAL_DELTAS)), dg_bounds, db_bounds)
        _add(float(default_x[0]), float(default_x[1]))

    n_perturb = max(0, int(cobyla_restarts) - len(starts))
    anchor = starts[0]
    span_dg = dg_bounds[1] - dg_bounds[0]
    span_db = db_bounds[1] - db_bounds[0]
    for _ in range(n_perturb):
        noise = gen.normal(0.0, float(cobyla_perturb_scale), size=2)
        _add(
            float(anchor[0] + noise[0] * span_dg),
            float(anchor[1] + noise[1] * span_db),
        )

    # ---- COBYLA restarts ---- #
    best_x = starts[0].copy()
    best_obj_val = obj(best_x)
    if warm and warm_x is not None and warm_obj is not None and warm_obj < best_obj_val:
        best_x = warm_x.copy()
        best_obj_val = float(warm_obj)

    restart_records: List[dict] = []
    for i, x0 in enumerate(starts):
        x_fin, obj_fin = _run_cobyla(
            obj, x0, constraints, cobyla_maxiter, cobyla_tol, rhobeg, dg_bounds, db_bounds,
        )
        restart_records.append({
            "restart_idx": i,
            "x0": x0.tolist(),
            "x_final": x_fin.tolist(),
            "obj": obj_fin,
        })
        if obj_fin < best_obj_val:
            best_obj_val = obj_fin
            best_x = x_fin
        if verbose and (i == 0 or obj_fin <= best_obj_val + 1e-12):
            print(
                f"  > COBYLA restart {i:2d}: obj={obj_fin:+.6f} "
                f"x={np.round(x_fin, 4).tolist()}"
            )

    # ---- Anti-regression: never worse than warm-start ---- #
    anti_regression_applied = False
    if anti_regression and warm and warm_x is not None and warm_obj is not None:
        if best_obj_val > warm_obj + 1e-12 * max(1.0, abs(warm_obj)):
            anti_regression_applied = True
            best_x = warm_x.copy()
            best_obj_val = float(warm_obj)
            if verbose:
                print(
                    f"  > Anti-regression: reverted to warm-start (obj={warm_obj:+.6f})"
                )

    # ---- Collapse guard: reject solutions with mean_p near zero ---- #
    trained_x = best_x.copy()  # record what training produced before any fallback
    dg_opt, db_opt = float(best_x[0]), float(best_x[1])
    betas_opt, gammas_opt = angles(dg_opt, db_opt)
    final_mean_p = _mean_p_succ(instances, train_n, betas_opt, gammas_opt)

    warm_mean_p: Optional[float] = None
    if warm and warm_x is not None:
        betas_w, gammas_w = angles(float(warm_x[0]), float(warm_x[1]))
        warm_mean_p = _mean_p_succ(instances, train_n, betas_w, gammas_w)

    train_rejected = False
    train_reject_reason = ""
    if float(dg_opt) > 0.0:
        train_rejected = True
        train_reject_reason = f"dg={dg_opt:+.4f} > 0"
    elif float(final_mean_p) < float(MIN_TRAIN_MEAN_P):
        train_rejected = True
        train_reject_reason = f"mean_p={final_mean_p:.4e} < {MIN_TRAIN_MEAN_P:g}"

    collapse_guard_applied = False
    if collapse_guard and train_rejected:
        fallback_x: Optional[np.ndarray] = None
        if (
            warm_x is not None
            and warm_mean_p is not None
            and float(warm_mean_p) >= float(MIN_TRAIN_MEAN_P)
        ):
            fallback_x = warm_x.copy()

        if fallback_x is not None:
            collapse_guard_applied = True
            best_x = fallback_x
            dg_opt, db_opt = float(best_x[0]), float(best_x[1])
            betas_opt, gammas_opt = angles(dg_opt, db_opt)
            final_mean_p = _mean_p_succ(instances, train_n, betas_opt, gammas_opt)
            best_obj_val = obj(best_x)
            if verbose:
                print(
                    f"  > Collapse guard: reverted to warm-start "
                    f"({train_reject_reason}) -> dg={dg_opt:+.4f} db={db_opt:+.4f}"
                )
        elif verbose:
            print(
                f"  > Collapse guard triggered ({train_reject_reason}); "
                "no acceptable warm start available"
            )
    elif train_rejected and verbose:
        print(
            f"  > Train reject flagged ({train_reject_reason}); "
            "collapse guard disabled — keeping COBYLA result"
        )

    params_concat = np.concatenate([gammas_opt, betas_opt])

    if verbose:
        print(
            f"  > Result: dg={dg_opt:+.6f} db={db_opt:+.6f} "
            f"obj={best_obj_val:+.6f} mean_p@train_n={final_mean_p:.4e}"
        )

    diag = {
        # Mode metadata — always emitted, never misleading
        "training_mode": training_mode,
        "train_n": train_n,
        "depth": depth,
        "objective": _OBJECTIVE_LABELS[training_mode],
        "simulator": "bm24",
        "uses_multi_n_training": False,
        "eval_slope_only": True,
        # Angle results
        "best_deltas": [dg_opt, db_opt],
        "trained_deltas": [float(trained_x[0]), float(trained_x[1])],
        # Training performance
        "best_train_objective": float(best_obj_val),
        "best_avg_train_p_succ": float(final_mean_p),
        # Guards
        "train_rejected": bool(train_rejected),
        "train_reject_reason": train_reject_reason,
        "collapse_guard_applied": bool(collapse_guard_applied),
        "anti_regression_applied": bool(anti_regression_applied),
        "warm_start_used": bool(warm),
        # Optimizer details
        "grid_skipped": not do_grid,
        "cobyla_restarts": int(len(starts)),
        "cobyla_rhobeg_used": float(rhobeg),
        "restart_records": restart_records,
    }
    return params_concat, diag


# --------------------------------------------------------------------------- #
# Eval-side regression helpers                                                 #
# --------------------------------------------------------------------------- #

def eval_median_runtime_reject(
    med_rt: Dict[int, float],
    prev_med_rt: Optional[Dict[int, float]],
    n_min: int,
    n_max: int,
    *,
    factor: float = DEFAULT_EVAL_RUNTIME_REGRESSION_FACTOR,
) -> Tuple[bool, str]:
    """
    Return (reject, reason) when median(1/p) at n_min or n_max exceeds
    ``factor`` times the previous depth's value at the same n.
    """
    if prev_med_rt is None:
        return False, ""
    n_lo, n_hi = int(n_min), int(n_max)

    def _get(d: Dict[int, float], n: int) -> float:
        if n in d:
            return float(d[n])
        if str(n) in d:
            return float(d[str(n)])
        return float("nan")

    cur_lo, cur_hi = _get(med_rt, n_lo), _get(med_rt, n_hi)
    prev_lo, prev_hi = _get(prev_med_rt, n_lo), _get(prev_med_rt, n_hi)
    reasons: List[str] = []
    if np.isfinite(prev_lo) and prev_lo > 0 and np.isfinite(cur_lo):
        if cur_lo > factor * prev_lo:
            reasons.append(f"n_min={n_lo}: {cur_lo:.4g} > {factor:g}× prev {prev_lo:.4g}")
    if np.isfinite(prev_hi) and prev_hi > 0 and np.isfinite(cur_hi):
        if cur_hi > factor * prev_hi:
            reasons.append(f"n_max={n_hi}: {cur_hi:.4g} > {factor:g}× prev {prev_hi:.4g}")
    if reasons:
        return True, "; ".join(reasons)
    return False, ""


def run_train_eval_with_retries(
    *,
    train_at_depth,
    evaluate_at_angles,
    prev_med_rt: Optional[Dict[int, float]],
    prev_accepted_deltas: Optional[Tuple[float, float]],
    n_min: int,
    n_max: int,
    max_retries: int = DEFAULT_EVAL_TRAIN_RETRIES,
    regression_factor: float = DEFAULT_EVAL_RUNTIME_REGRESSION_FACTOR,
    check_eval_regression: bool = True,
    fallback_to_previous_angles: bool = True,
    verbose: bool = True,
) -> dict:
    """
    Train, benchmark-evaluate, and on eval regression re-call ``train_at_depth``
    up to ``max_retries`` times before falling back to ``prev_accepted_deltas``.

    ``train_at_depth(retry_index)`` must return ``(dg, db, diag)``.
    ``evaluate_at_angles(dg, db)`` must return median(1/p_succ) per n.
    """
    max_retries = max(0, int(max_retries))
    last_diag: dict = {}
    last_reject_reason = ""

    for attempt in range(max_retries + 1):
        if attempt > 0 and verbose:
            print(
                f"  eval regression: re-training "
                f"(attempt {attempt + 1}/{max_retries + 1}) ..."
            )
        dg, db, diag = train_at_depth(int(attempt))
        last_diag = diag
        med_rt = evaluate_at_angles(float(dg), float(db))
        if check_eval_regression:
            reject, reject_reason = eval_median_runtime_reject(
                med_rt, prev_med_rt, int(n_min), int(n_max), factor=float(regression_factor),
            )
        else:
            reject, reject_reason = False, ""
        if not reject:
            return {
                "dg": float(dg), "db": float(db), "diag": diag, "med_rt": med_rt,
                "eval_rejected": False, "eval_reject_reason": None,
                "eval_train_attempts": int(attempt) + 1, "used_previous_angles": False,
            }
        last_reject_reason = reject_reason
        if verbose:
            print(f"  REJECT eval regression: {reject_reason}")

    used_previous = False
    if fallback_to_previous_angles and prev_accepted_deltas is not None:
        dg, db = float(prev_accepted_deltas[0]), float(prev_accepted_deltas[1])
        med_rt = evaluate_at_angles(dg, db)
        used_previous = True
        if verbose:
            print(
                f"  eval regression: kept prior angles after "
                f"{max_retries + 1} train attempt(s): dg={dg:.6f} db={db:.6f}"
            )
    else:
        dg, db = float(last_diag["best_deltas"][0]), float(last_diag["best_deltas"][1])
        med_rt = evaluate_at_angles(dg, db)
        if verbose:
            print(
                "  eval regression: no prior accepted angles; "
                f"keeping last train dg={dg:.6f} db={db:.6f}"
            )

    return {
        "dg": float(dg), "db": float(db), "diag": last_diag, "med_rt": med_rt,
        "eval_rejected": True, "eval_reject_reason": last_reject_reason,
        "eval_train_attempts": int(max_retries) + 1, "used_previous_angles": used_previous,
    }


# --------------------------------------------------------------------------- #
# Sanity checks                                                                #
# --------------------------------------------------------------------------- #

def verify_objectives(n: int = 4, depth: int = 1, seed: int = 42) -> None:
    """
    Verify objective implementations against direct hand-computed values.
    Uses n=4, depth=1 for speed. Raises AssertionError on failure.

    Checks:
    1. bm24_mean_p_fixed_n objective == -mean(p_succ) over instances.
    2. median_runtime_fixed_n objective == median(1/(p_succ+eps)) over instances.
    3. mean_log_runtime_fixed_n objective == mean(ln(1/(p_succ+eps))) over instances.
    4. train_angles_fixed_n always sets uses_multi_n_training=False.
    5. train_angles_fixed_n always sets eval_slope_only=True.
    """
    rng = np.random.default_rng(seed)
    k, r = 2, 2.0
    m = max(1, int(rng.poisson(r * n)))
    clauses = [generate_random_clause(n, k, rng) for _ in range(m)]
    h = build_h_diagonal(clauses, n)
    instances = [h]

    dg, db = -0.5, 1.0
    eps = 1e-12
    betas, gammas = make_lr_angles(dg, db, depth, angle_convention="bm24")
    psi = run_qaoa(h, betas, gammas, n)
    p = float(per_instance_success_probability(psi, h))

    # Check 1: bm24_mean_p_fixed_n = -mean(p_succ)
    obj_mean = _compute_objective("bm24_mean_p_fixed_n", instances, n, betas, gammas, eps=eps)
    expected = -p  # single instance: mean = p itself
    assert abs(obj_mean - expected) < 1e-10, (
        f"bm24_mean_p_fixed_n: got {obj_mean:.12g}, expected {expected:.12g}"
    )

    # Check 2: median_runtime_fixed_n = median(1/(p+eps))
    obj_med = _compute_objective("median_runtime_fixed_n", instances, n, betas, gammas, eps=eps)
    expected_med = 1.0 / max(p, eps)
    assert abs(obj_med - expected_med) < 1e-10, (
        f"median_runtime_fixed_n: got {obj_med:.12g}, expected {expected_med:.12g}"
    )

    # Check 3: mean_log_runtime_fixed_n = ln(1/(p+eps))  (single instance)
    obj_mlog = _compute_objective(
        "mean_log_runtime_fixed_n", instances, n, betas, gammas, eps=eps,
    )
    expected_mlog = float(np.log(1.0 / max(p, eps)))
    assert abs(obj_mlog - expected_mlog) < 1e-10, (
        f"mean_log_runtime_fixed_n: got {obj_mlog:.12g}, expected {expected_mlog:.12g}"
    )

    # Check 4 & 5: diag flags
    _, diag = train_angles_fixed_n(
        "bm24_mean_p_fixed_n", n, depth, instances,
        skip_grid=True, cobyla_restarts=1, cobyla_maxiter=5, verbose=False,
    )
    assert diag["uses_multi_n_training"] is False, (
        f"uses_multi_n_training should be False, got {diag['uses_multi_n_training']}"
    )
    assert diag["eval_slope_only"] is True, (
        f"eval_slope_only should be True, got {diag['eval_slope_only']}"
    )

    print("verify_objectives: all 5 checks passed.")


if __name__ == "__main__":
    verify_objectives()
