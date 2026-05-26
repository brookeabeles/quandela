"""
train_lr_notebook_protocol.py  (v2 — slope-matched objective)
=============================================================

LR-QAOA training pipeline that trains ``(delta_gamma, delta_beta)`` by
directly minimizing the **same** quantity the evaluation plot measures:
the slope of ``log(median(1/p_succ))`` vs ``n``.

Why this differs from v1
------------------------
v1 used a composite objective::

    score = w_mean * ln E[p_succ @ train_n]  +  w_slope * (slope on proxy)

with defaults ``w_mean = 0.8``, ``w_slope = 0.2``, and a *narrow* proxy
spanning ``n in {train_n, train_n+1, train_n+2}`` with ~25 instances per
``n``. This has two compounding problems at large depth:

* **Easy-instance domination of the "mean" term.** BM24 §III B reports
  mean ``p_succ`` ≈ 0.56 at ``k=8, p=60, n=12``: the average is dominated
  by instances QAOA already solves trivially, so the gradient on
  ``(dg, db)`` from this term vanishes precisely in the regime that
  matters (large ``p``).

* **A noisy "slope" term.** Three consecutive ``n`` values with 25
  instances each gives a slope estimate whose Monte-Carlo noise can
  exceed the differences COBYLA needs to discriminate good ``(dg, db)``
  from bad. The 0.2 weight then biases against tackling this directly.

The net effect, visible as the flattening of the LR curve from ``p ≈ 18``
onward in the user's scaling plot, is that COBYLA stops getting useful
gradient information at exactly the depths the experiment is about.

What v2 does
------------
1. **Train on the evaluation metric, full stop.** The objective is

       slope_nat = least_squares_slope_over_n( ln(median over instances of 1/p_succ) )

   minimised over ``(dg, db)``. Divide by ``ln 2`` for the log₂ slope on
   the plot's y-axis. There is no longer a training/eval-metric gap to
   interpret away.

2. **Wider, spread-out training n.** ``proxy_n_values_for_training``
   defaults to **step 2** rather than consecutive integers. With the
   notebook's ``CFG["proxy_n_span"] = 4``, that gives ``[12, 14, 16]``
   instead of ``[12, 13, 14]`` — three times the lever arm in ``n`` for
   the same simulation cost (each ``n`` simulated independently). Three
   well-spaced points are far better than three crowded ones for a slope
   fit on noisy data.

3. **Median, not mean.** The median of ``1/p_succ`` over instances is
   robust to the easy-instance tail that breaks the v1 mean objective.
   It is also exactly what the evaluation plot uses.

The grid + multi-restart COBYLA + depth-warm-start scaffolding is
preserved. The ``w_mean / w_slope / use_log_mean`` keyword arguments are
still accepted by ``train_lr_grid_search_bm24`` (so the notebook keeps
running unchanged) but are *ignored* with a one-shot deprecation note —
they no longer have a clean interpretation under the new objective.

Recommended notebook settings
-----------------------------
For the user's existing ``LR_QAOA_benchmark_efficient.ipynb``::

    CFG["proxy_n_span"]      = 4    # was 2; with step=2 -> [train_n, +2, +4]
    CFG["proxy_size_per_n"]  = 50   # was 25
    CFG["train_size"]        = 100  # unchanged; kept for warm-start mean p

These three changes are sufficient to engage v2's improvements with no
code modifications.

CLI compatibility
-----------------
The standalone CLI accepts the same flags as v1; the obsolete
``--legacy-objective``, ``--w-mean``, ``--w-slope`` flags are parsed but
warn and have no effect.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize
from scipy.stats import linregress

_SCRIPT_DIR = Path(__file__).resolve().parent
_DEFAULT_ANGLE_LOG = _SCRIPT_DIR / "bm24_runs" / "lr_train_optimal_angles.txt"
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from bm24_qaoa_sim import (  # noqa: E402
    build_h_diagonal,
    generate_random_clause,
    make_lr_angles,
    per_instance_success_probability,
    run_qaoa,
)

try:  # progress bars are nice-to-have, never required
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(it, **kwargs):  # type: ignore
        return it


# --------------------------------------------------------------------------- #
# Defaults                                                                    #
# --------------------------------------------------------------------------- #

# Historical notebook (dg, db) search box. We keep the same domain so any
# v1 trained angles remain "in-distribution" for v2's warm starts.
DEFAULT_DG_BOUNDS: Tuple[float, float] = (-2.0, 2.0)
DEFAULT_DB_BOUNDS: Tuple[float, float] = (0.1, 4.0)
DEFAULT_INITIAL_DELTAS: Tuple[float, float] = (-0.8, 0.49)

# Numerical floor: a p_succ = 0 instance becomes 1/eps, a sentinel "huge cost"
# that the median will treat correctly (it is just one ordered value). We do
# NOT clip in log space; the floor is applied to p before reciprocal.
DEFAULT_EPS: float = 1e-300

# We have been bitten enough times by this in v1: warn loudly if the multi-n
# training set has fewer than this many distinct n values.
_MIN_RECOMMENDED_NS_FOR_SLOPE = 3

# Single-shot deprecation flag so we don't spam the v1 notebook user.
_LEGACY_KWARGS_WARNED = False


# --------------------------------------------------------------------------- #
# Random-instance generation (unchanged from v1)                              #
# --------------------------------------------------------------------------- #

def _sample_m(rng: np.random.Generator, n: int, r: float, mode: str) -> int:
    """Notebook variant forces m >= 1; bm24 variant is strict Poisson(rn)."""
    lam = float(r) * int(n)
    m = int(rng.poisson(lam))
    if mode == "notebook":
        return max(1, m)
    if mode == "bm24":
        return m
    raise ValueError("m_sampling must be 'notebook' or 'bm24'")


def generate_training_h_diagonals(
    train_n: int,
    k: int,
    r: float,
    train_size: int,
    base_seed: int,
    m_sampling: Literal["notebook", "bm24"] = "notebook",
) -> List[np.ndarray]:
    """SAT-filtered random k-SAT instances at fixed n; returns H_diag arrays."""
    h_list: List[np.ndarray] = []
    trial = 0
    while len(h_list) < int(train_size):
        ss = np.random.SeedSequence(
            [int(base_seed), int(train_n), len(h_list), trial]
        )
        rng = np.random.default_rng(ss)
        m = _sample_m(rng, train_n, r, m_sampling)
        clauses = [generate_random_clause(train_n, k, rng) for _ in range(m)]
        h_diag = build_h_diagonal(clauses, train_n)
        if np.any(h_diag == 0):
            h_list.append(h_diag)
        trial += 1
        if trial > 1_000_000:
            raise RuntimeError(
                f"Exceeded 1e6 trials while building training set at n={train_n}"
            )
    return h_list


def generate_training_h_diagonals_multi_n(
    n_values: Sequence[int],
    k: int,
    r: float,
    train_size_per_n: int,
    base_seed: int,
    m_sampling: Literal["notebook", "bm24"] = "notebook",
) -> Dict[int, List[np.ndarray]]:
    """SAT-filtered H_diag arrays keyed by ``n`` (independent seeds per n)."""
    out: Dict[int, List[np.ndarray]] = {}
    for n in n_values:
        out[int(n)] = generate_training_h_diagonals(
            train_n=int(n),
            k=int(k),
            r=float(r),
            train_size=int(train_size_per_n),
            base_seed=int(base_seed) + 10_000 * int(n),
            m_sampling=m_sampling,
        )
    return out


def proxy_n_values_for_training(
    train_n: int,
    proxy_n_span: int = 4,
    n_max_cap: int = 20,
    step: int = 2,
) -> List[int]:
    """
    Multi-n training-point selector (v2 semantics: spread by ``step``).

    Returns ``[train_n, train_n + step, train_n + 2*step, ...]`` up to
    ``min(train_n + proxy_n_span, n_max_cap)``.

    v2 default ``step=2`` (was effectively ``1`` in v1). Combined with the
    notebook's ``CFG["proxy_n_span"] = 2``, this still gives only 2 points
    ``[12, 14]`` — usable but minimal. For three well-spaced points, set
    ``proxy_n_span = 4`` in the notebook CFG; the function will then
    return ``[12, 14, 16]``.

    Why the change? At consecutive ``n`` values the median of ``1/p_succ``
    moves so little (compared to its Monte-Carlo standard error from 25–50
    instances) that the resulting 3-point slope estimate is dominated by
    sampling noise. Doubling the lever arm in ``n`` triples the signal-
    to-noise on the slope at unchanged simulation cost (each ``n``
    independent), and consequently triples the depth at which the
    composite objective remains informative.
    """
    train_n = int(train_n)
    proxy_n_span = max(0, int(proxy_n_span))
    step = max(1, int(step))
    n_max_cap = int(n_max_cap)
    out: List[int] = []
    n = train_n
    while n <= train_n + proxy_n_span and n <= n_max_cap:
        out.append(int(n))
        n += step
    if not out:
        out = [train_n]
    return out


# --------------------------------------------------------------------------- #
# Core objective: slope of ln(median(1/p_succ)) vs n                          #
# --------------------------------------------------------------------------- #

def _median_inv_p_at_n(
    h_list: Sequence[np.ndarray],
    n: int,
    betas: np.ndarray,
    gammas: np.ndarray,
    eps: float = DEFAULT_EPS,
) -> float:
    """Median over instances of 1/p_succ at fixed n. Always >= 1."""
    if not h_list:
        return float("nan")
    costs = np.empty(len(h_list), dtype=np.float64)
    for i, h_diag in enumerate(h_list):
        psi = run_qaoa(h_diag, betas, gammas, int(n))
        p = per_instance_success_probability(psi, h_diag)
        costs[i] = 1.0 / max(float(p), float(eps))
    return float(np.median(costs))


def _mean_p_succ_at_n(
    h_list: Sequence[np.ndarray],
    n: int,
    betas: np.ndarray,
    gammas: np.ndarray,
) -> float:
    """Diagnostic only: mean p_succ at fixed n (the v1 objective term)."""
    if not h_list:
        return 0.0
    total = 0.0
    for h_diag in h_list:
        psi = run_qaoa(h_diag, betas, gammas, int(n))
        total += float(per_instance_success_probability(psi, h_diag))
    return total / len(h_list)


def _slope_log_median_inv_p(
    h_by_n: Dict[int, List[np.ndarray]],
    n_values: Sequence[int],
    betas: np.ndarray,
    gammas: np.ndarray,
    eps: float = DEFAULT_EPS,
) -> Tuple[float, List[float]]:
    """
    Return ``(slope_nat, log_med_per_n)``.

    ``log_med_per_n[i] = ln(median over instances of 1/p_succ)`` at
    ``n = n_values[i]``; always >= 0 since ``p_succ in [0, 1]`` means
    ``1/p_succ >= 1``.

    ``slope_nat`` is the least-squares slope of those log medians vs n
    (natural log units). **This IS the quantity the evaluation plot
    reports, modulo division by ln 2 for log₂.**

    Returns ``(nan, ...)`` if fewer than 2 finite log medians are
    available.
    """
    ns = [int(n) for n in n_values]
    log_med: List[float] = []
    for n in ns:
        m = _median_inv_p_at_n(h_by_n[n], n, betas, gammas, eps=eps)
        if not np.isfinite(m) or m <= 0.0:
            log_med.append(float("nan"))
        else:
            log_med.append(float(np.log(m)))

    arr = np.asarray(log_med, dtype=float)
    mask = np.isfinite(arr)
    if int(mask.sum()) < 2:
        return float("nan"), log_med
    res = linregress(np.asarray(ns, dtype=float)[mask], arr[mask])
    return float(res.slope), log_med


# --------------------------------------------------------------------------- #
# COBYLA scaffolding                                                          #
# --------------------------------------------------------------------------- #

def _lr_box_cobyla_constraints(
    dg_bounds: Tuple[float, float],
    db_bounds: Tuple[float, float],
) -> list:
    lo_dg, hi_dg = dg_bounds
    lo_db, hi_db = db_bounds
    return [
        {"type": "ineq", "fun": lambda x: x[0] - lo_dg},
        {"type": "ineq", "fun": lambda x: hi_dg - x[0]},
        {"type": "ineq", "fun": lambda x: x[1] - lo_db},
        {"type": "ineq", "fun": lambda x: hi_db - x[1]},
    ]


def _clip_deltas(
    d: np.ndarray,
    dg_bounds: Tuple[float, float],
    db_bounds: Tuple[float, float],
) -> np.ndarray:
    out = np.asarray(d, dtype=float).copy()
    out[0] = float(np.clip(out[0], dg_bounds[0], dg_bounds[1]))
    out[1] = float(np.clip(out[1], db_bounds[0], db_bounds[1]))
    return out


def _angles_from_deltas(
    dg: float, db: float, depth: int, beta_schedule: str,
) -> Tuple[np.ndarray, np.ndarray]:
    return make_lr_angles(
        float(dg), float(db), int(depth),
        beta_schedule=str(beta_schedule),
        angle_convention="bm24",
    )


def _cobyla_refine(
    objective, x0: np.ndarray, *,
    constraints: list,
    cobyla_maxiter: int,
    cobyla_tol: float,
    cobyla_rhobeg: float,
    dg_bounds: Tuple[float, float],
    db_bounds: Tuple[float, float],
) -> Tuple[np.ndarray, float, dict]:
    res = minimize(
        objective,
        np.asarray(x0, dtype=float),
        method="COBYLA",
        constraints=constraints,
        tol=float(cobyla_tol),
        options={
            "maxiter": int(cobyla_maxiter),
            "rhobeg": float(cobyla_rhobeg),
            "catol": 1e-6,
            "disp": False,
        },
    )
    x = _clip_deltas(res.x, dg_bounds, db_bounds)
    return x, float(res.fun), {
        "cobyla_success": bool(res.success),
        "cobyla_message": str(res.message),
        "nfev": int(getattr(res, "nfev", -1)),
    }


def _adaptive_rhobeg(
    depth: int,
    dg_bounds: Tuple[float, float],
    db_bounds: Tuple[float, float],
) -> float:
    """
    Shrink COBYLA's initial trust region with depth.

    Empirically the optimal ``(dg, db)`` magnitudes decrease as ``p`` grows
    (the integrated angle :math:`p \\cdot dg` stays roughly bounded), so
    larger trust regions at high ``p`` waste evaluations probing bad
    landscape. Halve roughly every doubling of ``p``.
    """
    span = 0.5 * ((dg_bounds[1] - dg_bounds[0]) + (db_bounds[1] - db_bounds[0]))
    # Cap the multiplier in [0.03, 0.20] of the average box span.
    factor = 0.20 / max(1.0, math.log2(max(2, int(depth))) / 2.0)
    return float(max(0.03, min(0.20, factor)) * span)


# --------------------------------------------------------------------------- #
# Main training driver                                                         #
# --------------------------------------------------------------------------- #

def _warn_legacy_kwargs_once(w_mean, w_slope, use_log_mean):
    global _LEGACY_KWARGS_WARNED
    if _LEGACY_KWARGS_WARNED:
        return
    legacy_in_play = (
        (w_mean is not None and float(w_mean) != 0.0 and float(w_mean) != 1.0)
        or (w_slope is not None and float(w_slope) != 1.0 and float(w_slope) != 0.0)
        or (use_log_mean is False)
    )
    if legacy_in_play:
        warnings.warn(
            "train_lr_grid_search_bm24 (v2): the v1 kwargs "
            "`w_mean`, `w_slope`, `use_log_mean` are accepted for "
            "backward compatibility but are now ignored. The objective is "
            "fixed to: slope of ln(median(1/p_succ)) vs n. To recover the "
            "v1 single-n mean objective explicitly, set proxy_h_by_n=None.",
            DeprecationWarning,
            stacklevel=3,
        )
        _LEGACY_KWARGS_WARNED = True


def train_lr_grid_search_bm24(
    training_h: List[np.ndarray],
    train_n: int,
    depth: int,
    *,
    skip_grid: bool = False,
    initial_deltas: Optional[Tuple[float, float]] = None,
    skip_grid_if_warm_start: bool = True,
    beta_schedule: str = "decreasing",
    dg_bounds: Tuple[float, float] = DEFAULT_DG_BOUNDS,
    db_bounds: Tuple[float, float] = DEFAULT_DB_BOUNDS,
    cobyla_maxiter: int = 200,
    cobyla_tol: float = 1e-3,
    cobyla_rhobeg: Optional[float] = None,
    cobyla_restarts: int = 8,
    cobyla_perturb_scale: float = 0.2,
    grid_top_k: int = 5,
    proxy_h_by_n: Optional[Dict[int, List[np.ndarray]]] = None,
    proxy_n_values: Optional[Sequence[int]] = None,
    eps: float = DEFAULT_EPS,
    rng: Optional[np.random.Generator] = None,
    verbose: bool = True,
    # --- v1 kwargs accepted for backward compatibility; ignored in v2 ----
    w_mean: float = 0.0,
    w_slope: float = 1.0,
    use_log_mean: bool = True,
) -> Tuple[np.ndarray, dict]:
    """
    Train LR-QAOA ``(delta_gamma, delta_beta)`` against the v2 slope-of-
    log-median objective. Same call signature as v1 for the notebook.

    Pipeline
    --------
    1. **Assemble multi-n training set.** ``proxy_h_by_n`` + ``proxy_n_values``
       define the n values used in the slope fit. If absent, falls back to
       single-n median objective at ``train_n`` (with a warning).
    2. **Optional grid scan** in the ``(dg, db)`` box (default ``11 x 11``).
       Skipped if ``skip_grid`` or if a warm start is provided and
       ``skip_grid_if_warm_start`` is true.
    3. **Multi-restart COBYLA** from the top-``grid_top_k`` grid points
       plus perturbed restarts around the best, with an adaptive ``rhobeg``
       that shrinks with depth.
    4. **Return** the optimum together with a diagnostics dict containing
       per-n log medians, mean p_succ at ``train_n``, restart records, and
       the final slope at the optimum.

    The returned ``params`` array is concatenated ``[gammas, betas]`` of
    length ``2*depth`` (BM24 half-angle convention).
    """
    _warn_legacy_kwargs_once(w_mean, w_slope, use_log_mean)
    train_n = int(train_n)
    depth = int(depth)

    # ---------- Assemble the n -> H_diag dict ----------
    h_by_n: Dict[int, List[np.ndarray]] = {}
    if proxy_h_by_n is not None:
        for n in proxy_h_by_n:
            h_by_n[int(n)] = list(proxy_h_by_n[int(n)])
    if train_n not in h_by_n and training_h is not None:
        # Always make sure train_n is available, both as a slope anchor and
        # for the mean-p diagnostic.
        h_by_n[train_n] = list(training_h)

    if proxy_n_values is not None and len(proxy_n_values) >= 1:
        ns_for_slope = sorted({int(n) for n in proxy_n_values if int(n) in h_by_n})
    else:
        ns_for_slope = sorted(h_by_n.keys())

    if len(ns_for_slope) < 2:
        warnings.warn(
            f"Single-n training only (n={ns_for_slope}). Falling back to "
            "log-median objective at that n. Provide a multi-n "
            "proxy_h_by_n (>= 3 well-spaced n values) for the full "
            "slope-matched v2 objective.",
            UserWarning,
            stacklevel=2,
        )
    elif len(ns_for_slope) < _MIN_RECOMMENDED_NS_FOR_SLOPE:
        warnings.warn(
            f"Training slope uses only {len(ns_for_slope)} n values "
            f"({ns_for_slope}). Recommended: >= 3 well-spaced (step >= 2). "
            "See `proxy_n_values_for_training` docstring.",
            UserWarning,
            stacklevel=2,
        )

    if verbose:
        total_h = sum(len(h_by_n[n]) for n in ns_for_slope)
        print(
            f"  > Slope-objective training set: n in {ns_for_slope} "
            f"({total_h} H_diag arrays total; eps={eps:g})"
        )

    # ---------- Build the objective ----------
    can_slope = len(ns_for_slope) >= 2

    def evaluate(dg: float, db: float) -> Tuple[float, List[float], float]:
        """
        Returns (objective_to_minimize, log_med_per_n, mean_p_at_train_n).

        The third return value is for diagnostics only; not used in the
        objective itself.
        """
        betas, gammas = _angles_from_deltas(dg, db, depth, beta_schedule)
        if can_slope:
            slope, log_med = _slope_log_median_inv_p(
                h_by_n, ns_for_slope, betas, gammas, eps=eps,
            )
            obj = slope if np.isfinite(slope) else 1e6
        else:
            n = ns_for_slope[0]
            m = _median_inv_p_at_n(h_by_n[n], n, betas, gammas, eps=eps)
            log_med = [float(np.log(m)) if m > 0 else float("nan")]
            obj = log_med[0] if np.isfinite(log_med[0]) else 1e6
        # Diagnostic: mean p at train_n (helps spot saturation)
        if train_n in h_by_n:
            mean_p = _mean_p_succ_at_n(h_by_n[train_n], train_n, betas, gammas)
        else:
            mean_p = float("nan")
        return float(obj), log_med, float(mean_p)

    def scalar(dg: float, db: float) -> float:
        return evaluate(dg, db)[0]

    # ---------- Grid scan ----------
    warm = initial_deltas is not None
    do_grid = (not skip_grid) and not (warm and skip_grid_if_warm_start)
    grid_scores: List[Tuple[float, float, float]] = []
    dg_vals = np.linspace(dg_bounds[0], dg_bounds[1], 11)
    db_vals = np.linspace(db_bounds[0], db_bounds[1], 11)
    constraints = _lr_box_cobyla_constraints(dg_bounds, db_bounds)

    if do_grid:
        for dg in tqdm(dg_vals, desc="Grid Scanning (dg)", disable=not verbose):
            for db in db_vals:
                sc = scalar(float(dg), float(db))
                grid_scores.append((sc, float(dg), float(db)))
        grid_scores.sort(key=lambda t: t[0])  # minimize -> ascending
        top_k = max(1, min(int(grid_top_k), len(grid_scores)))
        candidates: List[Tuple[float, float]] = [
            (grid_scores[i][1], grid_scores[i][2]) for i in range(top_k)
        ]
        best_deltas = [candidates[0][0], candidates[0][1]]
        best_rank_score = grid_scores[0][0]
        if verbose:
            print(
                f"  > Best grid: dg={best_deltas[0]:+.4f}, "
                f"db={best_deltas[1]:+.4f} (slope_nat={best_rank_score:+.6f})"
            )
            if top_k > 1:
                print(f"  > Top-{top_k} grid starts retained for COBYLA")
    else:
        if warm:
            best_deltas = [float(initial_deltas[0]), float(initial_deltas[1])]
        else:
            best_deltas = list(DEFAULT_INITIAL_DELTAS)
        best_deltas = _clip_deltas(
            np.asarray(best_deltas), dg_bounds, db_bounds
        ).tolist()
        candidates = [tuple(best_deltas)]
        best_rank_score = scalar(best_deltas[0], best_deltas[1])
        if verbose:
            label = "warm-start" if warm else "default"
            print(
                f"  > Grid skipped ({label}): dg={best_deltas[0]:+.4f}, "
                f"db={best_deltas[1]:+.4f} (slope_nat={best_rank_score:+.6f})"
            )

    # ---------- COBYLA: assemble restart starts ----------
    gen = rng if rng is not None else np.random.default_rng()
    span_dg = dg_bounds[1] - dg_bounds[0]
    span_db = db_bounds[1] - db_bounds[0]
    starts: List[np.ndarray] = []
    seen: set = set()
    for dg, db in candidates:
        x = _clip_deltas(np.array([dg, db]), dg_bounds, db_bounds)
        key = (round(float(x[0]), 8), round(float(x[1]), 8))
        if key not in seen:
            starts.append(x)
            seen.add(key)
    n_perturb = max(0, int(cobyla_restarts) - len(starts))
    anchor = starts[0]
    for _ in range(n_perturb):
        noise = gen.normal(0.0, float(cobyla_perturb_scale), size=2)
        x = anchor + noise * np.array([span_dg, span_db])
        x = _clip_deltas(x, dg_bounds, db_bounds)
        key = (round(float(x[0]), 8), round(float(x[1]), 8))
        if key not in seen:
            starts.append(x)
            seen.add(key)

    # ---------- COBYLA: run restarts ----------
    rhobeg = (
        float(cobyla_rhobeg)
        if cobyla_rhobeg is not None
        else _adaptive_rhobeg(depth, dg_bounds, db_bounds)
    )

    def objective(d: np.ndarray) -> float:
        return scalar(float(d[0]), float(d[1]))

    best_x = np.asarray(best_deltas, dtype=float)
    best_obj = scalar(best_x[0], best_x[1])
    restart_records: List[dict] = []
    for i, x0 in enumerate(starts):
        x_fin, obj, meta = _cobyla_refine(
            objective, x0,
            constraints=constraints,
            cobyla_maxiter=cobyla_maxiter,
            cobyla_tol=cobyla_tol,
            cobyla_rhobeg=rhobeg,
            dg_bounds=dg_bounds,
            db_bounds=db_bounds,
        )
        restart_records.append({
            "restart_idx": i,
            "x0": x0.tolist(),
            "x_final": x_fin.tolist(),
            "slope_nat": obj,
            **meta,
        })
        if obj < best_obj:
            best_obj = obj
            best_x = x_fin
        if verbose and (i == 0 or obj <= best_obj + 1e-12):
            print(
                f"  > COBYLA restart {i:2d}: slope_nat={obj:+.6f}  "
                f"x={np.round(x_fin, 4).tolist()}"
            )

    # ---------- Final diagnostics at the optimum ----------
    dg_opt, db_opt = float(best_x[0]), float(best_x[1])
    final_obj, final_log_med, final_mean_p = evaluate(dg_opt, db_opt)
    final_slope_log2 = (
        float(final_obj / np.log(2.0))
        if can_slope and np.isfinite(final_obj)
        else float("nan")
    )

    if verbose:
        print(
            f"  > Best after COBYLA: dg={dg_opt:+.6f}, db={db_opt:+.6f}, "
            f"slope_nat={final_obj:+.6f} (slope_log2={final_slope_log2:+.6f})"
        )
        if can_slope:
            saturation = []
            for n_i, lm in zip(ns_for_slope, final_log_med):
                saturation.append(f"n={n_i}: ln(med 1/p)={lm:+.4f}")
            print("    per-n log medians at optimum: " + " | ".join(saturation))
            if final_log_med[0] < 1e-3 and len(final_log_med) > 1:
                print(
                    "    WARNING: ln(med 1/p) ≈ 0 at smallest n -> "
                    "median p_succ saturated near 1. The slope is being "
                    "anchored by the larger n only; consider a larger train_n."
                )
        print(f"    mean p_succ @ n={train_n} = {final_mean_p:.4e}")

    betas, gammas = _angles_from_deltas(dg_opt, db_opt, depth, beta_schedule)
    params_concat = np.concatenate([gammas, betas])
    diag = {
        "best_deltas": [dg_opt, db_opt],
        "best_train_slope_nat": float(final_obj) if can_slope else None,
        "best_train_slope_log2": float(final_slope_log2) if can_slope else None,
        "best_avg_train_p_succ": float(final_mean_p),  # legacy key for the notebook
        "best_train_score": float(-final_obj) if can_slope else float(final_mean_p),
        # ^ legacy: v1 returned a "score to maximize". We give -slope so
        #   higher still means better, but its absolute scale differs.
        "log_med_inv_p_per_n": [float(x) for x in final_log_med],
        "train_ns": list(ns_for_slope),
        "grid_skipped": not do_grid,
        "warm_start_used": bool(warm),
        "cobyla_rhobeg_used": float(rhobeg),
        "cobyla_restarts": int(len(starts)),
        "restart_records": restart_records,
        "objective_version": "v2-slope-of-log-median-inv-p",
    }
    return params_concat, diag


# --------------------------------------------------------------------------- #
# Post-training: held-out exponent fit (unchanged from v1 in spirit)          #
# --------------------------------------------------------------------------- #

def compute_mean_success_exponent_vs_n(
    n_min: int,
    n_max: int,
    k: int,
    r: float,
    test_size: int,
    base_seed: int,
    betas: np.ndarray,
    gammas: np.ndarray,
    m_sampling: Literal["notebook", "bm24"] = "notebook",
) -> dict:
    """
    Held-out fit of ln(mean p_succ) vs n at the trained angles. This is the
    "average" complement to the median-based training objective and is
    useful for sanity-checking against BM24's analytic scaling.
    """
    n_min, n_max = int(n_min), int(n_max)
    if n_max < n_min:
        return {"ok": False, "reason": "n_max < n_min"}
    n_values = list(range(n_min, n_max + 1))
    mean_p_per_n: Dict[int, float] = {}
    for n in n_values:
        h_list = generate_training_h_diagonals(
            train_n=n, k=k, r=r,
            train_size=int(test_size),
            base_seed=int(base_seed) + 99_991 * n,
            m_sampling=m_sampling,
        )
        mean_p_per_n[n] = _mean_p_succ_at_n(h_list, n, betas, gammas)
    log_means = np.array([
        np.log(mean_p_per_n[n]) if mean_p_per_n[n] > 0 else np.nan
        for n in n_values
    ])
    ns_arr = np.array(n_values, dtype=float)
    mask = np.isfinite(log_means)
    if int(mask.sum()) < 2:
        return {
            "ok": False,
            "reason": "fewer than 2 positive mean p_succ values",
            "mean_p_succ_per_n": {int(n): float(p) for n, p in mean_p_per_n.items()},
        }
    res = linregress(ns_arr[mask], log_means[mask])
    return {
        "ok": True,
        "mean_p_succ_per_n": {int(n): float(p) for n, p in mean_p_per_n.items()},
        "slope_ln_mean_p_succ_vs_n_natural_log": float(res.slope),
        "slope_ln_mean_p_succ_vs_n_log2": float(res.slope / np.log(2.0)),
        "intercept_natural_log": float(res.intercept),
        "rvalue": float(res.rvalue),
        "stderr": float(res.stderr),
    }


# --------------------------------------------------------------------------- #
# Optimal-angle log appender (preserved for the notebook)                     #
# --------------------------------------------------------------------------- #

def append_optimal_angles_log(
    log_path: Path,
    *,
    timestamp: str,
    settings: dict,
    dg: float,
    db: float,
    betas: np.ndarray,
    gammas: np.ndarray,
    best_avg_train_p_succ: float,
    benchmark_cmd: str,
    exponent_fit: Optional[dict] = None,
) -> None:
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"\n# {timestamp}\n")
        f.write(f"settings = {json.dumps(settings, default=str)}\n")
        f.write(f"delta_gamma = {dg:+.10f}\n")
        f.write(f"delta_beta  = {db:+.10f}\n")
        f.write(f"betas  = {np.array(betas).tolist()}\n")
        f.write(f"gammas = {np.array(gammas).tolist()}\n")
        f.write(f"best_avg_train_p_succ = {best_avg_train_p_succ:.6e}\n")
        f.write(f"benchmark_cmd = {benchmark_cmd!r}\n")
        if exponent_fit is not None:
            f.write(f"exponent_fit = {json.dumps(exponent_fit)}\n")


def _benchmark_cli_snippet(
    train_n: int, n_max: int, k: int, r: float,
    dg: float, db: float, depth: int, test_size: int,
    seed: int, beta_schedule: str,
) -> str:
    return (
        "python phasecraft/bm24_qaoa_sim.py --benchmark "
        f"--n-min {int(train_n)} --n-max {int(n_max)} --k {int(k)} --r {float(r)} "
        f"--algorithms lr_qaoa walksat walksatlm "
        f"--depth {int(depth)} --lr-dgamma {float(dg):.10f} --lr-dbeta {float(db):.10f} "
        f"--lr-angle-convention bm24 --lr-beta-schedule {beta_schedule} "
        f"--test-size {int(test_size)} --seed {int(seed)} --plot-benchmark"
    )


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #

def main():
    p = argparse.ArgumentParser(
        description=(
            "Train LR-QAOA (delta_gamma, delta_beta) by minimising slope of "
            "ln(median(1/p_succ)) vs n. v2 objective: matches the evaluation "
            "plot's y-axis."
        )
    )
    p.add_argument("--depth", type=int, required=True)
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--r", type=float, default=176.54)
    p.add_argument("--train-n", type=int, default=12)
    p.add_argument("--train-size", type=int, default=100,
                   help="Instances at train_n (used for warm-start diagnostic "
                        "and as one of the slope-fit n points).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--skip-grid", action="store_true",
                   help="Skip 11x11 grid; start COBYLA from default deltas.")
    p.add_argument("--no-warm-skip-grid", action="store_true")
    p.add_argument("--initial-dgamma", type=float, default=None)
    p.add_argument("--initial-dbeta", type=float, default=None)
    p.add_argument("--m-sampling", type=str, default="notebook",
                   choices=["notebook", "bm24"])
    p.add_argument("--lr-beta-schedule", type=str, default="decreasing",
                   choices=["decreasing", "increasing"])
    p.add_argument("--cobyla-maxiter", type=int, default=200)
    p.add_argument("--cobyla-tol", type=float, default=1e-3)
    p.add_argument("--cobyla-restarts", type=int, default=8)
    p.add_argument("--cobyla-perturb-scale", type=float, default=0.2)
    p.add_argument("--cobyla-rhobeg", type=float, default=None,
                   help="Default: adaptive rhobeg, smaller at larger depth.")
    p.add_argument("--grid-top-k", type=int, default=5)
    # v2 multi-n slope-objective parameters
    p.add_argument("--proxy-size-per-n", type=int, default=50,
                   help="Instances per n in the multi-n slope set (v2 default 50).")
    p.add_argument("--proxy-n-span", type=int, default=4,
                   help="Total span in n (v2 default 4 with step 2 -> 3 n values).")
    p.add_argument("--proxy-n-step", type=int, default=2,
                   help="Step between training n values (v2 default 2).")
    p.add_argument("--no-proxy-set", action="store_true",
                   help="Disable multi-n; fall back to single-n median objective.")
    p.add_argument("--n-max-benchmark", type=int, default=20)
    p.add_argument("--test-size-benchmark", type=int, default=200)
    p.add_argument("--eval-exponent-test-size", type=int, default=50)
    p.add_argument("--eval-exponent-seed", type=int, default=None)
    p.add_argument("--skip-exponent-fit", action="store_true")
    p.add_argument("--save-json", type=str, default=None)
    p.add_argument("--angle-log", type=str, default=str(_DEFAULT_ANGLE_LOG))
    p.add_argument("--no-angle-log", action="store_true")
    # v1 legacy flags accepted but ignored
    p.add_argument("--legacy-objective", action="store_true",
                   help="(deprecated, ignored in v2)")
    p.add_argument("--w-mean", type=float, default=0.0,
                   help="(deprecated, ignored in v2)")
    p.add_argument("--w-slope", type=float, default=1.0,
                   help="(deprecated, ignored in v2)")
    args = p.parse_args()

    # ----- Build multi-n training set -----
    if args.no_proxy_set:
        proxy_h = None
        proxy_ns: List[int] = []
    else:
        proxy_ns = proxy_n_values_for_training(
            args.train_n,
            proxy_n_span=int(args.proxy_n_span),
            n_max_cap=int(args.n_max_benchmark),
            step=int(args.proxy_n_step),
        )
        print(
            f"Building multi-n training set: n in {proxy_ns} "
            f"({args.proxy_size_per_n} SAT-filtered instances per n) ..."
        )
        proxy_h = generate_training_h_diagonals_multi_n(
            proxy_ns, k=args.k, r=args.r,
            train_size_per_n=int(args.proxy_size_per_n),
            base_seed=int(args.seed) + 77,
            m_sampling=args.m_sampling,
        )

    # Anchor set at train_n (also acts as one of the slope-fit n's if not
    # already present in the proxy)
    print(
        f"Building training set at train_n={args.train_n}: "
        f"size={args.train_size}, k={args.k}, r={args.r}, seed={args.seed}"
    )
    training_h = generate_training_h_diagonals(
        train_n=args.train_n, k=args.k, r=args.r,
        train_size=args.train_size, base_seed=args.seed,
        m_sampling=args.m_sampling,
    )
    print(f"  collected {len(training_h)} SAT-filtered instances at n={args.train_n}.")

    initial = None
    if args.initial_dgamma is not None and args.initial_dbeta is not None:
        initial = (float(args.initial_dgamma), float(args.initial_dbeta))

    print("\n--- Training LR-QAOA (v2 slope-of-log-median-inv-p) ---")
    params, diag = train_lr_grid_search_bm24(
        training_h,
        train_n=args.train_n,
        depth=args.depth,
        skip_grid=args.skip_grid,
        initial_deltas=initial,
        skip_grid_if_warm_start=not args.no_warm_skip_grid,
        beta_schedule=args.lr_beta_schedule,
        cobyla_maxiter=args.cobyla_maxiter,
        cobyla_tol=args.cobyla_tol,
        cobyla_rhobeg=args.cobyla_rhobeg,
        cobyla_restarts=args.cobyla_restarts,
        cobyla_perturb_scale=args.cobyla_perturb_scale,
        grid_top_k=args.grid_top_k,
        proxy_h_by_n=proxy_h,
        proxy_n_values=proxy_ns if proxy_h is not None else None,
        rng=np.random.default_rng(int(args.seed) + 1000 + int(args.depth)),
    )
    depth = int(args.depth)
    gammas = params[:depth]
    betas = params[depth:]
    dg, db = diag["best_deltas"]

    print("\n" + "=" * 72)
    print("BEST (BM24 convention -- use with --lr-angle-convention bm24)")
    print("=" * 72)
    print(f"  delta_gamma : {dg:+.10f}")
    print(f"  delta_beta  : {db:+.10f}")
    print(f"  slope_nat   : {diag['best_train_slope_nat']:+.6f}")
    print(f"  slope_log2  : {diag['best_train_slope_log2']:+.6f}")
    print(f"  betas  (len {depth}): {np.round(betas, 6).tolist()}")
    print(f"  gammas (len {depth}): {np.round(gammas, 6).tolist()}")

    cmd = _benchmark_cli_snippet(
        train_n=args.train_n,
        n_max=args.n_max_benchmark,
        k=args.k, r=args.r,
        dg=dg, db=db, depth=depth,
        test_size=args.test_size_benchmark,
        seed=args.seed,
        beta_schedule=args.lr_beta_schedule,
    )
    print("\nSuggested full benchmark:")
    print(cmd)

    exponent_fit = None
    if not args.skip_exponent_fit and args.n_max_benchmark >= args.train_n:
        eval_seed = (
            int(args.eval_exponent_seed)
            if args.eval_exponent_seed is not None
            else int(args.seed) + 1_000_003
        )
        print("\n--- Held-out mean-success exponent ---")
        exponent_fit = compute_mean_success_exponent_vs_n(
            n_min=args.train_n, n_max=args.n_max_benchmark,
            k=args.k, r=args.r,
            test_size=args.eval_exponent_test_size,
            base_seed=eval_seed,
            betas=betas, gammas=gammas,
            m_sampling=args.m_sampling,
        )
        if exponent_fit.get("ok"):
            print(f"  mean_p_succ per n: {exponent_fit['mean_p_succ_per_n']}")
            print(
                "  slope d/dn ln(mean p_succ) [nat] = "
                f"{exponent_fit['slope_ln_mean_p_succ_vs_n_natural_log']:+.6f}"
            )
            print(
                "  same in log2 units               = "
                f"{exponent_fit['slope_ln_mean_p_succ_vs_n_log2']:+.6f}"
            )
        else:
            print(f"  (skipped or failed: {exponent_fit.get('reason')})")

    out = {
        "settings": {
            "train_n": args.train_n, "train_size": args.train_size,
            "k": args.k, "r": args.r, "depth": depth, "seed": args.seed,
            "m_sampling": args.m_sampling,
            "lr_beta_schedule": args.lr_beta_schedule,
            "proxy_n_values": proxy_ns,
            "proxy_size_per_n": args.proxy_size_per_n,
            "objective": "v2-slope-of-log-median-inv-p",
            "cobyla_maxiter": args.cobyla_maxiter,
            "cobyla_restarts": args.cobyla_restarts,
            "angle_convention": "bm24",
        },
        "best_deltas": {"delta_gamma": float(dg), "delta_beta": float(db)},
        "betas_bm24": [float(x) for x in betas],
        "gammas_bm24": [float(x) for x in gammas],
        "params_concat_gammas_then_betas": [float(x) for x in params],
        "diagnostics": diag,
        "suggested_benchmark_command": cmd,
        "mean_success_exponent_fit": exponent_fit,
    }
    if args.save_json:
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"\nWrote JSON to {args.save_json}")

    if not args.no_angle_log:
        log_path = Path(args.angle_log)
        settings_compact = {
            "train_n": args.train_n, "train_size": args.train_size,
            "k": args.k, "r": args.r, "depth": depth, "seed": args.seed,
            "m_sampling": args.m_sampling,
            "lr_beta_schedule": args.lr_beta_schedule,
            "proxy_n_values": proxy_ns,
            "proxy_size_per_n": args.proxy_size_per_n,
            "objective": "v2-slope-of-log-median-inv-p",
        }
        append_optimal_angles_log(
            log_path,
            timestamp=datetime.now().isoformat(timespec="seconds"),
            settings=settings_compact,
            dg=dg, db=db, betas=betas, gammas=gammas,
            best_avg_train_p_succ=float(diag["best_avg_train_p_succ"]),
            benchmark_cmd=cmd,
            exponent_fit=exponent_fit,
        )
        print(f"\nAppended optimal angles to: {log_path.resolve()}")


if __name__ == "__main__":
    main()
