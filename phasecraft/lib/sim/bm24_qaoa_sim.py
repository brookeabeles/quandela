"""
bm24_qaoa_sim.py
================

A clean, BM24-compatible finite-n QAOA simulator for random k-SAT, designed to
be mathematically compatible with the asymptotic (large-n) observable
implemented in `generalized_binomial_sum.py`. -> from the BM24 repository

This module deliberately contains NO training / grid-search / external solver
machinery. It implements exactly:

  (i)   the BM24 random k-SAT ensemble (Definition 1 of arXiv:2208.06909 /
        PRX Quantum 5, 030348 (2024)),
  (ii)  the BM24 QAOA state of Eq. (13) -- in particular, the factors of 1/2
        in BOTH the phase separator AND the mixer (not present in Leo's old code),
  (iii) the BM24 success observable
            E_{sigma ~ CNF(n,k,r)} [ <psi| Pi_{H[sigma]=0} |psi> ]
        over the UNCONDITIONAL ensemble (UNSAT instances kept with p_succ = 0),
  (iv)  the corresponding asymptotic exponent
            phi_full = phi_M + phi_pref,
        with phi_pref = -(r / 2^k)   [Convention-2 / all-subsets prefactor]
        and phi_M obtained from
            generalized_binomial_sum_full_exponent_ksat(...) .

Conventions (locked, matching BM24 verbatim):
---------------------------------------------
  * H_B    := sum_j X_j                              (BM24 Eq. 7)
  * H[sigma](y) := number of clauses in sigma not satisfied by y
                                                     (BM24 Eq. 11)
  * |psi(sigma, beta, gamma)>
        = exp(-i beta_{p-1}/2  H_B)
          exp(-i gamma_{p-1}/2  H[sigma])
          ...
          exp(-i beta_0/2      H_B)
          exp(-i gamma_0/2     H[sigma])
          |+>^{otimes n}                             (BM24 Eq. 13)
        NOTE the 1/2 in BOTH unitaries.
  * p_succ(sigma) := <psi| Pi_{H[sigma]=0} |psi>     (BM24 Eq. 12)
  * Ensemble: m ~ Poisson(r n); each clause = k literals drawn uniformly with
              replacement from {x_0, ~x_0, ..., x_{n-1}, ~x_{n-1}}.
              UNSAT instances are kept (p_succ = 0). (BM24 Definition 1)
  * Asymptotic prediction:
        log E[p_succ(n)]  ~  n * phi_full        (natural log)
        log_2 E[p_succ(n)] ~ n * phi_full / ln(2) (base-2 log)

Note on LR angle conventions (``make_lr_angles``):
----------------------------------------------
  * ``angle_convention="bm24"`` (default): ``delta_gamma`` / ``delta_beta`` are
    in the same units as ``run_qaoa`` (half-angle Hamiltonian convention).
  * ``angle_convention="notebook"``: deltas follow the legacy notebook ramp
    (full-angle ``exp(-i gamma H)`` style); ``make_lr_angles`` doubles them
    before building the ramp so returned ``(betas, gammas)`` are BM24-ready.

Bit-ordering convention used internally:
  Variable x_i corresponds to bit i of the integer y (LSB = variable 0).
  State |y> for integer y in {0, ..., 2^n - 1} is at index y of the
  state-vector array.

Validation status (k=2, p=1, r=2.0, beta=gamma=0.4) -- see comments in
`exact_finite_n_p1_prop4` and the GAP rows in the report:

  * The simulator's per-instance circuit was hand-verified for n=2 against
    the explicit BM24 Eq. (13) state vector.
  * The simulator's UNCONDITIONAL ensemble mean was brute-forced over the
    full Poisson(rn) clause distribution at n=2 (truncating m at the
    Poisson tail beyond ~1e-7), matching the simulator to ~0.1 percent.
  * `exact_finite_n_p1_prop4` (a direct re-implementation of BM24
    Proposition 4 / Eq. (A10)) matches the simulator within Monte Carlo
    error at every n tested in [2, 10].
  * The existing-code function `generalized_flip_symmetric_expected_success_p1`
    DOES NOT match the simulator, the brute force, or `exact_finite_n_p1_prop4`.
    Its values exceed the truth by a factor of approximately exp(0.126 * n).
  * Equivalently, `generalized_binomial_sum_full_exponent_ksat` returns
    phi_full ~= -0.453 at the test point, while a least-squares fit on
    `exact_finite_n_p1_prop4` over n in [20, 50] gives slope ~= -0.579.

Run:
  python bm24_qaoa_sim.py
"""

import argparse
import importlib.util
import json
import logging
import math
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from scipy.stats import linregress

# --- Theory side (existing repo code) ---------------------------------------
# Load the patched generalized-binomial implementation from this repository.
from phasecraft.lib.paths import bm24_qaoa_sim_cli_path, bm24_runs_dir, patched_gbs_path

_PATCHED_PATH = patched_gbs_path()
_KSAT_LIB = _PATCHED_PATH.parent.parent
_PATCHED_SPEC = importlib.util.spec_from_file_location(
    "phasecraft_gbs_patched",
    _PATCHED_PATH,
    submodule_search_locations=[str(_PATCHED_PATH.parent), str(_KSAT_LIB)],
)
if _PATCHED_SPEC is None or _PATCHED_SPEC.loader is None:
    raise ImportError(f"Cannot load patched module at {_PATCHED_PATH}")
_PATCHED_MOD = importlib.util.module_from_spec(_PATCHED_SPEC)
_PATCHED_SPEC.loader.exec_module(_PATCHED_MOD)

from phasecraft.lib.sim.bm24_run_io import (  # noqa: E402
    apply_figure_suptitle,
    format_benchmark_title,
    make_run_stem,
    resolve_bm24_run_output_paths,
)

generalized_binomial_sum_full_exponent_ksat = _PATCHED_MOD.generalized_binomial_sum_full_exponent_ksat
generalized_flip_symmetric_expected_success_p1 = _PATCHED_MOD.generalized_flip_symmetric_expected_success_p1
bm24_prefactor_exponent_ksat = _PATCHED_MOD.bm24_prefactor_exponent_ksat
bm24_prefactor_exponent_ksat_all_subsets = getattr(
    _PATCHED_MOD, "bm24_prefactor_exponent_ksat_all_subsets", None
)

LOG = logging.getLogger("bm24_qaoa_sim")

# Default folder for timestamped run artifacts (JSON + TXT).
BM24_DEFAULT_OUTPUT_DIR = str(bm24_runs_dir())

try:
    from numba import njit
    HAS_NUMBA = True
except Exception:  # pragma: no cover - optional acceleration
    HAS_NUMBA = False
    njit = None


# ---------------------------------------------------------------------------
# 1. Random formula generation -- BM24 Definition 1
# ---------------------------------------------------------------------------

def generate_random_clause(n: int, k: int, rng: np.random.Generator
                           ) -> List[Tuple[int, bool]]:
    """
    Sample a single OR clause of k literals chosen uniformly WITH REPLACEMENT
    from the set of 2n literals {x_0, ~x_0, ..., x_{n-1}, ~x_{n-1}}.

    Returns a list of (var_index, is_negated) pairs, length k.

    Implementation note:
      Uniform-with-replacement over 2n literals is equivalent to drawing
      var_index ~ Uniform({0,...,n-1}) and is_negated ~ Bernoulli(1/2)
      independently for each of the k literals.
      Tautological clauses (e.g. x_0 OR ~x_0 OR x_1) and clauses with repeated
      variables are kept -- this is part of the BM24 ensemble.
    """
    var_idx = rng.integers(0, n, size=k)
    neg = rng.integers(0, 2, size=k).astype(bool)
    return list(zip(var_idx.tolist(), neg.tolist()))


def generate_random_formula(n: int, k: int, r: float, rng: np.random.Generator
                            ) -> List[List[Tuple[int, bool]]]:
    """
    Sample a random k-SAT instance from CNF(n, k, r):
      m ~ Poisson(r * n)
      then m i.i.d. clauses from generate_random_clause.
    """
    m = int(rng.poisson(r * n))
    return [generate_random_clause(n, k, rng) for _ in range(m)]


# ---------------------------------------------------------------------------
# 2. Diagonal cost Hamiltonian -- BM24 Eq. (11)
# ---------------------------------------------------------------------------

def build_h_diagonal(clauses: List[List[Tuple[int, bool]]], n: int
                     ) -> np.ndarray:
    """
    Build the diagonal of H[sigma]:
        H[y] = #{ clauses not satisfied by y } .

    A clause C = (l_1, ..., l_k) is UNSATISFIED by y iff every literal in C
    evaluates to 0 under y:
        positive literal x_i unsatisfied  <=>  y_i = 0
        negative literal ~x_i unsatisfied <=>  y_i = 1

    Tautological clauses (variable appearing with both polarities) are
    automatically unsatisfied by no assignment -> contribute 0 everywhere.
    """
    H = np.zeros(2 ** n, dtype=np.int64)
    y = np.arange(2 ** n)
    for clause in clauses:
        unsat = np.ones(2 ** n, dtype=bool)
        for var_idx, is_negated in clause:
            bit = (y >> var_idx) & 1
            if is_negated:
                # literal ~x_i  is false iff y_i == 1
                unsat &= (bit == 1)
            else:
                # literal  x_i  is false iff y_i == 0
                unsat &= (bit == 0)
            if not unsat.any():
                break  # short-circuit tautologies / impossible clauses
        H[unsat] += 1
    return H


# ---------------------------------------------------------------------------
# 3. State-vector simulator -- BM24 Eq. (13)
# ---------------------------------------------------------------------------

def plus_state(n: int) -> np.ndarray:
    """ |+>^{otimes n}  as a length-2^n complex vector (uniform amplitude). """
    return np.full(2 ** n, 1.0 / np.sqrt(2 ** n), dtype=np.complex128)


def apply_phase_separator(psi: np.ndarray, H_diag: np.ndarray, gamma: float
                          ) -> np.ndarray:
    """
    psi <- exp(-i gamma/2 * H[sigma]) psi   (BM24 convention with 1/2)
    """
    return psi * np.exp(-0.5j * gamma * H_diag)


def apply_mixer(psi: np.ndarray, beta: float, n: int) -> np.ndarray:
    """
    psi <- exp(-i beta/2 * sum_j X_j) psi    (BM24 convention with 1/2)

    Uses the fact that the X_j commute, so the operator factorises into
    single-qubit X-rotations:
        exp(-i beta/2 X_j) = cos(beta/2) I - i sin(beta/2) X_j .
    Each is applied via a bit-flip indexing trick in O(2^n) per qubit,
    O(n 2^n) total per mixer layer.
    """
    c = np.cos(beta / 2.0)
    s = -1j * np.sin(beta / 2.0)
    idx = np.arange(2 ** n)
    for j in range(n):
        flipped = idx ^ (1 << j)
        # Crucial: read psi[flipped] BEFORE rebinding psi.
        psi = c * psi + s * psi[flipped]
    return psi


def run_qaoa(H_diag: np.ndarray, betas: np.ndarray, gammas: np.ndarray,
             n: int) -> np.ndarray:
    """
    Apply the BM24 p-layer QAOA circuit (Eq. 13). Layers are applied in the
    order  gamma_0, beta_0, gamma_1, beta_1, ..., gamma_{p-1}, beta_{p-1}
    starting from |+>^{otimes n}.
    """
    if len(betas) != len(gammas):
        raise ValueError("betas and gammas must have the same length p")
    psi = plus_state(n)
    for j in range(len(betas)):
        psi = apply_phase_separator(psi, H_diag, float(gammas[j]))
        psi = apply_mixer(psi, float(betas[j]), n)
    return psi


# ---------------------------------------------------------------------------
# 4. Per-instance and ensemble observables -- BM24 Eq. (12)
# ---------------------------------------------------------------------------

def per_instance_success_probability(psi: np.ndarray, H_diag: np.ndarray
                                     ) -> float:
    """
    p_succ(sigma) = <psi| Pi_{H=0} |psi>
                  = sum_{ y : H[sigma](y) = 0 } |psi[y]|^2 .
    For UNSAT instances the sum is empty and returns 0.0.
    """
    sat_mask = (H_diag == 0)
    if not sat_mask.any():
        return 0.0
    return float(np.sum(np.abs(psi[sat_mask]) ** 2))


def sample_p_succ_at_n(n: int, k: int, r: float,
                       betas: np.ndarray, gammas: np.ndarray,
                       num_instances: int, base_seed: int) -> np.ndarray:
    """
    Draw `num_instances` independent random instances from CNF(n, k, r) and
    compute p_succ(sigma) for each.  UNSAT instances are KEPT (with p_succ=0),
    matching the unconditional BM24 expectation.
    """
    # Make seeding fully reproducible across (base_seed, n) without
    # cross-talk between sizes.
    seed_seq = np.random.SeedSequence([int(base_seed), int(n)])
    rng = np.random.default_rng(seed_seq)
    out = np.zeros(num_instances, dtype=np.float64)
    for i in range(num_instances):
        clauses = generate_random_formula(n, k, r, rng)
        H_diag = build_h_diagonal(clauses, n)
        psi = run_qaoa(H_diag, betas, gammas, n)
        out[i] = per_instance_success_probability(psi, H_diag)
    return out


def make_lr_angles(delta_gamma: float, delta_beta: float, depth: int,
                   beta_schedule: str = "decreasing",
                   angle_convention: str = "bm24",
                   ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Construct LR-QAOA linear-ramp angles.

    beta_schedule="decreasing" (legacy notebook *shape*):
        gammas_j = scale * delta_gamma * (j+1)/p
        betas_j  = scale * delta_beta  * (1 - (j+1)/p)
    beta_schedule="increasing":
        gammas_j = scale * delta_gamma * (j+1)/p
        betas_j  = scale * delta_beta  * (j+1)/p

    angle_convention="bm24" (default):
        delta_gamma, delta_beta are in BM24 units (for ``run_qaoa``); scale=1.
    angle_convention="notebook":
        deltas are in the legacy full-angle notebook units; scale=2 so the
        returned arrays are BM24-convention angles ready for ``run_qaoa``.
    """
    if depth <= 0:
        raise ValueError("depth must be positive")
    if angle_convention == "bm24":
        scale = 1.0
    elif angle_convention == "notebook":
        scale = 2.0
    else:
        raise ValueError(
            f"angle_convention must be one of {{'bm24','notebook'}}, got {angle_convention!r}"
        )
    t = np.arange(1, depth + 1, dtype=float) / float(depth)
    gammas = scale * float(delta_gamma) * t
    if beta_schedule == "decreasing":
        betas = scale * float(delta_beta) * (1.0 - t)
    elif beta_schedule == "increasing":
        betas = scale * float(delta_beta) * t
    else:
        raise ValueError("beta_schedule must be one of {'decreasing','increasing'}")
    return betas, gammas


def clauses_to_arrays(clauses: List[List[Tuple[int, bool]]], k: int
                      ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert clause list to array form for classical solvers.
    c_unsat[i,j] = int(is_negated), literal true iff assignment[var] != unsat.
    """
    m = len(clauses)
    c_vars = np.zeros((m, k), dtype=np.int32)
    c_unsat = np.zeros((m, k), dtype=np.int32)
    for i, clause in enumerate(clauses):
        if len(clause) != k:
            raise ValueError(f"Clause {i} has len={len(clause)} but expected k={k}")
        for j, (var, is_negated) in enumerate(clause):
            c_vars[i, j] = int(var)
            c_unsat[i, j] = int(bool(is_negated))
    return c_vars, c_unsat


def assignment_satisfies_formula(assignment: np.ndarray,
                                 clauses: List[List[Tuple[int, bool]]]) -> bool:
    """Literal semantics: literal true iff assignment[var] != int(is_negated)."""
    for clause in clauses:
        if not any(int(assignment[var]) != int(is_negated) for var, is_negated in clause):
            return False
    return True


def generate_benchmark_dataset(n_values, k: int, r: float, test_size: int,
                               base_seed: int, require_sat: bool = True,
                               max_rejections: int = 100000) -> Dict[int, List[dict]]:
    """
    Build SAT-filtered formulas without storing H_diag (memory-safe).
    """
    dataset: Dict[int, List[dict]] = {}
    n_values = [int(n) for n in n_values]
    for n in n_values:
        accepted = 0
        trial = 0
        instances: List[dict] = []
        while accepted < int(test_size):
            if trial >= int(max_rejections):
                raise RuntimeError(
                    f"Exceeded max_rejections={max_rejections} while building dataset for n={n}"
                )
            seed_seq = np.random.SeedSequence([int(base_seed), int(n), int(accepted), int(trial)])
            rng = np.random.default_rng(seed_seq)
            clauses = generate_random_formula(n=n, k=k, r=r, rng=rng)
            if require_sat:
                H_diag = build_h_diagonal(clauses, n)
                num_solutions = int(np.sum(H_diag == 0))
                if num_solutions <= 0:
                    trial += 1
                    continue
            else:
                num_solutions = -1
            instances.append({"clauses": clauses, "num_solutions": num_solutions})
            accepted += 1
            trial += 1
        dataset[n] = instances
    return dataset


def _fit_log_slope_safe(n_values: np.ndarray, y_values: np.ndarray, positive_only: bool = True):
    n_values = np.asarray(n_values, dtype=float)
    y_values = np.asarray(y_values, dtype=float)
    mask = np.isfinite(y_values)
    if positive_only:
        mask &= (y_values > 0)
    if int(np.sum(mask)) < 2:
        return {"slope": float("nan"), "intercept": float("nan"), "stderr": float("nan")}
    res = linregress(n_values[mask], np.log(y_values[mask]))
    return {"slope": float(res.slope), "intercept": float(res.intercept), "stderr": float(res.stderr)}


def evaluate_qaoa_runtime_on_dataset(dataset: Dict[int, List[dict]],
                                     betas: np.ndarray, gammas: np.ndarray,
                                     eps: float = 1e-300) -> dict:
    """
    Compute success and runtime proxy (=1/max(success,eps)) on a shared dataset.
    """
    per_n = {}
    n_values = np.array(sorted(dataset.keys()), dtype=int)
    mean_successes = []
    median_runtimes = []
    for n in n_values:
        succ = []
        runtimes = []
        for inst in dataset[int(n)]:
            H_diag = build_h_diagonal(inst["clauses"], int(n))
            psi = run_qaoa(H_diag, np.asarray(betas, dtype=float), np.asarray(gammas, dtype=float), int(n))
            p = per_instance_success_probability(psi, H_diag)
            succ.append(float(p))
            runtimes.append(float(1.0 / max(float(p), float(eps))))
        succ_arr = np.asarray(succ, dtype=float)
        run_arr = np.asarray(runtimes, dtype=float)
        per_n[int(n)] = {
            "mean_success": float(np.mean(succ_arr)),
            "median_success": float(np.median(succ_arr)),
            "median_runtime": float(np.median(run_arr)),
            "inverse_mean_success": float(1.0 / max(float(np.mean(succ_arr)), float(eps))),
            "fraction_zero_success": float(np.mean(succ_arr <= 0.0)),
        }
        mean_successes.append(per_n[int(n)]["mean_success"])
        median_runtimes.append(per_n[int(n)]["median_runtime"])

    fit_mean = _fit_log_slope_safe(n_values, np.array(mean_successes), positive_only=True)
    fit_runtime = _fit_log_slope_safe(n_values, np.array(median_runtimes), positive_only=True)
    return {
        "per_n": per_n,
        "fitted_exponents_natural_log": {
            "mean_success_log_slope": float(fit_mean["slope"]),
            "median_runtime_log_slope": float(fit_runtime["slope"]),
        },
        "fitted_exponents_log2": {
            "mean_success_log_slope": float(fit_mean["slope"] / np.log(2.0)),
            "median_runtime_log_slope": float(fit_runtime["slope"] / np.log(2.0)),
        },
    }


def _eval_clause_truth_count(c_vars: np.ndarray, c_unsat: np.ndarray,
                             assignment: np.ndarray, ci: int) -> int:
    count = 0
    for j in range(c_vars.shape[1]):
        v = c_vars[ci, j]
        if int(assignment[v]) != int(c_unsat[ci, j]):
            count += 1
    return count


if HAS_NUMBA:
    @njit(cache=True)
    def _walksat_kernel_numba(c_vars, c_unsat, n, k, max_flips, p_noise, seed):
        np.random.seed(seed)
        m = c_vars.shape[0]
        assignment = np.random.randint(0, 2, size=n).astype(np.int8)
        unsat = np.empty(m, dtype=np.int32)
        unique_vars = np.empty(k, dtype=np.int32)

        for flip in range(1, max_flips + 1):
            num_unsat = 0
            for ci in range(m):
                true_count = 0
                for j in range(k):
                    v = c_vars[ci, j]
                    # literal true iff assignment[var] != unsat_value
                    if assignment[v] != c_unsat[ci, j]:
                        true_count += 1
                if true_count == 0:
                    unsat[num_unsat] = ci
                    num_unsat += 1
            if num_unsat == 0:
                return flip - 1, True

            ci = unsat[np.random.randint(0, num_unsat)]
            num_unique = 0
            for j in range(k):
                v = c_vars[ci, j]
                seen = False
                for t in range(num_unique):
                    if unique_vars[t] == v:
                        seen = True
                        break
                if not seen:
                    unique_vars[num_unique] = v
                    num_unique += 1

            if np.random.random() < p_noise:
                chosen = unique_vars[np.random.randint(0, num_unique)]
            else:
                old_true_counts = np.empty(m, dtype=np.int32)
                for cj in range(m):
                    tc = 0
                    for jj in range(k):
                        vv = c_vars[cj, jj]
                        if assignment[vv] != c_unsat[cj, jj]:
                            tc += 1
                    old_true_counts[cj] = tc
                best_break = 1 << 30
                best_vars = np.empty(k, dtype=np.int32)
                best_count = 0
                for t in range(num_unique):
                    v = unique_vars[t]
                    break_count = 0
                    assignment[v] ^= 1
                    for cj in range(m):
                        after_true = 0
                        for jj in range(k):
                            vv = c_vars[cj, jj]
                            if assignment[vv] != c_unsat[cj, jj]:
                                after_true += 1
                        if old_true_counts[cj] > 0 and after_true == 0:
                            break_count += 1
                    assignment[v] ^= 1
                    if break_count < best_break:
                        best_break = break_count
                        best_vars[0] = v
                        best_count = 1
                    elif break_count == best_break:
                        best_vars[best_count] = v
                        best_count += 1
                chosen = best_vars[np.random.randint(0, best_count)]

            assignment[chosen] ^= 1
        return max_flips, False


def _run_walksat_py(c_vars: np.ndarray, c_unsat: np.ndarray, n: int, k: int,
                    max_flips: int, p_noise: float, rng_seed: Optional[int],
                    return_assignment: bool = False):
    rng = np.random.default_rng(rng_seed)
    assignment = rng.integers(0, 2, size=n, dtype=np.int8)
    m = c_vars.shape[0]
    unique_buf = np.empty(k, dtype=np.int32)
    for flip in range(1, max_flips + 1):
        unsat = []
        for ci in range(m):
            if _eval_clause_truth_count(c_vars, c_unsat, assignment, ci) == 0:
                unsat.append(ci)
        if len(unsat) == 0:
            if return_assignment:
                return flip - 1, True, assignment.copy()
            return flip - 1, True

        ci = int(unsat[int(rng.integers(0, len(unsat)))])
        # unique variables from chosen clause
        num_unique = 0
        for j in range(k):
            v = int(c_vars[ci, j])
            if v not in unique_buf[:num_unique]:
                unique_buf[num_unique] = v
                num_unique += 1
        clause_vars = unique_buf[:num_unique].copy()

        if float(rng.random()) < float(p_noise):
            chosen = int(clause_vars[int(rng.integers(0, len(clause_vars)))])
        else:
            old_counts = np.zeros(m, dtype=np.int32)
            for cj in range(m):
                old_counts[cj] = _eval_clause_truth_count(c_vars, c_unsat, assignment, cj)
            best_break = None
            best = []
            for v in clause_vars:
                v = int(v)
                break_count = 0
                assignment[v] ^= 1
                for cj in range(m):
                    after = _eval_clause_truth_count(c_vars, c_unsat, assignment, cj)
                    if old_counts[cj] > 0 and after == 0:
                        break_count += 1
                assignment[v] ^= 1
                if (best_break is None) or (break_count < best_break):
                    best_break = break_count
                    best = [v]
                elif break_count == best_break:
                    best.append(v)
            chosen = int(best[int(rng.integers(0, len(best)))])
        assignment[chosen] ^= 1
    if return_assignment:
        return max_flips, False, assignment.copy()
    return max_flips, False


def _compute_break_make_lmake_for_var(c_vars: np.ndarray, c_unsat: np.ndarray,
                                      assignment: np.ndarray, num_true_lits: np.ndarray,
                                      affected_clauses: np.ndarray, v: int,
                                      w1: int, w2: int) -> Tuple[int, int]:
    break_count = 0
    make_1 = 0
    make_2 = 0
    for ci in affected_clauses:
        old = int(num_true_lits[ci])
        delta = 0
        for j in range(c_vars.shape[1]):
            if int(c_vars[ci, j]) == int(v):
                lit_true = int(assignment[v]) != int(c_unsat[ci, j])
                delta += -1 if lit_true else 1
        new = old + delta
        if old > 0 and new == 0:
            break_count += 1
        if old == 0 and new == 1:
            make_1 += 1
        if old == 1 and new == 2:
            make_2 += 1
    return break_count, int(w1 * make_1 + w2 * make_2)


def _build_var_to_clause_adjacency(c_vars: np.ndarray, n: int) -> List[np.ndarray]:
    adj = [set() for _ in range(n)]
    m, k = c_vars.shape
    for ci in range(m):
        for j in range(k):
            adj[int(c_vars[ci, j])].add(ci)
    return [np.array(sorted(s), dtype=np.int32) for s in adj]


def run_walksatlm_instance(clauses, n, k, max_flips: int = 100000, p_noise: float = 0.15,
                           w1: int = 6, w2: int = 5, rng_seed=None,
                           return_assignment: bool = False) -> Tuple[int, bool]:
    c_vars, c_unsat = clauses_to_arrays(clauses, k)
    rng = np.random.default_rng(rng_seed)
    assignment = rng.integers(0, 2, size=n, dtype=np.int8)
    m = c_vars.shape[0]
    num_true_lits = np.zeros(m, dtype=np.int32)
    for ci in range(m):
        num_true_lits[ci] = _eval_clause_truth_count(c_vars, c_unsat, assignment, ci)
    adj = _build_var_to_clause_adjacency(c_vars, n)

    for flip in range(1, max_flips + 1):
        unsat = np.flatnonzero(num_true_lits == 0)
        if unsat.size == 0:
            if return_assignment:
                return flip - 1, True, assignment.copy()
            return flip - 1, True
        ci = int(unsat[int(rng.integers(0, unsat.size))])

        unique_vars = []
        for j in range(k):
            v = int(c_vars[ci, j])
            if v not in unique_vars:
                unique_vars.append(v)

        metrics = []
        for v in unique_vars:
            brk, lmk = _compute_break_make_lmake_for_var(
                c_vars, c_unsat, assignment, num_true_lits, adj[v], v, w1, w2
            )
            metrics.append((v, brk, lmk))

        zero_break = [x for x in metrics if x[1] == 0]
        if zero_break:
            best_l = max(x[2] for x in zero_break)
            candidates = [x[0] for x in zero_break if x[2] == best_l]
            chosen = int(candidates[int(rng.integers(0, len(candidates)))])
        elif float(rng.random()) < float(p_noise):
            chosen = int(unique_vars[int(rng.integers(0, len(unique_vars)))])
        else:
            min_break = min(x[1] for x in metrics)
            subset = [x for x in metrics if x[1] == min_break]
            best_l = max(x[2] for x in subset)
            candidates = [x[0] for x in subset if x[2] == best_l]
            chosen = int(candidates[int(rng.integers(0, len(candidates)))])

        # Update num_true_lits by delta over affected clauses (handles repeats).
        for cj in adj[chosen]:
            delta = 0
            for j in range(k):
                if int(c_vars[cj, j]) == chosen:
                    lit_true = int(assignment[chosen]) != int(c_unsat[cj, j])
                    delta += -1 if lit_true else 1
            num_true_lits[cj] = int(num_true_lits[cj] + delta)
        assignment[chosen] ^= 1

    if return_assignment:
        return max_flips, False, assignment.copy()
    return max_flips, False


def run_walksat_instance(clauses, n, k, max_flips: int = 100000, p_noise: float = 0.5,
                         rng_seed=None, return_assignment: bool = False) -> Tuple[int, bool]:
    c_vars, c_unsat = clauses_to_arrays(clauses, k)
    if HAS_NUMBA and (not return_assignment):
        seed = int(0 if rng_seed is None else rng_seed)
        flips, solved = _walksat_kernel_numba(
            c_vars, c_unsat, int(n), int(k), int(max_flips), float(p_noise), seed
        )
        return int(flips), bool(solved)
    return _run_walksat_py(c_vars, c_unsat, n, k, max_flips, p_noise, rng_seed, return_assignment)


def evaluate_walksat_on_dataset(dataset, k, max_flips: int = 100000, p_noise: float = 0.5,
                                seed: int = 0, solver: str = "walksat",
                                walksatlm_w1: int = 6, walksatlm_w2: int = 5) -> dict:
    per_n = {}
    n_values = np.array(sorted(dataset.keys()), dtype=int)
    median_flips = []
    for n in n_values:
        flips = []
        solved = []
        for idx, inst in enumerate(dataset[int(n)]):
            run_seed = int(np.random.SeedSequence([seed, int(n), int(idx)]).generate_state(1)[0])
            if solver == "walksat":
                f, ok = run_walksat_instance(
                    inst["clauses"], n=int(n), k=k, max_flips=max_flips,
                    p_noise=p_noise, rng_seed=run_seed
                )
            elif solver == "walksatlm":
                f, ok = run_walksatlm_instance(
                    inst["clauses"], n=int(n), k=k, max_flips=max_flips,
                    p_noise=p_noise, w1=walksatlm_w1, w2=walksatlm_w2, rng_seed=run_seed
                )
            else:
                raise ValueError(f"Unknown solver {solver}")
            flips.append(float(f))
            solved.append(bool(ok))
        flips_arr = np.asarray(flips, dtype=float)
        solved_arr = np.asarray(solved, dtype=bool)
        per_n[int(n)] = {
            "median_flips": float(np.median(flips_arr)),
            "solve_rate": float(np.mean(solved_arr)),
            "mean_flips": float(np.mean(flips_arr)),
            "timeout_count": int(np.sum(~solved_arr)),
        }
        median_flips.append(per_n[int(n)]["median_flips"])
    fit_runtime = _fit_log_slope_safe(n_values, np.asarray(median_flips), positive_only=True)
    return {
        "per_n": per_n,
        "fitted_exponents_natural_log": {
            "median_runtime_or_flips_log_slope": float(fit_runtime["slope"])
        },
        "fitted_exponents_log2": {
            "median_runtime_or_flips_log_slope": float(fit_runtime["slope"] / np.log(2.0))
        },
    }


def run_algorithm_benchmark(
    n_min, n_max, k, r, test_size, base_seed,
    algorithms,
    depth=None,
    lr_delta_gamma=None,
    lr_delta_beta=None,
    lr_beta_schedule="decreasing",
    lr_angle_convention: str = "bm24",
    max_flips=100000,
    walksat_noise=0.5,
    walksatlm_noise=0.15,
    walksatlm_w1=6,
    walksatlm_w2=5,
    require_sat=True,
) -> dict:
    n_values = np.arange(int(n_min), int(n_max) + 1)
    dataset = generate_benchmark_dataset(
        n_values=n_values, k=int(k), r=float(r), test_size=int(test_size),
        base_seed=int(base_seed), require_sat=bool(require_sat)
    )

    algs = [str(a) for a in algorithms]
    out = {
        "settings": {
            "n_min": int(n_min), "n_max": int(n_max), "k": int(k), "r": float(r),
            "test_size": int(test_size), "base_seed": int(base_seed),
            "algorithms": algs, "require_sat": bool(require_sat),
            "max_flips": int(max_flips), "walksat_noise": float(walksat_noise),
            "walksatlm_noise": float(walksatlm_noise), "walksatlm_w1": int(walksatlm_w1),
            "walksatlm_w2": int(walksatlm_w2), "depth": depth,
            "lr_delta_gamma": lr_delta_gamma, "lr_delta_beta": lr_delta_beta,
            "lr_beta_schedule": lr_beta_schedule,
            "lr_angle_convention": str(lr_angle_convention),
        },
        "results": {},
        "fitted_exponents_natural_log": {},
        "fitted_exponents_log2": {},
    }

    if "lr_qaoa" in algs:
        if depth is None:
            raise ValueError("depth is required for lr_qaoa benchmark")
        if lr_delta_gamma is None or lr_delta_beta is None:
            raise ValueError("lr_qaoa requires --lr-dgamma and --lr-dbeta")
        betas, gammas = make_lr_angles(
            delta_gamma=float(lr_delta_gamma), delta_beta=float(lr_delta_beta),
            depth=int(depth), beta_schedule=str(lr_beta_schedule),
            angle_convention=str(lr_angle_convention),
        )
        out["settings"]["lr_qaoa_betas_bm24"] = [float(x) for x in betas]
        out["settings"]["lr_qaoa_gammas_bm24"] = [float(x) for x in gammas]
        qres = evaluate_qaoa_runtime_on_dataset(dataset, betas=betas, gammas=gammas)
        out["results"]["lr_qaoa"] = qres
        out["fitted_exponents_natural_log"]["lr_qaoa"] = qres["fitted_exponents_natural_log"]
        out["fitted_exponents_log2"]["lr_qaoa"] = qres["fitted_exponents_log2"]

    if "walksat" in algs:
        wres = evaluate_walksat_on_dataset(
            dataset, k=int(k), max_flips=int(max_flips),
            p_noise=float(walksat_noise), seed=int(base_seed), solver="walksat"
        )
        out["results"]["walksat"] = wres
        out["fitted_exponents_natural_log"]["walksat"] = wres["fitted_exponents_natural_log"]
        out["fitted_exponents_log2"]["walksat"] = wres["fitted_exponents_log2"]

    if "walksatlm" in algs:
        wlres = evaluate_walksat_on_dataset(
            dataset, k=int(k), max_flips=int(max_flips),
            p_noise=float(walksatlm_noise), seed=int(base_seed), solver="walksatlm",
            walksatlm_w1=int(walksatlm_w1), walksatlm_w2=int(walksatlm_w2),
        )
        out["results"]["walksatlm"] = wlres
        out["fitted_exponents_natural_log"]["walksatlm"] = wlres["fitted_exponents_natural_log"]
        out["fitted_exponents_log2"]["walksatlm"] = wlres["fitted_exponents_log2"]

    return out


# ---------------------------------------------------------------------------
# 5. Empirical exponent fit + bootstrap CI
# ---------------------------------------------------------------------------

def fit_log_slope(n_values: np.ndarray, mean_values: np.ndarray
                  ) -> Tuple[float, float, float]:
    """
    Fit  log(mean_values) = slope * n + intercept   in NATURAL log.

    Guard against log(0): mean_values that are <= 0 (only possible when *no*
    instance at that n produced any success, which becomes vanishingly rare
    once num_instances is large) are dropped from the fit and a warning is
    emitted. Returns (slope, intercept, stderr_of_slope).
    """
    n_values = np.asarray(n_values, dtype=float)
    mean_values = np.asarray(mean_values, dtype=float)
    mask = mean_values > 0
    if mask.sum() < 2:
        raise RuntimeError(
            "Fewer than 2 usable n-points (mean p_succ collapsed to 0); "
            "increase num_instances, n_max, or p_succ-friendlier angles."
        )
    if (~mask).any():
        LOG.warning("Dropping %d n-points where mean(p_succ)=0 from fit.",
                    int((~mask).sum()))
    res = linregress(n_values[mask], np.log(mean_values[mask]))
    return float(res.slope), float(res.intercept), float(res.stderr)


def bootstrap_slope_ci(per_n_samples: dict, n_resamples: int = 200,
                       rng_seed: int = 0) -> Tuple[float, float]:
    """
    Bootstrap the slope by resampling instances WITH replacement at each n.
    Returns (mean_slope, std_slope) across resamples.
    """
    rng = np.random.default_rng(rng_seed)
    n_arr = np.array(sorted(per_n_samples.keys()))
    slopes = []
    for _ in range(n_resamples):
        means = []
        for n in n_arr:
            arr = per_n_samples[n]
            sample = rng.choice(arr, size=len(arr), replace=True)
            means.append(sample.mean())
        means = np.array(means)
        m = means > 0
        if m.sum() < 2:
            continue
        slopes.append(linregress(n_arr[m], np.log(means[m])).slope)
    if not slopes:
        return float("nan"), float("nan")
    return float(np.mean(slopes)), float(np.std(slopes))


# ---------------------------------------------------------------------------
# 6. Theory side: phi_M, phi_pref, phi_full  (calls existing repo code)
# ---------------------------------------------------------------------------

def compute_theory_exponents(k: int, r: float,
                             betas: Iterable[float], gammas: Iterable[float],
                             num_iter: int = 200,
                             dz_threshold: float = 1e-6,
                             damping: float = 0.0):
    """
    Returns a dict with phi_M, phi_pref, phi_full (natural-log coefficients of
    n in E[p_succ] ~ exp(n * phi)) and the saddle-point diagnostics
    (iterations, residual).  k must be a power of 2.
    """
    q_float = np.log2(k)
    q = int(round(q_float))
    if 2 ** q != k:
        raise ValueError(
            f"k={k} is not a power of 2. The saddle-point implementation in "
            f"generalized_binomial_sum.py only supports 2^q-SAT.")
    betas = np.asarray(list(betas), dtype=float)
    gammas = np.asarray(list(gammas), dtype=float)
    # phi_pref alone, for sanity / reporting:
    if bm24_prefactor_exponent_ksat_all_subsets is not None:
        phi_pref_alone = bm24_prefactor_exponent_ksat_all_subsets(k=k, r=r)
    else:
        phi_pref_alone = bm24_prefactor_exponent_ksat(k=k, r=r, gammas=gammas)
    iters, z, residual, phi_m, phi_pref, phi_full, converged = (
        generalized_binomial_sum_full_exponent_ksat(
            q=q, r=r, betas=betas, gammas=gammas,
            num_iter=num_iter, dz_threshold=dz_threshold, damping=damping,
        )
    )
    # The full code may return complex phi (BM24 formula has Im part = 0 at
    # the saddle for the "real" regime, but numerically may have tiny Im).
    return {
        "k": k, "q": q, "r": r,
        "betas": betas.tolist(), "gammas": gammas.tolist(),
        "phi_M": complex(phi_m),
        "phi_pref": complex(phi_pref),
        "phi_pref_check": complex(phi_pref_alone),
        "phi_full": complex(phi_full),
        "saddle_iterations": int(iters),
        "saddle_residual": float(residual),
        "saddle_converged": bool(converged),
    }


def exact_finite_n_p1_prop4(k: int, r: float, beta: float, gamma: float,
                             n: int) -> float:
    """
    Exact finite-n unconditional E[p_succ] at p=1 implemented DIRECTLY from
    BM24 Proposition 4 / Eq. (A10).  This is the ground truth (verified
    against full Monte-Carlo brute force over the BM24 ensemble at n=2).

    NOTE on existing code: `generalized_flip_symmetric_expected_success_p1`
    in `generalized_binomial_sum.py` does NOT match this brute-force
    ground truth.  See the README / report header for evidence.
    """
    q = int(round(np.log2(k)))
    if 2 ** q != k:
        raise ValueError(f"k={k} not a power of 2")
    out = 0.0 + 0.0j
    pref = np.exp(-(r / 2 ** k) * n
                  * (1.0 + 4.0 * np.sin(gamma / 4.0) ** 2))
    cos2 = np.cos(beta / 2.0) ** 2
    sin2 = np.sin(beta / 2.0) ** 2
    sb2  = np.sin(beta) / 2.0
    e_minus = np.exp(-1j * gamma / 2.0)
    e_plus  = np.exp( 1j * gamma / 2.0)
    s2g4 = np.sin(gamma / 4.0) ** 2
    for na in range(n + 1):
        for nb in range(n + 1 - na):
            for nc in range(n + 1 - na - nb):
                nd = n - na - nb - nc
                # multinomial coefficient
                multi = math.factorial(n) / (
                    math.factorial(na) * math.factorial(nb)
                    * math.factorial(nc) * math.factorial(nd)
                )
                B_part = (cos2 ** na) * (sin2 ** nb) \
                         * (1j * sb2) ** nc * (-1j * sb2) ** nd
                exp_arg = r * n * (
                    4.0 * s2g4
                    * (((nb + na) / (2 * n)) ** k - (na / (2 * n)) ** k)
                    + (1.0 - e_minus) * ((nc + na) / (2 * n)) ** k
                    + (1.0 - e_plus)  * ((nd + na) / (2 * n)) ** k
                )
                out += multi * B_part * np.exp(exp_arg)
    return float(np.real(pref * out))


def exact_finite_n_p1_existing_code(k: int, r: float, beta: float,
                                    gamma: float, n: int) -> float:
    """
    Wrapper around `generalized_flip_symmetric_expected_success_p1` from the
    existing repo code.  KEPT FOR COMPARISON but NOT USED as ground truth
    because it is empirically inconsistent with brute force / BM24 Prop 4.
    """
    q = int(round(np.log2(k)))
    if 2 ** q != k:
        raise ValueError(f"k={k} not a power of 2")
    return float(np.real(generalized_flip_symmetric_expected_success_p1(
        q=q, r=r, betas=np.array([beta]), gammas=np.array([gamma]), n=n,
    )))


def asymptotic_slope_from_prop4(k: int, r: float, beta: float, gamma: float,
                                n_lo: int = 14, n_hi: int = 22) -> float:
    """
    Independent estimate of the natural-log asymptotic slope d/dn ln E[p_succ]
    at p=1, obtained by fitting `exact_finite_n_p1_prop4` over a window of
    moderate n.  Used as a 'second opinion' theory exponent for cross-check
    against `generalized_binomial_sum_full_exponent_ksat`.
    """
    ns = np.arange(n_lo, n_hi + 1)
    ys = np.array([exact_finite_n_p1_prop4(k, r, beta, gamma, int(n))
                   for n in ns])
    # Drop nonpositive (shouldn't happen for typical small angles)
    mask = ys > 0
    if mask.sum() < 2:
        return float("nan")
    return float(linregress(ns[mask], np.log(ys[mask])).slope)


# ---------------------------------------------------------------------------
# 7. End-to-end run / report
# ---------------------------------------------------------------------------

def run_comparison(n_min: int, n_max: int, k: int, r: float,
                   betas, gammas, num_instances: int, base_seed: int,
                   bootstrap_resamples: int = 200,
                   sanity_check_p1: bool = True) -> dict:
    """
    Main driver. Returns a dict containing:
      - empirical mean E[p_succ] per n with bootstrap CI,
      - the BM24 Prop-4 EXACT finite-n reference at p=1 (ground truth),
      - the existing-code finite-n value at p=1 (for comparison),
      - empirical exponent (slope of ln E[p_succ] vs n),
      - theory exponent from `generalized_binomial_sum_full_exponent_ksat`,
      - independent BM24 Prop-4-derived asymptotic slope (p=1 only),
      - all gaps in both natural-log and base-2 log units.
    """
    betas = np.asarray(betas, dtype=float)
    gammas = np.asarray(gammas, dtype=float)
    if len(betas) != len(gammas):
        raise ValueError("len(betas) must equal len(gammas)")
    p = len(betas)
    can_compare_p1_exact = (p == 1) and sanity_check_p1 \
        and (k & (k - 1) == 0)

    n_values = np.arange(n_min, n_max + 1)
    per_n_samples = {}
    means = np.zeros(len(n_values))
    stds = np.zeros(len(n_values))
    sat_fracs = np.zeros(len(n_values))
    exact_p1_prop4 = np.full(len(n_values), np.nan)
    exact_p1_code  = np.full(len(n_values), np.nan)
    timings = np.zeros(len(n_values))

    for idx, n in enumerate(n_values):
        t0 = time.time()
        samples = sample_p_succ_at_n(n=int(n), k=k, r=r,
                                     betas=betas, gammas=gammas,
                                     num_instances=num_instances,
                                     base_seed=base_seed)
        timings[idx] = time.time() - t0
        per_n_samples[int(n)] = samples
        means[idx] = samples.mean()
        stds[idx] = samples.std(ddof=1) / np.sqrt(num_instances)
        sat_fracs[idx] = float(np.mean(samples > 0))
        if can_compare_p1_exact:
            exact_p1_prop4[idx] = exact_finite_n_p1_prop4(
                k=k, r=r, beta=float(betas[0]), gamma=float(gammas[0]),
                n=int(n))
            exact_p1_code[idx] = exact_finite_n_p1_existing_code(
                k=k, r=r, beta=float(betas[0]), gamma=float(gammas[0]),
                n=int(n))
        LOG.info("n=%2d  E[p_succ]=%.6e  SE=%.2e  sat_frac=%.3f  "
                 "prop4=%s  code_exact=%s  (%.2fs)",
                 n, means[idx], stds[idx], sat_fracs[idx],
                 f"{exact_p1_prop4[idx]:.6e}" if not np.isnan(exact_p1_prop4[idx]) else "n/a",
                 f"{exact_p1_code[idx]:.6e}"  if not np.isnan(exact_p1_code[idx])  else "n/a",
                 timings[idx])

    slope, intercept, slope_se = fit_log_slope(n_values, means)
    boot_mean, boot_std = bootstrap_slope_ci(per_n_samples,
                                             n_resamples=bootstrap_resamples,
                                             rng_seed=base_seed + 1)

    theory = compute_theory_exponents(k=k, r=r, betas=betas, gammas=gammas)

    # Independent Prop-4-derived asymptotic slope (only available at p=1)
    if can_compare_p1_exact:
        prop4_slope = asymptotic_slope_from_prop4(
            k=k, r=r, beta=float(betas[0]), gamma=float(gammas[0]))
    else:
        prop4_slope = float("nan")

    LN2 = np.log(2.0)
    result = {
        "settings": {
            "n_min": n_min, "n_max": n_max,
            "k": k, "r": r,
            "betas": betas.tolist(), "gammas": gammas.tolist(),
            "p": p,
            "num_instances": num_instances,
            "base_seed": base_seed,
            "bootstrap_resamples": bootstrap_resamples,
        },
        "per_n": {
            "n": n_values.tolist(),
            "mean_p_succ": means.tolist(),
            "stderr_p_succ": stds.tolist(),
            "fraction_with_some_success": sat_fracs.tolist(),
            "exact_p1_prop4_ground_truth": [float(x) for x in exact_p1_prop4],
            "exact_p1_existing_code":      [float(x) for x in exact_p1_code],
            "wallclock_s": timings.tolist(),
        },
        "empirical_exponent_natural_log": {
            "slope": slope,
            "stderr_from_linregress": slope_se,
            "bootstrap_mean": boot_mean,
            "bootstrap_std": boot_std,
        },
        "empirical_exponent_log2": {
            "slope": slope / LN2,
            "stderr_from_linregress": slope_se / LN2,
            "bootstrap_mean": boot_mean / LN2,
            "bootstrap_std": boot_std / LN2,
        },
        "theory_exponent_existing_code_natural_log": {
            "phi_M":    theory["phi_M"].real,
            "phi_M_imag": theory["phi_M"].imag,
            "phi_pref": theory["phi_pref"].real,
            "phi_full": theory["phi_full"].real,
            "phi_full_imag": theory["phi_full"].imag,
            "saddle_iterations": theory["saddle_iterations"],
            "saddle_residual":   theory["saddle_residual"],
        },
        "theory_exponent_existing_code_log2": {
            "phi_M":    theory["phi_M"].real / LN2,
            "phi_pref": theory["phi_pref"].real / LN2,
            "phi_full": theory["phi_full"].real / LN2,
        },
        "theory_exponent_prop4_p1_natural_log": prop4_slope,
        "theory_exponent_prop4_p1_log2": prop4_slope / LN2 if not np.isnan(prop4_slope) else float("nan"),
        "gap_empirical_minus_existing_code_phi_full_nat": slope - theory["phi_full"].real,
        "gap_empirical_minus_existing_code_phi_full_log2": (slope - theory["phi_full"].real) / LN2,
        "gap_empirical_minus_prop4_nat": slope - prop4_slope,
        "gap_existing_code_minus_prop4_nat": theory["phi_full"].real - prop4_slope,
    }
    return result


# ---------------------------------------------------------------------------
# 8. CLI
# ---------------------------------------------------------------------------

def _format_report(res: dict) -> str:
    s = res["settings"]
    e_ln = res["empirical_exponent_natural_log"]
    e_l2 = res["empirical_exponent_log2"]
    t_ln = res["theory_exponent_existing_code_natural_log"]
    t_l2 = res["theory_exponent_existing_code_log2"]
    prop4 = res["theory_exponent_prop4_p1_natural_log"]

    lines = []
    lines.append("=" * 76)
    lines.append(" BM24 QAOA  --  finite-n simulator vs. asymptotic theory")
    lines.append("=" * 76)
    lines.append(f"k = {s['k']}   r = {s['r']}   p = {s['p']}")
    lines.append(f"betas  = {s['betas']}")
    lines.append(f"gammas = {s['gammas']}")
    lines.append(f"n in [{s['n_min']}, {s['n_max']}],  "
                 f"{s['num_instances']} instances per n,  seed={s['base_seed']}")
    lines.append("")
    lines.append(" Per-n results:")
    lines.append("   n   E[p_succ] empirical  SE         "
                 "Prop4 (BM24 GT)   Existing code")
    lines.append("   --  -------------------- ---------- "
                 "----------------- -----------------")
    pn = res["per_n"]
    for i, n in enumerate(pn["n"]):
        ex = pn["exact_p1_prop4_ground_truth"][i]
        co = pn["exact_p1_existing_code"][i]
        ex_str = f"{ex:.6e}" if not np.isnan(ex) else "       n/a       "
        co_str = f"{co:.6e}" if not np.isnan(co) else "       n/a       "
        lines.append(
            f"  {n:3d}  {pn['mean_p_succ'][i]:.6e}        "
            f"{pn['stderr_p_succ'][i]:.2e}   "
            f"{ex_str}     {co_str}"
        )
    lines.append("")
    lines.append(" Empirical exponent (natural log) -- slope of ln(E[p_succ]) vs n:")
    lines.append(f"   slope         = {e_ln['slope']:+.6f}")
    lines.append(f"   linregress SE = {e_ln['stderr_from_linregress']:.6f}")
    lines.append(f"   bootstrap     = {e_ln['bootstrap_mean']:+.6f}"
                 f"  +/- {e_ln['bootstrap_std']:.6f}")
    lines.append(" Empirical exponent (base-2 log = bits / variable):")
    lines.append(f"   slope         = {e_l2['slope']:+.6f}")
    lines.append(f"   bootstrap     = {e_l2['bootstrap_mean']:+.6f}"
                 f"  +/- {e_l2['bootstrap_std']:.6f}")
    lines.append("")
    lines.append(" Theory exponent (natural log) -- existing code"
                 " `generalized_binomial_sum_full_exponent_ksat`:")
    lines.append(f"   phi_M    = {t_ln['phi_M']:+.6f}    "
                 f"(Im={t_ln['phi_M_imag']:+.2e},  "
                 f"saddle iters={t_ln['saddle_iterations']}, "
                 f"residual={t_ln['saddle_residual']:.2e})")
    lines.append(f"   phi_pref = {t_ln['phi_pref']:+.6f}")
    lines.append(f"   phi_full = {t_ln['phi_full']:+.6f}    "
                 f"(Im={t_ln['phi_full_imag']:+.2e})")
    lines.append(" Theory exponent (base-2 log) -- existing code:")
    lines.append(f"   phi_M    = {t_l2['phi_M']:+.6f}")
    lines.append(f"   phi_pref = {t_l2['phi_pref']:+.6f}")
    lines.append(f"   phi_full = {t_l2['phi_full']:+.6f}")
    lines.append("")
    if not np.isnan(prop4):
        lines.append(" Theory exponent (natural log) -- INDEPENDENT BM24 Prop 4 fit (p=1 only):")
        lines.append(f"   slope of ln(prop4 exact) over n_lo..n_hi = {prop4:+.6f}")
        lines.append(f"                                   in base-2 = {prop4/np.log(2):+.6f}")
    lines.append("")
    lines.append(" GAPS (natural log):")
    lines.append(f"   empirical - existing-code phi_full      = "
                 f"{res['gap_empirical_minus_existing_code_phi_full_nat']:+.6f}")
    if not np.isnan(prop4):
        lines.append(f"   empirical - Prop 4 (ground truth)        = "
                     f"{res['gap_empirical_minus_prop4_nat']:+.6f}")
        lines.append(f"   existing-code phi_full - Prop 4          = "
                     f"{res['gap_existing_code_minus_prop4_nat']:+.6f}"
                     f"      <-- expected ~0 if existing code is correct;"
                     " nonzero indicates discrepancy.")
    lines.append("=" * 76)
    return "\n".join(lines)


def _format_benchmark_report(res: dict) -> str:
    s = res["settings"]
    lines = []
    lines.append("=" * 76)
    lines.append(" Benchmark mode: LR-QAOA vs WalkSAT vs WalkSATlm")
    lines.append("=" * 76)
    lines.append(
        f"k={s['k']} r={s['r']} n=[{s['n_min']},{s['n_max']}] "
        f"test_size={s['test_size']} sat_filter={s['require_sat']}"
    )
    lines.append(f"algorithms={s['algorithms']}")
    if "lr_qaoa" in s.get("algorithms", []):
        conv = s.get("lr_angle_convention", "bm24")
        lines.append(
            f"lr_qaoa: depth={s.get('depth')} schedule={s.get('lr_beta_schedule')} "
            f"angle_convention={conv} "
            f"dgamma={s.get('lr_delta_gamma')} dbeta={s.get('lr_delta_beta')}"
        )
        if "lr_qaoa_betas_bm24" in s and "lr_qaoa_gammas_bm24" in s:
            betas_bm = s["lr_qaoa_betas_bm24"]
            gammas_bm = s["lr_qaoa_gammas_bm24"]
            if len(betas_bm) <= 6:
                lines.append(f"  BM24-convention betas:  {betas_bm}")
                lines.append(f"  BM24-convention gammas: {gammas_bm}")
            else:
                lines.append(
                    f"  BM24-convention betas (first/last): "
                    f"{betas_bm[0]:+.4f} ... {betas_bm[-1]:+.4f}"
                )
                lines.append(
                    f"  BM24-convention gammas (first/last): "
                    f"{gammas_bm[0]:+.4f} ... {gammas_bm[-1]:+.4f}"
                )
    for alg, data in res["results"].items():
        lines.append("")
        lines.append(f"[{alg}]")
        per_n = data["per_n"]
        if alg == "lr_qaoa":
            lines.append(" n   mean_success   median_success   median_runtime")
            for n in sorted(per_n.keys()):
                d = per_n[n]
                lines.append(
                    f" {n:2d}  {d['mean_success']:.6e}  {d['median_success']:.6e}  {d['median_runtime']:.6e}"
                )
        else:
            lines.append(" n   median_flips   solve_rate   mean_flips   timeout_count")
            for n in sorted(per_n.keys()):
                d = per_n[n]
                lines.append(
                    f" {n:2d}  {d['median_flips']:.3f}  {d['solve_rate']:.3f}  "
                    f"{d['mean_flips']:.3f}  {d['timeout_count']}"
                )
        ln_fit = res["fitted_exponents_natural_log"].get(alg, {})
        l2_fit = res["fitted_exponents_log2"].get(alg, {})
        lines.append(f" natural-log exponents: {ln_fit}")
        lines.append(f" log2 exponents:        {l2_fit}")
    sc = summarize_benchmark_scaling(res)
    lines.append("")
    lines.append("--- Scaling comparison (notebook: ln median cost vs n) ---")
    lines.append(
        f" LR median(1/p)   log2 slope = {sc['lr_qaoa']['median_runtime_slope_log2']:.6f}"
    )
    lines.append(
        f" WalkSAT flips    log2 slope = {sc['walksat']['median_flips_slope_log2']:.6f}"
    )
    if "walksatlm" in res.get("results", {}):
        lines.append(
            f" WalkSATlm flips  log2 slope = {sc['walksatlm']['median_flips_slope_log2']:.6f}"
        )
    if sc["fit_ok"]:
        lines.append(
            f" LR better scaling than WalkSAT:    {sc['lr_beats_walksat_on_scaling']}"
        )
        lines.append(
            f" LR better scaling than WalkSATlm:  {sc['lr_beats_walksatlm_on_scaling']}"
        )
    else:
        lines.append(" (need >= 2 n values for scaling fit)")
    lines.append("=" * 76)
    return "\n".join(lines)


def _per_n_sorted(per_n: dict) -> List[int]:
    return sorted(int(k) for k in per_n.keys())


def _per_n_get(per_n: dict, n: int) -> dict:
    if n in per_n:
        return per_n[n]
    return per_n[str(n)]


def summarize_benchmark_scaling(res: dict) -> dict:
    """
  Notebook / BM24-style scaling comparison on a shared benchmark window.

  Fit ``ln(cost) ≈ slope_nat · n + const`` with ``cost = median(1/p_succ)`` for
  LR-QAOA and ``cost = median(flips)`` for WalkSAT / WalkSATlm (same fits as
  ``fitted_exponents_*`` on ``run_algorithm_benchmark`` output).

  **Lower** ``slope`` (natural log or log2) means cost grows more slowly with
  ``n`` — LR "wins on scaling" vs a classical solver when
  ``lr_slope < classical_slope`` (both finite).
    """
    fe_ln = res.get("fitted_exponents_natural_log", {})
    fe_l2 = res.get("fitted_exponents_log2", {})

    def _slope(alg: str, ln_key: str, l2_key: str) -> Tuple[float, float]:
        ln_alg = fe_ln.get(alg, {})
        l2_alg = fe_l2.get(alg, {})
        return (
            float(ln_alg.get(ln_key, float("nan"))),
            float(l2_alg.get(l2_key, float("nan"))),
        )

    lr_nat, lr_l2 = _slope(
        "lr_qaoa", "median_runtime_log_slope", "median_runtime_log_slope"
    )
    ws_nat, ws_l2 = _slope(
        "walksat", "median_runtime_or_flips_log_slope", "median_runtime_or_flips_log_slope"
    )
    lm_nat, lm_l2 = (
        _slope(
            "walksatlm",
            "median_runtime_or_flips_log_slope",
            "median_runtime_or_flips_log_slope",
        )
        if "walksatlm" in res.get("results", {})
        else (float("nan"), float("nan"))
    )

    def _beats(lr: float, cl: float) -> bool:
        return bool(np.isfinite(lr) and np.isfinite(cl) and lr < cl)

    beats_ws = _beats(lr_l2, ws_l2)
    beats_lm = _beats(lr_l2, lm_l2)
    ns = _per_n_sorted(res.get("results", {}).get("lr_qaoa", {}).get("per_n", {}))
    n_fit_ok = len(ns) >= 2

    return {
        "n_values": ns,
        "n_fit_points": len(ns),
        "fit_ok": n_fit_ok,
        "lr_qaoa": {
            "median_runtime_slope_nat": lr_nat,
            "median_runtime_slope_log2": lr_l2,
        },
        "walksat": {
            "median_flips_slope_nat": ws_nat,
            "median_flips_slope_log2": ws_l2,
        },
        "walksatlm": {
            "median_flips_slope_nat": lm_nat,
            "median_flips_slope_log2": lm_l2,
        },
        "lr_beats_walksat_on_scaling": beats_ws,
        "lr_beats_walksatlm_on_scaling": beats_lm,
        "beats_both_on_scaling": beats_ws and beats_lm,
    }


def check_benchmark_win(
    res: dict,
    *,
    stop_criterion: str = "scaling-exponent",
    win_mode: str = "all-n",
    equiv_flips_per_shot: float = 1.0,
    plot_summary: Optional[dict] = None,
) -> Tuple[bool, dict]:
    """
    Unified win check for sweeps and reports.

    ``stop_criterion``:
      - ``scaling-exponent``: LR log2 slope of ``median(1/p)`` < classical log2
        slope of ``median(flips)`` (notebook default).
      - ``per-n``: ``median(1/p)×equiv < median flips`` at each n (needs
        ``plot_summary`` from ``plot_benchmark_comparison`` or pass ``res`` only
        and we compute a lightweight per-n check).
      - ``both``: scaling-exponent **and** per-n ``all-n`` must hold.
    """
    if stop_criterion not in ("scaling-exponent", "per-n", "both"):
        raise ValueError("stop_criterion must be scaling-exponent, per-n, or both")

    scaling = summarize_benchmark_scaling(res)
    per_n_detail: dict = {}
    per_n_ok = False

    if stop_criterion in ("per-n", "both"):
        if plot_summary is None:
            plot_summary = _per_n_win_summary_from_res(res, equiv_flips_per_shot)
        ns = list(scaling["n_values"])
        ws_win = set(plot_summary.get("lr_beats_walksat_at_n", []))
        lm_win = set(plot_summary.get("lr_beats_walksatlm_at_n", []))
        if win_mode == "all-n":
            ok_ws = all(n in ws_win for n in ns)
            ok_lm = all(n in lm_win for n in ns)
        elif win_mode == "any-n":
            ok_ws = any(n in ws_win for n in ns)
            ok_lm = any(n in lm_win for n in ns)
        else:
            raise ValueError("win_mode must be 'all-n' or 'any-n'")
        per_n_ok = ok_ws and ok_lm
        per_n_detail = {
            "win_mode": win_mode,
            "lr_beats_walksat_at_n": sorted(ws_win),
            "lr_beats_walksatlm_at_n": sorted(lm_win),
            "beats_walksat": ok_ws,
            "beats_walksatlm": ok_lm,
            "beats_both": per_n_ok,
        }

    scaling_ok = bool(
        scaling["fit_ok"] and scaling["beats_both_on_scaling"]
    )
    if stop_criterion == "scaling-exponent":
        won = scaling_ok
    elif stop_criterion == "per-n":
        won = per_n_ok
    else:
        won = scaling_ok and per_n_ok

    detail = {
        "stop_criterion": stop_criterion,
        "win_mode": win_mode,
        "scaling": scaling,
        "per_n": per_n_detail if per_n_detail else None,
        "success": won,
    }
    return won, detail


def _per_n_win_summary_from_res(res: dict, equiv_flips_per_shot: float) -> dict:
    """Per-n LR vs classical cost without matplotlib."""
    results = res.get("results", {})
    lr_pn = results.get("lr_qaoa", {}).get("per_n", {})
    ws_pn = results.get("walksat", {}).get("per_n", {})
    ns = _per_n_sorted(lr_pn)
    equiv = float(equiv_flips_per_shot)
    lr_beats_ws = []
    lr_beats_lm = []
    for n in ns:
        med_rt = float(_per_n_get(lr_pn, n)["median_runtime"])
        qcost = med_rt * equiv
        if qcost < float(_per_n_get(ws_pn, n)["median_flips"]):
            lr_beats_ws.append(n)
        lm_pn = results.get("walksatlm", {}).get("per_n", {})
        if lm_pn:
            if qcost < float(_per_n_get(lm_pn, n)["median_flips"]):
                lr_beats_lm.append(n)
    out = {
        "equiv_flips_per_shot": equiv,
        "n_values": ns,
        "lr_beats_walksat_at_n": lr_beats_ws,
        "lr_beats_walksatlm_at_n": lr_beats_lm,
    }
    return out


def plot_benchmark_comparison(
    res: dict,
    output_path: Path,
    equiv_flips_per_shot: float = 1.0,
) -> dict:
    """
    Save a two-panel PNG comparing LR-QAOA vs WalkSAT / WalkSATlm.

    "LR beats classical at n" uses a tunable cost model (not a theorem):
        qaoa_cost(n) = median_runtime(n) * equiv_flips_per_shot
        classical_cost(n) = median_flips(n)   (WalkSAT or WalkSATlm)
    LR wins at n when qaoa_cost < classical_cost.

    Returns a small summary dict (winning n values, paths).
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError(
            "plot_benchmark_comparison requires matplotlib. "
            "Install with: pip install matplotlib"
        ) from e

    results = res.get("results", {})
    if "lr_qaoa" not in results:
        raise ValueError("benchmark results must include 'lr_qaoa' for plotting")
    if "walksat" not in results:
        raise ValueError("benchmark results must include 'walksat' for plotting")

    lr_pn = results["lr_qaoa"]["per_n"]
    ws_pn = results["walksat"]["per_n"]
    ns = _per_n_sorted(lr_pn)
    equiv = float(equiv_flips_per_shot)
    if equiv <= 0:
        raise ValueError("equiv_flips_per_shot must be positive")

    mean_succ = [float(_per_n_get(lr_pn, n)["mean_success"]) for n in ns]
    med_succ = [float(_per_n_get(lr_pn, n)["median_success"]) for n in ns]
    med_rt = [float(_per_n_get(lr_pn, n)["median_runtime"]) for n in ns]
    ws_flips = [float(_per_n_get(ws_pn, n)["median_flips"]) for n in ns]
    qaoa_cost = [rt * equiv for rt in med_rt]
    lr_beats_ws = [qc < wf for qc, wf in zip(qaoa_cost, ws_flips)]

    has_lm = "walksatlm" in results
    if has_lm:
        lm_pn = results["walksatlm"]["per_n"]
        lm_flips = [float(_per_n_get(lm_pn, n)["median_flips"]) for n in ns]
        lr_beats_lm = [qc < lf for qc, lf in zip(qaoa_cost, lm_flips)]
    else:
        lm_flips = None
        lr_beats_lm = None

    s = res.get("settings", {})
    depth = s.get("depth", "?")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    apply_figure_suptitle(
        fig,
        format_benchmark_title(
            {
                "k": s.get("k"),
                "r": s.get("r"),
                "n_min": s.get("n_min"),
                "n_max": s.get("n_max"),
                "test_size": s.get("test_size"),
                "seed": s.get("base_seed"),
                "depth": depth,
                "require_sat": s.get("require_sat"),
            },
            headline=f"Benchmark · equiv flips/shot={equiv:g}",
        ),
        fontsize=9,
    )

    ax = axes[0]
    ax.semilogy(ns, mean_succ, "o-", label="mean p_succ")
    ax.semilogy(ns, med_succ, "s--", label="median p_succ")
    ax.set_xlabel("n")
    ax.set_ylabel("success probability")
    ax.set_title("LR-QAOA success (same dataset)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, which="both", alpha=0.3)

    scaling = summarize_benchmark_scaling(res)

    ax = axes[1]
    ax.semilogy(ns, med_rt, "o-", color="C0", label="LR median 1/p")
    ax.semilogy(ns, ws_flips, "s-", color="C1", label="WalkSAT median flips")
    if lm_flips is not None:
        ax.semilogy(ns, lm_flips, "^--", color="C2", label="WalkSATlm median flips")
    ax.set_xlabel("n")
    ax.set_ylabel("cost (flips or 1/p)")
    lr_b2 = scaling["lr_qaoa"]["median_runtime_slope_log2"]
    ws_b2 = scaling["walksat"]["median_flips_slope_log2"]
    ax.set_title(
        f"Scaling costs (log2 slope: LR={lr_b2:.3f}, WS={ws_b2:.3f})"
        + (
            f", LM={scaling['walksatlm']['median_flips_slope_log2']:.3f}"
            if lm_flips is not None
            else ""
        )
    )
    ax.legend(loc="best", fontsize=7)
    ax.grid(True, which="both", alpha=0.3)

    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight", pad_inches=0.35)
    plt.close(fig)

    summary = {
        "plot_path": str(output_path),
        "equiv_flips_per_shot": equiv,
        "n_values": ns,
        "lr_beats_walksat_at_n": [n for n, ok in zip(ns, lr_beats_ws) if ok],
        "walksat_beats_lr_at_n": [n for n, ok in zip(ns, lr_beats_ws) if not ok],
        "scaling": scaling,
    }
    if lr_beats_lm is not None:
        summary["lr_beats_walksatlm_at_n"] = [n for n, ok in zip(ns, lr_beats_lm) if ok]
        summary["walksatlm_beats_lr_at_n"] = [n for n, ok in zip(ns, lr_beats_lm) if not ok]

    return summary


def _print_plot_summary(summary: dict) -> None:
    print("\n--- Benchmark plot summary ---")
    print(f"  saved: {summary['plot_path']}")
    print(f"  equiv_flips_per_shot: {summary['equiv_flips_per_shot']}")
    sc = summary.get("scaling") or {}
    if sc:
        lr_b2 = sc.get("lr_qaoa", {}).get("median_runtime_slope_log2", float("nan"))
        ws_b2 = sc.get("walksat", {}).get("median_flips_slope_log2", float("nan"))
        lm_b2 = sc.get("walksatlm", {}).get("median_flips_slope_log2", float("nan"))
        print(f"  scaling (log2 slope, lower=better): LR={lr_b2:.4f}  WalkSAT={ws_b2:.4f}  "
              f"WalkSATlm={lm_b2:.4f}")
        if sc.get("fit_ok"):
            print(f"  LR beats WalkSAT on scaling: {sc.get('lr_beats_walksat_on_scaling')}")
            print(f"  LR beats WalkSATlm on scaling: {sc.get('lr_beats_walksatlm_on_scaling')}")
        else:
            print("  scaling fit needs >= 2 n values in the benchmark window.")
    if summary["lr_beats_walksat_at_n"]:
        print(f"  LR beats WalkSAT at n = {summary['lr_beats_walksat_at_n']}")
    else:
        print("  LR beats WalkSAT at no n in this window (per-n equiv model).")
    if summary.get("walksat_beats_lr_at_n"):
        print(f"  WalkSAT beats LR at n = {summary['walksat_beats_lr_at_n']}")
    if "lr_beats_walksatlm_at_n" in summary:
        if summary["lr_beats_walksatlm_at_n"]:
            print(f"  LR beats WalkSATlm at n = {summary['lr_beats_walksatlm_at_n']}")
        else:
            print("  LR beats WalkSATlm at no n in this window (per-n equiv model).")
        if summary.get("walksatlm_beats_lr_at_n"):
            print(f"  WalkSATlm beats LR at n = {summary['walksatlm_beats_lr_at_n']}")
    print(
        "  Per-n: median(1/p)×equiv vs median flips. "
        "Scaling: ln(median cost) vs n (notebook / BM24 figure style)."
    )


def run_sanity_checks() -> None:
    # test_clause_semantics
    rng = np.random.default_rng(7)
    for n in [3, 4]:
        for _ in range(5):
            clauses = [generate_random_clause(n, 3, rng) for _ in range(4)]
            H = build_h_diagonal(clauses, n)
            for y in range(2 ** n):
                assignment = np.array([(y >> i) & 1 for i in range(n)], dtype=np.int8)
                sat = assignment_satisfies_formula(assignment, clauses)
                if sat != bool(H[y] == 0):
                    raise AssertionError("Clause semantics mismatch")

    # test_lr_angles (bm24 convention)
    betas, gammas = make_lr_angles(3.0, 6.0, depth=3, beta_schedule="decreasing")
    if not np.allclose(gammas, np.array([1.0, 2.0, 3.0])):
        raise AssertionError("LR gamma schedule mismatch")
    if not np.allclose(betas, np.array([4.0, 2.0, 0.0])):
        raise AssertionError("LR beta schedule mismatch")
    betas_nb, gammas_nb = make_lr_angles(
        3.0, 6.0, depth=3, beta_schedule="decreasing", angle_convention="notebook"
    )
    if not np.allclose(gammas_nb, 2.0 * np.array([1.0, 2.0, 3.0])):
        raise AssertionError("LR notebook convention should double gammas")
    if not np.allclose(betas_nb, 2.0 * np.array([4.0, 2.0, 0.0])):
        raise AssertionError("LR notebook convention should double betas")

    # test_qaoa_wrapper
    tiny = {2: [{"clauses": [[(0, False), (1, True)]], "num_solutions": 2}]}
    b = np.array([0.1]); g = np.array([0.2])
    out = evaluate_qaoa_runtime_on_dataset(tiny, b, g, eps=1e-300)
    H = build_h_diagonal(tiny[2][0]["clauses"], 2)
    p = per_instance_success_probability(run_qaoa(H, b, g, 2), H)
    if not np.isclose(out["per_n"][2]["mean_success"], p):
        raise AssertionError("QAOA wrapper mismatch")

    # test_walksat_return
    easy = [[(0, False), (0, False), (0, False)]]
    _, solved, assignment = run_walksat_instance(easy, n=1, k=3, max_flips=200,
                                                 p_noise=0.5, rng_seed=3, return_assignment=True)
    if solved and not assignment_satisfies_formula(assignment, easy):
        raise AssertionError("WalkSAT solved=True with invalid assignment")

    _, solved_lm, assignment_lm = run_walksatlm_instance(
        easy, n=1, k=3, max_flips=200, p_noise=0.5, w1=6, w2=5, rng_seed=9, return_assignment=True
    )
    if solved_lm and not assignment_satisfies_formula(assignment_lm, easy):
        raise AssertionError("WalkSATlm solved=True with invalid assignment")

    # test_cli_legacy
    legacy_cmd = [
        sys.executable, str(bm24_qaoa_sim_cli_path()),
        "--n-min", "2", "--n-max", "3", "--k", "2", "--r", "2",
        "--num-instances", "2", "--seed", "1", "--no-exact-check", "--quiet",
        "--no-auto-save",
    ]
    legacy = subprocess.run(legacy_cmd, capture_output=True, text=True, check=False)
    if legacy.returncode != 0:
        raise AssertionError(f"Legacy CLI smoke test failed: {legacy.stderr}")

    # test_cli_benchmark_smoke
    bench_cmd = [
        sys.executable, str(bm24_qaoa_sim_cli_path()),
        "--benchmark", "--n-min", "4", "--n-max", "5", "--k", "3", "--r", "4.2",
        "--test-size", "2", "--algorithms", "walksat", "walksatlm",
        "--max-flips", "1000", "--quiet", "--no-auto-save",
    ]
    bench = subprocess.run(bench_cmd, capture_output=True, text=True, check=False)
    if bench.returncode != 0:
        raise AssertionError(f"Benchmark CLI smoke test failed: {bench.stderr}")


def _shell_command_line() -> str:
    """Full command line: interpreter + sys.argv (copy-paste to re-run)."""
    return shlex.join([str(sys.executable)] + [str(x) for x in sys.argv])


def _build_run_meta(args: argparse.Namespace) -> dict:
    meta = {
        "shell_command": _shell_command_line(),
        "argv": [str(x) for x in sys.argv],
        "cwd": str(Path.cwd()),
        "python": sys.executable,
    }
    if args.benchmark:
        meta["mode"] = "benchmark"
        meta["algorithms"] = list(args.algorithms)
        meta["lr_beta_schedule"] = args.lr_beta_schedule
        meta["lr_angle_convention"] = args.lr_angle_convention
        meta["depth"] = args.depth
        meta["lr_dgamma"] = args.lr_dgamma
        meta["lr_dbeta"] = args.lr_dbeta
        if (
            "lr_qaoa" in args.algorithms
            and args.depth is not None
            and args.lr_dgamma is not None
            and args.lr_dbeta is not None
        ):
            betas_lr, gammas_lr = make_lr_angles(
                float(args.lr_dgamma),
                float(args.lr_dbeta),
                int(args.depth),
                str(args.lr_beta_schedule),
                angle_convention=str(args.lr_angle_convention),
            )
            meta["lr_qaoa_betas_bm24"] = [float(x) for x in betas_lr]
            meta["lr_qaoa_gammas_bm24"] = [float(x) for x in gammas_lr]
    else:
        meta["mode"] = "theory_comparison"
        meta["betas"] = [float(x) for x in args.betas]
        meta["gammas"] = [float(x) for x in args.gammas]
        meta["p_angles"] = len(args.betas)
    return meta


def _attach_run_meta(result: dict, run_meta: dict) -> dict:
    out = dict(result)
    out["run_meta"] = run_meta
    return out


def main():
    parser = argparse.ArgumentParser(
        description="BM24-compatible QAOA finite-n simulator and theory check.",
    )
    parser.add_argument("--n-min", type=int, default=8)
    parser.add_argument("--n-max", type=int, default=14)
    parser.add_argument("--k", type=int, default=8,
                        help="k-SAT clause length. Power of 2 for theory.")
    parser.add_argument("--r", type=float, default=176.54,
                        help="Clauses-to-variables ratio.")
    parser.add_argument("--betas", type=float, nargs="+", default=[0.5433996420760803],
                        help="QAOA mixer angles (length p).")
    parser.add_argument("--gammas", type=float, nargs="+", default=[0.7487887432117251],
                        help="QAOA phase angles (length p, keep small).")
    parser.add_argument("--num-instances", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20240501)
    parser.add_argument("--bootstrap-resamples", type=int, default=200)
    parser.add_argument("--no-exact-check", action="store_true",
                        help="Skip the p=1 exact-formula sanity column.")
    parser.add_argument("--save-json", type=str, default=None,
                        help="Also write JSON to this explicit path.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=BM24_DEFAULT_OUTPUT_DIR,
        help="Directory for auto-saved .json and .txt (MM-DD-HH-MM-SS stem, no year).",
    )
    parser.add_argument(
        "--no-auto-save",
        action="store_true",
        help="Do not write timestamped files under --output-dir.",
    )
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--benchmark", action="store_true",
                        help="Run benchmark mode (LR-QAOA / WalkSAT / WalkSATlm).")
    parser.add_argument("--algorithms", nargs="+",
                        default=["lr_qaoa", "walksat", "walksatlm"],
                        choices=["lr_qaoa", "walksat", "walksatlm"])
    parser.add_argument("--test-size", type=int, default=20)
    parser.add_argument("--depth", type=int, default=None)
    parser.add_argument("--lr-dgamma", type=float, default=None)
    parser.add_argument("--lr-dbeta", type=float, default=None)
    parser.add_argument("--lr-beta-schedule", type=str, default="decreasing",
                        choices=["decreasing", "increasing"])
    parser.add_argument(
        "--lr-angle-convention",
        type=str,
        default="bm24",
        choices=["bm24", "notebook"],
        help="Interpretation of --lr-dgamma/--lr-dbeta for LR-QAOA: bm24=native "
        "simulator units; notebook=legacy full-angle deltas (doubled to BM24).",
    )
    parser.add_argument("--max-flips", type=int, default=100000)
    parser.add_argument(
        "--walksat-noise",
        type=float,
        default=0.5,
        help="Random-walk probability for WalkSAT (notebook default 0.5).",
    )
    parser.add_argument(
        "--walksatlm-noise",
        type=float,
        default=0.15,
        help="Random-walk probability for WalkSATlm (default 0.15; WalkSAT uses 0.5).",
    )
    parser.add_argument("--walksatlm-w1", type=int, default=6)
    parser.add_argument("--walksatlm-w2", type=int, default=5)
    parser.add_argument("--sat-filter", dest="sat_filter", action="store_true")
    parser.add_argument("--no-sat-filter", dest="sat_filter", action="store_false")
    parser.set_defaults(sat_filter=True)
    parser.add_argument("--run-sanity-checks", action="store_true")
    parser.add_argument(
        "--plot-benchmark",
        action="store_true",
        help="After --benchmark, save a comparison PNG (requires matplotlib).",
    )
    parser.add_argument(
        "--plot-from-json",
        type=str,
        default=None,
        metavar="PATH",
        help="Only plot a saved benchmark JSON (no simulation).",
    )
    parser.add_argument(
        "--plot-path",
        type=str,
        default=None,
        help="Output PNG for --plot-benchmark (default: <output-dir>/<stem>-plot.png).",
    )
    parser.add_argument(
        "--plot-equiv-flips-per-shot",
        type=float,
        default=1.0,
        help="LR cost = median(1/p) × this; compare to WalkSAT median flips.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    if args.run_sanity_checks:
        run_sanity_checks()
        print("Sanity checks passed.")
        return

    if args.plot_from_json:
        with open(args.plot_from_json, encoding="utf-8") as f:
            plot_res = json.load(f)
        plot_path = Path(args.plot_path) if args.plot_path else (
            Path(args.plot_from_json).with_suffix(".png")
        )
        ps = plot_benchmark_comparison(
            plot_res, plot_path, equiv_flips_per_shot=args.plot_equiv_flips_per_shot
        )
        _print_plot_summary(ps)
        return

    # Always print the exact command first (including betas/gammas flags as passed).
    print(_shell_command_line(), flush=True)

    if args.benchmark:
        bres = run_algorithm_benchmark(
            n_min=args.n_min, n_max=args.n_max, k=args.k, r=args.r,
            test_size=args.test_size, base_seed=args.seed,
            algorithms=args.algorithms, depth=args.depth,
            lr_delta_gamma=args.lr_dgamma, lr_delta_beta=args.lr_dbeta,
            lr_beta_schedule=args.lr_beta_schedule,
            lr_angle_convention=args.lr_angle_convention,
            max_flips=args.max_flips, walksat_noise=args.walksat_noise,
            walksatlm_noise=args.walksatlm_noise, walksatlm_w1=args.walksatlm_w1,
            walksatlm_w2=args.walksatlm_w2, require_sat=args.sat_filter,
        )
        report_text = _format_benchmark_report(bres)
        print(report_text)
        res = _attach_run_meta(bres, _build_run_meta(args))
    else:
        res_raw = run_comparison(
            n_min=args.n_min, n_max=args.n_max,
            k=args.k, r=args.r,
            betas=args.betas, gammas=args.gammas,
            num_instances=args.num_instances,
            base_seed=args.seed,
            bootstrap_resamples=args.bootstrap_resamples,
            sanity_check_p1=not args.no_exact_check,
        )
        report_text = _format_report(res_raw)
        print(report_text)
        res = _attach_run_meta(res_raw, _build_run_meta(args))

    if not args.no_auto_save:
        out_dir = Path(args.output_dir).expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = make_run_stem("bench" if args.benchmark else "thy")
        run_paths = resolve_bm24_run_output_paths(out_dir, stem)
        json_path = run_paths["json"]
        txt_path = run_paths["run_dir"] / f"{stem}.txt"
        plot_stem = stem
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(_shell_command_line())
            f.write("\n\n")
            f.write(report_text)
            f.write("\n")
        print(f"\nSaved run to:\n  {json_path}\n  {txt_path}", flush=True)

    if args.benchmark and args.plot_benchmark:
        if args.plot_path:
            plot_path = Path(args.plot_path).expanduser().resolve()
        elif not args.no_auto_save:
            plot_path = run_paths["png"]
        else:
            plot_path = Path(BM24_DEFAULT_OUTPUT_DIR).expanduser().resolve() / (
                f"{make_run_stem('bench')}.png"
            )
        ps = plot_benchmark_comparison(
            res, plot_path, equiv_flips_per_shot=args.plot_equiv_flips_per_shot
        )
        _print_plot_summary(ps)
        if isinstance(res, dict) and not args.no_auto_save:
            res["plot_summary"] = ps
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(res, f, indent=2)

    if args.save_json:
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)
        print(f"\nSaved JSON (extra) to {args.save_json}", flush=True)


if __name__ == "__main__":
    main()
