"""Numba-accelerated WalkSAT baselines (CPU; unchanged from notebook)."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
from numba import njit
from tqdm import tqdm


@njit(cache=True)
def fast_walksat_solver(n, c_vars, c_signs, max_flips, p_noise):
    assignment = np.random.randint(0, 2, n)
    m, k_sat = c_vars.shape
    unsat_buffer = np.empty(m, dtype=np.int32)
    for flip in range(max_flips):
        unsat_count = 0
        for i in range(m):
            is_sat = False
            for kk in range(k_sat):
                v = c_vars[i, kk]
                if assignment[v] == c_signs[i, kk]:
                    is_sat = True
                    break
            if not is_sat:
                unsat_buffer[unsat_count] = i
                unsat_count += 1
        if unsat_count == 0:
            return flip + 1
        target_c_idx = unsat_buffer[np.random.randint(0, unsat_count)]
        if np.random.random() < p_noise:
            var_to_flip = c_vars[target_c_idx, np.random.randint(0, k_sat)]
        else:
            best_var = -1
            min_breaks = 999999
            for kk in range(k_sat):
                candidate_var = c_vars[target_c_idx, kk]
                assignment[candidate_var] = 1 - assignment[candidate_var]
                current_breaks = 0
                for i_scan in range(m):
                    c_sat = False
                    for kk2 in range(k_sat):
                        v_scan = c_vars[i_scan, kk2]
                        if assignment[v_scan] == c_signs[i_scan, kk2]:
                            c_sat = True
                            break
                    if not c_sat:
                        current_breaks += 1
                if current_breaks < min_breaks:
                    min_breaks = current_breaks
                    best_var = candidate_var
                assignment[candidate_var] = 1 - assignment[candidate_var]
            var_to_flip = best_var
        assignment[var_to_flip] = 1 - assignment[var_to_flip]
    return max_flips


@njit(cache=True)
def walksatlm_paper_kernel(n, c_vars, c_signs, max_flips, p_noise, w1, w2):
    m = c_vars.shape[0]
    k_sat = c_vars.shape[1]
    degrees = np.zeros(n, dtype=np.int32)
    for i in range(m):
        for kk in range(k_sat):
            degrees[c_vars[i, kk]] += 1
    max_degree = 0
    for i in range(n):
        if degrees[i] > max_degree:
            max_degree = degrees[i]
    adj_indices = np.full((n, max_degree), -1, dtype=np.int32)
    adj_signs = np.full((n, max_degree), -1, dtype=np.int32)
    current_fill = np.zeros(n, dtype=np.int32)
    for i in range(m):
        for kk in range(k_sat):
            v = c_vars[i, kk]
            s = c_signs[i, kk]
            pos = current_fill[v]
            adj_indices[v, pos] = i
            adj_signs[v, pos] = s
            current_fill[v] += 1
    assignment = np.random.randint(0, 2, n)
    num_true_lits = np.zeros(m, dtype=np.int32)
    for i in range(m):
        count = 0
        for kk in range(k_sat):
            if assignment[c_vars[i, kk]] == c_signs[i, kk]:
                count += 1
        num_true_lits[i] = count
    unsat_buffer = np.empty(m, dtype=np.int32)
    for flip in range(1, max_flips + 1):
        unsat_count = 0
        for i in range(m):
            if num_true_lits[i] == 0:
                unsat_buffer[unsat_count] = i
                unsat_count += 1
        if unsat_count == 0:
            return flip
        target_c_idx = unsat_buffer[np.random.randint(0, unsat_count)]
        candidates = c_vars[target_c_idx]
        cand_breaks = np.zeros(k_sat, dtype=np.int32)
        cand_lmakes = np.zeros(k_sat, dtype=np.int32)
        has_zero_break = False
        for kk in range(k_sat):
            var = candidates[kk]
            current_break = 0
            make_1 = 0
            make_2 = 0
            deg = current_fill[var]
            for idx in range(deg):
                c_idx = adj_indices[var, idx]
                s = adj_signs[var, idx]
                lit_count = num_true_lits[c_idx]
                if assignment[var] == s:
                    if lit_count == 1:
                        current_break += 1
                else:
                    if lit_count == 0:
                        make_1 += 1
                    elif lit_count == 1:
                        make_2 += 1
            cand_breaks[kk] = current_break
            cand_lmakes[kk] = w1 * make_1 + w2 * make_2
            if current_break == 0:
                has_zero_break = True
        best_var = -1
        if has_zero_break:
            best_val = -1e9
            for kk in range(k_sat):
                if cand_breaks[kk] == 0:
                    score = cand_lmakes[kk]
                    if score > best_val:
                        best_val = score
                        best_var = candidates[kk]
                    elif score == best_val and np.random.random() < 0.5:
                        best_var = candidates[kk]
        elif np.random.random() < p_noise:
            best_var = candidates[np.random.randint(0, k_sat)]
        else:
            min_b = 999999
            max_l = -999999
            for kk in range(k_sat):
                b = cand_breaks[kk]
                lmk = cand_lmakes[kk]
                if b < min_b:
                    min_b = b
                    max_l = lmk
                    best_var = candidates[kk]
                elif b == min_b:
                    if lmk > max_l:
                        max_l = lmk
                        best_var = candidates[kk]
                    elif lmk == max_l and np.random.random() < 0.5:
                        best_var = candidates[kk]
        assignment[best_var] = 1 - assignment[best_var]
        deg = current_fill[best_var]
        for idx in range(deg):
            c_idx = adj_indices[best_var, idx]
            s = adj_signs[best_var, idx]
            if assignment[best_var] == s:
                num_true_lits[c_idx] += 1
            else:
                num_true_lits[c_idx] -= 1
    return max_flips


def clauses_to_numba_arrays(clauses, k: int):
    m = len(clauses)
    c_vars = np.zeros((m, k), dtype=np.int32)
    c_signs = np.zeros((m, k), dtype=np.int32)
    for i, clause in enumerate(clauses):
        for j, (var, is_negated) in enumerate(clause):
            c_vars[i, j] = int(var)
            c_signs[i, j] = 1 - int(bool(is_negated))
    return c_vars, c_signs


def _seed_for_instance(base_seed: int, n: int, idx: int) -> int:
    return int(np.random.SeedSequence([base_seed, n, idx]).generate_state(1)[0])


def evaluate_classical_once(dataset: Dict[int, List[dict]], cfg: dict) -> dict:
    p_ws = float(cfg.get("walksat_p_noise", cfg.get("p_noise", 0.5)))
    p_lm = float(cfg.get("walksatlm_p_noise", 0.15))
    max_flips = int(cfg["max_flips"])
    w1, w2 = int(cfg["walksatlm_w1"]), int(cfg["walksatlm_w2"])
    base_seed = int(cfg["seed"])
    out = {"walksat": {}, "walksatlm": {}}
    for n in sorted(dataset.keys()):
        ws_flips, lm_flips = [], []
        for idx, inst in enumerate(tqdm(dataset[n], desc=f"classical n={n}", leave=False)):
            seed = _seed_for_instance(base_seed, n, idx)
            np.random.seed(seed)
            ws_flips.append(
                int(fast_walksat_solver(n, inst["c_vars"], inst["c_signs"], max_flips, p_ws))
            )
            np.random.seed(seed)
            lm_flips.append(
                int(walksatlm_paper_kernel(n, inst["c_vars"], inst["c_signs"], max_flips, p_lm, w1, w2))
            )
        out["walksat"][n] = float(np.median(ws_flips))
        out["walksatlm"][n] = float(np.median(lm_flips))
    return out
