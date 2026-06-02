"""
lr_eval_patch.py
================
Minimal, drop-in patch for the LR-QAOA / BM24 pipeline. Adds exactly four things,
each addressing a defect identified in the audit of LR_QAOA_benchmark_efficient.ipynb:

  (A) train/eval seed *namespace* separation + a runtime disjointness assertion
  (B) a single-fixed-n robust training objective (winsorized mean / median of ln(1/p))
  (C) a held-out bootstrap evaluation that reports the slope CI and P(slope<bar)
  (D) separated reporting of ANNEALED, TYPICAL, and MEDIAN exponents + the Jensen gap

It is written against the EXISTING simulator interface only:
    from bm24_qaoa_sim import (build_h_diagonal, generate_random_clause,
                               make_lr_angles, per_instance_success_probability, run_qaoa)
so it does not touch train_lr_notebook_protocol / phasecraft internals.

NOTHING heavy runs on import. `python lr_eval_patch.py --selftest` runs a synthetic
unit test of the metric math (no QAOA simulation), to validate the estimators.

Conventions
-----------
* p_succ in (0,1]; runtime = 1/p_succ.
* Exponent sign convention matches BM24: quantity ~ 2^{-c n}, so c = -slope of log2(quantity) vs n.
  - c_ann  from log2( mean_i p_i )
  - c_typ  from mean_i log2(p_i)              (geometric-mean exponent)
  - c_med  from log2( median_i p_i )          ( = -slope of median_i log2 p_i, monotone-invariant )
* Jensen gap per n (natural log, matches E[ln p] <= ln E[p]):  g(n) = ln(mean p) - mean(ln p) >= 0.
"""
from __future__ import annotations

import argparse
import hashlib
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

LN2 = float(np.log(2.0))
WALKSAT_BAR = 0.37  # default classical median-slope bar (k=8); make explicit, never hard-coded elsewhere

# --------------------------------------------------------------------------------------
# (A) Seed-namespace separation + disjointness assertion
# --------------------------------------------------------------------------------------
# Root problem in the notebook: the eval `dataset` (base_seed=27) and the training sets
# (training_h base_seed=27; proxy_h base_seed=27+77) live in OVERLAPPING n-ranges
# ({12..20} eval vs train_n=12 and proxy {12,14,16}). Whether instances coincide depends
# on SeedSequence construction inside the (unavailable) phasecraft generators -- i.e. it is
# NOT verifiable, and "probably disjoint" is not good enough for a head-to-head claim.
#
# Fix: assign train and eval to seed namespaces separated by a large, fixed offset, AND
# verify emptiness of the intersection at runtime by content-hashing the instances.

SEED_NAMESPACE_TRAIN = 1_000_000  # offsets are arbitrary but fixed + disjoint by construction
SEED_NAMESPACE_EVAL = 7_000_000
SEED_NAMESPACE_UNCOND = 9_000_000  # unconditional control set (no SAT filtering)


def namespaced_base_seed(user_seed: int, namespace: int) -> int:
    return int(user_seed) + int(namespace)


def instance_fingerprint(inst: dict) -> str:
    """Content hash of an instance, robust to dict ordering. Uses clauses if present,
    else the h_diag vector. Two instances with identical formula -> identical fingerprint."""
    h = hashlib.blake2b(digest_size=16)
    clauses = inst.get("clauses")
    if clauses is not None:
        # clauses: list of clause; clause: list of (var, is_negated)
        for clause in clauses:
            for (var, neg) in clause:
                h.update(int(var).to_bytes(4, "little", signed=False))
                h.update(b"\x01" if neg else b"\x00")
            h.update(b"|")
    else:
        hd = np.ascontiguousarray(inst["h_diag"]).astype(np.float64)
        h.update(hd.tobytes())
    return h.hexdigest()


def assert_disjoint(train_by_n: Dict[int, List[dict]],
                    eval_by_n: Dict[int, List[dict]]) -> Dict[int, int]:
    """Raise if any train instance equals any eval instance (per n). Returns overlap counts
    (all zero on success). Cheap: O(#instances) hashing."""
    overlaps: Dict[int, int] = {}
    for n in sorted(set(train_by_n) & set(eval_by_n)):
        tr = {instance_fingerprint(x) for x in train_by_n[n]}
        ev = [instance_fingerprint(x) for x in eval_by_n[n]]
        c = sum(1 for fp in ev if fp in tr)
        overlaps[n] = c
    bad = {n: c for n, c in overlaps.items() if c > 0}
    if bad:
        raise AssertionError(f"train/eval instance overlap detected (n -> #shared): {bad}")
    return overlaps


# --------------------------------------------------------------------------------------
# Thin simulator adapter (lets the file import + self-test without the real sim present)
# --------------------------------------------------------------------------------------
def _resolve_sim():
    """Return (make_lr_angles, run_qaoa, per_instance_success_probability) from bm24_qaoa_sim,
    or (None, None, None) if unavailable (self-test path uses a supplied p_succ_fn instead)."""
    try:
        from bm24_qaoa_sim import (  # type: ignore
            make_lr_angles,
            per_instance_success_probability,
            run_qaoa,
        )
        return make_lr_angles, run_qaoa, per_instance_success_probability
    except Exception:
        return None, None, None


def p_succ_vector(deltas: Tuple[float, float], depth: int, h_diags: Sequence[np.ndarray],
                  ns: Sequence[int], beta_schedule: str = "decreasing",
                  eps: float = 1e-300) -> np.ndarray:
    """Compute p_succ for each instance via the real simulator. ns[i] is the qubit count of h_diags[i]."""
    make_lr_angles, run_qaoa, per_p = _resolve_sim()
    if make_lr_angles is None:
        raise RuntimeError("bm24_qaoa_sim not importable; pass an explicit p_succ_fn for offline use.")
    dg, db = float(deltas[0]), float(deltas[1])
    out = np.empty(len(h_diags), dtype=np.float64)
    cache: Dict[int, Tuple] = {}
    for i, hd in enumerate(h_diags):
        n = int(ns[i])
        if n not in cache:
            cache[n] = make_lr_angles(dg, db, depth, beta_schedule=beta_schedule, angle_convention="bm24")
        betas, gammas = cache[n]
        psi = run_qaoa(hd, betas, gammas, n)
        out[i] = max(float(per_p(psi, hd)), eps)
    return out


# --------------------------------------------------------------------------------------
# (B) Fixed-n robust training objective
# --------------------------------------------------------------------------------------
def winsorized_mean(x: np.ndarray, lo: float = 0.10, hi: float = 0.10) -> float:
    x = np.sort(np.asarray(x, dtype=float))
    n = x.size
    klo, khi = int(np.floor(lo * n)), int(np.floor(hi * n))
    if klo + khi >= n:
        return float(np.median(x))
    if khi > 0:
        x = x[klo:n - khi]
    else:
        x = x[klo:]
    return float(np.mean(x))


def robust_log_runtime(p: np.ndarray, kind: str = "winsor", eps: float = 1e-300) -> float:
    """Objective to MINIMIZE: a robust aggregate of ln(1/p_succ) = -ln(p_succ).
    'winsor' = 10/10 winsorized mean; 'median' = median. Lower is better."""
    ell = -np.log(np.clip(np.asarray(p, dtype=float), eps, 1.0))  # ln(1/p) >= 0
    if kind == "median":
        return float(np.median(ell))
    return winsorized_mean(ell, 0.10, 0.10)


@dataclass
class TrainConfig:
    depth: int
    train_n: int = 16                     # single fixed n, past small-n saturation
    box_dg: Tuple[float, float] = (-1.8, -1.2)
    box_db: Tuple[float, float] = (0.7, 1.2)
    init: Tuple[float, float] = (-1.5, 1.0)
    objective: str = "winsor"             # "winsor" | "median"
    cobyla_maxiter: int = 120
    n_restarts: int = 4
    seed: int = 27
    beta_schedule: str = "decreasing"


def _clip_box(d, box_dg, box_db):
    return (float(np.clip(d[0], *box_dg)), float(np.clip(d[1], *box_db)))


def train_fixed_n(train_h: Sequence[np.ndarray], cfg: TrainConfig,
                  p_succ_fn: Optional[Callable] = None) -> dict:
    """Local box-constrained minimisation of the robust objective at a SINGLE fixed n.
    Uses SciPy COBYLA with box constraints + multi-restart. Returns best (dg,db) and trace.
    `p_succ_fn(deltas)->array` overrides the simulator (used in self-test / offline)."""
    from scipy.optimize import minimize

    box_dg, box_db = cfg.box_dg, cfg.box_db
    ns = [cfg.train_n] * len(train_h)

    def pvec(deltas):
        if p_succ_fn is not None:
            return np.asarray(p_succ_fn(deltas), dtype=float)
        return p_succ_vector(deltas, cfg.depth, train_h, ns, beta_schedule=cfg.beta_schedule)

    def obj(x):
        d = _clip_box(x, box_dg, box_db)
        return robust_log_runtime(pvec(d), kind=cfg.objective)

    cons = [
        {"type": "ineq", "fun": lambda x: x[0] - box_dg[0]},
        {"type": "ineq", "fun": lambda x: box_dg[1] - x[0]},
        {"type": "ineq", "fun": lambda x: x[1] - box_db[0]},
        {"type": "ineq", "fun": lambda x: box_db[1] - x[1]},
    ]
    rng = np.random.default_rng(namespaced_base_seed(cfg.seed, SEED_NAMESPACE_TRAIN) + cfg.depth)
    starts = [np.array(cfg.init, dtype=float)]
    for _ in range(cfg.n_restarts - 1):
        starts.append(np.array([rng.uniform(*box_dg), rng.uniform(*box_db)]))

    best = None
    trace = []
    for x0 in starts:
        res = minimize(obj, x0, method="COBYLA", constraints=cons,
                       options={"maxiter": cfg.cobyla_maxiter, "rhobeg": 0.15})
        d = _clip_box(res.x, box_dg, box_db)
        f = float(res.fun)
        trace.append({"start": x0.tolist(), "deltas": d, "obj": f})
        if best is None or f < best["obj"]:
            best = {"deltas": d, "obj": f}
    return {"best_deltas": best["deltas"], "best_obj": best["obj"],
            "objective": cfg.objective, "train_n": cfg.train_n, "trace": trace}


# --------------------------------------------------------------------------------------
# (C)+(D) Held-out evaluation: bootstrap slope CI + separated annealed/typical/median
# --------------------------------------------------------------------------------------
def _slope_log2(ns: np.ndarray, y: np.ndarray) -> float:
    """log2-slope of y vs n by OLS. y already in linear units; we fit log2(y)."""
    m = np.isfinite(y) & (y > 0)
    if m.sum() < 2:
        return float("nan")
    A = np.vstack([ns[m], np.ones(m.sum())]).T
    coef, *_ = np.linalg.lstsq(A, np.log2(y[m]), rcond=None)
    return float(coef[0])


def _exponents_from_psucc(per_n_p: Dict[int, np.ndarray]) -> dict:
    """Given p_succ arrays per n, return c_ann, c_typ, c_med and per-n aggregates."""
    ns = np.array(sorted(per_n_p), dtype=float)
    mean_p = np.array([per_n_p[int(n)].mean() for n in ns])
    mean_log2p = np.array([np.log2(per_n_p[int(n)]).mean() for n in ns])
    med_p = np.array([np.median(per_n_p[int(n)]) for n in ns])
    # c = -slope ; for the "typical" line we fit mean(log2 p) directly (already log space)
    c_ann = -_slope_log2(ns, mean_p)
    A = np.vstack([ns, np.ones(ns.size)]).T
    c_typ = -float(np.linalg.lstsq(A, mean_log2p, rcond=None)[0][0])
    c_med = -_slope_log2(ns, med_p)
    # Jensen gap per n (natural log): ln(mean p) - mean(ln p)
    jensen = {int(n): float(np.log(per_n_p[int(n)].mean()) - np.log(per_n_p[int(n)]).mean())
              for n in ns}
    # log-variance per variable (base-2), for the (ln2/2)*v gap-law check
    v_per_n = {int(n): float(np.var(np.log2(per_n_p[int(n)])) / n) for n in ns}
    skew_log = {int(n): float(_skew(np.log2(per_n_p[int(n)]))) for n in ns}
    return {
        "c_annealed": c_ann, "c_typical": c_typ, "c_median": c_med,
        "median_psucc_per_n": {int(n): float(med_p[i]) for i, n in enumerate(ns)},
        "mean_psucc_per_n": {int(n): float(mean_p[i]) for i, n in enumerate(ns)},
        "median_log2_inv_psucc_per_n": {int(n): float(-np.log2(med_p[i])) for i, n in enumerate(ns)},
        "jensen_gap_per_n_nat": jensen,
        "log2var_per_variable_per_n": v_per_n,
        "skew_log2p_per_n": skew_log,
    }


def _skew(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    s = x.std()
    if s == 0:
        return 0.0
    return float(np.mean(((x - x.mean()) / s) ** 3))


def evaluate_heldout(per_n_p: Dict[int, np.ndarray], bar: float = WALKSAT_BAR,
                     n_boot: int = 10000, seed: int = 12345) -> dict:
    """Full held-out report. `per_n_p[n]` = array of p_succ on the held-out set at size n.

    Returns metrics (1)-(11) requested:
      selected angles are recorded by the caller; here we report the scaling metrics.
    """
    ns = np.array(sorted(per_n_p), dtype=float)
    base = _exponents_from_psucc(per_n_p)

    # Median-cost slope = slope of log2( median(1/p) ) vs n. Note median(1/p)=1/median(p).
    med_runtime = np.array([1.0 / np.median(per_n_p[int(n)]) for n in ns])
    med_slope = _slope_log2(ns, med_runtime)

    # Bootstrap: resample instances WITHIN each n, recompute median-cost slope.
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot)
    arrs = [per_n_p[int(n)] for n in ns]
    sizes = [a.size for a in arrs]
    for b in range(n_boot):
        med_rt_b = np.empty(ns.size)
        for i, a in enumerate(arrs):
            idx = rng.integers(0, sizes[i], sizes[i])
            med_rt_b[i] = 1.0 / np.median(a[idx])
        boot[b] = _slope_log2(ns, med_rt_b)
    boot = boot[np.isfinite(boot)]
    ci = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))
    p_below = float(np.mean(boot < bar))

    # Monotonicity of per-n median p_succ (should be DECREASING in n)
    med_p = np.array([np.median(per_n_p[int(n)]) for n in ns])
    monotone = bool(np.all(np.diff(med_p) <= 0))

    # gap-law self-consistency: implied v from gap vs measured v
    gap_bits = base["c_typical"] - base["c_annealed"]
    implied_v = 2.0 * gap_bits / LN2  # if extensive-lognormal held
    measured_v = float(np.mean(list(base["log2var_per_variable_per_n"].values())))

    return {
        "median_cost_slope": med_slope,                     # (4)
        "median_cost_slope_CI95": ci,                       # (5)
        "P_slope_below_bar": p_below, "bar": bar,           # (6)
        "c_annealed": base["c_annealed"],                   # (7)
        "c_typical": base["c_typical"],                     # (8)
        "c_median": base["c_median"],
        "jensen_gap_per_n_nat": base["jensen_gap_per_n_nat"],   # (9)
        "median_psucc_per_n": base["median_psucc_per_n"],       # (2)
        "median_log2_inv_psucc_per_n": base["median_log2_inv_psucc_per_n"],  # (3)
        "per_n_median_monotone_decreasing": monotone,           # (10)
        "skew_log2p_per_n": base["skew_log2p_per_n"],
        "log2var_per_variable_per_n": base["log2var_per_variable_per_n"],
        "gap_law_implied_v": implied_v, "gap_law_measured_v": measured_v,
        "n_boot_effective": int(boot.size),
    }


def compare_sat_vs_uncond(per_n_p_sat: Dict[int, np.ndarray],
                          per_n_p_uncond: Dict[int, np.ndarray]) -> dict:
    """(11) Whether SAT-conditioned and unconditional median-cost slopes differ meaningfully."""
    ns = np.array(sorted(set(per_n_p_sat) & set(per_n_p_uncond)), dtype=float)
    s_sat = _slope_log2(ns, np.array([1.0 / np.median(per_n_p_sat[int(n)]) for n in ns]))
    s_unc = _slope_log2(ns, np.array([1.0 / np.median(per_n_p_uncond[int(n)]) for n in ns]))
    return {"slope_sat_conditioned": s_sat, "slope_unconditional": s_unc,
            "difference": float(s_sat - s_unc)}


# --------------------------------------------------------------------------------------
# Synthetic self-test of the metric math (no QAOA). Run: python lr_eval_patch.py --selftest
# --------------------------------------------------------------------------------------
def _selftest() -> None:
    rng = np.random.default_rng(0)
    # Build an exactly-lognormal ensemble: log2 p = -c_typ n + sqrt(v n) Z.
    c_typ_true, v_true = 0.45, 0.030
    ns = [14, 16, 18, 20]
    per_n_p = {}
    for n in ns:
        L = -c_typ_true * n + np.sqrt(v_true * n) * rng.standard_normal(4000)
        per_n_p[n] = np.clip(2.0 ** L, 1e-300, 1.0)
    rep = evaluate_heldout(per_n_p, n_boot=3000, seed=1)
    gap = rep["c_typical"] - rep["c_annealed"]
    print("SELF-TEST (exact log-normal ensemble)")
    print(f"  c_ann={rep['c_annealed']:.4f}  c_typ={rep['c_typical']:.4f}  c_med={rep['c_median']:.4f}")
    print(f"  measured gap c_typ-c_ann = {gap:.4f}   predicted (ln2/2)*v = {LN2/2*v_true:.4f}")
    print(f"  c_med - c_typ = {rep['c_median']-rep['c_typical']:+.4f}  (≈0 expected: symmetric log)")
    print(f"  median-cost slope = {rep['median_cost_slope']:.4f}  (≈ c_med = {rep['c_median']:.4f})")
    print(f"  slope 95% CI = [{rep['median_cost_slope_CI95'][0]:.3f}, {rep['median_cost_slope_CI95'][1]:.3f}]")
    print(f"  P(slope<{rep['bar']}) = {rep['P_slope_below_bar']:.3f}")
    print(f"  per-n median monotone decreasing = {rep['per_n_median_monotone_decreasing']}")
    print(f"  gap-law implied v={rep['gap_law_implied_v']:.4f} vs measured v={rep['gap_law_measured_v']:.4f}")
    # checks
    assert abs(gap - LN2 / 2 * v_true) < 0.004, "gap law off"
    assert abs(rep["c_median"] - rep["c_typical"]) < 0.01, "median!=typical under symmetry"
    assert all(g >= -1e-9 for g in rep["jensen_gap_per_n_nat"].values()), "Jensen gap negative!"
    # disjointness assertion smoke test
    a = {12: [{"clauses": [[(0, False), (1, True)]]}]}
    b = {12: [{"clauses": [[(0, False), (2, True)]]}]}
    assert assert_disjoint(a, b) == {12: 0}
    try:
        assert_disjoint(a, a); raise SystemExit("disjoint check failed to fire")
    except AssertionError:
        pass
    print("  all assertions passed.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        _selftest()
    else:
        print("Import this module; or run with --selftest. No experiment runs on import.")
