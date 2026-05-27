"""
Saddle dominance study with large multi-start Newton scans.

Implements:
  - Exhaustive multi-start at certified gamma=0.14 for p=1,2,3
  - Pairwise root separation + time-reversal symmetry diagnostics
  - Fine-gamma grid scan
  - Correct per-variable scaling exponent computation
"""

from __future__ import annotations

from dataclasses import dataclass
import argparse
import json
from pathlib import Path
import sys
from time import time

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from symmetry_reduction.core import alpha_linear_coefficients, bm24_clause_arity, build_structure_matrix, compute_b_s
from symmetry_reduction.newton_saddle_q import _get_active_indices, _compute_weights_from_u, newton_solve_q, solve_8sat_saddle_general


@dataclass(frozen=True)
class GeneralData:
    p: int
    q: int
    r: float
    beta: float
    gamma: float
    d: int
    active_idx: np.ndarray
    A: np.ndarray
    A_act: np.ndarray
    log_b: np.ndarray
    coeff: np.ndarray
    coeff_act: np.ndarray
    c_2q_act: np.ndarray


def build_data(p: int, beta: float, gamma: float, q: int = 3, r: float = 176.54) -> GeneralData:
    A = build_structure_matrix(p, q=q)
    d = int(A.shape[0])
    active_idx = _get_active_indices(d)
    A_act = A[active_idx]
    betas = np.full(p, beta, dtype=float)
    gammas = np.full(p, gamma, dtype=float)
    b_s = compute_b_s(p, betas)
    log_b = np.log(b_s + 1e-300 * (1 - np.sign(np.abs(b_s))))
    log_b = np.where(np.isfinite(log_b), log_b, -700.0)
    coeff = alpha_linear_coefficients(p, gammas, q=q, r=r)
    coeff_act = coeff[active_idx]
    c_2q_act = np.power(coeff_act, bm24_clause_arity(q))
    return GeneralData(
        p=p,
        q=q,
        r=float(r),
        beta=float(beta),
        gamma=float(gamma),
        d=d,
        active_idx=active_idx,
        A=A,
        A_act=A_act,
        log_b=log_b,
        coeff=coeff,
        coeff_act=coeff_act,
        c_2q_act=c_2q_act,
    )


def get_known_root_active(data: GeneralData) -> np.ndarray:
    betas = np.full(data.p, data.beta, dtype=float)
    gammas_target = np.full(data.p, data.gamma, dtype=float)
    y_full, _coeff, converged, residual = solve_8sat_saddle_general(
        p=data.p,
        betas=betas,
        gammas_target=gammas_target,
        q=data.q,
        r=data.r,
        t_start=1e-6,
        t_factor=1.08,
        t_max_steps=1500,
        newton_tol=1e-12,
        newton_max_iter=1200,
        target_residual=1e-6,
        max_newton_calls=6000,
        verbose=False,
    )
    if not converged and (not np.isfinite(residual) or residual > 1e-4):
        raise RuntimeError(f"Failed to build known root for p={data.p}, gamma={data.gamma}, residual={residual}")
    return y_full[data.active_idx]


def generate_multistart_inits(y_known: np.ndarray, n_starts: int, rng: np.random.Generator) -> list[np.ndarray]:
    n = len(y_known)
    max_scale = max(float(np.max(np.abs(y_known))), 1.0)
    starts: list[np.ndarray] = []

    n1 = n_starts // 10
    n2 = n_starts // 5
    n3 = 3 * n_starts // 10
    n4 = n_starts // 5
    n5 = n_starts - (n1 + n2 + n3 + n4)

    for _ in range(n1):
        eps = 0.001 * max_scale * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
        starts.append(y_known + eps)
    for _ in range(n2):
        mags = np.abs(y_known)
        mags = np.where(mags > 1, mags, max_scale)
        phases = rng.uniform(0, 2 * np.pi, n)
        starts.append(mags * np.exp(1j * phases))
    for _ in range(n3):
        log_scale = rng.uniform(-2, 2, n)
        mags = (10.0 ** log_scale) * np.maximum(np.abs(y_known), 1.0)
        phases = rng.uniform(0, 2 * np.pi, n)
        starts.append(mags * np.exp(1j * phases))
    for _ in range(n4):
        starts.append(max_scale * (rng.standard_normal(n) + 1j * rng.standard_normal(n)))
    for _ in range(n5):
        r_mag = max_scale * (10.0 ** rng.uniform(-1, 2))
        phases = rng.uniform(0, 2 * np.pi, n)
        starts.append(r_mag * np.exp(1j * phases) * rng.standard_normal(n))
    return starts


def run_one_start(y_start: np.ndarray, data: GeneralData) -> tuple[bool, np.ndarray, float, int]:
    u_start = data.coeff_act * y_start
    u_new, conv, res, nit = newton_solve_q(
        u_init=u_start,
        c_2q_active=data.c_2q_act,
        active_idx=data.active_idx,
        A=data.A,
        log_b=data.log_b,
        d=data.d,
        q=data.q,
        tol=1e-8,
        max_iter=200,
    )
    found = bool(conv) or (np.isfinite(res) and float(res) < 1e-6)
    y_found = np.zeros_like(y_start)
    if found:
        for i, c in enumerate(data.coeff_act):
            if abs(c) > 1e-14:
                y_found[i] = u_new[i] / c
    return found, y_found, float(res), int(nit)


def rel_dist(y: np.ndarray, rep: np.ndarray) -> float:
    scale = max(float(np.max(np.abs(y))), 1.0)
    return float(np.max(np.abs(y - rep)) / scale)


def cluster_roots(roots: list[np.ndarray], tol: float = 1e-4) -> list[dict]:
    clusters: list[dict] = []
    for y in roots:
        merged = False
        for c in clusters:
            if rel_dist(y, c["rep"]) < tol:
                c["count"] += 1
                merged = True
                break
        if not merged:
            clusters.append({"rep": y.copy(), "count": 1})
    return clusters


def polish_root(y_rep: np.ndarray, data: GeneralData) -> tuple[np.ndarray, float]:
    u_init = data.coeff_act * y_rep
    u_new, conv, res, _it = newton_solve_q(
        u_init=u_init,
        c_2q_active=data.c_2q_act,
        active_idx=data.active_idx,
        A=data.A,
        log_b=data.log_b,
        d=data.d,
        q=data.q,
        tol=1e-14,
        max_iter=2000,
    )
    y = np.zeros_like(y_rep)
    for i, c in enumerate(data.coeff_act):
        if abs(c) > 1e-14:
            y[i] = u_new[i] / c
    if (not conv) and (not np.isfinite(res) or float(res) > 1e-8):
        # Keep the best available point while preserving diagnostics.
        return y, float(res)
    return y, float(res)


def compute_scaling_exponent(y_active: np.ndarray, data: GeneralData) -> dict:
    y_full = np.zeros(data.d, dtype=complex)
    y_full[data.active_idx] = y_active
    coeff_y = data.coeff * y_full
    log_terms = data.log_b + (coeff_y @ data.A)
    shift = np.max(log_terms.real)
    exp_terms = np.exp(log_terms - shift)
    Z = np.sum(exp_terms)
    F_value = shift + np.log(Z)

    w = exp_terms / Z
    E_full = w @ data.A.T
    grad_full = data.coeff * E_full

    power_sum = np.sum(np.power(grad_full, 2 * data.q))
    scaling = F_value + (2**data.q - 1) * power_sum
    c_exp = -float(np.real(scaling)) / float(np.log(2.0))
    return {
        "F_value_real": float(np.real(F_value)),
        "F_value_imag": float(np.imag(F_value)),
        "power_sum_real": float(np.real(power_sum)),
        "power_sum_imag": float(np.imag(power_sum)),
        "scaling_Re": float(np.real(scaling)),
        "scaling_Im": float(np.imag(scaling)),
        "exponent_c": c_exp,
    }


def root_separation(y_a: np.ndarray, y_b: np.ndarray) -> dict:
    scale = max(float(np.max(np.abs(y_a))), float(np.max(np.abs(y_b))), 1.0)
    abs_sep = float(np.max(np.abs(y_a - y_b)))
    rel_sep = abs_sep / scale
    if rel_sep < 1e-6:
        diag = "SAME_ROOT"
    elif rel_sep > 1e-2:
        diag = "DIFFERENT"
    else:
        diag = "AMBIGUOUS"
    return {"abs_separation": abs_sep, "rel_separation": rel_sep, "diagnosis": diag}


def time_reversal_perm_on_active(p: int) -> np.ndarray:
    n = 2 * p + 1
    d = 2**n
    sigma_full = np.zeros(d, dtype=int)
    for alpha in range(d):
        new_alpha = 0
        for j in range(n):
            if (alpha >> j) & 1:
                new_j = 2 * p - j
                new_alpha |= 1 << new_j
        sigma_full[alpha] = new_alpha
    active = [a for a in range(d) if bin(a).count("1") >= 2]
    pos = {a: i for i, a in enumerate(active)}
    sigma = np.zeros(len(active), dtype=int)
    for i, a in enumerate(active):
        sigma[i] = pos[sigma_full[a]]
    return sigma


def check_symmetry_relation(y_a: np.ndarray, y_b: np.ndarray, sigma_active: np.ndarray, tol: float = 1e-4) -> dict:
    y_perm = y_a[sigma_active]
    scale = max(float(np.max(np.abs(y_a))), 1.0)
    rel_diff = float(np.max(np.abs(y_perm - y_b)) / scale)
    return {"rel_diff_after_permutation": rel_diff, "is_symmetry_related": bool(rel_diff < tol)}


def run_multistart_report(
    p: int,
    beta: float,
    gamma: float,
    r: float,
    q: int,
    n_starts: int,
    merge_tol: float,
    seed: int,
) -> dict:
    t0 = time()
    data = build_data(p=p, beta=beta, gamma=gamma, r=r, q=q)
    y_known = get_known_root_active(data)
    rng = np.random.default_rng(seed + p * 1000 + int(round(gamma * 1e6)))
    starts = generate_multistart_inits(y_known=y_known, n_starts=n_starts, rng=rng)

    found_roots: list[np.ndarray] = []
    n_diverged = 0
    for i, y0 in enumerate(starts, start=1):
        found, y, _res, _nit = run_one_start(y_start=y0, data=data)
        if found:
            found_roots.append(y)
        else:
            n_diverged += 1
        if i % max(1000, n_starts // 20) == 0:
            print(f"p={p}, gamma={gamma:.6f}: starts {i}/{n_starts}", flush=True)

    clusters = cluster_roots(found_roots, tol=merge_tol)
    roots_payload = []
    for rid, c in enumerate(clusters):
        y_pol, res_pol = polish_root(c["rep"], data)
        scale_info = compute_scaling_exponent(y_pol, data)
        roots_payload.append(
            {
                "root_id": rid,
                "n_starts_converged_here": int(c["count"]),
                "residual_polished": float(res_pol),
                "re_phi": float(scale_info["scaling_Re"]),
                "exponent_c": float(scale_info["exponent_c"]),
                "y_real": [float(np.real(v)) for v in y_pol],
                "y_imag": [float(np.imag(v)) for v in y_pol],
                "scaling_details": scale_info,
            }
        )

    return {
        "p": int(p),
        "beta": float(beta),
        "gamma": float(gamma),
        "r": float(r),
        "n_starts": int(n_starts),
        "n_converged": int(len(found_roots)),
        "n_diverged": int(n_diverged),
        "n_distinct_roots": int(len(clusters)),
        "merge_tol": float(merge_tol),
        "roots": roots_payload,
        "elapsed_seconds": float(time() - t0),
    }


def diagnose_root_pairs(report: dict) -> dict:
    roots = report["roots"]
    p = int(report["p"])
    if len(roots) < 2:
        return {"pairs": [], "note": "fewer than 2 roots"}
    sigma = time_reversal_perm_on_active(p)
    pairs = []
    y_list = [np.array(r["y_real"]) + 1j * np.array(r["y_imag"]) for r in roots]
    for i in range(len(y_list)):
        for j in range(i + 1, len(y_list)):
            sep = root_separation(y_list[i], y_list[j])
            sym = check_symmetry_relation(y_list[i], y_list[j], sigma_active=sigma, tol=1e-4)
            pairs.append({"root_pair": [i, j], **sep, "symmetry_check": sym})
    return {"pairs": pairs}


def default_starts_for_p(p: int) -> int:
    if p == 1:
        return 100_000
    if p == 2:
        return 50_000
    if p == 3:
        return 10_000
    raise ValueError("Unsupported p")


def main() -> None:
    parser = argparse.ArgumentParser(description="Saddle dominance multi-start diagnostics.")
    parser.add_argument("--mode", choices=["certified", "fine_grid", "single"], default="certified")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "multivariable" / "results" / "saddle_dominance_v2.json")
    parser.add_argument("--beta", type=float, default=-float(np.pi / 2))
    parser.add_argument("--gamma", type=float, default=0.14, help="Used when --mode=single")
    parser.add_argument("--p", type=int, default=2, help="Used when --mode=single")
    parser.add_argument("--r", type=float, default=176.54)
    parser.add_argument("--q", type=int, default=3)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--merge_tol", type=float, default=1e-4)
    parser.add_argument("--override_starts", type=int, default=None, help="If set, use same n_starts for every run.")
    args = parser.parse_args()

    outputs = {"mode": args.mode, "runs": []}
    if args.mode == "single":
        n = args.override_starts if args.override_starts is not None else default_starts_for_p(args.p)
        rep = run_multistart_report(args.p, args.beta, args.gamma, args.r, args.q, n, args.merge_tol, args.seed)
        rep["pair_diagnostics"] = diagnose_root_pairs(rep)
        outputs["runs"].append(rep)
    elif args.mode == "certified":
        for p in (1, 2, 3):
            n = args.override_starts if args.override_starts is not None else default_starts_for_p(p)
            rep = run_multistart_report(p, args.beta, 0.14, args.r, args.q, n, args.merge_tol, args.seed)
            rep["pair_diagnostics"] = diagnose_root_pairs(rep)
            if rep["n_distinct_roots"] == 1:
                rep["conclusion"] = "UNIQUE ROOT at certified parameters. Saddle dominance holds."
            else:
                rep["conclusion"] = "MULTIPLE ROOTS found; compare Re[Phi] and symmetry diagnostics."
            outputs["runs"].append(rep)

        # Required p=2 large-gamma diagnostics.
        for g in (11 * np.pi / 8, 3 * np.pi / 2, 2 * np.pi):
            n = args.override_starts if args.override_starts is not None else 50_000
            rep = run_multistart_report(2, args.beta, float(g), args.r, args.q, n, args.merge_tol, args.seed)
            rep["pair_diagnostics"] = diagnose_root_pairs(rep)
            outputs["runs"].append(rep)
    else:
        gammas = [0.02, 0.05, 0.08, 0.10, 0.12, 0.14, 0.16, 0.20, 0.30, 0.50]
        for p in (1, 2, 3):
            for g in gammas:
                n = args.override_starts if args.override_starts is not None else 50_000
                rep = run_multistart_report(p, args.beta, g, args.r, args.q, n, args.merge_tol, args.seed)
                rep["pair_diagnostics"] = diagnose_root_pairs(rep)
                outputs["runs"].append(rep)
                print(
                    f"p={p}, gamma={g:.2f}: {rep['n_distinct_roots']} root(s), "
                    f"{rep['n_converged']}/{rep['n_starts']} converged",
                    flush=True,
                )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(outputs, f, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
