"""Numerical Picard-Lefschetz feasibility probe for BM24 p=1.

This is a prototype, not a rigorous PL implementation.  It answers a narrower
question:

Can we build a local holomorphic action sheet around the BM24 saddles, verify
that the seed/competitor are Morse saddles on that sheet, and numerically trace
short upward flows with Im(S) approximately conserved?

What it does NOT yet compute:

* exact intersection numbers with the original BM24 integration cycle
* globally valid thimble contours across fractional-power branch cuts
* a proof that a high-Re algebraic saddle has zero physical intersection

Those require the original integration cycle represented on the same
multi-valued z-sheet used by the fractional-power saddle equations.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-phasecraft")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp


HERE = Path(__file__).resolve().parent
PACKAGE_ROOT = HERE.parents[2]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from phasecraft.bm24_saddle_audit_p1 import refined_transition_scan as refined
from phasecraft.bm24_saddle_audit_p1.audit import _x_to_z
from phasecraft.bm24_saddle_audit_p1.continue_seed_branch_certified import (
    certify_z,
    conv2_full_exponent,
    conv2_pref,
    lambda_abs_n_max,
    polish_to_residual,
)
from phasecraft.lib.saddles.krawczyk_p1_roots import (
    SaddleSystem,
    discover_roots,
    numerical_jacobian,
    parent_function_alpha_sum_sos,
    parent_function_s_sum_sos,
)
from phasecraft.lib.saddles.picard_lefschetz import compute_phi


Q = 3
K_CLAUSE = 8
R = 176.54
BETA = 0.5433996420760803
N_VALUES = list(range(18, 25))
DPS = 80
MIN_RESIDUAL = 1e-10

OUT = HERE / "results" / "picard_lefschetz_probe"


@dataclass
class LocalSheet:
    sheets: np.ndarray
    theta_ref: np.ndarray
    grad_inf_at_saddle: float
    action_at_saddle: complex
    phi_at_saddle: complex
    action_minus_phi: complex


def f_and_df(sys: SaddleSystem, z: np.ndarray) -> tuple[complex, np.ndarray, complex]:
    s_vec = np.exp(parent_function_alpha_sum_sos(0.5 * sys.c_root * z))
    log_arg = np.sum(sys.b * s_vec)
    F = np.log(log_arg)
    dF = sys.c_root * parent_function_s_sum_sos(0.5 * sys.b * s_vec) / log_arg
    return complex(F), dF, complex(log_arg)


def branch_log(z: np.ndarray, sheets: np.ndarray, theta_ref: np.ndarray) -> np.ndarray:
    """Local logarithm centered at the saddle, avoiding principal-cut jumps."""
    radius = np.abs(z)
    theta = np.angle(z)
    dtheta = (theta - theta_ref + math.pi) % (2.0 * math.pi) - math.pi
    theta_local = theta_ref + dtheta + 2.0 * math.pi * sheets
    return np.log(radius) + 1j * theta_local


def infer_local_sheet(sys: SaddleSystem, z: np.ndarray) -> LocalSheet:
    """Infer the local integer log sheets that make the action stationary."""
    k = 2 ** sys.q
    alpha = ((k - 1.0) / k) * k ** (-1.0 / (k - 1.0))
    m = k / (k - 1.0)
    F, dF, _ = f_and_df(sys, z)
    theta_ref = np.angle(z)

    sheets = np.zeros(sys.nvars, dtype=int)
    for i, zi in enumerate(z):
        target_root = (k ** (1.0 / (k - 1.0))) * (-dF[i])
        best_sheet = 0
        best_dist = math.inf
        for sheet in range(k - 1):
            cand = np.exp((np.log(abs(zi)) + 1j * (theta_ref[i] + 2.0 * math.pi * sheet)) / (k - 1.0))
            dist = abs(cand - target_root)
            if dist < best_dist:
                best_dist = float(dist)
                best_sheet = sheet
        sheets[i] = best_sheet

    S = local_action(sys, z, sheets, theta_ref)
    grad = local_grad(sys, z, sheets, theta_ref)
    phi = compute_phi(z, q=sys.q, r=sys.r, betas=sys.betas, gammas=sys.gammas)
    return LocalSheet(
        sheets=sheets,
        theta_ref=theta_ref,
        grad_inf_at_saddle=float(np.linalg.norm(grad, ord=np.inf)),
        action_at_saddle=S,
        phi_at_saddle=phi,
        action_minus_phi=S - phi,
    )


def local_action(sys: SaddleSystem, z: np.ndarray, sheets: np.ndarray, theta_ref: np.ndarray) -> complex:
    k = 2 ** sys.q
    alpha = ((k - 1.0) / k) * k ** (-1.0 / (k - 1.0))
    m = k / (k - 1.0)
    F, _, _ = f_and_df(sys, z)
    term = alpha * np.sum(np.exp(m * branch_log(z, sheets, theta_ref)))
    return complex(F + term)


def local_grad(sys: SaddleSystem, z: np.ndarray, sheets: np.ndarray, theta_ref: np.ndarray) -> np.ndarray:
    k = 2 ** sys.q
    alpha = ((k - 1.0) / k) * k ** (-1.0 / (k - 1.0))
    m = k / (k - 1.0)
    _, dF, _ = f_and_df(sys, z)
    return dF + alpha * m * np.exp((m - 1.0) * branch_log(z, sheets, theta_ref))


def pack(z: np.ndarray) -> np.ndarray:
    return np.concatenate([z.real, z.imag])


def unpack(x: np.ndarray, n: int) -> np.ndarray:
    return x[:n] + 1j * x[n:]


def upward_vector_field(sys: SaddleSystem, sheets: np.ndarray, theta_ref: np.ndarray):
    n = sys.nvars

    def vf(_t: float, x: np.ndarray) -> np.ndarray:
        z = unpack(x, n)
        try:
            g = local_grad(sys, z, sheets, theta_ref)
            if not np.all(np.isfinite(g)):
                return np.zeros_like(x)
            dz = np.conjugate(g)
            return np.concatenate([dz.real, dz.imag])
        except Exception:
            return np.zeros_like(x)

    return vf


def real_hessian_re_action(sys: SaddleSystem, z: np.ndarray, sheets: np.ndarray, theta_ref: np.ndarray) -> np.ndarray:
    x0 = pack(z)
    vf = upward_vector_field(sys, sheets, theta_ref)
    return numerical_jacobian(lambda x: vf(0.0, x), x0, eps=1e-6)


def discover_relevant_saddles(
    beta: float,
    gamma: float,
    starts: int,
) -> tuple[np.ndarray, list[dict[str, Any]], float]:
    z_seed, ok = refined.warm_seed_to(beta, gamma)
    if z_seed is None or not ok:
        raise RuntimeError(f"could not continue seed to beta={beta}, gamma={gamma}")

    sys = SaddleSystem.build(q=Q, r=R, betas=np.array([beta]), gammas=np.array([gamma]))
    lam = lambda_abs_n_max(Q, K_CLAUSE, R, beta, gamma, N_VALUES)
    seed_phi = compute_phi(z_seed, q=Q, r=R, betas=sys.betas, gammas=sys.gammas)
    seed_full = conv2_full_exponent(float(seed_phi.real), K_CLAUSE, R)
    saddles: list[dict[str, Any]] = [
        {
            "name": "seed",
            "z": z_seed,
            "re_phi": float(seed_phi.real),
            "im_phi": float(seed_phi.imag),
            "full_exponent": seed_full,
            "exact_gap_abs": abs(seed_full - lam),
            "certified": True,
            "residual_inf": float(np.linalg.norm(sys.G_complex(z_seed), ord=np.inf)),
        }
    ]

    roots = discover_roots(sys, num_starts=starts, seed=int(1000 * abs(gamma) + 1000 * beta), tol=1e-12)
    for idx, x in enumerate(roots):
        z = _x_to_z(x, sys.nvars)
        if np.linalg.norm(z - z_seed, ord=np.inf) < 1e-4:
            continue
        x_pol, res = polish_to_residual(sys, x, min_residual=MIN_RESIDUAL, dps=DPS)
        z_pol = _x_to_z(x_pol, sys.nvars)
        ok, _ = certify_z(sys, z_pol, dps=DPS)
        if not ok:
            continue
        phi = compute_phi(z_pol, q=Q, r=R, betas=sys.betas, gammas=sys.gammas)
        full = conv2_full_exponent(float(phi.real), K_CLAUSE, R)
        saddles.append(
            {
                "name": f"comp_{idx}",
                "z": z_pol,
                "re_phi": float(phi.real),
                "im_phi": float(phi.imag),
                "full_exponent": float(full),
                "exact_gap_abs": abs(full - lam),
                "certified": True,
                "residual_inf": float(res),
            }
        )

    return z_seed, saddles, lam


def select_probe_saddles(saddles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seed = saddles[0]
    competitors = saddles[1:]
    out = [seed]
    if competitors:
        best_exact = min(competitors, key=lambda s: s["exact_gap_abs"])
        best_exact = dict(best_exact)
        best_exact["name"] = "best_exact_competitor"
        out.append(best_exact)
        high_re = max(competitors, key=lambda s: s["full_exponent"])
        if np.linalg.norm(high_re["z"] - best_exact["z"], ord=np.inf) > 1e-8:
            high_re = dict(high_re)
            high_re["name"] = "highest_re_competitor"
            out.append(high_re)
    return out


def flow_probe_for_saddle(
    sys: SaddleSystem,
    saddle: dict[str, Any],
    t_max: float,
    eps: float,
    max_dirs: int,
) -> dict[str, Any]:
    z0 = saddle["z"]
    sheet = infer_local_sheet(sys, z0)
    H = real_hessian_re_action(sys, z0, sheet.sheets, sheet.theta_ref)
    eigvals, eigvecs = np.linalg.eig(H)
    eigvals = eigvals.real
    eigvecs = eigvecs.real
    pos = np.where(eigvals > 1e-7)[0]
    neg = np.where(eigvals < -1e-7)[0]
    order = pos[np.argsort(-eigvals[pos])]
    selected = order[:max_dirs]

    S0 = local_action(sys, z0, sheet.sheets, sheet.theta_ref)
    vf = upward_vector_field(sys, sheet.sheets, sheet.theta_ref)
    flows = []
    for eig_idx in selected:
        v = eigvecs[:, eig_idx]
        v = v / max(np.linalg.norm(v), 1e-300)
        for sign in [-1.0, 1.0]:
            x_start = pack(z0) + sign * eps * v
            sol = solve_ivp(
                vf,
                (0.0, t_max),
                x_start,
                rtol=1e-7,
                atol=1e-9,
                max_step=0.02,
            )
            z_path = np.array([unpack(sol.y[:, j], sys.nvars) for j in range(sol.y.shape[1])])
            S_path = np.array([local_action(sys, z, sheet.sheets, sheet.theta_ref) for z in z_path])
            _, _, log_arg_final = f_and_df(sys, z_path[-1])
            flows.append(
                {
                    "eig_index": int(eig_idx),
                    "eig_value": float(eigvals[eig_idx]),
                    "sign": float(sign),
                    "status": int(sol.status),
                    "n_steps": int(sol.t.size),
                    "t_final": float(sol.t[-1]),
                    "delta_re_S": float(S_path[-1].real - S0.real),
                    "max_abs_delta_im_S": float(np.max(np.abs(S_path.imag - S0.imag))),
                    "final_norm_z": float(np.linalg.norm(z_path[-1])),
                    "min_abs_z_coord": float(np.min(np.abs(z_path))),
                    "final_abs_log_arg": float(abs(log_arg_final)),
                }
            )

    return {
        "name": saddle["name"],
        "re_phi": saddle["re_phi"],
        "im_phi": saddle["im_phi"],
        "full_exponent": saddle["full_exponent"],
        "exact_gap_abs": saddle["exact_gap_abs"],
        "residual_inf": saddle["residual_inf"],
        "local_sheet": {
            "sheets": sheet.sheets.tolist(),
            "theta_ref": sheet.theta_ref.tolist(),
            "grad_inf_at_saddle": sheet.grad_inf_at_saddle,
            "action_at_saddle": [sheet.action_at_saddle.real, sheet.action_at_saddle.imag],
            "phi_at_saddle": [sheet.phi_at_saddle.real, sheet.phi_at_saddle.imag],
            "action_minus_phi": [sheet.action_minus_phi.real, sheet.action_minus_phi.imag],
        },
        "morse": {
            "positive_real_hessian_eigs": int(len(pos)),
            "negative_real_hessian_eigs": int(len(neg)),
            "near_zero_real_hessian_eigs": int(H.shape[0] - len(pos) - len(neg)),
            "top_positive_eigs": [float(eigvals[i]) for i in order[:8]],
            "most_negative_eigs": [float(x) for x in sorted(eigvals)[:8]],
            "condition_proxy": float(np.max(np.abs(eigvals)) / max(np.min(np.abs(eigvals)), 1e-300)),
        },
        "flows": flows,
    }


def plot_flow_summary(result: dict[str, Any]) -> str:
    rows = []
    labels = []
    for point in result["points"]:
        for s in point["saddles"]:
            vals = [f["max_abs_delta_im_S"] for f in s["flows"]]
            re_vals = [f["delta_re_S"] for f in s["flows"]]
            rows.append((np.median(vals) if vals else np.nan, np.median(re_vals) if re_vals else np.nan))
            labels.append(f"G={point['Gamma']:.3f}\n{s['name']}")

    fig, ax = plt.subplots(figsize=(max(8, 0.8 * len(rows)), 4.8))
    x = np.arange(len(rows))
    im_vals = [r[0] for r in rows]
    re_vals = [r[1] for r in rows]
    ax.bar(x - 0.18, im_vals, width=0.36, label="median |Delta Im S|")
    ax.bar(x + 0.18, re_vals, width=0.36, label="median Delta Re S")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_yscale("symlog", linthresh=1e-8)
    ax.set_title("Short upward-flow PL sanity checks")
    ax.set_ylabel("flow diagnostic")
    ax.legend()
    fig.tight_layout()
    path = OUT / "pl_upward_flow_sanity.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def write_report(result: dict[str, Any]) -> str:
    lines = [
        "# Picard-Lefschetz feasibility probe",
        "",
        "## Verdict",
        "",
        "A rigorous PL calculation is not yet possible from the current artifacts alone because the original BM24 integration cycle is not represented in the same multi-valued fractional-power z-sheet coordinates used by the saddle equations.  However, a local numerical PL probe is possible: the script constructs the local action sheet at selected saddles and traces short upward flows.",
        "",
        "## What was tested",
        "",
        f"- beta: {result['beta']}",
        f"- gammas: {result['gammas']}",
        f"- competitor starts per gamma: {result['starts']}",
        f"- flow t_max: {result['flow_t_max']}",
        f"- flow epsilon: {result['flow_eps']}",
        "",
        "## Local PL diagnostics",
        "",
        "| Gamma | saddle | gap to exact | grad_inf | Hessian + / - / 0 | median |Delta Im S| | median Delta Re S |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for point in result["points"]:
        for s in point["saddles"]:
            flows = s["flows"]
            med_im = float(np.median([f["max_abs_delta_im_S"] for f in flows])) if flows else float("nan")
            med_re = float(np.median([f["delta_re_S"] for f in flows])) if flows else float("nan")
            m = s["morse"]
            lines.append(
                f"| {point['Gamma']:.6f} | {s['name']} | {s['exact_gap_abs']:.6g} | "
                f"{s['local_sheet']['grad_inf_at_saddle']:.3e} | "
                f"{m['positive_real_hessian_eigs']} / {m['negative_real_hessian_eigs']} / {m['near_zero_real_hessian_eigs']} | "
                f"{med_im:.3e} | {med_re:.3e} |"
            )
    lines += [
        "",
        "## Interpretation",
        "",
        "- If `grad_inf` is tiny and the Hessian has 8 positive and 8 negative real-flow eigenvalues, the local sheet behaves like a valid holomorphic Morse saddle for p=1.",
        "- If upward flows increase `Re S` while conserving `Im S`, the local PL ODE is numerically coherent.",
        "- This still does not give intersection numbers.  The next missing object is the original BM24 integration cycle in z-sheet coordinates, plus a global branch-cut/singularity handling strategy.",
        "",
        "## Figure",
        "",
        f"- `{Path(result['figure']).name}`",
        "",
    ]
    path = OUT / "picard_lefschetz_probe_report.md"
    path.write_text("\n".join(lines))
    return str(path)


def run(args: argparse.Namespace) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {
        "metadata": {
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "caveat": "Local numerical PL probe only; no rigorous intersection numbers.",
        },
        "beta": args.beta,
        "gammas": args.gammas,
        "starts": args.starts,
        "flow_t_max": args.t_max,
        "flow_eps": args.eps,
        "points": [],
    }

    for gamma in args.gammas:
        print(f"=== beta={args.beta:.6f}, gamma={gamma:.6f} ===", flush=True)
        z_seed, saddles, lam = discover_relevant_saddles(args.beta, gamma, args.starts)
        sys = SaddleSystem.build(q=Q, r=R, betas=np.array([args.beta]), gammas=np.array([gamma]))
        selected = select_probe_saddles(saddles)
        point = {
            "gamma": gamma,
            "Gamma": -gamma,
            "lambda_abs": lam,
            "n_certified_saddles_found": len(saddles),
            "saddles": [],
        }
        for saddle in selected:
            print(f"  probing {saddle['name']} gap={saddle['exact_gap_abs']:.4g}", flush=True)
            point["saddles"].append(
                flow_probe_for_saddle(sys, saddle, args.t_max, args.eps, args.max_dirs)
            )
        result["points"].append(point)

    figure = plot_flow_summary(result)
    result["figure"] = figure
    report = write_report(result)
    result["report"] = report
    out_json = OUT / "picard_lefschetz_probe.json"
    with out_json.open("w") as f:
        json.dump(result, f, indent=2)
    print("\nWrote:")
    print(f"  {out_json}")
    print(f"  {report}")
    print(f"  {figure}")


def parse_gammas(value: str) -> list[float]:
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--beta", type=float, default=BETA)
    parser.add_argument("--gammas", type=parse_gammas, default=[-1.825, -1.85])
    parser.add_argument("--starts", type=int, default=300)
    parser.add_argument("--t-max", type=float, default=0.35)
    parser.add_argument("--eps", type=float, default=1e-5)
    parser.add_argument("--max-dirs", type=int, default=4)
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
