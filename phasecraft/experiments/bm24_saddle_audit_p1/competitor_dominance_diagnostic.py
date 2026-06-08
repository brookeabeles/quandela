"""
Focused discovered-competitor diagnostic at selected gamma values.

Stronger root search than continue_seed_branch_certified defaults; reports
signed Re Phi gaps vs an interpolated seed branch. Not a completeness proof.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.optimize import root

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from phasecraft.bm24_saddle_audit_p1.audit import AUDIT_DIR, _x_to_z
from phasecraft.bm24_saddle_audit_p1.continue_seed_branch_certified import (
    IM_PHI_WARNING,
    certify_z,
    conv2_full_exponent,
    polish_to_residual,
    step_from_previous_z,
)
from phasecraft.lib.saddles.krawczyk_p1_roots import SaddleSystem
from phasecraft.lib.saddles.picard_lefschetz import compute_phi

DEFAULT_GAMMAS = [-0.3, -0.6, -0.83, -1.0, -1.6, -2.0]
DEFAULT_CONTINUATION = (
    AUDIT_DIR / "results" / "run_seed_branch_g-2pi" / "seed_branch_continuation.json"
)
DEFAULT_PRIOR_COMPETITORS = (
    AUDIT_DIR / "results" / "run_seed_branch_g-2pi" / "competitor_saddles_by_gamma.json"
)
DISCLAIMER = (
    "Discovered competitor diagnostic only: reports certified saddles found by the "
    "stated random/seed-biased search. Not a global dominance or completeness proof."
)


def _load_continuation(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data.get("continuation", data)
    return [r for r in rows if r.get("certified") and not r.get("failed")]


def _z_from_row(row: dict) -> np.ndarray:
    return np.asarray(row["z_real"], dtype=float) + 1j * np.asarray(row["z_imag"], dtype=float)


def interpolate_seed_z(rows: list[dict], gamma: float) -> np.ndarray:
    """Linear interpolation of certified continuation z along gamma."""
    cert = sorted(rows, key=lambda r: float(r["gamma"]))
    if not cert:
        raise ValueError("no certified continuation rows")
    gs = [float(r["gamma"]) for r in cert]
    g = float(gamma)
    if g <= min(gs):
        return _z_from_row(cert[0 if gs[0] <= gs[-1] else -1])
    if g >= max(gs):
        return _z_from_row(cert[-1 if gs[0] <= gs[-1] else 0])
    for i in range(len(cert) - 1):
        g0, g1 = gs[i], gs[i + 1]
        if (g0 <= g <= g1) or (g1 <= g <= g0):
            t = (g - g0) / (g1 - g0) if abs(g1 - g0) > 1e-15 else 0.0
            z0, z1 = _z_from_row(cert[i]), _z_from_row(cert[i + 1])
            return (1.0 - t) * z0 + t * z1
    ref = min(cert, key=lambda r: abs(float(r["gamma"]) - g))
    return _z_from_row(ref)


def _nearest_prior_gamma_keys(by_gamma: dict, gamma: float, k: int = 2) -> list[str]:
    keys = sorted((float(kk) for kk in by_gamma.keys()), key=lambda x: abs(x - gamma))
    return [str(keys[i]) for i in range(min(k, len(keys)))]


def _prior_root_z_list(prior_path: Path, gamma: float) -> list[np.ndarray]:
    if not prior_path.is_file():
        return []
    data = json.loads(prior_path.read_text(encoding="utf-8"))
    by_gamma = data.get("by_gamma", {})
    out: list[np.ndarray] = []
    for key in _nearest_prior_gamma_keys(by_gamma, gamma):
        for entry in by_gamma.get(key, []):
            if not entry.get("certified"):
                continue
            if "z_real" not in entry or "z_imag" not in entry:
                continue
            z = np.asarray(entry["z_real"], dtype=float) + 1j * np.asarray(
                entry["z_imag"], dtype=float
            )
            out.append(z)
    return out


def _seed_biased_x0(
    seed_z: np.ndarray, nvars: int, *, n: int, rng: np.random.Generator, scales: tuple[float, ...]
) -> list[np.ndarray]:
    x_seed = np.concatenate([seed_z.real, seed_z.imag])
    starts = [x_seed.copy()]
    for scale in scales:
        for _ in range(max(1, n // len(scales))):
            noise = rng.normal(0.0, scale, size=2 * nvars)
            starts.append(x_seed + noise)
    return starts


def discover_roots_from_starts(
    sys: SaddleSystem,
    x0_list: list[np.ndarray],
    *,
    tol: float,
    dedup_tol: float,
    res_tol: float = 1e-7,
) -> list[np.ndarray]:
    roots: list[np.ndarray] = []
    for x0 in x0_list:
        sol = root(sys.G_real, x0, method="hybr", tol=tol)
        if not sol.success:
            continue
        x = sol.x
        res = float(np.linalg.norm(sys.G_real(x), ord=np.inf))
        if not np.isfinite(res) or res > res_tol:
            continue
        if any(np.linalg.norm(x - y, ord=np.inf) < dedup_tol for y in roots):
            continue
        roots.append(x)
    return roots


def build_start_list(
    sys: SaddleSystem,
    seed_z: np.ndarray,
    *,
    num_random: int,
    seed: int,
    prior_z: list[np.ndarray],
    near_seed_count: int,
    near_prior_count: int,
    near_prior_scale: float,
) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    nvars = sys.nvars
    starts: list[np.ndarray] = []

    for _ in range(num_random):
        starts.append(rng.normal(0.0, 1.0, size=2 * nvars))

    starts.extend(
        _seed_biased_x0(
            seed_z,
            nvars,
            n=near_seed_count,
            rng=rng,
            scales=(0.002, 0.01, 0.05, 0.15),
        )
    )

    x_seed = np.concatenate([seed_z.real, seed_z.imag])
    starts.append(x_seed)

    for z in prior_z:
        x = np.concatenate([z.real, z.imag])
        starts.append(x)
        for _ in range(near_prior_count):
            noise = rng.normal(0.0, near_prior_scale, size=2 * nvars)
            starts.append(x + noise)

    return starts


def certify_seed_at_gamma(
    q: int,
    K_clause: int,
    r: float,
    beta: float,
    gamma: float,
    z_init: np.ndarray,
    *,
    min_residual: float,
    dps: int,
) -> tuple[np.ndarray, dict]:
    z_cert, row = step_from_previous_z(
        q,
        K_clause,
        r,
        beta,
        gamma,
        z_init,
        step_index=0,
        n_values=list(range(12, 23)),
        match_tol=0.02,
        min_residual=min_residual,
        dps=dps,
    )
    if z_cert is None or not row.certified:
        raise RuntimeError(
            f"seed branch not certified at gamma={gamma}: {row.failure_reason}"
        )
    return z_cert, {
        "re_phi_m": row.re_phi_m,
        "full_conv2_exponent": row.full_conv2_exponent,
        "residual_inf": row.residual_inf,
        "source": "continuation_interpolate",
    }


def run_gamma_diagnostic(
    *,
    q: int,
    K_clause: int,
    r: float,
    beta: float,
    gamma: float,
    seed_z_init: np.ndarray,
    prior_z: list[np.ndarray],
    num_random_starts: int,
    seed: int,
    seed_branch_tol: float,
    dedup_tol: float,
    min_residual: float,
    dps: int,
) -> dict:
    seed_z, seed_info = certify_seed_at_gamma(
        q, K_clause, r, beta, gamma, seed_z_init, min_residual=min_residual, dps=dps
    )
    seed_re_phi = float(seed_info["re_phi_m"])
    seed_full = float(seed_info["full_conv2_exponent"])

    sys = SaddleSystem.build(q=q, r=r, betas=np.array([beta]), gammas=np.array([gamma]))
    x_starts = build_start_list(
        sys,
        seed_z,
        num_random=num_random_starts,
        seed=seed,
        prior_z=prior_z,
        near_seed_count=400,
        near_prior_count=8,
        near_prior_scale=0.08,
    )
    roots_x = discover_roots_from_starts(
        sys, x_starts, tol=1e-12, dedup_tol=dedup_tol, res_tol=1e-7
    )

    entries: list[dict] = []
    for x in roots_x:
        z = _x_to_z(x, sys.nvars)
        z_dist = float(np.linalg.norm(z - seed_z, ord=np.inf))
        is_seed = z_dist < seed_branch_tol
        x_pol, res = polish_to_residual(sys, x, min_residual=1e-8, dps=dps)
        z_pol = _x_to_z(x_pol, sys.nvars)
        ok, proof = certify_z(sys, z_pol, dps=dps)
        phi = compute_phi(z_pol, q=q, r=r, betas=sys.betas, gammas=sys.gammas)
        re_phi = float(phi.real)
        full = conv2_full_exponent(re_phi, K_clause, r)
        signed_gap = re_phi - seed_re_phi
        entries.append(
            {
                "gamma": float(gamma),
                "certified": bool(ok),
                "residual_inf": float(res),
                "re_phi_m": re_phi,
                "im_phi_m": float(phi.imag),
                "full_conv2_exponent": full,
                "signed_re_phi_gap_vs_seed": signed_gap,
                "z_distance_from_seed": z_dist,
                "is_seed_branch": is_seed,
                "krawczyk_contraction": float(proof.get("contraction_bound", math.inf)),
                "z_real": z_pol.real.tolist(),
                "z_imag": z_pol.imag.tolist(),
            }
        )

    certified = [e for e in entries if e["certified"]]
    competitors = [e for e in certified if not e["is_seed_branch"]]
    gaps = [e["signed_re_phi_gap_vs_seed"] for e in competitors]
    if gaps:
        max_gap = max(gaps)
        nearest_gap = min(gaps, key=lambda g: abs(g))
        max_comp_re = max(e["re_phi_m"] for e in competitors)
    else:
        max_gap = float("-inf")
        nearest_gap = float("nan")
        max_comp_re = float("nan")

    return {
        "gamma": float(gamma),
        "seed_re_phi": seed_re_phi,
        "seed_full_conv2_exponent": seed_full,
        "seed_source": seed_info["source"],
        "num_roots_found": len(entries),
        "num_certified": len(certified),
        "num_certified_competitors": len(competitors),
        "max_competitor_re_phi": max_comp_re,
        "max_signed_gap_vs_seed": max_gap,
        "seed_has_largest_discovered_re_phi": bool(not gaps or max_gap <= 0.0),
        "nearest_signed_gap_vs_seed": nearest_gap,
        "competitors": competitors,
    }


def run_diagnostic(
    *,
    gammas: list[float],
    continuation_json: Path,
    prior_competitors_json: Path,
    out_dir: Optional[Path],
    q: int = 3,
    K_clause: int = 8,
    r: float = 176.54,
    beta: float = 0.5433996420760803,
    num_random_starts: int = 3000,
    seed: int = 0,
    seed_branch_tol: float = 1e-6,
    dedup_tol: float = 1e-5,
    min_residual: float = 1e-10,
    dps: int = 80,
) -> Path:
    ts = datetime.now(timezone.utc).strftime("run_%m-%d_%H-%M-%SZ")
    if out_dir is not None:
        run_dir = out_dir
    elif continuation_json.parent.name == "run_seed_branch_g-2pi":
        run_dir = continuation_json.parent / "competitor_dominance"
    else:
        run_dir = AUDIT_DIR / "results" / f"{ts}_competitor_dominance"
    run_dir.mkdir(parents=True, exist_ok=True)

    cont_rows = _load_continuation(continuation_json)
    per_gamma: list[dict] = []
    for i, g in enumerate(gammas):
        print(f"gamma={g} ...", flush=True)
        z_init = interpolate_seed_z(cont_rows, g)
        prior_z = _prior_root_z_list(prior_competitors_json, g)
        row = run_gamma_diagnostic(
            q=q,
            K_clause=K_clause,
            r=r,
            beta=beta,
            gamma=g,
            seed_z_init=z_init,
            prior_z=prior_z,
            num_random_starts=num_random_starts,
            seed=seed + i * 10007,
            seed_branch_tol=seed_branch_tol,
            dedup_tol=dedup_tol,
            min_residual=min_residual,
            dps=dps,
        )
        print(
            f"  certified={row['num_certified']} competitors={row['num_certified_competitors']} "
            f"max_signed_gap={row['max_signed_gap_vs_seed']:.4g} "
            f"seed_largest={row['seed_has_largest_discovered_re_phi']}",
            flush=True,
        )
        per_gamma.append(row)

    summary = {
        "metadata": {
            "q": q,
            "p": 1,
            "K_clause": K_clause,
            "r": r,
            "beta": beta,
            "gammas": gammas,
            "num_random_starts": num_random_starts,
            "seed": seed,
            "seed_branch_tol": seed_branch_tol,
            "dedup_tol": dedup_tol,
            "continuation_json": str(continuation_json.resolve()),
            "prior_competitors_json": str(prior_competitors_json.resolve()),
            "convention": "signed_gap = Re Phi_competitor - Re Phi_seed (algebraic, discovered roots only)",
            "disclaimer": DISCLAIMER,
            "im_phi_warning": IM_PHI_WARNING,
            "timestamp": ts,
        },
        "diagnostic_type": "discovered_competitor_diagnostic",
        "per_gamma": per_gamma,
    }
    out_json = run_dir / "competitor_dominance_summary.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {out_json}")
    return run_dir


def main() -> None:
    p = argparse.ArgumentParser(
        description="Discovered competitor diagnostic (not a global dominance proof)"
    )
    p.add_argument("--gammas", type=float, nargs="*", default=DEFAULT_GAMMAS)
    p.add_argument("--continuation-json", type=str, default=str(DEFAULT_CONTINUATION))
    p.add_argument("--prior-competitors-json", type=str, default=str(DEFAULT_PRIOR_COMPETITORS))
    p.add_argument("--out-dir", type=str, default="")
    p.add_argument("--num-random-starts", type=int, default=3000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--seed-branch-tol", type=float, default=1e-6)
    p.add_argument("--dedup-tol", type=float, default=1e-5)
    p.add_argument("--dps", type=int, default=80)
    p.add_argument(
        "--quick",
        action="store_true",
        help="Smoke: 400 starts, gammas -0.3 and -1.0 only",
    )
    args = p.parse_args()
    gammas = list(args.gammas)
    num_starts = args.num_random_starts
    if args.quick:
        gammas = [-0.3, -1.0]
        num_starts = 400
    run_diagnostic(
        gammas=gammas,
        continuation_json=Path(args.continuation_json),
        prior_competitors_json=Path(args.prior_competitors_json),
        out_dir=Path(args.out_dir) if args.out_dir else None,
        num_random_starts=num_starts,
        seed=args.seed,
        seed_branch_tol=args.seed_branch_tol,
        dedup_tol=args.dedup_tol,
        dps=args.dps,
    )


if __name__ == "__main__":
    main()
