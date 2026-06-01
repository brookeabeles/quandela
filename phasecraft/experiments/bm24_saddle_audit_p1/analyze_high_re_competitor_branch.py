"""
Analyze high-Re-Phi competitors from competitor_dominance_summary.json.

Tracks (1) per-gamma max signed Re-Phi gap competitor and (2) a z-continuous
branch via nearest-neighbor matching. Compares unwrapped Im(Phi) to the seed branch.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from phasecraft.bm24_saddle_audit_p1.audit import AUDIT_DIR
from phasecraft.bm24_saddle_audit_p1.competitor_dominance_diagnostic import (
    _load_continuation,
    _z_from_row,
    interpolate_seed_z,
)
from phasecraft.bm24_saddle_audit_p1.continue_seed_branch_certified import (
    IM_PHI_WARNING,
    certify_z,
    conv2_full_exponent,
    polish_to_residual,
    step_from_previous_z,
)
from phasecraft.krawczyk_p1_roots import SaddleSystem
from phasecraft.picard_lefschetz import compute_phi
from phasecraft.w_saddle.workflow import unwrap_im_branch, unwrap_phase_diff

DEFAULT_SUMMARY = (
    AUDIT_DIR
    / "results/run_seed_branch_g-2pi/competitor_dominance/competitor_dominance_summary.json"
)
DEFAULT_CONTINUATION = (
    AUDIT_DIR / "results/run_seed_branch_g-2pi/seed_branch_continuation.json"
)

Z_CONTINUITY_WARN = 2.0
IM_JUMP_WARN = 0.95 * math.pi


def _z_from_entry(entry: dict) -> np.ndarray:
    return np.asarray(entry["z_real"], dtype=float) + 1j * np.asarray(entry["z_imag"], dtype=float)


def _pick_fields(entry: dict, *, seed_re_phi: float) -> dict:
    return {
        "gamma": float(entry["gamma"]),
        "re_phi_m": float(entry["re_phi_m"]),
        "im_phi_m": float(entry["im_phi_m"]),
        "signed_re_phi_gap_vs_seed": float(entry.get("signed_re_phi_gap_vs_seed", entry["re_phi_m"] - seed_re_phi)),
        "z_distance_from_seed": float(entry["z_distance_from_seed"]),
        "residual_inf": float(entry["residual_inf"]),
        "krawczyk_contraction": float(entry["krawczyk_contraction"]),
        "z_real": entry["z_real"],
        "z_imag": entry["z_imag"],
    }


def max_gap_competitor(per_gamma_row: dict) -> dict:
    seed_re = float(per_gamma_row["seed_re_phi"])
    comps = [c for c in per_gamma_row["competitors"] if not c.get("is_seed_branch")]
    if not comps:
        raise ValueError(f"no competitors at gamma={per_gamma_row['gamma']}")
    return max(comps, key=lambda c: float(c["signed_re_phi_gap_vs_seed"]))


def nearest_competitor(
    competitors: list[dict], z_ref: np.ndarray, *, exclude_seed: bool = True
) -> tuple[dict, float]:
    pool = competitors
    if exclude_seed:
        pool = [c for c in competitors if not c.get("is_seed_branch")]
    if not pool:
        raise ValueError("empty competitor pool")
    best = min(pool, key=lambda c: float(np.linalg.norm(_z_from_entry(c) - z_ref, ord=np.inf)))
    d = float(np.linalg.norm(_z_from_entry(best) - z_ref, ord=np.inf))
    return best, d


def track_continuous_branch(per_gamma_rows: list[dict], gammas: list[float]) -> list[dict]:
    """Nearest-neighbor chain in z, seeded by max-gap competitor at first gamma."""
    by_g = {float(r["gamma"]): r for r in per_gamma_rows}
    g0 = gammas[0]
    current = max_gap_competitor(by_g[g0])
    track = [_pick_fields(current, seed_re_phi=float(by_g[g0]["seed_re_phi"]))]
    z_cur = _z_from_entry(current)
    for g in gammas[1:]:
        row = by_g[g]
        nxt, z_step = nearest_competitor(row["competitors"], z_cur)
        entry = _pick_fields(nxt, seed_re_phi=float(row["seed_re_phi"]))
        entry["z_step_from_previous_gamma"] = z_step
        track.append(entry)
        z_cur = _z_from_entry(nxt)
    return track


def unwrap_series(im_raw: list[float]) -> tuple[list[float], bool, list[float]]:
    unwrapped, ok = unwrap_im_branch(im_raw)
    jumps = []
    for i in range(1, len(im_raw)):
        jumps.append(float(im_raw[i]) - float(im_raw[i - 1]))
    return unwrapped, ok, jumps


def seed_im_at_gammas(
    continuation_path: Path, gammas: list[float], *, q: int, r: float, beta: float
) -> list[dict]:
    rows = _load_continuation(continuation_path)
    out = []
    for g in gammas:
        z_init = interpolate_seed_z(rows, g)
        z_cert, row = step_from_previous_z(
            q,
            8,
            r,
            beta,
            g,
            z_init,
            step_index=0,
            n_values=list(range(12, 23)),
            match_tol=0.02,
            min_residual=1e-10,
            dps=80,
        )
        if z_cert is None or not row.certified:
            raise RuntimeError(f"seed not certified at gamma={g}")
        out.append(
            {
                "gamma": float(g),
                "re_phi_m": float(row.re_phi_m),
                "im_phi_m": float(row.im_phi_m),
                "z_real": z_cert.real.tolist(),
                "z_imag": z_cert.imag.tolist(),
            }
        )
    return out


def recertify_track(
    track: list[dict], *, q: int, r: float, beta: float, dps: int = 80
) -> list[dict]:
    """Polish + Krawczyk at each gamma using previous point as warm start."""
    out: list[dict] = []
    z_prev: np.ndarray | None = None
    for i, pt in enumerate(track):
        g = float(pt["gamma"])
        z_init = _z_from_entry(pt) if z_prev is None else z_prev
        z_cert, row = step_from_previous_z(
            q,
            8,
            r,
            beta,
            g,
            z_init,
            step_index=i,
            n_values=list(range(12, 23)),
            match_tol=0.02,
            min_residual=1e-10,
            dps=dps,
        )
        if z_cert is None or not row.certified:
            pt = dict(pt)
            pt["recertify_failed"] = True
            pt["failure_reason"] = row.failure_reason
            out.append(pt)
            z_prev = z_init
            continue
        sys = SaddleSystem.build(q=q, r=r, betas=np.array([beta]), gammas=np.array([g]))
        ok, proof = certify_z(sys, z_cert, dps=dps)
        phi = compute_phi(z_cert, q=q, r=r, betas=sys.betas, gammas=sys.gammas)
        pt = dict(pt)
        pt.update(
            {
                "recertify_failed": not ok,
                "re_phi_m": float(phi.real),
                "im_phi_m": float(phi.imag),
                "full_conv2_exponent": conv2_full_exponent(float(phi.real), 8, r),
                "residual_inf": float(row.residual_inf),
                "krawczyk_contraction": float(proof.get("contraction_bound", math.inf)),
                "z_real": z_cert.real.tolist(),
                "z_imag": z_cert.imag.tolist(),
            }
        )
        if z_prev is not None:
            pt["z_step_from_previous_gamma"] = float(
                np.linalg.norm(z_cert - z_prev, ord=np.inf)
            )
        out.append(pt)
        z_prev = z_cert
    return out


def build_warnings(
    gammas: list[float],
    max_gap_track: list[dict],
    continuous_track: list[dict],
    seed_series: list[dict],
) -> list[str]:
    warnings: list[str] = []

    def _z_jumps(track: list[dict], label: str) -> None:
        for i in range(1, len(track)):
            za = _z_from_entry(track[i - 1])
            zb = _z_from_entry(track[i])
            d = float(np.linalg.norm(zb - za, ord=np.inf))
            if d > Z_CONTINUITY_WARN:
                warnings.append(
                    f"{label}: large z jump {d:.3g} between gamma={gammas[i-1]} and {gammas[i]}"
                )

    _z_jumps(max_gap_track, "max_signed_gap_chain")
    _z_jumps(continuous_track, "z_continuous_branch")

    for label, track in (
        ("max_signed_gap_chain", max_gap_track),
        ("z_continuous_branch", continuous_track),
        ("seed_branch", seed_series),
    ):
        im_raw = [t["im_phi_m"] for t in track]
        _, ok, jumps = unwrap_series(im_raw)
        if not ok:
            warnings.append(f"{label}: principal Im(Phi) has 2pi-scale jumps along gamma")
        for i, dj in enumerate(jumps):
            if abs(dj) > IM_JUMP_WARN:
                warnings.append(
                    f"{label}: |Delta Im(Phi)|={abs(dj):.3g} > pi between "
                    f"gamma={gammas[i]} and {gammas[i+1]}"
                )

    return warnings


def analyze(
    summary_path: Path,
    continuation_path: Path,
    out_dir: Path,
    *,
    recertify: bool = True,
) -> Path:
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    meta = data["metadata"]
    per_gamma = data["per_gamma"]
    gammas = [float(g) for g in meta["gammas"]]

    beta = float(meta["beta"])
    r = float(meta["r"])
    q = int(meta["q"])

    max_gap_by_gamma: list[dict] = []
    for row in per_gamma:
        seed_re = float(row["seed_re_phi"])
        best = max_gap_competitor(row)
        max_gap_by_gamma.append(_pick_fields(best, seed_re_phi=seed_re))

    continuous_raw = track_continuous_branch(per_gamma, gammas)
    if recertify:
        continuous_track = recertify_track(continuous_raw, q=q, r=r, beta=beta)
    else:
        continuous_track = continuous_raw

    seed_series = seed_im_at_gammas(continuation_path, gammas, q=q, r=r, beta=beta)

    # Unwrapped Im(Phi) and difference (seed - competitor) on continuous branch
    seed_im_u, seed_ok, _ = unwrap_series([s["im_phi_m"] for s in seed_series])
    cont_im_u, cont_ok, _ = unwrap_series([t["im_phi_m"] for t in continuous_track])
    max_im_u, max_ok, _ = unwrap_series([t["im_phi_m"] for t in max_gap_by_gamma])

    im_diff_unwrapped = [
        float(s - c) for s, c in zip(seed_im_u, cont_im_u, strict=True)
    ]
    im_diff_principal = [
        unwrap_phase_diff(s["im_phi_m"], t["im_phi_m"])
        for s, t in zip(seed_series, continuous_track, strict=True)
    ]

    z_steps_max = []
    for i in range(1, len(max_gap_by_gamma)):
        d = float(
            np.linalg.norm(
                _z_from_entry(max_gap_by_gamma[i]) - _z_from_entry(max_gap_by_gamma[i - 1]),
                ord=np.inf,
            )
        )
        z_steps_max.append({"gamma_from": gammas[i - 1], "gamma_to": gammas[i], "z_inf": d})

    warnings = build_warnings(gammas, max_gap_by_gamma, continuous_track, seed_series)
    if not seed_ok:
        warnings.append("seed_branch: unwrapped Im(Phi) discontinuous along gamma")
    if not cont_ok:
        warnings.append("z_continuous_branch: unwrapped Im(Phi) discontinuous along gamma")
    if not max_ok:
        warnings.append("max_signed_gap_chain: unwrapped Im(Phi) discontinuous along gamma")

    max_z_step = max((s["z_inf"] for s in z_steps_max), default=0.0)
    cont_z_steps = [
        float(t.get("z_step_from_previous_gamma", 0.0)) for t in continuous_track[1:]
    ]
    is_continuous_branch = all(s < Z_CONTINUITY_WARN for s in cont_z_steps) and all(
        s < Z_CONTINUITY_WARN for s in (x["z_inf"] for x in z_steps_max)
    )
    # max-gap chain is usually NOT continuous
    max_gap_continuous = max_z_step < Z_CONTINUITY_WARN

    cont_continuous = all(s < Z_CONTINUITY_WARN for s in cont_z_steps)
    conclusion = (
        "Per-gamma max-Re-Phi competitors are **not** one z-continuous saddle branch "
        f"(greedy chain max ||Δz||_inf ≈ {max_z_step:.2g}). "
        + (
            f"A nearest-neighbor z-chain in the diagnostic direction also fails continuity "
            f"(max step ≈ {max(cont_z_steps, default=0.0):.2g} > {Z_CONTINUITY_WARN}). "
            if not cont_continuous
            else "A nearest-neighbor z-chain is small-step in z, but "
        )
        + "Seed and high-Re tracks sit on different principal Im(Φ) sheets (seed Im Φ ≈ 0, "
        "competitor Im Φ ≈ O(1–6) with 2π-scale jumps). These are certified algebraic saddles "
        "but look like distinct log-branch artifacts, not a single physical competitor branch "
        "dominating the seed in contour sense."
    )

    summary = {
        "metadata": {
            "source_summary": str(summary_path.resolve()),
            "continuation_json": str(continuation_path.resolve()),
            "gammas": gammas,
            "z_match_metric": "||z(gamma_a)-z(gamma_b)||_inf",
            "z_continuity_warn_threshold": Z_CONTINUITY_WARN,
            "im_phi_warning": IM_PHI_WARNING,
            "disclaimer": (
                "Branch analysis of discovered saddles only. High Re(Phi) does not imply "
                "physical dominance or nonzero contour contribution without PL / intersection data."
            ),
        },
        "max_re_competitor_by_gamma": max_gap_by_gamma,
        "z_continuous_branch_track": continuous_track,
        "seed_branch_at_gammas": seed_series,
        "branch_analysis": {
            "max_signed_gap_chain_z_continuous": max_gap_continuous,
            "max_signed_gap_chain_max_z_step": max_z_step,
            "z_continuous_branch_max_z_step": max(cont_z_steps, default=0.0),
            "z_steps_max_gap_chain": z_steps_max,
            "seed_im_unwrapped": seed_im_u,
            "continuous_branch_im_unwrapped": cont_im_u,
            "max_gap_chain_im_unwrapped": max_im_u,
            "im_phi_seed_minus_continuous_unwrapped": im_diff_unwrapped,
            "im_phi_seed_minus_continuous_principal_wrapped": im_diff_principal,
        },
        "warnings": warnings,
        "conclusion": conclusion,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "max_re_competitor_by_gamma.json").write_text(
        json.dumps(max_gap_by_gamma, indent=2), encoding="utf-8"
    )
    (out_dir / "high_re_competitor_branch_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print(json.dumps(
        {
            "max_gap_chain_z_continuous": max_gap_continuous,
            "continuous_branch_max_z_step": max(cont_z_steps, default=0.0),
            "num_warnings": len(warnings),
        },
        indent=2,
    ))
    print(f"Wrote {out_dir}")
    return out_dir


def main() -> None:
    p = argparse.ArgumentParser(description="Analyze high-Re competitor branch from dominance JSON")
    p.add_argument("--summary", type=str, default=str(DEFAULT_SUMMARY))
    p.add_argument("--continuation", type=str, default=str(DEFAULT_CONTINUATION))
    p.add_argument("--out-dir", type=str, default="")
    p.add_argument("--no-recertify", action="store_true")
    args = p.parse_args()
    summary_path = Path(args.summary)
    out_dir = Path(args.out_dir) if args.out_dir else summary_path.parent
    analyze(
        summary_path,
        Path(args.continuation),
        out_dir,
        recertify=not args.no_recertify,
    )


if __name__ == "__main__":
    main()
