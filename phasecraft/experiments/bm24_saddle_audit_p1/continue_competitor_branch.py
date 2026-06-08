"""
Dense gamma continuation of a fixed high-Re competitor saddle (anchor gamma).

Tracks one z-branch by polish + Krawczyk from an anchor point in both gamma
directions; compares to the seed branch at each gamma. Not a completeness proof.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from phasecraft.bm24_saddle_audit_p1.competitor_dominance_diagnostic import (
    DEFAULT_CONTINUATION,
    interpolate_seed_z,
    _load_continuation,
)
from phasecraft.bm24_saddle_audit_p1.continue_seed_branch_certified import (
    IM_PHI_WARNING,
    certify_z,
    polish_to_residual,
    step_from_previous_z,
)
from phasecraft.bm24_saddle_audit_p1.audit import AUDIT_DIR, _x_to_z
from phasecraft.lib.saddles.krawczyk_p1_roots import SaddleSystem
from phasecraft.lib.saddles.picard_lefschetz import compute_phi
from phasecraft.w_saddle.workflow import unwrap_im_branch

DEFAULT_MAX_RE_JSON = (
    AUDIT_DIR
    / "results/run_seed_branch_g-2pi/competitor_dominance/max_re_competitor_by_gamma.json"
)
OUT_JSON_NAME = "competitor_branch_from_gamma_-0.83.json"

Z_JUMP_FLAG = 1.0
IM_JUMP_FLAG = math.pi


@dataclass
class BranchStep:
    gamma: float
    direction: str
    step_index: int
    certified: bool
    failed: bool
    failure_reason: str
    residual_inf: float
    krawczyk_contraction: float
    re_phi_m: float
    im_phi_m: float
    im_phi_m_unwrapped: float
    z_distance_from_previous: float
    z_distance_from_seed: float
    seed_re_phi_m: float
    seed_im_phi_m: float
    seed_im_phi_m_unwrapped: float
    signed_re_phi_gap_vs_seed: float
    z_jump_gt_1: bool
    im_phi_jump_gt_pi: bool
    z_real: list[float]
    z_imag: list[float]

    def to_dict(self) -> dict:
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, (np.bool_, bool)):
                d[k] = bool(v)
            elif isinstance(v, (np.floating, np.integer)):
                d[k] = float(v)
        return d


def _z_from_dict(d: dict) -> np.ndarray:
    return np.asarray(d["z_real"], dtype=float) + 1j * np.asarray(d["z_imag"], dtype=float)


def _load_anchor_z(max_re_path: Path, anchor_gamma: float) -> np.ndarray:
    rows = json.loads(max_re_path.read_text(encoding="utf-8"))
    for row in rows:
        if abs(float(row["gamma"]) - anchor_gamma) < 1e-9:
            return _z_from_dict(row)
    raise KeyError(f"anchor gamma={anchor_gamma} not in {max_re_path}")


def _seed_at_gamma(
    q: int,
    K_clause: int,
    r: float,
    beta: float,
    gamma: float,
    cont_rows: list[dict],
    *,
    dps: int,
) -> tuple[np.ndarray, float, float]:
    z_init = interpolate_seed_z(cont_rows, gamma)
    z_s, row = step_from_previous_z(
        q,
        K_clause,
        r,
        beta,
        gamma,
        z_init,
        step_index=0,
        n_values=list(range(12, 23)),
        match_tol=0.02,
        min_residual=1e-10,
        dps=dps,
    )
    if z_s is None or not row.certified:
        raise RuntimeError(f"seed not certified at gamma={gamma}: {row.failure_reason}")
    return z_s, float(row.re_phi_m), float(row.im_phi_m)


def _competitor_step(
    q: int,
    K_clause: int,
    r: float,
    beta: float,
    gamma: float,
    z_prev: np.ndarray,
    z_seed: np.ndarray,
    *,
    direction: str,
    step_index: int,
    dps: int,
    im_prev_raw: float,
) -> tuple[np.ndarray | None, BranchStep]:
    sys = SaddleSystem.build(q=q, r=r, betas=np.array([beta]), gammas=np.array([gamma]))
    x_prev = np.concatenate([z_prev.real, z_prev.imag])
    x_pol, res = polish_to_residual(sys, x_prev, min_residual=1e-10, dps=dps)
    z_pol = _x_to_z(x_pol, sys.nvars)
    z_dist = float(np.linalg.norm(z_pol - z_prev, ord=np.inf))
    z_dist_seed = float(np.linalg.norm(z_pol - z_seed, ord=np.inf))
    ok, proof = certify_z(sys, z_pol, dps=dps)
    phi = compute_phi(z_pol, q=q, r=r, betas=sys.betas, gammas=sys.gammas)
    im_raw = float(phi.imag)
    im_jump = abs(im_raw - im_prev_raw) > IM_JUMP_FLAG
    failed = (not ok) or res > 1e-10 or not np.isfinite(res)
    reason = "" if not failed else (str(proof.get("reason", "krawczyk_failed")) if not ok else "residual")
    row = BranchStep(
        gamma=float(gamma),
        direction=direction,
        step_index=step_index,
        certified=bool(ok) and not failed,
        failed=failed,
        failure_reason=reason,
        residual_inf=float(res),
        krawczyk_contraction=float(proof.get("contraction_bound", math.inf)),
        re_phi_m=float(phi.real),
        im_phi_m=im_raw,
        im_phi_m_unwrapped=im_raw,  # filled post-pass
        z_distance_from_previous=z_dist,
        z_distance_from_seed=z_dist_seed,
        seed_re_phi_m=0.0,
        seed_im_phi_m=0.0,
        seed_im_phi_m_unwrapped=0.0,
        signed_re_phi_gap_vs_seed=0.0,
        z_jump_gt_1=z_dist > Z_JUMP_FLAG,
        im_phi_jump_gt_pi=im_jump,
        z_real=z_pol.real.tolist(),
        z_imag=z_pol.imag.tolist(),
    )
    if failed:
        return None, row
    return z_pol, row


def _continue_direction(
    *,
    q: int,
    K_clause: int,
    r: float,
    beta: float,
    z_start: np.ndarray,
    gamma_start: float,
    gamma_end: float,
    gamma_step: float,
    min_step: float,
    direction: str,
    cont_rows: list[dict],
    dps: int,
    im_anchor: float,
) -> tuple[list[BranchStep], np.ndarray]:
    """direction 'toward_zero' (increasing gamma) or 'toward_neg' (decreasing)."""
    rows: list[BranchStep] = []
    z_cur = z_start
    gamma = float(gamma_start)
    g_end = float(gamma_end)
    step = float(gamma_step)
    step_index = 0
    im_prev = float(im_anchor)

    toward_zero = direction == "toward_zero"
    while True:
        if toward_zero:
            if gamma >= g_end - 1e-12:
                break
            target = min(g_end, gamma + step)
        else:
            if gamma <= g_end + 1e-12:
                break
            target = max(g_end, gamma - step)

        z_seed, seed_re, seed_im = _seed_at_gamma(
            q, K_clause, r, beta, target, cont_rows, dps=dps
        )
        z_next, row = _competitor_step(
            q,
            K_clause,
            r,
            beta,
            target,
            z_cur,
            z_seed,
            direction=direction,
            step_index=step_index,
            dps=dps,
            im_prev_raw=im_prev,
        )
        row.seed_re_phi_m = seed_re
        row.seed_im_phi_m = seed_im
        row.signed_re_phi_gap_vs_seed = float(row.re_phi_m - seed_re)

        if z_next is None:
            step *= 0.5
            if step < min_step:
                rows.append(row)
                break
            continue

        rows.append(row)
        z_cur = z_next
        gamma = target
        im_prev = row.im_phi_m
        step_index += 1
        step = min(float(gamma_step), step * 1.25)

    return rows, z_cur


def _anchor_step(
    q: int,
    K_clause: int,
    r: float,
    beta: float,
    z_anchor: np.ndarray,
    anchor_gamma: float,
    cont_rows: list[dict],
    dps: int,
) -> BranchStep:
    z_seed, seed_re, seed_im = _seed_at_gamma(
        q, K_clause, r, beta, anchor_gamma, cont_rows, dps=dps
    )
    sys = SaddleSystem.build(q=q, r=r, betas=np.array([beta]), gammas=np.array([anchor_gamma]))
    x_pol, res = polish_to_residual(
        sys, np.concatenate([z_anchor.real, z_anchor.imag]), min_residual=1e-10, dps=dps
    )
    z_pol = _x_to_z(x_pol, sys.nvars)
    ok, proof = certify_z(sys, z_pol, dps=dps)
    phi = compute_phi(z_pol, q=q, r=r, betas=sys.betas, gammas=sys.gammas)
    return BranchStep(
        gamma=float(anchor_gamma),
        direction="anchor",
        step_index=0,
        certified=bool(ok) and res <= 1e-10,
        failed=not ok or res > 1e-10,
        failure_reason="" if ok else str(proof.get("reason", "krawczyk_failed")),
        residual_inf=float(res),
        krawczyk_contraction=float(proof.get("contraction_bound", math.inf)),
        re_phi_m=float(phi.real),
        im_phi_m=float(phi.imag),
        im_phi_m_unwrapped=float(phi.imag),
        z_distance_from_previous=0.0,
        z_distance_from_seed=float(np.linalg.norm(z_pol - z_seed, ord=np.inf)),
        seed_re_phi_m=seed_re,
        seed_im_phi_m=seed_im,
        seed_im_phi_m_unwrapped=seed_im,
        signed_re_phi_gap_vs_seed=float(phi.real - seed_re),
        z_jump_gt_1=False,
        im_phi_jump_gt_pi=False,
        z_real=z_pol.real.tolist(),
        z_imag=z_pol.imag.tolist(),
    )


def _apply_unwrap(steps: list[BranchStep]) -> None:
    ordered = sorted(steps, key=lambda s: s.gamma)
    comp_im, ok_c = unwrap_im_branch([s.im_phi_m for s in ordered])
    seed_im, ok_s = unwrap_im_branch([s.seed_im_phi_m for s in ordered])
    by_g = {s.gamma: s for s in steps}
    for s, cu, su in zip(ordered, comp_im, seed_im, strict=True):
        by_g[s.gamma].im_phi_m_unwrapped = float(cu)
        by_g[s.gamma].seed_im_phi_m_unwrapped = float(su)
    return ok_c, ok_s


def _plots(steps: list[BranchStep], out_dir: Path) -> None:
    ordered = sorted(steps, key=lambda s: s.gamma)
    g = [s.gamma for s in ordered]
    comp_re = [s.re_phi_m for s in ordered]
    seed_re = [s.seed_re_phi_m for s in ordered]
    gap = [s.signed_re_phi_gap_vs_seed for s in ordered]
    im_diff = [
        s.seed_im_phi_m_unwrapped - s.im_phi_m_unwrapped for s in ordered
    ]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(g, seed_re, "o-", label="Seed branch Re $\\Phi$")
    ax.plot(g, comp_re, "s-", label="Competitor branch Re $\\Phi$")
    ax.set_xlabel(r"$\gamma$")
    ax.set_ylabel(r"$\mathrm{Re}\,\Phi$")
    ax.set_title(r"Seed vs high-Re competitor continuation (anchor $\gamma=-0.83$)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "competitor_branch_re_phi_vs_seed.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(g, gap, "o-", color="C3")
    ax.axhline(0.0, color="k", lw=0.8, alpha=0.5)
    ax.set_xlabel(r"$\gamma$")
    ax.set_ylabel(r"$\mathrm{Re}\,\Phi_{\mathrm{comp}} - \mathrm{Re}\,\Phi_{\mathrm{seed}}$")
    ax.set_title("Signed Re $\\Phi$ gap along competitor branch")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "competitor_branch_signed_re_gap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(g, im_diff, "o-", color="C2")
    ax.axhline(0.0, color="k", lw=0.8, alpha=0.5)
    ax.set_xlabel(r"$\gamma$")
    ax.set_ylabel(r"$\mathrm{Im}\,\Phi_{\mathrm{seed}} - \mathrm{Im}\,\Phi_{\mathrm{comp}}$ (unwrapped)")
    ax.set_title("Unwrapped Im $\\Phi$ difference along continuation")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "competitor_branch_im_phi_difference.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def run(
    *,
    anchor_gamma: float = -0.83,
    gamma_toward_zero_end: float = -0.05,
    gamma_toward_neg_end: float = -2.0,
    gamma_step: float = 0.01,
    min_step: float = 0.0005,
    max_re_json: Path,
    continuation_json: Path,
    out_dir: Path,
    dps: int = 80,
) -> Path:
    q, K_clause, r, beta = 3, 8, 176.54, 0.5433996420760803
    cont_rows = _load_continuation(continuation_json)
    z_anchor = _load_anchor_z(max_re_json, anchor_gamma)

    anchor = _anchor_step(q, K_clause, r, beta, z_anchor, anchor_gamma, cont_rows, dps)
    z_cert = _z_from_dict(anchor.to_dict())
    im0 = anchor.im_phi_m

    rows_neg, _ = _continue_direction(
        q=q,
        K_clause=K_clause,
        r=r,
        beta=beta,
        z_start=z_cert,
        gamma_start=anchor_gamma,
        gamma_end=gamma_toward_neg_end,
        gamma_step=gamma_step,
        min_step=min_step,
        direction="toward_neg",
        cont_rows=cont_rows,
        dps=dps,
        im_anchor=im0,
    )
    rows_pos, _ = _continue_direction(
        q=q,
        K_clause=K_clause,
        r=r,
        beta=beta,
        z_start=z_cert,
        gamma_start=anchor_gamma,
        gamma_end=gamma_toward_zero_end,
        gamma_step=gamma_step,
        min_step=min_step,
        direction="toward_zero",
        cont_rows=cont_rows,
        dps=dps,
        im_anchor=im0,
    )

    all_steps = rows_neg + [anchor] + rows_pos
    ok_c, ok_s = _apply_unwrap(all_steps)
    flags = {
        "z_jump_gt_1": [s.to_dict() for s in all_steps if s.z_jump_gt_1],
        "im_phi_jump_gt_pi": [s.to_dict() for s in all_steps if s.im_phi_jump_gt_pi],
    }
    certified = [s for s in all_steps if s.certified and not s.failed]
    z_steps = [s.z_distance_from_previous for s in certified if s.direction != "anchor"]

    payload = {
        "metadata": {
            "anchor_gamma": anchor_gamma,
            "gamma_toward_zero_end": gamma_toward_zero_end,
            "gamma_toward_neg_end": gamma_toward_neg_end,
            "gamma_step_initial": gamma_step,
            "min_step": min_step,
            "source_max_re_json": str(max_re_json.resolve()),
            "continuation_json": str(continuation_json.resolve()),
            "z_jump_flag_threshold": Z_JUMP_FLAG,
            "im_phi_jump_flag_threshold": IM_JUMP_FLAG,
            "im_phi_warning": IM_PHI_WARNING,
            "disclaimer": (
                "Dense continuation of one discovered high-Re saddle from anchor gamma. "
                "Not a global branch uniqueness or dominance proof."
            ),
            "timestamp": datetime.now(timezone.utc).strftime("run_%m-%d_%H-%M-%SZ"),
        },
        "unwrap_ok": {"competitor": ok_c, "seed": ok_s},
        "flags": flags,
        "summary": {
            "num_steps": len(all_steps),
            "num_certified": len(certified),
            "num_failed": sum(1 for s in all_steps if s.failed),
            "max_z_step": max(z_steps, default=0.0),
            "num_z_jump_gt_1": len(flags["z_jump_gt_1"]),
            "num_im_jump_gt_pi": len(flags["im_phi_jump_gt_pi"]),
            "gamma_range": [min(s.gamma for s in certified), max(s.gamma for s in certified)]
            if certified
            else None,
        },
        "steps": [s.to_dict() for s in sorted(all_steps, key=lambda s: s.gamma)],
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / OUT_JSON_NAME
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _plots(all_steps, out_dir)
    print(json.dumps(payload["summary"], indent=2))
    print(f"Wrote {out_json}")
    return out_dir


def main() -> None:
    p = argparse.ArgumentParser(description="Continue high-Re competitor branch from anchor gamma")
    p.add_argument("--anchor-gamma", type=float, default=-0.83)
    p.add_argument("--gamma-toward-zero-end", type=float, default=-0.05)
    p.add_argument("--gamma-toward-neg-end", type=float, default=-2.0)
    p.add_argument("--gamma-step", type=float, default=0.01)
    p.add_argument("--min-step", type=float, default=0.0005)
    p.add_argument("--max-re-json", type=str, default=str(DEFAULT_MAX_RE_JSON))
    p.add_argument("--continuation-json", type=str, default=str(DEFAULT_CONTINUATION))
    p.add_argument("--out-dir", type=str, default="")
    p.add_argument("--quick", action="store_true", help="Coarse: step 0.05, ends -1.2 / -0.2")
    args = p.parse_args()
    out = (
        Path(args.out_dir)
        if args.out_dir
        else DEFAULT_MAX_RE_JSON.parent
    )
    kwargs = dict(
        anchor_gamma=args.anchor_gamma,
        gamma_toward_zero_end=args.gamma_toward_zero_end,
        gamma_toward_neg_end=args.gamma_toward_neg_end,
        gamma_step=args.gamma_step,
        min_step=args.min_step,
        max_re_json=Path(args.max_re_json),
        continuation_json=Path(args.continuation_json),
        out_dir=out,
    )
    if args.quick:
        kwargs.update(gamma_toward_zero_end=-0.2, gamma_toward_neg_end=-1.2, gamma_step=0.05)
    run(**kwargs)


if __name__ == "__main__":
    main()
