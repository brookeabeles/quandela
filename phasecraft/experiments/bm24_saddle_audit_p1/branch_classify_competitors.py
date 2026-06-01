"""
Classify discovered BM24 competitors by local gamma continuation from each anchor.

Separates algebraic high-Re roots from stable continuous branches vs seed collapse.
Not a PL/intersection dominance proof.
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

from phasecraft.bm24_saddle_audit_p1.audit import AUDIT_DIR, _x_to_z
from phasecraft.bm24_saddle_audit_p1.competitor_dominance_diagnostic import (
    DEFAULT_CONTINUATION,
    _load_continuation,
    interpolate_seed_z,
)
from phasecraft.bm24_saddle_audit_p1.continue_seed_branch_certified import (
    IM_PHI_WARNING,
    certify_z,
    polish_to_residual,
    step_from_previous_z,
)
from phasecraft.krawczyk_p1_roots import SaddleSystem
from phasecraft.picard_lefschetz import compute_phi
from phasecraft.w_saddle.workflow import unwrap_im_branch

DEFAULT_GAMMAS = [-0.3, -0.6, -0.83, -1.0, -1.6, -2.0]
LOCAL_DELTAS = [0.001, 0.002, 0.005, 0.01, 0.02]
DEFAULT_DOMINANCE = (
    AUDIT_DIR
    / "results/run_seed_branch_g-2pi/competitor_dominance/competitor_dominance_summary.json"
)
DEFAULT_OUT_DIR = DEFAULT_DOMINANCE.parent

Z_STEP_MAX = 1.0
Z_STEP_PREFER = 0.5
SEED_Z_TOL = 0.05
SEED_RE_TOL = 0.05
HIGH_RE_GAP = 0.5
NEAR_SEED_GAP = 0.2


def _z_from_entry(entry: dict) -> np.ndarray:
    return np.asarray(entry["z_real"], dtype=float) + 1j * np.asarray(entry["z_imag"], dtype=float)


def _seed_at_gamma(
    q: int,
    K: int,
    r: float,
    beta: float,
    gamma: float,
    cont_rows: list[dict],
    *,
    dps: int,
) -> tuple[np.ndarray, float, float]:
    z_init = interpolate_seed_z(cont_rows, gamma)
    z_s, row = step_from_previous_z(
        q, K, r, beta, gamma, z_init,
        step_index=0, n_values=list(range(12, 23)),
        match_tol=0.02, min_residual=1e-10, dps=dps,
    )
    if z_s is None or not row.certified:
        raise RuntimeError(f"seed not certified at gamma={gamma}")
    return z_s, float(row.re_phi_m), float(row.im_phi_m)


def _probe_from_anchor(
    q: int,
    K: int,
    r: float,
    beta: float,
    gamma0: float,
    z_anchor: np.ndarray,
    gamma_target: float,
    cont_rows: list[dict],
    *,
    dps: int,
) -> dict:
    sys = SaddleSystem.build(q=q, r=r, betas=np.array([beta]), gammas=np.array([gamma_target]))
    x_pol, res = polish_to_residual(
        sys, np.concatenate([z_anchor.real, z_anchor.imag]), min_residual=1e-10, dps=dps
    )
    z_pol = _x_to_z(x_pol, sys.nvars)
    z_dist_anchor = float(np.linalg.norm(z_pol - z_anchor, ord=np.inf))
    z_seed, seed_re, seed_im = _seed_at_gamma(q, K, r, beta, gamma_target, cont_rows, dps=dps)
    z_dist_seed = float(np.linalg.norm(z_pol - z_seed, ord=np.inf))
    ok, proof = certify_z(sys, z_pol, dps=dps)
    phi = compute_phi(z_pol, q=q, r=r, betas=sys.betas, gammas=sys.gammas)
    re_phi = float(phi.real)
    im_phi = float(phi.imag)
    return {
        "gamma": float(gamma_target),
        "delta_gamma": float(gamma_target - gamma0),
        "certified": bool(ok) and res <= 1e-10,
        "residual_inf": float(res),
        "krawczyk_contraction": float(proof.get("contraction_bound", math.inf)),
        "re_phi_m": re_phi,
        "im_phi_m": im_phi,
        "signed_re_phi_gap_vs_seed": re_phi - seed_re,
        "z_distance_from_anchor": z_dist_anchor,
        "z_distance_from_seed": z_dist_seed,
        "collapsed_to_seed": bool(
            z_dist_seed < SEED_Z_TOL and abs(re_phi - seed_re) < SEED_RE_TOL
        ),
        "seed_re_phi_m": seed_re,
    }


def _select_anchors(per_gamma_row: dict) -> list[dict]:
    """Anchors: max-gap, nearest-gap, and other high-Re roots (deduped by z)."""
    comps = [c for c in per_gamma_row["competitors"] if c.get("certified") and not c.get("is_seed_branch")]
    if not comps:
        return []

    max_gap = max(comps, key=lambda c: float(c["signed_re_phi_gap_vs_seed"]))
    nearest = min(comps, key=lambda c: abs(float(c["signed_re_phi_gap_vs_seed"])))
    candidates = [max_gap, nearest]
    candidates.extend(
        c for c in comps
        if float(c["signed_re_phi_gap_vs_seed"]) >= HIGH_RE_GAP
        and c is not max_gap
    )

    selected: list[dict] = []
    for c in candidates:
        z = _z_from_entry(c)
        if any(np.linalg.norm(z - _z_from_entry(s), ord=np.inf) < 1e-4 for s in selected):
            continue
        selected.append(c)
    return selected


def _classify_anchor(
    gamma0: float,
    anchor_entry: dict,
    probes: list[dict],
) -> tuple[str, dict]:
    """Classify from anchor + local offset probes (sorted by |delta|)."""
    ordered = sorted(probes, key=lambda p: abs(p["delta_gamma"]))
    certified = [p for p in ordered if p["certified"]]
    if not certified:
        return "failed_or_unstable", {"reason": "no_certified_probes"}

    z_steps = [p["z_distance_from_anchor"] for p in ordered if p["delta_gamma"] != 0.0]
    max_z = max(z_steps, default=0.0)
    n_collapsed = sum(1 for p in ordered if p.get("collapsed_to_seed"))
    im_raw = [p["im_phi_m"] for p in ordered]
    im_u, im_continuous = unwrap_im_branch(im_raw)
    im_jumps = [
        abs(im_raw[i] - im_raw[i - 1])
        for i in range(1, len(im_raw))
        if ordered[i]["delta_gamma"] != 0.0 or ordered[i - 1]["delta_gamma"] != 0.0
    ]
    max_im_jump = max(im_jumps, default=0.0)

    meta = {
        "max_z_step_from_anchor": max_z,
        "n_probes": len(ordered),
        "n_certified": len(certified),
        "n_collapsed_to_seed": n_collapsed,
        "max_im_phi_raw_jump": max_im_jump,
        "im_unwrap_continuous": im_continuous,
        "anchor_re_phi_m": float(anchor_entry["re_phi_m"]),
        "anchor_signed_gap": float(anchor_entry["signed_re_phi_gap_vs_seed"]),
    }

    if n_collapsed >= max(1, len(ordered) // 2):
        return "collapses_to_seed", meta

    if max_z >= Z_STEP_MAX or len(certified) < len(ordered) - 1:
        if max_z >= Z_STEP_MAX:
            return "failed_or_unstable", {**meta, "reason": f"z_step>={Z_STEP_MAX}"}
        return "failed_or_unstable", {**meta, "reason": "certification_gaps"}

    if max_im_jump > math.pi and not im_continuous:
        return "log_sheet_artifact", {**meta, "reason": "im_phi_2pi_jumps"}

    if max_z < Z_STEP_PREFER and n_collapsed == 0 and len(certified) == len(ordered):
        return "stable_continuous_branch", meta

    if max_z < Z_STEP_MAX and n_collapsed == 0:
        return "stable_continuous_branch", {**meta, "note": "z_step_between_prefer_and_max"}

    return "log_sheet_artifact", {**meta, "reason": "marginal_z_or_im_behavior"}


@dataclass
class ClassifiedAnchor:
    anchor_gamma: float
    anchor_id: str
    classification: str
    branch_valid: bool
    re_phi_m: float
    signed_re_phi_gap_vs_seed: float
    z_distance_from_seed: float
    classification_meta: dict
    probes: list[dict]

    def to_dict(self) -> dict:
        return asdict(self)


def run_classification(
    *,
    dominance_path: Path,
    continuation_path: Path,
    out_dir: Path,
    gammas: list[float],
    dps: int = 80,
) -> Path:
    q, K, r, beta = 3, 8, 176.54, 0.5433996420760803
    dominance = json.loads(dominance_path.read_text(encoding="utf-8"))
    cont_rows = _load_continuation(continuation_path)
    by_gamma = {float(r["gamma"]): r for r in dominance["per_gamma"]}

    classified: list[ClassifiedAnchor] = []
    for g0 in gammas:
        if g0 not in by_gamma:
            continue
        row = by_gamma[g0]
        seed_re = float(row["seed_re_phi"])
        anchors = _select_anchors(row)
        print(f"gamma0={g0}: {len(anchors)} anchors", flush=True)
        for j, anc in enumerate(anchors):
            z0 = _z_from_entry(anc)
            probes: list[dict] = []
            # anchor point
            probes.append(
                _probe_from_anchor(q, K, r, beta, g0, z0, g0, cont_rows, dps=dps)
            )
            for d in LOCAL_DELTAS:
                for sign in (-1.0, 1.0):
                    gt = g0 + sign * d
                    probes.append(
                        _probe_from_anchor(q, K, r, beta, g0, z0, gt, cont_rows, dps=dps)
                    )
            label, cmeta = _classify_anchor(g0, anc, probes)
            branch_valid = label == "stable_continuous_branch"
            aid = f"g{g0}_a{j}_re{anc['re_phi_m']:.4f}"
            classified.append(
                ClassifiedAnchor(
                    anchor_gamma=g0,
                    anchor_id=aid,
                    classification=label,
                    branch_valid=branch_valid,
                    re_phi_m=float(anc["re_phi_m"]),
                    signed_re_phi_gap_vs_seed=float(anc["signed_re_phi_gap_vs_seed"]),
                    z_distance_from_seed=float(anc["z_distance_from_seed"]),
                    classification_meta=cmeta,
                    probes=probes,
                )
            )
            print(f"  {aid} -> {label} (gap={anc['signed_re_phi_gap_vs_seed']:.3g})", flush=True)

    # Per-gamma dominance summary
    per_gamma_summary: list[dict] = []
    for g0 in gammas:
        if g0 not in by_gamma:
            continue
        row = by_gamma[g0]
        seed_re = float(row["seed_re_phi"])
        comps = [c for c in row["competitors"] if c.get("certified") and not c.get("is_seed_branch")]
        alg_max_gap = float(row.get("max_signed_gap_vs_seed", float("-inf")))
        valid = [c for c in classified if c.anchor_gamma == g0 and c.branch_valid]
        valid_max_gap = max((c.signed_re_phi_gap_vs_seed for c in valid), default=float("-inf"))
        if valid_max_gap == float("-inf"):
            valid_max_gap = None
        per_gamma_summary.append(
            {
                "gamma": g0,
                "seed_re_phi": seed_re,
                "num_discovered_certified_competitors": len(comps),
                "max_algebraic_signed_re_gap_vs_seed": alg_max_gap,
                "seed_has_largest_algebraic_re_phi": bool(row.get("seed_has_largest_discovered_re_phi", True)),
                "num_branch_valid_competitors": len(valid),
                "max_branch_valid_signed_re_gap_vs_seed": valid_max_gap,
                "branch_valid_dominates_seed": bool(valid_max_gap is not None and valid_max_gap > 0),
                "disclaimer": (
                    "branch_valid requires local continuation stability; "
                    "not PL/intersection dominance."
                ),
            }
        )

    ts = datetime.now(timezone.utc).strftime("run_%m-%d_%H-%M-%SZ")
    meta = {
        "gammas": gammas,
        "local_deltas": LOCAL_DELTAS,
        "classification_thresholds": {
            "z_step_max": Z_STEP_MAX,
            "z_step_prefer": Z_STEP_PREFER,
            "seed_z_tol": SEED_Z_TOL,
            "seed_re_tol": SEED_RE_TOL,
            "high_re_gap": HIGH_RE_GAP,
            "near_seed_gap": NEAR_SEED_GAP,
        },
        "inputs": {
            "competitor_dominance_summary": str(dominance_path.resolve()),
            "continuation_json": str(continuation_path.resolve()),
        },
        "im_phi_warning": IM_PHI_WARNING,
        "timestamp": ts,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    classified_out = {
        "metadata": meta,
        "anchors": [c.to_dict() for c in classified],
    }
    (out_dir / "branch_classified_competitors.json").write_text(
        json.dumps(classified_out, indent=2), encoding="utf-8"
    )
    (out_dir / "branch_valid_dominance_summary.json").write_text(
        json.dumps({"metadata": meta, "per_gamma": per_gamma_summary}, indent=2),
        encoding="utf-8",
    )

    _plots(per_gamma_summary, out_dir)
    print(f"Wrote {out_dir / 'branch_classified_competitors.json'}")
    return out_dir


def _plots(per_gamma: list[dict], out_dir: Path) -> None:
    g = [r["gamma"] for r in per_gamma]
    alg = [r["max_algebraic_signed_re_gap_vs_seed"] for r in per_gamma]
    valid = [
        r["max_branch_valid_signed_re_gap_vs_seed"]
        if r["max_branch_valid_signed_re_gap_vs_seed"] is not None
        else float("nan")
        for r in per_gamma
    ]
    n_valid = [r["num_branch_valid_competitors"] for r in per_gamma]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(g, alg, "o-", label="max algebraic (discovered)")
    ax.plot(g, valid, "s-", label="max branch-valid")
    ax.axhline(0.0, color="k", lw=0.8, alpha=0.5)
    ax.set_xlabel(r"$\gamma$")
    ax.set_ylabel(r"signed $\mathrm{Re}\,\Phi_{\mathrm{comp}} - \mathrm{Re}\,\Phi_{\mathrm{seed}}$")
    ax.set_title("Algebraic vs branch-valid Re $\\Phi$ gap (not PL dominance)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "gamma_vs_algebraic_and_branch_valid_gap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(g, n_valid, width=0.08, color="C2", alpha=0.85)
    ax.set_xlabel(r"$\gamma$")
    ax.set_ylabel("count")
    ax.set_title("Branch-valid competitors per $\\gamma$")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_dir / "gamma_vs_num_branch_valid.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Classify competitors by local gamma continuation")
    p.add_argument("--dominance-json", type=str, default=str(DEFAULT_DOMINANCE))
    p.add_argument("--continuation-json", type=str, default=str(DEFAULT_CONTINUATION))
    p.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR))
    p.add_argument("--quick", action="store_true", help="Only gamma=-0.83, deltas 0.01 only")
    args = p.parse_args()
    gammas = list(DEFAULT_GAMMAS)
    if args.quick:
        gammas = [-0.83]
    run_classification(
        dominance_path=Path(args.dominance_json),
        continuation_path=Path(args.continuation_json),
        out_dir=Path(args.out_dir),
        gammas=gammas,
    )


if __name__ == "__main__":
    main()
