"""
Branch-valid competitor onset scan: discovery + local continuation classification
at fine gamma spacing between -0.65 and -0.83.

Outputs only under competitor_dominance/onset_scan/ (does not overwrite parent files).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from phasecraft.bm24_saddle_audit_p1.branch_classify_competitors import (
    LOCAL_DELTAS,
    ClassifiedAnchor,
    _classify_anchor,
    _probe_from_anchor,
    _select_anchors,
    _z_from_entry,
)
from phasecraft.bm24_saddle_audit_p1.competitor_dominance_diagnostic import (
    DEFAULT_CONTINUATION,
    DEFAULT_PRIOR_COMPETITORS,
    _load_continuation,
    _prior_root_z_list,
    interpolate_seed_z,
    run_gamma_diagnostic,
)
from phasecraft.bm24_saddle_audit_p1.audit import AUDIT_DIR
from phasecraft.bm24_saddle_audit_p1.continue_seed_branch_certified import IM_PHI_WARNING

ONSET_GAMMAS = [-0.65, -0.70, -0.75, -0.78, -0.80, -0.82, -0.83]
DEFAULT_OUT_DIR = (
    AUDIT_DIR
    / "results/run_seed_branch_g-2pi/competitor_dominance/onset_scan"
)
DISCLAIMER = (
    "Branch-valid algebraic competitor onset scan only. "
    "Not a PL/intersection-number proof or global dominance certification."
)


def _classify_gamma_row(
    per_gamma_row: dict,
    cont_rows: list[dict],
    *,
    q: int,
    K: int,
    r: float,
    beta: float,
    dps: int,
) -> list[ClassifiedAnchor]:
    g0 = float(per_gamma_row["gamma"])
    classified: list[ClassifiedAnchor] = []
    for j, anc in enumerate(_select_anchors(per_gamma_row)):
        z0 = _z_from_entry(anc)
        probes = [_probe_from_anchor(q, K, r, beta, g0, z0, g0, cont_rows, dps=dps)]
        for d in LOCAL_DELTAS:
            for sign in (-1.0, 1.0):
                gt = g0 + sign * d
                probes.append(
                    _probe_from_anchor(q, K, r, beta, g0, z0, gt, cont_rows, dps=dps)
                )
        label, cmeta = _classify_anchor(g0, anc, probes)
        classified.append(
            ClassifiedAnchor(
                anchor_gamma=g0,
                anchor_id=f"g{g0}_a{j}_re{anc['re_phi_m']:.4f}",
                classification=label,
                branch_valid=label == "stable_continuous_branch",
                re_phi_m=float(anc["re_phi_m"]),
                signed_re_phi_gap_vs_seed=float(anc["signed_re_phi_gap_vs_seed"]),
                z_distance_from_seed=float(anc["z_distance_from_seed"]),
                classification_meta=cmeta,
                probes=probes,
            )
        )
    return classified


def _leading_branch_valid(classified: list[ClassifiedAnchor]) -> dict | None:
    valid = [c for c in classified if c.branch_valid]
    if not valid:
        return None
    best = max(valid, key=lambda c: c.signed_re_phi_gap_vs_seed)
    return {
        "anchor_id": best.anchor_id,
        "classification": best.classification,
        "re_phi_m": best.re_phi_m,
        "signed_re_phi_gap_vs_seed": best.signed_re_phi_gap_vs_seed,
        "z_distance_from_seed": best.z_distance_from_seed,
        "classification_meta": best.classification_meta,
    }


def run_onset_scan(
    *,
    gammas: list[float],
    continuation_path: Path,
    prior_competitors_json: Path,
    out_dir: Path,
    num_random_starts: int = 3000,
    seed: int = 0,
    dps: int = 80,
) -> Path:
    q, K, r, beta = 3, 8, 176.54, 0.5433996420760803
    out_dir.mkdir(parents=True, exist_ok=True)
    cont_rows = _load_continuation(continuation_path)

    all_classified: list[ClassifiedAnchor] = []
    per_gamma_summary: list[dict] = []
    discovery_rows: list[dict] = []

    for i, g in enumerate(gammas):
        print(f"=== gamma={g} discovery + classification ===", flush=True)
        z_init = interpolate_seed_z(cont_rows, g)
        prior_z = _prior_root_z_list(prior_competitors_json, g)
        row = run_gamma_diagnostic(
            q=q,
            K_clause=K,
            r=r,
            beta=beta,
            gamma=g,
            seed_z_init=z_init,
            prior_z=prior_z,
            num_random_starts=num_random_starts,
            seed=seed + i * 10007,
            seed_branch_tol=1e-6,
            dedup_tol=1e-5,
            min_residual=1e-10,
            dps=dps,
        )
        discovery_rows.append(row)

        classified = _classify_gamma_row(row, cont_rows, q=q, K=K, r=r, beta=beta, dps=dps)
        all_classified.extend(classified)
        valid = [c for c in classified if c.branch_valid]
        valid_max = max((c.signed_re_phi_gap_vs_seed for c in valid), default=None)
        leading = _leading_branch_valid(classified)
        summary = {
            "gamma": g,
            "seed_re_phi": row["seed_re_phi"],
            "num_discovered_certified_competitors": row["num_certified_competitors"],
            "max_algebraic_signed_re_gap_vs_seed": row["max_signed_gap_vs_seed"],
            "seed_has_largest_algebraic_re_phi": row["seed_has_largest_discovered_re_phi"],
            "num_branch_valid_competitors": len(valid),
            "max_branch_valid_signed_re_gap_vs_seed": valid_max,
            "branch_valid_dominates_seed": bool(valid_max is not None and valid_max > 0),
            "leading_branch_valid_competitor": leading,
        }
        per_gamma_summary.append(summary)
        print(
            f"  algebraic_max_gap={row['max_signed_gap_vs_seed']:.4g} "
            f"n_valid={len(valid)} valid_max_gap={valid_max}",
            flush=True,
        )

    ts = datetime.now(timezone.utc).strftime("run_%m-%d_%H-%M-%SZ")
    meta = {
        "scan_type": "branch_valid_algebraic_competitor_onset_scan",
        "q": q,
        "p": 1,
        "K_clause": K,
        "r": r,
        "beta": beta,
        "gammas": gammas,
        "local_deltas": LOCAL_DELTAS,
        "num_random_starts": num_random_starts,
        "continuation_json": str(continuation_path.resolve()),
        "prior_competitors_json": str(prior_competitors_json.resolve()),
        "im_phi_warning": IM_PHI_WARNING,
        "disclaimer": DISCLAIMER,
        "timestamp": ts,
    }

    onset_summary = {"metadata": meta, "per_gamma": per_gamma_summary}
    (out_dir / "branch_valid_onset_scan.json").write_text(
        json.dumps(onset_summary, indent=2), encoding="utf-8"
    )
    (out_dir / "branch_classified_onset_competitors.json").write_text(
        json.dumps(
            {
                "metadata": meta,
                "discovery_per_gamma": [
                    {k: v for k, v in r.items() if k != "competitors"}
                    for r in discovery_rows
                ],
                "anchors": [c.to_dict() for c in all_classified],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    _plots_onset(per_gamma_summary, out_dir)
    print(f"Wrote {out_dir}")
    return out_dir


def _plots_onset(per_gamma: list[dict], out_dir: Path) -> None:
    g = [r["gamma"] for r in per_gamma]
    alg = [r["max_algebraic_signed_re_gap_vs_seed"] for r in per_gamma]
    valid_gap = [
        r["max_branch_valid_signed_re_gap_vs_seed"]
        if r["max_branch_valid_signed_re_gap_vs_seed"] is not None
        else float("nan")
        for r in per_gamma
    ]
    n_valid = [r["num_branch_valid_competitors"] for r in per_gamma]
    seed_re = [r["seed_re_phi"] for r in per_gamma]
    lead_re = []
    for r in per_gamma:
        lead = r.get("leading_branch_valid_competitor")
        lead_re.append(lead["re_phi_m"] if lead else float("nan"))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(g, valid_gap, "o-", color="C0", label="max branch-valid gap")
    ax.plot(g, alg, "s--", alpha=0.6, label="max algebraic gap")
    ax.axhline(0.0, color="k", lw=0.8, alpha=0.5)
    ax.set_xlabel(r"$\gamma$")
    ax.set_ylabel(r"signed $\mathrm{Re}\,\Phi_{\mathrm{comp}} - \mathrm{Re}\,\Phi_{\mathrm{seed}}$")
    ax.set_title("Onset scan: branch-valid vs algebraic max gap")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "gamma_vs_max_branch_valid_gap_onset.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(g, n_valid, width=0.015, color="C2", alpha=0.85)
    ax.set_xlabel(r"$\gamma$")
    ax.set_ylabel("count")
    ax.set_title("Branch-valid competitors per $\\gamma$ (onset scan)")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_dir / "gamma_vs_num_branch_valid_competitors_onset.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(g, seed_re, "o-", label="Seed Re $\\Phi$")
    ax.plot(g, lead_re, "s-", label="Leading branch-valid Re $\\Phi$")
    ax.set_xlabel(r"$\gamma$")
    ax.set_ylabel(r"$\mathrm{Re}\,\Phi$")
    ax.set_title("Seed vs leading branch-valid competitor (onset scan)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "seed_vs_leading_branch_valid_competitor_onset.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Branch-valid competitor onset scan")
    p.add_argument("--continuation-json", type=str, default=str(DEFAULT_CONTINUATION))
    p.add_argument("--prior-competitors-json", type=str, default=str(DEFAULT_PRIOR_COMPETITORS))
    p.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR))
    p.add_argument("--num-random-starts", type=int, default=3000)
    p.add_argument("--quick", action="store_true", help="400 starts, gammas -0.80,-0.83 only")
    args = p.parse_args()
    gammas = list(ONSET_GAMMAS)
    num_starts = args.num_random_starts
    if args.quick:
        gammas = [-0.80, -0.83]
        num_starts = 400
    run_onset_scan(
        gammas=gammas,
        continuation_path=Path(args.continuation_json),
        prior_competitors_json=Path(args.prior_competitors_json),
        out_dir=Path(args.out_dir),
        num_random_starts=num_starts,
    )


if __name__ == "__main__":
    main()
