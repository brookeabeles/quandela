#!/usr/bin/env python3
"""Overlay p=1 q=3 seed-branch tracking: w,u-saddle vs z-saddle."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_Z_SEED = REPO_ROOT / "experiments/w_saddle/runs/compare_q3_matched/z/seed_branch_z.json"
DEFAULT_Z_RESOLVED = REPO_ROOT / "experiments/w_saddle/runs/compare_q3_matched/z/resolved_robust_z.json"
DEFAULT_Z_SWEEP = REPO_ROOT / "experiments/w_saddle/runs/compare_q3_matched/z/competitors_robust_z.json"
DEFAULT_WU_SEED = REPO_ROOT / "experiments/w_saddle/runs/compare_q3_matched/wu/seed_branch_wu.json"
DEFAULT_WU_RESOLVED = REPO_ROOT / "experiments/w_saddle/runs/compare_q3_matched/wu/resolved_robust_wu.json"
DEFAULT_WU_SWEEP = REPO_ROOT / "experiments/w_saddle/runs/compare_q3_matched/wu/competitors_robust_wu.json"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def seed_curve(payload: dict) -> tuple[np.ndarray, np.ndarray]:
    pts = sorted(payload["points"], key=lambda p: float(p["gamma"]), reverse=True)
    g = np.array([float(p["gamma"]) for p in pts])
    re = np.array([float(p["Phi_eff_real"]) for p in pts])
    return g, re


def sweep_max_re(payload: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gammas, max_re, n_roots = [], [], []
    if payload.get("gamma_slices"):
        for sl in payload["gamma_slices"]:
            g = float(sl["gamma"])
            comps = [
                n for n in sl.get("nodes", [])
                if not n.get("is_seed_sheet") and not n.get("is_seed_duplicate")
            ]
            if not comps:
                continue
            gammas.append(g)
            max_re.append(max(float(n["Phi_eff_real"]) for n in comps))
            n_roots.append(len(comps))
    else:
        for pt in payload.get("points", []):
            sweep = pt.get("competitor_sweep") or {}
            roots = sweep.get("certified_roots") or []
            comps = [r for r in roots if not r.get("is_seed_duplicate")]
            if not comps:
                continue
            g = float(pt["gamma"])
            gammas.append(g)
            max_re.append(max(float(r["Phi_eff_real"]) for r in comps))
            n_roots.append(len(comps))
    if not gammas:
        return np.array([]), np.array([]), np.array([])
    order = np.argsort(gammas)[::-1]
    g = np.asarray(gammas)[order]
    return g, np.asarray(max_re)[order], np.asarray(n_roots)[order]


def branch_stats(resolved: dict) -> dict:
    branches = resolved.get("branches", [])
    comp = [b for b in branches if not b.get("is_seed_sheet")]
    lengths = [int(b.get("num_points", len(b.get("points", [])))) for b in comp]
    return {
        "num_branches": int(resolved.get("num_branches", len(branches))),
        "num_competitor_branches": len(comp),
        "max_track_len": max(lengths) if lengths else 0,
        "median_track_len": float(np.median(lengths)) if lengths else 0.0,
        "num_crossings": len(resolved.get("delta_re_crossing_intervals_mesh", [])),
    }


def plot_comparison(
    z_seed: dict,
    wu_seed: dict,
    z_sweep: dict,
    wu_sweep: dict,
    z_resolved: dict,
    wu_resolved: dict,
    out_path: Path,
) -> None:
    z_g, z_re = seed_curve(z_seed)
    wu_g, wu_re = seed_curve(wu_seed)
    z_sg, z_max, z_cnt = sweep_max_re(z_sweep)
    wu_sg, wu_max, wu_cnt = sweep_max_re(wu_sweep)
    z_stats = branch_stats(z_resolved)
    wu_stats = branch_stats(wu_resolved)

    beta = float(z_seed.get("beta", wu_seed.get("beta", float("nan"))))
    r = float(z_seed.get("r", wu_seed.get("r", float("nan"))))

    fig = plt.figure(figsize=(14, 11))
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.38, wspace=0.28)

    ax_seed = fig.add_subplot(gs[0, :])
    ax_seed.plot(z_g, z_re, "C3o-", ms=3, lw=1.6, label=f"z seed (8D, {len(z_g)} pts)")
    ax_seed.plot(wu_g, wu_re, "C0s-", ms=4, lw=1.8, label=f"w,u seed (4D, {len(wu_g)} pts)")
    ax_seed.axhline(0, color="k", lw=0.5, ls="--")
    ax_seed.set_xlabel("γ")
    ax_seed.set_ylabel("Re Φ on certified seed branch")
    ax_seed.set_title(
        "Seed-branch tracking: Φ_z vs Φ_wu are different actions (not directly comparable magnitudes)"
    )
    ax_seed.legend(fontsize=9)
    ax_seed.grid(True, alpha=0.25)

    ax_zmax = fig.add_subplot(gs[1, 0])
    ax_zmax.plot(z_sg, z_max, "C3o-", ms=3, lw=1.4)
    ax_zmax.set_xlabel("γ")
    ax_zmax.set_ylabel("max competitor Re Φ")
    ax_zmax.set_title(f"z-saddle sweep ({len(z_sg)} γ slices)")
    ax_zmax.grid(True, alpha=0.25)

    ax_wmax = fig.add_subplot(gs[1, 1])
    ax_wmax.plot(wu_sg, wu_max, "C0s-", ms=3, lw=1.4)
    ax_wmax.set_xlabel("γ")
    ax_wmax.set_ylabel("max competitor Re Φ")
    ax_wmax.set_title(f"w,u-saddle sweep ({len(wu_sg)} γ slices)")
    ax_wmax.grid(True, alpha=0.25)

    ax_cnt = fig.add_subplot(gs[2, 0])
    ax_cnt.plot(z_sg, z_cnt, "C3o-", ms=3, lw=1.4, label="z competitors/slice")
    ax_cnt.plot(wu_sg, wu_cnt, "C0s--", ms=3, lw=1.4, label="w,u competitors/slice")
    ax_cnt.set_xlabel("γ")
    ax_cnt.set_ylabel("# certified competitors per γ")
    ax_cnt.set_title("Competitor discovery density")
    ax_cnt.legend(fontsize=9)
    ax_cnt.grid(True, alpha=0.25)

    ax_sum = fig.add_subplot(gs[2, 1])
    ax_sum.axis("off")
    lines = [
        f"p=1, q=3 (k=8), r={r:.2f}, β={beta:.4f}",
        "",
        "Resolved branch tracking",
        f"  z:  {z_stats['num_competitor_branches']} competitor sheets, "
        f"max len {z_stats['max_track_len']}, "
        f"{z_stats['num_crossings']} ΔRe crossings",
        f"  w,u: {wu_stats['num_competitor_branches']} competitor sheets, "
        f"max len {wu_stats['max_track_len']}, "
        f"{wu_stats['num_crossings']} ΔRe crossings",
        "",
        "Key structural difference",
        "  z: 8D chart, c_root = (-c)^(1/8) fractional root",
        "  w,u: 4D chart, u = i g(w), no fractional root",
        "",
        "At γ=-0.01 seed Re Φ:",
        f"  z  = {z_re[-1]:+.4f}",
        f"  w,u = {wu_re[-1]:+.4f}",
    ]
    ax_sum.text(0.02, 0.98, "\n".join(lines), va="top", ha="left", fontsize=10, family="monospace")

    fig.suptitle(
        "p=1 q=3 saddle system tracking: w,u vs z\n"
        "(seed continuation + competitor branch resolve)",
        fontsize=12,
        y=0.98,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot → {out_path}")
    print(json.dumps({"z": z_stats, "wu": wu_stats}, indent=2))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--z-seed", type=Path, default=DEFAULT_Z_SEED)
    p.add_argument("--z-resolved", type=Path, default=DEFAULT_Z_RESOLVED)
    p.add_argument("--z-sweep", type=Path, default=DEFAULT_Z_SWEEP)
    p.add_argument("--wu-seed", type=Path, default=DEFAULT_WU_SEED)
    p.add_argument("--wu-resolved", type=Path, default=DEFAULT_WU_RESOLVED)
    p.add_argument("--wu-sweep", type=Path, default=DEFAULT_WU_SWEEP)
    p.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "experiments/w_saddle/runs/compare_z_wu_tracking.png",
    )
    args = p.parse_args()
    plot_comparison(
        load_json(args.z_seed),
        load_json(args.wu_seed),
        load_json(args.z_sweep),
        load_json(args.wu_sweep),
        load_json(args.z_resolved),
        load_json(args.wu_resolved),
        args.output,
    )


if __name__ == "__main__":
    main()
