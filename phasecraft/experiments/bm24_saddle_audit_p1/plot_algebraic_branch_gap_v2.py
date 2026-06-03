"""
v2: Algebraic vs same-branch competitor dominance along gamma.

Fixes misleading v1 comparisons:
  1. Green = gap along ONE Krawczyk-tracked competitor sheet (dense continuation
     from anchor gamma), not max over loosely probed anchors.
  2. Blue = max algebraic gap only at gammas with discovery data (markers, no
     false line through missing branch-valid points).
  5. Dense gamma mesh from branch continuation (~0.01), not six sparse points.

Also plots Conv2 full-exponent gaps (not Re Phi alone) and marks z-jump events.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from phasecraft.bm24_saddle_audit_p1.continue_seed_branch_certified import (
    IM_PHI_WARNING,
    conv2_full_exponent,
    _style_gamma_axis_reading_zero_to_negative,
)

DEFAULT_DOMINANCE_DIR = (
    REPO_ROOT
    / "phasecraft/experiments/bm24_saddle_audit_p1/results/run_seed_branch_g-2pi/competitor_dominance"
)

DISCLAIMER = (
    "Blue: max Re gap over all certified discoveries at that γ (may be different roots/sheets). "
    "Green: same high-Re branch tracked by z-continuation from γ_anchor. "
    "Not PL / intersection dominance. Conv2 exponent gap uses BM24 A41 prefactor."
)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _seed_series(continuation_path: Path) -> dict[float, dict[str, float]]:
    data = _load_json(continuation_path)
    out: dict[float, dict[str, float]] = {}
    for row in data.get("continuation", []):
        if not row.get("certified") or row.get("failed"):
            continue
        g = float(row["gamma"])
        out[g] = {
            "re_phi_m": float(row["re_phi_m"]),
            "full_conv2": float(row["full_conv2_exponent"]),
        }
    return out


def _interp_seed(seed_by_g: dict[float, dict[str, float]], gamma: float) -> dict[str, float]:
    keys = sorted(seed_by_g.keys())
    if not keys:
        raise ValueError("empty seed series")
    if gamma in seed_by_g:
        return seed_by_g[gamma]
    if gamma >= keys[0]:
        return seed_by_g[keys[0]]
    if gamma <= keys[-1]:
        return seed_by_g[keys[-1]]
    for i in range(len(keys) - 1):
        g0, g1 = keys[i], keys[i + 1]
        if (g0 >= gamma >= g1) or (g1 >= gamma >= g0):
            t = (gamma - g0) / (g1 - g0) if abs(g1 - g0) > 1e-15 else 0.0
            s0, s1 = seed_by_g[g0], seed_by_g[g1]
            return {
                "re_phi_m": (1 - t) * s0["re_phi_m"] + t * s1["re_phi_m"],
                "full_conv2": (1 - t) * s0["full_conv2"] + t * s1["full_conv2"],
            }
    ref = min(keys, key=lambda k: abs(k - gamma))
    return seed_by_g[ref]


def _branch_track(branch_path: Path, *, K: int, r: float) -> list[dict[str, Any]]:
    data = _load_json(branch_path)
    rows = []
    for s in data.get("steps", []):
        if not s.get("certified") or s.get("failed"):
            continue
        g = float(s["gamma"])
        re_c = float(s["re_phi_m"])
        seed_re = float(s.get("seed_re_phi_m", 0.0))
        rows.append(
            {
                "gamma": g,
                "re_phi_m": re_c,
                "full_conv2": conv2_full_exponent(re_c, K, r),
                "signed_re_gap": float(s.get("signed_re_phi_gap_vs_seed", re_c - seed_re)),
                "z_step": float(s.get("z_distance_from_previous", 0.0)),
                "z_jump": bool(s.get("z_jump_gt_1", False)),
                "direction": s.get("direction", ""),
            }
        )
    return sorted(rows, key=lambda r: r["gamma"], reverse=True)


def _algebraic_max_from_saddles(
    saddles_path: Path,
    seed_by_g: dict[float, dict[str, float]],
    *,
    K: int,
    r: float,
) -> dict[float, dict[str, float]]:
    """Max signed Re gap over certified non-seed competitors per γ key in by_gamma."""
    data = _load_json(saddles_path)
    by_gamma = data.get("by_gamma") or {}
    out: dict[float, dict[str, float]] = {}
    for key, comps in by_gamma.items():
        g = float(key)
        certified = [
            c
            for c in comps
            if c.get("certified") and c.get("label") == "certified_competitor"
        ]
        if not certified:
            continue
        gap_key = "signed_re_phi_gap_vs_seed"
        if gap_key not in certified[0]:
            gap_key = "re_action_gap_vs_seed_branch"
        best = max(certified, key=lambda c: float(c[gap_key]))
        re_c = float(best["re_phi_m"])
        gap = float(best[gap_key])
        seed = _interp_seed(seed_by_g, g)
        comp_full = float(best.get("full_conv2_exponent", conv2_full_exponent(re_c, K, r)))
        out[g] = {
            "signed_re_gap": gap,
            "full_conv2_gap": comp_full - seed["full_conv2"],
            "n_certified": len(certified),
        }
    return out


def _algebraic_max_from_dominance(dominance_path: Path, *, K: int, r: float) -> dict[float, dict[str, float]]:
    data = _load_json(dominance_path)
    out: dict[float, dict[str, float]] = {}
    for row in data.get("per_gamma", []):
        g = float(row["gamma"])
        gap = float(row["max_signed_gap_vs_seed"])
        seed_re = float(row["seed_re_phi"])
        re_comp = seed_re + gap
        out[g] = {
            "signed_re_gap": gap,
            "full_conv2_gap": conv2_full_exponent(re_comp, K, r) - conv2_full_exponent(seed_re, K, r),
            "n_certified": int(row.get("num_certified_competitors", 0)),
        }
    return out


def plot_algebraic_branch_gap_v2(
    *,
    dominance_dir: Path,
    continuation_json: Optional[Path] = None,
    branch_json: Optional[Path] = None,
    saddles_json: Optional[Path] = None,
    out_path: Optional[Path] = None,
    anchor_gamma: float = -0.83,
    dpi: int = 170,
) -> Path:
    dom_dir = Path(dominance_dir)
    cont_path = continuation_json or (dom_dir.parent / "seed_branch_continuation.json")
    branch_path = branch_json or (dom_dir / "competitor_branch_from_gamma_-0.83.json")
    saddles_path = saddles_json or (dom_dir.parent / "competitor_saddles_by_gamma.json")
    dom_summary = dom_dir / "competitor_dominance_summary.json"

    meta_dom = _load_json(dom_summary).get("metadata", {}) if dom_summary.is_file() else {}
    meta_br = _load_json(branch_path).get("metadata", {}) if branch_path.is_file() else {}
    K = int(meta_dom.get("K_clause", meta_br.get("K_clause", 8)))
    r = float(meta_dom.get("r", meta_br.get("r", 176.54)))
    q = int(meta_dom.get("q", 3))

    seed_by_g = _seed_series(cont_path)
    track = _branch_track(branch_path, K=K, r=r)

    # Prefer sparse discovery grid (every ~0.05) when available; else 6 diagnostic γ.
    if saddles_path.is_file():
        alg_by_g = _algebraic_max_from_saddles(saddles_path, seed_by_g, K=K, r=r)
        alg_source = "competitor_saddles_by_gamma (max certified at each key)"
    elif dom_summary.is_file():
        alg_by_g = _algebraic_max_from_dominance(dom_summary, K=K, r=r)
        alg_source = "competitor_dominance_summary (6 diagnostic γ)"
    else:
        alg_by_g = {}
        alg_source = "none"

    g_track = np.array([row["gamma"] for row in track])
    gap_re_track = np.array([row["signed_re_gap"] for row in track])
    gap_conv2_track = []
    z_jump_g: list[float] = []
    for row in track:
        seed = _interp_seed(seed_by_g, row["gamma"])
        gap_conv2_track.append(row["full_conv2"] - seed["full_conv2"])
        if row["z_jump"]:
            z_jump_g.append(row["gamma"])
    gap_conv2_track = np.array(gap_conv2_track)

    alg_g = np.array(sorted(alg_by_g.keys(), reverse=True))
    gap_re_alg = np.array([alg_by_g[g]["signed_re_gap"] for g in alg_g])
    gap_conv2_alg = np.array([alg_by_g[g]["full_conv2_gap"] for g in alg_g])
    alg_n = [alg_by_g[g]["n_certified"] for g in alg_g]

    fig, axes = plt.subplots(2, 1, figsize=(11, 8.5), sharex=True, constrained_layout=True)
    fig.suptitle(
        f"Competitor dominance: algebraic discovery vs same tracked branch  v2\n"
        f"$q={q}$, $r={r}$, anchor $\\gamma={anchor_gamma}$  |  {DISCLAIMER}",
        fontsize=9,
    )

    for ax, y_track, y_alg, ylab in zip(
        axes,
        [gap_re_track, gap_conv2_track],
        [gap_re_alg, gap_conv2_alg],
        [
            r"signed $\mathrm{Re}\,\Phi_{\mathrm{comp}} - \mathrm{Re}\,\Phi_{\mathrm{seed}}$",
            r"signed Conv2 exponent gap ($\mathrm{Re}\,\Phi_M + \phi_{\mathrm{pref}}$)",
        ],
    ):
        ax.plot(
            g_track,
            y_track,
            "-",
            color="C2",
            lw=1.4,
            alpha=0.9,
            label=f"same branch tracked from $\\gamma={anchor_gamma}$ ({len(g_track)} steps)",
            zorder=2,
        )
        ax.scatter(
            alg_g,
            y_alg,
            s=55,
            c="C0",
            edgecolors="k",
            linewidths=0.4,
            zorder=4,
            label=f"max algebraic discovery ({alg_source})",
        )
        for gi, nv, yv in zip(alg_g, alg_n, y_alg):
            ax.annotate(
                str(nv),
                (gi, yv),
                textcoords="offset points",
                xytext=(0, 7),
                ha="center",
                fontsize=6,
                color="C0",
            )
        if z_jump_g and ax is axes[0]:
            for gz in z_jump_g:
                ax.axvline(gz, color="C3", ls=":", lw=0.7, alpha=0.35)
            ax.plot([], [], ":", color="C3", label=r"$||\Delta z||_\infty>1$ on branch (sheet risk)")
        ax.axhline(0.0, color="k", lw=0.6, alpha=0.45)
        ax.set_ylabel(ylab)
        ax.legend(fontsize=7, loc="upper left")
        ax.grid(True, alpha=0.25)

    # Shade γ window where tracked branch actually dominates Re Φ (gap > 0.1)
    dom_mask = gap_re_track > 0.1
    if np.any(dom_mask):
        g_dom = g_track[dom_mask]
        axes[0].axvspan(float(np.min(g_dom)), float(np.max(g_dom)), color="C2", alpha=0.08)

    axes[1].set_xlabel(r"$\gamma$")
    _style_gamma_axis_reading_zero_to_negative(axes[1])
    axes[0].text(
        0.98,
        0.02,
        IM_PHI_WARNING[:80] + "…",
        transform=axes[0].transAxes,
        fontsize=6,
        ha="right",
        va="bottom",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3),
    )

    out = out_path or (dom_dir / "gamma_vs_algebraic_and_branch_valid_gap_v2.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    # Sidecar summary for reproducibility
    summary = {
        "plot": str(out.resolve()),
        "anchor_gamma": anchor_gamma,
        "n_branch_steps": len(track),
        "n_algebraic_keys": len(alg_by_g),
        "algebraic_source": alg_source,
        "gamma_track_range": [float(g_track.min()), float(g_track.max())] if len(g_track) else None,
        "re_gap_track_at_diagnostic": {
            str(g): float(
                track[
                    min(range(len(track)), key=lambda i: abs(track[i]["gamma"] - g))
                ]["signed_re_gap"]
            )
            for g in (-0.3, -0.6, -0.83, -1.0, -1.6, -2.0)
            if track
        },
        "note": (
            "At γ=-0.3,-0.6 the tracked branch matches seed (gap≈0) while algebraic max remains "
            "large — dominance is sheet-local, not global."
        ),
    }
    sidecar = out.with_suffix(".summary.json")
    sidecar.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Plot algebraic vs same-branch gap v2.")
    p.add_argument("--dominance-dir", type=str, default=str(DEFAULT_DOMINANCE_DIR))
    p.add_argument("--continuation-json", type=str, default="")
    p.add_argument("--branch-json", type=str, default="")
    p.add_argument("--saddles-json", type=str, default="")
    p.add_argument("--anchor-gamma", type=float, default=-0.83)
    p.add_argument("--out", type=str, default="")
    args = p.parse_args()

    out = plot_algebraic_branch_gap_v2(
        dominance_dir=Path(args.dominance_dir),
        continuation_json=Path(args.continuation_json) if args.continuation_json else None,
        branch_json=Path(args.branch_json) if args.branch_json else None,
        saddles_json=Path(args.saddles_json) if args.saddles_json else None,
        out_path=Path(args.out) if args.out else None,
        anchor_gamma=args.anchor_gamma,
    )
    print(f"Wrote {out}")
    print(f"Wrote {out.with_suffix('.summary.json')}")


if __name__ == "__main__":
    main()
