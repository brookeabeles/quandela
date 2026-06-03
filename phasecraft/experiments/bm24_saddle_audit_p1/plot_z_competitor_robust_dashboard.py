"""
2×3 competitor robust dashboard for BM24 z-saddle (current audit parameters).

Mirrors ``phasecraft.w_saddle.workflow.plot_competitor_analysis`` layout using
existing JSON artifacts (no full z resolve pipeline required):

  - Seed sheet: ``seed_branch_continuation.json``
  - Main competitor sheet: ``competitor_branch_from_gamma_-0.83.json``
  - Extra sheets: local probe tracks from ``branch_classified_competitors.json``
    (stable_continuous_branch anchors only)
  - Sweep stats: ``competitor_saddles_by_gamma.json`` on the seed γ mesh
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from phasecraft.bm24_saddle_audit_p1.continue_seed_branch_certified import (
    IM_PHI_WARNING,
    _style_gamma_axis_reading_zero_to_negative,
)
from phasecraft.w_saddle.workflow import (
    LOSS_COLOR,
    NONE_COLOR,
    SEED_PLOT_COLOR,
    WIN_COLOR,
    _branch_color_map,
    _plot_branch_curves_on_ax,
    _plot_sweep_counts_on_ax,
    _plot_sweep_gap_scatter_on_ax,
    _plot_sweep_summary_bar_on_ax,
)

DEFAULT_RUN_DIR = (
    REPO_ROOT
    / "phasecraft/experiments/bm24_saddle_audit_p1/results/run_seed_branch_g-2pi"
)
DISCLAIMER = (
    "z-saddle BM24 diagnostic dashboard. Sheets from certified γ-continuation / local probes. "
    "Not PL dominance; Re Φ_M gaps ≠ Conv2 exponent unless noted."
)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _nearest_key(keys: list[float], gamma: float, tol: float) -> Optional[float]:
    if not keys:
        return None
    best = min(keys, key=lambda k: abs(k - gamma))
    return best if abs(best - gamma) <= tol else None


def build_sweep_payload(
    run_dir: Path,
    *,
    saddles_tol: float = 0.03,
) -> dict[str, Any]:
    """Per-γ sweep stats from competitor_saddles_by_gamma on the seed continuation mesh."""
    cont = _load(run_dir / "seed_branch_continuation.json")
    meta = cont.get("metadata", {})
    saddles_path = run_dir / "competitor_saddles_by_gamma.json"
    saddles_by: dict[float, list[dict]] = {}
    if saddles_path.is_file():
        raw = _load(saddles_path).get("by_gamma", {})
        saddles_by = {float(k): v for k, v in raw.items()}
    saddle_keys = sorted(saddles_by.keys())

    rows = [r for r in cont.get("continuation", []) if r.get("certified") and not r.get("failed")]
    rows = sorted(rows, key=lambda r: float(r["gamma"]), reverse=True)

    points: list[dict[str, Any]] = []
    for row in rows:
        g = float(row["gamma"])
        seed_re = float(row["re_phi_m"])
        key = _nearest_key(saddle_keys, g, saddles_tol)
        comps: list[dict] = []
        if key is not None:
            comps = [
                c
                for c in saddles_by[key]
                if c.get("certified") and c.get("label") == "certified_competitor"
            ]
        gaps = [float(c["re_action_gap_vs_seed_branch"]) for c in comps]
        min_gap = min(gaps) if gaps else None
        best_re = max((float(c["re_phi_m"]) for c in comps), default=seed_re)
        points.append(
            {
                "gamma": g,
                "Phi_eff_real": seed_re,
                "num_certified_roots": len(comps) + 1,
                "num_certified_competitors": len(comps),
                "min_certified_gap_re": min_gap,
                "min_positive_gap_re": min((x for x in gaps if x > 0), default=None),
                "best_competitor_re_phi": best_re if comps else None,
                "seed_dominates_all": min_gap is None or min_gap > 0,
            }
        )

    return {
        "r": float(meta.get("r", 176.54)),
        "q": int(meta.get("q", 3)),
        "points": points,
        "refined_window": _detect_wall_window(points),
    }


def _detect_wall_window(
    points: list[dict[str, Any]],
    *,
    gap_lo: float = 0.5,
    margin: float = 0.12,
    fallback: tuple[float, float] = (-1.05, -0.72),
) -> dict[str, float]:
    """γ interval where sparse discovery shows strong algebraic threat (or BM24 wall band)."""
    flagged = [
        float(p["gamma"])
        for p in points
        if p.get("min_certified_gap_re") is not None and float(p["min_certified_gap_re"]) <= gap_lo
    ]
    if len(flagged) >= 2:
        lo, hi = min(flagged), max(flagged)
        return {"gamma_lo": lo - margin, "gamma_hi": hi + margin, "method": "auto_sparse_saddles"}
    return {
        "gamma_lo": fallback[0],
        "gamma_hi": fallback[1],
        "method": "default_wall_band",
    }


def _point(
    *,
    branch_id: int,
    gamma: float,
    re_phi: float,
    seed_re: float,
    status: str = "competitor_certified_local",
    z_dist_seed: float = 0.0,
) -> dict[str, Any]:
    return {
        "branch_id": branch_id,
        "gamma": gamma,
        "Phi_eff_real": re_phi,
        "Phi_eff_imag": 0.0,
        "status": status,
        "DeltaRe_vs_seed": float(re_phi - seed_re),
        "distance_to_seed_w": z_dist_seed,
        "krawczyk_certified": True,
    }


def build_resolved_payload(
    run_dir: Path,
    dom_dir: Path,
    *,
    anchor_gamma: float = -0.83,
    include_probe_branches: bool = True,
    max_probe_branches: int = 6,
) -> dict[str, Any]:
    cont = _load(run_dir / "seed_branch_continuation.json")
    meta = cont.get("metadata", {})
    r = float(meta.get("r", 176.54))

    seed_rows = [r for r in cont.get("continuation", []) if r.get("certified") and not r.get("failed")]
    seed_by_g = {float(r["gamma"]): r for r in seed_rows}

    branches: list[dict[str, Any]] = []
    bid = 0

    # Seed sheet
    seed_pts = [
        _point(
            branch_id=bid,
            gamma=float(r["gamma"]),
            re_phi=float(r["re_phi_m"]),
            seed_re=float(r["re_phi_m"]),
            status="seed_certified_local",
        )
        for r in sorted(seed_rows, key=lambda x: float(x["gamma"]), reverse=True)
    ]
    branches.append(
        {
            "branch_id": bid,
            "label": "seed sheet",
            "is_seed_sheet": True,
            "points": seed_pts,
        }
    )
    bid += 1

    # Main continued competitor sheet
    branch_path = dom_dir / "competitor_branch_from_gamma_-0.83.json"
    if branch_path.is_file():
        bdata = _load(branch_path)
        steps = [
            s
            for s in bdata.get("steps", [])
            if s.get("certified") and not s.get("failed")
        ]
        comp_pts = []
        for s in sorted(steps, key=lambda x: float(x["gamma"]), reverse=True):
            g = float(s["gamma"])
            seed_re = float(s.get("seed_re_phi_m", seed_by_g.get(g, {}).get("re_phi_m", 0.0)))
            if g not in seed_by_g and seed_rows:
                # interpolate nearest seed re
                gs = sorted(seed_by_g.keys())
                ref = min(gs, key=lambda k: abs(k - g))
                seed_re = float(seed_by_g[ref]["re_phi_m"])
            comp_pts.append(
                _point(
                    branch_id=bid,
                    gamma=g,
                    re_phi=float(s["re_phi_m"]),
                    seed_re=seed_re,
                    z_dist_seed=float(s.get("z_distance_from_seed", 0.0)),
                )
            )
        branches.append(
            {
                "branch_id": bid,
                "label": f"competitor sheet (anchor γ={anchor_gamma})",
                "is_seed_sheet": False,
                "points": comp_pts,
            }
        )
        bid += 1

    # Local probe tracks from branch classification
    if include_probe_branches:
        cl_path = dom_dir / "branch_classified_competitors.json"
        if cl_path.is_file():
            anchors = [
                a
                for a in _load(cl_path).get("anchors", [])
                if a.get("branch_valid") and a.get("classification") == "stable_continuous_branch"
            ]
            # Prefer highest gap anchors; cap count for readability
            anchors = sorted(
                anchors,
                key=lambda a: float(a.get("signed_re_phi_gap_vs_seed", 0.0)),
                reverse=True,
            )[:max_probe_branches]
            for anc in anchors:
                probe_pts = []
                for pr in sorted(anc.get("probes", []), key=lambda p: float(p["gamma"]), reverse=True):
                    if not pr.get("certified"):
                        continue
                    g = float(pr["gamma"])
                    seed_re = float(pr.get("seed_re_phi_m", 0.0))
                    probe_pts.append(
                        _point(
                            branch_id=bid,
                            gamma=g,
                            re_phi=float(pr["re_phi_m"]),
                            seed_re=seed_re,
                            z_dist_seed=float(pr.get("z_distance_from_seed", 0.0)),
                        )
                    )
                if len(probe_pts) < 2:
                    continue
                branches.append(
                    {
                        "branch_id": bid,
                        "label": f"probe sheet {anc['anchor_id']}",
                        "is_seed_sheet": False,
                        "points": probe_pts,
                    }
                )
                bid += 1

    crossings = _detect_crossings(branches)
    return {
        "r": r,
        "q": int(meta.get("q", 3)),
        "branches": branches,
        "crossing_certificates": crossings,
        "disclaimer": DISCLAIMER,
    }


def _detect_crossings(branches: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Bracket γ intervals where a competitor sheet's ΔRe vs seed changes sign."""
    seed_br = next((b for b in branches if b.get("is_seed_sheet")), None)
    if seed_br is None:
        return []
    out: list[dict[str, Any]] = []
    for br in branches:
        if br.get("is_seed_sheet"):
            continue
        bid = int(br["branch_id"])
        pts = sorted(br["points"], key=lambda p: float(p["gamma"]), reverse=True)
        for i in range(len(pts) - 1):
            g0, g1 = float(pts[i]["gamma"]), float(pts[i + 1]["gamma"])
            d0 = float(pts[i]["DeltaRe_vs_seed"])
            d1 = float(pts[i + 1]["DeltaRe_vs_seed"])
            if d0 * d1 < 0:
                out.append(
                    {
                        "crossing_branch_id": bid,
                        "gamma_interval": [g0, g1],
                        "delta_re_endpoints": [d0, d1],
                        "suppressed_duplicate_same_sheet": False,
                    }
                )
    return out


def _plot_crossing_brackets_z(ax, resolved: dict[str, Any], br_colors: dict[int, Any]) -> int:
    n = 0
    for iv in resolved.get("crossing_certificates") or []:
        g0, g1 = iv["gamma_interval"]
        bid = int(iv["crossing_branch_id"])
        color = br_colors.get(bid, LOSS_COLOR)
        ax.axvspan(g1, g0, alpha=0.25, color=color, label=rf"branch {bid} crossing")
        n += 1
    ax.axhline(0.0, color="gray", ls="--", lw=0.6, alpha=0.5)
    ax.set_xlabel(r"$\gamma$")
    ax.set_ylabel("(interval)")
    ax.set_title(r"$\Delta\mathrm{Re}=0$ brackets (probe/continuation mesh)", fontsize=9)
    if n:
        ax.legend(fontsize=6)
    ax.grid(True, alpha=0.25)
    return n


def plot_z_competitor_robust_dashboard(
    run_dir: Path,
    *,
    out_path: Optional[Path] = None,
    anchor_gamma: float = -0.83,
    dpi: int = 160,
) -> Path:
    run_dir = Path(run_dir)
    dom_dir = run_dir / "competitor_dominance"
    sweep = build_sweep_payload(run_dir)
    resolved = build_resolved_payload(run_dir, dom_dir, anchor_gamma=anchor_gamma)

    pts = sorted(sweep["points"], key=lambda p: float(p["gamma"]), reverse=True)
    gamma = np.array([float(p["gamma"]) for p in pts])
    min_gap = np.array(
        [float(p["min_certified_gap_re"]) if p.get("min_certified_gap_re") is not None else np.nan for p in pts]
    )
    n_comp = np.array([int(p.get("num_certified_competitors", 0) or 0) for p in pts])
    n_cert = np.array([int(p.get("num_certified_roots", 0) or 0) for p in pts])

    branches = resolved["branches"]
    br_colors = _branch_color_map(branches)
    n_comp_sheets = sum(1 for b in branches if not b.get("is_seed_sheet"))
    n_cross = len(resolved.get("crossing_certificates") or [])

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), constrained_layout=True)

    ax = axes[0, 0]
    _plot_branch_curves_on_ax(
        ax, branches, br_colors, y_key="DeltaRe_vs_seed", skip_seed=True, legend=True
    )
    ax.axhline(0.0, color="gray", ls="--", lw=0.8)
    ax.set_xlabel(r"$\gamma$")
    ax.set_ylabel(r"$\Delta\mathrm{Re}(\gamma)$")
    ax.set_title(r"Tracked sheets: $\mathrm{Re}\,\Phi_M^{\mathrm{comp}}-\mathrm{Re}\,\Phi_M^{\mathrm{seed}}$", fontsize=9)
    ax.grid(True, alpha=0.25)
    _style_gamma_axis_reading_zero_to_negative(ax)

    ax = axes[0, 1]
    _plot_branch_curves_on_ax(
        ax, branches, br_colors, y_key="Phi_eff_real", skip_seed=False, legend=True, label_max=12
    )
    ax.set_xlabel(r"$\gamma$")
    ax.set_ylabel(r"$\mathrm{Re}\,\Phi_M$")
    ax.set_title(r"$\mathrm{Re}\,\Phi_M$ on seed + competitor sheets", fontsize=9)
    ax.grid(True, alpha=0.25)
    _style_gamma_axis_reading_zero_to_negative(ax)

    ax = axes[0, 2]
    _plot_crossing_brackets_z(ax, resolved, br_colors)
    _style_gamma_axis_reading_zero_to_negative(ax)

    ax = axes[1, 0]
    ref = sweep.get("refined_window")
    if ref:
        lo, hi = float(ref["gamma_lo"]), float(ref["gamma_hi"])
        if lo > hi:
            lo, hi = hi, lo
        any_line = False
        for br in branches:
            if br.get("is_seed_sheet"):
                continue
            bid = int(br["branch_id"])
            pts = sorted(br["points"], key=lambda p: float(p["gamma"]), reverse=True)
            g = [float(p["gamma"]) for p in pts if lo <= float(p["gamma"]) <= hi]
            y = [float(p["DeltaRe_vs_seed"]) for p in pts if lo <= float(p["gamma"]) <= hi]
            if g:
                any_line = True
                ax.plot(
                    g,
                    y,
                    "o-",
                    ms=3,
                    lw=1.2,
                    color=br_colors[bid],
                    label=br.get("label", f"branch {bid}")[:40],
                )
        ax.axhline(0.0, color="gray", ls="--", lw=0.8)
        ax.set_title(rf"Wall zoom: $\gamma\in[{lo:.3g},{hi:.3g}]$", fontsize=9)
        ax.set_ylabel(r"$\Delta\mathrm{Re}$")
        if any_line:
            ax.legend(fontsize=6)
    else:
        _plot_sweep_gap_scatter_on_ax(ax, gamma, min_gap, wall_only=False)
    ax.set_xlabel(r"$\gamma$")
    ax.grid(True, alpha=0.25)
    _style_gamma_axis_reading_zero_to_negative(ax)

    ax = axes[1, 1]
    _plot_sweep_counts_on_ax(ax, gamma, n_comp, n_cert)
    _style_gamma_axis_reading_zero_to_negative(ax)

    ax = axes[1, 2]
    _plot_sweep_summary_bar_on_ax(ax, pts)

    r = sweep.get("r", 176.54)
    ref = sweep.get("refined_window")
    title = f"BM24 z-saddle competitor analysis  r={r}  N={len(pts)}"
    if ref:
        title += f"  wall [{ref.get('gamma_lo'):.3g}, {ref.get('gamma_hi'):.3g}]"
    subtitle = (
        f"Top: {n_comp_sheets} competitor sheet(s), {n_cross} ΔRe sign-change bracket(s). "
        f"Bottom: sparse discovery counts (competitor_saddles_by_gamma). {DISCLAIMER}"
    )
    fig.suptitle(title + "\n" + subtitle, fontsize=10)

    out = out_path or (dom_dir / "competitors_robust_analysis_z.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    # Persist payloads for replot
    (dom_dir / "sweep_payload_z.json").write_text(json.dumps(sweep, indent=2), encoding="utf-8")
    (dom_dir / "resolved_payload_z.json").write_text(json.dumps(resolved, indent=2), encoding="utf-8")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="BM24 z-saddle 2x3 competitor robust dashboard.")
    p.add_argument("--run-dir", type=str, default=str(DEFAULT_RUN_DIR))
    p.add_argument("--anchor-gamma", type=float, default=-0.83)
    p.add_argument("--out", type=str, default="")
    args = p.parse_args()
    out = plot_z_competitor_robust_dashboard(
        Path(args.run_dir),
        out_path=Path(args.out) if args.out else None,
        anchor_gamma=args.anchor_gamma,
    )
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
