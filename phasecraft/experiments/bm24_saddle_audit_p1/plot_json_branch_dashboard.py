"""
Multi-panel dashboard from BM24 z-saddle competitor / branch-classification JSON.

Reads (by default) the run_seed_branch_g-2pi competitor_dominance bundle:
  - branch_valid_dominance_summary.json
  - branch_classified_competitors.json
  - competitor_dominance_summary.json
  - seed_branch_continuation.json (parent directory)

Optional:
  - run_seed_competitors/competitor_scan_summary.json

Does NOT claim PL dominance; labels match JSON disclaimers.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from phasecraft.bm24_saddle_audit_p1.continue_seed_branch_certified import (
    _style_gamma_axis_reading_zero_to_negative,
)

DISCLAIMER = (
    "Algebraic Re Φ gaps ≠ physical dominance. Branch-valid = local γ-continuation stability only. "
    "Not PL / intersection / global uniqueness."
)

CLASS_COLORS = {
    "stable_continuous_branch": "#2ca02c",
    "collapses_to_seed": "#ff7f0e",
    "failed_or_unstable": "#d62728",
    "log_sheet_artifact": "#9467bd",
}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _continuation_series(cont_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = _load_json(cont_path)
    rows = [r for r in data.get("continuation", []) if r.get("certified") and not r.get("failed")]
    if not rows:
        raise ValueError(f"no certified continuation rows in {cont_path}")
    rows = sorted(rows, key=lambda r: float(r["gamma"]), reverse=True)
    g = np.array([float(r["gamma"]) for r in rows])
    seed_exp = np.array([float(r["full_conv2_exponent"]) for r in rows])
    exact = np.array([float(r["lambda_abs_n_max"]) for r in rows])
    return g, seed_exp, exact


def _anchors_by_gamma(classified: dict[str, Any]) -> dict[float, list[dict]]:
    out: dict[float, list[dict]] = defaultdict(list)
    for a in classified.get("anchors", []):
        out[float(a["anchor_gamma"])].append(a)
    return dict(out)


def _competitor_arrays(per_gamma_row: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    comps = [
        c
        for c in per_gamma_row.get("competitors", [])
        if c.get("certified") and not c.get("is_seed_branch")
    ]
    if not comps:
        return np.array([]), np.array([]), np.array([])
    gap = np.array([float(c["signed_re_phi_gap_vs_seed"]) for c in comps])
    zd = np.array([float(c["z_distance_from_seed"]) for c in comps])
    re_phi = np.array([float(c["re_phi_m"]) for c in comps])
    return gap, zd, re_phi


def plot_dashboard(
    dominance_dir: Path,
    *,
    continuation_json: Optional[Path] = None,
    scan_summary_json: Optional[Path] = None,
    out_path: Optional[Path] = None,
    highlight_gamma: float = -0.83,
    dpi: int = 160,
) -> Path:
    dom_dir = Path(dominance_dir)
    bv_path = dom_dir / "branch_valid_dominance_summary.json"
    cl_path = dom_dir / "branch_classified_competitors.json"
    dom_path = dom_dir / "competitor_dominance_summary.json"

    for p in (bv_path, cl_path, dom_path):
        if not p.is_file():
            raise FileNotFoundError(p)

    bv = _load_json(bv_path)
    classified = _load_json(cl_path)
    dominance = _load_json(dom_path)

    cont_path = continuation_json or (dom_dir.parent / "seed_branch_continuation.json")
    per_gamma_bv = sorted(bv["per_gamma"], key=lambda r: float(r["gamma"]), reverse=True)
    per_gamma_dom = {float(r["gamma"]): r for r in dominance["per_gamma"]}
    anchors_by_g = _anchors_by_gamma(classified)
    meta = bv.get("metadata", {})
    r_val = meta.get("r") or dominance.get("metadata", {}).get("r", "?")
    q_val = dominance.get("metadata", {}).get("q", 3)

    g_cont, seed_exp, exact_exp = _continuation_series(cont_path)

    fig, axes = plt.subplots(2, 3, figsize=(16, 9.5), constrained_layout=True)
    fig.suptitle(
        f"BM24 z-saddle branch diagnostic  $q={q_val}$, $r={r_val}$\n{DISCLAIMER}",
        fontsize=10,
    )

    # --- (0,0) Algebraic vs branch-valid max Re Φ gap ---
    ax = axes[0, 0]
    g = [float(r["gamma"]) for r in per_gamma_bv]
    alg = [float(r["max_algebraic_signed_re_gap_vs_seed"]) for r in per_gamma_bv]
    valid = [
        float(r["max_branch_valid_signed_re_gap_vs_seed"])
        if r["max_branch_valid_signed_re_gap_vs_seed"] is not None
        else np.nan
        for r in per_gamma_bv
    ]
    ax.plot(g, alg, "o-", color="C0", lw=1.5, ms=6, label="max algebraic (all certified)")
    valid_arr = np.asarray(valid, dtype=float)
    mask = np.isfinite(valid_arr)
    if np.any(mask):
        ax.plot(
            np.asarray(g)[mask],
            valid_arr[mask],
            "s",
            color="C2",
            ms=8,
            label="max branch-valid (probed anchors only)",
        )
    ax.axhline(0.0, color="k", lw=0.6, alpha=0.4)
    ax.set_ylabel(r"signed $\mathrm{Re}\,\Phi_{\mathrm{comp}} - \mathrm{Re}\,\Phi_{\mathrm{seed}}$")
    ax.set_title("Algebraic vs branch-valid gap")
    ax.legend(fontsize=7, loc="best")
    ax.grid(True, alpha=0.25)
    ax.invert_xaxis()

    # --- (0,1) Anchor classification counts per γ ---
    ax = axes[0, 1]
    labels_order = [
        "stable_continuous_branch",
        "collapses_to_seed",
        "failed_or_unstable",
        "log_sheet_artifact",
    ]
    bottom = np.zeros(len(g))
    width = 0.07
    for lab in labels_order:
        counts = []
        for gi in g:
            n = sum(1 for a in anchors_by_g.get(gi, []) if a["classification"] == lab)
            counts.append(n)
        if max(counts) == 0:
            continue
        ax.bar(
            g,
            counts,
            width=width,
            bottom=bottom,
            label=lab.replace("_", " "),
            color=CLASS_COLORS.get(lab, "gray"),
            alpha=0.9,
        )
        bottom = bottom + np.array(counts, dtype=float)
    ax.set_ylabel("# probed anchors")
    ax.set_title("Local continuation classification (anchors)")
    ax.legend(fontsize=6, loc="upper left")
    ax.grid(True, alpha=0.25, axis="y")
    ax.invert_xaxis()

    # --- (0,2) Scatter at highlight γ: gap vs z-distance ---
    ax = axes[0, 2]
    row = per_gamma_dom.get(highlight_gamma)
    if row is None:
        ax.text(0.5, 0.5, f"no data at γ={highlight_gamma}", ha="center", va="center", transform=ax.transAxes)
    else:
        gap, zd, _ = _competitor_arrays(row)
        ax.scatter(zd, gap, s=12, alpha=0.35, c="C0", label=f"all certified ({len(gap)})")
        bv_anchors = [
            a
            for a in anchors_by_g.get(highlight_gamma, [])
            if a.get("branch_valid")
        ]
        if bv_anchors:
            ax.scatter(
                [a["z_distance_from_seed"] for a in bv_anchors],
                [a["signed_re_phi_gap_vs_seed"] for a in bv_anchors],
                s=80,
                facecolors="none",
                edgecolors="C2",
                linewidths=1.5,
                label=f"branch-valid anchors ({len(bv_anchors)})",
            )
        ax.axhline(0.0, color="k", lw=0.5, alpha=0.5)
        ax.set_xlabel(r"$\|z_{\mathrm{comp}} - z_{\mathrm{seed}}\|_\infty$")
        ax.set_ylabel(r"signed Re $\Phi$ gap vs seed")
        ax.set_title(f"Discovered competitors at $\\gamma={highlight_gamma}$")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.25)

    # --- (1,0) Seed branch exponent vs exact finite-n ---
    ax = axes[1, 0]
    step = max(1, len(g_cont) // 400)
    ax.plot(
        g_cont[::step],
        seed_exp[::step],
        ".-",
        ms=2,
        lw=0.8,
        color="C0",
        alpha=0.85,
        label="seed Conv2 exponent",
    )
    ax.plot(
        g_cont[::step],
        exact_exp[::step],
        ".--",
        ms=2,
        lw=0.8,
        color="C1",
        alpha=0.85,
        label=r"$\lambda_{\mathrm{abs}}(n_{\max})$",
    )
    for gi in g:
        ax.axvline(gi, color="gray", ls=":", lw=0.6, alpha=0.5)
    ax.set_ylabel("exponent (nat. log)")
    ax.set_title("Seed branch track vs exact finite-$n$")
    _style_gamma_axis_reading_zero_to_negative(ax)
    ax.legend(fontsize=7, loc="best")
    ax.grid(True, alpha=0.25)

    # --- (1,1) Box/strip: competitor gap distributions at diagnostic γ ---
    ax = axes[1, 1]
    positions = []
    data = []
    for gi in g:
        row = per_gamma_dom.get(gi)
        if not row:
            continue
        gap, _, _ = _competitor_arrays(row)
        if gap.size:
            positions.append(gi)
            data.append(gap)
    if data:
        bp = ax.boxplot(
            data,
            positions=positions,
            widths=0.06,
            patch_artist=True,
            showfliers=False,
            medianprops=dict(color="black", lw=1.2),
        )
        for patch in bp["boxes"]:
            patch.set_facecolor("C0")
            patch.set_alpha(0.35)
        ax.scatter(positions, [np.max(d) for d in data], c="C3", s=28, zorder=3, label="max algebraic")
        bv_g = [float(r["gamma"]) for r in per_gamma_bv]
        bv_max = [
            float(r["max_branch_valid_signed_re_gap_vs_seed"])
            if r.get("max_branch_valid_signed_re_gap_vs_seed") is not None
            else np.nan
            for r in per_gamma_bv
        ]
        ax.scatter(
            bv_g,
            bv_max,
            marker="s",
            c="C2",
            s=40,
            zorder=4,
            label="max branch-valid",
        )
    ax.axhline(0.0, color="k", lw=0.5, alpha=0.4)
    ax.set_ylabel(r"signed Re $\Phi$ gap")
    ax.set_title("Distribution of competitor gaps (certified)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.25, axis="y")
    ax.invert_xaxis()

    # --- (1,2) Counts: discovered vs branch-valid + finite-n scan if present ---
    ax = axes[1, 2]
    n_disc = [int(r["num_discovered_certified_competitors"]) for r in per_gamma_bv]
    n_bv = [int(r["num_branch_valid_competitors"]) for r in per_gamma_bv]
    x = np.arange(len(g))
    w = 0.35
    ax.bar(x - w / 2, n_disc, w, label="certified competitors", color="C0", alpha=0.7)
    ax.bar(x + w / 2, n_bv, w, label="branch-valid", color="C2", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{gi:g}" for gi in g], rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("count")
    ax.set_title("Discovery vs branch-valid filter")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.25, axis="y")

    lines = []
    if scan_summary_json and Path(scan_summary_json).is_file():
        scan = _load_json(Path(scan_summary_json))
        lines.append("Finite-$n$ scan: see companion plot")
    for r in per_gamma_bv:
        if not r.get("seed_has_largest_algebraic_re_phi"):
            lines.append(
                f"γ={r['gamma']}: seed NOT max Re Φ "
                f"({r['num_discovered_certified_competitors']} alg. competitors)"
            )
    if lines:
        ax.text(
            0.02,
            0.98,
            "\n".join(lines[:4]),
            transform=ax.transAxes,
            fontsize=6.5,
            va="top",
            ha="left",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3),
        )

    out = out_path or (dom_dir / "branch_diagnostic_dashboard.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_collapse_example(
    classified_path: Path,
    *,
    out_path: Optional[Path] = None,
    gamma: float = -0.3,
    anchor_id_prefix: str = "g-0.3_a0",
) -> Optional[Path]:
    """One-panel: local γ probes showing collapse to seed (branch artifact)."""
    data = _load_json(classified_path)
    anchor = next(
        (a for a in data["anchors"] if a["anchor_id"].startswith(anchor_id_prefix)),
        None,
    )
    if anchor is None:
        return None
    probes = sorted(anchor["probes"], key=lambda p: float(p["delta_gamma"]))
    dg = [float(p["delta_gamma"]) for p in probes]
    re_gap = [float(p["signed_re_phi_gap_vs_seed"]) for p in probes]
    collapsed = [bool(p.get("collapsed_to_seed")) for p in probes]

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ["C2" if c else "C0" for c in collapsed]
    ax.scatter(dg, re_gap, c=colors, s=60, zorder=3)
    for p, c in zip(probes, collapsed):
        if c:
            ax.annotate("→ seed", (p["delta_gamma"], p["signed_re_phi_gap_vs_seed"]), fontsize=7, alpha=0.8)
    ax.axhline(0.0, color="k", lw=0.6)
    ax.set_xlabel(r"$\Delta\gamma$ from anchor")
    ax.set_ylabel(r"signed Re $\Phi$ gap vs seed")
    ax.set_title(
        f"Anchor {anchor['anchor_id']}: {anchor['classification']}\n"
        f"(high Re at anchor; many probes collapse to seed sheet)"
    )
    ax.grid(True, alpha=0.25)
    out = out_path or (classified_path.parent / "collapse_probe_example.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_finite_n_comparison(scan_path: Path, out_path: Optional[Path] = None) -> Path:
    """γ scan: seed vs best-match competitor vs max-algebraic exponent vs λ_abs."""
    scan = _load_json(scan_path)
    rows = sorted(scan["per_gamma"], key=lambda r: float(r["gamma"]), reverse=True)
    g = np.array([float(r["gamma"]) for r in rows])
    lam = np.array([float(r["lambda_abs_n_max"]) for r in rows])
    seed = np.array([float(r["seed_full_conv2_exponent"]) for r in rows])
    best = []
    max_alg = []
    for r in rows:
        bm = r.get("best_match") or {}
        ma = r.get("max_algebraic_competitor") or {}
        best.append(float(bm.get("full_conv2_exponent", np.nan)))
        max_alg.append(float(ma.get("full_conv2_exponent", np.nan)))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(g, lam, "k--", lw=1.2, label=r"$\lambda_{\mathrm{abs}}(n_{\max})$")
    ax.plot(g, seed, "o-", ms=3, label="seed branch exponent")
    ax.plot(g, best, "s-", ms=3, label="best match to $\\lambda$ (discovered)")
    ax.plot(g, max_alg, "^-", ms=3, alpha=0.7, label="max algebraic competitor")
    ax.set_ylabel("exponent (nat. log)")
    ax.set_title("Exact finite-$n$ vs seed vs competitors (discovered only)")
    _style_gamma_axis_reading_zero_to_negative(ax)
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)
    out = out_path or (scan_path.parent / "gamma_vs_exponents_scan_from_json.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Plot BM24 branch diagnostics from JSON artifacts.")
    p.add_argument(
        "--dominance-dir",
        type=str,
        default=str(
            REPO_ROOT
            / "phasecraft/experiments/bm24_saddle_audit_p1/results/run_seed_branch_g-2pi/competitor_dominance"
        ),
    )
    p.add_argument("--continuation-json", type=str, default="")
    p.add_argument(
        "--scan-summary",
        type=str,
        default=str(REPO_ROOT / "phasecraft/results/bm24_saddle_audit_p1/run_seed_competitors/competitor_scan_summary.json"),
    )
    p.add_argument("--highlight-gamma", type=float, default=-0.83)
    p.add_argument("--out-dashboard", type=str, default="")
    p.add_argument("--no-collapse-panel", action="store_true")
    args = p.parse_args()

    dom_dir = Path(args.dominance_dir)
    cont = Path(args.continuation_json) if args.continuation_json else None
    scan = Path(args.scan_summary) if args.scan_summary else None

    dash_out = plot_dashboard(
        dom_dir,
        continuation_json=cont,
        scan_summary_json=scan if scan and scan.is_file() else None,
        out_path=Path(args.out_dashboard) if args.out_dashboard else None,
        highlight_gamma=args.highlight_gamma,
    )
    print(f"Wrote {dash_out}")

    if not args.no_collapse_panel:
        col = plot_collapse_example(dom_dir / "branch_classified_competitors.json")
        if col:
            print(f"Wrote {col}")

    if scan and scan.is_file():
        fin = plot_finite_n_comparison(scan)
        print(f"Wrote {fin}")


if __name__ == "__main__":
    main()
