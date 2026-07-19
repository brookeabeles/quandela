"""
Readable Re(Phi_M) vs gamma plots from z_robust_full (resolved + sweep JSON).

Focus: tracked sheets (seed + top disjoint competitor branches), not 251-line spaghetti.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from phasecraft.bm24_saddle_audit_p1.continue_seed_branch_certified import (
    GAMMA_AXIS_LEFT,
    GAMMA_AXIS_RIGHT,
    _style_gamma_axis_reading_zero_to_negative,
)

DEFAULT_DIR = (
    REPO_ROOT
    / "phasecraft/experiments/bm24_saddle_audit_p1/results/run_seed_branch_g-2pi/competitor_dominance/z_robust_full"
)

# BM24 seed-branch Re Phi_M stays roughly here; sweep also certifies distant log sheets.
DEFAULT_RE_PHYS_LO = -2.0
DEFAULT_RE_PHYS_HI = 1.0
BRANCH_MATCH_STEP_TOL = 0.35
GAMMA_MATCH_TOL = 0.006

STATUS_LOCAL = "competitor_certified_local"
STATUS_PL = "competitor_pl_active"
STATUS_DUP = "seed_duplicate"
STATUS_SEED = "seed_certified_local"

# Matches thesis-ready figures (gamma_vs_exponents-full_gamma, etc.).
_THESIS_PAPER_RCPARAMS = {
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "axes.unicode_minus": False,
    "font.size": 14,
    "axes.labelsize": 16,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
}

# Highlight colours for key branches in tracked v2 plot.
V2_BRANCH_COLORS: dict[int, str] = {
    43: "#2ca02c",
    46: "#ff7f0e",
    2: "#d62728",
}

GAMMA_ZERO_TARGET = -0.01
RE_SAME_SHEET_TOL = 0.35
IM_OPPOSITE_TOL = 0.5


@dataclass
class TrackPoint:
    gamma: float
    re: float
    im: float
    z: np.ndarray
    source: str  # resolved | sweep | extension


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _re_in_physical_band(re: float, *, re_lo: float, re_hi: float) -> bool:
    return re_lo <= float(re) <= re_hi


def _sweep_competitor_records(sw: dict[str, Any]) -> list[dict[str, Any]]:
    """Competitors only (seed sheet excluded), falling back to non-seed roots."""
    comps = list(sw.get("competitors") or [])
    if comps:
        return comps
    return [r for r in sw.get("certified_roots") or [] if not r.get("is_seed_branch")]


def _z_from_sweep_rec(rec: dict[str, Any]) -> np.ndarray:
    if "z_real" in rec:
        return np.asarray(rec["z_real"], dtype=float) + 1j * np.asarray(rec["z_imag"], dtype=float)
    return np.asarray(rec["w_real"], dtype=float) + 1j * np.asarray(rec["w_imag"], dtype=float)


def _branch_points_by_gamma(
    resolved: dict[str, Any],
) -> dict[float, list[tuple[int, np.ndarray, float, bool]]]:
    """Map γ → list of (branch_id, z, Re Φ_M, is_seed_sheet) from resolve step."""
    by_g: dict[float, list[tuple[int, np.ndarray, float, bool]]] = {}
    for br in resolved.get("branches", []):
        bid = int(br["branch_id"])
        is_seed = bool(br.get("is_seed_sheet"))
        for p in br.get("points", []):
            g = round(float(p["gamma"]), 6)
            z = np.asarray(p["w_real"], dtype=float) + 1j * np.asarray(p["w_imag"], dtype=float)
            by_g.setdefault(g, []).append((bid, z, float(p["Phi_eff_real"]), is_seed))
    return by_g


def _match_sweep_to_branch(
    gamma: float,
    z: np.ndarray,
    by_gamma: dict[float, list[tuple[int, np.ndarray, float, bool]]],
    *,
    step_tol: float = BRANCH_MATCH_STEP_TOL,
) -> Optional[int]:
    """Match sweep root to resolved track (same rule as track_branches_z proximity)."""
    candidates: list[tuple[int, np.ndarray, float, bool]] = []
    for g_key, entries in by_gamma.items():
        if abs(g_key - gamma) <= GAMMA_MATCH_TOL:
            candidates.extend(entries)
    if not candidates:
        return None
    z = np.asarray(z, dtype=complex)
    best_id: Optional[int] = None
    best_d = float("inf")
    for bid, z_b, _re_b, _is_seed in candidates:
        d = float(np.linalg.norm(z - z_b, ord=np.inf))
        scale = max(float(np.max(np.abs(z))), float(np.max(np.abs(z_b))), 1e-3) * max(step_tol, 0.05)
        if d <= scale and d < best_d:
            best_d = d
            best_id = bid
    return best_id


def _filter_sweep_records(
    records: list[dict[str, Any]],
    *,
    re_lo: float,
    re_hi: float,
) -> tuple[list[dict[str, Any]], int]:
    kept = [r for r in records if _re_in_physical_band(r["Phi_eff_real"], re_lo=re_lo, re_hi=re_hi)]
    return kept, len(records) - len(kept)


def _valid_sheet_points(
    branch: dict[str, Any],
    *,
    re_lo: float = DEFAULT_RE_PHYS_LO,
    re_hi: float = DEFAULT_RE_PHYS_HI,
) -> list[dict[str, Any]]:
    """Points on a sheet that are Krawczyk-disjoint from seed (real competitors)."""
    if branch.get("is_seed_sheet"):
        pts = list(branch.get("points", []))
    else:
        pts = [
            p
            for p in branch.get("points", [])
            if p.get("box_disjoint_from_seed")
            and p.get("status") not in (STATUS_DUP, STATUS_SEED)
        ]
    return [p for p in pts if _re_in_physical_band(p["Phi_eff_real"], re_lo=re_lo, re_hi=re_hi)]


def _branch_score(branch: dict[str, Any]) -> tuple[float, int, float]:
    pts = _valid_sheet_points(branch)
    if not pts:
        return (-1e9, 0, 0.0)
    re_vals = [float(p["Phi_eff_real"]) for p in pts]
    return (max(re_vals), len(pts), float(np.mean(re_vals)))


def select_branches(
    resolved: dict[str, Any],
    *,
    top_n: int = 10,
    min_valid_pts: int = 10,
) -> list[dict[str, Any]]:
    branches = resolved.get("branches", [])
    seed = [b for b in branches if b.get("is_seed_sheet")]
    if not seed:
        raise ValueError("no seed sheet in resolved_robust_z.json")
    seed_br = seed[0]

    ranked: list[tuple[float, int, dict[str, Any]]] = []
    for b in branches:
        if b.get("is_seed_sheet"):
            continue
        pts = _valid_sheet_points(b)
        if len(pts) < min_valid_pts:
            continue
        re_max, n, _ = _branch_score(b)
        ranked.append((re_max, n, b))
    ranked.sort(key=lambda t: (t[0], t[1]), reverse=True)

    out = [seed_br]
    seen: set[int] = {int(seed_br["branch_id"])}
    for _re, _n, b in ranked:
        bid = int(b["branch_id"])
        if bid in seen:
            continue
        out.append(b)
        seen.add(bid)
        if len(out) > top_n:
            break
    return out


def _series_from_points(
    points: list[dict[str, Any]],
    *,
    key: str = "Phi_eff_real",
) -> tuple[np.ndarray, np.ndarray]:
    pts = sorted(points, key=lambda p: float(p["gamma"]), reverse=True)
    g = np.array([float(p["gamma"]) for p in pts], dtype=float)
    y = np.array([float(p[key]) for p in pts], dtype=float)
    return g, y


def plot_tracked_re_sheets(
    resolved: dict[str, Any],
    out_path: Path,
    *,
    top_n: int = 10,
    gamma_lo: float = GAMMA_AXIS_RIGHT,
    gamma_hi: float = GAMMA_AXIS_LEFT,
    dpi: int = 160,
) -> Path:
    picked = select_branches(resolved, top_n=top_n)
    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)

    ax_re, ax_gap = axes
    cmap = plt.cm.tab10(np.linspace(0, 1, max(len(picked), 1)))

    for i, br in enumerate(picked):
        pts = _valid_sheet_points(br)
        if len(pts) < 2:
            continue
        g, re = _series_from_points(pts, key="Phi_eff_real")
        mask = (g <= gamma_hi + 1e-9) & (g >= gamma_lo - 1e-9)
        g, re = g[mask], re[mask]
        if br.get("is_seed_sheet"):
            ax_re.plot(g, re, color="black", lw=2.8, zorder=10, label="seed sheet")
            ax_gap.plot(g, np.zeros_like(g), color="black", lw=2.8, zorder=10, label="seed (ΔRe=0)")
        else:
            lbl = br.get("label", f"branch {br['branch_id']}")
            ax_re.plot(g, re, color=cmap[i], lw=1.8, label=lbl)
            dg, d_re = _series_from_points(pts, key="DeltaRe_vs_seed")
            dg, d_re = dg[mask], d_re[mask]
            ax_gap.plot(dg, d_re, color=cmap[i], lw=1.8, alpha=0.9)

    ax_re.axhline(0.0, color="gray", lw=0.6, alpha=0.4)
    ax_re.set_ylim(DEFAULT_RE_PHYS_LO - 0.15, DEFAULT_RE_PHYS_HI + 0.15)
    ax_re.set_ylabel(r"Re $\Phi_M$ (certified, per sheet)")
    ax_re.set_title(
        f"Tracked sheets: seed + top {min(top_n, len(picked) - 1)} disjoint competitor branches "
        f"(from {resolved.get('num_branches', '?')} total tracks)"
    )
    ax_re.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=8, framealpha=0.95)
    ax_re.grid(True, alpha=0.25)

    ax_gap.axhline(0.0, color="k", lw=0.8, alpha=0.5)
    ax_gap.set_ylabel(r"$\Delta$Re vs seed  ($\mathrm{Re}\,\Phi_{\mathrm{seed}} - \mathrm{Re}\,\Phi_{\mathrm{sheet}}$)")
    ax_gap.set_title("Dominance gap along each sheet (positive = seed larger on Re axis)")
    ax_gap.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=7, framealpha=0.95)
    ax_gap.grid(True, alpha=0.25)

    _style_gamma_axis_reading_zero_to_negative(ax_gap)
    fig.suptitle(
        "BM24 z-robust branch resolve — sheet-local Re tracking (disjoint boxes only for competitors)",
        fontsize=11,
        y=1.01,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_re_zoom_wall(
    resolved: dict[str, Any],
    out_path: Path,
    *,
    top_n: int = 8,
    gamma_hi: float = -0.5,
    gamma_lo: float = -2.5,
    dpi: int = 160,
) -> Path:
    """Zoom where high-Re competitor sheets separate from seed."""
    picked = select_branches(resolved, top_n=top_n)
    fig, ax = plt.subplots(figsize=(11, 6))
    cmap = plt.cm.tab10(np.linspace(0, 1, max(len(picked), 1)))

    for i, br in enumerate(picked):
        pts = _valid_sheet_points(br)
        g, re = _series_from_points(pts, key="Phi_eff_real")
        mask = (g <= gamma_hi + 1e-9) & (g >= gamma_lo - 1e-9)
        g, re = g[mask], re[mask]
        if len(g) < 2:
            continue
        if br.get("is_seed_sheet"):
            ax.plot(g, re, "k-", lw=3, label="seed sheet", zorder=10)
        else:
            ax.plot(g, re, color=cmap[i], lw=2, marker="o", ms=3, label=br.get("label", ""))

    ax.set_xlim(gamma_hi, gamma_lo)
    ax.set_ylabel(r"Re $\Phi_M$")
    ax.set_xlabel(r"$\gamma$")
    ax.set_title(rf"Wall region zoom: $\gamma \in [{gamma_lo:.2f}, {gamma_hi:.2f}]$")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_sweep_re_envelope(
    sweep: dict[str, Any],
    out_path: Path,
    *,
    re_lo: float = DEFAULT_RE_PHYS_LO,
    re_hi: float = DEFAULT_RE_PHYS_HI,
    dpi: int = 160,
) -> Path:
    """Per γ: seed Re + max competitor Re in physical Re band only."""
    gammas: list[float] = []
    seed_re: list[float] = []
    max_re: list[float] = []
    best_comp_re: list[float] = []
    n_hidden: list[int] = []

    for row in sorted(sweep.get("points", []), key=lambda r: float(r["gamma"]), reverse=True):
        if not row.get("krawczyk_certified", True):
            continue
        sw = row.get("competitor_sweep") or {}
        comps_raw = _sweep_competitor_records(sw)
        comps, n_h = _filter_sweep_records(comps_raw, re_lo=re_lo, re_hi=re_hi)
        if not comps_raw:
            continue
        g = float(row["gamma"])
        s_re = float(row["Phi_eff_real"])
        re_comp = [float(r["Phi_eff_real"]) for r in comps]
        gammas.append(g)
        seed_re.append(s_re)
        max_re.append(max(re_comp) if re_comp else float("nan"))
        best_comp_re.append(max(re_comp) if re_comp else float("nan"))
        n_hidden.append(n_h)

    g = np.array(gammas)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(g, seed_re, "k-", lw=2.5, label="continued seed (row)")
    ax.plot(g, max_re, "r.-", ms=5, lw=1.2, label=f"max competitor Re in [{re_lo}, {re_hi}]")
    ax.fill_between(g, seed_re, max_re, alpha=0.15, color="tomato", label="Re gap band @ γ")
    ax.axhline(0.0, color="gray", lw=0.5)
    ax.set_ylim(re_lo - 0.15, re_hi + 0.15)
    ax.set_ylabel(r"Re $\Phi_M$")
    ax.set_title(
        f"Per-γ sweep: seed vs competitors with {re_lo} ≤ Re Φ_M ≤ {re_hi} "
        f"(hid {sum(n_hidden)} distant-sheet roots)"
    )
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.25)
    _style_gamma_axis_reading_zero_to_negative(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_sweep_re_scatter(
    sweep: dict[str, Any],
    out_path: Path,
    *,
    re_lo: float = DEFAULT_RE_PHYS_LO,
    re_hi: float = DEFAULT_RE_PHYS_HI,
    max_roots_per_gamma: int = 20,
    dpi: int = 140,
) -> Path:
    """Scatter competitor Re in the physical BM24 band (excludes distant log sheets)."""
    xs: list[float] = []
    ys: list[float] = []
    colors: list[float] = []
    n_hidden_total = 0
    n_shown = 0

    for row in sweep.get("points", []):
        if not row.get("krawczyk_certified", True):
            continue
        g = float(row["gamma"])
        sw = row.get("competitor_sweep") or {}
        comps_raw = _sweep_competitor_records(sw)
        comps, n_h = _filter_sweep_records(comps_raw, re_lo=re_lo, re_hi=re_hi)
        n_hidden_total += n_h
        if len(comps) > max_roots_per_gamma:
            comps = sorted(comps, key=lambda r: -float(r["Phi_eff_real"]))[:max_roots_per_gamma]
        seed_re = float(row["Phi_eff_real"])
        for r in comps:
            re = float(r["Phi_eff_real"])
            xs.append(g)
            ys.append(re)
            colors.append(re - seed_re)
            n_shown += 1

    fig, ax = plt.subplots(figsize=(12, 6))
    sc = ax.scatter(xs, ys, c=colors, cmap="RdYlBu_r", s=22, alpha=0.85, vmin=-0.8, vmax=0.8)
    plt.colorbar(sc, ax=ax, label=r"Re(comp) $-$ Re(seed)")
    sg = [float(r["gamma"]) for r in sweep["points"] if r.get("krawczyk_certified")]
    sr = [float(r["Phi_eff_real"]) for r in sweep["points"] if r.get("krawczyk_certified")]
    order = np.argsort(sg)[::-1]
    sg = np.array(sg)[order]
    sr = np.array(sr)[order]
    ax.plot(sg, sr, "k-", lw=2.8, zorder=5, label="seed continuation")
    ax.set_ylim(re_lo - 0.1, re_hi + 0.1)
    ax.set_ylabel(r"Re $\Phi_M$ (competitors in band)")
    ax.set_title(
        f"Competitor sweep: {re_lo} ≤ Re Φ_M ≤ {re_hi}  "
        f"({n_shown} shown, {n_hidden_total} distant-sheet roots omitted)"
    )
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.2)
    _style_gamma_axis_reading_zero_to_negative(ax)
    fig.text(
        0.01,
        0.01,
        "Out-of-band certified roots (other log sheets) are excluded; they dominated the old plot.",
        fontsize=8,
        color="0.35",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_sweep_re_scatter_by_branch_id(
    sweep: dict[str, Any],
    resolved: dict[str, Any],
    out_path: Path,
    *,
    re_lo: float = DEFAULT_RE_PHYS_LO,
    re_hi: float = DEFAULT_RE_PHYS_HI,
    legend_top_n: int = 12,
    dpi: int = 140,
) -> Path:
    """Scatter competitors colored by resolved branch_id; faint polylines show tracks."""
    by_gamma = _branch_points_by_gamma(resolved)
    branches_by_id = {int(b["branch_id"]): b for b in resolved.get("branches", [])}

    xs: list[float] = []
    ys: list[float] = []
    bids: list[int] = []
    n_hidden = 0
    match_counts: dict[int, int] = {}

    for row in sweep.get("points", []):
        if not row.get("krawczyk_certified", True):
            continue
        g = float(row["gamma"])
        sw = row.get("competitor_sweep") or {}
        comps_raw = _sweep_competitor_records(sw)
        comps, n_h = _filter_sweep_records(comps_raw, re_lo=re_lo, re_hi=re_hi)
        n_hidden += n_h
        for r in comps:
            z = _z_from_sweep_rec(r)
            bid = _match_sweep_to_branch(g, z, by_gamma)
            xs.append(g)
            ys.append(float(r["Phi_eff_real"]))
            if bid is None:
                bids.append(-1)
            else:
                bids.append(bid)
                match_counts[bid] = match_counts.get(bid, 0) + 1

    ranked_ids = [bid for bid, _ in sorted(match_counts.items(), key=lambda t: -t[1])]
    seed_id = next(
        (int(b["branch_id"]) for b in resolved.get("branches", []) if b.get("is_seed_sheet")),
        None,
    )
    palette_ids: list[int] = []
    if seed_id is not None and seed_id in match_counts:
        palette_ids.append(seed_id)
    for bid in ranked_ids:
        if bid not in palette_ids:
            palette_ids.append(bid)
        if len(palette_ids) >= legend_top_n:
            break

    cmap = plt.cm.tab20(np.linspace(0, 1, max(len(palette_ids), 1)))
    id_to_color: dict[int, tuple[float, float, float, float]] = {
        bid: cmap[i] for i, bid in enumerate(palette_ids)
    }
    unassigned_color = (0.75, 0.75, 0.75, 0.35)

    with plt.rc_context(_THESIS_PAPER_RCPARAMS):
        fig, ax = plt.subplots(figsize=(13, 6.5))

        for bid in palette_ids:
            br = branches_by_id.get(bid)
            if not br:
                continue
            pts = sorted(br.get("points", []), key=lambda p: float(p["gamma"]), reverse=True)
            g_line = [float(p["gamma"]) for p in pts if re_lo <= float(p["Phi_eff_real"]) <= re_hi]
            re_line = [float(p["Phi_eff_real"]) for p in pts if re_lo <= float(p["Phi_eff_real"]) <= re_hi]
            if len(g_line) < 2:
                continue
            lbl = br.get("label", f"branch {bid}")
            lw = 2.5 if br.get("is_seed_sheet") else 1.4
            ax.plot(
                g_line,
                re_line,
                "-",
                color=id_to_color[bid],
                lw=lw,
                alpha=0.85 if br.get("is_seed_sheet") else 0.55,
                zorder=4,
                label=lbl,
            )

        for g, y, bid in zip(xs, ys, bids):
            if bid < 0:
                ax.scatter(g, y, c=[unassigned_color], s=14, edgecolors="none", zorder=2)
            else:
                col = id_to_color.get(bid, (0.5, 0.5, 0.5, 0.6))
                ax.scatter(g, y, c=[col], s=26, edgecolors="k", linewidths=0.2, zorder=5)

        sg = [float(r["gamma"]) for r in sweep["points"] if r.get("krawczyk_certified")]
        sr = [float(r["Phi_eff_real"]) for r in sweep["points"] if r.get("krawczyk_certified")]
        order = np.argsort(sg)[::-1]
        ax.plot(
            np.array(sg)[order],
            np.array(sr)[order],
            "k--",
            lw=1.2,
            alpha=0.5,
            zorder=3,
            label="seed row (sweep)",
        )

        ax.set_ylim(re_lo - 0.1, re_hi + 0.1)
        ax.set_ylabel(r"Re $\Phi_M$")
        ax.grid(True, alpha=0.2)
        _style_gamma_axis_reading_zero_to_negative(ax)
        fig.text(
            0.01,
            0.01,
            "Lines = resolved tracks; dots = sweep hits matched by z (branch_step_tol=0.35). Gray = no track match.",
            fontsize=8,
            color="0.35",
        )
        fig.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
    return out_path


def plot_sweep_outliers_per_gamma(
    sweep: dict[str, Any],
    out_path: Path,
    *,
    re_lo: float = DEFAULT_RE_PHYS_LO,
    re_hi: float = DEFAULT_RE_PHYS_HI,
    dpi: int = 140,
) -> Path:
    """How many certified competitors fall outside the physical Re band at each γ."""
    gammas: list[float] = []
    n_out: list[int] = []
    n_in: list[int] = []

    for row in sorted(sweep.get("points", []), key=lambda r: float(r["gamma"]), reverse=True):
        if not row.get("krawczyk_certified", True):
            continue
        sw = row.get("competitor_sweep") or {}
        comps = _sweep_competitor_records(sw)
        if not comps:
            continue
        _, n_h = _filter_sweep_records(comps, re_lo=re_lo, re_hi=re_hi)
        gammas.append(float(row["gamma"]))
        n_out.append(n_h)
        n_in.append(len(comps) - n_h)

    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.bar(gammas, n_in, width=0.008, color="steelblue", alpha=0.7, label=f"in [{re_lo},{re_hi}]")
    ax.bar(gammas, n_out, width=0.008, bottom=n_in, color="salmon", alpha=0.7, label="out of band")
    ax.set_ylabel("# certified competitors")
    ax.set_title("Distant log-sheet competitors per γ (stacked counts)")
    ax.legend(loc="upper left", fontsize=8)
    _style_gamma_axis_reading_zero_to_negative(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _z_vec_point(p: dict[str, Any]) -> np.ndarray:
    return np.asarray(p["w_real"], dtype=float) + 1j * np.asarray(p["w_imag"], dtype=float)


def _z_continuity_tol(z_a: np.ndarray, z_b: np.ndarray) -> float:
    scale = max(float(np.max(np.abs(z_a))), float(np.max(np.abs(z_b))), 1e-3)
    return max(scale * BRANCH_MATCH_STEP_TOL, 0.08)


def _is_z_continuous(z_a: np.ndarray, z_b: np.ndarray) -> bool:
    return float(np.linalg.norm(z_a - z_b, ord=np.inf)) <= _z_continuity_tol(z_a, z_b)


def _track_point_from_resolved(p: dict[str, Any]) -> TrackPoint:
    return TrackPoint(
        gamma=float(p["gamma"]),
        re=float(p["Phi_eff_real"]),
        im=float(p["Phi_eff_imag"]),
        z=_z_vec_point(p),
        source="resolved",
    )


def _track_point_from_sweep(rec: dict[str, Any], gamma: float, *, source: str = "sweep") -> TrackPoint:
    return TrackPoint(
        gamma=float(gamma),
        re=float(rec["Phi_eff_real"]),
        im=float(rec.get("Phi_eff_imag", 0.0)),
        z=_z_from_sweep_rec(rec),
        source=source,
    )


def _merge_track_points(*groups: list[TrackPoint]) -> list[TrackPoint]:
    by_g: dict[float, TrackPoint] = {}
    priority = {"resolved": 0, "sweep": 1, "extension": 2}
    for pts in groups:
        for p in pts:
            gk = round(p.gamma, 6)
            if gk not in by_g or priority[p.source] < priority[by_g[gk].source]:
                by_g[gk] = p
    return sorted(by_g.values(), key=lambda p: p.gamma, reverse=True)


def _split_track_segments(points: list[TrackPoint]) -> list[list[TrackPoint]]:
    if not points:
        return []
    segs: list[list[TrackPoint]] = [[points[0]]]
    for p in points[1:]:
        if _is_z_continuous(segs[-1][-1].z, p.z):
            segs[-1].append(p)
        else:
            segs.append([p])
    return segs


def _load_seed_track_from_continuation(
    seed_path: Path,
    *,
    re_lo: float,
    re_hi: float,
) -> list[TrackPoint]:
    data = _load(seed_path)
    out: list[TrackPoint] = []
    for row in data.get("continuation", []):
        if not row.get("certified") or row.get("failed"):
            continue
        re = float(row["re_phi_m"])
        if not _re_in_physical_band(re, re_lo=re_lo, re_hi=re_hi):
            continue
        z = np.asarray(row["z_real"], dtype=float) + 1j * np.asarray(row["z_imag"], dtype=float)
        out.append(
            TrackPoint(
                gamma=float(row["gamma"]),
                re=re,
                im=float(row["im_phi_m"]),
                z=z,
                source="resolved",
            )
        )
    return sorted(out, key=lambda p: p.gamma, reverse=True)


def _resolved_branch_track(
    branch: dict[str, Any],
    *,
    re_lo: float,
    re_hi: float,
) -> list[TrackPoint]:
    if branch.get("is_seed_sheet"):
        pts = list(branch.get("points", []))
    else:
        pts = [
            p
            for p in branch.get("points", [])
            if p.get("box_disjoint_from_seed")
            and p.get("status") not in (STATUS_DUP, STATUS_SEED)
        ]
    kept = [p for p in pts if _re_in_physical_band(p["Phi_eff_real"], re_lo=re_lo, re_hi=re_hi)]
    return [_track_point_from_resolved(p) for p in kept]


def _collect_sweep_hits_by_branch(
    sweep: dict[str, Any],
    resolved: dict[str, Any],
    *,
    re_lo: float,
    re_hi: float,
) -> dict[int, list[TrackPoint]]:
    by_gamma = _branch_points_by_gamma(resolved)
    hits: dict[int, list[TrackPoint]] = {}
    for row in sweep.get("points", []):
        if not row.get("krawczyk_certified", True):
            continue
        g = float(row["gamma"])
        sw = row.get("competitor_sweep") or {}
        comps_raw = _sweep_competitor_records(sw)
        comps, _ = _filter_sweep_records(comps_raw, re_lo=re_lo, re_hi=re_hi)
        for rec in comps:
            z = _z_from_sweep_rec(rec)
            bid = _match_sweep_to_branch(g, z, by_gamma)
            if bid is None:
                continue
            hits.setdefault(bid, []).append(_track_point_from_sweep(rec, g))
    return hits


def _extend_track_toward_zero(
    track: list[TrackPoint],
    sweep: dict[str, Any],
    resolved: dict[str, Any],
    *,
    branch_id: int,
    re_lo: float,
    re_hi: float,
    gamma_target: float = GAMMA_ZERO_TARGET,
    re_anchor_tol: float = RE_SAME_SHEET_TOL,
    re_family_ids: Optional[set[int]] = None,
) -> list[TrackPoint]:
    """Greedy extension toward γ=0: prefer z-continuous steps, fall back to Re-anchored picks."""
    if not track:
        return track
    track = sorted(track, key=lambda p: p.gamma, reverse=True)
    g_hi = track[0].gamma  # closest existing point to γ=0
    if g_hi >= gamma_target - 1e-6:
        return track

    re_anchor = float(np.median([p.re for p in track[: min(5, len(track))]]))
    by_gamma = _branch_points_by_gamma(resolved)
    sweep_rows = [r for r in sweep.get("points", []) if r.get("krawczyk_certified", True)]
    sweep_by_g = {round(float(r["gamma"]), 6): r for r in sweep_rows}
    # γ runs 0 → −2π; walk from g_hi toward 0 one mesh step at a time.
    gammas_between = sorted(
        float(r["gamma"])
        for r in sweep_rows
        if g_hi + 1e-9 < float(r["gamma"]) <= gamma_target + 1e-6
    )

    family = re_family_ids if re_family_ids is not None else {branch_id}
    extended: list[TrackPoint] = []
    z_prev = track[0].z

    # Step γ-by-γ toward 0 (one mesh step at a time).
    for g in gammas_between:
        row = sweep_by_g.get(round(g, 6))
        if row is None:
            continue
        sw = row.get("competitor_sweep") or {}
        comps_raw = _sweep_competitor_records(sw)
        comps, _ = _filter_sweep_records(comps_raw, re_lo=re_lo, re_hi=re_hi)
        best: Optional[TrackPoint] = None
        best_score = float("inf")
        for rec in comps:
            z = _z_from_sweep_rec(rec)
            re_c = float(rec["Phi_eff_real"])
            d_re = abs(re_c - re_anchor)
            if d_re > re_anchor_tol:
                continue
            bid = _match_sweep_to_branch(g, z, by_gamma)
            z_ok = _is_z_continuous(z_prev, z)
            d_z = float(np.linalg.norm(z - z_prev, ord=np.inf))
            score = d_re
            if z_ok:
                score += 0.02 * d_z
            else:
                # Re-family extension before branch_id appears in resolve (43/46 onset gap).
                score += 2.0 + 0.05 * d_z
            if bid in family:
                score *= 0.12
            elif bid is not None:
                score *= 0.85
            if score < best_score:
                best_score = score
                best = _track_point_from_sweep(rec, g, source="extension")
        if best is None or best_score > re_anchor_tol + 2.5:
            break
        extended.append(best)
        z_prev = best.z

    return _merge_track_points(extended, track)


def _diagnose_pair_same_re_sheet(
    pts_a: list[TrackPoint],
    pts_b: list[TrackPoint],
    *,
    gamma_pair_tol: float = 0.011,
) -> dict[str, Any]:
    """Lightweight 43/46-style same-sheet gauge on track points."""
    by_a = {round(p.gamma, 6): p for p in pts_a}
    by_b = {round(p.gamma, 6): p for p in pts_b}
    pairs: list[tuple[TrackPoint, TrackPoint]] = []
    for gk, pa in by_a.items():
        for gkb, pb in by_b.items():
            if abs(gk - gkb) <= gamma_pair_tol:
                pairs.append((pa, pb))
                break
    if not pairs:
        return {"n_pairs": 0, "same_z_sheet": False, "same_re_sheet": False, "im_opposite_fraction": 0.0}

    z_close = 0
    re_close = 0
    im_opp = 0
    for pa, pb in pairs:
        if _is_z_continuous(pa.z, pb.z):
            z_close += 1
        if abs(pa.re - pb.re) <= RE_SAME_SHEET_TOL:
            re_close += 1
        if abs(pa.im + pb.im) <= IM_OPPOSITE_TOL:
            im_opp += 1
    n = len(pairs)
    same_z = z_close / n >= 0.5 and re_close / n >= 0.5
    same_re = re_close / n >= 0.9 and max(abs(pa.re - pb.re) for pa, pb in pairs) < 0.5
    return {
        "n_pairs": n,
        "same_z_sheet": same_z,
        "same_re_sheet": same_re,
        "im_opposite_fraction": im_opp / n,
        "re_close_fraction": re_close / n,
    }


def _select_v2_branch_ids(
    resolved: dict[str, Any],
    sweep_hits: dict[int, list[TrackPoint]],
    *,
    top_n: int = 8,
    force_ids: tuple[int, ...] = (43, 46),
) -> list[int]:
    ranked = sorted(sweep_hits.items(), key=lambda t: len(t[1]), reverse=True)
    out: list[int] = []
    for bid in force_ids:
        if bid not in out and any(int(b["branch_id"]) == bid for b in resolved.get("branches", [])):
            out.append(bid)
    for bid, _hits in ranked:
        if bid in out:
            continue
        br = next((b for b in resolved["branches"] if int(b["branch_id"]) == bid), None)
        if br is None or br.get("is_seed_sheet"):
            continue
        out.append(bid)
        if len(out) >= top_n:
            break
    return out


def _build_branch_track_v2(
    branch: dict[str, Any],
    sweep_hits: list[TrackPoint],
    sweep: dict[str, Any],
    resolved: dict[str, Any],
    *,
    re_lo: float,
    re_hi: float,
    extend_to_zero: bool,
    re_family_ids: Optional[set[int]] = None,
) -> list[TrackPoint]:
    bid = int(branch["branch_id"])
    resolved_pts = _resolved_branch_track(branch, re_lo=re_lo, re_hi=re_hi)
    merged = _merge_track_points(resolved_pts, sweep_hits)
    if extend_to_zero:
        merged = _extend_track_toward_zero(
            merged,
            sweep,
            resolved,
            branch_id=bid,
            re_lo=re_lo,
            re_hi=re_hi,
            re_family_ids=re_family_ids,
        )
    return merged


def plot_sweep_tracked_re_v2(
    sweep: dict[str, Any],
    resolved: dict[str, Any],
    out_path: Path,
    *,
    seed_continuation: Path,
    re_lo: float = DEFAULT_RE_PHYS_LO,
    re_hi: float = DEFAULT_RE_PHYS_HI,
    top_n: int = 8,
    extend_ids: tuple[int, ...] = (43, 46),
    dpi: int = 160,
) -> tuple[Path, dict[str, Any]]:
    """
    Longest-possible Re Φ branch tracks: seed from continuation (0→−2π),
    competitors from resolved+sweep with z-continuous extension toward γ=0.
    """
    seed_track = _load_seed_track_from_continuation(seed_continuation, re_lo=re_lo, re_hi=re_hi)
    sweep_hits = _collect_sweep_hits_by_branch(sweep, resolved, re_lo=re_lo, re_hi=re_hi)
    branch_ids = _select_v2_branch_ids(resolved, sweep_hits, top_n=top_n, force_ids=extend_ids)
    branches_by_id = {int(b["branch_id"]): b for b in resolved.get("branches", [])}
    re_family_43_46: set[int] = {43, 46}

    meta: dict[str, Any] = {
        "seed_source": str(seed_continuation),
        "seed_n_points": len(seed_track),
        "seed_gamma_range": [float(seed_track[-1].gamma), float(seed_track[0].gamma)] if seed_track else [],
        "branches": {},
        "pair_diagnoses": {},
    }

    fig, ax = plt.subplots(figsize=(13, 7))

    # Seed: full continuation, black.
    seed_segs = _split_track_segments(seed_track)
    for si, seg in enumerate(seed_segs):
        g = [p.gamma for p in seg]
        re = [p.re for p in seg]
        lbl = "seed branch (continuation)" if si == 0 else "_seed_cont"
        ax.plot(g, re, color="black", lw=3.0, zorder=12, label=lbl)

    cmap = plt.cm.tab10(np.linspace(0, 1, max(len(branch_ids), 1)))
    for i, bid in enumerate(branch_ids):
        br = branches_by_id.get(bid)
        if br is None:
            continue
        extend = bid in extend_ids
        family = re_family_43_46 if bid in re_family_43_46 else {bid}
        track = _build_branch_track_v2(
            br,
            sweep_hits.get(bid, []),
            sweep,
            resolved,
            re_lo=re_lo,
            re_hi=re_hi,
            extend_to_zero=extend,
            re_family_ids=family,
        )
        if len(track) < 2:
            continue

        diag = {
            "n_resolved": len(_resolved_branch_track(br, re_lo=re_lo, re_hi=re_hi)),
            "n_sweep_hits": len(sweep_hits.get(bid, [])),
            "n_merged": len(track),
            "gamma_max": float(track[0].gamma),
            "gamma_min": float(track[-1].gamma),
            "extended_toward_zero": extend,
            "n_segments": len(_split_track_segments(track)),
        }
        meta["branches"][str(bid)] = diag

        color = V2_BRANCH_COLORS.get(bid, cmap[i % len(cmap)])
        base_lbl = br.get("label", f"branch {bid}")
        ext_tag = ""
        if extend:
            n_ext = sum(1 for p in track if p.source == "extension")
            if track[0].gamma >= GAMMA_ZERO_TARGET - 0.05:
                ext_tag = f" (extended to γ≈0, +{n_ext} pts)"
            elif n_ext > 0:
                ext_tag = f" (+{n_ext} ext pts to γ={track[0].gamma:.2f})"

        segs = _split_track_segments(track)
        if bid not in extend_ids:
            segs = [s for s in segs if len(s) >= 4]
            if not segs:
                segs = [max(_split_track_segments(track), key=len)]
        for si, seg in enumerate(segs):
            g = [p.gamma for p in seg]
            re = [p.re for p in seg]
            ext_pts = [p for p in seg if p.source == "extension"]
            ax.plot(
                g,
                re,
                color=color,
                lw=2.2 if bid in extend_ids else 1.6,
                ls="-" if si == 0 else "--",
                alpha=0.92,
                zorder=8,
                label=f"{base_lbl}{ext_tag}" if si == 0 else f"_{bid}_{si}",
            )
            if ext_pts:
                ax.scatter(
                    [p.gamma for p in ext_pts],
                    [p.re for p in ext_pts],
                    c=[color],
                    s=28,
                    marker="^",
                    edgecolors="k",
                    linewidths=0.3,
                    zorder=9,
                    alpha=0.85,
                )
            # Faint sweep confirmation dots (non-extension).
            conf = [p for p in seg if p.source == "sweep"]
            if conf:
                ax.scatter(
                    [p.gamma for p in conf],
                    [p.re for p in conf],
                    c=[color],
                    s=12,
                    alpha=0.35,
                    edgecolors="none",
                    zorder=6,
                )

    if 43 in branch_ids and 46 in branch_ids:
        t43 = _build_branch_track_v2(
            branches_by_id[43],
            sweep_hits.get(43, []),
            sweep,
            resolved,
            re_lo=re_lo,
            re_hi=re_hi,
            extend_to_zero=True,
            re_family_ids=re_family_43_46,
        )
        t46 = _build_branch_track_v2(
            branches_by_id[46],
            sweep_hits.get(46, []),
            sweep,
            resolved,
            re_lo=re_lo,
            re_hi=re_hi,
            extend_to_zero=True,
            re_family_ids=re_family_43_46,
        )
        pair = _diagnose_pair_same_re_sheet(t43, t46)
        meta["pair_diagnoses"]["43_46"] = pair
        if pair.get("same_re_sheet") and not pair.get("same_z_sheet"):
            ax.annotate(
                "43/46: same Re Φ family (Im±), distinct z — merge for exponent only",
                xy=(0.02, 0.97),
                xycoords="axes fraction",
                fontsize=8,
                color="#444",
                va="top",
                bbox=dict(boxstyle="round,pad=0.3", fc="wheat", alpha=0.35),
            )

    ax.set_ylim(re_lo - 0.1, re_hi + 0.1)
    ax.set_ylabel(r"Re $\Phi$")
    ax.set_xlabel(r"$\gamma$")
    ax.set_title(
        f"Branch tracks (Re Φ): seed continuation + sweep-augmented competitors "
        f"({re_lo} ≤ Re Φ ≤ {re_hi})"
    )
    ax.axhline(0.0, color="gray", lw=0.5, alpha=0.4)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=7, framealpha=0.95)
    ax.grid(True, alpha=0.22)
    _style_gamma_axis_reading_zero_to_negative(ax)
    foot = (
        "Black = seed from continuation (γ: 0→−2π). Solid/dashed = z-continuous segments. "
        "△ = backward extension from sweep. Faint dots = sweep hits on same branch_id."
    )
    fig.text(0.01, 0.01, foot, fontsize=7.5, color="0.35")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    meta_path = out_path.with_suffix(".json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return out_path, meta


def write_summary(resolved: dict[str, Any], out_path: Path, *, top_n: int = 12) -> None:
    picked = select_branches(resolved, top_n=top_n)
    rows = []
    for b in picked:
        pts = _valid_sheet_points(b)
        if not pts:
            continue
        g = [float(p["gamma"]) for p in pts]
        re = [float(p["Phi_eff_real"]) for p in pts]
        rows.append(
            {
                "branch_id": b["branch_id"],
                "label": b.get("label"),
                "is_seed_sheet": bool(b.get("is_seed_sheet")),
                "n": len(pts),
                "gamma_min": min(g),
                "gamma_max": max(g),
                "re_min": min(re),
                "re_max": max(re),
                "re_end": re[0] if g[0] >= g[-1] else re[-1],
            }
        )
    out_path.write_text(json.dumps({"top_branches": rows}, indent=2), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description="Readable Re(Phi_M) sheet plots from z_robust_full")
    p.add_argument("--dir", type=str, default=str(DEFAULT_DIR))
    p.add_argument("--top-n", type=int, default=10)
    p.add_argument("--dpi", type=int, default=160)
    p.add_argument("--re-lo", type=float, default=DEFAULT_RE_PHYS_LO)
    p.add_argument("--re-hi", type=float, default=DEFAULT_RE_PHYS_HI)
    p.add_argument(
        "--seed-continuation",
        type=str,
        default="",
        help="seed_branch_continuation.json (default: run_dir/seed_branch_continuation.json)",
    )
    args = p.parse_args()

    d = Path(args.dir)
    resolved_path = d / "resolved_robust_z.json"
    sweep_path = d / "competitors_robust_z.json"
    if not resolved_path.is_file():
        raise SystemExit(f"missing {resolved_path}")

    resolved = _load(resolved_path)
    write_summary(resolved, d / "re_phi_top_sheets_summary.json", top_n=args.top_n)

    seed_cont = Path(args.seed_continuation) if args.seed_continuation.strip() else d.parent.parent / "seed_branch_continuation.json"

    outs: list[Path] = [
        plot_tracked_re_sheets(
            resolved, d / "re_phi_tracked_sheets.png", top_n=args.top_n, dpi=args.dpi
        ),
        plot_re_zoom_wall(resolved, d / "re_phi_wall_zoom.png", top_n=8, dpi=args.dpi),
    ]
    if sweep_path.is_file():
        sweep = _load(sweep_path)
        re_kw = {"re_lo": args.re_lo, "re_hi": args.re_hi, "dpi": args.dpi}
        outs.extend(
            [
                plot_sweep_re_envelope(sweep, d / "re_phi_sweep_envelope.png", **re_kw),
                plot_sweep_re_scatter(sweep, d / "re_phi_sweep_scatter.png", **re_kw),
                plot_sweep_re_scatter_by_branch_id(
                    sweep,
                    resolved,
                    d / "re_phi_sweep_scatter_by_branch.png",
                    **re_kw,
                ),
                plot_sweep_outliers_per_gamma(sweep, d / "re_phi_sweep_outliers.png", **re_kw),
            ]
        )
        if seed_cont.is_file():
            v2_path, v2_meta = plot_sweep_tracked_re_v2(
                sweep,
                resolved,
                d / "re_phi_sweep_tracked_v2.png",
                seed_continuation=seed_cont,
                re_lo=args.re_lo,
                re_hi=args.re_hi,
                top_n=max(8, args.top_n),
                dpi=args.dpi,
            )
            outs.append(v2_path)
            print(f"  tracked v2 meta: n_seed={v2_meta.get('seed_n_points')} branches={list(v2_meta.get('branches', {}))}")

    for o in outs:
        print(f"Wrote {o}")


if __name__ == "__main__":
    main()
