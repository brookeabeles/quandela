"""
Scan certified competitor saddles along γ to find which best matches exact finite-n.

Uses the seed-branch continuation (interpolated z*) as the branch-exclusion reference.
Writes to ``phasecraft/results/bm24_saddle_audit_p1/run_seed_competitors/`` by default.

Not a global completeness proof — only saddles found by the stated random search.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np

_PKG_DIR = Path(__file__).resolve().parent
PHASECRAFT_ROOT = _PKG_DIR.parents[1]  # …/phasecraft (works through experiments/ symlink)
REPO_ROOT = PHASECRAFT_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from phasecraft.bm24_saddle_audit_p1.competitor_dominance_diagnostic import (
    DISCLAIMER,
    interpolate_seed_z,
    _load_continuation,
)
from phasecraft.bm24_saddle_audit_p1.continue_seed_branch_certified import (
    IM_PHI_WARNING,
    LABEL_EXACT_FINITE_N_EXPONENT,
    conv2_full_exponent,
    discover_competitors_at_gamma,
    lambda_abs_n_max,
    selected_competitor_gammas,
    _style_gamma_axis_reading_zero_to_negative,
)
from phasecraft.picard_lefschetz import compute_phi

DEFAULT_RESULTS_ROOT = PHASECRAFT_ROOT / "results" / "bm24_saddle_audit_p1"
DEFAULT_SEED_BRANCH = DEFAULT_RESULTS_ROOT / "run_seed_branch_g-2pi"
DEFAULT_OUT = DEFAULT_RESULTS_ROOT / "run_seed_competitors"

LABEL_SEED_SADDLE = (
    r"Krawczyk-certified seed-saddle exponent ($\mathrm{Re}\,\Phi_M + \mathrm{BM24}$ A41 prefactor)"
)
LABEL_BEST_MATCH = r"Best match to exact finite-$n$ (discovered competitor)"
LABEL_MAX_ALGEBRAIC = r"Max certified competitor exponent (algebraic Re$\,\Phi$)"
TITLE_WITH_COMPETITORS = (
    r"Exact finite-$n$ vs seed vs competitors (BM24 $q{=}3$, $p{=}1$)"
)


def _continuation_rows_from_json(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [r for r in data.get("continuation", data) if r.get("certified") and not r.get("failed")]


def _continuation_rows_from_csv(path: Path) -> list[dict]:
    import csv

    rows: list[dict] = []
    with path.open(encoding="utf-8") as fh:
        for d in csv.DictReader(fh):
            if d.get("certified", "").lower() == "true" and d.get("failed", "").lower() != "true":
                rows.append(d)
    return rows


def load_seed_continuation(branch_dir: Path) -> list[dict]:
    json_path = branch_dir / "seed_branch_continuation.json"
    csv_path = branch_dir / "seed_branch_continuation.csv"
    if json_path.is_file():
        return _continuation_rows_from_json(json_path)
    if csv_path.is_file():
        return _continuation_rows_from_csv(csv_path)
    raise FileNotFoundError(f"No continuation under {branch_dir}")


def _comps_for_gamma(by_gamma: dict[str, list[dict]], gamma: float, tol: float = 1e-5) -> list[dict]:
    key = str(gamma)
    if key in by_gamma:
        return by_gamma[key]
    for k, comps in by_gamma.items():
        if abs(float(k) - gamma) <= tol:
            return comps
    return []


def competitor_scan_gamma_list(
    *,
    gamma_left: float = 0.0,
    gamma_right: float = -2.0 * math.pi,
    coarse_every: float = math.pi / 4.0,
    dense_below: float = -0.6 * math.pi,
    dense_every: float = 0.08,
) -> list[float]:
    """Coarse π/4 grid plus denser samples where seed vs exact diverges."""
    coarse = selected_competitor_gammas(gamma_left, gamma_right, coarse_every)
    dense = selected_competitor_gammas(dense_below, gamma_right, dense_every)
    return sorted({round(g, 10) for g in coarse + dense}, reverse=True)


def _pick_best_match(
    comps: list[dict], *, lambda_exact: float, include_seed: Optional[dict] = None
) -> Optional[dict]:
    candidates: list[dict] = [c for c in comps if c.get("certified")]
    if include_seed is not None:
        candidates.append(include_seed)
    if not candidates:
        return None
    return min(candidates, key=lambda c: abs(float(c["full_conv2_exponent"]) - lambda_exact))


def _pick_max_algebraic(comps: list[dict]) -> Optional[dict]:
    cert = [c for c in comps if c.get("certified")]
    if not cert:
        return None
    return max(cert, key=lambda c: float(c["full_conv2_exponent"]))


def _seed_entry_at_gamma(
    *,
    q: int,
    K_clause: int,
    r: float,
    beta: float,
    gamma: float,
    cont_rows: list[dict],
    lam: float,
) -> dict[str, Any]:
    seed_z = interpolate_seed_z(cont_rows, gamma)
    betas = np.array([beta])
    gammas = np.array([gamma])
    seed_phi = float(compute_phi(seed_z, q=q, r=r, betas=betas, gammas=gammas).real)
    seed_full = conv2_full_exponent(seed_phi, K_clause, r)
    return {
        "gamma": float(gamma),
        "certified": True,
        "label": "seed_branch",
        "full_conv2_exponent": seed_full,
        "re_phi_m": seed_phi,
        "gap_to_exact": float(seed_full - lam),
    }


def _summarize_gamma_scan(
    *,
    gamma: float,
    comps: list[dict],
    cont_rows: list[dict],
    q: int,
    K_clause: int,
    r: float,
    beta: float,
    n_values: list[int],
) -> dict[str, Any]:
    lam = lambda_abs_n_max(q, K_clause, r, beta, gamma, n_values)
    seed_entry = _seed_entry_at_gamma(
        q=q, K_clause=K_clause, r=r, beta=beta, gamma=gamma, cont_rows=cont_rows, lam=lam
    )
    seed_full = float(seed_entry["full_conv2_exponent"])
    best = _pick_best_match(comps, lambda_exact=lam, include_seed=seed_entry)
    max_alg = _pick_max_algebraic(comps)
    return {
        "gamma": float(gamma),
        "lambda_abs_n_max": float(lam),
        "seed_full_conv2_exponent": seed_full,
        "seed_gap_to_exact": float(seed_full - lam),
        "num_certified_competitors": sum(1 for c in comps if c.get("certified")),
        "best_match": None if best is None else _summarize_candidate(best, lam),
        "max_algebraic_competitor": None if max_alg is None else _summarize_candidate(max_alg, lam),
    }


def _summarize_candidate(c: dict, lam: float) -> dict[str, Any]:
    full = float(c["full_conv2_exponent"])
    return {
        "label": c.get("label", "unknown"),
        "full_conv2_exponent": full,
        "re_phi_m": float(c.get("re_phi_m", math.nan)),
        "im_phi_m": float(c.get("im_phi_m", math.nan)),
        "gap_to_exact": float(full - lam),
        "abs_gap_to_exact": float(abs(full - lam)),
        "z_distance_from_seed_branch": float(c.get("z_distance_from_seed_branch", math.nan)),
        "residual_inf": float(c.get("residual_inf", math.nan)),
        "krawczyk_contraction": float(c.get("krawczyk_contraction", math.nan)),
    }


def run_scan(
    *,
    q: int = 3,
    K_clause: int = 8,
    r: float = 176.54,
    beta: float = 0.5433996420760803,
    seed_branch_dir: Path,
    out_dir: Path,
    gammas: list[float],
    n_min: int = 12,
    n_max: int = 22,
    num_starts: int = 500,
    branch_tol: float = 1e-4,
    min_residual: float = 1e-10,
    dps: int = 80,
    seed: int = 0,
    plot_only: bool = False,
    prior_competitors: Optional[Path] = None,
    discover_missing: bool = True,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    cont_rows = load_seed_continuation(seed_branch_dir)
    n_values = list(range(n_min, n_max + 1))
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")

    by_gamma: dict[str, list[dict]] = {}
    if prior_competitors and prior_competitors.is_file():
        prior = json.loads(prior_competitors.read_text(encoding="utf-8"))
        by_gamma.update(prior.get("by_gamma", {}))

    per_gamma_summary: list[dict[str, Any]] = []

    if not plot_only:
        for i, g in enumerate(gammas):
            comps = _comps_for_gamma(by_gamma, g)
            if comps:
                pass
            elif not discover_missing:
                print(f"skip gamma={g:.4f} (no prior competitors)", flush=True)
                continue
            else:
                print(f"[{i + 1}/{len(gammas)}] discovering competitors at gamma={g:.4f}", flush=True)
                seed_z = interpolate_seed_z(cont_rows, g)
                comps = discover_competitors_at_gamma(
                    q,
                    K_clause,
                    r,
                    beta,
                    g,
                    seed_z,
                    num_starts=num_starts,
                    seed=seed + int(1000 * abs(g)),
                    branch_tol=branch_tol,
                    min_residual=min_residual,
                    dps=dps,
                )
                by_gamma[str(g)] = comps
            if not comps:
                continue
            per_gamma_summary.append(
                _summarize_gamma_scan(
                    gamma=g,
                    comps=comps,
                    cont_rows=cont_rows,
                    q=q,
                    K_clause=K_clause,
                    r=r,
                    beta=beta,
                    n_values=n_values,
                )
            )

        with open(out_dir / "competitor_saddles_by_gamma.json", "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "metadata": {
                        "q": q,
                        "K_clause": K_clause,
                        "r": r,
                        "beta": beta,
                        "gammas": gammas,
                        "num_starts": num_starts,
                        "seed_branch_dir": str(seed_branch_dir),
                        "disclaimer": DISCLAIMER,
                        "im_phi_warning": IM_PHI_WARNING,
                        "timestamp": ts,
                    },
                    "by_gamma": by_gamma,
                },
                fh,
                indent=2,
            )
        with open(out_dir / "competitor_scan_summary.json", "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "metadata": {"timestamp": ts, "disclaimer": DISCLAIMER},
                    "per_gamma": per_gamma_summary,
                },
                fh,
                indent=2,
            )
    else:
        summary_path = out_dir / "competitor_scan_summary.json"
        saddles_path = out_dir / "competitor_saddles_by_gamma.json"
        if prior_competitors and prior_competitors.is_file() and not saddles_path.is_file():
            by_gamma = json.loads(prior_competitors.read_text(encoding="utf-8")).get("by_gamma", {})
            saddles_path = prior_competitors  # summarize from prior file
        if summary_path.is_file():
            per_gamma_summary = json.loads(summary_path.read_text(encoding="utf-8"))["per_gamma"]
        elif saddles_path.is_file():
            by_gamma = json.loads(saddles_path.read_text())["by_gamma"]
            for g in sorted((float(k) for k in by_gamma), reverse=True):
                comps = _comps_for_gamma(by_gamma, g)
                per_gamma_summary.append(
                    _summarize_gamma_scan(
                        gamma=g,
                        comps=comps,
                        cont_rows=cont_rows,
                        q=q,
                        K_clause=K_clause,
                        r=r,
                        beta=beta,
                        n_values=n_values,
                    )
                )
        else:
            raise FileNotFoundError("plot-only requires competitor_scan_summary.json or competitor_saddles_by_gamma.json")

    _plot_exponents_with_competitors(cont_rows, per_gamma_summary, out_dir)
    _plot_best_match_gap(per_gamma_summary, out_dir)
    if per_gamma_summary and not (out_dir / "competitor_scan_summary.json").is_file():
        with open(out_dir / "competitor_scan_summary.json", "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "metadata": {
                        "timestamp": ts,
                        "disclaimer": DISCLAIMER,
                        "num_gamma_summarized": len(per_gamma_summary),
                    },
                    "per_gamma": per_gamma_summary,
                },
                fh,
                indent=2,
            )
    print(f"Wrote results to {out_dir}")
    return out_dir


def _plot_exponents_with_competitors(
    cont_rows: list[dict],
    scan_summary: list[dict[str, Any]],
    out_dir: Path,
) -> None:
    cont_sorted = sorted(cont_rows, key=lambda r: float(r["gamma"]), reverse=True)
    g_cont = [float(r["gamma"]) for r in cont_sorted]
    seed_exp = [float(r["full_conv2_exponent"]) for r in cont_sorted]
    exact_exp = [float(r["lambda_abs_n_max"]) for r in cont_sorted]

    scan_sorted = sorted(scan_summary, key=lambda s: float(s["gamma"]), reverse=True)
    g_scan = [float(s["gamma"]) for s in scan_sorted]
    best_exp = [
        float(s["best_match"]["full_conv2_exponent"]) if s.get("best_match") else math.nan
        for s in scan_sorted
    ]
    max_exp = [
        float(s["max_algebraic_competitor"]["full_conv2_exponent"])
        if s.get("max_algebraic_competitor")
        else math.nan
        for s in scan_sorted
    ]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(g_cont, seed_exp, "o-", ms=2, lw=1.2, label=LABEL_SEED_SADDLE)
    ax.plot(g_cont, exact_exp, "s--", ms=2, lw=1.2, label=LABEL_EXACT_FINITE_N_EXPONENT)
    ax.plot(g_scan, best_exp, "D-", ms=5, lw=1.5, color="C2", label=LABEL_BEST_MATCH)
    ax.plot(g_scan, max_exp, "^:", ms=4, lw=1.0, color="C3", alpha=0.85, label=LABEL_MAX_ALGEBRAIC)
    ax.set_ylabel("exponent (natural log)")
    ax.set_title(TITLE_WITH_COMPETITORS)
    _style_gamma_axis_reading_zero_to_negative(ax)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "gamma_vs_exponents_with_competitors.png", dpi=150)
    plt.close(fig)


def _plot_best_match_gap(scan_summary: list[dict[str, Any]], out_dir: Path) -> None:
    scan_sorted = sorted(scan_summary, key=lambda s: float(s["gamma"]), reverse=True)
    g_scan = [float(s["gamma"]) for s in scan_sorted]
    seed_gap = [float(s.get("seed_gap_to_exact", math.nan)) for s in scan_sorted]
    best_gap = [
        float(s["best_match"]["gap_to_exact"]) if s.get("best_match") else math.nan for s in scan_sorted
    ]
    best_label = [
        (s.get("best_match") or {}).get("label", "") for s in scan_sorted
    ]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(g_scan, seed_gap, "o-", label="Seed branch gap to exact")
    ax.plot(g_scan, best_gap, "D-", label="Best-match gap to exact")
    ax.axhline(0.0, color="k", lw=0.8, alpha=0.5)
    ax.set_ylabel(r"exponent $-$ exact $\lambda(n_{\max})$")
    ax.set_title("Gap to exact finite-$n$ exponent (BM24 $q{=}3$, $p{=}1$)")
    _style_gamma_axis_reading_zero_to_negative(ax)
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "gamma_vs_gap_to_exact_competitors.png", dpi=150)
    plt.close(fig)

    # Text summary: where best match is not seed
    flips = [
        s
        for s in scan_sorted
        if s.get("best_match") and s["best_match"].get("label") != "seed_branch"
    ]
    with open(out_dir / "best_match_not_seed.json", "w", encoding="utf-8") as fh:
        json.dump({"count": len(flips), "per_gamma": flips}, fh, indent=2)


def main(argv: Optional[list[str]] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed-branch-dir", type=Path, default=DEFAULT_SEED_BRANCH)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--gamma-every", type=float, default=math.pi / 4.0, help="Coarse grid step (rad)")
    p.add_argument("--dense-below", type=float, default=-0.6 * math.pi)
    p.add_argument("--dense-every", type=float, default=0.08)
    p.add_argument("--num-starts", type=int, default=500)
    p.add_argument("--plot-only", action="store_true")
    p.add_argument(
        "--reuse-prior",
        type=Path,
        default=DEFAULT_SEED_BRANCH / "competitor_saddles_by_gamma.json",
        help="Reuse competitor lists from this JSON when keys match",
    )
    p.add_argument(
        "--include-all-prior-gammas",
        action="store_true",
        default=True,
        help="Also summarize every gamma present in --reuse-prior (default: on)",
    )
    p.add_argument("--quick", action="store_true")
    args = p.parse_args(argv)

    gammas = competitor_scan_gamma_list(
        coarse_every=args.gamma_every,
        dense_below=args.dense_below,
        dense_every=args.dense_every,
    )
    kwargs: dict[str, Any] = dict(
        seed_branch_dir=args.seed_branch_dir,
        out_dir=args.out_dir,
        gammas=gammas,
        num_starts=args.num_starts,
        prior_competitors=args.reuse_prior if args.reuse_prior.is_file() else None,
    )
    if args.reuse_prior.is_file() and args.include_all_prior_gammas:
        prior_g = sorted(
            {float(k) for k in json.loads(args.reuse_prior.read_text())["by_gamma"]},
            reverse=True,
        )
        gammas = sorted({round(g, 10) for g in list(gammas) + prior_g}, reverse=True)

    if args.quick:
        kwargs.update(
            gammas=selected_competitor_gammas(0.0, -2.0 * math.pi, math.pi / 2.0),
            num_starts=200,
        )
    kwargs["gammas"] = gammas

    if args.plot_only:
        kwargs["plot_only"] = True
        kwargs["discover_missing"] = False
        run_scan(**kwargs)
        return

    run_scan(**kwargs)


if __name__ == "__main__":
    main()
