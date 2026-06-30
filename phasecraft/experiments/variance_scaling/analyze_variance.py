"""
analyze_variance.py — estimate v = lim_n n·Var(Xₙ) from per-instance JSONL.

Reads variance_rows.jsonl (from evaluate_variance.py). For each
(objective, train_n, p) combination produces:
  1. n·Var(Xₙ) vs n with √(2/N) relative error bands
  2. Richardson fit  n·Var(Xₙ) = v + a/n
  3. Bulk vs full variance (trim top-T% by X)
  4. Skewness vs n
  5. Mean Xₙ vs n (sanity-check c_ann recovery)
  6. Verdict: c_typ = c_ann + (ln2/2)·v

Usage:
    python -m phasecraft.experiments.variance_scaling.analyze_variance \\
        [--rows PATH] [--out-dir DIR] [--trim-frac FLOAT]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[3]
for _p in (_REPO_ROOT, _REPO_ROOT / "phasecraft"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from phasecraft.lib.paths import bm24_runs_dir

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

LN2 = math.log(2.0)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_rows(path: Path) -> List[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def group_rows(
    rows: List[dict],
) -> Dict[Tuple[str, int, int], Dict[int, dict]]:
    """
    Returns {(objective, train_n, p): {n: {"X": [...], "unsat": [...]}}}
    """
    out: Dict[Tuple[str, int, int], Dict[int, dict]] = {}
    for row in rows:
        key = (row["objective"], int(row["train_n"]), int(row["p"]))
        n = int(row["n"])
        x = float(row["X"])
        unsat = bool(row.get("unsat", False))
        if key not in out:
            out[key] = {}
        if n not in out[key]:
            out[key][n] = {"X": [], "unsat": []}
        out[key][n]["X"].append(x)
        out[key][n]["unsat"].append(unsat)
    return out


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def richardson_fit(ns: np.ndarray, nvar: np.ndarray) -> Tuple[float, float, float]:
    """
    Fit  n·Var(Xₙ) = v + a/n  by least-squares in (1/n, nVar) space.
    Returns (v, a, r²).
    """
    inv_n = 1.0 / ns
    # Design matrix [1, 1/n]
    A = np.column_stack([np.ones_like(inv_n), inv_n])
    coef, *_ = np.linalg.lstsq(A, nvar, rcond=None)
    v, a = float(coef[0]), float(coef[1])
    resid = nvar - (v + a * inv_n)
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((nvar - nvar.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return v, a, r2


def sample_skewness(x: np.ndarray) -> float:
    """Adjusted Fisher-Pearson skewness."""
    n = len(x)
    if n < 3:
        return float("nan")
    mu = x.mean()
    sigma = x.std(ddof=1)
    if sigma == 0:
        return float("nan")
    return float(np.sum(((x - mu) / sigma) ** 3) * n / ((n - 1) * (n - 2)))


def analyze_combo(
    x_by_n: Dict[int, dict],
    trim_frac: float = 0.05,
) -> dict:
    """
    Compute statistics for one (objective, train_n, p) combination.
    x_by_n: {n: {"X": [...], "unsat": [...]}}
    Returns a dict of per-n and summary stats.
    """
    ns = sorted(x_by_n.keys())
    per_n = {}
    for n in ns:
        X = np.array(x_by_n[n]["X"], dtype=float)
        unsat_flags = np.array(x_by_n[n]["unsat"], dtype=bool)
        N = len(X)
        var_full = float(np.var(X, ddof=1))
        nvar_full = n * var_full

        # Bulk variance: trim top trim_frac by X
        n_trim = max(1, int(np.floor(N * trim_frac)))
        thresh = np.sort(X)[-n_trim]
        X_bulk = X[X < thresh]
        var_bulk = float(np.var(X_bulk, ddof=1)) if len(X_bulk) > 1 else float("nan")
        nvar_bulk = n * var_bulk

        # Error bar on nVar: relative error sqrt(2/N) (Gaussian approximation)
        nvar_err = nvar_full * math.sqrt(2.0 / max(N - 1, 1))

        per_n[n] = {
            "N": N,
            "mean_X": float(X.mean()),
            "var_X": var_full,
            "nVar_X": nvar_full,
            "nVar_X_err": nvar_err,
            "nVar_X_bulk": nvar_bulk,
            "skewness": sample_skewness(X),
            "n_unsat": int(unsat_flags.sum()),
        }

    ns_arr = np.array(ns, dtype=float)
    nvar_arr = np.array([per_n[n]["nVar_X"] for n in ns])

    v, a, r2 = richardson_fit(ns_arr, nvar_arr) if len(ns) >= 3 else (float("nan"), float("nan"), float("nan"))

    # c_ann from mean X at largest n (slope of mean_X vs n is c_typ)
    if len(ns) >= 2:
        from scipy.stats import linregress
        mean_xs = np.array([per_n[n]["mean_X"] for n in ns])
        res = linregress(ns_arr, mean_xs)
        c_typ_slope = float(res.slope)
        c_typ_intercept = float(res.intercept)
    else:
        c_typ_slope = float("nan")
        c_typ_intercept = float("nan")

    # Verdict: c_typ = c_ann + (ln2/2) * v   (Gaussian log-normal relation)
    c_typ_from_v = c_typ_intercept + (LN2 / 2.0) * v if math.isfinite(v) else float("nan")

    return {
        "per_n": per_n,
        "richardson": {"v": v, "a": a, "r2": r2},
        "c_typ_slope": c_typ_slope,
        "c_typ_intercept": c_typ_intercept,
        "c_typ_from_v": c_typ_from_v,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_combo(
    key: Tuple[str, int, int],
    result: dict,
    out_dir: Path,
) -> None:
    if not HAS_MPL:
        return
    obj, tn, p = key
    per_n = result["per_n"]
    ns = sorted(per_n.keys())

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # --- Panel 1: n·Var(Xₙ) vs n ---
    ax = axes[0]
    nvar_vals = [per_n[n]["nVar_X"] for n in ns]
    nvar_errs = [per_n[n]["nVar_X_err"] for n in ns]
    nvar_bulk = [per_n[n]["nVar_X_bulk"] for n in ns]
    ax.errorbar(ns, nvar_vals, yerr=nvar_errs, fmt="o-", label="full", capsize=4)
    ax.plot(ns, nvar_bulk, "s--", label="bulk (trim 5%)")

    v = result["richardson"]["v"]
    a = result["richardson"]["a"]
    r2 = result["richardson"]["r2"]
    if math.isfinite(v):
        ns_fine = np.linspace(min(ns) * 0.9, max(ns) * 1.1, 100)
        ax.plot(ns_fine, v + a / ns_fine, "k--", lw=1, label=f"v+a/n (v={v:.3f})")
        ax.axhline(v, color="gray", ls=":", lw=1, label=f"v={v:.3f}")

    ax.set_xlabel("n")
    ax.set_ylabel("n·Var(Xₙ)")
    ax.set_title(f"{obj}, train_n={tn}, p={p}")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # --- Panel 2: mean Xₙ vs n ---
    ax = axes[1]
    mean_xs = [per_n[n]["mean_X"] for n in ns]
    ax.plot(ns, mean_xs, "o-")
    c_slope = result["c_typ_slope"]
    c_int = result["c_typ_intercept"]
    if math.isfinite(c_slope):
        ns_arr = np.array(ns, dtype=float)
        ax.plot(ns_arr, c_slope * ns_arr + c_int, "k--", lw=1,
                label=f"slope={c_slope:.4f}")
        ax.legend(fontsize=7)
    ax.set_xlabel("n")
    ax.set_ylabel("mean Xₙ  = −(1/n)⟨log₂ p⟩")
    ax.set_title("c_typ check")
    ax.grid(True, alpha=0.3)

    # --- Panel 3: skewness vs n ---
    ax = axes[2]
    skews = [per_n[n]["skewness"] for n in ns]
    ax.plot(ns, skews, "o-")
    ax.axhline(0, color="gray", ls="--", lw=1)
    ax.set_xlabel("n")
    ax.set_ylabel("skewness(Xₙ)")
    ax.set_title("Skewness")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fname = f"variance_{obj}_tn{tn}_p{p}.png"
    fig.savefig(out_dir / fname, dpi=130)
    plt.close(fig)


def _plot_overview(
    all_results: Dict[Tuple[str, int, int], dict],
    out_dir: Path,
    *,
    objective: str,
    train_n: int,
) -> None:
    """One plot per (objective, train_n): n·Var vs n for all p on one axes."""
    if not HAS_MPL:
        return
    keys = [(obj, tn, p) for (obj, tn, p) in all_results if obj == objective and tn == train_n]
    if not keys:
        return
    keys.sort(key=lambda k: k[2])

    fig, ax = plt.subplots(figsize=(7, 5))
    for key in keys:
        p = key[2]
        per_n = all_results[key]["per_n"]
        ns = sorted(per_n.keys())
        nvar = [per_n[n]["nVar_X"] for n in ns]
        nerr = [per_n[n]["nVar_X_err"] for n in ns]
        ax.errorbar(ns, nvar, yerr=nerr, fmt="o-", label=f"p={p}", capsize=3)

    ax.set_xlabel("n")
    ax.set_ylabel("n·Var(Xₙ)")
    ax.set_title(f"{objective}, train_n={train_n} — variance plateau?")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fname = f"variance_overview_{objective}_tn{train_n}.png"
    fig.savefig(out_dir / fname, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    default_rows = bm24_runs_dir() / "variance_scaling" / "variance_rows.jsonl"
    default_out = bm24_runs_dir() / "variance_scaling" / "analysis"

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=Path, default=default_rows)
    parser.add_argument("--out-dir", type=Path, default=default_out)
    parser.add_argument(
        "--trim-frac", type=float, default=0.05,
        help="Fraction of top-X instances trimmed for bulk variance"
    )
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.rows} ...")
    rows = load_rows(args.rows)
    print(f"  {len(rows)} rows loaded.")

    grouped = group_rows(rows)
    all_results: Dict[Tuple[str, int, int], dict] = {}

    print("\n=== Per-(objective, train_n, p) analysis ===\n")
    for key in sorted(grouped.keys(), key=lambda k: (k[0], k[1], k[2])):
        obj, tn, p = key
        result = analyze_combo(grouped[key], trim_frac=args.trim_frac)
        all_results[key] = result

        ric = result["richardson"]
        v, a, r2 = ric["v"], ric["a"], ric["r2"]
        c_typ_from_v = result["c_typ_from_v"]
        c_slope = result["c_typ_slope"]

        ns = sorted(grouped[key].keys())
        nvar_row = " ".join(
            f"n={n}: {result['per_n'][n]['nVar_X']:.4f}±{result['per_n'][n]['nVar_X_err']:.4f}"
            for n in ns
        )
        print(f"[{obj}, train_n={tn}, p={p}]")
        print(f"  n·Var: {nvar_row}")
        print(f"  Richardson: v={v:.4f}, a={a:.4f}, R²={r2:.4f}")
        print(f"  c_typ slope={c_slope:.4f}")
        print(f"  Verdict: c_typ_from_v = c_ann + (ln2/2)·v = {c_typ_from_v:.4f}")
        print()

        if not args.no_plots:
            _plot_combo(key, result, out_dir)

    # Overview plots per (objective, train_n)
    if not args.no_plots:
        seen_combos = set()
        for (obj, tn, _) in all_results:
            if (obj, tn) not in seen_combos:
                seen_combos.add((obj, tn))
                _plot_overview(all_results, out_dir, objective=obj, train_n=tn)

    # Save summary JSON
    summary: dict = {}
    for (obj, tn, p), result in all_results.items():
        k_str = f"{obj}|{tn}|{p}"
        ric = result["richardson"]
        summary[k_str] = {
            "v": ric["v"],
            "a": ric["a"],
            "r2": ric["r2"],
            "c_typ_slope": result["c_typ_slope"],
            "c_typ_from_v": result["c_typ_from_v"],
            "per_n": {
                str(n): {
                    "N": d["N"],
                    "mean_X": d["mean_X"],
                    "nVar_X": d["nVar_X"],
                    "nVar_X_err": d["nVar_X_err"],
                    "nVar_X_bulk": d["nVar_X_bulk"],
                    "skewness": d["skewness"],
                }
                for n, d in result["per_n"].items()
            },
        }

    summary_path = out_dir / "variance_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {summary_path}")
    if not args.no_plots:
        print(f"Plots written to {out_dir}/")


if __name__ == "__main__":
    main()
