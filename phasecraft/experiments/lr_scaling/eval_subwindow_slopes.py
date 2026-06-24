#!/usr/bin/env python3
"""
Sub-window slope analysis — no new QAOA evaluation needed.

Loads existing per_n median-runtime data (from the rebenchmark cache produced
by plot_compare_run1_run4.py) and refits the c_typ slope over rolling
sub-windows of the n range.

If c_typ is stable across all sub-windows, the n=12-18 estimate is reliable.
If it drops monotonically from left to right (small windows give higher slopes),
the original estimate is inflated by finite-n bias at small n.

Example::

    python experiments/lr_scaling/eval_subwindow_slopes.py

    python experiments/lr_scaling/eval_subwindow_slopes.py \\
        --per-n-cache results/bm24_runs/06-09/run1/train12-tr100-te200-n12-18-per_n-all.json \\
        --window-size 4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress

_PHASECRAFT = Path(__file__).resolve().parents[2]
for _p in (_PHASECRAFT.parent, _PHASECRAFT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

LN2 = float(np.log(2))

_MODE_LABEL = {
    "bm24_mean_p_fixed_n": "mean_p",
    "median_runtime_fixed_n": "median_rt",
}
_MODE_COLOR = {
    "bm24_mean_p_fixed_n": "#4C72B0",
    "median_runtime_fixed_n": "#C44E52",
}

_DEFAULT_CACHE = (
    _PHASECRAFT / "results/bm24_runs/06-09/run1"
    / "train12-tr100-te200-n12-18-per_n-all.json"
)


def _fit_log2_slope(ns: List[int], ys: List[float]) -> float:
    n_arr = np.asarray(ns, dtype=float)
    y_arr = np.asarray(ys, dtype=float)
    mask = np.isfinite(y_arr) & (y_arr > 0)
    if int(mask.sum()) < 2:
        return float("nan")
    return float(linregress(n_arr[mask], np.log(y_arr[mask])).slope / LN2)


def load_per_n_cache(cache_path: Path) -> Dict[str, List[dict]]:
    """Return {mode_label: [{depth, median_runtime_per_n}, ...]}."""
    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    modes = payload.get("modes", {})
    out: Dict[str, List[dict]] = {}
    for mode_label, rows in modes.items():
        out[mode_label] = sorted(rows, key=lambda r: int(r["depth"]))
    return out


def subwindow_slopes(
    per_n: Dict[str, float],  # {n_str: median_runtime}
    window_size: int,
) -> List[Tuple[Tuple[int, int], float]]:
    """Fit slope over every contiguous sub-window of length window_size."""
    all_ns = sorted(int(k) for k in per_n)
    results: List[Tuple[Tuple[int, int], float]] = []
    for i in range(len(all_ns) - window_size + 1):
        win_ns = all_ns[i : i + window_size]
        win_ys = [float(per_n[str(n)]) for n in win_ns]
        slope = _fit_log2_slope(win_ns, win_ys)
        results.append(((win_ns[0], win_ns[-1]), slope))
    return results


def analyze(
    modes_data: Dict[str, List[dict]],
    window_size: int,
) -> Dict[str, List[dict]]:
    """
    Return {mode: [{depth, full_slope, subwindow_slopes: [(lo,hi,slope), ...]}]}.
    """
    out: Dict[str, List[dict]] = {}
    for mode_label, rows in modes_data.items():
        out[mode_label] = []
        for row in rows:
            depth = int(row["depth"])
            per_n = row.get("median_runtime_per_n", {})
            if not per_n:
                continue
            all_ns = sorted(int(k) for k in per_n)
            all_ys = [float(per_n[str(n)]) for n in all_ns]
            full_slope = _fit_log2_slope(all_ns, all_ys)
            subs = subwindow_slopes(per_n, window_size)
            out[mode_label].append({
                "depth": depth,
                "full_slope": full_slope,
                "all_ns": all_ns,
                "subwindow_slopes": [
                    {"n_lo": lo, "n_hi": hi, "slope": s}
                    for (lo, hi), s in subs
                ],
            })
    return out


def plot_subwindow(
    analysis: Dict[str, List[dict]],
    *,
    out_path: Path,
    window_size: int,
    run_label: str = "",
) -> None:
    modes = list(analysis.keys())
    n_modes = len(modes)
    fig, axes = plt.subplots(1, n_modes, figsize=(7 * n_modes, 5), squeeze=False)

    for col, mode in enumerate(modes):
        ax = axes[0][col]
        rows = analysis[mode]
        if not rows:
            continue

        color = _MODE_COLOR.get(mode, "#666666")
        mode_label = _MODE_LABEL.get(mode, mode)
        depths = [r["depth"] for r in rows]

        # For each sub-window, plot c_typ(p) across depths
        # First collect all unique sub-windows
        if not rows[0]["subwindow_slopes"]:
            continue
        sub_windows = [(s["n_lo"], s["n_hi"]) for s in rows[0]["subwindow_slopes"]]

        cmap = plt.cm.Blues_r if "mean" in mode else plt.cm.Reds_r
        n_wins = len(sub_windows)
        alphas = np.linspace(0.4, 1.0, n_wins)

        for wi, (n_lo, n_hi) in enumerate(sub_windows):
            slopes = []
            for row in rows:
                match = next(
                    (s["slope"] for s in row["subwindow_slopes"]
                     if s["n_lo"] == n_lo and s["n_hi"] == n_hi),
                    float("nan"),
                )
                slopes.append(match)
            is_full = (n_lo == rows[0]["all_ns"][0] and n_hi == rows[0]["all_ns"][-1])
            lw = 2.5 if is_full else 1.2
            ls = "-" if is_full else "--"
            ax.plot(
                depths, slopes,
                ls=ls, lw=lw,
                color=cmap(wi / max(n_wins - 1, 1)),
                label=f"n={n_lo}–{n_hi}" + (" (full)" if is_full else ""),
                alpha=alphas[wi],
            )

        ax.set_xlabel("QAOA depth p")
        ax.set_ylabel(r"$c_\mathrm{typ}$ (log₂ slope vs $n$)")
        ax.set_title(f"{mode_label} — window_size={window_size}")
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.3)

    all_ns_example = analysis[modes[0]][0]["all_ns"] if analysis[modes[0]] else []
    fig.suptitle(
        f"Sub-window slope stability test  [n ∈ {{{','.join(str(n) for n in all_ns_example)}}}]\n"
        f"Parallel lines = exponent stable; downward drift right → left = small-n inflation.\n"
        f"{run_label}",
        fontsize=10,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_stability_heatmap(
    analysis: Dict[str, List[dict]],
    *,
    out_path: Path,
    window_size: int,
) -> None:
    """Heatmap: row=depth, col=sub-window centre, value=slope. Flat = stable."""
    modes = list(analysis.keys())
    n_modes = len(modes)
    fig, axes = plt.subplots(1, n_modes, figsize=(6 * n_modes, 5), squeeze=False)

    for col, mode in enumerate(modes):
        ax = axes[0][col]
        rows = analysis[mode]
        if not rows:
            continue
        mode_label = _MODE_LABEL.get(mode, mode)

        depths = [r["depth"] for r in rows]
        if not rows[0]["subwindow_slopes"]:
            continue
        windows = [(s["n_lo"], s["n_hi"]) for s in rows[0]["subwindow_slopes"]]
        win_labels = [f"{lo}–{hi}" for lo, hi in windows]

        mat = np.full((len(depths), len(windows)), float("nan"))
        for ri, row in enumerate(rows):
            for wi, (n_lo, n_hi) in enumerate(windows):
                match = next(
                    (s["slope"] for s in row["subwindow_slopes"]
                     if s["n_lo"] == n_lo and s["n_hi"] == n_hi),
                    float("nan"),
                )
                mat[ri, wi] = match

        im = ax.imshow(mat, aspect="auto", cmap="RdYlGn_r", origin="upper")
        ax.set_xticks(range(len(windows)))
        ax.set_xticklabels(win_labels, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(len(depths)))
        ax.set_yticklabels([str(d) for d in depths])
        ax.set_xlabel("n sub-window")
        ax.set_ylabel("depth p")
        ax.set_title(f"{mode_label}")
        fig.colorbar(im, ax=ax, label=r"$c_\mathrm{typ}$")

        # Annotate cells
        for ri in range(len(depths)):
            for wi in range(len(windows)):
                v = mat[ri, wi]
                if np.isfinite(v):
                    ax.text(wi, ri, f"{v:.3f}", ha="center", va="center",
                            fontsize=6, color="black")

    fig.suptitle(
        f"Sub-window slope heatmap (window_size={window_size})\n"
        "Uniform columns = no finite-n bias; left-to-right drop = small-n inflates slope.",
        fontsize=10,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--per-n-cache", type=Path, default=_DEFAULT_CACHE,
        help="Path to per_n rebenchmark cache JSON (from plot_compare_run1_run4.py)",
    )
    ap.add_argument(
        "--window-size", type=int, default=4,
        help="Number of n values per sub-window (default 4)",
    )
    ap.add_argument(
        "--output-dir", type=Path, default=None,
    )
    args = ap.parse_args()

    cache_path = (
        (_PHASECRAFT / args.per_n_cache).resolve()
        if not args.per_n_cache.is_absolute()
        else args.per_n_cache.resolve()
    )
    if not cache_path.is_file():
        print(f"Cache not found: {cache_path}", file=sys.stderr)
        print("Run plot_compare_run1_run4.py first to build the per_n cache.", file=sys.stderr)
        sys.exit(1)

    out_dir = (args.output_dir or (
        _PHASECRAFT / "results/bm24_runs/analysis/subwindow_slopes"
    )).resolve()

    modes_data = load_per_n_cache(cache_path)
    print(f"Loaded {cache_path.name}")
    for mode, rows in modes_data.items():
        depths = [r["depth"] for r in rows]
        ns = sorted(int(k) for k in rows[0].get("median_runtime_per_n", {}).keys()) if rows else []
        print(f"  {mode}: {len(rows)} depths {depths},  n={ns}")

    result = analyze(modes_data, window_size=args.window_size)

    # Print summary table
    print(f"\nSub-window slopes (window_size={args.window_size})")
    for mode, rows in result.items():
        mode_label = _MODE_LABEL.get(mode, mode)
        print(f"\n{mode_label}:")
        if not rows:
            continue
        windows = [(s["n_lo"], s["n_hi"]) for s in rows[0]["subwindow_slopes"]]
        header = f"  {'p':>4}  {'full':>7}" + "".join(f"  {lo}-{hi}".rjust(9) for lo, hi in windows)
        print(header)
        for row in rows:
            line = f"  {row['depth']:>4}  {row['full_slope']:>7.4f}"
            for (lo, hi) in windows:
                s = next(
                    (x["slope"] for x in row["subwindow_slopes"] if x["n_lo"] == lo and x["n_hi"] == hi),
                    float("nan"),
                )
                line += f"  {s:>7.4f}"
            print(line)

        # Stability check: is there a monotone trend across windows at large depth?
        deep = [r for r in rows if r["depth"] >= 20]
        if deep and windows:
            r = deep[-1]
            sub_slopes = [
                next((x["slope"] for x in r["subwindow_slopes"]
                      if x["n_lo"] == lo and x["n_hi"] == hi), float("nan"))
                for lo, hi in windows
            ]
            valid = [s for s in sub_slopes if np.isfinite(s)]
            if valid:
                span = max(valid) - min(valid)
                trend = "STABLE" if span < 0.03 else f"DRIFTS by {span:.4f} (max-min)"
                print(f"  → at p={r['depth']}: sub-window spread = {span:.4f}  [{trend}]")

    run_label = cache_path.stem
    plot_subwindow(
        result,
        out_path=out_dir / f"subwindow_slopes_w{args.window_size}.png",
        window_size=args.window_size,
        run_label=run_label,
    )
    plot_stability_heatmap(
        result,
        out_path=out_dir / f"subwindow_heatmap_w{args.window_size}.png",
        window_size=args.window_size,
    )

    json_out = out_dir / f"subwindow_slopes_w{args.window_size}.json"
    json_out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Wrote {json_out}")


if __name__ == "__main__":
    main()
