#!/usr/bin/env python3
"""
Re-evaluate trained LR-QAOA angles on a wider / higher n window.

Takes an existing run JSON (with saved delta_gamma, delta_beta per depth),
generates a fresh benchmark dataset at a configurable n range, and recomputes
the c_typ scaling exponent.  Plots original vs new window side by side for
all available depths and both training modes.

This tests whether the n=12-18 slopes from training are finite-n artifacts or
reliable estimates of the asymptotic exponent.  If c_typ(p) shifts down when
evaluated at larger n, the original estimates were inflated by small-n effects.

Results are cached to disk so large runs can be interrupted and resumed.

Example::

    # Default: re-eval run1 on n=16-22 (100 instances each n)
    python experiments/lr_scaling/eval_wider_n_window.py \\
        results/bm24_runs/06-09/run1/train12-tr100-te200-n12-18.json

    # Higher n (slower — n=22 is ~4M state entries per instance)
    python experiments/lr_scaling/eval_wider_n_window.py RUN.json \\
        --n-min 18 --n-max 24 --test-size 50

    # Multiple runs on the same new window
    python experiments/lr_scaling/eval_wider_n_window.py run1.json run4.json \\
        --n-min 16 --n-max 22 --test-size 100

    # Skip depths you already have cached
    python experiments/lr_scaling/eval_wider_n_window.py RUN.json --depths 30,40,50
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress

_PHASECRAFT = Path(__file__).resolve().parents[2]
_LR = Path(__file__).resolve().parent
for _p in (_PHASECRAFT.parent, _PHASECRAFT, _LR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from benchmark_dataset_cache import (  # noqa: E402
    benchmark_dataset_cache_path,
    load_benchmark_dataset_clauses,
    save_benchmark_dataset_clauses,
)
from phasecraft.lib.sim.bm24_qaoa_sim import (  # noqa: E402
    build_h_diagonal,
    evaluate_qaoa_runtime_on_dataset,
    generate_benchmark_dataset,
    make_lr_angles,
)

LN2 = float(np.log(2))

_MODE_LABEL = {
    "bm24_mean_p_fixed_n": "mean_p",
    "median_runtime_fixed_n": "median_rt",
    "mean_log_runtime_fixed_n": "mean_log_rt",
}
_MODE_COLOR = {
    "bm24_mean_p_fixed_n": "#4C72B0",
    "median_runtime_fixed_n": "#C44E52",
    "mean_log_runtime_fixed_n": "#55A868",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _traces(payload: dict) -> Dict[str, List[dict]]:
    if "traces_by_mode" in payload:
        return {k: list(v) for k, v in payload["traces_by_mode"].items()}
    out: Dict[str, List[dict]] = {}
    for key in (
        "trace_bm24_mean_p_fixed_n",
        "trace_median_runtime_fixed_n",
        "trace_mean_log_runtime_fixed_n",
    ):
        if key in payload and payload[key]:
            out[key.replace("trace_", "")] = list(payload[key])
    return out


def _fit_log2_slope(ns: Sequence[int], ys: Sequence[float]) -> float:
    n_arr = np.asarray(list(ns), dtype=float)
    y_arr = np.asarray(list(ys), dtype=float)
    mask = np.isfinite(y_arr) & (y_arr > 0)
    if int(mask.sum()) < 2:
        return float("nan")
    return float(linregress(n_arr[mask], np.log(y_arr[mask])).slope / LN2)


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _cache_key(mode: str, depth: int, n_min: int, n_max: int, test_size: int) -> str:
    return f"{mode}|p{depth}|n{n_min}-{n_max}|te{test_size}"


def _load_cache(cache_path: Path) -> dict:
    if cache_path.is_file():
        try:
            return json.loads(cache_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _save_cache(cache_path: Path, cache: dict) -> None:
    # Merge with on-disk version to survive concurrent runs
    on_disk: dict = {}
    if cache_path.is_file():
        try:
            on_disk = json.loads(cache_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    merged = dict(on_disk)
    merged.update(cache)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(merged, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Per-n incremental dataset builder
# ---------------------------------------------------------------------------

def _per_n_cache_path(run_dir: Path, *, n: int, k: int, r: float,
                      test_size: int, base_seed: int) -> Path:
    return benchmark_dataset_cache_path(
        run_dir, k=k, r=r, base_seed=base_seed, test_size=test_size, n_values=[n]
    )


def _load_per_n(run_dir: Path, *, n: int, k: int, r: float,
                test_size: int, base_seed: int) -> Optional[List[dict]]:
    meta = {"version": 1, "k": k, "r": float(r), "seed": base_seed,
            "test_size": test_size, "n_values": [n]}
    p = _per_n_cache_path(run_dir, n=n, k=k, r=r, test_size=test_size, base_seed=base_seed)
    ds = load_benchmark_dataset_clauses(p, meta=meta)
    return ds[n] if ds and n in ds else None


def _save_per_n(run_dir: Path, *, n: int, instances: List[dict],
                k: int, r: float, test_size: int, base_seed: int) -> None:
    meta = {"version": 1, "k": k, "r": float(r), "seed": base_seed,
            "test_size": test_size, "n_values": [n]}
    p = _per_n_cache_path(run_dir, n=n, k=k, r=r, test_size=test_size, base_seed=base_seed)
    save_benchmark_dataset_clauses(p, meta=meta, dataset={n: instances})


def _build_dataset_per_n(
    *,
    run_dir: Path,
    n_values: List[int],
    k: int,
    r: float,
    test_size: int,
    base_seed: int,
) -> Dict[int, List[dict]]:
    """Build (and incrementally cache) one n at a time so partial results survive."""
    dataset: Dict[int, List[dict]] = {}
    for n in sorted(n_values):
        cached = _load_per_n(run_dir, n=n, k=k, r=r, test_size=test_size, base_seed=base_seed)
        if cached is not None:
            print(f"  dataset n={n}: loaded from cache ({len(cached)} instances)", flush=True)
            dataset[n] = cached
            continue
        print(f"  dataset n={n}: generating {test_size} SAT instances "
              f"(k={k}, r={r}, seed={base_seed}) ...", flush=True)
        t0 = time.time()
        raw = generate_benchmark_dataset([n], k, r, test_size, base_seed, require_sat=True)
        instances = raw[n]
        elapsed = time.time() - t0
        print(f"    done in {elapsed:.1f}s", flush=True)
        _save_per_n(run_dir, n=n, instances=instances, k=k, r=r,
                    test_size=test_size, base_seed=base_seed)
        dataset[n] = instances
    return dataset


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def eval_run_on_window(
    run_json: Path,
    *,
    n_min: int,
    n_max: int,
    test_size: int,
    modes: Optional[List[str]] = None,
    depths_filter: Optional[List[int]] = None,
    cache_path: Optional[Path] = None,
) -> dict:
    """Return slope data for a run re-evaluated on [n_min, n_max]."""
    payload = json.loads(run_json.read_text(encoding="utf-8"))
    cfg = payload["config"]
    traces = _traces(payload)
    k, r = int(cfg["k"]), float(cfg["r"])
    seed = int(cfg["seed"])
    beta_schedule = cfg.get("lr_beta_schedule", "decreasing")
    n_values = list(range(n_min, n_max + 1))

    if cache_path is None:
        window_tag = f"n{n_min}-{n_max}-te{test_size}"
        cache_path = run_json.parent / f"{run_json.stem}-wider-{window_tag}.json"

    cache = _load_cache(cache_path)

    dataset: Optional[Dict[int, List[dict]]] = None

    rows: List[dict] = []
    for mode, trace in traces.items():
        if modes and mode not in modes:
            continue
        for row in sorted(trace, key=lambda r: int(r["depth"])):
            depth = int(row["depth"])
            if depths_filter is not None and depth not in depths_filter:
                continue

            ckey = _cache_key(mode, depth, n_min, n_max, test_size)
            if ckey in cache:
                rows.append(cache[ckey])
                print(f"  cache {_MODE_LABEL.get(mode, mode)} p={depth}", flush=True)
                continue

            if dataset is None or not all(n in dataset for n in n_values):
                dataset = _build_dataset_per_n(
                    run_dir=run_json.parent,
                    n_values=n_values,
                    k=k,
                    r=r,
                    test_size=test_size,
                    base_seed=seed,
                )

            dg, db = float(row["delta_gamma"]), float(row["delta_beta"])
            betas, gammas = make_lr_angles(
                delta_gamma=dg,
                delta_beta=db,
                depth=depth,
                beta_schedule=beta_schedule,
                angle_convention="bm24",
            )

            t0 = time.time()
            print(f"  eval {_MODE_LABEL.get(mode, mode)} p={depth} ...", flush=True)
            ev = evaluate_qaoa_runtime_on_dataset(dataset, betas, gammas)
            ns = sorted(int(n) for n in ev["per_n"])
            med_rts = [float(ev["per_n"][n]["median_runtime"]) for n in ns]
            mean_ps = [float(ev["per_n"][n]["mean_success"]) for n in ns]
            c_typ = _fit_log2_slope(ns, med_rts)
            c_mean = _fit_log2_slope(ns, mean_ps)

            entry = {
                "mode": mode,
                "depth": depth,
                "delta_gamma": dg,
                "delta_beta": db,
                "ns": ns,
                "c_typ": c_typ,
                "c_emp_mean": c_mean,
                "median_rt_per_n": {int(n): float(v) for n, v in zip(ns, med_rts)},
                "mean_p_per_n": {int(n): float(v) for n, v in zip(ns, mean_ps)},
                "c_typ_original": float(row.get("lr_log2_slope", float("nan"))),
            }
            print(
                f"    c_typ={c_typ:.5f}  orig={entry['c_typ_original']:.5f}"
                f"  Δ={c_typ - entry['c_typ_original']:+.5f}  ({time.time()-t0:.1f}s)",
                flush=True,
            )
            cache[ckey] = entry
            if cache_path:
                _save_cache(cache_path, cache)
            rows.append(entry)

    return {
        "source": str(run_json),
        "config": cfg,
        "window": {"n_min": n_min, "n_max": n_max, "test_size": test_size},
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_comparison(
    results: List[dict],
    *,
    out_path: Path,
    modes: Optional[List[str]] = None,
) -> None:
    """Side-by-side: original window (stored) vs new window (re-eval), per mode."""
    all_modes = modes or sorted({r["mode"] for res in results for r in res["rows"]})
    n_modes = len(all_modes)
    fig, axes = plt.subplots(1, max(n_modes, 1), figsize=(6 * max(n_modes, 1), 4.8), squeeze=False)

    run_linestyles = ["-", "--", "-."]
    run_markers = ["o", "s", "^"]

    for col, mode in enumerate(all_modes):
        ax = axes[0][col]
        mode_label = _MODE_LABEL.get(mode, mode)
        base_color = _MODE_COLOR.get(mode, "#666666")

        for ri, res in enumerate(results):
            cfg = res["config"]
            run_label = (
                f"train_n={cfg.get('train_n')}, "
                f"n={cfg.get('n_min')}–{cfg.get('n_max')}"
            )
            w = res["window"]
            new_label = f"n={w['n_min']}–{w['n_max']} (new)"
            orig_label = f"n={cfg.get('n_min')}–{cfg.get('n_max')} (original)"

            rows_m = sorted([r for r in res["rows"] if r["mode"] == mode],
                            key=lambda r: int(r["depth"]))
            if not rows_m:
                continue

            depths = [int(r["depth"]) for r in rows_m]
            c_orig = [r.get("c_typ_original", float("nan")) for r in rows_m]
            c_new = [r.get("c_typ", float("nan")) for r in rows_m]
            delta = [cn - co for cn, co in zip(c_new, c_orig)]

            ls = run_linestyles[ri % len(run_linestyles)]
            mk = run_markers[ri % len(run_markers)]

            # Blend color slightly per run
            color_orig = base_color
            color_new = base_color

            ax.plot(depths, c_orig, marker=mk, color=color_orig, ls=":", lw=1.5, alpha=0.6,
                    label=f"{run_label} | {orig_label}")
            ax.plot(depths, c_new, marker=mk, color=color_new, ls=ls, lw=2,
                    label=f"{run_label} | {new_label}")

            # Annotate Δ at each depth
            for d, co, cn in zip(depths, c_orig, c_new):
                if np.isfinite(co) and np.isfinite(cn):
                    ax.annotate(
                        f"{cn-co:+.3f}",
                        (d, cn),
                        textcoords="offset points",
                        xytext=(0, 6),
                        fontsize=6,
                        ha="center",
                        color=color_new,
                        alpha=0.85,
                    )

        ax.set_xlabel("QAOA depth p")
        ax.set_ylabel(r"$c_\mathrm{typ}$ (log₂ slope vs $n$)")
        ax.set_title(f"{mode_label} training")
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.3)

    # Shared title
    new_w = results[0]["window"] if results else {}
    fig.suptitle(
        f"c_typ: original n window (dotted) vs wider window "
        f"n={new_w.get('n_min')}–{new_w.get('n_max')} (solid)\n"
        f"Annotations = Δ(new − orig). Downward shift = small-n upward bias.",
        fontsize=10,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_delta(results: List[dict], *, out_path: Path, modes: Optional[List[str]] = None) -> None:
    """Plot Δ c_typ (new − original) to isolate the n-window bias."""
    all_modes = modes or sorted({r["mode"] for res in results for r in res["rows"]})
    fig, ax = plt.subplots(figsize=(9, 4.5))

    run_markers = ["o", "s", "^"]
    for ri, res in enumerate(results):
        cfg = res["config"]
        w = res["window"]
        run_label = f"train_n={cfg.get('train_n')} | new n={w['n_min']}–{w['n_max']}"
        for mode in all_modes:
            rows_m = sorted([r for r in res["rows"] if r["mode"] == mode],
                            key=lambda r: int(r["depth"]))
            if not rows_m:
                continue
            color = _MODE_COLOR.get(mode, "#666666")
            mode_label = _MODE_LABEL.get(mode, mode)
            depths = [int(r["depth"]) for r in rows_m]
            deltas = [
                r.get("c_typ", float("nan")) - r.get("c_typ_original", float("nan"))
                for r in rows_m
            ]
            mk = run_markers[ri % len(run_markers)]
            ax.plot(depths, deltas, marker=mk, color=color, label=f"{run_label} | {mode_label}")

    ax.axhline(0, color="k", lw=0.8, alpha=0.4)
    ax.set_xlabel("QAOA depth p")
    ax.set_ylabel(r"$\Delta c_\mathrm{typ}$ = new window − original")
    ax.set_title(
        "n-window bias: negative = original n=12-18 overestimates the exponent\n"
        "Zero = exponent is stable; no finite-n inflation."
    )
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "run_jsons", nargs="*", type=Path,
        default=[Path("results/bm24_runs/06-09/run1/train12-tr100-te200-n12-18.json")],
    )
    ap.add_argument(
        "--n-min", type=int, default=16,
        help="Lower bound of new evaluation n range (default 16)",
    )
    ap.add_argument(
        "--n-max", type=int, default=22,
        help="Upper bound of new evaluation n range (default 22). "
             "n=22 has 4M state entries — increase test-size slowly.",
    )
    ap.add_argument(
        "--test-size", type=int, default=100,
        help="Instances per n for the new window (default 100)",
    )
    ap.add_argument(
        "--modes", default="bm24_mean_p_fixed_n,median_runtime_fixed_n",
    )
    ap.add_argument(
        "--depths", default=None,
        help="Comma-separated depths to evaluate (default: all in run JSON)",
    )
    ap.add_argument(
        "--output-dir", type=Path, default=None,
    )
    ap.add_argument(
        "--cache-dir", type=Path, default=None,
        help="Directory for per-run eval caches (default: beside each run JSON)",
    )
    args = ap.parse_args()

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    depths_filter = (
        [int(x) for x in args.depths.split(",") if x.strip()]
        if args.depths else None
    )
    run_jsons = [
        (_PHASECRAFT / p).resolve() if not p.is_absolute() else p.resolve()
        for p in args.run_jsons
    ]

    window_tag = f"n{args.n_min}-{args.n_max}-te{args.test_size}"
    out_dir = (args.output_dir or (
        _PHASECRAFT / "results/bm24_runs/analysis/wider_n_window"
    )).resolve()

    results: List[dict] = []
    for run_json in run_jsons:
        if not run_json.is_file():
            print(f"SKIP {run_json} (not found)", file=sys.stderr)
            continue

        if args.cache_dir:
            cache_path = (args.cache_dir / f"{run_json.stem}-wider-{window_tag}.json").resolve()
        else:
            cache_path = run_json.parent / f"{run_json.stem}-wider-{window_tag}.json"

        print(f"\n{'='*60}")
        print(f"Run: {run_json.name}")
        print(f"New window: n={args.n_min}..{args.n_max}, test_size={args.test_size}")
        print(f"Cache: {cache_path.name}")
        print(f"{'='*60}")

        res = eval_run_on_window(
            run_json,
            n_min=args.n_min,
            n_max=args.n_max,
            test_size=args.test_size,
            modes=modes,
            depths_filter=depths_filter,
            cache_path=cache_path,
        )
        results.append(res)

        # Print comparison table
        print(f"\n  {'mode':16s}  {'p':>4}  {'c_orig':>8}  {'c_new':>8}  {'delta':>8}")
        for r in sorted(res["rows"], key=lambda x: (x["mode"], x["depth"])):
            ml = _MODE_LABEL.get(r["mode"], r["mode"])
            print(
                f"  {ml:16s}  {r['depth']:>4}  "
                f"{r['c_typ_original']:>8.5f}  {r['c_typ']:>8.5f}  "
                f"{r['c_typ'] - r['c_typ_original']:>+8.5f}"
            )

    if not results:
        print("No results produced.", file=sys.stderr)
        sys.exit(1)

    # Save consolidated JSON
    json_out = out_dir / f"wider_n_{window_tag}_summary.json"
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nWrote {json_out}")

    # Plots
    plot_comparison(
        results,
        out_path=out_dir / f"wider_n_{window_tag}_comparison.png",
        modes=modes,
    )
    plot_delta(
        results,
        out_path=out_dir / f"wider_n_{window_tag}_delta.png",
        modes=modes,
    )


if __name__ == "__main__":
    main()
