#!/usr/bin/env python3
"""Plot run1 vs run4 with both eval windows (n=12-18 and n=14-18)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress

_PHASECRAFT = Path(__file__).resolve().parents[2]
_REPO = _PHASECRAFT.parent
_LR = Path(__file__).resolve().parent
for _p in (_REPO, _PHASECRAFT, _LR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from benchmark_dataset_cache import load_or_build_benchmark_dataset  # noqa: E402
from phasecraft.lib.sim.bm24_qaoa_sim import (  # noqa: E402
    evaluate_qaoa_runtime_on_dataset,
    generate_benchmark_dataset,
    make_lr_angles,
)

LN2 = float(np.log(2))
MODE_KEYS = {
    "mean_p": "bm24_mean_p_fixed_n",
    "median_rt": "median_runtime_fixed_n",
}
WINDOWS: Tuple[Tuple[int, int], ...] = ((12, 18), (14, 18))


def fit_log2_slope(ns: Sequence[int], ys: Sequence[float]) -> float:
    n_arr = np.asarray(list(ns), dtype=float)
    y_arr = np.asarray(list(ys), dtype=float)
    mask = np.isfinite(y_arr) & (y_arr > 0)
    if int(mask.sum()) < 2:
        return float("nan")
    return float(linregress(n_arr[mask], np.log(y_arr[mask])).slope / LN2)


def _traces(payload: dict) -> Dict[str, List[dict]]:
    out: Dict[str, List[dict]] = {}
    for key in ("trace_bm24_mean_p_fixed_n", "trace_median_runtime_fixed_n"):
        if key in payload and payload[key]:
            out[key.replace("trace_", "")] = list(payload[key])
    return out


def slope_from_per_n(per_n: dict, n_lo: int, n_hi: int) -> float:
    ns = [n for n in range(n_lo, n_hi + 1)]
    ys = [float(per_n.get(str(n), per_n.get(n))) for n in ns]
    return fit_log2_slope(ns, ys)


def rebenchmark_run1(
    run_json: Path,
    *,
    cache_path: Path,
    test_size: int,
) -> dict:
    payload = json.loads(run_json.read_text(encoding="utf-8"))
    cfg = payload["config"]
    traces = _traces(payload)
    n_values = list(range(int(cfg["n_min"]), int(cfg["n_max"]) + 1))
    run_dir = run_json.parent

    if cache_path.is_file():
        out = json.loads(cache_path.read_text(encoding="utf-8"))
        if out.get("test_size") == test_size and out.get("complete"):
            return out
    else:
        out = {"source": str(run_json), "test_size": test_size, "complete": False, "modes": {}}

    dataset = load_or_build_benchmark_dataset(
        run_dir=run_dir,
        n_values=n_values,
        k=int(cfg["k"]),
        r=float(cfg["r"]),
        test_size=test_size,
        base_seed=int(cfg["seed"]),
        use_cache=True,
        build_fn=lambda: generate_benchmark_dataset(
            n_values,
            int(cfg["k"]),
            float(cfg["r"]),
            test_size,
            int(cfg["seed"]),
            require_sat=True,
        ),
    )
    print(f"Dataset ready (n={n_values[0]}..{n_values[-1]}, test_size={test_size})", flush=True)

    for mode_label, trace_key in MODE_KEYS.items():
        rows_out: List[dict] = list(out["modes"].get(mode_label, []))
        done_depths = {int(r["depth"]) for r in rows_out}
        for row in traces[trace_key]:
            depth = int(row["depth"])
            if depth in done_depths:
                continue
            print(f"  rebenchmark run1 {mode_label} p={depth} ...", flush=True)
            betas, gammas = make_lr_angles(
                delta_gamma=float(row["delta_gamma"]),
                delta_beta=float(row["delta_beta"]),
                depth=depth,
                beta_schedule=cfg.get("lr_beta_schedule", "decreasing"),
                angle_convention="bm24",
            )
            ev = evaluate_qaoa_runtime_on_dataset(dataset, betas, gammas)
            med_rt = {int(n): float(d["median_runtime"]) for n, d in ev["per_n"].items()}
            rows_out.append({"depth": depth, "median_runtime_per_n": med_rt})
            rows_out.sort(key=lambda r: int(r["depth"]))
            out["modes"][mode_label] = rows_out
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    out["complete"] = True
    cache_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def series_from_run4(payload: dict) -> Dict[str, Dict[Tuple[int, int], Dict[int, float]]]:
    traces = _traces(payload)
    out: Dict[str, Dict[Tuple[int, int], Dict[int, float]]] = {
        mode: {} for mode in MODE_KEYS
    }
    for mode_label, trace_key in MODE_KEYS.items():
        for row in traces[trace_key]:
            per_n = row.get("median_runtime_per_n")
            if not per_n:
                continue
            depth = int(row["depth"])
            for n_lo, n_hi in WINDOWS:
                out[mode_label].setdefault((n_lo, n_hi), {})[depth] = slope_from_per_n(
                    per_n, n_lo, n_hi
                )
    return out


def series_from_run1(
    payload: dict,
    rebench: dict,
) -> Dict[str, Dict[Tuple[int, int], Dict[int, float]]]:
    traces = _traces(payload)
    out: Dict[str, Dict[Tuple[int, int], Dict[int, float]]] = {
        mode: {} for mode in MODE_KEYS
    }
    reb_by_mode = rebench.get("modes", {})
    for mode_label, trace_key in MODE_KEYS.items():
        reb_rows = {int(r["depth"]): r for r in reb_by_mode.get(mode_label, [])}
        for row in traces[trace_key]:
            depth = int(row["depth"])
            for n_lo, n_hi in WINDOWS:
                if n_lo == 12 and n_hi == 18:
                    slope = float(row.get("lr_log2_slope", float("nan")))
                else:
                    reb = reb_rows.get(depth)
                    if reb and reb.get("median_runtime_per_n"):
                        slope = slope_from_per_n(reb["median_runtime_per_n"], n_lo, n_hi)
                    else:
                        slope = float("nan")
                out[mode_label].setdefault((n_lo, n_hi), {})[depth] = slope
    return out


def plot_compare(
    run1_series: Dict[str, Dict[Tuple[int, int], Dict[int, float]]],
    run4_series: Dict[str, Dict[Tuple[int, int], Dict[int, float]]],
    *,
    out_path: Path,
) -> None:
    depths = sorted(
        set().union(
            *[
                set(d for win in s.values() for d in win)
                for s in [run1_series["mean_p"], run4_series["mean_p"]]
            ]
        )
    )
    styles = {
        (12, (12, 18)): dict(color="#4C72B0", ls="-", marker="o", lw=2),
        (12, (14, 18)): dict(color="#DD8452", ls="--", marker="o", lw=2),
        (16, (12, 18)): dict(color="#55A868", ls="-", marker="s", lw=2),
        (16, (14, 18)): dict(color="#C44E52", ls="--", marker="s", lw=2),
    }
    labels = {
        (12, (12, 18)): "train_n=12 (n=12–18)",
        (12, (14, 18)): "train_n=12 (n=14–18)",
        (16, (12, 18)): "train_n=16 (n=12–18)",
        (16, (14, 18)): "train_n=16 (n=14–18)",
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, mode in zip(axes, ("mean_p", "median_rt")):
        for train_n, src in ((12, run1_series), (16, run4_series)):
            for window in WINDOWS:
                ys = [src[mode].get(window, {}).get(d, float("nan")) for d in depths]
                st = styles[(train_n, window)]
                ax.plot(
                    depths,
                    ys,
                    label=labels[(train_n, window)],
                    **st,
                )
        ax.set_title(mode)
        ax.set_xlabel("QAOA depth p")
        ax.set_xticks(depths)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Eval: log₂ slope of median(1/p_succ) vs n")
    axes[0].legend(fontsize=8, loc="upper right")
    fig.suptitle(
        "run1 vs run4 — both eval windows (n=12–18 and n=14–18)\n"
        "train_size=100 · test=200 · seed=27",
        fontsize=11,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--run1",
        type=Path,
        default=Path("results/bm24_runs/06-09/run1/train12-tr100-te200-n12-18.json"),
    )
    p.add_argument(
        "--run4",
        type=Path,
        default=Path("results/bm24_runs/06-09/run4/train16-tr100-te200-n14-18.json"),
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("results/bm24_runs/06-09/compare-run1-vs-run4-corrected.png"),
    )
    p.add_argument("--test-size", type=int, default=200)
    p.add_argument(
        "--rebench-cache",
        type=Path,
        default=Path("results/bm24_runs/06-09/run1/train12-tr100-te200-n12-18-per_n-all.json"),
    )
    args = p.parse_args()

    run1_path = (_PHASECRAFT / args.run1).resolve() if not args.run1.is_absolute() else args.run1
    run4_path = (_PHASECRAFT / args.run4).resolve() if not args.run4.is_absolute() else args.run4
    out_path = (_PHASECRAFT / args.output).resolve() if not args.output.is_absolute() else args.output
    cache_path = (
        (_PHASECRAFT / args.rebench_cache).resolve()
        if not args.rebench_cache.is_absolute()
        else args.rebench_cache
    )

    run1_payload = json.loads(run1_path.read_text(encoding="utf-8"))
    run4_payload = json.loads(run4_path.read_text(encoding="utf-8"))

    print(f"Re-benchmarking run1 (cache: {cache_path.name})...")
    rebench = rebenchmark_run1(run1_path, cache_path=cache_path, test_size=args.test_size)

    run1_series = series_from_run1(run1_payload, rebench)
    run4_series = series_from_run4(run4_payload)

    plot_compare(run1_series, run4_series, out_path=out_path)

    mirror = _PHASECRAFT / "bm24_runs" / "06-09" / out_path.name
    if mirror.resolve() != out_path.resolve():
        plot_compare(run1_series, run4_series, out_path=mirror)


if __name__ == "__main__":
    main()
