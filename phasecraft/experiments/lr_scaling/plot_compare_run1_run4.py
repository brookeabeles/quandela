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

from benchmark_dataset_cache import (  # noqa: E402
    benchmark_dataset_cache_path,
    load_benchmark_clauses_only,
    load_benchmark_dataset_clauses,
    load_or_build_benchmark_dataset,
    save_benchmark_dataset_clauses,
)
from phasecraft.lib.sim.bm24_qaoa_sim import (  # noqa: E402
    build_h_diagonal,
    evaluate_qaoa_runtime_on_dataset,
    generate_benchmark_dataset,
    generate_random_formula,
    make_lr_angles,
)

LN2 = float(np.log(2))
MODE_KEYS = {
    "mean_p": "bm24_mean_p_fixed_n",
    "median_rt": "median_runtime_fixed_n",
}
WINDOWS: Tuple[Tuple[int, int], ...] = ((12, 18), (14, 18))
WINDOWS_NMAX20: Tuple[Tuple[int, int], ...] = ((12, 20), (14, 20))
FAIR_WINDOW = (14, 18)
RUN4_CANDIDATES = (
    Path("results/bm24_runs/06-09/run4/train16-tr100-te200-n14-18.json"),
    Path("results/bm24_runs/06-09/run4/train16-tr100-te200-n12-18.json"),
    Path("bm24_runs/06-09/run4/train16-tr100-te200-n14-18.json"),
    Path("bm24_runs/06-09/run4/train16-tr100-te200-n12-18.json"),
)
CLASSICAL_CANDIDATES = (
    Path("bm24_runs/06-10/run1/scaling-tn14.json"),
    Path("results/bm24_runs/06-10/run1/scaling-tn14.json"),
)


def _resolve_run_json(explicit: Path, candidates: Sequence[Path]) -> Path:
    tried: List[Path] = []
    for cand in (explicit, *candidates):
        p = (_PHASECRAFT / cand).resolve() if not cand.is_absolute() else cand.resolve()
        if p in tried:
            continue
        tried.append(p)
        if p.is_file():
            return p
    raise FileNotFoundError(
        "No run JSON found. Tried:\n  " + "\n  ".join(str(p) for p in tried)
    )


def fit_log2_slope(ns: Sequence[int], ys: Sequence[float]) -> float:
    slope, _stderr = fit_log2_slope_stderr(ns, ys)
    return slope


def fit_log2_slope_stderr(ns: Sequence[int], ys: Sequence[float]) -> Tuple[float, float]:
    """Return (log2 slope, 1σ stderr) from OLS on log(y) vs n."""
    n_arr = np.asarray(list(ns), dtype=float)
    y_arr = np.asarray(list(ys), dtype=float)
    mask = np.isfinite(y_arr) & (y_arr > 0)
    if int(mask.sum()) < 2:
        return float("nan"), float("nan")
    res = linregress(n_arr[mask], np.log(y_arr[mask]))
    slope = float(res.slope / LN2)
    stderr = float(res.stderr / LN2) if res.stderr is not None else float("nan")
    return slope, stderr


def _resolve_classical_json(explicit: Path | None) -> Path | None:
    if explicit is not None:
        p = (_PHASECRAFT / explicit).resolve() if not explicit.is_absolute() else explicit.resolve()
        return p if p.is_file() else None
    for cand in CLASSICAL_CANDIDATES:
        p = (_PHASECRAFT / cand).resolve()
        if p.is_file():
            return p
    return None


def load_classical_baselines(classical_json: Path | None) -> dict | None:
    """Median-flip classical baselines keyed by n (str), from a scaling JSON setup."""
    path = _resolve_classical_json(classical_json)
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    classical = payload.get("setup", {}).get("classical")
    return classical if classical else None


def classical_slopes(
    classical: dict,
    n_lo: int,
    n_hi: int,
) -> Tuple[float, float]:
    ns = list(range(n_lo, n_hi + 1))
    ws = [float(classical["walksat"][str(n)]) for n in ns]
    lm = [float(classical["walksatlm"][str(n)]) for n in ns]
    return fit_log2_slope(ns, ws), fit_log2_slope(ns, lm)


def _classical_legend_labels(
    ws_slope: float,
    lm_slope: float,
    window: Tuple[int, int],
    *,
    latex: bool = False,
) -> Tuple[str, str]:
    n_lo, n_hi = window
    if latex:
        n_rng = rf"$n{{\in}}[{n_lo},{n_hi}]$"
        ws = rf"WalkSAT ({ws_slope:.3f}, {n_rng})"
        lm = rf"WalkSATlm ({lm_slope:.3f}, {n_rng})"
    else:
        ws = f"WalkSAT ({ws_slope:.3f}, n={n_lo}–{n_hi})"
        lm = f"WalkSATlm ({lm_slope:.3f}, n={n_lo}–{n_hi})"
    return ws, lm


def add_classical_baselines(
    ax,
    classical: dict | None,
    window: Tuple[int, int],
) -> None:
    if not classical:
        return
    ws, lm = classical_slopes(classical, window[0], window[1])
    ws_label, lm_label = _classical_legend_labels(ws, lm, window, latex=False)
    if np.isfinite(ws):
        ax.axhline(
            ws,
            color="C1",
            linestyle=":",
            linewidth=1.5,
            label=ws_label,
        )
    if np.isfinite(lm):
        ax.axhline(
            lm,
            color="C2",
            linestyle=":",
            linewidth=1.5,
            label=lm_label,
        )


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


def slope_stderr_from_per_n(
    per_n: dict,
    n_lo: int,
    n_hi: int,
) -> Tuple[float, float]:
    ns = [n for n in range(n_lo, n_hi + 1)]
    ys = [float(per_n.get(str(n), per_n.get(n))) for n in ns]
    return fit_log2_slope_stderr(ns, ys)


# mode -> (train_n, window) -> depth -> (slope, stderr)
SeriesWithErr = Dict[str, Dict[Tuple[int, Tuple[int, int]], Dict[int, Tuple[float, float]]]]


def _normalize_per_n(per_n: dict) -> Dict[int, float]:
    return {int(k): float(v) for k, v in per_n.items()}


def _merge_per_n(base: dict, extra: dict) -> Dict[int, float]:
    merged = _normalize_per_n(base)
    merged.update(_normalize_per_n(extra))
    return merged


def _extend_dataset_to_test_size(
    instances: List[dict],
    *,
    n: int,
    k: int,
    r: float,
    test_size: int,
    base_seed: int,
) -> List[dict]:
    """Append SAT instances using the same SeedSequence scheme as generate_benchmark_dataset."""
    if len(instances) >= test_size:
        return instances[:test_size]
    out = list(instances)
    accepted = len(out)
    trial = 0
    max_rejections = 100000
    while accepted < int(test_size):
        if trial >= int(max_rejections):
            raise RuntimeError(
                f"Exceeded max_rejections={max_rejections} while extending dataset for n={n}"
            )
        seed_seq = np.random.SeedSequence(
            [int(base_seed), int(n), int(accepted), int(trial)]
        )
        rng = np.random.default_rng(seed_seq)
        clauses = generate_random_formula(n=n, k=k, r=r, rng=rng)
        H_diag = build_h_diagonal(clauses, n)
        num_solutions = int(np.sum(H_diag == 0))
        if num_solutions <= 0:
            trial += 1
            continue
        out.append({"clauses": clauses, "num_solutions": num_solutions})
        accepted += 1
        trial += 1
    return out


def _build_extra_n_dataset(
    *,
    run_dir: Path,
    extra_ns: Sequence[int],
    k: int,
    r: float,
    test_size: int,
    base_seed: int,
    fallback_test_size: int | None = 50,
) -> dict:
    """Load or build benchmark instances one n at a time (resumable)."""
    dataset: dict = {}
    meta_base = {
        "version": 1,
        "k": int(k),
        "r": float(r),
        "seed": int(base_seed),
        "test_size": int(test_size),
    }
    for n in sorted(int(x) for x in extra_ns):
        n_values = [n]
        meta = {**meta_base, "n_values": n_values}
        cache_path = benchmark_dataset_cache_path(
            run_dir,
            k=k,
            r=r,
            base_seed=base_seed,
            test_size=test_size,
            n_values=n_values,
        )
        loaded = load_benchmark_dataset_clauses(cache_path, meta=meta)
        if loaded is not None:
            print(f"  dataset n={n}: loaded te{test_size} cache", flush=True)
            dataset[n] = loaded[n]
            continue

        # Prefer an existing smaller test_size cache (same SeedSequence prefix).
        prefix_loaded: dict | None = None
        for smaller in sorted(
            {int(fallback_test_size)} if fallback_test_size else set()
            | {50, 60, 100},
            reverse=True,
        ):
            if smaller >= test_size:
                continue
            small_meta = {
                "version": 1,
                "k": int(k),
                "r": float(r),
                "seed": int(base_seed),
                "test_size": int(smaller),
                "n_values": n_values,
            }
            small_path = benchmark_dataset_cache_path(
                run_dir,
                k=k,
                r=r,
                base_seed=base_seed,
                test_size=smaller,
                n_values=n_values,
            )
            small_ds = load_benchmark_clauses_only(small_path, meta=small_meta)
            if small_ds is not None and n in small_ds:
                print(
                    f"  dataset n={n}: using te{smaller} cache "
                    f"({len(small_ds[n])} instances; same draw as te{test_size} prefix)",
                    flush=True,
                )
                dataset[n] = small_ds[n]
                prefix_loaded = small_ds
                break

        if prefix_loaded is not None:
            continue

        print(
            f"  dataset n={n}: generating {test_size} SAT instances ...",
            flush=True,
        )
        raw = generate_benchmark_dataset(
            n_values, int(k), float(r), test_size, int(base_seed), require_sat=True
        )
        save_benchmark_dataset_clauses(cache_path, meta=meta, dataset=raw)
        dataset[n] = raw[n]
        print(f"  dataset n={n}: saved -> {cache_path.name}", flush=True)
    return dataset


def _wider_n_scale_factors(wider_json: Path) -> Tuple[float, float]:
    """Median-runtime ratios n=19/18 and n=20/18 from wider-n te50 re-eval."""
    payload = json.loads(wider_json.read_text(encoding="utf-8"))
    r19: List[float] = []
    r20: List[float] = []
    for row in payload.values():
        per_n = row.get("median_rt_per_n", {})
        if "18" not in per_n:
            continue
        m18 = float(per_n["18"])
        if m18 <= 0:
            continue
        if "19" in per_n:
            r19.append(float(per_n["19"]) / m18)
        if "20" in per_n:
            r20.append(float(per_n["20"]) / m18)
    if not r19 or not r20:
        raise ValueError(f"No n=19,20 wider-n medians in {wider_json}")
    return float(np.mean(r19)), float(np.mean(r20))


def _augment_per_n_with_wider_ratios(
    per_n: dict,
    *,
    r19: float,
    r20: float,
) -> Dict[int, float]:
    merged = _normalize_per_n(per_n)
    if 18 not in merged:
        raise ValueError("per_n missing n=18 anchor for wider-n extrapolation")
    m18 = merged[18]
    merged[19] = m18 * r19
    merged[20] = m18 * r20
    return merged


def merged_per_n_for_run1_wider(
    rebench: dict,
    wider_json: Path,
) -> Dict[str, Dict[int, dict]]:
    r19, r20 = _wider_n_scale_factors(wider_json)
    out: Dict[str, Dict[int, dict]] = {m: {} for m in MODE_KEYS}
    for mode_label in MODE_KEYS:
        for row in rebench.get("modes", {}).get(mode_label, []):
            depth = int(row["depth"])
            per_n = row.get("median_runtime_per_n", {})
            if not per_n:
                continue
            merged = _augment_per_n_with_wider_ratios(per_n, r19=r19, r20=r20)
            out[mode_label][depth] = {"median_runtime_per_n": merged}
    return out


def merged_per_n_for_run4_wider(
    run4_payload: dict,
    wider_json: Path,
) -> Dict[str, Dict[int, dict]]:
    r19, r20 = _wider_n_scale_factors(wider_json)
    out: Dict[str, Dict[int, dict]] = {m: {} for m in MODE_KEYS}
    for mode_label, trace_key in MODE_KEYS.items():
        for row in _traces(run4_payload).get(trace_key, []):
            per_n = row.get("median_runtime_per_n")
            if not per_n:
                continue
            depth = int(row["depth"])
            merged = _augment_per_n_with_wider_ratios(per_n, r19=r19, r20=r20)
            out[mode_label][depth] = {"median_runtime_per_n": merged}
    return out


def eval_extra_per_n(
    payload: dict,
    *,
    run_dir: Path,
    extra_ns: Sequence[int],
    test_size: int,
    cache_path: Path,
    dataset_dir: Path | None = None,
) -> dict:
    """Evaluate stored angles at extra n values; cache merged per-n medians."""
    cfg = payload["config"]
    traces = _traces(payload)
    extra_ns = [int(n) for n in extra_ns]
    if cache_path.is_file():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if (
            cached.get("complete")
            and cached.get("test_size") == test_size
            and cached.get("extra_ns") == extra_ns
        ):
            return cached

    clause_dir = dataset_dir if dataset_dir is not None else run_dir
    dataset = _build_extra_n_dataset(
        run_dir=clause_dir,
        extra_ns=extra_ns,
        k=int(cfg["k"]),
        r=float(cfg["r"]),
        test_size=test_size,
        base_seed=int(cfg["seed"]),
    )
    print(
        f"  extra-n dataset ready n={extra_ns[0]}..{extra_ns[-1]} "
        f"(test_size={test_size})",
        flush=True,
    )

    out: dict = {
        "source": str(cache_path),
        "test_size": test_size,
        "extra_ns": extra_ns,
        "complete": False,
        "modes": {},
    }
    if cache_path.is_file():
        try:
            out = json.loads(cache_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass

    for mode_label, trace_key in MODE_KEYS.items():
        rows_out: List[dict] = list(out.get("modes", {}).get(mode_label, []))
        done_depths = {int(r["depth"]) for r in rows_out}
        for row in traces[trace_key]:
            depth = int(row["depth"])
            if depth in done_depths:
                continue
            print(f"  extra-n eval {mode_label} p={depth} ...", flush=True)
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
            out.setdefault("modes", {})[mode_label] = rows_out
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    out["complete"] = True
    cache_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def merged_per_n_for_run1(
    rebench: dict,
    extra_cache: dict | None,
) -> Dict[str, Dict[int, dict]]:
    """mode -> depth -> row with merged median_runtime_per_n (12..n_max)."""
    out: Dict[str, Dict[int, dict]] = {m: {} for m in MODE_KEYS}
    extra_by_mode = (extra_cache or {}).get("modes", {})
    for mode_label in MODE_KEYS:
        reb_rows = {int(r["depth"]): r for r in rebench.get("modes", {}).get(mode_label, [])}
        extra_rows = {int(r["depth"]): r for r in extra_by_mode.get(mode_label, [])}
        for depth, row in reb_rows.items():
            per_n = row.get("median_runtime_per_n", {})
            extra = extra_rows.get(depth, {}).get("median_runtime_per_n", {})
            merged = _merge_per_n(per_n, extra)
            out[mode_label][depth] = {"median_runtime_per_n": merged}
    return out


def merged_per_n_for_run4(
    run4_payload: dict,
    extra_cache: dict | None,
) -> Dict[str, Dict[int, dict]]:
    out: Dict[str, Dict[int, dict]] = {m: {} for m in MODE_KEYS}
    extra_by_mode = (extra_cache or {}).get("modes", {})
    for mode_label, trace_key in MODE_KEYS.items():
        extra_rows = {int(r["depth"]): r for r in extra_by_mode.get(mode_label, [])}
        for row in _traces(run4_payload).get(trace_key, []):
            depth = int(row["depth"])
            per_n = row.get("median_runtime_per_n", {})
            extra = extra_rows.get(depth, {}).get("median_runtime_per_n", {})
            merged = _merge_per_n(per_n, extra)
            out[mode_label][depth] = {"median_runtime_per_n": merged}
    return out


def build_series_with_stderr(
    run1_payload: dict,
    run4_payload: dict,
    rebench: dict,
    *,
    windows: Sequence[Tuple[int, int]] = WINDOWS,
    run1_per_n: Dict[str, Dict[int, dict]] | None = None,
    run4_per_n: Dict[str, Dict[int, dict]] | None = None,
) -> SeriesWithErr:
    """Slopes + regression stderr from per-n median(1/p) for every train/window."""
    out: SeriesWithErr = {m: {} for m in MODE_KEYS}
    if run1_per_n is None:
        run1_per_n = merged_per_n_for_run1(rebench, None)
    if run4_per_n is None:
        run4_per_n = merged_per_n_for_run4(run4_payload, None)

    for mode_label in MODE_KEYS:
        for window in windows:
            by_depth: Dict[int, Tuple[float, float]] = {}
            for depth, row in run1_per_n.get(mode_label, {}).items():
                per_n = row.get("median_runtime_per_n")
                if per_n:
                    by_depth[depth] = slope_stderr_from_per_n(per_n, window[0], window[1])
            if by_depth:
                out[mode_label][(12, window)] = by_depth

    for mode_label in MODE_KEYS:
        for depth, row in run4_per_n.get(mode_label, {}).items():
            per_n = row.get("median_runtime_per_n")
            if not per_n:
                continue
            for window in windows:
                out[mode_label].setdefault((16, window), {})[depth] = slope_stderr_from_per_n(
                    per_n, window[0], window[1]
                )
    return out


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


def plot_legacy_corrected(
    run1_series: Dict[str, Dict[Tuple[int, int], Dict[int, float]]],
    run4_series: Dict[str, Dict[Tuple[int, int], Dict[int, float]]],
    *,
    out_path: Path,
    classical: dict | None = None,
) -> None:
    """Original 3-line plot (run1 both windows + run4 n=14-18 only)."""
    depths = sorted(
        set(run1_series["mean_p"].get((12, 18), {}))
        & set(run4_series["mean_p"].get((14, 18), {}))
    )
    series = [
        (12, (12, 18), run1_series, dict(color="#8ebad9", ls=":", marker="o", lw=2),
         "train_n=12 (n=12–18)"),
        (12, (14, 18), run1_series, dict(color="#DD8452", ls="-", marker="o", lw=2),
         "train_n=12 (n=14–18 corrected)"),
        (16, (14, 18), run4_series, dict(color="#55A868", ls="--", marker="s", lw=2),
         "train_n=16 (n=14–18)"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, mode in zip(axes, ("mean_p", "median_rt")):
        for _tn, win, src, st, label in series:
            ys = [src[mode].get(win, {}).get(d, float("nan")) for d in depths]
            ax.plot(depths, ys, label=label, **st)
        add_classical_baselines(ax, classical, FAIR_WINDOW)
        ax.set_title(mode)
        ax.set_xlabel("QAOA depth p")
        ax.set_xticks(depths)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Eval: log₂ slope of median(1/p_succ) vs n")
    axes[0].legend(fontsize=7, loc="upper right")
    fig.suptitle(
        "run1 vs run4 — run1 re-evaluated on common n=14–18 window",
        fontsize=11,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def _finite_series(
    src: Dict[str, Dict[Tuple[int, int], Dict[int, float]]],
    mode: str,
    window: Tuple[int, int],
) -> Tuple[List[int], List[float]]:
    """Depth/value pairs with finite slopes only (keeps lines connected)."""
    dmap = src[mode].get(window, {})
    pts = sorted(
        (int(d), float(v))
        for d, v in dmap.items()
        if np.isfinite(v)
    )
    if not pts:
        return [], []
    depths, ys = zip(*pts)
    return list(depths), list(ys)


def _apply_paper_rcparams(*, compact: bool = False) -> None:
    size = 9 if compact else 10
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "dejavuserif",
            "font.size": size,
            "axes.labelsize": size + 1,
            "axes.titlesize": size + 1,
            "legend.fontsize": size - 0.5,
            "xtick.labelsize": size - 0.5,
            "ytick.labelsize": size - 0.5,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.6,
            "lines.markersize": 5.5,
            "savefig.dpi": 300,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.45,
        }
    )


def _reset_rcparams() -> None:
    import matplotlib as mpl
    mpl.rcdefaults()


# Color = train_n (Okabe–Ito); linestyle = eval window in supplement only.
_TRAIN_N_STYLE = {
    12: dict(color="#0072B2", marker="o"),
    16: dict(color="#D55E00", marker="s"),
}

_SUPP_WINDOW_STYLE = {
    (12, 18): dict(ls="-"),
    (14, 18): dict(ls="--"),
}


def _series_err_points(
    series_err: SeriesWithErr,
    mode: str,
    train_n: int,
    window: Tuple[int, int],
    *,
    depth_min: int = 1,
) -> Tuple[List[int], List[float], List[float]]:
    dmap = series_err.get(mode, {}).get((train_n, window), {})
    pts = sorted(
        (int(d), float(v[0]), float(v[1]))
        for d, v in dmap.items()
        if int(d) >= depth_min and np.isfinite(v[0])
    )
    if not pts:
        return [], [], []
    depths, slopes, errs = zip(*pts)
    errs = [e if np.isfinite(e) else 0.0 for e in errs]
    return list(depths), list(slopes), list(errs)


def _draw_baseline_refs(
    ax,
    ws_slope: float,
    lm_slope: float,
    *,
    label_right: bool = False,
    label_in_axes: bool = False,
    color: str = "0.45",
    linewidth: float = 0.75,
    label_offsets: Tuple[Tuple[float, str], Tuple[float, str]] | None = None,
) -> None:
    from matplotlib.transforms import blended_transform_factory

    entries = (
        (ws_slope, "-", "WalkSAT"),
        (lm_slope, "--", "WalkSATlm"),
    )
    for i, (y, ls, name) in enumerate(entries):
        if not np.isfinite(y):
            continue
        ax.axhline(y, color=color, linestyle=ls, linewidth=linewidth, zorder=2)
        dy, va = (0.0, "center")
        if label_offsets is not None:
            dy, va = label_offsets[i]
        if label_in_axes:
            trans = blended_transform_factory(ax.transAxes, ax.transData)
            ax.text(
                0.97,
                y + dy,
                name,
                transform=trans,
                va=va,
                ha="right",
                fontsize=8,
                color=color,
                clip_on=True,
            )
        elif label_right:
            ax.text(
                1.01,
                y + dy,
                name,
                transform=ax.get_yaxis_transform(),
                va=va,
                ha="left",
                fontsize=8,
                color=color,
            )


def _y_limits_from_data(
    series_err: SeriesWithErr,
    *,
    modes: Sequence[str],
    train_ns: Sequence[int],
    windows: Sequence[Tuple[int, int]],
    depth_min: int,
    baselines: Tuple[float, float],
    pad_scale: float = 1.0,
) -> Tuple[float, float]:
    ys: List[float] = []
    for mode in modes:
        for tn in train_ns:
            for win in windows:
                _, slopes, _ = _series_err_points(
                    series_err, mode, tn, win, depth_min=depth_min
                )
                ys.extend(slopes)
    for v in baselines:
        if np.isfinite(v):
            ys.append(v)
    if not ys:
        return 0.0, 1.0
    span = max(ys) - min(ys)
    pad = pad_scale * max(0.012, 0.06 * span)
    return min(ys) - pad, max(ys) + pad


def plot_paper_main(
    series_err: SeriesWithErr,
    *,
    out_path: Path,
    classical: dict | None = None,
    window: Tuple[int, int] = FAIR_WINDOW,
    depth_min: int = 5,
) -> None:
    """Two-column main-text figure: training objective panels, fair eval window only."""
    _apply_paper_rcparams(compact=True)

    ws_slope = lm_slope = float("nan")
    if classical:
        ws_slope, lm_slope = classical_slopes(classical, window[0], window[1])

    y_lo, y_hi = _y_limits_from_data(
        series_err,
        modes=("mean_p", "median_rt"),
        train_ns=(12, 16),
        windows=(window,),
        depth_min=depth_min,
        baselines=(ws_slope, lm_slope),
    )

    fig, axes = plt.subplots(
        1, 2, figsize=(7.2, 3.0), sharey=True, gridspec_kw={"wspace": 0.28}
    )
    panel_cfg = (
        ("mean_p", "(a)  Mean-success training"),
        ("median_rt", "(b)  Median-runtime training"),
    )

    legend_handles: List = []
    legend_labels: List[str] = []

    for ax, (mode, title) in zip(axes, panel_cfg):
        for train_n in (12, 16):
            xs, ys, yerr = _series_err_points(
                series_err, mode, train_n, window, depth_min=depth_min
            )
            if not xs:
                continue
            st = _TRAIN_N_STYLE[train_n]
            eb = ax.errorbar(
                xs,
                ys,
                yerr=yerr,
                fmt=f"{st['marker']}-",
                color=st["color"],
                mfc="white",
                mec=st["color"],
                mew=1.1,
                elinewidth=0.9,
                capsize=2.2,
                capthick=0.9,
                zorder=4,
                label=rf"train $n={train_n}$",
            )
            if mode == "mean_p":
                legend_handles.append(eb)
                legend_labels.append(rf"train $n={train_n}$")

        _draw_baseline_refs(
            ax, ws_slope, lm_slope, label_right=(ax is axes[-1])
        )
        ax.set_title(title, loc="left", pad=4, fontsize=10)
        ax.set_xlabel(r"QAOA depth $p$")
        xticks = sorted(
            {
                d
                for mode in ("mean_p", "median_rt")
                for tn in (12, 16)
                for d in series_err.get(mode, {}).get((tn, window), {})
                if int(d) >= depth_min
            }
        )
        ax.set_xticks(xticks)
        ax.set_ylim(y_lo, y_hi)
        ax.tick_params(direction="in", top=False, right=False)

    axes[0].set_ylabel(r"Median-runtime exponent $c_{\mathrm{med}}$")
    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        frameon=False,
        handlelength=2.2,
        columnspacing=1.5,
    )
    fig.subplots_adjust(left=0.11, right=0.90, top=0.82, bottom=0.22, wspace=0.28)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    _reset_rcparams()
    print(f"Wrote {out_path}")


def plot_paper_supplement(
    series_err: SeriesWithErr,
    *,
    out_path: Path,
    classical: dict | None = None,
    eval_window_for_baselines: Tuple[int, int] = (12, 18),
    depth_min: int = 5,
    fig_height: float = 3.2,
    y_pad_scale: float = 1.0,
    panel_wspace: float = 0.26,
    compact_legend: bool = False,
    baseline_labels_first_panel: bool = False,
    windows: Sequence[Tuple[int, int]] = WINDOWS,
    win_labels: Dict[Tuple[int, int], str] | None = None,
    suptitle: str | None = None,
) -> None:
    """Supplementary robustness figure: both train sizes and eval windows."""
    _apply_paper_rcparams(compact=True)

    if win_labels is None:
        win_labels = {
            (12, 18): r"$n_{\mathrm{eval}}{\in}[12,18]$",
            (14, 18): r"$n_{\mathrm{eval}}{\in}[14,18]$",
            (12, 20): r"$n_{\mathrm{eval}}{\in}[12,20]$",
            (14, 20): r"$n_{\mathrm{eval}}{\in}[14,20]$",
        }

    ws_slope = lm_slope = float("nan")
    if classical:
        ws_slope, lm_slope = classical_slopes(
            classical, eval_window_for_baselines[0], eval_window_for_baselines[1]
        )

    y_lo, y_hi = _y_limits_from_data(
        series_err,
        modes=("mean_p", "median_rt"),
        train_ns=(12, 16),
        windows=windows,
        depth_min=depth_min,
        baselines=(ws_slope, lm_slope),
        pad_scale=y_pad_scale,
    )

    baseline_color = "black" if baseline_labels_first_panel else "0.45"
    baseline_lw = 1.15 if baseline_labels_first_panel else 0.75

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.2, fig_height),
        sharey=True,
        gridspec_kw={"wspace": panel_wspace},
    )
    panel_cfg = (
        ("mean_p", "(a)  Mean-success training"),
        ("median_rt", "(b)  Median-runtime training"),
    )

    legend_handles: List = []
    legend_labels: List[str] = []

    window_styles = {
        **dict(_SUPP_WINDOW_STYLE),
        (12, 20): dict(ls="-"),
        (14, 20): dict(ls="--"),
    }

    for ax, (mode, title) in zip(axes, panel_cfg):
        for train_n in (12, 16):
            for win in windows:
                xs, ys, yerr = _series_err_points(
                    series_err, mode, train_n, win, depth_min=depth_min
                )
                if not xs:
                    continue
                st = {**_TRAIN_N_STYLE[train_n], **window_styles.get(win, dict(ls="-"))}
                lab = rf"train $n={train_n}$, {win_labels[win]}"
                eb = ax.errorbar(
                    xs,
                    ys,
                    yerr=yerr,
                    fmt=f"{st['marker']}{st['ls']}",
                    color=st["color"],
                    mfc="white",
                    mec=st["color"],
                    mew=1.0,
                    elinewidth=0.85,
                    capsize=2.0,
                    zorder=4,
                    label=lab,
                )
                if mode == "mean_p":
                    legend_handles.append(eb)
                    legend_labels.append(lab)

        _draw_baseline_refs(
            ax,
            ws_slope,
            lm_slope,
            label_right=(
                not baseline_labels_first_panel and ax is axes[-1]
            ),
            label_in_axes=(baseline_labels_first_panel and ax is axes[0]),
            color=baseline_color,
            linewidth=baseline_lw,
        )
        ax.set_title(title, loc="left", pad=4, fontsize=10)
        ax.set_xlabel(r"QAOA depth $p$")
        xticks = sorted(
            {
                d
                for mode in ("mean_p", "median_rt")
                for tn in (12, 16)
                for win in windows
                for d in series_err.get(mode, {}).get((tn, win), {})
                if int(d) >= depth_min
            }
        )
        ax.set_xticks(xticks)
        ax.set_ylim(y_lo, y_hi)
        ax.tick_params(direction="in", top=False, right=False)

    axes[0].set_ylabel(r"Median-runtime exponent $c_{\mathrm{med}}$")
    if suptitle:
        fig.suptitle(suptitle, fontsize=10, y=1.02 if compact_legend else 1.04)
    if compact_legend:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.96),
            ncol=2,
            frameon=False,
            fontsize=7.5,
            handlelength=2.4,
            columnspacing=1.0,
        )
        fig.subplots_adjust(
            left=0.11, right=0.98, top=0.88, bottom=0.12, wspace=panel_wspace
        )
        pad_inches = 0.02
    else:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.01),
            ncol=2,
            frameon=False,
            fontsize=7.5,
            handlelength=2.4,
            columnspacing=1.0,
        )
        fig.subplots_adjust(
            left=0.11, right=0.90, top=0.72, bottom=0.18, wspace=panel_wspace
        )
        pad_inches = 0.08
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        out_path, dpi=300, bbox_inches="tight", pad_inches=pad_inches, facecolor="white"
    )
    plt.close(fig)
    _reset_rcparams()
    print(f"Wrote {out_path}")


def write_paper_captions(out_dir: Path) -> None:
    main = """\
Figure X. Median runtime exponent c_med for random 8-SAT LR-QAOA as a function of depth p,
comparing angles trained by maximizing mean success probability (panel a) with angles trained
by minimizing median runtime (panel b). Curves show train n = 12 and train n = 16; exponents
are linear fits of log2(median_sigma(1/p_succ)) versus n over the evaluation window n in [14, 18]
(test size 200, seed 27). Error bars are 1 sigma from the log-linear regression over n.
Thin horizontal lines: WalkSAT and WalkSATlm median-flip baselines on the same n window.
The near-overlap of the two panels indicates that median-runtime training does not materially
change the depth-scaling exponent relative to BM24-style mean-success training; train n = 16
yields modestly lower exponents at large p (see Supplement).
"""
    supp = """\
Figure Sx. Robustness of c_med to training size and evaluation window. Same protocol as
Figure X, but with eval windows n in [12, 18] (solid) and n in [14, 18] (dashed); color
encodes train n. Shaded error bars as in the main figure. At p = 2 (not shown) exponents
are ~0.67 for all curves; the x-axis begins at p = 5 to resolve the high-depth regime.
"""
    (out_dir / "compare-run1-vs-run4-paper-main-caption.txt").write_text(
        main, encoding="utf-8"
    )
    (out_dir / "compare-run1-vs-run4-paper-supplement-caption.txt").write_text(
        supp, encoding="utf-8"
    )


def plot_all_windows(
    run1_series: Dict[str, Dict[Tuple[int, int], Dict[int, float]]],
    run4_series: Dict[str, Dict[Tuple[int, int], Dict[int, float]]],
    *,
    out_path: Path,
    classical: dict | None = None,
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

    fig, axes = plt.subplots(1, 2, figsize=(12, 7.5), sharey=True)
    for ax, mode in zip(axes, ("mean_p", "median_rt")):
        for train_n, src in ((12, run1_series), (16, run4_series)):
            for window in WINDOWS:
                xs, ys = _finite_series(src, mode, window)
                st = styles[(train_n, window)]
                ax.plot(
                    xs,
                    ys,
                    label=labels[(train_n, window)],
                    **st,
                )
        add_classical_baselines(ax, classical, (12, 18))
        ax.set_title(mode)
        ax.set_xlabel("QAOA depth p")
        ax.set_xticks(depths)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Eval: log₂ slope of median(1/p_succ) vs n")
    axes[0].legend(fontsize=7, loc="upper right")
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


def build_summary(
    run1_series: Dict[str, Dict[Tuple[int, int], Dict[int, float]]],
    run4_series: Dict[str, Dict[Tuple[int, int], Dict[int, float]]],
    *,
    run1_path: Path,
    run4_path: Path,
) -> dict:
    rows: List[dict] = []
    for mode in MODE_KEYS:
        for window in WINDOWS:
            d1 = run1_series[mode].get(window, {})
            d4 = run4_series[mode].get(window, {})
            for depth in sorted(set(d1) & set(d4)):
                v1, v4 = float(d1[depth]), float(d4[depth])
                rows.append({
                    "mode": mode,
                    "window": f"n={window[0]}–{window[1]}",
                    "depth": int(depth),
                    "run1_c_typ": v1,
                    "run4_c_typ": v4,
                    "delta_run4_minus_run1": v4 - v1,
                })
    fair = [r for r in rows if r["window"] == f"n={FAIR_WINDOW[0]}–{FAIR_WINDOW[1]}"]
    return {
        "run1": str(run1_path),
        "run4": str(run4_path),
        "fair_window": f"n={FAIR_WINDOW[0]}–{FAIR_WINDOW[1]}",
        "rows": rows,
        "fair_window_mean_abs_delta": {
            mode: float(np.nanmean([abs(r["delta_run4_minus_run1"]) for r in fair if r["mode"] == mode]))
            for mode in MODE_KEYS
        },
    }


def plot_train_n_delta(
    run1_series: Dict[str, Dict[Tuple[int, int], Dict[int, float]]],
    run4_series: Dict[str, Dict[Tuple[int, int], Dict[int, float]]],
    *,
    out_path: Path,
    window: Tuple[int, int] = FAIR_WINDOW,
) -> None:
    """run4 − run1 at the common eval window (train_n=16 vs 12)."""
    depths = sorted(
        set(run1_series["mean_p"].get(window, {}))
        & set(run4_series["mean_p"].get(window, {}))
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True)
    for ax, mode in zip(axes, ("mean_p", "median_rt")):
        delta = [
            run4_series[mode].get(window, {}).get(d, float("nan"))
            - run1_series[mode].get(window, {}).get(d, float("nan"))
            for d in depths
        ]
        ax.axhline(0, color="k", lw=0.8, alpha=0.35)
        ax.plot(depths, delta, "o-", color="#55A868" if mode == "mean_p" else "#C44E52")
        ax.set_xlabel("QAOA depth p")
        ax.set_ylabel("Δ c_typ (train_n=16 − 12)")
        ax.set_title(mode)
        ax.set_xticks(depths)
        ax.grid(True, alpha=0.3)
    fig.suptitle(
        f"Effect of train_n at fair eval window (n={window[0]}–{window[1]})",
        fontsize=11,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_with_theory(
    run1_series: Dict[str, Dict[Tuple[int, int], Dict[int, float]]],
    run4_series: Dict[str, Dict[Tuple[int, int], Dict[int, float]]],
    *,
    run1_payload: dict,
    run4_payload: dict,
    theory_cache_path: Path,
    out_path: Path,
    window: Tuple[int, int] = FAIR_WINDOW,
    theory_max_depth: int = 10,
) -> None:
    """Overlay −c_ann (BM24 saddle) on c_typ for both runs at shared depths."""
    from plot_ctyp_vs_cann import (  # noqa: WPS433
        _merge_cache_on_disk,
        _theory_cache_valid,
        _theory_with_fallback,
    )

    cache: dict = {}
    if theory_cache_path.is_file():
        cache = json.loads(theory_cache_path.read_text(encoding="utf-8"))
    cfg = run1_payload["config"]
    k, r = int(cfg["k"]), float(cfg["r"])
    beta_schedule = cfg.get("lr_beta_schedule", "decreasing")

    def theory_for(payload: dict, run_label: str, depth: int) -> float:
        traces = _traces(payload)
        row = next(tr for tr in traces["bm24_mean_p_fixed_n"] if int(tr["depth"]) == depth)
        legacy_key = f"theory|bm24_mean_p_fixed_n|{depth}"
        key = f"theory|bm24_mean_p_fixed_n|{depth}|{run_label}"
        for k in (legacy_key if run_label == "run1" else None, key):
            if k and k in cache.get("theory", {}) and _theory_cache_valid(cache["theory"][k]):
                return float(cache["theory"][k]["c_ann_rt"])
        th = _theory_with_fallback(
            k=k, r=r, depth=depth,
            delta_gamma=float(row["delta_gamma"]),
            delta_beta=float(row["delta_beta"]),
            beta_schedule=beta_schedule,
            num_iter=250, damping=0.15, dz_threshold=1e-2,
            timeout_s=600.0, use_subprocess=True, skip_fallback=True,
        )
        cache.setdefault("theory", {})[key] = th
        _merge_cache_on_disk(theory_cache_path, cache)
        return float(th["c_ann_rt"]) if th.get("theory_ok") else float("nan")

    depths = sorted(
        set(run1_series["mean_p"].get(window, {}))
        & set(run4_series["mean_p"].get(window, {}))
    )
    depths = [d for d in depths if d <= theory_max_depth]

    fig, ax = plt.subplots(figsize=(7, 4.2))
    for label, series, color, marker in (
        ("train_n=12", run1_series, "#DD8452", "o"),
        ("train_n=16", run4_series, "#55A868", "s"),
    ):
        ys = [series["mean_p"].get(window, {}).get(d, float("nan")) for d in depths]
        ax.plot(depths, ys, f"{marker}-", color=color, label=f"{label} c_typ")
    ann1, ann4 = [], []
    for d in depths:
        ann1.append(theory_for(run1_payload, "run1", d))
        ann4.append(theory_for(run4_payload, "run4", d))
    if any(np.isfinite(x) for x in ann1):
        ax.plot(depths, ann1, ":", color="#DD8452", alpha=0.7, label="−c_ann (run1 theory)")
    if any(np.isfinite(x) for x in ann4):
        ax.plot(depths, ann4, ":", color="#55A868", alpha=0.7, label="−c_ann (run4 theory)")
    ax.set_xlabel("QAOA depth p")
    ax.set_ylabel("log₂ slope vs n")
    ax.set_title(f"mean_p train · eval n={window[0]}–{window[1]}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.suptitle("c_typ vs BM24 annealed runtime proxy (−c_ann)", fontsize=11)
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
        default=Path("results/bm24_runs/06-09/run4/train16-tr100-te200-n12-18.json"),
    )
    p.add_argument(
        "--all-windows-output",
        type=Path,
        default=Path("results/bm24_runs/06-09/compare-run1-vs-run4-all-windows.png"),
    )
    p.add_argument(
        "--paper-output",
        type=Path,
        default=Path(
            "results/bm24_runs/06-09/compare-run1-vs-run4-paper-main.png"
        ),
        help="Main-text paper figure (fair eval window, two train sizes).",
    )
    p.add_argument(
        "--paper-supplement-output",
        type=Path,
        default=Path(
            "results/bm24_runs/06-09/compare-run1-vs-run4-paper-supplement.png"
        ),
        help="Supplementary robustness figure (all eval windows).",
    )
    p.add_argument(
        "--paper-supplement2-output",
        type=Path,
        default=Path(
            "results/bm24_runs/06-09/compare-run1-vs-run4-paper-supplement2.png"
        ),
        help="Supplement figure with taller y-axis viewport (same data/colors).",
    )
    p.add_argument(
        "--paper-supplement2-nmax20-output",
        type=Path,
        default=Path(
            "results/bm24_runs/06-09/compare-run1-vs-run4-paper-supplement2-nmax20.png"
        ),
        help="Supplement2 layout with eval windows extended to n=20.",
    )
    p.add_argument(
        "--n-max-eval",
        type=int,
        default=18,
        help="Upper n for eval-window slopes (requires per-n data through this n).",
    )
    p.add_argument(
        "--only-nmax20",
        action="store_true",
        help="Only regenerate the nmax20 supplement figure (skip other plots).",
    )
    p.add_argument(
        "--wider-n-json",
        type=Path,
        default=Path(
            "bm24_runs/06-09/run1/train12-tr100-te200-n12-18-wider-n16-20-te50.json"
        ),
        help="Wider-n te50 cache used to anchor n=19,20 from te200 n=18 medians.",
    )
    p.add_argument(
        "--extra-n-qaoa-eval",
        action="store_true",
        help="QAOA re-eval at n=19,20 (slow). Default: scale from wider-n ratios.",
    )
    p.add_argument(
        "--legacy-output",
        type=Path,
        default=Path("results/bm24_runs/06-09/compare-run1-vs-run4-corrected.png"),
    )
    p.add_argument(
        "--delta-output",
        type=Path,
        default=Path("results/bm24_runs/06-09/compare-run1-vs-run4-train-n-delta.png"),
    )
    p.add_argument(
        "--theory-output",
        type=Path,
        default=Path("results/bm24_runs/06-09/compare-run1-vs-run4-with-theory.png"),
    )
    p.add_argument(
        "--summary-json",
        type=Path,
        default=Path("results/bm24_runs/06-09/compare-run1-vs-run4-summary.json"),
    )
    p.add_argument("--test-size", type=int, default=200)
    p.add_argument(
        "--rebench-cache",
        type=Path,
        default=Path("results/bm24_runs/06-09/run1/train12-tr100-te200-n12-18-per_n-all.json"),
    )
    p.add_argument(
        "--skip-rebench",
        action="store_true",
        help="Plot only from existing JSON/cache (no QAOA re-eval).",
    )
    p.add_argument(
        "--with-theory",
        action="store_true",
        help="Also plot c_typ vs −c_ann (slow: saddle at run4 angles).",
    )
    p.add_argument("--theory-max-depth", type=int, default=10)
    p.add_argument(
        "--classical-json",
        type=Path,
        default=None,
        help="Scaling JSON with setup.classical (default: scaling-tn14.json)",
    )
    args = p.parse_args()

    run1_path = _resolve_run_json(args.run1, (
        Path("bm24_runs/06-09/run1/train12-tr100-te200-n12-18.json"),
    ))
    run4_path = _resolve_run_json(args.run4, RUN4_CANDIDATES)
    all_windows_out = (
        (_PHASECRAFT / args.all_windows_output).resolve()
        if not args.all_windows_output.is_absolute()
        else args.all_windows_output
    )
    paper_out = (
        (_PHASECRAFT / args.paper_output).resolve()
        if not args.paper_output.is_absolute()
        else args.paper_output
    )
    paper_supp_out = (
        (_PHASECRAFT / args.paper_supplement_output).resolve()
        if not args.paper_supplement_output.is_absolute()
        else args.paper_supplement_output
    )
    paper_supp2_out = (
        (_PHASECRAFT / args.paper_supplement2_output).resolve()
        if not args.paper_supplement2_output.is_absolute()
        else args.paper_supplement2_output
    )
    paper_supp2_nmax20_out = (
        (_PHASECRAFT / args.paper_supplement2_nmax20_output).resolve()
        if not args.paper_supplement2_nmax20_output.is_absolute()
        else args.paper_supplement2_nmax20_output
    )
    legacy_out = (
        (_PHASECRAFT / args.legacy_output).resolve()
        if not args.legacy_output.is_absolute()
        else args.legacy_output
    )
    delta_out = (
        (_PHASECRAFT / args.delta_output).resolve()
        if not args.delta_output.is_absolute()
        else args.delta_output
    )
    theory_out = (
        (_PHASECRAFT / args.theory_output).resolve()
        if not args.theory_output.is_absolute()
        else args.theory_output
    )
    summary_out = (
        (_PHASECRAFT / args.summary_json).resolve()
        if not args.summary_json.is_absolute()
        else args.summary_json
    )
    cache_path = _resolve_run_json(
        args.rebench_cache,
        (Path("bm24_runs/06-09/run1/train12-tr100-te200-n12-18-per_n-all.json"),),
    )

    run1_payload = json.loads(run1_path.read_text(encoding="utf-8"))
    run4_payload = json.loads(run4_path.read_text(encoding="utf-8"))
    print(f"run1: {run1_path.name}  run4: {run4_path.name}", flush=True)

    if args.skip_rebench and cache_path.is_file():
        rebench = json.loads(cache_path.read_text(encoding="utf-8"))
    else:
        print(f"Re-benchmarking run1 (cache: {cache_path.name})...", flush=True)
        rebench = rebenchmark_run1(run1_path, cache_path=cache_path, test_size=args.test_size)

    run1_series = series_from_run1(run1_payload, rebench)
    run4_series = series_from_run4(run4_payload)
    series_err = build_series_with_stderr(run1_payload, run4_payload, rebench)
    classical = load_classical_baselines(args.classical_json)

    n_max_eval = int(args.n_max_eval)
    cfg_n_max = int(run1_payload["config"]["n_max"])
    series_err_nmax20 = None
    if n_max_eval > cfg_n_max:
        wider_path = (
            (_PHASECRAFT / args.wider_n_json).resolve()
            if not args.wider_n_json.is_absolute()
            else args.wider_n_json
        )
        if args.extra_n_qaoa_eval:
            extra_ns = list(range(cfg_n_max + 1, n_max_eval + 1))
            print(
                f"Extending per-n medians to n={n_max_eval} "
                f"(QAOA eval n={extra_ns[0]}..{extra_ns[-1]})...",
                flush=True,
            )
            run1_extra_cache = (
                run1_path.parent / f"{run1_path.stem}-per_n-n{extra_ns[0]}-{extra_ns[-1]}.json"
            )
            run4_extra_cache = (
                run4_path.parent / f"{run4_path.stem}-per_n-n{extra_ns[0]}-{extra_ns[-1]}.json"
            )

            def _load_extra(path: Path) -> dict | None:
                if not path.is_file():
                    return None
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError):
                    return None
                return payload if payload.get("complete") else None

            run1_extra = _load_extra(run1_extra_cache)
            if run1_extra is None:
                run1_extra = eval_extra_per_n(
                    run1_payload,
                    run_dir=run1_path.parent,
                    extra_ns=extra_ns,
                    test_size=args.test_size,
                    cache_path=run1_extra_cache,
                    dataset_dir=run1_path.parent,
                )
            run4_extra = _load_extra(run4_extra_cache)
            if run4_extra is None:
                run4_extra = eval_extra_per_n(
                    run4_payload,
                    run_dir=run4_path.parent,
                    extra_ns=extra_ns,
                    test_size=args.test_size,
                    cache_path=run4_extra_cache,
                    dataset_dir=run1_path.parent,
                )
            run1_per_n = merged_per_n_for_run1(rebench, run1_extra)
            run4_per_n = merged_per_n_for_run4(run4_payload, run4_extra)
        else:
            if not wider_path.is_file():
                raise FileNotFoundError(
                    f"Wider-n JSON not found for n=19,20 anchor: {wider_path}"
                )
            r19, r20 = _wider_n_scale_factors(wider_path)
            print(
                f"Extending per-n to n={n_max_eval} via wider-n ratios "
                f"(n19/n18={r19:.3f}, n20/n18={r20:.3f} from {wider_path.name})",
                flush=True,
            )
            run1_per_n = merged_per_n_for_run1_wider(rebench, wider_path)
            run4_per_n = merged_per_n_for_run4_wider(run4_payload, wider_path)

        series_err_nmax20 = build_series_with_stderr(
            run1_payload,
            run4_payload,
            rebench,
            windows=WINDOWS_NMAX20,
            run1_per_n=run1_per_n,
            run4_per_n=run4_per_n,
        )

    summary = build_summary(run1_series, run4_series, run1_path=run1_path, run4_path=run4_path)

    supp2_nmax20_kw = dict(
        fig_height=5.8,
        y_pad_scale=2.2,
        panel_wspace=0.08,
        compact_legend=True,
        baseline_labels_first_panel=True,
        windows=WINDOWS_NMAX20,
        eval_window_for_baselines=(12, 20),
        suptitle=rf"Eval window extended to $n_{{\max}}=20$",
    )
    if series_err_nmax20 is not None:
        plot_paper_supplement(
            series_err_nmax20,
            out_path=paper_supp2_nmax20_out,
            classical=classical,
            **supp2_nmax20_kw,
        )
        mirror = _PHASECRAFT / "bm24_runs" / "06-09" / paper_supp2_nmax20_out.name
        if mirror.resolve() != paper_supp2_nmax20_out.resolve():
            plot_paper_supplement(
                series_err_nmax20,
                out_path=mirror,
                classical=classical,
                **supp2_nmax20_kw,
            )

    if args.only_nmax20:
        return

    summary_out.parent.mkdir(parents=True, exist_ok=True)
    summary_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {summary_out}")

    outputs = [
        (all_windows_out, plot_all_windows),
        (legacy_out, plot_legacy_corrected),
        (delta_out, plot_train_n_delta),
    ]
    for out_path, plot_fn in outputs:
        kwargs = {"classical": classical} if plot_fn is not plot_train_n_delta else {}
        plot_fn(run1_series, run4_series, out_path=out_path, **kwargs)
        mirror = _PHASECRAFT / "bm24_runs" / "06-09" / out_path.name
        if mirror.resolve() != out_path.resolve():
            plot_fn(run1_series, run4_series, out_path=mirror, **kwargs)

    plot_paper_main(series_err, out_path=paper_out, classical=classical)
    plot_paper_supplement(series_err, out_path=paper_supp_out, classical=classical)
    plot_paper_supplement(
        series_err,
        out_path=paper_supp2_out,
        classical=classical,
        fig_height=5.8,
        y_pad_scale=2.2,
        panel_wspace=0.08,
        compact_legend=True,
        baseline_labels_first_panel=True,
    )
    write_paper_captions(paper_out.parent)
    for path, supp_kw in (
        (paper_out, None),
        (paper_supp_out, None),
        (paper_supp2_out, dict(
            fig_height=5.8,
            y_pad_scale=2.2,
            panel_wspace=0.08,
            compact_legend=True,
            baseline_labels_first_panel=True,
        )),
        (paper_supp2_nmax20_out, dict(
            fig_height=5.8,
            y_pad_scale=2.2,
            panel_wspace=0.08,
            compact_legend=True,
            baseline_labels_first_panel=True,
            windows=WINDOWS_NMAX20,
            eval_window_for_baselines=(12, 20),
            suptitle=rf"Eval window extended to $n_{{\max}}=20$",
        ) if series_err_nmax20 is not None else None),
    ):
        if supp_kw is None and path == paper_supp2_nmax20_out:
            continue
        mirror = _PHASECRAFT / "bm24_runs" / "06-09" / path.name
        if mirror.resolve() == path.resolve():
            continue
        if path == paper_out:
            plot_paper_main(series_err, out_path=mirror, classical=classical)
        elif supp_kw:
            data = series_err_nmax20 if path == paper_supp2_nmax20_out else series_err
            plot_paper_supplement(
                data, out_path=mirror, classical=classical, **supp_kw
            )
        else:
            plot_paper_supplement(series_err, out_path=mirror, classical=classical)
    cap_mirror = _PHASECRAFT / "bm24_runs" / "06-09"
    if cap_mirror.resolve() != paper_out.parent.resolve():
        write_paper_captions(cap_mirror)

    if args.with_theory:
        theory_cache = run1_path.parent / f"{run1_path.stem}-ctyp-cann-cache.json"
        plot_with_theory(
            run1_series, run4_series,
            run1_payload=run1_payload, run4_payload=run4_payload,
            theory_cache_path=theory_cache, out_path=theory_out,
            theory_max_depth=int(args.theory_max_depth),
        )
        mirror = _PHASECRAFT / "bm24_runs" / "06-09" / theory_out.name
        if mirror.resolve() != theory_out.resolve():
            plot_with_theory(
                run1_series, run4_series,
                run1_payload=run1_payload, run4_payload=run4_payload,
                theory_cache_path=theory_cache, out_path=mirror,
                theory_max_depth=int(args.theory_max_depth),
            )


if __name__ == "__main__":
    main()
