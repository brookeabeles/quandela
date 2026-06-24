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
FAIR_WINDOW = (14, 18)
RUN4_CANDIDATES = (
    Path("results/bm24_runs/06-09/run4/train16-tr100-te200-n14-18.json"),
    Path("results/bm24_runs/06-09/run4/train16-tr100-te200-n12-18.json"),
    Path("bm24_runs/06-09/run4/train16-tr100-te200-n14-18.json"),
    Path("bm24_runs/06-09/run4/train16-tr100-te200-n12-18.json"),
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


def plot_legacy_corrected(
    run1_series: Dict[str, Dict[Tuple[int, int], Dict[int, float]]],
    run4_series: Dict[str, Dict[Tuple[int, int], Dict[int, float]]],
    *,
    out_path: Path,
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
        ax.set_title(mode)
        ax.set_xlabel("QAOA depth p")
        ax.set_xticks(depths)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Eval: log₂ slope of median(1/p_succ) vs n")
    axes[0].legend(fontsize=8, loc="upper right")
    fig.suptitle(
        "run1 vs run4 — run1 re-evaluated on common n=14–18 window",
        fontsize=11,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_all_windows(
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

    summary = build_summary(run1_series, run4_series, run1_path=run1_path, run4_path=run4_path)
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    summary_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {summary_out}")

    outputs = [
        (all_windows_out, plot_all_windows),
        (legacy_out, plot_legacy_corrected),
        (delta_out, plot_train_n_delta),
    ]
    for out_path, plot_fn in outputs:
        plot_fn(run1_series, run4_series, out_path=out_path)
        mirror = _PHASECRAFT / "bm24_runs" / "06-09" / out_path.name
        if mirror.resolve() != out_path.resolve():
            plot_fn(run1_series, run4_series, out_path=mirror)

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
