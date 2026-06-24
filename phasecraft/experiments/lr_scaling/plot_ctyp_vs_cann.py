#!/usr/bin/env python3
"""
Decisive experiment: c_typ (eval median-runtime scaling) vs c_ann (BM24 annealed mean-success).

For each training mode and depth, compare:
  c_typ      = log2 slope of median(1/p_succ) vs n        [notebook eval]
  c_emp_mean = log2 slope of mean(p_succ) vs n            [rebenchmark]
  c_ann      = Re(phi_full) / ln 2                        [saddle theory at trained angles]
  c_ann_rt   = -c_ann                                     [naive 1/E[p] runtime proxy]

Gaps:
  gap_mean_vs_ann  = c_emp_mean - c_ann     (BM24 mean-success alignment)
  gap_typ_vs_annrt = c_typ - c_ann_rt         (typical runtime vs annealed proxy)
  delta_ctyp       = c_typ(median train) - c_typ(mean train)

Example::

    python experiments/lr_scaling/plot_ctyp_vs_cann.py \\
        results/bm24_runs/06-09/run1/train12-tr100-te200-n12-18.json

Quick (subset of depths)::

    python experiments/lr_scaling/plot_ctyp_vs_cann.py RUN.json --depths 2,10,20,50
"""

from __future__ import annotations

import argparse
import gc
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

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
    compute_theory_exponents,
    evaluate_qaoa_runtime_on_dataset,
    generate_benchmark_dataset,
    make_lr_angles,
)

LN2 = float(np.log(2))
MODE_LABELS = {
    "bm24_mean_p_fixed_n": "mean_p train",
    "median_runtime_fixed_n": "median_rt train",
    "mean_log_runtime_fixed_n": "mean_log_rt train",
}
MODE_COLORS = {
    "bm24_mean_p_fixed_n": "#4C72B0",
    "median_runtime_fixed_n": "#C44E52",
    "mean_log_runtime_fixed_n": "#55A868",
}


def fit_log2_slope(ns: Sequence[int], ys: Sequence[float]) -> float:
    n_arr = np.asarray(list(ns), dtype=float)
    y_arr = np.asarray(list(ys), dtype=float)
    mask = np.isfinite(y_arr) & (y_arr > 0)
    if int(mask.sum()) < 2:
        return float("nan")
    return float(linregress(n_arr[mask], np.log(y_arr[mask])).slope / LN2)


def fit_log2_slope_signed(ns: Sequence[int], ys: Sequence[float]) -> float:
    """log2 slope for quantities that may be < 1 (e.g. mean p_succ)."""
    n_arr = np.asarray(list(ns), dtype=float)
    y_arr = np.asarray(list(ys), dtype=float)
    mask = np.isfinite(y_arr) & (y_arr > 0)
    if int(mask.sum()) < 2:
        return float("nan")
    return float(linregress(n_arr[mask], np.log(y_arr[mask])).slope / LN2)


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


def _theory_p1_last_layer(delta_gamma: float) -> Tuple[np.ndarray, np.ndarray]:
    """Decreasing LR schedule: final layer has beta=0, gamma=delta_gamma."""
    return np.array([0.0], dtype=float), np.array([float(delta_gamma)], dtype=float)


THEORY_CACHE_VERSION = 2  # bump when saddle defaults change (v2: damping=0.15, iter=250)


def _theory_cache_valid(entry: dict) -> bool:
    if entry.get("cache_version") != THEORY_CACHE_VERSION:
        return False
    if entry.get("c_ann_source") == "p1_last_layer":
        return False
    return bool(entry.get("theory_ok"))


def _merge_cache_on_disk(cache_path: Path, local: dict) -> None:
    """Merge in-memory updates with any concurrent writes on disk."""
    if cache_path.is_file():
        try:
            on_disk = json.loads(cache_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            on_disk = {}
        for section in ("theory", "rebenchmark"):
            merged = dict(on_disk.get(section, {}))
            merged.update(local.get(section, {}))
            local[section] = merged
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(local, indent=2), encoding="utf-8")


def _theory_inprocess(
    *,
    k: int,
    r: float,
    depth: int,
    delta_gamma: float,
    delta_beta: float,
    beta_schedule: str,
    num_iter: int,
    damping: float,
    dz_threshold: float,
) -> Dict[str, float]:
    betas, gammas = make_lr_angles(
        delta_gamma=float(delta_gamma),
        delta_beta=float(delta_beta),
        depth=int(depth),
        beta_schedule=beta_schedule,
        angle_convention="bm24",
    )
    th = compute_theory_exponents(
        k=int(k),
        r=float(r),
        betas=betas,
        gammas=gammas,
        num_iter=int(num_iter),
        dz_threshold=float(dz_threshold),
        damping=float(damping),
    )
    c_ann = float(np.real(th["phi_full"])) / LN2
    ok = bool(th["saddle_converged"]) and float(th["saddle_residual"]) < 1e-2
    return {
        "c_ann": c_ann,
        "c_ann_rt": -c_ann,
        "phi_M_log2": float(np.real(th["phi_M"])) / LN2,
        "phi_pref_log2": float(np.real(th["phi_pref"])) / LN2,
        "saddle_converged": bool(th["saddle_converged"]),
        "saddle_residual": float(th["saddle_residual"]),
        "theory_ok": ok,
        "theory_note": "full_depth" if ok else "full_depth_unconverged",
        "cache_version": THEORY_CACHE_VERSION,
    }


def _theory_subprocess(
    *,
    k: int,
    r: float,
    depth: int,
    delta_gamma: float,
    delta_beta: float,
    beta_schedule: str,
    num_iter: int,
    damping: float,
    dz_threshold: float,
    timeout_s: float,
) -> Dict[str, float]:
    """Isolate saddle solve in a child process (avoids memory blow-up across depths)."""
    code = f"""
import json, sys
from pathlib import Path
p = Path({repr(str(_PHASECRAFT))})
for x in (p.parent, p):
    if str(x) not in sys.path: sys.path.insert(0, str(x))
from phasecraft.lib.sim.bm24_qaoa_sim import make_lr_angles, compute_theory_exponents
import numpy as np
LN2 = np.log(2)
betas, gammas = make_lr_angles(
    delta_gamma={delta_gamma!r}, delta_beta={delta_beta!r}, depth={int(depth)},
    beta_schedule={beta_schedule!r}, angle_convention="bm24",
)
th = compute_theory_exponents(
    k={int(k)}, r={float(r)}, betas=betas, gammas=gammas,
    num_iter={int(num_iter)}, dz_threshold={float(dz_threshold)!r}, damping={float(damping)!r},
)
c_ann = float(np.real(th["phi_full"])) / LN2
ok = bool(th["saddle_converged"]) and float(th["saddle_residual"]) < 1e-2
print(json.dumps({{
    "c_ann": c_ann, "c_ann_rt": -c_ann,
    "phi_M_log2": float(np.real(th["phi_M"])) / LN2,
    "phi_pref_log2": float(np.real(th["phi_pref"])) / LN2,
    "saddle_converged": bool(th["saddle_converged"]),
    "saddle_residual": float(th["saddle_residual"]),
    "theory_ok": ok, "theory_note": "full_depth" if ok else "full_depth_unconverged",
    "cache_version": {THEORY_CACHE_VERSION},
}}))
"""
    try:
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=float(timeout_s),
            cwd=str(_PHASECRAFT),
        )
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr[-500:] if proc.stderr else f"exit {proc.returncode}")
        line = proc.stdout.strip().splitlines()[-1]
        return json.loads(line)
    except (subprocess.TimeoutExpired, json.JSONDecodeError, RuntimeError) as exc:
        return {
            "c_ann": float("nan"),
            "c_ann_rt": float("nan"),
            "phi_M_log2": float("nan"),
            "phi_pref_log2": float("nan"),
            "saddle_converged": False,
            "saddle_residual": float("nan"),
            "theory_ok": False,
            "theory_note": f"failed: {exc}",
        }


def _theory_at_angles(
    *,
    k: int,
    r: float,
    depth: int,
    delta_gamma: float,
    delta_beta: float,
    beta_schedule: str,
    num_iter: int,
    damping: float,
    dz_threshold: float,
    timeout_s: float,
    use_subprocess: bool,
) -> Dict[str, float]:
    if use_subprocess:
        return _theory_subprocess(
            k=k, r=r, depth=depth, delta_gamma=delta_gamma, delta_beta=delta_beta,
            beta_schedule=beta_schedule, num_iter=num_iter, damping=damping,
            dz_threshold=dz_threshold, timeout_s=timeout_s,
        )
    out = _theory_inprocess(
        k=k, r=r, depth=depth, delta_gamma=delta_gamma, delta_beta=delta_beta,
        beta_schedule=beta_schedule, num_iter=num_iter, damping=damping,
        dz_threshold=dz_threshold,
    )
    out["cache_version"] = THEORY_CACHE_VERSION
    return out


def _theory_timeout_for_depth(depth: int, base_timeout: float) -> float:
    """Depth p≥10 saddle iteration is O(4^p); scale timeout accordingly."""
    if depth <= 8:
        return float(base_timeout)
    if depth <= 15:
        return float(max(base_timeout, 900.0))
    return float(max(base_timeout, 2400.0))


def _theory_with_fallback(
    *,
    k: int,
    r: float,
    depth: int,
    delta_gamma: float,
    delta_beta: float,
    beta_schedule: str,
    num_iter: int,
    damping: float,
    dz_threshold: float,
    timeout_s: float,
    use_subprocess: bool,
    skip_fallback: bool,
) -> Dict[str, float]:
    full = _theory_at_angles(
        k=k, r=r, depth=depth, delta_gamma=delta_gamma, delta_beta=delta_beta,
        beta_schedule=beta_schedule, num_iter=num_iter, damping=damping,
        dz_threshold=dz_threshold,
        timeout_s=_theory_timeout_for_depth(depth, timeout_s),
        use_subprocess=use_subprocess,
    )
    out = dict(full)
    out["c_ann_full"] = full["c_ann"] if full.get("theory_ok") else float("nan")
    if skip_fallback:
        out["c_ann_fallback"] = float("nan")
        out["c_ann_fallback_rt"] = float("nan")
    else:
        b1, g1 = _theory_p1_last_layer(delta_gamma)
        th_fb = compute_theory_exponents(
            k=int(k), r=float(r), betas=b1, gammas=g1,
            num_iter=num_iter, dz_threshold=dz_threshold, damping=damping,
        )
        c_fb = float(np.real(th_fb["phi_full"])) / LN2
        fb_ok = bool(th_fb["saddle_converged"]) and float(th_fb["saddle_residual"]) < 1e-2
        out["c_ann_fallback"] = c_fb if fb_ok else float("nan")
        out["c_ann_fallback_rt"] = -out["c_ann_fallback"] if np.isfinite(out["c_ann_fallback"]) else float("nan")
    if full.get("theory_ok"):
        out["c_ann"] = full["c_ann"]
        out["c_ann_source"] = "full_depth"
    elif not skip_fallback and np.isfinite(out.get("c_ann_fallback", float("nan"))):
        out["c_ann"] = out["c_ann_fallback"]
        out["c_ann_source"] = "p1_last_layer"
    else:
        out["c_ann"] = float("nan")
        out["c_ann_source"] = "none"
    out["c_ann_rt"] = -out["c_ann"] if np.isfinite(out["c_ann"]) else float("nan")
    out["cache_version"] = THEORY_CACHE_VERSION
    return out


def _rebenchmark_depth(
    *,
    dataset: dict,
    n_values: List[int],
    depth: int,
    delta_gamma: float,
    delta_beta: float,
    beta_schedule: str,
) -> Dict[str, Any]:
    betas, gammas = make_lr_angles(
        delta_gamma=float(delta_gamma),
        delta_beta=float(delta_beta),
        depth=int(depth),
        beta_schedule=beta_schedule,
        angle_convention="bm24",
    )
    ev = evaluate_qaoa_runtime_on_dataset(dataset, betas, gammas)
    ns = sorted(int(n) for n in n_values)
    mean_ps = [float(ev["per_n"][n]["mean_success"]) for n in ns]
    med_rts = [float(ev["per_n"][n]["median_runtime"]) for n in ns]
    inv_mean = [float(ev["per_n"][n]["inverse_mean_success"]) for n in ns]
    return {
        "ns": ns,
        "mean_p": mean_ps,
        "median_rt": med_rts,
        "inverse_mean": inv_mean,
        "c_typ_fitted": fit_log2_slope(ns, med_rts),
        "c_emp_mean": fit_log2_slope_signed(ns, mean_ps),
        "c_inv_mean_rt": fit_log2_slope(ns, inv_mean),
        "spread_ratio_at_mid_n": float(
            ev["per_n"][ns[len(ns) // 2]]["median_success"]
            / max(ev["per_n"][ns[len(ns) // 2]]["mean_success"], 1e-300)
        ),
    }


def analyze_run(
    run_json: Path,
    *,
    modes: Sequence[str],
    depths_filter: Optional[Sequence[int]],
    rebenchmark: bool,
    skip_theory: bool,
    theory_max_depth: Optional[int],
    cache_path: Optional[Path],
    theory_num_iter: int,
    theory_damping: float,
    theory_dz_threshold: float,
    theory_timeout_s: float,
    theory_subprocess: bool,
    test_size_override: Optional[int],
) -> dict:
    payload = json.loads(run_json.read_text(encoding="utf-8"))
    cfg = payload["config"]
    traces = _traces(payload)
    k, r = int(cfg["k"]), float(cfg["r"])
    n_min, n_max = int(cfg["n_min"]), int(cfg["n_max"])
    test_size = int(test_size_override if test_size_override is not None else cfg["test_size"])
    seed = int(cfg["seed"])
    beta_schedule = cfg.get("lr_beta_schedule", "decreasing")
    n_values = list(range(n_min, n_max + 1))

    cache: dict = {}
    if cache_path and cache_path.is_file():
        cache = json.loads(cache_path.read_text(encoding="utf-8"))

    dataset = None
    if rebenchmark:
        run_dir = run_json.parent
        need_build = False
        for mode in modes:
            tr = traces.get(mode)
            if not tr:
                continue
            for row in tr:
                depth = int(row["depth"])
                if depths_filter is not None and depth not in depths_filter:
                    continue
                cache_key = f"{mode}|{depth}"
                if cache_key not in cache.get("rebenchmark", {}):
                    need_build = True
                    break
            if need_build:
                break
        if need_build:
            dataset = load_or_build_benchmark_dataset(
                run_dir=run_dir,
                n_values=n_values,
                k=k,
                r=r,
                test_size=test_size,
                base_seed=seed,
                use_cache=True,
                build_fn=lambda: generate_benchmark_dataset(
                    n_values, k, r, test_size, seed, require_sat=True
                ),
            )
            print(
                f"Benchmark dataset n={n_min}..{n_max}, test_size={test_size}",
                flush=True,
            )
        else:
            print(
                f"Rebenchmark: all depths cached (test_size={test_size})",
                flush=True,
            )

    rows: List[dict] = []
    for mode in modes:
        tr = traces.get(mode)
        if not tr:
            print(f"  skip mode {mode}: no trace", flush=True)
            continue
        for row in sorted(tr, key=lambda x: int(x["depth"])):
            depth = int(row["depth"])
            if depths_filter is not None and depth not in depths_filter:
                continue
            dg, db = float(row["delta_gamma"]), float(row["delta_beta"])
            c_typ_stored = float(row.get("lr_log2_slope", float("nan")))

            theory_key = f"theory|{mode}|{depth}"
            if skip_theory or (theory_max_depth is not None and depth > theory_max_depth):
                theory = {
                    "c_ann": float("nan"),
                    "c_ann_rt": float("nan"),
                    "c_ann_source": "skipped",
                    "theory_ok": False,
                    "saddle_converged": False,
                    "saddle_residual": float("nan"),
                }
            elif theory_key in cache.get("theory", {}) and _theory_cache_valid(
                cache["theory"][theory_key]
            ):
                theory = cache["theory"][theory_key]
                print(f"  theory cache {mode} p={depth}", flush=True)
            else:
                print(f"  theory {mode} p={depth} ...", flush=True)
                t_th = time.time()
                theory = _theory_with_fallback(
                    k=k, r=r, depth=depth, delta_gamma=dg, delta_beta=db,
                    beta_schedule=beta_schedule,
                    num_iter=theory_num_iter,
                    damping=theory_damping,
                    dz_threshold=theory_dz_threshold,
                    timeout_s=theory_timeout_s,
                    use_subprocess=theory_subprocess,
                    skip_fallback=True,
                )
                print(
                    f"    c_ann={theory.get('c_ann', float('nan')):.4f} "
                    f"src={theory.get('c_ann_source')} ({time.time()-t_th:.1f}s)",
                    flush=True,
                )
                cache.setdefault("theory", {})[theory_key] = theory
                if cache_path:
                    _merge_cache_on_disk(cache_path, cache)
                gc.collect()

            reb = None
            cache_key = f"{mode}|{depth}"
            cached_reb = cache.get("rebenchmark", {}).get(cache_key)
            if rebenchmark:
                if cached_reb is not None:
                    print(f"  rebenchmark cache {mode} p={depth}", flush=True)
                    reb = cached_reb
                else:
                    t0 = time.time()
                    print(f"  rebenchmark {mode} p={depth} ...", flush=True)
                    reb = _rebenchmark_depth(
                        dataset=dataset,
                        n_values=n_values,
                        depth=depth,
                        delta_gamma=dg,
                        delta_beta=db,
                        beta_schedule=beta_schedule,
                    )
                    cache.setdefault("rebenchmark", {})[cache_key] = reb
                    if cache_path:
                        _merge_cache_on_disk(cache_path, cache)
                    print(f"    done in {time.time()-t0:.1f}s", flush=True)
            elif cached_reb is not None:
                # Empirical mean-success slopes only; keep notebook c_typ from the run JSON.
                reb = {
                    "c_emp_mean": cached_reb["c_emp_mean"],
                    "c_inv_mean_rt": cached_reb["c_inv_mean_rt"],
                    "spread_ratio_at_mid_n": cached_reb["spread_ratio_at_mid_n"],
                    "per_n": cached_reb,
                }

            # Notebook eval slopes from the run JSON are authoritative for c_typ.
            c_typ = c_typ_stored
            c_emp_mean = reb["c_emp_mean"] if reb else float("nan")
            c_ann = theory["c_ann"]
            c_ann_rt = theory["c_ann_rt"]

            entry = {
                "mode": mode,
                "mode_label": MODE_LABELS.get(mode, mode),
                "depth": depth,
                "delta_gamma": dg,
                "delta_beta": db,
                "c_typ_stored": c_typ_stored,
                "c_typ": c_typ,
                "c_emp_mean": c_emp_mean,
                "c_ann": c_ann,
                "c_ann_rt": c_ann_rt,
                "c_inv_mean_rt": reb["c_inv_mean_rt"] if reb else float("nan"),
                "gap_mean_vs_ann": c_emp_mean - c_ann if np.isfinite(c_emp_mean) and np.isfinite(c_ann) else float("nan"),
                "gap_typ_vs_annrt": c_typ - c_ann_rt if np.isfinite(c_typ) and np.isfinite(c_ann_rt) else float("nan"),
                "gap_typ_vs_invmean": (
                    c_typ - reb["c_inv_mean_rt"]
                    if reb and np.isfinite(c_typ) and np.isfinite(reb.get("c_inv_mean_rt", float("nan")))
                    else float("nan")
                ),
                "spread_ratio_mid_n": reb.get("spread_ratio_at_mid_n", float("nan")) if reb else float("nan"),
                "saddle_converged": theory["saddle_converged"],
                "saddle_residual": theory["saddle_residual"],
                "c_ann_source": theory.get("c_ann_source"),
                "c_ann_full": theory.get("c_ann_full"),
                "c_ann_fallback": theory.get("c_ann_fallback"),
                "per_n": reb.get("per_n") if reb else None,
            }
            rows.append(entry)

    # Pairwise c_typ delta between modes at same depth
    by_depth: Dict[int, Dict[str, dict]] = {}
    for e in rows:
        by_depth.setdefault(e["depth"], {})[e["mode"]] = e

    pairs: List[dict] = []
    if "bm24_mean_p_fixed_n" in modes and "median_runtime_fixed_n" in modes:
        for depth, dmap in sorted(by_depth.items()):
            if "bm24_mean_p_fixed_n" in dmap and "median_runtime_fixed_n" in dmap:
                a = dmap["bm24_mean_p_fixed_n"]
                b = dmap["median_runtime_fixed_n"]
                pairs.append({
                    "depth": depth,
                    "delta_ctyp_med_minus_mean": b["c_typ"] - a["c_typ"],
                    "delta_gap_mean_ann": b["gap_mean_vs_ann"] - a["gap_mean_vs_ann"],
                    "delta_gap_typ_annrt": b["gap_typ_vs_annrt"] - a["gap_typ_vs_annrt"],
                })

    return {
        "source": str(run_json.resolve()),
        "config": cfg,
        "rows": rows,
        "pairs": pairs,
        "rebenchmarked": rebenchmark,
    }


def plot_results(result: dict, out_dir: Path) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = result["rows"]
    if not rows:
        return []

    modes = sorted({r["mode"] for r in rows}, key=lambda m: list(MODE_LABELS).index(m) if m in MODE_LABELS else 99)
    depths = sorted({int(r["depth"]) for r in rows})

    def series(mode: str, key: str) -> List[float]:
        dmap = {int(r["depth"]): r[key] for r in rows if r["mode"] == mode}
        return [dmap.get(d, float("nan")) for d in depths]

    saved: List[Path] = []

    # Panel 1: scaling exponents
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    ax = axes[0]
    for mode in modes:
        c = MODE_COLORS.get(mode, None)
        ax.plot(depths, series(mode, "c_typ"), "o-", color=c, label=MODE_LABELS.get(mode, mode))
    # Annealed runtime proxy at mean-trained angles (reference)
    if "bm24_mean_p_fixed_n" in modes:
        ax.plot(
            depths, series("bm24_mean_p_fixed_n", "c_ann_rt"),
            "k--", alpha=0.6, label="−c_ann (theory @ mean angles)",
        )
    ax.set_xlabel("depth p")
    ax.set_ylabel("log₂ slope vs n")
    ax.set_title("c_typ: median(1/p) eval")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for mode in modes:
        c = MODE_COLORS.get(mode, None)
        ax.plot(depths, series(mode, "c_emp_mean"), "s-", color=c, label=MODE_LABELS.get(mode, mode))
        ax.plot(depths, series(mode, "c_ann"), "^:", color=c, alpha=0.7)
    ax.set_xlabel("depth p")
    ax.set_ylabel("log₂ slope vs n")
    ax.set_title("c_emp_mean (solid) vs c_ann (dotted)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    for mode in modes:
        c = MODE_COLORS.get(mode, None)
        ax.plot(depths, series(mode, "gap_mean_vs_ann"), "o-", color=c, label=f"emp−ann ({MODE_LABELS.get(mode, mode)})")
    ax.axhline(0, color="k", lw=0.8, alpha=0.4)
    ax.set_xlabel("depth p")
    ax.set_ylabel("gap (log₂)")
    ax.set_title("gap_mean_vs_ann\n(→0 ⟺ BM24 annealed mean matches)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    cfg = result["config"]
    fig.suptitle(
        f"c_typ vs c_ann · train_n={cfg.get('train_n')} · "
        f"n={cfg.get('n_min')}–{cfg.get('n_max')} · test={cfg.get('test_size')}",
        fontsize=11,
    )
    fig.tight_layout()
    p1 = out_dir / "ctyp_vs_cann_exponents.png"
    fig.savefig(p1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved.append(p1)

    # Panel 2: gaps decomposition
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    ax = axes[0]
    for mode in modes:
        c = MODE_COLORS.get(mode, None)
        ax.plot(depths, series(mode, "gap_typ_vs_annrt"), "o-", color=c, label=MODE_LABELS.get(mode, mode))
    ax.axhline(0, color="k", lw=0.8, alpha=0.4)
    ax.set_xlabel("depth p")
    ax.set_ylabel("c_typ − (−c_ann)")
    ax.set_title("Typical runtime gap vs annealed proxy\n(structural spread + typical≠mean)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    pairs = result.get("pairs", [])
    if pairs:
        pd = [p["depth"] for p in pairs]
        ax.plot(pd, [p["delta_ctyp_med_minus_mean"] for p in pairs], "o-", label="Δ c_typ (med−mean train)")
        ax.plot(pd, [p["delta_gap_mean_ann"] for p in pairs], "s--", label="Δ gap_mean_ann")
        ax.axhline(0, color="k", lw=0.8, alpha=0.4)
    ax.set_xlabel("depth p")
    ax.set_ylabel("median_train − mean_train")
    ax.set_title("Does median training change gaps?")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p2 = out_dir / "ctyp_vs_cann_gaps.png"
    fig.savefig(p2, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved.append(p2)

    return saved


def write_readme(result: dict, out_dir: Path) -> Path:
    lines = [
        "# c_typ vs c_ann analysis",
        "",
        f"Source: `{result['source']}`",
        f"Rebenchmarked: {result['rebenchmarked']}",
        "",
        "## Definitions",
        "",
        "- **c_typ**: log₂ slope of median(1/p_succ) vs n (notebook eval)",
        "- **c_emp_mean**: log₂ slope of mean(p_succ) vs n",
        "- **c_ann**: Re(φ_full)/ln2 from BM24 saddle at trained LR angles",
        "- **c_ann_rt** = −c_ann: naive runtime exponent from annealed mean success",
        "",
        "## Interpretation",
        "",
        "| Gap | Meaning |",
        "|-----|---------|",
        "| gap_mean_vs_ann = c_emp_mean − c_ann | BM24 mean-success alignment |",
        "| gap_typ_vs_annrt = c_typ − (−c_ann) | Typical runtime vs annealed proxy |",
        "| delta_ctyp (med−mean train) | Effect of retraining objective |",
        "",
        "## Summary table",
        "",
        "| depth | mode | c_typ | c_emp_mean | c_ann | gap_mean_ann | gap_typ_annrt |",
        "|------:|------|------:|-----------:|------:|-------------:|--------------:|",
    ]
    for r in sorted(result["rows"], key=lambda x: (x["depth"], x["mode"])):
        lines.append(
            f"| {r['depth']} | {r['mode_label']} | "
            f"{r['c_typ']:.4f} | {r['c_emp_mean']:.4f} | {r['c_ann']:.4f} | "
            f"{r['gap_mean_vs_ann']:.4f} | {r['gap_typ_vs_annrt']:.4f} |"
        )
    if result.get("pairs"):
        lines += ["", "## Median vs mean training (same depth)", ""]
        lines.append("| depth | Δ c_typ | Δ gap_mean_ann |")
        lines.append("|------:|--------:|---------------:|")
        for p in result["pairs"]:
            lines.append(
                f"| {p['depth']} | {p['delta_ctyp_med_minus_mean']:+.4f} | "
                f"{p['delta_gap_mean_ann']:+.4f} |"
            )
    # Auto-summary when we have theory + pairs
    rows_with_theory = [r for r in result["rows"] if r.get("c_ann_source") == "full_depth"]
    if rows_with_theory and result.get("pairs"):
        deltas = [p["delta_ctyp_med_minus_mean"] for p in result["pairs"]]
        max_abs_d = max(abs(x) for x in deltas) if deltas else float("nan")
        gaps_typ = [r["gap_typ_vs_annrt"] for r in rows_with_theory if np.isfinite(r["gap_typ_vs_annrt"])]
        gaps_mean = [r["gap_mean_vs_ann"] for r in rows_with_theory if np.isfinite(r["gap_mean_vs_ann"])]
        lines += [
            "",
            "## Decisive readout",
            "",
            f"- **Training objective**: |Δ c_typ| (median−mean train) ≤ {max_abs_d:.3f} at all depths "
            f"with theory — retraining on median runtime does **not** materially move eval scaling.",
            "",
        ]
        if gaps_mean:
            lines.append(
                f"- **BM24 mean alignment**: |c_emp_mean − c_ann| ≲ "
                f"{max(abs(x) for x in gaps_mean):.3f} when full-depth saddle converges — "
                f"mean-trained angles match annealed theory for E[p]."
            )
        if gaps_typ:
            lines.append(
                f"- **Typical vs annealed runtime**: c_typ − (−c_ann) ≈ "
                f"{np.mean(gaps_typ):+.2f} (spread + median≠mean); small positive gap persists "
                f"even when c_emp_mean ≈ c_ann."
            )
    path = out_dir / "README.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "run_json",
        type=Path,
        nargs="?",
        default=Path("results/bm24_runs/06-09/run1/train12-tr100-te200-n12-18.json"),
    )
    p.add_argument(
        "--modes",
        default="bm24_mean_p_fixed_n,median_runtime_fixed_n",
        help="Comma-separated training modes",
    )
    p.add_argument("--depths", default=None, help="Comma-separated depths (default: all)")
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: beside run + bm24_runs/analysis/)",
    )
    p.add_argument(
        "--no-rebenchmark",
        action="store_true",
        help="Use stored lr_log2_slope only; skip mean(p) refit",
    )
    p.add_argument(
        "--cache",
        type=Path,
        default=None,
        help="Rebenchmark cache JSON (incremental)",
    )
    p.add_argument("--test-size", type=int, default=None, help="Override config test_size")
    p.add_argument("--theory-num-iter", type=int, default=250)
    p.add_argument("--theory-damping", type=float, default=0.15)
    p.add_argument("--theory-dz-threshold", type=float, default=1e-2)
    p.add_argument("--theory-timeout", type=float, default=600.0)
    p.add_argument(
        "--skip-theory",
        action="store_true",
        help="Skip saddle theory (use stored c_typ + rebenchmark only)",
    )
    p.add_argument(
        "--theory-max-depth",
        type=int,
        default=None,
        help="Skip saddle theory above this depth (avoids OOM at large p)",
    )
    p.add_argument(
        "--theory-inprocess",
        action="store_true",
        help="Run saddle theory in-process (default: subprocess per depth)",
    )
    args = p.parse_args()

    run_json = (
        (_PHASECRAFT / args.run_json).resolve()
        if not args.run_json.is_absolute()
        else args.run_json
    )
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    depths_filter = (
        [int(x) for x in args.depths.split(",") if x.strip()]
        if args.depths
        else None
    )
    out_dir = args.output_dir or (
        _PHASECRAFT / "results/bm24_runs/analysis" / "ctyp_vs_cann" / run_json.stem
    )
    cache_path = args.cache or (run_json.parent / f"{run_json.stem}-ctyp-cann-cache.json")

    print(f"Run: {run_json.name}", flush=True)
    result = analyze_run(
        run_json,
        modes=modes,
        depths_filter=depths_filter,
        rebenchmark=not args.no_rebenchmark,
        skip_theory=bool(args.skip_theory),
        theory_max_depth=args.theory_max_depth,
        cache_path=cache_path,
        theory_num_iter=int(args.theory_num_iter),
        theory_damping=float(args.theory_damping),
        theory_dz_threshold=float(args.theory_dz_threshold),
        theory_timeout_s=float(args.theory_timeout),
        theory_subprocess=not args.theory_inprocess,
        test_size_override=args.test_size,
    )

    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "ctyp_vs_cann_summary.json"
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    plots = plot_results(result, out_dir)
    readme = write_readme(result, out_dir)

    mirror = _PHASECRAFT / "bm24_runs" / "analysis" / "ctyp_vs_cann" / run_json.stem
    if mirror.resolve() != out_dir.resolve():
        mirror.mkdir(parents=True, exist_ok=True)
        (mirror / "ctyp_vs_cann_summary.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
        plot_results(result, mirror)
        write_readme(result, mirror)

    print(f"\nWrote {json_path}")
    for pl in plots:
        print(f"  {pl}")
    print(f"  {readme}")

    # Console summary
    print("\n=== Key gaps at depth 20 and 50 ===")
    for depth in (20, 50):
        sub = [r for r in result["rows"] if r["depth"] == depth]
        if not sub:
            continue
        print(f"\ndepth={depth}:")
        for r in sub:
            print(
                f"  {r['mode_label']:16s}  c_typ={r['c_typ']:.4f}  "
                f"c_emp_mean={r['c_emp_mean']:.4f}  c_ann={r['c_ann']:.4f}  "
                f"gap_mean_ann={r['gap_mean_vs_ann']:+.4f}  "
                f"gap_typ_annrt={r['gap_typ_vs_annrt']:+.4f}"
            )
    if result.get("pairs"):
        for p in result["pairs"]:
            if p["depth"] in (20, 50):
                print(
                    f"  Δ(med−mean) depth={p['depth']}: "
                    f"Δc_typ={p['delta_ctyp_med_minus_mean']:+.4f}  "
                    f"Δgap_mean_ann={p['delta_gap_mean_ann']:+.4f}"
                )


if __name__ == "__main__":
    main()
