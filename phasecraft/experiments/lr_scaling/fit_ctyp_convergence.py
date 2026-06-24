#!/usr/bin/env python3
"""
Fit the asymptotic convergence of c_typ(p) as p → ∞.

c_typ(p) is the log2 slope of median(1/p_succ) vs n, measured from a
training run at each depth.  At finite p it is still dropping; this script
extrapolates to c_inf = lim_{p→∞} c_typ(p) by fitting three functional forms:

  power:  c_inf + A · p^(-alpha)   [3 params]
  inv:    c_inf + A / p             [2 params]
  log:    c_inf + A / ln(p)         [2 params]

Outputs:
  - fit summary table to stdout
  - PNG: data + all fits extrapolated to p_max, with c_inf ± 1σ band
  - JSON: fit parameters and uncertainties

Example::

    python experiments/lr_scaling/fit_ctyp_convergence.py \\
        results/bm24_runs/06-09/run1/train12-tr100-te200-n12-18.json

    # Compare two runs:
    python experiments/lr_scaling/fit_ctyp_convergence.py \\
        results/bm24_runs/06-09/run1/train12-tr100-te200-n12-18.json \\
        results/bm24_runs/06-09/run4/train16-tr100-te200-n14-18.json

    # Only fit depths p >= 10 (skip small-p transient):
    python experiments/lr_scaling/fit_ctyp_convergence.py RUN.json --p-min-fit 10
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

_PHASECRAFT = Path(__file__).resolve().parents[2]
for _p in (_PHASECRAFT.parent, _PHASECRAFT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

LN2 = float(np.log(2))

# Mode display
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
# Functional forms
# ---------------------------------------------------------------------------

def _f_power(p, c_inf, A, alpha):
    return c_inf + A * np.asarray(p, dtype=float) ** (-alpha)


def _f_inv(p, c_inf, A):
    return c_inf + A / np.asarray(p, dtype=float)


def _f_log(p, c_inf, A):
    return c_inf + A / np.log(np.asarray(p, dtype=float))


_MODELS_DEF = {
    "power": (_f_power, r"$c_\infty + A\,p^{-\alpha}$"),
    "inv":   (_f_inv,   r"$c_\infty + A/p$"),
    "log":   (_f_log,   r"$c_\infty + A/\ln p$"),
}
# Physical bounds: c_inf ∈ [0, 1], A > 0, alpha > 0
_BOUNDS = {
    "power": ([0.0, 0.0, 0.01], [1.0, 10.0, 5.0]),
    "inv":   ([0.0, 0.0],       [1.0, 10.0]),
    "log":   ([0.0, 0.0],       [1.0, 10.0]),
}
# Multiple initial guesses for power (most degenerate)
_P0_GRID_POWER = [
    [0.20, 0.6, 0.40],
    [0.25, 0.5, 0.30],
    [0.30, 0.4, 0.50],
    [0.15, 0.8, 0.25],
]


def _aic(residuals: np.ndarray, n_params: int) -> float:
    n = len(residuals)
    ssr = float(np.sum(residuals ** 2))
    if ssr <= 0 or n <= n_params:
        return float("nan")
    log_lik = -n / 2 * np.log(ssr / n)
    return 2 * n_params - 2 * log_lik


def _fit_one(fn, p0, bounds, ps_f, ys_f, maxfev=10000):
    popt, pcov = curve_fit(fn, ps_f, ys_f, p0=p0, bounds=bounds, maxfev=maxfev)
    perr = np.sqrt(np.diag(pcov))
    pred = fn(ps_f, *popt)
    ssr = float(np.sum((ys_f - pred) ** 2))
    return popt, perr, ssr


def fit_models(
    ps: np.ndarray,
    ys: np.ndarray,
    p_min_fit: int = 1,
) -> Dict[str, dict]:
    mask = np.isfinite(ys) & (ps >= p_min_fit)
    ps_f, ys_f = ps[mask], ys[mask]
    results: Dict[str, dict] = {}
    for name, (fn, label) in _MODELS_DEF.items():
        bounds = _BOUNDS[name]
        p0_list = _P0_GRID_POWER if name == "power" else [[0.25, 0.5]]
        best_popt, best_perr, best_ssr = None, None, float("inf")
        last_exc = None
        for p0 in p0_list:
            try:
                popt, perr, ssr = _fit_one(fn, p0, bounds, ps_f, ys_f)
                if ssr < best_ssr:
                    best_popt, best_perr, best_ssr = popt, perr, ssr
            except Exception as exc:
                last_exc = exc
        if best_popt is None:
            results[name] = {"ok": False, "error": str(last_exc)}
            continue
        pred = fn(ps_f, *best_popt)
        resid = ys_f - pred
        aic = _aic(resid, len(best_popt))
        c_inf = float(best_popt[0])
        c_inf_err = float(best_perr[0])
        # Flag as unstable if 1σ on c_inf exceeds the value itself
        stable = c_inf_err < abs(c_inf) and c_inf_err < 0.5
        results[name] = {
            "params": best_popt.tolist(),
            "param_err": best_perr.tolist(),
            "c_inf": c_inf,
            "c_inf_err": c_inf_err,
            "aic": aic,
            "label": label,
            "n_fit": int(mask.sum()),
            "stable": stable,
            "ok": True,
        }
    return results


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


def load_series(run_json: Path, modes: Optional[List[str]] = None) -> Dict[str, dict]:
    """Return {mode: {ps: array, ys: array, label: str, cfg: dict}}."""
    payload = json.loads(run_json.read_text(encoding="utf-8"))
    cfg = payload.get("config", {})
    traces = _traces(payload)
    out: Dict[str, dict] = {}
    for mode, rows in traces.items():
        if modes and mode not in modes:
            continue
        sorted_rows = sorted(rows, key=lambda r: int(r["depth"]))
        ps = np.array([int(r["depth"]) for r in sorted_rows], dtype=float)
        ys = np.array([float(r.get("lr_log2_slope", float("nan"))) for r in sorted_rows])
        out[mode] = {"ps": ps, "ys": ys, "cfg": cfg}
    return out


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_convergence(
    all_series: List[Tuple[str, str, Dict[str, dict]]],  # (run_label, mode, fit_results)
    data_by_run: List[Tuple[str, str, np.ndarray, np.ndarray]],  # (run_label, mode, ps, ys)
    *,
    out_path: Path,
    p_max_extrap: int = 300,
    p_min_fit: int = 1,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax_main = axes[0]
    ax_cinf = axes[1]

    p_extrap = np.linspace(max(2, p_min_fit), p_max_extrap, 500)

    linestyles = ["-", "--", "-.", ":"]
    model_colors = {"power": "#333333", "inv": "#888888", "log": "#bbbbbb"}
    model_ls = {"power": "-", "inv": "--", "log": ":"}

    for (run_label, mode, fits), (_, _, ps, ys) in zip(all_series, data_by_run):
        base_color = _MODE_COLOR.get(mode, "#666666")
        mode_label = _MODE_LABEL.get(mode, mode)
        data_label = f"{run_label} {mode_label}"
        # Plot data points
        marker = "o" if "mean_p" in mode else "s"
        ax_main.plot(ps, ys, marker=marker, color=base_color, lw=1.5,
                     label=data_label, zorder=5)

    # Plot fit curves from the first series only (to avoid clutter)
    # Pick the best-fit model for each series
    best_fits: List[Tuple[str, str, str, dict, np.ndarray, np.ndarray]] = []
    for (run_label, mode, fits), (_, _, ps, ys) in zip(all_series, data_by_run):
        base_color = _MODE_COLOR.get(mode, "#666666")
        mode_label = _MODE_LABEL.get(mode, mode)
        # Prefer stable fits; among those pick lowest AIC
        valid = {k: v for k, v in fits.items() if v.get("ok")}
        stable = {k: v for k, v in valid.items() if v.get("stable")}
        pool = stable if stable else valid
        if not pool:
            continue
        best_name = min(pool, key=lambda k: pool[k].get("aic", 1e99))
        best = pool[best_name]
        fn = _MODELS_DEF[best_name][0]
        y_extrap = fn(p_extrap, *best["params"])
        ax_main.plot(p_extrap, y_extrap, color=base_color, lw=1, ls="--", alpha=0.6,
                     label=f"  {mode_label} best fit ({best_name}, c∞={best['c_inf']:.4f}±{best['c_inf_err']:.4f})")
        # c_inf band
        ax_main.axhline(best["c_inf"], color=base_color, lw=0.8, ls=":", alpha=0.5)
        best_fits.append((run_label, mode, best_name, best, ps, ys))

    ax_main.set_xlabel("QAOA depth p")
    ax_main.set_ylabel(r"$c_\mathrm{typ}$  (log₂ slope vs $n$)")
    ax_main.set_title(f"c_typ(p) convergence  [fit p ≥ {p_min_fit}]")
    ax_main.legend(fontsize=7, loc="upper right")
    ax_main.grid(True, alpha=0.3)

    # Right panel: c_inf per model per series
    all_cinf_data: List[dict] = []
    labels_for_bar: List[str] = []
    for (run_label, mode, fits), (_, _, ps, ys) in zip(all_series, data_by_run):
        mode_label = _MODE_LABEL.get(mode, mode)
        series_label = f"{run_label}\n{mode_label}"
        for mname, res in fits.items():
            if not res.get("ok"):
                continue
            all_cinf_data.append({
                "series": series_label,
                "model": mname,
                "c_inf": res["c_inf"],
                "c_inf_err": res["c_inf_err"],
                "aic": res["aic"],
            })

    if all_cinf_data:
        models_present = sorted({d["model"] for d in all_cinf_data}, key=lambda m: list(_MODELS_DEF).index(m))
        series_present = list(dict.fromkeys(d["series"] for d in all_cinf_data))
        x = np.arange(len(series_present))
        w = 0.25
        offsets = np.linspace(-(len(models_present) - 1) * w / 2, (len(models_present) - 1) * w / 2, len(models_present))
        mcolors = ["#2c7bb6", "#d7191c", "#888888"]
        for i, mname in enumerate(models_present):
            cinf_vals, cinf_errs = [], []
            for s in series_present:
                match = [d for d in all_cinf_data if d["series"] == s and d["model"] == mname]
                if match:
                    cinf_vals.append(match[0]["c_inf"])
                    cinf_errs.append(match[0]["c_inf_err"])
                else:
                    cinf_vals.append(float("nan"))
                    cinf_errs.append(0.0)
            ax_cinf.bar(x + offsets[i], cinf_vals, w * 0.9, yerr=cinf_errs,
                        label=mname, color=mcolors[i % len(mcolors)], alpha=0.8,
                        capsize=4, error_kw={"lw": 1.5})
        ax_cinf.set_xticks(x)
        ax_cinf.set_xticklabels(series_present, fontsize=8)
        ax_cinf.set_ylabel(r"Extrapolated $c_\infty = \lim_{p\to\infty} c_\mathrm{typ}(p)$")
        ax_cinf.set_title("c_inf estimates by model (error bars = 1σ)")
        ax_cinf.legend(fontsize=8)
        ax_cinf.grid(True, axis="y", alpha=0.3)

        # Mark current p=50 value for reference
        if data_by_run:
            _, _, ps0, ys0 = data_by_run[0]
            last_idx = np.where(np.isfinite(ys0))[0]
            if len(last_idx):
                last_val = float(ys0[last_idx[-1]])
                ax_cinf.axhline(last_val, color="k", lw=0.8, ls="--", alpha=0.4,
                                label=f"c_typ at p={int(ps0[last_idx[-1]])} (last measured)")

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
        "--modes", default="bm24_mean_p_fixed_n,median_runtime_fixed_n",
        help="Comma-separated training modes to fit",
    )
    ap.add_argument(
        "--p-min-fit", type=int, default=5,
        help="Exclude depths below this from the fit (default 5, skips small-p transient)",
    )
    ap.add_argument(
        "--p-max-extrap", type=int, default=300,
        help="Extrapolate fit up to this depth in the plot",
    )
    ap.add_argument(
        "--output-dir", type=Path, default=None,
        help="Output directory (default: beside first run JSON)",
    )
    args = ap.parse_args()

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    run_jsons = [
        (_PHASECRAFT / p).resolve() if not p.is_absolute() else p.resolve()
        for p in args.run_jsons
    ]

    out_dir = args.output_dir or (
        _PHASECRAFT / "results/bm24_runs/analysis/ctyp_convergence"
    )
    out_dir = out_dir.resolve()

    all_series: List[Tuple[str, str, Dict[str, dict]]] = []
    data_by_run: List[Tuple[str, str, np.ndarray, np.ndarray]] = []
    json_rows: List[dict] = []

    print(f"\n{'='*70}")
    print(f"Fitting c_typ(p) convergence  [p_min_fit={args.p_min_fit}]")
    print(f"{'='*70}")

    for run_json in run_jsons:
        if not run_json.is_file():
            print(f"  SKIP {run_json} (not found)", flush=True)
            continue
        run_label = run_json.stem[:30]
        series = load_series(run_json, modes=modes)
        for mode, sv in series.items():
            ps, ys = sv["ps"], sv["ys"]
            fits = fit_models(ps, ys, p_min_fit=args.p_min_fit)
            all_series.append((run_label, mode, fits))
            data_by_run.append((run_label, mode, ps, ys))

            mode_label = _MODE_LABEL.get(mode, mode)
            print(f"\n{run_label}  |  {mode_label}  (n_data={np.isfinite(ys).sum()}, p_fit≥{args.p_min_fit})")
            print(f"  {'Model':<8}  {'c_inf':>8}  {'±1σ':>7}  {'AIC':>9}  {'stable':>6}  params")
            for mname, res in fits.items():
                if not res.get("ok"):
                    print(f"  {mname:<8}  FAILED: {res.get('error', '')}")
                    continue
                params_str = "  ".join(f"{v:.4f}" for v in res["params"])
                flag = "ok" if res.get("stable") else "UNSTBL"
                print(f"  {mname:<8}  {res['c_inf']:>8.5f}  ±{res['c_inf_err']:>6.5f}  {res['aic']:>9.2f}  {flag:>6}  [{params_str}]")
                json_rows.append({
                    "run": str(run_json),
                    "run_label": run_label,
                    "mode": mode,
                    "model": mname,
                    **res,
                })

    if not all_series:
        print("No data loaded — check run JSON paths.", file=sys.stderr)
        sys.exit(1)

    # Compare c_inf across models — print the spread as a reliability check
    print(f"\n{'='*70}")
    print("c_inf spread across models (smaller spread = more reliable extrapolation)")
    print(f"{'='*70}")
    for run_label, mode, fits in all_series:
        mode_label = _MODE_LABEL.get(mode, mode)
        valid = {k: v for k, v in fits.items() if v.get("ok")}
        stable = {k: v for k, v in valid.items() if v.get("stable")}
        pool = stable if stable else valid
        if not pool:
            continue
        cinfs = [v["c_inf"] for v in pool.values()]
        print(f"  {run_label}  {mode_label}:  "
              f"c_inf ∈ [{min(cinfs):.5f}, {max(cinfs):.5f}]  "
              f"spread={max(cinfs)-min(cinfs):.5f}  "
              f"({'stable' if stable else 'all unstable'} models)")
        best = min(pool, key=lambda k: pool[k].get("aic", 1e99))
        br = pool[best]
        print(f"    Best stable model ({best}):  c_inf = {br['c_inf']:.5f} ± {br['c_inf_err']:.5f}")

    png_path = out_dir / "ctyp_convergence_fit.png"
    plot_convergence(
        all_series, data_by_run,
        out_path=png_path,
        p_max_extrap=args.p_max_extrap,
        p_min_fit=args.p_min_fit,
    )

    json_path = out_dir / "ctyp_convergence_fit.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps({"fits": json_rows}, indent=2), encoding="utf-8")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
