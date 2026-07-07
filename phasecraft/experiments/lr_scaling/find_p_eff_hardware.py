#!/usr/bin/env python3
"""
Hardware-style p_eff for BM24 k-SAT (step beyond find_p_eff.py).

Implements the LR-QAOA QPU protocol more faithfully:

1. **Depth-scaling noise** — layer infidelity F(p) = (1 - ε)^p mixes the ideal
   state toward uniform; performance rises then falls like paper Fig. 2(d):

       p_noisy(p) = F(p) · p_ideal(p) + (1 - F(p)) · p_uniform

2. **Paper random baseline** — uniform random *bitstrings* (not random angles),
   99.73% threshold = μ + 3σ from bootstrap subsets (Sec. A.3). Flat in p.

3. **Reports** — p_peak (best noisy depth), p_eff (max p with r > random + 3σ),
   r_eff (Eq. 8 analog at p_eff).

Example::

    python experiments/lr_scaling/find_p_eff_hardware.py \\
        --run-json results/bm24_runs/06-09/run1/train12-tr100-te200-n12-18.json

    # Device sweep (per-layer infidelity ε):
    python experiments/lr_scaling/find_p_eff_hardware.py --sweep-eps 0.01,0.02,0.05,0.1
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Sequence

_MPLCONFIGDIR = Path(os.environ.get("MPLCONFIGDIR", os.environ.get("TMPDIR", "/tmp")))
if "MPLCONFIGDIR" not in os.environ:
    _MPLCONFIGDIR = _MPLCONFIGDIR / "phasecraft-matplotlib"
    _MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(_MPLCONFIGDIR)

import numpy as np

_PHASECRAFT = Path(__file__).resolve().parents[2]
for _p in (_PHASECRAFT.parent, _PHASECRAFT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from phasecraft.experiments.lr_scaling.find_p_eff import (  # noqa: E402
    load_trained_trace,
    mixed_state_baseline,
)
from phasecraft.lib.sim.bm24_qaoa_sim import (  # noqa: E402
    build_h_diagonal,
    generate_benchmark_dataset,
    make_lr_angles,
    per_instance_success_probability,
    run_qaoa,
)


def _n_qubits(h_diag: np.ndarray) -> int:
    return int(round(np.log2(int(h_diag.size))))


def _uniform_random_p_succ(h_diag: np.ndarray) -> float:
    return float(np.sum(h_diag == 0.0)) / float(h_diag.size)


def layer_fidelity(depth: int, eps_per_layer: float) -> float:
    """Survival probability after p QAOA layers (one infidelity ε per layer)."""
    depth = int(depth)
    eps = float(eps_per_layer)
    if depth <= 0:
        return 1.0
    return float((1.0 - eps) ** depth)


def apply_depth_noise(p_ideal: float, p_uniform: float, fidelity: float) -> float:
    """Mix ideal success probability toward fully-mixed measurement baseline."""
    f = float(np.clip(fidelity, 0.0, 1.0))
    return f * float(p_ideal) + (1.0 - f) * float(p_uniform)


def ideal_mean_p_succ(
    h_list: Sequence[np.ndarray],
    betas: np.ndarray,
    gammas: np.ndarray,
) -> float:
    vals: List[float] = []
    for h_diag in h_list:
        n = _n_qubits(h_diag)
        psi = run_qaoa(h_diag, betas, gammas, n)
        vals.append(per_instance_success_probability(psi, h_diag))
    return float(np.mean(vals)) if vals else float("nan")


def bootstrap_mean_ci(
    values: Sequence[float],
    *,
    num_bootstrap: int,
    seed: int,
    ci_level: float,
) -> dict:
    """Bootstrap a mean confidence interval over held-out instances."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    if arr.size == 1 or int(num_bootstrap) <= 0:
        mean = float(np.mean(arr))
        return {"mean": mean, "ci_low": mean, "ci_high": mean}

    rng = np.random.default_rng(int(seed))
    idx = rng.integers(0, arr.size, size=(int(num_bootstrap), arr.size))
    means = np.mean(arr[idx], axis=1)
    alpha = max(0.0, min(1.0, 1.0 - float(ci_level)))
    lo = 100.0 * (alpha / 2.0)
    hi = 100.0 * (1.0 - alpha / 2.0)
    return {
        "mean": float(np.mean(arr)),
        "ci_low": float(np.percentile(means, lo)),
        "ci_high": float(np.percentile(means, hi)),
    }


def eval_trained_depth(
    dg: float,
    db: float,
    depth: int,
    h_list: Sequence[np.ndarray],
    *,
    beta_schedule: str,
    eps_per_layer: float,
) -> dict:
    """Return ideal/uniform/noisy probabilities, including per-instance values."""
    betas, gammas = make_lr_angles(
        dg, db, depth, beta_schedule=beta_schedule, angle_convention="bm24",
    )
    ideals: List[float] = []
    uniforms: List[float] = []
    for h_diag in h_list:
        n = _n_qubits(h_diag)
        psi = run_qaoa(h_diag, betas, gammas, n)
        p_i = per_instance_success_probability(psi, h_diag)
        p_u = _uniform_random_p_succ(h_diag)
        f = layer_fidelity(depth, eps_per_layer)
        ideals.append(p_i)
        uniforms.append(p_u)
    ideal_arr = np.asarray(ideals, dtype=float)
    uniform_arr = np.asarray(uniforms, dtype=float)
    f = layer_fidelity(depth, eps_per_layer)
    noisy_arr = f * ideal_arr + (1.0 - f) * uniform_arr
    return {
        "p_ideal": float(np.mean(ideal_arr)),
        "p_uniform": float(np.mean(uniform_arr)),
        "p_noisy": float(np.mean(noisy_arr)),
        "p_ideal_by_instance": ideal_arr,
        "p_uniform_by_instance": uniform_arr,
        "p_noisy_by_instance": noisy_arr,
    }


def bitstring_random_baseline(
    h_list: Sequence[np.ndarray],
    *,
    num_bootstrap: int = 100,
    samples_per_subset: int = 200,
    seed: int = 0,
) -> dict:
    """
    Paper Sec. A.3 baseline: uniform random bitstrings, bootstrap μ ± 3σ.

    For each bootstrap replicate, draw ``samples_per_subset`` random assignments
    per instance and average the SAT indicator; then aggregate over instances.
    """
    rng = np.random.default_rng(int(seed))
    replicate_means: List[float] = []
    for _ in range(int(num_bootstrap)):
        inst_scores: List[float] = []
        for h_diag in h_list:
            n = _n_qubits(h_diag)
            n_sat = int(np.sum(h_diag == 0.0))
            n_total = int(h_diag.size)
            # Hypergeometric: k successes in samples without enumerating 2^n.
            k = min(int(samples_per_subset), n_total)
            hits = int(rng.hypergeometric(n_sat, n_total - n_sat, k))
            inst_scores.append(hits / float(k))
        replicate_means.append(float(np.mean(inst_scores)))
    arr = np.asarray(replicate_means, dtype=float)
    mu = float(np.mean(arr))
    sigma = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    return {
        "mean": mu,
        "std": sigma,
        "threshold_3sigma": mu + 3.0 * sigma,
        "threshold_lo_3sigma": mu - 3.0 * sigma,
        "replicate_means": arr,
    }


def find_p_peak(depths: Sequence[int], scores: Dict[int, float]) -> int | None:
    finite = [(int(p), scores[int(p)]) for p in depths if np.isfinite(scores[int(p)])]
    if not finite:
        return None
    return max(finite, key=lambda t: t[1])[0]


def classify_threshold_result(
    *,
    mean: float,
    ci_low: float,
    ci_high: float,
    threshold: float,
) -> str:
    """Classify a depth relative to the random threshold."""
    if not np.isfinite(mean) or not np.isfinite(threshold):
        return "invalid"
    if np.isfinite(ci_low) and ci_low > threshold:
        return "confirmed_above"
    if np.isfinite(ci_high) and ci_high <= threshold:
        return "confirmed_at_or_below"
    if mean > threshold:
        return "mean_above_uncertain"
    return "mean_at_or_below_uncertain"


def _format_p_eff_report(estimate: int | None, status: str) -> str:
    if estimate is None:
        return "none"
    if status == "right_censored":
        return f">={estimate}"
    return str(estimate)


def measure_p_eff_from_rows(
    depths: Sequence[int],
    rows_by_depth: Dict[int, dict],
    threshold: float,
) -> dict:
    """
    Measure p_eff from sampled depths without hiding censoring.

    The legacy point estimate is the largest sampled depth on the descending
    branch with mean noisy performance above threshold.  If the scan never
    observes a failure after the peak, the estimate is right-censored and must
    be read as a lower bound.
    """
    sorted_depths = sorted(int(p) for p in depths)
    noisy_scores = {p: float(rows_by_depth[p]["p_noisy"]) for p in sorted_depths}
    p_peak = find_p_peak(sorted_depths, noisy_scores)
    if p_peak is None:
        return {
            "p_peak": None,
            "p_eff": None,
            "p_eff_status": "no_finite_scores",
            "p_eff_report": "none",
            "p_eff_lower_bound": None,
            "p_eff_upper_bound": None,
            "first_fail_depth": None,
            "crossing_bracket": None,
            "p_eff_confirmed_lower_bound": None,
            "p_eff_confirmed_status": "no_finite_scores",
        }

    tail = [p for p in sorted_depths if p >= p_peak]
    last_pass: int | None = None
    first_fail: int | None = None
    for p in tail:
        if rows_by_depth[p]["passes_mean"]:
            last_pass = p
        else:
            first_fail = p
            break

    if last_pass is None:
        status = "none_after_peak"
        lower = None
        upper = first_fail
    elif first_fail is None:
        status = "right_censored"
        lower = last_pass
        upper = None
    else:
        status = "bracketed"
        lower = last_pass
        upper = first_fail

    confirmed_lower: int | None = None
    confirmed_stop: int | None = None
    confirmed_stop_state: str | None = None
    for p in tail:
        state = str(rows_by_depth[p]["threshold_state"])
        if state == "confirmed_above":
            confirmed_lower = p
            continue
        confirmed_stop = p
        confirmed_stop_state = state
        break
    if confirmed_lower is None:
        confirmed_status = "not_confirmed_at_peak"
    elif confirmed_stop is None:
        confirmed_status = "right_censored"
    else:
        confirmed_status = f"stopped_by_{confirmed_stop_state}"

    crossing = [last_pass, first_fail] if last_pass is not None and first_fail is not None else None
    return {
        "p_peak": p_peak,
        "p_eff": last_pass,
        "p_eff_status": status,
        "p_eff_report": _format_p_eff_report(last_pass, status),
        "p_eff_lower_bound": lower,
        "p_eff_upper_bound": upper,
        "first_fail_depth": first_fail,
        "crossing_bracket": crossing,
        "p_eff_confirmed_lower_bound": confirmed_lower,
        "p_eff_confirmed_status": confirmed_status,
        "p_eff_confirmed_stop_depth": confirmed_stop,
        "p_eff_confirmed_stop_state": confirmed_stop_state,
    }


def plot_hardware_p_eff(
    *,
    depths: List[int],
    ideal: List[float],
    noisy: List[float],
    rand_thresh: float,
    rand_lo: float,
    rand_mu: float,
    mixed: float,
    p_eff: int | None,
    p_peak: int | None,
    eps_per_layer: float,
    out_path: Path,
    title: str,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.5, 5))
    d = np.array(depths, dtype=float)
    ax.plot(d, ideal, "^:", color="#abd9e9", lw=1.2, ms=5, alpha=0.9,
            label="noise-free trained")
    ax.plot(d, noisy, "o-", color="#2c7bb6", lw=2, ms=6,
            label=rf"noisy ($F(p)=(1-\varepsilon)^p$, $\varepsilon={eps_per_layer:g}$)")
    ax.axhspan(rand_lo, rand_thresh, color="#d7191c", alpha=0.12,
               label="random bitstrings 99.73% band")
    ax.axhline(rand_mu, color="#d7191c", ls="--", lw=1, alpha=0.7,
               label=rf"random bitstrings $\mu={rand_mu:.4g}$")
    ax.axhline(mixed, color="#fdae61", ls=":", lw=1.2, alpha=0.8,
               label=rf"exact $p_{{\mathrm{{uniform}}}}$ ({mixed:.4g})")
    if p_peak is not None:
        ax.axvline(p_peak, color="#7570b3", ls="-.", lw=1.2, alpha=0.8,
                   label=rf"$p_{{\mathrm{{peak}}}}={p_peak}$")
    if p_eff is not None:
        ax.axvline(p_eff, color="black", ls="--", lw=1.3,
                   label=rf"$p_{{\mathrm{{eff}}}}={p_eff}$")
    ax.set_xlabel("LR-QAOA depth $p$")
    ax.set_ylabel("mean $p_{\\mathrm{succ}}$")
    ax.set_title(title)
    ax.legend(fontsize=7.5, loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_sweep_grid(
    summaries: List[dict],
    *,
    rand_thresh: float,
    rand_lo: float,
    rand_mu: float,
    mixed: float,
    out_path: Path,
    eval_n: int,
) -> None:
    import matplotlib.pyplot as plt

    n = len(summaries)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.6 * nrows), squeeze=False)
    for i, summary in enumerate(summaries):
        ax = axes[i // ncols][i % ncols]
        eps = float(summary["eps_per_layer"])
        rows = summary["depths"]
        depths = [int(r["depth"]) for r in rows]
        d = np.array(depths, dtype=float)
        ideal = [float(r["p_ideal"]) for r in rows]
        noisy = [float(r["p_noisy"]) for r in rows]
        ax.plot(d, ideal, "^:", color="#abd9e9", lw=1, ms=3, alpha=0.8)
        ax.plot(d, noisy, "o-", color="#2c7bb6", lw=1.5, ms=4)
        ax.axhspan(rand_lo, rand_thresh, color="#d7191c", alpha=0.10)
        ax.axhline(rand_mu, color="#d7191c", ls="--", lw=0.8, alpha=0.6)
        pp, ph, pe = summary.get("p_peak"), summary.get("p_half"), summary.get("p_eff")
        if pp is not None:
            ax.axvline(pp, color="#7570b3", ls="-.", lw=0.9, alpha=0.7)
        if pe is not None:
            ax.axvline(pe, color="black", ls="--", lw=0.9, alpha=0.75)
        pe_report = summary.get("p_eff_report", pe)
        ax.set_title(
            rf"$\varepsilon={eps:g}$  $p_{{\mathrm{{peak}}}}={pp}$  "
            rf"$p_{{\mathrm{{eff}}}}={pe_report}$",
            fontsize=9,
        )
        ax.grid(True, alpha=0.25)
        if i // ncols == nrows - 1:
            ax.set_xlabel("$p$")
        if i % ncols == 0:
            ax.set_ylabel(r"mean $p_{\mathrm{succ}}$")
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    fig.suptitle(rf"Hardware-style $p_{{\mathrm{{eff}}}}$ sweep ($n={eval_n}$)", fontsize=11, y=1.02)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def run_one(
    *,
    run_json: Path,
    eval_n: int,
    test_size: int,
    eps_per_layer: float,
    num_bootstrap: int,
    perf_bootstrap: int,
    ci_level: float,
    samples_per_subset: int,
    random_seed: int,
    depth_max: int | None,
    write_files: bool = False,
    out_dir: Path | None = None,
) -> dict:
    cfg, trace = load_trained_trace(run_json)
    k = int(cfg.get("k", 8))
    r = float(cfg.get("r", 176.54))
    seed = int(cfg.get("seed", 27))
    beta_schedule = str(cfg.get("lr_beta_schedule", "decreasing"))

    dataset = generate_benchmark_dataset(
        [eval_n], k=k, r=r,
        test_size=int(test_size), base_seed=seed + 999_001,
        require_sat=True,
    )
    h_list = [build_h_diagonal(inst["clauses"], eval_n) for inst in dataset[eval_n]]
    mixed = mixed_state_baseline(h_list)

    rb = bitstring_random_baseline(
        h_list,
        num_bootstrap=int(num_bootstrap),
        samples_per_subset=int(samples_per_subset),
        seed=int(random_seed),
    )
    threshold = rb["threshold_3sigma"]

    depths = [int(row["depth"]) for row in trace]
    if depth_max is not None:
        depths = [p for p in depths if p <= int(depth_max)]
        trace = [row for row in trace if int(row["depth"]) in set(depths)]

    rows: List[dict] = []
    rows_by_depth: Dict[int, dict] = {}
    for row in trace:
        p = int(row["depth"])
        eval_row = eval_trained_depth(
            float(row["delta_gamma"]), float(row["delta_beta"]), p, h_list,
            beta_schedule=beta_schedule, eps_per_layer=float(eps_per_layer),
        )
        f = layer_fidelity(p, eps_per_layer)
        noisy_ci = bootstrap_mean_ci(
            eval_row["p_noisy_by_instance"],
            num_bootstrap=int(perf_bootstrap),
            seed=int(random_seed) + 1_000_003 * p + int(round(10_000 * float(eps_per_layer))),
            ci_level=float(ci_level),
        )
        state = classify_threshold_result(
            mean=float(eval_row["p_noisy"]),
            ci_low=float(noisy_ci["ci_low"]),
            ci_high=float(noisy_ci["ci_high"]),
            threshold=threshold,
        )
        passes = bool(eval_row["p_noisy"] > threshold)
        saved_row = {
            "depth": p,
            "fidelity_F": f,
            "p_ideal": float(eval_row["p_ideal"]),
            "p_uniform_instance_mean": float(eval_row["p_uniform"]),
            "p_noisy": float(eval_row["p_noisy"]),
            "p_noisy_ci_low": float(noisy_ci["ci_low"]),
            "p_noisy_ci_high": float(noisy_ci["ci_high"]),
            "ci_level": float(ci_level),
            "threshold_state": state,
            "passes_mean": passes,
            "passes": passes,
            "delta_gamma": float(row["delta_gamma"]),
            "delta_beta": float(row["delta_beta"]),
        }
        rows.append(saved_row)
        rows_by_depth[p] = saved_row
        print(
            f"  ε={eps_per_layer:g} p={p:3d}  F={f:.4f}  "
            f"ideal={eval_row['p_ideal']:.5f}  noisy={eval_row['p_noisy']:.5f}  "
            f"CI{100*float(ci_level):.0f}%=[{noisy_ci['ci_low']:.5f}, {noisy_ci['ci_high']:.5f}]  "
            f"thresh={threshold:.5f}  {state}"
        )

    measurement = measure_p_eff_from_rows(depths, rows_by_depth, threshold)
    p_eff = measurement["p_eff"]
    p_peak = measurement["p_peak"]

    peak_val = rows_by_depth[p_peak]["p_noisy"] if p_peak is not None else float("nan")
    half_level = 0.5 * peak_val if np.isfinite(peak_val) else float("nan")
    p_half: int | None = None
    if p_peak is not None and np.isfinite(half_level):
        tail = sorted(p for p in depths if int(p) >= p_peak)
        for p in reversed(tail):
            if rows_by_depth[int(p)]["p_noisy"] >= half_level:
                p_half = int(p)
                break

    r_eff = float("nan")
    if p_eff is not None:
        r_max = rows_by_depth[p_eff]["p_noisy"]
        r_rand = threshold
        denom = 1.0 - r_rand
        r_eff = (r_max - r_rand) / denom if denom > 0 else float("nan")

    summary = {
        "mode": "hardware_depth_scaling",
        "run_json": str(run_json),
        "eval_n": eval_n,
        "test_size": int(test_size),
        "eps_per_layer": float(eps_per_layer),
        "noise_model": "p_noisy = F(p)*p_ideal + (1-F(p))*p_uniform, F=(1-eps)^p",
        "random_baseline": "uniform_bitstring_bootstrap",
        "num_bootstrap": int(num_bootstrap),
        "performance_bootstrap": int(perf_bootstrap),
        "ci_level": float(ci_level),
        "samples_per_subset": int(samples_per_subset),
        "mixed_state_exact": mixed,
        "random_bitstring_mean": rb["mean"],
        "random_bitstring_std": rb["std"],
        "random_threshold_3sigma": threshold,
        "random_threshold_lo_3sigma": rb["threshold_lo_3sigma"],
        "p_peak": p_peak,
        "p_peak_noisy_value": peak_val,
        "p_half": p_half,
        "p_eff": p_eff,
        "p_eff_status": measurement["p_eff_status"],
        "p_eff_report": measurement["p_eff_report"],
        "p_eff_lower_bound": measurement["p_eff_lower_bound"],
        "p_eff_upper_bound": measurement["p_eff_upper_bound"],
        "first_fail_depth": measurement["first_fail_depth"],
        "crossing_bracket": measurement["crossing_bracket"],
        "p_eff_confirmed_lower_bound": measurement["p_eff_confirmed_lower_bound"],
        "p_eff_confirmed_status": measurement["p_eff_confirmed_status"],
        "p_eff_confirmed_stop_depth": measurement["p_eff_confirmed_stop_depth"],
        "p_eff_confirmed_stop_state": measurement["p_eff_confirmed_stop_state"],
        "r_eff_at_p_eff": r_eff,
        "depths": rows,
    }

    if write_files and out_dir is not None:
        stem = run_json.stem[:36]
        tag = f"p_eff_hw_{stem}_n{eval_n}_eps{eps_per_layer:g}"
        json_path = out_dir / f"{tag}.json"
        json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        plot_hardware_p_eff(
            depths=depths,
            ideal=[r["p_ideal"] for r in rows],
            noisy=[rows_by_depth[p]["p_noisy"] for p in depths],
            rand_thresh=threshold,
            rand_lo=rb["threshold_lo_3sigma"],
            rand_mu=rb["mean"],
            mixed=mixed,
            p_eff=p_eff,
            p_peak=p_peak,
            eps_per_layer=float(eps_per_layer),
            out_path=out_dir / f"{tag}.png",
            title=rf"Hardware $p_{{\mathrm{{eff}}}}$ ($n={eval_n}$, $\varepsilon={eps_per_layer:g}$)",
        )

    print(
        f"  -> p_peak={p_peak}, p_half={p_half}, "
        f"p_eff={measurement['p_eff_report']} ({measurement['p_eff_status']}), "
        f"confirmed_lower={measurement['p_eff_confirmed_lower_bound']}, "
        f"r_eff={r_eff:.4f}"
    )
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--run-json",
        type=Path,
        default=Path("results/bm24_runs/06-09/run1/train12-tr100-te200-n12-18.json"),
    )
    ap.add_argument("--eval-n", type=int, default=12)
    ap.add_argument("--test-size", type=int, default=40)
    ap.add_argument("--eps-per-layer", type=float, default=0.03,
                    help="Per-QAOA-layer infidelity ε in F(p)=(1-ε)^p")
    ap.add_argument("--sweep-eps", type=str, default=None,
                    help="Comma-separated ε values; writes hardware_sweep_summary.json")
    ap.add_argument("--num-bootstrap", type=int, default=100)
    ap.add_argument("--perf-bootstrap", type=int, default=1000,
                    help="Bootstrap replicates for CI over held-out instance means")
    ap.add_argument("--ci-level", type=float, default=0.95,
                    help="Confidence level for noisy performance CI")
    ap.add_argument("--samples-per-subset", type=int, default=200)
    ap.add_argument("--random-seed", type=int, default=4242)
    ap.add_argument("--depth-max", type=int, default=None,
                    help="Maximum trained depth to evaluate; default uses all depths in the run JSON")
    ap.add_argument("--output-dir", type=Path, default=None)
    args = ap.parse_args()

    run_json = (_PHASECRAFT / args.run_json).resolve() if not args.run_json.is_absolute() else args.run_json
    out_dir = args.output_dir or (_PHASECRAFT / "results/bm24_runs/analysis/p_eff")
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    eps_list = (
        [float(x.strip()) for x in args.sweep_eps.split(",") if x.strip()]
        if args.sweep_eps
        else [float(args.eps_per_layer)]
    )

    summaries: List[dict] = []
    for eps in eps_list:
        print(f"\n{'='*60}\nε_per_layer = {eps:g}\n{'='*60}")
        summaries.append(run_one(
            run_json=run_json,
            eval_n=int(args.eval_n),
            test_size=int(args.test_size),
            eps_per_layer=eps,
            num_bootstrap=int(args.num_bootstrap),
            perf_bootstrap=int(args.perf_bootstrap),
            ci_level=float(args.ci_level),
            samples_per_subset=int(args.samples_per_subset),
            random_seed=int(args.random_seed),
            depth_max=int(args.depth_max) if args.depth_max else None,
            write_files=(len(eps_list) == 1),
            out_dir=out_dir,
        ))

    payload = {
        "run_json": str(run_json),
        "eval_n": int(args.eval_n),
        "test_size": int(args.test_size),
        "num_bootstrap": int(args.num_bootstrap),
        "performance_bootstrap": int(args.perf_bootstrap),
        "ci_level": float(args.ci_level),
        "random_threshold_3sigma": summaries[0]["random_threshold_3sigma"],
        "random_bitstring_mean": summaries[0]["random_bitstring_mean"],
        "rows": [
            {
                "eps_per_layer": s["eps_per_layer"],
                "p_peak": s["p_peak"],
                "p_half": s["p_half"],
                "p_eff": s["p_eff"],
                "p_eff_status": s["p_eff_status"],
                "p_eff_report": s["p_eff_report"],
                "p_eff_lower_bound": s["p_eff_lower_bound"],
                "p_eff_upper_bound": s["p_eff_upper_bound"],
                "first_fail_depth": s["first_fail_depth"],
                "crossing_bracket": s["crossing_bracket"],
                "p_eff_confirmed_lower_bound": s["p_eff_confirmed_lower_bound"],
                "p_eff_confirmed_status": s["p_eff_confirmed_status"],
                "p_eff_confirmed_stop_depth": s["p_eff_confirmed_stop_depth"],
                "p_eff_confirmed_stop_state": s["p_eff_confirmed_stop_state"],
                "r_eff_at_p_eff": s["r_eff_at_p_eff"],
                "depths": s["depths"],
            }
            for s in summaries
        ],
    }
    json_path = out_dir / ("hardware_sweep.json" if len(eps_list) > 1 else "hardware.json")
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nWrote {json_path}")

    if len(eps_list) > 1:
        plot_sweep_grid(
            summaries,
            rand_thresh=summaries[0]["random_threshold_3sigma"],
            rand_lo=summaries[0]["random_threshold_lo_3sigma"],
            rand_mu=summaries[0]["random_bitstring_mean"],
            mixed=summaries[0]["mixed_state_exact"],
            out_path=out_dir / "hardware_sweep.png",
            eval_n=int(args.eval_n),
        )
        print(f"Wrote {out_dir / 'hardware_sweep.png'}")
    elif summaries:
        plot_hardware_p_eff(
            depths=[int(r["depth"]) for r in summaries[0]["depths"]],
            ideal=[float(r["p_ideal"]) for r in summaries[0]["depths"]],
            noisy=[float(r["p_noisy"]) for r in summaries[0]["depths"]],
            rand_thresh=summaries[0]["random_threshold_3sigma"],
            rand_lo=summaries[0]["random_threshold_lo_3sigma"],
            rand_mu=summaries[0]["random_bitstring_mean"],
            mixed=summaries[0]["mixed_state_exact"],
            p_eff=summaries[0]["p_eff"],
            p_peak=summaries[0]["p_peak"],
            eps_per_layer=float(summaries[0]["eps_per_layer"]),
            out_path=out_dir / "hardware.png",
            title=rf"Hardware $p_{{\mathrm{{eff}}}}$ ($n={args.eval_n}$)",
        )
        print(f"Wrote {out_dir / 'hardware.png'}")


if __name__ == "__main__":
    main()
