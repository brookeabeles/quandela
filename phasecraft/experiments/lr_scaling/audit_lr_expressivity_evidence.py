#!/usr/bin/env python3
"""Audit LR-QAOA vs full-QAOA expressivity evidence (no new asymptotic claims)."""

from __future__ import annotations

import json
import math
import os
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "phasecraft-mpl"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress

_PHASECRAFT = Path(__file__).resolve().parents[2]
for _p in (_PHASECRAFT.parent, _PHASECRAFT, Path(__file__).resolve().parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from phasecraft.lib.sim.bm24_qaoa_sim import (  # noqa: E402
    build_h_diagonal,
    make_lr_angles,
    per_instance_success_probability,
    run_qaoa,
)
from plot_compare_run1_run4 import classical_slopes, load_classical_baselines  # noqa: E402

LN2 = math.log(2.0)
BM24_AMP = 0.69
BM24_EXP = 0.32
EVAL_WINDOW = (12, 18)
DEPTHS_FIG3 = [2, 10, 20, 40, 50, 60, 80, 100]
OUT_DIR = _PHASECRAFT / "bm24_runs/analysis/expressivity_audit"

QAOA_REG_JSONS = [
    _PHASECRAFT / "qaoa-reg/results/cpu-prx-small.json",
    _PHASECRAFT / "qaoa-reg/results/smoke.json",
    _PHASECRAFT / "qaoa-reg/results/batched-smoke.json",
    _PHASECRAFT / "results/gpu_qaoa_vs_lr/smoke-qaoa-vs-lr-gpu.json",
]

MULTI_SEED_CACHES = [
    (_PHASECRAFT / "results/bm24_runs/multi_seed_ctyp_cann/06-24/run1/train12-tr100-te200-n12-18-seed0-ctyp-cann-cache.json", 0),
    (_PHASECRAFT / "results/bm24_runs/multi_seed_ctyp_cann/06-24/run2/train12-tr100-te200-n12-18-seed42-ctyp-cann-cache.json", 42),
    (_PHASECRAFT / "results/bm24_runs/06-09/run1/train12-tr100-te200-n12-18-ctyp-cann-cache.json", 27),
]


def fit_log2_slope(ns: Sequence[int], ys: Sequence[float]) -> float:
    n_arr = np.asarray(ns, dtype=float)
    y_arr = np.asarray(ys, dtype=float)
    mask = np.isfinite(y_arr) & (y_arr > 0)
    if int(mask.sum()) < 2:
        return float("nan")
    res = linregress(n_arr[mask], np.log(y_arr[mask]))
    return float(res.slope / LN2)


def bm24_analytic_c(p: float) -> float:
    return float(BM24_AMP * (float(p) ** (-BM24_EXP)))


def fit_lr_schedule(
    betas: np.ndarray,
    gammas: np.ndarray,
    *,
    beta_schedule: str = "decreasing",
) -> Tuple[float, float, np.ndarray, np.ndarray, float, float]:
    """Least-squares fit of (delta_gamma, delta_beta) to full angle vectors."""
    p = len(betas)
    t = np.arange(1, p + 1, dtype=float) / float(p)
    s = 1.0 - t if beta_schedule == "decreasing" else t
    dg = float(np.dot(gammas, t) / np.dot(t, t))
    db = float(np.dot(betas, s) / np.dot(s, s))
    lr_betas, lr_gammas = make_lr_angles(dg, db, p, beta_schedule=beta_schedule)
    full = np.concatenate([gammas, betas])
    proj = np.concatenate([lr_gammas, lr_betas])
    resid = full - proj
    abs_res = float(np.linalg.norm(resid))
    norm_full = float(np.linalg.norm(full))
    rel_res = abs_res / norm_full if norm_full > 0 else float("nan")
    return dg, db, lr_betas, lr_gammas, abs_res, rel_res


def eval_angles_on_dataset(
    dataset: Dict[int, List[np.ndarray]],
    betas: np.ndarray,
    gammas: np.ndarray,
) -> Dict[int, dict]:
    out: Dict[int, dict] = {}
    for n, insts in dataset.items():
        probs = []
        runtimes = []
        for h in insts:
            psi = run_qaoa(h, betas, gammas, int(n))
            p = float(per_instance_success_probability(psi, h))
            probs.append(p)
            runtimes.append(1.0 / max(p, 1e-300))
        probs_arr = np.asarray(probs, dtype=float)
        rt_arr = np.asarray(runtimes, dtype=float)
        out[int(n)] = {
            "mean_p_succ": float(np.mean(probs_arr)),
            "median_p_succ": float(np.median(probs_arr)),
            "median_runtime": float(np.median(rt_arr)),
            "probs": probs_arr,
        }
    return out


def generate_eval_dataset(
    *,
    k: int,
    r: float,
    eval_seed: int,
    test_size: int,
    n_values: Sequence[int],
) -> Dict[int, List[np.ndarray]]:
    qaoa_reg = _PHASECRAFT / "qaoa-reg"
    if str(qaoa_reg) not in sys.path:
        sys.path.insert(0, str(qaoa_reg))
    from run_benchmark import generate_benchmark_dataset as gen_ds  # noqa: E402

    return gen_ds(
        n_values,
        k=k,
        r=r,
        test_size=test_size,
        base_seed=eval_seed,
        require_sat=True,
        m_sampling="notebook",
        max_trials_per_n=1_000_000,
    )


def bootstrap_gap_cis(
    by_n: Dict[int, dict],
    *,
    n_boot: int,
    rng: np.random.Generator,
) -> Dict[str, Tuple[float, float]]:
    ns = sorted(by_n)
    vals = {k: np.empty(n_boot) for k in ("c_rt", "c_sp", "delta_obj", "delta_bench", "delta_total")}
    for b in range(n_boot):
        med_rt = []
        inv_mean = []
        for n in ns:
            probs = by_n[n]["probs"]
            idx = rng.integers(0, len(probs), len(probs))
            ps = probs[idx]
            med_rt.append(float(np.median(1.0 / np.maximum(ps, 1e-300))))
            inv_mean.append(float(np.mean(1.0 / np.maximum(ps, 1e-300))))
        c_rt = fit_log2_slope(ns, med_rt)
        c_sp = fit_log2_slope(ns, inv_mean)
        vals["c_rt"][b] = c_rt
        vals["c_sp"][b] = c_sp
        vals["delta_obj"][b] = c_rt - c_sp
        # benchmark gap needs depth; filled by caller
    return {k: (float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))) for k, v in vals.items()}


@dataclass
class ProvenanceRow:
    label: str
    value: float
    source_file: str
    field_name: str
    objective: str
    seeds: str
    n_window: str
    fit_model: str
    value_kind: str  # raw_depth | fitted_asymptote | horizontal_reference | analytic_formula


def collect_provenance() -> List[ProvenanceRow]:
    rows: List[ProvenanceRow] = []

    summary = json.loads(
        (_PHASECRAFT / "results/bm24_runs/analysis/multi_seed_ctyp_cann/multi_seed_summary.json").read_text()
    )
    for r in summary["rows"]:
        if r["depth"] == 100:
            rows.append(
                ProvenanceRow(
                    label=f"LR c_rt seed{r['seed']} @ p=100",
                    value=r["c_typ_mean_p"],
                    source_file="multi_seed_summary.json",
                    field_name="c_typ_mean_p",
                    objective="bm24_mean_p_fixed_n (angles); median(1/p) vs n",
                    seeds=str(r["seed"]),
                    n_window="12-18",
                    fit_model="log2 linear regression on median(1/p_succ) vs n",
                    value_kind="raw_depth",
                )
            )

    exp = json.loads(
        (_PHASECRAFT / "results/bm24_runs/analysis/ctyp_convergence/objective_exp_convergence.json").read_text()
    )
    for f in exp["fits"]:
        if f["panel"] == "multiseed":
            rows.append(
                ProvenanceRow(
                    label=f"c_inf multiseed {f['mode_label']}",
                    value=f["c_inf"],
                    source_file="objective_exp_convergence.json",
                    field_name="c_inf",
                    objective=f["mode_label"],
                    seeds="0,27,42",
                    n_window="12-18",
                    fit_model=f"c_inf + A p^(-beta), p>=10",
                    value_kind="fitted_asymptote",
                )
            )

    classical = load_classical_baselines(None)
    if classical:
        ws, lm = classical_slopes(classical, 12, 18)
        rows.append(
            ProvenanceRow(
                label="WalkSAT c",
                value=ws,
                source_file="scaling-tn14.json setup.classical",
                field_name="walksat median flips log2 slope",
                objective="classical baseline",
                seeds="n/a",
                n_window="12-18",
                fit_model="log2 linear on median flips vs n",
                value_kind="horizontal_reference",
            )
        )
        rows.append(
            ProvenanceRow(
                label="BM24 analytic @ p=100",
                value=bm24_analytic_c(100),
                source_file="plot_multi_seed_objective_ab.py constants",
                field_name="0.69*p^{-0.32}",
                objective="analytic reference",
                seeds="n/a",
                n_window="n/a",
                fit_model="pure power (no offset)",
                value_kind="analytic_formula",
            )
        )

    for jp in QAOA_REG_JSONS:
        if not jp.is_file():
            continue
        payload = json.loads(jp.read_text())
        cfg = payload["config"]
        n_rng = f"{cfg['n_min']}-{cfg['n_max']}"
        finite = int(cfg["n_max"]) <= 13
        for tr in payload.get("trace", []):
            d = int(tr["depth"])
            q_per_n = tr.get("qaoa_mean_p_succ_per_n") or {}
            l_per_n = tr.get("lr_eval", {}).get("per_n") or {}
            if not q_per_n or not l_per_n:
                continue
            n_key = "12" if "12" in q_per_n else sorted(q_per_n.keys(), key=int)[0]
            rows.append(
                ProvenanceRow(
                    label=f"full QAOA mean_p @ n={n_key} p={d} ({jp.stem})",
                    value=float(q_per_n[n_key]),
                    source_file=jp.name,
                    field_name=f"qaoa_mean_p_succ_per_n['{n_key}']",
                    objective="bm24_mean_p_fixed_n",
                    seeds=str(cfg["seed"]),
                    n_window=n_rng,
                    fit_model="n/a",
                    value_kind="raw_depth (finite-size benchmark)" if finite else "raw_depth",
                )
            )
            rows.append(
                ProvenanceRow(
                    label=f"LR mean_p @ n={n_key} p={d} ({jp.stem})",
                    value=float(l_per_n[n_key]["mean_p_succ"]),
                    source_file=jp.name,
                    field_name=f"lr_eval.per_n['{n_key}'].mean_p_succ",
                    objective="bm24_mean_p_fixed_n",
                    seeds=str(cfg["seed"]),
                    n_window=n_rng,
                    fit_model="n/a",
                    value_kind="raw_depth (finite-size benchmark)" if finite else "raw_depth",
                )
            )
    return rows


def audit_qaoa_angle_projections() -> List[dict]:
    results = []
    for jp in QAOA_REG_JSONS:
        if not jp.is_file():
            continue
        payload = json.loads(jp.read_text())
        cfg = payload["config"]
        for tr in payload["trace"]:
            betas = np.asarray(tr["qaoa_betas"], dtype=float)
            gammas = np.asarray(tr["qaoa_gammas"], dtype=float)
            dg, db, lb, lg, abs_r, rel_r = fit_lr_schedule(
                betas, gammas, beta_schedule=str(cfg.get("lr_beta_schedule", "decreasing"))
            )
            results.append(
                {
                    "source": jp.name,
                    "depth": int(tr["depth"]),
                    "seed": int(cfg["seed"]),
                    "n_window": f"{cfg['n_min']}-{cfg['n_max']}",
                    "train_size": int(cfg["train_size"]),
                    "test_size": int(cfg["test_size"]),
                    "delta_gamma_fit": dg,
                    "delta_beta_fit": db,
                    "angle_residual_l2": abs_r,
                    "angle_residual_normalized": rel_r,
                    "full_gammas": gammas.tolist(),
                    "full_betas": betas.tolist(),
                    "proj_gammas": lg.tolist(),
                    "proj_betas": lb.tolist(),
                    "finite_size_benchmark": int(cfg["n_max"]) <= 13,
                }
            )
    return results


def audit_projection_performance(projections: List[dict]) -> List[dict]:
    perf = []
    qaoa_reg = _PHASECRAFT / "qaoa-reg"
    if str(qaoa_reg) not in sys.path:
        sys.path.insert(0, str(qaoa_reg))
    from run_benchmark import generate_benchmark_dataset  # noqa: E402

    for row in projections:
        jp = next(p for p in QAOA_REG_JSONS if p.name == row["source"])
        cfg = json.loads(jp.read_text())["config"]
        n_values = list(range(int(cfg["n_min"]), int(cfg["n_max"]) + 1))
        dataset = generate_benchmark_dataset(
            n_values,
            k=int(cfg["k"]),
            r=float(cfg["r"]),
            test_size=int(cfg["test_size"]),
            base_seed=int(cfg["eval_seed"]),
            require_sat=True,
            m_sampling="notebook",
            max_trials_per_n=1_000_000,
        )
        betas = np.asarray(row["full_betas"])
        gammas = np.asarray(row["full_gammas"])
        pb = np.asarray(row["proj_betas"])
        pg = np.asarray(row["proj_gammas"])
        full_ev = eval_angles_on_dataset(dataset, betas, gammas)
        proj_ev = eval_angles_on_dataset(dataset, pb, pg)
        for n in n_values:
            fp = full_ev[n]["mean_p_succ"]
            pp = proj_ev[n]["mean_p_succ"]
            fr = full_ev[n]["median_runtime"]
            pr = proj_ev[n]["median_runtime"]
            perf.append(
                {
                    "source": row["source"],
                    "depth": row["depth"],
                    "n": n,
                    "finite_size_benchmark": row["finite_size_benchmark"],
                    "mean_p_full": fp,
                    "mean_p_proj": pp,
                    "mean_p_loss": fp - pp,
                    "mean_p_loss_frac": (fp - pp) / fp if fp > 0 else float("nan"),
                    "median_rt_full": fr,
                    "median_rt_proj": pr,
                    "median_rt_loss_frac": (pr - fr) / fr if fr > 0 else float("nan"),
                }
            )
    return perf


def figure3_gap_decomposition() -> Tuple[List[dict], List[dict]]:
    """Per-seed gaps + seed-averaged with SEM."""
    per_seed = []
    n_lo, n_hi = EVAL_WINDOW
    ns = list(range(n_lo, n_hi + 1))

    for cache_path, seed in MULTI_SEED_CACHES:
        if not cache_path.is_file():
            continue
        reb = json.loads(cache_path.read_text())["rebenchmark"]
        for p in DEPTHS_FIG3:
            key = f"bm24_mean_p_fixed_n|{p}"
            if key not in reb:
                continue
            e = reb[key]
            c_rt = float(e["c_typ_fitted"])
            c_sp = float(e["c_inv_mean_rt"])
            c_bm24 = bm24_analytic_c(p)
            d_obj = c_rt - c_sp
            d_bench = c_sp - c_bm24
            per_seed.append(
                {
                    "seed": seed,
                    "depth": p,
                    "c_rt": c_rt,
                    "c_sp": c_sp,
                    "c_bm24": c_bm24,
                    "delta_objective": d_obj,
                    "delta_benchmark": d_bench,
                    "delta_total": d_obj + d_bench,
                    "source": cache_path.name,
                    "n_window": f"{n_lo}-{n_hi}",
                    "fit_model_rt": "log2 slope median(1/p) vs n",
                    "fit_model_sp": "log2 slope mean(1/p) vs n",
                    "c_bm24_model": "0.69*p^{-0.32} analytic (no offset)",
                }
            )

    agg = []
    for p in DEPTHS_FIG3:
        sub = [r for r in per_seed if r["depth"] == p]
        if not sub:
            continue
        for key in ("delta_objective", "delta_benchmark", "delta_total", "c_rt", "c_sp"):
            vals = [r[key] for r in sub]
            agg.append(
                {
                    "depth": p,
                    "metric": key,
                    "mean": float(np.mean(vals)),
                    "sem": float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0,
                    "n_seeds": len(vals),
                }
            )
    return per_seed, agg


def bootstrap_fig3_from_n500(n_boot: int = 1000) -> List[dict]:
    """Instance bootstrap for depths in N500 CSV (train_n=12, mean_p objective)."""
    import csv

    csv_path = _PHASECRAFT / "results/bm24_runs/bm24_gap_audit/N500_n12-20_all.csv"
    if not csv_path.is_file():
        return []
    buckets: Dict[Tuple[int, int], Dict[int, List[float]]] = {}
    with csv_path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row["objective"] != "mean_p" or int(row["train_n"]) != 12:
                continue
            p = int(row["p"])
            n = int(row["n"])
            if n < EVAL_WINDOW[0] or n > EVAL_WINDOW[1]:
                continue
            ps = float(row["p_succ"])
            if not (math.isfinite(ps) and ps > 0):
                continue
            buckets.setdefault((12, p), {}).setdefault(n, []).append(ps)

    rng = np.random.default_rng(0)
    out = []
    for (_, p), by_n in sorted(buckets.items()):
        ns = sorted(by_n)
        if len(ns) < 3:
            continue
        boots_obj = []
        boots_bench = []
        boots_tot = []
        for _ in range(n_boot):
            med_rt = []
            inv_m = []
            for n in ns:
                ps = np.asarray(by_n[n])
                idx = rng.integers(0, len(ps), len(ps))
                samp = ps[idx]
                med_rt.append(float(np.median(1.0 / np.maximum(samp, 1e-300))))
                inv_m.append(float(np.mean(1.0 / np.maximum(samp, 1e-300))))
            c_rt = fit_log2_slope(ns, med_rt)
            c_sp = fit_log2_slope(ns, inv_m)
            c_bm = bm24_analytic_c(p)
            boots_obj.append(c_rt - c_sp)
            boots_bench.append(c_sp - c_bm)
            boots_tot.append(c_rt - c_bm)
        out.append(
            {
                "depth": p,
                "delta_objective_ci": tuple(np.quantile(boots_obj, [0.025, 0.975]).tolist()),
                "delta_benchmark_ci": tuple(np.quantile(boots_bench, [0.025, 0.975]).tolist()),
                "delta_total_ci": tuple(np.quantile(boots_tot, [0.025, 0.975]).tolist()),
                "bootstrap_n": n_boot,
                "source": "N500_n12-20_all.csv",
                "n_window": f"{EVAL_WINDOW[0]}-{EVAL_WINDOW[1]}",
                "note": "frozen angles from angles_grid; not multi-seed Figure 3 angles",
            }
        )
    return out


def build_summary_table(
    provenance: List[ProvenanceRow],
    projections: List[dict],
    perf: List[dict],
    per_seed_gaps: List[dict],
    boot: List[dict],
) -> str:
    lines = [
        "# Expressivity evidence audit summary table",
        "",
        "## A. Key number provenance (selected)",
        "",
        "| Label | Value | Source | Field | Objective | Seeds | n-window | Fit model | Kind |",
        "|-------|------:|--------|-------|-----------|-------|----------|-----------|------|",
    ]
    for r in provenance:
        lines.append(
            f"| {r.label} | {r.value:.4f} | {r.source_file} | {r.field_name} | {r.objective} | {r.seeds} | {r.n_window} | {r.fit_model} | {r.value_kind} |"
        )

    lines += [
        "",
        "## B. Full-QAOA angle projection onto LR schedule (LS fit, decreasing beta)",
        "",
        "| Source | p | |g_perp|/|g| | Δγ_fit | Δβ_fit | Finite-n benchmark? |",
        "|--------|--:|--------:|-------:|-------:|:-------------------:|",
    ]
    for r in projections:
        lines.append(
            f"| {r['source']} | {r['depth']} | {r['angle_residual_normalized']:.4f} | {r['delta_gamma_fit']:.4f} | {r['delta_beta_fit']:.4f} | {r['finite_size_benchmark']} |"
        )

    lines += [
        "",
        "## C. Projection performance loss (full QAOA → nearest LR angles, same held-out set)",
        "",
        "| Source | p | n | Δ mean_p | frac loss mean_p | frac worse median_rt | Finite-n? |",
        "|--------|--:|--:|---------:|-----------------:|---------------------:|:---------:|",
    ]
    for r in perf:
        lines.append(
            f"| {r['source']} | {r['depth']} | {r['n']} | {r['mean_p_loss']:.6f} | {r['mean_p_loss_frac']:.4f} | {r['median_rt_loss_frac']:.4f} | {r['finite_size_benchmark']} |"
        )

    lines += [
        "",
        "## D. Figure 3 gap decomposition (per seed, mean_p-trained LR, n=12–18)",
        "",
        "| Seed | p | Δ_obj | Δ_bench | Δ_total | c_rt | c_sp | c_BM24 |",
        "|-----:|--:|------:|--------:|--------:|-----:|-----:|-------:|",
    ]
    for r in sorted(per_seed_gaps, key=lambda x: (x["depth"], x["seed"])):
        lines.append(
            f"| {r['seed']} | {r['depth']} | {r['delta_objective']:.4f} | {r['delta_benchmark']:.4f} | {r['delta_total']:.4f} | {r['c_rt']:.4f} | {r['c_sp']:.4f} | {r['c_bm24']:.4f} |"
        )

    if boot:
        lines += [
            "",
            "## E. Instance bootstrap CIs (N500 frozen angles, n=12–18)",
            "",
            "| p | Δ_obj CI | Δ_bench CI | Δ_total CI |",
            "|--:|:---------|:-----------|:-----------|",
        ]
        for r in boot:
            lines.append(
                f"| {r['depth']} | [{r['delta_objective_ci'][0]:.4f}, {r['delta_objective_ci'][1]:.4f}] | "
                f"[{r['delta_benchmark_ci'][0]:.4f}, {r['delta_benchmark_ci'][1]:.4f}] | "
                f"[{r['delta_total_ci'][0]:.4f}, {r['delta_total_ci'][1]:.4f}] |"
            )
    return "\n".join(lines) + "\n"


def plot_gap_decomposition(per_seed: List[dict], boot: List[dict], out_path: Path) -> None:
    depths = DEPTHS_FIG3
    metrics = ("delta_objective", "delta_benchmark", "delta_total")
    titles = (
        r"$\Delta_{\mathrm{objective}} = c_{\mathrm{rt}}^{\mathrm{LR}} - c_{\mathrm{sp}}^{\mathrm{LR}}$",
        r"$\Delta_{\mathrm{benchmark}} = c_{\mathrm{sp}}^{\mathrm{LR}} - c_{\mathrm{sp}}^{\mathrm{BM24}}$",
        r"$\Delta_{\mathrm{total}} = \Delta_{\mathrm{objective}} + \Delta_{\mathrm{benchmark}}$",
    )
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.8), sharex=True)
    boot_map = {r["depth"]: r for r in boot}
    for ax, metric, title in zip(axes, metrics, titles):
        means, sems = [], []
        for p in depths:
            vals = [r[metric] for r in per_seed if r["depth"] == p]
            means.append(float(np.mean(vals)) if vals else float("nan"))
            sems.append(
                float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
            )
        ax.errorbar(depths, means, yerr=sems, fmt="o-", color="#4C72B0", capsize=3, lw=1.8, ms=6)
        if boot and metric in ("delta_objective", "delta_benchmark", "delta_total"):
            ci_key = {
                "delta_objective": "delta_objective_ci",
                "delta_benchmark": "delta_benchmark_ci",
                "delta_total": "delta_total_ci",
            }[metric]
            for p in depths:
                if p not in boot_map:
                    continue
                lo, hi = boot_map[p][ci_key]
                ax.fill_between([p - 0.8, p + 0.8], lo, hi, color="#DD8452", alpha=0.2, linewidth=0)
        ax.axhline(0, color="0.5", lw=0.8)
        ax.set_title(title, fontsize=9.5)
        ax.set_xlabel(r"QAOA depth $p$")
        ax.grid(True, alpha=0.2)
    axes[0].set_ylabel(r"Exponent gap")
    fig.suptitle(
        "Figure 3 gap decomposition (seed mean ± SEM; orange band = N500 instance bootstrap 95% CI)",
        fontsize=10,
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_report(
    provenance: List[ProvenanceRow],
    projections: List[dict],
    perf: List[dict],
    per_seed_gaps: List[dict],
    boot: List[dict],
) -> str:
    p5 = next((r for r in projections if r["source"] == "cpu-prx-small.json" and r["depth"] == 5), None)
    perf_p5_n12 = [r for r in perf if r["source"] == "cpu-prx-small.json" and r["depth"] == 5 and r["n"] == 12]
    gap100 = [r for r in per_seed_gaps if r["depth"] == 100]

    lines = [
        "# LR-QAOA expressivity evidence audit (no new asymptotic claims)",
        "",
        "## 1. Structural restriction",
        "",
        "LR-QAOA uses 2 parameters `(Δγ, Δβ)` expanded via a linear ramp (`make_lr_angles`).",
        "Full regular QAOA uses `2p` independent layer angles.",
        "",
    ]
    if p5:
        lines += [
            f"- At `p=5` (`cpu-prx-small.json`), LS projection residual "
            f"`||θ_full − θ_LR|| / ||θ_full|| = {p5['angle_residual_normalized']:.3f}`.",
            "- Optimized full-QAOA β schedule is visibly non-linear; it cannot be represented exactly on the LR manifold.",
            "",
        ]

    lines += [
        "## 2. Projection loss (same held-out instances)",
        "",
        "Projecting full-QAOA angles onto the nearest LR schedule and re-evaluating:",
        "",
    ]
    if perf_p5_n12:
        r = perf_p5_n12[0]
        lines.append(
            f"- `p=5`, `n=12`, `N=20`: mean success drops by `{r['mean_p_loss_frac']*100:.1f}%`; "
            f"median runtime worsens by `{r['median_rt_loss_frac']*100:.1f}%`."
        )
    lines += [
        "- All qaoa-reg comparisons use `n ≤ 13` only → **finite-size matched benchmarks**, not scaling estimates.",
        "",
        "## 3. Objective gap (Figure 3, LR only)",
        "",
        "`Δ_objective = c_rt^LR − c_sp^LR` from `*-ctyp-cann-cache.json`, seeds {0,27,42}, n=12–18.",
        "",
    ]
    if gap100:
        m = np.mean([r["delta_objective"] for r in gap100])
        lines.append(f"- At `p=100`: mean Δ_objective ≈ `{m:.3f}` (median-runtime exponent exceeds inverse-mean exponent).")
    lines += [
        "",
        "## 4. Benchmark gap (not expressivity)",
        "",
        "`Δ_benchmark = c_sp^LR − 0.69 p^{-0.32}` compares LR to the BM24 analytic reference.",
        "This is **not** a controlled full-QAOA comparison.",
        "",
    ]
    if gap100:
        m = np.mean([r["delta_benchmark"] for r in gap100])
        lines.append(f"- At `p=100`: mean Δ_benchmark ≈ `{m:.3f}`.")
    lines += [
        "",
        "## 5. Optimizer / finite-size limitations",
        "",
        "- No production-scale full-QAOA depth sweep exists (`train_size=100`, `n=12–18`, depths 2–100).",
        "- `c_inf` values in `objective_exp_convergence.json` are **fitted offsets** (`c_inf + A p^{-β}`),",
        "  distinct from per-depth values on the thesis Figure 3.",
        "- N500 bootstrap CIs use **frozen** angles from `angles_grid_existing.json`, not the multi-seed training runs.",
        "",
        "## 6. What is *not* demonstrated",
        "",
        "- A production-scale expressivity gap vs optimized full QAOA.",
        "- Lift-and-release refinement from LR init.",
        "- Gradient orthogonality `R_⊥`.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    provenance = collect_provenance()
    projections = audit_qaoa_angle_projections()
    print("Evaluating projection performance on held-out instances...")
    perf = audit_projection_performance(projections)
    per_seed_gaps, agg_gaps = figure3_gap_decomposition()
    boot = bootstrap_fig3_from_n500()

    payload = {
        "provenance": [asdict(r) for r in provenance],
        "angle_projections": projections,
        "projection_performance": perf,
        "figure3_per_seed_gaps": per_seed_gaps,
        "figure3_aggregated": agg_gaps,
        "figure3_n500_bootstrap": boot,
    }
    json_path = OUT_DIR / "expressivity_audit.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {json_path}")

    table_md = build_summary_table(provenance, projections, perf, per_seed_gaps, boot)
    (OUT_DIR / "expressivity_audit_table.md").write_text(table_md, encoding="utf-8")
    (OUT_DIR / "expressivity_audit_report.md").write_text(
        write_report(provenance, projections, perf, per_seed_gaps, boot), encoding="utf-8"
    )
    plot_gap_decomposition(per_seed_gaps, boot, OUT_DIR / "figure3_gap_decomposition.png")
    print(f"Wrote outputs under {OUT_DIR}")


if __name__ == "__main__":
    main()
