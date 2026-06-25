#!/usr/bin/env python3
"""
Aggregate multi-seed mean_p vs median_rt A/B runs.

Reads manifest from run_multi_seed_objective_ab.py (plus optional seed-27 baseline).

Example::

    python experiments/lr_scaling/plot_multi_seed_objective_ab.py
    python experiments/lr_scaling/plot_multi_seed_objective_ab.py --with-ctyp-cann
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np

_PHASECRAFT = Path(__file__).resolve().parents[2]
_LR = Path(__file__).resolve().parent
for _p in (_PHASECRAFT.parent, _PHASECRAFT, _LR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
from phasecraft.lib.paths import bm24_runs_dir  # noqa: E402

_DEFAULT_MANIFEST = bm24_runs_dir() / "multi_seed_ctyp_cann" / "manifest.json"
_BASELINE = bm24_runs_dir() / "06-09/run1/train12-tr100-te200-n12-18.json"


def _traces(payload: dict) -> Dict[str, List[dict]]:
    if "traces_by_mode" in payload:
        return {k: list(v) for k, v in payload["traces_by_mode"].items()}
    out: Dict[str, List[dict]] = {}
    for key in ("trace_bm24_mean_p_fixed_n", "trace_median_runtime_fixed_n"):
        if key in payload and payload[key]:
            out[key.replace("trace_", "")] = list(payload[key])
    return out


def load_runs(manifest_path: Path, include_baseline: bool) -> List[dict]:
    runs: List[dict] = []
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for rec in manifest.get("runs", []):
            jp = rec.get("json")
            if jp and Path(jp).is_file() and rec.get("status") in ("ok", "baseline", None):
                runs.append({"seed": rec["seed"], "path": Path(jp)})
    if include_baseline and _BASELINE.is_file():
        if not any(r["seed"] == 27 for r in runs):
            runs.append({"seed": 27, "path": _BASELINE})
    runs.sort(key=lambda r: int(r["seed"]))
    return runs


def extract_deltas(run: dict) -> List[dict]:
    payload = json.loads(run["path"].read_text(encoding="utf-8"))
    traces = _traces(payload)
    mean_tr = {int(r["depth"]): r for r in traces.get("bm24_mean_p_fixed_n", [])}
    med_tr = {int(r["depth"]): r for r in traces.get("median_runtime_fixed_n", [])}
    rows: List[dict] = []
    for depth in sorted(set(mean_tr) & set(med_tr)):
        c_mean = float(mean_tr[depth]["lr_log2_slope"])
        c_med = float(med_tr[depth]["lr_log2_slope"])
        rows.append({
            "seed": int(run["seed"]),
            "depth": depth,
            "c_typ_mean_p": c_mean,
            "c_typ_median_rt": c_med,
            "delta_ctyp": c_med - c_mean,
        })
    return rows


def _filter_rows(all_rows: List[dict], depths: Optional[List[int]]) -> Tuple[List[dict], List[int]]:
    if depths:
        depth_set = set(depths)
        rows = [r for r in all_rows if int(r["depth"]) in depth_set]
        use_depths = [d for d in depths if any(int(r["depth"]) == d for r in rows)]
    else:
        rows = list(all_rows)
        use_depths = sorted({int(r["depth"]) for r in rows})
    return rows, use_depths


def extract_ctyp_by_seed(runs: List[dict], *, mode: str = "bm24_mean_p_fixed_n") -> List[dict]:
    """Per-seed eval c_typ for a single fixed training mode."""
    rows: List[dict] = []
    for run in runs:
        payload = json.loads(run["path"].read_text(encoding="utf-8"))
        traces = _traces(payload)
        for row in traces.get(mode, []):
            rows.append({
                "seed": int(run["seed"]),
                "depth": int(row["depth"]),
                "c_typ": float(row["lr_log2_slope"]),
            })
    return rows


def plot_seed_convergence(
    ctyp_rows: List[dict],
    out_dir: Path,
    *,
    focus_depths: Optional[List[int]] = None,
    training_label: str = "mean_p fixed n (standard)",
) -> Path:
    """Same training protocol across seeds: do eval c_typ curves agree / converge?"""
    out_dir.mkdir(parents=True, exist_ok=True)
    depths = focus_depths or sorted({int(r["depth"]) for r in ctyp_rows})
    seeds = sorted({int(r["seed"]) for r in ctyp_rows})

    cmap = plt.cm.tab10
    seed_colors = {seed: cmap(i % 10) for i, seed in enumerate(seeds)}

    by_seed: Dict[int, Dict[int, float]] = {s: {} for s in seeds}
    for r in ctyp_rows:
        if int(r["depth"]) in depths:
            by_seed[int(r["seed"])][int(r["depth"])] = float(r["c_typ"])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), gridspec_kw={"width_ratios": [2.2, 1]})

    # Left: c_typ vs depth, one curve per seed
    ax = axes[0]
    all_vals: List[float] = []
    curves: Dict[int, List[float]] = {}
    for seed in seeds:
        ys = [by_seed[seed].get(d, float("nan")) for d in depths]
        curves[seed] = ys
        for y in ys:
            if not np.isnan(y):
                all_vals.append(y)
        ax.plot(
            depths, ys, "o-", color=seed_colors[seed], lw=2.2, ms=8,
            label=f"seed {seed}", alpha=0.9, zorder=3,
        )

    seed_mean = []
    seed_lo = []
    seed_hi = []
    for d in depths:
        vals = [by_seed[s][d] for s in seeds if d in by_seed[s]]
        seed_mean.append(float(np.mean(vals)) if vals else float("nan"))
        seed_lo.append(float(np.min(vals)) if vals else float("nan"))
        seed_hi.append(float(np.max(vals)) if vals else float("nan"))

    ax.fill_between(depths, seed_lo, seed_hi, color="#888888", alpha=0.22, zorder=1, label="seed min–max")
    ax.plot(
        depths, seed_mean, "k--", lw=2.5, ms=0, zorder=4,
        label="mean across seeds",
    )

    y_min = float(np.nanmin(all_vals))
    y_max = float(np.nanmax(all_vals))
    ax.set_ylim(y_min - 0.02, y_max + 0.05)
    ax.set_xlabel("depth p", fontsize=11)
    ax.set_ylabel(r"$c_{\mathrm{typ}}$ (eval log₂ slope)", fontsize=11)
    ax.set_title(f"Same training: {training_label}", fontsize=11)
    ax.set_xticks(depths)
    ax.set_xticklabels([str(d) for d in depths])
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="upper right", framealpha=0.95)

    # Right: seed disagreement vs depth
    ax2 = axes[1]
    spreads = [hi - lo for lo, hi in zip(seed_lo, seed_hi)]
    ax2.plot(depths, spreads, "s-", color="#4C72B0", lw=2.2, ms=8)
    for d, spread, n in zip(depths, spreads, [len([s for s in seeds if d in by_seed[s]]) for d in depths]):
        ax2.annotate(f"n={n}", (d, spread), textcoords="offset points", xytext=(0, 6),
                     ha="center", fontsize=8, color="#333333")
    ax2.set_xlabel("depth p", fontsize=11)
    ax2.set_ylabel(r"seed spread (max $-$ min)", fontsize=11)
    ax2.set_title("Do seeds agree?", fontsize=11)
    ax2.set_xticks(depths)
    ax2.set_xticklabels([str(d) for d in depths])
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, max(spreads) * 1.25 if spreads else 0.05)

    fig.suptitle(
        f"Multi-seed convergence ({len(seeds)} seeds, identical protocol)",
        fontsize=12, y=1.02,
    )
    fig.tight_layout()

    png = out_dir / "multi_seed_convergence.png"
    fig.savefig(png, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {png}")
    return png


def plot_seed_overlay(
    all_rows: List[dict],
    out_dir: Path,
    *,
    focus_depths: Optional[List[int]] = None,
) -> Path:
    """Single-panel overlay: all seeds × both training objectives on one axis."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, depths = _filter_rows(all_rows, focus_depths)
    seeds = sorted({int(r["seed"]) for r in rows})

    cmap = plt.cm.tab10
    seed_colors = {seed: cmap(i % 10) for i, seed in enumerate(seeds)}
    mean_dashes = (8, 4)

    fig, ax = plt.subplots(figsize=(8.5, 5.2))

    mean_by_depth: Dict[int, List[float]] = {d: [] for d in depths}
    all_vals: List[float] = []

    for seed in seeds:
        sub = [r for r in rows if int(r["seed"]) == seed]
        mean_map = {int(r["depth"]): r["c_typ_mean_p"] for r in sub}
        med_map = {int(r["depth"]): r["c_typ_median_rt"] for r in sub}
        color = seed_colors[seed]
        mean_ys = [mean_map.get(d, float("nan")) for d in depths]
        med_ys = [med_map.get(d, float("nan")) for d in depths]
        for d, ym, ymed in zip(depths, mean_ys, med_ys):
            if not np.isnan(ym):
                mean_by_depth[d].append(ym)
                all_vals.append(ym)
            if not np.isnan(ymed):
                all_vals.append(ymed)
        ax.plot(
            depths,
            mean_ys,
            linestyle="--",
            dashes=mean_dashes,
            color=color,
            lw=2.0,
            ms=8,
            marker="o",
            markerfacecolor="white",
            markeredgecolor=color,
            markeredgewidth=1.8,
            alpha=0.9,
            zorder=2,
        )
        ax.plot(
            depths,
            med_ys,
            linestyle="-",
            color=color,
            lw=2.4,
            ms=8,
            marker="o",
            markerfacecolor=color,
            markeredgecolor=color,
            markeredgewidth=1.2,
            alpha=0.95,
            zorder=3,
        )

    mean_lo = [min(mean_by_depth[d]) if mean_by_depth[d] else float("nan") for d in depths]
    mean_hi = [max(mean_by_depth[d]) if mean_by_depth[d] else float("nan") for d in depths]
    ax.fill_between(depths, mean_lo, mean_hi, color="#888888", alpha=0.15, zorder=0)

    y_min = float(np.nanmin(all_vals))
    y_max = float(np.nanmax(all_vals))
    y_pad_lo = 0.02
    y_pad_hi = 0.05
    ax.set_ylim(y_min - y_pad_lo, y_max + y_pad_hi)

    ax.set_xlabel("depth p", fontsize=11)
    ax.set_ylabel(r"$c_{\mathrm{typ}}$ (eval log₂ slope)", fontsize=11)
    ax.set_title(f"Multi-seed overlay ({len(seeds)} seeds)", fontsize=11)
    ax.set_xticks(depths)
    ax.set_xticklabels([str(d) for d in depths])
    ax.grid(True, alpha=0.3)

    style_handles = [
        Line2D(
            [0], [0],
            color="#333333",
            linestyle="--",
            dashes=mean_dashes,
            lw=2.5,
            marker="o",
            markerfacecolor="white",
            markeredgecolor="#333333",
            markeredgewidth=1.8,
            ms=8,
            label="mean_p training (dashed, open markers)",
        ),
        Line2D(
            [0], [0],
            color="#333333",
            linestyle="-",
            lw=2.5,
            marker="o",
            markerfacecolor="#333333",
            markeredgecolor="#333333",
            ms=8,
            label="median_rt training (solid, filled markers)",
        ),
    ]
    seed_handles = [
        Line2D(
            [0], [0],
            color=seed_colors[seed],
            linestyle="-",
            lw=2.4,
            marker="o",
            markerfacecolor=seed_colors[seed],
            ms=8,
            label=f"seed {seed}",
        )
        for seed in seeds
    ]
    leg_style = ax.legend(
        handles=style_handles,
        fontsize=9,
        loc="upper right",
        framealpha=0.95,
        title="Line style = training objective",
        title_fontsize=9,
    )
    ax.add_artist(leg_style)
    ax.legend(
        handles=seed_handles,
        fontsize=9,
        loc="upper center",
        framealpha=0.95,
        title="Colour = random seed",
        title_fontsize=9,
        ncol=len(seeds),
    )
    fig.tight_layout()

    png = out_dir / "multi_seed_ctyp_overlay.png"
    fig.savefig(png, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {png}")
    return png


def plot_summary(
    all_rows: List[dict],
    out_dir: Path,
    *,
    focus_depths: Optional[List[int]] = None,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, depths = _filter_rows(all_rows, focus_depths)
    seeds = sorted({int(r["seed"]) for r in rows})

    cmap = plt.cm.tab10
    seed_colors = {seed: cmap(i % 10) for i, seed in enumerate(seeds)}

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # Left: c_typ eval slopes — both training objectives
    ax = axes[0]
    for seed in seeds:
        sub = [r for r in rows if int(r["seed"]) == seed]
        mean_map = {int(r["depth"]): r["c_typ_mean_p"] for r in sub}
        med_map = {int(r["depth"]): r["c_typ_median_rt"] for r in sub}
        color = seed_colors[seed]
        xs = depths
        ax.plot(
            xs,
            [mean_map.get(d, float("nan")) for d in xs],
            "o--",
            color=color,
            alpha=0.55,
            ms=5,
            label=f"seed={seed} mean_p",
        )
        ax.plot(
            xs,
            [med_map.get(d, float("nan")) for d in xs],
            "o-",
            color=color,
            alpha=0.95,
            ms=5,
            label=f"seed={seed} median_rt",
        )
    ax.set_xlabel("depth p")
    ax.set_ylabel("c_typ (eval log₂ slope)")
    ax.set_title("Eval scaling by training objective")
    ax.set_xticks(depths)
    ax.set_xticklabels([str(d) for d in depths])
    ax.legend(fontsize=6.5, ncol=2, loc="upper right")
    ax.grid(True, alpha=0.3)

    # Middle: Δ c_typ per seed
    ax = axes[1]
    for seed in seeds:
        sub = [r for r in rows if int(r["seed"]) == seed]
        dmap = {int(r["depth"]): r["delta_ctyp"] for r in sub}
        ys = [dmap.get(d, float("nan")) for d in depths]
        ax.plot(depths, ys, "o-", label=f"seed={seed}", color=seed_colors[seed], alpha=0.9)
    ax.axhline(0, color="k", lw=0.8, alpha=0.35)
    ax.set_xlabel("depth p")
    ax.set_ylabel("Δ c_typ (median_rt − mean_p)")
    ax.set_title("Training objective effect")
    ax.set_xticks(depths)
    ax.set_xticklabels([str(d) for d in depths])
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Right: mean |Δ| per depth with per-seed scatter
    ax = axes[2]
    mean_abs = []
    std_delta = []
    n_at_depth = []
    for d in depths:
        deltas = [r["delta_ctyp"] for r in rows if int(r["depth"]) == d]
        mean_abs.append(float(np.mean(np.abs(deltas))))
        std_delta.append(float(np.std(deltas)) if len(deltas) > 1 else 0.0)
        n_at_depth.append(len(deltas))
        jitter = np.linspace(-0.12, 0.12, max(len(deltas), 1))
        for j, delta in enumerate(deltas):
            ax.scatter(d + jitter[j], abs(delta), s=28, color="#555555", alpha=0.65, zorder=3)
    x = np.arange(len(depths))
    bars = ax.bar(x, mean_abs, yerr=std_delta, capsize=4, color="#4C72B0", alpha=0.75, zorder=2)
    for i, (bar, n) in enumerate(zip(bars, n_at_depth)):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + std_delta[i] + 0.001,
            f"n={n}",
            ha="center",
            va="bottom",
            fontsize=7,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([str(d) for d in depths])
    ax.set_xlabel("depth p")
    ax.set_ylabel("mean |Δ c_typ| across seeds")
    ax.set_title(f"{len(seeds)} seeds")
    ax.grid(True, alpha=0.3, axis="y")

    depth_label = ",".join(str(d) for d in depths)
    fig.suptitle(
        f"Multi-seed robustness (p ∈ {{{depth_label}}}): median vs mean training",
        fontsize=11,
    )
    fig.tight_layout()
    png = out_dir / "multi_seed_delta_ctyp.png"
    fig.savefig(png, dpi=150, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "n_seeds": len(seeds),
        "seeds": seeds,
        "depths": depths,
        "rows": rows,
        "per_depth": {
            str(d): {
                "n_seeds": len([r for r in rows if int(r["depth"]) == d]),
                "mean_delta": float(np.mean([r["delta_ctyp"] for r in rows if int(r["depth"]) == d])),
                "mean_abs_delta": float(np.mean(np.abs([r["delta_ctyp"] for r in rows if int(r["depth"]) == d]))),
                "std_delta": float(np.std([r["delta_ctyp"] for r in rows if int(r["depth"]) == d])),
            }
            for d in depths
        },
    }
    json_path = out_dir / "multi_seed_summary.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {png}")
    print(f"Wrote {json_path}")
    return png


def run_ctyp_cann(json_path: Path) -> None:
    script = _LR / "plot_ctyp_vs_cann.py"
    out_name = json_path.stem
    out_dir = _PHASECRAFT / "bm24_runs/analysis/ctyp_vs_cann" / out_name
    cmd = [
        sys.executable, str(script), str(json_path),
        "--theory-max-depth", "10",
        "--no-rebenchmark",
        "--output-dir", str(out_dir),
    ]
    print(f"  ctyp_vs_cann: {json_path.name}", flush=True)
    subprocess.run(cmd, cwd=str(_PHASECRAFT), check=False)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, default=_DEFAULT_MANIFEST)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=_PHASECRAFT / "bm24_runs/analysis/multi_seed_ctyp_cann",
    )
    p.add_argument("--include-baseline-seed27", action="store_true", default=True)
    p.add_argument(
        "--depths",
        default="10,20,40,50,60",
        help="Comma-separated depths to plot (default: 10,20,40,50,60).",
    )
    p.add_argument(
        "--with-ctyp-cann",
        action="store_true",
        help="Also run plot_ctyp_vs_cann per JSON (theory p≤10, no rebenchmark).",
    )
    args = p.parse_args()

    runs = load_runs(args.manifest.resolve(), include_baseline=bool(args.include_baseline_seed27))
    if not runs:
        print(f"No runs found (manifest: {args.manifest})", file=sys.stderr)
        sys.exit(1)

    all_rows: List[dict] = []
    for run in runs:
        all_rows.extend(extract_deltas(run))
        if args.with_ctyp_cann:
            run_ctyp_cann(run["path"])

    focus_depths = [int(x) for x in args.depths.split(",") if x.strip()]
    out_dir = args.output_dir.resolve()
    ctyp_rows = extract_ctyp_by_seed(runs)
    plot_seed_convergence(ctyp_rows, out_dir, focus_depths=focus_depths)
    plot_seed_overlay(all_rows, out_dir, focus_depths=focus_depths)
    plot_summary(all_rows, out_dir, focus_depths=focus_depths)
    mirror = _PHASECRAFT / "results/bm24_runs/analysis/multi_seed_ctyp_cann"
    if mirror.resolve() != out_dir.resolve():
        plot_seed_convergence(ctyp_rows, mirror, focus_depths=focus_depths)
        plot_seed_overlay(all_rows, mirror, focus_depths=focus_depths)
        plot_summary(all_rows, mirror, focus_depths=focus_depths)

    _, depths = _filter_rows(all_rows, focus_depths)
    print("\nPer-depth seed spread (max−min, mean_p training):")
    by_seed = extract_ctyp_by_seed(runs)
    for d in depths:
        vals = [r["c_typ"] for r in by_seed if int(r["depth"]) == d]
        if vals:
            print(f"  p={d:2d}: spread={max(vals)-min(vals):.4f}  seeds={len(vals)}")
    print("\nPer-depth |Δ c_typ| (mean_p vs median_rt, different training):")
    for d in depths:
        deltas = [abs(r["delta_ctyp"]) for r in all_rows if int(r["depth"]) == d]
        print(f"  p={d:2d}: {np.mean(deltas):.4f}  (seeds={len(deltas)})")


if __name__ == "__main__":
    main()
