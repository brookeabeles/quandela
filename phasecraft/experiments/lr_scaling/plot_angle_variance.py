#!/usr/bin/env python3
"""Plot variance of optimized LR angles (dg, db) across saved bm24 runs."""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_REPO = Path(__file__).resolve().parents[2]
_PC = _REPO / "phasecraft"
for _p in (_REPO, _PC):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from lib.paths import bm24_runs_dir


def _load_rows(root: Path) -> list[dict]:
    rows: list[dict] = []
    for p in sorted(root.rglob("*.json")):
        if "partial" in p.name or "bench" in p.name or "obj-compare" in str(p):
            continue
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(d, dict):
            continue
        cfg = d.get("config", {}) or {}
        stem = str(d.get("run_stem") or d.get("stem") or p.stem)
        traces: list[tuple[dict, str]] = []
        if d.get("trace"):
            mode = str(cfg.get("training_mode") or "?")
            traces = [(r, str(r.get("training_mode") or mode)) for r in d["trace"] if "delta_gamma" in r]
        else:
            if d.get("traces_by_mode"):
                for mode, tr in d["traces_by_mode"].items():
                    for r in tr:
                        traces.append((r, str(r.get("training_mode", mode))))
            else:
                for key in (
                    "trace_bm24_mean_p_fixed_n",
                    "trace_median_runtime_fixed_n",
                    "trace_mean_log_runtime_fixed_n",
                ):
                    if d.get(key):
                        for r in d[key]:
                            traces.append((r, str(r.get("training_mode", key))))
        for r, mode in traces:
            dg, db = float(r["delta_gamma"]), float(r["delta_beta"])
            rows.append(
                {
                    "stem": stem,
                    "mode": mode,
                    "depth": int(r["depth"]),
                    "dg": dg,
                    "db": db,
                    "train_n": cfg.get("train_n"),
                    "n_max": cfg.get("n_max"),
                }
            )
    return rows


def _stats_by_depth(rows: list[dict], *, depths: list[int]) -> dict:
    out = {}
    for d in depths:
        pts = [r for r in rows if r["depth"] == d]
        if len(pts) < 2:
            continue
        dgs = np.array([r["dg"] for r in pts], dtype=float)
        dbs = np.array([r["db"] for r in pts], dtype=float)
        out[d] = {
            "n": len(pts),
            "dg_mean": float(dgs.mean()),
            "dg_std": float(dgs.std(ddof=1)) if len(pts) > 1 else 0.0,
            "db_mean": float(dbs.mean()),
            "db_std": float(dbs.std(ddof=1)) if len(pts) > 1 else 0.0,
            "pts": pts,
        }
    return out


def main() -> Path:
    root = bm24_runs_dir()
    rows = _load_rows(root)

    # Primary cohort: BM24 mean-p @ train_n=12, negative dg (physical basin)
    cohort = [
        r
        for r in rows
        if r["train_n"] == 12
        and r["dg"] < 0
        and r["mode"] in ("bm24_mean_p_fixed_n", "?", "median_runtime_fixed_n")
    ]
    mean_p = [r for r in cohort if r["mode"] == "bm24_mean_p_fixed_n"]
    median_p = [r for r in cohort if r["mode"] == "median_runtime_fixed_n"]

    depths = sorted({r["depth"] for r in cohort})
    stats_mp = _stats_by_depth(mean_p, depths=depths)
    stats_med = _stats_by_depth(median_p, depths=depths)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5), constrained_layout=True)
    fig.suptitle(
        "LR angle variance across saved bm24 runs\n"
        "train_n=12, k=8, r=176.54 · bands = mean ± 1σ · dots = individual runs",
        fontsize=12,
        y=1.02,
    )

    # --- Panel A: delta_gamma vs depth ---
    ax = axes[0, 0]
    for label, stats, color, marker in (
        ("mean_p fixed_n", stats_mp, "#2563eb", "o"),
        ("median runtime", stats_med, "#dc2626", "s"),
    ):
        if not stats:
            continue
        ds = sorted(stats)
        m = [stats[d]["dg_mean"] for d in ds]
        s = [stats[d]["dg_std"] for d in ds]
        ax.errorbar(ds, m, yerr=s, fmt=f"{marker}-", capsize=3, label=label, color=color, lw=1.8, ms=5)
        for d in ds:
            ys = [r["dg"] for r in stats[d]["pts"]]
            xs = np.full(len(ys), d) + (np.random.default_rng(d).uniform(-0.15, 0.15, len(ys)))
            ax.scatter(xs, ys, s=18, alpha=0.45, color=color, edgecolors="none", zorder=3)
    ax.set_xlabel("QAOA depth p")
    ax.set_ylabel("delta_gamma (dg)")
    ax.set_title("delta_gamma vs depth")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)

    # --- Panel B: delta_beta vs depth ---
    ax = axes[0, 1]
    for label, stats, color, marker in (
        ("mean_p fixed_n", stats_mp, "#2563eb", "o"),
        ("median runtime", stats_med, "#dc2626", "s"),
    ):
        if not stats:
            continue
        ds = sorted(stats)
        m = [stats[d]["db_mean"] for d in ds]
        s = [stats[d]["db_std"] for d in ds]
        ax.errorbar(ds, m, yerr=s, fmt=f"{marker}-", capsize=3, label=label, color=color, lw=1.8, ms=5)
        for d in ds:
            ys = [r["db"] for r in stats[d]["pts"]]
            xs = np.full(len(ys), d) + (np.random.default_rng(d + 100).uniform(-0.15, 0.15, len(ys)))
            ax.scatter(xs, ys, s=18, alpha=0.45, color=color, edgecolors="none", zorder=3)
    ax.set_xlabel("QAOA depth p")
    ax.set_ylabel("delta_beta (db)")
    ax.set_title("delta_beta vs depth")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)

    # --- Panel C: dg vs db scatter, colored by depth ---
    ax = axes[1, 0]
    cmap = plt.cm.viridis
    dmin, dmax = min(depths), max(depths)
    for r in mean_p:
        t = (r["depth"] - dmin) / max(dmax - dmin, 1)
        ax.scatter(r["dg"], r["db"], c=[cmap(t)], s=28, alpha=0.75, edgecolors="k", linewidths=0.2)
    # high-depth cluster ellipse (p>=30)
    hi = [r for r in mean_p if r["depth"] >= 30]
    if len(hi) >= 2:
        dgs = np.array([r["dg"] for r in hi])
        dbs = np.array([r["db"] for r in hi])
        cx, cy = dgs.mean(), dbs.mean()
        sx, sy = dgs.std(ddof=1), dbs.std(ddof=1)
        th = np.linspace(0, 2 * np.pi, 100)
        ax.plot(cx + sx * np.cos(th), cy + sy * np.sin(th), "k--", lw=1, alpha=0.6)
        ax.annotate(
            f"p≥30 mean_p\nσ_dg={sx:.3f}, σ_db={sy:.3f}",
            (cx, cy),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=8,
        )
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(dmin, dmax))
    fig.colorbar(sm, ax=ax, label="depth p")
    ax.set_xlabel("delta_gamma (dg)")
    ax.set_ylabel("delta_beta (db)")
    ax.set_title("Angle cloud (mean_p, train_n=12)")
    ax.grid(True, alpha=0.25)

    # --- Panel D: coefficient of variation vs depth ---
    ax = axes[1, 1]
    for label, stats, color in (
        ("|dg| CV (mean_p)", stats_mp, "#2563eb"),
        ("db CV (mean_p)", stats_mp, "#93c5fd"),
        ("|dg| CV (median)", stats_med, "#dc2626"),
    ):
        if not stats:
            continue
        ds = sorted(stats)
        if "median" in label:
            cv = [stats[d]["dg_std"] / max(abs(stats[d]["dg_mean"]), 1e-9) for d in ds]
            ls = "--"
        elif "db CV" in label:
            cv = [stats[d]["db_std"] / max(abs(stats[d]["db_mean"]), 1e-9) for d in ds]
            ls = ":"
        else:
            cv = [stats[d]["dg_std"] / max(abs(stats[d]["dg_mean"]), 1e-9) for d in ds]
            ls = "-"
        ax.plot(ds, cv, ls, marker="o", ms=4, label=label, color=color, lw=1.5)
    ax.set_xlabel("QAOA depth p")
    ax.set_ylabel("coefficient of variation (σ / |mean|)")
    ax.set_title("Relative spread shrinks at high depth")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)

    out_dir = root / "06-09" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "lr-angle-variance-across-runs.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")
    print(f"Cohort: {len(mean_p)} mean_p points, {len(median_p)} median_p points, {len(set(r['stem'] for r in cohort))} runs")
    if stats_mp:
        d50 = stats_mp.get(50) or stats_mp.get(max(stats_mp))
        if d50:
            print(
                f"At p={max(stats_mp)} mean_p: dg={d50['dg_mean']:.4f}±{d50['dg_std']:.4f}, "
                f"db={d50['db_mean']:.4f}±{d50['db_std']:.4f} (n={d50['n']} runs)"
            )
    return out


if __name__ == "__main__":
    main()
