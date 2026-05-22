#!/usr/bin/env python3
"""
Sweep LR-QAOA depth p = depth_min .. depth_max (train + benchmark each p).

Stop when LR beats **both** WalkSAT and WalkSATlm.  Default stop rule matches
the notebook / BM24 figure style (**scaling exponent**):

    Fit  ln(median cost) ≈ slope · n + const  over n in [n_min, n_max]
    LR cost      = median(1/p_succ)
    classical    = median(flips)
    LR wins when log2 slope(LR) < log2 slope(classical)  (lower = better)

Use ``--stop-criterion per-n`` for the older pointwise rule
``median(1/p)×equiv < median flips`` at each n (``--win-mode all-n`` / ``any-n``).

Example (notebook-like test window)::

    cd /Users/b/Quandela
    caffeinate -dimsu python phasecraft/sweep_lr_depth_until_win.py \\
      --depth-min 2 --depth-max 30 \\
      --n-min 18 --n-max 21 --k 8 --r 176.54 \\
      --test-size 100 --seed 27 \\
      --stop-criterion scaling-exponent
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from bm24_qaoa_sim import (  # noqa: E402
    check_benchmark_win,
    plot_benchmark_comparison,
    run_algorithm_benchmark,
    summarize_benchmark_scaling,
)
from bm24_run_io import format_benchmark_title, make_run_stem  # noqa: E402
from train_lr_notebook_protocol import (  # noqa: E402
    generate_training_h_diagonals,
    train_lr_grid_search_bm24,
)


def _parse_angle_log(path: Path) -> Dict[int, Tuple[float, float]]:
    """Return latest (delta_gamma, delta_beta) per depth from training log."""
    if not path.is_file():
        return {}
    text = path.read_text(encoding="utf-8")
    blocks = text.split("-" * 78)
    out: Dict[int, Tuple[float, float]] = {}
    for block in blocks:
        m_depth = re.search(r'"depth":\s*(\d+)', block)
        m_dg = re.search(r"delta_gamma.*:\s*([+-]?\d+\.?\d*(?:e[+-]?\d+)?)", block)
        m_db = re.search(r"delta_beta.*:\s*([+-]?\d+\.?\d*(?:e[+-]?\d+)?)", block)
        if m_depth and m_dg and m_db:
            p = int(m_depth.group(1))
            out[p] = (float(m_dg.group(1)), float(m_db.group(1)))
    return out


def _print_scaling(sc: dict) -> None:
    lr = sc["lr_qaoa"]["median_runtime_slope_log2"]
    ws = sc["walksat"]["median_flips_slope_log2"]
    lm = sc["walksatlm"]["median_flips_slope_log2"]
    print(f"  scaling log2 slopes: LR={lr:.4f}  WalkSAT={ws:.4f}  WalkSATlm={lm:.4f}")
    if sc["fit_ok"]:
        print(f"  LR better scaling than WalkSAT:   {sc['lr_beats_walksat_on_scaling']}")
        print(f"  LR better scaling than WalkSATlm: {sc['lr_beats_walksatlm_on_scaling']}")
    else:
        print("  scaling fit: need >= 2 distinct n in [n_min, n_max]")


def plot_scaling_vs_depth(
    trace: list[dict],
    output_path: Path,
    *,
    settings: dict | None = None,
    k: int | None = None,
    r: float | None = None,
    n_min: int | None = None,
    n_max: int | None = None,
    test_size: int | None = None,
    seed: int | None = None,
    depth_min: int | None = None,
    depth_max: int | None = None,
    train_n: int | None = None,
    train_size: int | None = None,
) -> Path:
    """Plot log2 scaling slopes vs QAOA depth p from a scaling-vs-depth JSON trace."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError(
            "plot_scaling_vs_depth requires matplotlib. Install with: pip install matplotlib"
        ) from e

    if not trace:
        raise ValueError("empty scaling trace")

    depths = [int(row["depth"]) for row in trace]
    lr = [float(row["lr_log2_slope"]) for row in trace]
    ws = [float(row["walksat_log2_slope"]) for row in trace]
    lm = [float(row["walksatlm_log2_slope"]) for row in trace]
    win_depths = [int(row["depth"]) for row in trace if row.get("lr_beats_both_scaling")]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(depths, lr, "o-", color="C0", linewidth=2, markersize=7, label="LR-QAOA (median 1/p)")
    ax.axhline(ws[0], color="C1", linestyle="--", linewidth=1.5, label=f"WalkSAT ({ws[0]:.3f})")
    ax.axhline(lm[0], color="C2", linestyle="--", linewidth=1.5, label=f"WalkSATlm ({lm[0]:.3f})")

    first_win = min(win_depths) if win_depths else None
    if first_win is not None:
        ax.axvline(first_win, color="0.4", linestyle=":", linewidth=1.2, alpha=0.8)
        idx = depths.index(first_win)
        ax.scatter([first_win], [lr[idx]], s=120, facecolors="none", edgecolors="C0", linewidths=2, zorder=5)
        ax.annotate(
            f"first win p={first_win}",
            xy=(first_win, lr[idx]),
            xytext=(8, 12),
            textcoords="offset points",
            fontsize=9,
        )

    ax.set_xlabel("QAOA depth p")
    ax.set_ylabel(r"$\log_2$ slope of median cost vs $n$")
    title_settings = dict(settings or {})
    for key, val in (
        ("k", k), ("r", r), ("n_min", n_min), ("n_max", n_max),
        ("test_size", test_size), ("seed", seed),
        ("depth_min", depth_min), ("depth_max", depth_max),
        ("train_n", train_n), ("train_size", train_size),
    ):
        if val is not None:
            title_settings[key] = val
    ax.set_title(
        format_benchmark_title(
            title_settings,
            headline="LR depth sweep",
            depths=depths,
        ),
        fontsize=10,
    )
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(depths)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def _load_settings_for_title(settings_json: Path | None) -> dict:
    if not settings_json or not settings_json.is_file():
        return {}
    with open(settings_json, encoding="utf-8") as f:
        payload = json.load(f)
    return payload.get("settings", payload)


def main() -> None:
    p = argparse.ArgumentParser(description="Sweep LR depth until LR beats WalkSAT + WalkSATlm.")
    p.add_argument("--depth-min", type=int, default=2)
    p.add_argument("--depth-max", type=int, default=30)
    p.add_argument("--n-min", type=int, default=18,
                   help="Benchmark n window (notebook uses 18–21).")
    p.add_argument("--n-max", type=int, default=21)
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--r", type=float, default=176.54)
    p.add_argument("--test-size", type=int, default=100)
    p.add_argument("--seed", type=int, default=27,
                   help="Benchmark seed (notebook default SEED=27).")
    p.add_argument("--train-n", type=int, default=12)
    p.add_argument("--train-size", type=int, default=50)
    p.add_argument("--skip-grid", action="store_true")
    p.add_argument("--skip-train", action="store_true",
                   help="Only use angles from --angle-log (skip depths missing there).")
    p.add_argument("--angle-log", type=str,
                   default=str(_SCRIPT_DIR / "bm24_runs" / "lr_train_optimal_angles.txt"))
    p.add_argument("--lr-beta-schedule", default="decreasing", choices=["decreasing", "increasing"])
    p.add_argument("--plot-equiv-flips-per-shot", type=float, default=1.0)
    p.add_argument(
        "--stop-criterion",
        choices=["scaling-exponent", "per-n", "both"],
        default="scaling-exponent",
        help="scaling-exponent: notebook/BM24 log2 slope crossover (default); "
        "per-n: median(1/p)×equiv vs flips; both: require both.",
    )
    p.add_argument("--win-mode", choices=["all-n", "any-n"], default="all-n",
                   help="For per-n stop only: all-n or any-n.")
    p.add_argument("--max-flips", type=int, default=100000)
    p.add_argument("--output-dir", type=str, default=str(_SCRIPT_DIR / "bm24_runs"))
    p.add_argument("--no-plot", action="store_true", help="Skip final scaling-vs-depth plot.")
    p.add_argument(
        "--save-per-depth",
        action="store_true",
        help="Write {run_stem}-p{N}.json per depth (off by default; see depth_sweep_until_win.jsonl).",
    )
    p.add_argument(
        "--per-depth-plots",
        action="store_true",
        help="Per-depth benchmark comparison PNGs (requires --save-per-depth or still runs benchmark).",
    )
    p.add_argument("--cobyla-maxiter", type=int, default=200)
    p.add_argument(
        "--plot-scaling-json",
        type=str,
        default="",
        help="Only plot scaling-vs-depth JSON (no sweep). Output: same stem + -plot.png unless --plot-scaling-out.",
    )
    p.add_argument("--plot-scaling-out", type=str, default="")
    p.add_argument(
        "--plot-scaling-settings",
        type=str,
        default="",
        help="Optional benchmark JSON for title (k, r, n_min, n_max).",
    )
    args = p.parse_args()

    if args.plot_scaling_json:
        trace_path = Path(args.plot_scaling_json).expanduser().resolve()
        with open(trace_path, encoding="utf-8") as f:
            payload = json.load(f)
        extra = _load_settings_for_title(
            Path(args.plot_scaling_settings).expanduser().resolve()
            if args.plot_scaling_settings
            else None
        )
        if isinstance(payload, dict) and "trace" in payload:
            trace = payload["trace"]
            plot_settings = {**extra, **payload.get("settings", {})}
        else:
            trace = payload
            plot_settings = extra
        if args.plot_scaling_out:
            out = Path(args.plot_scaling_out).expanduser().resolve()
        elif trace_path.suffix.lower() == ".json":
            out = trace_path.with_suffix(".png")
        else:
            out = trace_path.with_name(trace_path.stem + ".png")
        plot_scaling_vs_depth(trace, out, settings=plot_settings)
        print(f"Wrote {out}")
        return

    if args.depth_max < args.depth_min:
        raise SystemExit("depth-max must be >= depth-min")
    if args.n_max < args.n_min:
        raise SystemExit("n-max must be >= n-min")
    if args.stop_criterion in ("scaling-exponent", "both") and args.n_max == args.n_min:
        raise SystemExit(
            "scaling-exponent needs at least two n values (set n_max > n_min)."
        )
    if args.depth_min == 1 and args.lr_beta_schedule == "decreasing":
        print(
            "Note: depth=1 with decreasing schedule gives beta=0 (no mixer). "
            "Consider --lr-beta-schedule increasing for p=1, or start --depth-min 2."
        )

    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    cached = _parse_angle_log(Path(args.angle_log))
    sweep_log = out_dir / "depth_sweep_until_win.jsonl"
    run_stem = make_run_stem("sweep-scaling")
    sweep_t0 = time.time()

    print(
        f"Depth sweep p in [{args.depth_min}, {args.depth_max}], "
        f"n in [{args.n_min}, {args.n_max}], stop={args.stop_criterion}, "
        f"equiv={args.plot_equiv_flips_per_shot}"
    )
    if args.stop_criterion in ("per-n", "both"):
        print(f"  per-n win_mode={args.win_mode}")
    print(f"Run id: {run_stem}")
    print(f"Logging to {sweep_log}")

    exponent_trace: List[dict] = []
    stop_depth: int | None = None

    def _write_final_outputs(stopped_early: bool) -> tuple[Path, Path | None]:
        payload = {
            "run_stem": run_stem,
            "settings": {
                "depth_min": args.depth_min,
                "depth_max": args.depth_max,
                "n_min": args.n_min,
                "n_max": args.n_max,
                "k": args.k,
                "r": args.r,
                "test_size": args.test_size,
                "seed": args.seed,
                "train_n": args.train_n,
                "train_size": args.train_size,
                "stop_criterion": args.stop_criterion,
                "win_mode": args.win_mode,
                "plot_equiv_flips_per_shot": args.plot_equiv_flips_per_shot,
            },
            "trace": exponent_trace,
            "stopped_early": stopped_early,
            "stop_depth": stop_depth,
            "elapsed_s": time.time() - sweep_t0,
        }
        trace_path = out_dir / f"{run_stem}.json"
        with open(trace_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        summary_plot = None
        if not args.no_plot:
            summary_plot = plot_scaling_vs_depth(
                exponent_trace,
                out_dir / f"{run_stem}.png",
                settings={
                    "k": args.k,
                    "r": args.r,
                    "n_min": args.n_min,
                    "n_max": args.n_max,
                    "test_size": args.test_size,
                    "seed": args.seed,
                    "depth_min": args.depth_min,
                    "depth_max": args.depth_max,
                    "train_n": args.train_n,
                    "train_size": args.train_size,
                    "require_sat": True,
                },
            )
        return trace_path, summary_plot

    for depth in range(args.depth_min, args.depth_max + 1):
        t0 = time.time()
        print(f"\n{'=' * 72}\nDepth p = {depth}\n{'=' * 72}")

        if args.skip_train:
            if depth not in cached:
                print(f"  skip: no angles in {args.angle_log} for depth={depth}")
                continue
            dg, db = cached[depth]
            print(f"  using cached angles: dg={dg:.6f}, db={db:.6f}")
        else:
            print("  training LR angles (notebook protocol, BM24 simulator)...")
            training_h = generate_training_h_diagonals(
                train_n=args.train_n, k=args.k, r=args.r,
                train_size=args.train_size, base_seed=args.seed,
                m_sampling="notebook",
            )
            _, diag = train_lr_grid_search_bm24(
                training_h, train_n=args.train_n, depth=depth,
                skip_grid=args.skip_grid,
                beta_schedule=args.lr_beta_schedule,
                cobyla_maxiter=args.cobyla_maxiter,
            )
            dg, db = diag["best_deltas"]
            print(f"  trained: dg={dg:.6f}, db={db:.6f}, train_p_succ={diag['best_avg_train_p_succ']:.6e}")

        if depth == 1 and args.lr_beta_schedule == "decreasing" and abs(db) > 0:
            print("  warning: db has no effect at p=1 with decreasing schedule (beta[0]=0).")

        print("  benchmarking...")
        bres = run_algorithm_benchmark(
            n_min=args.n_min, n_max=args.n_max, k=args.k, r=args.r,
            test_size=args.test_size, base_seed=args.seed,
            algorithms=["lr_qaoa", "walksat", "walksatlm"],
            depth=depth, lr_delta_gamma=dg, lr_delta_beta=db,
            lr_beta_schedule=args.lr_beta_schedule,
            lr_angle_convention="bm24",
            max_flips=args.max_flips,
            require_sat=True,
        )

        json_path = None
        if args.save_per_depth:
            json_path = out_dir / f"{run_stem}-p{depth}.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(bres, f, indent=2)

        plot_summary = None
        if args.per_depth_plots:
            plot_path = out_dir / f"{run_stem}-p{depth}-bench.png"
            plot_summary = plot_benchmark_comparison(
                bres, plot_path, equiv_flips_per_shot=args.plot_equiv_flips_per_shot
            )
            print(f"  plot: {plot_path}")

        scaling = summarize_benchmark_scaling(bres)
        _print_scaling(scaling)

        won, win_detail = check_benchmark_win(
            bres,
            stop_criterion=args.stop_criterion,
            win_mode=args.win_mode,
            equiv_flips_per_shot=args.plot_equiv_flips_per_shot,
            plot_summary=plot_summary,
        )

        if args.stop_criterion in ("per-n", "both") and win_detail.get("per_n"):
            pn = win_detail["per_n"]
            print(f"  LR beats WalkSAT at n:   {pn.get('lr_beats_walksat_at_n', [])}")
            print(f"  LR beats WalkSATlm at n: {pn.get('lr_beats_walksatlm_at_n', [])}")

        exponent_trace.append({
            "depth": depth,
            "lr_log2_slope": scaling["lr_qaoa"]["median_runtime_slope_log2"],
            "walksat_log2_slope": scaling["walksat"]["median_flips_slope_log2"],
            "walksatlm_log2_slope": scaling["walksatlm"]["median_flips_slope_log2"],
            "lr_beats_both_scaling": scaling["beats_both_on_scaling"],
        })

        record = {
            "run_stem": run_stem,
            "depth": depth,
            "delta_gamma": dg,
            "delta_beta": db,
            "elapsed_s": time.time() - t0,
            "benchmark": bres,
            "benchmark_json": str(json_path) if json_path else None,
            "plot_summary": plot_summary,
            "scaling": scaling,
            "win_check": win_detail,
            "success_stop": won,
        }
        with open(sweep_log, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=float) + "\n")

        print(f"  stop satisfied ({args.stop_criterion}): {won}")

        if won:
            stop_depth = depth
            print(f"\n*** Stop: depth p={depth} satisfies {args.stop_criterion} ***")
            print(f"  angles: --lr-dgamma {dg} --lr-dbeta {db} --depth {depth}")
            trace_path, summary_plot = _write_final_outputs(stopped_early=True)
            print(f"  aggregate: {trace_path}")
            if summary_plot:
                print(f"  scaling plot: {summary_plot}")
            return

    print(f"\nNo depth in [{args.depth_min}, {args.depth_max}] met stop ({args.stop_criterion}).")
    trace_path, summary_plot = _write_final_outputs(stopped_early=False)
    print(f"Aggregate: {trace_path}")
    if summary_plot:
        print(f"Scaling plot: {summary_plot}")
    print(f"Per-depth detail: {sweep_log}")


if __name__ == "__main__":
    main()
