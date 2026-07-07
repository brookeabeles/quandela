#!/usr/bin/env python3
"""Design the next BM24 gap-audit runs needed to test asymptotic persistence.

This consumes the current finite-size scaling report and row-level audit CSV.
It asks a practical question:

    At what future n would the positive-limit fit and the best flexible
    zero-limit fit make measurably different predictions?

The output is a Markdown/JSON run design, including approximate N requirements
from current bootstrap noise and concrete evaluate.py commands.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from plot_asymptotic_gap_certificate import (  # noqa: E402
    load_cells,
    per_n_stats,
)


ZERO_LIMIT_MODELS = (
    "pure_1_over_n_decay",
    "curved_zero_limit_decay",
    "slow_power_decay_to_zero",
)


def parse_ints(spec: str) -> list[int]:
    return [int(x) for x in str(spec).split(",") if x.strip()]


def predict(model_name: str, params: Mapping[str, float], n: float) -> float:
    if model_name == "constant_positive_limit":
        return float(params["gap_inf"])
    if model_name == "positive_limit_plus_1_over_n":
        return float(params["gap_inf"]) + float(params["inv_n_coeff"]) / float(n)
    if model_name == "pure_1_over_n_decay":
        return float(params["inv_n_coeff"]) / float(n)
    if model_name == "curved_zero_limit_decay":
        return float(params["inv_n_coeff"]) / float(n) + float(params["inv_n2_coeff"]) / float(n * n)
    if model_name == "slow_power_decay_to_zero":
        return float(params["power_coeff"]) * (float(n) ** (-float(params["alpha"])))
    if model_name == "linear_in_n_diagnostic":
        return float(params["intercept"]) + float(params["slope_n"]) * float(n)
    raise ValueError(f"unknown model {model_name}")


def bootstrap_point_se(by_n: Mapping[int, dict], metric: str, *, n_boot: int, seed: int) -> dict[int, float]:
    rng = np.random.default_rng(seed)
    out: dict[int, float] = {}
    for n, row in sorted(by_n.items()):
        vals = np.empty(int(n_boot), dtype=float)
        ps = np.asarray(row["p_succ"], dtype=float)
        xs = np.asarray(row["X"], dtype=float)
        for b in range(int(n_boot)):
            idx = rng.integers(0, len(ps), len(ps))
            vals[b] = float(per_n_stats(ps[idx], xs[idx], int(n))[metric])
        out[int(n)] = float(np.std(vals, ddof=1))
    return out


def current_tail_noise(by_n: Mapping[int, dict], metric: str, *, tail_n_min: int, n_boot: int, seed: int) -> float:
    se_by_n = bootstrap_point_se(by_n, metric, n_boot=n_boot, seed=seed)
    tail = [se for n, se in se_by_n.items() if int(n) >= int(tail_n_min)]
    if not tail:
        tail = list(se_by_n.values())
    return float(np.median(tail))


def best_zero_model(models: Mapping[str, Mapping[str, object]]) -> str:
    return min(ZERO_LIMIT_MODELS, key=lambda name: float(models[name]["aicc"]))


def n_required(
    *,
    current_n: int,
    current_se: float,
    separation: float,
    z: float,
    current_N: int,
) -> float:
    if separation <= 0.0 or not math.isfinite(separation):
        return float("inf")
    # Very rough exact-simulator extrapolation: keep the current bootstrap SE,
    # but inflate by sqrt(2^(n-current_n)) only through the user's compute budget?
    # Here we report statistical N only; computational cost is handled separately.
    return float(current_N * (z * current_se / separation) ** 2)


def build_eval_command(
    *,
    angles: Path,
    output: Path,
    n_values: Sequence[int],
    N: int,
    eval_seed: int,
    train_ns: Sequence[int],
    depths: Sequence[int],
    require_gpu: bool,
) -> str:
    parts = [
        "python",
        "experiments/lr_scaling/evaluate.py",
        "--angles",
        str(angles),
        "--output",
        str(output),
        "--n-values",
        ",".join(str(int(n)) for n in n_values),
        "--N",
        str(int(N)),
        "--eval-seed",
        str(int(eval_seed)),
        "--objectives",
        "mean_p",
        "--train-ns",
        ",".join(str(int(n)) for n in train_ns),
        "--depths",
        ",".join(str(int(d)) for d in depths),
        "--m-sampling",
        "notebook",
    ]
    if require_gpu:
        parts.append("--require-gpu")
    return " ".join(parts)


def evaluate_design(
    *,
    csv_path: Path,
    scaling_json: Path,
    angles: Path,
    train_ns: Sequence[int],
    depths: Sequence[int],
    future_ns: Sequence[int],
    metric: str,
    current_N: int,
    tail_n_min: int,
    bootstrap: int,
    seed: int,
    z: float,
    output_prefix: Path,
) -> dict:
    scaling = json.loads(scaling_json.read_text(encoding="utf-8"))
    cells = load_cells(csv_path, "mean_p")
    row_lookup = {
        (int(row["train_n"]), int(row["depth"])): row
        for row in scaling["rows"]
    }

    rows = []
    for key in sorted(row_lookup):
        train_n, depth = key
        if train_n not in set(train_ns) or depth not in set(depths):
            continue
        if key not in cells:
            continue
        metric_report = row_lookup[key]["metrics"][metric]
        models = metric_report["fits"]["models"]
        pos_model = "positive_limit_plus_1_over_n"
        zero_model = best_zero_model(models)
        pos_params = models[pos_model]["params"]
        zero_params = models[zero_model]["params"]
        tail_se = current_tail_noise(
            cells[key],
            metric,
            tail_n_min=tail_n_min,
            n_boot=bootstrap,
            seed=seed + 1000 * train_n + depth,
        )
        current_max_n = max(cells[key])
        forecasts = []
        for n in future_ns:
            pos_pred = predict(pos_model, pos_params, float(n))
            zero_pred = predict(zero_model, zero_params, float(n))
            sep = abs(pos_pred - zero_pred)
            req = n_required(
                current_n=current_max_n,
                current_se=tail_se,
                separation=sep,
                z=z,
                current_N=current_N,
            )
            forecasts.append(
                {
                    "n": int(n),
                    "positive_limit_prediction": float(pos_pred),
                    "zero_limit_model": zero_model,
                    "zero_limit_prediction": float(zero_pred),
                    "absolute_separation": float(sep),
                    "estimated_N_for_z_separation": float(req),
                }
            )
        rows.append(
            {
                "train_n": int(train_n),
                "depth": int(depth),
                "metric": metric,
                "current_tail_bootstrap_se": float(tail_se),
                "positive_model": pos_model,
                "zero_model": zero_model,
                "zero_model_delta_aicc": float(models[zero_model]["delta_aicc"]),
                "forecasts": forecasts,
            }
        )

    cpu_probe_ns = [int(n) for n in future_ns if 21 <= int(n) <= 22] or [int(min(future_ns))]
    gpu_decision_ns = [int(n) for n in future_ns if 23 <= int(n) <= 24]
    stage2_ns = [int(n) for n in future_ns if int(n) > 24]
    shard_paths: list[Path] = []

    cpu_path = output_prefix.with_name(output_prefix.name + "_cpu_probe_n" + f"{min(cpu_probe_ns)}-{max(cpu_probe_ns)}.csv")
    shard_paths.append(cpu_path)
    commands = [
        {
            "stage": "stage1_cpu_probe",
            "purpose": "Cheap extension beyond n=20; useful sanity check before committing GPU time.",
            "command": build_eval_command(
                angles=angles,
                output=cpu_path,
                n_values=cpu_probe_ns,
                N=current_N,
                eval_seed=100027,
                train_ns=train_ns,
                depths=depths,
                require_gpu=False,
            ),
        }
    ]
    if gpu_decision_ns:
        gpu_path = output_prefix.with_name(output_prefix.name + "_gpu_decision_n" + f"{min(gpu_decision_ns)}-{max(gpu_decision_ns)}.csv")
        shard_paths.append(gpu_path)
        commands.append(
            {
                "stage": "stage2_gpu_decision_window",
                "purpose": "First genuinely discriminating window; p=50 and n=24 should already separate positive-limit and curved-decay fits at current N.",
                "command": build_eval_command(
                    angles=angles,
                    output=gpu_path,
                    n_values=gpu_decision_ns,
                    N=current_N,
                    eval_seed=100027,
                    train_ns=train_ns,
                    depths=depths,
                    require_gpu=True,
                ),
            }
        )
    if stage2_ns:
        large_path = output_prefix.with_name(output_prefix.name + "_larger_n" + f"{min(stage2_ns)}-{max(stage2_ns)}.csv")
        shard_paths.append(large_path)
        commands.append(
            {
                "stage": "stage3_larger_n_if_still_ambiguous",
                "purpose": "Push to the next window only if n=23..24 remains compatible with both persistence and decay.",
                "command": build_eval_command(
                    angles=angles,
                    output=large_path,
                    n_values=stage2_ns,
                    N=current_N,
                    eval_seed=100027,
                    train_ns=train_ns,
                    depths=depths,
                    require_gpu=True,
                ),
            }
        )
    for extra_seed in (200027, 300027):
        repl_path = output_prefix.with_name(output_prefix.name + f"_replicate_seed{extra_seed}_n{min(cpu_probe_ns)}-{max(cpu_probe_ns)}.csv")
        shard_paths.append(repl_path)
        commands.append(
            {
                "stage": f"replicate_eval_seed_{extra_seed}",
                "purpose": "Check whether the first extension is eval-root stable before spending more GPU time.",
                "command": build_eval_command(
                    angles=angles,
                    output=repl_path,
                    n_values=cpu_probe_ns,
                    N=current_N,
                    eval_seed=extra_seed,
                    train_ns=train_ns,
                    depths=depths,
                    require_gpu=False,
                ),
            }
        )
    combined_path = output_prefix.with_name(output_prefix.name + "_combined_with_current.csv")
    merge_inputs = [csv_path, *shard_paths]
    commands.append(
        {
            "stage": "merge_current_and_followup_rows",
            "purpose": "Create the single CSV consumed by the diagnostics.",
            "command": (
                "python experiments/lr_scaling/merge_gap_audit_csvs.py "
                + " ".join(str(path) for path in merge_inputs)
                + f" --output {combined_path} --dedupe --ignore-missing"
            ),
        }
    )
    commands.append(
        {
            "stage": "reanalyze_combined_rows",
            "purpose": "Regenerate mechanism, gate, and finite-size scaling diagnostics on the combined data.",
            "command": (
                "python experiments/lr_scaling/run_gap_persistence_analysis.py "
                f"--csv {combined_path} "
                "--output-dir results/bm24_runs/bm24_gap_audit/CLOSE_gap_analysis/followup_reanalysis "
                "--tail-n-min 20 --window-width 9"
            ),
        }
    )

    return {
        "schema_version": 1,
        "kind": "bm24_asymptotic_gap_followup_design",
        "csv": str(csv_path),
        "scaling_json": str(scaling_json),
        "angles": str(angles),
        "metric": metric,
        "current_N": int(current_N),
        "future_ns": [int(n) for n in future_ns],
        "z_separation": float(z),
        "rows": rows,
        "commands": commands,
        "interpretation": {
            "main_obstruction": (
                "The current n-window cannot rule out flexible zero-limit curves. "
                "A finite-data claim must pre-register a zero-limit family or bring in "
                "theory that lower-bounds the decay exponent."
            ),
            "decisive_pattern_for_persistence": (
                "Later windows keep Delta_n and the Jensen component positive while "
                "the best zero-limit models become AICc-worse across independent eval roots."
            ),
            "decisive_pattern_against_persistence": (
                "Delta_n, Jensen, tilted integral, or scaled n Var(X_n) trend downward "
                "and become well fit by a zero-limit model."
            ),
        },
    }


def fmt(x: object, digits: int = 4) -> str:
    if isinstance(x, (int, float)):
        if not math.isfinite(float(x)):
            return str(x)
        return f"{float(x):.{digits}f}"
    return str(x)


def write_markdown(payload: Mapping[str, object], out_path: Path) -> None:
    lines = [
        "# Follow-Up Design For The Asymptotic Gap Question",
        "",
        "The current data identify the finite-window mechanism but do not resolve the limit.",
        "This design asks what larger-n runs would actually discriminate persistence from flexible decay-to-zero behavior.",
        "",
        "## Core Implication",
        "",
        "No finite dataset can rule out an arbitrarily slow decay such as `Delta_n = a n^{-alpha}` with `alpha` extremely close to zero.",
        "Therefore the asymptotic question needs either a theoretical lower bound on the allowed decay rate or a pre-registered finite family of zero-limit alternatives.",
        "",
        "## Forecast Table",
        "",
        "| train_n | depth | n | positive-limit pred | best zero-limit model | zero-limit pred | separation | rough N for 2-sigma separation |",
        "|---:|---:|---:|---:|---|---:|---:|---:|",
    ]
    for row in payload["rows"]:  # type: ignore[index]
        for forecast in row["forecasts"]:
            lines.append(
                "| {tn} | {depth} | {n} | {pos} | `{zero_model}` | {zero} | {sep} | {reqN} |".format(
                    tn=row["train_n"],
                    depth=row["depth"],
                    n=forecast["n"],
                    pos=fmt(forecast["positive_limit_prediction"]),
                    zero_model=forecast["zero_limit_model"],
                    zero=fmt(forecast["zero_limit_prediction"]),
                    sep=fmt(forecast["absolute_separation"]),
                    reqN=fmt(forecast["estimated_N_for_z_separation"], 0),
                )
            )
    lines.extend(
        [
            "",
            "The `N` column is only a statistical guide based on current bootstrap noise. Exact simulation cost still grows rapidly with `n`.",
            "",
            "## Runnable Stages",
            "",
        ]
    )
    for cmd in payload["commands"]:  # type: ignore[index]
        lines.extend(
            [
                f"### {cmd['stage']}",
                "",
                cmd["purpose"],
                "",
                "```bash",
                cmd["command"],
                "```",
                "",
            ]
        )
    lines.extend(
        [
            "## Decision Rule",
            "",
            "After collecting each new shard, merge the current and follow-up rows, then run the three reanalysis stages above on the combined CSV.",
            "Evidence for persistence strengthens only if later windows keep the gap/Jensen/tilted-integral positive and make the flexible zero-limit models worse across multiple eval roots.",
            "Evidence against persistence appears if the gap, Jensen component, tilted integral, or scaled variance trends toward zero and the zero-limit models become stable winners.",
            "",
        ]
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=Path("results/bm24_runs/bm24_gap_audit/N500_n12-20_all.csv"))
    parser.add_argument(
        "--scaling-json",
        type=Path,
        default=Path("results/bm24_runs/bm24_gap_audit/CLOSE_gap_analysis/finite_size_scaling_verdict.json"),
    )
    parser.add_argument("--angles", type=Path, default=Path("results/bm24_runs/bm24_gap_audit/angles_grid_existing.json"))
    parser.add_argument("--train-ns", default="12,16")
    parser.add_argument("--depths", default="20,50")
    parser.add_argument("--future-ns", default="21,22,23,24,25,26,28")
    parser.add_argument("--metric", default="point_gap", choices=["point_gap", "jensen_gap"])
    parser.add_argument("--current-N", type=int, default=500)
    parser.add_argument("--tail-n-min", type=int, default=16)
    parser.add_argument("--bootstrap", type=int, default=300)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--z", type=float, default=2.0)
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("results/bm24_runs/bm24_gap_audit/followup_gap_persistence"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("results/bm24_runs/bm24_gap_audit/CLOSE_gap_analysis/asymptotic_gap_followup_design.json"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("results/bm24_runs/bm24_gap_audit/CLOSE_gap_analysis/asymptotic_gap_followup_design.md"),
    )
    args = parser.parse_args()

    payload = evaluate_design(
        csv_path=args.csv,
        scaling_json=args.scaling_json,
        angles=args.angles,
        train_ns=parse_ints(args.train_ns),
        depths=parse_ints(args.depths),
        future_ns=parse_ints(args.future_ns),
        metric=args.metric,
        current_N=int(args.current_N),
        tail_n_min=int(args.tail_n_min),
        bootstrap=int(args.bootstrap),
        seed=int(args.seed),
        z=float(args.z),
        output_prefix=args.output_prefix,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_markdown(payload, args.output_md)
    print(args.output_json)
    print(args.output_md)


if __name__ == "__main__":
    main()
