#!/usr/bin/env python3
"""Evaluate asymptotic-persistence protocol gates on BM24 gap-audit CSV rows."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from plot_asymptotic_gap_certificate import (  # noqa: E402
    CellKey,
    bootstrap_slope_cis,
    compute_tilted_summaries,
    load_cells,
    summarize_cell,
)


def parse_ints(spec: str) -> List[int]:
    return [int(x) for x in str(spec).split(",") if x.strip()]


def parse_windows(spec: str | None, available_ns: Sequence[int], width: int) -> List[Tuple[int, int]]:
    if spec:
        out = []
        for part in str(spec).split(","):
            if not part.strip():
                continue
            lo_s, hi_s = part.replace("..", "-").split("-", 1)
            out.append((int(lo_s), int(hi_s)))
        return out
    ns = sorted(set(int(n) for n in available_ns))
    if not ns:
        return []
    if len(ns) <= width:
        return [(min(ns), max(ns))]
    return [(ns[i], ns[i + width - 1]) for i in range(0, len(ns) - width + 1)]


def filter_window(cells: Mapping[CellKey, Mapping[int, dict]], window: Tuple[int, int]) -> Dict[CellKey, Dict[int, dict]]:
    lo, hi = window
    out: Dict[CellKey, Dict[int, dict]] = {}
    for key, by_n in cells.items():
        sub = {n: row for n, row in by_n.items() if lo <= int(n) <= hi}
        if len(sub) >= 3:
            out[key] = sub
    return out


def protocol_thresholds(protocol: Mapping[str, object]) -> dict:
    gates = protocol.get("gates", {}) if isinstance(protocol, dict) else {}
    gap = gates.get("A_stable_positive_gap", {}).get("recommended_threshold", {})  # type: ignore[union-attr]
    jensen = gates.get("B_stable_positive_jensen", {}).get("recommended_threshold", {})  # type: ignore[union-attr]
    tilted = gates.get("C_stable_tilted_integral", {}).get("recommended_threshold", {})  # type: ignore[union-attr]
    tail = gates.get("E_tail_robustness", {}).get("warning_thresholds", {})  # type: ignore[union-attr]
    multi = gates.get("F_multiseed_replication", {})  # type: ignore[union-attr]
    return {
        "gap_lower_ci_min": float(gap.get("lower_95_bootstrap_CI_gap", 0.01)),
        "required_late_windows": int(gap.get("required_late_windows", 2)),
        "jensen_lower_ci_min": float(jensen.get("lower_95_bootstrap_CI_jensen", 0.0)),
        "jensen_fraction_min": float(jensen.get("jensen_fraction_of_gap_min", 0.5)),
        "tilted_late_mean_min": float(tilted.get("late_window_mean_tilted_J_min", 0.02)),
        "top1_warn": float(tail.get("top1_mass_share", 0.25)),
        "top5_warn": float(tail.get("top5_mass_share", 0.5)),
        "loo_warn": float(tail.get("leave_one_out_delta_shift", 0.005)),
        "minimum_seed_families": int(multi.get("minimum_seed_families", 3)),
    }


def row_verdict(row: dict, thresholds: Mapping[str, float]) -> dict:
    ci = row["bootstrap_ci"]
    gap_ci_low = float(ci["gap"][0])
    jensen_ci_low = float(ci["jensen"][0])
    gap = float(row["measured_gap_slope"])
    jensen = float(row["jensen_slope"])
    tilted = row["tilted_variance"]
    jensen_fraction = jensen / gap if gap > 0 else float("nan")
    return {
        "gap_pass": bool(gap_ci_low >= thresholds["gap_lower_ci_min"]),
        "jensen_pass": bool(
            jensen_ci_low >= thresholds["jensen_lower_ci_min"]
            and np.isfinite(jensen_fraction)
            and jensen_fraction >= thresholds["jensen_fraction_min"]
        ),
        "tilted_pass": bool(float(tilted["tilted_integral_point_tail"]) >= thresholds["tilted_late_mean_min"]),
        "tail_pass": bool(
            float(row["tail_top1_mass_share"]) <= thresholds["top1_warn"]
            and float(row["tail_top5_mass_share"]) <= thresholds["top5_warn"]
            and float(row["tail_loo_max_abs_shift"]) <= thresholds["loo_warn"]
        ),
        "jensen_fraction_of_gap": float(jensen_fraction),
    }


def evaluate(
    *,
    csv_path: Path,
    protocol_path: Path,
    objective: str,
    train_ns: Sequence[int],
    depths: Sequence[int],
    windows: Sequence[Tuple[int, int]],
    tail_n_min: int,
    bootstrap: int,
    seed: int,
    lambda_grid_size: int,
) -> dict:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    thresholds = protocol_thresholds(protocol)
    all_cells = load_cells(csv_path, objective)
    all_cells = {
        key: by_n
        for key, by_n in all_cells.items()
        if int(key[0]) in set(train_ns) and int(key[1]) in set(depths)
    }
    rng = np.random.default_rng(seed)
    lambdas = np.linspace(0.0, 1.0, int(lambda_grid_size))
    rows: List[dict] = []
    for window in windows:
        window_cells = filter_window(all_cells, window)
        summaries = {key: summarize_cell(by_n, int(tail_n_min)) for key, by_n in window_cells.items()}
        tilted = compute_tilted_summaries(cells=window_cells, lambdas=lambdas, tail_n_min=int(tail_n_min))
        for key, summary in sorted(summaries.items()):
            cis = bootstrap_slope_cis(window_cells[key], n_boot=int(bootstrap), rng=rng)
            train_n, depth = key
            row = {
                "window": {"lo": int(window[0]), "hi": int(window[1])},
                "train_n": int(train_n),
                "depth": int(depth),
                **summary,
                "bootstrap_ci": cis,
                "tilted_variance": tilted[key],
            }
            row["gate_verdict"] = row_verdict(row, thresholds)
            rows.append(row)

    # Overall status is intentionally conservative: if there are not enough late windows,
    # we cannot claim protocol-level persistence even if all current rows pass.
    row_passes = [r["gate_verdict"] for r in rows]
    stable_rows = [
        r for r in rows
        if r["gate_verdict"]["gap_pass"]
        and r["gate_verdict"]["jensen_pass"]
        and r["gate_verdict"]["tilted_pass"]
        and r["gate_verdict"]["tail_pass"]
    ]
    distinct_windows = sorted({(r["window"]["lo"], r["window"]["hi"]) for r in stable_rows})
    overall = {
        "row_count": len(rows),
        "stable_row_count": len(stable_rows),
        "stable_distinct_windows": [
            {"lo": int(lo), "hi": int(hi)} for lo, hi in distinct_windows
        ],
        "passes_current_rows": bool(rows and len(stable_rows) == len(rows)),
        "passes_protocol_persistence": bool(len(distinct_windows) >= thresholds["required_late_windows"]),
        "reason": (
            "enough stable late windows"
            if len(distinct_windows) >= thresholds["required_late_windows"]
            else "insufficient number of stable late windows for asymptotic-persistence evidence"
        ),
    }
    return {
        "schema_version": 1,
        "kind": "bm24_asymptotic_persistence_gate_evaluation",
        "csv": str(csv_path),
        "protocol": str(protocol_path),
        "objective": objective,
        "thresholds": thresholds,
        "windows": [{"lo": int(lo), "hi": int(hi)} for lo, hi in windows],
        "rows": rows,
        "overall": overall,
    }


def fmt(x: object, digits: int = 4) -> str:
    if isinstance(x, (int, float)):
        return f"{float(x):.{digits}f}"
    return str(x)


def write_markdown(payload: Mapping[str, object], out_path: Path) -> None:
    rows = payload["rows"]  # type: ignore[index]
    lines = [
        "# Asymptotic Persistence Gate Evaluation",
        "",
        f"CSV: `{payload['csv']}`",
        f"Protocol: `{payload['protocol']}`",
        "",
        "## Overall",
        "",
        "```json",
        json.dumps(payload["overall"], indent=2),  # type: ignore[index]
        "```",
        "",
        "## Rows",
        "",
        "| window | train_n | depth | Gap | gap CI | Jensen | Skew | tilted J | Var1/Var0 | top1 | top5 | LOO | row pass |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:  # type: ignore[assignment]
        ci = row["bootstrap_ci"]["gap"]
        gv = row["gate_verdict"]
        row_pass = gv["gap_pass"] and gv["jensen_pass"] and gv["tilted_pass"] and gv["tail_pass"]
        tilted = row["tilted_variance"]
        lines.append(
            "| {lo}-{hi} | {train_n} | {depth} | {gap} | [{ci_lo}, {ci_hi}] | "
            "{jensen} | {skew} | {tilted_j} | {flatness} | {top1} | {top5} | {loo} | {row_pass} |".format(
                lo=row["window"]["lo"],
                hi=row["window"]["hi"],
                train_n=row["train_n"],
                depth=row["depth"],
                gap=fmt(row["measured_gap_slope"]),
                ci_lo=fmt(ci[0]),
                ci_hi=fmt(ci[1]),
                jensen=fmt(row["jensen_slope"]),
                skew=fmt(row["skew_slope"]),
                tilted_j=fmt(tilted["tilted_integral_slope"]),
                flatness=fmt(tilted["tail_nvar_flatness_ratio"]),
                top1=fmt(row["tail_top1_mass_share"], 3),
                top5=fmt(row["tail_top5_mass_share"], 3),
                loo=fmt(row["tail_loo_max_abs_shift"], 4),
                row_pass="yes" if row_pass else "no",
            )
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("results/bm24_runs/bm24_gap_audit/N500_n12-20_all.csv"),
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path(
            "results/bm24_runs/bm24_gap_audit/CLOSE_gap_analysis/"
            "asymptotic_persistence_protocol.json"
        ),
    )
    parser.add_argument("--objective", default="mean_p")
    parser.add_argument("--train-ns", default="12,16")
    parser.add_argument("--depths", default="20,50")
    parser.add_argument("--windows", default=None, help="Comma-separated windows like 12-20,16-24")
    parser.add_argument("--window-width", type=int, default=9)
    parser.add_argument("--tail-n-min", type=int, default=16)
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--seed", type=int, default=321)
    parser.add_argument("--lambda-grid-size", type=int, default=101)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path(
            "results/bm24_runs/bm24_gap_audit/CLOSE_gap_analysis/"
            "asymptotic_persistence_gate_evaluation.json"
        ),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path(
            "results/bm24_runs/bm24_gap_audit/CLOSE_gap_analysis/"
            "asymptotic_persistence_gate_evaluation.md"
        ),
    )
    args = parser.parse_args()

    csv_path = args.csv
    protocol_path = args.protocol
    train_ns = parse_ints(args.train_ns)
    depths = parse_ints(args.depths)
    available = load_cells(csv_path, args.objective)
    available_ns = sorted({n for key, by_n in available.items() for n in by_n})
    windows = parse_windows(args.windows, available_ns, int(args.window_width))
    payload = evaluate(
        csv_path=csv_path,
        protocol_path=protocol_path,
        objective=args.objective,
        train_ns=train_ns,
        depths=depths,
        windows=windows,
        tail_n_min=int(args.tail_n_min),
        bootstrap=int(args.bootstrap),
        seed=int(args.seed),
        lambda_grid_size=int(args.lambda_grid_size),
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_markdown(payload, args.output_md)
    print(args.output_json)
    print(args.output_md)


if __name__ == "__main__":
    main()
