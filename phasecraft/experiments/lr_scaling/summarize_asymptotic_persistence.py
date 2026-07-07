#!/usr/bin/env python3
"""Write a compact persistence decision table from a gap diagnostic summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List


def fmt(x: object, digits: int = 4) -> str:
    if isinstance(x, (int, float)):
        return f"{float(x):.{digits}f}"
    return str(x)


def ci_text(ci: object) -> str:
    if not isinstance(ci, dict):
        return ""
    gap = ci.get("gap")
    if not isinstance(gap, list) or len(gap) != 2:
        return ""
    return f"[{float(gap[0]):.4f}, {float(gap[1]):.4f}]"


def row_for_summary(row: dict) -> List[str]:
    tilted = row.get("tilted_variance") or {}
    return [
        f"{min(row['ns'])}-{max(row['ns'])}",
        str(row["train_n"]),
        str(row["depth"]),
        fmt(row["measured_gap_slope"]),
        ci_text(row.get("bootstrap_ci")),
        fmt(row["jensen_slope"]),
        fmt(row["skew_slope"]),
        fmt(tilted.get("tilted_integral_slope", "")),
        fmt(tilted.get("tail_nvar_flatness_ratio", "")),
        fmt(row["tail_top1_mass_share"], 3),
        fmt(row["tail_top5_mass_share"], 3),
        fmt(row["tail_loo_max_abs_shift"], 4),
    ]


def write_markdown(rows: Iterable[dict], out_path: Path) -> None:
    headers = [
        "window",
        "train_n",
        "depth",
        "Gap",
        "CI",
        "Jensen",
        "Skew",
        "tilted J",
        "Var1/Var0",
        "top1",
        "top5",
        "LOO",
    ]
    lines = [
        "# Current Persistence Decision Table",
        "",
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row_for_summary(row)) + " |")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path(
            "results/bm24_runs/bm24_gap_audit/CLOSE_gap_analysis/"
            "04_asymptotic_gap_certificate_summary.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/bm24_runs/bm24_gap_audit/CLOSE_gap_analysis/"
            "asymptotic_persistence_current_decision_table.md"
        ),
    )
    parser.add_argument("--depths", default="20,50")
    args = parser.parse_args()

    payload = json.loads(args.summary.read_text(encoding="utf-8"))
    depths = {int(x) for x in str(args.depths).split(",") if x.strip()}
    rows = [r for r in payload["rows"] if int(r["depth"]) in depths]
    write_markdown(rows, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
