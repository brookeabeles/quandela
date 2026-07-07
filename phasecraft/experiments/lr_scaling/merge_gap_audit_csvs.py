#!/usr/bin/env python3
"""Merge BM24 gap-audit CSV shards while preserving one header."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--dedupe",
        action="store_true",
        help="Drop duplicate rows by seed/n/p/objective/train_n/angle_id/eval_index.",
    )
    parser.add_argument(
        "--ignore-missing",
        action="store_true",
        help="Skip missing input shards; useful while staged follow-up runs are incomplete.",
    )
    args = parser.parse_args()

    fieldnames: list[str] | None = None
    rows: list[dict[str, str]] = []
    seen: set[tuple[str, ...]] = set()
    key_cols = ("seed", "n", "p", "objective", "train_n", "angle_id", "eval_index")
    missing: list[Path] = []
    for path in args.inputs:
        if not path.exists():
            if args.ignore_missing:
                missing.append(path)
                continue
            raise FileNotFoundError(path)
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise ValueError(f"{path} has no header")
            if fieldnames is None:
                fieldnames = list(reader.fieldnames)
            elif list(reader.fieldnames) != fieldnames:
                raise ValueError(f"{path} header differs from first input")
            for row in reader:
                if args.dedupe:
                    key = tuple(str(row.get(col, "")) for col in key_cols)
                    if key in seen:
                        continue
                    seen.add(key)
                rows.append(row)

    if fieldnames is None:
        raise ValueError("no input rows")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(args.output)
    print(f"rows={len(rows)}")
    if missing:
        print(f"ignored_missing={len(missing)}")


if __name__ == "__main__":
    main()
