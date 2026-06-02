#!/usr/bin/env python3
"""Organize ``bm24_runs/`` artifacts (dated run folders or legacy flatten)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO.parent) not in sys.path:
    sys.path.insert(0, str(_REPO.parent))

from phasecraft.lib.paths import bm24_runs_dir  # noqa: E402
from phasecraft.lib.sim.bm24_run_io import (  # noqa: E402
    flatten_bm24_scaling_artifacts,
    migrate_flat_bm24_runs_to_dated,
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=str, default=str(bm24_runs_dir()))
    p.add_argument(
        "--migrate-dated-runs",
        action="store_true",
        help="Move flat scaling JSON/PNG into MM-DD/runN/ (default action).",
    )
    p.add_argument(
        "--flatten",
        action="store_true",
        help="Pull artifacts from legacy runtime_scaling/ success_scaling/ subfolders to flat root.",
    )
    p.add_argument("--no-replot", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    root = Path(args.root).expanduser().resolve()

    if args.flatten:
        moves = flatten_bm24_scaling_artifacts(
            root,
            replot=not args.no_replot,
            dry_run=bool(args.dry_run),
        )
        print(f"{'Would move' if args.dry_run else 'Moved'} {len(moves)} file(s) to {root}")
        for src, dest in moves[:25]:
            print(f"  {src.parent.name}/{src.name} -> {dest.name}")
        if len(moves) > 25:
            print(f"  ... and {len(moves) - 25} more")
        return

    moves = migrate_flat_bm24_runs_to_dated(root, dry_run=bool(args.dry_run))
    report = {
        "root": str(root),
        "moves": len(moves),
        "dry_run": bool(args.dry_run),
        "sample": [
            {"from": str(s), "to": str(d)} for s, d in moves[:30]
        ],
    }
    print(json.dumps(report, indent=2))
    if args.dry_run:
        print("(dry run — no files changed)")


if __name__ == "__main__":
    main()
