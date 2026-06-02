#!/usr/bin/env python3
"""Pull scaling artifacts from legacy subfolders into flat bm24_runs/ and replot PNGs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO.parent) not in sys.path:
    sys.path.insert(0, str(_REPO.parent))

from phasecraft.lib.paths import bm24_runs_dir  # noqa: E402
from phasecraft.lib.sim.bm24_run_io import flatten_bm24_scaling_artifacts  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=str, default=str(bm24_runs_dir()))
    p.add_argument("--no-replot", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    root = Path(args.root).expanduser().resolve()
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


if __name__ == "__main__":
    main()
