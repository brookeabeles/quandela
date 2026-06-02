#!/usr/bin/env python3
"""Split a mixed lr_train_optimal_angles.txt into legacy and v2/v3 logs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO.parent) not in sys.path:
    sys.path.insert(0, str(_REPO.parent))

from phasecraft.lib.paths import (  # noqa: E402
    bm24_runs_dir,
    lr_train_optimal_angles_compat_path,
    lr_train_optimal_angles_legacy_path,
    lr_train_optimal_angles_v2_path,
)
from phasecraft.lib.sim.bm24_run_io import split_mixed_angle_log  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--mixed",
        type=str,
        default=str(lr_train_optimal_angles_compat_path()),
        help="Mixed angle log to split (default: bm24_runs/lr_train_optimal_angles.txt)",
    )
    p.add_argument("--legacy", type=str, default=str(lr_train_optimal_angles_legacy_path()))
    p.add_argument("--v2", type=str, default=str(lr_train_optimal_angles_v2_path()))
    p.add_argument("--no-merge-existing", action="store_true")
    p.add_argument("--no-compat-symlink", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    report = split_mixed_angle_log(
        Path(args.mixed),
        legacy_path=Path(args.legacy),
        v2_path=Path(args.v2),
        merge_existing=not args.no_merge_existing,
        compat_symlink=not args.no_compat_symlink,
        dry_run=bool(args.dry_run),
    )
    print(json.dumps(report, indent=2))
    if args.dry_run:
        print("(dry run — no files changed)")


if __name__ == "__main__":
    main()
