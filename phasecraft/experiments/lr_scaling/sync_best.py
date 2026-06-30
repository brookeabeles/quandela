#!/usr/bin/env python3
"""Synchronize curated ``results/best`` LR-scaling figures from source runs.

The curated train plots are regenerated from their JSON run artifacts instead
of copied from stale PNGs, so sub-window WalkSAT lines are refit consistently.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

_PHASECRAFT = Path(__file__).resolve().parents[2]
_REPO = _PHASECRAFT.parent
_LR = Path(__file__).resolve().parent
for _p in (_REPO, _PHASECRAFT, _LR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from compare_objectives_fixed_n import replot_from_json  # noqa: E402


CLASSICAL_JSON = Path("results/bm24_runs/06-10/run1/scaling-tn14.json")


@dataclass(frozen=True)
class ReplotBestFigure:
    source_json: Path
    best_png: Path
    eval_n_min: int | None = None
    eval_n_max: int | None = None
    refit_eval_window: bool = False


BEST_FIGURES: tuple[ReplotBestFigure, ...] = (
    ReplotBestFigure(
        source_json=Path(
            "results/bm24_runs/multi_seed_ctyp_cann/06-24/run1/"
            "train12-tr100-te200-n12-18-seed0.json"
        ),
        best_png=Path("results/best/train12-tr100-te200-n12-18-seed0.png"),
    ),
    ReplotBestFigure(
        source_json=Path(
            "results/bm24_runs/multi_seed_ctyp_cann/06-24/run2/"
            "train12-tr100-te200-n12-18-seed42.json"
        ),
        best_png=Path("results/best/train12-tr100-te200-n12-18-seed42.png"),
    ),
    ReplotBestFigure(
        source_json=Path("results/bm24_runs/06-09/run4/train16-tr100-te200-n12-18.json"),
        best_png=Path("results/best/train16-tr100-te200-n12-18.png"),
    ),
    ReplotBestFigure(
        source_json=Path("results/bm24_runs/06-09/run4/train16-tr100-te200-n12-18.json"),
        best_png=Path("results/best/train16-tr100-te200-n14-18.png"),
        eval_n_min=14,
        eval_n_max=18,
        refit_eval_window=True,
    ),
)


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else (_PHASECRAFT / path).resolve()


def _deps_for(entry: ReplotBestFigure, classical_json: Path) -> tuple[Path, ...]:
    return (
        _resolve(entry.source_json),
        _resolve(classical_json),
        Path(__file__).resolve(),
        _resolve(Path("experiments/lr_scaling/compare_objectives_fixed_n.py")),
        _resolve(Path("experiments/lr_scaling/plot_compare_run1_run4.py")),
    )


def _is_stale(output: Path, deps: Iterable[Path]) -> bool:
    if not output.exists():
        return True
    output_mtime = output.stat().st_mtime
    return any(dep.exists() and dep.stat().st_mtime > output_mtime for dep in deps)


def sync_best(*, force: bool = False, check: bool = False, classical_json: Path = CLASSICAL_JSON) -> int:
    stale: list[ReplotBestFigure] = []
    for entry in BEST_FIGURES:
        source = _resolve(entry.source_json)
        output = _resolve(entry.best_png)
        deps = _deps_for(entry, classical_json)
        missing = [p for p in deps if not p.exists()]
        if missing:
            missing_text = "\n  ".join(str(p) for p in missing)
            raise FileNotFoundError(f"Missing dependency for {output.name}:\n  {missing_text}")
        if force or _is_stale(output, deps):
            stale.append(entry)

    if check:
        for entry in stale:
            print(f"STALE {entry.best_png}")
        if not stale:
            print("results/best train figures are up to date")
        return 1 if stale else 0

    for entry in stale:
        output = _resolve(entry.best_png)
        output.parent.mkdir(parents=True, exist_ok=True)
        replot_from_json(
            _resolve(entry.source_json),
            output,
            eval_n_min=entry.eval_n_min,
            eval_n_max=entry.eval_n_max,
            classical_json=_resolve(classical_json),
            refit_eval_window=entry.refit_eval_window,
        )
        print(f"updated {entry.best_png}")

    if not stale:
        print("results/best train figures are up to date")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Regenerate all managed figures.")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Report stale managed figures and exit non-zero if any need regeneration.",
    )
    parser.add_argument(
        "--classical-json",
        type=Path,
        default=CLASSICAL_JSON,
        help="JSON containing setup.classical median-flip baselines.",
    )
    args = parser.parse_args()
    raise SystemExit(
        sync_best(
            force=bool(args.force),
            check=bool(args.check),
            classical_json=args.classical_json,
        )
    )


if __name__ == "__main__":
    main()
