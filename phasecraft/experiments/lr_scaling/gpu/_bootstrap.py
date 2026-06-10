"""Put repo root and phasecraft on sys.path (no hardcoded user paths)."""

from __future__ import annotations

import sys
from pathlib import Path


def phasecraft_root() -> Path:
    # .../phasecraft/experiments/lr_scaling/gpu/_bootstrap.py -> parents[3]
    return Path(__file__).resolve().parents[3]


def repo_root() -> Path:
    return phasecraft_root().parent


def results_bm24_dir() -> Path:
    """Canonical benchmark output root (``results/bm24_runs``)."""
    out = phasecraft_root() / "results" / "bm24_runs"
    out.mkdir(parents=True, exist_ok=True)
    return out


def ensure_paths() -> tuple[Path, Path]:
    pc = phasecraft_root()
    repo = repo_root()
    for p in (repo, pc):
        s = str(p)
        if s not in sys.path:
            sys.path.insert(0, s)
    lr_scaling = pc / "experiments" / "lr_scaling"
    sim = pc / "lib" / "sim"
    for extra in (lr_scaling, sim):
        s = str(extra)
        if s not in sys.path:
            sys.path.insert(0, s)
    return repo, pc
