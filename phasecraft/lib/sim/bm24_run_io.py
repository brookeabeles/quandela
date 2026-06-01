"""Filename helpers and plot titles for ``phasecraft/results/bm24_runs/`` artifacts."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from phasecraft.lib.paths import bm24_runs_dir

# Stems look like: 05-19_1645-sweep-scaling  (month-day, 24h time, kind tag; no year)
STEM_TIME_FMT = "%m-%d_%H%M"


def make_run_stem(kind: str, when: float | None = None) -> str:
    """Return a sortable filename stem ``MM-DD_HHMM-{kind}`` (no calendar year)."""
    loc = time.localtime(when if when is not None else time.time())
    return f"{time.strftime(STEM_TIME_FMT, loc)}-{kind}"


def default_bm24_runs_dir() -> Path:
    return bm24_runs_dir()


def _fmt_r(r: Any) -> str:
    if isinstance(r, float):
        return f"{r:g}"
    return str(r)


def format_benchmark_title(
    settings: Mapping[str, Any],
    *,
    headline: str = "LR scaling vs depth",
    depth_min: Optional[int] = None,
    depth_max: Optional[int] = None,
    depths: Optional[Sequence[int]] = None,
) -> str:
    """One-line matplotlib title with the main benchmark parameters."""
    s = dict(settings)
    parts = [headline]

    k, r = s.get("k"), s.get("r")
    if k is not None and r is not None:
        parts.append(f"k={k}, r={_fmt_r(r)}")

    n_min, n_max = s.get("n_min"), s.get("n_max")
    if n_min is not None and n_max is not None:
        parts.append(f"n∈[{n_min},{n_max}]")

    test_size = s.get("test_size")
    if test_size is not None:
        parts.append(f"test_size={test_size}")

    train_size = s.get("train_size")
    if train_size is not None:
        parts.append(f"train_size={train_size}")

    seed = s.get("seed", s.get("base_seed"))
    if seed is not None:
        parts.append(f"seed={seed}")

    p_lo = depth_min if depth_min is not None else s.get("depth_min")
    p_hi = depth_max if depth_max is not None else s.get("depth_max")
    if depths:
        p_lo = min(int(d) for d in depths)
        p_hi = max(int(d) for d in depths)
    elif s.get("depth") is not None:
        p_lo = p_hi = int(s["depth"])
    if p_lo is not None and p_hi is not None:
        parts.append(f"p∈[{p_lo},{p_hi}]" if p_lo != p_hi else f"p={p_lo}")

    train_n = s.get("train_n")
    if train_n is not None:
        parts.append(f"train_n={train_n}")

    if s.get("require_sat") is True:
        parts.append("SAT-only")

    return " · ".join(parts)
