#!/usr/bin/env python3
"""Analyze BM24 gap-audit per-instance CSV rows."""

from __future__ import annotations

import sys
from pathlib import Path

_PHASECRAFT = Path(__file__).resolve().parents[2]
_REPO = _PHASECRAFT.parent
for _p in (_REPO, _PHASECRAFT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from phasecraft.experiments.lr_scaling.bm24_gap_audit import main_analyze


def main() -> None:
    main_analyze()


if __name__ == "__main__":
    main()
