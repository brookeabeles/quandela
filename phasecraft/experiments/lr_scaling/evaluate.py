"""Evaluation compatibility exports for LR scaling."""

from __future__ import annotations

import sys
from pathlib import Path

_PHASECRAFT = Path(__file__).resolve().parents[2]
_REPO = _PHASECRAFT.parent
for _p in (_REPO, _PHASECRAFT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from phasecraft.experiments.lr_scaling.bm24_gap_audit import (
    main_evaluate as _main_evaluate,
)
from phasecraft.experiments.lr_scaling.train_lr_notebook_protocol import (
    compute_mean_success_exponent_vs_n,
)

__all__ = ["compute_mean_success_exponent_vs_n"]


def main() -> None:
    """CLI: evaluate frozen BM24 gap-audit angles without re-optimizing."""
    _main_evaluate()


if __name__ == "__main__":
    main()
