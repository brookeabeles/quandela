"""Training compatibility exports for LR scaling."""

from __future__ import annotations

import sys
from pathlib import Path

_PHASECRAFT = Path(__file__).resolve().parents[2]
_REPO = _PHASECRAFT.parent
for _p in (_REPO, _PHASECRAFT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from phasecraft.experiments.lr_scaling.bm24_gap_audit import (
    main_train as _main_train,
)
from phasecraft.experiments.lr_scaling.train_lr_notebook_protocol import (
    run_train_eval_with_retries,
    train_lr_grid_search_bm24,
    train_mean_p_collapse_reject,
)

__all__ = [
    "run_train_eval_with_retries",
    "train_lr_grid_search_bm24",
    "train_mean_p_collapse_reject",
]


def main() -> None:
    """CLI: freeze BM24 gap-audit angles to JSON."""
    _main_train()


if __name__ == "__main__":
    main()
