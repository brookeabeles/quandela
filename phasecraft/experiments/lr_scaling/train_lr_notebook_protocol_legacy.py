"""Compatibility wrapper for the archived legacy LR-QAOA trainer.

New code should use :mod:`phasecraft.experiments.lr_scaling.train_lr_fixed_n`.
This module remains so old scripts and ``--legacy-objective`` imports keep
working while the large superseded implementation lives under ``archive/``.
"""

from __future__ import annotations

import sys
from pathlib import Path

_PHASECRAFT_ROOT = Path(__file__).resolve().parents[2]
_REPO_ROOT = _PHASECRAFT_ROOT.parent
for _p in (_REPO_ROOT, _PHASECRAFT_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from phasecraft.experiments.lr_scaling.archive.train_lr_notebook_protocol_legacy import *  # noqa: F401,F403,E402
from phasecraft.experiments.lr_scaling.archive.train_lr_notebook_protocol_legacy import main as _main  # noqa: E402


if __name__ == "__main__":
    _main()
