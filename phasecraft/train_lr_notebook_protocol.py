"""Compatibility CLI entrypoint. Canonical: ``phasecraft.experiments.lr_scaling.train_lr_notebook_protocol``."""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_PHASECRAFT = Path(__file__).resolve().parent
for _p in (_ROOT, _PHASECRAFT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from phasecraft.experiments.lr_scaling.train_lr_notebook_protocol import *  # noqa: F401,F403

if __name__ == "__main__":
    from phasecraft.experiments.lr_scaling.train_lr_notebook_protocol import main

    main()
