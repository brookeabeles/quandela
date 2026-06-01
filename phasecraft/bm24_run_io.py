"""Compatibility re-export. Canonical: ``phasecraft.lib.sim.bm24_run_io``."""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from phasecraft.lib.sim.bm24_run_io import *  # noqa: F401,F403
