"""LR-scaling configuration defaults.

This module centralizes reusable defaults while legacy scripts remain runnable.
"""

from __future__ import annotations

from typing import Tuple

DEFAULT_DG_BOUNDS: Tuple[float, float] = (-2.0, -0.01)
DEFAULT_DB_BOUNDS: Tuple[float, float] = (0.1, 4.0)
DEFAULT_INITIAL_DELTAS: Tuple[float, float] = (-0.8, 0.49)
MIN_TRAIN_MEAN_P_SUCC: float = 1e-4
TRAIN_MEAN_P_REGRESSION_FACTOR: float = 10.0
DEFAULT_EPS: float = 1e-300
MIN_RECOMMENDED_NS_FOR_SLOPE: int = 3

