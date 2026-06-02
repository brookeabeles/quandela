"""Dataset generation compatibility exports for LR scaling."""

from __future__ import annotations

from phasecraft.experiments.lr_scaling.train_lr_notebook_protocol import (
    generate_training_h_diagonals,
    generate_training_h_diagonals_multi_n,
    proxy_n_values_for_training,
)

__all__ = [
    "generate_training_h_diagonals",
    "generate_training_h_diagonals_multi_n",
    "proxy_n_values_for_training",
]

