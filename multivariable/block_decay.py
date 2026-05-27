"""
Multivariable-facing wrappers for 8-SAT block-structure utilities.

The implementation source of truth remains in
symmetry_reduction.saddle_hessian_research.direction2_block_structure.
"""

from symmetry_reduction.saddle_hessian_research.direction2_block_structure import (
    compute_cov_block_structure,
    plot_block_heatmaps,
    run_block_and_truncation,
)

__all__ = [
    "compute_cov_block_structure",
    "plot_block_heatmaps",
    "run_block_and_truncation",
]
