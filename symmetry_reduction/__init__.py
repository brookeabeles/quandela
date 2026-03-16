"""
QAOA Hessian spectral concentration numerical suite (BM24).

Modules: core, saddle, spectral, mechanisms, diagnostics, validation, plotting, tables.
"""

from .core import (
    build_structure_matrix,
    compute_b_s,
    compute_c_alpha,
    covariance_from_weights,
    hessian_log_at_y,
    softmax_weights,
    weights_at_zero,
    popcount,
    indices_by_size,
    DEFAULT_BETA,
    DEFAULT_R,
)
