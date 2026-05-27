"""
Multivariable-facing accessors for the q=3 Newton saddle solver.

The implementation source of truth remains in symmetry_reduction.newton_saddle_q.
"""

from symmetry_reduction.newton_saddle_q import (
    active_indices_for_p,
    newton_solve_q,
    solve_8sat_saddle,
)

__all__ = [
    "active_indices_for_p",
    "newton_solve_q",
    "solve_8sat_saddle",
]
