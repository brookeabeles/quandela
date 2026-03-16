"""
Saddle path continuation and Stokes-curve visualisation for the BM24 QAOA phase Φ.

This package implements:
- Step A: forward continuation of a chosen saddle as a function of γ̃;
- Step B: backward continuation of a rival saddle branch born at large γ̃;
- Step C: 2D complex-plane visualisations (Re Φ "strips", gradient field, and saddles).

See `saddle_tracking.py` for the main API.
"""

from .saddle_tracking import (
    newton_solve_saddle,
    continue_saddle_branch,
    track_primary_and_rival_branches,
    find_stokes_crossing,
    plot_stokes_slices,
)

__all__ = [
    "newton_solve_saddle",
    "continue_saddle_branch",
    "track_primary_and_rival_branches",
    "find_stokes_crossing",
    "plot_stokes_slices",
]

