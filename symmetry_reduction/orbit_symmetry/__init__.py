"""
Orbit symmetry reduction and structural analysis (S_p × S_p × Z_2).

- Orbit classification (a, b, c) and reduced Hessian
- Complement vs normal truncation analysis
- Diagonal-orbit (a = c) analysis
- Symmetry commutation checks
"""

from .symmetry import (
    classify_alpha,
    canonical_orbit,
    build_orbit_map,
    orbit_averaged_hessian,
    symmetry_analysis,
    print_symmetry_analysis,
    run_symmetry_sweep,
    check_symmetry_commutation,
    complement_truncation_analysis,
    print_complement_analysis,
    run_complement_sweep,
    diagonal_orbit_analysis,
)

__all__ = [
    "classify_alpha",
    "canonical_orbit",
    "build_orbit_map",
    "orbit_averaged_hessian",
    "symmetry_analysis",
    "print_symmetry_analysis",
    "run_symmetry_sweep",
    "check_symmetry_commutation",
    "complement_truncation_analysis",
    "print_complement_analysis",
    "run_complement_sweep",
    "diagonal_orbit_analysis",
]
