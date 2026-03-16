"""
Eigenvalue decay analysis for H_log: numerical characterization and path to proof.

Goal: establish |λ_k| ≤ M exp(-c' k) for eigenvalues of H_log (ordered by magnitude)
to derive rigorous bounds on determinant-ratio truncation error.

Modules:
- eigenvalue_decay: eigenvalue-by-index decay, fits (exp/power/stretched), c'(p), block decomposition
- fourier_agreement: Fourier expansion of agreement function A_αs on Boolean hypercube
- bonami_beckner: comparison with noise-operator model
- determinant_bound: theoretical |R(k0)-1| bound from eigenvalue decay
"""

from .eigenvalue_decay import (
    eigenvalue_decay_analysis,
    decay_rate_vs_p,
    eigenvalue_block_decomposition,
)
from .determinant_bound import determinant_ratio_bound_from_decay, k0_for_epsilon

__all__ = [
    "eigenvalue_decay_analysis",
    "decay_rate_vs_p",
    "eigenvalue_block_decomposition",
    "determinant_ratio_bound_from_decay",
    "k0_for_epsilon",
]
