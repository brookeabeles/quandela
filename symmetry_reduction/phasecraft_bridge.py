"""
Bridge utilities to run symmetry-reduction Hessian analysis with Phasecraft formulas.

This keeps the existing saddle/spectral machinery, but replaces b_s and c_alpha
with the definitions used in `phasecraft/exact_ksat.py`.
"""

from __future__ import annotations

import numpy as np
from phasecraft.optimal_angles import optimal_angles, optimal_angles_biased


def phasecraft_b_s(betas: np.ndarray) -> np.ndarray:
    """Phasecraft b_s definition: b = 0.5 * B(betas, s)."""
    p = len(betas)
    all_s = np.arange(2 ** (2 * p + 1))
    betas = np.asarray(betas, dtype=float)
    return 0.5 * _phasecraft_B(betas, all_s)


def _phasecraft_B(betas: np.ndarray, s: np.ndarray) -> np.ndarray:
    """
    Local copy of Phasecraft B(betas, s) from `phasecraft/generalized_binomial_sum.py`.
    Kept here to avoid package-import issues in that module.
    """
    p = len(betas)
    s = np.asarray(s)
    factors = []
    for j in range(p):
        eq_count = (
            (((s >> j) & 1) == ((s >> (j + 1)) & 1)).astype(int)
            + (((s >> (2 * p - j)) & 1) == ((s >> (2 * p - j - 1)) & 1)).astype(int)
        )
        neq_count = (
            (((s >> j) & 1) != ((s >> (j + 1)) & 1)).astype(int)
            + (((s >> (2 * p - j)) & 1) != ((s >> (2 * p - j - 1)) & 1)).astype(int)
        )
        factors.append(
            np.cos(betas[j] / 2.0) ** eq_count * (1j * np.sin(betas[j] / 2.0)) ** neq_count
        )
    sign = (-1) ** (((s & 1) != ((s >> p) & 1)))
    return sign * np.prod(factors, axis=0)


def phasecraft_c_alpha(gammas: np.ndarray, r: float) -> np.ndarray:
    """
    Phasecraft c_alpha definition used in generalized_binomial_sum_scaling_exponent_ksat.

    c_alpha = r * Π_j [prod_elts[j] if alpha_j=1 else 1]
    with prod_elts = [exp(-i gamma/2)-1, -1, exp(+i gamma_rev/2)-1] (BM24 Eq. A34).
    """
    gammas = np.asarray(gammas, dtype=float)
    p = len(gammas)
    all_alpha = np.arange(2 ** (2 * p + 1))
    prod_elts = np.concatenate(
        (np.exp(-0.5j * gammas) - 1.0, [(-1.0)], np.exp(0.5j * gammas[::-1]) - 1.0)
    )
    factors = [
        prod_elts[j] * ((all_alpha >> j) & 1) + 1.0 * ((~all_alpha >> j) & 1)
        for j in range(2 * p + 1)
    ]
    return r * np.prod(factors, axis=0)


def get_phasecraft_angles(
    k: int, r: float, p: int, *, biased: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Return (betas, gammas) from Phasecraft angle tables for (k, r, p)."""
    table = optimal_angles_biased if biased else optimal_angles
    key = (k, float(r))
    if key not in table or p not in table[key]:
        raise KeyError(f"No Phasecraft angles for (k={k}, r={r}, p={p}, biased={biased})")
    entry = table[key][p]
    return np.asarray(entry["betas"], dtype=float), np.asarray(entry["gammas"], dtype=float)
