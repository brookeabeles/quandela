#!/usr/bin/env python3
"""
Diagnostic: inspect b_s and w = weights_at_zero(b_s) for complex/real behavior.
Run from repo root: python -m symmetry_reduction.diagnostic_weights
"""
import numpy as np
from symmetry_reduction.core import (
    build_structure_matrix,
    compute_b_s,
    compute_c_alpha,
    weights_at_zero,
    popcount,
    DEFAULT_BETA,
)

p = 5
betas = np.full(p, DEFAULT_BETA)
gamma = 0.30
gammas = np.full(p, gamma)

A = build_structure_matrix(p)
b_s = compute_b_s(p, betas)
c_alpha = compute_c_alpha(p, gammas)
w = weights_at_zero(b_s)

print(f"b_s: dtype={b_s.dtype}, sum={np.sum(b_s):.6f}")
print(f"  max |imag(b_s)| = {np.max(np.abs(b_s.imag)):.2e}")
print(f"  max |real(b_s)| = {np.max(np.abs(b_s.real)):.2e}")
print(f"  any negative real? {np.any(b_s.real < -1e-15)}")
print()
print(f"w=b_s/sum(b_s): dtype={w.dtype}, sum={np.sum(w):.6f}")
print(f"  max |imag(w)| = {np.max(np.abs(w.imag)):.2e}")
print(f"  min real(w) = {np.min(w.real):.6e}")
print()

# E_w[A_alpha] by subset size
d = A.shape[0]
E_A = w @ A.T  # complex

from collections import defaultdict
by_size = defaultdict(list)
for alpha in range(d):
    m = popcount(alpha)
    if m >= 2:
        by_size[m].append(E_A[alpha])

print(f"{'|α|':>4} | {'mean Re(E_A)':>14} | {'mean Im(E_A)':>14} | {'mean |E_A|':>14} | {'mean 2*Re(E_A)':>14} | {'count':>5}")
print("-" * 80)
for m in sorted(by_size.keys()):
    vals = np.array(by_size[m])
    print(f"{m:>4} | {np.mean(vals.real):>14.6e} | {np.mean(vals.imag):>14.6e} | "
          f"{np.mean(np.abs(vals)):>14.6e} | {2*np.mean(vals.real):>14.6e} | {len(vals):>5}")
