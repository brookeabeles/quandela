"""
Fourier expansion of the agreement function (Phase 2, Task 2.1).

f_α(s) = A_αs = (1/2) · 1[∀ j,j' ∈ α: s_j = s_{j'}]
on the Boolean hypercube {0,1}^{2p+1} with uniform measure.

Theory: χ_S(s) = (-1)^{∑_{j∈S} s_j}; f̂_α(S) = E_s[f_α(s) χ_S(s)].
- f̂_α(S) = 0 unless S ⊆ α
- f̂_α(S) = 2^{-|α|} if S ⊆ α and |S| is even; 0 if |S| odd

Numerical check: build A from core, compute Fourier coefficients of each row (as function of s), verify.
"""

from __future__ import annotations

import numpy as np

import sys
from pathlib import Path
if __name__ == "__main__":
    _root = Path(__file__).resolve().parent.parent.parent
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

from symmetry_reduction.core import build_structure_matrix, popcount


def parity_S(s: int, S: int, n: int) -> int:
    """χ_S(s) = (-1)^{∑_{j∈S} s_j}. S and s are bitmasks; n = number of bits."""
    val = 0
    for j in range(n):
        if (S >> j) & 1:
            val += (s >> j) & 1
    return 1 if (val % 2 == 0) else -1


def fourier_coefficient_uniform(f: np.ndarray, S: int, n: int) -> float:
    """f̂(S) = 2^{-n} ∑_s f(s) χ_S(s). f is length 2^n."""
    n_s = len(f)
    assert n_s == (1 << n)
    total = 0.0
    for s in range(n_s):
        total += f[s] * parity_S(s, S, n)
    return total / n_s


def fourier_coefficients_full(f: np.ndarray, n: int) -> np.ndarray:
    """Compute f̂(S) for all S ⊆ [n]. Returns array of length 2^n."""
    n_s = 1 << n
    assert len(f) == n_s
    # FFT-like: use that χ_S(s) = ∏_{j∈S} (-1)^{s_j} = ∏_j (1 if j∉S else (-1)^{s_j})
    # For Boolean functions, f̂(S) = 2^{-n} ∑_s f(s) (-1)^{s·S} (dot mod 2).
    # So f̂ = 2^{-n} * Walsh-Hadamard transform of f (with -1 convention).
    f = np.asarray(f, dtype=float)
    # Walsh-Hadamard: H_{s,S} = (-1)^{s·S}, f̂ = 2^{-n} H f.
    # Iterative WH transform:
    g = f.copy()
    for j in range(n):
        stride = 1 << j
        for start in range(0, n_s, 2 * stride):
            for i in range(stride):
                a = g[start + i]
                b = g[start + i + stride]
                g[start + i] = a + b
                g[start + i + stride] = a - b
    return g / n_s


def verify_agreement_fourier_expansion(p: int, alpha: int | None = None, tol: float = 1e-10) -> dict:
    """
    For the agreement function f_α(s) = A[α, s] (row α of structure matrix),
    compute Fourier coefficients and verify:
    - f̂_α(S) = 0 unless S ⊆ α
    - f̂_α(S) = 2^{-|α|} for S ⊆ α with |S| even; 0 for |S| odd

    If alpha is None, check one α per block size (one per |α|).
    Returns dict with 'passed', 'max_error', 'details' per α.
    """
    A = build_structure_matrix(p)
    n = 2 * p + 1
    d = 1 << n
    m_alpha = popcount(alpha) if alpha is not None else None

    if alpha is not None:
        alphas_to_check = [alpha]
    else:
        # one representative per block size
        alphas_to_check = []
        for m in range(n + 1):
            # first α with |α| = m: α = (1<<m) - 1 for first m bits
            if m == 0:
                alphas_to_check.append(0)
            else:
                alphas_to_check.append((1 << m) - 1)

    details = []
    max_err = 0.0
    for alpha in alphas_to_check:
        f = A[alpha, :]  # (n_s,)
        f_hat = fourier_coefficients_full(f, n)
        m = popcount(alpha)
        error_sup = 0.0
        for S in range(d):
            # S ⊆ α iff (S | α) == α, i.e. (S & ~alpha) == 0
            if (S | alpha) != alpha:
                # S ⊈ α: expect f̂(S) = 0
                err = abs(f_hat[S])
                error_sup = max(error_sup, err)
            else:
                # S ⊆ α. For α=∅ (m=0), f is constant 1/2 so f̂(∅)=1/2
                if m == 0:
                    expected = 0.5 if S == 0 else 0.0
                else:
                    expected = (2.0 ** (-m)) if (popcount(S) % 2 == 0) else 0.0
                err = abs(f_hat[S] - expected)
                error_sup = max(error_sup, err)
        details.append({"alpha": alpha, "|alpha|": m, "max_error": error_sup})
        max_err = max(max_err, error_sup)

    passed = max_err < tol
    return {
        "passed": passed,
        "max_error": max_err,
        "tol": tol,
        "p": p,
        "n": n,
        "details": details,
    }


def run_verification(p_values: list[int] = (2, 3)) -> None:
    """Run Fourier expansion verification for a few p and print summary."""
    for p in p_values:
        out = verify_agreement_fourier_expansion(p, alpha=None)
        print("Fourier agreement verification p=%d: passed=%s max_error=%.2e" % (p, out["passed"], out["max_error"]))
        for d in out["details"]:
            print("  |α|=%d alpha=%d max_error=%.2e" % (d["|alpha|"], d["alpha"], d["max_error"]))


if __name__ == "__main__":
    run_verification([2, 3])
