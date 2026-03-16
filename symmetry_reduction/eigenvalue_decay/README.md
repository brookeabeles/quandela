# Eigenvalue decay of \(H_{\log}\)

This folder implements the numerical and analytic work toward proving **exponential eigenvalue decay** of the QAOA Hessian \(H_{\log}(y)\): \(|\lambda_k| \leq M e^{-c' k}\), which implies a rigorous bound on the determinant-ratio truncation error and hence on the number of directions needed for \(\epsilon\)-accurate saddle-point integration (BM24).

## Contents

| File | Description |
|------|-------------|
| **eigenvalue_decay.py** | Phase 1: eigenvalue-by-index decay, fits (exponential / power-law / stretched-exponential), \(c'(p)\) vs \(p\), block decomposition heatmap; **block-wise** \(\max|\lambda^{(m)}|\) vs \(m\), fit \(M e^{-c_m m}\), \(c_m(p)\) vs \(p\) |
| **fourier_agreement.py** | Fourier expansion of the agreement function \(A_{\alpha s}\) on \(\{0,1\}^{2p+1}\); numerical verification |
| **bonami_beckner.py** | Comparison of eigenvalue decay: \(H_{\log}\) vs \(H_{\mathrm{unif}}\) vs \(H_\rho\) (noise-operator model with \(\rho = \Gamma/\sqrt{2}\)) |
| **determinant_bound.py** | Theoretical bound \(|R(k_0)-1| \leq (8M/c') e^{-c' k_0}\) and \(k_0(\epsilon)\); overlay on probe1 plots |
| **eigenvalue_decay_proof.md** | Analytic write-up: Fourier expansion, covariance, Bonami–Beckner, hypercontractivity, determinant-ratio bound |
| **run_eigenvalue_decay.py** | Driver: run all analyses and save figures under `symmetry_reduction/figures/eigenvalue_decay/` |

## Quick run

From the **repository root** (Quandela):

```bash
python -m symmetry_reduction.eigenvalue_decay.run_eigenvalue_decay
```

This will:

- Verify the Fourier expansion of the agreement function (Task 2.1)
- Plot \(\log|\lambda_k|\) vs \(k\) and \(c'(p)\) vs \(p\) (Tasks 1.1–1.2)
- Generate block-decomposition heatmaps (Task 1.3)
- Compare \(H_{\log}\) with the Bonami–Beckner–style model (Task 1.4)

Figures are written to `symmetry_reduction/figures/eigenvalue_decay/` (combined to reduce file count):

- **eigenvalue_decay_and_c_prime_y0.png** — log\|λ_k\| vs k (3 panels) + c'(p) vs p (y=0)
- **eigenvalue_decay_and_c_prime_saddle.png** — same layout at saddle
- **eigenvalue_block_decomposition_p3_gamma1.00_combined.png** — block heatmaps for y=0 and saddle side by side
- **eigenvalue_block_max_decay_y0.png**, **eigenvalue_block_max_decay_saddle.png** — block-wise \(\max|\lambda^{(m)}|\) vs \(m\) with exponential fit; \(c_m(p)\) vs \(p\)
- **bonami_comparison_all.png** — 2×3 grid of (p, γ) Bonami–Beckner comparisons

## Determinant-ratio bound overlay

If probe1 data exists (`symmetry_reduction/data/probe1_saddle_p{p}.npz`), you can plot the theoretical bound overlay:

```python
from symmetry_reduction.plotting import load_probe1
from symmetry_reduction.eigenvalue_decay.determinant_bound import plot_probe1_with_bound_overlay

probe1 = load_probe1(5)
if probe1:
    plot_probe1_with_bound_overlay(probe1, p=5)
```

This computes \(M\), \(c'\) from `eigenvalue_decay_analysis(p, gamma, at_saddle=True)` for each \(\gamma\) and overlays the curve \((8M/c') e^{-c' k_0}\).

## Conventions

- Same as parent package: \(\beta_j = \pi/4\), \(r=1\), uniform \(\gamma_j = \gamma\).
- \(H_{\log}\) is **complex symmetric**; use `numpy.linalg.eig` (not `eigh`). Eigenvalues sorted by \(|\lambda_k|\) descending.
- \(\Gamma = 2|\sin(\gamma/4)|\); effective noise parameter \(\rho = \Gamma/\sqrt{2}\); hypercontractive regime \(\gamma < \pi\).

## References

- BM24: Boulebnane & Montanaro, PRX Quantum 5, 030348 (2024)
- Instructions: `EIGENVALUE_DECAY_CURSOR_INSTRUCTIONS.md`
- Parent: `symmetry_reduction/README_hessian_spectral.md`, `STABLE_RANK_THEOREM.md`
