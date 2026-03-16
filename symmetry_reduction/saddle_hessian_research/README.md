# QAOA Saddle-Point Hessian Research

Extension of the PRX Quantum saddle-point analysis for QAOA on random k-SAT. Goals:

1. **Direction 1**: Rigorous link between saddle integral truncation error and spectral tail quantities of the Hessian.
2. **Direction 2**: Blockwise spectral structure in subset-size coordinates (|α|, |α'|) and normal vs complement truncation.

Findings and figures live in `figures/`. Scripts import from the parent `symmetry_reduction` package.

## Direction 1 — Spectral tail vs truncation error

- **Goal**: Bound |R(k₀) − 1| by a function of tail eigenvalues λ_j (j > k₀).
- **Quantities**: R(k₀) = ∏_{j>k₀} (½)/(½ − λ_j); log R(k₀) = Σ_{j>k₀} −log(1 − 2λ_j).
- **Candidate controllers**: nuclear tail Σ_{j>k₀}|λ_j|, Frobenius tail Σ|λ_j|², operator tail max_{j>k₀}|λ_j|.
- **Script**: `direction1_spectral_tail.py` — computes and plots |R(k₀)−1| vs each tail metric; scaling over p and γ.

## Direction 2 — Blockwise structure (|α|, |α'|)

- **Goal**: Describe spectral mass in blocks (m, m') = (|α|, |α'|); compare normal (|α| ≤ m) vs complement (|α| ≥ n−m) truncation.
- **Outputs**: Block Frobenius/operator norms, heatmaps, truncation error and energy capture per strategy.
- **Script**: `direction2_block_structure.py` — block norms, heatmaps, truncation experiments, decay fits.

## Running

From **repo root** (e.g. `Quandela/`):

```bash
python -m symmetry_reduction.saddle_hessian_research.direction1_spectral_tail   # Direction 1
python -m symmetry_reduction.saddle_hessian_research.direction2_block_structure   # Direction 2
```

**Output:** By default each script generates **a few summary figures** (3 for Direction 1, 2 for Direction 2). Use `--all` to also generate per-p or per-(p,γ) figures.

CLI options:
- **Direction 1**: `--p 2 3 4 5 --gamma 0.3 1.0 3.14 6.28`, `--single` (one (p,γ)), `--no-plot`, `--all` (add 4 per-p scaling plots).
- **Direction 2**: `--p 2 3 4 --gamma 0.3 1.0`, `--all` (add per-(p,γ) heatmaps and truncation plots).

See **FINDINGS.md** for the candidate analytic bound (nuclear tail controls \(|\log R(k_0)|\)) and target theorem for blockwise decay.
