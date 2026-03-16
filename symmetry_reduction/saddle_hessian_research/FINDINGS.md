# Saddle-Point Hessian Research — Findings & Candidate Bounds

## Direction 1: Spectral tail vs truncation error

### Analytic starting point

- **Determinant-ratio**: \(R(k_0) = \prod_{j>k_0} \frac{1/2}{1/2 - \lambda_j}\) (eigenvalues \(\lambda_j\) of \(H_{\log}\) sorted by \(|\lambda_j|\) descending).
- **Log-ratio**: \(\log R(k_0) = \sum_{j>k_0} \bigl(-\log(1 - 2\lambda_j)\bigr)\). For \(|2\lambda_j| < 1\), \(-\log(1-2\lambda_j) \approx 2\lambda_j + O(\lambda_j^2)\).
- **Candidate bound**: \(|\log R(k_0)| \leq C \sum_{j>k_0} |\lambda_j|\), so the **nuclear tail** \(\sum_{j>k_0} |\lambda_j|\) is a candidate controller for truncation error.

### Numerical objectives

- Compare \(|R(k_0)-1|\) vs:
  - **Nuclear tail**: \(\sum_{j>k_0} |\lambda_j|\)
  - **Frobenius tail**: \(\sum_{j>k_0} |\lambda_j|^2\)
  - **Operator tail**: \(\max_{j>k_0} |\lambda_j|\)
- Identify which gives the tightest predictive relationship across \(p\) and \(\gamma\).
- Scaling plots: how tail quantities and error evolve with depth \(p\) and \(\gamma\).

### Script outputs

- `direction1_err_vs_tails_*.png`: scatter of \(|R(k_0)-1|\) vs each tail metric.
- `direction1_logR_vs_nuclear_tail.png`: \(|\log R(k_0)|\) vs nuclear tail (theory line).
- `direction1_scaling_gamma_p*.png`: error and tail at \(k_0 = k_{99}\) vs \(\gamma\).

---

## Direction 2: Blockwise structure in \((|\alpha|, |\alpha'|)\)

### Goals

- Hessian blocks by subset sizes \((m, m') = (|\alpha|, |\alpha'|)\): Frobenius and operator norm per block.
- Heatmaps of spectral energy; track shift of dominant blocks with \(\gamma\).
- Compare **normal** truncation (keep \(|\alpha| \leq m\)) vs **complement** truncation (keep \(|\alpha| \geq n-m\)): Frobenius energy captured and determinant-ratio error of the restricted matrix.

### Target theorem (long-term)

- Blockwise decay: \(\|H_{\mathrm{block}}(m,m')\| \leq g(m, m', \gamma, p)\) with spectral mass concentrating along a ridge in subset-size coordinates.

### Script outputs

- `direction2_block_heatmap_*.png`: fractional energy \(f_{m,m'}\) and \(\log_{10}\) block operator norm.
- `direction2_truncation_*.png`: energy captured and \(|R(0)-1|\) for normal vs complement by \(m\).

---

## Combined objective

- Determine which spectral tail quantity controls saddle truncation error.
- Explain why the Hessian effectively behaves as low-dimensional.
- Characterize how spectral structure evolves with QAOA depth \(p\) and angle \(\gamma\).
