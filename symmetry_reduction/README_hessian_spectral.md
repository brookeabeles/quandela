# QAOA Hessian Spectral Concentration — Numerical Suite

This folder implements the numerical probe of the QAOA Hessian perturbation \(H_{\log}(y)\) from the instructions (based on BM24: Boulebnane & Montanaro, PRX Quantum 5, 030348 (2024)) and the extended proof.

- **STABLE_RANK_THEOREM.md** — Theorem: \(r_s \leq C\) (independent of \(p\), \(\gamma\)) for \(H_{\log}(0)\); proof strategy via operator/Frobenius norms and entry-wise bounds.

## Structure

- **core.py** — Index sets \(S\), \(\mathcal{A}\); structure matrix \(A_{\alpha s}\); mixing coefficients \(b_s\); phase coefficients \(c_\alpha\); \(H_{\log}(y) = \text{diag}(\sqrt{c})\,\text{Cov}_{w(y)}(A)\,\text{diag}(\sqrt{c})\).
- **saddle.py** — Fixed-point iteration (BM24 Algorithm 2) for the true saddle \(y_0^*(\gamma)\).
- **spectral.py** — SVD of \(H_{\log}\), Frobenius/spectral norms, stable rank, effective dimensions \(k_{90}, k_{95}, k_{99}, k_{99.9}\); tail mass, spectral tail, and `approximation_error_proxies` for the k_99 vs stable-rank investigation.
- **mechanisms.py** — Frobenius energy by subset size \(E_{m,m'}\), mechanism isolation (phase, agreement, variance).
- **diagnostics.py** — Softmax weight diagnostics at saddle (κ, IPR, entropy).
- **validation.py** — Sanity checks: \(\sum_s b_s=1\), rank(A), rank(Cov), saddle residual, etc.
- **plotting.py** — All plots (k_99 vs γ, stable rank, heatmaps, mechanism decomposition, etc.).
- **tables.py** — Formatted summary tables.
- **run_sweep.py** — Main driver: sweep over \((p,\gamma)\), save npz, optional validation.
- **determinant_ratio_decay.py** — Fits power-law/exponential decay of \(|R(k_0)| - 1\) vs \(k_0\); reports rate and \(k_0\) for 1% accuracy per \((\gamma, p)\).

## Usage

From repo root:

```bash
# Run sweep for p=1..5 (discrete γ), save to symmetry_reduction/data/
python -m symmetry_reduction.run_sweep --p-max 5

# Dense γ sweep (80 points) for plots
python -m symmetry_reduction.run_sweep --p-max 5 --dense

# Run validation checks once per p
python -m symmetry_reduction.run_sweep --p-max 3 --validate

# Skip saddle (y=0 only)
python -m symmetry_reduction.run_sweep --p-max 5 --no-saddle
```

Generate plots from saved data:

```python
from symmetry_reduction.plotting import generate_all_plots_from_saved
generate_all_plots_from_saved(p_max=5)
```

Print summary tables:

```python
from symmetry_reduction.tables import print_all_tables
print_all_tables(p_max=5)
```

## Output

- **data/spectral_data_p{p}_y0.npz** — At \(y=0\): `gamma_values`, `singular_values`, `k90`, `k95`, `k99`, `k999`, `stable_rank`, `frobenius_norm`, `spectral_norm`, `energy_by_size`.
- **data/spectral_data_p{p}_saddle.npz** — Same plus `saddle_converged`, `saddle_residual`, `kappa`, `ipr`, `entropy`.
- **figures/** — PNGs from `plotting.generate_all_plots_from_saved()`, including `approximation_error_metric_investigation_p{p}_y0.png` and `_saddle.png`.

## Approximation error metric investigation (k_99 vs stable rank)

A central question (Direction 1 in the QAOA Hessian spectral findings): does the saddle-point integral’s accuracy depend on **k_99** (number of directions for 99% Frobenius capture) or on **stable rank** (ℓ²-weighted effective dimension)? Error may scale like (A) \(\sum_{j>k} \sigma_j^2\) (tail mass → favors stable rank) or (B) \(\max_{j>k} \sigma_j\) (spectral tail → favors k_99). The suite computes:

- **Tail mass** \(1 - E(k)\): with \(k = k_{99}\) this is ~1% by definition; with \(k = \lceil r_s \rceil\) it can be large (e.g. 50–90%), so using “stable rank” dimensions does *not* fix Frobenius truncation error.
- **Spectral tail** \(\sigma_{k+1}/\sigma_1\): first dropped singular value for \(k = k_{99}\) vs \(k = \lceil r_s \rceil\). If the integral error scales like the spectral tail (e.g. inverse approximation), then the plot shows which truncation gives the smaller tail.

Use `plot_approximation_error_metric_investigation(data, p, at_saddle)` (or generate all plots) to produce the comparison figures. Interpretation: if spectral tail at \(\lceil r_s \rceil\) is already tiny everywhere, stable rank may still control practical accuracy; if tail mass at \(\lceil r_s \rceil\) is large, k_99 is needed for Frobenius-based error control.

## Probe 1: Saddle — determinant-ratio (does stable rank or k_99 control integral error?)

Does truncating the saddle-point Gaussian integral at \(\lceil r_s \rceil\) directions give \(|R - 1| \ll 1\) (stable rank wins), or do we need \(k_{99}\) directions (k_99 wins)? Probe 1 computes the **determinant ratio** at the saddle: eigenvalues \(\lambda_k\) of \(H_{\log}\) (via `eig`, since \(H_{\log}\) is complex symmetric), sorted by \(|\lambda_k|\) descending, non-null only. Then \(R(k_0) = \prod_{k>k_0} \tfrac{1/2}{1/2 - \lambda_k}\) (ratio of truncated to full Gaussian integral when treating directions beyond \(k_0\) as free). The plot shows \(|R(k_0)| - 1\) vs \(k_0\), with markers at \(k_0 = \lceil r_s \rceil\) (square) and \(k_0 = k_{99}\) (circle). If the square sits near 0 and the circle is redundant, stable rank controls the error; if the square is large and the circle is near 0, you need k_99.

**Run the probe** (saddle + eig per γ):

```bash
python -m symmetry_reduction.run_sweep --probe1
```

This writes **data/probe1_saddle_p5.npz**. **Plot from saved data:**

```python
from symmetry_reduction.plotting import load_probe1, plot_probe1_saddle_determinant_ratio
plot_probe1_saddle_determinant_ratio(load_probe1(5))
```

Figure **figures/probe1_saddle_determinant_ratio_p5.png**: \(|R(k_0)| - 1\) vs \(k_0\) for each \(\gamma\); square = \(\lceil r_s \rceil\), circle = \(k_{99}\). Interpret by whether the square or the circle lies near zero.

### Decay rate and “how many directions for 1% accuracy?”

The determinant-ratio error decays roughly as a power law or exponential in \(k_0\); the rate depends on \(\gamma\) and \(p\). The module **determinant_ratio_decay** fits \(|R(k_0)| - 1\) vs \(k_0\) over a sensible range and reports:

- **Power-law**: \(|R(k_0)| - 1 \approx A\,k_0^{-\alpha}\). Rate \(\alpha\) depends on \(\gamma\) and \(p\).
- **Exponential**: \(|R(k_0)| - 1 \approx A\,e^{-c k_0}\). Rate \(c\) depends on \(\gamma\) and \(p\).

The better fit (by R²) is chosen per \((\gamma, p)\). From the fit we estimate **\(k_0\) needed for 1% accuracy** (i.e. \(|R(k_0)| - 1 \leq 0.01\)).

```bash
python -m symmetry_reduction.determinant_ratio_decay
```

This prints a table of decay rates and \(k_0(1\%)\) per \(\gamma\) and \(p\), and saves **figures/saddle-pt-integral-error/probe1_decay_rate_vs_gamma.png** (rate vs \(\gamma\), and \(k_0\) for 1% vs \(\gamma\)). So the practical question — *"how many effective dimensions does the BM24 saddle-point method need for 1% accuracy?"* — is answered numerically as a function of \(\gamma\) and \(p\).

### Rigorous \(e^{-c k}\) bound: what would be needed

Proving \(|R(k_0)| - 1 \leq C e^{-c k_0}\) for some \(c = c(\gamma) > 0\) would give a **rigorous** bound on the number of directions needed for a given \(\epsilon\): \(k_0 \geq c^{-1}\bigl(\ln(C/\epsilon)\bigr)\) implies \(|R(k_0)| - 1 \leq \epsilon\).

Recall \(R(k_0) = \prod_{k > k_0} \frac{1/2}{1/2 - \lambda_k}\), where \(\lambda_k\) are eigenvalues of \(H_{\log}\) (saddle Hessian of the log-integrand), ordered by \(|\lambda_k|\) descending. So \(R(k_0) \to 1\) as \(k_0\) increases iff the tail product tends to 1.

**Sufficient condition for exponential decay of \(|R(k_0)| - 1\):** If \(|\lambda_k| \leq M e^{-c' k}\) for all \(k\) and some \(c' = c'(\gamma) > 0\), then each factor \(\frac{1/2}{1/2 - \lambda_k} = 1 + O(|\lambda_k|)\), so \(\log R(k_0) = O\bigl(\sum_{k > k_0} |\lambda_k|\bigr) = O(e^{-c' k_0})\). Hence \(|R(k_0)| - 1 \leq C e^{-c k_0}\) for some \(c\) proportional to \(c'\).

So a **rigorous** \(e^{-c k}\) bound reduces to: **eigenvalue decay of \(H_{\log}\)** — prove that (in an appropriate ordering) \(|\lambda_k| \leq M e^{-c' k}\) with \(c' = c'(\gamma) > 0\) (and ideally \(M, c'\) bounded as \(p\) grows). That would require controlling the spectrum of \(H_{\log}\) at the saddle (e.g. via structure of the covariance matrix and the \(c_\alpha\) weights). The existing stable-rank and spectral-concentration results (see **STABLE_RANK_THEOREM.md**) give Frobenius/operator-norm and effective-dimension bounds but do not yet give eigenvalue-by-index decay; that would be a natural next step to close the gap between the abstract spectral analysis and the concrete computational guarantee for the saddle-point method.

## Conventions

- \(\beta_j = \pi/4\), \(r=1\). Uniform \(\gamma_j = \gamma\).
- Subsets \(\alpha \subseteq \{0,\ldots,2p\}\) as bitmask in \([0, 2^{2p+1})\); configurations \(s \in \{0,1\}^{2p+1}\) as bitmask.
- \(H_{\log}\) is complex symmetric (use SVD, not eigh). Principal branch for \(\sqrt{c_\alpha}\).
