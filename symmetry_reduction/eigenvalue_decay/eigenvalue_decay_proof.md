# Eigenvalue decay of \(H_{\log}\): toward a rigorous bound

## Goal

Prove (or build toward proving) that the eigenvalues of the QAOA Hessian \(H_{\log}(y)\), ordered by magnitude, satisfy
\[
|\lambda_k| \leq M e^{-c' k}
\]
for constants \(M, c' > 0\) depending on \(\gamma\) (and ideally bounded as \(p \to \infty\)). This would close the gap between the existing spectral concentration numerics and a rigorous computational guarantee for the BM24 saddle-point method.

---

## 1. Fourier expansion of the agreement function

### 1.1 Definition

\(A_{\alpha s} = \frac{1}{2} \mathbf{1}[\forall j,j' \in \alpha: s_j = s_{j'}]\) is the agreement indicator on the Boolean hypercube \(\{0,1\}^{2p+1}\). With the uniform measure, the Fourier basis is \(\chi_S(s) = (-1)^{\sum_{j \in S} s_j}\).

### 1.2 Fourier coefficients

For \(f_\alpha(s) = A_{\alpha s}\):
- \(\hat{f}_\alpha(S) = 0\) unless \(S \subseteq \alpha\).
- For \(S \subseteq \alpha\): \(\hat{f}_\alpha(S) = 2^{-|\alpha|}\) if \(|S|\) is even; \(\hat{f}_\alpha(S) = 0\) if \(|S|\) is odd.

**Check:** \(\mathbb{E}[f_\alpha] = 2^{-|\alpha|}\), so \(\hat{f}_\alpha(\emptyset) = 2^{-|\alpha|}\). For \(S \subseteq \alpha\), \(|S| \geq 1\), conditioning on “all bits in \(\alpha\) agree”: they are either all 0 or all 1; \(\chi_S = (-1)^{|S|}\) in the all-1 case and \(1\) in the all-0 case, so \(\mathbb{E}[f_\alpha \chi_S] = \frac{1}{2} \cdot 2^{-|\alpha|}(1 + (-1)^{|S|})\), giving \(2^{-|\alpha|}\) for \(|S|\) even and 0 for \(|S|\) odd.

**Numerical verification:** implemented in `fourier_agreement.verify_agreement_fourier_expansion()`.

---

## 2. Covariance in Fourier basis

\[
[\operatorname{Cov}_w(A)]_{\alpha \alpha'} = \sum_s w_s f_\alpha(s) f_{\alpha'}(s) - \left(\sum_s w_s f_\alpha(s)\right)\left(\sum_s w_s f_{\alpha'}(s)\right).
\]

At \(y = 0\) with \(w_s = b_s / Z\), this is the covariance of two agreement functions under the mixing distribution. In the uniform case (\(w_s = 2^{-n}\)):
\[
\operatorname{Cov}_{\mathrm{unif}}(f_\alpha, f_{\alpha'}) = \sum_{S \neq \emptyset,\, S \subseteq \alpha \cap \alpha',\, |S| \text{ even}} 2^{-|\alpha|} \cdot 2^{-|\alpha'|}.
\]

After weighting by \(\sqrt{c_\alpha}\sqrt{c_{\alpha'}}\):
\[
[H_{\log}]_{\alpha \alpha'} = \sqrt{c_\alpha c_{\alpha'}} \cdot \operatorname{Cov}_w(f_\alpha, f_{\alpha'}).
\]

If \(|c_\alpha| \lesssim \Gamma^{|\alpha|}\) with \(\Gamma = 2|\sin(\gamma/4)|\), the effective weight on block \((m, m')\) is \(\Gamma^{(m+m')/2} \cdot 2^{-(m+m')} \cdot (\text{covariance factor})\).

---

## 3. Bonami–Beckner and hypercontractivity

**Noise operator:** \(T_\rho\) on functions \(f : \{0,1\}^n \to \mathbb{C}\) satisfies \(\widehat{T_\rho f}(S) = \rho^{|S|} \hat{f}(S)\).

**Noise-weighted covariance (idealized):**
\[
\operatorname{Cov}^{(\rho)}(f_\alpha, f_{\alpha'}) = \sum_{\substack{S \subseteq \alpha \cap \alpha',\\ |S| \geq 2,\, |S| \text{ even}}} \rho^{|S|} \cdot 2^{-|\alpha|} \cdot 2^{-|\alpha'|}.
\]

**Effective noise parameter:** \(\rho = \Gamma/\sqrt{2}\). The hypercontractive regime is \(\rho < 1\), i.e. \(\Gamma < \sqrt{2}\), i.e. \(\gamma < \pi\). For \(\gamma < \pi\), the Bonami–Beckner machinery can be used to control higher-level contributions and hence eigenvalue decay.

**Strategy:** If \(H_{\log}\) is well-approximated by \(D T D\) with \(D = \operatorname{diag}(\sqrt{c})\) and \(T\) close to a noise operator with \(\rho = \Gamma/\sqrt{2}\), then the eigenvalue structure inherits Fourier-level decay \(\rho^{|S|}\), which (after counting multiplicities \(\binom{n}{|S|}\)) yields exponential decay in the eigenvalue index \(k\). **Numerical test:** `bonami_beckner.bonami_beckner_comparison()` compares eigenvalue decay of \(H_{\log}\), \(H_{\mathrm{unif}}\), and \(H_\rho\).

### Block-Wise Spectral Bounding

Instead of bounding eigenvalues by the global index \(k\), we bound the spectral norm of the \(m\)-th Fourier block: let \(\lambda^{(m)}\) denote the eigenvalues whose dominant (argmax) block energy is at level \(m\). Then we ask whether
\[
\max |\lambda^{(m)}| \leq \mathcal{O}(\rho^m)
\]
for some \(\rho \in (0,1)\) (e.g. \(\rho = \Gamma/\sqrt{2}\)). This block-wise bound is stable under scaling of \(p\) because it does not mix with the binomial multiplicity \(\binom{2p+1}{m}\) of the block; the global index \(k\) grows with \(p\) due to that multiplicity, which is why the global exponential fit \(|\lambda_k| \leq M e^{-c' k}\) degrades at large \(p\). **Numerical test:** `eigenvalue_decay.block_max_eigenvalue_analysis()` and `plot_block_max_decay()` extract \(\max|\lambda^{(m)}|\) per block and fit \(M e^{-c_m m}\); the rate \(c_m(p)\) can then be checked for stability as \(p\) increases.

The determinant ratio bound must be modified when using block-wise bounds: the tail contribution is summed over block levels \(m\), with each level contributing at most \(\binom{2p+1}{m}\) eigenvalues of magnitude at most \(\mathcal{O}(\rho^m)\). So the tail sum is controlled by \(\sum_{m \geq m_0} \binom{2p+1}{m} \rho^m\), which can be bounded via the binomial theorem and decay in \(m\).

---

## 4. Determinant ratio bound (Phase 3)

If \(|\lambda_k| \leq M e^{-c' k}\) for \(k \geq 1\), then
\[
\log R(k_0) = \sum_{k > k_0} \log \frac{1/2}{1/2 - \lambda_k} = \sum_{k > k_0} \log \frac{1}{1 - 2\lambda_k}.
\]
For \(|2\lambda_k| < 1\) (large \(k\)):
\[
|\log R(k_0)| \leq \sum_{k > k_0} \frac{2M e^{-c'k}}{1 - 2M e^{-c'k}}.
\]
For \(k_0\) such that \(2M e^{-c' k_0} < 1/2\):
\[
|\log R(k_0)| \leq \frac{4M e^{-c'(k_0+1)}}{1 - e^{-c'}} \leq \frac{4M}{c'} e^{-c' k_0}.
\]
Hence \(|R(k_0) - 1| \leq \frac{8M}{c'} e^{-c' k_0}\) for small \(|\log R|\).

**\(k_0(\epsilon)\):** To have \(|R(k_0) - 1| \leq \epsilon\),
\[
k_0 \geq \frac{1}{c'}\left(\log \frac{8M}{c'} + \log \frac{1}{\epsilon}\right) = O(\log(1/\epsilon)).
\]
Implemented in `determinant_bound.determinant_ratio_bound_from_decay()` and `determinant_bound.k0_for_epsilon()`.

---

## 5. Non-uniform weights (saddle)

At the saddle \(y^*\), \(w(y^*)\) is not uniform. Write \(w(y^*) = w(0)(1 + \delta)\). If the softmax condition number \(\kappa\) is bounded, \(\|\delta\|\) is bounded and the eigenvalue decay for \(w(0)\) transfers to \(w(y^*)\) with modified constants. Numerically, \(r_s\) is often smaller at the saddle (better concentration), so saddle weights can help rather than hurt.

---

## 6. Regime \(\gamma > \pi\)

For \(\gamma > \pi\), \(\rho = \Gamma/\sqrt{2} > 1\) and the noise-operator interpretation breaks down. Concentration at the saddle (low \(r_s\)) suggests a different mechanism: highly peaked \(w(y^*)\) effectively projects onto a low-dimensional subspace, so rank of the covariance (and number of significant eigenvalues) is limited by the number of effectively non-zero weights. A perturbation argument around the saddle may extend the bound to this regime.

---

## References

- BM24: Boulebnane & Montanaro, PRX Quantum 5, 030348 (2024)
- O’Donnell, *Analysis of Boolean Functions*, Cambridge (2014), Ch. 2 (Fourier), Ch. 9 (hypercontractivity)
- Existing code: `symmetry_reduction/core.py`, `spectral.py`, `mechanisms.py`, `STABLE_RANK_THEOREM.md`
