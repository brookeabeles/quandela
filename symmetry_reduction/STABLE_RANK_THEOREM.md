# Stable rank universality theorem (Direction 1)

## Theorem (conjecture with numerical support)

**Statement.** For the QAOA Hessian log-term \(H_{\log}(y) = \operatorname{diag}(\sqrt{c})\,\operatorname{Cov}_{w(y)}(A)\,\operatorname{diag}(\sqrt{c})\) (BM24, \(d = 2^{2p+1}\), uniform \(\gamma\)):

- **At \(y=0\):** The stable rank of \(H_{\log}(0)\) satisfies
  \[
  r_s = \frac{\|H_{\log}\|_F^2}{\|H_{\log}\|^2} \leq C
  \]
  for a constant \(C\) **independent of \(p\)** and of \(\gamma\).

- **Numerical evidence:** Over the full range of \(p\) and \(\gamma\) tested, \(r_s \leq 5.3\).

So the effective dimension in the sense of stable rank is **bounded uniformly** in system size and angle.

---

## Proof strategy

### 1. Weighted covariance at \(y=0\)

At \(y=0\), \(w(0)\) is proportional to \(b_s\), so \(\operatorname{Cov}_{w(0)}(A)\) is the covariance of the structure matrix \(A\) under the mixing distribution. The Hessian is
\[
H_{\log}(0) = \operatorname{diag}(\sqrt{c})\,\operatorname{Cov}_{w(0)}(A)\,\operatorname{diag}(\sqrt{c}).
\]

**Goal:** Show that after the \(\sqrt{c}\) weighting, both the **operator norm** and the **Frobenius norm** of \(H_{\log}(0)\) are \(O(1)\) (or at least grow at the **same rate** in \(p\)), so that \(r_s = \|H_{\log}\|_F^2/\|H_{\log}\|^2\) stays bounded.

- **Operator norm:** \(\|H_{\log}\|\) is at most (spectral norm of \(A\) under the weight) times a concentration ratio. Bounding the operator norm of \(\operatorname{Cov}_{w(0)}(A)\) and the effect of \(\operatorname{diag}(\sqrt{c})\) yields \(\|H_{\log}\| = O(1)\) (or the same scaling as \(\|H_{\log}\|_F\)).

- **Frobenius norm squared:**
  \[
  \|H_{\log}\|_F^2 = \sum_{\alpha,\alpha'} \bigl|[H_{\log}]_{\alpha\alpha'}\bigr|^2.
  \]
  Although there are \(2^{4p+2}\) terms, **entry-wise bounds** can make this sum \(O(1)\) or grow like the same function of \(p\) as \(\|H_{\log}\|^2\).

### 2. Entry-wise bounds (implemented in the codebase)

From BM24 and the suite:

- **Phase:** \(\Gamma = 2|\sin(\gamma/4)|\), and \(|\sqrt{c_\alpha}| \leq \Gamma^{(|\alpha|-1)/2}\) (see `mechanisms.phase_bound_Gamma` and `analytical_bounds`).
- **Agreement / variance:** \(\operatorname{Var}_{w}(A_\alpha)\) and covariance entries are controlled by the “agreement” probability and decay with \(|\alpha|\), e.g. factors of the form \(2^{-(|\alpha|+|\alpha'|)/2}\) (or similar) from the structure of \(A_{\alpha s} \in \{0, 1/2\}\).

So
\[
\bigl|[H_{\log}]_{\alpha\alpha'}\bigr| \;\lesssim\; \Gamma^{(|\alpha|+|\alpha'|-2)/2} \cdot 2^{-(|\alpha|+|\alpha'|)/2} \cdot (\text{concentration factor}),
\]
i.e. a factor \(\Gamma^{|\alpha|+|\alpha'|} \cdot 2^{-(|\alpha|+|\alpha'|)/2}\) (up to normalization) in the bound. For \(\Gamma \leq 1\) this decays in \(|\alpha|+|\alpha'|\); for \(\Gamma > 1\) one needs the \(2^{-(|\alpha|+|\alpha'|)/2}\) to dominate or a more refined bound.

**Key point:** Summing \(\sum_{\alpha,\alpha'} |[H_{\log}]_{\alpha\alpha'}|^2\) over \(d^2 = 2^{4p+2}\) indices, each term carries a weight that decays in \(|\alpha|,|\alpha'|\). The number of pairs with given \((|\alpha|,|\alpha'|)\) is \(\binom{2p+1}{|\alpha|}\binom{2p+1}{|\alpha'|}\); the product of (bound)² and count can still converge or grow slowly enough that \(\|H_{\log}\|_F^2\) is \(O(1)\) or has the same scaling as \(\|H_{\log}\|^2\).

### 3. Why the ratio stays bounded

- **Stable rank:** \(r_s = \|H_{\log}\|_F^2 / \|H_{\log}\|^2\).
- If **both** \(\|H_{\log}\|_F^2\) and \(\|H_{\log}\|^2\) are \(O(1)\) in \(p\), then \(r_s = O(1)\).
- If both grow at the **same rate** in \(p\) (e.g. both \(\sim f(p)\)), then \(r_s = O(1)\) again.

So the proof reduces to:

1. **Frobenius:** Bound \(\|H_{\log}\|_F^2\) by summing the entry-wise bounds over \((\alpha,\alpha')\) and show it is \(O(1)\) or controlled.
2. **Spectral:** Show \(\|H_{\log}\|\) is dominated by the same scaling (e.g. the leading singular value comes from the same block structure and is of the same order as the Frobenius norm contribution from the “important” blocks).

This is **tractable random-matrix / structured-matrix analysis**: the matrix is not generic; it has a clear block structure in \((|\alpha|,|\alpha'|)\) and explicit decay, so \(\|H_{\log}\|\) and \(\|H_{\log}\|_F\) can be analyzed together.

---

## Connection to the codebase

| Concept | Location |
|--------|----------|
| \(H_{\log}(y)\), \(\operatorname{Cov}_{w(y)}(A)\) | `core.hessian_log_at_y`, `core.covariance_from_weights` |
| Stable rank \(r_s = \|H\|_F^2/\|H\|^2\) | `spectral.stable_rank`, `spectral.spectral_summary` |
| \(\Gamma\), phase bound \(\Gamma^{|\alpha|-1}\) | `mechanisms.phase_bound_Gamma`, `mechanisms.analytical_bounds` |
| Frobenius by block \(E_{m,m'}\) | `mechanisms.energy_by_block` |
| Numerical \(r_s\) over \((p,\gamma)\) | `run_sweep` → `spectral_data_p*_y0.npz` / `_saddle.npz`, plots `stable_rank_vs_gamma_*` |

---

## Contribution

A proof that \(r_s \leq C\) for all \(p\) and \(\gamma\) would be a **clean theorem** with real consequences:

- It justifies using a **fixed number of directions** (of order \(r_s\)) for the saddle-point approximation, independent of \(p\).
- It belongs to the **random matrix theory of structured covariance matrices** (covariance under a fixed distribution \(w(0)\), weighted by phase coefficients \(c\)).

The numerical evidence (\(r_s \leq 5.3\)) strongly suggests that such a constant \(C\) exists; the entry-wise and block structure in the codebase provide the right objects (norms, \(\Gamma\), agreement bounds) to turn this into a formal proof.
