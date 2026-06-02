# 8-SAT Universality: Recent Implementation Changes (for Claude)

## High-level objective (what the code is now trying to produce)
The repo now supports the *universality via covariance decomposition* evidence set:

1. **Row/Layer 1 (q=3, small γ):** compute the exact q=3 saddle contribution to the Hessian as an existence/structure check in the regime where the q=3 solve is numerically stable.
2. **Row/Layer 2 (q=1, large γ):** compute the full q=1 Hessian blocks \(H_{\log}=\nabla^2 F\) across a broad γ range.
3. **Row/Layer 3 (universal part):** compute the **pure covariance** block energies of \(\mathrm{Cov}_w(A)\), intended to be (empirically) γ-independent for fixed \((p,\beta,r)\).

The plotting script now also produces a **9-panel grid** with:
- vertical: `p = 2, 3, 4`
- horizontal: `gamma = pi/4, pi, 2*pi`
using the **pure covariance** \( \mathrm{Cov}_w(A)\) block energy.

## Summary of the concrete code changes

### 1) New q>1 saddle solver: Newton in u-space (Newton + γ continuation)
**New file:** `symmetry_reduction/newton_saddle_q.py`

Key behavior:
- Solves the BM24 q>1 saddle fixed-point system (Eq. 20 framework) using a **Newton method in u-space**:
  - \(u_\alpha = \text{coeff}_\alpha \, y_\alpha\)
  - restricts to **active indices** \(|\alpha|\ge 2\) (phantom \(|\alpha|\le 1\) components are treated as constant / frozen)
- Uses **γ-homotopy continuation** (adaptive geometric steps + backtracking).
- Supports **warmstarts** (`u_init_active` and `gamma_start_override`) to accelerate successive γ values.
- Tracks `converged` based on whether the final residual meets a target threshold; the calling code also marks nonconverged panels in red.

### 2) Covariance decomposition helper added for universality evidence
**Modified file:** `symmetry_reduction/saddle_hessian_research/direction2_block_structure.py`

Added function:
- `compute_cov_block_structure(p, gamma, betas=None, gammas=None, q=1, r=1.0)`

What it computes (at the q=1 saddle):
- `f_mm_H`: fractional block energy for the **full q=1** Hessian \(H_{\log}=\mathrm{diag}(\sqrt c)\,\mathrm{Cov}\,\mathrm{diag}(\sqrt c)\)
- `f_mm_cov`: fractional block energy for the **pure** covariance \(\mathrm{Cov}_w(A)\)
- plus `converged` and `saddle_residual` diagnostics from the q=1 saddle solve.

This cleanly separates the “universal” covariance structure from the coefficient-weighting evolution.

### 3) Universality plotting script rewritten + updated γ/p grids
**Modified file:** `run_8sat_block_decay.py`

What it produces now:
1. A **layer plot** (still 3 rows) saved as:
   - `symmetry_reduction/figures/8sat_block_decay/8sat_universality_summary_layers.png`
2. A **requested 9-panel p×gamma grid** saved as:
   - `symmetry_reduction/figures/8sat_block_decay/8sat_universality_summary.png`

The 9-panel grid uses:
- vertical axis (rows): `p = 2, 3, 4`
- horizontal axis (columns): `gamma = pi/4, pi, 2*pi`
- shown quantity: **Row 3 / pure covariance** \( \mathrm{Cov}_w(A)\) fractional block energy.

Quick mode:
- `QUANDELA_QUICK=1` reduces the number of expensive computations (and applies fail-fast caps for hard points) so the script can run quickly for sanity checks.

## Small-γ limitation (important for interpreting the results)
Even with the Newton u-space solver + γ continuation, **large γ** values such as `pi/2`, `pi`, `2*pi` are currently **not reliably accessible** in the **q=3 r=176.54** saddle solve regime (residuals can remain above the convergence target for some p,γ).

Therefore:
- The **q=3** “exact” portion of the evidence set is computed at **small γ** only (e.g. `0.001`, `0.003/0.005/0.01`, depending on quick mode).
- The **universal covariance structure** is intended to be verified over the broader γ sweep using the q=1 saddle, via the new covariance decomposition helper.

## Files for Claude to use (necessary change files)
For Claude to reproduce/extend the universality evidence set, the minimal set of relevant files is:
1. `symmetry_reduction/newton_saddle_q.py`
2. `symmetry_reduction/saddle_hessian_research/direction2_block_structure.py` (includes `compute_cov_block_structure`)
3. `run_8sat_block_decay.py` (rewritten to generate the new figures + grids)

Optional context (only if Claude needs it):
4. `docs/NEW_CHAT_HANDOFF.md` (high-level narrative + solver motivation)

## How to run
From repo root:
```bash
cd /Users/b/Quandela
python -u run_8sat_block_decay.py
```

Quick validation:
```bash
QUANDELA_QUICK=1 python -u run_8sat_block_decay.py
```

