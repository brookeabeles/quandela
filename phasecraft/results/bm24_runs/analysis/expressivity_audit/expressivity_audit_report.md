# LR-QAOA expressivity evidence audit (no new asymptotic claims)

## 1. Structural restriction

LR-QAOA uses 2 parameters `(Δγ, Δβ)` expanded via a linear ramp (`make_lr_angles`).
Full regular QAOA uses `2p` independent layer angles.

- At `p=5` (`cpu-prx-small.json`), LS projection residual `||θ_full − θ_LR|| / ||θ_full|| = 0.272`.
- Optimized full-QAOA β schedule is visibly non-linear; it cannot be represented exactly on the LR manifold.

## 2. Projection loss (same held-out instances)

Projecting full-QAOA angles onto the nearest LR schedule and re-evaluating:

- `p=5`, `n=12`, `N=20`: mean success drops by `64.1%`; median runtime worsens by `207.0%`.
- All qaoa-reg comparisons use `n ≤ 13` only → **finite-size matched benchmarks**, not scaling estimates.

## 3. Objective gap (Figure 3, LR only)

`Δ_objective = c_rt^LR − c_sp^LR` from `*-ctyp-cann-cache.json`, seeds {0,27,42}, n=12–18.

- At `p=100`: mean Δ_objective ≈ `0.098` (median-runtime exponent exceeds inverse-mean exponent).

## 4. Benchmark gap (not expressivity)

`Δ_benchmark = c_sp^LR − 0.69 p^{-0.32}` compares LR to the BM24 analytic reference.
This is **not** a controlled full-QAOA comparison.

- At `p=100`: mean Δ_benchmark ≈ `0.059`.

## 5. Optimizer / finite-size limitations

- No production-scale full-QAOA depth sweep exists (`train_size=100`, `n=12–18`, depths 2–100).
- `c_inf` values in `objective_exp_convergence.json` are **fitted offsets** (`c_inf + A p^{-β}`),
  distinct from per-depth values on the thesis Figure 3.
- N500 bootstrap CIs use **frozen** angles from `angles_grid_existing.json`, not the multi-seed training runs.

## 6. What is *not* demonstrated

- A production-scale expressivity gap vs optimized full QAOA.
- Lift-and-release refinement from LR init.
- Gradient orthogonality `R_⊥`.
