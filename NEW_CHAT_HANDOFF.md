# New chat handoff (8‑SAT q=3 saddle + block heatmaps)

## Goal
Build and validate an **8‑SAT (k=8, q=3, r≈176.54)** pipeline (BM24 PRX Quantum 5, 030348) that:
- solves the **q=3 saddle / fixed-point system** (BM24 Eq. 20 / Proposition 1 framework),
- computes a Hessian-based block energy heatmap in \((|\alpha|,|\alpha'|)\) coordinates,
- produces a **single 4×4 summary PNG** for \(p\in\{2,3,4,5\}\) and a γ sweep (see “How to run”),
- reports convergence and residuals per panel and visually flags nonconverged panels.

## Current status (high-signal)
### q=1
- The existing q=1 solver (`saddle_fixed_point` / `saddle_with_adaptive_damping`) generally converges across a wide γ range with damping, but is **not rigorously guaranteed** for all γ.

### q=3 (the important part)
- The original **fixed-point iteration** for q=3 at threshold \(r=176.54\) is numerically unstable: the map’s effective scale grows like \(r^{2q-1}=r^5\sim 10^{11}\), so direct iteration (even with damping, r-homotopy, etc.) cannot reliably reach large r at moderate/large γ.
- The current q=3 implementation therefore uses a **Newton solver in u-space** (see below), and the 8‑SAT summary figure should be treated as **verified only for panels** with `converged=True` and small `saddle_residual`.
- **Small‑γ limitation (important)**: for large γ values such as \(\pi/2, \pi, 2\pi\), the q=3 saddle at \(r=176.54\) is currently **numerically inaccessible with reliable residuals** using the present formulation/solver. Empirically, even γ≈0.005–0.01 can be borderline depending on p. This is why the default γ sweep is “small γ”.

## What the current q>1 Hessian is
For q>1 we compute the **Hessian of F**, i.e. \(\nabla^2 F\) where
\[
F(y)=\log\sum_s b_s \exp\Big(\sum_\alpha \text{coeff}_\alpha A_{\alpha s} y_\alpha\Big),
\quad
\text{coeff}_\alpha = r\cdot (-c_{\alpha,\mathrm{phase}})^{1/(2q)}.
\]
This is **not** the full action Hessian \(\nabla^2\Phi^\*\) from BM24 Eq. (15). This is documented in code.

## Key files (attach/reupload in new chat)
Minimum set to recreate context and reason correctly:
- `symmetry_reduction/core.py`
- `symmetry_reduction/saddle.py`
- `symmetry_reduction/saddle_hessian_research/direction2_block_structure.py`
- `symmetry_reduction/newton_saddle_q.py`  (**NEW**: q>1 Newton u-space saddle solver)
- `symmetry_reduction/mechanisms.py`
- `run_8sat_block_decay.py`
- Instructions provided by user:
  - `8SAT_HESSIAN_FIXES.md`
  - `Q3_CONVERGENCE_FIX.md`

Optional but helpful:
- `symmetry_reduction/figures/8sat_block_decay/8sat_block_heatmap_summary.png`
- `symmetry_reduction/saddle_hessian_research/figures/direction2_block_heatmap_summary.png`

## How to run (current)
From repo root:
```bash
cd /Users/b/Quandela
python -u run_8sat_block_decay.py
```
Output:
- `symmetry_reduction/figures/8sat_block_decay/8sat_block_heatmap_summary.png`

The script prints:
- `|coeff_α|` magnitude range/mean (diagnostic),
- `converged` and `saddle_residual`,
and draws a red border + “⚠” on nonconverged panels.

### Quick mode (recommended for sanity checks)
To avoid extremely slow/hard cases (notably p=5 and large γ), run:
```bash
QUANDELA_QUICK=1 python -u run_8sat_block_decay.py
```
In quick mode, the script reduces to p ∈ {2,3,4} and a reduced γ sweep (small γ only).

### γ sweep values
- Default (full): γ ∈ {0.001, 0.003, 0.005, 0.01}
- Quick: γ ∈ {0.001, 0.003, 0.005}

These are intentionally **small γ** because large γ like \(\pi/2,\pi,2\pi\) currently does not converge at \(r=176.54\) with small residual using the present solver/formulation.

## Changes made after receiving the instruction docs

### `symmetry_reduction/saddle_hessian_research/direction2_block_structure.py`
- Threaded residual through the stack:
  - `_get_saddle_hessian` returns `(H, converged, residual)`
  - `run_block_and_truncation` stores `saddle_residual`
- Added doc note: for q>1 returns **∇²F** not **∇²Φ\***
- q>1 branch was rewritten to use **Newton in u-space + adaptive γ-homotopy**:
  - calls `solve_8sat_saddle` from `symmetry_reduction/newton_saddle_q.py`
  - solves only the **active** subset indices |α|≥2 (|α|≤1 are “phantom” / constant A_{αs}=1/2)
  - computes Hessian as `H = hessian_F_at_y(y_star, A, b_s, coeff_alpha)`
  - includes an in-module **warmstart cache across γ** for fixed (p,β,q,r)
  - includes a **cap** on total Newton attempts to prevent stalls on hard γ

### `run_8sat_block_decay.py`
- Implements “Fix 1/2/3” from `8SAT_HESSIAN_FIXES.md`:
  - checks `res["converged"]` and prints warnings,
  - adds visual red-border marker on nonconverged panels,
  - includes residual output,
  - suptitle explicitly says **8‑SAT ∇²F (q=3, r=176.54)**
- Implements diagnostic print from `Q3_CONVERGENCE_FIX.md`:
  - prints `|coeff_α|` range/mean per (p,γ)
- Updated γ sweep to **small γ** (see “How to run”), because larger γ is not reliably convergent at threshold.
- Added a `QUANDELA_QUICK=1` mode to run fewer p values and (in quick mode) omit the hardest γ point.

### `symmetry_reduction/saddle.py`
- `saddle_fixed_point_q`:
  - added `y_init` warmstart support,
  - added overflow/NaN guard (skip update if `y_new` non-finite)
- `saddle_with_adaptive_damping_q`:
  - passes through `y_init`
- Implemented `saddle_fixed_point_q_homotopy`:
  - initially recursive subdivider; then rewritten to **adaptive backtracking/bisection** approach
  - still stalls before reaching r=176.54 for γ=0.2 (p=2) in testing
  - NOTE: this fixed-point homotopy is kept for experimentation but is no longer the default q>1 saddle path.

## Evidence of current failure mode
Empirically, q=3 at \(r=176.54\) is difficult:
- The old **fixed-point** approach stalls/diverges well before threshold at moderate γ.
- The new **Newton u-space** approach can converge at small γ, but begins failing (residual above threshold) as γ grows (often already around γ≈0.005–0.01 depending on p).

## Next recommended implementation (for the new chat)
Clarify the research goal given the **small‑γ limitation**:
- If the scientific question is “does block structure persist for 8‑SAT at threshold?”, a **small‑γ sweep** may be sufficient (and is consistent with BM24’s small‑γ regime / initialization).
- If the scientific question specifically requires \(\gamma \in \{\pi/2,\pi,2\pi\}\), then we likely need a **different numerical approach or reformulation** (e.g. alternative continuation path, different variable scaling, or solving a different but equivalent saddle system) because the current solver cannot reliably reach those γ with small residual at \(r=176.54\).

