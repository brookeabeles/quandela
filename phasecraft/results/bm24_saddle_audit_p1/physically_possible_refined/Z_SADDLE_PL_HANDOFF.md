# Z Saddle / PL Diagnostic Handoff

Date: 2026-07-07

This note summarizes the current state of the BM24 p=1 z-saddle crossover / Picard-Lefschetz diagnostic work. It is meant as a handoff for a new chat.

## Main Conclusion

The strongest current result is:

- On all clean beta slices except the excluded nonmonotone `beta=0.60` slice, the rate-selected competitor and the PL-tracked competitor agree as the same numerical root in direct z-coordinates at about `1e-14`.
- The finite-n residual crossover `D_n = 0` coincides with equal-real-action `Delta Re Phi = 0` for `n=18..24`, but this is not an independent finite-n stability result. It is an algebraic consequence of `lambda_n` lying above both saddle rates in the audited regime.
- No thimble intersection numbers have been computed. All PL statements are necessary-condition diagnostics only.

## Most Relevant Files

Primary report:

- `/Users/b/Quandela/phasecraft/results/bm24_saddle_audit_p1/physically_possible_refined/physically_possible_refined_report.md`

Primary figure:

- `/Users/b/Quandela/phasecraft/results/bm24_saddle_audit_p1/physically_possible_refined/separated_crossover_and_pl_conditions.png`
- `/Users/b/Quandela/phasecraft/results/bm24_saddle_audit_p1/physically_possible_refined/separated_crossover_and_pl_conditions.pdf`

Key JSON outputs:

- `/Users/b/Quandela/phasecraft/results/bm24_saddle_audit_p1/physically_possible_refined/physically_possible_refined.json`
- `/Users/b/Quandela/phasecraft/results/bm24_saddle_audit_p1/physically_possible_refined/physically_possible_refined_summary.json`
- `/Users/b/Quandela/phasecraft/results/bm24_saddle_audit_p1/physically_possible_refined/finite_n_crossover_stability.json`
- `/Users/b/Quandela/phasecraft/results/bm24_saddle_audit_p1/physically_possible_refined/finite_n_mechanism_audit.json`
- `/Users/b/Quandela/phasecraft/results/bm24_saddle_audit_p1/physically_possible_refined/equal_real_action_refined_brackets.json`
- `/Users/b/Quandela/phasecraft/results/bm24_saddle_audit_p1/physically_possible_refined/interval_comparisons.json`
- `/Users/b/Quandela/phasecraft/results/bm24_saddle_audit_p1/physically_possible_refined/beta035_left_censor_resolution.json`

Key scripts:

- `/Users/b/Quandela/phasecraft/experiments/bm24_saddle_audit_p1/physically_possible_refined_figures.py`
- `/Users/b/Quandela/phasecraft/experiments/bm24_saddle_audit_p1/pl_homotopy_tracker.py`
- `/Users/b/Quandela/phasecraft/experiments/bm24_saddle_audit_p1/continue_seed_branch_certified.py`

## Current Figure Meaning

The final main figure is:

- Panel (a): rate-residual diagnostic
  `D_24 = |lambda_comp - lambda_24| - |lambda_seed - lambda_24|`.
- Panel (b): slice-wise PL necessary-condition candidates.

Important visual conventions:

- Purple points/line: rate-residual crossover. The guide line is broken across `beta=0.60`.
- No line is connected through left-censored `beta=0.35`.
- Red cross: nonmonotone `beta=0.60` slice.
- Gray square candidates: phase-alignment roots. The red phase-family connection was removed because anchor robustness failed.
- Black points/brackets: equal-real-action candidates.

The figure should be described with “brackets”, not “error bars”.

## Clean Slice Identity Check

For the clean slices, the report compares the best discovered rate-admissible competitor on the rate-crossover side with the PL-tracked competitor at the nearest stored Gamma.

Result:

| beta | status |
|---:|---|
| 0.350000 | same numerical root, but left-censored in finite-n table |
| 0.400000 | same numerical root |
| 0.450000 | same numerical root |
| 0.500000 | same numerical root |
| 0.543400 | same numerical root |
| 0.575000 | same numerical root |
| 0.600000 | excluded/nonmonotone, unresolved tracked row |
| 0.625000 | same numerical root |
| 0.650000 | same numerical root |

Direct coordinate agreement is about `1e-14` on the clean, resolved slices. Symmetry minimization was deliberately not implemented because direct agreement is already stronger.

## Finite-n Stability / Mechanism Audit

The finite-n pass recomputes only `lambda_n` for `n=18..24`. It does not rerun saddle searches and does not reselect competitors independently by n.

Current outcome:

- `Gamma_cross(beta;n)` is flat across `n=18..24` at the reported precision for all bracketed clean slices.
- This flatness should not be oversold as an independent finite-n phenomenon.
- The mechanism audit shows `lambda_n` lies above both saddle rates, so
  `D_n = |E_comp - lambda_n| - |E_seed - lambda_n|`
  reduces algebraically to
  `D_n = E_seed - E_comp`.
- Therefore `D_n = 0` is equivalent to `E_seed = E_comp`, i.e. equal real action, in this regime.

## Equal-real-action Brackets

Clean equal-real-action brackets were root-refined by continuing/polishing the same seed and competitor branches and Krawczyk-certifying endpoint saddles.

The scalar Gamma root itself is not rigorously interval-certified; only the endpoint saddle roots are Krawczyk-certified.

Current refined widths are about `7.81e-04`, satisfying the requested `<= 1e-3`.

See:

- `/Users/b/Quandela/phasecraft/results/bm24_saddle_audit_p1/physically_possible_refined/equal_real_action_refined_brackets.json`

## Interval Comparisons

The report includes:

- `I_AS` vs `I_cross,24`
- `I_AS` vs selected `I_S`

Because the mechanism audit shows `D_24 = 0` is algebraically equivalent to equal action in this regime, no redundant second bisection was run for `I_cross,24`.

For the phase condition:

- `I_S` overlaps `I_AS` near beta around `0.50`, `beta_opt`, and `0.575`.
- It does not overlap at the outer clean slices.
- Because the phase-family anchor test failed, do not claim a unique global phase-alignment family.

## Phase-family Anchor Robustness

Anchors tested:

- `beta=0.50`
- `beta=beta_opt`
- `beta=0.575`

Result:

- Not robust.
- The three anchors do not select one common phase-alignment family across all common betas.
- The red dashed phase-family line was removed.
- The figure shows all phase roots as pale gray candidates.

Also state:

- Agreement at `beta_opt` is not independent because the original family was selected by proximity to the equal-real-action candidate there.

## Special Slices

### beta = 0.35

Initially left-censored because the tracked competitor branch first appears already competitor-favored.

A continuation attempt from `Gamma=1.70` toward smaller Gamma was made. At `Gamma=1.65`, the continued branch merged with the seed:

- status: `continued_branch_merged_with_seed`

So `beta=0.35` remains left-censored rather than resolved.

See:

- `/Users/b/Quandela/phasecraft/results/bm24_saddle_audit_p1/physically_possible_refined/beta035_left_censor_resolution.json`

### beta = 0.60

Still excluded from the smooth boundary.

Reason:

- rate classification is nonmonotone;
- PL tracked row is unresolved/missing for the first rate-selected point;
- multiple transition intervals appear in the refined grid.

Optional future work:

- targeted dense scan over `Gamma in [1.60, 1.92]` with step `0.005` or `0.01`;
- store z coordinates and rate-best competitor identity at every point;
- determine whether multiple crossings are due to competitor identity changes, same-branch recrossing, missing continuation, or lambda oscillation.

## Caveats to Preserve

Do not claim:

- thimble intersection numbers were computed;
- the PL mechanism is proven;
- saddle search is exhaustive;
- scalar Gamma roots are rigorously interval-certified;
- phase-family uniqueness.

Safe wording:

- “PL necessary-condition diagnostic.”
- “Krawczyk-certified endpoint saddles.”
- “The finite random-start saddle search is not exhaustive.”
- “The same competitor is used in both panels on all clean slices.”
- “`D_n = 0` coincides with equal action here as an algebraic consequence of the observed ordering.”

## Verification Already Run

The following passed:

```bash
python -m py_compile \
  experiments/bm24_saddle_audit_p1/physically_possible_refined_figures.py \
  experiments/bm24_saddle_audit_p1/pl_homotopy_tracker.py \
  experiments/bm24_saddle_audit_p1/continue_seed_branch_certified.py
```

## Git / Worktree Notes

The worktree was already dirty before these edits. Do not blindly revert unrelated changes.

Files touched by the z-saddle finalization include:

- `experiments/bm24_saddle_audit_p1/physically_possible_refined_figures.py`
- `experiments/bm24_saddle_audit_p1/pl_homotopy_tracker.py`
- `experiments/bm24_saddle_audit_p1/continue_seed_branch_certified.py`
- regenerated result files under `results/bm24_saddle_audit_p1/physically_possible_refined/`
- regenerated PL cache/report files under `results/bm24_saddle_audit_p1/pl_homotopy_all_slices/`

No files were intentionally deleted during this finalization.
