# BM24 saddle push-further diagnostics

Generated from existing audit artifacts; no new root searches were run.

## Headline

The strongest empirical story is an explanatory saddle transition: the BM24 seed saddle controls in the QAOA-relevant window, while the branch 43/46 conjugate pair controls the large-|gamma| plateau.

## Key numerical outputs

- Primary Anti-Stokes DeltaRe crossing near QAOA window: -1.830842
- Other DeltaRe zeroes seen in the onset window: [-1.830842334140663, -1.69802612700208, -1.68001081946482]
- DeltaIm mod 2pi range on crossing plot: 0 to 0.276
- Finite-n drift linear extrapolation from sparse grid: gamma_cross(infty) ~= -1.8636
- Scatter points: 965 certified competitors; 85 high-Re roots with bad exact-exponent match.
- Large-gamma RMSE, merged 43/46: 0.00609; seed: 1.31362.
- Branch 43/46 shared gamma count: 148; median |Re43-Re46|: 1.48e-08.

## Interpretation of 43/46

The dominance is not challenged by two unrelated saddles. Branches 43 and 46 are best treated as a conjugate/symmetric pair: they are distinct z-roots, but they have nearly the same real action over the overlap. For the exponential rate, their shared Re Phi is what matters; for prefactors and oscillatory interference, the pair should remain distinct.

## Generated figures

- `physical_phase_diagram`: `/Users/b/Quandela/phasecraft/experiments/bm24_saddle_audit_p1/results/push_groundbreaking/physical_dominance_phase_diagram_beta_gamma.png`
- `anti_stokes`: `/Users/b/Quandela/phasecraft/experiments/bm24_saddle_audit_p1/results/push_groundbreaking/anti_stokes_crossing_gamma.png`
- `finite_n_drift`: `/Users/b/Quandela/phasecraft/experiments/bm24_saddle_audit_p1/results/push_groundbreaking/finite_n_crossing_drift.png`
- `algebraic_vs_physical`: `/Users/b/Quandela/phasecraft/experiments/bm24_saddle_audit_p1/results/push_groundbreaking/algebraic_vs_physical_saddles_scatter.png`
- `large_gamma_plateau`: `/Users/b/Quandela/phasecraft/experiments/bm24_saddle_audit_p1/results/push_groundbreaking/large_gamma_plateau.png`
- `branch_43_46_pair`: `/Users/b/Quandela/phasecraft/experiments/bm24_saddle_audit_p1/results/push_groundbreaking/branch_43_46_pair_diagnostic.png`
