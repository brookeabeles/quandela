# PL homotopy diagnostics across refined beta slices

This batch applies the beta-opt PL homotopy diagnostic to every refined beta slice.  Each slice uses the longest contiguous competitor-best run as its anchor branch.  The result is still a necessary-condition PL diagnostic, not rigorous intersection numbers.

## Summary

| beta | refined regime | anchor Gamma | anti-Stokes brackets | exact-switch brackets | PL aligned? | notes |
|---:|---|---:|---|---|---|---|
| 0.350000 | robust_single_switch | 1.700000 | [] | [] | yes | right-censored: 7 pre-onset branch-merge points filtered |
| 0.400000 | robust_single_switch | 1.750000 | [[1.7, 1.75]] | [[1.7, 1.75]] | yes | 3 small-Gamma branch-merge points filtered |
| 0.450000 | robust_single_switch | 1.800000 | [[1.75, 1.8]] | [[1.75, 1.8]] | yes | 3 small-Gamma branch-merge points filtered |
| 0.500000 | robust_single_switch | 1.825000 | [[1.8, 1.825]] | [[1.8, 1.825]] | yes | 3 small-Gamma branch-merge points filtered |
| 0.543400 | robust_single_switch | 1.850000 | [[1.825, 1.85]] | [[1.825, 1.85]] | yes | 3 small-Gamma branch-merge points filtered |
| 0.575000 | robust_single_switch | 1.850000 | [[1.825, 1.85]] | [[1.825, 1.85]] | yes | 3 small-Gamma branch-merge points filtered |
| 0.600000 | ambiguous_or_nonmonotone | 1.875000 | [] | [] | no | LinAlgError('SVD did not converge') |
| 0.625000 | robust_single_switch | 1.875000 | [[1.85, 1.875]] | [[1.85, 1.875]] | yes | 3 small-Gamma branch-merge points filtered |
| 0.650000 | ambiguous_or_nonmonotone | 1.900000 | [[1.875, 1.9]] | [[1.875, 1.9], [2.05, 2.1]] | yes | 3 small-Gamma branch-merge points filtered |

## Figures

- `pl_homotopy_all_slices_summary.png`
- `pl_homotopy_all_slices_summary.pdf`
