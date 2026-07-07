# Beta=0.60 Branch Resolution

Generated: 2026-07-07T13:18:02.780560+00:00

## Branch continuation from side anchors

Endpoint classification: `distinct_branches`.

| Gamma | classification | ||z_left-z_right||_inf | |Delta E| |
|---:|---|---:|---:|
| 1.830 | same_branch | 2.678e-14 | 4.441e-16 |
| 1.840 | distinct_branches | 6.071e-01 | 4.441e-16 |
| 1.850 | same_branch | 3.764e-14 | 8.882e-16 |
| 1.860 | distinct_branches | 1.224e+00 | 1.296e-03 |
| 1.870 | same_branch | 4.454e-14 | 1.554e-15 |
| 1.880 | same_branch | 2.063e-14 | 6.661e-16 |

## beta=0.60 equal-action roots

| branch ID | Gamma_lo | Gamma_hi | midpoint | certification | branch-safe? |
|---|---:|---:|---:|---|---|
| branch_left | 1.857500 | 1.860000 | 1.858750 | partial | False |
| branch_right | 1.858750 | 1.859375 | 1.859063 | partial | False |

## Old red cross

Classification: `branch_selection_artifact`.

The old beta=0.60 marker came from the gridwise rate-best competitor selection, not from a single branch definition. The table in `beta060_branch_assignment.json` assigns each original grid-selected root to the nearest continued branch family and reports the tracked branch `f_AS` values alongside `D_24`.

## Local 2D structure

Branch-safe local root counts: `{'branch_left': 1, 'branch_right': 1}`.

## Main-figure decision

Decision case: `D`.
Distinct side-anchored branches reach beta=0.60; the old nonmonotonicity should be treated as branch switching.

No thimble intersection numbers are computed here; the PL phase points remain necessary-condition diagnostics only.
