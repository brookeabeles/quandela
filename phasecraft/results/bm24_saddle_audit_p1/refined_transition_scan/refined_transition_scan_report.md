# Refined transition scan

## Method

A certified saddle is not declared physically dominant merely because it has larger `Re Phi`. For each grid point the script compares saddle exponents against the exact finite-n exponent `lambda_abs = log|P_n|/n` over the configured `n` range.  A large-Re saddle is marked as a decoy challenge when it beats the seed algebraically but the seed remains the closest match to the exact exponent.  This is an empirical/certified filter, not a Picard-Lefschetz intersection proof.

## Scan

- beta points: 9
- Gamma points: 16
- successful grid points: 144 / 144
- competitor random starts per point: 200
- finite-n values: [18, 19, 20, 21, 22, 23, 24]
- decoy margin: `0.02` in conv2 exponent units

## Classification counts

- seed controls cleanly: 8
- seed controls despite high-Re decoys: 70
- competitor controls by exact-exponent match: 66

## Transition estimates

| beta | transition interval in Gamma | midpoint | notes |
|---:|---:|---:|---|
| 0.350000 | 1.6500-1.7000 | 1.6750 | 4 decoy-challenge points, 11 competitor-best points |
| 0.400000 | 1.7000-1.7500 | 1.7250 | 5 decoy-challenge points, 10 competitor-best points |
| 0.450000 | 1.7500-1.8000 | 1.7750 | 6 decoy-challenge points, 9 competitor-best points |
| 0.500000 | 1.8000-1.8250 | 1.8125 | 7 decoy-challenge points, 8 competitor-best points |
| 0.543400 | 1.8250-1.8500 | 1.8375 | 7 decoy-challenge points, 7 competitor-best points |
| 0.575000 | 1.8250-1.8500 | 1.8375 | 8 decoy-challenge points, 7 competitor-best points |
| 0.600000 | 1.6500-1.7000, 1.7000-1.7500, 1.8500-1.8750, 1.8750-1.9000 | 1.6750 | 13 decoy-challenge points, 2 competitor-best points |
| 0.625000 | 1.8500-1.8750 | 1.8625 | 10 decoy-challenge points, 6 competitor-best points |
| 0.650000 | 1.7000-1.7500, 1.7500-1.8000, 1.8750-1.9000 | 1.7250 | 10 decoy-challenge points, 6 competitor-best points |

## Strongest decoy challenge

- beta=0.600000, Gamma=1.825000
- max competitor advantage over seed: +5.390839
- seed exact gap: 0.013448
- best competitor exact gap: 0.0190081

## First competitor-control evidence near beta_opt

- beta=0.543400, Gamma=1.850000
- seed exact gap: 0.0179472
- best competitor exact gap: 0.0146365
- max competitor advantage over seed: +1.797901

## Figures

- `refined_transition_map.png`
- `refined_decoy_pressure_and_exact_gap.png`
- `refined_boundary_curve.png`
