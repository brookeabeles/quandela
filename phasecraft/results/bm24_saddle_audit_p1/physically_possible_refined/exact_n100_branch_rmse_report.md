# Exact n=100 Branch RMSE Against lambda_abs

Generated: 2026-07-07T22:18:30.677659+00:00

## Method

For each clean beta slice with a resolved equal-action point, I used the Krawczyk-tracked seed and competitor rows from `pl_homotopy_all_slices.json`.

For each sampled Gamma, I recomputed the exact finite-n baseline at `n=100` using:

```python
finite_n_exponent_grid(K_clause=8, q=3, r=176.54, beta=beta, gamma=-Gamma, n_values=[100])
```

This is the same exact finite-n lambda_abs machinery used by the finite-size/Fig. 6 workflow.  The primary comparison is against `lambda_abs[100]` from the Conv2 all-subsets baseline.

The RMSE columns compare `lambda_abs(100)` to the full saddle exponent `E = Re Phi + phi_pref`, because `lambda_abs` includes the same prefactor convention.  Raw `Re Phi` RMSE values are also stored in the JSON for auditability, but they are offset by the prefactor and are not the physically matched comparison.

Rows are split by the resolved equal-action crossing `Gamma_AS(beta)`: inside means `|gamma| = Gamma < Gamma_AS`; outside means `Gamma > Gamma_AS`.

## Summary Table

| beta | Gamma_AS | n_in | RMSE seed in | RMSE comp in | n_out | RMSE seed out | RMSE comp out |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.4 | 1.746484 | 5 | 0.0020222 | 0.0476642 | 10 | 0.0662089 | 0.00238704 |
| 0.45 | 1.782422 | 6 | 0.00237538 | 0.0480939 | 9 | 0.0568396 | 0.00252082 |
| 0.5 | 1.816797 | 7 | 0.00293606 | 0.0486883 | 8 | 0.0501607 | 0.00304561 |
| 0.5433996421 | 1.833203 | 8 | 0.00321174 | 0.0488961 | 7 | 0.0465461 | 0.003245 |
| 0.575 | 1.848828 | 8 | 0.00270423 | 0.0512663 | 7 | 0.0425496 | 0.00406105 |
| 0.625 | 1.873047 | 9 | 0.00295954 | 0.0519584 | 6 | 0.0405634 | 0.00410274 |
| 0.65 | 1.876953 | 10 | 0.0036618 | 0.0510508 | 5 | 0.0420832 | 0.0374665 |

## Interpretation

Inside the equal-action crossing, the seed branch is generally the closer saddle-rate approximation to the exact `lambda_abs(100)`.  Outside the crossing, the tracked competitor branch becomes comparable to or better than the seed on the clean slices.  This is an RMSE diagnostic of branch rates against the exact finite-n baseline; it does not compute or prove thimble intersection numbers.

## Outputs

- `exact_n100_branch_rmse_summary.json`
- `exact_n100_branch_rmse_points.csv`
- `exact_n100_branch_rmse_summary.csv`
- `exact_n100_lambda_cache.json`

Number of point rows: `105`.
