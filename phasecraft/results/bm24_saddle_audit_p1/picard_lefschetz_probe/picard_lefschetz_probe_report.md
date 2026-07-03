# Picard-Lefschetz feasibility probe

## Verdict

A rigorous PL calculation is not yet possible from the current artifacts alone because the original BM24 integration cycle is not represented in the same multi-valued fractional-power z-sheet coordinates used by the saddle equations.  However, a local numerical PL probe is possible: the script constructs the local action sheet at selected saddles and traces short upward flows.

## What was tested

- beta: 0.5433996420760803
- gammas: [-1.825, -1.85]
- competitor starts per gamma: 80
- flow t_max: 0.1
- flow epsilon: 1e-05

## Local PL diagnostics

| Gamma | saddle | gap to exact | grad_inf | Hessian + / - / 0 | median |Delta Im S| | median Delta Re S |
|---:|---|---:|---:|---:|---:|---:|
| 1.825000 | seed | 0.0146103 | 1.665e-15 | 8 / 8 / 0 | 8.004e-15 | 3.485e-12 |
| 1.825000 | best_exact_competitor | 0.0162387 | 1.665e-15 | 8 / 8 / 0 | 1.931e-14 | 2.451e-12 |
| 1.825000 | highest_re_competitor | 1.05535 | 1.776e-15 | 8 / 8 / 0 | 1.799e-14 | 1.106e-10 |
| 1.850000 | seed | 0.0178743 | 2.635e-15 | 8 / 8 / 0 | 9.456e-15 | 3.367e-12 |
| 1.850000 | best_exact_competitor | 0.0146365 | 1.671e-15 | 8 / 8 / 0 | 2.749e-14 | 2.191e-12 |
| 1.850000 | highest_re_competitor | 1.45777 | 1.887e-15 | 8 / 8 / 0 | 1.155e-14 | 1.230e-11 |

## Interpretation

- If `grad_inf` is tiny and the Hessian has 8 positive and 8 negative real-flow eigenvalues, the local sheet behaves like a valid holomorphic Morse saddle for p=1.
- If upward flows increase `Re S` while conserving `Im S`, the local PL ODE is numerically coherent.
- This still does not give intersection numbers.  The next missing object is the original BM24 integration cycle in z-sheet coordinates, plus a global branch-cut/singularity handling strategy.

## Figure

- `pl_upward_flow_sanity.png`
