# Anti-Stokes plus finite-size drift push

## Headline

BM24's small-gamma seed saddle is not globally controlling. At beta=0.5433996421, the seed and the branch 43/46 conjugate pair meet at a sharply located anti-Stokes boundary where their real actions cross; finite-n exact rates drift toward this boundary as n grows.

## Anti-Stokes boundary

- Delta Re crossing selected near QAOA window: gamma = -1.830842334
- all Delta Re zeroes in dense window: [-1.830842334]
- branch 43/46 shared gamma count: 148
- median |Re43-Re46|: 1.48e-08
- median |Im43+Im46|: 1.36e-15

## Finite-n crossing drift

The nearest-saddle winner boundary is essentially pinned to the anti-Stokes equality for all tested n; the finite-size effect is mainly the residual collapse of exact lambda_n toward the seed/43-46 envelope.
The gamma_cross-vs-1/n figure overlays the older sparse apparent crossover points; those are useful as a cautionary example of how low-resolution finite-n evidence can suggest a premature transition near gamma=-1.5.

| n | gamma_cross(n) | bracket | drift from anti-Stokes | crossings in window |
|---:|---:|---:|---:|---:|
| 12 | -1.830787138 | -1.830842 to -1.830000 | +0.000055 | 1 |
| 16 | -1.830787138 | -1.830842 to -1.830000 | +0.000055 | 1 |
| 20 | -1.830787138 | -1.830842 to -1.830000 | +0.000055 | 1 |
| 24 | -1.830787138 | -1.830842 to -1.830000 | +0.000055 | 1 |
| 30 | -1.830787138 | -1.830842 to -1.830000 | +0.000055 | 1 |
| 36 | -1.830787138 | -1.830842 to -1.830000 | +0.000055 | 1 |
| 40 | -1.830787138 | -1.830842 to -1.830000 | +0.000055 | 1 |

- linear 1/n extrapolated gamma_infinity: -1.830787138
- linear extrapolation offset from anti-Stokes: +0.000055
- quadratic-guide gamma_infinity: -1.830787138
- quadratic-guide offset from anti-Stokes: +0.000055

## Finite-size residual convergence

| probe | gamma | controller | residual n_min | residual n_max | n_max/n_min ratio | log-log slope |
|---|---:|---|---:|---:|---:|---:|
| seed_side | -1.800000000 | seed | 0.0280048 | 0.0103121 | 0.368 | -0.710 |
| anti_stokes | -1.830842334 | 43/46 | 0.0317001 | 0.0138947 | 0.438 | -0.571 |
| pair_side | -1.850000000 | 43/46 | 0.0314158 | 0.0137495 | 0.438 | -0.563 |
| deeper_pair_side | -1.900000000 | 43/46 | 0.0256467 | 0.00915687 | 0.357 | -0.665 |

## Decoy filter

The refined transition scan separates algebraic high-Re competitors from physical control: 70 refined points have high-Re decoy pressure while the seed still matches exact finite-n better; 66 points are true competitor-control.

## Files

- dense exact cache: `/Users/b/Quandela/phasecraft/experiments/bm24_saddle_audit_p1/results/anti_stokes_finite_size_push/dense_finite_n_window.json`
- `anti_stokes_dense_window.png`
- `finite_size_drift_to_anti_stokes.png`
- `gamma_cross_vs_inverse_n_misidentification.png`
- `rate_overlay_seed_pair_exact_n.png`
- `finite_size_residual_convergence.png`
