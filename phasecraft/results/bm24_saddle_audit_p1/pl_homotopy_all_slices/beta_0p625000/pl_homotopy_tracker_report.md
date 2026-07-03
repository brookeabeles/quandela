# PL homotopy branch tracker

## Verdict

This is the first practical homotopy-style PL diagnostic: it follows one competitor branch from the post-crossing region back toward small Gamma and asks where a Stokes jump could plausibly turn on its intersection number.  It is still not a rigorous intersection-number computation.

## Setup

- beta: 0.625
- anchor Gamma: 1.875
- anchor competitor search starts: 80
- Stokes tolerance: 0.08
- anti-Stokes tolerance: 0.02

## Crossing Summary

- anti-Stokes brackets: [[1.85, 1.875]]
- physical exact-match switch brackets: [[1.85, 1.875]]
- candidate simultaneous PL jump points: [1.75, 1.8, 1.825, 1.85, 1.875, 1.9, 1.95]

## Branch Table

| Gamma | Delta Re Phi | |Delta Im Phi| mod 2pi | seed gap | comp gap | exact winner | PL candidate? |
|---:|---:|---:|---:|---:|---|---|
| 1.000000 | -0.110072 | 0.321442 | 0.00809372 | 0.101978 | seed |  |
| 1.300000 | -0.0825847 | 0.147457 | 0.00611545 | 0.0764693 | seed |  |
| 1.500000 | -0.0564852 | 0.0704161 | 0.00367435 | 0.0528108 | seed |  |
| 1.600000 | -0.0418345 | 0.0415669 | 0.00133219 | 0.0405023 | seed |  |
| 1.700000 | -0.0264927 | 0.019088 | 0.00285187 | 0.0293446 | seed |  |
| 1.750000 | -0.0186913 | 0.0104279 | 0.00611411 | 0.0248054 | seed | yes |
| 1.800000 | -0.010915 | 0.00379526 | 0.0104892 | 0.0214043 | seed | yes |
| 1.825000 | -0.00708273 | 0.00143126 | 0.0131702 | 0.0202529 | seed | yes |
| 1.850000 | -0.00333258 | 3.52595e-05 | 0.0162059 | 0.0195385 | seed | yes |
| 1.875000 | +0.00135666 | 1.33227e-15 | 0.0195859 | 0.0182292 | competitor | yes |
| 1.900000 | +0.00683836 | 4.78099e-05 | 0.0231513 | 0.016313 | competitor | yes |
| 1.950000 | +0.0196255 | 0.00150248 | 0.0318566 | 0.0122311 | competitor | yes |
| 2.000000 | +0.034923 | 0.00355547 | 0.0434655 | 0.00854248 | competitor |  |
| 2.050000 | +0.0521202 | 0.00581906 | 0.0576853 | 0.00556513 | competitor |  |
| 2.100000 | +0.0708589 | 0.0081655 | 0.0741891 | 0.00333024 | competitor |  |

## Interpretation

- Start of homotopy assumption: at small Gamma, seed intersection number is 1 and this competitor branch is off.
- A necessary place for the competitor intersection number to change is a Stokes condition, `Delta Im Phi = 0 mod 2pi`.
- Exponential dominance changes at anti-Stokes, `Delta Re Phi = 0`.
- If the Stokes and anti-Stokes windows coincide, that is the strongest numerical evidence for the PL mechanism behind the observed physical transition.

## Figures

- `pl_homotopy_stokes_anti_stokes.png`
- `pl_homotopy_exact_gap_switch.png`
