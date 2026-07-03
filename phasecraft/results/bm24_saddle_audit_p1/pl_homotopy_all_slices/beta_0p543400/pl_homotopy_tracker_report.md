# PL homotopy branch tracker

## Verdict

This is the first practical homotopy-style PL diagnostic: it follows one competitor branch from the post-crossing region back toward small Gamma and asks where a Stokes jump could plausibly turn on its intersection number.  It is still not a rigorous intersection-number computation.

## Setup

- beta: 0.5433996421
- anchor Gamma: 1.85
- anchor competitor search starts: 80
- Stokes tolerance: 0.08
- anti-Stokes tolerance: 0.02

## Crossing Summary

- anti-Stokes brackets: [[1.825, 1.85]]
- physical exact-match switch brackets: [[1.825, 1.85]]
- candidate simultaneous PL jump points: [1.75, 1.8, 1.825, 1.85, 1.875, 1.9]

## Branch Table

| Gamma | Delta Re Phi | |Delta Im Phi| mod 2pi | seed gap | comp gap | exact winner | PL candidate? |
|---:|---:|---:|---:|---:|---|---|
| 1.000000 | -0.0988912 | 0.316247 | 0.00764958 | 0.0912416 | seed |  |
| 1.300000 | -0.0753016 | 0.140701 | 0.00572153 | 0.06958 | seed |  |
| 1.500000 | -0.0500154 | 0.0638006 | 0.00347831 | 0.0465371 | seed |  |
| 1.600000 | -0.0355461 | 0.0355471 | 0.00119253 | 0.0343536 | seed |  |
| 1.700000 | -0.020396 | 0.0141625 | 0.00320146 | 0.0235975 | seed |  |
| 1.750000 | -0.0127589 | 0.00635629 | 0.00677653 | 0.0195354 | seed | yes |
| 1.800000 | -0.00527666 | 0.00104444 | 0.0116417 | 0.0169183 | seed | yes |
| 1.825000 | -0.00162845 | 8.88178e-16 | 0.0146103 | 0.0162387 | seed | yes |
| 1.850000 | +0.0033107 | 0 | 0.0179472 | 0.0146365 | competitor | yes |
| 1.875000 | +0.0086737 | 0.000437994 | 0.021311 | 0.0126373 | competitor | yes |
| 1.900000 | +0.0149982 | 0.00130416 | 0.0255695 | 0.0105713 | competitor | yes |
| 1.950000 | +0.0297238 | 0.00346057 | 0.0364645 | 0.00674068 | competitor |  |
| 2.000000 | +0.0465981 | 0.00585707 | 0.050258 | 0.0036599 | competitor |  |
| 2.050000 | +0.0651706 | 0.00833964 | 0.0665812 | 0.00141063 | competitor |  |
| 2.100000 | +0.0851474 | 0.0108389 | 0.0850311 | 0.000116323 | competitor |  |

## Interpretation

- Start of homotopy assumption: at small Gamma, seed intersection number is 1 and this competitor branch is off.
- A necessary place for the competitor intersection number to change is a Stokes condition, `Delta Im Phi = 0 mod 2pi`.
- Exponential dominance changes at anti-Stokes, `Delta Re Phi = 0`.
- If the Stokes and anti-Stokes windows coincide, that is the strongest numerical evidence for the PL mechanism behind the observed physical transition.

## Figures

- `pl_homotopy_stokes_anti_stokes.png`
- `pl_homotopy_exact_gap_switch.png`
