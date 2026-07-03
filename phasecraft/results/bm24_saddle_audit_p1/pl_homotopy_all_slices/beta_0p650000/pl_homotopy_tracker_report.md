# PL homotopy branch tracker

## Verdict

This is the first practical homotopy-style PL diagnostic: it follows one competitor branch from the post-crossing region back toward small Gamma and asks where a Stokes jump could plausibly turn on its intersection number.  It is still not a rigorous intersection-number computation.

## Setup

- beta: 0.65
- anchor Gamma: 1.9
- anchor competitor search starts: 80
- Stokes tolerance: 0.08
- anti-Stokes tolerance: 0.02

## Crossing Summary

- anti-Stokes brackets: [[1.875, 1.9]]
- physical exact-match switch brackets: [[1.875, 1.9], [2.05, 2.1]]
- candidate simultaneous PL jump points: [1.8, 1.825, 1.85, 1.875, 1.9, 1.95]

## Branch Table

| Gamma | Delta Re Phi | |Delta Im Phi| mod 2pi | seed gap | comp gap | exact winner | PL candidate? |
|---:|---:|---:|---:|---:|---|---|
| 1.000000 | -0.113716 | 0.322366 | 0.00820641 | 0.10551 | seed |  |
| 1.300000 | -0.0848005 | 0.148744 | 0.0061853 | 0.0786152 | seed |  |
| 1.500000 | -0.0582743 | 0.071784 | 0.00366402 | 0.0546103 | seed |  |
| 1.600000 | -0.0435035 | 0.0428637 | 0.00128242 | 0.0422211 | seed |  |
| 1.700000 | -0.0280636 | 0.0202044 | 0.00289297 | 0.0309565 | seed |  |
| 1.750000 | -0.0202077 | 0.0113924 | 0.00610937 | 0.026317 | seed |  |
| 1.800000 | -0.0123594 | 0.00453121 | 0.0103983 | 0.0227577 | seed | yes |
| 1.825000 | -0.00847736 | 0.00199329 | 0.0130212 | 0.0214986 | seed | yes |
| 1.850000 | -0.00465911 | 0.000283534 | 0.0159934 | 0.0206525 | seed | yes |
| 1.875000 | -0.000340951 | 1.77636e-15 | 0.019318 | 0.0196589 | seed | yes |
| 1.900000 | +0.00510861 | 1.77636e-15 | 0.0229402 | 0.0178316 | competitor | yes |
| 1.950000 | +0.0174619 | 0.00109526 | 0.0312015 | 0.0137397 | competitor | yes |
| 2.000000 | +0.0323767 | 0.00303009 | 0.0423072 | 0.00993052 | competitor |  |
| 2.050000 | +0.0492417 | 0.00522154 | 0.0560277 | 0.00678601 | competitor |  |
| 2.100000 | +0.151531 | 0.0349554 | 0.0720552 | 0.0794753 | seed |  |

## Interpretation

- Start of homotopy assumption: at small Gamma, seed intersection number is 1 and this competitor branch is off.
- A necessary place for the competitor intersection number to change is a Stokes condition, `Delta Im Phi = 0 mod 2pi`.
- Exponential dominance changes at anti-Stokes, `Delta Re Phi = 0`.
- If the Stokes and anti-Stokes windows coincide, that is the strongest numerical evidence for the PL mechanism behind the observed physical transition.

## Figures

- `pl_homotopy_stokes_anti_stokes.png`
- `pl_homotopy_exact_gap_switch.png`
