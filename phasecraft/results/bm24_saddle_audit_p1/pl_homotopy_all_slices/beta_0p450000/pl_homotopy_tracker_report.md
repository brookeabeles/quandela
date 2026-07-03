# PL homotopy branch tracker

## Verdict

This is the first practical homotopy-style PL diagnostic: it follows one competitor branch from the post-crossing region back toward small Gamma and asks where a Stokes jump could plausibly turn on its intersection number.  It is still not a rigorous intersection-number computation.

## Setup

- beta: 0.45
- anchor Gamma: 1.8
- anchor competitor search starts: 80
- Stokes tolerance: 0.08
- anti-Stokes tolerance: 0.02

## Crossing Summary

- anti-Stokes brackets: [[1.75, 1.8]]
- physical exact-match switch brackets: [[1.75, 1.8]]
- candidate simultaneous PL jump points: [1.7, 1.75, 1.8, 1.825, 1.85]

## Branch Table

| Gamma | Delta Re Phi | |Delta Im Phi| mod 2pi | seed gap | comp gap | exact winner | PL candidate? |
|---:|---:|---:|---:|---:|---|---|
| 1.000000 | -0.0870181 | 0.304729 | 0.00687697 | 0.0801411 | seed |  |
| 1.300000 | -0.0661432 | 0.127445 | 0.00476355 | 0.0613797 | seed |  |
| 1.500000 | -0.0408428 | 0.0519645 | 0.00255398 | 0.0382888 | seed |  |
| 1.600000 | -0.026303 | 0.0253551 | 0.000111627 | 0.0261913 | seed |  |
| 1.700000 | -0.0113408 | 0.00662885 | 0.0050631 | 0.016404 | seed | yes |
| 1.750000 | -0.00410188 | 0.0010215 | 0.0094248 | 0.0135267 | seed | yes |
| 1.800000 | +0.0039948 | 0.000139682 | 0.0149674 | 0.0109726 | competitor | yes |
| 1.825000 | +0.0095894 | 0.000970296 | 0.0184311 | 0.00884165 | competitor | yes |
| 1.850000 | +0.016145 | 0.00205598 | 0.0228366 | 0.00669162 | competitor | yes |
| 1.875000 | +0.0234597 | 0.00326423 | 0.0281321 | 0.00467235 | competitor |  |
| 1.900000 | +0.03141 | 0.00454116 | 0.0342737 | 0.0028637 | competitor |  |
| 1.950000 | +0.0488922 | 0.00719584 | 0.0488941 | 1.85366e-06 | competitor |  |
| 2.000000 | +0.0681134 | 0.00989388 | 0.0662337 | 0.00187972 | competitor |  |
| 2.050000 | +0.0887611 | 0.0125777 | 0.0857853 | 0.00297583 | competitor |  |
| 2.100000 | +0.110611 | 0.0152175 | 0.107084 | 0.00352712 | competitor |  |

## Interpretation

- Start of homotopy assumption: at small Gamma, seed intersection number is 1 and this competitor branch is off.
- A necessary place for the competitor intersection number to change is a Stokes condition, `Delta Im Phi = 0 mod 2pi`.
- Exponential dominance changes at anti-Stokes, `Delta Re Phi = 0`.
- If the Stokes and anti-Stokes windows coincide, that is the strongest numerical evidence for the PL mechanism behind the observed physical transition.

## Figures

- `pl_homotopy_stokes_anti_stokes.png`
- `pl_homotopy_exact_gap_switch.png`
