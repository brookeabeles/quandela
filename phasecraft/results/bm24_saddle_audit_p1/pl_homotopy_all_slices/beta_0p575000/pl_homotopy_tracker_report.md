# PL homotopy branch tracker

## Verdict

This is the first practical homotopy-style PL diagnostic: it follows one competitor branch from the post-crossing region back toward small Gamma and asks where a Stokes jump could plausibly turn on its intersection number.  It is still not a rigorous intersection-number computation.

## Setup

- beta: 0.575
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
| 1.000000 | -0.103104 | 0.318705 | 0.00783883 | 0.0952654 | seed |  |
| 1.300000 | -0.0781583 | 0.143804 | 0.00590896 | 0.0722493 | seed |  |
| 1.500000 | -0.0526556 | 0.0667586 | 0.00360156 | 0.049054 | seed |  |
| 1.600000 | -0.0381497 | 0.0382006 | 0.00130835 | 0.0368414 | seed |  |
| 1.700000 | -0.0229426 | 0.0162897 | 0.00296722 | 0.0259098 | seed |  |
| 1.750000 | -0.0152393 | 0.0080777 | 0.00638794 | 0.0216272 | seed | yes |
| 1.800000 | -0.00762234 | 0.00212129 | 0.0110231 | 0.0186454 | seed | yes |
| 1.825000 | -0.00391842 | 0.000318609 | 0.0138651 | 0.0177835 | seed | yes |
| 1.850000 | +0.00123311 | 4.44089e-16 | 0.0180449 | 0.0168118 | competitor | yes |
| 1.875000 | +0.00547143 | 3.51128e-05 | 0.0204477 | 0.0149762 | competitor | yes |
| 1.900000 | +0.0112901 | 0.000650934 | 0.0242264 | 0.0129363 | competitor | yes |
| 1.950000 | +0.0252074 | 0.0025779 | 0.0341644 | 0.00895702 | competitor |  |
| 2.000000 | +0.0414149 | 0.00484964 | 0.0470105 | 0.00559563 | competitor |  |
| 2.050000 | +0.0594038 | 0.00725117 | 0.0624332 | 0.00302943 | competitor |  |
| 2.100000 | +0.0788553 | 0.0096937 | 0.0800604 | 0.00120504 | competitor |  |

## Interpretation

- Start of homotopy assumption: at small Gamma, seed intersection number is 1 and this competitor branch is off.
- A necessary place for the competitor intersection number to change is a Stokes condition, `Delta Im Phi = 0 mod 2pi`.
- Exponential dominance changes at anti-Stokes, `Delta Re Phi = 0`.
- If the Stokes and anti-Stokes windows coincide, that is the strongest numerical evidence for the PL mechanism behind the observed physical transition.

## Figures

- `pl_homotopy_stokes_anti_stokes.png`
- `pl_homotopy_exact_gap_switch.png`
