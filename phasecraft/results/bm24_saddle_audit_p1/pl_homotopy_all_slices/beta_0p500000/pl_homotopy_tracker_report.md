# PL homotopy branch tracker

## Verdict

This is the first practical homotopy-style PL diagnostic: it follows one competitor branch from the post-crossing region back toward small Gamma and asks where a Stokes jump could plausibly turn on its intersection number.  It is still not a rigorous intersection-number computation.

## Setup

- beta: 0.5
- anchor Gamma: 1.825
- anchor competitor search starts: 80
- Stokes tolerance: 0.08
- anti-Stokes tolerance: 0.02

## Crossing Summary

- anti-Stokes brackets: [[1.8, 1.825]]
- physical exact-match switch brackets: [[1.8, 1.825]]
- candidate simultaneous PL jump points: [1.7, 1.75, 1.8, 1.825, 1.85, 1.875]

## Branch Table

| Gamma | Delta Re Phi | |Delta Im Phi| mod 2pi | seed gap | comp gap | exact winner | PL candidate? |
|---:|---:|---:|---:|---:|---|---|
| 1.000000 | -0.09329 | 0.311769 | 0.00733871 | 0.0859513 | seed |  |
| 1.300000 | -0.071215 | 0.135347 | 0.00536741 | 0.0658476 | seed |  |
| 1.500000 | -0.0460398 | 0.0588925 | 0.00317783 | 0.042862 | seed |  |
| 1.600000 | -0.0315651 | 0.0312439 | 0.000862214 | 0.0307029 | seed |  |
| 1.700000 | -0.0164827 | 0.0108499 | 0.00379605 | 0.0202788 | seed | yes |
| 1.750000 | -0.00896878 | 0.0038186 | 0.00766854 | 0.0166373 | seed | yes |
| 1.800000 | -0.00178942 | 1.33227e-15 | 0.0129277 | 0.0147171 | seed | yes |
| 1.825000 | +0.00284766 | 3.10862e-15 | 0.0160503 | 0.0132027 | competitor | yes |
| 1.850000 | +0.00813418 | 0.00052499 | 0.0193211 | 0.0111869 | competitor | yes |
| 1.875000 | +0.0144216 | 0.00145815 | 0.0235125 | 0.00909088 | competitor | yes |
| 1.900000 | +0.0214771 | 0.00255143 | 0.0285524 | 0.00707536 | competitor |  |
| 1.950000 | +0.037418 | 0.00496813 | 0.0410073 | 0.00358926 | competitor |  |
| 2.000000 | +0.0553153 | 0.00751761 | 0.056315 | 0.000999718 | competitor |  |
| 2.050000 | +0.0747887 | 0.0101016 | 0.0740432 | 0.000745488 | competitor |  |
| 2.100000 | +0.0955776 | 0.0126718 | 0.0937504 | 0.00182724 | competitor |  |

## Interpretation

- Start of homotopy assumption: at small Gamma, seed intersection number is 1 and this competitor branch is off.
- A necessary place for the competitor intersection number to change is a Stokes condition, `Delta Im Phi = 0 mod 2pi`.
- Exponential dominance changes at anti-Stokes, `Delta Re Phi = 0`.
- If the Stokes and anti-Stokes windows coincide, that is the strongest numerical evidence for the PL mechanism behind the observed physical transition.

## Figures

- `pl_homotopy_stokes_anti_stokes.png`
- `pl_homotopy_exact_gap_switch.png`
