# PL homotopy branch tracker

## Verdict

This is the first practical homotopy-style PL diagnostic: it follows one competitor branch from the post-crossing region back toward small Gamma and asks where a Stokes jump could plausibly turn on its intersection number.  It is still not a rigorous intersection-number computation.

## Setup

- beta: 0.4
- anchor Gamma: 1.75
- anchor competitor search starts: 80
- Stokes tolerance: 0.08
- anti-Stokes tolerance: 0.02

## Crossing Summary

- anti-Stokes brackets: [[1.7, 1.75]]
- physical exact-match switch brackets: [[1.7, 1.75]]
- candidate simultaneous PL jump points: [1.7, 1.75, 1.8, 1.825]

## Branch Table

| Gamma | Delta Re Phi | |Delta Im Phi| mod 2pi | seed gap | comp gap | exact winner | PL candidate? |
|---:|---:|---:|---:|---:|---|---|
| 1.000000 | -0.0808354 | 0.295212 | 0.00625182 | 0.0745835 | seed |  |
| 1.300000 | -0.0605027 | 0.117418 | 0.00383454 | 0.0566682 | seed |  |
| 1.500000 | -0.0348169 | 0.0435855 | 0.00142514 | 0.0333917 | seed |  |
| 1.600000 | -0.0201889 | 0.0185286 | 0.00133865 | 0.0215276 | seed |  |
| 1.700000 | -0.0055487 | 0.00239672 | 0.00738977 | 0.0129385 | seed | yes |
| 1.750000 | +0.00158592 | 5.50753e-09 | 0.0122398 | 0.0106539 | competitor | yes |
| 1.800000 | +0.0126902 | 0.00181949 | 0.0189227 | 0.00623255 | competitor | yes |
| 1.825000 | +0.019733 | 0.0030806 | 0.0237584 | 0.00402543 | competitor | yes |
| 1.850000 | +0.0274966 | 0.00442198 | 0.0295218 | 0.00202517 | competitor |  |
| 1.875000 | +0.0358709 | 0.00580748 | 0.0361661 | 0.000295224 | competitor |  |
| 1.900000 | +0.0447758 | 0.00721598 | 0.043635 | 0.00114078 | competitor |  |
| 1.950000 | +0.0639438 | 0.0100526 | 0.0607778 | 0.00316608 | competitor |  |
| 2.000000 | +0.0846354 | 0.0128679 | 0.0803758 | 0.00425963 | competitor |  |
| 2.050000 | +0.106594 | 0.0156303 | 0.10188 | 0.00471412 | competitor |  |
| 2.100000 | +0.12963 | 0.0183232 | 0.124844 | 0.00478565 | competitor |  |

## Interpretation

- Start of homotopy assumption: at small Gamma, seed intersection number is 1 and this competitor branch is off.
- A necessary place for the competitor intersection number to change is a Stokes condition, `Delta Im Phi = 0 mod 2pi`.
- Exponential dominance changes at anti-Stokes, `Delta Re Phi = 0`.
- If the Stokes and anti-Stokes windows coincide, that is the strongest numerical evidence for the PL mechanism behind the observed physical transition.

## Figures

- `pl_homotopy_stokes_anti_stokes.png`
- `pl_homotopy_exact_gap_switch.png`
