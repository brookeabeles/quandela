# PL homotopy branch tracker

## Verdict

This is the first practical homotopy-style PL diagnostic: it follows one competitor branch from the post-crossing region back toward small Gamma and asks where a Stokes jump could plausibly turn on its intersection number.  It is still not a rigorous intersection-number computation.

## Setup

- beta: 0.35
- anchor Gamma: 1.75
- anchor competitor search starts: 120
- Stokes tolerance: 0.08
- anti-Stokes tolerance: 0.02

## Crossing Summary

- anti-Stokes brackets: []
- physical exact-match switch brackets: []
- candidate simultaneous PL jump points: [1.7, 1.75]

## Branch Table

| Gamma | Delta Re Phi | |Delta Im Phi| mod 2pi | seed gap | comp gap | exact winner | PL candidate? |
|---:|---:|---:|---:|---:|---|---|
| 1.700000 | +0.000655942 | 8.88178e-16 | 0.0114466 | 0.0107907 | competitor | yes |
| 1.750000 | +0.0112005 | 0.00192978 | 0.0173466 | 0.0061461 | competitor | yes |
| 1.800000 | +0.0259358 | 0.00473103 | 0.0274205 | 0.00148474 | competitor |  |
| 1.825000 | +0.034355 | 0.00621168 | 0.0339041 | 0.00045086 | competitor |  |
| 1.850000 | +0.0433395 | 0.00771112 | 0.0412843 | 0.00205511 | competitor |  |
| 1.875000 | +0.0528194 | 0.00921571 | 0.0494941 | 0.00332533 | competitor |  |
| 1.900000 | +0.0627394 | 0.0107164 | 0.0584559 | 0.00428347 | competitor |  |
| 1.950000 | +0.0837261 | 0.0136836 | 0.0783077 | 0.00541832 | competitor |  |
| 2.000000 | +0.106017 | 0.0165825 | 0.100212 | 0.0058052 | competitor |  |
| 2.050000 | +0.129406 | 0.0193983 | 0.12365 | 0.0057552 | competitor |  |
| 2.100000 | +0.153732 | 0.0221244 | 0.148243 | 0.00548844 | competitor |  |

## Interpretation

- Start of homotopy assumption: at small Gamma, seed intersection number is 1 and this competitor branch is off.
- A necessary place for the competitor intersection number to change is a Stokes condition, `Delta Im Phi = 0 mod 2pi`.
- Exponential dominance changes at anti-Stokes, `Delta Re Phi = 0`.
- If the Stokes and anti-Stokes windows coincide, that is the strongest numerical evidence for the PL mechanism behind the observed physical transition.

## Figures

- `pl_homotopy_stokes_anti_stokes.png`
- `pl_homotopy_exact_gap_switch.png`
