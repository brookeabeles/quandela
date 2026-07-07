# Rate-admissible refined saddle audit

This redraws the refined transition figures after filtering certified competitors whose standalone conv2 exponent `E` is positive. Saddles with `E > 0` were excluded from the rate-admissible single-saddle set. Such saddles could only be compatible with a bounded observable if their thimble coefficient vanishes or exponentially precise cancellations occur.

Important caveat: these figures compare certified saddle rates and necessary Stokes/anti-Stokes-style conditions.  They do not compute thimble intersection numbers or prove that a competitor controls the original BM24 contour.

Panel (a) colors `D_24 = |lambda_comp - lambda_24| - |lambda_seed - lambda_24|`, using the existing `n=max(n_values)=24` finite-n reference from the refined scan. The finite-`n` stability section below recomputes `lambda_n` for `n=18..24` on the same tracked competitor branch.

The purple connecting curve is a guide to the eye through separately bracketed finite-n crossover slices, not a certified two-parameter branch continuation.

## Filter

- possible exponent criterion: `E <= 0.0`
- beta points: 9
- Gamma points: 16
- competitor starts per recomputed point: 200
- certified competitor roots recomputed: 6965
- possible-exponent competitors kept: 6682
- positive-exponent competitors excluded as single-saddle controllers: 283
- grid points where the all-root max-Re controller was removed by the filter: 120 / 144

Search completeness caveat: Krawczyk certification applies to roots that were found.  The finite random-start competitor search is not a proof that every saddle root was discovered.

## Classification after filter

- seed gives smaller rate residual cleanly: 3
- seed gives smaller rate residual; only excluded high-Re saddles challenged it before filtering: 18
- seed gives smaller rate residual despite certified rate-admissible challengers: 58
- rate-admissible competitor gives smaller rate residual: 65

## Transition estimates after filter

| beta | transition interval in Gamma | midpoint | notes |
|---:|---:|---:|---|
| 0.350000 | 1.6500-1.7000 | 1.6750 | 3 possible-decoy, 0 impossible-only-decoy, 11 possible-competitor points |
| 0.400000 | 1.7000-1.7500 | 1.7250 | 5 possible-decoy, 1 impossible-only-decoy, 10 possible-competitor points |
| 0.450000 | 1.7500-1.8000 | 1.7750 | 6 possible-decoy, 1 impossible-only-decoy, 9 possible-competitor points |
| 0.500000 | 1.8000-1.8250 | 1.8125 | 7 possible-decoy, 1 impossible-only-decoy, 8 possible-competitor points |
| 0.543400 | 1.8250-1.8500 | 1.8375 | 7 possible-decoy, 1 impossible-only-decoy, 7 possible-competitor points |
| 0.575000 | 1.8250-1.8500 | 1.8375 | 6 possible-decoy, 3 impossible-only-decoy, 7 possible-competitor points |
| 0.600000 | 1.6500-1.7000, 1.7000-1.7500, 1.8500-1.8750, 1.8750-1.9000 | 1.6750 | 10 possible-decoy, 4 impossible-only-decoy, 2 possible-competitor points |
| 0.625000 | 1.8500-1.8750 | 1.8625 | 7 possible-decoy, 3 impossible-only-decoy, 6 possible-competitor points |
| 0.650000 | 1.8750-1.9000 | 1.8875 | 7 possible-decoy, 4 impossible-only-decoy, 5 possible-competitor points |

## Competitor identity check

This compares panel (a)'s best discovered rate-admissible competitor on the competitor side of the crossover with panel (b)'s tracked competitor branch at the nearest stored Gamma.  The current distance is a direct z-coordinate sup-norm; explicit minimization over saddle-coordinate symmetries is not yet implemented.

| beta | Gamma_rate | Gamma_tracked | d_z | classification |
|---:|---:|---:|---:|---|
| 0.350000 | 1.7000 | 1.7000 | 3.005e-14 | same numerical root |
| 0.400000 | 1.7500 | 1.7500 | 2.459e-14 | same numerical root |
| 0.450000 | 1.8000 | 1.8000 | 3.619e-14 | same numerical root |
| 0.500000 | 1.8250 | 1.8250 | 1.575e-14 | same numerical root |
| 0.543400 | 1.8500 | 1.8500 | 1.808e-14 | same numerical root |
| 0.575000 | 1.8500 | 1.8500 | 9.607e-15 | same numerical root |
| 0.600000 | 1.7000 | nan | nan | needs recomputation with stored z-coordinates |
| 0.625000 | 1.8750 | 1.8750 | 3.743e-14 | same numerical root |
| 0.650000 | 1.9000 | 1.9000 | 1.603e-14 | same numerical root |

## Finite-n crossover stability

This uses the same validated tracked competitor branch for all `n`; it does not reselect the best competitor independently at each finite size and does not rerun saddle searches.  Only `lambda_n = n^{-1} log |P_n|` is recomputed.

| beta | Gamma_cross(18) | Gamma_cross(19) | Gamma_cross(20) | Gamma_cross(21) | Gamma_cross(22) | Gamma_cross(23) | Gamma_cross(24) | range |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.350000 | <= 1.7000 | <= 1.7000 | <= 1.7000 | <= 1.7000 | <= 1.7000 | <= 1.7000 | <= 1.7000 | not bracketed |
| 0.400000 | 1.7389 | 1.7389 | 1.7389 | 1.7389 | 1.7389 | 1.7389 | 1.7389 | 0.0000 |
| 0.450000 | 1.7753 | 1.7753 | 1.7753 | 1.7753 | 1.7753 | 1.7753 | 1.7753 | 0.0000 |
| 0.500000 | 1.8096 | 1.8096 | 1.8096 | 1.8096 | 1.8096 | 1.8096 | 1.8096 | 0.0000 |
| 0.543400 | 1.8332 | 1.8332 | 1.8332 | 1.8332 | 1.8332 | 1.8332 | 1.8332 | 0.0000 |
| 0.575000 | 1.8440 | 1.8440 | 1.8440 | 1.8440 | 1.8440 | 1.8440 | 1.8440 | 0.0000 |
| 0.625000 | 1.8678 | 1.8678 | 1.8678 | 1.8678 | 1.8678 | 1.8678 | 1.8678 | 0.0000 |
| 0.650000 | 1.8766 | 1.8766 | 1.8766 | 1.8766 | 1.8766 | 1.8766 | 1.8766 | 0.0000 |

Nonmonotone rate slices kept separate from this table: `0.600000`.  No smooth crossover boundary is inferred through those slices.

`<=` entries are left-censored by the tracked-branch cache: the first valid tracked-competitor row is already competitor-favored, so the crossing lies at or below the displayed Gamma.
`*` marks a first positive-to-negative crossing in a series with multiple sign changes.

## Absolute-value mechanism audit

This audits why the finite-n rate-residual crossover agrees with the equal-real-action condition on the clean tracked branch.  At each displayed crossing, `lambda_n` lies outside the interval between the two saddle rates, so the absolute-value residual equation collapses algebraically to equality of the two saddle rates.  Thus the observed `Gamma_cross(n) = Gamma_AS` is an algebraic consequence in this regime, not an independent finite-n numerical coincidence.

| beta | n | E_seed | E_comp | lambda_n | ordering | D_n mechanism |
|---:|---:|---:|---:|---:|---|---|
| 0.400000 | 18 | -0.67784546 | -0.67784546 | -0.66173365 | lambda_n above both saddle rates | D_n = E_seed - E_comp |
| 0.400000 | 19 | -0.67784546 | -0.67784546 | -0.66332714 | lambda_n above both saddle rates | D_n = E_seed - E_comp |
| 0.400000 | 20 | -0.67784546 | -0.67784546 | -0.66456314 | lambda_n above both saddle rates | D_n = E_seed - E_comp |
| 0.400000 | 21 | -0.67784546 | -0.67784546 | -0.66548705 | lambda_n above both saddle rates | D_n = E_seed - E_comp |
| 0.400000 | 22 | -0.67784546 | -0.67784546 | -0.6661435 | lambda_n above both saddle rates | D_n = E_seed - E_comp |
| 0.400000 | 23 | -0.67784546 | -0.67784546 | -0.66657572 | lambda_n above both saddle rates | D_n = E_seed - E_comp |
| 0.400000 | 24 | -0.67784546 | -0.67784546 | -0.66682478 | lambda_n above both saddle rates | D_n = E_seed - E_comp |
| 0.450000 | 18 | -0.67273222 | -0.67273222 | -0.65744718 | lambda_n above both saddle rates | D_n = E_seed - E_comp |
| 0.450000 | 19 | -0.67273222 | -0.67273222 | -0.65865772 | lambda_n above both saddle rates | D_n = E_seed - E_comp |
| 0.450000 | 20 | -0.67273222 | -0.67273222 | -0.65952839 | lambda_n above both saddle rates | D_n = E_seed - E_comp |
| 0.450000 | 21 | -0.67273222 | -0.67273222 | -0.66011678 | lambda_n above both saddle rates | D_n = E_seed - E_comp |
| 0.450000 | 22 | -0.67273222 | -0.67273222 | -0.6604769 | lambda_n above both saddle rates | D_n = E_seed - E_comp |
| 0.450000 | 23 | -0.67273222 | -0.67273222 | -0.6606583 | lambda_n above both saddle rates | D_n = E_seed - E_comp |
| 0.450000 | 24 | -0.67273222 | -0.67273222 | -0.66070528 | lambda_n above both saddle rates | D_n = E_seed - E_comp |
| 0.500000 | 18 | -0.67010742 | -0.67010742 | -0.65390731 | lambda_n above both saddle rates | D_n = E_seed - E_comp |
| 0.500000 | 19 | -0.67010742 | -0.67010742 | -0.65481956 | lambda_n above both saddle rates | D_n = E_seed - E_comp |
| 0.500000 | 20 | -0.67010742 | -0.67010742 | -0.65542925 | lambda_n above both saddle rates | D_n = E_seed - E_comp |
| 0.500000 | 21 | -0.67010742 | -0.67010742 | -0.65580074 | lambda_n above both saddle rates | D_n = E_seed - E_comp |
| 0.500000 | 22 | -0.67010742 | -0.67010742 | -0.65599142 | lambda_n above both saddle rates | D_n = E_seed - E_comp |
| 0.500000 | 23 | -0.67010742 | -0.67010742 | -0.65605111 | lambda_n above both saddle rates | D_n = E_seed - E_comp |
| 0.500000 | 24 | -0.67010742 | -0.67010742 | -0.65602186 | lambda_n above both saddle rates | D_n = E_seed - E_comp |
| 0.543400 | 18 | -0.6679447 | -0.6679447 | -0.65060262 | lambda_n above both saddle rates | D_n = E_seed - E_comp |
| 0.543400 | 19 | -0.6679447 | -0.6679447 | -0.65134441 | lambda_n above both saddle rates | D_n = E_seed - E_comp |
| 0.543400 | 20 | -0.6679447 | -0.6679447 | -0.65182395 | lambda_n above both saddle rates | D_n = E_seed - E_comp |
| 0.543400 | 21 | -0.6679447 | -0.6679447 | -0.65210609 | lambda_n above both saddle rates | D_n = E_seed - E_comp |
| 0.543400 | 22 | -0.6679447 | -0.6679447 | -0.65224626 | lambda_n above both saddle rates | D_n = E_seed - E_comp |
| 0.543400 | 23 | -0.6679447 | -0.6679447 | -0.65229054 | lambda_n above both saddle rates | D_n = E_seed - E_comp |
| 0.543400 | 24 | -0.6679447 | -0.6679447 | -0.6522761 | lambda_n above both saddle rates | D_n = E_seed - E_comp |
| 0.575000 | 18 | -0.66617156 | -0.66617156 | -0.64752075 | lambda_n above both saddle rates | D_n = E_seed - E_comp |
| 0.575000 | 19 | -0.66617156 | -0.66617156 | -0.64819322 | lambda_n above both saddle rates | D_n = E_seed - E_comp |
| 0.575000 | 20 | -0.66617156 | -0.66617156 | -0.64863311 | lambda_n above both saddle rates | D_n = E_seed - E_comp |
| 0.575000 | 21 | -0.66617156 | -0.66617156 | -0.64890272 | lambda_n above both saddle rates | D_n = E_seed - E_comp |
| 0.575000 | 22 | -0.66617156 | -0.66617156 | -0.64905383 | lambda_n above both saddle rates | D_n = E_seed - E_comp |
| 0.575000 | 23 | -0.66617156 | -0.66617156 | -0.64912829 | lambda_n above both saddle rates | D_n = E_seed - E_comp |
| 0.575000 | 24 | -0.66617156 | -0.66617156 | -0.64915893 | lambda_n above both saddle rates | D_n = E_seed - E_comp |
| 0.625000 | 18 | -0.66581861 | -0.66581861 | -0.64548158 | lambda_n above both saddle rates | D_n = E_seed - E_comp |
| 0.625000 | 19 | -0.66581861 | -0.66581861 | -0.64609295 | lambda_n above both saddle rates | D_n = E_seed - E_comp |
| 0.625000 | 20 | -0.66581861 | -0.66581861 | -0.64651167 | lambda_n above both saddle rates | D_n = E_seed - E_comp |
| 0.625000 | 21 | -0.66581861 | -0.66581861 | -0.64679532 | lambda_n above both saddle rates | D_n = E_seed - E_comp |
| 0.625000 | 22 | -0.66581861 | -0.66581861 | -0.64698988 | lambda_n above both saddle rates | D_n = E_seed - E_comp |
| 0.625000 | 23 | -0.66581861 | -0.66581861 | -0.64713098 | lambda_n above both saddle rates | D_n = E_seed - E_comp |
| 0.625000 | 24 | -0.66581861 | -0.66581861 | -0.64724532 | lambda_n above both saddle rates | D_n = E_seed - E_comp |
| 0.650000 | 18 | -0.66588203 | -0.66588203 | -0.64440444 | lambda_n above both saddle rates | D_n = E_seed - E_comp |
| 0.650000 | 19 | -0.66588203 | -0.66588203 | -0.64501336 | lambda_n above both saddle rates | D_n = E_seed - E_comp |
| 0.650000 | 20 | -0.66588203 | -0.66588203 | -0.64544687 | lambda_n above both saddle rates | D_n = E_seed - E_comp |
| 0.650000 | 21 | -0.66588203 | -0.66588203 | -0.64575936 | lambda_n above both saddle rates | D_n = E_seed - E_comp |
| 0.650000 | 22 | -0.66588203 | -0.66588203 | -0.64599344 | lambda_n above both saddle rates | D_n = E_seed - E_comp |
| 0.650000 | 23 | -0.66588203 | -0.66588203 | -0.64618155 | lambda_n above both saddle rates | D_n = E_seed - E_comp |
| 0.650000 | 24 | -0.66588203 | -0.66588203 | -0.64634752 | lambda_n above both saddle rates | D_n = E_seed - E_comp |

## Equal-real-action endpoint refinement

These are equal-real-action brackets with Krawczyk-certified endpoint saddles.  The saddle roots at the endpoints are certified; the scalar Gamma root is not claimed to be rigorously interval-certified because no interval proof in Gamma is implemented.

| beta | Gamma_lo | Gamma_hi | midpoint | width | f_AS(lo) | f_AS(hi) | status |
|---:|---:|---:|---:|---:|---:|---:|---|
| 0.400000 | 1.746094 | 1.746875 | 1.746484 | 7.81e-04 | -1.041e-04 | +4.441e-16 | ok |
| 0.450000 | 1.782031 | 1.782812 | 1.782422 | 7.81e-04 | -3.537e-04 | +4.441e-16 | ok |
| 0.500000 | 1.816406 | 1.817188 | 1.816797 | 7.81e-04 | -3.830e-04 | +1.272e-03 | ok |
| 0.543400 | 1.832812 | 1.833594 | 1.833203 | 7.81e-04 | -2.178e-04 | +9.083e-04 | ok |
| 0.575000 | 1.848438 | 1.849219 | 1.848828 | 7.81e-04 | -6.172e-05 | +9.162e-05 | ok |
| 0.625000 | 1.872656 | 1.873438 | 1.873047 | 7.81e-04 | -9.913e-04 | +1.027e-03 | ok |
| 0.650000 | 1.876562 | 1.877344 | 1.876953 | 7.81e-04 | -1.440e-03 | +1.409e-04 | ok |

## Interval comparisons

Because `D_24 = 0` is algebraically equivalent to `Delta Re Phi = 0` in the audited ordering regime, no redundant second bisection was run for `I_cross,24`; the comparison below uses the local sign-change brackets.

| beta | I_AS | I_cross,24 | overlap? | midpoint distance |
|---:|---|---|---|---:|
| 0.400000 | [1.7000,1.7500] | [1.7000,1.7500] | yes | 0 |
| 0.450000 | [1.7500,1.8000] | [1.7500,1.8000] | yes | 0 |
| 0.500000 | [1.8000,1.8250] | [1.8000,1.8250] | yes | 0 |
| 0.543400 | [1.8250,1.8500] | [1.8250,1.8500] | yes | 0 |
| 0.575000 | [1.8250,1.8500] | [1.8250,1.8500] | yes | 0 |
| 0.625000 | [1.8500,1.8750] | [1.8500,1.8750] | yes | 0 |
| 0.650000 | [1.8750,1.9000] | [1.8750,1.9000] | yes | 0 |

| beta | I_AS | I_S | overlap? | interval separation |
|---:|---|---|---|---:|
| 0.400000 | [1.7000,1.7500] | [1.8250,1.8500] | no | 0.075 |
| 0.450000 | [1.7500,1.8000] | [1.8250,1.8500] | no | 0.025 |
| 0.500000 | [1.8000,1.8250] | [1.8250,1.8250] | yes | 0 |
| 0.543400 | [1.8250,1.8500] | [1.8250,1.8250] | yes | 0 |
| 0.575000 | [1.8250,1.8500] | [1.8000,1.8250] | yes | 0 |
| 0.625000 | [1.8500,1.8750] | [1.8000,1.8250] | no | 0.025 |
| 0.650000 | [1.8750,1.9000] | [1.8000,1.8250] | no | 0.05 |

## Phase-family anchor robustness

Anchors tested: `0.500000, 0.543400, 0.575000`.
The three anchors do not select one common phase-alignment family, so the red connecting family should not be interpreted as established.
Agreement at `beta_opt` is not independent because the original family was selected by proximity to the equal-real-action candidate there.

| beta | selected Gamma spread across anchors | same family? |
|---:|---:|---|
| 0.350000 | 0.000e+00 | yes |
| 0.400000 | 0.000e+00 | yes |
| 0.450000 | 0.000e+00 | yes |
| 0.500000 | 1.713e-02 | no |
| 0.543400 | 2.500e-02 | no |
| 0.575000 | 2.826e-02 | no |
| 0.625000 | 3.125e-02 | no |
| 0.650000 | 5.764e-02 | no |

## Beta 0.35 left-censor check

Status: `continued_branch_merged_with_seed`.

| Gamma | f_AS | comp certified? | comp residual | z-step distance |
|---:|---:|---|---:|---:|
| 1.6500 | +0.000e+00 | yes | 3.553e-15 | 1.911e+00 |

## Phase-alignment candidate check

The red phase-family connection is suppressed because the selected phase family is not anchor-robust.  The figure therefore shows phase-alignment roots as pale gray candidates only.

## Figures

- `separated_crossover_and_pl_conditions.png`
- `separated_crossover_and_pl_conditions.pdf`
- `possible_exponent_transition_map_with_pl.png`
- `possible_exponent_boundary_curve_with_pl.png`
- `possible_vs_impossible_competitor_counts.png`
- `possible_filter_removed_rate_and_exact_gap.png`
- `finite_n_crossover_stability.png`

## Final Figure Caption

Saddle crossover beyond the small-gamma regime. Panel (a) shows the rate-residual diagnostic `D_24 = |E_comp-lambda_24| - |E_seed-lambda_24|`; plotted intervals are brackets from the sampled Gamma grid and lines are guides between branch-resolved slices. In the audited ordering regime, `D_24=0` coincides algebraically with equal real action. Panel (b) shows branch-resolved equal-real-action candidates and numerical phase-alignment candidates. The beta=0.60 inset shows `f_AS(Gamma)` for the continued side-anchored branches; the old nonmonotone marker is resolved as a branch-selection artifact, not a certified single-branch continuation. PL phase points are necessary-condition candidates only; no thimble intersection numbers are computed.
