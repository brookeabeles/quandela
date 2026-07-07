# BM24 saddle legitimate-findings audit

Date: 2026-07-03

This folder re-analyzes the push-groundbreaking saddle evidence with a stricter physical filter. The purpose is to separate certified algebraic roots from roots that can legitimately explain the BM24 success-probability exponent.

## Legitimacy rule

For the success probability, `0 <= E[p_succ] <= 1`, so the asymptotic exponent `lambda = log(E[p_succ]) / n` must satisfy `lambda <= 0`.

The saddle quantity used here is therefore not raw `Re Phi_M`; the physical comparison is:

`full_conv2_exponent = phi_pref + Re Phi_M`, with `phi_pref = -0.689609375`.

A candidate with `full_conv2_exponent > 0` is marked algebraic-only for `p_succ` dominance claims.

## Headline findings

- Anchor competitor roots checked: 965.
- Positive-full-exponent roots: 64; all 64 also miss the exact finite-n exponent by more than 0.1.
- High-Re bad exact matches: 85. These are certified algebraic roots, but they should not be used as physical dominance evidence.
- Strict competitor exact matches (`gap <= 0.02`): 3; strict and near-real-phase matches: 1.
- Branch 43/46 merged pair points: 148; positive-full points: 0.
- Branch 43/46 large-|gamma| RMSE: 0.000734286, compared with seed RMSE 1.31116.
- Main Re crossing of merged 43/46 against seed: gamma ~= -1.830782.

## What survives

1. The BM24 seed saddle remains the legitimate explanatory saddle through the QAOA-relevant window in the existing scans.
2. The branch 43/46 pair is legitimate at larger `|gamma|`: its full exponent is nonpositive, it is Krawczyk-certified along a robust tracked family, its phase is near the real locus over the plateau, and it matches the finite-n exact plateau far better than the seed.
3. The very high-`Re Phi_M` roots are real certified algebraic roots, but many imply `phi_pref + Re Phi_M > 0`. They should be reported as algebraic-sheet artifacts/nonphysical for `p_succ` unless future Picard-Lefschetz intersection data says otherwise and the exponent feasibility problem is resolved.

## Per-gamma anchor summary

| gamma | total competitors | positive full | physical feasible | strict exact | strict + near-real | best physical gap |
|---:|---:|---:|---:|---:|---:|---:|
| -0.30 | 87 | 2 | 85 | 0 | 0 | 0.071626 |
| -0.60 | 121 | 6 | 115 | 2 | 0 | 0.0186375 |
| -0.83 | 145 | 10 | 135 | 0 | 0 | 0.0919363 |
| -1.00 | 147 | 12 | 135 | 0 | 0 | 0.0236635 |
| -1.60 | 207 | 9 | 198 | 0 | 0 | 0.0220865 |
| -2.00 | 258 | 25 | 233 | 1 | 1 | 0.00284432 |

## Branch 43/46 checks

| branch | points | full min | full max | positive full | median phase mod 2pi | max phase mod 2pi |
|---:|---:|---:|---:|---:|---:|---:|
| 43 | 220 | -0.918315 | -0.588081 | 0 | 8.88178e-16 | 1.44871 |
| 46 | 231 | -0.689665 | -0.588923 | 0 | 8.88178e-16 | 0.154728 |

The pair should be merged for exponent-envelope claims and kept separate for prefactor or interference questions.

## Generated artifacts

- `legitimate_findings_summary.json`: machine-readable summary.
- `anchor_competitor_legitimacy_table.csv`: per-root anchor competitor classifications.
- `branch_43_46_merged_legitimacy_table.csv`: merged branch-pair evidence along gamma.
- `01_legitimacy_filter_funnel.png`: survival funnel under physical/exact filters.
- `02_per_gamma_legitimacy_counts.png`: per-gamma legitimacy tiers.
- `03_full_exponent_vs_exact_anchor_competitors.png`: full exponent vs exact finite-n lambda.
- `04_seed_vs_branch_43_46_legitimate_story.png`: seed-to-43/46 transition evidence.

## Caveats

This is still based on existing artifacts; no new root search was run. The filters establish physical feasibility and finite-n explanatory consistency, not a full Picard-Lefschetz intersection-number proof. The strongest defensible statement is therefore an empirical/explanatory saddle-transition claim, not a global contour-dominance theorem.
