# Certified root census

## Scope

This is a repeated-budget numerical census of certified roots of the BM24 p=1 saddle equations. Each discovered root is polished and Krawczyk-certified.  The test is strong evidence for local root-discovery stability, but it is not a Picard-Lefschetz contour/intersection proof and therefore not a mathematical completeness theorem.

## Parameters

- beta: `0.5433996420760803`
- Gamma values: `[1.75, 1.825, 1.85, 2.0]`
- budgets: `[300, 800, 1600]`
- n values for exact exponent: `[18, 19, 20, 21, 22, 23, 24]`

## Results

| Gamma | certified competitors by budget | new roots at final budget | max gap vs seed | best exact match | seed exact gap | best exact gap | high-Re decoys | status |
|---:|---:|---:|---:|---|---:|---:|---:|---|
| 1.750000 | [57, 125, 177] | 96 | +4.789853 | seed | 0.00677653 | 0.00677653 | 27 / 27 | not_saturated |
| 1.825000 | [64, 106, 174] | 101 | +3.941716 | seed | 0.0146103 | 0.0146103 | 25 / 25 | not_saturated |
| 1.850000 | [67, 120, 173] | 95 | +4.689314 | competitor | 0.0179472 | 0.0146365 | 23 / 23 | not_saturated |
| 2.000000 | [65, 116, 169] | 91 | +3.579937 | competitor | 0.050258 | 0.0036599 | 27 / 28 | not_saturated |

## Interpretation

The important column is not the largest discovered real action by itself.  The exact-exponent match separates physical candidates from high-Re decoys.  If root counts and top gaps stabilize with budget, the census supports the claim that the observed seed/competitor transition is not a random-start artifact.

## Figures

- `root_count_saturation.png`
- `max_re_gap_saturation.png`
