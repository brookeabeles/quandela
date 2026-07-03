# PL homotopy beta sweep

## Question

Does the PL/Stokes alignment seen at beta_opt also occur at the other refined robust transition points?

## Method

For each robust single-switch beta slice, anchor on the best exact-matching competitor just after the refined transition, continue the same branch backward/forward in Gamma, remove points where it merges into the seed, and compare the continued-branch anti-Stokes bracket to the refined exact-exponent switch bracket.

- competitor starts at each anchor: 80
- tested robust single-switch slices: 7
- aligned slices: 7
- skipped ambiguous/nonmonotone slices: 2

## Results

| beta | refined switch | anti-Stokes brackets | overlap? | max phase dist near switch | merged small-Gamma rows |
|---:|---|---|---|---:|---:|
| 0.350000 | [1.6500, 1.7000] | [[1.6749999999999998, 1.7]] | yes | 0.011 | 14 |
| 0.400000 | [1.7000, 1.7500] | [[1.725, 1.75]] | yes | 0.153 | 3 |
| 0.450000 | [1.7500, 1.8000] | [[1.775, 1.8]] | yes | 0.00663 | 3 |
| 0.500000 | [1.8000, 1.8250] | [[1.8, 1.8125]] | yes | 0.00382 | 3 |
| 0.543400 | [1.8250, 1.8500] | [[1.825, 1.8375]] | yes | 0.0013 | 3 |
| 0.575000 | [1.8250, 1.8500] | [[1.8375, 1.85]] | yes | 0.00212 | 3 |
| 0.625000 | [1.8500, 1.8750] | [[1.8625, 1.875]] | yes | 0.0038 | 3 |

## Skipped

- beta=0.600000: 4 transition(s), not a robust single switch
- beta=0.650000: 3 transition(s), not a robust single switch

## Interpretation

- `overlap=yes` means the continued branch's anti-Stokes bracket intersects the exact-exponent switch bracket from the refined scan.
- This is still a necessary-condition PL test, not an intersection-number proof.
- Rows merged into the seed at small Gamma are explicitly filtered because they otherwise create fake Stokes/anti-Stokes events.

## Figure

- `pl_beta_sweep_alignment.png`
