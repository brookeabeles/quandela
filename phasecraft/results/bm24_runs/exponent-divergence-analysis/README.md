# Mean success vs median runtime exponent analysis

## What “tight instance-to-instance spread” means

At fixed n, each random k-SAT instance gives a success probability pᵢ.
**Tight spread** = all pᵢ are close (low variance / IQR). Then mean(p) ≈ median(p)
and median(1/p) ≈ 1/mean(p), so success and runtime exponents agree.

**Loose spread** = a tail of hard instances (small pᵢ). mean(p) is pulled down less
than median(p), but median(1/p) is dominated by typical hardness → runtime exponent
is steeper (larger) than |mean-p exponent|.

Proxy used here: `median(p) / mean(p)` — near 1 is tight, falling below ~0.8 is loose.

## Where to see mean-p vs median-rt training (06-09 run1)

| depth | eval slope (mean-p train) | eval slope (median-rt train) | Δ |
|------:|--------------------------:|-----------------------------:|--:|
| 2 | 0.6732 | 0.6733 | +0.0001 |
| 5 | 0.5628 | 0.5789 | +0.0161 |
| 8 | 0.5224 | 0.5258 | +0.0034 |
| 10 | 0.5132 | 0.5187 | +0.0055 |
| 15 | 0.4764 | 0.4631 | -0.0133 |
| 20 | 0.4219 | 0.4202 | -0.0017 |
| 30 | 0.3930 | 0.3935 | +0.0005 |
| 40 | 0.3687 | 0.3827 | +0.0140 |
| 50 | 0.3603 | 0.3710 | +0.0107 |

File: `results/bm24_runs/06-09/run1/06-09_1649-train12-tr100-te200-n12-18.json`
Keys: `trace_bm24_mean_p_fixed_n` vs `trace_median_runtime_fixed_n`, field `lr_log2_slope`.

## Benchmark exponent gap (05-29 depth-15)

- slope ln(mean p): -0.4208
- slope ln(median 1/p): +0.4633
- gap (median_rt + mean_p): +0.0425 log₂
- source: `05-29_1630-bench`

### All benchmarks with per_n data
| source | n range | |mean_p| | median_rt | gap |
|--------|---------|--------:|----------:|----:|
| 05-29_1630-bench | 12–20 | 0.4208 | 0.4633 | +0.0425 |
| 05-19-10-58-21-bench | 10–12 | 0.5188 | 0.6176 | +0.0988 |
