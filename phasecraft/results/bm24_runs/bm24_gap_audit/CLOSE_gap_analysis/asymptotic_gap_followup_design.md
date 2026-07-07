# Follow-Up Design For The Asymptotic Gap Question

The current data identify the finite-window mechanism but do not resolve the limit.
This design asks what larger-n runs would actually discriminate persistence from flexible decay-to-zero behavior.

## Core Implication

No finite dataset can rule out an arbitrarily slow decay such as `Delta_n = a n^{-alpha}` with `alpha` extremely close to zero.
Therefore the asymptotic question needs either a theoretical lower bound on the allowed decay rate or a pre-registered finite family of zero-limit alternatives.

## Forecast Table

| train_n | depth | n | positive-limit pred | best zero-limit model | zero-limit pred | separation | rough N for 2-sigma separation |
|---:|---:|---:|---:|---|---:|---:|---:|
| 12 | 20 | 21 | 0.0366 | `curved_zero_limit_decay` | 0.0337 | 0.0029 | 3067 |
| 12 | 20 | 22 | 0.0373 | `curved_zero_limit_decay` | 0.0333 | 0.0039 | 1679 |
| 12 | 20 | 23 | 0.0378 | `curved_zero_limit_decay` | 0.0329 | 0.0050 | 1055 |
| 12 | 20 | 24 | 0.0384 | `curved_zero_limit_decay` | 0.0324 | 0.0060 | 725 |
| 12 | 20 | 25 | 0.0389 | `curved_zero_limit_decay` | 0.0319 | 0.0070 | 531 |
| 12 | 20 | 26 | 0.0393 | `curved_zero_limit_decay` | 0.0313 | 0.0080 | 408 |
| 12 | 20 | 28 | 0.0401 | `curved_zero_limit_decay` | 0.0302 | 0.0099 | 266 |
| 12 | 50 | 21 | 0.0411 | `curved_zero_limit_decay` | 0.0370 | 0.0042 | 1870 |
| 12 | 50 | 22 | 0.0428 | `curved_zero_limit_decay` | 0.0370 | 0.0057 | 997 |
| 12 | 50 | 23 | 0.0442 | `curved_zero_limit_decay` | 0.0370 | 0.0073 | 617 |
| 12 | 50 | 24 | 0.0456 | `curved_zero_limit_decay` | 0.0368 | 0.0088 | 419 |
| 12 | 50 | 25 | 0.0468 | `curved_zero_limit_decay` | 0.0365 | 0.0103 | 304 |
| 12 | 50 | 26 | 0.0480 | `curved_zero_limit_decay` | 0.0362 | 0.0118 | 232 |
| 12 | 50 | 28 | 0.0501 | `curved_zero_limit_decay` | 0.0353 | 0.0147 | 150 |
| 16 | 20 | 21 | 0.0342 | `curved_zero_limit_decay` | 0.0317 | 0.0025 | 3840 |
| 16 | 20 | 22 | 0.0346 | `curved_zero_limit_decay` | 0.0313 | 0.0034 | 2080 |
| 16 | 20 | 23 | 0.0350 | `curved_zero_limit_decay` | 0.0308 | 0.0043 | 1298 |
| 16 | 20 | 24 | 0.0354 | `curved_zero_limit_decay` | 0.0303 | 0.0052 | 888 |
| 16 | 20 | 25 | 0.0358 | `curved_zero_limit_decay` | 0.0297 | 0.0060 | 648 |
| 16 | 20 | 26 | 0.0361 | `curved_zero_limit_decay` | 0.0292 | 0.0069 | 496 |
| 16 | 20 | 28 | 0.0367 | `curved_zero_limit_decay` | 0.0281 | 0.0086 | 322 |
| 16 | 50 | 21 | 0.0390 | `curved_zero_limit_decay` | 0.0349 | 0.0040 | 2098 |
| 16 | 50 | 22 | 0.0404 | `curved_zero_limit_decay` | 0.0350 | 0.0055 | 1136 |
| 16 | 50 | 23 | 0.0418 | `curved_zero_limit_decay` | 0.0349 | 0.0069 | 709 |
| 16 | 50 | 24 | 0.0431 | `curved_zero_limit_decay` | 0.0347 | 0.0084 | 485 |
| 16 | 50 | 25 | 0.0442 | `curved_zero_limit_decay` | 0.0344 | 0.0098 | 354 |
| 16 | 50 | 26 | 0.0453 | `curved_zero_limit_decay` | 0.0341 | 0.0112 | 271 |
| 16 | 50 | 28 | 0.0472 | `curved_zero_limit_decay` | 0.0333 | 0.0139 | 176 |

The `N` column is only a statistical guide based on current bootstrap noise. Exact simulation cost still grows rapidly with `n`.

## Runnable Stages

### stage1_cpu_probe

Cheap extension beyond n=20; useful sanity check before committing GPU time.

```bash
python experiments/lr_scaling/evaluate.py --angles results/bm24_runs/bm24_gap_audit/angles_grid_existing.json --output results/bm24_runs/bm24_gap_audit/followup_gap_persistence_cpu_probe_n21-22.csv --n-values 21,22 --N 500 --eval-seed 100027 --objectives mean_p --train-ns 12,16 --depths 20,50 --m-sampling notebook
```

### stage2_gpu_decision_window

First genuinely discriminating window; p=50 and n=24 should already separate positive-limit and curved-decay fits at current N.

```bash
python experiments/lr_scaling/evaluate.py --angles results/bm24_runs/bm24_gap_audit/angles_grid_existing.json --output results/bm24_runs/bm24_gap_audit/followup_gap_persistence_gpu_decision_n23-24.csv --n-values 23,24 --N 500 --eval-seed 100027 --objectives mean_p --train-ns 12,16 --depths 20,50 --m-sampling notebook --require-gpu
```

### stage3_larger_n_if_still_ambiguous

Push to the next window only if n=23..24 remains compatible with both persistence and decay.

```bash
python experiments/lr_scaling/evaluate.py --angles results/bm24_runs/bm24_gap_audit/angles_grid_existing.json --output results/bm24_runs/bm24_gap_audit/followup_gap_persistence_larger_n25-28.csv --n-values 25,26,28 --N 500 --eval-seed 100027 --objectives mean_p --train-ns 12,16 --depths 20,50 --m-sampling notebook --require-gpu
```

### replicate_eval_seed_200027

Check whether the first extension is eval-root stable before spending more GPU time.

```bash
python experiments/lr_scaling/evaluate.py --angles results/bm24_runs/bm24_gap_audit/angles_grid_existing.json --output results/bm24_runs/bm24_gap_audit/followup_gap_persistence_replicate_seed200027_n21-22.csv --n-values 21,22 --N 500 --eval-seed 200027 --objectives mean_p --train-ns 12,16 --depths 20,50 --m-sampling notebook
```

### replicate_eval_seed_300027

Check whether the first extension is eval-root stable before spending more GPU time.

```bash
python experiments/lr_scaling/evaluate.py --angles results/bm24_runs/bm24_gap_audit/angles_grid_existing.json --output results/bm24_runs/bm24_gap_audit/followup_gap_persistence_replicate_seed300027_n21-22.csv --n-values 21,22 --N 500 --eval-seed 300027 --objectives mean_p --train-ns 12,16 --depths 20,50 --m-sampling notebook
```

### merge_current_and_followup_rows

Create the single CSV consumed by the diagnostics.

```bash
python experiments/lr_scaling/merge_gap_audit_csvs.py results/bm24_runs/bm24_gap_audit/N500_n12-20_all.csv results/bm24_runs/bm24_gap_audit/followup_gap_persistence_cpu_probe_n21-22.csv results/bm24_runs/bm24_gap_audit/followup_gap_persistence_gpu_decision_n23-24.csv results/bm24_runs/bm24_gap_audit/followup_gap_persistence_larger_n25-28.csv results/bm24_runs/bm24_gap_audit/followup_gap_persistence_replicate_seed200027_n21-22.csv results/bm24_runs/bm24_gap_audit/followup_gap_persistence_replicate_seed300027_n21-22.csv --output results/bm24_runs/bm24_gap_audit/followup_gap_persistence_combined_with_current.csv --dedupe --ignore-missing
```

### reanalyze_combined_rows

Regenerate mechanism, gate, and finite-size scaling diagnostics on the combined data.

```bash
python experiments/lr_scaling/run_gap_persistence_analysis.py --csv results/bm24_runs/bm24_gap_audit/followup_gap_persistence_combined_with_current.csv --output-dir results/bm24_runs/bm24_gap_audit/CLOSE_gap_analysis/followup_reanalysis --tail-n-min 20 --window-width 9
```

## Decision Rule

After collecting each new shard, merge the current and follow-up rows, then run the three reanalysis stages above on the combined CSV.
Evidence for persistence strengthens only if later windows keep the gap/Jensen/tilted-integral positive and make the flexible zero-limit models worse across multiple eval roots.
Evidence against persistence appears if the gap, Jensen component, tilted integral, or scaled variance trends toward zero and the zero-limit models become stable winners.
