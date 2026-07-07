# Asymptotic Persistence Gate Evaluation

CSV: `results/bm24_runs/bm24_gap_audit/N500_n12-20_all.csv`
Protocol: `results/bm24_runs/bm24_gap_audit/CLOSE_gap_analysis/asymptotic_persistence_protocol.json`

## Overall

```json
{
  "row_count": 4,
  "stable_row_count": 4,
  "stable_distinct_windows": [
    {
      "lo": 12,
      "hi": 20
    }
  ],
  "passes_current_rows": true,
  "passes_protocol_persistence": false,
  "reason": "insufficient number of stable late windows for asymptotic-persistence evidence"
}
```

## Rows

| window | train_n | depth | Gap | gap CI | Jensen | Skew | tilted J | Var1/Var0 | top1 | top5 | LOO | row pass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 12-20 | 12 | 20 | 0.0498 | [0.0299, 0.0661] | 0.0427 | 0.0071 | 0.0427 | 0.5217 | 0.055 | 0.201 | 0.0009 | yes |
| 12-20 | 12 | 50 | 0.0734 | [0.0571, 0.0907] | 0.0685 | 0.0050 | 0.0685 | 0.3583 | 0.050 | 0.198 | 0.0007 | yes |
| 12-20 | 16 | 20 | 0.0428 | [0.0268, 0.0595] | 0.0398 | 0.0031 | 0.0398 | 0.5391 | 0.054 | 0.198 | 0.0009 | yes |
| 12-20 | 16 | 50 | 0.0698 | [0.0534, 0.0841] | 0.0632 | 0.0066 | 0.0632 | 0.3810 | 0.049 | 0.196 | 0.0007 | yes |
