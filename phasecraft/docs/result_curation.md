# Result Curation

`results/best/` is for thesis-ready artifacts, not merely good-looking plots.

## Promotion Rule

Promote a file into `results/best/` only when it has:

1. A source run folder.
2. A generating script or notebook.
3. A short interpretation.
4. A known thesis claim or open question it supports.

## Keeping Curated Train Plots Fresh

The managed LR-scaling train plots in `results/best/` should be regenerated
from their source JSONs whenever the original run or shared classical baseline
changes:

```bash
python experiments/lr_scaling/sync_best.py
```

To check without rewriting files:

```bash
python experiments/lr_scaling/sync_best.py --check
```

The script currently manages:

- `train12-tr100-te200-n12-18-seed0.png`
- `train12-tr100-te200-n12-18-seed42.png`
- `train16-tr100-te200-n12-18.png`
- `train16-tr100-te200-n14-18.png`

For the first three files, the WalkSAT / WalkSATlm lines are the same because
the eval window is `n=12-18`. The `train16...n14-18` plot intentionally refits
both LR and classical slopes on `n=14-18`, so its lines differ.

## Current Curated Files

| File | Source run/script | Supports | Notes |
|------|-------------------|----------|-------|
| `results/best/angle-variance-average.png` | TODO | TODO | Add provenance before using in thesis. |
| `results/best/gamma_vs_exponents.png` | TODO | TODO | Add provenance before using in thesis. |

## Artifact Classes

| Class | Goes where | Example |
|-------|------------|---------|
| Raw run output | Pipeline-specific `results/<pipeline>/...` | JSON, CSV, logs, intermediate plots |
| Curated output | `results/best/` | Final thesis figure or table |
| Historical reference | `results/archive/` | Legacy outputs kept for comparison |
| Partial run | `_archive_partial/` under the relevant pipeline | Aborted or sanity-only runs |
