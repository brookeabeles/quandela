# Result Curation

`results/best/` is for thesis-ready artifacts, not merely good-looking plots.

## Promotion Rule

Promote a file into `results/best/` only when it has:

1. A source run folder.
2. A generating script or notebook.
3. A short interpretation.
4. A known thesis claim or open question it supports.

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

