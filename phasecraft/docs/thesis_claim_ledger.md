# Thesis Claim Ledger

Use this as the single place where thesis claims become auditable.

| Claim ID | Claim | Status | Evidence path | Figure/table | Citation | Caveat |
|----------|-------|--------|---------------|--------------|----------|--------|
| C-001 | TODO | draft | TODO | TODO | BM24 | TODO |

## Status Labels

| Status | Meaning |
|--------|---------|
| `draft` | Plausible, not yet fully checked. |
| `supported` | Evidence path exists and reproduces the claim. |
| `thesis-ready` | Wording, citation, and figure/table are ready for the thesis. |
| `blocked` | Needs theory, supervisor input, compute, or a missing artifact. |
| `retired` | Superseded or found to be wrong. |

## Claim Checklist

Before promoting a claim to `thesis-ready`, confirm:

- The evidence path points to a concrete file or run folder.
- The script or command that generated it is known.
- The figure/table is present in `results/best/` or intentionally excluded.
- The claim distinguishes finite-n evidence, certified saddles, and contour
  dominance where relevant.
- The citation is precise enough to find the source again.

