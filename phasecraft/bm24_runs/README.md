# BM24 / LR-QAOA run artifacts

Outputs from `bm24_qaoa_sim.py`, `sweep_lr_depth_until_win.py`, and
`LR_QAOA_benchmark_efficient.ipynb`.

## Filename pattern

```
MM-DD_HHMM-<kind>.<ext>
```

Examples: `05-19_1645-sweep-scaling.png`, `05-19_1145-efficient-scaling.json`

No calendar year in the stem (same convention as older `05-19-11-45-28-…` files).
Timestamp = run start (local). Plot titles include full parameters (k, r, n, inst, seed, p, …).

## Kinds

| Kind | Source |
|------|--------|
| `sweep-scaling` | `sweep_lr_depth_until_win.py` aggregate JSON + PNG |
| `efficient-scaling` | `LR_QAOA_benchmark_efficient.ipynb` |
| `bench` / `thy` | `bm24_qaoa_sim.py` |

## Logs (not stamped)

| File | Purpose |
|------|---------|
| `depth_sweep_until_win.jsonl` | One JSON line per depth (includes full benchmark) |
| `lr_train_optimal_angles.txt` | LR training `(dg, db)` by depth |

Per-depth `*-p{N}.json` only with `--save-per-depth` on the sweep script.
