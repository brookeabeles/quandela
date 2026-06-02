# BM24 / LR-QAOA run artifacts

Outputs from `bm24_qaoa_sim.py`, `sweep_lr_depth_until_win.py`, and
`experiments/lr_scaling/notebooks/LR_QAOA_benchmark_efficient.ipynb`.

Depth-scaling JSON/PNG files live in this directory (flat layout).

To pull files out of legacy `runtime_scaling/` or `success_scaling/` subfolders:

```bash
python phasecraft/experiments/lr_scaling/organize_bm24_runs.py
```

## Filename pattern

```
MM-DD_HHMM-<kind>.<ext>
```

Examples: `05-26_1944-efficient-scaling.png`, `05-19_1645-sweep-scaling.png`

No calendar year in the stem (same convention as older `05-19-11-45-28-…` files).
Timestamp = run start (local). Plot titles use wrapped two-line headers.

## Kinds

| Kind | Source |
|------|--------|
| `sweep-scaling` | `sweep_lr_depth_until_win.py` aggregate JSON + PNG |
| `efficient-scaling` | `experiments/lr_scaling/notebooks/LR_QAOA_benchmark_efficient.ipynb` |
| `bench` / `thy` | `bm24_qaoa_sim.py` (root) |

## Logs (not stamped)

| File | Purpose |
|------|---------|
| `depth_sweep_until_win.jsonl` | One JSON line per depth (includes full benchmark) |
| `lr_train_optimal_angles_legacy.txt` | Legacy training: median/mean p_succ @ `train_n` |
| `lr_train_optimal_angles_v2.txt` | v2/v3 slope-objective training `(dg, db)` by depth |
| `lr_train_optimal_angles.txt` | Symlink → `lr_train_optimal_angles_v2.txt` (compat `--angle-log` path) |
| `lr_train_optimal_angles_mixed_archive.txt` | Pre-split combined log (kept for reference) |

Split a mixed log:

```bash
python phasecraft/experiments/lr_scaling/split_lr_angle_log.py
```

Per-depth `*-p{N}.json` only with `--save-per-depth` on the sweep script.
