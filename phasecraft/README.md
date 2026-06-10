# phasecraft

BM24 / LR-QAOA simulation, saddle certification, and experiment pipelines.

## Layout

```
phasecraft/
  lib/           # importable library code
  experiments/   # runnable study scripts & notebooks
  docs/          # research/thesis navigation notes
  results/       # generated artifacts (JSON, PNG, logs)
  shims/         # legacy CLI implementations
  *.py           # thin compatibility re-exports at package root
```

| Path | Purpose |
|------|---------|
| `lib/angles/` | Optimal QAOA angles |
| `lib/ksat/` | Exact k-SAT sums; `variants/generalized_binomial_sum.PATCHED.py` is the equation set used in production |
| `lib/saddles/` | z-Krawczyk, Picard–Lefschetz; `variants/krawczyk_p1_roots_fix.py` |
| `lib/certificates/` | Proof / certificate wording |
| `lib/sim/` | `bm24_qaoa_sim.py`, `bm24_run_io.py` |
| `lib/paths.py` | Canonical paths (`bm24_runs_dir()`, `patched_gbs_path()`, …) |
| `docs/organization.md` | Working map for BM24 research, result curation, and thesis evidence |
| `experiments/lr_scaling/` | LR trainer, sweep, notebooks |
| `experiments/w_saddle/` | w-coordinate Krawczyk pipeline |
| `experiments/bm24_saddle_audit_p1/` | p=1 saddle audit |
| `results/bm24_runs/` | LR benchmark outputs (`MM-DD/runN/`; `bm24_runs` → symlink) |
| `results/w_saddle_runs/` | w-saddle `runN/` folders |
| `results/bm24_saddle_audit_p1/` | Audit run outputs |
| `results/archive/` | Legacy flat outputs, `og_krawczyk_data/`, unused helpers |

## Commands (from repo root `Quandela/`)

```bash
python phasecraft/train_lr_notebook_protocol.py --depth 14 --seed 27
python phasecraft/bm24_qaoa_sim.py --benchmark --help
python -m phasecraft.w_saddle pipeline --preset r1-neg-gamma --plots
python -m phasecraft.bm24_saddle_audit_p1.audit --help
make -f phasecraft/experiments/w_saddle/Makefile.w_saddle pipeline
```

Notebook: `experiments/lr_scaling/notebooks/LR_QAOA_benchmark.ipynb`

## Compatibility

Root modules (`optimal_angles.py`, `bm24_qaoa_sim.py`, …) re-export canonical code under `lib/` or `shims/` so existing imports and shell habits keep working. Prefer `from phasecraft.lib...` in new code.
