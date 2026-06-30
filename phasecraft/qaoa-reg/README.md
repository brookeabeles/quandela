# qaoa-reg

Regular QAOA benchmark code for the BM24 / PRX Quantum random k-SAT setup.

This folder is intentionally separate from `experiments/lr_scaling`.  It reuses
the repository's BM24 simulator and random k-SAT instance generator, but trains
full regular-QAOA angles:

```text
gammas = [gamma_0, ..., gamma_{p-1}]
betas  = [beta_0,  ..., beta_{p-1}]
```

The default training protocol follows the PRX Quantum numerical setup:

- train fixed angles on `train_n=12`;
- use 100 SAT-filtered random training instances by default;
- maximize empirical mean `p_succ`;
- initialize every layer with `beta=+0.01`, `gamma=-0.01`;
- evaluate held-out scaling by fitting `log2(median(1/p_succ))` versus `n`.

## Quick smoke run

```bash
python qaoa-reg/run_benchmark.py \
  --depths 1,2 \
  --train-size 4 \
  --test-size 4 \
  --n-min 4 \
  --n-max 5 \
  --train-n 4 \
  --k 3 \
  --r 4.2 \
  --maxiter 20 \
  --restarts 1 \
  --lr-cobyla-maxiter 20 \
  --lr-cobyla-restarts 1 \
  --force-cpu \
  --run-stem smoke
```

## PRX-style regular-QAOA run

```bash
python qaoa-reg/run_benchmark.py \
  --depths 2,5,8,10,15,20 \
  --train-n 12 \
  --train-size 100 \
  --n-min 12 \
  --n-max 20 \
  --test-size 200 \
  --training-mode bm24_mean_p_fixed_n \
  --compare-lr \
  --gpu-batch-size 16 \
  --require-gpu
```

Outputs are written under `qaoa-reg/results/`:

- `<run_stem>.json`: checkpoint with trained regular-QAOA angles, per-`n`
  median runtimes, slopes, and optional LR-QAOA comparison rows.
- `<run_stem>.png`: slope-versus-depth plot for regular QAOA and LR-QAOA.

## Notes

- The simulator uses the BM24 half-angle convention from PRX Quantum Eq. (13).
- Packed regular-QAOA angles are stored as `[gammas, betas]`, matching the
  convention used elsewhere in this repository.
- GPU compatibility uses the existing CuPy batched backend from
  `experiments/lr_scaling/gpu/backend.py`.  Regular-QAOA training, regular-QAOA
  evaluation, LR-QAOA training, and LR-QAOA evaluation all use this backend when
  CUDA is detected.
- Use `--require-gpu` for production runs so the command fails immediately
  instead of silently falling back to CPU.  Tune `--gpu-batch-size` downward if
  the GPU runs out of memory at larger `n`.
- `--batch-on-cpu` exercises the same batched backend with NumPy and is useful
  for debugging, but it is not a substitute for CUDA performance.
- `--compare-lr` trains LR-QAOA on the same training set and evaluates it on the
  same held-out dataset, so the regular-QAOA and LR-QAOA slopes are directly
  comparable.
- To overlay an existing LR checkpoint instead of retraining LR, pass
  `--no-compare-lr --lr-checkpoint /path/to/lr_run.json`.
