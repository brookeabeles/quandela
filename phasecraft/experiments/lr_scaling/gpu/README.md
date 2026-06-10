# LR-QAOA GPU pipeline

GPU-accelerated copy of `train_lr_fixed_n.py` and `notebooks/LR_QAOA_benchmark.ipynb`.
Original files are untouched; this folder is a minimal terminal-runnable variant for the Quandela GPU VM.

## What is faster

- **QAOA simulation** (`run_qaoa` + success probability) runs in **batched** form on CUDA via [CuPy](https://docs.cupy.dev/) when available.
- **COBYLA**, instance generation, WalkSAT baselines, and plotting reuse the existing CPU code.

CUDA is detected at startup; if CuPy or a GPU is missing, the same code path uses NumPy on CPU.

## Setup (Ubuntu, Python 3.12)

From the repository root (`Quandela/`):

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install numpy scipy matplotlib numba tqdm

# Match your CUDA driver (12.x on the Quandela GPU VM):
pip install cupy-cuda12x
```

Or install everything listed in `requirements.txt` (CuPy line is Linux-only).

## Verify GPU backend + objectives

```bash
cd Quandela
python -m phasecraft.experiments.lr_scaling.gpu.train_lr_fixed_n
```

Expected: `verify_objectives: all 5 checks passed.` and a device line (`CUDA (...)` or `CPU via NumPy`).

## Run full benchmark (notebook pipeline)

Small smoke test (CPU or GPU):

```bash
cd Quandela
python -m phasecraft.experiments.lr_scaling.gpu.run_benchmark \
  --train-n 12 --train-size 16 \
  --n-min 12 --n-max 14 --test-size 20 \
  --depths 2,5 \
  --cobyla-maxiter 40 --cobyla-restarts 2 \
  --skip-grid \
  --run-stem gpu-smoke
```

Full-scale run (matches notebook defaults):

```bash
cd Quandela
python -m phasecraft.experiments.lr_scaling.gpu.run_benchmark \
  --train-n 14 --train-size 100 \
  --n-min 12 --n-max 20 --test-size 200 \
  --depths 2,5,8,10,15,20,30,40,50,60,100 \
  --cobyla-maxiter 160 --cobyla-restarts 8 \
  --run-stem scaling-tn14-gpu
```

Resume after disconnect:

```bash
python -m phasecraft.experiments.lr_scaling.gpu.run_benchmark \
  --run-stem scaling-tn14-gpu --auto-resume
```

Force CPU (debug / no GPU):

```bash
python -m phasecraft.experiments.lr_scaling.gpu.run_benchmark --force-cpu ...
```

Reduce GPU memory at large `n_max` (default batch size is 32):

```bash
python -m phasecraft.experiments.lr_scaling.gpu.run_benchmark --gpu-batch-size 8 ...
```

## Outputs

Checkpoints and plots are written under:

`phasecraft/results/bm24_runs/MM-DD/runN/`

(same layout as the notebook; `bm24_runs` is a symlink to `results/bm24_runs`).

Each run produces `{run_stem}.json`, `{run_stem}.partial.json`, and `{run_stem}.png`.

## Files

| File | Role |
|------|------|
| `backend.py` | CUDA detection, batched QAOA on CuPy, NumPy fallback |
| `train_lr_fixed_n.py` | Patches objectives in the original trainer |
| `run_benchmark.py` | Terminal port of `LR_QAOA_benchmark.ipynb` |
| `classical.py` | Numba WalkSAT baselines (CPU) |
| `_bootstrap.py` | `sys.path` + `results/` paths (no hardcoded home dirs) |
