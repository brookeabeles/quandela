"""
GPU-accelerated drop-in for ``experiments.lr_scaling.train_lr_fixed_n``.

Patches the hot objective paths to use batched CUDA simulation when available.
All other logic (COBYLA, grid search, guards) is reused from the original module.
"""

from __future__ import annotations

from typing import Optional

from ._bootstrap import ensure_paths

ensure_paths()

from phasecraft.experiments.lr_scaling import train_lr_fixed_n as _base  # noqa: E402
from .backend import (  # noqa: E402
    DeviceInfo,
    compute_objective_batch,
    detect_device,
    mean_p_succ_batch,
)

_DEVICE: Optional[DeviceInfo] = None
_BATCH_SIZE: Optional[int] = None


def configure_gpu(*, device: DeviceInfo | None = None, batch_size: int | None = None) -> DeviceInfo:
    """Select device and optional GPU chunk size for instance batches."""
    global _DEVICE, _BATCH_SIZE
    _DEVICE = device or detect_device()
    _BATCH_SIZE = batch_size
    return _DEVICE


def _patched_compute_objective(training_mode, instances, n, betas, gammas, eps=_base.DEFAULT_EPS):
    return compute_objective_batch(
        training_mode, instances, n, betas, gammas, eps,
        device=_DEVICE, batch_size=_BATCH_SIZE,
    )


def _patched_mean_p_succ(instances, n, betas, gammas):
    return mean_p_succ_batch(
        instances, n, betas, gammas, device=_DEVICE, batch_size=_BATCH_SIZE,
    )


_base._compute_objective = _patched_compute_objective  # type: ignore[attr-defined]
_base._mean_p_succ = _patched_mean_p_succ  # type: ignore[attr-defined]

# Re-export public API unchanged.
SUPPORTED_TRAINING_MODES = _base.SUPPORTED_TRAINING_MODES
DEFAULT_DG_BOUNDS = _base.DEFAULT_DG_BOUNDS
DEFAULT_DB_BOUNDS = _base.DEFAULT_DB_BOUNDS
DEFAULT_INITIAL_DELTAS = _base.DEFAULT_INITIAL_DELTAS
DEFAULT_EPS = _base.DEFAULT_EPS
MIN_TRAIN_MEAN_P = _base.MIN_TRAIN_MEAN_P
DEFAULT_EVAL_RUNTIME_REGRESSION_FACTOR = _base.DEFAULT_EVAL_RUNTIME_REGRESSION_FACTOR
DEFAULT_EVAL_TRAIN_RETRIES = _base.DEFAULT_EVAL_TRAIN_RETRIES
MODE_RNG_OFFSET = _base.MODE_RNG_OFFSET

generate_training_instances = _base.generate_training_instances
train_angles_fixed_n = _base.train_angles_fixed_n
eval_median_runtime_reject = _base.eval_median_runtime_reject
run_train_eval_with_retries = _base.run_train_eval_with_retries
verify_objectives = _base.verify_objectives


def main() -> None:
    info = configure_gpu()
    print(info.summary())
    _base.verify_objectives()


if __name__ == "__main__":
    main()
