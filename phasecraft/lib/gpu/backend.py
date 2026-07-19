"""Public GPU/backend compatibility helpers.

The implementation delegates to the maintained batched backend in
``experiments.lr_scaling.gpu.backend``.  This keeps the stable
``phasecraft.lib.gpu`` import path while avoiding the old per-instance loop.
"""

from __future__ import annotations

import numpy as np

from phasecraft.experiments.lr_scaling.gpu.backend import (
    BackendName,
    detect_device,
    get_xp,
    run_qaoa_batch,
    success_probability_batch,
)


def _device_for_backend(backend: str):
    if backend == "numpy":
        return detect_device(prefer="numpy")
    if backend == "cupy":
        return detect_device(prefer="cupy")
    raise ValueError(f"Unsupported backend: {backend!r}")


def get_array_module(backend: BackendName = "numpy"):
    """Return NumPy or CuPy for callers that still choose backends manually."""
    return get_xp(_device_for_backend(backend))


def _as_numpy_array(value) -> np.ndarray:
    if hasattr(value, "get"):
        return np.asarray(value.get())
    return np.asarray(value)


def _infer_num_qubits(state_size: int) -> int:
    n = int(state_size).bit_length() - 1
    if (1 << n) != int(state_size):
        raise ValueError("Second dimension must be a power of two (2**n).")
    return n


def qaoa_success_probabilities_batch(
    h_diag_batch,
    betas,
    gammas,
    *,
    backend: str = "numpy",
    batch_size: int | None = None,
):
    """
    Compute success probabilities for a batch of BM24 QAOA instances.

    h_diag_batch: shape (B, 2**n)
    betas: beta angles
    gammas: gamma angles
    backend: "numpy" for now
    batch_size: optional chunk size for memory control

    returns: success probabilities with shape (B,)
    """
    h_diag_batch_np = _as_numpy_array(h_diag_batch)
    if h_diag_batch_np.ndim != 2:
        raise ValueError("h_diag_batch must have shape (B, 2**n)")

    n = _infer_num_qubits(int(h_diag_batch_np.shape[1]))
    device = _device_for_backend(backend)
    xp = get_xp(device)
    chunk = int(batch_size) if batch_size else int(h_diag_batch_np.shape[0])
    if chunk <= 0:
        raise ValueError("batch_size must be positive when provided.")

    out = np.empty(int(h_diag_batch_np.shape[0]), dtype=np.float64)
    for start in range(0, int(h_diag_batch_np.shape[0]), chunk):
        stop = min(start + chunk, int(h_diag_batch_np.shape[0]))
        stack = h_diag_batch_np[start:stop]
        psi = run_qaoa_batch(stack, np.asarray(betas), np.asarray(gammas), n, xp=xp)
        out[start:stop] = success_probability_batch(psi, stack, xp=xp)
    return out
