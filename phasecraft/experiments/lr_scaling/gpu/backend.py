"""
CUDA/GPU batched BM24 QAOA simulation with CPU fallback.

Uses CuPy when a CUDA device is available; otherwise NumPy on CPU.
Batched ``run_qaoa`` amortizes kernel launch and vectorizes the mixer loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np

BackendName = Literal["cupy", "numpy"]

_CUPY = None
_CUDA_AVAILABLE = False
_DEVICE_NAME = "cpu"


def _try_import_cupy():
    global _CUPY, _CUDA_AVAILABLE, _DEVICE_NAME
    if _CUPY is not None:
        return
    try:
        import cupy as cp  # type: ignore[import-untyped]

        if cp.cuda.is_available():
            _CUPY = cp
            _CUDA_AVAILABLE = True
            try:
                _DEVICE_NAME = str(cp.cuda.Device().name)
            except Exception:
                _DEVICE_NAME = "cuda"
        else:
            _CUPY = False  # type: ignore[assignment]
    except ImportError:
        _CUPY = False  # type: ignore[assignment]


@dataclass(frozen=True)
class DeviceInfo:
    cuda_available: bool
    backend: BackendName
    device_name: str

    def summary(self) -> str:
        if self.cuda_available:
            return f"CUDA ({self.device_name}) via CuPy"
        return "CPU via NumPy"


def detect_device(*, prefer: BackendName | None = None) -> DeviceInfo:
    """Detect CUDA; fall back to NumPy on CPU when CuPy/CUDA is unavailable."""
    _try_import_cupy()
    if prefer == "numpy":
        return DeviceInfo(False, "numpy", "cpu")
    if _CUDA_AVAILABLE and prefer != "numpy":
        return DeviceInfo(True, "cupy", _DEVICE_NAME)
    return DeviceInfo(False, "numpy", "cpu")


def get_xp(info: DeviceInfo | None = None):
    info = info or detect_device()
    if info.backend == "cupy":
        _try_import_cupy()
        return _CUPY
    return np


def _as_numpy(arr) -> np.ndarray:
    if hasattr(arr, "get"):
        return arr.get()
    return np.asarray(arr)


def run_qaoa_batch(
    h_diag_batch,
    betas: np.ndarray,
    gammas: np.ndarray,
    n: int,
    *,
    xp,
) -> np.ndarray:
    """
    Batched BM24 QAOA (Eq. 13).

    Parameters
    ----------
    h_diag_batch : (B, 2**n) Hamiltonian diagonals
  betas, gammas : length-p angle arrays
    n : number of qubits

    Returns
    -------
    psi : (B, 2**n) complex state vectors (NumPy)
    """
    h_diag_batch = xp.asarray(h_diag_batch)
    if h_diag_batch.ndim != 2:
        raise ValueError("h_diag_batch must have shape (B, 2**n)")
    betas = xp.asarray(betas, dtype=xp.float64)
    gammas = xp.asarray(gammas, dtype=xp.float64)
    if len(betas) != len(gammas):
        raise ValueError("betas and gammas must have the same length")

    bsz, dim = int(h_diag_batch.shape[0]), int(h_diag_batch.shape[1])
    if dim != (1 << int(n)):
        raise ValueError(f"Second dimension must be 2**n={1 << int(n)}, got {dim}")

    psi = xp.full((bsz, dim), 1.0 / xp.sqrt(float(dim)), dtype=xp.complex128)
    idx = xp.arange(dim, dtype=xp.int64)
    for layer in range(len(betas)):
        gamma = float(gammas[layer])
        beta = float(betas[layer])
        psi = psi * xp.exp(-0.5j * gamma * h_diag_batch)
        c = xp.cos(beta / 2.0)
        s = -1j * xp.sin(beta / 2.0)
        for q in range(int(n)):
            flipped = idx ^ (1 << q)
            psi = c * psi + s * psi[:, flipped]
    return _as_numpy(psi)


def success_probability_batch(
    psi_batch: np.ndarray,
    h_diag_batch,
    *,
    xp,
) -> np.ndarray:
    """Per-instance p_succ for batched states; returns NumPy float64 (B,)."""
    psi_batch = xp.asarray(psi_batch)
    h_diag_batch = xp.asarray(h_diag_batch)
    sat_mask = h_diag_batch == 0
    probs = xp.sum(xp.abs(psi_batch) ** 2 * sat_mask, axis=1)
    return _as_numpy(probs).astype(np.float64, copy=False)


def success_probs_for_instances(
    instances: Sequence[np.ndarray],
    n: int,
    betas: np.ndarray,
    gammas: np.ndarray,
    *,
    device: DeviceInfo | None = None,
    batch_size: int | None = None,
) -> np.ndarray:
    """
    Success probabilities for a list of H_diag arrays at fixed n.

    Chunks large batches to limit GPU memory (important at n≈20).
    """
    if not instances:
        return np.array([], dtype=np.float64)

    info = device or detect_device()
    xp = get_xp(info)
    n_inst = len(instances)
    chunk = int(batch_size) if batch_size else n_inst
    if chunk <= 0:
        raise ValueError("batch_size must be positive when provided")

    out = np.empty(n_inst, dtype=np.float64)
    for start in range(0, n_inst, chunk):
        stop = min(start + chunk, n_inst)
        stack = np.stack([np.asarray(instances[i], dtype=np.int64) for i in range(start, stop)])
        psi = run_qaoa_batch(stack, betas, gammas, int(n), xp=xp)
        out[start:stop] = success_probability_batch(psi, stack, xp=xp)
    return out


def mean_p_succ_batch(
    instances: Sequence[np.ndarray],
    n: int,
    betas: np.ndarray,
    gammas: np.ndarray,
    *,
    device: DeviceInfo | None = None,
    batch_size: int | None = None,
) -> float:
    probs = success_probs_for_instances(
        instances, n, betas, gammas, device=device, batch_size=batch_size,
    )
    if probs.size == 0:
        return 0.0
    return float(np.mean(probs))


def compute_objective_batch(
    training_mode: str,
    instances: Sequence[np.ndarray],
    n: int,
    betas: np.ndarray,
    gammas: np.ndarray,
    eps: float,
    *,
    device: DeviceInfo | None = None,
    batch_size: int | None = None,
) -> float:
    """Scalar objective to MINIMIZE (matches ``train_lr_fixed_n._compute_objective``)."""
    probs = success_probs_for_instances(
        instances, n, betas, gammas, device=device, batch_size=batch_size,
    )
    if probs.size == 0:
        return float("nan")
    if training_mode == "bm24_mean_p_fixed_n":
        return -float(np.mean(probs))
    if training_mode == "median_runtime_fixed_n":
        costs = 1.0 / np.maximum(probs, float(eps))
        return float(np.median(costs))
    if training_mode == "mean_log_runtime_fixed_n":
        costs = np.log(1.0 / np.maximum(probs, float(eps)))
        return float(np.mean(costs))
    raise ValueError(f"Unknown training_mode {training_mode!r}")
