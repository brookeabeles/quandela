"""Backend selection and future batch-evaluation boundary."""

from __future__ import annotations


def get_array_module(backend: str = "numpy"):
    if backend == "numpy":
        import numpy as np

        return np
    raise ValueError(f"Unsupported backend: {backend}")


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
    xp = get_array_module(backend=backend)
    from phasecraft.lib.sim.bm24_qaoa_sim import (
        per_instance_success_probability,
        run_qaoa,
    )

    h_diag_batch = xp.asarray(h_diag_batch)
    if h_diag_batch.ndim != 2:
        raise ValueError("h_diag_batch must have shape (B, 2**n)")

    state_size = int(h_diag_batch.shape[1])
    n = int(round(state_size.bit_length() - 1))
    if (1 << n) != state_size:
        raise ValueError("Second dimension must be a power of two (2**n).")

    betas = xp.asarray(betas)
    gammas = xp.asarray(gammas)
    chunk = int(batch_size) if batch_size else int(h_diag_batch.shape[0])
    if chunk <= 0:
        raise ValueError("batch_size must be positive when provided.")

    out = xp.empty(int(h_diag_batch.shape[0]), dtype=xp.float64)
    for start in range(0, int(h_diag_batch.shape[0]), chunk):
        stop = min(start + chunk, int(h_diag_batch.shape[0]))
        for i in range(start, stop):
            h_diag = xp.asarray(h_diag_batch[i])
            psi = run_qaoa(h_diag, betas, gammas, n)
            out[i] = per_instance_success_probability(psi, h_diag)
    return out

