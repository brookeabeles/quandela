"""
Softmax weight diagnostics at the true saddle (fourth mechanism: weight concentration).
"""

import numpy as np


def _real_normalized_weights(w: np.ndarray) -> np.ndarray:
    """Real part, clip to nonnegative, renormalize to sum 1 (for complex weights from softmax)."""
    w = np.real(np.asarray(w).ravel())
    w = np.maximum(w, 0.0)
    total = np.sum(w)
    if total < 1e-300:
        return np.ones_like(w) / len(w)
    return w / total


def weight_concentration_ratio(w: np.ndarray) -> float:
    """κ = max_s w_s / min_s w_s. Uses real part of weights, renormalized."""
    w = _real_normalized_weights(w)
    w_pos = w[w > 1e-300]
    if len(w_pos) == 0:
        return np.nan
    return float(np.max(w_pos) / np.min(w_pos))


def inverse_participation_ratio(w: np.ndarray) -> float:
    """IPR = 1 / ∑_s w_s² (effective number of contributing configurations). Uses real, renormalized weights."""
    w = _real_normalized_weights(w)
    return float(1.0 / (np.sum(w ** 2) + 1e-300))


def shannon_entropy(w: np.ndarray) -> float:
    """H = -∑_s w_s log(w_s). Uses real part of weights, renormalized."""
    w = _real_normalized_weights(w)
    w = w[w > 1e-300]
    return float(-np.sum(w * np.log(w)))


def fraction_top_k(w: np.ndarray, k_values: tuple[int, ...] = (1, 2, 5, 10)) -> dict[int, float]:
    """Fraction of weight in top k configurations. Uses real, renormalized weights."""
    w = _real_normalized_weights(w)
    order = np.argsort(w)[::-1]
    cum = np.cumsum(w[order])
    return {k: float(cum[min(k, len(cum)) - 1]) for k in k_values}


def softmax_diagnostics(w: np.ndarray) -> dict:
    """Return dict: kappa, ipr, entropy, max_weight, min_weight, fraction_top_1, 2, 5, 10. Uses real, renormalized weights."""
    w = _real_normalized_weights(w)
    w_pos = w[w > 1e-300]
    out = {
        "kappa": weight_concentration_ratio(w),
        "ipr": inverse_participation_ratio(w),
        "entropy": shannon_entropy(w),
        "max_weight": float(np.max(w)) if len(w) else 0.0,
        "min_weight": float(np.min(w_pos)) if len(w_pos) else 0.0,
    }
    out.update(fraction_top_k(w))
    return out
