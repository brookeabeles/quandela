"""
Theoretical bound on |R(k₀) − 1| from eigenvalue decay (Phase 3, Task 3.1).

If |λ_k| ≤ M e^{-c' k} for k ≥ 1, then
  |log R(k₀)| ≤ (4M/(1−e^{-c'})) e^{-c'(k₀+1)} ≤ (4M/c') e^{-c' k₀}
  and |R(k₀) − 1| ≤ (8M/c') e^{-c' k₀} for small |log R|.

Also: k₀(ε) ≥ (1/c')(log(8M/c') + log(1/ε)) for |R(k₀) − 1| ≤ ε.
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Any


def determinant_ratio_bound_from_decay(
    k0_axis: np.ndarray,
    M: float,
    c_prime: float,
) -> np.ndarray:
    """
    Upper bound on |R(k₀) − 1| from |λ_k| ≤ M e^{-c' k}.

    For k₀ large enough that 2 M e^{-c' k₀} < 1/2:
      |log R(k₀)| ≤ 4 M e^{-c'(k₀+1)} / (1 - e^{-c'}) ≤ (4M/c') e^{-c' k₀}
      |R(k₀) − 1| ≤ (8M/c') e^{-c' k₀}  (for small |log R|: e^|log R| - 1 ≤ 2|log R|).

    Returns array of length len(k0_axis) with the bound at each k₀.
    """
    k0_axis = np.asarray(k0_axis, dtype=float)
    if c_prime <= 0 or not np.isfinite(M) or M <= 0:
        return np.full_like(k0_axis, np.nan)
    # Safe regime: 2 M e^{-c' k₀} < 1/2  =>  k₀ > (1/c') log(4M)
    k0_safe = np.maximum(k0_axis, 0)
    log_R_bound = (4.0 * M / c_prime) * np.exp(-c_prime * k0_safe)
    # |R - 1| ≤ e^{|log R|} - 1 ≤ 2|log R| for |log R| small
    err_bound = np.minimum(2.0 * log_R_bound, 1e10)  # cap for huge k0
    return err_bound


def k0_for_epsilon(epsilon: float, M: float, c_prime: float) -> float:
    """
    Minimum k₀ such that the bound gives |R(k₀) − 1| ≤ ε.

    From (8M/c') e^{-c' k₀} ≤ ε:
      k₀ ≥ (1/c') ( log(8M/c') + log(1/ε) ).
    """
    if c_prime <= 0 or epsilon <= 0 or not np.isfinite(M) or M <= 0:
        return np.nan
    return (1.0 / c_prime) * (np.log(8.0 * M / c_prime) + np.log(1.0 / epsilon))


def overlay_bound_on_probe1(
    k0_axis: np.ndarray,
    err_array: np.ndarray,
    M: float,
    c_prime: float,
) -> dict[str, Any]:
    """
    Compute the theoretical bound curve and return data for overlaying on probe1 plot.
    Returns dict with 'k0_axis', 'err_array' (observed), 'bound_array' (theoretical).
    """
    bound_array = determinant_ratio_bound_from_decay(k0_axis, M, c_prime)
    return {
        "k0_axis": np.asarray(k0_axis),
        "err_array": np.asarray(err_array),
        "bound_array": bound_array,
        "M": M,
        "c_prime": c_prime,
        "k0_for_1pct": k0_for_epsilon(0.01, M, c_prime),
    }


def plot_probe1_with_bound_overlay(
    probe1_results: list[dict],
    p: int,
    eigenvalue_fits: dict | None = None,
    out_path: Path | None = None,
) -> None:
    """
    Plot |R(k0)-1| vs k0 (from probe1_results) and overlay theoretical bound
    |R(k0)-1| <= (8M/c') e^{-c' k0} when eigenvalue decay holds.
    eigenvalue_fits: optional dict mapping (gamma_index or gamma) -> {M, c_prime};
    if None, M and c' are computed by running eigenvalue_decay_analysis for each curve.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    try:
        from symmetry_reduction.eigenvalue_decay.eigenvalue_decay import eigenvalue_decay_analysis
    except Exception:
        from .eigenvalue_decay import eigenvalue_decay_analysis

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(probe1_results)))
    for i, r in enumerate(probe1_results):
        k0 = np.asarray(r["k0_axis"])
        err = np.asarray(r["err_array"])
        if len(k0) == 0 or len(err) == 0:
            continue
        gamma = r["gamma"]
        err_pos = np.maximum(np.abs(err), 1e-16)
        ax.semilogy(k0, err_pos, "-", color=colors[i], label=rf"$\gamma$={gamma:.2f}")
        if eigenvalue_fits is not None and i in eigenvalue_fits:
            fit = eigenvalue_fits[i]
        elif eigenvalue_fits is not None and gamma in eigenvalue_fits:
            fit = eigenvalue_fits[gamma]
        else:
            try:
                out = eigenvalue_decay_analysis(p, gamma, at_saddle=True)
                fit = out["fits"]["exponential"]
            except Exception:
                fit = None
        if fit and np.isfinite(fit.get("M")) and np.isfinite(fit.get("c_prime")) and fit.get("c_prime", 0) > 0:
            M, c = fit["M"], fit["c_prime"]
            bound = determinant_ratio_bound_from_decay(k0, M, c)
            ax.semilogy(k0, np.maximum(bound, 1e-16), "--", color=colors[i], alpha=0.8, label=rf"bound $\gamma$={gamma:.2f}")
    ax.axhline(0.01, color="gray", linestyle=":", alpha=0.5, label="1%")
    ax.set_xlabel(r"$k_0$")
    ax.set_ylabel(r"$||R(k_0)| - 1|$")
    ax.set_title(rf"Determinant-ratio error with theoretical bound (p={p}, saddle)")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if out_path is None:
        out_path = Path(__file__).resolve().parent.parent / "figures" / "eigenvalue_decay" / f"probe1_with_bound_p{p}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
