"""
Probe whether a direct bound of the form

    |R(k0) - 1|  \lesssim  C * Frobenius_tail^alpha

is consistent with the numerical data from Direction 1.

Concretely, this script:
  - loads per-(p, gamma) results via `direction1_spectral_tail.run_sweep`
  - flattens all (k0, p, gamma) triples
  - fits log10 |R-1| = a + alpha * log10(Frobenius tail)
  - reports the fitted alpha, and quantiles of the ratio
        |R-1| / Frobenius_tail^alpha
    across the dataset.

This does NOT prove any inequality; it only checks whether a *uniform*
constant C looks numerically plausible for the explored regime.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.linear_model import LinearRegression  # type: ignore[import-untyped]

from direction1_spectral_tail import (  # type: ignore[import]
    FIG_DIR,
    run_sweep,
)


@dataclass
class FrobeniusFitSummary:
    alpha: float
    intercept: float
    r2: float
    ratio_quantiles: Tuple[float, float, float, float, float]


def _load_flat_frobenius_and_error(results: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Flatten all (p, gamma, k0) samples to (frob_tail, |R-1|) pairs."""
    frob_list: list[np.ndarray] = []
    err_list: list[np.ndarray] = []
    for d in results:
        err = np.asarray(d["err"], dtype=float)
        frob = np.asarray(d["frobenius_tail"], dtype=float)
        mask = (err > 0) & (frob > 0)
        if not np.any(mask):
            continue
        frob_list.append(frob[mask])
        err_list.append(err[mask])
    if not frob_list:
        return np.array([], dtype=float), np.array([], dtype=float)
    frob_all = np.concatenate(frob_list)
    err_all = np.concatenate(err_list)
    return frob_all, err_all


def fit_power_law(frob_tail: np.ndarray, err: np.ndarray) -> FrobeniusFitSummary:
    """
    Fit log10 |R-1| = a + alpha * log10(FrobTail).

    Returns (alpha, intercept, R^2, ratio quantiles) where ratio is
        |R-1| / FrobTail^alpha.
    """
    # Work in base-10 logs to match plots.
    x = np.log10(frob_tail).reshape(-1, 1)
    y = np.log10(err)
    reg = LinearRegression().fit(x, y)
    alpha = float(reg.coef_[0])
    intercept = float(reg.intercept_)
    r2 = float(reg.score(x, y))

    # For this fitted alpha, look at the distribution of the implied constant C.
    with np.errstate(over="ignore"):
        C_vals = err / (frob_tail**alpha)
    C_vals = C_vals[np.isfinite(C_vals) & (C_vals > 0)]
    if C_vals.size == 0:
        quantiles = (np.nan, np.nan, np.nan, np.nan, np.nan)
    else:
        qs = np.quantile(C_vals, [0.05, 0.25, 0.5, 0.75, 0.95])
        quantiles = tuple(float(q) for q in qs)
    return FrobeniusFitSummary(alpha=alpha, intercept=intercept, r2=r2, ratio_quantiles=quantiles)


def plot_ratio_histogram(
    frob_tail: np.ndarray,
    err: np.ndarray,
    alpha: float,
    out_path: Path | None = None,
) -> Path:
    """
    Plot histogram of log10( |R-1| / FrobTail^alpha ).
    A tight, narrow histogram suggests a nearly-constant C.
    """
    with np.errstate(over="ignore"):
        ratio = err / (frob_tail**alpha)
    ratio = ratio[np.isfinite(ratio) & (ratio > 0)]
    log_ratio = np.log10(ratio)

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.hist(log_ratio, bins=80, density=True, alpha=0.7, color="C1")
    ax.set_xlabel(r"$\log_{10}\left(|R(k_0)-1| / \mathrm{FrobTail}^\alpha\right)$")
    ax.set_ylabel("density")
    ax.set_title(r"Distribution of implied constant $C$ in $|R-1| \approx C\,\mathrm{FrobTail}^\alpha$")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if out_path is None:
        out_path = FIG_DIR / "direction1_frobenius_bound_ratio_hist.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Probe whether a direct bound |R(k0)-1| <= C * Frobenius_tail^alpha "
            "is numerically consistent with Direction 1 data."
        ),
    )
    parser.add_argument("--p", type=int, nargs="+", default=[2, 3, 4, 5], help="QAOA depths")
    parser.add_argument(
        "--gamma",
        type=float,
        nargs="+",
        default=[0.3, 1.0, float(np.pi), 2.0 * float(np.pi)],
        help="γ values",
    )
    args = parser.parse_args()

    results = run_sweep(p_values=tuple(args.p), gamma_values=tuple(args.gamma))
    frob_tail, err = _load_flat_frobenius_and_error(results)
    if frob_tail.size == 0:
        print("No valid (Frobenius tail, error) samples found.")
        return

    summary = fit_power_law(frob_tail, err)
    print("Fitted power-law relation |R-1| ≈ C * FrobTail^alpha")
    print(f"  alpha     = {summary.alpha:.3f}")
    print(f"  intercept = {summary.intercept:.3f}  (base-10 log C)")
    print(f"  R^2       = {summary.r2:.4f}")
    q05, q25, q50, q75, q95 = summary.ratio_quantiles
    print("  Quantiles of implied C = |R-1| / FrobTail^alpha:")
    print(f"    5%  = {q05:.3e}")
    print(f"    25% = {q25:.3e}")
    print(f"    50% = {q50:.3e}")
    print(f"    75% = {q75:.3e}")
    print(f"    95% = {q95:.3e}")

    out_hist = plot_ratio_histogram(frob_tail, err, alpha=summary.alpha)
    print("Histogram of log10(C) saved to", out_hist)


if __name__ == "__main__":
    main()

