"""
Decay-rate analysis for Probe 1 (determinant-ratio error |R(k₀)| − 1).

Fits power-law and exponential decay in k₀ (number of kept directions) and reports:
- Best-fit exponent α (power-law: err ~ k₀^{-α}) or rate c (exponential: err ~ e^{-c k₀})
- Estimated k₀ required for a target relative error ε (e.g. 1% accuracy).

This bridges the spectral structure to the practical question: "How many effective
dimensions does the BM24 saddle-point method need for a given accuracy?"

Theoretical bound (eigenvalue decay): if |λ_k| ≤ M e^{-c' k} for eigenvalues of H_log,
then |R(k₀) − 1| ≤ (8M/c') e^{-c' k₀} and k₀(ε) ≥ (1/c')(log(8M/c') + log(1/ε)).
See symmetry_reduction/eigenvalue_decay/determinant_bound.py for the bound and
plot_probe1_with_bound_overlay() to overlay it on probe1 plots.
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Any

OUT_DIR = Path(__file__).resolve().parent
DATA_DIR = OUT_DIR / "data"
FIG_DIR = OUT_DIR / "figures"

# Fit only in range where error is meaningful (avoid k0=0 and numerical noise)
ERR_MIN = 1e-8
ERR_MAX_FIT = 0.5
MIN_POINTS_FIT = 4


def load_probe1_all_p(p_values: list[int] | None = None) -> dict[int, list[dict[str, Any]]]:
    """Load Probe 1 results for all available p. Returns {p: list of result dicts}."""
    if p_values is None:
        p_values = [1, 2, 3, 4, 5]
    out = {}
    for p in p_values:
        path = DATA_DIR / f"probe1_saddle_p{p}.npz"
        if not path.exists():
            continue
        d = dict(np.load(path, allow_pickle=True))
        if "err_array" not in d:
            continue
        n = len(d["gamma"])
        out[p] = [
            {
                "gamma": float(d["gamma"][i]),
                "k_99": int(d["k99"][i]),
                "stable_rank": float(d["stable_rank"][i]),
                "k0_axis": np.asarray(d["k0_axis"][i]),
                "err_array": np.asarray(d["err_array"][i]),
            }
            for i in range(n)
        ]
    return out


def _fit_range(
    k0: np.ndarray,
    err: np.ndarray,
    err_min: float = ERR_MIN,
    err_max: float = ERR_MAX_FIT,
) -> np.ndarray:
    """Boolean mask for k0/err points used in fitting (exclude k0=0, too small/large err)."""
    err_pos = np.maximum(np.abs(err), 1e-20)
    use = (k0 >= 1) & (err_pos >= err_min) & (err_pos <= err_max)
    return use


def fit_power_law(
    k0: np.ndarray,
    err: np.ndarray,
) -> tuple[float, float, float] | None:
    """
    Fit |R(k₀)| − 1 ≈ A * k₀^{-α} (linear regression in log-log).
    Returns (α, log_A, R²) or None if fit fails.
    """
    k0 = np.asarray(k0, dtype=float)
    err = np.asarray(err, dtype=float)
    use = _fit_range(k0, err)
    if np.sum(use) < MIN_POINTS_FIT:
        return None
    x = np.log(k0[use] + 0.5)  # avoid log(0)
    y = np.log(np.maximum(np.abs(err[use]), 1e-20))
    # y = log A - α * log k0  =>  regress y on log(k0)
    log_k = np.log(k0[use] + 0.5)
    n = len(log_k)
    sxy = np.sum((log_k - np.mean(log_k)) * (y - np.mean(y)))
    sxx = np.sum((log_k - np.mean(log_k)) ** 2)
    if sxx < 1e-30:
        return None
    alpha = -sxy / sxx
    log_A = np.mean(y) + alpha * np.mean(log_k)
    ss_res = np.sum((y - (log_A - alpha * log_k)) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 1e-30 else 0.0
    return (float(alpha), float(log_A), float(r2))


def fit_exponential(
    k0: np.ndarray,
    err: np.ndarray,
) -> tuple[float, float, float] | None:
    """
    Fit |R(k₀)| − 1 ≈ A * exp(−c * k₀) (linear regression in log vs k₀).
    Returns (c, log_A, R²) or None if fit fails.
    """
    k0 = np.asarray(k0, dtype=float)
    err = np.asarray(err, dtype=float)
    use = _fit_range(k0, err)
    if np.sum(use) < MIN_POINTS_FIT:
        return None
    x = k0[use]
    y = np.log(np.maximum(np.abs(err[use]), 1e-20))
    n = len(x)
    sxy = np.sum((x - np.mean(x)) * (y - np.mean(y)))
    sxx = np.sum((x - np.mean(x)) ** 2)
    if sxx < 1e-30:
        return None
    c = -sxy / sxx
    log_A = np.mean(y) + c * np.mean(x)
    ss_res = np.sum((y - (log_A - c * x)) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 1e-30 else 0.0
    return (float(c), float(log_A), float(r2))


def k0_for_target_error(
    target_err: float,
    rate: float | None,
    log_A: float | None,
    power_law: bool,
) -> float | None:
    """
    From power-law err = A k₀^{-α}: k₀ = (A / target_err)^{1/α} (rate = α).
    From exponential err = A e^{-c k₀}: k₀ = (log A - log target_err) / c (rate = c).
    Returns estimated k₀ or None.
    """
    if power_law and rate is not None and log_A is not None and rate > 0:
        A = np.exp(log_A)
        if A <= 0 or target_err <= 0:
            return None
        return float((A / target_err) ** (1.0 / rate))
    if not power_law and rate is not None and log_A is not None and rate > 0:
        log_target = np.log(max(target_err, 1e-20))
        return float((log_A - log_target) / rate)
    return None


def analyze_one_curve(
    k0: np.ndarray,
    err: np.ndarray,
    target_epsilon: float = 0.01,
    n_dims: int | None = None,
) -> dict[str, Any]:
    """
    Fit power-law and exponential to one (γ, p) curve; choose better fit by R²;
    estimate k₀ for target_epsilon. If n_dims is set, k₀ estimate is capped to [0, n_dims].
    """
    pl = fit_power_law(k0, err)
    ex = fit_exponential(k0, err)
    best_alpha: float | None = None
    best_log_A: float | None = None
    best_c: float | None = None
    best_r2 = -np.inf
    best_model = "none"

    if pl is not None:
        alpha, log_A, r2 = pl
        if r2 > best_r2:
            best_r2 = r2
            best_model = "power_law"
            best_alpha = alpha
            best_log_A = log_A

    if ex is not None:
        c, log_A, r2 = ex
        if r2 > best_r2:
            best_r2 = r2
            best_model = "exponential"
            best_c = c
            best_log_A = log_A

    k0_target: float | None = None
    if best_model == "power_law" and best_alpha is not None and best_log_A is not None:
        k0_target = k0_for_target_error(target_epsilon, best_alpha, best_log_A, power_law=True)
    elif best_model == "exponential" and best_c is not None and best_log_A is not None:
        k0_target = k0_for_target_error(target_epsilon, best_c, best_log_A, power_law=False)

    if k0_target is not None and n_dims is not None:
        if k0_target < 0 or k0_target > n_dims or not np.isfinite(k0_target):
            k0_target = None  # unreasonable estimate

    return {
        "model": best_model,
        "alpha": best_alpha,
        "c": best_c,
        "log_A": best_log_A,
        "r2": best_r2,
        "k0_for_1pct": k0_target,
        "target_epsilon": target_epsilon,
    }


def run_decay_analysis(
    p_values: list[int] | None = None,
    target_epsilon: float = 0.01,
) -> list[dict[str, Any]]:
    """
    Load all available Probe 1 data, fit decay for each (γ, p), return list of
    records: gamma, p, model, alpha, c, r2, k0_for_1pct.
    """
    data = load_probe1_all_p(p_values)
    rows = []
    for p, curves in data.items():
        for r in curves:
            k0 = np.asarray(r["k0_axis"])
            err = np.asarray(r["err_array"])
            if len(k0) == 0 or len(err) == 0:
                continue
            n_dims = len(k0)
            res = analyze_one_curve(k0, err, target_epsilon=target_epsilon, n_dims=n_dims)
            rows.append({
                "gamma": r["gamma"],
                "p": p,
                "k_99": r["k_99"],
                "stable_rank": r["stable_rank"],
                **res,
            })
    return rows


def print_decay_table(rows: list[dict[str, Any]]) -> None:
    """Print a text table of decay rates and k₀ for 1% accuracy."""
    if not rows:
        print("No data.")
        return
    print("Determinant-ratio decay analysis (|R(k₀)| − 1 vs k₀)")
    print("Target relative error ε = 1%")
    print("-" * 80)
    print(f"{'γ':>8} {'p':>3} {'model':>12} {'α (power)':>12} {'c (exp)':>12} {'R²':>8} {'k₀(1%)':>10} {'k_99':>6}")
    print("-" * 80)
    for r in sorted(rows, key=lambda x: (x["p"], x["gamma"])):
        alpha_s = f"{r['alpha']:.4f}" if r["alpha"] is not None else "—"
        c_s = f"{r['c']:.4f}" if r["c"] is not None else "—"
        k0_s = f"{r['k0_for_1pct']:.1f}" if r["k0_for_1pct"] is not None else "—"
        print(f"{r['gamma']:>8.3f} {r['p']:>3} {r['model']:>12} {alpha_s:>12} {c_s:>12} {r['r2']:>8.3f} {k0_s:>10} {r['k_99']:>6}")
    print("-" * 80)


def plot_rate_vs_gamma(
    rows: list[dict[str, Any]],
    out_path: Path | None = None,
) -> None:
    """
    Plot decay rate vs γ, one subplot per p. For power-law use α; for exponential use c.
    Also plot k₀ required for 1% accuracy vs γ.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    p_vals = sorted(set(r["p"] for r in rows))
    fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    ax_rate, ax_k0 = axes[0], axes[1]
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(p_vals)))

    for i, p in enumerate(p_vals):
        sub = [r for r in rows if r["p"] == p]
        sub = sorted(sub, key=lambda x: x["gamma"])
        gammas = np.array([r["gamma"] for r in sub])
        x = gammas / np.pi
        rate = []
        for r in sub:
            if r["model"] == "power_law" and r["alpha"] is not None:
                rate.append(r["alpha"])
            elif r["model"] == "exponential" and r["c"] is not None:
                rate.append(r["c"])
            else:
                rate.append(np.nan)
        k0_1pct = [
            r["k0_for_1pct"] if r["k0_for_1pct"] is not None and np.isfinite(r["k0_for_1pct"]) else np.nan
            for r in sub
        ]
        ax_rate.plot(x, rate, "o-", color=colors[i], label=f"p={p}", markersize=4)
        ax_k0.plot(x, k0_1pct, "o-", color=colors[i], label=f"p={p}", markersize=4)

    ax_rate.set_ylabel("Decay rate (α or c)")
    ax_rate.set_title("Determinant-ratio error decay: rate vs γ/π (α for power-law, c for exponential)")
    ax_rate.legend()
    ax_rate.grid(True, alpha=0.3)

    ax_k0.set_xlabel(r"$\gamma/\pi$")
    ax_k0.set_xticks([0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0])
    ax_k0.set_xticklabels(["0", r"$\pi/4$", r"$\pi/2$", r"$3\pi/4$", r"$\pi$", r"$5\pi/4$", r"$3\pi/2$", r"$7\pi/4$", r"$2\pi$"])
    ax_k0.set_ylabel(r"$k_0$ for 1% accuracy")
    ax_k0.set_title("Estimated directions needed for |R(k₀)| − 1 ≤ 0.01")
    ax_k0.legend()
    ax_k0.grid(True, alpha=0.3)

    fig.tight_layout()
    if out_path is None:
        out_path = FIG_DIR / "saddle-pt-integral-error" / "probe1_decay_rate_vs_gamma.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    rows = run_decay_analysis(target_epsilon=0.01)
    print_decay_table(rows)
    if rows:
        plot_rate_vs_gamma(rows)
        print("\nPlot saved to figures/saddle-pt-integral-error/probe1_decay_rate_vs_gamma.png")


if __name__ == "__main__":
    main()
