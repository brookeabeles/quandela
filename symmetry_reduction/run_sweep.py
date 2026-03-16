"""
Main driver: sweep over (p, γ), compute H_log at y=0 and at saddle y_0^*, save results.
"""

import numpy as np
from pathlib import Path

from .core import (
    build_structure_matrix,
    compute_b_s,
    compute_c_alpha,
    covariance_from_weights,
    hessian_log_at_y,
    softmax_weights,
    weights_at_zero,
    DEFAULT_BETA,
)
from .saddle import saddle_with_adaptive_damping
from .spectral import (
    spectral_summary,
    singular_values,
    determinant_ratio_probe,
    approximation_error_proxies,
    kept_dropped_alpha_analysis,
    effective_dimension_block_structure,
)
from .mechanisms import energy_by_block, mechanism_per_alpha
from .diagnostics import softmax_diagnostics
from .validation import run_all_checks

# Gamma sweep: key regimes + dense for plotting
GAMMA_SMALL = [0.05, 0.1, 0.2, 0.3, 0.5]
GAMMA_MODERATE = [0.7, 1.0, 1.5, 2.0]
GAMMA_LARGE = [2.5, np.pi, 3.5, 4.0, 5.0, 2 * np.pi]
GAMMA_DISCRETE = list(dict.fromkeys(GAMMA_SMALL + GAMMA_MODERATE + GAMMA_LARGE))
GAMMA_DENSE = np.linspace(0.05, 2 * np.pi, 80).tolist()

OUT_DIR = Path(__file__).resolve().parent
DATA_DIR = OUT_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)


def _print_kept_dropped_analysis(analysis: dict, k: int, p: int, label: str, top_n: int = 10):
    """Print a short summary of kept/dropped α analysis to stdout."""
    n_blocks = 2 * p + 2
    bw_kept = analysis["block_weights_kept"]
    bw_drop = analysis["block_weights_dropped"]
    print(f"  [{label}] k_99 = {k}")
    print("    Block |α|:  kept (frac)   dropped (frac)")
    for m in range(n_blocks):
        print(f"      {m:2d}:     {bw_kept[m]:.4f}         {bw_drop[m]:.4f}")
    top_kept = analysis["top_alpha_kept"][:top_n]
    top_drop = analysis["top_alpha_dropped"][:top_n]
    print(f"    Top α (kept):   {top_kept.tolist()}")
    print(f"    Top α (dropped): {top_drop.tolist()}")


def _print_effective_structure(eff: dict, label: str):
    """Print structure of the effective-dimension matrix (rank-k_99 approximation)."""
    print(f"  [{label}] Effective matrix (rank-k) block structure:")
    print(f"    Diagonal fraction (m=m'):    {eff['diagonal_fraction']:.4f}")
    print(f"    Off-diagonal fraction:       {eff['off_diagonal_fraction']:.4f}")
    f = eff["f_mm_eff"]
    # Peak block (m, m')
    i, j = np.unravel_index(np.argmax(f), f.shape)
    print(f"    Peak block (|α|,|α'|):       ({i}, {j}) = {f[i, j]:.4f}")
    # Per-vector: which block is dominant for first few singular vectors
    pvb = eff["per_vector_block"]
    n_blocks = pvb.shape[1]
    print("    Per-vector dominant block (first 5 directions): ", end="")
    dom = np.argmax(pvb[: min(5, pvb.shape[0])], axis=1)
    print(dom.tolist())


def run_single_p_gamma(
    p: int,
    gamma: float,
    betas: np.ndarray | None = None,
    run_saddle: bool = True,
    run_validation: bool = False,
    analyze_kept_dims: bool = False,
    A: np.ndarray | None = None,
    b_s: np.ndarray | None = None,
):
    """Compute H_log at y=0 and optionally at saddle for one (p, γ). Returns dicts.
    If A and/or b_s are provided (e.g. from a multi-gamma sweep), they are reused; they do not depend on gamma.
    If analyze_kept_dims is True, run full SVD and attach which α contribute to kept/dropped k_99 subspace (and print a summary)."""
    if betas is None:
        betas = np.full(p, DEFAULT_BETA)
    gammas = np.full(p, gamma)
    if A is not None:
        d = A.shape[0]
    else:
        d = 1 << (2 * p + 1)
        A = build_structure_matrix(p)
    if b_s is None:
        b_s = compute_b_s(p, betas)
    c_alpha = compute_c_alpha(p, gammas)

    if run_validation:
        run_all_checks(p, A, b_s, c_alpha)

    w0 = weights_at_zero(b_s)
    y0 = np.zeros(d, dtype=complex)
    Cov0 = covariance_from_weights(A, w0)
    sigmas_cov0 = np.linalg.svd(Cov0, compute_uv=False)
    cum_cov0 = np.cumsum(sigmas_cov0**2) / (np.sum(sigmas_cov0**2) + 1e-30)
    k99_cov0 = int(np.searchsorted(cum_cov0, 0.99) + 1) if len(cum_cov0) else 0

    H0 = hessian_log_at_y(y0, A, b_s, c_alpha)
    spec0 = spectral_summary(H0)

    E_mm_0, f_mm_0 = energy_by_block(H0, p)
    mech0, _ = mechanism_per_alpha(p, A, w0, c_alpha)

    out_y0 = {
        "gamma": gamma,
        "singular_values": spec0["singular_values"],
        "frobenius_norm": spec0["frobenius_norm"],
        "spectral_norm": spec0["spectral_norm"],
        "stable_rank": spec0["stable_rank"],
        "k_90": spec0["k_90"],
        "k_95": spec0["k_95"],
        "k_99": spec0["k_99"],
        "k_999": spec0["k_999"],
        "cumulative_energy": spec0["cumulative_energy"],
        "energy_by_size": E_mm_0,
        "f_mm": f_mm_0,
        "mechanism": mech0,
        "k99_cov": k99_cov0,
    }
    if analyze_kept_dims:
        kd_y0 = kept_dropped_alpha_analysis(H0, spec0["k_99"], p)
        out_y0["kept_dropped"] = kd_y0
        eff_y0 = effective_dimension_block_structure(H0, spec0["k_99"], p)
        out_y0["effective_block_structure"] = eff_y0
        print("--- Kept/dropped α analysis (y=0) ---")
        _print_kept_dropped_analysis(kd_y0, spec0["k_99"], p, "y=0")
        print("--- Effective-dimension matrix structure (y=0) ---")
        _print_effective_structure(eff_y0, "y=0")

    out_saddle = None
    if run_saddle:
        y_star, converged, residual = saddle_with_adaptive_damping(A, b_s, c_alpha)
        w_star = softmax_weights(y_star, A, b_s, np.sqrt(c_alpha + 0j))
        Cov_star = covariance_from_weights(A, w_star)
        sigmas_cov_s = np.linalg.svd(Cov_star, compute_uv=False)
        cum_cov_s = np.cumsum(sigmas_cov_s**2) / (np.sum(sigmas_cov_s**2) + 1e-30)
        k99_cov_s = int(np.searchsorted(cum_cov_s, 0.99) + 1) if len(cum_cov_s) else 0

        H_star = hessian_log_at_y(y_star, A, b_s, c_alpha)
        spec_star = spectral_summary(H_star)
        E_mm_s, f_mm_s = energy_by_block(H_star, p)
        mech_s, _ = mechanism_per_alpha(p, A, w_star, c_alpha)
        diag = softmax_diagnostics(w_star)

        out_saddle = {
            "gamma": gamma,
            "y_star": y_star,
            "converged": converged,
            "saddle_residual": residual,
            "singular_values": spec_star["singular_values"],
            "frobenius_norm": spec_star["frobenius_norm"],
            "spectral_norm": spec_star["spectral_norm"],
            "stable_rank": spec_star["stable_rank"],
            "k_90": spec_star["k_90"],
            "k_95": spec_star["k_95"],
            "k_99": spec_star["k_99"],
            "k_999": spec_star["k_999"],
            "cumulative_energy": spec_star["cumulative_energy"],
            "energy_by_size": E_mm_s,
            "f_mm": f_mm_s,
            "mechanism": mech_s,
            "kappa": diag["kappa"],
            "ipr": diag["ipr"],
            "entropy": diag["entropy"],
            "max_weight": diag["max_weight"],
            "min_weight": diag["min_weight"],
            "k99_cov": k99_cov_s,
        }
        for k in (1, 2, 5, 10):
            out_saddle[f"fraction_top_{k}"] = diag.get(k, np.nan)
        if analyze_kept_dims:
            kd_saddle = kept_dropped_alpha_analysis(H_star, spec_star["k_99"], p)
            out_saddle["kept_dropped"] = kd_saddle
            eff_saddle = effective_dimension_block_structure(H_star, spec_star["k_99"], p)
            out_saddle["effective_block_structure"] = eff_saddle
            print("--- Kept/dropped α analysis (saddle) ---")
            _print_kept_dropped_analysis(kd_saddle, spec_star["k_99"], p, "saddle")
            print("--- Effective-dimension matrix structure (saddle) ---")
            _print_effective_structure(eff_saddle, "saddle")

    return out_y0, out_saddle


def run_sweep_p(
    p: int,
    gamma_list: list[float] | np.ndarray,
    run_saddle: bool = True,
    run_validation_once: bool = True,
):
    """Run sweep over gamma for fixed p. Returns (results_y0, results_saddle)."""
    gamma_list = np.atleast_1d(gamma_list)
    d = 1 << (2 * p + 1)
    n_gamma = len(gamma_list)

    results_y0 = {
        "p": p,
        "gamma_values": np.array(gamma_list),
        "singular_values": np.zeros((n_gamma, d)),
        "k90": np.zeros(n_gamma, dtype=int),
        "k95": np.zeros(n_gamma, dtype=int),
        "k99": np.zeros(n_gamma, dtype=int),
        "k999": np.zeros(n_gamma, dtype=int),
        "stable_rank": np.zeros(n_gamma),
        "frobenius_norm": np.zeros(n_gamma),
        "spectral_norm": np.zeros(n_gamma),
        "energy_by_size": np.zeros((n_gamma, 2 * p + 2, 2 * p + 2)),
        "k99_cov": np.zeros(n_gamma, dtype=int),
    }

    results_saddle = None
    if run_saddle:
        results_saddle = {
            "p": p,
            "gamma_values": np.array(gamma_list),
            "singular_values": np.zeros((n_gamma, d)),
            "k90": np.zeros(n_gamma, dtype=int),
            "k95": np.zeros(n_gamma, dtype=int),
            "k99": np.zeros(n_gamma, dtype=int),
            "k999": np.zeros(n_gamma, dtype=int),
            "stable_rank": np.zeros(n_gamma),
            "frobenius_norm": np.zeros(n_gamma),
            "spectral_norm": np.zeros(n_gamma),
            "energy_by_size": np.zeros((n_gamma, 2 * p + 2, 2 * p + 2)),
            "saddle_converged": np.zeros(n_gamma, dtype=bool),
            "saddle_residual": np.zeros(n_gamma),
            "kappa": np.zeros(n_gamma),
            "ipr": np.zeros(n_gamma),
            "entropy": np.zeros(n_gamma),
            "k99_cov": np.zeros(n_gamma, dtype=int),
        }

    A = build_structure_matrix(p)
    betas = np.full(p, DEFAULT_BETA)
    b_s = compute_b_s(p, betas)
    for i, gamma in enumerate(gamma_list):
        out0, out_s = run_single_p_gamma(
            p, float(gamma), run_saddle=run_saddle, run_validation=(run_validation_once and i == 0),
            A=A, b_s=b_s,
        )
        results_y0["singular_values"][i] = out0["singular_values"]
        results_y0["k90"][i] = out0["k_90"]
        results_y0["k95"][i] = out0["k_95"]
        results_y0["k99"][i] = out0["k_99"]
        results_y0["k999"][i] = out0["k_999"]
        results_y0["stable_rank"][i] = out0["stable_rank"]
        results_y0["frobenius_norm"][i] = out0["frobenius_norm"]
        results_y0["spectral_norm"][i] = out0["spectral_norm"]
        results_y0["energy_by_size"][i] = out0["energy_by_size"]
        results_y0["k99_cov"][i] = out0["k99_cov"]

        if run_saddle and out_s is not None:
            results_saddle["singular_values"][i] = out_s["singular_values"]
            results_saddle["k90"][i] = out_s["k_90"]
            results_saddle["k95"][i] = out_s["k_95"]
            results_saddle["k99"][i] = out_s["k_99"]
            results_saddle["k999"][i] = out_s["k_999"]
            results_saddle["stable_rank"][i] = out_s["stable_rank"]
            results_saddle["frobenius_norm"][i] = out_s["frobenius_norm"]
            results_saddle["spectral_norm"][i] = out_s["spectral_norm"]
            results_saddle["energy_by_size"][i] = out_s["energy_by_size"]
            results_saddle["saddle_converged"][i] = out_s["converged"]
            results_saddle["saddle_residual"][i] = out_s["saddle_residual"]
            results_saddle["kappa"][i] = out_s["kappa"]
            results_saddle["ipr"][i] = out_s["ipr"]
            results_saddle["entropy"][i] = out_s["entropy"]
            results_saddle["k99_cov"][i] = out_s["k99_cov"]

    return results_y0, results_saddle


def run_probe1_saddle_determinant_ratio(
    p: int = 5,
    gamma_list: list[float] | np.ndarray | None = None,
    A: np.ndarray | None = None,
    b_s: np.ndarray | None = None,
) -> list[dict]:
    """
    Probe 1 (saddle): determinant-ratio probe. For each γ at the saddle, compute
    R(k₀) = ∏_{k>k₀} (½)/(½ − λ_k) from eigenvalues of H_log (sorted by |λ_k| desc,
    non-null only). Plot |R(k₀)| − 1 vs k₀; mark k₀ = ⌈r_s⌉ and k₀ = k_99.
    Directly tests whether truncating at stable rank gives |R−1| ≪ 1 (stable rank
    wins) or we need k_99 (k_99 wins).
    Returns list of dicts: gamma, k_99, stable_rank, k0_axis, err_array, converged.
    """
    if gamma_list is None:
        gamma_list = [0.3, 0.5, 1.0, 1.5, 2.0, np.pi, 3.5, 2 * np.pi]
    gamma_list = np.atleast_1d(gamma_list)
    if A is None:
        A = build_structure_matrix(p)
    if b_s is None:
        b_s = compute_b_s(p, np.full(p, DEFAULT_BETA))
    results = []
    for gamma in gamma_list:
        gamma = float(gamma)
        gammas = np.full(p, gamma)
        c_alpha = compute_c_alpha(p, gammas)
        y_star, converged, _ = saddle_with_adaptive_damping(A, b_s, c_alpha)
        H = hessian_log_at_y(y_star, A, b_s, c_alpha)
        spec = spectral_summary(H)
        k_99 = spec["k_99"]
        r_s = spec["stable_rank"]
        k0_axis, err_array = determinant_ratio_probe(H)
        results.append({
            "gamma": gamma,
            "k_99": k_99,
            "stable_rank": r_s,
            "k0_axis": k0_axis,
            "err_array": err_array,
            "converged": converged,
        })
    return results


def save_npz(results_y0: dict, results_saddle: dict | None, p: int, suffix: str = ""):
    """Save to spectral_data_p{p}_y0.npz and spectral_data_p{p}_saddle.npz."""
    base = DATA_DIR / f"spectral_data_p{p}{suffix}"
    np.savez_compressed(
        base.with_name(base.name + "_y0.npz"),
        gamma_values=results_y0["gamma_values"],
        singular_values=results_y0["singular_values"],
        k90=results_y0["k90"],
        k95=results_y0["k95"],
        k99=results_y0["k99"],
        k999=results_y0["k999"],
        stable_rank=results_y0["stable_rank"],
        frobenius_norm=results_y0["frobenius_norm"],
        spectral_norm=results_y0["spectral_norm"],
        energy_by_size=results_y0["energy_by_size"],
        k99_cov=results_y0["k99_cov"],
    )
    if results_saddle is not None:
        np.savez_compressed(
            base.with_name(base.name + "_saddle.npz"),
            gamma_values=results_saddle["gamma_values"],
            singular_values=results_saddle["singular_values"],
            k90=results_saddle["k90"],
            k95=results_saddle["k95"],
            k99=results_saddle["k99"],
            k999=results_saddle["k999"],
            stable_rank=results_saddle["stable_rank"],
            frobenius_norm=results_saddle["frobenius_norm"],
            spectral_norm=results_saddle["spectral_norm"],
            energy_by_size=results_saddle["energy_by_size"],
            saddle_converged=results_saddle["saddle_converged"],
            saddle_residual=results_saddle["saddle_residual"],
            kappa=results_saddle["kappa"],
            ipr=results_saddle["ipr"],
            entropy=results_saddle["entropy"],
            k99_cov=results_saddle["k99_cov"],
        )


def main():
    """Run sweep for p=1..5 on discrete gamma, save data, then optional dense for plots."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--p-max", type=int, default=5)
    parser.add_argument("--p", type=int, default=None, help="Single p (use with --gamma to run one point)")
    parser.add_argument("--gamma", type=float, default=None, help="Single gamma (use with --p to run one point)")
    parser.add_argument("--dense", action="store_true", help="Use dense gamma sweep (80 points)")
    parser.add_argument("--no-saddle", action="store_true", help="Skip saddle computation")
    parser.add_argument("--validate", action="store_true", help="Run validation checks once per p")
    parser.add_argument("--analyze-kept-dims", action="store_true", help="Compute and print which α dimensions are kept/dropped for k_99 (single --p/--gamma only; uses full SVD)")
    parser.add_argument("--plot", action="store_true", help="With --analyze-kept-dims: save plots to symmetry_reduction/figures/")
    parser.add_argument("--probe1", action="store_true", help="Run Probe 1 (saddle integral error: Frobenius vs spectral tail)")
    args = parser.parse_args()

    if args.p is not None and args.gamma is not None:
        p, gamma = args.p, args.gamma
        print(f"Single run: p={p}, γ={gamma}, saddle={not args.no_saddle}, analyze_kept_dims={args.analyze_kept_dims}")
        out_y0, out_saddle = run_single_p_gamma(
            p, gamma, run_saddle=not args.no_saddle, analyze_kept_dims=args.analyze_kept_dims
        )
        print(f"  y=0: k_99={out_y0['k_99']}, stable_rank={out_y0['stable_rank']:.2f}")
        if out_saddle is not None:
            print(f"  saddle: k_99={out_saddle['k_99']}, stable_rank={out_saddle['stable_rank']:.2f}")
        if args.plot and args.analyze_kept_dims:
            from .plotting import plot_analyze_kept_dims
            plot_analyze_kept_dims(out_y0, out_saddle, p, gamma)
        return

    if args.probe1:
        p = 5
        gamma_list = [0.3, 0.5, 1.0, 1.5, 2.0, np.pi, 3.5, 2 * np.pi]
        print(f"Running Probe 1 (saddle determinant-ratio, p={p}) over {len(gamma_list)} γ values...")
        results = run_probe1_saddle_determinant_ratio(p=p, gamma_list=gamma_list)
        out = DATA_DIR / f"probe1_saddle_p{p}.npz"
        np.savez_compressed(
            out,
            gamma=np.array([r["gamma"] for r in results]),
            k99=np.array([r["k_99"] for r in results]),
            stable_rank=np.array([r["stable_rank"] for r in results]),
            k0_axis=np.array([r["k0_axis"] for r in results], dtype=object),
            err_array=np.array([r["err_array"] for r in results], dtype=object),
        )
        print(f"  Saved {out}")
        print("  Plot with: from symmetry_reduction.plotting import load_probe1, plot_probe1_saddle_determinant_ratio; plot_probe1_saddle_determinant_ratio(load_probe1(5))")
        return

    gamma_list = GAMMA_DENSE if args.dense else GAMMA_DISCRETE
    for p in range(1, args.p_max + 1):
        print(f"Running p={p} (d={1 << (2*p+1)}) over {len(gamma_list)} γ values...")
        r0, r1 = run_sweep_p(p, gamma_list, run_saddle=not args.no_saddle, run_validation_once=args.validate)
        save_npz(r0, r1, p, suffix="_dense" if args.dense else "")
        print(f"  Saved data for p={p}.")
    print("Done.")


if __name__ == "__main__":
    main()
