"""
Scaling analysis: fit k_99(p) = C * p^alpha for each gamma.
Also fit exponential k_99(p) = a * b^p and compare R^2.
"""

import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"


def load_k99_vs_p(p_values, saddle=False):
    """Load k99 arrays for each p. Returns dict {p: {"gamma_values": array, "k99": array}}."""
    suffix = "_saddle" if saddle else "_y0"
    all_data = {}
    for p in p_values:
        path = DATA_DIR / f"spectral_data_p{p}{suffix}.npz"
        if path.exists():
            d = np.load(path)
            all_data[p] = {"gamma_values": d["gamma_values"], "k99": d["k99"]}
    return all_data


def load_stable_rank_vs_p(p_values, saddle: bool = False):
    """
    Load stable rank arrays for each p.

    Returns dict {p: {"gamma_values": array, "stable_rank": array}}.
    """
    suffix = "_saddle" if saddle else "_y0"
    all_data = {}
    for p in p_values:
        path = DATA_DIR / f"spectral_data_p{p}{suffix}.npz"
        if path.exists():
            d = np.load(path)
            all_data[p] = {"gamma_values": d["gamma_values"], "stable_rank": d["stable_rank"]}
    return all_data


def fit_power_law(p_arr, k99_arr):
    """Fit log(k99) = alpha * log(p) + log(C). Returns (alpha, C, R^2)."""
    mask = (p_arr >= 2) & (k99_arr > 0)
    if np.sum(mask) < 2:
        return np.nan, np.nan, np.nan
    log_p = np.log(p_arr[mask].astype(float))
    log_k = np.log(k99_arr[mask].astype(float))
    coeffs = np.polyfit(log_p, log_k, 1)
    alpha, log_C = coeffs
    C = np.exp(log_C)
    predicted = np.polyval(coeffs, log_p)
    ss_res = np.sum((log_k - predicted) ** 2)
    ss_tot = np.sum((log_k - np.mean(log_k)) ** 2)
    R2 = 1 - ss_res / (ss_tot + 1e-30)
    return alpha, C, R2


def fit_exponential(p_arr, k99_arr):
    """Fit log(k99) = p * log(b) + log(a). Returns (a, b, R^2)."""
    mask = (p_arr >= 2) & (k99_arr > 0)
    if np.sum(mask) < 2:
        return np.nan, np.nan, np.nan
    p_vals = p_arr[mask].astype(float)
    log_k = np.log(k99_arr[mask].astype(float))
    coeffs = np.polyfit(p_vals, log_k, 1)
    log_b, log_a = coeffs
    a, b = np.exp(log_a), np.exp(log_b)
    predicted = np.polyval(coeffs, p_vals)
    ss_res = np.sum((log_k - predicted) ** 2)
    ss_tot = np.sum((log_k - np.mean(log_k)) ** 2)
    R2 = 1 - ss_res / (ss_tot + 1e-30)
    return a, b, R2


def scaling_table(p_values=(2, 3, 4, 5), saddle=False):
    """
    Print table: for each gamma, power-law and exponential fits.
    Also print predicted k_99 at p=14.
    If p=6 is in p_values and has fewer gamma points, use that gamma list and for each p
    take k99 at the nearest gamma.
    """
    all_data = load_k99_vs_p(p_values, saddle=saddle)
    if not all_data:
        print("No data found.")
        return None

    p_arr = np.array(sorted(all_data.keys()))
    # Use finest gamma grid that all p have, or if p=6 present use its (shorter) gamma list
    if 6 in all_data and len(all_data[6]["gamma_values"]) < len(all_data[p_arr[0]]["gamma_values"]):
        gamma_values = all_data[6]["gamma_values"]
        def get_k99(p, gi):
            g = gamma_values[gi]
            idx = np.argmin(np.abs(all_data[p]["gamma_values"] - g))
            return all_data[p]["k99"][idx]
    else:
        gamma_values = all_data[p_arr[0]]["gamma_values"]
        def get_k99(p, gi):
            return all_data[p]["k99"][gi]

    label = "saddle" if saddle else "y=0"
    print(f"\nScaling analysis ({label}):")
    print(
        f"{'γ':>6} | {'α (power)':>10} | {'C':>8} | {'R²_pow':>7} | "
        f"{'b (exp)':>8} | {'R²_exp':>7} | {'k99(p=14) pow':>14} | {'k99(p=14) exp':>14}"
    )
    print("-" * 100)

    for gi in range(len(gamma_values)):
        gamma = gamma_values[gi]
        k99_vals = np.array([get_k99(p, gi) for p in p_arr])
        alpha, C, R2_pow = fit_power_law(p_arr, k99_vals)
        a, b, R2_exp = fit_exponential(p_arr, k99_vals)

        pred_pow = C * 14**alpha if not np.isnan(alpha) else np.nan
        pred_exp = a * b**14 if not np.isnan(b) else np.nan

        print(
            f"{gamma:>6.3f} | {alpha:>10.3f} | {C:>8.2f} | {R2_pow:>7.4f} | "
            f"{b:>8.3f} | {R2_exp:>7.4f} | {pred_pow:>14.0f} | {pred_exp:>14.0f}"
        )
    return all_data


def get_alphas_per_gamma(p_values=(2, 3, 4, 5), saddle=False):
    """For each gamma index, return power-law exponent alpha. Returns (gamma_values, alpha_array)."""
    all_data = load_k99_vs_p(p_values, saddle=saddle)
    if not all_data:
        return np.array([]), np.array([])
    p_arr = np.array(sorted(all_data.keys()))
    if 6 in all_data and len(all_data[6]["gamma_values"]) < len(all_data[p_arr[0]]["gamma_values"]):
        gamma_values = all_data[6]["gamma_values"]
        def get_k99(p, gi):
            g = gamma_values[gi]
            idx = np.argmin(np.abs(all_data[p]["gamma_values"] - g))
            return all_data[p]["k99"][idx]
    else:
        gamma_values = all_data[p_arr[0]]["gamma_values"]
        get_k99 = lambda p, gi: all_data[p]["k99"][gi]
    alphas = []
    for gi in range(len(gamma_values)):
        k99_vals = np.array([get_k99(p, gi) for p in p_arr])
        alpha, _, _ = fit_power_law(p_arr, k99_vals)
        alphas.append(alpha)
    return gamma_values, np.array(alphas)


def get_alphas_with_leave_one_out(p_values=(2, 3, 4, 5, 6), saddle: bool = False):
    """
    For each gamma index, compute:
      - alpha_full: power-law exponent using all available p in p_values
      - alpha_min/alpha_max: extrema of leave-one-out fits,
        where each fit drops a single p from p_values.

    Returns (gamma_values, alpha_full, alpha_min, alpha_max).
    """
    all_data = load_k99_vs_p(p_values, saddle=saddle)
    if not all_data:
        return np.array([]), np.array([]), np.array([]), np.array([])

    p_arr = np.array(sorted(all_data.keys()))
    if 6 in all_data and len(all_data[6]["gamma_values"]) < len(all_data[p_arr[0]]["gamma_values"]):
        gamma_values = all_data[6]["gamma_values"]

        def get_k99(p, gi):
            g = gamma_values[gi]
            idx = np.argmin(np.abs(all_data[p]["gamma_values"] - g))
            return all_data[p]["k99"][idx]
    else:
        gamma_values = all_data[p_arr[0]]["gamma_values"]

        def get_k99(p, gi):
            return all_data[p]["k99"][gi]

    alphas_full = []
    alphas_min = []
    alphas_max = []
    n_p = len(p_arr)

    for gi in range(len(gamma_values)):
        k99_vals = np.array([get_k99(p, gi) for p in p_arr], dtype=float)
        alpha_full, _, _ = fit_power_law(p_arr, k99_vals)

        loo_alphas = []
        for drop_idx in range(n_p):
            p_loo = np.delete(p_arr, drop_idx)
            k_loo = np.delete(k99_vals, drop_idx)
            alpha_loo, _, _ = fit_power_law(p_loo, k_loo)
            if not np.isnan(alpha_loo):
                loo_alphas.append(alpha_loo)

        if loo_alphas:
            a_min = float(np.min(loo_alphas))
            a_max = float(np.max(loo_alphas))
        else:
            a_min = float("nan")
            a_max = float("nan")

        alphas_full.append(alpha_full)
        alphas_min.append(a_min)
        alphas_max.append(a_max)

    return (
        gamma_values,
        np.array(alphas_full),
        np.array(alphas_min),
        np.array(alphas_max),
    )


def stable_rank_k99_summary(p_values=(2, 3, 4, 5, 6), saddle: bool = False):
    """
    For each gamma where data exists for all p in p_values, collect
    (p, k99, stable_rank, k99 / stable_rank) arrays.

    Returns (gamma_values, summary) where:
      - gamma_values is a 1D array of shared gamma points
      - summary is dict mapping gamma_index -> dict with keys:
          "p": 1D array of p values
          "k99": 1D array of k99(p, gamma)
          "stable_rank": 1D array of r_s(p, gamma)
          "ratio": 1D array of k99 / r_s
    """
    k99_data = load_k99_vs_p(p_values, saddle=saddle)
    rs_data = load_stable_rank_vs_p(p_values, saddle=saddle)
    if not k99_data or not rs_data:
        return np.array([]), {}

    p_arr = np.array(sorted(k99_data.keys()))
    # Determine a common gamma grid by intersecting all available gamma_values
    gamma_ref = None
    for p in p_arr:
        g = k99_data[p]["gamma_values"]
        gamma_ref = g if gamma_ref is None else np.intersect1d(gamma_ref, g)
    if gamma_ref is None or len(gamma_ref) == 0:
        return np.array([]), {}

    def nearest_idx(gamma_array, target):
        return int(np.argmin(np.abs(gamma_array - target)))

    summary = {}
    for gi, g in enumerate(gamma_ref):
        k_vals = []
        r_vals = []
        for p in p_arr:
            g_arr_k = k99_data[p]["gamma_values"]
            g_arr_r = rs_data[p]["gamma_values"]
            idx_k = nearest_idx(g_arr_k, g)
            idx_r = nearest_idx(g_arr_r, g)
            k_vals.append(k99_data[p]["k99"][idx_k])
            r_vals.append(rs_data[p]["stable_rank"][idx_r])
        k_vals = np.asarray(k_vals, dtype=float)
        r_vals = np.asarray(r_vals, dtype=float)
        ratio = np.where(r_vals > 0, k_vals / r_vals, np.nan)
        summary[gi] = {
            "p": p_arr.copy(),
            "k99": k_vals,
            "stable_rank": r_vals,
            "ratio": ratio,
        }
    return gamma_ref, summary
