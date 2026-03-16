"""
Formatted summary tables (Section 6.2): dimensions, ranks, spectral norms, k_η, top singular values.
"""

import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"


def load_p_data(p: int, saddle: bool = False) -> dict | None:
    path = DATA_DIR / f"spectral_data_p{p}_{'saddle' if saddle else 'y0'}.npz"
    if not path.exists():
        return None
    return dict(np.load(path, allow_pickle=True))


def table_dimensions_ranks(p_max: int = 5) -> str:
    """Table: dimensions and ranks at each p (from first gamma point)."""
    lines = [
        "Table: Dimensions and ranks",
        "p  | d=2^(2p+1) | |S|   | rank(A) | rank(Cov_w0) | # zero eig H_log(0)",
        "-" * 70,
    ]
    for p in range(1, p_max + 1):
        d = 1 << (2 * p + 1)
        n_s = d
        # rank(A) = 2^{2p}, rank(Cov) = 2^{2p}-1, zero eig = 2^{2p}+1
        rank_A = 1 << (2 * p)
        rank_Cov = rank_A - 1
        zero_eig = rank_A + 1
        lines.append(f"{p:2} | {d:10} | {n_s:5} | {rank_A:8} | {rank_Cov:12} | {zero_eig:19}")
    return "\n".join(lines)


def table_spectral_norms_stable_rank(data_y0: dict, data_saddle: dict | None, p: int, gamma_idx: int = 0) -> str:
    """Table: ‖H_log‖, ‖H_log‖_F, r_s at (p, γ) for y=0 and saddle."""
    g = data_y0["gamma_values"][gamma_idx]
    lines = [
        f"p={p}, γ={g:.4f}",
        "         | ‖H_log‖   | ‖H_log‖_F | stable_rank",
        "-" * 50,
        f"y=0     | {data_y0['spectral_norm'][gamma_idx]:.6f}  | {data_y0['frobenius_norm'][gamma_idx]:.6f}  | {data_y0['stable_rank'][gamma_idx]:.4f}",
    ]
    if data_saddle is not None:
        lines.append(f"y_0*    | {data_saddle['spectral_norm'][gamma_idx]:.6f}  | {data_saddle['frobenius_norm'][gamma_idx]:.6f}  | {data_saddle['stable_rank'][gamma_idx]:.4f}")
    return "\n".join(lines)


def table_effective_dimensions(data: dict, p: int, gamma_idx: int = 0) -> str:
    """Table: k_90, k_95, k_99, k_99.9 at (p, γ)."""
    g = data["gamma_values"][gamma_idx]
    return (
        f"p={p}, γ={g:.4f}\n"
        f"k_90 = {data['k90'][gamma_idx]}, k_95 = {data['k95'][gamma_idx]}, "
        f"k_99 = {data['k99'][gamma_idx]}, k_99.9 = {data['k999'][gamma_idx]}"
    )


def table_top_singular_values(data: dict, p: int, gamma_idx: int = 0, top_k: int = 10) -> str:
    """Table: Top 10 singular values at (p, γ)."""
    sigmas = data["singular_values"][gamma_idx]
    top = sigmas[:top_k]
    g = data["gamma_values"][gamma_idx]
    lines = [f"p={p}, γ={g:.4f} — top {top_k} singular values:", " ".join(f"{s:.6f}" for s in top)]
    return "\n".join(lines)


def print_all_tables(p_max: int = 5):
    """Print formatted tables from saved data."""
    print(table_dimensions_ranks(p_max))
    print()
    for p in range(1, min(p_max + 1, 4)):  # smaller p for readability
        d0 = load_p_data(p, saddle=False)
        d1 = load_p_data(p, saddle=True)
        if d0 is not None:
            print(table_spectral_norms_stable_rank(d0, d1, p))
            print(table_effective_dimensions(d0, p))
            print(table_top_singular_values(d0, p))
            print()
