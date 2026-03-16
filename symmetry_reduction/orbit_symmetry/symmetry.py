"""
Symmetry reduction of H_log under S_p × S_p × Z_2 (uniform angles).

Each subset α is classified by the orbit (a, b, c) where:
  a = |α ∩ {0,...,p-1}|
  b = 1[p ∈ α]
  c = |α ∩ {p+1,...,2p}|

With Z_2 mirror symmetry: (a,b,c) ~ (c,b,a).
"""

import numpy as np
import random
from collections import defaultdict

from ..core import (
    popcount,
    build_structure_matrix,
    compute_b_s,
    compute_c_alpha,
    hessian_log_at_y,
    weights_at_zero,
    DEFAULT_BETA,
)
from ..spectral import spectral_summary
from ..saddle import saddle_with_adaptive_damping


def classify_alpha(alpha, p):
    """
    Given subset α as bitmask and depth p, return (a, b, c).
    Positions: {0,...,p-1} are "forward", p is "middle", {p+1,...,2p} are "backward".
    """
    a = 0  # count of forward positions
    for j in range(p):
        if (alpha >> j) & 1:
            a += 1
    b = (alpha >> p) & 1  # middle position
    c = 0  # count of backward positions
    for j in range(p + 1, 2 * p + 1):
        if (alpha >> j) & 1:
            c += 1
    return (a, b, c)


def canonical_orbit(a, b, c):
    """Canonical representative under Z_2: (a,b,c) ~ (c,b,a). Return sorted."""
    return (min(a, c), b, max(a, c))


def build_orbit_map(p):
    """
    Map each α to its canonical orbit (a,b,c).

    Returns:
      orbit_of_alpha: dict {alpha_int: (a,b,c) canonical}
      orbits: sorted list of distinct canonical (a,b,c) triples
      orbit_members: dict {(a,b,c): [list of alpha_int in this orbit]}
      n_orbits: number of distinct orbits
    """
    n = 2 * p + 1
    d = 1 << n
    orbit_of_alpha = {}
    orbit_members = defaultdict(list)

    for alpha in range(d):
        a, b, c = classify_alpha(alpha, p)
        canon = canonical_orbit(a, b, c)
        orbit_of_alpha[alpha] = canon
        orbit_members[canon].append(alpha)

    orbits = sorted(orbit_members.keys())
    return orbit_of_alpha, orbits, dict(orbit_members), len(orbits)


def orbit_averaged_hessian(H_log, p):
    """
    Project H_log onto orbit-averaged basis.

    For each pair of orbits (O_i, O_j), the averaged entry is:
      H_avg[i,j] = (1/|O_i|) * (1/|O_j|) * sum_{α∈O_i, α'∈O_j} H_log[α, α']

    But for a proper projection, we want the restriction of H_log to the
    orbit-symmetric subspace. This is:
      H_red[i,j] = sum_{α∈O_i, α'∈O_j} H_log[α, α'] / sqrt(|O_i| * |O_j|)

    Returns: (H_red, orbits, orbit_members)
    """
    orbit_of_alpha, orbits, orbit_members, n_orbits = build_orbit_map(p)

    # Build projection: for each orbit i, the basis vector is
    # e_i = (1/sqrt(|O_i|)) * sum_{α ∈ O_i} e_α
    H_red = np.zeros((n_orbits, n_orbits), dtype=complex)

    for i, orb_i in enumerate(orbits):
        members_i = orbit_members[orb_i]
        for j, orb_j in enumerate(orbits):
            members_j = orbit_members[orb_j]
            # Sum H_log[α, α'] over all α in O_i, α' in O_j
            total = 0.0 + 0j
            for alpha in members_i:
                for alpha_p in members_j:
                    total += H_log[alpha, alpha_p]
            H_red[i, j] = total / np.sqrt(len(members_i) * len(members_j))

    return H_red, orbits, orbit_members


def symmetry_analysis(p, gamma, beta=DEFAULT_BETA, saddle=False):
    """
    Full symmetry reduction analysis for one (p, γ).

    If saddle=True, compute H_log at y_0^* (saddle) instead of y=0.

    Returns dict with:
      - n_orbits: number of distinct orbits
      - H_red: reduced Hessian (n_orbits × n_orbits)
      - singular_values_red: SVs of reduced Hessian
      - k99_red: k_99 of reduced Hessian
      - stable_rank_red: stable rank of reduced Hessian
      - k99_full: k_99 of full Hessian (for comparison)
      - stable_rank_full: stable rank of full Hessian
      - energy_captured: fraction of full Frobenius energy in symmetric subspace
    """
    d = 1 << (2 * p + 1)
    betas = np.full(p, beta)
    gammas = np.full(p, gamma)

    A = build_structure_matrix(p)
    b_s = compute_b_s(p, betas)
    c_alpha = compute_c_alpha(p, gammas)

    if saddle:
        y_star, converged, residual = saddle_with_adaptive_damping(A, b_s, c_alpha)
        y0 = y_star
    else:
        y0 = np.zeros(d, dtype=complex)

    # Full Hessian at y=0 or y_star
    H_full = hessian_log_at_y(y0, A, b_s, c_alpha)
    spec_full = spectral_summary(H_full)

    # Orbit structure
    orbit_of_alpha, orbits, orbit_members, n_orbits = build_orbit_map(p)

    # Reduced Hessian
    H_red, _, _ = orbit_averaged_hessian(H_full, p)
    spec_red = spectral_summary(H_red)

    # Energy captured by symmetric subspace
    frob_full_sq = np.sum(np.abs(H_full) ** 2)
    frob_red_sq = np.sum(np.abs(H_red) ** 2)
    energy_captured = frob_red_sq / (frob_full_sq + 1e-30)

    # Orbit sizes
    orbit_sizes = [(orb, len(orbit_members[orb])) for orb in orbits]

    return {
        "p": p,
        "gamma": gamma,
        "saddle": saddle,
        "d_full": d,
        "n_orbits": n_orbits,
        "H_red": H_red,
        "sv_red": spec_red["singular_values"],
        "k99_red": spec_red["k_99"],
        "k95_red": spec_red["k_95"],
        "k90_red": spec_red["k_90"],
        "stable_rank_red": spec_red["stable_rank"],
        "k99_full": spec_full["k_99"],
        "k95_full": spec_full["k_95"],
        "k90_full": spec_full["k_90"],
        "stable_rank_full": spec_full["stable_rank"],
        "energy_captured": energy_captured,
        "orbit_sizes": orbit_sizes,
    }


def print_symmetry_analysis(result):
    """Print formatted symmetry analysis results."""
    r = result
    loc = "saddle" if r.get("saddle") else "y=0"
    print(f"\n{'='*60}")
    print(f"Symmetry reduction: p={r['p']}, γ={r['gamma']:.4f} ({loc})")
    print(f"{'='*60}")
    print(f"Full dimension:     d = {r['d_full']}")
    print(f"Number of orbits:   {r['n_orbits']}")
    print(f"Reduction factor:   {r['d_full'] / r['n_orbits']:.1f}x")
    print(f"Energy captured:    {r['energy_captured']:.6f} ({r['energy_captured']*100:.2f}%)")
    print(f"")
    print(f"                    Full         Reduced")
    print(f"  k_90:             {r['k90_full']:>6}       {r['k90_red']:>6}")
    print(f"  k_95:             {r['k95_full']:>6}       {r['k95_red']:>6}")
    print(f"  k_99:             {r['k99_full']:>6}       {r['k99_red']:>6}")
    print(f"  Stable rank:      {r['stable_rank_full']:>8.3f}     {r['stable_rank_red']:>8.3f}")
    print(f"")

    # Show orbit structure
    print(f"Orbit (a,b,c) → size (showing largest and smallest):")
    sizes = sorted(r["orbit_sizes"], key=lambda x: -x[1])
    for orb, sz in sizes[:5]:
        print(f"  {orb}: {sz} subsets")
    print(f"  ...")
    for orb, sz in sizes[-3:]:
        print(f"  {orb}: {sz} subsets")


def run_symmetry_sweep(
    p_values=(2, 3, 4, 5),
    gamma_values=(0.3, 1.0, np.pi, 2 * np.pi),
    saddle=False,
):
    """Run symmetry analysis across p and gamma values."""
    results = []
    for p in p_values:
        for gamma in gamma_values:
            print(f"Running p={p}, γ={gamma:.2f} (saddle={saddle})...")
            r = symmetry_analysis(p, gamma, saddle=saddle)
            print_symmetry_analysis(r)
            results.append(r)

    # Summary table
    print(f"\n\n{'='*80}")
    print(f"SUMMARY: k_99 (full → reduced)  [{'saddle' if saddle else 'y=0'}]")
    print(f"{'='*80}")
    print(
        f"{'p':>3} | {'γ':>6} | {'d':>6} | {'#orbits':>7} | {'k99_full':>8} | {'k99_red':>7} | {'rs_full':>7} | {'rs_red':>6} | {'E_capt':>6}"
    )
    print("-" * 80)
    for r in results:
        print(
            f"{r['p']:>3} | {r['gamma']:>6.2f} | {r['d_full']:>6} | {r['n_orbits']:>7} | "
            f"{r['k99_full']:>8} | {r['k99_red']:>7} | {r['stable_rank_full']:>7.2f} | "
            f"{r['stable_rank_red']:>6.2f} | {r['energy_captured']:>6.4f}"
        )

    return results


def check_symmetry_commutation(H_log, p, n_samples=10):
    """Check that H_log commutes with S_p × S_p × Z_2."""
    n = 2 * p + 1
    d = 1 << n

    max_err = 0.0
    for _ in range(n_samples):
        # Random permutation: shuffle {0,...,p-1} and {p+1,...,2p} independently
        perm_fwd = list(range(p))
        random.shuffle(perm_fwd)
        perm_bwd = list(range(p + 1, 2 * p + 1))
        random.shuffle(perm_bwd)

        apply_mirror = random.random() < 0.5

        # Build full position permutation: pos_perm[old_pos] = new_pos
        pos_perm = [0] * n
        if apply_mirror:
            # Z_2: swap forward and backward blocks (with optional shuffle within each)
            for i in range(p):
                pos_perm[perm_fwd[i]] = perm_bwd[i]
                pos_perm[perm_bwd[i]] = perm_fwd[i]
            pos_perm[p] = p
        else:
            for j in range(p):
                pos_perm[j] = perm_fwd[j]
            pos_perm[p] = p
            for k in range(p):
                pos_perm[p + 1 + k] = perm_bwd[k]

        # Build subset permutation: α → π(α)
        subset_perm = np.zeros(d, dtype=int)
        for alpha in range(d):
            new_alpha = 0
            for j in range(n):
                if (alpha >> j) & 1:
                    new_alpha |= 1 << pos_perm[j]
            subset_perm[alpha] = new_alpha

        # Permute H_log: H_perm[α, α'] = H_log[π^{-1}(α), π^{-1}(α')]
        inv_perm = np.argsort(subset_perm)
        H_perm = H_log[np.ix_(inv_perm, inv_perm)]
        err = np.max(np.abs(H_log - H_perm))
        max_err = max(max_err, err)

    print(f"  Symmetry commutation check: max error = {max_err:.2e}")
    return max_err < 1e-8


# --- Complement truncation (Task 2) ---


def complement_truncation_analysis(
    p,
    gamma,
    beta=DEFAULT_BETA,
    max_complement_sizes=(0, 1, 2, 3, 4, 5),
):
    """
    For each complement cutoff m, restrict H_log to subsets with |α^c| ≤ m
    and compute k_99 and Frobenius energy capture.

    Also do the dual: restrict to |α| ≤ m (normal truncation).

    Returns dict with results for each m.
    """
    d = 1 << (2 * p + 1)
    n = 2 * p + 1
    betas = np.full(p, beta)
    gammas = np.full(p, gamma)

    A = build_structure_matrix(p)
    b_s = compute_b_s(p, betas)
    c_alpha = compute_c_alpha(p, gammas)

    y0 = np.zeros(d, dtype=complex)
    H_full = hessian_log_at_y(y0, A, b_s, c_alpha)
    frob_full_sq = np.sum(np.abs(H_full) ** 2)
    spec_full = spectral_summary(H_full)

    results = {
        "full": {"k99": spec_full["k_99"], "stable_rank": spec_full["stable_rank"]}
    }

    for m in max_complement_sizes:
        # Complement truncation: keep α where |α^c| ≤ m, i.e., |α| ≥ n - m
        complement_mask = np.array(
            [popcount(alpha) >= n - m for alpha in range(d)], dtype=bool
        )
        n_kept_comp = np.sum(complement_mask)

        if n_kept_comp > 0 and n_kept_comp < d:
            idx_comp = np.where(complement_mask)[0]
            H_comp = H_full[np.ix_(idx_comp, idx_comp)]
            spec_comp = spectral_summary(H_comp)
            frob_comp_sq = np.sum(np.abs(H_comp) ** 2)

            results[f"complement_le_{m}"] = {
                "n_kept": int(n_kept_comp),
                "k99": spec_comp["k_99"],
                "stable_rank": spec_comp["stable_rank"],
                "energy_captured": float(frob_comp_sq / (frob_full_sq + 1e-30)),
            }

        # Normal truncation: keep α where |α| ≤ m
        normal_mask = np.array(
            [popcount(alpha) <= m for alpha in range(d)], dtype=bool
        )
        n_kept_norm = np.sum(normal_mask)

        if n_kept_norm > 0 and n_kept_norm < d:
            idx_norm = np.where(normal_mask)[0]
            H_norm = H_full[np.ix_(idx_norm, idx_norm)]
            spec_norm = spectral_summary(H_norm)
            frob_norm_sq = np.sum(np.abs(H_norm) ** 2)

            results[f"normal_le_{m}"] = {
                "n_kept": int(n_kept_norm),
                "k99": spec_norm["k_99"],
                "stable_rank": spec_norm["stable_rank"],
                "energy_captured": float(frob_norm_sq / (frob_full_sq + 1e-30)),
            }

    return results


def print_complement_analysis(results, p, gamma):
    """Print complement truncation results."""
    n = 2 * p + 1
    print(f"\n{'='*70}")
    print(f"Complement vs Normal truncation: p={p}, γ={gamma:.4f}, n={n}")
    print(f"{'='*70}")
    print(
        f"Full: k_99={results['full']['k99']}, r_s={results['full']['stable_rank']:.3f}"
    )
    print()
    print(f"{'Truncation':>20} | {'n_kept':>6} | {'k99':>5} | {'r_s':>6} | {'Energy%':>8}")
    print("-" * 60)

    for key in sorted(results.keys()):
        if key == "full":
            continue
        r = results[key]
        label = key.replace("complement_le_", "|α^c|≤").replace("normal_le_", "|α|≤")
        print(
            f"{label:>20} | {r['n_kept']:>6} | {r['k99']:>5} | {r['stable_rank']:>6.2f} | {r['energy_captured']*100:>7.2f}%"
        )


def run_complement_sweep(
    p_values=(3, 4, 5),
    gamma_values=(0.3, 1.0, np.pi, 2 * np.pi),
):
    """Run complement analysis across p and gamma."""
    for p in p_values:
        for gamma in gamma_values:
            results = complement_truncation_analysis(p, gamma)
            print_complement_analysis(results, p, gamma)


# --- Diagonal orbit analysis (Task 5) ---


def diagonal_orbit_analysis(p, gamma, beta=DEFAULT_BETA):
    """
    Check if energy concentrates on diagonal orbits (a = c).

    Diagonal orbits: (a, b, a) for a = 0,...,p and b ∈ {0,1}.
    Number of diagonal orbits: 2(p+1).
    """
    d = 1 << (2 * p + 1)
    betas = np.full(p, beta)
    gammas = np.full(p, gamma)

    A = build_structure_matrix(p)
    b_s = compute_b_s(p, betas)
    c_alpha = compute_c_alpha(p, gammas)
    y0 = np.zeros(d, dtype=complex)
    H_full = hessian_log_at_y(y0, A, b_s, c_alpha)
    frob_full_sq = np.sum(np.abs(H_full) ** 2)

    orbit_of_alpha, orbits, orbit_members, n_orbits = build_orbit_map(p)

    # Partition orbits into diagonal (a=c) and off-diagonal (a≠c)
    diagonal_orbits = [orb for orb in orbits if orb[0] == orb[2]]
    off_diagonal_orbits = [orb for orb in orbits if orb[0] != orb[2]]

    # Indices of subsets in diagonal orbits
    diag_alphas = []
    for orb in diagonal_orbits:
        diag_alphas.extend(orbit_members[orb])
    diag_alphas = sorted(set(diag_alphas))

    # Frobenius energy in diagonal block
    idx_diag = np.array(diag_alphas, dtype=int)
    H_diag = H_full[np.ix_(idx_diag, idx_diag)]
    frob_diag_sq = np.sum(np.abs(H_diag) ** 2)
    energy_diagonal = frob_diag_sq / (frob_full_sq + 1e-30)

    # k_99 of H restricted to diagonal orbits
    spec_diag = spectral_summary(H_diag)

    return {
        "p": p,
        "gamma": gamma,
        "n_diagonal_orbits": len(diagonal_orbits),
        "n_off_diagonal_orbits": len(off_diagonal_orbits),
        "n_diagonal_subsets": len(diag_alphas),
        "energy_diagonal": energy_diagonal,
        "k99_diagonal": spec_diag["k_99"],
        "stable_rank_diagonal": spec_diag["stable_rank"],
    }
