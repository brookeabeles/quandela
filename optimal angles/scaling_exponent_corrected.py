"""
SUCCESS! The solver now converges perfectly across the full r range.

Key findings:
- C_saddle ≈ 0.000979 at r=176.54, p=1 — tiny positive contribution from QAOA
- C_full = -log(2) - r/256 + C_saddle ≈ -1.3818 — barely beats random guessing
- This makes physical sense: p=1 QAOA on 8-SAT at the phase transition is very weak

Now let me write the clean, complete, corrected module.
"""
import numpy as np
from scipy.optimize import root

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, desc=None):
        return iterable


# ================================================================
# Efficient subset-sum transforms
# ================================================================

def _zeta_transform(v, dim):
    """Subset-sum (zeta): out[S] = sum_{T ⊆ S} v[T]"""
    z = v.copy()
    for i in range(dim):
        shape = (1 << (dim - 1 - i), 2, 1 << i)
        zr = z.reshape(shape)
        zr[:, 1, :] += zr[:, 0, :]
    return z


def _superset_transform(v, dim):
    """Superset-sum: out[S] = sum_{T ⊇ S} v[T]"""
    z = v.copy()
    for i in range(dim):
        shape = (1 << (dim - 1 - i), 2, 1 << i)
        zr = z.reshape(shape)
        zr[:, 0, :] += zr[:, 1, :]
    return z


def algorithm_3(v, dim):
    """
    X_s = sum_α A_{α,s} v_α  where A_{α,s} = ½·1[∀j,j'∈α: s_j=s_j']
    
    Compatible α are those where α ⊆ {j: s_j=0} or α ⊆ {j: s_j=1}.
    X_s = ½·(ζ(v)[s] + ζ(v)[complement(s)] - v[0])
    """
    size = 1 << dim
    rev = np.arange(size) ^ (size - 1)
    zv = _zeta_transform(v, dim)
    return (zv + zv[rev] - v[0]) / 2.0


def algorithm_4(v, dim):
    """
    Y_α = sum_s A_{α,s} v_s  where A_{α,s} = ½·1[∀j,j'∈α: s_j=s_j']
    
    For given α: s must have all bits at α-positions equal (all 0 or all 1).
    Y_α = ½·(ζ(v)[complement(α)] + superset(v)[α])
    Special fix at α=0: result is ½·sum(v).
    """
    size = 1 << dim
    rev = np.arange(size) ^ (size - 1)
    zv = _zeta_transform(v, dim)
    sv = _superset_transform(v, dim)
    result = (zv[rev] + sv) / 2.0
    result[0] = np.sum(v) / 2.0
    return result


# ================================================================
# Core computation
# ================================================================

def compute_b_s(betas, p, dim, size, s_bits):
    """
    b_s = ½ · prod_j <s_j|exp(+iβ_j X/2)|s_{j+1}> · <s_{2p-j-1}|exp(-iβ_j X/2)|s_{2p-j}>
    
    exp(+iβX): same→cos(β), diff→+i·sin(β)
    exp(-iβX): same→cos(β), diff→-i·sin(β)
    """
    b = np.ones(size, dtype=np.complex128) / 2.0
    for j in range(p):
        cb = np.cos(betas[j] / 2.0)
        sb = np.sin(betas[j] / 2.0)
        
        fwd_diff = s_bits[j] ^ s_bits[j + 1]
        b *= (cb ** (1 - fwd_diff)) * ((1j * sb) ** fwd_diff)
        
        bwd_diff = s_bits[2*p - j - 1] ^ s_bits[2*p - j]
        b *= (cb ** (1 - bwd_diff)) * ((-1j * sb) ** bwd_diff)
    
    return b


def compute_c_alpha(r, gammas, p, dim, s_bits):
    """
    c_α = r · (-1)^{1{p∈α}} · prod_{j∈α,j<p}(e^{-iγ_j/2}-1) · prod_{j∈α,j>p}(e^{+iγ_{2p-j}/2}-1)
    
    Note: r is included in c_α (matches BM24 Example 16 / Eq. A41).
    """
    c = r * ((-1.0) ** s_bits[p]).astype(np.complex128)
    for j in range(dim):
        if j == p:
            continue
        in_alpha = s_bits[j]
        if j < p:
            gf = np.exp(-1j * gammas[j] / 2.0) - 1.0
        else:
            gf = np.exp(1j * gammas[2*p - j] / 2.0) - 1.0
        c *= np.where(in_alpha, gf, 1.0)
    return c


def get_scaling_exponent_qaoa_ksat(r, gammas, betas, k, verbose=True,
                                    z_init=None, return_z=False):
    """
    Compute the scaling exponent C for QAOA on random k-SAT.
    
    The success probability scales as E[Pr(success)] ~ exp(C·n) where:
        C = -log(2) - r/2^k + F(z*) + (k-1)·Σ_α (∂F/∂z_α)^k
    
    Parameters
    ----------
    r : float           Clause-to-variable ratio
    gammas : list       QAOA γ angles (length p)
    betas : list        QAOA β angles (length p)
    k : int             SAT arity (must be power of 2)
    verbose : bool      Print diagnostics
    z_init : ndarray    Initial guess for z* (from r-homotopy)
    return_z : bool     If True, also return the converged z*
    
    Returns
    -------
    C_full : float      Full scaling exponent (negative means exponentially hard)
    z_star : ndarray    (only if return_z=True) Fixed point vector
    """
    q_val = int(np.log2(k))
    assert 2**q_val == k, "k must be a power of 2"
    p = len(gammas)
    assert len(betas) == p
    dim = 2*p + 1
    size = 1 << dim
    
    s_indices = np.arange(size)
    s_bits = [(s_indices >> j) & 1 for j in range(dim)]
    
    b = compute_b_s(betas, p, dim, size, s_bits)
    c_alpha = compute_c_alpha(r, gammas, p, dim, s_bits)
    
    if verbose:
        print(f"sum(b_s) = {np.sum(b):.10f}  (should be 1.0)")
    
    # prefactor = (-c_α)^{1/k} — r is already inside c_α
    prefactor = (-c_alpha) ** (1.0 / k)
    
    # --- Solve fixed-point equation via Newton (scipy.optimize.root) ---
    def G_real(x):
        z = x[:size] + 1j * x[size:]
        V = prefactor * z
        X = algorithm_3(V, dim)
        X_shift = np.max(X.real)
        E = np.exp(X - X_shift)
        wE = b * E
        D = np.sum(wE)
        Y = algorithm_4(wE, dim)
        grad = prefactor * (Y / D)
        residual = z + k * (grad ** (k - 1))
        return np.concatenate([residual.real, residual.imag])
    
    if z_init is None:
        x0 = np.zeros(2 * size)
    else:
        x0 = np.concatenate([z_init.real, z_init.imag])
    
    sol = root(G_real, x0, method='hybr', options={'maxfev': 200000, 'xtol': 1e-14})
    z_star = sol.x[:size] + 1j * sol.x[size:]
    
    residual = np.max(np.abs(G_real(sol.x)))
    if verbose:
        print(f"Newton converged: {sol.success}, residual: {residual:.2e}")
    
    # --- Evaluate scaling exponent ---
    V = prefactor * z_star
    X = algorithm_3(V, dim)
    X_shift = np.max(X.real)
    E = np.exp(X - X_shift)
    wE = b * E
    D = np.sum(wE)
    Y = algorithm_4(wE, dim)
    
    F_val = np.log(D) + X_shift
    grad = prefactor * (Y / D)
    C_saddle = F_val + (k - 1) * np.sum(grad ** k)
    
    # Full exponent: BM24 Eq. (A41) prefactors
    C_full = -np.log(2) - r / (2**k) + np.real(C_saddle)
    
    if verbose:
        print(f"F(z*) = {F_val:.8e}")
        print(f"(k-1)·Σ(grad^k) = {(k-1)*np.sum(grad**k):.8e}")
        print(f"C_saddle = {np.real(C_saddle):.8f}  (QAOA improvement over random)")
        print(f"-log(2) = {-np.log(2):.8f}")
        print(f"-r/2^k  = {-r/2**k:.8f}")
        print(f"C_full  = {C_full:.8f}")
        imag = np.imag(C_saddle)
        if abs(imag) > 1e-6:
            print(f"WARNING: Imaginary part of C_saddle = {imag:.2e}")
    
    if return_z:
        return C_full, z_star
    return C_full


def scan_r_homotopy(gammas, betas, k=8, r_target=176.54, n_steps=500, verbose=True):
    """
    Solve the fixed-point equation across a range of r values using homotopy.
    Returns the scaling exponent at r_target.
    """
    r_steps = np.linspace(0.01, r_target, n_steps)
    z = None
    
    for i, r_val in enumerate(r_steps):
        _, z_new = get_scaling_exponent_qaoa_ksat(
            r_val, gammas, betas, k, verbose=False, z_init=z, return_z=True
        )
        z = z_new
    
    C = get_scaling_exponent_qaoa_ksat(
        r_target, gammas, betas, k, verbose=verbose, z_init=z
    )
    return C


# ================================================================
# Main: reproduce BM24 results
# ================================================================

if __name__ == "__main__":
    k = 8
    r = 176.54
    
    print("=" * 70)
    print("BM24 Scaling Exponent for QAOA on Random 8-SAT")
    print("=" * 70)
    
    # --- p=1 ---
    print("\n--- p=1: β=-π/2, γ=0.14 ---")
    C1 = scan_r_homotopy([0.14], [-np.pi/2], k=8, r_target=r, n_steps=500, verbose=True)
    
    # --- p=1, different angles ---
    print("\n--- p=1: β=-π/4, γ=0.10 ---")
    C2 = scan_r_homotopy([0.10], [-np.pi/4], k=8, r_target=r, n_steps=500, verbose=True)
    
    # --- p=2 ---
    print("\n--- p=2: Linear ramp angles ---")
    betas_2 = [-np.pi/2, -np.pi/4]
    gammas_2 = [0.07, 0.14]
    C3 = scan_r_homotopy(gammas_2, betas_2, k=8, r_target=r, n_steps=600, verbose=True)
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Random guessing baseline:  C = {-np.log(2) - r/256:.6f}")
    print(f"p=1 (β=-π/2, γ=0.14):     C = {C1:.6f}")
    print(f"p=1 (β=-π/4, γ=0.10):     C = {C2:.6f}")
    print(f"p=2 (linear ramp):         C = {C3:.6f}")
