#!/usr/bin/env python3
"""
=============================================================================
Scaling Exponent for QAOA on Random k-SAT
Reference: Boulebnane & Montanaro, PRX Quantum 5, 030348 (2024) [BM24]
=============================================================================

WHAT THIS COMPUTES
------------------
The ground-truth scaling exponent C_GT such that:
    E_σ[<Ψ_QAOA|1{H[σ]=0}|Ψ_QAOA>]  ~  exp(C_GT · n)    as n → ∞


THE r-FACTOR CONVENTION
-----------------------
BM24 Eq. 11 defines F with explicit (r/2) and c_α WITHOUT r:

    F(z) = log Σ_s b_s exp( (r/2) Σ_{α,compat} (-c_α)^{1/k} z_α )

where r/2 = r × (1/2), the 1/2 being A_{α,s} = ½·1[compat].

BM24 Proposition 1 claims C_GT = F(z*) + (k-1)·Σ(∂F)^k directly with this F.
However, r/2 ≈ 88 for 8-SAT makes gradients scale as ~88^7 ≈ 4×10¹³,
causing Newton's method to diverge. This is inherent numerical stiffness.

SOLUTION: Apply Proposition 7 to S_q with c^{S_q} = r·ĉ (r inside c).
The prefactor becomes (-r·ĉ)^{1/k} ≈ r^{1/k} ≈ 1.9, which is stable.
This gives (1/n)·log S_q, and we add the Eq. A41 prefactors:

    C_GT = -log(2) - r/2^k + [F_A(z*) + (k-1)·Σ(∂_α F_A)^k]
            ↑         ↑        ↑_________________________________↑
         from 1/2^n  from        = (1/n) log S_q  via Prop. 7
                   e^{-rn/2^k}

Both approaches give identical C_GT. The Eq. 11 convention absorbs the
constants into z*; Convention A keeps them explicit.
"""

import numpy as np
from scipy.optimize import root

# ==========================================================================
# Sum-over-subsets transforms (O(dim · 2^dim) each)
# ==========================================================================

def _zeta(v, dim):
    """Subset-sum (zeta): out[S] = Σ_{T⊆S} v[T]"""
    z = v.copy()
    for i in range(dim):
        zr = z.reshape(1 << (dim-1-i), 2, 1 << i)
        zr[:, 1, :] += zr[:, 0, :]
    return z

def _superset(v, dim):
    """Superset-sum: out[S] = Σ_{T⊇S} v[T]"""
    z = v.copy()
    for i in range(dim):
        zr = z.reshape(1 << (dim-1-i), 2, 1 << i)
        zr[:, 0, :] += zr[:, 1, :]
    return z


def algorithm_3(v, dim):
    """
    BM24 Algorithm 3 (p.030348-27):  X_s = Σ_α A_{α,s} v_α
    
    A_{α,s} = ½·1[∀j,j'∈α: s_j=s_j']. Compatible α satisfy α⊆zeros(s) or α⊆ones(s).
    
    X_s = ( ζ(v)[s] + ζ(v)[s̄] - v[∅] ) / 2
    
    Both terms are the SAME zeta transform evaluated at s and complement(s).
    """
    size = 1 << dim
    rev = np.arange(size) ^ (size - 1)
    zv = _zeta(v, dim)
    return (zv + zv[rev] - v[0]) / 2.0


def algorithm_4(v, dim):
    """
    BM24 Algorithm 4 (p.030348-28):  Y_α = Σ_s A_{α,s} v_s
    
    Compatible s have all bits at α-positions equal (all-0 or all-1).
    Term 1 (all-0): Σ_{s⊆ᾱ} v_s = ζ(v)[ᾱ]     (subset-sum at complement)
    Term 2 (all-1): Σ_{s⊇α} v_s = superset(v)[α]
    
    Y_α = ( ζ(v)[ᾱ] + superset(v)[α] ) / 2,   with fix at α=∅.
    """
    size = 1 << dim
    rev = np.arange(size) ^ (size - 1)
    zv = _zeta(v, dim)
    sv = _superset(v, dim)
    result = (zv[rev] + sv) / 2.0
    result[0] = np.sum(v) / 2.0  # α=∅: both terms give Σv, correct is Σv/2
    return result


# ==========================================================================
# Parameter construction
# ==========================================================================

def compute_b_s(betas, dim, size, s_bits):
    """
    BM24 Eq. 25:  b_s = ½ · Π_{j=0}^{p-1} <s_j|e^{+iβ_jX/2}|s_{j+1}>
                                             · <s_{2p-j-1}|e^{-iβ_jX/2}|s_{2p-j}>
    
    Matrix elements (BM24 Eq. 24):
        <s|e^{+iβX}|s'> = cos β  if s=s',   +i sin β  if s≠s'
        <s|e^{-iβX}|s'> = cos β  if s=s',   -i sin β  if s≠s'
                                              ^^^ MINUS for backward
    """
    p = len(betas)
    b = np.ones(size, dtype=np.complex128) / 2.0
    for j in range(p):
        cb, sb = np.cos(betas[j] / 2), np.sin(betas[j] / 2)
        fd = s_bits[j] ^ s_bits[j + 1]             # forward differs
        b *= (cb ** (1 - fd)) * ((1j * sb) ** fd)   # +i sin for forward
        bd = s_bits[2*p - j - 1] ^ s_bits[2*p - j]  # backward differs
        b *= (cb ** (1 - bd)) * ((-1j * sb) ** bd)   # -i sin for backward
    return b


def compute_c_hat(gammas, dim, size, s_bits):
    """
    BM24 Eq. 16:  ĉ_α = (-1)^{1{p∈α}} · Π_{j∈α,j<p}(e^{-iγ_j/2}-1)
                                          · Π_{j∈α,j>p}(e^{+iγ_{2p-j}/2}-1)
    
    This is c WITHOUT r. The r factor belongs to the S_q parameter mapping
    (BM24 Example 16): c^{S_q}_α = r · ĉ_α.
    """
    p = len(gammas)
    c = ((-1.0) ** s_bits[p]).astype(np.complex128)
    for j in range(dim):
        if j == p:
            continue
        in_alpha = s_bits[j]
        if j < p:
            gf = np.exp(-1j * gammas[j] / 2) - 1.0
        else:
            gf = np.exp(1j * gammas[2*p - j] / 2) - 1.0
        c *= np.where(in_alpha, gf, 1.0)
    return c


# ==========================================================================
# Core solver
# ==========================================================================

def scaling_exponent(r, gammas, betas, k=8, n_homotopy=500, verbose=True):
    """
    Compute C_GT = -log(2) - r/2^k + [F_A(z*) + (k-1)·Σ(∂F_A)^k].
    
    Uses Proposition 7 convention: c^{S_q} = r·ĉ, prefactor = (-r·ĉ)^{1/k}.
    Solves via Newton + r-homotopy for numerical stability.
    
    Parameters
    ----------
    r         : float – clause-to-variable ratio
    gammas    : list  – QAOA γ angles (length p)
    betas     : list  – QAOA β angles (length p)
    k         : int   – SAT arity (power of 2)
    n_homotopy: int   – number of r-homotopy steps
    
    Returns
    -------
    C_GT : float – full scaling exponent
    info : dict  – diagnostic breakdown
    """
    q = int(np.log2(k))
    assert 2**q == k
    p = len(gammas)
    assert len(betas) == p
    dim = 2*p + 1
    size = 1 << dim

    si = np.arange(size)
    s_bits = [(si >> j) & 1 for j in range(dim)]

    b = compute_b_s(betas, dim, size, s_bits)
    c_hat = compute_c_hat(gammas, dim, size, s_bits)

    b_sum = np.sum(b)
    assert abs(b_sum - 1.0) < 1e-8, f"sum(b)={b_sum}, expect 1"

    # --- r-homotopy Newton solve ---
    z = None
    for rv in np.linspace(0.01, r, n_homotopy):
        pf = (-(rv * c_hat)) ** (1.0 / k)

        def G(x, pf=pf):
            zz = x[:size] + 1j * x[size:]
            X = algorithm_3(pf * zz, dim)
            Xm = np.max(X.real)
            wE = b * np.exp(X - Xm)
            D = np.sum(wE)
            g = pf * (algorithm_4(wE, dim) / D)
            res = zz + k * (g ** (k - 1))
            return np.concatenate([res.real, res.imag])

        x0 = np.zeros(2 * size) if z is None else np.concatenate([z.real, z.imag])
        sol = root(G, x0, method='hybr', options={'maxfev': 200000, 'xtol': 1e-14})
        z = sol.x[:size] + 1j * sol.x[size:]

    # --- Evaluate at final z* ---
    pf = (-(r * c_hat)) ** (1.0 / k)
    X = algorithm_3(pf * z, dim)
    Xm = np.max(X.real)
    wE = b * np.exp(X - Xm)
    D = np.sum(wE)
    F_val = np.log(D) + Xm
    grad = pf * (algorithm_4(wE, dim) / D)
    sum_gk = np.sum(grad ** k)

    fp_res = np.max(np.abs(z + k * (grad ** (k - 1))))
    Sq = np.real(F_val) + (k - 1) * np.real(sum_gk)
    C_GT = -np.log(2) - r / 2**k + Sq

    info = dict(p=p, k=k, r=r, neg_log2=-np.log(2), neg_r_2k=-r/2**k,
                F_zstar=np.real(F_val), km1_sum=np.real((k-1)*sum_gk),
                Sq_exponent=Sq, C_GT=C_GT, imag=np.imag(F_val+(k-1)*sum_gk),
                fp_residual=fp_res)

    if verbose:
        print(f"  p={p}  β={[round(x,4) for x in betas]}  γ={[round(x,4) for x in gammas]}")
        print(f"  -log2={-np.log(2):.6f}  -r/2^k={-r/2**k:.6f}  "
              f"F(z*)={info['F_zstar']:.6f}  (k-1)Σg^k={info['km1_sum']:.6f}")
        print(f"  (1/n)logSq={Sq:.8f}   C_GT={C_GT:.8f}   res={fp_res:.1e}  im={info['imag']:.1e}")

    return C_GT, info


# ==========================================================================
# Validation
# ==========================================================================

def _build_A(dim):
    size = 1 << dim
    A = np.zeros((size, size))
    for a in range(size):
        for s in range(size):
            bits = [(s >> j) & 1 for j in range(dim) if (a >> j) & 1]
            if len(bits) == 0 or all(b == bits[0] for b in bits):
                A[a, s] = 0.5
    return A


def validate():
    k = 8; r = 176.54
    C_rand = -np.log(2) - r / 2**k
    all_pass = True

    # V1: Algorithms vs brute-force
    print("V1: Algorithm 3/4 vs brute-force")
    for d in [3, 5, 7]:
        sz = 1 << d; A = _build_A(d)
        rng = np.random.default_rng(42)
        v = rng.standard_normal(sz) + 1j * rng.standard_normal(sz)
        w = rng.standard_normal(sz) + 1j * rng.standard_normal(sz)
        e3 = np.max(np.abs(A.T @ v - algorithm_3(v, d)))
        e4 = np.max(np.abs(A @ w - algorithm_4(w, d)))
        ok = e3 < 1e-10 and e4 < 1e-10
        print(f"  dim={d}: Alg3={e3:.1e} Alg4={e4:.1e} {'PASS' if ok else 'FAIL'}")
        all_pass &= ok

    # V2: b normalization
    print("V2: Σb_s = 1")
    for tp in [1, 2, 3]:
        rng = np.random.default_rng(tp)
        tb = rng.uniform(-np.pi, np.pi, tp)
        dim = 2*tp+1; sz = 1 << dim; si = np.arange(sz)
        sb = [(si >> j) & 1 for j in range(dim)]
        bs = np.sum(compute_b_s(tb, dim, sz, sb))
        ok = abs(bs - 1) < 1e-10
        print(f"  p={tp}: Σb={bs:.12f} {'PASS' if ok else 'FAIL'}")
        all_pass &= ok

    # V3: γ→0 = random guessing
    print("V3: γ→0 limit")
    C0, _ = scaling_exponent(r, [1e-10], [-np.pi/2], k, n_homotopy=200, verbose=False)
    ok = abs(C0 - C_rand) < 1e-4
    print(f"  C(γ≈0)={C0:.8f}  C_rand={C_rand:.8f}  diff={abs(C0-C_rand):.1e} {'PASS' if ok else 'FAIL'}")
    all_pass &= ok

    # V4: Real + converged
    print("V4: Real-valued + converged")
    C4, i4 = scaling_exponent(r, [0.14], [-np.pi/2], k, verbose=False)
    ok_im = abs(i4['imag']) < 1e-6
    ok_fp = i4['fp_residual'] < 1e-8
    print(f"  imag={i4['imag']:.1e} {'PASS' if ok_im else 'FAIL'}  "
          f"res={i4['fp_residual']:.1e} {'PASS' if ok_fp else 'FAIL'}")
    all_pass &= ok_im and ok_fp

    # V5: C_GT < 0
    print("V5: C_GT < 0")
    ok = C4 < 0
    print(f"  C_GT={C4:.6f} {'PASS' if ok else 'FAIL'}")
    all_pass &= ok

    print(f"\n{'ALL VALIDATIONS PASSED' if all_pass else 'SOME VALIDATIONS FAILED'}")
    return all_pass


# ==========================================================================
# Main
# ==========================================================================

if __name__ == "__main__":
    validate()

    k = 8; r = 176.54
    C_rand = -np.log(2) - r / 2**k

    print("\n" + "="*70)
    print(f"RESULTS: Random {k}-SAT, r={r}")
    print("="*70)
    print(f"Random guessing baseline: C = {C_rand:.8f}\n")

    configs = [
        ("p=1  β=-π/2     γ=0.14",   [0.14],       [-np.pi/2]),
        ("p=1  β=-π/2     γ=0.44",   [0.44],       [-np.pi/2]),
        ("p=1  β=-1.30    γ=0.44",   [0.44],       [-1.30]),
        ("p=2  linear ramp",         [0.07, 0.14], [-np.pi/2, -np.pi/4]),
        ("p=2  wider γ",            [0.15, 0.30], [-np.pi/2, -np.pi/4]),
        ("p=3  linear ramp",         [0.05, 0.10, 0.15],
                                     [-np.pi/2, -np.pi/3, -np.pi/6]),
    ]

    print(f"{'Config':<28} {'C_GT':>11} {'vs_rand':>10} {'α=-C/ln2':>9}")
    print("-" * 62)
    for name, gam, bet in configs:
        ns = 400 + len(gam) * 100
        C, info = scaling_exponent(r, gam, bet, k, n_homotopy=ns, verbose=False)
        alpha = -C / np.log(2)
        print(f"{name:<28} {C:>11.6f} {C-C_rand:>+10.6f} {alpha:>9.4f}")
