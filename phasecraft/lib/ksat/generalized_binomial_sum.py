#   Copyright 2023 Phasecraft Ltd.
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License

import logging
import numpy as np

try:
    from .exact_ksat import multinomial_sum
except (ImportError, ModuleNotFoundError):
    try:
        from exact_ksat import multinomial_sum
    except (ImportError, ModuleNotFoundError):
        from phasecraft.lib.ksat.exact_ksat import multinomial_sum

if not hasattr(np, "product"):
    np.product = np.prod


def B(betas, s):
    p = len(betas)
    s = np.array(s)
    return (-1) ** (s & 1 != (s >> p) & 1) * np.product(
        [
            np.cos(betas[j] / 2) ** (((s >> j) & 1 == (s >> (j + 1)) & 1).astype(int) + ((s >> (2 * p - j)) & 1 == (s >> (2 * p - j - 1)) & 1).astype(int)) *
            (1j * np.sin(betas[j] / 2)) ** (((s >> j) & 1 != (s >> (j + 1)) & 1).astype(int) + ((s >> (2 * p - j)) & 1 != (s >> (2 * p - j - 1)) & 1).astype(int))
            for j in range(p)
        ],
        axis=0
    )


def generalized_flip_symmetric_multinomial_sum_p1(q, A, b, c, n):
    "Computes an exact finite-n exact quantity via multinomial sum (exact_ksat.multinomial_sum)"

    if A.shape[1] != 8:
        raise ValueError("invalid shape for A matrix (should have 8 columns)")
    if not np.allclose(A[:, :4], A[:, :-5:-1]):
        raise ValueError("A is not flip-symmetric")
    if not np.allclose(b[:4], b[:-5:-1]):
        raise ValueError("b is not flip-symmetric")
    return multinomial_sum(
        n,
        lambda n012, n12, n02, n01: np.exp(
            n * np.sum(
                c * (A[:, :4] @ np.array([n012, n12, n02, n01]) / n) ** (2 ** q)
            )
        ),
        *(2 * b[:4])
    )

#" generalized..._generic and _ksat solve saddle fixed-pt eqts and return Phi=F-(1-2^-q)*sum(z*dF)"
def generalized_binomial_sum_scaling_exponent_generic(
    F_dF,
    q,
    c,
    num_iter=100,
    dz_threshold=1e-2,
    init_z=None,
    damping=0.0,
    debug_logging=False,
    abs_tol=None,
    rel_tol=None,
):
    z = np.copy(init_z) if init_z is not None else np.zeros(c.size)
    abs_tol = float(dz_threshold if abs_tol is None else abs_tol)
    rel_tol = float(dz_threshold if rel_tol is None else rel_tol)
    converged = False
    F, dF = F_dF(z)
    for it in range(num_iter):
        if debug_logging:
            logging.info(f"------------------------- iteration {it} -------------------------")
        prev_z = z
        z = 2 ** q * (-dF) ** (2 ** q - 1)
        z = damping * prev_z + (1 - damping) * z
        if not np.all(np.isfinite(z)):
            break
        F, dF = F_dF(z)
        if not (np.all(np.isfinite(dF)) and np.isfinite(F)):
            break
        map_z = 2 ** q * (-dF) ** (2 ** q - 1)
        delta = z - map_z
        if not np.all(np.isfinite(delta)):
            break
        abs_residual = float(np.linalg.norm(delta, ord=np.inf))
        z_scale = 1.0 + float(np.linalg.norm(z, ord=np.inf))
        scaled_residual = abs_residual / z_scale
        if abs_residual < abs_tol and scaled_residual < rel_tol:
            converged = True
            break
        if debug_logging:
            logging.info(f"abs_residual={abs_residual} scaled_residual={scaled_residual}")
    map_z = 2 ** q * (-dF) ** (2 ** q - 1)
    residual = np.linalg.norm(z - map_z, ord=np.inf) if np.all(np.isfinite(z)) and np.all(np.isfinite(dF)) else np.inf
    phi = F - (1 - 2 ** (-q)) * np.sum(z * dF) if np.isfinite(F) and np.all(np.isfinite(dF)) and np.all(np.isfinite(z)) else np.nan + 0.0j
    return it + 1, z, float(residual), phi, converged


def generalized_binomial_sum_scaling_exponent_ksat(
    q,
    r,
    betas,
    gammas,
    num_iter=100,
    dz_threshold=1e-2,
    init_z=None,
    damping=0.0,
    debug_logging=False,
    abs_tol=None,
    rel_tol=None,
):
    """general infinite-n"""
    p = len(gammas)
    all_s = np.arange(2 ** (2 * p + 1))
    b = 0.5 * B(betas, all_s)
    # BM24 Eq. (A34): exp(-i gamma_j/2) for j < p, exp(+i gamma_{2p-j}/2) for j > p.
    prod_elts = np.concatenate((np.exp(-0.5j * gammas) - 1, [(-1)], np.exp(0.5j * gammas[::-1]) - 1))
    c = r * np.product([prod_elts[j] * ((all_s >> j) & 1) + 1 * ((~all_s >> j) & 1) for j in range(2 * p + 1)], axis=0)
    c_root = (-c) ** (1 / 2 ** q)
    def F_dF(z):
        #s_vector = np.exp(parent_function_alpha_sum_fft(0.5 * c_root * z))
        s_vector = np.exp(parent_function_alpha_sum_sos(0.5 * c_root * z))
        log_arg = np.sum(b * s_vector)
        #return np.log(log_arg), c_root * parent_function_s_sum_fft(0.5 * b * s_vector) / log_arg
        return np.log(log_arg), c_root * parent_function_s_sum_sos(0.5 * b * s_vector) / log_arg
    return generalized_binomial_sum_scaling_exponent_generic(
        F_dF,
        q,
        c,
        num_iter,
        dz_threshold,
        init_z,
        damping,
        debug_logging,
        abs_tol=abs_tol,
        rel_tol=rel_tol,
    )

# B-note: np.log(log_arg) uses the principal branch of the complex logarith pointwise. As you scan params, log_arg winds around
# the origin and the principle value of the log can jump by 2πi -> then Im(F) and Im(Φ) can jump by 2π even if the underlyign saddle track is actually varying smoothly
# this can look like a stokes event even when its not. 
# Lesson: dont trust raw pointwise Im(Phi). 
# You should use continuously unwrapped phase if you are following the same saddle continuously or if B or gamma cahnges

def parent_function_alpha_sum_sos(z):
    n = int(np.log2(z.size))
    # Sum over subsets of 1 bits in s
    A1 = z.copy()
    for i in range(n):
        for mask in range(1 << n):
            if (mask >> i) & 1:
                A1[mask] += A1[mask ^ (1 << i)]
    # Sum over subsets of 0 bits in s
    A0 = z.copy()[::-1]
    for i in range(n):
        for mask in range(1 << n):
            if (~mask >> i) & 1:
                A0[mask] += A0[mask ^ (1 << i)]
    return A1 + A0 - z[0]


def parent_function_s_sum_sos(z):
    n = int(np.log2(z.size))
    # Sum with bits in alpha set to 0 in s
    A0 = z.copy()[::-1]
    for i in range(n):
        for mask in range(1 << n):
            if (~mask >> i) & 1:
                A0[mask] += A0[mask ^ (1 << i)]
    # Sum with bits in alpha set to 1 in s
    A1 = z.copy()
    for i in range(n):
        for mask in range(1 << n):
            if (~mask >> i) & 1:
                A1[mask] += A1[mask ^ (1 << i)]
    A = A0 + A1
    A[0] -= A0[0]
    return A

        
def generalized_binomial_sum_random_pow2_sat_data(r, betas, gammas):
    p = len(betas)
    all_J = np.arange(2 ** (2 * p + 1))
    all_s = np.arange(2 ** (2 * p + 1))
    A = 0.5 * (((all_J[:, None] & all_s[None, :]) == all_J[:, None]) | ((all_J[:, None] & ~all_s[None, :]) == all_J[:, None]))
    b = 0.5 * B(betas, all_s)
    prod_elts = np.concatenate((np.exp(-0.5j * gammas) - 1, [(-1)], np.exp(0.5j * gammas[::-1]) - 1))
    c = r * np.product([prod_elts[j] * ((all_s >> j) & 1) + 1 * ((~all_s >> j) & 1) for j in range(2 * p + 1)], axis=0)
    return A, b, c
