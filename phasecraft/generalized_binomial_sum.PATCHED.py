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
except ImportError:
    from exact_ksat import multinomial_sum

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
    """Computes an exact finite-n exact quantity via multinomial sum (exact_ksat.multinomial_sum)"""

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
def generalized_binomial_sum_scaling_exponent_generic(F_dF, q, c, num_iter=100, dz_threshold=1e-2, init_z=None, damping=0.0, debug_logging=False):
    z = np.copy(init_z) if init_z is not None else np.zeros(c.size)
    F, dF = F_dF(z)
    for it in range(num_iter):
        if debug_logging:
            logging.info(f"------------------------- iteration {it} -------------------------")
        prev_z = z
        z = 2 ** q * (-dF) ** (2 ** q - 1)
        z = damping * prev_z + (1 - damping) * z
        F, dF = F_dF(z)
        dz = np.max(np.abs(z - 2 ** q * (-dF) ** (2 ** q - 1)) / np.abs(z))
        if dz < dz_threshold:
            break
        if debug_logging:
            logging.info(f"dz = {dz}")
    return it + 1, \
        z, \
        np.linalg.norm(z - 2 ** q * (-dF) ** (2 ** q - 1)), \
        F - (1 - 2 ** (-q)) * np.sum(z * dF)


def generalized_binomial_sum_scaling_exponent_ksat(q, r, betas, gammas, num_iter=100, dz_threshold=1e-2, init_z=None, damping=0.0, debug_logging=False):
    """general infinite-n"""
    p = len(gammas)
    all_s = np.arange(2 ** (2 * p + 1))
    b = 0.5 * B(betas, all_s)
    # BM24 Eq. (A34): for j < p the per-index factor is (e^{-i gamma_j / 2} - 1);
    # for j > p it is (e^{+i gamma_{2p-j} / 2} - 1).  Earlier versions of this
    # file had the i-signs swapped, which produced a per-n exponent error of
    # the form +(r/2^k) * ... that grew like exp(0.13 n) at the test point
    # (k=2, r=2, beta=gamma=0.4).
    prod_elts = np.concatenate((np.exp(-0.5j * gammas) - 1, [(-1)], np.exp(0.5j * gammas[::-1]) - 1))
    c = r * np.product([prod_elts[j] * ((all_s >> j) & 1) + 1 * ((~all_s >> j) & 1) for j in range(2 * p + 1)], axis=0)
    c_root = (-c) ** (1 / 2 ** q)
    def F_dF(z):
        #s_vector = np.exp(parent_function_alpha_sum_fft(0.5 * c_root * z))
        s_vector = np.exp(parent_function_alpha_sum_sos(0.5 * c_root * z))
        log_arg = np.sum(b * s_vector)
        #return np.log(log_arg), c_root * parent_function_s_sum_fft(0.5 * b * s_vector) / log_arg
        return np.log(log_arg), c_root * parent_function_s_sum_sos(0.5 * b * s_vector) / log_arg
    return generalized_binomial_sum_scaling_exponent_generic(F_dF, q, c, num_iter, dz_threshold, init_z, damping, debug_logging)

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
    # See note above generalized_binomial_sum_scaling_exponent_ksat: i-signs in
    # prod_elts must follow BM24 Eq. (A34) -- negative i for j < p, positive i
    # for j > p.
    prod_elts = np.concatenate((np.exp(-0.5j * gammas) - 1, [(-1)], np.exp(0.5j * gammas[::-1]) - 1))
    c = r * np.product([prod_elts[j] * ((all_s >> j) & 1) + 1 * ((~all_s >> j) & 1) for j in range(2 * p + 1)], axis=0)
    return A, b, c


def bm24_prefactor_exponent_ksat(k, r, gammas):
    """
    Convention-1 prefactor exponent: the n-independent coefficient of n in

        exp(-(r/2^k) n (1 + 4 sum_j sin^2(gamma_j/4)))     [BM24 Eq. (A8)]

    i.e. -(r/2^k) (1 + 4 sum_j sin^2(gamma_j/4)).

    NOTE.  In BM24 there are TWO multinomial-sum representations of the
    expected success probability:

      - Convention 1, Eq. (A8): the index set A runs over subsets J of
        [2p+1] with |J| >= 2, and the explicit prefactor is the one above.
      - Convention 2, Eq. (A41): the index set A runs over ALL subsets of
        [2p+1] (including |J| = 0 and |J| = 1), the prefactor is just
        exp(-(r/2^k) n), and the 4 sum sin^2(gamma_j/4) piece has been
        absorbed into the multinomial sum through the |J| = 1 terms.

    The other helpers in this file --
    `generalized_binomial_sum_random_pow2_sat_data`,
    `generalized_flip_symmetric_multinomial_sum_p1`,
    `generalized_binomial_sum_scaling_exponent_ksat` -- all build the
    multinomial sum data over ALL subsets, i.e. they implement Convention 2.
    The Convention-2 prefactor exponent is therefore -(r/2^k) (independent
    of gamma); see `bm24_prefactor_exponent_ksat_all_subsets` below.

    This function is kept (and unchanged) because callers may use it
    directly with a Convention-1 multinomial sum.  Wrappers in this file
    were previously combining this Convention-1 prefactor with a
    Convention-2 multinomial sum, which double-counted the sin^2(gamma/4)
    piece and produced an O(0.13)-per-n discrepancy at the p=1 test point
    (k=2, r=2, beta=gamma=0.4); see commit message / README.
    """
    gammas = np.array(gammas, dtype=float)
    return -(r / (2 ** k)) * (1.0 + 4.0 * np.sum(np.sin(gammas / 4.0) ** 2))


def bm24_prefactor_exponent_ksat_all_subsets(k, r):
    """
    Convention-2 prefactor exponent (BM24 Eq. (A41)):
        exp(-(r/2^k) n)
    -- i.e. -(r/2^k).  Independent of beta and gamma.  This is the prefactor
    that must be combined with a multinomial sum over ALL subsets of [2p+1].
    """
    return -(r / (2 ** k))


def generalized_binomial_sum_full_exponent_ksat(q, r, betas, gammas, num_iter=100, dz_threshold=1e-2, init_z=None, damping=0.0, debug_logging=False):
    """
    Full expected-success scaling exponent for fixed p:
        phi_full = phi_M + phi_prefactor

    `generalized_binomial_sum_scaling_exponent_ksat` builds its multinomial
    sum over ALL subsets of [2p+1], so phi_M is the Convention-2 saddle
    exponent and the Convention-2 prefactor -(r/2^k) is what must be added,
    NOT the Convention-1 prefactor -(r/2^k) (1 + 4 sum sin^2(gamma_j/4)).
    Earlier versions of this wrapper added the Convention-1 prefactor; the
    resulting double-count produced an O(0.13)-per-n error in phi_full at
    the p=1 test point (k=2, r=2, beta=gamma=0.4).
    """
    iters, z, residual, phi_m = generalized_binomial_sum_scaling_exponent_ksat(
        q=q,
        r=r,
        betas=betas,
        gammas=gammas,
        num_iter=num_iter,
        dz_threshold=dz_threshold,
        init_z=init_z,
        damping=damping,
        debug_logging=debug_logging,
    )
    k = 2 ** q
    phi_pref = bm24_prefactor_exponent_ksat_all_subsets(k=k, r=r)
    phi_full = phi_m + phi_pref
    return iters, z, residual, phi_m, phi_pref, phi_full


def generalized_flip_symmetric_expected_success_p1(q, r, betas, gammas, n):
    """
    Exact finite-n expected-success probability for p = 1, computed via
    BM24 Eq. (A41) (Convention 2):
        E[p_succ] = exp(-(r/2^k) n) * M_n
    where M_n = `generalized_flip_symmetric_multinomial_sum_p1(..., n)`.

    Earlier versions multiplied by exp(n * (-(r/2^k))(1 + 4 sin^2(gamma/4)))
    instead of exp(-(r/2^k) n), which double-counted the
    4 sin^2(gamma/4) contribution that is already inside M_n via the
    |J| = 1 terms.
    """
    if len(betas) != 1 or len(gammas) != 1:
        raise ValueError("generalized_flip_symmetric_expected_success_p1 requires p=1")
    A, b, c = generalized_binomial_sum_random_pow2_sat_data(r, betas, gammas)
    m_n = generalized_flip_symmetric_multinomial_sum_p1(q=q, A=A, b=b, c=c, n=n)
    k = 2 ** q
    phi_pref = bm24_prefactor_exponent_ksat_all_subsets(k=k, r=r)
    return np.exp(n * phi_pref) * m_n
