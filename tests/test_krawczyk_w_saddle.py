"""Regression tests for phasecraft/w_saddle/core.py (parent-action slice)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from phasecraft.w_saddle.core import (
    M_VEC,
    SLICE_BETA,
    SLICE_P,
    Delta,
    Delta_beta_closed_p1,
    Delta_star,
    WSaddleSystem,
    delta_terms_nontrivial_index_set,
    discover_w_roots,
    grad_Delta,
    grad_Delta_star,
    grad_log_Delta,
    hess_Delta,
    hess_Delta_star,
    krawczyk_certify_w_root,
    moment_vec,
)

M_EXPECTED = np.array(
    [0.25 + 0.25j, 0.5 + 0.0j, 0.25 - 0.25j, 0.25 + 0.0j],
    dtype=complex,
)

# Certified reference at r=1, gamma=-0.05, beta=-pi/2 (krawczyk_w_saddle run).
R_REF = 1.0
GAMMA_REF = -0.05
PHI_EFF_REAL_REF = 0.006365254809074513
W_STAR_REF = np.array(
    [
        -0.012420594111515942 - 0.012575416485166227j,
        1.0752527433150951e-21 - 0.0006211111593887804j,
        0.012420594111515942 - 0.012575416485166227j,
        -5.434408136362247e-22 + 0.00031248397687636036j,
    ],
    dtype=complex,
)

BETA_TEST_VALUES = (-np.pi / 2, -1.0, -0.3, 0.0, 0.7)


class TestDeltaEqualsDeltaStar:
    """At beta=-pi/2, general Delta matches closed slice Delta_star."""

    def test_terms_match_at_slice(self):
        terms = delta_terms_nontrivial_index_set(beta=SLICE_BETA, p=SLICE_P)
        for (coef_gen, e_gen), (coef_star, e_star) in zip(
            terms,
            [
                (complex(0.5, 0.0), (1, 1, 1, 1)),
                (complex(0.0, 0.5), (1, 0, 0, 0)),
                (complex(0.5, 0.0), (0, 1, 0, 0)),
                (complex(0.0, -0.5), (0, 0, 1, 0)),
            ],
        ):
            assert e_gen == e_star
            assert abs(coef_gen - coef_star) < 1e-14

    @pytest.mark.parametrize("seed", range(8))
    def test_pointwise_equality(self, seed: int):
        rng = np.random.default_rng(seed)
        w = rng.normal(size=4) + 1j * rng.normal(size=4)
        d_gen = Delta(w, beta=SLICE_BETA, p=SLICE_P)
        d_star = Delta_star(w)
        assert abs(d_gen - d_star) < 1e-14


class TestDeltaBetaClosedP1:
    @pytest.mark.parametrize("beta", BETA_TEST_VALUES)
    def test_delta_zero_is_one(self, beta: float):
        w0 = np.zeros(4, dtype=complex)
        assert abs(Delta(w0, beta=beta) - 1.0) < 1e-12
        assert abs(Delta_beta_closed_p1(w0, beta) - 1.0) < 1e-12

    @pytest.mark.parametrize("beta", BETA_TEST_VALUES)
    def test_matches_closed_formula(self, beta: float):
        rng = np.random.default_rng(int(abs(beta * 100)) % 10000)
        w = 0.05 * (rng.normal(size=4) + 1j * rng.normal(size=4))
        assert abs(Delta(w, beta=beta) - Delta_beta_closed_p1(w, beta)) < 1e-12

    def test_slice_matches_star_derivatives(self):
        rng = np.random.default_rng(3)
        w = rng.normal(size=4) + 1j * rng.normal(size=4)
        assert abs(Delta(w, beta=SLICE_BETA) - Delta_star(w)) < 1e-14
        assert np.max(np.abs(grad_Delta(w, beta=SLICE_BETA) - grad_Delta_star(w))) < 1e-14
        assert np.max(np.abs(hess_Delta(w, beta=SLICE_BETA) - hess_Delta_star(w))) < 1e-14


class TestGradLogDeltaAtOrigin:
    """grad log Delta(0) = i m(beta); at slice, m = M_VEC."""

    def test_m_vector_fixture(self):
        assert np.max(np.abs(M_VEC - M_EXPECTED)) < 1e-14

    def test_moment_vec_at_slice(self):
        assert np.max(np.abs(moment_vec(beta=SLICE_BETA) - M_EXPECTED)) < 1e-14

    def test_grad_log_at_zero(self):
        w0 = np.zeros(4, dtype=complex)
        g0 = grad_log_Delta(w0, beta=SLICE_BETA, p=SLICE_P)
        assert np.max(np.abs(g0 - 1j * M_EXPECTED)) < 1e-14


class TestEffectiveParentModeReference:
    """r=1, gamma=-0.05, beta=-pi/2 reference saddle and Phi_eff."""

    @pytest.fixture
    def system(self) -> WSaddleSystem:
        return WSaddleSystem(r=R_REF, gamma=GAMMA_REF, beta=SLICE_BETA)

    def test_phi_eff_at_reference_w(self, system: WSaddleSystem):
        phi = system.Phi_eff(W_STAR_REF)
        assert abs(phi.real - PHI_EFF_REAL_REF) < 1e-12
        assert abs(phi.imag) < 1e-20

    def test_discovered_w_matches_reference(self, system: WSaddleSystem):
        roots = discover_w_roots(system, num_starts=30, seed=0)
        assert len(roots) >= 1
        w_found = roots[0]
        assert np.linalg.norm(w_found - W_STAR_REF, ord=np.inf) < 1e-9
        assert np.linalg.norm(system.F_complex(w_found), ord=np.inf) < 1e-18
        assert abs(system.Phi_eff(w_found).real - PHI_EFF_REAL_REF) < 1e-12


class TestKrawczykBetaRegression:
    def test_slice_beta_certifies_reference(self):
        sys = WSaddleSystem(r=R_REF, gamma=GAMMA_REF, beta=SLICE_BETA)
        roots = discover_w_roots(sys, num_starts=30, seed=0)
        assert roots
        w = roots[0]
        ok, info = krawczyk_certify_w_root(sys, w, dps=60)
        assert ok
        assert float(info["delta_lower"]) > 0
        assert np.linalg.norm(w - W_STAR_REF, ord=np.inf) < 1e-9
        assert abs(sys.Phi_eff(w).real - PHI_EFF_REAL_REF) < 1e-12

    def test_beta_minus_one_certifies_seed(self):
        sys = WSaddleSystem(r=R_REF, gamma=GAMMA_REF, beta=-1.0)
        roots = discover_w_roots(sys, num_starts=30, seed=0)
        assert roots
        w = roots[0]
        res = float(np.linalg.norm(sys.F_complex(w), ord=np.inf))
        assert res < 1e-8
        ok, info = krawczyk_certify_w_root(sys, w, dps=60)
        assert ok
        assert float(info["delta_lower"]) > 0

    def test_beta_zero_delta_valid(self):
        w0 = np.zeros(4, dtype=complex)
        assert abs(Delta(w0, beta=0.0) - 1.0) < 1e-12
        sys = WSaddleSystem(r=R_REF, gamma=GAMMA_REF, beta=0.0)
        assert np.isfinite(sys.Phi_eff(w0))
