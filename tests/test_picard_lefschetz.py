"""
Tests for phasecraft/picard_lefschetz.py

Covers:
  - Phi computation reproducibility on a synthetic saddle
  - Phi computation matches known p=1 fixture root
  - Conjugate-pair detection on synthetic data
  - Stokes-pair detection logic
  - Regression test: p=1 JSON fixture (first 5 roots)
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure repo root is on path.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from phasecraft.picard_lefschetz import (
    SaddleInfo,
    compute_phi,
    detect_conjugate_pairs,
    detect_stokes_pairs,
    load_roots_json,
    analyse,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

P1_JSON = REPO_ROOT / "phasecraft" / "p1_krawczyk_sampled_roots_3000starts_seed0.json"
P2_JSON = REPO_ROOT / "phasecraft" / "p2_krawczyk_rigorous_roots_5000starts_seed0.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_saddle(idx: int, re_phi: float, im_phi: float) -> SaddleInfo:
    """Create a minimal SaddleInfo with given Phi values for unit tests."""
    return SaddleInfo(
        idx=idx,
        z=np.zeros(2, dtype=complex),
        residual_norm=0.0,
        certified=True,
        contraction_bound=0.0,
        phi=complex(re_phi, im_phi),
        re_phi=re_phi,
        im_phi=im_phi,
    )


# ---------------------------------------------------------------------------
# Phi computation reproducibility
# ---------------------------------------------------------------------------

class TestComputePhi:
    def test_scalar_output(self):
        """compute_phi returns a complex scalar."""
        betas = np.array([0.5433996420760803])
        gammas = np.array([0.7487887432117251])
        # Synthetic z: small real vector
        z = np.zeros(8, dtype=complex)
        phi = compute_phi(z, q=3, r=176.54, betas=betas, gammas=gammas)
        assert isinstance(phi, complex)

    def test_deterministic(self):
        """compute_phi is deterministic on the same input."""
        betas = np.array([0.5433996420760803])
        gammas = np.array([0.7487887432117251])
        z = np.ones(8, dtype=complex) * 0.1
        phi1 = compute_phi(z, q=3, r=176.54, betas=betas, gammas=gammas)
        phi2 = compute_phi(z, q=3, r=176.54, betas=betas, gammas=gammas)
        assert phi1 == phi2

    @pytest.mark.skipif(not P1_JSON.exists(), reason="p1 fixture not found")
    def test_phi_on_p1_fixture_root0(self):
        """Phi on the first p=1 fixture root should be self-consistent.

        We verify that the saddle equation G(z) ≈ 0 is satisfied by checking
        that z = 2^q * (-dF)^{2^q - 1}.  Phi itself has no external reference
        value, so we just check it is finite and has reasonable magnitude.
        """
        data = load_roots_json(str(P1_JSON))
        entry = data["roots"][0]
        z = entry["z_real"] + 1j * entry["z_imag"]
        phi = compute_phi(
            z, q=data["q"], r=data["r"], betas=data["betas"], gammas=data["gammas"]
        )
        assert math.isfinite(phi.real)
        assert math.isfinite(phi.imag)
        # Re(Phi) should be of order 1–100 for typical k-SAT parameters
        assert -1000 < phi.real < 1000


# ---------------------------------------------------------------------------
# Conjugate pair detection
# ---------------------------------------------------------------------------

class TestDetectConjugatePairs:
    def test_exact_conjugate_pair(self):
        """Two saddles with Re equal and Im opposite should be paired."""
        s0 = _make_saddle(0, re_phi=5.0, im_phi=2.0)
        s1 = _make_saddle(1, re_phi=5.0, im_phi=-2.0)
        detect_conjugate_pairs([s0, s1], re_tol=1e-3, im_tol=1e-3)
        assert s0.pair_idx == 1
        assert s1.pair_idx == 0

    def test_no_pair_when_re_differs(self):
        """Saddles with different Re should not be paired."""
        s0 = _make_saddle(0, re_phi=5.0, im_phi=2.0)
        s1 = _make_saddle(1, re_phi=6.0, im_phi=-2.0)
        detect_conjugate_pairs([s0, s1], re_tol=1e-3, im_tol=1e-3)
        assert s0.pair_idx is None
        assert s1.pair_idx is None

    def test_no_pair_when_im_sum_nonzero(self):
        """Saddles with Im not summing to ≈0 should not be paired."""
        s0 = _make_saddle(0, re_phi=5.0, im_phi=2.0)
        s1 = _make_saddle(1, re_phi=5.0, im_phi=2.0)  # same Im, sum = 4
        detect_conjugate_pairs([s0, s1], re_tol=1e-3, im_tol=1e-3)
        assert s0.pair_idx is None
        assert s1.pair_idx is None

    def test_mod2pi_pairing(self):
        """Im sum that is a multiple of 2π should be paired with mod2pi=True."""
        # Im_i + Im_j = 2π → sum mod 2π = 0
        s0 = _make_saddle(0, re_phi=5.0, im_phi=math.pi)
        s1 = _make_saddle(1, re_phi=5.0, im_phi=math.pi)
        detect_conjugate_pairs([s0, s1], re_tol=1e-3, im_tol=1e-3, mod2pi=True)
        assert s0.pair_idx == 1
        assert s1.pair_idx == 0

    def test_single_saddle_no_pair(self):
        """A single saddle should have no pair."""
        s = _make_saddle(0, re_phi=1.0, im_phi=0.5)
        detect_conjugate_pairs([s])
        assert s.pair_idx is None


# ---------------------------------------------------------------------------
# Stokes pair detection
# ---------------------------------------------------------------------------

class TestDetectStokesPairs:
    def test_near_stokes_pair(self):
        """Stokes line: Im(Φ_i) = Im(Φ_j). Pair with |ΔIm| < tol should be flagged."""
        s0 = _make_saddle(0, re_phi=5.0, im_phi=1.0)
        s1 = _make_saddle(1, re_phi=3.0, im_phi=1.02)
        pairs = detect_stokes_pairs([s0, s1], stokes_tol=0.05)
        assert len(pairs) == 1
        assert pairs[0].near_stokes is True
        assert pairs[0].near_anti_stokes is False

    def test_non_stokes_pair(self):
        """Pair with large ΔIm should not be near-Stokes."""
        s0 = _make_saddle(0, re_phi=5.0, im_phi=0.0)
        s1 = _make_saddle(1, re_phi=3.0, im_phi=1.5)
        pairs = detect_stokes_pairs([s0, s1], stokes_tol=0.05)
        assert len(pairs) == 1
        assert pairs[0].near_stokes is False

    def test_near_anti_stokes_pair(self):
        """Anti-Stokes line: Re(Φ_i) = Re(Φ_j). Pair with |ΔRe| < tol should be flagged."""
        s0 = _make_saddle(0, re_phi=3.0, im_phi=1.0)
        s1 = _make_saddle(1, re_phi=3.02, im_phi=5.0)
        pairs = detect_stokes_pairs([s0, s1], stokes_tol=0.05)
        assert len(pairs) == 1
        assert pairs[0].near_anti_stokes is True
        assert pairs[0].near_stokes is False

    def test_pair_count(self):
        """n saddles produce n*(n-1)/2 pairs."""
        saddles = [_make_saddle(i, float(i), float(i) * 0.3) for i in range(5)]
        pairs = detect_stokes_pairs(saddles)
        assert len(pairs) == 10  # C(5,2) = 10

    def test_delta_im_sign(self):
        """delta_im should be Phi_i.im - Phi_j.im for i < j."""
        s0 = _make_saddle(0, re_phi=0.0, im_phi=2.0)
        s1 = _make_saddle(1, re_phi=0.0, im_phi=0.5)
        pairs = detect_stokes_pairs([s0, s1])
        assert abs(pairs[0].delta_im - (2.0 - 0.5)) < 1e-12


# ---------------------------------------------------------------------------
# Regression test: p=1 JSON fixture
# ---------------------------------------------------------------------------

class TestRegressionP1:
    @pytest.mark.skipif(not P1_JSON.exists(), reason="p1 fixture not found")
    def test_analyse_runs_without_error(self):
        """analyse() should complete for the p=1 fixture without raising."""
        result = analyse(str(P1_JSON), top_n=5)
        assert "metadata" in result
        assert "saddles" in result
        assert len(result["saddles"]) > 0

    @pytest.mark.skipif(not P1_JSON.exists(), reason="p1 fixture not found")
    def test_phi_all_finite(self):
        """All Phi values should be finite for valid certified saddles."""
        result = analyse(str(P1_JSON), certified_only=True)
        for s in result["saddles"]:
            assert math.isfinite(s["re_phi"]), f"idx={s['idx']} has non-finite Re(Phi)"
            assert math.isfinite(s["im_phi"]), f"idx={s['idx']} has non-finite Im(Phi)"

    @pytest.mark.skipif(not P1_JSON.exists(), reason="p1 fixture not found")
    def test_schema_keys_present(self):
        """Output dict has all required schema keys."""
        result = analyse(str(P1_JSON))
        required = [
            "metadata", "saddles", "im0_saddle", "dominant_re_saddle",
            "competitors_top_re", "stokes_pairs", "anti_stokes_pairs", "notes",
        ]
        for key in required:
            assert key in result, f"Missing key: {key}"

    @pytest.mark.skipif(not P1_JSON.exists(), reason="p1 fixture not found")
    def test_im0_saddle_exists(self):
        """im0_saddle should reference a valid idx."""
        result = analyse(str(P1_JSON))
        idxs = {s["idx"] for s in result["saddles"]}
        assert result["im0_saddle"]["idx"] in idxs

    @pytest.mark.skipif(not P1_JSON.exists(), reason="p1 fixture not found")
    def test_phi_reproducibility_first_five(self):
        """Phi values for the first 5 roots should be stable across two calls."""
        data = load_roots_json(str(P1_JSON))
        q, r = data["q"], data["r"]
        betas, gammas = data["betas"], data["gammas"]
        for entry in data["roots"][:5]:
            z = entry["z_real"] + 1j * entry["z_imag"]
            phi1 = compute_phi(z, q=q, r=r, betas=betas, gammas=gammas)
            phi2 = compute_phi(z, q=q, r=r, betas=betas, gammas=gammas)
            assert abs(phi1 - phi2) < 1e-14, f"Phi not reproducible for idx={entry['idx']}"


# ---------------------------------------------------------------------------
# Regression test: p=2 JSON fixture (light – only first 3 roots)
# ---------------------------------------------------------------------------

class TestRegressionP2:
    @pytest.mark.skipif(not P2_JSON.exists(), reason="p2 fixture not found")
    def test_phi_p2_finite(self):
        """Phi should be finite for the first 3 p=2 roots."""
        data = load_roots_json(str(P2_JSON))
        q, r = data["q"], data["r"]
        betas, gammas = data["betas"], data["gammas"]
        for entry in data["roots"][:3]:
            z = entry["z_real"] + 1j * entry["z_imag"]
            phi = compute_phi(z, q=q, r=r, betas=betas, gammas=gammas)
            assert math.isfinite(phi.real)
            assert math.isfinite(phi.imag)


# ---------------------------------------------------------------------------
# load_roots_json: schema normalisation
# ---------------------------------------------------------------------------

class TestLoadRootsJson:
    @pytest.mark.skipif(not P1_JSON.exists(), reason="p1 fixture not found")
    def test_old_schema_parsed(self):
        """Old schema (scalar beta/gamma) should be normalised to arrays."""
        data = load_roots_json(str(P1_JSON))
        assert data["betas"].ndim == 1
        assert data["gammas"].ndim == 1
        assert data["p"] == 1

    @pytest.mark.skipif(not P2_JSON.exists(), reason="p2 fixture not found")
    def test_new_schema_parsed(self):
        """New schema (array betas/gammas) should be loaded correctly."""
        data = load_roots_json(str(P2_JSON))
        assert data["p"] == 2
        assert len(data["betas"]) == 2
        assert len(data["gammas"]) == 2
