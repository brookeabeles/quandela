"""Tests for phasecraft/certificate_language.py guardrails and theorem text."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from phasecraft.certificate_language import (
    CERTIFICATE_TYPE_DISCRETE,
    CERTIFICATE_TYPE_TUBE,
    COMPETITOR_DISCLAIMER,
    finalize_payload_certificate,
    generate_theorem_statements,
    guard_interval_theorem_language,
)


def test_level_c_raises_without_tube_certificate():
    payload = {
        "certificate_type": CERTIFICATE_TYPE_DISCRETE,
        "gamma_convention": "code_gamma_negative_seed_chamber",
        "num_points_completed": 10,
        "gamma_mesh": [-0.1, -0.2],
    }
    with pytest.raises(ValueError, match="gamma_interval_tube"):
        generate_theorem_statements(payload, levels=("C"))


def test_level_c_allowed_with_tube_type():
    payload = {
        "certificate_type": CERTIFICATE_TYPE_TUBE,
        "gamma_convention": "code_gamma_negative_seed_chamber",
        "num_points_completed": 10,
        "gamma_mesh": [-0.1, -0.2],
    }
    out = generate_theorem_statements(payload, levels=("C"), allow_level_c=True)
    assert "level_C" in out
    assert "union of certified slabs" in out["level_C"]


def test_guard_interval_language_on_discrete():
    with pytest.raises(ValueError, match="interval"):
        guard_interval_theorem_language(
            "For every gamma in the interval we certify roots.",
            CERTIFICATE_TYPE_DISCRETE,
        )


def test_finalize_lean_includes_disclaimer_only():
    payload = {
        "certificate_type": CERTIFICATE_TYPE_DISCRETE,
        "points": [{"gamma": -0.05, "krawczyk_certified": True}],
        "gamma_mesh": [-0.05],
        "num_points_completed": 1,
    }
    finalize_payload_certificate(payload)
    assert COMPETITOR_DISCLAIMER in payload["proof_note"]
    assert "certificate_summary" not in payload
    assert "theorem_statements" not in payload


def test_finalize_proof_metadata_full():
    payload = {
        "certificate_type": CERTIFICATE_TYPE_DISCRETE,
        "points": [{"gamma": -0.05, "krawczyk_certified": True}],
        "gamma_mesh": [-0.05],
        "num_points_completed": 1,
    }
    finalize_payload_certificate(payload, proof_metadata=True)
    assert "K(X)" in payload["certificate_summary"]["krawczyk_inclusion"]
    assert payload["theorem_statements"]["level_A"]
    assert payload["certificate_summary"]["level_C_allowed"] is False
