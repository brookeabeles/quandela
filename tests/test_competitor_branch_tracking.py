"""Tests for branch-resolved competitor tracking."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from phasecraft.competitor_branch_tracking import (
    STATUS_COMPETITOR_CERTIFIED_LOCAL,
    STATUS_COMPETITOR_PL_ACTIVE,
    STATUS_SEED_CERTIFIED_LOCAL,
    STATUS_SEED_DUPLICATE,
    classify_vs_seed,
    diagnose_same_sheet_pairs,
    find_delta_re_crossing_intervals,
    interval_straddles_zero,
    krawczyk_boxes_disjoint,
    merge_same_sheet_branches,
    track_branches_from_nodes,
)


def _synthetic_slices():
    """Two gammas, one competitor sheet moving in w."""
    w_seed = np.array([0.1, 0.0, 0.1, 0.0]) + 0.1j * np.array([1.0, 0.0, 1.0, 0.0])
    w_comp_a = w_seed + 0.01
    w_comp_b = w_seed + 0.02

    def node(w, gamma, is_seed, re, rho=1e-12):
        n = {
            "gamma": gamma,
            "w_real": w.real.tolist(),
            "w_imag": w.imag.tolist(),
            "Phi_eff_real": re,
            "Phi_eff_imag": 0.0,
            "box_radius": rho,
            "delta_lower": 1.0,
            "krawczyk_certified": True,
            "distance_to_seed_w": float(np.linalg.norm(w - w_seed, ord=np.inf)),
            "is_seed_sheet": is_seed,
        }
        n.update(classify_vs_seed(n, w_seed, 1e-12))
        return n

    return [
        {
            "gamma": -0.55,
            "r": 1.0,
            "seed_Phi_real": 0.1,
            "seed_Phi_imag": 0.0,
            "seed_RePhi_interval": [0.1, 0.1],
            "nodes": [
                node(w_seed, -0.55, True, 0.1),
                node(w_comp_a, -0.55, False, 0.05),
            ],
        },
        {
            "gamma": -0.65,
            "r": 1.0,
            "seed_Phi_real": 0.12,
            "seed_Phi_imag": 0.0,
            "seed_RePhi_interval": [0.12, 0.12],
            "nodes": [
                node(w_seed, -0.65, True, 0.12),
                node(w_comp_b, -0.65, False, 0.15),
            ],
        },
    ]


def test_track_branch_and_crossing():
    branches = track_branches_from_nodes(_synthetic_slices(), branch_step_tol=0.5)
    comp = [b for b in branches if not b.is_seed_sheet]
    assert len(comp) >= 1
    assert len(comp[0].points) == 2
    ivs = find_delta_re_crossing_intervals(comp[0])
    assert len(ivs) == 1
    assert ivs[0]["DeltaRe_hi"] * ivs[0]["DeltaRe_lo"] < 0


def test_pl_active_status():
    branches = track_branches_from_nodes(_synthetic_slices(), branch_step_tol=0.5)
    comp_pts = [p for b in branches if not b.is_seed_sheet for p in b.points]
    assert any(p.status == STATUS_COMPETITOR_PL_ACTIVE for p in comp_pts)
    for b in branches:
        if b.is_seed_sheet:
            assert all(p.status == STATUS_SEED_CERTIFIED_LOCAL for p in b.points)


def test_seed_sheet_is_longest_low_distance_branch():
    w_seed = np.array([0.1, 0.0, 0.1, 0.0]) + 0.1j * np.array([1.0, 0.0, 1.0, 0.0])
    slices = []
    for k in range(8):
        g = -0.5 - 0.02 * k
        nodes = [
            {
                "gamma": g,
                "w_real": w_seed.real.tolist(),
                "w_imag": w_seed.imag.tolist(),
                "Phi_eff_real": 0.1,
                "Phi_eff_imag": 0.0,
                "box_radius": 1e-12,
                "delta_lower": 1.0,
                "krawczyk_certified": True,
                "distance_to_seed_w": 1e-8,
                "is_seed_sheet": True,
            }
        ]
        nodes[0].update(classify_vs_seed(nodes[0], w_seed, 1e-12))
        if k == 3:
            dup = {
                "gamma": g,
                "w_real": (w_seed + 5.0).real.tolist(),
                "w_imag": (w_seed + 5.0).imag.tolist(),
                "Phi_eff_real": 2.0,
                "Phi_eff_imag": 0.0,
                "box_radius": 1e-12,
                "delta_lower": 1.0,
                "krawczyk_certified": True,
                "distance_to_seed_w": 0.0,
                "is_seed_sheet": True,
            }
            dup.update(classify_vs_seed(dup, w_seed, 1e-12))
            nodes.append(dup)
        slices.append(
            {
                "gamma": g,
                "r": 1.0,
                "seed_Phi_real": 0.1,
                "seed_Phi_imag": 0.0,
                "seed_RePhi_interval": [0.1, 0.1],
                "nodes": nodes,
            }
        )
    branches = track_branches_from_nodes(slices, branch_step_tol=0.5)
    seed = [b for b in branches if b.is_seed_sheet]
    assert len(seed) == 1
    assert len(seed[0].points) >= 7


def test_seed_duplicate_overlap():
    w_seed = np.zeros(4, dtype=complex)
    node = {
        "w_real": [0.0, 0.0, 0.0, 0.0],
        "w_imag": [0.0, 0.0, 0.0, 0.0],
        "box_radius": 1e-6,
        "is_seed_sheet": False,
    }
    c = classify_vs_seed(node, w_seed, 1e-6)
    assert c["is_seed_duplicate"]
    assert not c["box_disjoint_from_seed"]


def test_boxes_disjoint():
    w1 = np.array([1.0, 0, 0, 0], dtype=complex)
    w2 = np.array([10.0, 0, 0, 0], dtype=complex)
    assert krawczyk_boxes_disjoint(w1, 0.1, w2, 0.1)
    w_near = np.array([1.5, 0, 0, 0], dtype=complex)
    assert not krawczyk_boxes_disjoint(w1, 1.0, w_near, 1.0)


def test_same_sheet_diagnosis_and_merge():
    from phasecraft.competitor_branch_tracking import BranchPoint, CompetitorBranch

    def pt(bid, g, re, w_shift=0.0):
        return BranchPoint(
            branch_id=bid,
            gamma=g,
            w_real=[0.5 + w_shift, 0.1, -0.1, 0.2],
            w_imag=[0.0, 0.0, 0.0, 0.0],
            Phi_eff_real=re,
            Phi_eff_imag=0.0,
            Phi_eff_imag_unwrapped=0.0,
            box_radius=1e-10,
            delta_lower=1.0,
            status=STATUS_COMPETITOR_CERTIFIED_LOCAL,
            distance_to_seed_w=1.0,
            DeltaRe_vs_seed=-re,
            DeltaIm_vs_seed=0.0,
            box_disjoint_from_seed=True,
        )

    a = CompetitorBranch(branch_id=5, points=[pt(5, g, 2.7) for g in [-0.7, -0.65, -0.62]])
    b = CompetitorBranch(branch_id=6, points=[pt(6, g, 2.7001, w_shift=1e-4) for g in [-0.7, -0.65, -0.62]])
    diag = diagnose_same_sheet_pairs([a, b])
    assert diag and diag[0]["likely_same_sheet"]
    merged = merge_same_sheet_branches([a, b], diag)
    assert len([x for x in merged if x.points]) == 1


def test_interval_straddles_zero():
    assert interval_straddles_zero([-1.0, 1.0])
    assert not interval_straddles_zero([0.1, 0.2])
