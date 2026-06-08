"""
Picard-Lefschetz (PL) analysis for Phasecraft k-SAT QAOA saddle points.

This module loads certified saddle points (from krawczyk_p1_roots.py output),
computes the action Phi at each saddle, detects conjugate pairs, identifies
near-Stokes conditions, and produces a heuristic contribution ranking.

RIGOUR CAVEAT
-------------
* Phi computation: exact (same formulas as generalized_binomial_sum.PATCHED.py).
* Conjugate-pair detection: exact up to numerical tolerance.
* Stokes-condition detection: exact for the pair-wise Im(Phi) difference;
  the branch offset check is heuristic (mod 2*pi).
* Contribution classification: HEURISTIC.  Full rigorous Picard-Lefschetz
  intersection-number computation is NOT implemented.  Labels should be
  treated as supporting evidence only.

Branch choices
--------------
(-c)^{1/2^q} is computed via numpy's default branch of the complex power,
which places the branch cut on the negative real axis.  Results depend on
this choice; different branches give different z_s values but the same Phi
(because Phi is a function of the original contour integral, not of z).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Path bootstrap so this file can be run as __main__ from any directory.
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from phasecraft.lib.saddles.krawczyk_p1_roots import (
    B,
    SaddleSystem,
    parent_function_alpha_sum_sos,
    parent_function_s_sum_sos,
)


# ---------------------------------------------------------------------------
# Phi computation
# ---------------------------------------------------------------------------

def compute_phi(
    z: np.ndarray,
    q: int,
    r: float,
    betas: np.ndarray,
    gammas: np.ndarray,
) -> complex:
    """Compute the action Phi at a saddle point z.

    Uses the identical formulas as
    ``generalized_binomial_sum_scaling_exponent_ksat`` (PATCHED) so there is no
    possibility of formula drift.

    Parameters
    ----------
    z:
        Complex saddle point vector (length 2^{2p+1}).
    q:
        Integer; k = 2^q.
    r:
        Clause-to-variable ratio.
    betas, gammas:
        QAOA angle arrays of length p.

    Returns
    -------
    Phi as a complex number.
        Phi = F - (1 - 2^{-q}) * sum_s z_s * dF_s
    """
    sys = SaddleSystem.build(q=q, r=r, betas=np.array(betas, dtype=float), gammas=np.array(gammas, dtype=float))
    s_vec = np.exp(parent_function_alpha_sum_sos(0.5 * sys.c_root * z))
    log_arg = np.sum(sys.b * s_vec)
    F = np.log(log_arg)
    dF = sys.c_root * parent_function_s_sum_sos(0.5 * sys.b * s_vec) / log_arg
    phi = F - (1.0 - 2.0 ** (-q)) * np.sum(z * dF)
    return complex(phi)


# ---------------------------------------------------------------------------
# JSON loading
# ---------------------------------------------------------------------------

def load_roots_json(path: str) -> dict:
    """Load a krawczyk roots JSON and normalise to a canonical schema.

    Handles both the old schema (scalar ``beta``/``gamma``, no ``p`` field)
    and the new schema (arrays ``betas``/``gammas``, explicit ``p``).

    Returns a dict with keys:
        q, k, r, p, betas (ndarray), gammas (ndarray),
        roots (list of dicts with z_real, z_imag, idx, residual_norm,
               krawczyk_certified, krawczyk_contraction_bound),
        raw (original parsed JSON)
    """
    with open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)

    q = int(raw["q"])
    k = int(raw.get("k", 2 ** q))
    r = float(raw["r"])

    # Angle arrays – handle old scalar and new list schemas.
    if "betas" in raw:
        betas = np.array(raw["betas"], dtype=float)
        gammas = np.array(raw["gammas"], dtype=float)
        p = int(raw.get("p", len(betas)))
    elif "beta" in raw:
        betas = np.array([raw["beta"]], dtype=float)
        gammas = np.array([raw["gamma"]], dtype=float)
        p = 1
    else:
        raise ValueError("JSON has neither 'betas' nor 'beta' field")

    roots = []
    for entry in raw.get("roots", []):
        roots.append(
            {
                "idx": int(entry["idx"]),
                "residual_norm": float(entry["residual_norm"]),
                "krawczyk_certified": bool(entry.get("krawczyk_certified", False)),
                "krawczyk_contraction_bound": float(
                    entry.get("krawczyk_contraction_bound", math.inf)
                ),
                "z_real": np.array(entry["z_real"], dtype=float),
                "z_imag": np.array(entry["z_imag"], dtype=float),
            }
        )

    return {
        "q": q,
        "k": k,
        "r": r,
        "p": p,
        "betas": betas,
        "gammas": gammas,
        "roots": roots,
        "raw": raw,
    }


# ---------------------------------------------------------------------------
# Saddle data container
# ---------------------------------------------------------------------------

@dataclass
class SaddleInfo:
    idx: int
    z: np.ndarray          # complex saddle point vector
    residual_norm: float
    certified: bool
    contraction_bound: float
    phi: complex            # action value
    re_phi: float
    im_phi: float
    pair_idx: Optional[int] = None   # index of conjugate partner (if found)
    labels: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "idx": self.idx,
            "re_phi": self.re_phi,
            "im_phi": self.im_phi,
            "residual_norm": self.residual_norm,
            "certified": self.certified,
            "contraction_bound": self.contraction_bound,
            "pair_idx": self.pair_idx,
            "labels": self.labels,
            "z_real": self.z.real.tolist(),
            "z_imag": self.z.imag.tolist(),
        }


# ---------------------------------------------------------------------------
# Conjugate pair detection
# ---------------------------------------------------------------------------

def detect_conjugate_pairs(
    saddles: list[SaddleInfo],
    re_tol: float = 1e-4,
    im_tol: float = 1e-4,
    mod2pi: bool = True,
) -> None:
    """Label conjugate saddle pairs in-place.

    Two saddles i, j are considered a conjugate pair when:
        |Re(Phi_i) - Re(Phi_j)| < re_tol
    and
        |Im(Phi_i) + Im(Phi_j)| < im_tol  [mod 2*pi if mod2pi is True]

    Only the closest partner (by combined metric) is recorded.
    Self-pairing is excluded (i != j).
    """
    n = len(saddles)
    re_arr = np.array([s.re_phi for s in saddles])
    im_arr = np.array([s.im_phi for s in saddles])

    for i in range(n):
        best_j = None
        best_dist = math.inf
        for j in range(n):
            if i == j:
                continue
            delta_re = abs(re_arr[i] - re_arr[j])
            if delta_re >= re_tol:
                continue
            im_sum = im_arr[i] + im_arr[j]
            if mod2pi:
                im_sum = im_sum % (2 * math.pi)
                # Map to [-pi, pi]
                if im_sum > math.pi:
                    im_sum -= 2 * math.pi
            if abs(im_sum) >= im_tol:
                continue
            dist = delta_re + abs(im_sum)
            if dist < best_dist:
                best_dist = dist
                best_j = j
        saddles[i].pair_idx = best_j


# ---------------------------------------------------------------------------
# Stokes-condition diagnostics
# ---------------------------------------------------------------------------

@dataclass
class StokesPair:
    i: int
    j: int
    delta_re: float
    delta_im: float
    near_stokes: bool       # |ΔIm| < tol  (Stokes line: Im(Φ_i) = Im(Φ_j))
    near_anti_stokes: bool  # |ΔRe| < tol  (anti-Stokes line: Re(Φ_i) = Re(Φ_j))

    def to_dict(self) -> dict:
        return asdict(self)


def detect_stokes_pairs(
    saddles: list[SaddleInfo],
    stokes_tol: float = 0.1,
) -> list[StokesPair]:
    """Return all ordered pairs (i, j) with i < j and their Stokes diagnostics.

    Stokes line     : Im(Φ_i) = Im(Φ_j)  →  near_stokes      when |ΔIm| < stokes_tol
    Anti-Stokes line: Re(Φ_i) = Re(Φ_j)  →  near_anti_stokes when |ΔRe| < stokes_tol
    """
    pairs: list[StokesPair] = []
    n = len(saddles)
    for i in range(n):
        for j in range(i + 1, n):
            d_re = saddles[i].re_phi - saddles[j].re_phi
            d_im = saddles[i].im_phi - saddles[j].im_phi
            near_stokes = abs(d_im) < stokes_tol
            near_anti_stokes = abs(d_re) < stokes_tol
            pairs.append(
                StokesPair(
                    i=i, j=j,
                    delta_re=d_re, delta_im=d_im,
                    near_stokes=near_stokes,
                    near_anti_stokes=near_anti_stokes,
                )
            )
    return pairs


# ---------------------------------------------------------------------------
# Heuristic PL contribution classifier
# ---------------------------------------------------------------------------

_CONTRIBUTION_LABELS = {
    "im0": "Im≈0 saddle (likely contributes on original contour)",
    "im_small": "Small |Im(Phi)| (<π/4): likely contributes",
    "conjugate_im0": "Conjugate partner of an Im≈0 saddle",
    "dominant_re": "Globally dominant Re(Phi)",
    "near_stokes": "Near a Stokes line [Im(Φ_i)=Im(Φ_j)] – intersection number may jump",
    "near_anti_stokes": "Near an anti-Stokes line [Re(Φ_i)=Re(Φ_j)] – saddles exchange dominance",
    "large_im": "Large |Im(Phi)| (>π): oscillatory suppression, likely zero intersection",
    "uncertain": "Uncertain (heuristic classifier)",
}

HEURISTIC_CAVEAT = (
    "HEURISTIC: contribution labels are NOT rigorous PL intersection numbers. "
    "Stokes line: Im(Φ_i)=Im(Φ_j); anti-Stokes line: Re(Φ_i)=Re(Φ_j). "
    "Labels use Im(Phi) proximity to 0, conjugate pairing, and Stokes/anti-Stokes "
    "proximity as proxies. Full rigorous thimble tracing is not implemented."
)


def classify_contributions(
    saddles: list[SaddleInfo],
    stokes_pairs: list[StokesPair],
    im0_tol: float = 0.05,
) -> None:
    """Assign heuristic contribution labels to each saddle in-place."""
    # Find im≈0 saddles
    im0_set: set[int] = set()
    for s in saddles:
        im_mod = s.im_phi % (2 * math.pi)
        if im_mod > math.pi:
            im_mod -= 2 * math.pi
        if abs(im_mod) < im0_tol:
            s.labels.append("im0")
            im0_set.add(s.idx)

    # Mark conjugate partners of im≈0 saddles
    idx_to_saddle = {s.idx: s for s in saddles}
    for s in saddles:
        if s.pair_idx is not None and s.pair_idx in im0_set:
            if "im0" not in s.labels:
                s.labels.append("conjugate_im0")

    # Dominant Re saddle
    max_re = max(s.re_phi for s in saddles)
    for s in saddles:
        if abs(s.re_phi - max_re) < 1e-8:
            s.labels.append("dominant_re")

    # Near-Stokes / anti-Stokes
    stokes_idxs: set[int] = set()
    anti_stokes_idxs: set[int] = set()
    for sp in stokes_pairs:
        if sp.near_stokes:
            stokes_idxs.add(sp.i)
            stokes_idxs.add(sp.j)
        if sp.near_anti_stokes:
            anti_stokes_idxs.add(sp.i)
            anti_stokes_idxs.add(sp.j)
    for i, s in enumerate(saddles):
        if i in stokes_idxs:
            s.labels.append("near_stokes")
        if i in anti_stokes_idxs:
            s.labels.append("near_anti_stokes")

    # Remaining size-based labels
    for s in saddles:
        if s.labels:
            continue
        im_mod = abs(s.im_phi % (2 * math.pi))
        if im_mod > math.pi:
            im_mod = 2 * math.pi - im_mod
        if im_mod < math.pi / 4:
            s.labels.append("im_small")
        elif im_mod > math.pi:
            s.labels.append("large_im")
        else:
            s.labels.append("uncertain")


# ---------------------------------------------------------------------------
# Ranking report
# ---------------------------------------------------------------------------

def build_ranking_report(
    saddles: list[SaddleInfo],
    stokes_pairs: list[StokesPair],
    top_n: int = 10,
) -> dict:
    """Build a structured ranking report dict."""
    # Im≈0 saddle: smallest |Im(Phi) mod 2π|
    def im_mod_dist(s: SaddleInfo) -> float:
        v = s.im_phi % (2 * math.pi)
        if v > math.pi:
            v -= 2 * math.pi
        return abs(v)

    sorted_by_im = sorted(saddles, key=im_mod_dist)
    im0_saddle = sorted_by_im[0]

    sorted_by_re = sorted(saddles, key=lambda s: -s.re_phi)
    dominant_re_saddle = sorted_by_re[0]

    competitors = []
    for s in sorted_by_re[:top_n]:
        competitors.append(
            {
                "idx": s.idx,
                "re_phi": s.re_phi,
                "im_phi": s.im_phi,
                "delta_re_vs_im0": s.re_phi - im0_saddle.re_phi,
                "pair_idx": s.pair_idx,
                "labels": s.labels,
                "certified": s.certified,
            }
        )

    return {
        "im0_saddle": {"idx": im0_saddle.idx, "re_phi": im0_saddle.re_phi, "im_phi": im0_saddle.im_phi},
        "dominant_re_saddle": {
            "idx": dominant_re_saddle.idx,
            "re_phi": dominant_re_saddle.re_phi,
            "im_phi": dominant_re_saddle.im_phi,
        },
        "competitors_top_re": competitors,
    }


# ---------------------------------------------------------------------------
# Main analysis pipeline
# ---------------------------------------------------------------------------

def analyse(
    roots_json: str,
    im0_tol: float = 0.05,
    pair_re_tol: float = 1e-4,
    pair_im_tol: float = 1e-4,
    stokes_tol: float = 0.1,
    top_n: int = 10,
    certified_only: bool = False,
) -> dict:
    """Run the full PL analysis pipeline and return the result dict.

    Parameters
    ----------
    roots_json:
        Path to a krawczyk roots JSON file.
    im0_tol:
        Tolerance for classifying Im(Phi) ≈ 0 (mod 2π).
    pair_re_tol, pair_im_tol:
        Tolerances for conjugate-pair detection.
    stokes_tol:
        Tolerance for near-Stokes classification (|ΔIm mod 2π| < stokes_tol).
    top_n:
        Number of top-Re saddles to include in the ranking report.
    certified_only:
        If True, skip uncertified saddles.

    Returns
    -------
    Result dict suitable for JSON serialisation.
    """
    data = load_roots_json(roots_json)
    q, r, p = data["q"], data["r"], data["p"]
    betas, gammas = data["betas"], data["gammas"]

    saddles: list[SaddleInfo] = []
    for entry in data["roots"]:
        if certified_only and not entry["krawczyk_certified"]:
            continue
        z = entry["z_real"] + 1j * entry["z_imag"]
        phi = compute_phi(z, q=q, r=r, betas=betas, gammas=gammas)
        saddles.append(
            SaddleInfo(
                idx=entry["idx"],
                z=z,
                residual_norm=entry["residual_norm"],
                certified=entry["krawczyk_certified"],
                contraction_bound=entry["krawczyk_contraction_bound"],
                phi=phi,
                re_phi=phi.real,
                im_phi=phi.imag,
            )
        )

    detect_conjugate_pairs(
        saddles, re_tol=pair_re_tol, im_tol=pair_im_tol
    )
    stokes_pairs = detect_stokes_pairs(saddles, stokes_tol=stokes_tol)
    classify_contributions(saddles, stokes_pairs, im0_tol=im0_tol)
    ranking = build_ranking_report(saddles, stokes_pairs, top_n=top_n)

    result = {
        "metadata": {
            "q": q,
            "k": data["k"],
            "r": r,
            "p": p,
            "betas": betas.tolist(),
            "gammas": gammas.tolist(),
            "input_file": str(roots_json),
            "tolerances": {
                "im0_tol": im0_tol,
                "pair_re_tol": pair_re_tol,
                "pair_im_tol": pair_im_tol,
                "stokes_tol": stokes_tol,
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_saddles": len(saddles),
            "certified_saddles": sum(1 for s in saddles if s.certified),
        },
        "saddles": [s.to_dict() for s in saddles],
        "im0_saddle": ranking["im0_saddle"],
        "dominant_re_saddle": ranking["dominant_re_saddle"],
        "competitors_top_re": ranking["competitors_top_re"],
        "stokes_pairs": [
            sp.to_dict() for sp in stokes_pairs if sp.near_stokes
        ],
        "anti_stokes_pairs": [
            sp.to_dict() for sp in stokes_pairs if sp.near_anti_stokes
        ],
        "notes": [
            HEURISTIC_CAVEAT,
            "Branch cut for (-c)^{1/2^q} placed on negative real axis (numpy default).",
            "Im(Phi) values are not reduced mod 2π in the saddle list; raw values are reported.",
            "Stokes pairs listed are only those flagged near_stokes=True.",
        ],
    }
    return result


# ---------------------------------------------------------------------------
# Human-readable summary
# ---------------------------------------------------------------------------

def print_summary(result: dict) -> None:
    meta = result["metadata"]
    print(f"\n{'='*60}")
    print(f"Picard-Lefschetz Analysis")
    print(f"  Input : {meta['input_file']}")
    print(f"  q={meta['q']}, k={meta['k']}, r={meta['r']}, p={meta['p']}")
    print(f"  Saddles analysed : {meta['total_saddles']}  (certified: {meta['certified_saddles']})")
    print(f"{'='*60}")

    im0 = result["im0_saddle"]
    dom = result["dominant_re_saddle"]
    print(f"\nIm≈0 saddle     : idx={im0['idx']:4d}  Re(Φ)={im0['re_phi']:+.6f}  Im(Φ)={im0['im_phi']:+.6f}")
    print(f"Dominant Re(Φ)  : idx={dom['idx']:4d}  Re(Φ)={dom['re_phi']:+.6f}  Im(Φ)={dom['im_phi']:+.6f}")

    print(f"\nTop-{len(result['competitors_top_re'])} competitors by Re(Φ):")
    print(f"  {'idx':>4}  {'Re(Φ)':>12}  {'Im(Φ)':>12}  {'ΔRe vs Im≈0':>14}  {'pair':>5}  labels")
    for c in result["competitors_top_re"]:
        pair_str = str(c["pair_idx"]) if c["pair_idx"] is not None else "—"
        cert_str = "✓" if c["certified"] else "✗"
        print(
            f"  {c['idx']:>4}  {c['re_phi']:>12.6f}  {c['im_phi']:>12.6f}  "
            f"{c['delta_re_vs_im0']:>+14.6f}  {pair_str:>5}  "
            f"[{cert_str}] {', '.join(c['labels'])}"
        )

    near_stokes = result["stokes_pairs"]
    near_anti = result["anti_stokes_pairs"]
    print(f"\nNear-Stokes pairs [ΔIm≈0]: {len(near_stokes)}")
    for sp in near_stokes[:10]:
        print(f"  ({sp['i']},{sp['j']})  ΔRe={sp['delta_re']:+.4f}  ΔIm={sp['delta_im']:+.6f}")
    print(f"\nNear-anti-Stokes pairs [ΔRe≈0]: {len(near_anti)}")
    for sp in near_anti[:10]:
        print(f"  ({sp['i']},{sp['j']})  ΔRe={sp['delta_re']:+.6f}  ΔIm={sp['delta_im']:+.4f}")

    print(f"\nNOTE: {result['notes'][0]}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Picard-Lefschetz analysis for Phasecraft saddle points"
    )
    parser.add_argument("--roots-json", required=True, help="Path to krawczyk roots JSON")
    parser.add_argument("--out", default="", help="Output JSON path (optional)")
    parser.add_argument("--im0-tol", type=float, default=0.05)
    parser.add_argument("--pair-re-tol", type=float, default=1e-4)
    parser.add_argument("--pair-im-tol", type=float, default=1e-4)
    parser.add_argument("--stokes-tol", type=float, default=0.1)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument(
        "--certified-only",
        action="store_true",
        help="Only analyse Krawczyk-certified saddles",
    )
    args = parser.parse_args()

    result = analyse(
        roots_json=args.roots_json,
        im0_tol=args.im0_tol,
        pair_re_tol=args.pair_re_tol,
        pair_im_tol=args.pair_im_tol,
        stokes_tol=args.stokes_tol,
        top_n=args.top_n,
        certified_only=args.certified_only,
    )

    print_summary(result)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2)
        print(f"Results written to {args.out}")


if __name__ == "__main__":
    main()
