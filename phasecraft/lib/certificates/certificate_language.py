"""
Proof/certificate language, guardrails, and theorem text for w-saddle Krawczyk runs.

See phasecraft.w_saddle.core for the interval operator; this module fixes wording and
what may be claimed from each certificate_type.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional

# ---------------------------------------------------------------------------
# Certificate types
# ---------------------------------------------------------------------------
CERTIFICATE_TYPE_DISCRETE = "discrete_gamma_points"
CERTIFICATE_TYPE_TUBE = "gamma_interval_tube"
CERTIFICATE_TYPE_TRACKED_COMPETITOR = "discrete_gamma_points_tracked_competitor"
CERTIFICATE_TYPE_BRANCH_RESOLVED = "branch_resolved_competitor_sheets"

GAMMA_CONVENTION_CODE_NEGATIVE = "code_gamma_negative_seed_chamber"
GAMMA_CONVENTION_PHYS_NEG = "physical_gamma_positive_with_gamma_code_equals_minus_gamma"
GAMMA_CONVENTION_PHYS_DIRECT = "physical_gamma_positive_direct"

ALLOWED_GAMMA_CONVENTIONS = frozenset(
    {
        GAMMA_CONVENTION_CODE_NEGATIVE,
        GAMMA_CONVENTION_PHYS_NEG,
        GAMMA_CONVENTION_PHYS_DIRECT,
    }
)

COMPETITOR_DISCLAIMER = (
    "These certificates prove local uniqueness only in the explicitly listed "
    "boxes/tubes per sheet. They do not prove global uniqueness, seed dominance, "
    "absence of other certified saddles, thimble/PL dominance, or the QAOA exponent. "
    "A competitor certified outside the seed box does not contradict seed-local "
    "uniqueness."
)

NO_DOMINANCE_NOTE = (
    "Larger Re(Phi) on a competitor sheet is not seed dominance; it is at most a "
    "diagnostic PL-competition candidate unless separately proved."
)

KRAWCZYK_INCLUSION_STATEMENT = (
    "Krawczyk inclusion: K(X) subset int(X), where after realifying w in C^4 to "
    "x in R^8, K(X) = x_0 - Y F(x_0) + (I - Y DF(X))(X - x_0) with F = F_tilde_R, "
    "X = x_0 + [-rho, rho]^8, and Y approximately DF(x_0)^{-1}."
)

NONDEGENERACY_STATEMENT = (
    "If sup_{J in DF(X)} ||I - Y J||_inf < 1 with Y invertible, then every "
    "J in DF(X) is invertible (Neumann series). Hence the root certified inside "
    "X is nondegenerate. The recorded contraction_bound is sqrt(2) times an "
    "upper bound on sup_{J in DF(X)} ||I - Y J||_inf."
)

ACTION_NUMERICAL_NOTE = (
    "Phi_eff is evaluated at the Newton/Krawczyk center (numerical); the certified "
    "root lies in X_gamma. Residual and box_radius are recorded."
)

ACTION_INTERVAL_NOTE = (
    "An interval enclosure of Phi(w_*(gamma)) is obtained by evaluating Phi_eff "
    "over the certified box X_gamma with the log(Delta_star) branch fixed by "
    "certified ratio-lifts ell_{j+1} = ell_j + Log_principal(Delta(X_{j+1})/Delta(X_j)) "
    "along the mesh, with each ratio step certified to avoid the negative real axis."
)

INTERVAL_LANGUAGE_FORBIDDEN = (
    "for every gamma in the interval",
    "for all gamma in the interval",
    "on the union of certified slabs",
    "true interval theorem",
    "theorem level c:",
    "level_c theorem",
)


def infer_gamma_convention_from_mesh(gamma_mesh: Iterable[float]) -> str:
    """Infer convention from stored mesh values (continuation default)."""
    gammas = [float(g) for g in gamma_mesh]
    if not gammas:
        return GAMMA_CONVENTION_CODE_NEGATIVE
    if all(g < 0 for g in gammas):
        return GAMMA_CONVENTION_CODE_NEGATIVE
    if all(g > 0 for g in gammas):
        return GAMMA_CONVENTION_PHYS_DIRECT
    return GAMMA_CONVENTION_CODE_NEGATIVE


def gamma_mesh_supports_positive_theorem(gamma_convention: str) -> bool:
    return gamma_convention in (
        GAMMA_CONVENTION_PHYS_DIRECT,
        GAMMA_CONVENTION_PHYS_NEG,
    )


def guard_interval_theorem_language(text: str, certificate_type: str) -> None:
    """Raise if text claims an interval-in-gamma theorem without tube certificates."""
    if certificate_type == CERTIFICATE_TYPE_TUBE:
        return
    lower = text.lower()
    for phrase in INTERVAL_LANGUAGE_FORBIDDEN:
        if phrase in lower:
            raise ValueError(
                f"Certificate type {certificate_type!r} does not support interval "
                f"theorem language (matched {phrase!r}). Use "
                f"{CERTIFICATE_TYPE_TUBE!r} or restrict to Level A/B pointwise claims."
            )


def generate_theorem_level_a_seed(certificate_type: str, n_points: int) -> str:
    return (
        f"Level A_seed (pointwise): On the seed continuation mesh (|Gamma|={n_points}), "
        "each gamma_i has a Krawczyk-certified unique nondegenerate seed-sheet root "
        f"in X_i^{{seed}} only ({certificate_type}). "
        f"{NO_DOMINANCE_NOTE}"
    )


def generate_theorem_level_a_competitor(n_competitor_sheets: int) -> str:
    return (
        f"Level A_competitor (pointwise, per sheet): The sweep certifies "
        f"{n_competitor_sheets} locally unique competitor-sheet roots in their "
        "own boxes; coexistence with the seed sheet is allowed. "
        f"{NO_DOMINANCE_NOTE}"
    )


def generate_theorem_level_b(certificate_type: str) -> str:
    return (
        "Level B (LOC mesh extension, seed sheet only): With the analytic LOC "
        "certificate near gamma=0, pointwise Krawczyk boxes extend the BM24 seed "
        f"sheet on the sampled mesh beyond the Banach radius ({certificate_type}). "
        "This does not certify that no other sheets exist at those gamma_i."
    )


def generate_theorem_level_crossing() -> str:
    return (
        "Level crossing (branch-tracked): On a gamma interval where branch-tracked "
        "DeltaRe = Re(Phi_seed) - Re(Phi_comp) changes sign, refined midpoint "
        "certificates bracket a real-action crossing (DeltaRe interval contains 0). "
        "Im behavior is recorded separately; no PL/Stokes jump is claimed."
    )


def generate_theorem_level_c() -> str:
    return (
        "Level C (interval tube): On the union of certified slabs "
        "[gamma_i, gamma_{i+1}], interval Krawczyk certifies a unique "
        "nondegenerate root in the continuation tube."
    )


def generate_theorem_statements(
    payload: dict[str, Any],
    *,
    levels: Iterable[str] = ("A", "B"),
    allow_level_c: bool = False,
) -> dict[str, Any]:
    """Build Level A/B/C theorem strings with guardrails."""
    cert_type = str(payload.get("certificate_type", CERTIFICATE_TYPE_DISCRETE))
    gamma_conv = str(
        payload.get("gamma_convention", GAMMA_CONVENTION_CODE_NEGATIVE)
    )
    n_pts = int(payload.get("num_points_completed", len(payload.get("points", []))))

    out: dict[str, Any] = {
        "gamma_convention": gamma_conv,
        "certificate_type": cert_type,
        "levels_requested": list(levels),
    }

    n_comp = int(payload.get("num_branches", 0)) - int(payload.get("num_seed_sheets", 1))
    if n_comp < 0:
        n_comp = int(payload.get("num_competitor_sheets", 0))

    if "A" in levels or "a" in levels:
        stmt_a = generate_theorem_level_a_seed(cert_type, n_pts)
        if not gamma_mesh_supports_positive_theorem(gamma_conv):
            stmt_a += (
                " Mesh uses code_gamma_negative_seed_chamber; map signs explicitly "
                "for physical gamma."
            )
        guard_interval_theorem_language(stmt_a, cert_type)
        out["level_A_seed"] = stmt_a
        out["level_A"] = stmt_a  # legacy alias

    if "A_comp" in levels or "a_comp" in levels:
        out["level_A_competitor"] = generate_theorem_level_a_competitor(n_comp)

    if "B" in levels or "b" in levels:
        stmt_b = generate_theorem_level_b(cert_type)
        guard_interval_theorem_language(stmt_b, cert_type)
        out["level_B"] = stmt_b

    if "C" in levels or "c" in levels:
        if not allow_level_c and cert_type != CERTIFICATE_TYPE_TUBE:
            raise ValueError(
                f"Level C requires certificate_type={CERTIFICATE_TYPE_TUBE!r}; "
                f"got {cert_type!r}. Interval tubes have not been certified."
            )
        stmt_c = generate_theorem_level_c()
        out["level_C"] = stmt_c

    if "crossing" in levels or "X" in levels:
        if payload.get("crossing_certificates"):
            out["level_crossing"] = generate_theorem_level_crossing()
        else:
            out["level_crossing"] = None

    return out


def build_proof_note(certificate_type: str, *, action_enclosure: bool) -> str:
    parts = [
        KRAWCZYK_INCLUSION_STATEMENT,
        NONDEGENERACY_STATEMENT,
        COMPETITOR_DISCLAIMER,
    ]
    if certificate_type == CERTIFICATE_TYPE_DISCRETE:
        parts.insert(
            2,
            "Each mesh point has a local Krawczyk box; this is NOT an "
            "interval-in-gamma certificate unless certificate_type is "
            f"{CERTIFICATE_TYPE_TUBE}.",
        )
    elif certificate_type == CERTIFICATE_TYPE_TUBE:
        parts.insert(
            2,
            "Interval-in-gamma tube certification: overlapping gamma slabs required.",
        )
    if action_enclosure:
        parts.append(ACTION_INTERVAL_NOTE)
    else:
        parts.append(ACTION_NUMERICAL_NOTE)
    return " ".join(parts)


def _count_certified_mesh_points(payload: dict[str, Any]) -> int:
    if payload.get("points"):
        return sum(1 for p in payload["points"] if p.get("krawczyk_certified"))
    n = 0
    for br in payload.get("tracked_competitor_branches") or []:
        for pt in br.get("points") or []:
            comp = pt.get("competitor") or {}
            if comp.get("krawczyk_certified"):
                n += 1
    return n


def _count_action_interval_points(payload: dict[str, Any]) -> int:
    if payload.get("points"):
        return sum(1 for p in payload["points"] if p.get("action_interval_real"))
    n = 0
    for br in payload.get("tracked_competitor_branches") or []:
        for pt in br.get("points") or []:
            if (pt.get("competitor") or {}).get("action_interval_real"):
                n += 1
    return n


def build_certificate_summary(payload: dict[str, Any]) -> dict[str, Any]:
    """Aggregate summary block stored on JSON outputs."""
    cert_type = str(payload.get("certificate_type", CERTIFICATE_TYPE_DISCRETE))
    gamma_conv = str(
        payload.get("gamma_convention", GAMMA_CONVENTION_CODE_NEGATIVE)
    )
    mesh = _gamma_mesh_from_payload(payload)
    n_cert = _count_certified_mesh_points(payload)
    n_action_iv = _count_action_interval_points(payload)

    theorems = generate_theorem_statements(payload, levels=("A", "B"))
    level_c_available = cert_type == CERTIFICATE_TYPE_TUBE

    return {
        "gamma_convention": gamma_conv,
        "certificate_type": cert_type,
        "gamma_mesh_min": float(min(mesh)) if mesh else None,
        "gamma_mesh_max": float(max(mesh)) if mesh else None,
        "num_mesh_points": len(mesh),
        "num_krawczyk_certified_points": n_cert,
        "num_action_interval_enclosures": n_action_iv,
        "krawczyk_inclusion": KRAWCZYK_INCLUSION_STATEMENT,
        "nondegeneracy_implication": NONDEGENERACY_STATEMENT,
        "competitor_disclaimer": COMPETITOR_DISCLAIMER,
        "action_note": ACTION_INTERVAL_NOTE if n_action_iv else ACTION_NUMERICAL_NOTE,
        "theorem_level_A": theorems.get("level_A"),
        "theorem_level_B": theorems.get("level_B"),
        "theorem_level_C": generate_theorem_level_c() if level_c_available else None,
        "level_C_allowed": level_c_available,
        "claims_interval_in_gamma": cert_type == CERTIFICATE_TYPE_TUBE,
    }


def _gamma_mesh_from_payload(payload: dict[str, Any]) -> list[float]:
    if payload.get("gamma_mesh"):
        return [float(g) for g in payload["gamma_mesh"]]
    if payload.get("points"):
        return [float(p["gamma"]) for p in payload["points"]]
    gammas: list[float] = []
    for br in payload.get("tracked_competitor_branches") or []:
        for pt in br.get("points") or []:
            if "gamma" in pt:
                gammas.append(float(pt["gamma"]))
    return gammas


def lean_proof_note(certificate_type: str) -> str:
    """Minimal JSON note without long Krawczyk/theorem boilerplate."""
    return (
        f"certificate_type={certificate_type}; pointwise Krawczyk mesh only "
        "(not interval-in-gamma unless gamma_interval_tube). "
        f"{COMPETITOR_DISCLAIMER}"
    )


def finalize_payload_certificate(
    payload: dict[str, Any],
    *,
    gamma_convention: Optional[str] = None,
    enclose_action: bool = False,
    proof_metadata: bool = False,
    theorem_levels: Iterable[str] = ("A", "B"),
    allow_level_c: bool = False,
) -> dict[str, Any]:
    """Attach gamma_convention; optional full proof/theorem blocks (--proof-metadata)."""
    mesh = _gamma_mesh_from_payload(payload)
    if mesh and "gamma_mesh" not in payload:
        payload["gamma_mesh"] = mesh
    conv = gamma_convention or infer_gamma_convention_from_mesh(mesh)
    if conv not in ALLOWED_GAMMA_CONVENTIONS:
        raise ValueError(f"Unknown gamma_convention: {conv!r}")

    payload["gamma_convention"] = conv
    cert_type = str(payload.get("certificate_type", CERTIFICATE_TYPE_DISCRETE))

    if proof_metadata:
        payload["proof_note"] = build_proof_note(
            cert_type, action_enclosure=enclose_action
        )
        payload["certificate_summary"] = build_certificate_summary(payload)
        theorems = generate_theorem_statements(
            payload,
            levels=theorem_levels,
            allow_level_c=allow_level_c or cert_type == CERTIFICATE_TYPE_TUBE,
        )
        payload["theorem_statements"] = theorems
        for key in ("level_A", "level_B", "level_C"):
            if theorems.get(key):
                guard_interval_theorem_language(theorems[key], cert_type)
    else:
        payload["proof_note"] = lean_proof_note(cert_type)
    return payload


def format_certificate_summary_for_print(payload: dict[str, Any]) -> str:
    """Human-readable summary; safe for discrete_gamma_points only."""
    summary = payload.get("certificate_summary") or build_certificate_summary(payload)
    lines = [
        "=== Certificate summary ===",
        f"gamma_convention: {summary['gamma_convention']}",
        f"certificate_type: {summary['certificate_type']}",
        f"claims_interval_in_gamma: {summary['claims_interval_in_gamma']}",
        "",
        summary["krawczyk_inclusion"],
        "",
        summary["nondegeneracy_implication"],
        "",
        f"Action: {summary['action_note']}",
        "",
        summary["competitor_disclaimer"],
        "",
    ]
    if summary.get("theorem_level_A"):
        lines.append(summary["theorem_level_A"])
    if summary.get("theorem_level_B"):
        lines.append("")
        lines.append(summary["theorem_level_B"])
    if summary.get("theorem_level_C"):
        lines.append("")
        lines.append(summary["theorem_level_C"])
    elif not summary.get("level_C_allowed"):
        lines.append("")
        lines.append(
            "Interval tube certification (gamma_interval_tube) has not been performed."
        )
    return "\n".join(lines)
