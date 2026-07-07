"""Anti-Stokes finite-size narrative figure.

This figure is deliberately conservative: it compares exact finite-n p=1
rates with certified local saddle data and diagnostic candidate combinations.
It does not claim that the original contour decomposition is known.
"""

from __future__ import annotations

import json
import math
import os
import shutil
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-phasecraft")

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RESULTS = REPO / "results" / "bm24_saddle_audit_p1"
OUT = RESULTS / "PAPER-RESULTS"
SUB_OUT = OUT / "two-saddle-sum"
JSON_DIR = OUT / "json"

BETA = 0.5434
GAMMA_CROSS = -1.83
GAMMA_AS = -1.830842334140663
PHI_PREF = -0.689609375
GAMMA_PAIR_TOL = 0.011

BLUE = "#2b6cb0"
ORANGE = "#dd6b20"
GREEN = "#2f855a"
PURPLE = "#6b46c1"
RED = "#c53030"
BLACK = "#1a202c"
GRAY = "#718096"
LIGHT_GREEN = "#c6f6d5"


def load_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def interp(x: float, xs: Iterable[float], ys: Iterable[float]) -> float:
    xs_arr = np.asarray(list(xs), dtype=float)
    ys_arr = np.asarray(list(ys), dtype=float)
    order = np.argsort(xs_arr)
    return float(np.interp(x, xs_arr[order], ys_arr[order]))


def load_two_saddle_row(gamma: float) -> dict:
    data = load_json(JSON_DIR / "two_saddle_crossing.json")
    rows = [
        row
        for row in data["results"]
        if row.get("competitor_found") and abs(float(row["gamma"]) - gamma) < 0.005
    ]
    if not rows:
        raise KeyError(f"No crossing row found near gamma={gamma}")
    return min(rows, key=lambda row: abs(float(row["gamma"]) - gamma))


def load_seed_series() -> dict[str, np.ndarray]:
    data = load_json(RESULTS / "run_seed_branch_g-2pi" / "seed_branch_continuation.json")
    rows = [
        row
        for row in data["continuation"]
        if row.get("certified") and not row.get("failed")
    ]
    return {
        "gamma": np.array([float(row["gamma"]) for row in rows]),
        "re": np.array([float(row["re_phi_m"]) for row in rows]),
        "im": np.array([float(row["im_phi_m"]) for row in rows]),
        "full": np.array([float(row["full_conv2_exponent"]) for row in rows]),
    }


def branch_series(branch_id: int) -> dict[str, np.ndarray]:
    resolved = load_json(
        RESULTS
        / "run_seed_branch_g-2pi"
        / "archive"
        / "competitor_dominance"
        / "z_robust_full"
        / "resolved_robust_z.json"
    )
    branch = next(b for b in resolved["branches"] if int(b["branch_id"]) == branch_id)
    rows = [
        row
        for row in branch["points"]
        if row.get("krawczyk_certified", True) and row.get("box_disjoint_from_seed", True)
    ]
    return {
        "gamma": np.array([float(row["gamma"]) for row in rows]),
        "re": np.array([float(row["Phi_eff_real"]) for row in rows]),
        "im": np.array([float(row["Phi_eff_imag"]) for row in rows]),
        "full": np.array([PHI_PREF + float(row["Phi_eff_real"]) for row in rows]),
    }


def merged_43_46_series() -> dict[str, np.ndarray]:
    b43 = branch_series(43)
    b46 = branch_series(46)
    t43 = {
        round(float(g), 6): {"re": float(r), "im": float(im)}
        for g, r, im in zip(b43["gamma"], b43["re"], b43["im"])
    }
    t46 = {
        round(float(g), 6): {"re": float(r), "im": float(im)}
        for g, r, im in zip(b46["gamma"], b46["re"], b46["im"])
    }
    gammas: list[float] = []
    merged_re: list[float] = []
    for g in sorted(set(t43) | set(t46)):
        vals: list[float] = []
        for table in (t43, t46):
            if g in table:
                vals.append(table[g]["re"])
                continue
            close = [(abs(g - g2), row["re"]) for g2, row in table.items() if abs(g - g2) <= GAMMA_PAIR_TOL]
            if close:
                vals.append(min(close, key=lambda item: item[0])[1])
        if vals:
            gammas.append(g)
            merged_re.append(float(np.mean(vals)))
    seed = load_seed_series()
    return {
        "gamma": np.array(gammas),
        "re": np.array(merged_re),
        "delta_re": np.array(merged_re) - np.array([interp(g, seed["gamma"], seed["re"]) for g in gammas]),
    }


def logsumexp_rate(ns: np.ndarray, exponents: list[float], multiplicities: list[float] | None = None) -> np.ndarray:
    weights = np.ones(len(exponents), dtype=float) if multiplicities is None else np.asarray(multiplicities, dtype=float)
    exps = np.asarray(exponents, dtype=float)
    out = []
    for n in ns:
        scaled = n * exps
        m = float(np.max(scaled))
        out.append((m + math.log(float(np.sum(weights * np.exp(scaled - m))))) / n)
    return np.asarray(out, dtype=float)


def signed_equal_prefactor_rate(ns: np.ndarray, exponents: list[float], coeffs: tuple[int, ...]) -> np.ndarray:
    exps = np.asarray(exponents, dtype=float)
    co = np.asarray(coeffs, dtype=float)
    out = []
    for n in ns:
        scaled = n * exps
        m = float(np.max(scaled))
        signed = float(np.sum(co * np.exp(scaled - m)))
        if abs(signed) < 1e-300:
            out.append(float("nan"))
        else:
            out.append((m + math.log(abs(signed))) / n)
    return np.asarray(out, dtype=float)


def best_fixed_signed_candidate(
    ns: np.ndarray,
    exact: np.ndarray,
    exponents: list[float],
    allowed: list[tuple[int, ...]],
) -> tuple[tuple[int, ...], np.ndarray]:
    best_coeffs: tuple[int, ...] | None = None
    best_curve: np.ndarray | None = None
    best_rmse = float("inf")
    for coeffs in allowed:
        curve = signed_equal_prefactor_rate(ns, exponents, coeffs)
        if not np.all(np.isfinite(curve)):
            continue
        score = rmse(curve, exact)
        if score < best_rmse:
            best_rmse = score
            best_coeffs = coeffs
            best_curve = curve
    if best_coeffs is None or best_curve is None:
        raise RuntimeError("No finite signed candidate curve was available")
    return best_coeffs, best_curve


def coeff_label(coeffs: tuple[int, ...]) -> str:
    return "(" + ",".join("+" if c > 0 else "-" if c < 0 else "0" for c in coeffs) + ")"


def exact_series(row: dict) -> tuple[np.ndarray, np.ndarray]:
    ns = np.array([int(k) for k in row["lam_exact"]], dtype=float)
    vals = np.array([float(row["lam_exact"][str(int(n))]) for n in ns], dtype=float)
    order = np.argsort(ns)
    return ns[order], vals[order]


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def style_panel(ax: plt.Axes) -> None:
    ax.grid(True, alpha=0.24)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def plot_anti_stokes_narrative() -> Path:
    seed = load_seed_series()
    b43 = branch_series(43)
    b46 = branch_series(46)
    merged = merged_43_46_series()
    row = load_two_saddle_row(GAMMA_CROSS)

    ns, exact = exact_series(row)
    seed_full = float(row["full_seed"])
    b43_full = interp(GAMMA_CROSS, b43["gamma"], b43["full"])
    b46_full = interp(GAMMA_CROSS, b46["gamma"], b46["full"])
    best_comp_full = b43_full if abs(b43_full - exact[-1]) <= abs(b46_full - exact[-1]) else b46_full

    seed_curve = np.full_like(ns, seed_full)
    best_comp_curve = np.full_like(ns, best_comp_full)
    seed_43_diag = logsumexp_rate(ns, [seed_full, b43_full])
    coeff_4346, seed_43_46_diag = best_fixed_signed_candidate(
        ns,
        exact,
        [seed_full, b43_full, b46_full],
        [
            (s, c43, c46)
            for s in (-1, 1)
            for c43 in (-1, 1)
            for c46 in (-1, 1)
        ],
    )

    fig = plt.figure(figsize=(15.2, 6.9))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.0, 1.0], wspace=0.28)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])

    # (a) Real actions near the anti-Stokes crossing.
    g_lo, g_hi = -1.93, -1.74
    for series, color, label, ls, lw in [
        (seed, BLUE, "seed", "-", 2.4),
        (b43, ORANGE, "branch 43", "--", 2.0),
        (b46, PURPLE, "branch 46", ":", 2.4),
    ]:
        mask = (series["gamma"] >= g_lo) & (series["gamma"] <= g_hi)
        order = np.argsort(series["gamma"][mask])
        ax_a.plot(series["gamma"][mask][order], series["re"][mask][order], color=color, ls=ls, lw=lw, label=label)
    ax_a.axvline(GAMMA_AS, color=RED, lw=1.8, ls="-.", label=rf"$\gamma_{{AS}}={GAMMA_AS:.6f}$")
    y_star = 0.5 * (
        interp(GAMMA_AS, seed["gamma"], seed["re"])
        + interp(GAMMA_AS, merged["gamma"], merged["re"])
    )
    ax_a.plot(GAMMA_AS, y_star, marker="*", ms=15, color="gold", mec=BLACK, mew=0.7, zorder=5)
    ax_a.set_title("(a) Equal-real-action crossing", fontsize=11, fontweight="bold")
    ax_a.set_xlabel(r"$\gamma$")
    ax_a.set_ylabel(r"$\mathrm{Re}\,\Phi_j(\gamma)$")
    ax_a.legend(fontsize=8, loc="best")
    style_panel(ax_a)

    # (b) Exact finite-size exponent and candidate diagnostics at gamma ~= gamma_AS.
    ax_b.plot(ns, exact, "o", color=BLACK, ms=5, mfc="white", mew=1.2, label="exact finite-n p=1")
    ax_b.plot(ns, seed_curve, "--", color=BLUE, lw=1.8, label="seed-only local saddle")
    ax_b.plot(ns, best_comp_curve, "--", color=ORANGE, lw=1.8, label="best individual competitor")
    ax_b.plot(ns, seed_43_diag, "-", color=GREEN, lw=2.6, label="phase-aligned seed+43 diagnostic")
    ax_b.plot(
        ns,
        seed_43_46_diag,
        "-",
        color=PURPLE,
        lw=2.2,
        alpha=0.88,
        label=f"signed seed+43+46 candidate nu={coeff_label(coeff_4346)}",
    )
    ax_b.fill_between(ns, seed_curve, seed_43_diag, color=LIGHT_GREEN, alpha=0.28, zorder=0)
    ax_b.set_title(rf"(b) Finite-$n$ rates near $\gamma_{{AS}}$", fontsize=11, fontweight="bold")
    ax_b.set_xlabel(r"$n$")
    ax_b.set_ylabel(r"$\log|P_n|/n$")
    ax_b.legend(fontsize=7.5, loc="lower right")
    style_panel(ax_b)
    ax_b.text(
        0.03,
        0.04,
        rf"$\gamma={GAMMA_CROSS}$ is within $8.4\times10^{{-4}}$ of $\gamma_{{AS}}$.",
        transform=ax_b.transAxes,
        fontsize=8,
        color=GRAY,
    )

    # (c) Absolute error versus n.
    err_seed = np.abs(seed_curve - exact)
    err_comp = np.abs(best_comp_curve - exact)
    err_seed43 = np.abs(seed_43_diag - exact)
    err_seed4346 = np.abs(seed_43_46_diag - exact)
    ax_c.plot(ns, err_seed, "o--", color=BLUE, lw=1.8, ms=4, label=f"seed-only (RMSE {rmse(seed_curve, exact):.4f})")
    ax_c.plot(ns, err_comp, "s--", color=ORANGE, lw=1.8, ms=4, label=f"best competitor (RMSE {rmse(best_comp_curve, exact):.4f})")
    ax_c.plot(ns, err_seed43, "D-", color=GREEN, lw=2.4, ms=4, label=f"seed+43 diagnostic (RMSE {rmse(seed_43_diag, exact):.4f})")
    ax_c.plot(
        ns,
        err_seed4346,
        "^-",
        color=PURPLE,
        lw=2.0,
        ms=4,
        label=f"seed+43+46 nu={coeff_label(coeff_4346)} (RMSE {rmse(seed_43_46_diag, exact):.4f})",
    )
    ax_c.set_title("(c) Absolute approximation error", fontsize=11, fontweight="bold")
    ax_c.set_xlabel(r"$n$")
    ax_c.set_ylabel(r"$|$approximation $-$ exact$|$")
    ax_c.set_ylim(bottom=0)
    ax_c.legend(fontsize=7.2, loc="upper right")
    style_panel(ax_c)

    fig.suptitle(
        "Finite-size evidence for competing saddle contributions near the anti-Stokes crossing",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    footer = (
        "Krawczyk certification proves local existence, uniqueness in the certification box, and nonsingularity; "
        "it does not prove contour contribution. Contour coefficients/intersection numbers remain unknown. "
        "Log-sum-exp/sign curves are equal-prefactor diagnostic models with fixed candidate coefficients. "
        "All comparisons use exact finite-n p=1 values."
    )
    fig.text(0.01, -0.02, footer, ha="left", va="top", fontsize=8.2, color=GRAY, wrap=True)

    OUT.mkdir(parents=True, exist_ok=True)
    SUB_OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / "stokes_crossing_single_vs_two_narrative_v3.png"
    fig.savefig(out, dpi=190, bbox_inches="tight")
    plt.close(fig)

    sub_out = SUB_OUT / out.name
    shutil.copy2(out, sub_out)

    caption = (
        "Finite-size evidence for competing saddle contributions near the anti-Stokes crossing.\n\n"
        "If the relevant contour decomposes into the certified seed and competitor thimbles with nonzero "
        "coefficients, standard saddle-point asymptotics imply that the subdominant contribution is "
        "exponentially suppressed away from the equal-real-action crossing, while both contributions occur "
        "at the same exponential order at the crossing. The unknown intersection numbers prevent this "
        "conditional expansion from being promoted to a proof of the contour decomposition.\n\n"
        "Away from the anti-Stokes crossing, the exact finite-size exponent is accurately described by the "
        "continued seed saddle. Near the crossing, neither bare saddle action individually explains the exact "
        "rate. Candidate combinations of the certified saddle contributions provide a closer finite-size "
        "description. This supports a multi-saddle interpretation of the discrepancy, but does not determine "
        "the contributing thimbles or their intersection numbers.\n\n"
        "Krawczyk certification proves local existence, uniqueness within the certification box, and "
        "nonsingularity; it does not prove that the saddle contributes to the original integration contour. "
        "The contour coefficients remain unknown. The phase-aligned log-sum-exp and signed finite-set "
        "curves are diagnostic equal-prefactor models with fixed candidate coefficients, not fitted "
        "independently at each n. All comparisons use exact finite-n p=1 values.\n"
    )
    for caption_path in [
        OUT / "stokes_crossing_single_vs_two_narrative_v3_caption.txt",
        SUB_OUT / "stokes_crossing_single_vs_two_narrative_v3_caption.txt",
    ]:
        caption_path.write_text(caption)
    return sub_out


def main() -> None:
    path = plot_anti_stokes_narrative()
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
