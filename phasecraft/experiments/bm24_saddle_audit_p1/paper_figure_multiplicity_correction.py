"""Finite-size multiplicity correction at the anti-Stokes crossing."""

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
BETA_FULL = 0.5433996420760803
GAMMA_AS = -1.830842334140663
GAMMA_CONTROL = -1.0
PHI_PREF = -0.689609375

BLUE = "#2b6cb0"
ORANGE = "#dd6b20"
GREEN = "#2f855a"
PURPLE = "#6b46c1"
RED = "#c53030"
BLACK = "#1a202c"
GRAY = "#718096"


def load_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def interp(x: float, xs: Iterable[float], ys: Iterable[float]) -> float:
    xs_arr = np.asarray(list(xs), dtype=float)
    ys_arr = np.asarray(list(ys), dtype=float)
    order = np.argsort(xs_arr)
    return float(np.interp(x, xs_arr[order], ys_arr[order]))


def load_seed_series() -> dict[str, np.ndarray]:
    data = load_json(RESULTS / "run_seed_branch_g-2pi" / "seed_branch_continuation.json")
    rows = [row for row in data["continuation"] if row.get("certified") and not row.get("failed")]
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


def exact_lambda_as() -> tuple[np.ndarray, np.ndarray]:
    path = JSON_DIR / "exact_lambda_gamma_AS_n18_100.json"
    data = load_json(path)
    ns = np.array(data["n_values"], dtype=float)
    lam = np.array([float(data["lambda_abs"][str(int(n))]) for n in ns], dtype=float)
    order = np.argsort(ns)
    return ns[order], lam[order]


def fixed_control_c1() -> float:
    data = load_json(JSON_DIR / "prefactor_convergence.json")
    rows = [
        row
        for row in data["results"]
        if abs(float(row["beta"]) - BETA) < 1e-4 and abs(float(row["gamma"]) - GAMMA_CONTROL) < 1e-9
    ]
    if not rows:
        raise KeyError("Missing off-crossing control prefactor row")
    return float(rows[0]["fit_c1"])


def explicit_three_saddle_D(ns: np.ndarray, delta43: float, delta46: float, rho43: complex, rho46: complex) -> np.ndarray:
    vals = []
    for n in ns:
        amp = 1.0 + rho43 * np.exp(1j * n * delta43) + rho46 * np.exp(1j * n * delta46)
        vals.append(math.log(max(abs(amp), 1e-300)))
    return np.asarray(vals, dtype=float)


def style_panel(ax: plt.Axes) -> None:
    ax.grid(True, alpha=0.24)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def make_figure() -> Path:
    ns, exact = exact_lambda_as()
    seed = load_seed_series()
    b43 = branch_series(43)
    b46 = branch_series(46)
    c1_control = fixed_control_c1()

    seed_full = interp(GAMMA_AS, seed["gamma"], seed["full"])
    seed_im = interp(GAMMA_AS, seed["gamma"], seed["im"])
    theta43 = interp(GAMMA_AS, b43["gamma"], b43["im"]) - seed_im
    theta46 = interp(GAMMA_AS, b46["gamma"], b46["im"]) - seed_im

    lambda_seed_corr = seed_full - c1_control / ns
    lambda_eff_two = lambda_seed_corr + math.log(2.0) / ns
    D_three = explicit_three_saddle_D(ns, theta43, theta46, rho43=1.0 + 0.0j, rho46=1.0 + 0.0j)
    lambda_three = lambda_seed_corr + D_three / ns

    D_exact = ns * (exact - lambda_seed_corr)
    residual_seed = np.abs(lambda_seed_corr - exact)
    residual_two = np.abs(lambda_eff_two - exact)
    residual_three = np.abs(lambda_three - exact)

    fig = plt.figure(figsize=(15.2, 6.8))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.15, 1.0], wspace=0.28)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])

    ax_a.plot(ns, exact, "o", ms=3.6, color=BLACK, mfc="white", mew=1.0, label=r"exact $\lambda_{\rm abs}(n)$")
    ax_a.plot(ns, lambda_seed_corr, "-", color=BLUE, lw=2.1, label=r"corrected seed $\lambda_{\rm seed,corr}$")
    ax_a.plot(ns, lambda_eff_two, "-", color=GREEN, lw=2.3, label="effective two-channel")
    ax_a.plot(ns, lambda_three, "-", color=PURPLE, lw=2.0, label="explicit seed+43+46 candidate")
    ax_a.set_title(r"(a) Finite-$n$ rate", fontsize=11, fontweight="bold")
    ax_a.set_xlabel(r"$n$")
    ax_a.set_ylabel(r"$\lambda_{\rm abs}(n)=n^{-1}\log|P_n|$")
    ax_a.legend(fontsize=7.8, loc="upper right")
    style_panel(ax_a)

    ax_b.plot(ns, D_exact, "o-", color=BLACK, ms=3.5, lw=1.8, label=r"exact $D_n$")
    ax_b.axhline(0.0, color=BLUE, ls="--", lw=1.4, alpha=0.85, label="0: seed-only")
    ax_b.axhline(math.log(2.0), color=GREEN, ls="-.", lw=1.7, alpha=0.9, label=r"$\log 2$: two channels")
    ax_b.axhline(math.log(3.0), color=PURPLE, ls=":", lw=2.0, alpha=0.95, label=r"$\log 3$: three aligned")
    ax_b.plot(ns, D_three, "-", color=PURPLE, lw=1.4, alpha=0.55, label="explicit phase model")
    ax_b.set_title(r"(b) Rescaled excess $D_n$", fontsize=11, fontweight="bold")
    ax_b.set_xlabel(r"$n$")
    ax_b.set_ylabel(r"$D_n=n[\lambda_{\rm abs}(n)-\lambda_{\rm seed,corr}(n)]$")
    ax_b.legend(fontsize=7.4, loc="lower right")
    style_panel(ax_b)

    ax_c.semilogy(ns, residual_seed, "-", color=BLUE, lw=2.0, label="corrected seed")
    ax_c.semilogy(ns, residual_two, "-", color=GREEN, lw=2.2, label="effective two-channel")
    ax_c.semilogy(ns, residual_three, "-", color=PURPLE, lw=2.0, label="explicit seed+43+46")
    ax_c.set_title("(c) Residual error", fontsize=11, fontweight="bold")
    ax_c.set_xlabel(r"$n$")
    ax_c.set_ylabel(r"$|$prediction $-$ exact$|$")
    ax_c.legend(fontsize=8, loc="upper right")
    style_panel(ax_c)

    fig.suptitle(
        "Finite-size multiplicity correction at the certified anti-Stokes crossing",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    footer = (
        rf"$\gamma_{{AS}}={GAMMA_AS:.12f}$; seed correction fixed from the seed-controlled point "
        rf"$\gamma={GAMMA_CONTROL}$ with $c_1={c1_control:.6f}$ and reused for all crossing candidates. "
        "Explicit seed+43+46 curve uses rho43=rho46=1 and unknown contour coefficients are not determined."
    )
    fig.text(0.01, -0.015, footer, ha="left", va="top", fontsize=8.3, color=GRAY, wrap=True)

    OUT.mkdir(parents=True, exist_ok=True)
    SUB_OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / "anti_stokes_multiplicity_correction_v1.png"
    fig.savefig(out, dpi=190, bbox_inches="tight")
    plt.close(fig)

    sub_out = SUB_OUT / out.name
    shutil.copy2(out, sub_out)
    caption = (
        "Finite-size multiplicity correction at the certified anti-Stokes crossing.\n\n"
        f"The certified crossing used here is gamma_AS = {GAMMA_AS:.15f}. "
        f"The finite-n corrected seed baseline is lambda_seed_corr(n)=lambda_seed - c1/n, "
        f"with c1={c1_control:.12f} determined at the off-crossing seed-controlled point "
        f"beta={BETA}, gamma={GAMMA_CONTROL}; no crossing data are used to fit this correction. "
        "The seed saddle underestimates lambda_abs(n) at the crossing.\n\n"
        "At the certified equal-real-action crossing, the exact finite-size rate lies above the "
        "continued seed-saddle prediction. After removing the ordinary single-saddle prefactor "
        "correction, the remaining discrepancy is compared with the parameter-free log(2)/n and "
        "log(3)/n corrections expected from two and three equal-magnitude saddle channels. "
        "Agreement with one of these corrections provides numerical evidence for a multi-saddle "
        "contribution, although the unknown contour coefficients prevent identification of the "
        "exact Picard-Lefschetz decomposition.\n\n"
        "The effective two-channel model is seed plus a combined competitor-pair channel. The "
        "explicit seed+43+46 model uses D_model(n)=log|1+rho_43 exp(i n Delta theta_43)+"
        "rho_46 exp(i n Delta theta_46)| with rho_43=rho_46=1 for this equal-prefactor candidate. "
        "Replacing these rho values requires Gaussian-prefactor ratios and contour coefficients.\n"
    )
    for caption_path in [
        OUT / "anti_stokes_multiplicity_correction_v1_caption.txt",
        SUB_OUT / "anti_stokes_multiplicity_correction_v1_caption.txt",
    ]:
        caption_path.write_text(caption)
    return sub_out


def main() -> None:
    path = make_figure()
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
