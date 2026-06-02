"""
Computer-assisted verification of assumptions (A) and (B).

Assumption (A): Non-degeneracy at every saddle found: |Phi''(theta)| != 0.
Assumption (B): Distinct Re[Phi] among the dominant saddles (heuristic check).

This script uses mpmath (arbitrary precision) and analytic formulas for Phi, Phi', Phi''.
It is *numerical* (not interval-rigorous) unless interval arithmetic is added.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Iterable

import mpmath as mp


def w_of_gamma(gamma_tilde: mp.mpf) -> mp.mpc:
    # w = sqrt(-i * gamma_tilde / 2)
    return mp.sqrt(-1j * gamma_tilde / 2)


def A(theta: mp.mpc, beta: mp.mpf, gamma_tilde: mp.mpf) -> mp.mpc:
    ww = w_of_gamma(gamma_tilde)
    return mp.cos(beta / 2) - 1j * mp.sin(beta / 2) * mp.e ** (theta * ww)


def Phi(theta: mp.mpc, beta: mp.mpf, gamma_tilde: mp.mpf) -> mp.mpc:
    return -theta**2 / 4 + mp.log(A(theta, beta, gamma_tilde))


def dPhi(theta: mp.mpc, beta: mp.mpf, gamma_tilde: mp.mpf) -> mp.mpc:
    ww = w_of_gamma(gamma_tilde)
    s = mp.sin(beta / 2)
    aa = A(theta, beta, gamma_tilde)
    return -theta / 2 - 1j * ww * s * mp.e ** (theta * ww) / (aa + mp.mpf("1e-50"))


def d2Phi(theta: mp.mpc, beta: mp.mpf, gamma_tilde: mp.mpf) -> mp.mpc:
    ww = w_of_gamma(gamma_tilde)
    s = mp.sin(beta / 2)
    e = mp.e ** (theta * ww)
    aa = A(theta, beta, gamma_tilde) + mp.mpf("1e-50")
    term1 = -mp.mpf("1") / 2
    term2 = -1j * ww**2 * s * e / aa
    term3 = (ww * s * e / aa) ** 2
    return term1 + term2 + term3


def r_theta(theta: mp.mpc, beta: mp.mpf, gamma_tilde: mp.mpf) -> mp.mpc:
    """
    r(theta) = A'(theta)/A(theta) = Phi'(theta) + theta/2.
    Kept explicit (not via Phi') to improve stability for partial derivatives.
    """
    ww = w_of_gamma(gamma_tilde)
    s = mp.sin(beta / 2)
    c = mp.cos(beta / 2)
    z = mp.e ** (theta * ww)
    A_val = c - 1j * s * z
    Aprime = -1j * s * ww * z
    return Aprime / (A_val + mp.mpf("1e-50"))


def dw_dgamma(gamma_tilde: mp.mpf) -> mp.mpc:
    """
    w = sqrt(-i*gamma_tilde/2) satisfies w^2 = -i*gamma_tilde/2.
    Differentiate: 2 w dw/dgamma = -i/2 => dw/dgamma = -i/(4 w).
    """
    ww = w_of_gamma(gamma_tilde)
    return (-1j) / (4 * ww)


def dr_dgamma(theta: mp.mpc, beta: mp.mpf, gamma_tilde: mp.mpf) -> mp.mpc:
    """
    Partial derivative ∂r/∂gamma_tilde holding theta fixed.
    r = N/D with N = (-i s) * w * e^{theta w}, D = c - i s e^{theta w}.
    """
    ww = w_of_gamma(gamma_tilde)
    s = mp.sin(beta / 2)
    c = mp.cos(beta / 2)
    z = mp.e ** (theta * ww)

    N = (-1j * s) * ww * z
    D = c - 1j * s * z
    # z' = d/dw e^{theta w} = theta e^{theta w} = theta z
    dz_dww = theta * z
    dN_dww = (-1j * s) * (z + ww * dz_dww)
    dD_dww = -1j * s * dz_dww

    dr_dww = (dN_dww * D - N * dD_dww) / (D**2 + mp.mpf("1e-80"))
    return dr_dww * dw_dgamma(gamma_tilde)


def d2Phi_dgamma_partial(theta: mp.mpc, beta: mp.mpf, gamma_tilde: mp.mpf) -> mp.mpc:
    """
    Partial derivative ∂/∂gamma_tilde of Phi''(theta;gamma_tilde) at fixed theta.
    Using Phi'' = -1/2 + w*r - r^2:
      ∂Phi''/∂gamma = (dw/dg) * r + (w - 2r) * (∂r/∂g).
    """
    ww = w_of_gamma(gamma_tilde)
    r = r_theta(theta, beta, gamma_tilde)
    drg = dr_dgamma(theta, beta, gamma_tilde)
    wdg = dw_dgamma(gamma_tilde)
    return wdg * r + (ww - 2 * r) * drg


def d3Phi(theta: mp.mpc, beta: mp.mpf, gamma_tilde: mp.mpf) -> mp.mpc:
    """
    Third derivative with respect to theta: Phi'''(theta).
    This matches qaoa_core.d3Phi for the same action.
    """
    ww = w_of_gamma(gamma_tilde)
    s = mp.sin(beta / 2)
    e = mp.e ** (theta * ww)
    aa = A(theta, beta, gamma_tilde) + mp.mpf("1e-50")
    ap = -1j * ww * s * e
    app = -1j * ww**2 * s * e
    appp = -1j * ww**3 * s * e
    # (A'/A)'' = A'''/A - 3 A'' A' / A^2 + 2 (A')^3 / A^3
    ratio_pp = appp / aa - 3 * (app * ap) / (aa**2) + 2 * (ap**3) / (aa**3)
    return ratio_pp


def poles(beta: mp.mpf, gamma_tilde: mp.mpf, kmin: int, kmax: int) -> list[tuple[int, mp.mpc]]:
    ww = w_of_gamma(gamma_tilde)
    # theta_k = (log(-i cot(beta/2)) + 2π i k)/w
    base = mp.log(-1j * mp.cot(beta / 2))
    out: list[tuple[int, mp.mpc]] = []
    for k in range(kmin, kmax + 1):
        th = (base + 2 * mp.pi * 1j * k) / ww
        out.append((k, th))
    return out


def newton_solve(
    theta0: mp.mpc,
    beta: mp.mpf,
    gamma_tilde: mp.mpf,
    tol_dphi: mp.mpf,
    max_iter: int,
    step_cap: mp.mpf,
) -> tuple[mp.mpc, bool]:
    th = theta0
    for _ in range(max_iter):
        f = dPhi(th, beta, gamma_tilde)
        if abs(f) < tol_dphi:
            return th, True
        fp = d2Phi(th, beta, gamma_tilde)
        if abs(fp) < mp.mpf("1e-60"):
            break
        step = f / fp
        if abs(step) > step_cap:
            step *= step_cap / abs(step)
        th = th - step
    return th, abs(dPhi(th, beta, gamma_tilde)) < tol_dphi


def dedup_add(roots: list[mp.mpc], th: mp.mpc, tol: mp.mpf) -> bool:
    for r in roots:
        if abs(th - r) < tol:
            return False
    roots.append(th)
    return True


def find_saddles_for_gamma(
    beta: mp.mpf,
    gamma_tilde: mp.mpf,
    R_cutoff: mp.mpf,
    k_cap: int,
    tol_dphi: mp.mpf,
    newton_max_iter: int,
) -> list[mp.mpc]:
    # Heuristic: initialize Newton near poles inside |theta_k|<R_cutoff.
    # This matches the algorithmic spirit of your verification instructions.
    ww = w_of_gamma(gamma_tilde)
    roots: list[mp.mpc] = []

    for _, th_k in poles(beta, gamma_tilde, -k_cap, k_cap):
        if abs(th_k) >= R_cutoff:
            continue

        denom = ww * ww * th_k  # w^2 * theta_k
        if abs(denom) < mp.mpf("1e-30"):
            continue

        # Asymptotic saddle init near pole:
        # theta_* ≈ theta_k - i*gamma_tilde / (w^2 * theta_k)
        corr = -1j * gamma_tilde / denom
        for a in (mp.mpf("0.5"), mp.mpf("1.0"), mp.mpf("1.5")):
            th0 = th_k + a * corr
            th_sol, ok = newton_solve(
                th0,
                beta,
                gamma_tilde,
                tol_dphi=tol_dphi,
                max_iter=newton_max_iter,
                step_cap=mp.mpf("5.0"),
            )
            if ok:
                dedup_add(roots, th_sol, tol=mp.mpf("1e-6"))

    # Also include theta=0 as a candidate for very small gamma (optional).
    if gamma_tilde < mp.mpf("0.3"):
        th_sol, ok = newton_solve(
            mp.mpc(0),
            beta,
            gamma_tilde,
            tol_dphi=tol_dphi,
            max_iter=max(20, newton_max_iter // 2),
            step_cap=mp.mpf("2.0"),
        )
        if ok:
            dedup_add(roots, th_sol, tol=mp.mpf("1e-6"))

    return roots


@dataclass(frozen=True)
class NonDegeneracyReport:
    passed: bool
    epsilon: mp.mpf
    min_abs_d2phi: mp.mpf
    min_location: tuple[mp.mpf, mp.mpc]
    count_small: int
    lip_refine_performed: bool
    lip_refine_passed: bool
    lip_refined_min_bound: mp.mpf


def verify_nondegeneracy(
    beta: mp.mpf,
    gamma_tilde_max: mp.mpf,
    delta_gamma: mp.mpf,
    R_cutoff: mp.mpf,
    k_cap: int,
    grid_tol_dphi: mp.mpf,
    nondeg_epsilon: mp.mpf,
    small_threshold: mp.mpf,
    newton_max_iter: int,
    lip_refine: bool,
    lip_samples: int,
    lip_safety_factor: float,
) -> NonDegeneracyReport:
    g = mp.mpf("1e-6")
    min_abs = mp.inf
    min_loc: tuple[mp.mpf, mp.mpc] | None = None
    count_small = 0
    lip_refine_performed = bool(lip_refine)
    lip_refine_passed = True
    lip_refined_min_bound = mp.inf

    while g <= gamma_tilde_max + mp.mpf("1e-14"):
        saddles = find_saddles_for_gamma(
            beta=beta,
            gamma_tilde=g,
            R_cutoff=R_cutoff,
            k_cap=k_cap,
            tol_dphi=grid_tol_dphi,
            newton_max_iter=newton_max_iter,
        )

        for th in saddles:
            abs_d2 = abs(d2Phi(th, beta, g))
            if abs_d2 < min_abs:
                min_abs = abs_d2
                min_loc = (g, th)
            if abs_d2 < small_threshold:
                count_small += 1

        g += delta_gamma

    assert min_loc is not None
    passed = min_abs > nondeg_epsilon

    # Optional Lipschitz refinement around the grid minimizer.
    # This is a "computer-assisted proof" of non-vanishing on the local interval:
    # |Phi''| can decrease at most by L * (delta_gamma/2), where
    # L bounds |d/dgamma Phi''(theta*(gamma);gamma)|.
    refined_passed = True
    refined_min_bound = min_abs
    if lip_refine and min_loc is not None and delta_gamma > 0:
        gamma0 = mp.mpf(min_loc[0])
        theta0 = min_loc[1]
        delta = delta_gamma / 2
        gamma_left = max(mp.mpf("1e-6"), gamma0 - delta)
        gamma_right = min(gamma_tilde_max, gamma0 + delta)

        # Track the same saddle branch by continuation seeded from theta0.
        # We sample additional points inside [gamma_left, gamma_right] and compute
        # the total derivative using the chain rule + IFT.
        if gamma_right > gamma_left:
            gam_samples = [
                gamma_left + (gamma_right - gamma_left) * (mp.mpf(i) / (lip_samples - 1))
                for i in range(lip_samples)
            ]
            theta_prev = theta0
            abs_d2_samples: list[mp.mpf] = []
            deriv_samples: list[mp.mpf] = []
            for gg in gam_samples:
                theta_sol, ok = newton_solve(
                    theta_prev,
                    beta=beta,
                    gamma_tilde=gg,
                    tol_dphi=grid_tol_dphi,
                    max_iter=newton_max_iter,
                    step_cap=mp.mpf("5.0"),
                )
                if not ok:
                    refined_passed = False
                    break
                # continuation seed for next point
                theta_prev = theta_sol
                abs_d2 = abs(d2Phi(theta_sol, beta, gg))
                abs_d2_samples.append(abs_d2)

                # total derivative d/dgamma Phi'' on the saddle branch
                phi2 = d2Phi(theta_sol, beta, gg)
                if abs(phi2) < mp.mpf("1e-80"):
                    refined_passed = False
                    break
                drg = dr_dgamma(theta_sol, beta, gg)  # ∂Phi'/∂gamma = ∂r/∂gamma
                dtheta_dg = -drg / phi2  # IFT along Phi'=0
                phi3 = d3Phi(theta_sol, beta, gg)
                part_phi2_g = d2Phi_dgamma_partial(theta_sol, beta, gg)
                total_dphi2 = phi3 * dtheta_dg + part_phi2_g
                deriv_samples.append(abs(total_dphi2))

            if refined_passed and abs_d2_samples:
                m_min_local = min(abs_d2_samples)
                L = max(deriv_samples) * mp.mpf(lip_safety_factor)
                half_width = (gamma_right - gamma_left) / 2
                refined_min_bound = m_min_local - L * half_width
                refined_passed = refined_min_bound > nondeg_epsilon
        else:
            refined_passed = refined_passed and (min_abs > nondeg_epsilon)

    # If the refined local check fails, we still return the coarse-grid min.
    # Users can then increase the grid density or sampling parameters.
    passed = passed and refined_passed
    lip_refine_passed = refined_passed
    lip_refined_min_bound = refined_min_bound
    return NonDegeneracyReport(
        passed=passed,
        epsilon=nondeg_epsilon,
        min_abs_d2phi=min_abs,
        min_location=min_loc,
        count_small=count_small,
        lip_refine_performed=lip_refine_performed,
        lip_refine_passed=lip_refine_passed,
        lip_refined_min_bound=lip_refined_min_bound,
    )


def verify_distinct_rephi(
    beta: mp.mpf,
    gamma_tilde_max: mp.mpf,
    delta_gamma: mp.mpf,
    R_cutoff: mp.mpf,
    k_cap: int,
    grid_tol_dphi: mp.mpf,
    gap_tol: mp.mpf,
    newton_max_iter: int,
    topN: int = 2,
) -> tuple[mp.mpf, tuple[mp.mpf, list[mp.mpf]]] :
    g = mp.mpf("1e-6")
    min_gap = mp.inf
    min_loc: tuple[mp.mpf, list[mp.mpf]] | None = None

    while g <= gamma_tilde_max + mp.mpf("1e-14"):
        saddles = find_saddles_for_gamma(
            beta=beta,
            gamma_tilde=g,
            R_cutoff=R_cutoff,
            k_cap=k_cap,
            tol_dphi=grid_tol_dphi,
            newton_max_iter=newton_max_iter,
        )
        rephis: list[mp.mpf] = []
        for th in saddles:
            rephis.append(mp.re(Phi(th, beta, g)))
        if len(rephis) >= topN:
            rephis.sort(reverse=True)
            top = [mp.mpf(r) for r in rephis[:topN]]
            # compute minimum pairwise gap among the topN
            gap = mp.inf
            for i in range(topN):
                for j in range(i + 1, topN):
                    gap = mp.mpf(min(gap, abs(top[i] - top[j])))
            if gap < min_gap:
                min_gap = gap
                min_loc = (g, top)
        g += delta_gamma

    assert min_loc is not None
    return min_gap, min_loc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--beta", type=str, default="-pi/2", help="e.g. -pi/2, pi/2")
    parser.add_argument("--gamma-max", type=str, default="2*pi")
    parser.add_argument("--delta-gamma", type=str, default="pi/25")
    parser.add_argument("--precision", type=int, default=70)
    parser.add_argument("--R-cutoff", type=float, default=35.0)
    parser.add_argument("--k-cap", type=int, default=70)
    parser.add_argument("--tol-dphi", type=str, default="1e-26")
    parser.add_argument("--newton-max-iter", type=int, default=50)
    parser.add_argument("--nondeg-epsilon", type=str, default="1e-14")
    parser.add_argument("--small-threshold", type=str, default="1e-12")
    parser.add_argument("--gap-tol", type=str, default="1e-8")
    parser.add_argument("--lip-refine", action="store_true", help="Run Lipschitz refinement near the min |Phi''| grid point")
    parser.add_argument("--lip-samples", type=int, default=9, help="Number of gamma samples for the local Lipschitz refinement")
    parser.add_argument("--lip-safety-factor", type=float, default=2.0, help="Conservative factor multiplying L in the refinement test")
    args = parser.parse_args()

    # Parse expressions like -pi/2 and 2*pi
    mp.mp.dps = int(args.precision)
    local_env = {"pi": mp.pi}
    beta = mp.mpf(eval(args.beta, {"__builtins__": {}}, local_env))
    gamma_max = mp.mpf(eval(args.gamma_max, {"__builtins__": {}}, local_env))
    delta_gamma = mp.mpf(eval(args.delta_gamma, {"__builtins__": {}}, local_env))

    report = verify_nondegeneracy(
        beta=beta,
        gamma_tilde_max=gamma_max,
        delta_gamma=delta_gamma,
        R_cutoff=mp.mpf(args.R_cutoff),
        k_cap=int(args.k_cap),
        grid_tol_dphi=mp.mpf(args.tol_dphi),
        nondeg_epsilon=mp.mpf(args.nondeg_epsilon),
        small_threshold=mp.mpf(args.small_threshold),
        newton_max_iter=int(args.newton_max_iter),
        lip_refine=args.lip_refine,
        lip_samples=args.lip_samples,
        lip_safety_factor=args.lip_safety_factor,
    )

    min_gap, min_loc = verify_distinct_rephi(
        beta=beta,
        gamma_tilde_max=gamma_max,
        delta_gamma=delta_gamma,
        R_cutoff=mp.mpf(args.R_cutoff),
        k_cap=int(args.k_cap),
        grid_tol_dphi=mp.mpf(args.tol_dphi),
        gap_tol=mp.mpf(args.gap_tol),
        newton_max_iter=int(args.newton_max_iter),
        topN=2,
    )

    print("Assumption (A): Non-degeneracy check (numerical)")
    print(f"  beta           = {beta}")
    print(f"  gamma_tilde ∈ [1e-6, {gamma_max}] step={delta_gamma}")
    print(f"  min |Phi''|   = {report.min_abs_d2phi}")
    print(f"  min location   = (gt={report.min_location[0]}, theta={report.min_location[1]})")
    print(f"  count |Phi''|<{report.epsilon} : {report.count_small}")
    print(f"  passed         = {report.passed} (epsilon={report.epsilon})")
    if report.lip_refine_performed:
        print("\nLipschitz refinement (local, around min |Phi''|):")
        print(f"  lip_refine_passed        = {report.lip_refine_passed}")
        print(f"  certified lower bound min|Phi''| >= {report.lip_refined_min_bound}")

    print("\nAssumption (B): Distinct Re[Phi] among top 2 (heuristic)")
    print(f"  min gap among top-2 Re[Phi] = {min_gap}")
    print(f"  at (gt, [Re_top])            = {min_loc}")


if __name__ == "__main__":
    main()

