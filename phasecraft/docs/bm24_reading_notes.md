# BM24 Reading Notes

Primary reference: <https://arxiv.org/abs/2208.06909>

Local PDF copy noted by workspace context:
`/Users/b/Downloads/PRXQuantum.5.030348 (1).pdf`

## Local Naming

| Paper object | Local name | Where used |
|--------------|------------|------------|
| BM24 finite-n success probability | Exact finite-n exponent / `lambda_abs` | `lib/sim/bm24_qaoa_sim.py`, `experiments/lr_scaling/`, `experiments/bm24_saddle_audit_p1/` |
| Proposition 4 expression | `BM24 Prop. 4` | `lib/sim/bm24_qaoa_sim.py`, `experiments/bm24_saddle_audit_p1/audit.py` |
| Appendix finite-n expression | `BM24 Eq. A10` | `lib/sim/bm24_qaoa_sim.py`, `experiments/bm24_saddle_audit_p1/audit.py` |
| BM24 fixed-point iterate | `BM24 seed saddle` | `experiments/bm24_saddle_audit_p1/` |
| Krawczyk-certified saddle root | `certified saddle` | `lib/saddles/`, audit outputs |

## Equations To Track

| Label | Meaning | Local implementation | Notes |
|-------|---------|----------------------|-------|
| BM24 Prop. 4 | Exact finite-n success probability | `lib/sim/bm24_qaoa_sim.py::exact_finite_n_p1_prop4`; `experiments/bm24_saddle_audit_p1/audit.py::log_p_succ_prop4` | Used as the finite-n reference for p=1 exponent matching. |
| BM24 Eq. A10 | Finite-n expression used in pipeline | Same functions as above | Current code labels this as Convention 1. |
| BM24 Eq. A41 | All-subsets finite-n representation | `lib/ksat/variants/generalized_binomial_sum.PATCHED.py::generalized_flip_symmetric_expected_success_p1` | Current patched implementation uses Convention 2 prefactor `exp(-(r/2^k)n)`. |
| Saddle equations | Stationary equations for saddle analysis | `lib/ksat/variants/generalized_binomial_sum.PATCHED.py::generalized_binomial_sum_scaling_exponent_ksat`; `lib/saddles/krawczyk_p1_roots.py`; `experiments/w_saddle/core.py` | Distinguish z-coordinate and w-coordinate charts. |
| Prefactor convention | Conv1 vs Conv2 offset | `bm24_prefactor_exponent_ksat`; `bm24_prefactor_exponent_ksat_all_subsets` | Important for matching `lambda_abs` to `Re Phi`. |

## Thesis-Ready Definitions

TODO: Convert each into polished thesis prose.

- **Finite-n evidence:** Direct evaluation or simulation at finite problem sizes.
- **Saddle certification:** Krawczyk/interval evidence that a saddle root exists
  near a numerical candidate.
- **Contour dominance:** Stronger claim about which saddle controls the contour;
  do not conflate with local certification.
- **Competitor saddle:** A certified saddle discovered by search or continuation
  that may challenge the BM24 seed branch.

## Reading Log

| Date | Section | What changed in my understanding | Follow-up |
|------|---------|-----------------------------------|-----------|
| 2026-06-10 | Local code map | The repo has both a direct Prop. 4 / Eq. A10 p=1 implementation and a patched Convention-2 all-subsets implementation. | Confirm which one should be cited for each thesis figure. |
