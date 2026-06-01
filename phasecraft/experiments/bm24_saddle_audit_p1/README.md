# BM24 p=1 saddle audit (q=3, k=8, r=176.54)

Numerical pipeline comparing **exact finite-n** success probabilities (BM24 Proposition 4 / Eq. A10) against **certified saddle actions** from the original **z-coordinate** Krawczyk stack (`krawczyk_p1_roots.py`), not the `(u,w)` chart.

## Run

From the repo root:

```bash
python -m phasecraft.bm24_saddle_audit_p1.audit
```

Quick smoke (fewer starts, smaller grids):

```bash
python -m phasecraft.bm24_saddle_audit_p1.audit --quick
```

### Certified seed-branch continuation

Track the BM24 iterator seed saddle from small negative γ toward more negative γ
(Krawczyk-certified, Conv2 primary, no silent root jumps):

```bash
python -m phasecraft.bm24_saddle_audit_p1.continue_seed_branch_certified
python -m phasecraft.bm24_saddle_audit_p1.continue_seed_branch_certified --quick
```

Writes `results/<run>_seed_branch/` with `seed_branch_continuation.csv/json`,
`competitor_saddles_by_gamma.json`, and γ-scan plots (exponent, gap, residual,
nearest competitor Re-action gap). Does **not** claim contour dominance.

### Focused discovered-competitor diagnostic

Stronger competitor search at selected γ (default: `-0.3, -0.6, -0.83, -1.0, -1.6, -2.0`):

```bash
python -m phasecraft.bm24_saddle_audit_p1.competitor_dominance_diagnostic
python -m phasecraft.bm24_saddle_audit_p1.competitor_dominance_diagnostic --quick
```

Uses continuation interpolation for the seed when `run_seed_branch_g-2pi` data
is present (≥3000 random starts). Outputs `competitor_dominance_summary.json` only.
Labeled **discovered competitor diagnostic** — not a global dominance proof.

```bash
python -m phasecraft.bm24_saddle_audit_p1.analyze_high_re_competitor_branch
python -m phasecraft.bm24_saddle_audit_p1.continue_competitor_branch
```

**Plots to keep** (under `run_seed_branch_g-2pi/competitor_dominance/`):

| Plot | Purpose |
|------|---------|
| `competitor_branch_re_phi_vs_seed.png` | Dense γ continuation: high-Re island at −0.83 vs seed elsewhere |
| `competitor_branch_signed_re_gap.png` | Signed Re Φ gap along that continuation |
| `competitor_branch_im_phi_difference.png` | Unwrapped Im Φ(seed − competitor) |

**JSON:** `competitor_dominance_summary.json`, `high_re_competitor_branch_summary.json`,
`competitor_branch_from_gamma_-0.83.json`, `max_re_competitor_by_gamma.json`.

Branch classification (local continuation from each anchor):

```bash
python -m phasecraft.bm24_saddle_audit_p1.branch_classify_competitors
```

Writes `branch_classified_competitors.json`, `branch_valid_dominance_summary.json`, and
`gamma_vs_algebraic_and_branch_valid_gap.png`, `gamma_vs_num_branch_valid.png`.

Seed-branch plots from the −2π continuation live in `run_seed_branch_g-2pi/` (not `competitor_dominance/`).

Outputs land in `phasecraft/bm24_saddle_audit_p1/results/<run-name>/`:

Run folder names use readable UTC names `run_MM-DD_HH-MM-SSZ` generated at launch.
If a run aborts early, its folder can be partially populated (or empty).
See `phasecraft/bm24_saddle_audit_p1/results/RUN_INDEX.md` for the current
"latest recommended" and relevant historical runs.

| File | Contents |
|------|----------|
| `saddle_table.csv` / `saddle_table.json` | Per-saddle certification, Φ, det(H), action gaps, finite-n match flags |
| `object_comparison_table.csv` / `.json` | BM24 iterator vs Newton-polished vs Krawczyk-certified object-level comparison |
| `finite_n_exponents.json` | `lambda_abs`, `lambda_local` on the n-grid |
| `gamma_scan.json` | Stokes / anti-Stokes pairs along the γ sweep |
| `sanity_checks.json` | q=1 small-γ preflight check (`γ=1e-3,1e-2,5e-2`) |
| `exponent_vs_saddles.png` | λ curves vs Re Φ_M, Re Φ_M+φ_pref, and φ_pref lines |
| `gamma_stokes_scan.png` | Near Stokes/anti-Stokes events vs γ |

## Defaults

- `q=3`, `k=8`, `r=176.54`, `p=1`
- Angles from `phasecraft.optimal_angles[(8, 176.54)][1]` unless `--beta` / `--gamma` are passed
- Rigorous interval Krawczyk certification (`--rigorous`, default on). Use `--max-certify N` to cap certification cost when many roots are found.
- `matches_lambda_abs_full`: |λ_abs − (Re Φ_M + φ_pref)| within tolerance (Prop4 / Convention-1 prefactor).
- `re_phi_plus_pref_conv2`: Re Φ_M + (−r/2^k) (Convention-2 prefactor for the all-subsets saddle chart).
- `controls_finite_n`: certified and `matches_lambda_abs_full`.
- `is_bm24_seed_saddle`: certified root closest in Re Φ_M to the fixed-point seed from `generalized_binomial_sum_scaling_exponent_ksat`.

**Convention note.** Exact finite-n Prop4 uses Convention-1 prefactor
exp(−(r/k)·n·(1+4 sin²(γ/4))). Saddle Φ_M comes from the Convention-2
(all-subsets) equations; the match to λ_abs(n) is Re Φ_M + φ_pref (Conv. 1),
not Re Φ_M alone. The BM24 seed track often matches λ while other certified
saddles are conjugate branches with different Re Φ_M.

## Dependencies

Reuses `generalized_binomial_sum.PATCHED.py`, `krawczyk_p1_roots.py`, `picard_lefschetz.compute_phi`, and `scipy` / `mpmath`.
