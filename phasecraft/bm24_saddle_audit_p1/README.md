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

Outputs land in `phasecraft/bm24_saddle_audit_p1/results/<timestamp>/`:

| File | Contents |
|------|----------|
| `saddle_table.csv` / `saddle_table.json` | Per-saddle certification, Φ, det(H), action gaps, finite-n match flags |
| `finite_n_exponents.json` | `lambda_abs`, `lambda_local` on the n-grid |
| `gamma_scan.json` | Stokes / anti-Stokes pairs along the γ sweep |
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
