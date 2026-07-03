# c_typ vs c_ann analysis

Source: `/Users/b/Quandela/phasecraft/results/bm24_runs/06-09/run1/train12-tr100-te200-n12-18.json`
Rebenchmarked: True

## Definitions

- **c_typ**: log₂ slope of median(1/p_succ) vs n (notebook eval)
- **c_emp_mean**: log₂ slope of mean(p_succ) vs n
- **c_ann**: Re(φ_full)/ln2 from BM24 saddle at trained LR angles
- **c_ann_rt** = −c_ann: naive runtime exponent from annealed mean success

## Interpretation

| Gap | Meaning |
|-----|---------|
| gap_mean_vs_ann = c_emp_mean − c_ann | BM24 mean-success alignment |
| gap_typ_vs_annrt = c_typ − (−c_ann) | Typical runtime vs annealed proxy |
| delta_ctyp (med−mean train) | Effect of retraining objective |

## Summary table

| depth | mode | c_typ | c_emp_mean | c_ann | gap_mean_ann | gap_typ_annrt |
|------:|------|------:|-----------:|------:|-------------:|--------------:|
| 10 | mean_p train | 0.5132 | -0.4687 | nan | nan | nan |
| 10 | median_rt train | 0.5187 | -0.4735 | nan | nan | nan |
| 20 | mean_p train | 0.4219 | -0.3815 | nan | nan | nan |
| 20 | median_rt train | 0.4202 | -0.3836 | nan | nan | nan |
| 40 | mean_p train | 0.3687 | -0.3048 | nan | nan | nan |
| 40 | median_rt train | 0.3827 | -0.3146 | nan | nan | nan |
| 50 | mean_p train | 0.3603 | -0.2807 | nan | nan | nan |
| 50 | median_rt train | 0.3710 | -0.2895 | nan | nan | nan |
| 80 | mean_p train | 0.3334 | -0.2315 | nan | nan | nan |
| 80 | median_rt train | 0.3340 | -0.2392 | nan | nan | nan |
| 100 | mean_p train | 0.3178 | -0.2096 | nan | nan | nan |
| 100 | median_rt train | 0.3215 | -0.2126 | nan | nan | nan |

## Median vs mean training (same depth)

| depth | Δ c_typ | Δ gap_mean_ann |
|------:|--------:|---------------:|
| 10 | +0.0055 | +nan |
| 20 | -0.0017 | +nan |
| 40 | +0.0140 | +nan |
| 50 | +0.0107 | +nan |
| 80 | +0.0006 | +nan |
| 100 | +0.0037 | +nan |