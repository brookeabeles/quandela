# c_typ vs c_ann analysis

Source: `/Users/b/Quandela/phasecraft/results/bm24_runs/multi_seed_ctyp_cann/06-24/run2/train12-tr100-te200-n12-18-seed42.json`
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
| 2 | mean_p train | 0.6766 | -0.6865 | nan | nan | nan |
| 2 | median_rt train | 0.6988 | -0.6953 | nan | nan | nan |

## Median vs mean training (same depth)

| depth | Δ c_typ | Δ gap_mean_ann |
|------:|--------:|---------------:|
| 2 | +0.0221 | +nan |