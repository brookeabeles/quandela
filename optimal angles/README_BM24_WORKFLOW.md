# BM24 QAOA-Angles Workflow

This folder now contains a staged implementation aligned to the instructions in
`optimal_qaoa_angles (1).md`:

- `qaoa_angles/toy_model.py`: Level 1 toy model (exact finite-`n`, continuation saddle, optimization, validation)
- `qaoa_angles/ksat_p1.py`: Level 2 exact `p=1` expected-success formulas and landscape scan
- `qaoa_angles/ksat_general.py`: Level 3 general-`p` saddle-point helpers with continuation/fixed-point
- `qaoa_angles/subset_sum.py`: BM24-style subset-sum transforms (Algorithms 3-4 style)
- `qaoa_angles/optimization.py`: wrappers for toy / p=1 / general-p optimization
- `qaoa_angles/validation.py`: validation helpers
- `qaoa_angles/plots.py`: simple plotting helper
- `run_qaoa_angles.py`: CLI entry point

## Quick commands

```bash
python3 "optimal angles/run_qaoa_angles.py" --mode toy-opt --out "optimal angles/toy_opt.json"
python3 "optimal angles/run_qaoa_angles.py" --mode toy-validate --beta -1.57079632679 --gamma 3.14159265359 --out "optimal angles/toy_val.json"
python3 "optimal angles/run_qaoa_angles.py" --mode p1-opt --k 8 --r 176.54 --n-eval 30 --out "optimal angles/p1_opt_k8.json"
python3 "optimal angles/run_qaoa_angles.py" --mode general-opt --k 8 --p 2 --r 176.54 --out "optimal angles/general_opt_k8_p2.json"
```

## Notes

- Exponents are treated as `(1/n) log E[p_succ]` and should be `<= 0`.
- The optimization wrappers clamp unphysical positive values as a safety guard.
- SciPy is required for optimization and special functions.
