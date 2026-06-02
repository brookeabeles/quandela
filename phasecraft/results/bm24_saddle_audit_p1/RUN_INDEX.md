# BM24 Saddle Audit Run Index

Result folders are named in readable UTC format:

- `run_MM-DD_HH-MM-SSZ`
- Example: `run_05-27_13-13-28Z` means `May 27, 13:13:28 UTC`.

This is intentional and generated automatically by `run_audit()`.

## Latest Recommended Run

- `run_05-27_14-18-48Z` (Conv2 primary baseline, anchor γ near optimum)

## Small-γ certification sweep (lean outputs, no plots)

- `run_05-27_15-06-27Z_g+0.001` — γ = +1e-3, `num_certified=3`
- `run_05-27_15-06-33Z_g-0.001` — γ = -1e-3, `num_certified=1`
- `run_05-27_15-06-38Z_g+0.01` — γ = +1e-2, `num_certified=7`
- `run_05-27_15-06-47Z_g-0.01` — γ = -1e-2, `num_certified=5`
- `run_05-27_15-06-54Z_g+0.05` — γ = +5e-2, `num_certified=14`
- `run_05-27_15-07-05Z_g-0.05` — γ = -5e-2, `num_certified=17`

Each keeps: `summary.json`, `finite_n_exponents.json`, `saddle_table.json`, `object_comparison_table.json`.

## Older full runs

- `run_05-27_13-13-28Z`
  - Most complete and up-to-date output schema
  - Includes:
    - `summary.json`
    - `saddle_table.json` / `.csv`
    - `object_comparison_table.json` / `.csv`
    - `finite_n_exponents.json`
    - `gamma_scan.json`
    - `sanity_checks.json`
    - plots (`exponent_vs_saddles.png`, `gamma_stokes_scan.png`)

## Relevant Past Runs

- `run_05-27_11-10-19Z`
  - First run clearly showing BM24 iterate matching Conv1 finite-n exponent while not necessarily satisfying `G(z)=0`.
  - Good baseline for comparison with newer labeling.

- `run_05-27_13-06-13Z`
  - Transitional run with explicit Conv1/Conv2 split and q=1 sanity metadata in `summary.json`.
  - Uses a slightly different finite-n file naming style (`finite_n_exponents_conv1.json`, `finite_n_exponents_conv2.json`).

  - (Archived as partial) `run_05-27_13-12-29Z` contains only sanity preflight output.

## Incomplete / Partial Artifacts (Archived)

Partial/aborted runs are moved to:

- `results/_archive_partial/`

Currently archived examples include:

- `run_05-27_11-01-13Z` (empty aborted run)
- `run_05-27_10-59-13Z`, `run_05-27_10-59-28Z` (early partial bring-up)
- `run_05-27_13-12-29Z` (sanity-only preflight)

For analysis/reporting, prefer the latest recommended run above.
