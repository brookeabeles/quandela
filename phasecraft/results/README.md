# Results Directory

All experiment outputs live here. Each subdirectory corresponds to one pipeline.

---

## bm24_saddle_audit_p1/

Z-coordinate Krawczyk saddle audit for BM24 p=1.

- Timestamped run folders: `run_MM-DD_HH-MM-SSZ`
- Active seed-continuation run: `run_seed_branch_g-2pi/`
- Competitor scan outputs: `run_seed_competitors/`
- Aborted/partial runs: `_archive_partial/`
- **Navigation:** see `bm24_saddle_audit_p1/RUN_INDEX.md` for the latest recommended run

Entry point: `experiments/bm24_saddle_audit_p1/audit.py`

---

## w_saddle_runs/

W-coordinate Krawczyk pipeline outputs (`run1/` through `run7/`).

- `run7/` is the most recent
- Each run folder contains JSON files: `competitors_*.json`, `resolved_*.json`, `seed_*.json`, `tracked_*.json`

Entry point: `experiments/w_saddle/` (`python -m phasecraft.w_saddle`)

---

## bm24_runs/

BM24 QAOA finite-n benchmark results, organised by date (`05-18/`, `05-19/`, …).

- Each date folder has one or more `run*/` subdirs with JSON scaling results
- `.txt` files at top level are training logs

Entry point: `experiments/lr_scaling/`

---

## archive/

Legacy reference data — do not use for new analyses.

- `legacy_w_saddle_flat/` — flat JSON from pre-refactor w-saddle runs
- `og_krawczyk_data/` — original z-coordinate Krawczyk outputs (pre-lib)

---

## best/

Placeholder for curated / publication-ready outputs. Currently empty.
