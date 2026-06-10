# Research Organization

This repo supports three overlapping jobs:

1. Reproduce and interrogate BM24 numerics.
2. Preserve evidence from exploratory runs.
3. Feed a thesis with traceable claims, figures, and references.

The organization should make those jobs easy to separate. Code can be messy while
it is being discovered; conclusions should not be.

## Working Map

| Area | Role | Rule of thumb |
|------|------|---------------|
| `lib/` | Reusable, importable research machinery | New code here only when two experiments need it, or when it encodes a stable mathematical object. |
| `experiments/` | Runnable studies and notebooks | Each subfolder should answer one research question and own its entry points. |
| `results/` | Generated evidence | Never rely on memory; every important run gets an index entry or a short note. |
| `results/best/` | Thesis-ready figures and tables | Curated outputs only; each file should be reproducible from a named run or script. |
| `results/archive/` | Historical context | Keep for provenance, but do not use for new claims without a note explaining why. |
| `docs/` | Human navigation layer | Put maps, thesis notes, claim ledgers, and reading notes here. |

## BM24 Vocabulary

In this repo, **BM24** refers to the paper at <https://arxiv.org/abs/2208.06909>.
When writing thesis notes or code comments, prefer explicit labels:

- `BM24 Prop. 4` for exact finite-n success probability expressions.
- `BM24 Eq. A10` for the finite-n expression used by the benchmark pipeline.
- `BM24 seed saddle` for the saddle reached from the BM24 fixed-point iterator.
- `certified saddle` for a Krawczyk-certified root of the saddle equations.
- `discovered competitor` for numerically found certified alternatives that have
  not been globally proven contour-dominant.

This distinction matters because the repo currently compares exact finite-n
exponents, BM24 iterator objects, and Krawczyk-certified saddle objects.

## Thesis Evidence Workflow

Use this loop whenever a result becomes thesis-relevant:

1. **Question.** State the claim in one sentence before running more code.
2. **Run.** Save outputs under the existing pipeline-specific `results/` folder.
3. **Index.** Add the run to the relevant `RUN_INDEX.md` or README with date,
   parameters, and whether it is recommended, exploratory, partial, or obsolete.
4. **Curate.** Copy only thesis-ready figures/tables into `results/best/`.
5. **Claim.** Record what the output supports, and what it does not prove.
6. **Cite.** Link the claim back to BM24, local scripts, and exact output files.

The best thesis notes are boringly traceable: claim -> script -> run folder ->
figure/table -> citation.

## Suggested Docs To Add Next

These files would make the project easier to navigate without moving code:

| File | Purpose |
|------|---------|
| `docs/bm24_reading_notes.md` | Definitions, equations, and local naming choices for BM24. |
| `docs/thesis_claim_ledger.md` | A table of thesis claims, evidence paths, figures, and confidence levels. |
| `docs/result_curation.md` | Rules for promoting outputs into `results/best/`. |
| `docs/open_questions.md` | Live research questions, dead ends, and decisions needed from supervisors. |

These files now exist as starter templates; fill them incrementally instead of
waiting for a perfect reorganization pass.

## Naming Habits

- Use `run_MM-DD_HH-MM-SSZ` for timestamped runs when a pipeline already does so.
- Use descriptive suffixes for parameter sweeps, for example `g-2pi` or
  `sweep_beta_gamma_full`.
- Avoid names like `final`, `new`, `test`, or `fixed` for artifacts that may be
  read later.
- Put old notebooks under an `archive/` folder once a script becomes the source
  of truth.

## Current Pain Points

The repo is usable, but these are the places most likely to waste thesis time:

- `results/best/` has curated-looking figures but no provenance note yet.
- `results/bm24_saddle_audit_p1/RUN_INDEX.md` is valuable; other result areas
  would benefit from the same pattern.
- Notebook archives exist, but the canonical relationship between notebooks and
  scripts should be stated more explicitly.
- The distinction between finite-n evidence, saddle certification, and contour
  dominance should be repeated anywhere thesis-facing claims are made.
