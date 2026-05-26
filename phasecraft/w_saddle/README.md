# W-saddle Krawczyk pipeline

Certified saddles for the parent-action equations in **w**-coordinates  
(corrected slice β = −π/2, p = 1, q = 1). Outputs live in **`runs/`** next to this README.

## Folder layout

```
phasecraft/w_saddle/
  core.py           # interval Krawczyk, WSaddleSystem, root discovery
  workflow.py       # γ continuation, competitor sweeps, branch resolve, plots
  cli.py            # command-line (also: python -m phasecraft.w_saddle)
  __init__.py
  README.md         # this file
  runs/             # JSON + PNG artifacts (gitignored except .gitkeep)
    run1/           # one folder per pipeline execution
    run2/
    ...
```

Parent shims (optional, for old habits):  
`phasecraft/krawczyk_w_saddle.py`, `continue_w_saddle_branch.py`, `competitor_branch_tracking.py`, `run_w_saddle.py`

Proof wording: `phasecraft/certificate_language.py` (not in this folder).

---

## Commands (from repo root)

| Command | What it does |
|---------|----------------|
| `python -m phasecraft.w_saddle --help` | List subcommands |
| `python -m phasecraft.w_saddle certify --r 1 --gamma -0.05 --escalate` | One γ: random starts → Newton → Krawczyk boxes |
| `python -m phasecraft.w_saddle continue ...` | Track **seed branch** along negative γ mesh |
| `python -m phasecraft.w_saddle sweep --in runs/seed_....json --out runs/competitors.json` | At each γ: discover all roots, certify, gap vs seed |
| `python -m phasecraft.w_saddle resolve --sweep-in ... --seed-branch-in ...` | Match competitors into sheets, optional crossing refine |
| `python -m phasecraft.w_saddle track ...` | Continue one competitor sheet from sweep anchors |
| `python -m phasecraft.w_saddle pipeline --preset r1-neg-gamma --plots` | **continue → sweep → resolve** into `runs/` |
| `make -f phasecraft/Makefile.w_saddle pipeline` | Same as pipeline preset |
| `python -m phasecraft.w_saddle.core --selftest` | Memo identities + small-γ certification smoke |

### Match / beat the May 2026 robust study (recommended)

```bash
cd /path/to/Quandela   # repo root

python -m phasecraft.w_saddle pipeline \
  --preset r1-neg-gamma --robust-sweep \
  --r 1.0 --dps 80 --competitor-starts 500 --seed 0 \
  --plots --enclose-action --proof-metadata
```

This is the robust workflow in one shot:

1. **continue** — 100-point γ mesh −0.01 → −1.0  
2. **sweep** — 500 random starts per γ, then **auto-detect** where competitors threaten the seed and **dense resweep** that γ interval (no fixed −0.98/−0.82)  
3. **resolve** — branch-track; **auto-refine every ΔRe sign-change** on disjoint sheets

Writes under a new `runs/runN/` (e.g. `run3/seed_r1_neg_gamma.json`, `run3/competitors_robust.json`, …).

Or: `make -f phasecraft/Makefile.w_saddle pipeline-robust`

Reuse a run folder: `python -m phasecraft.w_saddle pipeline --run 2 --plots …`

### Quick (coarse) full run

```bash
python -m phasecraft.w_saddle pipeline --preset r1-neg-gamma --r 1.0 --dps 80 --plots
```

Writes `competitors.json` / `resolved.json` (no wall refine — can look worse near γ ≈ −0.92).

### Step-by-step (same paths)

```bash
# Manual steps: pick a run folder (pipeline creates runs/runN automatically)
RUN=phasecraft/w_saddle/runs/run1

python -m phasecraft.w_saddle continue \
  --r 1.0 --gamma-start -0.01 --gamma-stop -1.0 --num-points 100 --dps 80 \
  --out $RUN/seed_r1_neg_gamma.json --plot $RUN/seed_branch.png

python -m phasecraft.w_saddle sweep \
  --in $RUN/seed_r1_neg_gamma.json --out $RUN/competitors.json \
  --plot-analysis $RUN/competitors_analysis.png

python -m phasecraft.w_saddle resolve \
  --sweep-in $RUN/competitors.json --seed-branch-in $RUN/seed_r1_neg_gamma.json \
  --out $RUN/resolved.json --plot $RUN/resolved.png
```

---

## What each script does

| Module | Role |
|--------|------|
| **`core.py`** | Defines Δ*(w), couplings, `F_tilde(w)=0`, mpmath interval Krawczyk (`krawczyk_certify_w_root`), `discover_w_roots`, `Phi_eff`. |
| **`workflow.py`** | Warm-start γ continuation; per-γ competitor sweep; branch tracking / resolve; matplotlib plots. |
| **`cli.py`** | Subcommands `certify`, `continue`, `sweep`, `resolve`, `track`, `pipeline`; saves JSON via `certificate_language`. |
| **`certificate_language.py`** (parent) | Theorem text, disclaimers, `finalize_payload_certificate` on JSON outputs. |

**Main entry:** `python -m phasecraft.w_saddle` → `cli.py`.

---

## Dependencies

| Dependency | Used for |
|------------|----------|
| **numpy** | Newton centers, realified root finding |
| **scipy** (`optimize.root`) | Numerical roots before certification |
| **mpmath** (`iv`) | Rigorous interval Krawczyk + divisor bounds |
| **matplotlib** | PNG plots (optional; only when `--plot` / `--plots`) |
| **`symmetry_reduction.core`** (`compute_b_s`) | Cross-check Δ vs general nontrivial-index formula in tests |
| **`phasecraft.certificate_language`** | JSON proof metadata (local import) |

Not used by this pipeline: Julia certifiers, `krawczyk_p1_roots` (old **z** variables), QAOA training code.

---

## Outsourcing / scope

- **In scope:** Local Krawczyk box per (r, γ), discrete γ meshes, competitor **diagnostics** (Re Φ gaps).
- **Out of scope:** Interval-in-γ tubes (unless you add tube certs), Stokes/PL dominance, global uniqueness, QAOA exponent proofs.

Old flat outputs under `phasecraft/w_branch_*` are from earlier runs; new runs should use `w_saddle/runs/`.

---

## Plot quantities (equations on PNG titles)

**Competitor sweep** (`competitors_*_analysis.png`) — for **branch curves**, run full `pipeline --plots` (resolve redraws this file with one color per `branch_id`). Sweep-only PNG is green/red per mesh \(\gamma\), not branch tracking.

| Panel | Quantity |
|-------|----------|
| Gap | \(\Delta_{\min}(\gamma)=\min_{j\in\mathcal{C}_\gamma\setminus\{s\}}(\mathrm{Re}\,\Phi_{\mathrm{eff}}(w_s)-\mathrm{Re}\,\Phi_{\mathrm{eff}}(w_j))\) |
| Diagnostic | \(\max_j \mathrm{Re}\,\Phi_{\mathrm{eff}}(w_j)\) at each \(\gamma\) (root can switch → segment breaks) |
| Positive gap | \(\Delta_{+}=\min_{j:\,\Delta_j>0}\Delta_j\) |

**Sweep-only plot** (after `sweep`, before `resolve`): green/red per mesh \(\gamma\) only — **not** branch colors.

**Sweep + resolve** (full `pipeline --plots`): `competitors_*_analysis.png` is redrawn with **one color per tracked `branch_id`** (same as `resolved_*.png`). Use that for separate branches.

**Resolved** (`resolved_*.png`) — dedicated 2×2 branch view; one **color per `branch_id`** in every panel; seed sheet = blue dashed.

\(\Delta\mathrm{Re}(\gamma)=\mathrm{Re}\,\Phi_{\mathrm{eff}}(w_s)-\mathrm{Re}\,\Phi_{\mathrm{eff}}(w_{\mathrm{branch}})\) along tracked sheets.
