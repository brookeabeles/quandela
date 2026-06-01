# Research directions from corrected 8-SAT (p=1, q=3) diagnostics

## What the fixed p=1 run already shows

1. **Saddle rank collapse is universal** — at every converged γ, \(k_{99}\) drops from 2 (at \(y=0\)) to 1 at the saddle. Stable rank \(\approx 1\). The Hessian is effectively one-dimensional in the \(d=8\) reduced space.
2. **γ = π/2 is on the convergence cliff** at \(r=176.54\): residual \(\sim 1.1\times 10^{-2}\) with the same homotopy budget that succeeds elsewhere.
3. **p=1 is a degenerate toy model** for block/truncation/probe1 plots — use it to validate the pipeline, not to infer large-\(p\) scaling.

## Most interesting directions

| Priority | Direction | Why it matters |
|----------|-----------|----------------|
| **1** | **Saddle vs \(y=0\) spectral collapse across p** | TRUNCATION_NUMERICS shows at \(p=6\), \(k_{99}\) can fall 1007→243 at large γ. Does corrected Eq.(20) reproduce that compression? |
| **2** | **\(k_{99}\) vs \(p\) scaling at \(y=0\)** | Power law \(k_{99}\sim C p^\alpha\) with \(\alpha\approx 2\)–4 sets BM24 / block-decay exponents; cheap without saddle. |
| **3** | **Audit: \(k=2^q\) vs \(k=2q\)** | Wrong map is a *different* fixed point — saddles and all Hessian plots from pre-fix code need re-run. |
| **4** | **Integral error: \(k_{99}\) vs stable rank** | At large p, stable rank \(\ll k_{99}\); probe1 asks which controls Gaussian truncation error. |
| **5** | **Link to z-Krawczyk / PL** | Old pipeline in `phasecraft/og_krawczyk_data` certifies roots in **z**; u/y 8-SAT saddle is a different chart — compare only after explicit change of variables. |
| **6** | **γ-homotopy engineering** | Cold starts from \(\gamma\sim 10^{-12}\) need \(\mathcal{O}(10^2)\) steps to reach \(\gamma\sim 0.14\); warmstart along γ sweeps is mandatory for \(p\ge 2\). |

## New figures (`explore_directions.py`)

```bash
python3.12 symmetry_reduction/fixed/explore_directions.py
```

| File | Insight |
|------|---------|
| `insight_rank_collapse_gamma.png` | Compression ratio \(k_{99}^{saddle}/k_{99}^{y=0}\) vs γ at p=1 |
| `insight_k99_y0_vs_p.png` | \(k_{99}(y=0)\) power-law growth with p (no saddle) |
| `insight_block_y0_p123.png` | When block off-diagonal structure becomes visible |
| `insight_wrong_vs_correct_k.png` | Bug impact: different weights and \(k_{99}\) for \(k=8\) vs \(k=6\) |
| `insight_saddle_collapse_p12.png` | Does collapse persist for p=2? |
| `insight_convergence_phase.png` | \((p,\gamma)\) convergence map |

Data: `data/insight_summary.json`
