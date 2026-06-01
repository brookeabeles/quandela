# Fixed p=1, q=3 diagnostics (corrected BM24 Eq. 20)

Regenerates a **small set of high-value plots** for 8-SAT at threshold using the
corrected saddle map (**k = 2^q**, not 2q) via `symmetry_reduction/newton_saddle_q.py`.

## Run

```bash
cd /path/to/Quandela
python3.12 symmetry_reduction/fixed/regenerate_p1_q3.py          # p=1 (default)
python3.12 symmetry_reduction/fixed/regenerate_p1_q3.py --p 3   # p>1 (slower homotopy)
```

For `p≥2`, the script uses smaller γ steps, more homotopy/Newton budget, sorted γ sweeps with warmstart, and an automatic ultra-conservative retry on failure.

## Parameters

- `p=1`, `q=3`, `r=176.54`, `β=−π/2`
- Panel γ: `π/8`, `π/4`, `0.14`, `0.5` (π/2 does not converge at this r)
- Sweep γ: 10 points from `0.05` to `1.0` (not micro-sweeps near 0)

## Outputs

| File | Content |
|------|---------|
| `figures/block_heatmap_p1_q3.png` | Block fractional energy at 4 representative γ |
| `figures/k99_vs_gamma_p1_q3.png` | k₉₉ and stable rank: y=0 vs saddle |
| `figures/sv_decay_p1_q3.png` | Singular value decay at saddle |
| `figures/hessian_spectrum_p1_q3.png` | Eigenvalue spectrum: y=0 vs saddle |
| `figures/softmax_weights_p1_q3.png` | Weight concentration at saddle |
| `figures/saddle_convergence_p1_q3.png` | Newton/homotopy residual vs γ |
| `figures/truncation_p1_q3_pi4.png` | Block truncation at γ=π/4 |
| `figures/truncation_p1_q3_g0p5.png` | Block truncation at γ=0.5 |
| `data/regeneration_p1_q3_summary.json` | Convergence table |

Data: `data/insight_summary.json`

## Follow-up insights

```bash
python3.12 symmetry_reduction/fixed/explore_directions.py
```

See `INSIGHTS.md` for research directions and `figures/insight_*.png` (6 plots).

