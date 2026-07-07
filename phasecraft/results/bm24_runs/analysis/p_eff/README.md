# p_eff diagnostic

Two scripts, **three files** in this folder when you run them:

| File | From |
|------|------|
| `hardware_sweep.json` + `hardware_sweep.png` | `find_p_eff_hardware.py --sweep-eps …` |
| `step1.json` + `step1.png` | `find_p_eff.py` (optional) |

## Step 1 — random-angle baseline (noise-free sim)

```bash
python experiments/lr_scaling/find_p_eff.py \
  --run-json results/bm24_runs/06-09/run1/train12-tr100-te200-n12-18.json \
  --output-dir results/bm24_runs/analysis/p_eff \
  --eval-n 12 --test-size 40 --num-random 80 --depth-max 50
```

Writes `step1.json` / `step1.png` (override default names via `--output-dir`).

Compares trained mean $p_{\mathrm{succ}}$ to a **random LR-angle** band ($\mu + 3\sigma$). Mostly a sanity check — the random-angle ceiling rises with $p$, so $p_{\mathrm{eff}} \approx 10$ here is a statistical artifact, not hardware.

## Step 2 — hardware-style (depth noise + bitstring baseline)

```bash
python experiments/lr_scaling/find_p_eff_hardware.py \
  --run-json results/bm24_runs/06-09/run1/train12-tr100-te200-n12-18.json \
  --eval-n 12 --test-size 30 --depth-max 100 \
  --num-bootstrap 100 --perf-bootstrap 1000 --ci-level 0.95 \
  --sweep-eps 0.01,0.02,0.03,0.05,0.08,0.12
```

Writes **one** `hardware_sweep.json` (all $\varepsilon$ rows + depth curves) and **one** `hardware_sweep.png` (multi-panel).

- Noise: $p_{\mathrm{noisy}} = F(p)\,p_{\mathrm{ideal}} + (1-F(p))\,p_{\mathrm{uniform}}$, $F=(1-\varepsilon)^p$
- Random band: uniform bitstrings, bootstrap 99.73% threshold (paper Sec. A.3)
- Performance uncertainty: bootstrap CI over held-out instance means
- Metrics per $\varepsilon$: $p_{\mathrm{peak}}$, $p_{\mathrm{half}}$, $p_{\mathrm{eff}}$ plus censoring / bracket status

### Current sweep (run1, $n=12$)

Random-bitstring threshold: $1.3533\times 10^{-3}$.

| $\varepsilon$/layer | $p_{\mathrm{peak}}$ | $p_{\mathrm{half}}$ | $p_{\mathrm{eff}}$ report | status | confirmed lower bound |
|---------------------|---------------------|---------------------|----------------------------|--------|-----------------------|
| 0.01 | 50 | 100 | $\ge 100$ | right-censored | 100 |
| 0.02 | 40 | 80 | $\ge 100$ | right-censored | 100 |
| 0.03 | 30 | 50 | $\ge 100$ | right-censored | 100 |
| 0.05 | 20 | 40 | $\ge 100$ | right-censored | 100 |
| 0.08 | 15 | 30 | 50, bracketed by 80 | mean crossing bracket [50, 80] | 50 |
| 0.12 | 10 | 20 | 40, bracketed by 50 | mean crossing bracket [40, 50] | 40 |

$p_{\mathrm{peak}}$ is still the useful readout at BM24 threshold ($r=176.54$): the bitstring band sits near $10^{-3}$, so most $p_{\mathrm{eff}}$ rows are lower bounds rather than exact cutoffs.

Interpretation rule:

- `right_censored`: the scan never reached the random band; read `$p_{\mathrm{eff}}\ge$ last tested depth`.
- `bracketed`: the mean crossed between the last passing sampled depth and the next sampled depth.
- `confirmed lower bound`: every sampled depth through that value has a 95% held-out-instance bootstrap lower CI above the random threshold.

## References

- Montañez-Barrera et al., arXiv:2502.06471
- Angles: `results/bm24_runs/06-09/run1/train12-tr100-te200-n12-18.json`
