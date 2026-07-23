# QAOA vs LR-QAOA finite-size expressivity probe
Protocol: BM24 random 8-SAT, r=176.54; train full regular QAOA and LR-QAOA on the same 20 SAT-filtered n=12 formulas; evaluate both on the same 20 held-out formulas at each n=12,13,14; depths p=2,3,5; three independent train/eval seeds. Objective is BM24 mean p_succ at fixed n. CPU serial, COBYLA maxiter=80, restarts=2.
This is a controlled finite-size probe, not an asymptotic production benchmark. Lower log2 slope is better.
| p | QAOA slope mean [min,max] | LR slope mean [min,max] | mean LR-QAOA gap | train mean-p ratio QAOA/LR | n=14 runtime ratio LR/QAOA | nearest-LR angle residual | n=14 runtime ratio nearest-LR/QAOA |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 0.436 [0.297,0.533] | 0.616 [0.479,0.707] | 0.180 | 2.94x | 3.69x | 0.362 | 5.46x |
| 3 | 0.389 [0.249,0.465] | 0.492 [0.353,0.564] | 0.104 | 2.42x | 2.80x | 0.313 | 4.00x |
| 5 | 0.282 [0.103,0.415] | 0.383 [0.171,0.529] | 0.100 | 2.45x | 3.12x | 0.285 | 3.47x |

Interpretation: in all 9 matched depth/seed comparisons, full regular QAOA has a lower held-out median-runtime slope than optimized LR-QAOA. The training objective gap is larger: full QAOA achieves roughly 2.4-3.0x higher mean success at n=12. The learned full-QAOA schedules are not close to the nearest LR ramp, especially at p=3 and p=5, and simply projecting them to the nearest LR ramp worsens n=14 median runtime by about 2.4-5.3x.

Main caveat: n=12..14 and N=20 are small; the slopes should be read as finite-size diagnostics. A production answer would repeat this with GPU/batched backend, larger N, and deeper p.
