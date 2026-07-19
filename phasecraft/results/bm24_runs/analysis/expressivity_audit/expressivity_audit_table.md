# Expressivity evidence audit summary table

## A. Key number provenance (selected)

| Label | Value | Source | Field | Objective | Seeds | n-window | Fit model | Kind |
|-------|------:|--------|-------|-----------|-------|----------|-----------|------|
| LR c_rt seed0 @ p=100 | 0.3051 | multi_seed_summary.json | c_typ_mean_p | bm24_mean_p_fixed_n (angles); median(1/p) vs n | 0 | 12-18 | log2 linear regression on median(1/p_succ) vs n | raw_depth |
| LR c_rt seed27 @ p=100 | 0.3178 | multi_seed_summary.json | c_typ_mean_p | bm24_mean_p_fixed_n (angles); median(1/p) vs n | 27 | 12-18 | log2 linear regression on median(1/p_succ) vs n | raw_depth |
| LR c_rt seed42 @ p=100 | 0.3223 | multi_seed_summary.json | c_typ_mean_p | bm24_mean_p_fixed_n (angles); median(1/p) vs n | 42 | 12-18 | log2 linear regression on median(1/p_succ) vs n | raw_depth |
| c_inf multiseed mean_p | 0.3133 | objective_exp_convergence.json | c_inf | mean_p | 0,27,42 | 12-18 | c_inf + A p^(-beta), p>=10 | fitted_asymptote |
| c_inf multiseed median_rt | 0.3159 | objective_exp_convergence.json | c_inf | median_rt | 0,27,42 | 12-18 | c_inf + A p^(-beta), p>=10 | fitted_asymptote |
| WalkSAT c | 0.4071 | scaling-tn14.json setup.classical | walksat median flips log2 slope | classical baseline | n/a | 12-18 | log2 linear on median flips vs n | horizontal_reference |
| BM24 analytic @ p=100 | 0.1581 | plot_multi_seed_objective_ab.py constants | 0.69*p^{-0.32} | analytic reference | n/a | n/a | pure power (no offset) | analytic_formula |
| full QAOA mean_p @ n=12 p=2 (cpu-prx-small) | 0.0154 | cpu-prx-small.json | qaoa_mean_p_succ_per_n['12'] | bm24_mean_p_fixed_n | 27 | 12-13 | n/a | raw_depth (finite-size benchmark) |
| LR mean_p @ n=12 p=2 (cpu-prx-small) | 0.0056 | cpu-prx-small.json | lr_eval.per_n['12'].mean_p_succ | bm24_mean_p_fixed_n | 27 | 12-13 | n/a | raw_depth (finite-size benchmark) |
| full QAOA mean_p @ n=12 p=5 (cpu-prx-small) | 0.0673 | cpu-prx-small.json | qaoa_mean_p_succ_per_n['12'] | bm24_mean_p_fixed_n | 27 | 12-13 | n/a | raw_depth (finite-size benchmark) |
| LR mean_p @ n=12 p=5 (cpu-prx-small) | 0.0267 | cpu-prx-small.json | lr_eval.per_n['12'].mean_p_succ | bm24_mean_p_fixed_n | 27 | 12-13 | n/a | raw_depth (finite-size benchmark) |
| full QAOA mean_p @ n=4 p=1 (smoke) | 0.3620 | smoke.json | qaoa_mean_p_succ_per_n['4'] | bm24_mean_p_fixed_n | 27 | 4-5 | n/a | raw_depth (finite-size benchmark) |
| LR mean_p @ n=4 p=1 (smoke) | 0.1250 | smoke.json | lr_eval.per_n['4'].mean_p_succ | bm24_mean_p_fixed_n | 27 | 4-5 | n/a | raw_depth (finite-size benchmark) |
| full QAOA mean_p @ n=4 p=2 (smoke) | 0.4962 | smoke.json | qaoa_mean_p_succ_per_n['4'] | bm24_mean_p_fixed_n | 27 | 4-5 | n/a | raw_depth (finite-size benchmark) |
| LR mean_p @ n=4 p=2 (smoke) | 0.3458 | smoke.json | lr_eval.per_n['4'].mean_p_succ | bm24_mean_p_fixed_n | 27 | 4-5 | n/a | raw_depth (finite-size benchmark) |
| full QAOA mean_p @ n=4 p=1 (batched-smoke) | 0.3620 | batched-smoke.json | qaoa_mean_p_succ_per_n['4'] | bm24_mean_p_fixed_n | 27 | 4-5 | n/a | raw_depth (finite-size benchmark) |
| LR mean_p @ n=4 p=1 (batched-smoke) | 0.1250 | batched-smoke.json | lr_eval.per_n['4'].mean_p_succ | bm24_mean_p_fixed_n | 27 | 4-5 | n/a | raw_depth (finite-size benchmark) |
| full QAOA mean_p @ n=12 p=1 (smoke-qaoa-vs-lr-gpu) | 0.0066 | smoke-qaoa-vs-lr-gpu.json | qaoa_mean_p_succ_per_n['12'] | bm24_mean_p_fixed_n | 27 | 12-13 | n/a | raw_depth (finite-size benchmark) |
| LR mean_p @ n=12 p=1 (smoke-qaoa-vs-lr-gpu) | 0.0005 | smoke-qaoa-vs-lr-gpu.json | lr_eval.per_n['12'].mean_p_succ | bm24_mean_p_fixed_n | 27 | 12-13 | n/a | raw_depth (finite-size benchmark) |
| full QAOA mean_p @ n=12 p=2 (smoke-qaoa-vs-lr-gpu) | 0.0178 | smoke-qaoa-vs-lr-gpu.json | qaoa_mean_p_succ_per_n['12'] | bm24_mean_p_fixed_n | 27 | 12-13 | n/a | raw_depth (finite-size benchmark) |
| LR mean_p @ n=12 p=2 (smoke-qaoa-vs-lr-gpu) | 0.0066 | smoke-qaoa-vs-lr-gpu.json | lr_eval.per_n['12'].mean_p_succ | bm24_mean_p_fixed_n | 27 | 12-13 | n/a | raw_depth (finite-size benchmark) |

## B. Full-QAOA angle projection onto LR schedule (LS fit, decreasing beta)

| Source | p | |g_perp|/|g| | Δγ_fit | Δβ_fit | Finite-n benchmark? |
|--------|--:|--------:|-------:|-------:|:-------------------:|
| cpu-prx-small.json | 2 | 0.3574 | -1.1393 | 1.4413 | True |
| cpu-prx-small.json | 5 | 0.2720 | -1.3383 | 1.1149 | True |
| smoke.json | 1 | nan | -1.5364 | nan | True |
| smoke.json | 2 | 0.2819 | -2.0203 | 2.6045 | True |
| batched-smoke.json | 1 | nan | -1.5364 | nan | True |
| smoke-qaoa-vs-lr-gpu.json | 1 | nan | -0.9120 | nan | True |
| smoke-qaoa-vs-lr-gpu.json | 2 | 0.3594 | -1.1444 | 1.4482 | True |

## C. Projection performance loss (full QAOA → nearest LR angles, same held-out set)

| Source | p | n | Δ mean_p | frac loss mean_p | frac worse median_rt | Finite-n? |
|--------|--:|--:|---------:|-----------------:|---------------------:|:---------:|
| cpu-prx-small.json | 2 | 12 | 0.011525 | 0.7486 | 3.0177 | True |
| cpu-prx-small.json | 2 | 13 | 0.010273 | 0.7743 | 4.8006 | True |
| cpu-prx-small.json | 5 | 12 | 0.043139 | 0.6410 | 2.0695 | True |
| cpu-prx-small.json | 5 | 13 | 0.036549 | 0.6601 | 3.0024 | True |
| smoke.json | 1 | 4 | nan | nan | nan | True |
| smoke.json | 1 | 5 | nan | nan | nan | True |
| smoke.json | 2 | 4 | 0.173468 | 0.3496 | 0.6152 | True |
| smoke.json | 2 | 5 | 0.133365 | 0.4665 | 1.9827 | True |
| batched-smoke.json | 1 | 4 | nan | nan | nan | True |
| batched-smoke.json | 1 | 5 | nan | nan | nan | True |
| smoke-qaoa-vs-lr-gpu.json | 1 | 12 | nan | nan | nan | True |
| smoke-qaoa-vs-lr-gpu.json | 1 | 13 | nan | nan | nan | True |
| smoke-qaoa-vs-lr-gpu.json | 2 | 12 | 0.013424 | 0.7525 | 2.8325 | True |
| smoke-qaoa-vs-lr-gpu.json | 2 | 13 | 0.010376 | 0.7526 | 3.6328 | True |

## D. Figure 3 gap decomposition (per seed, mean_p-trained LR, n=12–18)

| Seed | p | Δ_obj | Δ_bench | Δ_total | c_rt | c_sp | c_BM24 |
|-----:|--:|------:|--------:|--------:|-----:|-----:|-------:|
| 0 | 2 | -0.0292 | 0.1196 | 0.0905 | 0.6432 | 0.6724 | 0.5527 |
| 27 | 2 | -0.0231 | 0.1238 | 0.1007 | 0.6535 | 0.6766 | 0.5527 |
| 42 | 2 | -0.0099 | 0.1337 | 0.1239 | 0.6766 | 0.6865 | 0.5527 |
| 0 | 10 | 0.0159 | 0.1359 | 0.1518 | 0.4820 | 0.4661 | 0.3303 |
| 27 | 10 | 0.0176 | 0.1385 | 0.1561 | 0.4863 | 0.4687 | 0.3303 |
| 42 | 10 | 0.0337 | 0.1474 | 0.1812 | 0.5114 | 0.4777 | 0.3303 |
| 0 | 20 | 0.0371 | 0.1081 | 0.1452 | 0.4098 | 0.3726 | 0.2646 |
| 27 | 20 | 0.0208 | 0.1169 | 0.1377 | 0.4022 | 0.3815 | 0.2646 |
| 42 | 20 | 0.0537 | 0.1222 | 0.1759 | 0.4404 | 0.3867 | 0.2646 |
| 0 | 40 | 0.0557 | 0.0904 | 0.1461 | 0.3580 | 0.3023 | 0.2119 |
| 27 | 40 | 0.0639 | 0.0929 | 0.1567 | 0.3687 | 0.3048 | 0.2119 |
| 42 | 40 | 0.0732 | 0.1011 | 0.1743 | 0.3862 | 0.3130 | 0.2119 |
| 0 | 50 | 0.0595 | 0.0838 | 0.1433 | 0.3406 | 0.2811 | 0.1973 |
| 27 | 50 | 0.0796 | 0.0834 | 0.1629 | 0.3603 | 0.2807 | 0.1973 |
| 42 | 50 | 0.0658 | 0.0932 | 0.1590 | 0.3564 | 0.2905 | 0.1973 |
| 0 | 60 | 0.0616 | 0.0776 | 0.1392 | 0.3253 | 0.2637 | 0.1861 |
| 42 | 60 | 0.0673 | 0.0851 | 0.1524 | 0.3385 | 0.2712 | 0.1861 |
| 0 | 80 | 0.0734 | 0.0685 | 0.1419 | 0.3117 | 0.2383 | 0.1698 |
| 27 | 80 | 0.1019 | 0.0617 | 0.1637 | 0.3334 | 0.2315 | 0.1698 |
| 42 | 80 | 0.0852 | 0.0730 | 0.1581 | 0.3279 | 0.2427 | 0.1698 |
| 0 | 100 | 0.0844 | 0.0626 | 0.1470 | 0.3051 | 0.2207 | 0.1581 |
| 27 | 100 | 0.1082 | 0.0516 | 0.1598 | 0.3178 | 0.2096 | 0.1581 |
| 42 | 100 | 0.1006 | 0.0636 | 0.1642 | 0.3223 | 0.2217 | 0.1581 |

## E. Instance bootstrap CIs (N500 frozen angles, n=12–18)

| p | Δ_obj CI | Δ_bench CI | Δ_total CI |
|--:|:---------|:-----------|:-----------|
| 2 | [-0.0251, 0.0096] | [0.1102, 0.1348] | [0.0918, 0.1394] |
| 5 | [-0.0432, -0.0017] | [0.1731, 0.2152] | [0.1442, 0.1976] |
| 10 | [-0.0642, -0.0066] | [0.1734, 0.2312] | [0.1404, 0.1947] |
| 20 | [-0.0811, -0.0045] | [0.1690, 0.2510] | [0.1361, 0.1986] |
| 50 | [-0.1578, 0.0253] | [0.1446, 0.3277] | [0.1373, 0.2048] |
