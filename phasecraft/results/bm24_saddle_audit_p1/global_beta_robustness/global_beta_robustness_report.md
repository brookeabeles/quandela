# Global beta robustness audit

Input: `sweep_beta_gamma_full.json` with 30 beta values, 20 gamma values, exact n=18..24, and 250 competitor starts per point.

## Regime counts

- Robust seed-to-competitor transitions: 3 beta slices (0.4, 0.5, 0.5434).
- Weak seed-to-competitor transitions: 9 beta slices (0.2, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 5, 5.5).
- Nonmonotone/ambiguous slices: 8 beta slices (0.05, 0.1, 0.3, 0.6, 0.7, 1.6, 1.7, 4.712).
- Always seed on grid: 10 beta slices (1.571, 1.8, 1.9, 2, 2.2, 2.5, 3.142, 3.5, 4, 6.283).
- Always competitor on grid: 0 beta slices (none).

## Strongest beta-wide conclusion

The BM24 seed-to-competitor transition is not an isolated beta=0.5434 accident. A coherent robust/near-robust transition band appears around beta=0.4, 0.5, 0.5434, and 0.6, with transition gamma values clustered near -1.75 to -1.87. Outside that band, the coarse grid shows either symmetry-protected seed behavior, immediate competitor dominance, or nonmonotone/ambiguous behavior that needs branch-resolved follow-up.

## Robust transition table

| beta | regime | primary gamma | uncertainty | avg eff gap | decoy frac | transitions |
|---:|---|---:|---:|---:|---:|---:|
| 0.05 | nonmonotone_or_ambiguous | -2.25 | 0.25 | 0.00188 | 0.55 | 2 |
| 0.1 | nonmonotone_or_ambiguous | -1.15 | 0.15 | 0.00239 | 0.70 | 4 |
| 0.2 | weak_seed_to_competitor | -1.4 | 0.1 | 0.00408 | 0.20 | 1 |
| 0.3 | nonmonotone_or_ambiguous | -0.45 | 0.15 | 0.00288 | 0.25 | 3 |
| 0.4 | robust_seed_to_competitor | -1.75 | 0.05 | 0.00283 | 0.20 | 1 |
| 0.5 | robust_seed_to_competitor | -1.815 | 0.015 | 0.00372 | 0.20 | 1 |
| 0.5434 | robust_seed_to_competitor | -1.865 | 0.035 | 0.00413 | 0.30 | 1 |
| 0.6 | nonmonotone_or_ambiguous | -1.65 | 0.05 | 0.00437 | 0.35 | 3 |
| 0.7 | nonmonotone_or_ambiguous | -1.865 | 0.035 | 0.00506 | 0.85 | 2 |
| 0.8 | weak_seed_to_competitor | -0.175 | 0.125 | 0.00428 | 0.05 | 1 |
| 0.9 | weak_seed_to_competitor | -0.175 | 0.125 | 0.00466 | 0.00 | 1 |
| 1 | weak_seed_to_competitor | -0.175 | 0.125 | 0.00614 | 0.00 | 1 |
| 1.1 | weak_seed_to_competitor | -0.175 | 0.125 | 0.00745 | 0.00 | 1 |
| 1.2 | weak_seed_to_competitor | -0.45 | 0.15 | 0.00604 | 0.10 | 1 |
| 1.3 | weak_seed_to_competitor | -0.8 | 0.2 | 0.00717 | 0.15 | 1 |
| 1.5708 | always_seed_on_grid | nan | nan | 0.00138 | 0.95 | 0 |
| 1.6 | nonmonotone_or_ambiguous | -0.45 | 0.15 | 0.00115 | 0.90 | 2 |
| 1.7 | nonmonotone_or_ambiguous | -0.45 | 0.15 | 0.000628 | 0.85 | 2 |
| 1.8 | always_seed_on_grid | nan | nan | 0.000384 | 0.95 | 0 |
| 1.9 | always_seed_on_grid | nan | nan | 0.000194 | 0.95 | 0 |
| 2 | always_seed_on_grid | nan | nan | 7.77e-05 | 0.80 | 0 |
| 2.2 | always_seed_on_grid | nan | nan | 1.61e-05 | 0.95 | 0 |
| 2.5 | always_seed_on_grid | nan | nan | 3.72e-06 | 0.80 | 0 |
| 3.14159 | always_seed_on_grid | nan | nan | 1.23e-15 | 0.00 | 0 |
| 3.5 | always_seed_on_grid | nan | nan | 1.09e-07 | 0.85 | 0 |
| 4 | always_seed_on_grid | nan | nan | 1.15e-05 | 0.85 | 0 |
| 4.71239 | nonmonotone_or_ambiguous | -1.95 | 0.05 | 0.00113 | 0.80 | 2 |
| 5 | weak_seed_to_competitor | -0.8 | 0.2 | 0.00252 | 0.15 | 1 |
| 5.5 | weak_seed_to_competitor | -0.45 | 0.15 | 0.00332 | 0.05 | 1 |
| 6.28319 | always_seed_on_grid | nan | nan | 6.99e-16 | 0.00 | 0 |

## Generated figures

- `atlas`: `/Users/b/Quandela/phasecraft/experiments/bm24_saddle_audit_p1/results/global_beta_robustness/global_beta_robustness_atlas.png`
- `boundary_reliability`: `/Users/b/Quandela/phasecraft/experiments/bm24_saddle_audit_p1/results/global_beta_robustness/global_beta_boundary_reliability.png`
- `representative_slices`: `/Users/b/Quandela/phasecraft/experiments/bm24_saddle_audit_p1/results/global_beta_robustness/global_beta_representative_slices.png`
