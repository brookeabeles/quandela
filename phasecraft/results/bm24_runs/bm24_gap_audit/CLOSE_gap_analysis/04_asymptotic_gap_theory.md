# Finite-Window Annealed-vs-Typical Gap Diagnostic

This note is generated with `plot_asymptotic_gap_certificate.py`.

## What Is Proven Exactly

For each finite evaluation window, define

```text
X_n = -(1/n) log2 p_succ(instance).
```

The empirical slope gap is the slope of

```text
log2(mean p_succ / median p_succ).
```

Equivalently, at point level,

```text
median(X_n) - c_ann(n)
  = [median(X_n) - mean(X_n)] + [mean(X_n) - c_ann(n)].
```

The second bracket is nonnegative by Jensen and is the piece targeted by
the second-order/log-normal approximation.

## What Is Only A Local Approximation

The cumulant expansion gives

```text
mean(X_n) - c_ann(n)
  = (ln2/2)n kappa_2(X_n)
    - (ln2)^2 n^2 kappa_3(X_n)/6
    + (ln2)^3 n^3 kappa_4(X_n)/24 - ...
```

If kappa_j(X_n) scales like n^{-(j-1)}, every displayed scaled cumulant
can contribute O(1). Therefore the variance-only formula is a finite-window
diagnostic, not an asymptotic theorem.

## Current N=500 Window

| train_n | depth | gap slope | Jensen slope | skew slope | second-order slope | top1 mass | top5 mass | LOO shift |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 12 | 20 | 0.0498 | 0.0427 | 0.0071 | 0.0570 | 0.055 | 0.201 | 0.0009 |
| 12 | 50 | 0.0734 | 0.0685 | 0.0050 | 0.1000 | 0.050 | 0.198 | 0.0007 |
| 16 | 20 | 0.0428 | 0.0398 | 0.0031 | 0.0525 | 0.054 | 0.198 | 0.0009 |
| 16 | 50 | 0.0698 | 0.0632 | 0.0066 | 0.0898 | 0.049 | 0.196 | 0.0007 |

## Defensible Claim

At p=20 and p=50 over n=12..20, the mean-vs-median exponent gap is
well resolved and is mostly a Jensen/annealed effect. The variance-only
term gives a useful local description of that Jensen component, but higher
scaled cumulants are visible and must be tracked before making any n -> inf
claim.

## Next Measurements

1. Repeat this diagnostic across independent training/evaluation seeds.
2. Extend n substantially beyond 20.
3. Track scaled cumulants n kappa2, n^2 kappa3, n^3 kappa4, ... directly.
4. Use tail-aware estimates of E[p_succ] if top-tail concentration grows.
