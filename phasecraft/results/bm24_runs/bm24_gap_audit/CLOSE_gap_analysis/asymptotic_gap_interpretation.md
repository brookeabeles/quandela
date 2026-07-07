# Interpretation of the Mean-vs-Median Gap Diagnostic

This note explains the claim:

> At n=12..20, for p=20 and p=50, the gap is real and mostly explained by
> finite-window spread in log-success probabilities. The variance term explains
> much of the Jensen piece, but higher cumulants are visible, so asymptotic
> persistence remains open.

The short version is:

> The data show a real finite-size difference between the typical runtime
> exponent and the exponent inferred from mean success probability. The
> difference is caused by instance-to-instance variation in success
> probabilities: easy instances raise the mean success probability above the
> median success probability. But the current n-window is too small to prove
> that this difference survives as n -> infinity.

## What Is Being Compared?

For each random SAT instance at size `n`, define

```text
X_n = -(1/n) log2 p_succ.
```

Here:

- `p_succ` is the QAOA success probability on that instance.
- Smaller `X_n` means an easier instance.
- Larger `X_n` means a harder instance.

The two exponents being compared are:

```text
c_typ  = exponent from median runtime, i.e. typical instances
c_ann  = exponent from mean success probability, i.e. annealed/mean behavior
```

Mean success probability and median runtime answer different questions.

The median runtime asks:

```text
How hard is a typical instance?
```

The mean success probability asks:

```text
What is the average success probability over instances?
```

Those differ when the instance distribution has a spread. Easy instances can
increase the mean success probability even if they are not typical.

## The Exact Finite-Window Identity

For a fixed depth and training choice, define

```text
mean_p(n)   = mean over instances of p_succ
median_p(n) = median over instances of p_succ
```

Then the finite-window exponent gap is the slope in `n` of

```text
log2(mean_p(n) / median_p(n)).
```

This is the clean finite-window fact. If this quantity grows with `n`, then
the exponent from mean success and the exponent from median runtime differ on
that window.

The key audit figure `01_spread_vs_n.png` already showed this growth. The new
figure `04_asymptotic_gap_certificate.png` plots the same object in panel (a)
for the depths where the gap is clearest.

## What "The Gap Is Real" Means

In the N=500 audit over `n=12..20`, the measured finite-window slope gaps are:

| train_n | depth p | measured gap slope | 95% bootstrap CI |
|---:|---:|---:|---:|
| 12 | 20 | 0.0498 | [0.0306, 0.0639] |
| 12 | 50 | 0.0734 | [0.0572, 0.0907] |
| 16 | 20 | 0.0428 | [0.0269, 0.0590] |
| 16 | 50 | 0.0698 | [0.0527, 0.0836] |

These intervals are well above zero. That is what "the gap is real" means in
this finite window: it is not just plotting noise or a single ambiguous slope.

It does not mean the gap has been proven asymptotically.

## Decomposing the Gap

At each `n`,

```text
median(X_n) - c_ann(n)
  = [median(X_n) - mean(X_n)] + [mean(X_n) - c_ann(n)].
```

The two brackets have different meanings.

### 1. The Skew Piece

```text
median(X_n) - mean(X_n)
```

This measures how the median of the instance hardness differs from the mean
of the instance hardness.

If the distribution of `X_n` has a hard-instance tail, then `mean(X_n)` can be
larger than `median(X_n)`, making this pointwise term negative.

### 2. The Jensen / Annealed Piece

```text
mean(X_n) - c_ann(n)
```

This is nonnegative by Jensen's inequality, because

```text
c_ann(n) = - (1/n) log2 E[2^{-n X_n}].
```

This term is the main reason the mean success probability exponent can be
smaller than the typical runtime exponent. It is caused by the exponential
tilt toward easier instances.

In the finite-window slope decomposition:

| train_n | depth p | total gap slope | Jensen slope | skew slope |
|---:|---:|---:|---:|---:|
| 12 | 20 | 0.0498 | 0.0427 | 0.0071 |
| 12 | 50 | 0.0734 | 0.0685 | 0.0050 |
| 16 | 20 | 0.0428 | 0.0398 | 0.0031 |
| 16 | 50 | 0.0698 | 0.0632 | 0.0066 |

So in this window, the gap is mostly a Jensen/annealed effect.

That is what "mostly explained by finite-window spread in log-success
probabilities" means.

## Why The Variance Term Appears

Let `L = ln 2`. The cumulant expansion gives

```text
mean(X_n) - c_ann(n)
  = (L/2) n kappa_2(X_n)
    - (L^2/6) n^2 kappa_3(X_n)
    + (L^3/24) n^3 kappa_4(X_n)
    - ...
```

The first term is the variance term:

```text
(ln 2 / 2) n Var(X_n).
```

If the distribution of `X_n` were approximately Gaussian in the relevant
tilted regime, the higher cumulants would be negligible and the variance term
would explain the Jensen piece.

That is the origin of the simple formula:

```text
Jensen piece ~= (ln 2 / 2) n Var(X_n).
```

But this is not an exact identity. It is a second-order approximation.

## A Better Exact View: Tilted Variance

There is a cleaner exact identity for the Jensen piece. Let

```text
p = p_succ
Y = ln p
```

and define the tilted instance distribution

```text
P_lambda(instance) proportional to p(instance)^lambda P(instance),
```

where `lambda` runs from `0` to `1`.

Then

```text
ln E[p] - E[ln p]
  = integral_0^1 (1 - lambda) Var_lambda(ln p) dlambda.
```

This identity is useful because the left-hand side is exactly the Jensen gap
between the log of the mean success probability and the mean log success
probability. The right-hand side says this gap is controlled by the variance
of log-success under a continuously tilted distribution.

In terms of

```text
X_n = -(1/n) log2 p,
```

the same identity becomes

```text
mean(X_n) - c_ann(n)
  = n ln(2) integral_0^1 (1 - lambda) Var_lambda(X_n) dlambda.
```

This is exact, up to the empirical finite-sample estimates.

The variance-only approximation used in the diagnostic figure is what happens
if we replace every tilted variance by the ordinary, untilted variance:

```text
Var_lambda(X_n) approximately Var_0(X_n).
```

Then

```text
mean(X_n) - c_ann(n)
  approximately n ln(2) Var_0(X_n) integral_0^1 (1 - lambda) dlambda
  = (ln 2 / 2) n Var(X_n).
```

So the variance term is not a new theorem. It is the simplest approximation to
an exact tilted-variance integral.

This gives a sharper diagnostic than just plotting `n Var(X_n)`: estimate

```text
Var_lambda(X_n),  lambda in [0,1],
```

and check whether it stays roughly flat. If it is flat, the variance-only
formula is a good local description. If it changes substantially, then higher
cumulants or tails are already important in the finite window.

## Follow-Up: The Tilted-Variance Diagnostic

The follow-up figure

```text
06_tilted_variance_diagnostics.png
```

does this test directly.

For each depth/training-size pair, it estimates

```text
n Var_lambda(X_n)
```

on a grid of `lambda` values from `0` to `1`, averaged over the tail window
`n>=16`.

The result is important: the tilted variance is not flat. It decreases as
`lambda` increases. Since positive `lambda` gives more weight to easier
instances, this means the easy-instance tilted distribution has smaller
variance than the original distribution.

That explains why the simple variance-only term overpredicts the measured
Jensen piece.

| train_n | depth p | measured Jensen slope | tilted-integral slope | lambda=0 variance slope | Var_1 / Var_0 |
|---:|---:|---:|---:|---:|---:|
| 12 | 20 | 0.0427 | 0.0427 | 0.0570 | 0.522 |
| 12 | 50 | 0.0685 | 0.0685 | 0.1000 | 0.358 |
| 16 | 20 | 0.0398 | 0.0398 | 0.0525 | 0.539 |
| 16 | 50 | 0.0632 | 0.0632 | 0.0898 | 0.381 |

The tilted-integral slope matches the measured Jensen slope because the
tilted-variance identity is exact. The `lambda=0` variance slope is the crude
second-order approximation, and it is too large because the variance curve
drops along the tilted path.

So the updated interpretation is:

> The Jensen piece is exactly explained by an integral of tilted variances.
> The ordinary variance at lambda=0 gives the right order of magnitude, but it
> overpredicts because the tilted variance decreases as the measure shifts
> toward easier instances.

This is stronger than the cumulant-only explanation. It shows directly where
the variance-only approximation succeeds and where it fails.

## What The Current Data Say About The Variance Term

The variance-only term explains a substantial part of the Jensen piece, but it
does not fully close the story.

| train_n | depth p | Jensen slope | variance-only slope |
|---:|---:|---:|---:|
| 12 | 20 | 0.0427 | 0.0570 |
| 12 | 50 | 0.0685 | 0.1000 |
| 16 | 20 | 0.0398 | 0.0525 |
| 16 | 50 | 0.0632 | 0.0898 |

The variance-only term is the right order of magnitude, but it tends to
overpredict the measured Jensen slope. This is exactly why we added
`05_cumulant_tail_diagnostics.png`.

## Why Higher Cumulants Matter

If the scaled cumulants behave like

```text
kappa_j(X_n) ~ a_j / n^{j-1},
```

then

```text
n kappa_2(X_n),
n^2 kappa_3(X_n),
n^3 kappa_4(X_n),
...
```

can each contribute an O(1) amount to `mean(X_n) - c_ann(n)`.

That means the variance term is not automatically dominant in the large-n
limit. Higher cumulants can remain important.

The current data show that higher cumulants are visible, especially at p=50.
This is why the revised figure no longer calls the variance plot an
"asymptotic certificate." It is a finite-window diagnostic.

## Tail Robustness

One concern is that `mean_p(n)` might be dominated by a few very easy
instances. If that happened, N=500 would be too small and the mean-success
estimate would be fragile.

The current audit checks this in two ways.

### Top-tail mass

In the tail window `n>=16`, the mean success mass is approximately:

| train_n | depth p | top 1% share | top 5% share |
|---:|---:|---:|---:|
| 12 | 20 | 0.055 | 0.201 |
| 12 | 50 | 0.050 | 0.198 |
| 16 | 20 | 0.054 | 0.198 |
| 16 | 50 | 0.049 | 0.196 |

So the top 1% do not dominate the sample mean. The top 5% carry about 20% of
the mass, which is noticeable but not catastrophic.

### Leave-one-out sensitivity

Removing the easiest observed instance changes the pointwise gap by less than
about

```text
0.001
```

in this audit.

This means the current N=500 estimate is not obviously a one-instance fluke.
However, this does not rule out unseen rarer easy instances at larger `n`.

## Finite-Size Scaling Implication

The follow-up report
`finite_size_scaling_verdict.md` compares positive-limit fits against
zero-limit fits for the pointwise gap

```text
Delta_n = median(X_n) - c_ann(n).
```

The tempting positive-limit model is

```text
Delta_n = Delta_inf + a/n.
```

If we compare this only against the rigid decay model `Delta_n = a/n`, the
current data look strongly positive. That is not a robust asymptotic argument,
because zero-limit behavior need not look like exactly `1/n` over `n=12..20`.

The stronger diagnostic also allows

```text
Delta_n = a/n + b/n^2
Delta_n = a n^{-alpha}
```

These are still simple decay-to-zero models, but they are flexible enough to
imitate a flat or slowly increasing curve on a short finite window. Once these
models are included, all four p=20,50 rows become underidentified. The fitted
positive intercepts remain encouraging finite-window evidence, but they do not
rule out slow or curved decay to zero.

This is the cleanest implication:

> The current data show a robust finite-window gap and identify its mechanism.
> They do not distinguish asymptotic persistence from very slow/curved decay.

The operational follow-up is written in
`asymptotic_gap_followup_design.md`. It forecasts how much the positive-limit
and best curved zero-limit fits separate at `n=21..28`, gives rough
sample-size requirements from current bootstrap noise, and lists staged
`evaluate.py` commands. The key design point is that `n=21..22` is only a
sanity probe; the first useful discrimination window is `n=23..24`, preferably
on GPU.

## What The Result Does And Does Not Prove

### It does prove, for this finite window

- The mean-vs-median exponent gap is real for p=20 and p=50.
- The gap is mostly a Jensen/annealed spread effect.
- The exact tilted-variance integral explains the Jensen piece.
- The simple `lambda=0` variance term captures the right scale but
  overpredicts because the tilted variance decreases with `lambda`.
- The mean is not dominated by a single easiest observed instance.

### It does not prove

- The gap persists as `n -> infinity`.
- `n Var(X_n)` converges to a positive constant.
- Higher cumulants vanish asymptotically.
- The distribution is asymptotically Gaussian.
- The same result holds for all training/evaluation seeds.

## Why Asymptotic Persistence Remains Open

To prove or strongly support asymptotic persistence, we would need evidence
that, as `n` grows:

1. The finite-window gap does not decay toward zero.
2. The Jensen piece remains positive.
3. The relevant scaled cumulants stabilize.
4. The mean success probability is not increasingly dominated by unseen rare
   easy instances.

The current data only cover `n=12..20`, less than a factor of two in system
size. That is not enough to extract a limit.

So the correct interpretation is:

> The data localize the mechanism of the observed finite-window gap, but they
> do not settle the large-n convergence question.

## Recommended Claim

The strongest defensible statement is:

> In the N=500 audit over n=12..20, for QAOA depths p=20 and p=50, the
> median-runtime exponent is measurably larger than the exponent inferred from
> mean success probability. The gap is mostly a Jensen/annealed effect caused
> by finite-window spread in X_n = -(1/n)log2 p_succ. The Jensen piece is
> exactly represented by a tilted-variance integral. The ordinary variance
> approximation explains the scale of this term but overpredicts it because
> the tilted variance decreases along lambda in [0,1]. Higher scaled cumulants
> are visible; therefore the data do not prove that the gap persists as
> n -> infinity.

## Next Steps

1. Repeat the full N=500 diagnostic for independent training and evaluation
   seeds.
2. Extend the `n` range beyond 20.
3. Track the scaled cumulants

```text
n kappa_2(X_n), n^2 kappa_3(X_n), n^3 kappa_4(X_n), ...
```

4. Estimate and plot the tilted variance curve

```text
Var_lambda(X_n),  lambda in [0,1].
```

This directly tests whether the variance-only approximation is valid or
whether higher cumulants/tails are active.

5. Plot the finite-window cumulant contributions to the Jensen piece.
6. Monitor top-tail mass and leave-one-out sensitivity as `n` grows.
7. If tail concentration increases, use tail-aware or importance-sampled
   estimates of `E[p_succ]`.
