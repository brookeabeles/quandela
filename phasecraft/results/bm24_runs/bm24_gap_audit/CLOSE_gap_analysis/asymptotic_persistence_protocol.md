# Protocol for Testing Asymptotic Persistence of the Mean-vs-Median Gap

Goal:

> Decide whether the gap between median-runtime scaling and mean-success
> scaling persists as `n -> infinity`.

The current data establish a finite-window gap for `n=12..20` at `p=20,50`.
They do **not** establish asymptotic persistence. This protocol defines what
would move the question from "finite-window mechanism" toward "large-n
evidence."

## 1. Core Objects

For each instance:

```text
X_n = -(1/n) log2 p_succ.
```

For each `n`, compute:

```text
mean_p(n)       = mean_i p_i
median_p(n)     = median_i p_i
c_ann(n)        = -(1/n) log2 mean_p(n)
median_X(n)     = median_i X_i
mean_X(n)       = mean_i X_i
```

The pointwise gap is:

```text
Delta_n = median_X(n) - c_ann(n)
        = (1/n) log2(mean_p(n) / median_p(n)).
```

The fitted exponent gap on a window `W` is:

```text
Gap(W) = slope_n log2(mean_p(n) / median_p(n)), n in W.
```

Persistence means something like:

```text
liminf_{n -> infinity} Delta_n > 0
```

or, operationally, that `Gap(W)` and tail-window estimates stop shrinking as
we move `W` to larger `n`.

## 2. Exact Decomposition

At each `n`:

```text
Delta_n = [median_X(n) - mean_X(n)] + [mean_X(n) - c_ann(n)].
```

Call these:

```text
Skew_n   = median_X(n) - mean_X(n)
Jensen_n = mean_X(n) - c_ann(n)
```

The exponent-level slopes are:

```text
Gap(W)    = slope_n [n Delta_n]
Skew(W)   = slope_n [n Skew_n]
Jensen(W) = slope_n [n Jensen_n]
```

and

```text
Gap(W) = Skew(W) + Jensen(W).
```

This decomposition must be reported for every window. Otherwise a positive
gap can be misleading.

## 3. Exact Jensen Object: Tilted Variance

The Jensen piece has the exact identity

```text
mean_X(n) - c_ann(n)
  = n ln(2) integral_0^1 (1 - lambda) Var_lambda(X_n) dlambda.
```

Here `Var_lambda` means variance under the tilted distribution

```text
P_lambda(instance) proportional to p_succ(instance)^lambda P(instance).
```

Therefore, the right large-n object is not simply `n Var_0(X_n)`. It is the
tilted-variance integral:

```text
J_n = n ln(2) integral_0^1 (1 - lambda) Var_lambda(X_n) dlambda.
```

The ordinary variance approximation

```text
(ln 2 / 2) n Var_0(X_n)
```

is valid only if `Var_lambda(X_n)` is nearly flat over `lambda in [0,1]`.

The current N=500 audit shows it is **not** flat: `Var_1 / Var_0` is roughly
`0.36..0.54` for `p=20,50`.

## 4. What Would Count As Evidence For Persistence?

No single finite window proves persistence. But the following would be strong
evidence:

### Gate A: Stable positive tail-window gap

For increasing windows such as

```text
12..20, 16..24, 20..28, 24..32
```

or comparable feasible windows, require:

```text
Gap(W) > 0
```

with bootstrap confidence intervals excluding zero, and no monotone trend
toward zero across later windows.

Recommended evidence threshold:

```text
lower 95% bootstrap CI of Gap(W) >= 0.01
```

for at least two late windows at the same depth and seed family.

### Gate B: Stable positive Jensen piece

Require:

```text
Jensen(W) > 0
```

with confidence interval excluding zero, and check whether it accounts for
most of the total gap.

This matters because the Jensen piece is the annealed-vs-typical mechanism.

### Gate C: Stable tilted-variance integral

For each `n`, compute:

```text
J_n = n ln(2) integral_0^1 (1 - lambda) Var_lambda(X_n) dlambda.
```

Persistence is supported if `J_n` is roughly stable and positive in late `n`.

Recommended evidence threshold:

```text
mean late-window J_n >= 0.02
```

and the late-window trend slope is statistically consistent with zero.

### Gate D: Higher cumulants monitored, not assumed away

Track:

```text
n kappa_2(X_n)
n^2 kappa_3(X_n)
n^3 kappa_4(X_n)
```

and optionally higher scaled cumulants if sample size allows.

Persistence evidence is stronger if these scaled cumulants stabilize. If they
grow or oscillate strongly, the finite-window extrapolation is not credible.

### Gate E: Mean is not tail-fragile

For each `n`, report:

```text
top 1% share of sum_i p_i
top 5% share of sum_i p_i
leave-one-out shift from removing easiest observed instance
```

Recommended warning flags:

```text
top 1% share > 0.25
top 5% share > 0.50
leave-one-out shift in Delta_n > 0.005
```

If these trigger, ordinary Monte Carlo estimates of `mean_p` are likely too
fragile for asymptotic claims.

### Gate F: Multi-seed replication

Repeat the same diagnostics across independent:

```text
training seeds
evaluation seed roots
```

At minimum, use three independent seed families. Stronger evidence would use
five or more.

The current `04/05/06` diagnostics are one main training/evaluation seed
family with train sizes `train_n=12,16`; this is not enough.

## 5. What Would Count Against Persistence?

Evidence against persistence would include:

- `Gap(W)` decreases steadily as the window moves to larger `n`.
- `Delta_n` decreases roughly like `1/n` or faster.
- The tilted-variance integral `J_n` decreases toward zero.
- `Jensen(W)` loses significance while only noisy skew remains.
- Tail mass becomes so concentrated that `mean_p` is under-sampled.
- Multi-seed runs disagree in sign or show large seed-to-seed instability.

## 6. Recommended Next Runs

The current data are `N=500`, `n=12..20`, depths `p=2,5,10,20,50`,
training sizes `train_n=12,16`.

The next useful run should not add more plots at the same `n`. It should extend
the `n` range.

Recommended staged plan:

### Stage 1: Feasible extension

```text
n = 12..24
N = 500
depths = 20,50
train_n = 12,16
seeds = current seed family
```

Purpose: check whether `n=21..24` continues or breaks the current trend.

### Stage 2: Late-window confirmation

```text
n = 16..28
N = 500 or 1000
depths = 20,50
train_n = 12,16
seeds = at least 3 independent evaluation roots
```

Purpose: form non-overlapping or less-overlapping later windows.

### Stage 3: Multi-seed robustness

```text
n = 12..24 or 16..28
N = 500
depths = 20,50
train_n = 12,16
training seeds = 0,27,42 or more
evaluation roots = independent per training seed
```

Purpose: separate intrinsic behavior from angle/training/evaluation seed
effects.

### Stage 4: Tail-aware check if needed

If top-tail mass grows, increase `N` or use tail-aware estimation for
`E[p_succ]`.

## 7. Minimal Decision Table

After each new run, produce this table:

| window | train_n | depth | Gap | CI | Jensen | Skew | tilted J | Var_1/Var_0 | top1 | top5 | LOO |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|

Interpretation:

- `Gap`: slope of `log2(mean_p / median_p)`.
- `Jensen`: slope of `n(mean_X - c_ann)`.
- `Skew`: slope of `n(median_X - mean_X)`.
- `tilted J`: slope or late-window mean of the exact tilted-variance integral.
- `Var_1/Var_0`: flatness of the tilted variance curve.
- `top1/top5/LOO`: tail robustness.

## 8. Current Status

The present data satisfy:

- finite-window positive gap for `p=20,50`;
- gap mostly explained by the Jensen piece;
- exact tilted-variance integral explains the Jensen piece;
- `lambda=0` variance has the right scale but overpredicts;
- top-tail mass and leave-one-out checks are not catastrophic.

The present data do **not** satisfy:

- large enough `n` range;
- multi-seed replication of the tilted-variance diagnostic;
- evidence that the tilted-variance integral stabilizes at large `n`;
- evidence that scaled cumulants converge.

Therefore:

> The current result is a mechanism identification, not an asymptotic
> persistence result.

The next decision-making step is to run the same tilted-variance and
decomposition audit on larger `n` windows and multiple seed families.

