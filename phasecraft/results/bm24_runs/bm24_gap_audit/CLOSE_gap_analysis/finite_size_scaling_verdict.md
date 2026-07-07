# Finite-Size Scaling Verdict

This report compares deliberately simple explanations for the observed finite-window gap:

- positive limit: `Delta_n = Delta_inf + a/n`;
- simple decay to zero: `Delta_n = a/n`;
- curved decay to zero: `Delta_n = a/n + b/n^2`;
- slow power decay to zero: `Delta_n = a n^{-alpha}`.

Because the current data only cover a short window, this is a discrimination diagnostic, not a proof.

## Overall

```json
{
  "row_count": 4,
  "all_rows_resolved": false,
  "resolved_row_count": 0,
  "underidentified_row_count": 4,
  "conclusion": "finite-size scaling is underidentified in this n-window"
}
```

## Model Table

| train_n | depth | metric | Delta_inf CI from c+a/n | best model | best zero-limit Delta AICc | power alpha CI | trend slope CI | verdict |
|---:|---:|---|---:|---|---:|---:|---:|---|
| 12 | 20 | `point_gap` | [0.0334, 0.0496, 0.0647] | `positive_limit_plus_1_over_n` | 1.7302 | [0.0200, 0.0200, 0.0200] | [0.00005, 0.00110, 0.00210] | `underidentified` |
| 12 | 20 | `jensen_gap` | [0.0349, 0.0439, 0.0525] | `constant_positive_limit` | 0.8655 | [0.0200, 0.0200, 0.1595] | [-0.00044, 0.00011, 0.00071] | `underidentified` |
| 12 | 20 | `skew_gap` | [-0.0094, 0.0058, 0.0181] | `pure_1_over_n_decay` | 0.0000 | [nan, nan, nan] | [0.00008, 0.00101, 0.00176] | `monitor_only` |
| 12 | 20 | `n_kappa2_X` | [0.1303, 0.1660, 0.2055] | `constant_positive_limit` | 1.7503 | [0.0200, 0.0200, 0.1064] | [-0.00113, 0.00115, 0.00366] | `monitor_only` |
| 12 | 50 | `point_gap` | [0.0617, 0.0773, 0.0928] | `curved_zero_limit_decay` | 0.0000 | [0.0200, 0.0200, 0.0200] | [0.00194, 0.00301, 0.00403] | `underidentified` |
| 12 | 50 | `jensen_gap` | [0.0589, 0.0701, 0.0805] | `curved_zero_limit_decay` | 0.0000 | [0.0200, 0.0200, 0.0200] | [0.00051, 0.00125, 0.00192] | `underidentified` |
| 12 | 50 | `skew_gap` | [-0.0090, 0.0074, 0.0218] | `pure_1_over_n_decay` | 0.0000 | [nan, nan, nan] | [0.00063, 0.00175, 0.00266] | `monitor_only` |
| 12 | 50 | `n_kappa2_X` | [0.2286, 0.2905, 0.3480] | `positive_limit_plus_1_over_n` | 1.2430 | [0.0200, 0.0200, 0.0200] | [0.00173, 0.00567, 0.00959] | `monitor_only` |
| 16 | 20 | `point_gap` | [0.0311, 0.0460, 0.0613] | `positive_limit_plus_1_over_n` | 0.1580 | [0.0200, 0.0200, 0.0200] | [0.00002, 0.00093, 0.00189] | `underidentified` |
| 16 | 20 | `jensen_gap` | [0.0323, 0.0407, 0.0498] | `constant_positive_limit` | 1.5799 | [0.0200, 0.0200, 0.1861] | [-0.00051, 0.00003, 0.00057] | `underidentified` |
| 16 | 20 | `skew_gap` | [-0.0095, 0.0050, 0.0177] | `pure_1_over_n_decay` | 0.0000 | [nan, nan, nan] | [0.00001, 0.00092, 0.00168] | `monitor_only` |
| 16 | 20 | `n_kappa2_X` | [0.1187, 0.1545, 0.1878] | `constant_positive_limit` | 2.5587 | [0.0200, 0.0200, 0.1462] | [-0.00139, 0.00092, 0.00305] | `monitor_only` |
| 16 | 50 | `point_gap` | [0.0561, 0.0710, 0.0864] | `positive_limit_plus_1_over_n` | 0.2259 | [0.0200, 0.0200, 0.0200] | [0.00181, 0.00270, 0.00365] | `underidentified` |
| 16 | 50 | `jensen_gap` | [0.0529, 0.0645, 0.0752] | `curved_zero_limit_decay` | 0.0000 | [0.0200, 0.0200, 0.0200] | [0.00034, 0.00105, 0.00172] | `underidentified` |
| 16 | 50 | `skew_gap` | [-0.0086, 0.0065, 0.0228] | `pure_1_over_n_decay` | 0.0000 | [nan, nan, nan] | [0.00061, 0.00164, 0.00263] | `monitor_only` |
| 16 | 50 | `n_kappa2_X` | [0.1989, 0.2620, 0.3207] | `positive_limit_plus_1_over_n` | 0.8175 | [0.0200, 0.0200, 0.0200] | [0.00101, 0.00460, 0.00849] | `monitor_only` |

## Reading The Table

`Delta_inf CI from c+a/n` is a bootstrap interval for the intercept of the positive-limit model.
A positive interval is encouraging, but it is not decisive unless the zero-limit alternatives are also clearly worse.
`best zero-limit Delta AICc` is the smallest AICc penalty among the zero-limit alternatives relative to the best model in this same short window.
A small power-law `alpha` is a warning sign: it means a very slow decay can mimic a constant on the available range.

The important conclusion is not the fitted intercept by itself. The important conclusion is whether the current `n` range can distinguish positive-limit behavior from decay-to-zero behavior. If the verdict is `underidentified`, the data support the finite-window mechanism but do not resolve the asymptotic question.
