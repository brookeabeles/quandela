#!/usr/bin/env python3
"""Finite-size scaling diagnostics for BM24 median-vs-mean gap persistence.

This script compares simple finite-size models for the pointwise gap

    Delta_n = median(X_n) - c_ann(n),   X_n = -log2(p_succ) / n,

where c_ann(n) = -log2(E[p_succ]) / n.

The goal is deliberately modest: quantify whether the available n-window can
distinguish a positive limiting gap from a decay-to-zero explanation. It should
not be read as an asymptotic proof when the n-range is short.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "phasecraft-mpl"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from plot_asymptotic_gap_certificate import (  # noqa: E402
    CellKey,
    load_cells,
    per_n_stats,
    setup_style,
)


def parse_ints(spec: str) -> list[int]:
    return [int(x) for x in str(spec).split(",") if x.strip()]


def _aicc(rss: float, n_obs: int, n_params: int) -> float:
    if n_obs <= n_params + 1:
        return float("inf")
    rss = max(float(rss), 1.0e-18)
    aic = float(n_obs * math.log(rss / n_obs) + 2 * n_params)
    return aic + float((2 * n_params * (n_params + 1)) / (n_obs - n_params - 1))


def _fit_linear_design(y: np.ndarray, design: np.ndarray, param_names: Sequence[str]) -> dict:
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    yhat = design @ beta
    resid = y - yhat
    rss = float(np.sum(resid**2))
    out = {
        "params": {name: float(beta[i]) for i, name in enumerate(param_names)},
        "rss": rss,
        "rmse": float(math.sqrt(rss / max(len(y), 1))),
        "aicc": _aicc(rss, len(y), len(param_names)),
        "yhat": [float(v) for v in yhat],
    }
    return out


def _finite_quantile(values: Sequence[float], probs: Sequence[float]) -> list[float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return [float("nan") for _ in probs]
    return [float(v) for v in np.quantile(arr, probs)]


def _fit_power_decay(ns: np.ndarray, ys: np.ndarray) -> dict:
    """Fit y = a n^{-alpha} by a one-dimensional alpha grid.

    This deliberately gives the zero-limit hypothesis a fairer chance than the
    rigid a/n model. A very small fitted alpha means "decays too slowly to
    distinguish from a constant in this window", not a real asymptotic estimate.
    """
    if np.any(ys <= 0.0):
        return {
            "params": {"power_coeff": float("nan"), "alpha": float("nan")},
            "rss": float("inf"),
            "rmse": float("inf"),
            "aicc": float("inf"),
            "yhat": [float("nan") for _ in ys],
        }
    alphas = np.linspace(0.02, 4.0, 600)
    best = None
    for alpha in alphas:
        col = ns ** (-float(alpha))
        coeff = float(np.dot(col, ys) / np.dot(col, col))
        yhat = coeff * col
        rss = float(np.sum((ys - yhat) ** 2))
        if best is None or rss < best[0]:
            best = (rss, coeff, float(alpha), yhat)
    assert best is not None
    rss, coeff, alpha, yhat = best
    return {
        "params": {"power_coeff": float(coeff), "alpha": float(alpha)},
        "rss": float(rss),
        "rmse": float(math.sqrt(rss / max(len(ys), 1))),
        "aicc": _aicc(float(rss), len(ys), 2),
        "yhat": [float(v) for v in yhat],
    }


def fit_models(ns: Sequence[int], ys: Sequence[float]) -> dict:
    n = np.asarray(ns, dtype=float)
    y = np.asarray(ys, dtype=float)
    inv_n = 1.0 / n
    models = {
        "constant_positive_limit": _fit_linear_design(
            y,
            np.ones((len(n), 1), dtype=float),
            ["gap_inf"],
        ),
        "positive_limit_plus_1_over_n": _fit_linear_design(
            y,
            np.column_stack([np.ones_like(n), inv_n]),
            ["gap_inf", "inv_n_coeff"],
        ),
        "pure_1_over_n_decay": _fit_linear_design(
            y,
            inv_n[:, None],
            ["inv_n_coeff"],
        ),
        "curved_zero_limit_decay": _fit_linear_design(
            y,
            np.column_stack([inv_n, inv_n**2]),
            ["inv_n_coeff", "inv_n2_coeff"],
        ),
        "slow_power_decay_to_zero": _fit_power_decay(n, y),
        "linear_in_n_diagnostic": _fit_linear_design(
            y,
            np.column_stack([np.ones_like(n), n]),
            ["intercept", "slope_n"],
        ),
    }
    best = min(models, key=lambda name: models[name]["aicc"])
    best_aicc = float(models[best]["aicc"])
    for row in models.values():
        row["delta_aicc"] = float(row["aicc"] - best_aicc)
    return {"best_model": best, "models": models}


def _resampled_metric(by_n: Mapping[int, dict], metric: str, rng: np.random.Generator) -> Tuple[list[int], list[float]]:
    ns = sorted(by_n)
    ys: list[float] = []
    for n in ns:
        ps = np.asarray(by_n[n]["p_succ"], dtype=float)
        xs = np.asarray(by_n[n]["X"], dtype=float)
        idx = rng.integers(0, len(ps), len(ps))
        st = per_n_stats(ps[idx], xs[idx], int(n))
        ys.append(float(st[metric]))
    return ns, ys


def bootstrap_scaling(
    by_n: Mapping[int, dict],
    *,
    metric: str,
    n_boot: int,
    seed: int,
) -> dict:
    rng = np.random.default_rng(seed)
    gap_inf = []
    pure_decay_delta = []
    curved_decay_delta = []
    power_decay_delta = []
    power_alpha = []
    trend_slope = []
    for _ in range(int(n_boot)):
        ns, ys = _resampled_metric(by_n, metric, rng)
        fits = fit_models(ns, ys)
        plus = fits["models"]["positive_limit_plus_1_over_n"]
        pure_decay = fits["models"]["pure_1_over_n_decay"]
        curved_decay = fits["models"]["curved_zero_limit_decay"]
        power_decay = fits["models"]["slow_power_decay_to_zero"]
        trend = fits["models"]["linear_in_n_diagnostic"]
        gap_inf.append(float(plus["params"]["gap_inf"]))
        pure_decay_delta.append(float(pure_decay["delta_aicc"]))
        curved_decay_delta.append(float(curved_decay["delta_aicc"]))
        power_decay_delta.append(float(power_decay["delta_aicc"]))
        power_alpha.append(float(power_decay["params"]["alpha"]))
        trend_slope.append(float(trend["params"]["slope_n"]))
    return {
        "gap_inf_plus_1_over_n_ci": _finite_quantile(gap_inf, [0.025, 0.5, 0.975]),
        "pure_decay_delta_aicc_ci": _finite_quantile(pure_decay_delta, [0.025, 0.5, 0.975]),
        "curved_decay_delta_aicc_ci": _finite_quantile(curved_decay_delta, [0.025, 0.5, 0.975]),
        "power_decay_delta_aicc_ci": _finite_quantile(power_decay_delta, [0.025, 0.5, 0.975]),
        "power_alpha_ci": _finite_quantile(power_alpha, [0.025, 0.5, 0.975]),
        "linear_trend_slope_n_ci": _finite_quantile(trend_slope, [0.025, 0.5, 0.975]),
    }


def verdict(metric_report: Mapping[str, object], *, positive_threshold: float) -> str:
    fits = metric_report["fits"]  # type: ignore[index]
    boot = metric_report["bootstrap"]  # type: ignore[index]
    plus = fits["models"]["positive_limit_plus_1_over_n"]  # type: ignore[index]
    pure_decay = fits["models"]["pure_1_over_n_decay"]  # type: ignore[index]
    curved_decay = fits["models"]["curved_zero_limit_decay"]  # type: ignore[index]
    power_decay = fits["models"]["slow_power_decay_to_zero"]  # type: ignore[index]
    gap_inf_ci = boot["gap_inf_plus_1_over_n_ci"]  # type: ignore[index]
    lower = float(gap_inf_ci[0])
    upper = float(gap_inf_ci[2])
    worst_zero_limit_delta = min(
        float(pure_decay["delta_aicc"]),
        float(curved_decay["delta_aicc"]),
        float(power_decay["delta_aicc"]),
    )
    plus_delta = float(plus["delta_aicc"])

    if lower >= positive_threshold and worst_zero_limit_delta >= 4.0:
        return "supports_positive_limit_in_window"
    if upper <= positive_threshold and plus_delta >= 4.0:
        return "supports_decay_to_zero_in_window"
    return "underidentified"


def evaluate(
    *,
    csv_path: Path,
    objective: str,
    train_ns: Sequence[int],
    depths: Sequence[int],
    n_min: int | None,
    n_max: int | None,
    n_boot: int,
    seed: int,
    positive_threshold: float,
    tail_n_min: int | None,
) -> dict:
    cells = load_cells(csv_path, objective)
    focus = {
        key: {
            n: row
            for n, row in by_n.items()
            if (n_min is None or int(n) >= n_min) and (n_max is None or int(n) <= n_max)
        }
        for key, by_n in cells.items()
        if int(key[0]) in set(train_ns) and int(key[1]) in set(depths)
    }
    focus = {key: by_n for key, by_n in focus.items() if len(by_n) >= 4}

    rows = []
    for key, by_n in sorted(focus.items()):
        ns = sorted(by_n)
        metrics: Dict[str, dict] = {}
        for metric in ("point_gap", "jensen_gap", "skew_gap", "n_kappa2_X"):
            ys = [float(by_n[n][metric]) for n in ns]
            fits = fit_models(ns, ys)
            boot = bootstrap_scaling(by_n, metric=metric, n_boot=n_boot, seed=seed + 17 * key[0] + key[1])
            metrics[metric] = {
                "ys": [float(v) for v in ys],
                "fits": fits,
                "bootstrap": boot,
                "verdict": verdict({"fits": fits, "bootstrap": boot}, positive_threshold=positive_threshold)
                if metric in {"point_gap", "jensen_gap"}
                else "monitor_only",
            }
        rows.append(
            {
                "train_n": int(key[0]),
                "depth": int(key[1]),
                "ns": [int(n) for n in ns],
                "metrics": metrics,
            }
        )

    underidentified = [
        row
        for row in rows
        if row["metrics"]["point_gap"]["verdict"] == "underidentified"
        or row["metrics"]["jensen_gap"]["verdict"] == "underidentified"
    ]
    overall = {
        "row_count": len(rows),
        "all_rows_resolved": bool(rows and not underidentified),
        "resolved_row_count": int(len(rows) - len(underidentified)),
        "underidentified_row_count": int(len(underidentified)),
        "conclusion": (
            "finite-size scaling is resolved in this window"
            if rows and not underidentified
            else "finite-size scaling is underidentified in this n-window"
        ),
    }
    return {
        "schema_version": 1,
        "kind": "bm24_gap_finite_size_scaling",
        "csv": str(csv_path),
        "objective": objective,
        "positive_threshold": float(positive_threshold),
        "tail_n_min": None if tail_n_min is None else int(tail_n_min),
        "rows": rows,
        "overall": overall,
    }


def plot_report(payload: Mapping[str, object], out_png: Path, out_pdf: Path | None) -> None:
    rows = payload["rows"]  # type: ignore[index]
    setup_style()
    fig, axes = plt.subplots(2, 2, figsize=(12.7, 8.1), sharex=False, sharey=False)
    axes = axes.ravel()
    colors = {"point_gap": "#4C72B0", "jensen_gap": "#55A868"}
    labels = {"point_gap": r"$\Delta_n$", "jensen_gap": "Jensen"}

    for ax, row in zip(axes, rows):  # type: ignore[assignment]
        ns = np.asarray(row["ns"], dtype=float)
        xfine = np.linspace(float(np.min(ns)), float(np.max(ns)), 160)
        for metric in ("point_gap", "jensen_gap"):
            report = row["metrics"][metric]
            ys = np.asarray(report["ys"], dtype=float)
            fits = report["fits"]["models"]
            color = colors[metric]
            ax.plot(ns, ys, "o", color=color, ms=5, label=labels[metric])

            plus = fits["positive_limit_plus_1_over_n"]["params"]
            y_plus = float(plus["gap_inf"]) + float(plus["inv_n_coeff"]) / xfine
            ax.plot(xfine, y_plus, "-", color=color, lw=1.4, alpha=0.82)

            decay = fits["pure_1_over_n_decay"]["params"]
            y_decay = float(decay["inv_n_coeff"]) / xfine
            ax.plot(xfine, y_decay, "--", color=color, lw=1.25, alpha=0.72)

            curved = fits["curved_zero_limit_decay"]["params"]
            y_curved = float(curved["inv_n_coeff"]) / xfine + float(curved["inv_n2_coeff"]) / (xfine**2)
            ax.plot(xfine, y_curved, ":", color=color, lw=1.35, alpha=0.78)

        ax.axhline(0.0, color="0.2", lw=0.8)
        ax.set_title(f"train n={row['train_n']}, p={row['depth']}")
        ax.set_xlabel("system size n")
        ax.set_ylabel("pointwise exponent gap")
        ax.text(
            0.03,
            0.05,
            "solid: c + a/n\n dashed: a/n\n dotted: a/n + b/n^2",
            transform=ax.transAxes,
            fontsize=8,
            va="bottom",
            bbox={"facecolor": "white", "edgecolor": "0.85", "alpha": 0.88, "pad": 3},
        )
    for ax in axes[len(rows) :]:
        ax.axis("off")
    handles, labels_seen = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_seen, loc="lower center", ncol=2, frameon=False)
    fig.suptitle("Finite-size scaling fits: positive-limit vs decay-to-zero models", y=0.985, fontsize=12)
    fig.tight_layout(rect=(0, 0.045, 1, 0.945))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    if out_pdf is not None:
        out_pdf.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def fmt(x: object, digits: int = 4) -> str:
    if isinstance(x, (float, int)):
        return f"{float(x):.{digits}f}"
    return str(x)


def write_markdown(payload: Mapping[str, object], out_path: Path) -> None:
    lines = [
        "# Finite-Size Scaling Verdict",
        "",
        "This report compares deliberately simple explanations for the observed finite-window gap:",
        "",
        "- positive limit: `Delta_n = Delta_inf + a/n`;",
        "- simple decay to zero: `Delta_n = a/n`;",
        "- curved decay to zero: `Delta_n = a/n + b/n^2`;",
        "- slow power decay to zero: `Delta_n = a n^{-alpha}`.",
        "",
        "Because the current data only cover a short window, this is a discrimination diagnostic, not a proof.",
        "",
        "## Overall",
        "",
        "```json",
        json.dumps(payload["overall"], indent=2),  # type: ignore[index]
        "```",
        "",
        "## Model Table",
        "",
        "| train_n | depth | metric | Delta_inf CI from c+a/n | best model | best zero-limit Delta AICc | power alpha CI | trend slope CI | verdict |",
        "|---:|---:|---|---:|---|---:|---:|---:|---|",
    ]
    for row in payload["rows"]:  # type: ignore[index]
        for metric in ("point_gap", "jensen_gap", "skew_gap", "n_kappa2_X"):
            report = row["metrics"][metric]
            fits = report["fits"]
            boot = report["bootstrap"]
            ci = boot["gap_inf_plus_1_over_n_ci"]
            trend_ci = boot["linear_trend_slope_n_ci"]
            zero_limit_deltas = [
                fits["models"]["pure_1_over_n_decay"]["delta_aicc"],
                fits["models"]["curved_zero_limit_decay"]["delta_aicc"],
                fits["models"]["slow_power_decay_to_zero"]["delta_aicc"],
            ]
            best_zero_delta = min(float(v) for v in zero_limit_deltas)
            alpha_ci = boot["power_alpha_ci"]
            lines.append(
                "| {tn} | {depth} | `{metric}` | [{lo}, {mid}, {hi}] | `{best}` | {zero_delta} | [{a_lo}, {a_mid}, {a_hi}] | [{tr_lo}, {tr_mid}, {tr_hi}] | `{verdict}` |".format(
                    tn=row["train_n"],
                    depth=row["depth"],
                    metric=metric,
                    lo=fmt(ci[0]),
                    mid=fmt(ci[1]),
                    hi=fmt(ci[2]),
                    best=fits["best_model"],
                    zero_delta=fmt(best_zero_delta),
                    a_lo=fmt(alpha_ci[0]),
                    a_mid=fmt(alpha_ci[1]),
                    a_hi=fmt(alpha_ci[2]),
                    tr_lo=fmt(trend_ci[0], 5),
                    tr_mid=fmt(trend_ci[1], 5),
                    tr_hi=fmt(trend_ci[2], 5),
                    verdict=report["verdict"],
                )
            )
    lines.extend(
        [
            "",
            "## Reading The Table",
            "",
            "`Delta_inf CI from c+a/n` is a bootstrap interval for the intercept of the positive-limit model.",
            "A positive interval is encouraging, but it is not decisive unless the zero-limit alternatives are also clearly worse.",
            "`best zero-limit Delta AICc` is the smallest AICc penalty among the zero-limit alternatives relative to the best model in this same short window.",
            "A small power-law `alpha` is a warning sign: it means a very slow decay can mimic a constant on the available range.",
            "",
            "The important conclusion is not the fitted intercept by itself. The important conclusion is whether the current `n` range can distinguish positive-limit behavior from decay-to-zero behavior. If the verdict is `underidentified`, the data support the finite-window mechanism but do not resolve the asymptotic question.",
            "",
        ]
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("results/bm24_runs/bm24_gap_audit/N500_n12-20_all.csv"),
    )
    parser.add_argument("--objective", default="mean_p")
    parser.add_argument("--train-ns", default="12,16")
    parser.add_argument("--depths", default="20,50")
    parser.add_argument("--n-min", type=int, default=None)
    parser.add_argument("--n-max", type=int, default=None)
    parser.add_argument(
        "--tail-n-min",
        type=int,
        default=None,
        help="Recorded metadata for consistency with the gate/mechanism diagnostics; model fits use --n-min/--n-max.",
    )
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--seed", type=int, default=911)
    parser.add_argument("--positive-threshold", type=float, default=0.01)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path(
            "results/bm24_runs/bm24_gap_audit/CLOSE_gap_analysis/"
            "finite_size_scaling_verdict.json"
        ),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path(
            "results/bm24_runs/bm24_gap_audit/CLOSE_gap_analysis/"
            "finite_size_scaling_verdict.md"
        ),
    )
    parser.add_argument(
        "--output-png",
        type=Path,
        default=Path(
            "results/bm24_runs/bm24_gap_audit/CLOSE_gap_analysis/"
            "07_finite_size_scaling_verdict.png"
        ),
    )
    parser.add_argument(
        "--output-pdf",
        type=Path,
        default=Path(
            "results/bm24_runs/bm24_gap_audit/CLOSE_gap_analysis/"
            "07_finite_size_scaling_verdict.pdf"
        ),
    )
    args = parser.parse_args()

    payload = evaluate(
        csv_path=args.csv,
        objective=args.objective,
        train_ns=parse_ints(args.train_ns),
        depths=parse_ints(args.depths),
        n_min=args.n_min,
        n_max=args.n_max,
        n_boot=int(args.bootstrap),
        seed=int(args.seed),
        positive_threshold=float(args.positive_threshold),
        tail_n_min=args.tail_n_min,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_markdown(payload, args.output_md)
    plot_report(payload, args.output_png, args.output_pdf)
    print(args.output_json)
    print(args.output_md)
    print(args.output_png)
    print(args.output_pdf)


if __name__ == "__main__":
    main()
