"""Filename helpers and plot titles for ``phasecraft/results/bm24_runs/`` artifacts."""

from __future__ import annotations

import json
import re
import textwrap
import time
from pathlib import Path
from typing import Any, Literal, Mapping, Optional, Sequence, Tuple, Union

from phasecraft.lib.paths import (
    bm24_runs_dir,
    lr_train_optimal_angles_legacy_path,
    lr_train_optimal_angles_v2_path,
)

# Stems look like: 05-19_1645-sweep-scaling  (month-day, 24h time, kind tag; no year)
STEM_TIME_FMT = "%m-%d_%H%M"

BM24_RUNTIME_SCALING = "runtime_scaling"
BM24_SUCCESS_SCALING = "success_scaling"
# Legacy folder names (sort script still recognizes these).
BM24_SCALING_SUBDIRS = (
    BM24_RUNTIME_SCALING,
    BM24_SUCCESS_SCALING,
    "scaling_median_cost",
    "scaling_mean_success",
)

EvalAxis = Literal["runtime", "success"]
EvalAggregation = Literal["median", "mean"]
# Back-compat alias used in older call sites.
ScalingMetricKind = Literal["median_cost", "mean_success"]


def make_run_stem(kind: str, when: float | None = None) -> str:
    """Return a sortable filename stem ``MM-DD_HHMM-{kind}`` (no calendar year)."""
    loc = time.localtime(when if when is not None else time.time())
    return f"{time.strftime(STEM_TIME_FMT, loc)}-{kind}"


def default_bm24_runs_dir() -> Path:
    return bm24_runs_dir()


def normalize_eval_axis(value: Any) -> EvalAxis:
    """Map config / CLI values to ``runtime`` (cost) vs ``success`` (p_succ)."""
    if value is None:
        return "runtime"
    s = str(value).strip().lower().replace("-", "_")
    if s in ("success", "success_scaling", "p_succ", "psucc"):
        return "success"
    if s in ("runtime", "runtime_scaling", "cost", "median_cost", "scaling_median_cost"):
        return "runtime"
    if s in ("mean_success", "scaling_mean_success"):
        return "success"
    raise ValueError(f"unknown eval_axis: {value!r} (use 'runtime' or 'success')")


def normalize_eval_aggregation(value: Any) -> EvalAggregation:
    if value is None:
        return "median"
    s = str(value).strip().lower()
    if s.startswith("mean"):
        return "mean"
    if s.startswith("median"):
        return "median"
    raise ValueError(f"unknown eval_aggregation: {value!r} (use 'median' or 'mean')")


def resolve_bm24_scaling_output_dir(
    *,
    eval_axis: EvalAxis | str = "runtime",
    legacy_mean_success: bool | None = None,
    base: Path | None = None,
) -> Path:
    """Flat ``bm24_runs/`` (``eval_axis`` is for titles/labels only, not subfolders)."""
    del eval_axis, legacy_mean_success
    root = Path(base) if base is not None else bm24_runs_dir()
    root.mkdir(parents=True, exist_ok=True)
    return root


def infer_eval_axis(
    *,
    path: Path | None = None,
    settings: Mapping[str, Any] | None = None,
    trace: Sequence[Mapping[str, Any]] | None = None,
) -> EvalAxis:
    """Infer plot/eval axis for folder layout (explicit config wins)."""
    if settings:
        if settings.get("eval_axis") is not None:
            return normalize_eval_axis(settings["eval_axis"])
        sm = settings.get("scaling_metric")
        if sm in ("median_cost", "runtime", "runtime_scaling"):
            return "runtime"
        if sm in ("mean_success", "success", "success_scaling"):
            return "success"
        if settings.get("legacy_objective") is True:
            return "success"

    rows = list(trace or ())
    if rows and rows[0].get("median_runtime_per_n"):
        return "runtime"
    if rows:
        obj = str(rows[0].get("training_objective", ""))
        if "inv-p" in obj or "log-inv-p" in obj:
            return "runtime"

    if path is not None:
        name = path.name.lower()
        parent = path.parent.name.lower()
        if parent in (BM24_SUCCESS_SCALING, "scaling_mean_success"):
            return "success"
        if parent in (BM24_RUNTIME_SCALING, "scaling_median_cost"):
            return "runtime"
        if "scaling-vs-depth" in name and "efficient" not in name and "runtime" not in name:
            return "success"
        if "efficient-scaling" in name or "sweep-scaling" in name or "runtime" in name:
            return "runtime"

    return "runtime"


def infer_eval_aggregation(
    *,
    settings: Mapping[str, Any] | None = None,
    trace: Sequence[Mapping[str, Any]] | None = None,
) -> EvalAggregation:
    if settings and settings.get("eval_aggregation") is not None:
        return normalize_eval_aggregation(settings["eval_aggregation"])
    if settings and settings.get("eval_stat") is not None:
        return normalize_eval_aggregation(settings["eval_stat"])
    return "median"


def scaling_ylabel(
    eval_axis: EvalAxis | str,
    eval_aggregation: EvalAggregation | str = "median",
) -> str:
    axis = normalize_eval_axis(eval_axis)
    agg = normalize_eval_aggregation(eval_aggregation)
    if axis == "runtime":
        if agg == "mean":
            return r"$\log_2$ slope of mean cost vs $n$"
        return r"$\log_2$ slope of median cost vs $n$"
    if agg == "mean":
        return r"$\log_2$ slope of mean $p_{\mathrm{succ}}$ vs $n$"
    return r"$\log_2$ slope of median $p_{\mathrm{succ}}$ vs $n$"


def scaling_lr_legend_label(
    eval_axis: EvalAxis | str,
    eval_aggregation: EvalAggregation | str = "median",
) -> str:
    axis = normalize_eval_axis(eval_axis)
    agg = normalize_eval_aggregation(eval_aggregation)
    if axis == "runtime":
        return f"LR-QAOA ({agg} cost)"
    return f"LR-QAOA ({agg} $p_{{\\mathrm{{succ}}}}$)"


def scaling_plot_headline(
    base: str,
    eval_axis: EvalAxis | str,
    eval_aggregation: EvalAggregation | str = "median",
) -> str:
    axis = normalize_eval_axis(eval_axis)
    agg = normalize_eval_aggregation(eval_aggregation)
    return f"{base} · {axis} · {agg}"


def infer_scaling_metric_kind(
    *,
    path: Path | None = None,
    settings: Mapping[str, Any] | None = None,
    trace: Sequence[Mapping[str, Any]] | None = None,
) -> ScalingMetricKind:
    """Deprecated: use ``infer_eval_axis`` + ``infer_eval_aggregation``."""
    if infer_eval_axis(path=path, settings=settings, trace=trace) == "success":
        return "mean_success"
    return "median_cost"


def _fmt_r(r: Any) -> str:
    if isinstance(r, float):
        return f"{r:g}"
    return str(r)


def format_benchmark_title(
    settings: Mapping[str, Any],
    *,
    headline: str = "LR scaling vs depth",
    depth_min: Optional[int] = None,
    depth_max: Optional[int] = None,
    depths: Optional[Sequence[int]] = None,
    max_param_width: int = 72,
) -> str:
    """Matplotlib title (headline + wrapped params) for benchmark / scaling plots."""
    s = dict(settings)
    param_parts: list[str] = []

    k, r = s.get("k"), s.get("r")
    if k is not None and r is not None:
        param_parts.append(f"k={k}, r={_fmt_r(r)}")

    n_min, n_max = s.get("n_min"), s.get("n_max")
    if n_min is not None and n_max is not None:
        param_parts.append(f"n∈[{n_min},{n_max}]")

    test_size = s.get("test_size")
    if test_size is not None:
        param_parts.append(f"inst={test_size}")

    train_size = s.get("train_size")
    if train_size is not None:
        param_parts.append(f"trn={train_size}")

    seed = s.get("seed", s.get("base_seed"))
    if seed is not None:
        param_parts.append(f"seed={seed}")

    p_lo = depth_min if depth_min is not None else s.get("depth_min")
    p_hi = depth_max if depth_max is not None else s.get("depth_max")
    if depths:
        p_lo = min(int(d) for d in depths)
        p_hi = max(int(d) for d in depths)
    elif s.get("depth") is not None:
        p_lo = p_hi = int(s["depth"])
    if p_lo is not None and p_hi is not None:
        param_parts.append(f"p∈[{p_lo},{p_hi}]" if p_lo != p_hi else f"p={p_lo}")

    train_n = s.get("train_n")
    if train_n is not None:
        param_parts.append(f"train_n={train_n}")

    if s.get("eval_axis") is not None:
        param_parts.append(f"axis={normalize_eval_axis(s['eval_axis'])}")
    if s.get("eval_aggregation") is not None:
        param_parts.append(f"agg={normalize_eval_aggregation(s['eval_aggregation'])}")
    elif s.get("eval_stat") is not None:
        param_parts.append(f"agg={normalize_eval_aggregation(s['eval_stat'])}")

    if s.get("require_sat") is True:
        param_parts.append("SAT-only")

    return _join_title_lines(headline, param_parts, max_width=max_param_width)


def _join_title_lines(
    headline: str,
    param_parts: Sequence[str],
    *,
    max_width: int = 88,
) -> str:
    if not param_parts:
        return headline
    line2 = " · ".join(param_parts)
    if len(line2) > max_width:
        line2 = "\n".join(
            textwrap.wrap(line2, width=max_width, break_long_words=False, break_on_hyphens=False)
        )
    return f"{headline}\n{line2}"


def apply_matplotlib_title(
    ax,
    title: str,
    *,
    fig=None,
    fontsize: int = 9,
    pad: float = 8.0,
    top: float = 0.88,
) -> None:
    """Set a wrapped axes title and reserve figure headroom so it is not clipped."""
    ax.set_title(title, fontsize=fontsize, loc="left", pad=pad)
    parent = fig if fig is not None else ax.figure
    parent.subplots_adjust(top=top)
    parent.tight_layout(rect=[0, 0, 1, top - 0.02])


def apply_figure_suptitle(
    fig,
    title: str,
    *,
    fontsize: int = 9,
    top: float = 0.90,
) -> None:
    """Set a wrapped figure suptitle with room above subplots."""
    fig.suptitle(title, fontsize=fontsize, y=0.995, va="top")
    fig.subplots_adjust(top=top)
    fig.tight_layout(rect=[0, 0, 1, top - 0.02])


def save_matplotlib_figure(fig, output_path: Path, *, dpi: int = 150) -> Path:
    """Save with padding so titles are not clipped at the figure edge."""
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", pad_inches=0.45)
    return output_path


def trace_depth_training_failed(row: Mapping[str, Any]) -> bool:
    """True when benchmark eval did not accept angles trained for this depth."""
    if "training_failed" in row:
        return bool(row["training_failed"])
    return bool(row.get("used_previous_angles")) or bool(row.get("eval_rejected"))


def trace_failed_depths(trace: Sequence[Mapping[str, Any]]) -> Tuple[List[int], List[float]]:
    """Return (depths, lr_log2_slopes) for depths where training failed."""
    depths: List[int] = []
    slopes: List[float] = []
    for row in trace:
        if trace_depth_training_failed(row):
            depths.append(int(row["depth"]))
            slopes.append(float(row["lr_log2_slope"]))
    return depths, slopes


def load_scaling_trace_json(path: Path) -> Tuple[list[dict], dict]:
    """Load a depth-scaling trace from JSON (list or ``{config, trace}`` wrapper)."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data, {}
    if isinstance(data, dict):
        trace = data.get("trace", data.get("exponent_trace", []))
        settings = dict(data.get("config", {}))
        if data.get("run_stem"):
            settings.setdefault("run_stem", data["run_stem"])
        return list(trace), settings
    raise ValueError(f"unsupported scaling JSON: {path}")


def scaling_output_path(
    stem: str,
    ext: str,
    *,
    eval_axis: EvalAxis | str = "runtime",
    eval_aggregation: EvalAggregation | str | None = None,
    legacy_mean_success: bool | None = None,
    base: Path | None = None,
) -> Path:
    """``bm24_runs/{runtime_scaling|success_scaling}/{stem}.{ext}``."""
    del eval_aggregation  # path depends only on axis
    out_dir = resolve_bm24_scaling_output_dir(
        eval_axis=eval_axis,
        legacy_mean_success=legacy_mean_success,
        base=base,
    )
    return out_dir / f"{stem}.{ext.lstrip('.')}"


def _is_scaling_artifact_stem(stem: str) -> bool:
    s = stem.replace(".partial", "")
    if "bench" in s and "scaling" not in s:
        return False
    return any(
        tag in s
        for tag in ("scaling", "efficient", "sweep-scaling", "runtime-train")
    )


def iter_bm24_scaling_artifacts(root: Path | None = None) -> list[Path]:
    """JSON/PNG depth-scaling files under ``bm24_runs/`` (root and legacy subdirs)."""
    root = Path(root) if root is not None else bm24_runs_dir()
    out: list[Path] = []

    def _maybe_add(p: Path) -> None:
        if p.suffix.lower() not in (".json", ".png"):
            return
        stem = p.stem.replace(".partial", "")
        if not _is_scaling_artifact_stem(stem):
            return
        out.append(p)

    for p in sorted(root.iterdir()):
        if p.is_file():
            _maybe_add(p)
    for sub in BM24_SCALING_SUBDIRS:
        sub_path = root / sub
        if sub_path.is_dir():
            for p in sorted(sub_path.iterdir()):
                if p.is_file():
                    _maybe_add(p)
    return out


def plot_scaling_vs_depth(
    trace: list[dict],
    output_path: Path,
    *,
    settings: Mapping[str, Any] | None = None,
    headline: str = "LR depth sweep",
    k: int | None = None,
    r: float | None = None,
    n_min: int | None = None,
    n_max: int | None = None,
    test_size: int | None = None,
    seed: int | None = None,
    depth_min: int | None = None,
    depth_max: int | None = None,
    train_n: int | None = None,
    train_size: int | None = None,
    annotate_first_win: bool = False,
) -> Path:
    """Plot log2 scaling slopes vs QAOA depth (runtime or legacy success metric)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError(
            "plot_scaling_vs_depth requires matplotlib. "
            "Install with: pip install matplotlib"
        ) from e

    if not trace:
        raise ValueError("empty scaling trace")

    depths = [int(row["depth"]) for row in trace]
    lr = [float(row["lr_log2_slope"]) for row in trace]
    ws = float(trace[0]["walksat_log2_slope"])
    lm = float(trace[0]["walksatlm_log2_slope"])
    title_settings = dict(settings or {})
    eval_axis = infer_eval_axis(settings=title_settings, trace=trace)
    eval_agg = infer_eval_aggregation(settings=title_settings, trace=trace)
    fail_depths, fail_slopes = trace_failed_depths(trace)
    win_depths = [
        int(row["depth"])
        for row in trace
        if row.get("lr_beats_both_scaling")
        or (
            row.get("lr_beats_walksat_scaling")
            and row.get("lr_beats_walksatlm_scaling")
        )
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    lr_label = scaling_lr_legend_label(eval_axis, eval_agg)
    ax.plot(depths, lr, "o-", color="C0", linewidth=2, markersize=7, label=lr_label)
    if fail_depths:
        ax.scatter(
            fail_depths,
            fail_slopes,
            marker="x",
            s=100,
            c="C3",
            linewidths=2.5,
            zorder=6,
            label="train failed (prior angles)",
        )
    ax.axhline(ws, color="C1", linestyle="--", linewidth=1.5, label=f"WalkSAT ({ws:.3f})")
    ax.axhline(lm, color="C2", linestyle="--", linewidth=1.5, label=f"WalkSATlm ({lm:.3f})")

    if annotate_first_win and win_depths:
        p0 = min(win_depths)
        ax.axvline(p0, color="0.4", linestyle=":", linewidth=1.2, alpha=0.8)
        idx = depths.index(p0)
        ax.scatter(
            [p0], [lr[idx]], s=120, facecolors="none", edgecolors="C0", linewidths=2, zorder=5
        )
        ax.annotate(
            f"first win p={p0}",
            xy=(p0, lr[idx]),
            xytext=(8, 12),
            textcoords="offset points",
            fontsize=9,
        )

    ax.set_xlabel("QAOA depth p")
    ax.set_ylabel(scaling_ylabel(eval_axis, eval_agg))
    title_settings.setdefault("eval_axis", eval_axis)
    title_settings.setdefault("eval_aggregation", eval_agg)
    for key, val in (
        ("k", k), ("r", r), ("n_min", n_min), ("n_max", n_max),
        ("test_size", test_size), ("seed", seed),
        ("depth_min", depth_min), ("depth_max", depth_max),
        ("train_n", train_n), ("train_size", train_size),
    ):
        if val is not None:
            title_settings[key] = val
    apply_matplotlib_title(
        ax,
        format_benchmark_title(title_settings, headline=headline, depths=depths),
        fig=fig,
        fontsize=9,
    )
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(depths)

    out = save_matplotlib_figure(fig, Path(output_path))
    plt.close(fig)
    return out


def sort_bm24_scaling_artifacts(
    root: Path | None = None,
    *,
    replot: bool = True,
    dry_run: bool = False,
) -> list[tuple[Path, Path]]:
    """
    Move depth-scaling JSON/PNG into ``runtime_scaling`` or ``success_scaling``
    using ``eval_axis`` from each trace config (inferred if missing).

    Optionally regenerate PNGs with axis/aggregation in the title.
    """
    root = Path(root) if root is not None else bm24_runs_dir()
    moves: list[tuple[Path, Path]] = []
    json_paths: list[Path] = []
    for p in iter_bm24_scaling_artifacts(root):
        if p.suffix.lower() == ".json":
            json_paths.append(p)

    for p in json_paths:
        trace, settings = load_scaling_trace_json(p)
        axis = infer_eval_axis(path=p, settings=settings, trace=trace)
        agg = infer_eval_aggregation(settings=settings, trace=trace)
        settings = {**settings, "eval_axis": axis, "eval_aggregation": agg}
        dest_dir = resolve_bm24_scaling_output_dir(eval_axis=axis, base=root)
        dest_json = dest_dir / p.name
        if dest_json.resolve() != p.resolve():
            moves.append((p, dest_json))
            if not dry_run:
                dest_json.parent.mkdir(parents=True, exist_ok=True)
                p.rename(dest_json)
            p = dest_json
        png = p.with_suffix(".png")
        if png.is_file() and png.parent.resolve() != dest_dir.resolve():
            dest_png = dest_dir / png.name
            moves.append((png, dest_png))
            if not dry_run:
                png.rename(dest_png)
        if replot and not dry_run:
            base_hl = (
                "Efficient LR sweep"
                if "efficient" in p.stem
                else "LR depth sweep"
            )
            plot_scaling_vs_depth(
                trace,
                p.with_suffix(".png"),
                settings=settings,
                headline=scaling_plot_headline(base_hl, axis, agg),
            )

    if not dry_run:
        for sub in BM24_SCALING_SUBDIRS:
            sub_dir = root / sub
            if sub_dir.is_dir() and not any(sub_dir.iterdir()):
                sub_dir.rmdir()
    return moves


def flatten_bm24_scaling_artifacts(
    root: Path | None = None,
    *,
    replot: bool = True,
    dry_run: bool = False,
) -> list[tuple[Path, Path]]:
    """Move scaling artifacts from subfolders back to flat ``bm24_runs/``."""
    root = Path(root) if root is not None else bm24_runs_dir()
    moves: list[tuple[Path, Path]] = []
    for sub in BM24_SCALING_SUBDIRS:
        sub_dir = root / sub
        if not sub_dir.is_dir():
            continue
        for p in sorted(sub_dir.iterdir()):
            if not p.is_file() or p.suffix.lower() not in (".json", ".png"):
                continue
            dest = root / p.name
            if dest.resolve() == p.resolve():
                continue
            if dest.exists() and p.suffix.lower() == ".json":
                continue
            if dest.exists() and p.suffix.lower() == ".png":
                if not dry_run:
                    p.unlink()
                continue
            moves.append((p, dest))
            if not dry_run:
                p.rename(dest)
    if not dry_run:
        for sub in BM24_SCALING_SUBDIRS:
            sub_dir = root / sub
            if sub_dir.is_dir() and not any(sub_dir.iterdir()):
                sub_dir.rmdir()
    return moves


# Default organize script flattens; sort is opt-in for subfolder layouts.
migrate_bm24_scaling_artifacts = flatten_bm24_scaling_artifacts


# --------------------------------------------------------------------------- #
# LR optimal-angle logs (legacy vs v2/v3)                                     #
# --------------------------------------------------------------------------- #

AngleLogKind = Literal["legacy", "v2v3"]

_ANGLE_LOG_SPLIT_RE = re.compile(r"(?:\n-{10,}\n|\n(?=# \d{4}-\d{2}-\d{2}))")
_ANGLE_TS_HASH_RE = re.compile(
    r"(?:^#\s*(\d{4}-\d{2}-\d{2}T[\d:.]+)|^timestamp:\s*(\d{4}-\d{2}-\d{2}T[\d:.]+))",
    re.MULTILINE,
)


def split_angle_log_text(text: str) -> list[str]:
    """Split a combined angle log into non-empty record blocks."""
    blocks = _ANGLE_LOG_SPLIT_RE.split(text)
    return [b.strip() for b in blocks if b.strip()]


def classify_angle_log_block(block: str) -> AngleLogKind:
    """Classify one record as legacy (median/mean @ train_n) or v2/v3 slope training."""
    if re.search(r'"legacy_objective"\s*:\s*true', block):
        return "legacy"
    if re.search(r'"objective"\s*:\s*"(?:legacy-)', block):
        return "legacy"
    if re.search(r"legacy-median|legacy-mean", block):
        return "legacy"
    if re.search(r'"objective"\s*:\s*"(?:v[23][^"]*)"', block):
        return "v2v3"
    if re.search(r'"proxy_n_values"\s*:\s*\[', block):
        return "v2v3"
    if re.search(r"proxy_n_values", block) and "null" not in block.split("proxy_n_values", 1)[-1][:20]:
        return "v2v3"
    if block.lstrip().startswith("# 20"):
        return "v2v3"
    if "betas_bm24:" in block or re.search(r"^timestamp:\s*", block, re.MULTILINE):
        return "legacy"
    if re.search(r"^settings\s*=", block, re.MULTILINE) and "proxy_n_values" in block:
        return "v2v3"
    return "legacy"


def angle_log_block_sort_key(block: str) -> str:
    """Sort key from embedded ISO timestamp (fallback: block text)."""
    m = _ANGLE_TS_HASH_RE.search(block)
    if m:
        return m.group(1) or m.group(2) or ""
    return block[:80]


def partition_angle_log_blocks(blocks: Sequence[str]) -> dict[AngleLogKind, list[str]]:
    out: dict[AngleLogKind, list[str]] = {"legacy": [], "v2v3": []}
    for block in blocks:
        out[classify_angle_log_block(block)].append(block)
    for kind in out:
        out[kind].sort(key=angle_log_block_sort_key)
    return out


def write_angle_log_blocks(path: Path, blocks: Sequence[str], *, trailing_newline: bool = True) -> None:
    """Write blocks joined with blank lines (preserves each block's internal format)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "\n\n".join(blocks)
    if trailing_newline and body:
        body += "\n"
    path.write_text(body, encoding="utf-8")


def split_mixed_angle_log(
    mixed_path: Path,
    *,
    legacy_path: Path | None = None,
    v2_path: Path | None = None,
    archive_path: Path | None = None,
    merge_existing: bool = True,
    compat_symlink: bool = True,
    dry_run: bool = False,
) -> dict[str, Any]:
    """
    Split a mixed ``lr_train_optimal_angles.txt`` into legacy + v2/v3 logs.

    Archives the original file, merges into existing legacy/v2 logs when
    ``merge_existing`` is true, and optionally replaces the mixed path with a
    symlink to the v2 log for backward-compatible ``--angle-log`` paths.
    """
    mixed_path = Path(mixed_path).expanduser().resolve()
    legacy_path = Path(legacy_path or lr_train_optimal_angles_legacy_path()).expanduser()
    v2_path = Path(v2_path or lr_train_optimal_angles_v2_path()).expanduser()
    archive_path = Path(
        archive_path or mixed_path.with_name("lr_train_optimal_angles_mixed_archive.txt")
    ).expanduser()

    if not mixed_path.is_file():
        raise FileNotFoundError(mixed_path)

    text = mixed_path.read_text(encoding="utf-8")
    if mixed_path.is_symlink():
        raise ValueError(f"refusing to split symlink: {mixed_path}")

    blocks = split_angle_log_text(text)
    parts = partition_angle_log_blocks(blocks)

    def _merge(kind: AngleLogKind, dest: Path) -> list[str]:
        merged = list(parts[kind])
        if merge_existing and dest.is_file() and dest.resolve() != mixed_path.resolve():
            existing = split_angle_log_text(dest.read_text(encoding="utf-8"))
            seen = {b.strip() for b in merged}
            for b in existing:
                bs = b.strip()
                if bs and bs not in seen:
                    merged.append(bs)
                    seen.add(bs)
            merged.sort(key=angle_log_block_sort_key)
        return merged

    legacy_blocks = _merge("legacy", legacy_path)
    v2_blocks = _merge("v2v3", v2_path)

    report = {
        "mixed": str(mixed_path),
        "archive": str(archive_path),
        "legacy_path": str(legacy_path),
        "v2_path": str(v2_path),
        "legacy_blocks": len(legacy_blocks),
        "v2_blocks": len(v2_blocks),
        "from_mixed_legacy": len(parts["legacy"]),
        "from_mixed_v2v3": len(parts["v2v3"]),
        "dry_run": dry_run,
        "compat_symlink": compat_symlink,
    }

    if dry_run:
        return report

    if archive_path.exists() and archive_path.resolve() != mixed_path.resolve():
        raise FileExistsError(f"archive already exists: {archive_path}")
    mixed_path.rename(archive_path)

    write_angle_log_blocks(legacy_path, legacy_blocks)
    write_angle_log_blocks(v2_path, v2_blocks)

    if compat_symlink:
        compat = mixed_path
        if compat.exists() or compat.is_symlink():
            compat.unlink()
        compat.symlink_to(v2_path.name)

    return report
