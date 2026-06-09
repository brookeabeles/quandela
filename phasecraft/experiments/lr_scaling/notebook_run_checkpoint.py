"""Checkpoint helpers for experiments/lr_scaling/notebooks/LR_QAOA_benchmark.ipynb."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from bm24_run_io import make_run_stem, resolve_bm24_run_output_paths


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def write_checkpoint(run_paths: dict[str, Path], payload: dict[str, Any]) -> Path:
    for key in ("partial_json", "json"):
        atomic_write_json(run_paths[key], payload)
    return run_paths["json"]


def find_checkpoint(output_dir: Path, run_stem: str) -> Path | None:
    output_dir = Path(output_dir)
    candidates = list(output_dir.rglob(f"{run_stem}.partial.json"))
    candidates += list(output_dir.rglob(f"{run_stem}.json"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def build_payload(
    *,
    run_stem: str,
    cfg: dict[str, Any],
    trace: list[dict[str, Any]],
    setup: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "run_stem": run_stem,
        "config": {k: (list(v) if isinstance(v, range) else v) for k, v in cfg.items()},
        "trace": trace,
    }
    if setup:
        payload["setup"] = setup
    return payload


def load_or_start_run(cfg: dict[str, Any]) -> tuple[str, dict[str, Path], list, dict]:
    output_dir = Path(cfg["output_dir"])
    recover_path = cfg.get("recover_from")
    if recover_path is not None:
        recover_path = Path(recover_path)
    elif cfg.get("auto_resume", True) and cfg.get("run_stem"):
        recover_path = find_checkpoint(output_dir, str(cfg["run_stem"]))

    if recover_path is not None and recover_path.is_file():
        with open(recover_path, encoding="utf-8") as f:
            saved = json.load(f)
        trace = list(saved.get("trace", []))
        saved_setup = dict(saved.get("setup") or {})
        run_stem = str(
            saved.get("run_stem")
            or cfg.get("run_stem")
            or recover_path.stem.replace(".partial", "")
        )
        run_dir = recover_path.parent
        print(f"Resumed from {recover_path.name} ({len(trace)} depth(s) done)")
    else:
        trace = []
        saved_setup = {}
        run_stem = str(cfg.get("run_stem") or make_run_stem("efficient-scaling"))
        run_dir = None

    run_paths = resolve_bm24_run_output_paths(output_dir, run_stem, run_dir=run_dir)
    return run_stem, run_paths, trace, saved_setup


def save_setup_checkpoint(
    run_paths: dict[str, Path],
    *,
    run_stem: str,
    cfg: dict[str, Any],
    trace: list[dict[str, Any]],
    classical: dict,
    ws_slope: float,
    lm_slope: float,
) -> Path:
    setup = {
        "classical": {
            algo: {str(n): float(v) for n, v in per_n.items()}
            for algo, per_n in classical.items()
        },
        "ws_slope": float(ws_slope),
        "lm_slope": float(lm_slope),
    }
    payload = build_payload(run_stem=run_stem, cfg=cfg, trace=trace, setup=setup)
    return write_checkpoint(run_paths, payload)
