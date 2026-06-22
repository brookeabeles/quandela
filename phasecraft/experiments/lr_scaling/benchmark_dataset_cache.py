"""Disk cache for SAT benchmark clauses (LR_QAOA_benchmark.ipynb setup)."""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence

from phasecraft.lib.sim.bm24_qaoa_sim import build_h_diagonal

Dataset = Dict[int, List[dict]]
BuildFn = Callable[[], Dataset]


def benchmark_dataset_cache_path(
    run_dir: Path,
    *,
    k: int,
    r: float,
    base_seed: int,
    test_size: int,
    n_values: Sequence[int],
) -> Path:
    """Sidecar path beside run checkpoints; does not touch JSON trace files."""
    n_lo, n_hi = int(min(n_values)), int(max(n_values))
    name = (
        f"benchmark_clauses_seed{int(base_seed)}_k{int(k)}"
        f"_r{float(r):g}_te{int(test_size)}_n{n_lo}-{n_hi}.json.gz"
    )
    return Path(run_dir) / name


def _cache_meta(
    *,
    k: int,
    r: float,
    base_seed: int,
    test_size: int,
    n_values: Sequence[int],
) -> dict[str, Any]:
    return {
        "version": 1,
        "k": int(k),
        "r": float(r),
        "seed": int(base_seed),
        "test_size": int(test_size),
        "n_values": [int(n) for n in n_values],
    }


def _meta_matches(payload: dict[str, Any], meta: dict[str, Any]) -> bool:
    for key in ("version", "k", "r", "seed", "test_size", "n_values"):
        if payload.get(key) != meta.get(key):
            return False
    return True


def _clauses_from_dataset(dataset: Dataset) -> dict[str, list]:
    return {str(int(n)): [inst["clauses"] for inst in instances] for n, instances in dataset.items()}


def _serialize_clauses(clauses_by_n: dict[str, list]) -> dict[str, list]:
    return {
        n_key: [
            [
                [[int(var), bool(neg)] for var, neg in clause]
                for clause in formula_clauses
            ]
            for formula_clauses in instances
        ]
        for n_key, instances in clauses_by_n.items()
    }


def _deserialize_clauses(serial: dict[str, list]) -> dict[str, list]:
    return {
        n_key: [
            [
                [(int(var), bool(neg)) for var, neg in clause]
                for clause in formula_clauses
            ]
            for formula_clauses in instances
        ]
        for n_key, instances in serial.items()
    }


def save_benchmark_dataset_clauses(path: Path, *, meta: dict[str, Any], dataset: Dataset) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        **meta,
        "clauses_by_n": _serialize_clauses(_clauses_from_dataset(dataset)),
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        json.dump(payload, f)
    tmp.replace(path)
    return path


def load_benchmark_dataset_clauses(path: Path, *, meta: dict[str, Any]) -> Dataset | None:
    path = Path(path)
    if not path.is_file():
        return None
    with gzip.open(path, "rt", encoding="utf-8") as f:
        payload = json.load(f)
    if not _meta_matches(payload, meta):
        return None
    clauses_by_n = _deserialize_clauses(payload["clauses_by_n"])
    dataset: Dataset = {}
    for n_key, instances in clauses_by_n.items():
        n = int(n_key)
        dataset[n] = [
            {"clauses": clauses, "h_diag": build_h_diagonal(clauses, n)}
            for clauses in instances
        ]
    return dataset


def load_or_build_benchmark_dataset(
    *,
    run_dir: Path,
    n_values: Sequence[int],
    k: int,
    r: float,
    test_size: int,
    base_seed: int,
    build_fn: BuildFn,
    use_cache: bool = True,
) -> Dataset:
    """
    Load benchmark clauses from a sidecar ``.json.gz`` in ``run_dir``, or build
    and save them. Rebuilds ``h_diag`` in memory on load (cheap vs generation).
    """
    meta = _cache_meta(
        k=k, r=r, base_seed=base_seed, test_size=test_size, n_values=n_values
    )
    cache_path = benchmark_dataset_cache_path(
        run_dir,
        k=k,
        r=r,
        base_seed=base_seed,
        test_size=test_size,
        n_values=n_values,
    )

    if use_cache:
        loaded = load_benchmark_dataset_clauses(cache_path, meta=meta)
        if loaded is not None:
            print(f"Loaded benchmark clauses from {cache_path.name} (rebuilt H_diag)")
            return loaded

    print("Building SAT benchmark dataset (cached H_diag)...")
    dataset = build_fn()
    if use_cache:
        save_benchmark_dataset_clauses(cache_path, meta=meta, dataset=dataset)
        print(f"Saved benchmark clauses -> {cache_path.name}")
    return dataset


def _main() -> None:
    """Pre-build clause cache from a run checkpoint JSON (does not modify the JSON)."""
    import argparse

    from phasecraft.lib.sim.bm24_qaoa_sim import generate_benchmark_dataset

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint_json", type=Path, help="e.g. bm24_runs/.../scaling-tn14.json")
    args = p.parse_args()

    payload = json.loads(Path(args.checkpoint_json).read_text(encoding="utf-8"))
    cfg = payload["config"]
    n_values = list(range(int(cfg["n_min"]), int(cfg["n_max"]) + 1))
    run_dir = Path(args.checkpoint_json).resolve().parent

    def build() -> Dataset:
        raw = generate_benchmark_dataset(
            n_values,
            int(cfg["k"]),
            float(cfg["r"]),
            int(cfg["test_size"]),
            int(cfg["seed"]),
            require_sat=True,
        )
        return {
            int(n): [
                {"clauses": inst["clauses"], "h_diag": build_h_diagonal(inst["clauses"], int(n))}
                for inst in raw[int(n)]
            ]
            for n in raw
        }

    load_or_build_benchmark_dataset(
        run_dir=run_dir,
        n_values=n_values,
        k=int(cfg["k"]),
        r=float(cfg["r"]),
        test_size=int(cfg["test_size"]),
        base_seed=int(cfg["seed"]),
        build_fn=build,
        use_cache=True,
    )


if __name__ == "__main__":
    _main()
