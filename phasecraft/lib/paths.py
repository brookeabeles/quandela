"""Canonical filesystem paths for the ``phasecraft`` package."""

from __future__ import annotations

from pathlib import Path


def phasecraft_root() -> Path:
    """Directory containing ``lib/``, ``experiments/``, ``results/``."""
    return Path(__file__).resolve().parents[1]


def repo_root() -> Path:
    """Repository root (parent of ``phasecraft/``)."""
    return phasecraft_root().parent


def bm24_runs_dir() -> Path:
    """LR / BM24 benchmark artifacts (symlink ``phasecraft/bm24_runs``)."""
    return phasecraft_root() / "bm24_runs"


def patched_gbs_path() -> Path:
    """BM24 equation implementation used by sim + Krawczyk."""
    return (
        phasecraft_root()
        / "lib"
        / "ksat"
        / "variants"
        / "generalized_binomial_sum.PATCHED.py"
    )


def bm24_qaoa_sim_cli_path() -> Path:
    """Stable CLI entrypoint (compat shim at package root)."""
    return phasecraft_root() / "bm24_qaoa_sim.py"


def train_lr_cli_path() -> Path:
    """Stable training CLI entrypoint (compat shim at package root)."""
    return phasecraft_root() / "train_lr_notebook_protocol.py"
