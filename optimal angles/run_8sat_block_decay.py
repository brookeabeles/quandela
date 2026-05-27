"""
Multivariable-facing entrypoint for 8-SAT block-decay experiments.

This preserves the existing runner logic in the repository root while
defaulting outputs to multivariable/figures/8sat_block_decay.
"""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)
    os.environ.setdefault(
        "QUANDELA_8SAT_OUT_DIR",
        str(repo_root / "multivariable" / "figures" / "8sat_block_decay"),
    )
    runpy.run_path(str(repo_root / "run_8sat_block_decay.py"), run_name="__main__")


if __name__ == "__main__":
    main()
