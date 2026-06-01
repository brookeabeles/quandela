"""
Patched-logic Krawczyk runner.

This wrapper reuses the original Krawczyk pipeline but patches in the
`generalized_binomial_sum.PATCHED.py` equations so results are directly
comparable with legacy outputs.
"""

from __future__ import annotations

import numpy as np
import importlib.util
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import phasecraft.krawczyk_p1_roots as base

_PATCHED_PATH = Path(__file__).resolve().with_name("generalized_binomial_sum.PATCHED.py")
_PATCHED_SPEC = importlib.util.spec_from_file_location(
    "phasecraft._generalized_binomial_sum_patched_fix",
    _PATCHED_PATH,
    submodule_search_locations=[str(_PATCHED_PATH.parent)],
)
_PATCHED_MOD = importlib.util.module_from_spec(_PATCHED_SPEC)
assert _PATCHED_SPEC is not None and _PATCHED_SPEC.loader is not None
sys.modules[_PATCHED_SPEC.name] = _PATCHED_MOD
_PATCHED_SPEC.loader.exec_module(_PATCHED_MOD)
B_patched = _PATCHED_MOD.B
alpha_sos_patched = _PATCHED_MOD.parent_function_alpha_sum_sos
s_sos_patched = _PATCHED_MOD.parent_function_s_sum_sos


class SaddleSystemPatched(base.SaddleSystem):
    @classmethod
    def build(cls, q: int, r: float, betas: np.ndarray, gammas: np.ndarray) -> "SaddleSystemPatched":
        p = int(len(betas))
        if len(gammas) != p:
            raise ValueError("betas and gammas must have same length p")
        all_s = np.arange(2 ** (2 * p + 1))
        b = 0.5 * B_patched(np.array(betas, dtype=float), all_s)
        # Patched BM24 (A34) sign convention.
        prod_elts = np.concatenate(
            (np.exp(-0.5j * np.array(gammas)) - 1.0, [(-1.0)], np.exp(0.5j * np.array(gammas)[::-1]) - 1.0)
        )
        c = r * np.prod(
            [prod_elts[j] * ((all_s >> j) & 1) + 1.0 * ((~all_s >> j) & 1) for j in range(2 * p + 1)],
            axis=0,
        )
        c_root = (-c) ** (1.0 / (2**q))
        return cls(
            q=q,
            r=r,
            p=p,
            betas=np.array(betas, dtype=float),
            gammas=np.array(gammas, dtype=float),
            b=b,
            c_root=c_root,
        )


# Monkey-patch the base module symbols so all downstream routines
# (interval certifier / discovery / main CLI) use patched equations.
base.SaddleSystem = SaddleSystemPatched
base.B = B_patched
base.parent_function_alpha_sum_sos = alpha_sos_patched
base.parent_function_s_sum_sos = s_sos_patched


if __name__ == "__main__":
    base.main()
