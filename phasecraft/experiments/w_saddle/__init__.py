"""
W-saddle Krawczyk pipeline (corrected beta=-pi/2, p=1, q=1 slice).

  python -m phasecraft.w_saddle --help

Modules:
  core      — interval certification, WSaddleSystem
  workflow  — gamma continuation, competitor sweeps, branch resolve
  cli       — command-line entry (also ``python -m phasecraft.w_saddle``)

Artifacts: ``phasecraft/w_saddle/runs/``
"""

from phasecraft.w_saddle.core import (
    WSaddleSystem,
    discover_w_roots,
    krawczyk_certify_w_root,
    krawczyk_certify_with_escalation,
    solve_w_from_init,
)
from phasecraft.w_saddle.workflow import continue_negative_gamma_branch

__all__ = [
    "WSaddleSystem",
    "continue_negative_gamma_branch",
    "discover_w_roots",
    "krawczyk_certify_w_root",
    "krawczyk_certify_with_escalation",
    "solve_w_from_init",
]
