"""
Export a numerical p=1, q=3 saddle anchor for downstream certification.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from multivariable.krawczyk_p1_q3_8sat import export_anchor_payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Export anchor for Julia/Arb certifier.")
    parser.add_argument("--beta", type=float, default=-float(np.pi / 2))
    parser.add_argument("--gamma", type=float, default=0.14)
    parser.add_argument("--r", type=float, default=176.54)
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "multivariable" / "results" / "anchor.json",
    )
    args = parser.parse_args()

    payload = export_anchor_payload(beta=args.beta, gamma=args.gamma, r=args.r)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    print(f"anchor written to {args.out}")
    print(f"anchor residual_inf={payload['anchor_residual_inf']:.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
