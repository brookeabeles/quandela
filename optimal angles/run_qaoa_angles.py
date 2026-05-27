#!/usr/bin/env python3
"""CLI entry point for BM24 staged optimization workflow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from datetime import datetime, timezone

from qaoa_angles_core import (
    THRESHOLDS,
    diagnostic_p1_crosscheck,
    optimize_ksat_general,
    optimize_ksat_p1,
    optimize_toy,
    scan_ksat_p1_landscape,
    validate_ksat_p1_scaling,
    validate_toy_point,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run toy/p1/general QAOA-angle tasks.")
    p.add_argument(
        "--mode",
        choices=["toy-opt", "toy-validate", "p1-opt", "p1-scan", "p1-validate", "p1-diagnostic", "general-opt"],
        required=True,
    )
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--r", type=float, default=None, help="If omitted, uses threshold for k when available.")
    p.add_argument("--p", type=int, default=1, help="QAOA depth for general-opt.")
    p.add_argument("--beta", type=float, default=-1.57079632679)
    p.add_argument("--gamma", type=float, default=3.14159265359)
    p.add_argument("--n-eval", type=int, default=30)
    p.add_argument("--n-beta", type=int, default=80)
    p.add_argument("--n-gamma", type=int, default=80)
    p.add_argument("--out", type=Path, default=Path("qaoa_angles_result.json"))
    p.add_argument(
        "--history",
        type=Path,
        default=Path("optimal angles/qaoa_angle_optimization_history.jsonl"),
        help="Append-only JSONL history file; one line per run.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    r = args.r if args.r is not None else THRESHOLDS.get(args.k)
    if r is None and args.mode != "toy-opt" and args.mode != "toy-validate":
        raise ValueError("Please provide --r for this k (no threshold constant found).")

    if args.mode == "toy-opt":
        payload = optimize_toy()
    elif args.mode == "toy-validate":
        payload = validate_toy_point(args.beta, args.gamma)
    elif args.mode == "p1-opt":
        payload = optimize_ksat_p1(args.k, float(r), n_eval=args.n_eval)
    elif args.mode == "p1-validate":
        payload = validate_ksat_p1_scaling(args.k, float(r), args.beta, args.gamma)
    elif args.mode == "p1-scan":
        b, g, land = scan_ksat_p1_landscape(args.k, float(r), n_eval=args.n_eval, n_beta=args.n_beta, n_gamma=args.n_gamma)
        payload = {"betas": b.tolist(), "gammas": g.tolist(), "landscape": land.tolist()}
    elif args.mode == "p1-diagnostic":
        payload = diagnostic_p1_crosscheck(args.k, float(r), n_eval=args.n_eval)
    else:
        payload = optimize_ksat_general(args.k, args.p, r=float(r))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # Append one machine-readable line per run for tracking best parameters over time.
    history_record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "k": args.k,
        "p": args.p,
        "r": r,
        "n_eval": args.n_eval,
        "n_beta": args.n_beta,
        "n_gamma": args.n_gamma,
        "beta_input": args.beta,
        "gamma_input": args.gamma,
        "out_file": str(args.out),
        "result": payload,
    }
    args.history.parent.mkdir(parents=True, exist_ok=True)
    with args.history.open("a", encoding="utf-8") as f:
        f.write(json.dumps(history_record, separators=(",", ":")) + "\n")

    print(json.dumps(payload, indent=2))
    print(f"Saved -> {args.out}")
    print(f"Appended -> {args.history}")


if __name__ == "__main__":
    main()
