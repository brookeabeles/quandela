#!/usr/bin/env python3
"""Run the BM24 gap-persistence diagnostic suite on one combined CSV."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str]) -> None:
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/bm24_runs/bm24_gap_audit/CLOSE_gap_analysis/followup_reanalysis"),
    )
    parser.add_argument("--train-ns", default="12,16")
    parser.add_argument("--depths", default="20,50")
    parser.add_argument("--tail-n-min", type=int, default=20)
    parser.add_argument("--window-width", type=int, default=9)
    parser.add_argument("--bootstrap", type=int, default=500)
    args = parser.parse_args()

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    py = sys.executable

    run(
        [
            py,
            "experiments/lr_scaling/plot_asymptotic_gap_certificate.py",
            "--csv",
            str(args.csv),
            "--train-ns",
            args.train_ns,
            "--focus-depths",
            args.depths,
            "--tail-n-min",
            str(args.tail_n_min),
            "--bootstrap",
            str(args.bootstrap),
            "--output-png",
            str(out / "04_asymptotic_gap_certificate.png"),
            "--output-pdf",
            str(out / "04_asymptotic_gap_certificate.pdf"),
            "--diagnostics-png",
            str(out / "05_cumulant_tail_diagnostics.png"),
            "--diagnostics-pdf",
            str(out / "05_cumulant_tail_diagnostics.pdf"),
            "--tilted-png",
            str(out / "06_tilted_variance_diagnostics.png"),
            "--tilted-pdf",
            str(out / "06_tilted_variance_diagnostics.pdf"),
            "--summary",
            str(out / "04_asymptotic_gap_certificate_summary.json"),
            "--theory-note",
            str(out / "04_asymptotic_gap_theory.md"),
        ]
    )
    run(
        [
            py,
            "experiments/lr_scaling/evaluate_asymptotic_persistence_gates.py",
            "--csv",
            str(args.csv),
            "--train-ns",
            args.train_ns,
            "--depths",
            args.depths,
            "--tail-n-min",
            str(args.tail_n_min),
            "--window-width",
            str(args.window_width),
            "--bootstrap",
            str(args.bootstrap),
            "--output-json",
            str(out / "asymptotic_persistence_gate_evaluation.json"),
            "--output-md",
            str(out / "asymptotic_persistence_gate_evaluation.md"),
        ]
    )
    run(
        [
            py,
            "experiments/lr_scaling/fit_asymptotic_gap_scaling.py",
            "--csv",
            str(args.csv),
            "--train-ns",
            args.train_ns,
            "--depths",
            args.depths,
            "--tail-n-min",
            str(args.tail_n_min),
            "--bootstrap",
            str(args.bootstrap),
            "--output-json",
            str(out / "finite_size_scaling_verdict.json"),
            "--output-md",
            str(out / "finite_size_scaling_verdict.md"),
            "--output-png",
            str(out / "07_finite_size_scaling_verdict.png"),
            "--output-pdf",
            str(out / "07_finite_size_scaling_verdict.pdf"),
        ]
    )
    print(out)


if __name__ == "__main__":
    main()
