#!/usr/bin/env python3
"""Smoke tests for hardware p_eff measurement semantics."""

from __future__ import annotations

from pathlib import Path
import sys

_PHASECRAFT = Path(__file__).resolve().parents[2]
for _p in (_PHASECRAFT.parent, _PHASECRAFT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from phasecraft.experiments.lr_scaling.find_p_eff_hardware import (  # noqa: E402
    classify_threshold_result,
    measure_p_eff_from_rows,
)


def _row(p_noisy: float, threshold: float, state: str | None = None) -> dict:
    if state is None:
        state = classify_threshold_result(
            mean=p_noisy,
            ci_low=p_noisy,
            ci_high=p_noisy,
            threshold=threshold,
        )
    return {
        "p_noisy": p_noisy,
        "passes_mean": p_noisy > threshold,
        "threshold_state": state,
    }


def test_bracketed_crossing() -> None:
    threshold = 0.01
    rows = {
        10: _row(0.05, threshold),
        20: _row(0.03, threshold),
        30: _row(0.005, threshold),
    }
    out = measure_p_eff_from_rows([10, 20, 30], rows, threshold)
    assert out["p_peak"] == 10
    assert out["p_eff"] == 20
    assert out["p_eff_status"] == "bracketed"
    assert out["p_eff_lower_bound"] == 20
    assert out["p_eff_upper_bound"] == 30
    assert out["crossing_bracket"] == [20, 30]


def test_right_censored_crossing() -> None:
    threshold = 0.01
    rows = {
        10: _row(0.05, threshold),
        20: _row(0.04, threshold),
        30: _row(0.03, threshold),
    }
    out = measure_p_eff_from_rows([10, 20, 30], rows, threshold)
    assert out["p_eff"] == 30
    assert out["p_eff_status"] == "right_censored"
    assert out["p_eff_report"] == ">=30"
    assert out["p_eff_lower_bound"] == 30
    assert out["p_eff_upper_bound"] is None


def test_confirmed_lower_bound_stops_at_uncertainty() -> None:
    threshold = 0.01
    rows = {
        10: _row(0.05, threshold, "confirmed_above"),
        20: _row(0.03, threshold, "confirmed_above"),
        30: _row(0.005, threshold, "mean_at_or_below_uncertain"),
        40: _row(0.002, threshold, "confirmed_at_or_below"),
    }
    out = measure_p_eff_from_rows([10, 20, 30, 40], rows, threshold)
    assert out["p_eff"] == 20
    assert out["p_eff_status"] == "bracketed"
    assert out["p_eff_confirmed_lower_bound"] == 20
    assert out["p_eff_confirmed_status"] == "stopped_by_mean_at_or_below_uncertain"
    assert out["p_eff_confirmed_stop_depth"] == 30


def main() -> None:
    test_bracketed_crossing()
    test_right_censored_crossing()
    test_confirmed_lower_bound_stops_at_uncertainty()
    print("p_eff hardware measurement tests passed")


if __name__ == "__main__":
    main()
