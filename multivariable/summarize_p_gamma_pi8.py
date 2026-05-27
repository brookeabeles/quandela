"""
Summarize p123 gamma-pi/8 sweep into flat CSV + compact JSON.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path


def main() -> None:
    src = Path("multivariable/results/p123_gamma_pi8_sweep.json")
    out_csv = Path("multivariable/results/p123_gamma_pi8_summary.csv")
    out_json = Path("multivariable/results/p123_gamma_pi8_summary.json")

    with src.open("r", encoding="utf-8") as f:
        data = json.load(f)

    rows: list[dict] = []
    for entry in data.get("results", []):
        p = int(entry["p"])
        gamma = float(entry["gamma"])
        roots = entry.get("roots", [])
        for k, root in enumerate(roots):
            rows.append(
                {
                    "p": p,
                    "gamma": gamma,
                    "root_id": k,
                    "n_roots_at_point": int(entry.get("n_roots_found", len(roots))),
                    "residual_inf": float(root["residual_inf"]),
                    "re_phi": float(root["re_phi"]),
                    "rho_Jg": float(root["rho_Jg"]),
                    "n_near_zero_components_full": int(root["n_near_zero_components_full"]),
                    "n_large_components_full": int(root["n_large_components_full"]),
                }
            )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "p",
                "gamma",
                "root_id",
                "n_roots_at_point",
                "residual_inf",
                "re_phi",
                "rho_Jg",
                "n_near_zero_components_full",
                "n_large_components_full",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    # Compact grouped summary for quick inspection.
    grouped: dict[str, dict] = {}
    for r in rows:
        key = f"p{r['p']}_g{r['gamma']:.12g}"
        grouped[key] = grouped.get(
            key,
            {
                "p": r["p"],
                "gamma": r["gamma"],
                "n_roots_at_point": r["n_roots_at_point"],
                "roots": [],
            },
        )
        grouped[key]["roots"].append(
            {
                "root_id": r["root_id"],
                "residual_inf": r["residual_inf"],
                "re_phi": r["re_phi"],
                "rho_Jg": r["rho_Jg"],
                "n_near_zero_components_full": r["n_near_zero_components_full"],
                "n_large_components_full": r["n_large_components_full"],
            }
        )

    with out_json.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "source": str(src),
                "n_points_total": len(data.get("results", [])),
                "n_roots_total": len(rows),
                "points": list(grouped.values()),
            },
            f,
            indent=2,
        )

    print(f"wrote {out_csv}")
    print(f"wrote {out_json}")
    print(f"points={len(data.get('results', []))}, roots={len(rows)}")


if __name__ == "__main__":
    main()
