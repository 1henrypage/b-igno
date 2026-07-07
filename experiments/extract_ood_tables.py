"""Extract OOD comparison table data from aggregated JSONs.

Reads the 4 *_aggregated.json files from the OOD structured results,
extracts cross_instance mean +/- std for in_domain_data_only and ood_data_only,
and prints LaTeX-formatted table rows matching tab:ood column order:
Accuracy, CRPS, Cov. 95%, CI width, mean_std.

Usage:
    python extract_ood_tables.py --base-dir experiments/results/structured
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


PROBLEMS = [
    "darcy_continuous",
    "darcy_piecewise",
    "eit",
    "burgers",
]

PROBLEM_LABELS = {
    "darcy_continuous": "Darcy continuous",
    "darcy_piecewise": "Darcy piecewise",
    "eit": "EIT",
    "burgers": "Burgers",
}

CONDITIONS = ["in_domain_data_only", "ood_data_only"]
CONDITION_LABELS = {"in_domain_data_only": "In-domain", "ood_data_only": "OOD"}

METRICS = ["a_err", "crps_a", "coverage_95", "ci_width", "mean_std"]
METRIC_HEADERS = ["Accuracy", "CRPS", "Cov. 95%", "CI width", "σ̄"]


def sig_figs(x: float, n: int = 2) -> str:
    if x == 0:
        return "0"
    if math.isnan(x):
        return "NaN"
    magnitude = math.floor(math.log10(abs(x)))
    if magnitude >= n - 1:
        return f"{round(x, -(magnitude - n + 1)):.0f}"
    else:
        decimals = n - 1 - magnitude
        return f"{x:.{decimals}f}"


def fmt_cell(mean: float, std: float) -> str:
    if math.isnan(mean):
        return "n/a"
    m_str = sig_figs(mean)
    s_str = sig_figs(std)
    return f"${m_str} \\pm {s_str}$"


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract OOD table data")
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("results/structured"),
        help="Base directory containing ood/ subdir with *_aggregated.json files",
    )
    args = parser.parse_args()

    ood_dir = args.base_dir / "ood"

    print("=" * 70)
    print("  OOD comparison table (tab:ood)")
    print("=" * 70)

    row_fmt = "{:<20} {:<10}" + " {:>20}" * len(METRICS)
    print(row_fmt.format("Problem", "Setting", *METRIC_HEADERS))
    print("-" * (30 + 21 * len(METRICS)))

    latex_rows = []

    for problem in PROBLEMS:
        fpath = ood_dir / f"{problem}_aggregated.json"
        with open(fpath) as f:
            data = json.load(f)

        ci = data["cross_instance"]

        for cond in CONDITIONS:
            stats = ci[cond]
            vals = []
            for m in METRICS:
                mean = stats[m]["mean"]
                std = stats[m]["std"]
                vals.append(fmt_cell(mean, std))

            label = CONDITION_LABELS[cond]
            prob_label = PROBLEM_LABELS[problem] if cond == CONDITIONS[0] else ""
            print(row_fmt.format(prob_label, label, *vals))

            latex_vals = " & ".join(vals)
            if cond == CONDITIONS[0]:
                latex_rows.append(
                    f"\t{PROBLEM_LABELS[problem]}& {label} & {latex_vals} \\\\"
                )
            else:
                latex_rows.append(f"\t& {label} & {latex_vals} \\\\")

        if problem != PROBLEMS[-1]:
            latex_rows.append("\t\\addlinespace")

    print()
    print("  LaTeX rows:")
    for row in latex_rows:
        print(row)

    print()
    print("  Raw cross_instance values (for verification):")
    for problem in PROBLEMS:
        fpath = ood_dir / f"{problem}_aggregated.json"
        with open(fpath) as f:
            data = json.load(f)
        ci = data["cross_instance"]
        for cond in CONDITIONS:
            stats = ci[cond]
            print(f"  {problem} {CONDITION_LABELS[cond]}:")
            for m in METRICS:
                mean = stats[m]["mean"]
                std = stats[m]["std"]
                print(f"    {m}: {mean:.6f} +/- {std:.6f} -> {sig_figs(mean)} +/- {sig_figs(std)}")


if __name__ == "__main__":
    main()
