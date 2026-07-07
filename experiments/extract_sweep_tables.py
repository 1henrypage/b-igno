"""Extract per-condition 3-seed means from noise_sweep and sensor_sweep JSONs.

Reads individual per-seed JSONs, groups by (problem, condition label),
deduplicates by seed (keeps latest timestamp), computes 3-seed means,
and prints LaTeX-ready table rows rounded to 2 significant figures.

Usage:
    python extract_sweep_tables.py --base-dir experiments/results/structured
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import sys
from pathlib import Path


NOISE_SWEEP_PROBLEMS = [
    "darcy_continuous",
    "darcy_piecewise",
    "eit",
    "burgers",
]

SENSOR_SWEEP_PROBLEMS = NOISE_SWEEP_PROBLEMS

NOISE_LEVELS = ["Clean", "SNR=50dB", "SNR=35dB", "SNR=25dB", "SNR=15dB"]
NOISE_LABELS_PRETTY = {
    "Clean": "Clean",
    "SNR=50dB": "50",
    "SNR=35dB": "35",
    "SNR=25dB": "25",
    "SNR=15dB": "15",
}

SENSOR_COUNTS = {
    "darcy_continuous": ["n_obs=25", "n_obs=50", "n_obs=100"],
    "darcy_piecewise": ["n_obs=25", "n_obs=50", "n_obs=100"],
    "eit": ["n_obs=31", "n_obs=62", "n_obs=124"],
    "burgers": ["n_obs=25", "n_obs=50", "n_obs=100"],
}

SENSOR_LABELS_PRETTY = {
    "n_obs=25": "25",
    "n_obs=50": "50",
    "n_obs=100": "100",
    "n_obs=31": "31",
    "n_obs=62": "62",
    "n_obs=124": "124",
}


def sig_figs(x: float, n: int = 2) -> str:
    if x == 0:
        return "0"
    magnitude = math.floor(math.log10(abs(x)))
    if magnitude >= n - 1:
        return f"{round(x, -(magnitude - n + 1)):.0f}"
    else:
        decimals = n - 1 - magnitude
        return f"{x:.{decimals}f}"


def load_sweep_data(
    base_dir: Path, experiment: str, problem: str
) -> dict[str, dict[int, dict]]:
    """Load all JSONs for a problem, return {label: {seed: metrics_dict}}.

    Deduplicates by keeping the latest timestamp per (seed, label).
    """
    pattern = str(base_dir / experiment / f"{problem}_*.json")
    files = sorted(glob.glob(pattern))

    entries: dict[str, list[tuple[str, int, dict]]] = {}

    for fpath in files:
        if fpath.endswith("_aggregated.json"):
            continue
        with open(fpath) as f:
            data = json.load(f)

        seed = data["seed"]
        ts = data.get("timestamp", "")

        baseline = data.get("baseline")
        if baseline:
            label = baseline["label"]
            entries.setdefault(label, []).append((ts, seed, baseline))

        for cond in data.get("sweep_conditions", []):
            label = cond["label"]
            entries.setdefault(label, []).append((ts, seed, cond))

    result: dict[str, dict[int, dict]] = {}
    for label, items in entries.items():
        by_seed: dict[int, dict] = {}
        seed_candidates: dict[int, list[tuple[str, dict]]] = {}
        for ts, seed, metrics in items:
            seed_candidates.setdefault(seed, []).append((ts, metrics))
        for seed, candidates in seed_candidates.items():
            candidates.sort(key=lambda x: x[0])
            by_seed[seed] = candidates[-1][1]
        result[label] = by_seed

    return result


def compute_means(by_seed: dict[int, dict], metrics: list[str]) -> dict[str, float]:
    vals = {m: [] for m in metrics}
    for seed, data in by_seed.items():
        for m in metrics:
            v = data.get(m)
            if v is not None:
                vals[m].append(v)
    return {m: sum(v) / len(v) if v else float("nan") for m, v in vals.items()}


def compute_stds(by_seed: dict[int, dict], metrics: list[str]) -> dict[str, float]:
    import statistics

    vals = {m: [] for m in metrics}
    for seed, data in by_seed.items():
        for m in metrics:
            v = data.get(m)
            if v is not None:
                vals[m].append(v)
    result = {}
    for m, v in vals.items():
        if len(v) >= 2:
            result[m] = statistics.stdev(v)
        else:
            result[m] = float("nan")
    return result


def print_noise_sweep_table(
    base_dir: Path, problem: str, metrics: list[str], header: str
) -> None:
    data = load_sweep_data(base_dir, "noise_sweep", problem)
    print(f"\n{'='*60}")
    print(f"  {header}")
    print(f"{'='*60}")

    metric_headers = [m.replace("_", " ") for m in metrics]
    row_fmt = "{:<8}" + " {:>10}" * len(metrics)
    print(row_fmt.format("SNR", *metric_headers))
    print("-" * (8 + 11 * len(metrics)))

    for label in NOISE_LEVELS:
        by_seed = data.get(label, {})
        if not by_seed:
            print(f"  WARNING: no data for {label}")
            continue
        means = compute_means(by_seed, metrics)
        stds = compute_stds(by_seed, metrics)
        n_seeds = len(by_seed)

        pretty = NOISE_LABELS_PRETTY[label]
        vals = [sig_figs(means[m]) for m in metrics]
        std_vals = [f"±{sig_figs(stds[m])}" if not math.isnan(stds[m]) else "" for m in metrics]
        print(row_fmt.format(pretty, *vals))
        print(row_fmt.format("", *std_vals))

        if n_seeds != 3:
            print(f"  WARNING: {n_seeds} seeds for {label} (expected 3)")

    print()
    print("  LaTeX rows:")
    for label in NOISE_LEVELS:
        by_seed = data.get(label, {})
        if not by_seed:
            continue
        means = compute_means(by_seed, metrics)
        pretty = NOISE_LABELS_PRETTY[label]
        vals = " & ".join(sig_figs(means[m]) for m in metrics)
        print(f"  {pretty} & {vals} \\\\")

    # Also print sigma values
    print()
    print("  Sigma values per condition:")
    for label in NOISE_LEVELS:
        by_seed = data.get(label, {})
        if not by_seed:
            continue
        sigmas = [d.get("sigma") for d in by_seed.values() if d.get("sigma") is not None]
        if sigmas:
            print(f"  {label}: sigma = {sigmas[0]}")


def print_sensor_sweep_table(base_dir: Path) -> None:
    print(f"\n{'='*60}")
    print(f"  Sensor sweep (all problems)")
    print(f"{'='*60}")

    all_rows = []

    for problem in SENSOR_SWEEP_PROBLEMS:
        data = load_sweep_data(base_dir, "sensor_sweep", problem)
        is_piecewise = "piecewise" in problem
        acc_metric = "a_err"
        metrics = [acc_metric, "crps_a", "coverage_95", "ci_width"]

        for label in SENSOR_COUNTS[problem]:
            by_seed = data.get(label, {})
            if not by_seed:
                print(f"  WARNING: no data for {problem} {label}")
                continue
            means = compute_means(by_seed, metrics)
            stds = compute_stds(by_seed, metrics)
            n_seeds = len(by_seed)

            pretty_label = SENSOR_LABELS_PRETTY[label]
            pretty_problem = {
                "darcy_continuous": "Darcy continuous",
                "darcy_piecewise": "Darcy piecewise",
                "eit": "EIT",
                "burgers": "Burgers",
            }[problem]

            vals = [sig_figs(means[m]) for m in metrics]
            std_vals = [f"±{sig_figs(stds[m])}" for m in metrics]

            all_rows.append((pretty_problem, pretty_label, vals, std_vals, n_seeds))

            if n_seeds != 3:
                print(f"  WARNING: {n_seeds} seeds for {problem} {label} (expected 3)")

    metric_headers = ["Accuracy", "CRPS", "Cov 95%", "CI width"]
    row_fmt = "{:<20} {:>5}" + " {:>10}" * 4
    print(row_fmt.format("Problem", "M", *metric_headers))
    print("-" * (25 + 11 * 4))

    for problem_name, m_val, vals, std_vals, n_seeds in all_rows:
        print(row_fmt.format(problem_name, m_val, *vals))
        print(row_fmt.format("", "", *std_vals))

    print()
    print("  LaTeX rows:")
    current_problem = None
    for problem_name, m_val, vals, std_vals, n_seeds in all_rows:
        prefix = problem_name if problem_name != current_problem else ""
        vals_str = " & ".join(vals)
        print(f"  {prefix}& {m_val} & {vals_str} \\\\")
        current_problem = problem_name

    # Print sigma values
    print()
    print("  Sigma values per condition:")
    for problem in SENSOR_SWEEP_PROBLEMS:
        data = load_sweep_data(base_dir, "sensor_sweep", problem)
        for label in SENSOR_COUNTS[problem]:
            by_seed = data.get(label, {})
            if not by_seed:
                continue
            sigmas = [d.get("sigma") for d in by_seed.values() if d.get("sigma") is not None]
            if sigmas:
                print(f"  {problem} {label}: sigma = {sigmas[0]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract sweep table data")
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("results/structured"),
        help="Base directory containing noise_sweep/ and sensor_sweep/ subdirs",
    )
    args = parser.parse_args()

    continuous_metrics = ["a_err", "crps_a", "coverage_95", "ci_width"]
    piecewise_metrics = ["a_err", "crps_a", "coverage_95", "ci_width"]

    print_noise_sweep_table(
        args.base_dir, "darcy_continuous", continuous_metrics,
        "Continuous Darcy noise sweep (a_err = rRMSE)"
    )
    print_noise_sweep_table(
        args.base_dir, "darcy_piecewise", piecewise_metrics,
        "Piecewise Darcy noise sweep (a_err = I_corr)"
    )
    print_noise_sweep_table(
        args.base_dir, "eit", continuous_metrics,
        "EIT noise sweep (a_err = rRMSE)"
    )
    print_noise_sweep_table(
        args.base_dir, "burgers", continuous_metrics,
        "Burgers noise sweep (a_err = rRMSE)"
    )

    print_sensor_sweep_table(args.base_dir)


if __name__ == "__main__":
    main()
