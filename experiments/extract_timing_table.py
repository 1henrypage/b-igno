"""Extract wall-clock timing data from baseline, physics, and map_laplace JSONs.

Reads individual per-seed JSONs, groups by (problem, seed), averages test
instances per seed, then computes mean +/- std across seeds.  Prints
formatted console output and LaTeX table rows rounded to 2 significant figures.

Usage:
    python extract_timing_table.py --base-dir experiments/results/structured
"""

from __future__ import annotations

import argparse
import glob
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


def load_experiment_data(
    base_dir: Path, experiment: str, problem: str
) -> list[dict]:
    """Load all per-instance JSONs for a problem, deduplicate by (seed, test_idx)."""
    pattern = str(base_dir / experiment / f"{problem}_*.json")
    files = sorted(glob.glob(pattern))

    candidates: dict[tuple[int, int], list[tuple[str, dict]]] = {}

    for fpath in files:
        if fpath.endswith("_aggregated.json"):
            continue
        with open(fpath) as f:
            data = json.load(f)

        seed = data["seed"]
        test_idx = data["test_idx"]
        ts = data.get("timestamp", "")
        candidates.setdefault((seed, test_idx), []).append((ts, data))

    result = []
    for key, items in candidates.items():
        items.sort(key=lambda x: x[0])
        result.append(items[-1][1])

    return result


def aggregate_by_seed(records: list[dict], extract_fn) -> dict[int, float]:
    """Group records by seed, average extracted values per seed."""
    by_seed: dict[int, list[float]] = {}
    for rec in records:
        val = extract_fn(rec)
        if val is not None:
            by_seed.setdefault(rec["seed"], []).append(val)

    return {seed: sum(vals) / len(vals) for seed, vals in by_seed.items()}


def mean_std(seed_means: dict[int, float]) -> tuple[float, float]:
    vals = list(seed_means.values())
    if not vals:
        return float("nan"), float("nan")
    m = sum(vals) / len(vals)
    if len(vals) >= 2:
        variance = sum((v - m) ** 2 for v in vals) / (len(vals) - 1)
        s = math.sqrt(variance)
    else:
        s = float("nan")
    return m, s


def fmt_cell(mean: float, std: float) -> str:
    if math.isnan(mean):
        return "n/a"
    m_str = sig_figs(mean)
    if math.isnan(std):
        return f"${m_str}$"
    s_str = sig_figs(std)
    return f"${m_str} \\pm {s_str}$"


def extract_baseline_timing(base_dir: Path) -> None:
    """Extract timing from baseline/ experiment (data-only MCMC)."""
    print("\n" + "=" * 70)
    print("  MCMC (data-only) timing — from baseline/")
    print("=" * 70)

    extractors = {
        "MAP": lambda d: d.get("map_time_s"),
        "Warmup": lambda d: d.get("condition", {}).get("warmup_time_s") if d.get("condition") else None,
        "Sampling": lambda d: d.get("condition", {}).get("sampling_time_s") if d.get("condition") else None,
    }

    all_rows = []

    for problem in PROBLEMS:
        records = load_experiment_data(base_dir, "baseline", problem)
        row = {"problem": problem}
        for col, extractor in extractors.items():
            seed_means = aggregate_by_seed(records, extractor)
            m, s = mean_std(seed_means)
            row[col] = (m, s)
            row[f"{col}_n"] = len(seed_means)
        all_rows.append(row)

    cols = list(extractors.keys())
    row_fmt = "{:<20}" + " {:>20}" * len(cols)
    print(row_fmt.format("Problem", *cols))
    print("-" * (20 + 21 * len(cols)))

    for row in all_rows:
        vals = [f"{sig_figs(row[c][0])} ± {sig_figs(row[c][1])}" for c in cols]
        print(row_fmt.format(PROBLEM_LABELS[row["problem"]], *vals))
        n_seeds = row[f"{cols[0]}_n"]
        if n_seeds != 3:
            print(f"  WARNING: {n_seeds} seeds (expected 3)")

    print("\n  LaTeX rows:")
    for row in all_rows:
        cells = " & ".join(fmt_cell(*row[c]) for c in cols)
        print(f"  {PROBLEM_LABELS[row['problem']]} & {cells} \\\\")

    return all_rows


def extract_physics_timing(base_dir: Path) -> None:
    """Extract timing from physics/ experiment (physics-constrained MCMC)."""
    print("\n" + "=" * 70)
    print("  MCMC (with physics) timing — from physics/")
    print("=" * 70)

    extractors = {
        "MAP": lambda d: d.get("map_time_s"),
        "Warmup": lambda d: (
            d.get("conditions", {}).get("physics", {}).get("warmup_time_s")
            if d.get("conditions") and d["conditions"].get("physics") else None
        ),
        "Sampling": lambda d: (
            d.get("conditions", {}).get("physics", {}).get("sampling_time_s")
            if d.get("conditions") and d["conditions"].get("physics") else None
        ),
    }

    all_rows = []

    for problem in PROBLEMS:
        records = load_experiment_data(base_dir, "physics", problem)
        row = {"problem": problem}
        for col, extractor in extractors.items():
            seed_means = aggregate_by_seed(records, extractor)
            m, s = mean_std(seed_means)
            row[col] = (m, s)
            row[f"{col}_n"] = len(seed_means)
        all_rows.append(row)

    cols = list(extractors.keys())
    row_fmt = "{:<20}" + " {:>20}" * len(cols)
    print(row_fmt.format("Problem", *cols))
    print("-" * (20 + 21 * len(cols)))

    for row in all_rows:
        vals = [f"{sig_figs(row[c][0])} ± {sig_figs(row[c][1])}" for c in cols]
        print(row_fmt.format(PROBLEM_LABELS[row["problem"]], *vals))
        n_seeds = row[f"{cols[0]}_n"]
        if n_seeds != 3:
            print(f"  WARNING: {n_seeds} seeds (expected 3)")

    print("\n  LaTeX rows:")
    for row in all_rows:
        cells = " & ".join(fmt_cell(*row[c]) for c in cols)
        print(f"  {PROBLEM_LABELS[row['problem']]} & {cells} \\\\")

    return all_rows


def extract_laplace_timing(base_dir: Path) -> None:
    """Extract timing from map_laplace/ experiment, filtering Hessian failures."""
    print("\n" + "=" * 70)
    print("  Laplace timing — from map_laplace/")
    print("=" * 70)

    def laplace_ok(d: dict) -> bool:
        lap = d.get("laplace")
        if not lap:
            return False
        return lap.get("n_negative_eigenvalues", 1) == 0

    extractors = {
        "MAP": lambda d: d.get("laplace", {}).get("map_time_s") if d.get("laplace") else None,
        "Hessian": lambda d: d.get("laplace", {}).get("hessian_time_s") if laplace_ok(d) else None,
        "Sampling": lambda d: d.get("laplace", {}).get("sampling_time_s") if laplace_ok(d) else None,
    }

    all_rows = []

    for problem in PROBLEMS:
        records = load_experiment_data(base_dir, "map_laplace", problem)
        row = {"problem": problem}
        for col, extractor in extractors.items():
            seed_means = aggregate_by_seed(records, extractor)
            m, s = mean_std(seed_means)
            row[col] = (m, s)
            row[f"{col}_n"] = len(seed_means)

        n_ok = sum(1 for r in records if laplace_ok(r))
        n_total = len(records)
        row["n_ok"] = n_ok
        row["n_total"] = n_total
        all_rows.append(row)

    cols = list(extractors.keys())
    row_fmt = "{:<20}" + " {:>20}" * len(cols) + " {:>12}"
    print(row_fmt.format("Problem", *cols, "OK/Total"))
    print("-" * (20 + 21 * len(cols) + 13))

    for row in all_rows:
        vals = [f"{sig_figs(row[c][0])} ± {sig_figs(row[c][1])}" for c in cols]
        ok_str = f"{row['n_ok']}/{row['n_total']}"
        print(row_fmt.format(PROBLEM_LABELS[row["problem"]], *vals, ok_str))

    print("\n  LaTeX rows:")
    for row in all_rows:
        cells = " & ".join(fmt_cell(*row[c]) for c in cols)
        print(f"  {PROBLEM_LABELS[row['problem']]} & {cells} \\\\")

    return all_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract wall-clock timing data")
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("results/structured"),
        help="Base directory containing baseline/, physics/, map_laplace/ subdirs",
    )
    args = parser.parse_args()

    extract_baseline_timing(args.base_dir)
    extract_physics_timing(args.base_dir)
    extract_laplace_timing(args.base_dir)


if __name__ == "__main__":
    main()
