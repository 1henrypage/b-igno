"""Cross-seed Laplace aggregation script.

CLI usage:
    # Single problem:
    python aggregate_laplace.py --problem darcy_continuous

    # All problems:
    python aggregate_laplace.py --all

    # Custom base dir:
    python aggregate_laplace.py --all --base-dir ~/school/actual/experiments/results/structured

Reads map_laplace JSON files, groups by test instance, excludes runs where
the Hessian failed (n_negative_eigenvalues > 0), and writes
    {base_dir}/map_laplace/{problem}_laplace_aggregated.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from experiment_utils import load_cross_seed_results

ALL_PROBLEMS = [
    "darcy_continuous", "darcy_piecewise",
    "darcy_piecewise_5v10", "darcy_piecewise_5v100", "darcy_piecewise_5v1000",
    "eit", "burgers",
]

LAPLACE_METRICS = [
    "a_err", "u_err", "crps_a", "coverage_95", "ci_width", "mean_std",
    "nll_a", "map_a_err", "map_u_err", "spearman_rho_error_std",
    "spearman_pvalue_error_std",
]

BASE_DIR = Path(__file__).parent / "results" / "structured"


def _summarise(values: List[float]) -> Dict:
    if not values:
        return {"mean": float("nan"), "std": float("nan"), "n": 0, "values": []}
    arr = np.array(values)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1) if len(arr) > 1 else 0.0),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "n": len(arr),
        "values": [float(v) for v in values],
    }


def aggregate_laplace(problem: str, base_dir: Path = BASE_DIR) -> Optional[Dict]:
    results = load_cross_seed_results("map_laplace", problem, base_dir=base_dir)
    if not results:
        print(f"  [SKIP] map_laplace/{problem}: no result files found.")
        return None

    total = len(results)
    valid = [r for r in results if r.laplace and r.laplace.n_negative_eigenvalues == 0]
    failed = [r for r in results if not r.laplace or r.laplace.n_negative_eigenvalues > 0]

    print(f"\n{'='*72}")
    print(f"  LAPLACE / {problem}  ({total} runs, {len(valid)} valid, {len(failed)} failed)")
    print(f"{'='*72}")

    if failed:
        for r in failed:
            n_neg = r.laplace.n_negative_eigenvalues if r.laplace else "missing"
            print(f"    FAILED: seed={r.seed} test_idx={r.test_idx} n_negative_eigenvalues={n_neg}")

    # Group valid results by test_idx
    by_instance: Dict[int, list] = defaultdict(list)
    for r in valid:
        tidx = r.test_idx if r.test_idx is not None else 0
        by_instance[tidx].append(r)

    per_instance: Dict[str, Dict] = {}
    for tidx in sorted(by_instance):
        group = by_instance[tidx]
        seeds = [r.seed for r in group]
        print(f"\n  --- test_idx={tidx}  ({len(group)} valid seed(s): {seeds}) ---")

        summaries = {}
        for metric in LAPLACE_METRICS:
            values = []
            for r in group:
                val = getattr(r.laplace, metric, None)
                if val is not None and not (isinstance(val, float) and np.isnan(val)):
                    values.append(float(val))
            summaries[metric] = _summarise(values)
            if values:
                s = summaries[metric]
                print(f"    {metric:<28s}  {s['mean']:.4f} ± {s['std']:.4f}  (n={s['n']})")

        per_instance[str(tidx)] = {
            "n_seeds": len(group),
            "seeds": seeds,
            "metric_summaries": summaries,
        }

    # Cross-instance summary: mean ± std of per-instance means
    cross_instance: Dict[str, Dict] = {}
    instance_keys = sorted(per_instance.keys())
    if len(instance_keys) > 1:
        print("\n  Cross-instance summary (mean ± std across test instances):")
        for metric in LAPLACE_METRICS:
            instance_means = []
            for k in instance_keys:
                s = per_instance[k]["metric_summaries"].get(metric, {})
                m = s.get("mean", float("nan"))
                if not np.isnan(m):
                    instance_means.append(m)
            cross_instance[metric] = _summarise(instance_means)
            if instance_means:
                cs = cross_instance[metric]
                print(f"    {metric:<28s}  {cs['mean']:.4f} ± {cs['std']:.4f}  (n={cs['n']})")

    payload = {
        "experiment": "map_laplace",
        "problem": problem,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "total_runs": total,
        "valid_runs": len(valid),
        "failed_runs": len(failed),
        "failed_details": [
            {"seed": r.seed, "test_idx": r.test_idx,
             "n_negative_eigenvalues": r.laplace.n_negative_eigenvalues if r.laplace else None}
            for r in failed
        ],
        "per_instance": per_instance,
    }
    if cross_instance:
        payload["cross_instance"] = cross_instance

    out_dir = base_dir / "map_laplace"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{problem}_laplace_aggregated.json"
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\n  Saved → {out_path}")

    return payload


def main() -> None:
    p = argparse.ArgumentParser(description="Aggregate Laplace approximation results.")
    p.add_argument("--problem", choices=ALL_PROBLEMS, help="Problem name.")
    p.add_argument("--all", action="store_true", help="Aggregate all problems.")
    p.add_argument("--base-dir", type=Path, default=BASE_DIR, help="Base directory.")
    args = p.parse_args()

    if args.all:
        problems = ALL_PROBLEMS
    elif args.problem:
        problems = [args.problem]
    else:
        p.error("Provide --problem or --all.")
        return

    for problem in problems:
        aggregate_laplace(problem, base_dir=args.base_dir)


if __name__ == "__main__":
    main()
