"""Cross-seed aggregation script for all experiments.

CLI usage:
    # Single experiment + problem:
    python aggregate_seeds.py --experiment baseline --problem darcy_continuous

    # Single experiment + problem + specific test instance:
    python aggregate_seeds.py --experiment baseline --problem darcy_continuous --test-idx 1

    # All problems for one experiment:
    python aggregate_seeds.py --experiment physics --all

    # All experiments and all problems:
    python aggregate_seeds.py --all

Loads per-seed ExperimentResult JSON files from
    experiments/results/structured/{experiment}/{problem}_*.json
runs cross-seed consistency and comparison-robustness tests, prints formatted tables to stdout, and writes
    experiments/results/structured/{experiment}/{problem}_aggregated.json

When multiple test instances are present (test_idx field), the output JSON
contains both per-instance aggregations and a cross-instance summary
(mean ± std across test instances).
"""

from __future__ import annotations

import argparse
import json
import sys
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from experiment_utils import (
    cross_seed_consistency_test,
    cross_seed_comparison_robustness,
    cross_seed_metric_summary,
    format_cross_seed_table,
    load_cross_seed_results,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALL_PROBLEMS = [
    "darcy_continuous", "darcy_piecewise",
    "darcy_piecewise_5v10", "darcy_piecewise_5v100", "darcy_piecewise_5v1000",
    "eit", "burgers",
]
ALL_EXPERIMENTS = ["baseline", "physics", "ood", "noise_sweep", "sensor_sweep", "map_laplace"]

# Metrics reported for every experiment type.
SCALAR_METRICS = ["a_err", "u_err", "ess_min", "rhat_max", "coverage_95", "ci_width", "crps_a",
                  "mean_std", "nll_a", "n_div", "map_a_err", "map_u_err", "chi2_ppc_pvalue",
                  "spearman_rho_error_std", "spearman_pvalue_error_std"]

# Metrics available on PriorPredictiveResult.
PRIOR_METRICS = ["a_err", "crps_a", "coverage_95", "ci_width", "mean_std"]

# RQ2 condition names (physics comparison experiments).
RQ2_CONDITIONS = {
    "darcy_continuous": ("data_only", "physics"),
    "darcy_piecewise":  ("data_only", "physics"),
    "eit":              ("data_only", "physics"),
    "burgers":          ("data_only", "physics"),
}

# OOD comparison pairs (4-condition experiments: pick the OOD pair for robustness test).
OOD_CONDITIONS = {
    "darcy_continuous": ("ood_data_only", "ood_physics"),
    "darcy_piecewise":  ("ood_data_only", "ood_physics"),
    "eit":              ("ood_data_only", "ood_physics"),
    "burgers":          ("ood_data_only", "ood_physics"),
}

BASE_DIR = Path(__file__).parent / "results" / "structured"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _detect_experiment_type(results) -> Optional[str]:
    if not results:
        return None
    return results[0].experiment_type


def _detect_sweep_var_values(results) -> Optional[List[float]]:
    """Try to extract sweep variable numeric values from sweep_conditions labels."""
    if not results:
        return None
    r = results[0]
    if r.experiment_type != "sweep" or not r.sweep_conditions:
        return None
    vals = []
    for sc in r.sweep_conditions:
        try:
            vals.append(float(sc.label))
        except (ValueError, AttributeError):
            return None
    return vals


def _run_consistency_tests(
    results,
    exp_type: str,
    condition_key: Optional[str] = None,
) -> List[Dict]:
    """Run cross-seed consistency (CV) test for each scalar metric."""
    tests = []
    for metric in SCALAR_METRICS:
        try:
            t = cross_seed_consistency_test(
                results, metric=metric, condition_key=condition_key
            )
            tests.append(t)
        except Exception as exc:
            tests.append({"metric": metric, "error": str(exc)})
    return tests


def _run_comparison_tests(
    results, condition_a: str, condition_b: str
) -> List[Dict]:
    """Run comparison robustness tests for each scalar metric (RQ2)."""
    tests = []
    for metric in SCALAR_METRICS:
        try:
            t = cross_seed_comparison_robustness(
                results, condition_a=condition_a, condition_b=condition_b, metric=metric
            )
            tests.append(t)
        except Exception as exc:
            tests.append({"metric": metric, "error": str(exc)})
    return tests


def _print_section(title: str) -> None:
    width = 72
    print()
    print("=" * width)
    print(f"  {title}")
    print("=" * width)


def _print_tests(tests: List[Dict], label: str) -> None:
    print(f"\n  {label}:")
    for t in tests:
        if "error" in t:
            print(f"    [{t['metric']}]  ERROR: {t['error']}")
        elif "interpretation" in t:
            print(f"    {t['interpretation']}")


def _prior_metric_summary(
    results,
    metric: str,
    ood: bool = False,
) -> Dict[str, float]:
    """Compute mean ± std of a PriorPredictiveResult metric across seeds.

    Args:
        results: list of ExperimentResult (one per seed).
        metric: attribute name on PriorPredictiveResult.
        ood: if True, read from ``prior_ood`` instead of ``prior``.

    Returns:
        Dict with keys ``mean``, ``std``, ``min``, ``max``, ``n``, ``values``.
    """
    values = []
    for r in results:
        prior = r.prior_ood if ood else r.prior
        if prior is None:
            continue
        val = getattr(prior, metric, None)
        if val is not None:
            values.append(float(val))
    if not values:
        return {"mean": float("nan"), "std": float("nan"), "min": float("nan"),
                "max": float("nan"), "n": 0, "values": []}
    arr = np.array(values)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1) if len(arr) > 1 else 0.0),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "n": len(arr),
        "values": values,
    }


def _collect_prior_summaries(results, ood: bool = False) -> Dict[str, Dict]:
    """Collect cross-seed summaries for all prior metrics. Returns empty dict if no prior data."""
    summaries = {}
    for metric in PRIOR_METRICS:
        s = _prior_metric_summary(results, metric, ood=ood)
        if s["n"] > 0:
            summaries[metric] = s
    return summaries


def _format_prior_row(results, label: str = "Prior", ood: bool = False, metrics: Optional[List[str]] = None) -> Optional[str]:
    """Format a summary table row for prior metrics. Returns None if no prior data."""
    if metrics is None:
        metrics = ["a_err", "ess_min", "coverage_95", "n_div"]
    col_w = 18
    has_any = any(
        (r.prior_ood if ood else r.prior) is not None for r in results
    )
    if not has_any:
        return None
    cells = []
    for m in metrics:
        if m in PRIOR_METRICS:
            s = _prior_metric_summary(results, m, ood=ood)
            if s["n"] > 0:
                cells.append(f"{s['mean']:.4f}±{s['std']:.4f}(n={s['n']})")
            else:
                cells.append("—")
        else:
            cells.append("—")
    return f"{label:<28}" + "".join(f"{c:>{col_w}}" for c in cells)


# ---------------------------------------------------------------------------
# Sweep reassembly (condition-split sweep scripts)
# ---------------------------------------------------------------------------

def reassemble_sweep_results(experiment: str, problem: str, base_dir: Optional[Path] = None) -> list:
    """Reconstruct sweep-shaped ExperimentResults from condition-split JSONs.

    Condition-split scripts produce individual JSONs with sweep_value set.
    This function groups them back into sweep-shaped ExperimentResults for
    cross-seed aggregation.
    """
    from collections import defaultdict
    from results_schema import ExperimentResult

    all_results = load_cross_seed_results(experiment, problem, base_dir=base_dir)

    sweep_origin = [r for r in all_results if r.sweep_value is not None]
    native = [r for r in all_results if r.sweep_value is None]

    if not sweep_origin:
        return native

    groups: dict = defaultdict(list)
    for r in sweep_origin:
        groups[(r.seed, r.test_idx)].append(r)

    reassembled = []
    for (seed, test_idx), results in groups.items():
        results.sort(key=lambda r: r.sweep_value if r.sweep_value is not None else -1)

        baselines = [r for r in results if r.is_sweep_baseline]
        conditions = [r for r in results if not r.is_sweep_baseline]

        # Comparison-type splits (physics sweeps) keep their conditions dict
        # intact — aggregate() groups them by sweep_value instead.
        if conditions and conditions[0].experiment_type == "comparison":
            reassembled.extend(conditions)
            if baselines:
                reassembled.extend(baselines)
            continue

        baseline_mcmc = baselines[0].condition if baselines else None
        sweep_mcmc = [r.condition for r in conditions]

        sweep_var = results[0].sweep_var

        synthetic = ExperimentResult(
            experiment=experiment,
            problem=problem,
            experiment_type="sweep",
            timestamp=results[0].timestamp,
            seed=seed,
            test_idx=test_idx,
            sweep_var=sweep_var,
            baseline=baseline_mcmc,
            sweep_conditions=sweep_mcmc,
            prior=baselines[0].prior if baselines else results[0].prior,
            laplace=baselines[0].laplace if baselines else None,
        )
        reassembled.append(synthetic)

    return native + reassembled


# ---------------------------------------------------------------------------
# Aggregate one (experiment, problem) pair
# ---------------------------------------------------------------------------

def _aggregate_group(
    experiment: str,
    problem: str,
    results,
    exp_type: Optional[str],
    base_dir: Path,
    test_idx_label: Optional[int] = None,
) -> Dict:
    """Run cross-seed aggregation for a single group of results (one test instance).

    Returns a dict with keys: seeds, n_seeds, metric_summaries (for single),
    consistency_tests, comparison_robustness, prior_summaries, etc.
    """
    seeds = [r.seed for r in results]
    n = len(results)

    label = f"test_idx={test_idx_label}" if test_idx_label is not None else "all"
    print(f"\n  --- Instance: {label}  ({n} seed(s): {seeds}) ---")

    condition_key: Optional[str] = None
    if exp_type == "comparison":
        first_cond = list((results[0].conditions or {}).keys())
        condition_key = first_cond[0] if first_cond else None

    try:
        # Pass test_idx to restrict format_cross_seed_table if supported
        table = format_cross_seed_table(
            experiment, [problem], condition_key=condition_key, base_dir=base_dir,
            test_idx=test_idx_label,
        )
        print(table)
    except TypeError:
        # format_cross_seed_table doesn't support test_idx yet — fall back
        try:
            table = format_cross_seed_table(
                experiment, [problem], condition_key=condition_key, base_dir=base_dir
            )
            print(table)
        except Exception as exc:
            print(f"  (could not render summary table: {exc})")
    except Exception as exc:
        print(f"  (could not render summary table: {exc})")

    prior_row = _format_prior_row(results, label="Prior")
    if prior_row:
        print(prior_row)
    prior_ood_row = _format_prior_row(results, label="Prior (OOD)", ood=True)
    if prior_ood_row:
        print(prior_ood_row)

    instance_payload: Dict = {"n_seeds": n, "seeds": seeds}

    if exp_type == "single":
        tests = _run_consistency_tests(results, exp_type)
        _print_tests(tests, "Consistency tests (single)")
        instance_payload["consistency_tests"] = tests

        summaries = {}
        for metric in SCALAR_METRICS:
            try:
                summaries[metric] = cross_seed_metric_summary(results, metric=metric)
            except Exception:
                pass
        instance_payload["metric_summaries"] = summaries

    elif exp_type == "comparison":
        all_cond_keys = list((results[0].conditions or {}).keys())
        instance_payload["consistency_tests"] = {}
        for ck in all_cond_keys:
            tests = _run_consistency_tests(results, exp_type, condition_key=ck)
            _print_tests(tests, f"Consistency tests (condition={ck!r})")
            instance_payload["consistency_tests"][ck] = tests

        if len(all_cond_keys) == 2:
            ca, cb = all_cond_keys
        elif problem in OOD_CONDITIONS and all(k in all_cond_keys for k in OOD_CONDITIONS[problem]):
            ca, cb = OOD_CONDITIONS[problem]
        elif problem in RQ2_CONDITIONS:
            ca, cb = RQ2_CONDITIONS[problem]
        else:
            ca, cb = None, None

        if ca and cb and ca in all_cond_keys and cb in all_cond_keys:
            comp_tests = _run_comparison_tests(results, ca, cb)
            _print_tests(comp_tests, f"Comparison robustness ({ca!r} vs {cb!r})")
            instance_payload["comparison_robustness"] = {
                "condition_a": ca,
                "condition_b": cb,
                "tests": comp_tests,
            }

    elif exp_type == "sweep":
        sweep_var_values = _detect_sweep_var_values(results)
        tests = _run_consistency_tests(results, exp_type)
        _print_tests(tests, "Consistency tests (baseline, sweep)")
        instance_payload["consistency_tests"] = tests

    prior_summaries = _collect_prior_summaries(results)
    if prior_summaries:
        instance_payload["prior_summaries"] = prior_summaries
    prior_ood_summaries = _collect_prior_summaries(results, ood=True)
    if prior_ood_summaries:
        instance_payload["prior_ood_summaries"] = prior_ood_summaries

    return instance_payload


def _group_results(results) -> Dict[tuple, list]:
    """Group results by (test_idx, sweep_value).

    For non-sweep results sweep_value is None, so grouping collapses to
    test_idx only (backward compatible).  For condition-split comparison
    results each sweep_value gets its own group.
    """
    groups: Dict[tuple, list] = {}
    for r in results:
        tidx = r.test_idx if r.test_idx is not None else 0
        sv = getattr(r, "sweep_value", None)
        groups.setdefault((tidx, sv), []).append(r)
    return dict(sorted(groups.items(), key=lambda x: (x[0][0], x[0][1] or -1)))


def _compute_cross_instance_summary(
    per_instance: Dict[str, Dict],
    exp_type: Optional[str],
) -> Dict[str, Dict]:
    """Compute mean ± std across test instances for each scalar metric.

    For 'single' experiments, reads per-instance metric_summaries[metric]['mean'].
    For 'comparison' experiments, returns {condition: {metric: {mean, std, n, values}}}
    by reading per-instance consistency_tests[condition][metric].summary.mean.
    For other types, skips (returns empty dict).
    """
    if exp_type == "comparison":
        return _compute_cross_instance_summary_comparison(per_instance)
    if exp_type != "single":
        return {}

    cross: Dict[str, Dict] = {}
    for metric in SCALAR_METRICS:
        means = []
        for inst in per_instance.values():
            ms = inst.get("metric_summaries", {})
            s = ms.get(metric)
            if s and not np.isnan(s.get("mean", float("nan"))):
                means.append(s["mean"])
        if not means:
            cross[metric] = {"mean": float("nan"), "std": float("nan"), "n": 0, "values": []}
            continue
        arr = np.array(means)
        cross[metric] = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr, ddof=1) if len(arr) > 1 else 0.0),
            "n": len(arr),
            "values": means,
        }
    return cross


def _compute_cross_instance_summary_comparison(
    per_instance: Dict[str, Dict],
) -> Dict[str, Dict[str, Dict]]:
    """Cross-instance summary for comparison experiments.

    Returns {condition_name: {metric: {mean, std, n, values}}}.
    """
    condition_keys: set = set()
    for inst in per_instance.values():
        ct = inst.get("consistency_tests", {})
        if isinstance(ct, dict):
            condition_keys.update(ct.keys())

    cross: Dict[str, Dict[str, Dict]] = {}
    for cond in sorted(condition_keys):
        cross[cond] = {}
        for metric in SCALAR_METRICS:
            means = []
            for inst in per_instance.values():
                ct = inst.get("consistency_tests", {})
                tests = ct.get(cond, [])
                for t in tests:
                    if t.get("metric") == metric:
                        val = t.get("summary", {}).get("mean", float("nan"))
                        if not np.isnan(val):
                            means.append(val)
                        break
            if not means:
                cross[cond][metric] = {"mean": float("nan"), "std": float("nan"), "n": 0, "values": []}
                continue
            arr = np.array(means)
            cross[cond][metric] = {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr, ddof=1) if len(arr) > 1 else 0.0),
                "n": len(arr),
                "values": [float(v) for v in means],
            }
    return cross


def _print_cross_instance_table(cross_instance: Dict[str, Dict]) -> None:
    """Print a summary table of cross-instance means and stds."""
    if not cross_instance:
        return

    first_val = next(iter(cross_instance.values()))
    is_comparison = isinstance(first_val, dict) and "mean" not in first_val

    if is_comparison:
        col_w = 22
        conditions = sorted(cross_instance.keys())
        print("\n  Cross-instance summary (mean ± std across test instances):")
        header = f"  {'Metric':<20}" + "".join(f"{c:>{col_w}}" for c in conditions)
        print(header)
        print("  " + "-" * (20 + col_w * len(conditions)))
        all_metrics = set()
        for cond_data in cross_instance.values():
            all_metrics.update(m for m, s in cond_data.items() if s.get("n", 0) > 0)
        for m in SCALAR_METRICS:
            if m not in all_metrics:
                continue
            cells = []
            for c in conditions:
                s = cross_instance[c].get(m, {})
                if s.get("n", 0) > 0:
                    cells.append(f"{s['mean']:.4f}±{s['std']:.4f}(n={s['n']})")
                else:
                    cells.append("—")
            print(f"  {m:<20}" + "".join(f"{c:>{col_w}}" for c in cells))
        return

    metrics_to_show = [m for m in SCALAR_METRICS if cross_instance.get(m, {}).get("n", 0) > 0]
    if not metrics_to_show:
        return
    print("\n  Cross-instance summary (mean ± std across test instances):")
    col_w = 22
    header = f"  {'Metric':<20}" + "".join(f"{'Value':>{col_w}}")
    print(header)
    print("  " + "-" * (20 + col_w))
    for m in metrics_to_show:
        s = cross_instance[m]
        cell = f"{s['mean']:.4f}±{s['std']:.4f}(n={s['n']})"
        print(f"  {m:<20}{cell:>{col_w}}")


def aggregate(
    experiment: str,
    problem: str,
    base_dir: Path = BASE_DIR,
    test_idx: Optional[int] = None,
) -> Optional[Dict]:
    """Aggregate cross-seed results for one experiment/problem.

    Prints a summary table and test results to stdout.
    Saves {problem}_aggregated.json in the experiment directory.

    When multiple test instances are present (test_idx field on results),
    the output JSON contains both per-instance and cross-instance sections.
    Use test_idx to restrict aggregation to a single test instance.

    Returns the aggregated payload dict (or None if no data found).
    """
    if "sweep" in experiment:
        all_results = reassemble_sweep_results(experiment, problem, base_dir=base_dir)
    else:
        all_results = load_cross_seed_results(experiment, problem, base_dir=base_dir)

    if not all_results:
        print(f"  [SKIP] {experiment}/{problem}: no result files found.")
        return None

    # Filter to a specific test_idx if requested
    if test_idx is not None:
        all_results = [r for r in all_results if (r.test_idx if r.test_idx is not None else 0) == test_idx]
        if not all_results:
            print(f"  [SKIP] {experiment}/{problem}: no results for test_idx={test_idx}.")
            return None

    exp_type = _detect_experiment_type(all_results)
    total_seeds = sorted({r.seed for r in all_results if r.seed is not None})

    _print_section(
        f"{experiment.upper()} / {problem}  "
        f"({len(all_results)} result(s), seeds={total_seeds})"
    )

    # Group by (test_idx, sweep_value) — sweep_value is None for
    # non-condition-split results so the grouping collapses to test_idx.
    groups = _group_results(all_results)
    multi_instance = len(groups) > 1

    per_instance: Dict[str, Dict] = {}
    for (tidx, sv), group_results in groups.items():
        group_exp_type = _detect_experiment_type(group_results)
        label_parts = []
        if tidx is not None:
            label_parts.append(f"test_idx={tidx}")
        if sv is not None:
            sweep_var = getattr(group_results[0], "sweep_var", None) or "sweep"
            label_parts.append(f"{sweep_var}={sv:g}")
        label = ", ".join(label_parts) if label_parts else None

        per_instance[str((tidx, sv))] = _aggregate_group(
            experiment, problem, group_results, group_exp_type, base_dir,
            test_idx_label=label if multi_instance else None,
        )

    # Cross-instance summary (only meaningful with >1 instance)
    cross_instance = _compute_cross_instance_summary(per_instance, exp_type) if multi_instance else {}

    if cross_instance:
        _print_cross_instance_table(cross_instance)

    payload: Dict = {
        "experiment": experiment,
        "problem": problem,
        "experiment_type": exp_type,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }

    if multi_instance:
        payload["per_instance"] = per_instance
        payload["cross_instance"] = cross_instance
    else:
        # Single instance: flatten into top-level for backward compatibility
        only = per_instance[str(next(iter(groups)))]
        payload.update(only)

    out_dir = base_dir / experiment
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{problem}_aggregated.json"
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\n  Saved → {out_path}")

    return payload


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Aggregate cross-seed experiment results for J-IGNO.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--experiment", choices=ALL_EXPERIMENTS, help="Experiment key (e.g. baseline).")
    p.add_argument(
        "--problem",
        choices=ALL_PROBLEMS,
        help="Problem name (e.g. darcy_continuous).",
    )
    p.add_argument(
        "--all",
        action="store_true",
        help="Aggregate all experiments × problems (ignores --experiment / --problem).",
    )
    p.add_argument(
        "--base-dir",
        type=Path,
        default=BASE_DIR,
        help="Override base directory for structured results.",
    )
    p.add_argument(
        "--test-idx",
        type=int,
        default=None,
        metavar="N",
        help="Restrict aggregation to a single test instance (e.g. 0, 1, 2).",
    )
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    base_dir: Path = args.base_dir

    if args.all:
        experiments = ALL_EXPERIMENTS
        problems = ALL_PROBLEMS
    elif args.experiment and not args.problem:
        experiments = [args.experiment]
        problems = ALL_PROBLEMS
    elif args.experiment and args.problem:
        experiments = [args.experiment]
        problems = [args.problem]
    else:
        parser.error("Provide --experiment [--problem] or --all.")
        return

    any_found = False
    for experiment in experiments:
        for problem in problems:
            result = aggregate(experiment, problem, base_dir=base_dir, test_idx=args.test_idx)
            if result is not None:
                any_found = True

    if not any_found:
        print(
            "\nNo result files found. Make sure experiments have been run and "
            "results saved to experiments/results/structured/{experiment}/{problem}_*.json"
        )


if __name__ == "__main__":
    main()
