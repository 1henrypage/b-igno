"""Result building, saving, loading, and cross-seed aggregation."""
import json
import numpy as np
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

from ._metrics import mcmc_reliability_flag


def format_significance_table(
    results: Dict[str, Dict],
    title: str = 'Statistical Significance Summary',
) -> None:
    """Print a formatted significance table."""
    width = 82
    print('─' * width)
    print(f'  {title}')
    print('─' * width)

    for label, res in results.items():
        if 'ci_lo' in res and 'mean_diff' not in res:
            est = res['estimate']
            lo, hi = res['ci_lo'], res['ci_hi']
            print(f'  {label:<38s}  {est:>10.5f}  95% CI [{lo:.5f}, {hi:.5f}]')
        elif 'mean_diff' in res:
            diff = res['mean_diff']
            lo, hi = res['ci_lo'], res['ci_hi']
            sig = '*' if res['significant'] else 'n.s.'
            print(f'  {label:<38s}  diff={diff:+.5f}  CI [{lo:+.5f}, {hi:+.5f}]  {sig}')
        else:
            print(f'  {label:<38s}  {res}')

    print('─' * width)
    print('  Significance: ** p<0.01  * p<0.05  ~ p<0.1  n.s. not significant')
    print('  Bootstrap CIs reflect MCMC finite-sample variability, not posterior uncertainty.')
    print('─' * width)


def build_mcmc_result(
    run_result: Dict,
    num_warmup: int,
    num_samples: int,
    num_chains: int,
) -> "MCMCResult":
    """Build an MCMCResult from a raw ``run_condition`` result dict.

    Args:
        run_result: dict returned by a ``run_condition`` function in an RQ
            notebook. Expected keys: 'sigma', 'ess_min', 'rhat_max',
            'rhat_mean', 'n_div', 'reliability_flag', 'a_err', 'crps_a',
            'coverage', 'ci_width', 'mean_std', 'cal_levels', 'cal_empirical'.
            Optional: 'label', 'reliability_explanation', 'u_err',
            'map_a_err', 'map_u_err'.
        num_warmup: NUTS warmup steps.
        num_samples: NUTS draw steps (per chain).
        num_chains: number of parallel chains.

    Returns:
        MCMCResult dataclass instance (JSON-serialisable).
    """
    from results_schema import MCMCResult

    flag = run_result.get("reliability_flag", "UNKNOWN")
    ess_min = float(run_result["ess_min"])
    rhat_max = float(run_result["rhat_max"])
    n_div = int(run_result["n_div"])
    explanation = run_result.get(
        "reliability_explanation",
        mcmc_reliability_flag(ess_min, rhat_max, n_div, num_chains * num_samples)[1],
    )

    return MCMCResult(
        sigma=float(run_result["sigma"]),
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        ess_min=ess_min,
        rhat_max=rhat_max,
        rhat_mean=float(run_result.get("rhat_mean", float("nan"))),
        n_div=n_div,
        reliability_flag=flag,
        reliability_explanation=explanation,
        a_err=float(run_result["a_err"]),
        crps_a=float(run_result["crps_a"]),
        coverage_95=float(run_result.get("coverage", run_result.get("coverage_95", float("nan")))),
        ci_width=float(run_result["ci_width"]),
        mean_std=float(run_result["mean_std"]),
        nll_a=float(run_result["nll_a"]) if "nll_a" in run_result else None,
        cal_levels=[float(v) for v in run_result["cal_levels"]],
        cal_empirical=[float(v) for v in run_result["cal_empirical"]],
        u_err=float(run_result["u_err"]) if "u_err" in run_result else None,
        label=str(run_result.get("label", "")),
        chi2_ppc=float(run_result["chi2_ppc"]) if "chi2_ppc" in run_result else None,
        chi2_ppc_pvalue=float(run_result["chi2_ppc_pvalue"]) if "chi2_ppc_pvalue" in run_result else None,
        map_a_err=float(run_result['map_a_err']) if 'map_a_err' in run_result else None,
        map_u_err=float(run_result['map_u_err']) if 'map_u_err' in run_result else None,
        warmup_time_s=run_result.get("warmup_time_s"),
        sampling_time_s=run_result.get("sampling_time_s"),
        step_time_s=run_result.get("step_time_s"),
        spearman_rho_error_std=run_result.get('spearman_rho_error_std'),
        spearman_pvalue_error_std=run_result.get('spearman_pvalue_error_std'),
        a_err_per_sample=run_result.get('a_err_per_sample'),
    )


def build_laplace_result(run_result: Dict) -> "LaplaceResult":
    """Build a LaplaceResult from a raw result dict.

    Expected keys mirror ``LaplaceResult`` fields.  See the Laplace block in
    the experiment scripts for how ``run_result`` is assembled.
    """
    from results_schema import LaplaceResult

    raw_sigma = run_result.get("sigma")
    return LaplaceResult(
        sigma=float(raw_sigma) if raw_sigma is not None else None,
        n_samples=int(run_result["n_samples"]),
        map_max_iter=int(run_result["map_max_iter"]),
        hessian_reg_lambda=float(run_result["hessian_reg_lambda"]),
        neg_log_posterior_at_map=float(run_result["neg_log_posterior_at_map"]),
        grad_norm_at_map=float(run_result["grad_norm_at_map"]),
        map_converged=bool(run_result["map_converged"]),
        hessian_min_eigenvalue=float(run_result["hessian_min_eigenvalue"]),
        hessian_condition_number=float(run_result["hessian_condition_number"]),
        n_negative_eigenvalues=int(run_result["n_negative_eigenvalues"]),
        fraction_clipped=float(run_result["fraction_clipped"]),
        a_err=float(run_result["a_err"]),
        crps_a=float(run_result["crps_a"]),
        coverage_95=float(run_result.get("coverage", run_result.get("coverage_95", float("nan")))),
        ci_width=float(run_result["ci_width"]),
        mean_std=float(run_result["mean_std"]),
        nll_a=float(run_result["nll_a"]) if "nll_a" in run_result else None,
        cal_levels=[float(v) for v in run_result["cal_levels"]],
        cal_empirical=[float(v) for v in run_result["cal_empirical"]],
        u_err=float(run_result["u_err"]) if "u_err" in run_result else None,
        label=str(run_result.get("label", "")),
        chi2_ppc=float(run_result["chi2_ppc"]) if "chi2_ppc" in run_result else None,
        chi2_ppc_pvalue=float(run_result["chi2_ppc_pvalue"]) if "chi2_ppc_pvalue" in run_result else None,
        map_a_err=float(run_result["map_a_err"]) if "map_a_err" in run_result else None,
        map_u_err=float(run_result["map_u_err"]) if "map_u_err" in run_result else None,
        map_time_s=run_result.get("map_time_s"),
        hessian_time_s=run_result.get("hessian_time_s"),
        sampling_time_s=run_result.get("sampling_time_s"),
        spearman_rho_error_std=run_result.get("spearman_rho_error_std"),
        spearman_pvalue_error_std=run_result.get("spearman_pvalue_error_std"),
    )


def build_prior_result(prior_dict: Dict) -> "PriorPredictiveResult":
    """Build a PriorPredictiveResult from a raw dict (e.g., from compute_prior_predictive)."""
    from results_schema import PriorPredictiveResult
    return PriorPredictiveResult(
        n_prior=int(prior_dict['n_prior']),
        a_err=float(prior_dict['a_err']),
        crps_a=float(prior_dict['crps_a']),
        coverage_95=float(prior_dict['coverage_95']),
        ci_width=float(prior_dict['ci_width']),
        mean_std=float(prior_dict['mean_std']),
    )


def save_experiment_result(
    result: "ExperimentResult",
    base_dir: Optional[Path] = None,
) -> Path:
    """Serialise an ExperimentResult to JSON.

    Output path: ``{base_dir}/{experiment}/{problem}_{timestamp}.json``
    Defaults to ``experiments/results/structured/`` relative to this file.

    Args:
        result: populated ExperimentResult dataclass.
        base_dir: override for the structured results root directory.

    Returns:
        Path to the written file.
    """
    from results_schema import ExperimentResult  # noqa: F401

    if base_dir is None:
        base_dir = Path(__file__).parent.parent / "results" / "structured"

    out_dir = base_dir / result.experiment
    out_dir.mkdir(parents=True, exist_ok=True)

    seed_suffix = f"_seed{result.seed}" if result.seed is not None else ""
    test_suffix = f"_test{result.test_idx}" if result.test_idx is not None else ""
    fname = f"{result.problem}_{result.timestamp.replace(':', '-').replace(' ', 'T')}{seed_suffix}{test_suffix}.json"
    out_path = out_dir / fname

    def _convert(obj):
        """Recursively convert numpy scalars/arrays to plain Python types."""
        if isinstance(obj, dict):
            return {k: _convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_convert(v) for v in obj]
        if hasattr(obj, "item"):          # numpy scalar
            return obj.item()
        if hasattr(obj, "tolist"):        # numpy array
            return obj.tolist()
        return obj

    data = _convert(asdict(result))
    out_path.write_text(json.dumps(data, indent=2))
    return out_path


def load_experiment_result(path: Path) -> "ExperimentResult":
    """Load an ExperimentResult from a JSON file written by save_experiment_result.

    Args:
        path: path to the JSON file.

    Returns:
        Reconstructed ExperimentResult dataclass.
    """
    from results_schema import (
        ExperimentResult, MCMCResult, LaplaceResult, PriorPredictiveResult,
    )

    raw = json.loads(Path(path).read_text())

    def _load_mcmc(d: Optional[Dict]) -> Optional[MCMCResult]:
        if d is None:
            return None
        return MCMCResult(**{k: v for k, v in d.items() if k in MCMCResult.__dataclass_fields__})

    def _load_laplace(d: Optional[Dict]) -> Optional[LaplaceResult]:
        if d is None:
            return None
        return LaplaceResult(**{k: v for k, v in d.items() if k in LaplaceResult.__dataclass_fields__})

    def _load_prior(d: Optional[Dict]) -> Optional[PriorPredictiveResult]:
        if d is None:
            return None
        return PriorPredictiveResult(**{k: v for k, v in d.items() if k in PriorPredictiveResult.__dataclass_fields__})

    conditions = None
    if raw.get("conditions") is not None:
        conditions = {k: _load_mcmc(v) for k, v in raw["conditions"].items()}

    sweep_conditions = None
    if raw.get("sweep_conditions") is not None:
        sweep_conditions = [_load_mcmc(v) for v in raw["sweep_conditions"]]

    return ExperimentResult(
        experiment=raw["experiment"],
        problem=raw["problem"],
        experiment_type=raw["experiment_type"],
        timestamp=raw["timestamp"],
        seed=raw.get("seed"),
        test_idx=raw.get("test_idx"),
        prior=_load_prior(raw.get("prior")),
        prior_ood=_load_prior(raw.get("prior_ood")),
        condition=_load_mcmc(raw.get("condition")),
        conditions=conditions,
        sweep_var=raw.get("sweep_var"),
        baseline=_load_mcmc(raw.get("baseline")),
        sweep_conditions=sweep_conditions,
        laplace=_load_laplace(raw.get("laplace")),
        map_time_s=raw.get("map_time_s"),
        total_time_s=raw.get("total_time_s"),
        sweep_value=raw.get("sweep_value"),
        is_sweep_baseline=raw.get("is_sweep_baseline"),
    )


# ---------------------------------------------------------------------------
# Cross-seed aggregation utilities
# ---------------------------------------------------------------------------

def load_cross_seed_results(
    experiment: str,
    problem: str,
    base_dir: Optional[Path] = None,
    test_idx: Optional[int] = None,
) -> List["ExperimentResult"]:
    """Load all seed results for a given experiment / problem combination.

    Finds JSON files in ``{base_dir}/{experiment}/`` whose names start with
    ``{problem}_`` and loads each as an ExperimentResult.  Results with
    ``seed=None`` (old files lacking the field) are included so that
    backward-compatibility is maintained.

    Args:
        experiment: experiment key, e.g. ``"baseline"``.
        problem: problem name, e.g. ``"darcy_continuous"``.
        base_dir: root of the structured results directory.  Defaults to
            ``experiments/results/structured/`` relative to this file.

    Returns:
        List of ExperimentResult, one per JSON file found, sorted by seed.
    """
    if base_dir is None:
        base_dir = Path(__file__).parent.parent / "results" / "structured"

    experiment_dir = base_dir / experiment
    if not experiment_dir.exists():
        return []

    results = [
        load_experiment_result(p)
        for p in sorted(experiment_dir.glob(f"{problem}_*.json"))
        if not p.stem.endswith("_aggregated")
    ]
    results = [r for r in results if r.problem == problem]
    # Sort: None seeds last, numbered seeds ascending
    results.sort(key=lambda r: (r.seed is None, r.seed if r.seed is not None else 0))
    if test_idx is not None:
        results = [r for r in results if r.test_idx == test_idx]
    return results


def cross_seed_metric_summary(
    results: List["ExperimentResult"],
    metric: str = "a_err",
    condition_key: Optional[str] = None,
) -> Dict[str, float]:
    """Compute mean ± std of a scalar metric across seeds.

    For ``"single"`` experiments, reads ``result.condition.<metric>``.
    For ``"comparison"`` experiments, pass ``condition_key`` to select which
    named condition to aggregate over.
    For ``"sweep"`` experiments, reads ``result.baseline.<metric>``.

    Args:
        results: list of ExperimentResult (one per seed).
        metric: attribute name on MCMCResult, e.g. ``"a_err"``, ``"ess_min"``.
        condition_key: for comparison experiments, the condition name to read.

    Returns:
        Dict with keys ``mean``, ``std``, ``min``, ``max``, ``n``, ``values``.
    """
    values = []
    for r in results:
        if r.experiment_type == "single":
            mcmc = r.condition
        elif r.experiment_type == "comparison":
            if condition_key is None:
                raise ValueError("condition_key required for comparison experiment_type")
            mcmc = (r.conditions or {})[condition_key]
        else:  # sweep
            mcmc = r.baseline
        if mcmc is not None:
            val = getattr(mcmc, metric)
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


def cross_seed_consistency_test(
    results: List["ExperimentResult"],
    metric: str = "a_err",
    condition_key: Optional[str] = None,
    cv_threshold: float = 0.20,
) -> Dict:
    """Test whether a metric is consistent across seeds using coefficient of variation.

    A CV below ``cv_threshold`` (default 20%) is considered consistent.

    Args:
        results: list of ExperimentResult (one per seed).
        metric: attribute name on MCMCResult.
        condition_key: for comparison experiments, the condition name to read.
        cv_threshold: CV fraction at which results are flagged as inconsistent.

    Returns:
        Dict with ``cv``, ``consistent`` (bool), ``summary`` (from
        cross_seed_metric_summary), and a human-readable ``interpretation``.
    """
    summary = cross_seed_metric_summary(results, metric=metric, condition_key=condition_key)
    cv = summary["std"] / abs(summary["mean"]) if summary["mean"] != 0 else float("inf")
    consistent = cv < cv_threshold
    interpretation = (
        f"{metric}: mean={summary['mean']:.4f}, std={summary['std']:.4f}, "
        f"CV={cv:.2%} → {'CONSISTENT' if consistent else 'INCONSISTENT'} "
        f"(threshold {cv_threshold:.0%})"
    )
    return {
        "metric": metric,
        "cv": float(cv),
        "consistent": consistent,
        "summary": summary,
        "interpretation": interpretation,
    }


def cross_seed_comparison_robustness(
    results: List["ExperimentResult"],
    condition_a: str,
    condition_b: str,
    metric: str = "a_err",
) -> Dict:
    """Test whether a pairwise comparison (RQ2) is robust across seeds.

    For each seed's ExperimentResult (comparison type), computes
    ``condition_a[metric] - condition_b[metric]`` and checks whether the
    difference is consistently signed across all seeds (sign consistency).
    Also runs a one-sample Wilcoxon signed-rank test on the differences.

    Args:
        results: list of ExperimentResult with ``experiment_type="comparison"``.
        condition_a: first condition name (e.g. ``"with_pde"``).
        condition_b: second condition name (e.g. ``"without_pde"``).
        metric: attribute name on MCMCResult.

    Returns:
        Dict with ``differences``, ``sign_consistent``, ``wilcoxon_p``,
        ``interpretation``.
    """
    from scipy.stats import wilcoxon as _wilcoxon

    diffs = []
    for r in results:
        conds = r.conditions or {}
        raw_a = getattr(conds[condition_a], metric)
        raw_b = getattr(conds[condition_b], metric)
        if raw_a is None or raw_b is None:
            continue
        diffs.append(float(raw_a) - float(raw_b))

    if not diffs:
        return {
            "metric": metric, "differences": [], "mean_diff": float("nan"),
            "sign_consistent": False, "wilcoxon_stat": float("nan"),
            "wilcoxon_p": float("nan"),
            "interpretation": f"{metric}: no non-None values to compare",
        }
    diffs_arr = np.array(diffs)
    sign_consistent = bool(np.all(diffs_arr > 0) or np.all(diffs_arr < 0))

    if len(diffs_arr) >= 2:
        try:
            stat, p_val = _wilcoxon(diffs_arr)
        except ValueError:
            stat, p_val = float("nan"), float("nan")
    else:
        stat, p_val = float("nan"), float("nan")

    direction = f"{condition_a} {'>' if np.mean(diffs_arr) > 0 else '<'} {condition_b}"
    interpretation = (
        f"{metric} diff ({condition_a} - {condition_b}): "
        f"mean={np.mean(diffs_arr):.4f}, "
        f"sign_consistent={sign_consistent}, "
        f"Wilcoxon p={p_val:.3f} → "
        f"{'ROBUST' if sign_consistent else 'NOT ROBUST'} ({direction})"
    )
    return {
        "metric": metric,
        "differences": diffs,
        "mean_diff": float(np.mean(diffs_arr)),
        "sign_consistent": sign_consistent,
        "wilcoxon_stat": float(stat),
        "wilcoxon_p": float(p_val),
        "interpretation": interpretation,
    }


def format_cross_seed_table(
    experiment: str,
    problems: List[str],
    metrics: Optional[List[str]] = None,
    condition_key: Optional[str] = None,
    base_dir: Optional[Path] = None,
    test_idx: Optional[int] = None,
) -> str:
    """Format a cross-seed summary table as a plain-text string.

    Loads results for each problem, computes mean ± std for each metric,
    and returns a formatted table suitable for printing or embedding in a
    report.

    Args:
        experiment: experiment key, e.g. ``"baseline"``.
        problems: list of problem names to include as rows.
        metrics: list of MCMCResult attributes to show as columns.
            Defaults to ``["a_err", "ess_min", "coverage_95", "n_div"]``.
        condition_key: passed to cross_seed_metric_summary for comparison
            experiments.
        base_dir: override for the structured results root directory.
        test_idx: if given, only include results for this test index.

    Returns:
        Formatted table string.
    """
    if metrics is None:
        metrics = ["a_err", "ess_min", "coverage_95", "n_div"]

    col_w = 18
    header = f"{'Problem':<28}" + "".join(f"{m:>{col_w}}" for m in metrics)
    sep = "-" * len(header)
    rows = [f"{experiment.upper()} Cross-Seed Summary", sep, header, sep]

    for prob in problems:
        res = load_cross_seed_results(experiment, prob, base_dir=base_dir, test_idx=test_idx)
        if not res:
            row = f"{prob:<28}" + "".join(f"{'(no data)':>{col_w}}" for _ in metrics)
        else:
            cells = []
            for m in metrics:
                try:
                    s = cross_seed_metric_summary(res, metric=m, condition_key=condition_key)
                    n = s["n"]
                    cells.append(f"{s['mean']:.4f}±{s['std']:.4f}(n={n})")
                except (KeyError, AttributeError, TypeError):
                    cells.append("error")
            row = f"{prob:<28}" + "".join(f"{c:>{col_w}}" for c in cells)
        rows.append(row)

    rows.append(sep)
    return "\n".join(rows)


# Rename for clarity (keep old name as alias for transition)
build_unconditional_prior_result = build_prior_result


def print_cross_seed_summary(experiment, problem_name):
    """Load and print a cross-seed summary table for one experiment/problem.

    Args:
        experiment: experiment key, e.g. "baseline"
        problem_name: problem name, e.g. "darcy_continuous"
    """
    results = load_cross_seed_results(experiment, problem_name)
    if len(results) <= 1:
        print(f"Only {len(results)} seed result(s) found — skipping cross-seed summary")
        return

    exp_type = results[0].experiment_type
    if exp_type == "comparison":
        condition_keys = list((results[0].conditions or {}).keys())
    else:
        condition_keys = [None]

    for cond_key in condition_keys:
        if cond_key is not None:
            print(f"\n--- Condition: {cond_key} ---")
        print(f"Cross-Seed Summary ({len(results)} seeds: {[r.seed for r in results]})\n")
        print(f"{'Metric':<16s}  {'Mean':>10s}  {'Std':>10s}  {'Min':>10s}  {'Max':>10s}")
        print("-" * 62)
        for m in ["a_err", "u_err", "crps_a", "coverage_95", "ci_width", "mean_std",
                  "ess_min", "rhat_max", "n_div"]:
            try:
                s = cross_seed_metric_summary(results, metric=m, condition_key=cond_key)
                if s["mean"] is not None:
                    print(f"{m:<16s}  {s['mean']:>10.4f}  {s['std']:>10.4f}  "
                          f"{s['min']:>10.4f}  {s['max']:>10.4f}")
            except (AttributeError, KeyError, TypeError):
                pass


def print_per_chain_table(chain_metrics, num_chains):
    """Print a formatted per-chain metrics table.

    Args:
        chain_metrics: dict from compute_per_chain_metrics
        num_chains: number of chains
    """
    print("\nPer-chain consistency:")
    print(f"  {'Chain':>5s}  {'CRPS':>8s}  {'Cov95':>8s}  {'CI W':>8s}  {'Sharp':>8s}")
    for c in range(num_chains):
        print(f"  {c+1:5d}  {chain_metrics['crps'][c]:.5f}  "
              f"{chain_metrics['coverage_95'][c]:.4f}  "
              f"{chain_metrics['ci_width'][c]:.5f}  "
              f"{chain_metrics['sharpness'][c]:.5f}")


def print_dimension_diagnostics(beta_samples_np, ess, rhat):
    """Print per-dimension ESS, R-hat, mean, and std.

    Args:
        beta_samples_np: (n_samples, d) numpy array of beta samples
        ess: per-dimension ESS array
        rhat: per-dimension R-hat array
    """
    import numpy as _np
    d = beta_samples_np.shape[1]
    print("Per-dimension diagnostics:")
    print(f"{'dim':>4s}  {'ESS':>8s}  {'R-hat':>8s}  {'mean':>10s}  {'std':>10s}")
    for i in range(d):
        print(f"{i:4d}  {float(_np.array(ess)[i] if hasattr(ess, '__getitem__') else ess):8.1f}  "
              f"{float(_np.array(rhat)[i] if hasattr(rhat, '__getitem__') else rhat):8.4f}  "
              f"{float(beta_samples_np[:, i].mean()):10.4f}  "
              f"{float(beta_samples_np[:, i].std()):10.4f}")
