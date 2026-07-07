"""Statistical metrics, bootstrap CIs, and MCMC reliability diagnostics."""
import numpy as np
from typing import Dict, List, Optional, Tuple

def crps_ensemble(samples: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Continuous Ranked Probability Score (energy form, exact for finite ensembles).

    Args:
        samples: (n_samples, n_points) ensemble predictions
        truth: (n_points,) ground truth

    Returns:
        crps: (n_points,) per-point CRPS values
    """
    sorted_samples = np.sort(samples, axis=0)
    n = samples.shape[0]
    mae = np.mean(np.abs(samples - truth[None, :]), axis=0)
    indices = np.arange(1, n + 1)
    weights = 2 * indices - n - 1
    spread = np.sum(weights[:, None] * sorted_samples, axis=0) / (n * n)
    return mae - spread


def compute_calibration(
    samples: np.ndarray,
    truth: np.ndarray,
    levels: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute empirical coverage at nominal credible interval levels.

    Args:
        samples: (n_samples, n_points) ensemble predictions
        truth: (n_points,) ground truth
        levels: nominal coverage levels (default: 10 levels from 0.1 to 0.95)

    Returns:
        (nominal_levels, empirical_coverages) both shape (n_levels,)
    """
    if levels is None:
        levels = np.linspace(0.1, 0.95, 10)

    empirical = np.empty_like(levels)
    for i, level in enumerate(levels):
        alpha = (1 - level) / 2
        lo = np.percentile(samples, 100 * alpha, axis=0)
        hi = np.percentile(samples, 100 * (1 - alpha), axis=0)
        empirical[i] = np.mean((truth >= lo) & (truth <= hi))

    return levels, empirical


def ci_width_95(samples: np.ndarray) -> float:
    """Mean 95% credible interval width across spatial locations."""
    lo = np.percentile(samples, 2.5, axis=0)
    hi = np.percentile(samples, 97.5, axis=0)
    return float(np.mean(hi - lo))


def chi2_ppc(obs_values: np.ndarray, pred_samples: np.ndarray, sigma: float) -> tuple:
    """Chi-squared posterior predictive check.

    Tests whether the posterior mean predictions at observation locations
    are consistent with the assumed Gaussian noise model.

    chi2 = sum((obs - pred_mean)^2) / sigma^2  ~  chi2(N_obs)

    Args:
        obs_values: (n_obs,) observed values
        pred_samples: (n_samples, n_obs) posterior predicted values at obs locations
        sigma: assumed observation noise std

    Returns:
        (chi2_stat, p_value) where p_value is from scipy.stats.chi2.sf
    """
    from scipy.stats import chi2
    pred_mean = np.mean(pred_samples, axis=0)
    n_obs = obs_values.shape[0]
    chi2_stat = float(np.sum((obs_values - pred_mean) ** 2) / sigma ** 2)
    p_value = float(chi2.sf(chi2_stat, df=n_obs))
    return chi2_stat, p_value


def nll_score(samples: np.ndarray, truth: np.ndarray) -> float:
    """Mean Negative Log-Likelihood (Gaussian log score) across spatial grid.

    Treats each spatial point independently using the marginal posterior (mean, std).
    NLL_i = 0.5*log(2*pi*sigma_i^2) + (truth_i - mean_i)^2 / (2*sigma_i^2)

    Args:
        samples: (n_samples, n_points) ensemble predictions
        truth: (n_points,) ground truth

    Returns:
        Scalar mean NLL averaged over spatial points.
    """
    mean = np.mean(samples, axis=0)
    std = np.std(samples, axis=0)
    std = np.maximum(std, 1e-12)
    nll_per_point = 0.5 * np.log(2 * np.pi * std**2) + (truth - mean)**2 / (2 * std**2)
    return float(np.mean(nll_per_point))


def bernoulli_nll_score(samples: np.ndarray, truth: np.ndarray,
                        k_low: float = 5.0, k_high: float = 10.0) -> float:
    """Bernoulli NLL for piecewise constant fields with values {k_low, k_high}."""
    p = np.mean(samples >= ((k_low + k_high) / 2), axis=0).astype(np.float64)
    p = np.clip(p, 1e-6, 1 - 1e-6)
    y = (truth - k_low) / (k_high - k_low)
    nll_per_point = -(y * np.log(p) + (1 - y) * np.log(1 - p))
    return float(np.mean(nll_per_point))


def compute_prior_predictive(
    prior_a_samples: np.ndarray,
    a_true: np.ndarray,
    error_fn=None,
) -> Dict[str, float]:
    """Compute prior predictive metrics (NF prior, no data conditioning).

    Computes the same quality metrics used for the posterior, but on samples
    drawn from the NF prior. This quantifies information gain from observations.

    Args:
        prior_a_samples: decoded a-field samples from NF prior, shape (n_prior, n_points)
        a_true: ground truth a-field, shape (n_points,)
        error_fn: optional callable(pred_mean, true) -> float. Default uses rmse
                  from src.evaluation.metrics. Pass cross_correlation for darcy_piecewise.

    Returns:
        dict with keys: n_prior, a_err, crps_a, coverage_95, ci_width, mean_std
    """
    import jax.numpy as jnp
    from src.evaluation.metrics import rmse as _rmse

    if error_fn is None:
        error_fn = lambda pred, true: float(_rmse(jnp.array(pred), jnp.array(true)))

    n_prior = prior_a_samples.shape[0]
    prior_mean = np.mean(prior_a_samples, axis=0)

    crps_vals = crps_ensemble(prior_a_samples, a_true)
    crps_a = float(np.mean(crps_vals))

    _, cal_emp = compute_calibration(prior_a_samples, a_true, np.array([0.95]))
    coverage_95 = float(cal_emp[0])

    ci_w = ci_width_95(prior_a_samples)

    mean_std = float(np.mean(np.std(prior_a_samples, axis=0)))

    a_err = error_fn(prior_mean, a_true)

    return {
        'n_prior': n_prior,
        'a_err': a_err,
        'crps_a': crps_a,
        'coverage_95': coverage_95,
        'ci_width': float(ci_w),
        'mean_std': mean_std,
    }


# ---------------------------------------------------------------------------
# MCMC Reliability Diagnostics
# ---------------------------------------------------------------------------

ESS_WARN_THRESHOLD = 50     # per-dimension minimum ESS — warn below this
ESS_FAIL_THRESHOLD = 10     # below this, results are unusable
RHAT_WARN_THRESHOLD = 1.05  # standard convergence threshold
RHAT_FAIL_THRESHOLD = 1.2   # severe non-convergence
DIV_WARN_FRAC = 0.01        # >1% divergences is concerning
DIV_FAIL_FRAC = 0.05        # >5% divergences is a failure


def mcmc_reliability_flag(
    ess_min: float,
    rhat_max: float,
    n_divergences: int,
    total_samples: int,
) -> Tuple[str, str]:
    """Classify MCMC run reliability based on standard diagnostics.

    Args:
        ess_min: Minimum effective sample size across latent dimensions
        rhat_max: Maximum R-hat (split Gelman-Rubin) across dimensions
        n_divergences: Total number of divergent transitions
        total_samples: Total posterior samples drawn (num_chains × num_samples)

    Returns:
        (flag, explanation) where flag is 'PASS', 'WARN', or 'FAIL'
    """
    issues = []
    flag = 'PASS'
    div_frac = n_divergences / max(total_samples, 1)

    if ess_min < ESS_FAIL_THRESHOLD:
        flag = 'FAIL'
        issues.append(f'ESS_min={ess_min:.1f} < {ESS_FAIL_THRESHOLD}')
    elif ess_min < ESS_WARN_THRESHOLD:
        flag = 'WARN'
        issues.append(f'ESS_min={ess_min:.1f} < {ESS_WARN_THRESHOLD}')

    if rhat_max > RHAT_FAIL_THRESHOLD:
        flag = 'FAIL'
        issues.append(f'R-hat_max={rhat_max:.3f} > {RHAT_FAIL_THRESHOLD}')
    elif rhat_max > RHAT_WARN_THRESHOLD:
        if flag != 'FAIL':
            flag = 'WARN'
        issues.append(f'R-hat_max={rhat_max:.3f} > {RHAT_WARN_THRESHOLD}')

    if div_frac > DIV_FAIL_FRAC:
        flag = 'FAIL'
        issues.append(f'divergences={div_frac:.1%} > {DIV_FAIL_FRAC:.0%}')
    elif div_frac > DIV_WARN_FRAC:
        if flag != 'FAIL':
            flag = 'WARN'
        issues.append(f'divergences={div_frac:.1%} > {DIV_WARN_FRAC:.0%}')

    explanation = '; '.join(issues) if issues else 'All diagnostics within thresholds'
    return flag, explanation


# ---------------------------------------------------------------------------
# Statistical Significance Testing
# ---------------------------------------------------------------------------

def bootstrap_metric_ci(
    samples: np.ndarray,
    truth: np.ndarray,
    metric_fn,
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, float]:
    """Bootstrap confidence interval for a posterior-based metric.

    Resamples posterior samples (rows) with replacement and recomputes the
    metric each time, yielding a distribution of the metric under Monte Carlo
    variability.

    Args:
        samples: (n_samples, n_points) posterior samples
        truth: (n_points,) ground truth
        metric_fn: callable(resampled_samples, truth) -> float
        n_bootstrap: number of bootstrap iterations
        ci_level: confidence level (default 0.95)
        rng: numpy random generator (created if None)

    Returns:
        dict with keys 'estimate', 'ci_lo', 'ci_hi', 'std'
    """
    if rng is None:
        rng = np.random.default_rng(0)
    n = samples.shape[0]
    estimates = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        estimates[i] = metric_fn(samples[idx], truth)
    alpha = (1 - ci_level) / 2
    return {
        'estimate': float(metric_fn(samples, truth)),
        'ci_lo': float(np.percentile(estimates, 100 * alpha)),
        'ci_hi': float(np.percentile(estimates, 100 * (1 - alpha))),
        'std': float(np.std(estimates)),
    }


def bootstrap_metric_difference_ci(
    samples_a: np.ndarray,
    samples_b: np.ndarray,
    truth: np.ndarray,
    metric_fn,
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, float]:
    """Bootstrap CI on the difference metric(A) - metric(B).

    If the CI excludes zero the difference is significant at the given level.

    Args:
        samples_a: (n_samples_a, n_points) posterior samples for condition A
        samples_b: (n_samples_b, n_points) posterior samples for condition B
        truth: (n_points,) ground truth
        metric_fn: callable(samples, truth) -> float
        n_bootstrap: number of bootstrap iterations
        ci_level: confidence level (default 0.95)
        rng: numpy random generator

    Returns:
        dict with 'mean_diff', 'ci_lo', 'ci_hi', 'significant'
    """
    if rng is None:
        rng = np.random.default_rng(0)
    na, nb = samples_a.shape[0], samples_b.shape[0]
    diffs = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx_a = rng.integers(0, na, size=na)
        idx_b = rng.integers(0, nb, size=nb)
        diffs[i] = metric_fn(samples_a[idx_a], truth) - metric_fn(samples_b[idx_b], truth)
    alpha = (1 - ci_level) / 2
    ci_lo = float(np.percentile(diffs, 100 * alpha))
    ci_hi = float(np.percentile(diffs, 100 * (1 - alpha)))
    return {
        'mean_diff': float(metric_fn(samples_a, truth) - metric_fn(samples_b, truth)),
        'ci_lo': ci_lo,
        'ci_hi': ci_hi,
        'significant': not (ci_lo <= 0 <= ci_hi),
    }


def compute_per_chain_metrics(
    a_samples_all: np.ndarray,
    a_true: np.ndarray,
    n_chains: int,
    k_low: Optional[float] = None,
    k_high: Optional[float] = None,
    a_err_fn=None,
) -> Dict[str, np.ndarray]:
    """Compute key posterior metrics separately for each MCMC chain.

    NumPyro concatenates chains in order, so reshaping recovers per-chain
    samples without re-decoding.

    Args:
        a_samples_all: (n_chains * n_samples_per_chain, n_points) pooled samples
        a_true: (n_points,) ground truth
        n_chains: number of chains

    Returns:
        dict mapping metric name -> ndarray of shape (n_chains,)
    """
    total = a_samples_all.shape[0]
    n_per_chain = total // n_chains
    by_chain = a_samples_all[:n_per_chain * n_chains].reshape(n_chains, n_per_chain, -1)

    results: Dict[str, np.ndarray] = {k: np.empty(n_chains) for k in
                                       ['crps', 'coverage_95', 'ci_width', 'sharpness', 'rel_l2']}
    for c in range(n_chains):
        s = by_chain[c]  # (n_per_chain, n_points)
        results['crps'][c] = float(np.mean(crps_ensemble(s, a_true)))
        _, emp = compute_calibration(s, a_true, np.array([0.95]))
        results['coverage_95'][c] = float(emp[0])
        results['ci_width'][c] = ci_width_95(s)
        results['sharpness'][c] = float(np.mean(np.std(s, axis=0)))
        if k_low is not None and k_high is not None and a_err_fn is not None:
            midpoint = (k_low + k_high) / 2
            chain_mean_thresh = np.where(np.mean(s, axis=0) >= midpoint, k_high, k_low)
            results['rel_l2'][c] = float(a_err_fn(chain_mean_thresh, a_true))
        else:
            norm = np.linalg.norm(a_true)
            results['rel_l2'][c] = float(np.linalg.norm(np.mean(s, axis=0) - a_true) / (norm + 1e-12))
    return results


def compute_metric_convergence(
    a_samples: np.ndarray,
    a_true: np.ndarray,
    sample_counts: Optional[np.ndarray] = None,
    k_low: Optional[float] = None,
    k_high: Optional[float] = None,
    a_err_fn=None,
) -> Dict[str, np.ndarray]:
    """Compute posterior metrics as a function of number of samples used.

    Useful for demonstrating that metrics have converged despite moderate ESS.

    Args:
        a_samples: (n_total, n_points) posterior samples
        a_true: (n_points,) ground truth
        sample_counts: evaluation points; if None, ~20 log-spaced from 50 to n_total

    Returns:
        dict with keys: sample_counts, crps_a, coverage_95, ci_width, a_err,
        mean_std (all ndarrays), and n_total (int)
    """
    n_total = a_samples.shape[0]

    if sample_counts is None:
        if n_total < 50:
            sample_counts = np.array([n_total])
        else:
            sample_counts = np.unique(np.geomspace(50, n_total, 20).astype(int))
            if sample_counts[-1] != n_total:
                sample_counts = np.append(sample_counts, n_total)

    crps_arr = np.empty(len(sample_counts))
    cov_arr = np.empty(len(sample_counts))
    ciw_arr = np.empty(len(sample_counts))
    aerr_arr = np.empty(len(sample_counts))
    std_arr = np.empty(len(sample_counts))

    for i, N in enumerate(sample_counts):
        prefix = a_samples[:N]
        crps_arr[i] = float(np.mean(crps_ensemble(prefix, a_true)))
        _, emp = compute_calibration(prefix, a_true, np.array([0.95]))
        cov_arr[i] = float(emp[0])
        ciw_arr[i] = ci_width_95(prefix)
        if k_low is not None and k_high is not None and a_err_fn is not None:
            midpoint = (k_low + k_high) / 2
            prefix_thresh = np.where(prefix.mean(axis=0) >= midpoint, k_high, k_low)
            aerr_arr[i] = float(a_err_fn(prefix_thresh, a_true))
        else:
            aerr_arr[i] = float(
                np.linalg.norm(prefix.mean(axis=0) - a_true)
                / (np.linalg.norm(a_true) + 1e-12)
            )
        std_arr[i] = float(np.mean(np.std(prefix, axis=0)))

    return {
        'sample_counts': sample_counts,
        'crps_a': crps_arr,
        'coverage_95': cov_arr,
        'ci_width': ciw_arr,
        'a_err': aerr_arr,
        'mean_std': std_arr,
        'n_total': n_total,
    }


# Rename for clarity (keep old name as alias for transition)
compute_unconditional_prior_metrics = compute_prior_predictive


def compute_standard_metrics(a_samples_np, a_true_np, a_mean_np=None, a_err_fn=None):
    """Compute standard posterior quality metrics in one call.

    Args:
        a_samples_np: (n_samples, n_points) posterior a-field samples
        a_true_np: (n_points,) ground truth a-field
        a_mean_np: (n_points,) posterior mean; computed if None
        a_err_fn: callable(pred_mean, truth) -> float; defaults to relative L2

    Returns:
        dict with keys: a_err, crps_a, nll_a, cal_levels, cal_empirical,
        coverage_95, ci_width, mean_std
    """
    if a_mean_np is None:
        a_mean_np = np.mean(a_samples_np, axis=0)

    if a_err_fn is None:
        def a_err_fn(pred, true):
            norm = np.linalg.norm(true)
            return float(np.linalg.norm(pred - true) / (norm + 1e-12))

    a_err = a_err_fn(a_mean_np, a_true_np)
    crps_a = float(np.mean(crps_ensemble(a_samples_np, a_true_np)))
    nll_a = nll_score(a_samples_np, a_true_np)
    cal_levels, cal_empirical = compute_calibration(a_samples_np, a_true_np)
    coverage_95 = float(cal_empirical[-1])
    ci_w = ci_width_95(a_samples_np)
    mean_std = float(np.mean(np.std(a_samples_np, axis=0)))

    return {
        'a_err': a_err,
        'crps_a': crps_a,
        'nll_a': nll_a,
        'cal_levels': cal_levels,
        'cal_empirical': cal_empirical,
        'coverage_95': coverage_95,
        'ci_width': float(ci_w),
        'mean_std': mean_std,
    }


def compute_piecewise_metrics(a_samples_np, a_true_np, k_low, k_high, a_err_fn,
                               n_bootstrap=1000, rng_seed=1):
    """Compute posterior metrics for piecewise constant fields."""
    midpoint = (k_low + k_high) / 2
    a_mean = np.mean(a_samples_np, axis=0)
    a_mean_thresholded = np.where(a_mean >= midpoint, k_high, k_low)

    a_err = a_err_fn(a_mean_thresholded, a_true_np)

    per_sample_errs = [a_err_fn(a_samples_np[i], a_true_np)
                       for i in range(a_samples_np.shape[0])]
    a_err_per_sample = float(np.mean(per_sample_errs))

    nll_a = bernoulli_nll_score(a_samples_np, a_true_np, k_low, k_high)

    crps_a = float(np.mean(crps_ensemble(a_samples_np, a_true_np)))
    cal_levels, cal_empirical = compute_calibration(a_samples_np, a_true_np)
    coverage_95 = float(cal_empirical[-1])
    ci_w = ci_width_95(a_samples_np)
    mean_std = float(np.mean(np.std(a_samples_np, axis=0)))

    def _a_err_boot(s, t):
        s_mean = np.where(np.mean(s, axis=0) >= midpoint, k_high, k_low)
        return a_err_fn(s_mean, t)
    def _crps_a(s, t): return float(np.mean(crps_ensemble(s, t)))
    def _coverage_95(s, t):
        _, emp = compute_calibration(s, t, np.array([0.95]))
        return float(emp[0])
    def _ci_width(s, t): return ci_width_95(s)
    def _sharpness(s, t): return float(np.mean(np.std(s, axis=0)))

    rng = np.random.default_rng(rng_seed)
    bootstrap_ci = {
        'a_err':       bootstrap_metric_ci(a_samples_np, a_true_np, _a_err_boot, n_bootstrap=n_bootstrap, rng=rng),
        'crps_a':      bootstrap_metric_ci(a_samples_np, a_true_np, _crps_a, n_bootstrap=n_bootstrap, rng=rng),
        'coverage_95': bootstrap_metric_ci(a_samples_np, a_true_np, _coverage_95, n_bootstrap=n_bootstrap, rng=rng),
        'ci_width':    bootstrap_metric_ci(a_samples_np, a_true_np, _ci_width, n_bootstrap=n_bootstrap, rng=rng),
        'sharpness':   bootstrap_metric_ci(a_samples_np, a_true_np, _sharpness, n_bootstrap=n_bootstrap, rng=rng),
    }

    return {
        'a_err': a_err,
        'a_err_per_sample': a_err_per_sample,
        'nll_a': nll_a,
        'crps_a': crps_a,
        'cal_levels': cal_levels,
        'cal_empirical': cal_empirical,
        'coverage_95': coverage_95,
        'ci_width': float(ci_w),
        'mean_std': mean_std,
        'bootstrap_ci': bootstrap_ci,
    }


def compute_bootstrap_ci_block(a_samples_np, a_true_np, rng_seed=1):
    """Compute bootstrap CIs for the 4 standard metrics.

    Returns:
        dict with keys 'crps_a', 'coverage_95', 'ci_width', 'sharpness',
        each a dict from bootstrap_metric_ci.
    """
    def _crps_a(s, t): return float(np.mean(crps_ensemble(s, t)))
    def _coverage_95(s, t):
        _, emp = compute_calibration(s, t, np.array([0.95]))
        return float(emp[0])
    def _ci_width(s, t): return ci_width_95(s)
    def _sharpness(s, t): return float(np.mean(np.std(s, axis=0)))

    rng = np.random.default_rng(rng_seed)
    return {
        'crps_a':      bootstrap_metric_ci(a_samples_np, a_true_np, _crps_a,      rng=rng),
        'coverage_95': bootstrap_metric_ci(a_samples_np, a_true_np, _coverage_95, rng=rng),
        'ci_width':    bootstrap_metric_ci(a_samples_np, a_true_np, _ci_width,    rng=rng),
        'sharpness':   bootstrap_metric_ci(a_samples_np, a_true_np, _sharpness,   rng=rng),
    }
