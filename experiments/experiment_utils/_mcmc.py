"""MCMC configuration, sigma tuning, and NF reparameterization."""
import numpy as np
from typing import Dict, List, Tuple

from ._metrics import compute_calibration, ci_width_95

def recommended_nuts_config(d: int, sigma: float) -> Dict:
    """Recommend NUTS hyperparameters based on latent dimension and observation noise.

    A smaller sigma creates a tighter (sharper) likelihood, requiring smaller step sizes
    and thus a higher target acceptance probability. High dimensions need lower target
    acceptance to avoid excessively short trajectories.

    Args:
        d: latent dimension
        sigma: observation noise standard deviation

    Returns:
        dict with 'target_accept_prob', 'max_tree_depth', and 'dense_mass'
    """
    if d <= 10:
        tap = 0.95 if sigma < 0.02 else 0.85
        max_depth = 10
        dense_mass = True
    elif d <= 50:
        tap = 0.85
        max_depth = 10
        dense_mass = False
    else:  # d=128+
        tap = 0.65
        max_depth = 8
        dense_mass = False
    return {'target_accept_prob': tap, 'max_tree_depth': max_depth, 'dense_mass': dense_mass}


def tune_sigma(
    model_fn_factory,
    beta_mode,
    sigma_candidates: List[float],
    rng_key,
    decode_fn,
    a_true: np.ndarray,
    pilot_warmup: int = 2000,
    pilot_samples: int = 500,
    pilot_chains: int = 2,
    target_accept_prob: float = 0.8,
    ess_min_threshold: float = 20.0,
    verbose: bool = True,
    sample_name: str = 'beta',
) -> Tuple[float, Dict]:
    """Select sigma by running pilot MCMC per candidate and picking closest to 95% coverage.

    Asymmetric criterion: over-coverage penalised 2x (short pilots overestimate coverage).
    """
    # Import here to avoid circular/conditional imports at module level
    from numpyro.infer import MCMC, NUTS, init_to_value
    from numpyro.diagnostics import effective_sample_size, split_gelman_rubin

    results = {}

    if verbose:
        print(f"  Sigma tuning ({len(sigma_candidates)} candidates, "
              f"{pilot_warmup} warmup + {pilot_samples} samples × {pilot_chains} chains):")

    for sigma in sigma_candidates:
        import jax
        rng_key, key = jax.random.split(rng_key)
        model_fn = model_fn_factory(sigma)

        kernel = NUTS(
            model_fn,
            init_strategy=init_to_value(values={sample_name: beta_mode}),
            target_accept_prob=target_accept_prob,
        )
        mcmc_pilot = MCMC(
            kernel,
            num_warmup=pilot_warmup,
            num_samples=pilot_samples,
            num_chains=pilot_chains,
            chain_method='sequential',
            progress_bar=False,
        )
        mcmc_pilot.run(key)

        beta_s = mcmc_pilot.get_samples()[sample_name]
        n_div = int(mcmc_pilot.get_extra_fields().get("diverging", np.array([])).sum())

        # ESS/R-hat (requires >=2 chains)
        if pilot_chains >= 2:
            beta_by_chain = np.array(mcmc_pilot.get_samples(group_by_chain=True)[sample_name])
            ess = effective_sample_size(beta_by_chain)
            rhat = split_gelman_rubin(beta_by_chain)
            ess_val = float(np.array(ess).min())
            rhat_val = float(np.array(rhat).max())
        else:
            ess_val = float('inf')
            rhat_val = 1.0

        # Decode to coefficient field and compute coverage
        a_samples = decode_fn(beta_s)  # (n_samples, n_points)
        _, emp = compute_calibration(a_samples, a_true, np.array([0.95]))
        width = ci_width_95(a_samples)

        results[sigma] = {
            'coverage': float(emp[0]),
            'ci_width': width,
            'ess_min': ess_val,
            'rhat_max': rhat_val,
            'n_div': n_div,
        }

        if verbose:
            viable = ess_val >= ess_min_threshold
            print(f"    sigma={sigma:.4f}  cov={float(emp[0]):.3f}  "
                  f"ESS_min={ess_val:.1f}  R-hat={rhat_val:.3f}  "
                  f"div={n_div}  {'OK' if viable else 'LOW-ESS'}")

    # Select among candidates with good ESS, preferring coverage closest to 95%.
    # Over-coverage is penalised 2× because short pilots tend to overestimate it.
    def _coverage_loss(cov: float, target: float = 0.95) -> float:
        err = cov - target
        return 2.0 * err if err > 0 else -err

    viable = {s: r for s, r in results.items() if r['ess_min'] >= ess_min_threshold}

    if not viable:
        print(f"  WARNING: No sigma candidate achieved ESS_min >= {ess_min_threshold}. "
              f"Selecting by highest ESS. Consider longer warmup or different candidates.")
        best = max(results.keys(), key=lambda s: results[s]['ess_min'])
    else:
        best = min(viable.keys(), key=lambda s: _coverage_loss(viable[s]['coverage']))

    if verbose:
        r = results[best]
        print(f"  Selected sigma={best}  (coverage={r['coverage']:.3f}, "
              f"ESS_min={r['ess_min']:.1f})")

    return best, results


def compute_sigma_from_map(
    u_pred_map: np.ndarray,
    u_obs: np.ndarray,
    noise_sigma: float = 0.0,
) -> float:
    """sigma = max(noise_sigma, RMSE(u_pred_map - u_obs))."""
    u_pred_flat = np.array(u_pred_map).ravel()
    u_obs_flat = np.array(u_obs).ravel()
    residual_rmse = float(np.sqrt(np.mean((u_pred_flat - u_obs_flat) ** 2)))
    return float(max(noise_sigma, residual_rmse))


def select_sweep_sigma(per_condition_sigmas: Dict, strategy: str = 'median') -> float:
    """Pick a single sigma across all sweep conditions to prevent confounding."""
    values = list(per_condition_sigmas.values())
    if strategy == 'median':
        return float(np.median(values))
    elif strategy == 'max':
        return float(np.max(values))
    else:
        raise ValueError(f"Unknown strategy: {strategy!r}. Use 'median' or 'max'.")


def make_nf_reparameterized_model(
    nf_model,
    nf_params: Dict,
    log_likelihood_fn,
    d: int,
    nf_alpha: float,
):
    """NUTS model reparameterized to sample in NF z-space instead of beta-space.

    p(z | data) ∝ p_z(z) × L(data | inverse(z)) — Jacobians cancel exactly,
    so only the data likelihood appears as a factor.
    """
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist

    def numpyro_model_z(sigma=0.1):
        z_01 = numpyro.sample("z", dist.Beta(nf_alpha, nf_alpha).expand([d]).to_event(1))
        z = 2.0 * z_01 - 1.0
        beta, _ = nf_model.apply(
            {'params': nf_params}, z[None, :], method=nf_model.inverse
        )
        beta = beta[0]
        # The NF prior is absorbed into the Beta sampling distribution, so no extra factor.
        numpyro.factor("data_lik", log_likelihood_fn(beta, sigma))

    return numpyro_model_z


def run_mcmc(model_fn, init_values, model_kwargs, mcmc_key,
             num_warmup, num_samples, num_chains,
             chain_method='vectorized', nuts_config=None):
    """Run NUTS MCMC with separate warmup/sampling timing.

    Args:
        model_fn: NumPyro model callable
        init_values: dict mapping sample name to init value (e.g. {"beta": beta_mode})
        model_kwargs: dict passed as **kwargs to warmup/run (e.g. {"sigma": 0.05})
        mcmc_key: JAX PRNG key
        num_warmup: number of warmup steps
        num_samples: number of sampling steps per chain
        num_chains: number of chains
        chain_method: 'vectorized' or 'sequential'
        nuts_config: dict from recommended_nuts_config (target_accept_prob, max_tree_depth, dense_mass)

    Returns:
        (mcmc, timing) where timing has keys warmup_time_s, sampling_time_s, step_time_s
    """
    import time
    from numpyro.infer import MCMC, NUTS, init_to_value

    if nuts_config is None:
        nuts_config = {}

    kernel = NUTS(
        model_fn,
        init_strategy=init_to_value(values=init_values),
        target_accept_prob=nuts_config.get('target_accept_prob', 0.8),
        max_tree_depth=nuts_config.get('max_tree_depth', 10),
        dense_mass=nuts_config.get('dense_mass', False),
    )

    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples,
                num_chains=num_chains, chain_method=chain_method, progress_bar=True)

    t0 = time.time()
    mcmc.warmup(mcmc_key, extra_fields=('diverging',), **model_kwargs)
    warmup_time = time.time() - t0

    t0 = time.time()
    mcmc.run(mcmc.post_warmup_state.rng_key, extra_fields=('diverging',), **model_kwargs)
    sampling_time = time.time() - t0

    total_steps = (num_warmup + num_samples) * num_chains
    step_time = (warmup_time + sampling_time) / total_steps

    print(f"MCMC warmup: {warmup_time:.1f}s, sampling: {sampling_time:.1f}s")

    return mcmc, {
        'warmup_time_s': warmup_time,
        'sampling_time_s': sampling_time,
        'step_time_s': step_time,
    }


def extract_mcmc_diagnostics(mcmc, sample_name="beta", total_samples=8000):
    """Extract samples, ESS, R-hat, divergences, and reliability flag from an MCMC run.

    Args:
        mcmc: completed MCMC object
        sample_name: name of the sampled variable
        total_samples: num_chains * num_samples (for divergence fraction)

    Returns:
        dict with keys: samples, by_chain, ess, rhat, ess_min, rhat_max,
        rhat_mean, n_div, flag, flag_explanation
    """
    from numpyro.diagnostics import effective_sample_size, split_gelman_rubin
    from ._metrics import mcmc_reliability_flag

    samples = mcmc.get_samples()[sample_name]
    by_chain = np.array(mcmc.get_samples(group_by_chain=True)[sample_name])

    ess = effective_sample_size(by_chain)
    rhat = split_gelman_rubin(by_chain)

    extra_fields = mcmc.get_extra_fields()
    n_div = int(extra_fields["diverging"].sum()) if "diverging" in extra_fields else 0

    ess_min = float(np.array(ess).min())
    rhat_max = float(np.array(rhat).max())
    rhat_mean = float(np.array(rhat).mean())

    flag, flag_explanation = mcmc_reliability_flag(ess_min, rhat_max, n_div, total_samples)

    print(f"ESS min={ess_min:.1f}, R-hat max={rhat_max:.4f}, "
          f"divergences={n_div}/{total_samples}, [{flag}] {flag_explanation}")

    return {
        'samples': samples,
        'by_chain': by_chain,
        'ess': ess,
        'rhat': rhat,
        'ess_min': ess_min,
        'rhat_max': rhat_max,
        'rhat_mean': rhat_mean,
        'n_div': n_div,
        'flag': flag,
        'flag_explanation': flag_explanation,
    }
