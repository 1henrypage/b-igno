# Experiment setup and workflow helpers.
# Callers MUST import load_this_before_everything_else BEFORE importing this module.
import time
import numpy as np
from typing import Callable, Dict, Optional, Tuple


def load_problem(problem, checkpoint_path):
    """Initialize models and load checkpoint for a ProblemInstance.

    Handles DarcyPiecewise's extra load_checkpoint_metadata via hasattr check.

    Returns:
        params: frozen parameter dict
    """
    sample_inputs = problem.get_sample_inputs(batch_size=1)
    problem.initialize_models(sample_inputs)
    ckpt_result = problem.load_checkpoint(checkpoint_path)

    if hasattr(problem, 'load_checkpoint_metadata'):
        problem.load_checkpoint_metadata(ckpt_result)

    return problem.params


def get_nf_mode(problem, params):
    """Compute NF mode: z=0 -> inverse NF -> beta_mode.

    Uses BETA_SIZE_A for EIT (6-dim coefficient latent), BETA_SIZE for others.

    Returns:
        (beta_mode, d) where d is the latent dimension used for MCMC.
    """
    import jax.numpy as jnp

    d = getattr(problem, 'BETA_SIZE_A', problem.BETA_SIZE)
    z_mode = jnp.zeros((1, d))
    beta_mode, _ = problem.models['nf'].apply(
        {'params': params['nf']}, z_mode, method=problem.models['nf'].inverse
    )
    return beta_mode[0], d


def make_log_prior(problem, params):
    """Factory returning log_prior_fn(beta) -> scalar log-probability under the NF prior."""
    def log_prior_fn(beta):
        beta_b = beta[None, :]
        log_prob = problem.log_prob_latent(params, beta_b)
        return log_prob[0]
    return log_prior_fn


def make_gaussian_log_likelihood(problem, params, mollifier_fn, x_obs, u_obs):
    """Factory returning Gaussian log-likelihood for MCMC.

    NOT for EIT (which uses Neumann flux observations, not Gaussian on u).

    Args:
        problem: ProblemInstance
        params: frozen parameter dict
        mollifier_fn: problem-specific mollifier (e.g., from darcy_continuous)
        x_obs: observation coordinates, shape (1, n_obs, d_in)
        u_obs: observed values, shape (1, n_obs, 1)

    Returns:
        log_likelihood_fn(beta, sigma) -> scalar log-probability
    """
    import jax.numpy as jnp

    def log_likelihood_fn(beta, sigma):
        beta_b = beta[None, :]
        u_pred = problem.models['u'].apply({'params': params['u']}, x_obs, beta_b)
        if u_pred.ndim == 2:
            u_pred = u_pred[..., None]
        u_pred = mollifier_fn(u_pred.squeeze(-1), x_obs)
        residual = u_pred - u_obs
        sq_err = jnp.sum(residual ** 2)
        n = u_obs.shape[1]
        return -0.5 * sq_err / (sigma ** 2) - 0.5 * n * jnp.log(2 * jnp.pi * sigma ** 2)
    return log_likelihood_fn


def make_numpyro_model(d, log_prior_fn, log_likelihood_fn, sample_name="beta"):
    """Factory for standard NumPyro model with Uniform prior + NF factor + data factor.

    NOT for DarcyPiecewise (use make_nf_reparameterized_model instead)
    or physics-informed models (use make_numpyro_model_physics).

    Returns:
        numpyro_model(sigma=0.1) callable
    """
    import numpyro
    import numpyro.distributions as dist

    def numpyro_model(sigma=0.1):
        beta = numpyro.sample(sample_name, dist.Uniform(-1., 1.).expand([d]).to_event(1))
        numpyro.factor("nf_prior", log_prior_fn(beta))
        numpyro.factor("data_lik", log_likelihood_fn(beta, sigma))
    return numpyro_model


def make_numpyro_model_physics(d, log_prior_fn, log_likelihood_fn, log_pde_fn, sample_name="beta"):
    """Factory for physics-informed NumPyro model.

    Like make_numpyro_model but adds a PDE residual factor.

    Returns:
        numpyro_model(sigma=0.1, rho_pde=1.0) callable
    """
    import numpyro
    import numpyro.distributions as dist

    def numpyro_model(sigma=0.1, rho_pde=1.0):
        beta = numpyro.sample(sample_name, dist.Uniform(-1., 1.).expand([d]).to_event(1))
        numpyro.factor("nf_prior", log_prior_fn(beta))
        numpyro.factor("data_lik", log_likelihood_fn(beta, sigma))
        numpyro.factor("pde_lik", log_pde_fn(beta, rho_pde))
    return numpyro_model


def setup_observations(problem, seed, test_idx, n_obs):
    """Set up observations for an experiment run.

    Args:
        problem: ProblemInstance
        seed: PRNG seed
        test_idx: test sample index
        n_obs: number of observation points

    Returns:
        (obs_data, obs_indices, rng) where obs_data is the dict from prepare_observations
        and rng is the updated PRNG key.
    """
    from jax import random

    rng = random.PRNGKey(seed)
    rng, key = random.split(rng)

    n_points = problem.get_n_points()
    obs_indices = problem.sample_observation_indices(n_points, n_obs, 'random', key)
    obs_data = problem.prepare_observations(
        sample_indices=[test_idx],
        obs_indices=obs_indices,
    )
    return obs_data, obs_indices, rng


def sample_unconditional_prior(problem, params, x_full, a_true_np, rng,
                               n_prior=500, batch_size=100, error_fn=None):
    """Sample from NF prior, decode, and compute unconditional prior metrics.

    Args:
        problem: ProblemInstance
        params: frozen parameter dict
        x_full: full grid coordinates, shape (1, n_points, d_in)
        a_true_np: ground truth a-field, shape (n_points,)
        rng: JAX PRNG key
        n_prior: number of prior samples
        batch_size: batch size for decoding
        error_fn: optional error function for metrics (default: RMSE)

    Returns:
        (prior_a_samples, prior_metrics, rng) where prior_a_samples is
        shape (n_prior, n_points) and rng is the updated key.
    """
    import jax
    import jax.numpy as jnp
    from ._metrics import compute_prior_predictive

    rng, prior_key = jax.random.split(rng)
    beta_prior = problem.sample_latent_from_nf(params, n_prior, prior_key)

    prior_a_list = []
    for i in range(0, n_prior, batch_size):
        batch = beta_prior[i:i + batch_size]
        x_tile = jnp.tile(x_full, (batch.shape[0], 1, 1))
        preds = problem.predict_from_beta(params, batch, x_tile)
        prior_a_list.append(np.array(preds['a_pred'][:, :, 0]))
    prior_a_samples = np.concatenate(prior_a_list, axis=0)

    prior_metrics = compute_prior_predictive(prior_a_samples, a_true_np, error_fn=error_fn)
    return prior_a_samples, prior_metrics, rng


def decode_posterior_batched(problem, params, beta_samples, x_full, batch_size=500):
    """Decode posterior beta samples to a and u fields in batches.

    NOT for DarcyPiecewise (which uses sigmoid decode).

    Args:
        problem: ProblemInstance
        params: frozen parameter dict
        beta_samples: shape (n_samples, d)
        x_full: shape (1, n_points, d_in)
        batch_size: decode batch size

    Returns:
        (a_pred_all, u_pred_all) as numpy arrays, shapes (n_samples, n_points).
    """
    import jax.numpy as jnp

    n_samples = beta_samples.shape[0]
    a_list, u_list = [], []

    for i in range(0, n_samples, batch_size):
        batch = beta_samples[i:i + batch_size]
        bs = batch.shape[0]
        x_tile = jnp.tile(x_full, (bs, 1, 1))
        preds = problem.predict_from_beta(params, batch, x_tile)

        a_pred = preds['a_pred']
        if a_pred.ndim == 2:
            a_pred = a_pred[..., None]
        a_list.append(np.array(a_pred[:, :, 0]))

        u_pred = preds['u_pred']
        if u_pred.ndim == 2:
            u_pred = u_pred[..., None]
        u_list.append(np.array(u_pred[:, :, 0]))

    return np.concatenate(a_list, axis=0), np.concatenate(u_list, axis=0)


def run_map_estimation(problem, params, x_obs, u_obs, x_full, inv_config, rng):
    """Run MAP estimation via IGNOInverter.

    Args:
        problem: ProblemInstance
        params: frozen parameter dict
        x_obs: observation coordinates
        u_obs: observed values
        x_full: full grid coordinates
        inv_config: InversionConfig
        rng: JAX PRNG key

    Returns:
        dict with keys: beta_map, preds_map, a_map, u_map, time_s,
        loss_history, rng (updated key).
    """
    import jax

    from src.evaluation.igno import IGNOInverter

    rng, inv_rng = jax.random.split(rng)
    inverter = IGNOInverter(problem, inv_rng)

    t0 = time.time()
    beta_map = inverter.invert(x_obs, u_obs, x_full, inv_config, verbose=True)
    map_time = time.time() - t0
    print(f"MAP completed in {map_time:.1f}s")

    preds_map = problem.predict_from_beta(params, beta_map, x_full)
    a_map = preds_map['a_pred'][0]
    u_map = preds_map['u_pred'][0]

    return {
        'beta_map': beta_map,
        'preds_map': preds_map,
        'a_map': a_map,
        'u_map': u_map,
        'time_s': map_time,
        'loss_history': inverter.loss_history,
        'rng': rng,
    }


def compute_sigma_from_map_residual(problem, params, mollifier_fn, beta_map, x_obs, u_obs):
    """Compute sigma from MAP u-prediction residuals at observation points.

    Forward-passes beta_map through the u-model + mollifier, then calls
    compute_sigma_from_map.

    Args:
        problem: ProblemInstance
        params: frozen parameter dict
        mollifier_fn: problem-specific mollifier
        beta_map: MAP estimate, shape (1, d)
        x_obs: observation coordinates, shape (1, n_obs, d_in)
        u_obs: observed values, shape (1, n_obs, 1)

    Returns:
        sigma (float)
    """
    import jax.numpy as jnp
    from ._mcmc import compute_sigma_from_map

    u_pred_map = problem.models['u'].apply({'params': params['u']}, x_obs, beta_map)
    if u_pred_map.ndim == 2:
        u_pred_map = u_pred_map[..., None]
    u_pred_map = mollifier_fn(u_pred_map.squeeze(-1), x_obs)

    return compute_sigma_from_map(
        np.array(u_pred_map).ravel(),
        np.array(u_obs).ravel(),
    )


def add_noise_snr_with_sigma(signal, snr_db, rng_key):
    """Add Gaussian noise at a target SNR and also return the noise sigma.

    Like ProblemInstance.add_noise_snr but returns (noisy_signal, noise_sigma).

    Args:
        signal: clean signal array
        snr_db: target signal-to-noise ratio in dB
        rng_key: JAX PRNG key

    Returns:
        (noisy_signal, noise_sigma)
    """
    import jax.numpy as jnp
    from jax import random

    signal_power = jnp.mean(signal ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise_sigma = float(jnp.sqrt(noise_power))
    noisy = signal + random.normal(rng_key, signal.shape) * noise_sigma
    return noisy, noise_sigma


def decode_initial_condition_burgers(problem, params, mollifier_fn, beta_samples):
    """Decode beta samples to initial condition a(x) for Burgers.

    Args:
        problem: ProblemInstance with models['u'] and xt_init attribute
        params: model parameters dict
        mollifier_fn: callable(u, x) -> u_mollified (e.g. mollifier_burgers)
        beta_samples: array shape (n_samples, d) where d=16 for Burgers

    Returns:
        a_pred: np.ndarray shape (n_samples, n_mesh) — predicted initial conditions
    """
    import jax.numpy as jnp

    n_s = beta_samples.shape[0]
    xt_init_tiled = jnp.tile(problem.xt_init[None, :, :], (n_s, 1, 1))
    a_raw = problem.models['u'].apply({'params': params['u']}, xt_init_tiled, beta_samples)
    if a_raw.ndim == 3:
        a_raw = a_raw.squeeze(-1)
    a_pred = mollifier_fn(a_raw, xt_init_tiled)
    return np.asarray(a_pred).squeeze(-1)


def make_igno_loss_fn(problem, params, inv_config, x_obs, u_obs, rng_key):
    """Return f(beta) -> scalar for the IGNO objective, suitable for jax.hessian.

    Closes over a fixed rng_key for loss_pde collocation sampling.
    """
    target_type = getattr(problem, 'data_target_type', 'u')
    w_pde = inv_config.loss_weights.pde
    w_data = inv_config.loss_weights.data

    def loss_fn(beta):
        beta = beta[None, :]  # jax.hessian passes (d,); sub-functions expect (batch, d)
        loss_pde = problem.loss_pde_from_beta(params, beta, rng_key)
        loss_data = problem.loss_data_from_beta(params, beta, x_obs, u_obs, target_type=target_type)
        return w_pde * loss_pde + w_data * loss_data

    return loss_fn
