# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Out-of-Distribution Robustness: Electrical Impedance Tomography
#
# - PDE: $-\nabla \cdot (a \nabla u) = 0$
# - Latent dimension: $d_a = 6$ (MCMC), $d_u = 26$ (with boundary encoding)
# - Observations: Neumann boundary flux

# %%
import sys, itertools, time
sys.path.insert(0, 'experiment_utils')
from _slurm import parse_slurm_task

PARAMETER_GRID = [
    {"seed": s, "test_idx": t}
    for s, t in itertools.product([42, 123, 7], [0, 1, 2])
]
_params, _task_id = parse_slurm_task(PARAMETER_GRID)

# %% [markdown]
# ## 0. Config

# %%
sys.path.insert(0, '..')
import load_this_before_everything_else

import jax
import jax.numpy as jnp
from jax import random
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, init_to_value
from numpyro.diagnostics import effective_sample_size, split_gelman_rubin

from src.problems.eit import (
    EIT,
    one_hot_g_l,
    compute_u_and_grad_eit,
    mollifier_eit,
    compute_pde_residual_eit_single_sample,
)
from src.evaluation.igno import IGNOInverter
from src.evaluation.metrics import rmse
from src.solver.config import InversionConfig, LossWeights, OptimizerConfig, SchedulerConfig

from experiment_utils import (
    crps_ensemble, compute_calibration, ci_width_95, chi2_ppc, nll_score,
    plot_field_comparison, plot_calibration, plot_posterior_gallery,
    plot_posterior_predictive, plot_trace, plot_metrics_table,
    plot_std_comparison_generic, plot_metrics_comparison_table_4way,
    plot_calibration_overlay,
    plot_eit_ground_truth,
    plot_eit_observation_data,
    bootstrap_metric_ci, bootstrap_metric_difference_ci,
    compute_per_chain_metrics, format_significance_table,
    compute_sigma_from_map, tune_sigma, recommended_nuts_config, mcmc_reliability_flag,
    plot_physics_benefit_comparison,
    compute_prior_predictive, build_prior_result,
    compute_error_std_correlation,
    load_problem, get_nf_mode, make_log_prior,
    make_numpyro_model, make_numpyro_model_physics,
    run_mcmc, extract_mcmc_diagnostics,
    compute_standard_metrics,
    build_mcmc_result, save_experiment_result, print_cross_seed_summary,
)

# Paths
CHECKPOINT_PATH = '../runs/final_eit/weights/best.pt'
IN_DATA_PATH = '../data/eit/inverse_EIT_in.mat'
OOD_DATA_PATH = '../data/eit/inverse_EIT_out.mat'

TEST_IDX = 0
if _task_id is not None:
    TEST_IDX = PARAMETER_GRID[_task_id]["test_idx"]
N_OBS = 124  # All boundary points

# PDE collocation
NC_PDE = 50

# Sweep settings
RHO_PDE_VALUES = [2.0, 1.0, 0.5, 0.3, 0.15, 0.1, 0.05]
SWEEP_WARMUP = 1000
SWEEP_SAMPLES = 500

# Full run settings
NUM_WARMUP = 5000
NUM_SAMPLES = 2000
NUM_CHAINS = 4
CHAIN_METHOD = 'vectorized'

SEEDS = [42, 123, 7]
if _task_id is not None:
    SEEDS = [PARAMETER_GRID[_task_id]["seed"]]

PROBLEM_NAME = 'ood_eit'
FIGURE_DIR = Path(f'figures/{PROBLEM_NAME}/tuning')
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

print(f"JAX: {jax.__version__}, NumPyro: {numpyro.__version__}")
print(f"Devices: {jax.devices()}")

# %% [markdown]
# ## 1. Setup

# %%
problem_in = EIT(seed=42, test_data_path=IN_DATA_PATH)
params = load_problem(problem_in, Path(CHECKPOINT_PATH))
beta_a_mode, d = get_nf_mode(problem_in, params)
log_prior_fn = make_log_prior(problem_in, params)

problem_ood = EIT(seed=42, test_data_path=OOD_DATA_PATH)
problem_ood.initialize_models(problem_in.get_sample_inputs(batch_size=1))
problem_ood.load_checkpoint(Path(CHECKPOINT_PATH))

n_points = problem_in.get_n_points()

print(f"Latent dim (coeff): {problem_in.BETA_SIZE_A}")
print(f"Latent dim (combined): {problem_in.BETA_SIZE_U}")
print(f"NF mode: {beta_a_mode.shape}, d={d}")

# %% [markdown]
# ## 2. EIT-Specific Closures
#
# Neumann flux likelihood and PDE closure factories. These must stay inline because
# EIT observations are fluxes (not u values) and EIT decoding uses `models['a'].apply` directly.

# %%
def make_log_likelihood(problem_inst, x_obs, u_obs):
    """Create a Neumann flux likelihood closure capturing a specific problem instance's state."""
    normals = problem_inst._active_boundary_normals_jax
    current_g_l = problem_inst._current_g_l  # captured at closure creation time

    def log_likelihood_fn(beta_a, sigma):
        beta_b = beta_a[None, :]
        g_l_onehot = one_hot_g_l(current_g_l)
        beta_u = jnp.concatenate([beta_b, g_l_onehot], axis=-1)

        g_l_scalar = current_g_l[0, 0]
        _, du_vals = compute_u_and_grad_eit(
            params['u'], problem_in.models['u'],
            x_obs[0], beta_u[0], g_l_scalar
        )

        a_vals = problem_in.models['a'].apply(
            {'params': params['a']}, x_obs, beta_b
        )[0]

        neumann_pred = a_vals * (du_vals[:, 0] * normals[:, 0] + du_vals[:, 1] * normals[:, 1])
        neumann_obs = u_obs[0, :, 0]

        sq_err = jnp.sum((neumann_pred - neumann_obs) ** 2)
        n = neumann_obs.shape[0]
        return -0.5 * sq_err / (sigma ** 2) - 0.5 * n * jnp.log(2 * jnp.pi * sigma ** 2)

    return log_likelihood_fn


# Fixed collocation points for PDE term
pde_rng = random.PRNGKey(123)
xc_fixed, R_fixed = problem_in.genPoint.weight_centers(
    n_center=NC_PDE, R_max=1e-4, R_min=1e-4, key=pde_rng,
)
x_pde_fixed = (problem_in.int_grid[None, :, :] * R_fixed + xc_fixed).reshape(-1, 2)

print(f"Fixed PDE grid: {x_pde_fixed.shape}  ({NC_PDE} centers x {problem_in.n_grid} pts)")


def make_log_pde(problem_inst):
    """Create a PDE closure capturing a specific problem instance's _current_g_l."""
    current_g_l = problem_inst._current_g_l

    def log_pde_fn(beta_a, rho_pde):
        beta_b = beta_a[None, :]
        g_l_onehot = one_hot_g_l(current_g_l)
        beta_u = jnp.concatenate([beta_b, g_l_onehot], axis=-1)

        x_pde_b = x_pde_fixed[None, :, :]
        a_decoded = problem_in.models['a'].apply({'params': params['a']}, x_pde_b, beta_b)
        a_vals = a_decoded[0, :, None]

        g_l_scalar = current_g_l[0, 0]
        residuals = compute_pde_residual_eit_single_sample(
            params['u'], problem_in.models['u'], beta_u[0], g_l_scalar,
            xc_fixed, R_fixed, problem_in.int_grid, problem_in.v, problem_in.dv_dr,
            a_vals, problem_in.n_grid,
        )
        return -0.5 * jnp.sum(residuals ** 2) / (rho_pde ** 2)

    return log_pde_fn


def decode_posterior_a(beta_a_samples, x_full, batch_size=200):
    """Decode beta_a samples -> coefficient field predictions (a-field only).

    EIT decodes only the a-field directly via models['a'].apply (not predict_from_beta).
    """
    n_s = beta_a_samples.shape[0]
    a_preds = []

    for i in range(0, n_s, batch_size):
        batch_beta = beta_a_samples[i:i + batch_size]
        n_b = batch_beta.shape[0]
        x_tile = jnp.tile(x_full, (n_b, 1, 1))
        a_b = problem_in.models['a'].apply({'params': params['a']}, x_tile, batch_beta)
        a_b = a_b[..., None] if a_b.ndim == 2 else a_b
        a_preds.append(np.array(a_b[:, :, 0]))

    return np.concatenate(a_preds, axis=0)

# %% [markdown]
# ## 3. Tuning Observations (Seed-42)
#
# CRITICAL: `prepare_observations` sets `_current_g_l` on each instance.
# Likelihood/PDE closures must be created AFTER `prepare_observations`.

# %%
_rng_tune = random.PRNGKey(42)
_rng_tune, _key_tune = random.split(_rng_tune)

_obs_indices_tune = problem_in.sample_observation_indices(n_points, N_OBS, 'random', _key_tune)

# In-domain observations for tuning (seed 42, sets problem_in._current_g_l)
obs_in = problem_in.prepare_observations(sample_indices=[TEST_IDX], obs_indices=_obs_indices_tune)
x_full_in = obs_in['x_full']      # (1, 1024, 2)
x_obs_in = obs_in['x_obs']        # (1, n_bd, 2)
u_obs_in = obs_in['u_obs']        # (1, n_bd, 1) - Neumann flux
a_true_in = obs_in['a_true']      # (1, 1024, 1)
g_l_in = obs_in['g_l']            # (1, 1)
u_true_in = obs_in.get('u_true', None)
normals_in = problem_in._active_boundary_normals_jax

# OOD observations for tuning (seed 42, sets problem_ood._current_g_l)
obs_ood = problem_ood.prepare_observations(sample_indices=[TEST_IDX], obs_indices=_obs_indices_tune)
x_full_ood = obs_ood['x_full']
x_obs_ood = obs_ood['x_obs']
u_obs_ood = obs_ood['u_obs']
a_true_ood = obs_ood['a_true']
g_l_ood = obs_ood['g_l']
u_true_ood = obs_ood.get('u_true', None)
normals_ood = problem_ood._active_boundary_normals_jax

print(f"In-domain: x_obs={x_obs_in.shape}, g_l={int(g_l_in[0, 0])}")
print(f"OOD:       x_obs={x_obs_ood.shape}, g_l={int(g_l_ood[0, 0])}")

# %%
# Ground truth fields: in-domain and OOD
plot_eit_ground_truth(
    np.array(x_full_in[0]),
    np.array(a_true_in[0, :, 0]),
    u_true=np.array(u_true_in[0, :, 0]) if u_true_in is not None else None,
    save_path=FIGURE_DIR / 'ground_truth_in.png',
)
plot_eit_ground_truth(
    np.array(x_full_ood[0]),
    np.array(a_true_ood[0, :, 0]),
    u_true=np.array(u_true_ood[0, :, 0]) if u_true_ood is not None else None,
    save_path=FIGURE_DIR / 'ground_truth_ood.png',
)

# %%
# Boundary condition g and Neumann flux: in-domain and OOD
plot_eit_observation_data(
    x_bd=np.array(x_obs_in[0]),
    g_l=int(g_l_in[0, 0]),
    neumann_obs=np.array(u_obs_in[0, :, 0]),
    save_path=FIGURE_DIR / 'observation_data_in.png',
)
plot_eit_observation_data(
    x_bd=np.array(x_obs_ood[0]),
    g_l=int(g_l_ood[0, 0]),
    neumann_obs=np.array(u_obs_ood[0, :, 0]),
    save_path=FIGURE_DIR / 'observation_data_ood.png',
)

# %% [markdown]
# ## 4. MAP Baselines and Sigma (Tuning)

# %%
inv_config = InversionConfig(
    epochs=200,
    loss_weights=LossWeights(pde=1.0, data=100.0),
    optimizer=OptimizerConfig(type='Adam', lr=0.01),
    scheduler=SchedulerConfig(type='StepLR', step_size=25, gamma=0.25),
)

# In-domain MAP (seed-42 observations for sigma and rho tuning)
_rng_tune, inv_rng = random.split(_rng_tune)
inverter_in_tune = IGNOInverter(problem_in, inv_rng)
beta_map_in_tune = inverter_in_tune.invert(x_obs_in, u_obs_in, x_full_in, inv_config, verbose=True)
preds_map_in_tune = problem_in.predict_from_beta(params, beta_map_in_tune, x_full_in)
a_map_in_tune = preds_map_in_tune['a_pred'][0]
print(f"In-domain MAP RMSE(a): {rmse(a_map_in_tune, a_true_in[0]):.6f}")

# Sigma from Neumann MAP residuals
g_l_onehot_map42 = one_hot_g_l(problem_in._current_g_l)
beta_u_map42 = jnp.concatenate([beta_map_in_tune, g_l_onehot_map42], axis=-1)
g_l_scalar_map42 = problem_in._current_g_l[0, 0]
_, du_vals_map42 = compute_u_and_grad_eit(
    params['u'], problem_in.models['u'],
    x_obs_in[0], beta_u_map42[0], g_l_scalar_map42
)
a_vals_map42 = problem_in.models['a'].apply({'params': params['a']}, x_obs_in, beta_map_in_tune)[0]
neumann_pred_map42 = a_vals_map42 * (du_vals_map42[:, 0] * normals_in[:, 0] + du_vals_map42[:, 1] * normals_in[:, 1])
neumann_pred_map42 = neumann_pred_map42[:, None]
SIGMA_MAP = compute_sigma_from_map(neumann_pred_map42, u_obs_in[0])
print(f"Sigma from MAP residual (seed-42, in-domain): {SIGMA_MAP:.6f}")

# %%
# Sigma tuning via pilot MCMC on in-domain observations
log_lik_in = make_log_likelihood(problem_in, x_obs_in, u_obs_in)

_model_tune = make_numpyro_model(d, log_prior_fn, log_lik_in, sample_name="beta_a")

def _model_factory_tune(sigma):
    def _model():
        _model_tune(sigma=sigma)
    return _model

x_full_tiled_tune = jnp.tile(x_full_in, (1, 1, 1))

def _decode_fn_tune(beta_samples):
    a_preds = []
    for i in range(len(beta_samples)):
        beta_i = beta_samples[i:i+1]
        a_pred = problem_in.models['a'].apply({'params': params['a']}, x_full_tiled_tune, beta_i)
        a_preds.append(np.array(a_pred[0]))
    return np.stack(a_preds)

_rng_tune, tune_key = random.split(_rng_tune)
SIGMA, _ = tune_sigma(
    model_fn_factory=_model_factory_tune,
    beta_mode=beta_a_mode,
    sigma_candidates=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    rng_key=tune_key,
    decode_fn=_decode_fn_tune,
    a_true=np.array(a_true_in[0, :, 0]),
    pilot_warmup=2000,
    pilot_samples=500,
    pilot_chains=2,
    sample_name='beta_a',
)
print(f"Tuned sigma (seed-42, in-domain): {SIGMA}")

# %% [markdown]
# ## 5. Rho_pde Tuning (In-Domain Only)
#
# Sweep physics constraint strength on in-domain data. Same rho applied to OOD.
# Uses threshold-based selection: keep candidates with a_err <= 2x data-only baseline.

# %%
log_pde_in = make_log_pde(problem_in)

# Quick in-domain data-only baseline (for rho selection threshold)
model_baseline_tune = make_numpyro_model(d, log_prior_fn, log_lik_in, sample_name="beta_a")
nuts_cfg_tune = recommended_nuts_config(d, SIGMA)
kernel_baseline = NUTS(model_baseline_tune,
                       init_strategy=init_to_value(values={"beta_a": beta_a_mode}),
                       target_accept_prob=nuts_cfg_tune['target_accept_prob'],
                       max_tree_depth=nuts_cfg_tune['max_tree_depth'],
                       dense_mass=nuts_cfg_tune.get('dense_mass', False))
_rng_tune, mcmc_key = random.split(_rng_tune)
mcmc_baseline = MCMC(kernel_baseline, num_warmup=SWEEP_WARMUP, num_samples=SWEEP_SAMPLES, progress_bar=True)
_t_pilot = time.time()
mcmc_baseline.run(mcmc_key, sigma=SIGMA)
print(f"  Pilot baseline completed in {time.time() - _t_pilot:.1f}s")

beta_baseline = mcmc_baseline.get_samples()["beta_a"]
a_baseline = decode_posterior_a(beta_baseline, x_full_in)
a_true_in_np_tune = np.array(a_true_in[0, :, 0])
dec_baseline = compute_standard_metrics(a_baseline, a_true_in_np_tune)
baseline_a_err = dec_baseline['a_err']
print(f"In-domain data-only baseline: a_err={baseline_a_err:.4f}, coverage={dec_baseline['coverage_95']:.2%}")

# %%
print(f"Rho sweep on IN-DOMAIN: {SWEEP_WARMUP} warmup, {SWEEP_SAMPLES} samples per rho")
print(f"rho values: {RHO_PDE_VALUES}")

sweep_results = []
for rho in RHO_PDE_VALUES:
    print(f"\n{'='*60}\nrho_pde = {rho}\n{'='*60}")
    model_phys_sweep = make_numpyro_model_physics(d, log_prior_fn, log_lik_in, log_pde_in, sample_name="beta_a")
    kernel_sw = NUTS(model_phys_sweep,
                     init_strategy=init_to_value(values={"beta_a": beta_a_mode}),
                     target_accept_prob=nuts_cfg_tune['target_accept_prob'],
                     max_tree_depth=nuts_cfg_tune['max_tree_depth'],
                     dense_mass=nuts_cfg_tune.get('dense_mass', False))
    _rng_tune, key = random.split(_rng_tune)
    mcmc_sw = MCMC(kernel_sw, num_warmup=SWEEP_WARMUP, num_samples=SWEEP_SAMPLES, progress_bar=True)
    _t_pilot = time.time()
    mcmc_sw.run(key, sigma=SIGMA, rho_pde=rho, extra_fields=('diverging',))
    print(f"  Pilot rho={rho} completed in {time.time() - _t_pilot:.1f}s")

    beta_s = mcmc_sw.get_samples()["beta_a"]
    a_sw = decode_posterior_a(beta_s, x_full_in)
    dec = compute_standard_metrics(a_sw, a_true_in_np_tune)
    ess = effective_sample_size(np.array(beta_s)[None])
    n_div = int(mcmc_sw.get_extra_fields()['diverging'].sum())

    sweep_results.append({
        'rho_pde': rho, 'a_err': dec['a_err'],
        'coverage': dec['coverage_95'], 'ess_min': float(ess.min()),
        'n_div': n_div, 'crps_a': dec['crps_a'],
    })
    print(f"  a_err={dec['a_err']:.4f}  coverage={dec['coverage_95']:.2%}  "
          f"ESS_min={float(ess.min()):.1f}  n_div={n_div}")

# Select best rho: EIT-specific threshold filter then closest to 95% coverage
a_err_threshold = baseline_a_err * 2
candidates = [(abs(r['coverage'] - 0.95), r) for r in sweep_results
              if r['a_err'] <= a_err_threshold]
if candidates:
    candidates.sort(key=lambda x: x[0])
    BEST_RHO_PDE = candidates[0][1]['rho_pde']
else:
    BEST_RHO_PDE = max(RHO_PDE_VALUES)
    print("No candidate with a_err <= 2x baseline; using largest rho.")

print(f"\nSelected BEST_RHO_PDE = {BEST_RHO_PDE}")

# %% [markdown]
# ## 6. Run Condition Helper

# %%
def run_condition(model_fn, model_kwargs, x_full, a_true_ref, label, seed):
    """Run MCMC for one EIT condition, decode a-field inline, and compute metrics."""
    print(f"\n{'='*60}\n  {label}\n{'='*60}")

    nuts_cfg = recommended_nuts_config(d, model_kwargs['sigma'])
    print(f"  NUTS config: {nuts_cfg}")

    rng_local = random.PRNGKey(seed)
    mcmc, timing = run_mcmc(
        model_fn, {"beta_a": beta_a_mode}, model_kwargs, rng_local,
        NUM_WARMUP, NUM_SAMPLES, NUM_CHAINS, CHAIN_METHOD, nuts_cfg,
    )
    mcmc.print_summary(exclude_deterministic=True)

    diag = extract_mcmc_diagnostics(mcmc, sample_name="beta_a",
                                    total_samples=NUM_CHAINS * NUM_SAMPLES)

    # Inline decode: EIT decodes a-field only via models['a'].apply
    a_samples = decode_posterior_a(diag['samples'], x_full)
    a_true_np = np.array(a_true_ref[0, :, 0])
    metrics = compute_standard_metrics(a_samples, a_true_np)

    a_std = np.std(a_samples, axis=0)
    a_mean = np.mean(a_samples, axis=0)
    spearman_rho, spearman_p = compute_error_std_correlation(a_true_np, a_mean, a_std)

    return {
        'label': label,
        'a_samples': a_samples,
        'a_mean': a_mean,
        'a_std': a_std,
        **metrics,
        'sigma': model_kwargs['sigma'],
        'ess_min': diag['ess_min'],
        'rhat_max': diag['rhat_max'],
        'rhat_mean': diag['rhat_mean'],
        'n_div': diag['n_div'],
        'reliability_flag': diag['flag'],
        'reliability_explanation': diag['flag_explanation'],
        'spearman_rho_error_std': float(spearman_rho),
        'spearman_pvalue_error_std': float(spearman_p),
        'warmup_time_s': timing['warmup_time_s'],
        'sampling_time_s': timing['sampling_time_s'],
        'step_time_s': timing['step_time_s'],
        'beta_by_chain': diag['by_chain'],
        'beta_for_trace': diag['by_chain'][0],
        'ess': diag['ess'],
        'rhat': diag['rhat'],
    }

# %% [markdown]
# ## 7. Multi-Seed Full MCMC Runs
#
# | | Data-only | Physics-informed |
# |---|---|---|
# | **In-domain** | (a) | (b) |
# | **OOD** | (c) | (d) |
#
# Each condition uses its own likelihood and PDE closures (capturing the correct `_current_g_l`).

# %%
from results_schema import ExperimentResult
from datetime import datetime

for SEED in SEEDS:
    print(f"\n{'#'*70}\n## SEED = {SEED}\n{'#'*70}")
    _t_total_start = time.time()

    rng = random.PRNGKey(SEED)

    # In-domain observations (sets problem_in._current_g_l)
    rng, key = random.split(rng)
    obs_indices_in = problem_in.sample_observation_indices(n_points, N_OBS, 'random', key)
    obs_in = problem_in.prepare_observations(sample_indices=[TEST_IDX], obs_indices=obs_indices_in)
    x_full_in = obs_in['x_full']
    x_obs_in = obs_in['x_obs']
    u_obs_in = obs_in['u_obs']
    a_true_in = obs_in['a_true']
    g_l_in = obs_in['g_l']
    u_true_in = obs_in.get('u_true', None)
    normals_in = problem_in._active_boundary_normals_jax

    # OOD observations (sets problem_ood._current_g_l; same key for same spatial indices)
    obs_indices_ood = problem_ood.sample_observation_indices(n_points, N_OBS, 'random', key)
    obs_ood = problem_ood.prepare_observations(sample_indices=[TEST_IDX], obs_indices=obs_indices_ood)
    x_full_ood = obs_ood['x_full']
    x_obs_ood = obs_ood['x_obs']
    u_obs_ood = obs_ood['u_obs']
    a_true_ood = obs_ood['a_true']
    g_l_ood = obs_ood['g_l']
    u_true_ood = obs_ood.get('u_true', None)
    normals_ood = problem_ood._active_boundary_normals_jax

    FIGURE_DIR = Path(f'figures/{PROBLEM_NAME}/seed_{SEED}')
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    # Likelihood and PDE closures capturing new observations/_current_g_l
    log_lik_in = make_log_likelihood(problem_in, x_obs_in, u_obs_in)
    log_lik_ood = make_log_likelihood(problem_ood, x_obs_ood, u_obs_ood)
    log_pde_in = make_log_pde(problem_in)
    log_pde_ood = make_log_pde(problem_ood)

    # === MAP Baselines ===
    rng, inv_rng = random.split(rng)
    inverter_in = IGNOInverter(problem_in, inv_rng)
    _t_map_in = time.time()
    beta_map_in = inverter_in.invert(x_obs_in, u_obs_in, x_full_in, inv_config, verbose=False)
    _map_in_time_s = time.time() - _t_map_in
    print(f"MAP (in-domain) completed in {_map_in_time_s:.1f}s")
    preds_map_in = problem_in.predict_from_beta(params, beta_map_in, x_full_in)
    a_map_in = preds_map_in['a_pred'][0]
    rmse_map_in = float(rmse(a_map_in, a_true_in[0]))
    print(f"[seed={SEED}] In-domain MAP RMSE(a): {rmse_map_in:.6f}")

    # Sigma from Neumann MAP residuals (in-domain, this seed)
    g_l_onehot_s = one_hot_g_l(problem_in._current_g_l)
    beta_u_s = jnp.concatenate([beta_map_in, g_l_onehot_s], axis=-1)
    g_l_scalar_s = problem_in._current_g_l[0, 0]
    _, du_vals_s = compute_u_and_grad_eit(
        params['u'], problem_in.models['u'],
        x_obs_in[0], beta_u_s[0], g_l_scalar_s
    )
    a_vals_s = problem_in.models['a'].apply({'params': params['a']}, x_obs_in, beta_map_in)[0]
    neumann_pred_s = a_vals_s * (du_vals_s[:, 0] * normals_in[:, 0] + du_vals_s[:, 1] * normals_in[:, 1])
    neumann_pred_s = neumann_pred_s[:, None]
    SIGMA_MAP_S = compute_sigma_from_map(neumann_pred_s, u_obs_in[0])
    print(f"Sigma from MAP residual: {SIGMA_MAP_S:.6f}")

    # Sigma tuning (this seed)
    _model_tune_s = make_numpyro_model(d, log_prior_fn, log_lik_in, sample_name="beta_a")

    def _model_factory_s(sigma):
        def _model():
            _model_tune_s(sigma=sigma)
        return _model

    x_full_tiled_s = jnp.tile(x_full_in, (1, 1, 1))

    def _decode_fn_s(beta_samples):
        a_preds = []
        for i in range(len(beta_samples)):
            beta_i = beta_samples[i:i+1]
            a_pred = problem_in.models['a'].apply({'params': params['a']}, x_full_tiled_s, beta_i)
            a_preds.append(np.array(a_pred[0]))
        return np.stack(a_preds)

    rng, tune_key = random.split(rng)
    SIGMA, _ = tune_sigma(
        model_fn_factory=_model_factory_s,
        beta_mode=beta_a_mode,
        sigma_candidates=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        rng_key=tune_key,
        decode_fn=_decode_fn_s,
        a_true=np.array(a_true_in[0, :, 0]),
        pilot_warmup=2000,
        pilot_samples=500,
        pilot_chains=2,
        sample_name='beta_a',
    )
    print(f"Tuned sigma (seed={SEED}, in-domain): {SIGMA}")

    rng, inv_rng = random.split(rng)
    inverter_ood = IGNOInverter(problem_ood, inv_rng)
    _t_map_ood = time.time()
    beta_map_ood = inverter_ood.invert(x_obs_ood, u_obs_ood, x_full_ood, inv_config, verbose=False)
    _map_ood_time_s = time.time() - _t_map_ood
    print(f"MAP (OOD) completed in {_map_ood_time_s:.1f}s")
    preds_map_ood = problem_ood.predict_from_beta(params, beta_map_ood, x_full_ood)
    a_map_ood = preds_map_ood['a_pred'][0]
    rmse_map_ood = float(rmse(a_map_ood, a_true_ood[0]))
    print(f"[seed={SEED}] OOD MAP RMSE(a): {rmse_map_ood:.6f}")

    # === Prior Predictive Baseline (EIT-specific: a-field only, models['a'].apply) ===
    N_PRIOR = 500
    rng, prior_key = random.split(rng)
    beta_a_prior = problem_in.sample_latent_from_nf(params, N_PRIOR, prior_key)

    prior_a_list = []
    PRIOR_BATCH = 100
    for i in range(0, N_PRIOR, PRIOR_BATCH):
        batch = beta_a_prior[i:i + PRIOR_BATCH]
        x_tile = jnp.tile(x_full_in, (batch.shape[0], 1, 1))
        a_pred = problem_in.models['a'].apply({'params': params['a']}, x_tile, batch)
        a_pred = a_pred[..., None] if a_pred.ndim == 2 else a_pred
        prior_a_list.append(np.array(a_pred[:, :, 0]))
    prior_a_samples = np.concatenate(prior_a_list, axis=0)

    a_true_in_np = np.array(a_true_in[0, :, 0])
    a_true_ood_np = np.array(a_true_ood[0, :, 0])
    prior_metrics_in = compute_prior_predictive(prior_a_samples, a_true_in_np)
    prior_metrics_ood = compute_prior_predictive(prior_a_samples, a_true_ood_np)
    print(f"Prior (in-domain): a_err={prior_metrics_in['a_err']:.4f}, CRPS={prior_metrics_in['crps_a']:.4f}")
    print(f"Prior (OOD):       a_err={prior_metrics_ood['a_err']:.4f}, CRPS={prior_metrics_ood['crps_a']:.4f}")

    # === Build per-seed models ===
    model_in_do = make_numpyro_model(d, log_prior_fn, log_lik_in, sample_name="beta_a")
    model_in_phys = make_numpyro_model_physics(d, log_prior_fn, log_lik_in, log_pde_in, sample_name="beta_a")
    model_ood_do = make_numpyro_model(d, log_prior_fn, log_lik_ood, sample_name="beta_a")
    model_ood_phys = make_numpyro_model_physics(d, log_prior_fn, log_lik_ood, log_pde_ood, sample_name="beta_a")

    # === 4 Conditions ===
    # (a) In-domain, data-only
    res_in_do = run_condition(
        model_in_do, {"sigma": SIGMA},
        x_full_in, a_true_in, "In-domain Data-Only", seed=SEED,
    )

    # (b) In-domain, physics
    res_in_phys = run_condition(
        model_in_phys, {"sigma": SIGMA, "rho_pde": BEST_RHO_PDE},
        x_full_in, a_true_in, "In-domain Physics", seed=SEED + 1,
    )

    # (c) OOD, data-only
    res_ood_do = run_condition(
        model_ood_do, {"sigma": SIGMA},
        x_full_ood, a_true_ood, "OOD Data-Only", seed=SEED + 2,
    )

    # (d) OOD, physics
    res_ood_phys = run_condition(
        model_ood_phys, {"sigma": SIGMA, "rho_pde": BEST_RHO_PDE},
        x_full_ood, a_true_ood, "OOD Physics", seed=SEED + 3,
    )

    # -- Chi2 PPC (Neumann flux, EIT-specific) --
    def _compute_neumann(beta_a_single, prob_inst, x_obs_c, normals_c):
        beta_b = beta_a_single[None, :]
        g_l_oh = one_hot_g_l(prob_inst._current_g_l)
        beta_u = jnp.concatenate([beta_b, g_l_oh], axis=-1)
        g_l_s = prob_inst._current_g_l[0, 0]
        _, du = compute_u_and_grad_eit(
            params['u'], problem_in.models['u'], x_obs_c[0], beta_u[0], g_l_s
        )
        a_v = problem_in.models['a'].apply({'params': params['a']}, x_obs_c, beta_b)[0]
        return a_v * (du[:, 0] * normals_c[:, 0] + du[:, 1] * normals_c[:, 1])

    _n_chi2 = 100
    for _res, _prob, _xobs, _nrm, _uobs in [
        (res_in_do,   problem_in,  x_obs_in,  normals_in,  u_obs_in),
        (res_in_phys, problem_in,  x_obs_in,  normals_in,  u_obs_in),
        (res_ood_do,  problem_ood, x_obs_ood, normals_ood, u_obs_ood),
        (res_ood_phys,problem_ood, x_obs_ood, normals_ood, u_obs_ood),
    ]:
        _betas_raw = jnp.array(_res['beta_by_chain'].reshape(-1, d))
        _ns = min(_n_chi2, _betas_raw.shape[0])
        _idx = np.linspace(0, _betas_raw.shape[0] - 1, _ns, dtype=int)
        _flux = np.stack(
            [np.array(_compute_neumann(_betas_raw[i], _prob, _xobs, _nrm)) for i in _idx], axis=0
        )
        _obs_np = np.array(_uobs[0, :, 0])
        _chi2, _pval = chi2_ppc(_obs_np, _flux, SIGMA)
        _res['chi2_ppc'] = _chi2
        _res['chi2_ppc_pvalue'] = _pval
        print(f"  {_res['label']}: chi2={_chi2:.2f}, p={_pval:.4f}")

    # -- Metrics Comparison --
    def make_metrics_dict(res):
        return {
            'Rel. L2 (a)': res['a_err'],
            'CRPS (a)': res['crps_a'],
            'NLL (a)': res['nll_a'],
            '95% Coverage': res['coverage_95'],
            'CI Width (a)': res['ci_width'],
            'Sharpness (mean std)': res['mean_std'],
            'ESS min': res['ess_min'],
            'R-hat max': res['rhat_max'],
            'R-hat mean': res['rhat_mean'],
            'Divergences': res['n_div'],
        }

    plot_metrics_comparison_table_4way(
        make_metrics_dict(res_in_do),
        make_metrics_dict(res_in_phys),
        make_metrics_dict(res_ood_do),
        make_metrics_dict(res_ood_phys),
        title=f'EIT — sigma={SIGMA}, rho_pde={BEST_RHO_PDE} (seed={SEED})',
    )

    # -- Significance of Pairwise Contrasts --
    def _crps_a(s, t): return float(np.mean(crps_ensemble(s, t)))
    def _coverage_95(s, t):
        _, emp = compute_calibration(s, t, np.array([0.95]))
        return float(emp[0])
    def _ci_width(s, t): return ci_width_95(s)

    rng_bs = np.random.default_rng(SEED)

    d_in_phys_crps  = bootstrap_metric_difference_ci(res_in_do['a_samples'],  res_in_phys['a_samples'],  a_true_in_np,  _crps_a, rng=rng_bs)
    d_in_phys_cov   = bootstrap_metric_difference_ci(res_in_do['a_samples'],  res_in_phys['a_samples'],  a_true_in_np,  _coverage_95, rng=rng_bs)
    d_in_phys_width = bootstrap_metric_difference_ci(res_in_do['a_samples'],  res_in_phys['a_samples'],  a_true_in_np,  _ci_width, rng=rng_bs)
    d_ood_phys_crps  = bootstrap_metric_difference_ci(res_ood_do['a_samples'], res_ood_phys['a_samples'], a_true_ood_np, _crps_a, rng=rng_bs)
    d_ood_phys_cov   = bootstrap_metric_difference_ci(res_ood_do['a_samples'], res_ood_phys['a_samples'], a_true_ood_np, _coverage_95, rng=rng_bs)
    d_ood_phys_width = bootstrap_metric_difference_ci(res_ood_do['a_samples'], res_ood_phys['a_samples'], a_true_ood_np, _ci_width, rng=rng_bs)
    d_ood_do_crps    = bootstrap_metric_difference_ci(res_ood_do['a_samples'], res_in_do['a_samples'],    a_true_ood_np, _crps_a, rng=rng_bs)
    d_ood_phys_deg   = bootstrap_metric_difference_ci(res_ood_phys['a_samples'],res_in_phys['a_samples'], a_true_ood_np, _crps_a, rng=rng_bs)

    format_significance_table({
        '(a) In-Phys CRPS diff (DO - Phys)':      d_in_phys_crps,
        '(a) In-Phys Coverage diff (DO - Phys)':  d_in_phys_cov,
        '(a) In-Phys CI Width diff (DO - Phys)':  d_in_phys_width,
        '(b) OOD-Phys CRPS diff (DO - Phys)':     d_ood_phys_crps,
        '(b) OOD-Phys Coverage diff (DO - Phys)': d_ood_phys_cov,
        '(b) OOD-Phys CI Width diff (DO - Phys)': d_ood_phys_width,
        '(c) OOD degradation CRPS (DO)':           d_ood_do_crps,
        '(d) OOD degradation CRPS (Phys)':         d_ood_phys_deg,
    }, title=f'EIT — RQ3: Pairwise Contrasts (Bootstrap, seed={SEED})')
    print("Negative CRPS/CI-Width diff = improved; positive Coverage diff = improved.")
    print("Note: 4 contrasts x 3 metrics = 12 tests; Bonferroni threshold 0.05/12 ≈ 0.004")

    # -- Physics Benefit Delta Analysis (RQ3 Hardening) --
    _rng_pb = np.random.default_rng(SEED + 5000)

    delta_in_crps  = bootstrap_metric_difference_ci(res_in_phys['a_samples'],  res_in_do['a_samples'],  a_true_in_np,  _crps_a, rng=_rng_pb)
    delta_ood_crps = bootstrap_metric_difference_ci(res_ood_phys['a_samples'], res_ood_do['a_samples'], a_true_ood_np, _crps_a, rng=_rng_pb)

    ci_in_do_pb    = bootstrap_metric_ci(res_in_do['a_samples'],   a_true_in_np,  _crps_a, rng=_rng_pb)
    ci_in_phys_pb  = bootstrap_metric_ci(res_in_phys['a_samples'], a_true_in_np,  _crps_a, rng=_rng_pb)
    ci_ood_do_pb   = bootstrap_metric_ci(res_ood_do['a_samples'],  a_true_ood_np, _crps_a, rng=_rng_pb)
    ci_ood_phys_pb = bootstrap_metric_ci(res_ood_phys['a_samples'],a_true_ood_np, _crps_a, rng=_rng_pb)

    _n_dod = 1000
    _dod_samples = np.empty(_n_dod)
    _rng_dod = np.random.default_rng(SEED + 6000)
    _na_in_do, _na_in_p   = res_in_do['a_samples'].shape[0],  res_in_phys['a_samples'].shape[0]
    _na_ood_do, _na_ood_p = res_ood_do['a_samples'].shape[0], res_ood_phys['a_samples'].shape[0]
    for _i in range(_n_dod):
        _b_in_do  = res_in_do['a_samples'][_rng_dod.integers(0, _na_in_do,  _na_in_do)]
        _b_in_p   = res_in_phys['a_samples'][_rng_dod.integers(0, _na_in_p,   _na_in_p)]
        _b_ood_do = res_ood_do['a_samples'][_rng_dod.integers(0, _na_ood_do, _na_ood_do)]
        _b_ood_p  = res_ood_phys['a_samples'][_rng_dod.integers(0, _na_ood_p,  _na_ood_p)]
        _d_in  = _crps_a(_b_in_p,  a_true_in_np)  - _crps_a(_b_in_do,  a_true_in_np)
        _d_ood = _crps_a(_b_ood_p, a_true_ood_np) - _crps_a(_b_ood_do, a_true_ood_np)
        _dod_samples[_i] = _d_ood - _d_in
    dod_crps = {
        'mean_diff': float(np.mean(_dod_samples)),
        'ci_lo':     float(np.percentile(_dod_samples, 2.5)),
        'ci_hi':     float(np.percentile(_dod_samples, 97.5)),
    }
    dod_crps['significant'] = not (dod_crps['ci_lo'] <= 0 <= dod_crps['ci_hi'])

    format_significance_table({
        'Δ CRPS In-Domain (phys − do)':         delta_in_crps,
        'Δ CRPS OOD (phys − do)':               delta_ood_crps,
        'Δ CRPS DoD (ood_delta − id_delta)':    dod_crps,
    }, title=f'EIT — Physics Benefit Analysis (seed={SEED})')
    print("Negative Δ CRPS = physics improves. Positive DoD = OOD benefits more from physics.")

    _id_pb  = [{'data_only': {'crps_a': ci_in_do_pb['estimate'],
                               'bootstrap_lo': ci_in_do_pb['ci_lo'],
                               'bootstrap_hi': ci_in_do_pb['ci_hi']},
                'physics':   {'crps_a': ci_in_phys_pb['estimate'],
                               'bootstrap_lo': ci_in_phys_pb['ci_lo'],
                               'bootstrap_hi': ci_in_phys_pb['ci_hi']},
                'delta_ci_lo': delta_in_crps['ci_lo'],
                'delta_ci_hi': delta_in_crps['ci_hi']}]
    _ood_pb = [{'data_only': {'crps_a': ci_ood_do_pb['estimate'],
                               'bootstrap_lo': ci_ood_do_pb['ci_lo'],
                               'bootstrap_hi': ci_ood_do_pb['ci_hi']},
                'physics':   {'crps_a': ci_ood_phys_pb['estimate'],
                               'bootstrap_lo': ci_ood_phys_pb['ci_lo'],
                               'bootstrap_hi': ci_ood_phys_pb['ci_hi']},
                'delta_ci_lo': delta_ood_crps['ci_lo'],
                'delta_ci_hi': delta_ood_crps['ci_hi']}]
    plot_physics_benefit_comparison(
        _id_pb, _ood_pb,
        metric_key='crps_a',
        metric_label='Δ CRPS (physics − data-only)',
        save_path=FIGURE_DIR / 'physics_benefit_comparison.png',
    )

    # -- Save Structured Result --
    conditions_raw = {
        "in_domain_data_only": res_in_do,
        "in_domain_physics": res_in_phys,
        "ood_data_only": res_ood_do,
        "ood_physics": res_ood_phys,
    }

    _map_a_err_for = {
        "in_domain_data_only": rmse_map_in,
        "in_domain_physics": rmse_map_in,
        "ood_data_only": rmse_map_ood,
        "ood_physics": rmse_map_ood,
    }

    def _build_condition(k, v):
        d_res = dict(v)
        d_res["map_a_err"] = _map_a_err_for[k]
        d_res["coverage"] = d_res.get("coverage_95")
        return build_mcmc_result(d_res, NUM_WARMUP, NUM_SAMPLES, NUM_CHAINS)

    experiment = ExperimentResult(
        experiment="ood",
        problem="eit",
        experiment_type="comparison",
        test_idx=TEST_IDX,
        timestamp=datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        seed=SEED,
        conditions={k: _build_condition(k, v) for k, v in conditions_raw.items()},
        prior=build_prior_result(prior_metrics_in),
        prior_ood=build_prior_result(prior_metrics_ood),
        map_time_s=_map_in_time_s + _map_ood_time_s,
        total_time_s=time.time() - _t_total_start,
    )

    out_path = save_experiment_result(experiment)
    print(f"Saved structured result to: {out_path}")


    # -- Plots --
    x_np_in = np.array(x_full_in[0])
    x_np_ood = np.array(x_full_ood[0])

    beta_true_in = problem_in.models['enc'].apply({'params': params['enc']}, a_true_in)[0]
    beta_true_ood = problem_in.models['enc'].apply({'params': params['enc']}, a_true_ood)[0]

    plot_std_comparison_generic(
        x_np_ood,
        std_a=np.array(res_ood_do['a_std']),
        std_b=np.array(res_ood_phys['a_std']),
        label_a='OOD (DO)',
        label_b='OOD (Phys)',
        grid_shape=(32, 32),
        suptitle='Uncertainty: OOD Data-Only vs OOD Physics',
        save_path=FIGURE_DIR / 'std_ood_do_vs_physics.png',
    )

    plot_calibration_overlay(
        [
            (res_in_do['cal_levels'],   res_in_do['cal_empirical'],   'In-Domain DO'),
            (res_in_phys['cal_levels'], res_in_phys['cal_empirical'], 'In-Domain Phys'),
            (res_ood_do['cal_levels'],  res_ood_do['cal_empirical'],  'OOD DO'),
            (res_ood_phys['cal_levels'],res_ood_phys['cal_empirical'],'OOD Phys'),
        ],
        save_path=FIGURE_DIR / 'calibration_overlay_4way.png',
    )

    a_map_ood_np = np.array(a_map_ood[:, 0])

    plot_field_comparison(
        x_np_ood, a_true_ood_np, a_map_ood_np,
        np.array(res_ood_do['a_mean']),
        np.array(res_ood_do['a_std']),
        grid_shape=(32, 32),
        obs_coords=np.array(x_obs_ood[0]),
        save_path=FIGURE_DIR / 'field_comparison_ood_data_only.png',
    )

    plot_field_comparison(
        x_np_ood, a_true_ood_np, a_map_ood_np,
        np.array(res_ood_phys['a_mean']),
        np.array(res_ood_phys['a_std']),
        grid_shape=(32, 32),
        obs_coords=np.array(x_obs_ood[0]),
        save_path=FIGURE_DIR / 'field_comparison_ood_physics.png',
    )

    plot_posterior_gallery(
        x_np_ood, res_ood_do['a_samples'], grid_shape=(32, 32),
        a_true=a_true_ood_np, n_show=6,
        save_path=FIGURE_DIR / 'posterior_gallery_ood_data_only.png',
    )

    plot_posterior_gallery(
        x_np_ood, res_ood_phys['a_samples'], grid_shape=(32, 32),
        a_true=a_true_ood_np, n_show=6,
        save_path=FIGURE_DIR / 'posterior_gallery_ood_physics.png',
    )

    # -- Posterior Predictive: Neumann Flux for OOD conditions --
    def compute_neumann_for_beta(beta_a_single, problem_inst):
        """Compute Neumann flux at boundary for a single beta_a sample."""
        beta_b = beta_a_single[None, :]
        g_l_onehot = one_hot_g_l(problem_inst._current_g_l)
        beta_u = jnp.concatenate([beta_b, g_l_onehot], axis=-1)
        g_l_scalar = problem_inst._current_g_l[0, 0]
        x_obs_inst = x_obs_ood
        normals_inst = problem_inst._active_boundary_normals_jax

        _, du_vals = compute_u_and_grad_eit(
            params['u'], problem_in.models['u'], x_obs_inst[0], beta_u[0], g_l_scalar
        )
        a_vals = problem_in.models['a'].apply({'params': params['a']}, x_obs_inst, beta_b)[0]
        return a_vals * (du_vals[:, 0] * normals_inst[:, 0] + du_vals[:, 1] * normals_inst[:, 1])

    beta_a_ood_do = jnp.array(res_ood_do['beta_by_chain'].reshape(-1, d))
    n_pred = min(100, beta_a_ood_do.shape[0])
    pred_idx = np.linspace(0, beta_a_ood_do.shape[0] - 1, n_pred, dtype=int)
    flux_pred_ood_do = np.stack(
        [np.array(compute_neumann_for_beta(beta_a_ood_do[idx], problem_ood)) for idx in pred_idx], axis=0
    )

    beta_a_ood_phys = jnp.array(res_ood_phys['beta_by_chain'].reshape(-1, d))
    n_pred_phys = min(100, beta_a_ood_phys.shape[0])
    pred_idx_phys = np.linspace(0, beta_a_ood_phys.shape[0] - 1, n_pred_phys, dtype=int)
    flux_pred_ood_phys = np.stack(
        [np.array(compute_neumann_for_beta(beta_a_ood_phys[idx], problem_ood)) for idx in pred_idx_phys], axis=0
    )

    neumann_obs_ood_np = np.array(u_obs_ood[0, :, 0])

    plot_posterior_predictive(
        neumann_obs_ood_np, flux_pred_ood_do,
        obs_label='Neumann flux observed (OOD)',
        save_path=FIGURE_DIR / 'posterior_predictive_ood_data_only.png',
    )

    plot_posterior_predictive(
        neumann_obs_ood_np, flux_pred_ood_phys,
        obs_label='Neumann flux observed (OOD)',
        save_path=FIGURE_DIR / 'posterior_predictive_ood_physics.png',
    )

    # -- Diagnostics --
    for _label, _res, _beta_true_np in [("OOD Data-Only", res_ood_do, np.array(beta_true_ood)),
                                         ("OOD Physics",   res_ood_phys, np.array(beta_true_ood))]:
        _bbc = _res['beta_by_chain']
        _ess = effective_sample_size(_bbc)
        _rhat = split_gelman_rubin(_bbc)

        print(f"\n{_label} per-dimension diagnostics ({NUM_CHAINS} chains):")
        print(f"{'dim':>4s}  {'ESS':>8s}  {'R-hat':>8s}  {'mean':>10s}  {'std':>10s}")
        for i in range(d):
            _b_all = _bbc[:, :, i].flatten()
            print(f"{i:4d}  {float(_ess[i]):8.1f}  {float(_rhat[i]):8.4f}  "
                  f"{float(_b_all.mean()):10.4f}  "
                  f"{float(_b_all.std()):10.4f}")
        print(f"Divergences: {_res['n_div']} / {NUM_CHAINS * NUM_SAMPLES}")
        print(f"ESS min: {float(_ess.min()):.1f}, R-hat max: {float(_rhat.max()):.4f}, R-hat mean: {float(_rhat.mean()):.4f}")

        _suffix = 'ood_do' if 'Data-Only' in _label else 'ood_phys'
        plot_trace(_res['beta_for_trace'], beta_true=_beta_true_np, num_warmup=0,
                   save_path=FIGURE_DIR / f'trace_plots_{_suffix}.png')


# %% [markdown]
# ## Cross-Seed Aggregation Summary

# %%
print_cross_seed_summary("ood", "eit")
