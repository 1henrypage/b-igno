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
# # Out-of-Distribution Robustness: Darcy Piecewise {5, 1000}
#
# - PDE: $-\nabla \cdot (a \nabla u) = 10$, piecewise constant coefficient function $\{5, 1000\}$ (200x contrast)
# - Latent dimension: $d = 200$

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

from src.problems.darcy_piecewise import (
    DarcyPiecewise5v1000 as DarcyPiecewise,
    mollifier,
    a_sample,
    compute_pde_residual_piecewise_single_sample,
)
from src.evaluation.igno import IGNOInverter
from src.evaluation.metrics import rmse, cross_correlation
from src.solver.config import InversionConfig, LossWeights, OptimizerConfig, SchedulerConfig

from experiment_utils import (
    crps_ensemble, compute_calibration, ci_width_95, chi2_ppc, nll_score,
    plot_field_comparison, plot_calibration, plot_posterior_gallery,
    plot_posterior_predictive, plot_trace, plot_metrics_table,
    plot_std_comparison_generic, plot_metrics_comparison_table_4way,
    plot_calibration_overlay,
    bootstrap_metric_ci, bootstrap_metric_difference_ci,
    compute_per_chain_metrics, format_significance_table,
    compute_sigma_from_map, recommended_nuts_config, mcmc_reliability_flag,
    plot_physics_benefit_comparison,
    compute_prior_predictive, build_prior_result,
    compute_error_std_correlation,
    load_problem,
    make_gaussian_log_likelihood,
    run_mcmc, extract_mcmc_diagnostics,
    compute_piecewise_metrics,
    build_mcmc_result, save_experiment_result, print_cross_seed_summary,
    sample_unconditional_prior,
)
from results_schema import ExperimentResult
from datetime import datetime

# Paths
CHECKPOINT_PATH = Path('../runs/darcy_piecewise_5v1000/weights/best.pt')
TRAIN_DATA_PATH = '../data/darcy_piecewise_5v1000/pwc_train_data10000.mat'
IN_DATA_PATH = '../data/darcy_piecewise_5v1000/pwc_test_in.mat'
OOD_DATA_PATH = '../data/darcy_piecewise_5v1000/pwc_test_out.mat'

TEST_IDX = 0
if _task_id is not None:
    TEST_IDX = PARAMETER_GRID[_task_id]["test_idx"]
N_OBS = 100

# PDE collocation (reduced: s1/s2 decoders make PDE eval more expensive)
NC_PDE = 30

# Sweep settings (extended to very small values for 200-dim)
RHO_PDE_VALUES = [5.0, 2.0, 1.0, 0.5, 0.3, 0.1, 0.05]
SWEEP_WARMUP = 2000
SWEEP_SAMPLES = 500

# Full run settings (200-dim needs more warmup)
NUM_WARMUP = 15000
NUM_SAMPLES = 5000
NUM_CHAINS = 4
CHAIN_METHOD = 'sequential'

SEEDS = [42, 123, 7]
if _task_id is not None:
    SEEDS = [PARAMETER_GRID[_task_id]["seed"]]

# Batch decoding to avoid OOM
DECODE_BATCH = 500

# NF reparameterization alpha
NF_ALPHA = 5.0

PROBLEM_NAME = 'ood_darcy_piecewise_5v1000'

print(f"JAX: {jax.__version__}, NumPyro: {numpyro.__version__}")
print(f"Devices: {jax.devices()}")

# %% [markdown]
# ## 1. Setup

# %%
# Two problem instances (both need train_data_path for normalization).
# Use load_problem for problem_in; manual init for problem_ood (different test data, same checkpoint).
problem_in = DarcyPiecewise(
    seed=42, train_data_path=TRAIN_DATA_PATH, test_data_path=IN_DATA_PATH,
)
params = load_problem(problem_in, CHECKPOINT_PATH)

problem_ood = DarcyPiecewise(
    seed=42, train_data_path=TRAIN_DATA_PATH, test_data_path=OOD_DATA_PATH,
)
problem_ood.initialize_models(problem_in.get_sample_inputs(batch_size=1))
ckpt_result_ood = problem_ood.load_checkpoint(CHECKPOINT_PATH)
if hasattr(problem_ood, 'load_checkpoint_metadata'):
    problem_ood.load_checkpoint_metadata(ckpt_result_ood)

# NF mode and latent dim
d = problem_in.BETA_SIZE

# z_init: center of Beta distribution in [0,1] space
z_init = 0.5 * jnp.ones(d)

n_points = problem_in.get_n_points()

print(f"Latent dim: {d}, n_points: {n_points}")
print(f"Normalization: a_mean={problem_in.a_mean is not None}, a_std={problem_in.a_std is not None}")

# %% [markdown]
# ## 2. PDE Collocation and Physics Closure

# %%
pde_rng = random.PRNGKey(123)
xc_fixed, R_fixed = problem_in.genPoint.weight_centers(
    n_center=NC_PDE, R_max=1e-4, R_min=1e-4, key=pde_rng,
)
x_pde_fixed = (problem_in.int_grid[None, :, :] * R_fixed + xc_fixed).reshape(-1, 2)

print(f"Fixed PDE grid: {x_pde_fixed.shape}  ({NC_PDE} centers x {problem_in.n_grid} pts)")


def log_pde_fn(beta, rho_pde):
    """PDE virtual-observable log-likelihood (mixed formulation with s1, s2)."""
    beta_b = beta[None, :]
    x_pde_b = x_pde_fixed[None, :, :]

    # Decode coefficient: logits -> sigmoid -> deterministic a_sample
    a_logits = problem_in.models['a'].apply({'params': params['a']}, x_pde_b, beta_b)
    a_prob = jax.nn.sigmoid(a_logits)
    a_vals = a_sample(a_prob[..., None], k_low=problem_in.K_LOW, k_high=problem_in.K_HIGH)  # rng=None -> deterministic
    a_vals_flat = a_vals[0]  # (nc*n_grid, 1)

    residuals = compute_pde_residual_piecewise_single_sample(
        params['u'], params['s1'], params['s2'],
        problem_in.models['u'], problem_in.models['s1'], problem_in.models['s2'],
        beta, xc_fixed, R_fixed,
        problem_in.int_grid, problem_in.v, problem_in.dv_dr,
        a_vals_flat, problem_in.n_grid,
    )
    return -0.5 * jnp.sum(residuals ** 2) / (rho_pde ** 2)


# %% [markdown]
# ## 3. Inline NumPyro Models (NF Reparameterization)
#
# DarcyPiecewise CANNOT use make_numpyro_model — it uses Beta(NF_ALPHA, NF_ALPHA) prior
# with z→beta NF transform. Models accept log_lik_fn as an argument to support
# fresh likelihood closures per seed.

# %%
def numpyro_model_data_only(sigma=0.1, log_lik_fn=None):
    """NF reparameterized data-only model."""
    z_01 = numpyro.sample("z", dist.Beta(NF_ALPHA, NF_ALPHA).expand([d]).to_event(1))
    z = 2.0 * z_01 - 1.0
    beta, _ = problem_in.models['nf'].apply(
        {'params': params['nf']}, z[None, :], method=problem_in.models['nf'].inverse
    )
    beta = beta[0]
    numpyro.factor("data_lik", log_lik_fn(beta, sigma))


def numpyro_model_physics(sigma=0.1, rho_pde=1.0, log_lik_fn=None):
    """NF reparameterized model with physics constraint."""
    z_01 = numpyro.sample("z", dist.Beta(NF_ALPHA, NF_ALPHA).expand([d]).to_event(1))
    z = 2.0 * z_01 - 1.0
    beta, _ = problem_in.models['nf'].apply(
        {'params': params['nf']}, z[None, :], method=problem_in.models['nf'].inverse
    )
    beta = beta[0]
    numpyro.factor("data_lik", log_lik_fn(beta, sigma))
    numpyro.factor("pde_lik", log_pde_fn(beta, rho_pde))


# %% [markdown]
# ## 4. Sigmoid Decode Helper
#
# DarcyPiecewise CANNOT use decode_posterior_batched — uses sigmoid decode via a_sample.

# %%
def _decode_piecewise(beta_samples, x_full):
    """Decode beta samples -> a_pred (n_samples, n_points, 1) and u_pred (n_samples, n_points).

    Uses sigmoid decode + a_sample (not predict_from_beta), as required for piecewise.
    """
    n_s = beta_samples.shape[0]
    a_pred_list = []
    u_pred_list = []

    for i in range(0, n_s, DECODE_BATCH):
        batch = beta_samples[i:i + DECODE_BATCH]
        bs = batch.shape[0]
        x_tile = jnp.tile(x_full, (bs, 1, 1))

        u_pred = problem_in.models['u'].apply({'params': params['u']}, x_tile, batch)
        if u_pred.ndim == 2:
            u_pred = u_pred[..., None]
        u_pred = mollifier(u_pred.squeeze(-1), x_tile)
        u_pred_list.append(np.array(u_pred))

        a_logits = problem_in.models['a'].apply({'params': params['a']}, x_tile, batch)
        a_prob = jax.nn.sigmoid(a_logits)
        a_decoded = a_sample(a_prob[..., None], k_low=problem_in.K_LOW, k_high=problem_in.K_HIGH)  # (bs, n_points, 1)
        a_pred_list.append(np.array(a_decoded))

    a_pred = np.concatenate(a_pred_list, axis=0)   # (n_samples, n_points, 1)
    u_pred = np.concatenate(u_pred_list, axis=0)   # (n_samples, n_points)
    return a_pred, u_pred


# %% [markdown]
# ## 5. Prepare Tuning Observations (Seed-42)

# %%
_rng_tune = random.PRNGKey(42)
_rng_tune, _key_tune = random.split(_rng_tune)
_obs_indices_tune = problem_in.sample_observation_indices(n_points, N_OBS, 'random', _key_tune)

obs_in_tune = problem_in.prepare_observations(sample_indices=[TEST_IDX], obs_indices=_obs_indices_tune)
obs_ood_tune = problem_ood.prepare_observations(sample_indices=[TEST_IDX], obs_indices=_obs_indices_tune)

x_full_in = obs_in_tune['x_full']
x_full_ood = obs_ood_tune['x_full']
a_true_in_tune = obs_in_tune['a_true']
a_true_ood_tune = obs_ood_tune['a_true']

print(f"In-domain a_true range: [{float(a_true_in_tune.min()):.1f}, {float(a_true_in_tune.max()):.1f}] (expect {{5, 1000}})")
print(f"OOD       a_true range: [{float(a_true_ood_tune.min()):.1f}, {float(a_true_ood_tune.max()):.1f}]")
print(f"d={d}, NF_ALPHA={NF_ALPHA}, z_init ready")

# %% [markdown]
# ## 6. MAP Baselines and Sigma (Tuning, Seed-42)

# %%
inv_config = InversionConfig(
    epochs=200,
    loss_weights=LossWeights(pde=1.0, data=1.0),
    optimizer=OptimizerConfig(type='Adam', lr=0.1),
    scheduler=SchedulerConfig(type='StepLR', step_size=40, gamma=0.1),
)

# In-domain MAP (seed-42 tuning)
_rng_tune, inv_rng = random.split(_rng_tune)
inverter_in = IGNOInverter(problem_in, inv_rng)
beta_map_in_tune = inverter_in.invert(
    obs_in_tune['x_obs'], obs_in_tune['u_obs'], x_full_in, inv_config, verbose=True,
)
print(f"In-domain MAP I_corr:  {cross_correlation(problem_in.predict_from_beta(params, beta_map_in_tune, x_full_in)['a_pred'][0], a_true_in_tune[0]):.6f}")

# Sigma from MAP residual on in-domain seed-42 observations
_preds_map_in_obs = problem_in.predict_from_beta(params, beta_map_in_tune, obs_in_tune['x_obs'])
SIGMA = compute_sigma_from_map(_preds_map_in_obs['u_pred'], obs_in_tune['u_obs'])
print(f"\nSIGMA = {SIGMA:.6f}  (MAP residual on in-domain seed-42 observations)")

# %% [markdown]
# ## 7. Rho_pde Tuning (In-Domain Only)
#
# Sweep physics constraint strength on in-domain data. Same rho applied to OOD.
# Note: I_corr (higher = better) is used as error metric (inverted compared to other problems).

# %%
# Quick in-domain data-only baseline for rho selection threshold
log_lik_in_tune = make_gaussian_log_likelihood(
    problem_in, params, mollifier, obs_in_tune['x_obs'], obs_in_tune['u_obs'],
)

nuts_cfg_sweep = {'target_accept_prob': 0.65, 'max_tree_depth': 8, 'dense_mass': False}
_rng_tune, mcmc_key = random.split(_rng_tune)
mcmc_baseline, _ = run_mcmc(
    numpyro_model_data_only, {"z": z_init},
    {"sigma": SIGMA, "log_lik_fn": log_lik_in_tune},
    mcmc_key, SWEEP_WARMUP, SWEEP_SAMPLES, 1, 'sequential', nuts_cfg_sweep,
)

z_01_bl = mcmc_baseline.get_samples()["z"]
z_bl = 2.0 * z_01_bl - 1.0
beta_bl, _ = problem_in.models['nf'].apply(
    {'params': params['nf']}, z_bl, method=problem_in.models['nf'].inverse
)
a_pred_bl, _ = _decode_piecewise(beta_bl, x_full_in)
a_np_bl = a_pred_bl[:, :, 0]
a_true_in_tune_np = np.array(a_true_in_tune[0, :, 0])
dec_baseline = compute_piecewise_metrics(a_np_bl, a_true_in_tune_np, 5.0, 1000.0, a_err_fn=lambda pred, true: cross_correlation(pred, true, k_low=5.0, k_high=1000.0))
baseline_a_err = dec_baseline['a_err']
print(f"In-domain data-only baseline: a_err(I_corr)={baseline_a_err:.4f}, coverage={dec_baseline['coverage_95']:.2%}")

# %%
print(f"Rho sweep on IN-DOMAIN with NF reparam: {SWEEP_WARMUP} warmup, {SWEEP_SAMPLES} samples per rho")
print(f"rho values: {RHO_PDE_VALUES}")

sweep_results = []
for rho in RHO_PDE_VALUES:
    print(f"\n{'='*60}\nrho_pde = {rho}\n{'='*60}")
    _rng_tune, key = random.split(_rng_tune)
    _t_pilot = time.time()
    mcmc_sw, _ = run_mcmc(
        numpyro_model_physics, {"z": z_init},
        {"sigma": SIGMA, "rho_pde": rho, "log_lik_fn": log_lik_in_tune},
        key, SWEEP_WARMUP, SWEEP_SAMPLES, 1, 'sequential', nuts_cfg_sweep,
    )
    print(f"  Pilot rho={rho} completed in {time.time() - _t_pilot:.1f}s")

    z_01_s = mcmc_sw.get_samples()["z"]
    z_s = 2.0 * z_01_s - 1.0
    beta_s, _ = problem_in.models['nf'].apply(
        {'params': params['nf']}, z_s, method=problem_in.models['nf'].inverse
    )
    a_pred_s, _ = _decode_piecewise(beta_s, x_full_in)
    a_np_s = a_pred_s[:, :, 0]
    dec = compute_piecewise_metrics(a_np_s, a_true_in_tune_np, 5.0, 1000.0, a_err_fn=lambda pred, true: cross_correlation(pred, true, k_low=5.0, k_high=1000.0))

    n_div = int(mcmc_sw.get_extra_fields().get('diverging', np.array([])).sum())
    ess = effective_sample_size(np.array(z_01_s)[None])

    sweep_results.append({
        'rho_pde': rho, 'a_err': dec['a_err'], 'coverage': dec['coverage_95'],
        'ess_min': float(ess.min()), 'n_div': n_div, 'crps_a': dec['crps_a'],
    })
    print(f"  a_err(I_corr)={dec['a_err']:.4f}  coverage={dec['coverage_95']:.2%}  "
          f"ESS_min={float(ess.min()):.1f}  n_div={n_div}")

# Select best rho (I_corr: higher = better, so threshold is >= 0.5x baseline)
a_err_threshold = baseline_a_err * 0.5
candidates = [(abs(r['coverage'] - 0.95), r) for r in sweep_results
              if r['a_err'] >= a_err_threshold]
if candidates:
    candidates.sort(key=lambda x: x[0])
    BEST_RHO_PDE = candidates[0][1]['rho_pde']
else:
    BEST_RHO_PDE = max(RHO_PDE_VALUES)
    print("No candidate with a_err >= 0.5x baseline (I_corr: higher=better); using largest rho.")

print(f"\nSelected BEST_RHO_PDE = {BEST_RHO_PDE}")

# %% [markdown]
# ## 8. Run Condition Helper

# %%
def run_condition(model_fn, model_kwargs, x_full, a_true_ref, obs_indices,
                  u_obs_np, label, seed):
    """Run NUTS for one condition, decode, and compute metrics."""
    print(f"\n{'='*60}\n  {label}\n{'='*60}")

    nuts_cfg = recommended_nuts_config(d, model_kwargs['sigma'])
    print(f"  NUTS config: {nuts_cfg}")
    print(f"  {NUM_WARMUP} warmup, {NUM_SAMPLES} samples, {NUM_CHAINS} chains ({CHAIN_METHOD})")

    rng_local = random.PRNGKey(seed)
    mcmc, timing = run_mcmc(
        model_fn, {"z": z_init}, model_kwargs,
        rng_local, NUM_WARMUP, NUM_SAMPLES, NUM_CHAINS, CHAIN_METHOD, nuts_cfg,
    )
    mcmc.print_summary(exclude_deterministic=True)

    diag = extract_mcmc_diagnostics(mcmc, sample_name="z",
                                    total_samples=NUM_CHAINS * NUM_SAMPLES)

    # Convert z -> beta via NF inverse
    z_01_s = diag['samples']
    z_s = 2.0 * z_01_s - 1.0
    beta_s, _ = problem_in.models['nf'].apply(
        {'params': params['nf']}, z_s, method=problem_in.models['nf'].inverse
    )

    # Sigmoid decode (piecewise-specific)
    a_pred, u_pred = _decode_piecewise(beta_s, x_full)
    a_samples = a_pred[:, :, 0]   # (n_samples, n_points)
    a_true_np = np.array(a_true_ref[0, :, 0])

    metrics = compute_piecewise_metrics(a_samples, a_true_np, 5.0, 1000.0, a_err_fn=lambda pred, true: cross_correlation(pred, true, k_low=5.0, k_high=1000.0))

    a_mean = np.mean(a_samples, axis=0)
    a_std = np.std(a_samples, axis=0)
    spearman_rho, spearman_p = compute_error_std_correlation(a_true_np, a_mean, a_std)

    u_at_obs = u_pred[:, obs_indices]   # (n_samples, n_obs)
    chi2_stat, chi2_pval = chi2_ppc(u_obs_np, u_at_obs, model_kwargs['sigma'])
    print(f"  {label}: chi2={chi2_stat:.2f}, p={chi2_pval:.4f}")

    return {
        'label': label,
        'a_samples': a_samples,
        'a_pred': a_pred,         # (n_samples, n_points, 1) — for plots expecting [..., 0]
        'u_pred': u_pred,         # (n_samples, n_points)
        'a_mean': a_mean,
        'a_std': a_std,
        'u_mean': np.mean(u_pred, axis=0),
        **metrics,
        'sigma': model_kwargs['sigma'],
        'ess_min': diag['ess_min'],
        'rhat_max': diag['rhat_max'],
        'rhat_mean': diag['rhat_mean'],
        'n_div': diag['n_div'],
        'reliability_flag': diag['flag'],
        'reliability_explanation': diag['flag_explanation'],
        'chi2_ppc': chi2_stat,
        'chi2_ppc_pvalue': chi2_pval,
        'spearman_rho_error_std': float(spearman_rho),
        'spearman_pvalue_error_std': float(spearman_p),
        'warmup_time_s': timing['warmup_time_s'],
        'sampling_time_s': timing['sampling_time_s'],
        'step_time_s': timing['step_time_s'],
        'beta_by_chain': diag['by_chain'],       # z-space, for trace plots
        'beta_for_trace': diag['by_chain'][0],   # z-space first chain
    }


# %% [markdown]
# ## 9. Multi-Seed Full MCMC Runs

# %%
for SEED in SEEDS:
    print(f"\n{'#'*70}\n## SEED = {SEED}\n{'#'*70}")
    _t_total_start = time.time()

    rng = random.PRNGKey(SEED)

    # Fresh observations per seed (shared indices for in-domain and OOD)
    rng, key = random.split(rng)
    obs_indices = problem_in.sample_observation_indices(n_points, N_OBS, 'random', key)

    obs_in = problem_in.prepare_observations(sample_indices=[TEST_IDX], obs_indices=obs_indices)
    obs_ood = problem_ood.prepare_observations(sample_indices=[TEST_IDX], obs_indices=obs_indices)

    x_full_in = obs_in['x_full']
    x_obs_in = obs_in['x_obs']
    u_obs_in = obs_in['u_obs']
    a_true_in = obs_in['a_true']
    u_true_in = obs_in['u_true']

    x_full_ood = obs_ood['x_full']
    x_obs_ood = obs_ood['x_obs']
    u_obs_ood = obs_ood['u_obs']
    a_true_ood = obs_ood['a_true']
    u_true_ood = obs_ood['u_true']

    a_true_in_np = np.array(a_true_in[0, :, 0])
    a_true_ood_np = np.array(a_true_ood[0, :, 0])

    FIGURE_DIR = Path(f'figures/{PROBLEM_NAME}/seed_{SEED}')
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    # === MAP Baselines ===
    rng, inv_rng_in, inv_rng_ood = random.split(rng, 3)

    inverter_in = IGNOInverter(problem_in, inv_rng_in)
    _t_map_in = time.time()
    beta_map_in = inverter_in.invert(x_obs_in, u_obs_in, x_full_in, inv_config, verbose=False)
    _map_in_time_s = time.time() - _t_map_in
    print(f"MAP (in-domain) completed in {_map_in_time_s:.1f}s")
    preds_map_in = problem_in.predict_from_beta(params, beta_map_in, x_full_in)
    a_map_in = preds_map_in['a_pred'][0]
    icorr_map_in = float(cross_correlation(a_map_in, a_true_in[0], k_low=5.0, k_high=1000.0))
    print(f"[seed={SEED}] In-domain MAP I_corr: {icorr_map_in:.6f}")
    print("  (Note: RMSE is meaningless for piecewise constant fields — use I_corr instead)")

    inverter_ood = IGNOInverter(problem_ood, inv_rng_ood)
    _t_map_ood = time.time()
    beta_map_ood = inverter_ood.invert(x_obs_ood, u_obs_ood, x_full_ood, inv_config, verbose=False)
    _map_ood_time_s = time.time() - _t_map_ood
    print(f"MAP (OOD) completed in {_map_ood_time_s:.1f}s")
    preds_map_ood = problem_ood.predict_from_beta(params, beta_map_ood, x_full_ood)
    a_map_ood = preds_map_ood['a_pred'][0]
    u_map_ood = np.array(preds_map_ood['u_pred'][0, :, 0])
    icorr_map_ood = float(cross_correlation(a_map_ood, a_true_ood[0], k_low=5.0, k_high=1000.0))
    print(f"[seed={SEED}] OOD MAP I_corr: {icorr_map_ood:.6f}")

    # === Prior Predictive Baseline ===
    _icorr_fn = lambda pred, true: float(cross_correlation(
        jnp.array(np.where(np.asarray(pred) >= 502.5, 1000.0, 5.0)), jnp.array(true),
        k_low=5.0, k_high=1000.0,
    ))
    prior_a_samples, prior_metrics_in, rng = sample_unconditional_prior(
        problem_in, params, x_full_in, a_true_in_np, rng, error_fn=_icorr_fn,
    )
    prior_metrics_ood = compute_prior_predictive(prior_a_samples, a_true_ood_np, error_fn=_icorr_fn)
    print(f"Prior (in-domain): a_err={prior_metrics_in['a_err']:.4f}, CRPS={prior_metrics_in['crps_a']:.4f}")
    print(f"Prior (OOD):       a_err={prior_metrics_ood['a_err']:.4f}, CRPS={prior_metrics_ood['crps_a']:.4f}")

    # === Build per-seed likelihood closures ===
    log_lik_in = make_gaussian_log_likelihood(problem_in, params, mollifier, x_obs_in, u_obs_in)
    log_lik_ood = make_gaussian_log_likelihood(problem_in, params, mollifier, x_obs_ood, u_obs_ood)

    _u_obs_in_np = np.array(u_obs_in[0, :, 0])
    _u_obs_ood_np = np.array(u_obs_ood[0, :, 0])

    # === 4 Conditions ===
    res_in_do = run_condition(
        numpyro_model_data_only,
        {"sigma": SIGMA, "log_lik_fn": log_lik_in},
        x_full_in, a_true_in, obs_indices, _u_obs_in_np,
        "In-domain Data-Only", seed=SEED + 0,
    )
    res_in_phys = run_condition(
        numpyro_model_physics,
        {"sigma": SIGMA, "rho_pde": BEST_RHO_PDE, "log_lik_fn": log_lik_in},
        x_full_in, a_true_in, obs_indices, _u_obs_in_np,
        "In-domain Physics", seed=SEED + 1,
    )
    res_ood_do = run_condition(
        numpyro_model_data_only,
        {"sigma": SIGMA, "log_lik_fn": log_lik_ood},
        x_full_ood, a_true_ood, obs_indices, _u_obs_ood_np,
        "OOD Data-Only", seed=SEED + 2,
    )
    res_ood_phys = run_condition(
        numpyro_model_physics,
        {"sigma": SIGMA, "rho_pde": BEST_RHO_PDE, "log_lik_fn": log_lik_ood},
        x_full_ood, a_true_ood, obs_indices, _u_obs_ood_np,
        "OOD Physics", seed=SEED + 3,
    )

    # === Metrics Comparison ===
    def make_metrics_dict(res):
        return {
            'I_corr': res['a_err'],
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
        title=f'Darcy Piecewise 5v1000 — sigma={SIGMA}, rho_pde={BEST_RHO_PDE} (seed={SEED})',
    )

    # === Significance of Pairwise Contrasts ===
    def _crps_a(s, t): return float(np.mean(crps_ensemble(s, t)))
    def _coverage_95(s, t):
        _, emp = compute_calibration(s, t, np.array([0.95]))
        return float(emp[0])
    def _ci_width(s, t): return ci_width_95(s)

    a_np_in_do   = res_in_do['a_samples']
    a_np_in_phys = res_in_phys['a_samples']
    a_np_ood_do  = res_ood_do['a_samples']
    a_np_ood_phys = res_ood_phys['a_samples']

    rng_bs = np.random.default_rng(SEED)

    d_in_phys_crps  = bootstrap_metric_difference_ci(a_np_in_do, a_np_in_phys, a_true_in_np, _crps_a, rng=rng_bs)
    d_in_phys_cov   = bootstrap_metric_difference_ci(a_np_in_do, a_np_in_phys, a_true_in_np, _coverage_95, rng=rng_bs)
    d_in_phys_width = bootstrap_metric_difference_ci(a_np_in_do, a_np_in_phys, a_true_in_np, _ci_width, rng=rng_bs)
    d_ood_phys_crps  = bootstrap_metric_difference_ci(a_np_ood_do, a_np_ood_phys, a_true_ood_np, _crps_a, rng=rng_bs)
    d_ood_phys_cov   = bootstrap_metric_difference_ci(a_np_ood_do, a_np_ood_phys, a_true_ood_np, _coverage_95, rng=rng_bs)
    d_ood_phys_width = bootstrap_metric_difference_ci(a_np_ood_do, a_np_ood_phys, a_true_ood_np, _ci_width, rng=rng_bs)
    d_ood_do_crps    = bootstrap_metric_difference_ci(a_np_ood_do, a_np_in_do, a_true_ood_np, _crps_a, rng=rng_bs)
    d_ood_phys_deg   = bootstrap_metric_difference_ci(a_np_ood_phys, a_np_in_phys, a_true_ood_np, _crps_a, rng=rng_bs)

    format_significance_table({
        '(a) In-Phys CRPS diff (DO - Phys)':      d_in_phys_crps,
        '(a) In-Phys Coverage diff (DO - Phys)':  d_in_phys_cov,
        '(a) In-Phys CI Width diff (DO - Phys)':  d_in_phys_width,
        '(b) OOD-Phys CRPS diff (DO - Phys)':     d_ood_phys_crps,
        '(b) OOD-Phys Coverage diff (DO - Phys)': d_ood_phys_cov,
        '(b) OOD-Phys CI Width diff (DO - Phys)': d_ood_phys_width,
        '(c) OOD degradation CRPS (DO)':           d_ood_do_crps,
        '(d) OOD degradation CRPS (Phys)':         d_ood_phys_deg,
    }, title=f'Darcy Piecewise 5v1000 — RQ3: Pairwise Contrasts (Bootstrap, seed={SEED})')
    print("Negative CRPS/CI-Width diff = improved; positive Coverage diff = improved.")
    print("Note: 4 contrasts x 3 metrics = 12 tests; Bonferroni threshold 0.05/12 ≈ 0.004")

    # === Physics Benefit Delta Analysis (RQ3 Hardening) ===
    _rng_pb = np.random.default_rng(SEED + 5000)

    delta_in_crps  = bootstrap_metric_difference_ci(a_np_in_phys,  a_np_in_do,  a_true_in_np,  _crps_a, rng=_rng_pb)
    delta_ood_crps = bootstrap_metric_difference_ci(a_np_ood_phys, a_np_ood_do, a_true_ood_np, _crps_a, rng=_rng_pb)

    ci_in_do_pb    = bootstrap_metric_ci(a_np_in_do,    a_true_in_np,  _crps_a, rng=_rng_pb)
    ci_in_phys_pb  = bootstrap_metric_ci(a_np_in_phys,  a_true_in_np,  _crps_a, rng=_rng_pb)
    ci_ood_do_pb   = bootstrap_metric_ci(a_np_ood_do,   a_true_ood_np, _crps_a, rng=_rng_pb)
    ci_ood_phys_pb = bootstrap_metric_ci(a_np_ood_phys, a_true_ood_np, _crps_a, rng=_rng_pb)

    _n_dod = 1000
    _dod_samples = np.empty(_n_dod)
    _rng_dod = np.random.default_rng(SEED + 6000)
    _na_in_do, _na_in_p   = a_np_in_do.shape[0],  a_np_in_phys.shape[0]
    _na_ood_do, _na_ood_p = a_np_ood_do.shape[0], a_np_ood_phys.shape[0]
    for _i in range(_n_dod):
        _b_in_do  = a_np_in_do[_rng_dod.integers(0, _na_in_do,  _na_in_do)]
        _b_in_p   = a_np_in_phys[_rng_dod.integers(0, _na_in_p,   _na_in_p)]
        _b_ood_do = a_np_ood_do[_rng_dod.integers(0, _na_ood_do, _na_ood_do)]
        _b_ood_p  = a_np_ood_phys[_rng_dod.integers(0, _na_ood_p,  _na_ood_p)]
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
    }, title=f'Darcy Piecewise 5v1000 — Physics Benefit Analysis (seed={SEED})')
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

    # === Save Structured Result ===
    conditions_raw = {
        "in_domain_data_only": res_in_do,
        "in_domain_physics": res_in_phys,
        "ood_data_only": res_ood_do,
        "ood_physics": res_ood_phys,
    }

    # MAP metric: I_corr for piecewise (higher = better)
    _map_a_err_for = {
        "in_domain_data_only": icorr_map_in,
        "in_domain_physics": icorr_map_in,
        "ood_data_only": icorr_map_ood,
        "ood_physics": icorr_map_ood,
    }

    def _build_condition(k, v):
        d_res = dict(v)
        d_res["map_a_err"] = _map_a_err_for[k]
        d_res["coverage"] = d_res.get("coverage_95")
        return build_mcmc_result(d_res, NUM_WARMUP, NUM_SAMPLES, NUM_CHAINS)

    experiment = ExperimentResult(
        experiment="ood",
        problem="darcy_piecewise_5v1000",
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


    # === Plots ===
    x_np_ood = np.array(x_full_ood[0])
    a_map_ood_np = np.array(a_map_ood[:, 0])

    plot_std_comparison_generic(
        x_np_ood,
        std_a=res_ood_do['a_std'],
        std_b=res_ood_phys['a_std'],
        label_a='OOD (DO)',
        label_b='OOD (Phys)',
        grid_shape=(29, 29),
        suptitle='Uncertainty: OOD Data-Only vs OOD Physics',
        save_path=FIGURE_DIR / 'std_ood_do_vs_physics.png',
    )

    plot_calibration_overlay(
        [
            (res_in_do['cal_levels'], res_in_do['cal_empirical'], 'In-Domain DO'),
            (res_in_phys['cal_levels'], res_in_phys['cal_empirical'], 'In-Domain Phys'),
            (res_ood_do['cal_levels'], res_ood_do['cal_empirical'], 'OOD DO'),
            (res_ood_phys['cal_levels'], res_ood_phys['cal_empirical'], 'OOD Phys'),
        ],
        save_path=FIGURE_DIR / 'calibration_overlay_4way.png',
    )

    plot_field_comparison(
        x_np_ood, a_true_ood_np, a_map_ood_np,
        res_ood_do['a_mean'],
        res_ood_do['a_std'],
        grid_shape=(29, 29),
        u_true=np.array(u_true_ood[0, :, 0]),
        u_map=u_map_ood,
        u_mean=res_ood_do['u_mean'],
        u_std=np.std(res_ood_do['u_pred'], axis=0),
        obs_coords=np.array(x_obs_ood[0]),
        save_path=FIGURE_DIR / 'field_comparison_ood_data_only.png',
        show_abs_error=False,
        piecewise_a_bounds=(5.0, 1000.0),
    )

    plot_field_comparison(
        x_np_ood, a_true_ood_np, a_map_ood_np,
        res_ood_phys['a_mean'],
        res_ood_phys['a_std'],
        grid_shape=(29, 29),
        u_true=np.array(u_true_ood[0, :, 0]),
        u_map=u_map_ood,
        u_mean=res_ood_phys['u_mean'],
        u_std=np.std(res_ood_phys['u_pred'], axis=0),
        obs_coords=np.array(x_obs_ood[0]),
        save_path=FIGURE_DIR / 'field_comparison_ood_physics.png',
        show_abs_error=False,
        piecewise_a_bounds=(5.0, 1000.0),
    )

    plot_posterior_gallery(
        x_np_ood, a_np_ood_do, grid_shape=(29, 29),
        piecewise_a_bounds=(5.0, 1000.0),
        a_true=a_true_ood_np, n_show=6,
        save_path=FIGURE_DIR / 'posterior_gallery_ood_data_only.png',
    )

    plot_posterior_gallery(
        x_np_ood, a_np_ood_phys,
        piecewise_a_bounds=(5.0, 1000.0), grid_shape=(29, 29),
        a_true=a_true_ood_np, n_show=6,
        save_path=FIGURE_DIR / 'posterior_gallery_ood_physics.png',
    )

    # === Diagnostics ===
    for _label, _res in [("OOD Data-Only", res_ood_do), ("OOD Physics", res_ood_phys)]:
        _bbc = _res['beta_by_chain']
        _ess = effective_sample_size(_bbc)
        _rhat = split_gelman_rubin(_bbc)

        print(f"\n{_label} multi-chain diagnostics ({NUM_CHAINS} chains, {d} dimensions):")
        print(f"  ESS  — min: {float(_ess.min()):.1f},  max: {float(_ess.max()):.1f},  "
              f"mean: {float(_ess.mean()):.1f},  median: {float(np.median(_ess)):.1f}")
        print(f"  R-hat — max: {float(_rhat.max()):.4f},  mean: {float(_rhat.mean()):.4f}")
        print(f"  dims with ESS < 10:  {int((_ess < 10).sum())} / {d}")
        print(f"  dims with ESS < 50:  {int((_ess < 50).sum())} / {d}")
        print(f"  dims with R-hat > 1.1: {int((_rhat > 1.1).sum())} / {d}")
        print(f"  dims with R-hat > 1.01: {int((_rhat > 1.01).sum())} / {d}")
        print(f"\n  Divergences: {_res['n_div']} / {NUM_CHAINS * NUM_SAMPLES}")

        _suffix = 'ood_do' if 'Data-Only' in _label else 'ood_phys'
        plot_trace(_res['beta_for_trace'][:, :8], beta_true=None, num_warmup=0,
                   save_path=FIGURE_DIR / f'trace_plots_{_suffix}_first8.png')


# %% [markdown]
# ## Cross-Seed Aggregation Summary

# %%
print_cross_seed_summary("ood", "darcy_piecewise_5v1000")
