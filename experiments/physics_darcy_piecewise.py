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

# %%
import sys, itertools, time, functools
sys.path.insert(0, 'experiment_utils')
from _slurm import parse_slurm_task

PARAMETER_GRID = [
    {"seed": s, "test_idx": t}
    for s, t in itertools.product([42, 123, 7], [0, 1, 2])
]
_params, _task_id = parse_slurm_task(PARAMETER_GRID)

# %% [markdown]
# # Physics Constraint Comparison: Darcy Piecewise
#
# - PDE: $-\nabla \cdot (a \nabla u) = 10$, piecewise constant coefficient function $\{5, 10\}$
# - Latent dimension: $d = 200$
# - Physics term: mixed PDE formulation (constitutive and weak) with stress variables $s_1, s_2$

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
    DarcyPiecewise,
    mollifier,
    a_sample,
    compute_pde_residual_piecewise_single_sample,
)
from src.evaluation.metrics import rmse, cross_correlation
from src.solver.config import InversionConfig, LossWeights, OptimizerConfig, SchedulerConfig

from experiment_utils import (
    crps_ensemble, compute_calibration, ci_width_95, nll_score, compute_piecewise_metrics,
    compute_prior_predictive, build_prior_result,
    plot_field_comparison, plot_calibration, plot_posterior_gallery,
    plot_posterior_predictive, plot_trace, plot_metrics_table,
    plot_rho_sweep, plot_sharpness_calibration_tradeoff,
    plot_std_comparison, plot_metrics_comparison_table,
    bootstrap_metric_ci, compute_per_chain_metrics,
    bootstrap_metric_difference_ci, format_significance_table,
    compute_sigma_from_map, recommended_nuts_config, mcmc_reliability_flag,
    make_nf_reparameterized_model,
    compute_metric_convergence, plot_metric_convergence,
    chi2_ppc,
    load_problem, make_log_prior,
    run_mcmc, extract_mcmc_diagnostics,
    compute_standard_metrics,
    build_mcmc_result, save_experiment_result,
    compute_error_std_correlation,
    load_cross_seed_results, cross_seed_metric_summary,
    sample_unconditional_prior,
    _use_science_style,
)

SEEDS = [42, 123, 7]
if _task_id is not None:
    SEEDS = [PARAMETER_GRID[_task_id]["seed"]]

TEST_IDX = 0
if _task_id is not None:
    TEST_IDX = PARAMETER_GRID[_task_id]["test_idx"]

print(f"JAX: {jax.__version__}, NumPyro: {numpyro.__version__}")
print(f"Devices: {jax.devices()}")

# %% [markdown]
# ## 1. Load Trained Model

# %%
CHECKPOINT_PATH = Path('../runs/final_darcy_piecewise_updated_dims/weights/best.pt')
TRAIN_DATA_PATH = '../data/darcy_piecewise/pwc_train_data1000.mat'
TEST_DATA_PATH = '../data/darcy_piecewise/pwc_test_in.mat'

problem = DarcyPiecewise(
    seed=42,
    train_data_path=TRAIN_DATA_PATH,
    test_data_path=TEST_DATA_PATH,
)
params = load_problem(problem, CHECKPOINT_PATH)
log_prior_fn = make_log_prior(problem, params)

d = problem.BETA_SIZE
n_points = problem.get_n_points()

# NF mode
z_mode = jnp.zeros((1, d))
beta_mode, _ = problem.models['nf'].apply(
    {'params': params['nf']}, z_mode, method=problem.models['nf'].inverse
)
beta_mode = beta_mode[0]
z_init = 0.5 * jnp.ones(d)

print(f"Latent dim: {d}")
print(f"Normalization: a_mean={problem.a_mean is not None}, a_std={problem.a_std is not None}")

# %% [markdown]
# ## 2. Config

# %%
N_OBS = 100
NC_PDE = 30

RHO_PDE_VALUES = [5.0, 2.0, 1.0, 0.5, 0.3, 0.1, 0.05]
SWEEP_WARMUP = 2000
SWEEP_SAMPLES = 500

NUM_WARMUP = 15000
NUM_SAMPLES = 5000
NUM_CHAINS = 4
CHAIN_METHOD = 'sequential'
SAMPLE_NAME = 'z'
DECODE_BATCH = 500
NF_ALPHA = 5.0

inv_config = InversionConfig(
    epochs=200,
    loss_weights=LossWeights(pde=1.0, data=1.0),
    optimizer=OptimizerConfig(type='Adam', lr=0.1),
    scheduler=SchedulerConfig(type='StepLR', step_size=40, gamma=0.1),
)

# %% [markdown]
# ## 3. Fixed PDE Collocation Points

# %%
pde_rng = random.PRNGKey(123)
xc_fixed, R_fixed = problem.genPoint.weight_centers(
    n_center=NC_PDE, R_max=1e-4, R_min=1e-4, key=pde_rng,
)
x_pde_fixed = (problem.int_grid[None, :, :] * R_fixed + xc_fixed).reshape(-1, 2)

print(f"Fixed PDE grid: {x_pde_fixed.shape}  ({NC_PDE} centers x {problem.n_grid} pts)")


def log_pde_fn(beta, rho_pde):
    """PDE virtual-observable log-likelihood (mixed formulation with s1, s2)."""
    beta_b = beta[None, :]
    x_pde_b = x_pde_fixed[None, :, :]

    a_logits = problem.models['a'].apply({'params': params['a']}, x_pde_b, beta_b)
    a_prob = jax.nn.sigmoid(a_logits)
    a_vals = a_sample(a_prob[..., None])
    a_vals_flat = a_vals[0]

    residuals = compute_pde_residual_piecewise_single_sample(
        params['u'], params['s1'], params['s2'],
        problem.models['u'], problem.models['s1'], problem.models['s2'],
        beta, xc_fixed, R_fixed,
        problem.int_grid, problem.v, problem.dv_dr,
        a_vals_flat, problem.n_grid,
    )
    return -0.5 * jnp.sum(residuals ** 2) / (rho_pde ** 2)


def _decode_piecewise_batched(beta_s, x_full_grid):
    """Decode beta samples to a and u fields for Darcy Piecewise (sigmoid decode)."""
    n_s = beta_s.shape[0]
    a_pred_list, u_pred_list = [], []
    for i in range(0, n_s, DECODE_BATCH):
        batch = beta_s[i:i+DECODE_BATCH]
        bs = batch.shape[0]
        x_tile = jnp.tile(x_full_grid, (bs, 1, 1))
        u_pred = problem.models['u'].apply({'params': params['u']}, x_tile, batch)
        if u_pred.ndim == 2:
            u_pred = u_pred[..., None]
        u_pred = mollifier(u_pred.squeeze(-1), x_tile)
        u_pred_list.append(u_pred)
        a_logits = problem.models['a'].apply({'params': params['a']}, x_tile, batch)
        a_prob = jax.nn.sigmoid(a_logits)
        a_decoded = a_sample(a_prob[..., None])
        a_pred_list.append(a_decoded)
    return jnp.concatenate(a_pred_list, axis=0), jnp.concatenate(u_pred_list, axis=0)


def _z_to_beta(z_01_samples):
    """Transform z_01 samples to beta-space via NF inverse."""
    z_s = 2.0 * z_01_samples - 1.0
    beta_s, _ = problem.models['nf'].apply(
        {'params': params['nf']}, z_s, method=problem.models['nf'].inverse
    )
    return beta_s

# %% [markdown]
# ## 4. Sigma (from MAP residual, seed-42 observations)

# %%
_rng_setup = random.PRNGKey(42)
_rng_setup, _key = random.split(_rng_setup)

_obs_indices_tune = problem.sample_observation_indices(n_points, N_OBS, 'random', _key)
_obs_data_tune = problem.prepare_observations(
    sample_indices=[TEST_IDX], obs_indices=_obs_indices_tune,
)
_x_full_tune = _obs_data_tune['x_full']
_x_obs_tune = _obs_data_tune['x_obs']
_u_obs_tune = _obs_data_tune['u_obs']
_a_true_tune = _obs_data_tune['a_true']

from src.evaluation.igno import IGNOInverter

_rng_setup, _inv_rng_tune = random.split(_rng_setup)
_inverter_tune = IGNOInverter(problem, _inv_rng_tune)
_beta_map_tune = _inverter_tune.invert(_x_obs_tune, _u_obs_tune, _x_full_tune, inv_config, verbose=False)
_preds_map_tune = problem.predict_from_beta(params, _beta_map_tune, _x_obs_tune)
SIGMA = compute_sigma_from_map(_preds_map_tune['u_pred'], _u_obs_tune)
print(f"\nSIGMA = {SIGMA:.6f}  (MAP reconstruction RMSE on seed-42 observations)")

# %% [markdown]
# ## 5. rho_pde Sweep (outside seed loop)

# %%
def _log_likelihood_tune(beta, sigma):
    """Gaussian log-likelihood using seed-42 tuning observations."""
    beta_b = beta[None, :]
    u_pred = problem.models['u'].apply({'params': params['u']}, _x_obs_tune, beta_b)
    if u_pred.ndim == 2:
        u_pred = u_pred[..., None]
    u_pred = mollifier(u_pred.squeeze(-1), _x_obs_tune)
    residual = u_pred - _u_obs_tune
    sq_err = jnp.sum(residual ** 2)
    n = _u_obs_tune.shape[1]
    return -0.5 * sq_err / (sigma ** 2) - 0.5 * n * jnp.log(2 * jnp.pi * sigma ** 2)


# Data-only sweep baseline (NF reparameterized)
_nf_model_fn_sweep_do = make_nf_reparameterized_model(
    nf_model=problem.models['nf'], nf_params=params['nf'],
    log_likelihood_fn=_log_likelihood_tune, d=d, nf_alpha=NF_ALPHA,
)

_rng_setup, _sweep_do_key = random.split(_rng_setup)
nuts_cfg_sweep = recommended_nuts_config(d, SIGMA)
_kernel_sweep_do = NUTS(
    functools.partial(_nf_model_fn_sweep_do, sigma=SIGMA),
    init_strategy=init_to_value(values={"z": z_init}),
    target_accept_prob=nuts_cfg_sweep['target_accept_prob'],
    max_tree_depth=nuts_cfg_sweep['max_tree_depth'],
)
_mcmc_sweep_do = MCMC(_kernel_sweep_do, num_warmup=SWEEP_WARMUP, num_samples=SWEEP_SAMPLES,
                      num_chains=1, progress_bar=True)
_mcmc_sweep_do.run(_sweep_do_key, extra_fields=('diverging',))

_beta_sweep_do = _z_to_beta(_mcmc_sweep_do.get_samples()["z"])
_a_pred_sweep_do, _ = _decode_piecewise_batched(_beta_sweep_do, _x_full_tune)
_a_mean_sweep_do = jnp.mean(_a_pred_sweep_do, axis=0)
_a_mean_thresh = np.where(np.array(_a_mean_sweep_do[:, 0]) >= 7.5, 10.0, 5.0)
_a_err_sweep_do = float(cross_correlation(jnp.array(_a_mean_thresh), _a_true_tune[0, :, 0]))
_a_lo_sweep_do = jnp.percentile(_a_pred_sweep_do, 2.5, axis=0)
_a_hi_sweep_do = jnp.percentile(_a_pred_sweep_do, 97.5, axis=0)
_coverage_sweep_do = float(((jnp.array(_a_true_tune[0]) >= _a_lo_sweep_do) & (jnp.array(_a_true_tune[0]) <= _a_hi_sweep_do)).mean())
_ci_width_sweep_do = float(jnp.mean(_a_hi_sweep_do - _a_lo_sweep_do))
_mean_std_sweep_do = float(jnp.mean(jnp.std(_a_pred_sweep_do[:, :, 0], axis=0)))
_sweep_data_only_baseline = {
    'a_err': _a_err_sweep_do,
    'coverage': _coverage_sweep_do,
    'ci_width': _ci_width_sweep_do,
    'mean_std': _mean_std_sweep_do,
}
print(f"Sweep data-only baseline a_err = {_a_err_sweep_do:.4f}")


# %%
def _numpyro_model_physics_sweep(sigma=0.1, rho_pde=1.0):
    """NF reparameterized model with physics constraint (uses tuning obs)."""
    z_01 = numpyro.sample("z", dist.Beta(NF_ALPHA, NF_ALPHA).expand([d]).to_event(1))
    z = 2.0 * z_01 - 1.0
    beta, _ = problem.models['nf'].apply(
        {'params': params['nf']}, z[None, :], method=problem.models['nf'].inverse
    )
    beta = beta[0]
    log_lik = _log_likelihood_tune(beta, sigma)
    numpyro.factor("data_lik", log_lik)
    log_pde = log_pde_fn(beta, rho_pde)
    numpyro.factor("pde_lik", log_pde)


def _run_sweep_one(rho_pde, warmup, n_samples, rng_key):
    """Run NUTS for one rho_pde value (using tuning obs), return metrics dict."""
    nuts_cfg = recommended_nuts_config(d, SIGMA)
    kernel = NUTS(
        _numpyro_model_physics_sweep,
        init_strategy=init_to_value(values={"z": z_init}),
        target_accept_prob=nuts_cfg['target_accept_prob'],
        max_tree_depth=nuts_cfg['max_tree_depth'],
    )
    mcmc = MCMC(kernel, num_warmup=warmup, num_samples=n_samples,
                num_chains=1, chain_method=CHAIN_METHOD, progress_bar=True)
    _t_pilot = time.time()
    mcmc.run(rng_key, sigma=SIGMA, rho_pde=rho_pde, extra_fields=('diverging',))
    print(f"  Pilot completed in {time.time() - _t_pilot:.1f}s")

    z_01_s = mcmc.get_samples()["z"]
    n_div = int(mcmc.get_extra_fields()['diverging'].sum())

    beta_s = _z_to_beta(z_01_s)

    beta_by_chain = np.array(z_01_s)[None, :, :]
    ess_min = float(effective_sample_size(beta_by_chain).min())

    _a_true_np_t = np.array(_a_true_tune[0, :, 0])
    a_pred, u_pred = _decode_piecewise_batched(beta_s, _x_full_tune)

    a_mean = jnp.mean(a_pred, axis=0)
    a_mean_thresh = jnp.where(a_mean >= 7.5, 10.0, 5.0)
    icorr = float(cross_correlation(a_mean_thresh, jnp.array(_a_true_tune[0])))
    u_err = float(jnp.linalg.norm(jnp.mean(u_pred, axis=0)[:, 0] - _obs_data_tune['u_true'][0, :, 0]) /
                  jnp.linalg.norm(_obs_data_tune['u_true'][0, :, 0]))

    a_lo = jnp.percentile(a_pred, 2.5, axis=0)
    a_hi = jnp.percentile(a_pred, 97.5, axis=0)
    coverage = float(((jnp.array(_a_true_tune[0]) >= a_lo) & (jnp.array(_a_true_tune[0]) <= a_hi)).mean())

    a_samples_np = np.array(a_pred[:, :, 0])
    crps_a = float(np.mean(crps_ensemble(a_samples_np, _a_true_np_t)))
    nll_a = nll_score(a_samples_np, _a_true_np_t)
    ci_width_val = float(jnp.mean(a_hi - a_lo))
    mean_std_val = float(jnp.mean(jnp.std(a_pred[:, :, 0], axis=0)))

    flag, _ = mcmc_reliability_flag(ess_min, 1.0, n_div, n_samples)

    return {
        "rho_pde": rho_pde,
        "a_err": icorr,
        "u_err": u_err,
        "icorr": icorr,
        "coverage": coverage,
        "ess_min": ess_min,
        "n_div": n_div,
        "crps_a": crps_a,
        "nll_a": nll_a,
        "ci_width": ci_width_val,
        "mean_std": mean_std_val,
        "rhat_max": None,
        "rhat_mean": None,
        "reliability_flag": flag,
    }


# %%
print(f"Screening sweep: {SWEEP_WARMUP} warmup, {SWEEP_SAMPLES} samples per rho")
print(f"rho values: {RHO_PDE_VALUES}")

sweep_results = []
for rho in RHO_PDE_VALUES:
    print(f"\n{'='*60}")
    print(f"rho_pde = {rho}")
    print(f"{'='*60}")
    _rng_setup, key = random.split(_rng_setup)
    res = _run_sweep_one(rho, SWEEP_WARMUP, SWEEP_SAMPLES, key)
    sweep_results.append(res)
    print(f"  a_err={res['a_err']:.4f}  u_err={res['u_err']:.4f}  "
          f"coverage={res['coverage']:.2%}  ESS_min={res['ess_min']:.1f}  "
          f"n_div={res['n_div']}  CRPS_a={res['crps_a']:.6f}")

# %%
header = f"{'rho_pde':>10} {'a_err':>8} {'u_err':>8} {'coverage':>10} {'ESS_min':>9} {'n_div':>6} {'CRPS_a':>10}"
print(header)
print("-" * len(header))
print(f"{'inf (DO)':>10}  {_a_err_sweep_do:>7.4f}  {'N/A':>7}  {'N/A':>9}  {'N/A':>8}  {'N/A':>5}  {'N/A':>9}")
for r in sweep_results:
    print(f"{r['rho_pde']:>10.2f}  {r['a_err']:>7.4f}  {r['u_err']:>7.4f}  "
          f"{r['coverage']:>9.2%}  {r['ess_min']:>8.1f}  {r['n_div']:>5}  {r['crps_a']:>9.6f}")

# %%
# cross_correlation is higher=better, so threshold is inverted
_a_err_threshold = _a_err_sweep_do * 0.5

candidates = [(abs(r['coverage'] - 0.95), r) for r in sweep_results
              if r['a_err'] >= _a_err_threshold]

if candidates:
    candidates.sort(key=lambda x: x[0])
    _best = candidates[0][1]
    BEST_RHO_PDE = _best['rho_pde']
    print(f"Best rho_pde = {BEST_RHO_PDE}")
    print(f"  a_err    = {_best['a_err']:.4f}  (baseline: {_a_err_sweep_do:.4f})")
    print(f"  coverage = {_best['coverage']:.2%}  (ideal: 95%)")
else:
    print("No candidate with a_err >= 0.5x baseline.")
    print("Setting BEST_RHO_PDE to largest value (weakest physics).")
    BEST_RHO_PDE = max(RHO_PDE_VALUES)

print(f"\nBEST_RHO_PDE = {BEST_RHO_PDE}")

# %% [markdown]
# ## 6. Multi-Seed Loop

# %%
from results_schema import ExperimentResult
from datetime import datetime

for SEED in SEEDS:
    _t_total_start = time.time()
    print(f"\n{'='*70}")
    print(f"SEED = {SEED}")
    print(f"{'='*70}")

    rng = random.PRNGKey(SEED)
    rng, key = random.split(rng)

    FIGURE_DIR = Path(f'figures/physics_darcy_piecewise/test{TEST_IDX}/seed{SEED}')
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Observations ----
    obs_indices = problem.sample_observation_indices(n_points, N_OBS, 'random', key)
    obs_data = problem.prepare_observations(
        sample_indices=[TEST_IDX], obs_indices=obs_indices,
    )
    x_full = obs_data['x_full']
    x_obs = obs_data['x_obs']
    u_obs = obs_data['u_obs']
    a_true = obs_data['a_true']
    u_true = obs_data['u_true']
    a_true_np = np.array(a_true[0, :, 0])

    # ---- Prior Predictive ----
    def _prior_icorr(pred_mean, true):
        thresh = np.where(np.array(pred_mean) >= 7.5, 10.0, 5.0)
        return float(cross_correlation(jnp.array(thresh), jnp.array(true)))
    prior_a_samples, prior_metrics, rng = sample_unconditional_prior(
        problem, params, x_full, a_true_np, rng, error_fn=_prior_icorr,
    )
    print(f"Prior predictive: a_err={prior_metrics['a_err']:.4f}, "
          f"CRPS={prior_metrics['crps_a']:.4f}, cov95={prior_metrics['coverage_95']:.4f}")

    # ---- Build per-seed log_likelihood and models ----
    def log_likelihood_fn(beta, sigma):
        beta_b = beta[None, :]
        u_pred = problem.models['u'].apply({'params': params['u']}, x_obs, beta_b)
        if u_pred.ndim == 2:
            u_pred = u_pred[..., None]
        u_pred = mollifier(u_pred.squeeze(-1), x_obs)
        residual = u_pred - u_obs
        sq_err = jnp.sum(residual ** 2)
        n = u_obs.shape[1]
        return -0.5 * sq_err / (sigma ** 2) - 0.5 * n * jnp.log(2 * jnp.pi * sigma ** 2)

    # NF reparameterized data-only model
    nf_model_fn = make_nf_reparameterized_model(
        nf_model=problem.models['nf'], nf_params=params['nf'],
        log_likelihood_fn=log_likelihood_fn, d=d, nf_alpha=NF_ALPHA,
    )

    # Physics model (inline — z-space sampling with PDE factor)
    def numpyro_model_physics(sigma=0.1, rho_pde=1.0):
        z_01 = numpyro.sample("z", dist.Beta(NF_ALPHA, NF_ALPHA).expand([d]).to_event(1))
        z = 2.0 * z_01 - 1.0
        beta, _ = problem.models['nf'].apply(
            {'params': params['nf']}, z[None, :], method=problem.models['nf'].inverse
        )
        beta = beta[0]
        numpyro.factor("data_lik", log_likelihood_fn(beta, sigma))
        numpyro.factor("pde_lik", log_pde_fn(beta, rho_pde))

    # ---- MAP Baseline ----
    rng, inv_rng = random.split(rng)
    inverter = IGNOInverter(problem, inv_rng)
    _t_map = time.time()
    beta_map = inverter.invert(x_obs, u_obs, x_full, inv_config, verbose=True)
    _map_time_s = time.time() - _t_map
    print(f"MAP completed in {_map_time_s:.1f}s")

    preds_map = problem.predict_from_beta(params, beta_map, x_full)
    a_map = preds_map['a_pred'][0]
    u_map = preds_map['u_pred'][0]

    rmse_map_a = rmse(a_map, a_true[0])
    rmse_map_u = rmse(u_map, u_true[0])
    icorr_map = cross_correlation(a_map, a_true[0])
    print(f"MAP I_corr: {icorr_map:.6f}")

    from src.utils.PlotFigure import Plot
    h = inverter.loss_history
    Plot.show_loss(
        [h['total'], h['weighted_pde'], h['weighted_data']],
        ['Total', f'PDE (x{inv_config.loss_weights.pde})', f'Data (x{inv_config.loss_weights.data})'],
        save_path=str(FIGURE_DIR / 'map_loss_curves.png'),
    )

    # ---- Data-Only MCMC Full Run (NF reparameterized) ----
    print(f"Data-only MCMC (NF reparam): {NUM_WARMUP} warmup, {NUM_SAMPLES} samples, sigma={SIGMA}, {NUM_CHAINS} chains ({CHAIN_METHOD})")

    nuts_cfg = recommended_nuts_config(d, SIGMA)
    rng, mcmc_key = random.split(rng)
    mcmc_do, timing_do = run_mcmc(
        functools.partial(nf_model_fn, sigma=SIGMA), {"z": z_init}, {}, mcmc_key,
        NUM_WARMUP, NUM_SAMPLES, NUM_CHAINS, CHAIN_METHOD, nuts_cfg,
    )
    mcmc_do.print_summary(exclude_deterministic=True)

    diag_do = extract_mcmc_diagnostics(mcmc_do, sample_name=SAMPLE_NAME, total_samples=NUM_CHAINS * NUM_SAMPLES)

    beta_do = _z_to_beta(diag_do['samples'])
    a_pred_do, u_pred_do = _decode_piecewise_batched(beta_do, x_full)

    a_do_np = np.array(a_pred_do[:, :, 0])
    u_do_np = np.array(u_pred_do[:, :, 0])
    a_std_do = np.std(a_do_np, axis=0)

    pw_do = compute_piecewise_metrics(a_do_np, a_true_np, 5.0, 10.0, a_err_fn=cross_correlation)
    icorr_do = pw_do['a_err']
    a_err_do = float(icorr_do)
    u_err_do = float(jnp.linalg.norm(jnp.mean(u_pred_do, axis=0)[:, 0] - u_true[0, :, 0]) / jnp.linalg.norm(u_true[0, :, 0]))

    crps_do_a = float(np.mean(crps_ensemble(a_do_np, a_true_np)))
    nll_do_a = pw_do['nll_a']
    cal_do_levels, cal_do_empirical = compute_calibration(a_do_np, a_true_np)
    ci_w_do = ci_width_95(a_do_np)
    coverage_do = float(((a_true[0] >= jnp.percentile(a_pred_do, 2.5, axis=0)) &
                         (a_true[0] <= jnp.percentile(a_pred_do, 97.5, axis=0))).mean())

    print(f"Data-only: a_err={a_err_do:.4f}  u_err={u_err_do:.4f}  coverage={coverage_do:.2%}")
    print(f"  ESS_min={diag_do['ess_min']:.1f}  I_corr={float(icorr_do):.6f}")

    # ---- Physics-Informed Full Run ----
    print(f"Physics MCMC: rho_pde={BEST_RHO_PDE}, {NUM_WARMUP} warmup, {NUM_SAMPLES} samples, {NUM_CHAINS} chains")

    rng, mcmc_key_phys = random.split(rng)
    mcmc_phys, timing_phys = run_mcmc(
        numpyro_model_physics, {"z": z_init},
        {"sigma": SIGMA, "rho_pde": BEST_RHO_PDE}, mcmc_key_phys,
        NUM_WARMUP, NUM_SAMPLES, NUM_CHAINS, CHAIN_METHOD, nuts_cfg,
    )

    diag_phys = extract_mcmc_diagnostics(mcmc_phys, sample_name=SAMPLE_NAME, total_samples=NUM_CHAINS * NUM_SAMPLES)

    beta_phys = _z_to_beta(diag_phys['samples'])
    a_pred_phys, u_pred_phys = _decode_piecewise_batched(beta_phys, x_full)

    a_phys_np = np.array(a_pred_phys[:, :, 0])
    u_phys_np = np.array(u_pred_phys[:, :, 0])
    a_std_phys = np.std(a_phys_np, axis=0)

    pw_phys = compute_piecewise_metrics(a_phys_np, a_true_np, 5.0, 10.0, a_err_fn=cross_correlation)
    icorr_phys = pw_phys['a_err']
    a_err_phys = float(icorr_phys)
    u_err_phys = float(jnp.linalg.norm(jnp.mean(u_pred_phys, axis=0)[:, 0] - u_true[0, :, 0]) / jnp.linalg.norm(u_true[0, :, 0]))

    crps_phys_a = float(np.mean(crps_ensemble(a_phys_np, a_true_np)))
    cal_phys_levels, cal_phys_empirical = compute_calibration(a_phys_np, a_true_np)
    ci_w_phys = ci_width_95(a_phys_np)
    coverage_phys = float(((a_true[0] >= jnp.percentile(a_pred_phys, 2.5, axis=0)) &
                           (a_true[0] <= jnp.percentile(a_pred_phys, 97.5, axis=0))).mean())

    print(f"Physics: a_err={a_err_phys:.4f}  u_err={u_err_phys:.4f}  coverage={coverage_phys:.2%}")

    # ---- Chi2 PPC ----
    u_obs_np = np.array(u_obs[0, :, 0])
    u_pred_at_obs_do = u_do_np[:, np.array(obs_indices)]
    u_pred_at_obs_phys = u_phys_np[:, np.array(obs_indices)]
    chi2_do, pval_do = chi2_ppc(u_obs_np, u_pred_at_obs_do, SIGMA)
    chi2_phys, pval_phys = chi2_ppc(u_obs_np, u_pred_at_obs_phys, SIGMA)
    print(f"  Chi2 PPC (data-only): chi2={chi2_do:.2f}, p={pval_do:.4f}")
    print(f"  Chi2 PPC (physics):   chi2={chi2_phys:.2f}, p={pval_phys:.4f}")

    # ---- Spearman error-std correlation ----
    spearman_rho_do, spearman_p_do = compute_error_std_correlation(
        a_true_np, a_do_np.mean(axis=0), a_std_do)
    spearman_rho_phys, spearman_p_phys = compute_error_std_correlation(
        a_true_np, a_phys_np.mean(axis=0), a_std_phys)

    # ---- Save Results ----
    sharpness_do = float(np.mean(a_std_do))
    sharpness_phys = float(np.mean(a_std_phys))

    do_result = {
        "sigma": SIGMA, "label": "data_only",
        "ess_min": diag_do['ess_min'], "rhat_max": diag_do['rhat_max'],
        "rhat_mean": diag_do['rhat_mean'], "n_div": diag_do['n_div'],
        "reliability_flag": diag_do['flag'],
        "reliability_explanation": diag_do['flag_explanation'],
        "a_err": a_err_do, "u_err": u_err_do,
        "a_err_per_sample": pw_do['a_err_per_sample'],
        "crps_a": crps_do_a, "nll_a": nll_do_a,
        "coverage": coverage_do,
        "ci_width": ci_w_do, "mean_std": sharpness_do,
        "cal_levels": cal_do_levels,
        "cal_empirical": cal_do_empirical,
        "chi2_ppc": chi2_do, "chi2_ppc_pvalue": pval_do,
        "map_a_err": float(icorr_map), "map_u_err": float(rmse_map_u),
        "spearman_rho_error_std": float(spearman_rho_do),
        "spearman_pvalue_error_std": float(spearman_p_do),
        "warmup_time_s": timing_do['warmup_time_s'],
        "sampling_time_s": timing_do['sampling_time_s'],
        "step_time_s": timing_do['step_time_s'],
    }

    phys_result = {
        "sigma": SIGMA, "label": "physics",
        "ess_min": diag_phys['ess_min'], "rhat_max": diag_phys['rhat_max'],
        "rhat_mean": diag_phys['rhat_mean'], "n_div": diag_phys['n_div'],
        "reliability_flag": diag_phys['flag'],
        "reliability_explanation": diag_phys['flag_explanation'],
        "a_err": a_err_phys, "u_err": u_err_phys,
        "a_err_per_sample": pw_phys['a_err_per_sample'],
        "crps_a": crps_phys_a, "nll_a": pw_phys['nll_a'],
        "coverage": coverage_phys,
        "ci_width": ci_w_phys, "mean_std": sharpness_phys,
        "cal_levels": cal_phys_levels,
        "cal_empirical": cal_phys_empirical,
        "chi2_ppc": chi2_phys, "chi2_ppc_pvalue": pval_phys,
        "map_a_err": float(icorr_map), "map_u_err": float(rmse_map_u),
        "spearman_rho_error_std": float(spearman_rho_phys),
        "spearman_pvalue_error_std": float(spearman_p_phys),
        "warmup_time_s": timing_phys['warmup_time_s'],
        "sampling_time_s": timing_phys['sampling_time_s'],
        "step_time_s": timing_phys['step_time_s'],
    }

    experiment = ExperimentResult(
        experiment="physics",
        problem="darcy_piecewise",
        experiment_type="comparison",
        seed=SEED,
        test_idx=TEST_IDX,
        timestamp=datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        conditions={
            "data_only": build_mcmc_result(do_result, NUM_WARMUP, NUM_SAMPLES, NUM_CHAINS),
            "physics": build_mcmc_result(phys_result, NUM_WARMUP, NUM_SAMPLES, NUM_CHAINS),
        },
        prior=build_prior_result(prior_metrics),
        map_time_s=_map_time_s,
        total_time_s=time.time() - _t_total_start,
    )

    out_path = save_experiment_result(experiment)
    print(f"Saved structured result to: {out_path}")


    # ---- Plots ----
    x_np = np.array(x_full[0])

    plot_metrics_comparison_table(
        {
            'Rel. L2 (a)': a_err_do,
            'Rel. L2 (u)': u_err_do,
            'I_corr': float(icorr_do),
            'CRPS (a)': crps_do_a,
            '95% Coverage': coverage_do,
            'CI Width (a)': ci_w_do,
            'Sharpness (mean std)': sharpness_do,
            'ESS min': diag_do['ess_min'],
            'Divergences': diag_do['n_div'],
        },
        {
            'Rel. L2 (a)': a_err_phys,
            'Rel. L2 (u)': u_err_phys,
            'I_corr': float(icorr_phys),
            'CRPS (a)': crps_phys_a,
            '95% Coverage': coverage_phys,
            'CI Width (a)': ci_w_phys,
            'Sharpness (mean std)': sharpness_phys,
            'ESS min': diag_phys['ess_min'],
            'Divergences': diag_phys['n_div'],
        },
        title=f'Darcy Piecewise seed={SEED} (sigma={SIGMA}, rho_pde={BEST_RHO_PDE})',
    )

    # Significance Tests
    def _crps_a(s, t): return float(np.mean(crps_ensemble(s, t)))
    def _coverage_95(s, t):
        _, emp = compute_calibration(s, t, np.array([0.95]))
        return float(emp[0])
    def _ci_width(s, t): return ci_width_95(s)
    def _icorr(s, t):
        s_mean = np.where(np.mean(s, axis=0) >= 7.5, 10.0, 5.0)
        return float(cross_correlation(jnp.array(s_mean), jnp.array(t)))

    rng_bs = np.random.default_rng(2)
    diff_icorr = bootstrap_metric_difference_ci(a_do_np, a_phys_np, a_true_np, _icorr, rng=rng_bs)
    diff_cov = bootstrap_metric_difference_ci(a_do_np, a_phys_np, a_true_np, _coverage_95, rng=rng_bs)
    diff_width = bootstrap_metric_difference_ci(a_do_np, a_phys_np, a_true_np, _ci_width, rng=rng_bs)

    format_significance_table({
        'I_corr (a) diff (DO - Phys)': diff_icorr,
        'Coverage 95% diff (DO - Phys)': diff_cov,
        'CI Width diff (DO - Phys)': diff_width,
    }, title=f'Darcy Piecewise seed={SEED} -- RQ2: Physics vs Data-Only (Bootstrap)')

    # Metric Convergence
    conv_do = compute_metric_convergence(a_do_np, a_true_np)
    conv_phys = compute_metric_convergence(a_phys_np, a_true_np)
    plot_metric_convergence(
        [conv_do, conv_phys],
        labels=['Data-only', 'Physics-informed'],
        save_path=FIGURE_DIR / 'metric_convergence.png',
    )

    plot_std_comparison(
        x_np,
        std_data_only=a_std_do,
        std_physics=a_std_phys,
        grid_shape=(29, 29),
        save_path=FIGURE_DIR / 'std_comparison.png',
    )

    plot_rho_sweep(
        sweep_results, _sweep_data_only_baseline,
        a_metric_key='icorr', a_metric_label='I_corr',
        save_path=FIGURE_DIR / 'rho_sweep.png',
    )

    plot_sharpness_calibration_tradeoff(
        sweep_results, _sweep_data_only_baseline,
        save_path=FIGURE_DIR / 'sharpness_calibration_tradeoff.png',
    )

    with plt.style.context(_use_science_style()):
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Ideal')
        ax.plot(cal_do_levels, cal_do_empirical, 'o-', markersize=6, label='Data-only')
        ax.plot(cal_phys_levels, cal_phys_empirical, 's-', markersize=6, label='Physics-informed')
        ax.set_xlabel('Nominal Coverage', fontsize=14)
        ax.set_ylabel('Empirical Coverage', fontsize=14)
        ax.tick_params(labelsize=13)
        ax.set_xlim(0.4, 1.0)
        ax.set_ylim(0.4, 1.05)
        ax.legend(fontsize=9, framealpha=0.3, facecolor='white', edgecolor='black')
        plt.tight_layout()
        fig.savefig(FIGURE_DIR / 'calibration_overlay.png', dpi=200, bbox_inches='tight')
        plt.show()

    plot_posterior_predictive(
        u_obs_np, u_pred_at_obs_do,
        obs_label='u observed',
        save_path=FIGURE_DIR / 'posterior_predictive_data_only.png',
    )
    plot_posterior_predictive(
        u_obs_np, u_pred_at_obs_phys,
        obs_label='u observed',
        save_path=FIGURE_DIR / 'posterior_predictive_physics.png',
    )

    plot_posterior_gallery(
        x_np, a_phys_np, grid_shape=(29, 29),
        piecewise_a_bounds=(5.0, 10.0),
        a_true=a_true_np, n_show=6,
        save_path=FIGURE_DIR / 'posterior_gallery_physics.png',
    )

    a_map_np = np.array(a_map[:, 0])
    plot_field_comparison(
        x_np, a_true_np, a_map_np, a_phys_np.mean(axis=0), a_std_phys,
        grid_shape=(29, 29),
        u_true=np.array(u_true[0, :, 0]),
        u_map=np.array(u_map[:, 0]),
        u_mean=u_phys_np.mean(axis=0),
        u_std=np.std(u_phys_np, axis=0),
        obs_coords=np.array(x_obs[0]),
        save_path=FIGURE_DIR / 'field_comparison_physics.png',
        show_abs_error=False,
        piecewise_a_bounds=(5.0, 10.0),
    )

    # Diagnostics
    print(f"ESS summary ({d} dimensions, physics-informed, {NUM_CHAINS} chains):")
    print(f"  min ESS:  {diag_phys['ess_min']:.1f}")
    print(f"  max ESS:  {float(np.array(diag_phys['ess']).max()):.1f}")
    print(f"  mean ESS: {float(np.array(diag_phys['ess']).mean()):.1f}")
    print(f"  R-hat max:  {diag_phys['rhat_max']:.4f}")
    print(f"\n  RELIABILITY (physics): [{diag_phys['flag']}] {diag_phys['flag_explanation']}")
    print(f"  RELIABILITY (data-only): [{diag_do['flag']}] {diag_do['flag_explanation']}")

    beta_true = problem.models['enc'].apply({'params': params['enc']}, a_true)[0]

    z_all_chains_do = np.array(mcmc_do.get_samples(group_by_chain=True)["z"])
    plot_trace(z_all_chains_do[0][:, :8], beta_true=None, num_warmup=0,
               save_path=FIGURE_DIR / 'trace_plots_data_only.png')

    z_phys_by_chain = np.array(mcmc_phys.get_samples(group_by_chain=True)["z"])
    plot_trace(z_phys_by_chain[0][:, :8], beta_true=None,
               save_path=FIGURE_DIR / 'trace_plots_first8.png')

    spearman_rho_do, spearman_p_do = compute_error_std_correlation(
        a_true_np, a_do_np.mean(axis=0), a_std_do,
        save_path=FIGURE_DIR / 'error_vs_std_data_only.png',
    )
    print(f'Spearman rho (data-only): {spearman_rho_do:.3f}, p = {spearman_p_do:.2e}')

    spearman_rho_phys, spearman_p_phys = compute_error_std_correlation(
        a_true_np, a_phys_np.mean(axis=0), a_std_phys,
        save_path=FIGURE_DIR / 'error_vs_std_physics.png',
    )
    print(f'Spearman rho (physics): {spearman_rho_phys:.3f}, p = {spearman_p_phys:.2e}')

# %% [markdown]
# ## Cross-Seed Aggregation Summary

# %%
results = load_cross_seed_results("physics", "darcy_piecewise")
if len(results) > 1:
    print(f"Cross-Seed Summary ({len(results)} seeds: {[r.seed for r in results]})")
    for cond in list(results[0].conditions.keys()):
        print(f"\n--- {cond} ---")
        print(f"{'Metric':<16s}  {'Mean':>10s}  {'Std':>10s}  {'Min':>10s}  {'Max':>10s}")
        print("-" * 62)
        for m in ["a_err", "u_err", "crps_a", "nll_a", "coverage_95", "ci_width", "mean_std", "ess_min", "rhat_max", "n_div"]:
            try:
                s = cross_seed_metric_summary(results, metric=m, condition_key=cond)
                if s["mean"] is not None:
                    print(f"{m:<16s}  {s['mean']:>10.4f}  {s['std']:>10.4f}  {s['min']:>10.4f}  {s['max']:>10.4f}")
            except (AttributeError, KeyError, TypeError):
                pass
else:
    print(f"Only {len(results)} seed result(s) found - skipping cross-seed summary")
