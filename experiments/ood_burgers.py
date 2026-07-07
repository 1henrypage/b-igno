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
# # Out-of-Distribution Robustness: Burgers Equation
#
# - PDE: $\partial_t u + u \partial_x u = \frac{0.1}{\pi} \partial_{xx} u$
# - Unknown: initial condition $a(x) = u(x, t=0)$
# - Latent dimension: $d = 16$

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
from numpyro.diagnostics import effective_sample_size

from src.problems.burgers import (
    Burgers, mollifier_burgers, compute_pde_residual_single_sample_burgers
)
from src.evaluation.metrics import rmse
from src.solver.config import InversionConfig, LossWeights, OptimizerConfig, SchedulerConfig

from experiment_utils import (
    crps_ensemble, compute_calibration, ci_width_95, chi2_ppc, nll_score,
    plot_calibration, plot_trace, plot_metrics_table,
    plot_calibration_overlay, plot_metrics_comparison_table_4way,
    bootstrap_metric_difference_ci, format_significance_table,
    bootstrap_metric_ci, compute_per_chain_metrics,
    plot_burgers_field_comparison, plot_burgers_std_comparison,
    plot_burgers_posterior_gallery, plot_physics_benefit_comparison,
    recommended_nuts_config, tune_sigma,
    compute_prior_predictive, build_prior_result,
    compute_error_std_correlation,
    load_problem, get_nf_mode, make_log_prior,
    make_gaussian_log_likelihood, make_numpyro_model, make_numpyro_model_physics,
    run_map_estimation, decode_initial_condition_burgers,
    sample_unconditional_prior, decode_posterior_batched,
    run_mcmc, extract_mcmc_diagnostics,
    compute_standard_metrics,
    build_mcmc_result, save_experiment_result, print_cross_seed_summary,
)

# Paths
CHECKPOINT_PATH = Path('../runs/final_burgers/weights/best.pt')
IN_DATA_PATH = '../data/burgers/viscid_test_in.mat'
OOD_DATA_PATH = '../data/burgers/viscid_test_out.mat'

TEST_IDX = 0
if _task_id is not None:
    TEST_IDX = PARAMETER_GRID[_task_id]["test_idx"]
N_OBS = 100

# PDE collocation
NC_PDE = 50

# Sweep settings
RHO_PDE_VALUES = [2.0, 1.0, 0.5, 0.3, 0.15, 0.1, 0.07, 0.05]
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

SIGMA_CANDIDATES = [0.001, 0.002, 0.003, 0.005, 0.007, 0.01]

PROBLEM_NAME = 'ood_burgers'
FIGURE_DIR = Path(f'figures/{PROBLEM_NAME}/tuning')
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

print(f"JAX: {jax.__version__}, NumPyro: {numpyro.__version__}")
print(f"Devices: {jax.devices()}")

# %% [markdown]
# ## 1. Setup

# %%
problem_in = Burgers(seed=42, test_data_path=IN_DATA_PATH)
params = load_problem(problem_in, CHECKPOINT_PATH)
beta_mode, d = get_nf_mode(problem_in, params)
log_prior_fn = make_log_prior(problem_in, params)

problem_ood = Burgers(seed=42, test_data_path=OOD_DATA_PATH)
problem_ood.initialize_models(problem_in.get_sample_inputs(batch_size=1))
problem_ood.load_checkpoint(CHECKPOINT_PATH)

x_mesh = problem_in.x_mesh
t_mesh = problem_in.t_mesh
x_mesh_np = np.array(x_mesh)
n_points = problem_in.get_n_points()

print(f"Latent dim: {d}, n_mesh: {problem_in.n_mesh}, n_time: {problem_in.n_time}")

# %% [markdown]
# ## 2. PDE Collocation and Physics Closure

# %%
pde_rng = random.PRNGKey(123)
xc_fixed, tc_fixed, R_fixed = problem_in.genPoint.weight_centers(
    n_center=NC_PDE, R_max=1e-4, R_min=1e-4, key=pde_rng
)


def log_pde_fn(beta, rho_pde):
    residuals = compute_pde_residual_single_sample_burgers(
        params['u'], problem_in.models['u'], beta,
        xc_fixed, tc_fixed, R_fixed,
        problem_in.int_grid, problem_in.v, problem_in.dv_dr,
        problem_in.n_grid, problem_in.LAMDA,
    )
    return -0.5 * jnp.sum(residuals ** 2) / (rho_pde ** 2)

# %% [markdown]
# ## 3. Prepare Tuning Observations (Seed-42)

# %%
_rng_tune = random.PRNGKey(42)
_rng_tune, _key_tune = random.split(_rng_tune)
_obs_indices_tune = problem_in.sample_observation_indices(n_points, N_OBS, 'random', _key_tune)

obs_in_tune = problem_in.prepare_observations(sample_indices=[TEST_IDX], obs_indices=_obs_indices_tune)
obs_ood_tune = problem_ood.prepare_observations(sample_indices=[TEST_IDX], obs_indices=_obs_indices_tune)

print(f"In-domain  a_true range: [{float(obs_in_tune['a_true'].min()):.3f}, {float(obs_in_tune['a_true'].max()):.3f}]")
print(f"OOD        a_true range: [{float(obs_ood_tune['a_true'].min()):.3f}, {float(obs_ood_tune['a_true'].max()):.3f}]")

# %%
fig, axes = plt.subplots(1, 4, figsize=(16, 3.5))
X, T = np.meshgrid(x_mesh, t_mesh)

axes[0].plot(x_mesh_np, np.array(obs_in_tune['a_true'][0, :, 0]), 'C0', lw=1.5)
axes[0].set_title('In-Domain IC $a(x)$', fontsize=14); axes[0].set_xlabel('$x$')

axes[1].plot(x_mesh_np, np.array(obs_ood_tune['a_true'][0, :, 0]), 'C1', lw=1.5)
axes[1].set_title('OOD IC $a(x)$', fontsize=14); axes[1].set_xlabel('$x$')

u_in_2d = np.array(obs_in_tune['u_true'][0, :, 0]).reshape(problem_in.n_time, problem_in.n_mesh)
im2 = axes[2].pcolormesh(X, T, u_in_2d, cmap='jet', shading='auto')
axes[2].scatter(np.array(obs_in_tune['x_obs'][0, :, 0]), np.array(obs_in_tune['x_obs'][0, :, 1]),
                c='k', s=8, zorder=5, alpha=0.7)
axes[2].set_title('In-Domain $u(x,t)$ + sensors', fontsize=14)
axes[2].set_xlabel('$x$'); axes[2].set_ylabel('$t$')
fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

u_ood_2d = np.array(obs_ood_tune['u_true'][0, :, 0]).reshape(problem_in.n_time, problem_in.n_mesh)
im3 = axes[3].pcolormesh(X, T, u_ood_2d, cmap='jet', shading='auto')
axes[3].scatter(np.array(obs_ood_tune['x_obs'][0, :, 0]), np.array(obs_ood_tune['x_obs'][0, :, 1]),
                c='k', s=8, zorder=5, alpha=0.7)
axes[3].set_title('OOD $u(x,t)$ + sensors', fontsize=14)
axes[3].set_xlabel('$x$'); axes[3].set_ylabel('$t$')
fig.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)

plt.tight_layout()
fig.savefig(FIGURE_DIR / 'test_cases.png', dpi=200, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## 4. Inversion Config and Sigma (Tuning)

# %%
inv_config = InversionConfig(
    epochs=500,
    loss_weights=LossWeights(pde=1.0, data=50.0),
    optimizer=OptimizerConfig(type='Adam', lr=0.01),
    scheduler=SchedulerConfig(type='StepLR', step_size=100, gamma=0.8),
)

_log_lik_ref = make_gaussian_log_likelihood(
    problem_in, params, mollifier_burgers,
    obs_in_tune['x_obs'], obs_in_tune['u_obs'],
)
_model_ref = make_numpyro_model(d, log_prior_fn, _log_lik_ref)

def _model_factory_ref(sigma):
    return lambda: _model_ref(sigma=sigma)

def _decode_fn_burgers(beta_samples):
    return decode_initial_condition_burgers(problem_in, params, mollifier_burgers, beta_samples)

_nuts_cfg_ref = recommended_nuts_config(d, sigma=0.005)
_rng_tune, _tune_key_ref = random.split(_rng_tune)
SIGMA, _ = tune_sigma(
    model_fn_factory=_model_factory_ref,
    beta_mode=beta_mode,
    sigma_candidates=SIGMA_CANDIDATES,
    rng_key=_tune_key_ref,
    decode_fn=_decode_fn_burgers,
    a_true=np.array(obs_in_tune['a_true'][0, :, 0]),
    target_accept_prob=_nuts_cfg_ref['target_accept_prob'],
)
print(f"Sigma (tuned, seed-42 in-domain reference): {SIGMA:.6f}")

print(f"NF mode log_prior: {float(log_prior_fn(beta_mode)):.2f}")

# %% [markdown]
# ## 5. Rho_pde Tuning (In-Domain Only)

# %%
log_lik_in_tune = make_gaussian_log_likelihood(
    problem_in, params, mollifier_burgers,
    obs_in_tune['x_obs'], obs_in_tune['u_obs'],
)

# Decode helper for rho sweep
def _decode_sweep(beta_s, x_full_ref, a_true_ref):
    a_samples, _ = decode_posterior_batched(problem_in, params, beta_s, x_full_ref, batch_size=200)
    a_true_np = np.array(a_true_ref[0, :, 0])
    return compute_standard_metrics(a_samples, a_true_np)

model_physics_tune = make_numpyro_model_physics(d, log_prior_fn, log_lik_in_tune, log_pde_fn)

print(f"Rho sweep on IN-DOMAIN condition: {SWEEP_WARMUP} warmup, {SWEEP_SAMPLES} samples")
sweep_results = []
for rho in RHO_PDE_VALUES:
    print(f"\n{'='*50}\nrho_pde = {rho}")
    kernel = NUTS(
        model_physics_tune,
        init_strategy=init_to_value(values={"beta": beta_mode}),
        target_accept_prob=0.8,
    )
    _rng_tune, key = random.split(_rng_tune)
    mcmc_sw = MCMC(kernel, num_warmup=SWEEP_WARMUP, num_samples=SWEEP_SAMPLES,
                   num_chains=1, progress_bar=True)
    _t_pilot = time.time()
    mcmc_sw.run(key, sigma=SIGMA, rho_pde=rho, extra_fields=('diverging',))
    print(f"  Pilot rho={rho} completed in {time.time() - _t_pilot:.1f}s")

    beta_s = mcmc_sw.get_samples()["beta"]
    n_div = int(mcmc_sw.get_extra_fields()['diverging'].sum())
    dec = _decode_sweep(beta_s, obs_in_tune['x_full'], obs_in_tune['a_true'])
    ess = effective_sample_size(np.array(beta_s)[None, :, :])
    sweep_results.append({'rho_pde': rho, 'n_div': n_div, 'ess_min': float(ess.min()),
                          'coverage': dec['coverage_95'], 'a_err': dec['a_err']})
    print(f"  a_err={dec['a_err']:.4f}  coverage={dec['coverage_95']:.2%}  "
          f"ess_min={float(ess.min()):.1f}  n_div={n_div}")

# %%
print("Rho selection -- pick closest 95% coverage with reasonable accuracy:")
for r in sweep_results:
    print(f"  rho={r['rho_pde']:.2f}  a_err={r['a_err']:.4f}  coverage={r['coverage']:.2%}  div={r['n_div']}")

BEST_RHO_PDE = min(sweep_results, key=lambda r: abs(r['coverage'] - 0.95))['rho_pde']
print(f"\nSelected BEST_RHO_PDE = {BEST_RHO_PDE}")


# %% [markdown]
# ## 6. Run Condition Helper

# %%
def run_condition(model_fn, model_kwargs, x_full_ref, a_true_ref, obs_indices,
                  u_obs_np, label, seed, problem_ref):
    """Run MCMC for one condition, decode, and compute metrics."""
    print(f"\n{'='*60}\n  {label}\n{'='*60}")

    nuts_cfg = recommended_nuts_config(d, model_kwargs['sigma'])
    print(f"  NUTS config: {nuts_cfg}")

    rng_local = random.PRNGKey(seed)
    mcmc, timing = run_mcmc(
        model_fn, {"beta": beta_mode}, model_kwargs, rng_local,
        NUM_WARMUP, NUM_SAMPLES, NUM_CHAINS, CHAIN_METHOD, nuts_cfg,
    )
    mcmc.print_summary(exclude_deterministic=True)

    diag = extract_mcmc_diagnostics(mcmc, sample_name="beta",
                                    total_samples=NUM_CHAINS * NUM_SAMPLES)

    a_samples, u_samples = decode_posterior_batched(
        problem_ref, params, diag['samples'], x_full_ref, batch_size=200,
    )

    a_true_np = np.array(a_true_ref[0, :, 0])
    metrics = compute_standard_metrics(a_samples, a_true_np)

    a_std = np.std(a_samples, axis=0)
    a_mean = np.mean(a_samples, axis=0)
    spearman_rho, spearman_p = compute_error_std_correlation(a_true_np, a_mean, a_std)

    u_pred_at_obs = u_samples[:, obs_indices]
    chi2_stat, chi2_pval = chi2_ppc(u_obs_np, u_pred_at_obs, model_kwargs['sigma'])
    print(f"  {label}: chi2={chi2_stat:.2f}, p={chi2_pval:.4f}")

    return {
        'label': label,
        'a_samples': a_samples,
        'u_samples': u_samples,
        'a_mean': a_mean,
        'a_std': a_std,
        'u_mean': np.mean(u_samples, axis=0),
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
        'beta_by_chain': diag['by_chain'],
        'beta_for_trace': diag['by_chain'][0],
        'ess': diag['ess'],
        'rhat': diag['rhat'],
    }


# %% [markdown]
# ## 7. Multi-Seed Full MCMC Runs

# %%
from results_schema import ExperimentResult
from datetime import datetime

for SEED in SEEDS:
    print(f"\n{'='*60}")
    print(f"SEED = {SEED}")
    print(f"{'='*60}")
    _t_total_start = time.time()

    FIGURE_DIR = Path(f'figures/ood_burgers/seed_{SEED}')
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    # Fresh observations per seed (shared indices for in-domain and OOD)
    _rng_seed = random.PRNGKey(SEED)
    _rng_seed, _key_obs = random.split(_rng_seed)
    _obs_indices_seed = problem_in.sample_observation_indices(n_points, N_OBS, 'random', _key_obs)

    obs_in = problem_in.prepare_observations(sample_indices=[TEST_IDX], obs_indices=_obs_indices_seed)
    obs_ood = problem_ood.prepare_observations(sample_indices=[TEST_IDX], obs_indices=_obs_indices_seed)

    x_full_in, x_obs_in, u_obs_in = obs_in['x_full'], obs_in['x_obs'], obs_in['u_obs']
    a_true_in, u_true_in = obs_in['a_true'], obs_in['u_true']

    x_full_ood, x_obs_ood, u_obs_ood = obs_ood['x_full'], obs_ood['x_obs'], obs_ood['u_obs']
    a_true_ood, u_true_ood = obs_ood['a_true'], obs_ood['u_true']

    # === MAP Baselines ===
    _rng_seed, key1, key2 = random.split(_rng_seed, 3)

    # In-domain MAP
    _rng_map_in = key1
    from src.evaluation.igno import IGNOInverter
    inverter_in = IGNOInverter(problem_in, _rng_map_in)
    _t_map_in = time.time()
    beta_map_in = inverter_in.invert(x_obs_in, u_obs_in, x_full_in, inv_config, verbose=False)
    _map_in_time_s = time.time() - _t_map_in
    print(f"MAP (in-domain) completed in {_map_in_time_s:.1f}s")
    preds_in = problem_in.predict_from_beta(params, beta_map_in, x_full_in)
    a_map_in = preds_in['a_pred'][0]
    u_map_in = np.array(preds_in['u_pred'][0, :, 0])

    _log_lik_in_seed = make_gaussian_log_likelihood(problem_in, params, mollifier_burgers, x_obs_in, u_obs_in)
    _model_in_seed = make_numpyro_model(d, log_prior_fn, _log_lik_in_seed)

    def _model_factory_seed(sigma):
        return lambda: _model_in_seed(sigma=sigma)

    _rng_seed, _tune_key_seed = random.split(_rng_seed)
    SIGMA, _ = tune_sigma(
        model_fn_factory=_model_factory_seed,
        beta_mode=beta_mode,
        sigma_candidates=SIGMA_CANDIDATES,
        rng_key=_tune_key_seed,
        decode_fn=_decode_fn_burgers,
        a_true=np.array(a_true_in[0, :, 0]),
        target_accept_prob=recommended_nuts_config(d, sigma=0.005)['target_accept_prob'],
    )
    print(f"Sigma (tuned, in-domain, seed={SEED}): {SIGMA:.6f}")

    # OOD MAP
    inverter_ood = IGNOInverter(problem_ood, key2)
    _t_map_ood = time.time()
    beta_map_ood = inverter_ood.invert(x_obs_ood, u_obs_ood, x_full_ood, inv_config, verbose=False)
    _map_ood_time_s = time.time() - _t_map_ood
    print(f"MAP (OOD) completed in {_map_ood_time_s:.1f}s")
    preds_ood = problem_ood.predict_from_beta(params, beta_map_ood, x_full_ood)
    a_map_ood = preds_ood['a_pred'][0]
    u_map_ood = np.array(preds_ood['u_pred'][0, :, 0])

    rmse_map_in = float(rmse(a_map_in, a_true_in[0]))
    rmse_map_ood = float(rmse(a_map_ood, a_true_ood[0]))
    print(f"MAP RMSE -- In-Domain: {rmse_map_in:.6f}")
    print(f"MAP RMSE -- OOD:       {rmse_map_ood:.6f}")

    # === Prior Predictive Baseline ===
    a_true_in_np = np.array(a_true_in[0, :, 0])
    a_true_ood_np = np.array(a_true_ood[0, :, 0])

    prior_a_samples, prior_metrics_in, _rng_seed = sample_unconditional_prior(
        problem_in, params, x_full_in, a_true_in_np, _rng_seed,
    )
    prior_metrics_ood = compute_prior_predictive(prior_a_samples, a_true_ood_np)
    print(f"Prior (in-domain): a_err={prior_metrics_in['a_err']:.4f}, CRPS={prior_metrics_in['crps_a']:.4f}")
    print(f"Prior (OOD):       a_err={prior_metrics_ood['a_err']:.4f}, CRPS={prior_metrics_ood['crps_a']:.4f}")

    # === Build per-seed likelihood closures and models ===
    log_lik_in = make_gaussian_log_likelihood(problem_in, params, mollifier_burgers, x_obs_in, u_obs_in)
    log_lik_ood = make_gaussian_log_likelihood(problem_in, params, mollifier_burgers, x_obs_ood, u_obs_ood)

    model_in_do = make_numpyro_model(d, log_prior_fn, log_lik_in)
    model_in_phys = make_numpyro_model_physics(d, log_prior_fn, log_lik_in, log_pde_fn)
    model_ood_do = make_numpyro_model(d, log_prior_fn, log_lik_ood)
    model_ood_phys = make_numpyro_model_physics(d, log_prior_fn, log_lik_ood, log_pde_fn)

    _u_obs_in_np = np.array(u_obs_in[0, :, 0])
    _u_obs_ood_np = np.array(u_obs_ood[0, :, 0])

    # === 4 Conditions ===
    res_in_do = run_condition(
        model_in_do, {"sigma": SIGMA},
        x_full_in, a_true_in, _obs_indices_seed, _u_obs_in_np,
        "In-Domain Data-Only", seed=SEED+0, problem_ref=problem_in,
    )
    res_in_phys = run_condition(
        model_in_phys, {"sigma": SIGMA, "rho_pde": BEST_RHO_PDE},
        x_full_in, a_true_in, _obs_indices_seed, _u_obs_in_np,
        "In-Domain Physics", seed=SEED+1, problem_ref=problem_in,
    )
    res_ood_do = run_condition(
        model_ood_do, {"sigma": SIGMA},
        x_full_ood, a_true_ood, _obs_indices_seed, _u_obs_ood_np,
        "OOD Data-Only", seed=SEED+2, problem_ref=problem_in,
    )
    res_ood_phys = run_condition(
        model_ood_phys, {"sigma": SIGMA, "rho_pde": BEST_RHO_PDE},
        x_full_ood, a_true_ood, _obs_indices_seed, _u_obs_ood_np,
        "OOD Physics", seed=SEED+3, problem_ref=problem_in,
    )

    # ## 8. Metrics Comparison

    def make_metrics_dict(res):
        return {
            'Rel. L2 (a)': res['a_err'],
            'CRPS (a)': res['crps_a'],
            'NLL (a)': res['nll_a'],
            '95% Coverage': res['coverage_95'],
            'CI Width (a)': res['ci_width'],
            'Sharpness': res['mean_std'],
            'ESS min': res['ess_min'],
            'R-hat max': res['rhat_max'],
            'Divergences': res['n_div'],
        }

    plot_metrics_comparison_table_4way(
        make_metrics_dict(res_in_do),
        make_metrics_dict(res_in_phys),
        make_metrics_dict(res_ood_do),
        make_metrics_dict(res_ood_phys),
        col_labels=['ID DO', 'ID Phys', 'OOD DO', 'OOD Phys'],
        title=f'Burgers OOD (sigma={SIGMA}, rho_pde={BEST_RHO_PDE})',
    )

    # ## 8b. Significance of Pairwise Contrasts

    def _crps_a(s, t): return float(np.mean(crps_ensemble(s, t)))
    def _coverage_95(s, t):
        _, emp = compute_calibration(s, t, np.array([0.95]))
        return float(emp[0])
    def _ci_width(s, t): return ci_width_95(s)

    rng_bs = np.random.default_rng(3)

    diff_crps_in = bootstrap_metric_difference_ci(res_in_do['a_samples'], res_in_phys['a_samples'], a_true_in_np, _crps_a, rng=rng_bs)
    diff_cov_in = bootstrap_metric_difference_ci(res_in_do['a_samples'], res_in_phys['a_samples'], a_true_in_np, _coverage_95, rng=rng_bs)
    diff_crps_ood = bootstrap_metric_difference_ci(res_ood_do['a_samples'], res_ood_phys['a_samples'], a_true_ood_np, _crps_a, rng=rng_bs)
    diff_cov_ood = bootstrap_metric_difference_ci(res_ood_do['a_samples'], res_ood_phys['a_samples'], a_true_ood_np, _coverage_95, rng=rng_bs)
    diff_width_ood_vs_in = bootstrap_metric_difference_ci(res_in_do['a_samples'], res_ood_do['a_samples'], a_true_in_np, _ci_width, rng=rng_bs)

    format_significance_table({
        'CRPS: ID (DO - Phys)':        diff_crps_in,
        'Coverage: ID (DO - Phys)':    diff_cov_in,
        'CRPS: OOD (DO - Phys)':       diff_crps_ood,
        'Coverage: OOD (DO - Phys)':   diff_cov_ood,
        'CI Width: OOD DO - ID DO':    diff_width_ood_vs_in,
    }, title='Burgers -- OOD: Pairwise Contrasts (Bootstrap)')
    print("Negative CRPS diff = physics helps; positive coverage diff = physics improves calibration.")

    # -- Physics Benefit Delta Analysis --
    _rng_pb = np.random.default_rng(SEED + 5000)

    delta_in_crps = bootstrap_metric_difference_ci(res_in_phys['a_samples'], res_in_do['a_samples'], a_true_in_np, _crps_a, rng=_rng_pb)
    delta_ood_crps = bootstrap_metric_difference_ci(res_ood_phys['a_samples'], res_ood_do['a_samples'], a_true_ood_np, _crps_a, rng=_rng_pb)

    ci_in_do_pb = bootstrap_metric_ci(res_in_do['a_samples'], a_true_in_np, _crps_a, rng=_rng_pb)
    ci_in_phys_pb = bootstrap_metric_ci(res_in_phys['a_samples'], a_true_in_np, _crps_a, rng=_rng_pb)
    ci_ood_do_pb = bootstrap_metric_ci(res_ood_do['a_samples'], a_true_ood_np, _crps_a, rng=_rng_pb)
    ci_ood_phys_pb = bootstrap_metric_ci(res_ood_phys['a_samples'], a_true_ood_np, _crps_a, rng=_rng_pb)

    # DoD bootstrap
    _n_dod = 1000
    _dod_samples = np.empty(_n_dod)
    _rng_dod = np.random.default_rng(SEED + 6000)
    _na_in_do, _na_in_p = res_in_do['a_samples'].shape[0], res_in_phys['a_samples'].shape[0]
    _na_ood_do, _na_ood_p = res_ood_do['a_samples'].shape[0], res_ood_phys['a_samples'].shape[0]
    for _i in range(_n_dod):
        _b_in_do = res_in_do['a_samples'][_rng_dod.integers(0, _na_in_do, _na_in_do)]
        _b_in_p = res_in_phys['a_samples'][_rng_dod.integers(0, _na_in_p, _na_in_p)]
        _b_ood_do = res_ood_do['a_samples'][_rng_dod.integers(0, _na_ood_do, _na_ood_do)]
        _b_ood_p = res_ood_phys['a_samples'][_rng_dod.integers(0, _na_ood_p, _na_ood_p)]
        _d_in = _crps_a(_b_in_p, a_true_in_np) - _crps_a(_b_in_do, a_true_in_np)
        _d_ood = _crps_a(_b_ood_p, a_true_ood_np) - _crps_a(_b_ood_do, a_true_ood_np)
        _dod_samples[_i] = _d_ood - _d_in
    dod_crps = {
        'mean_diff': float(np.mean(_dod_samples)),
        'ci_lo': float(np.percentile(_dod_samples, 2.5)),
        'ci_hi': float(np.percentile(_dod_samples, 97.5)),
    }
    dod_crps['significant'] = not (dod_crps['ci_lo'] <= 0 <= dod_crps['ci_hi'])

    format_significance_table({
        'Delta CRPS In-Domain (phys - do)':        delta_in_crps,
        'Delta CRPS OOD (phys - do)':              delta_ood_crps,
        'Delta CRPS DoD (ood_delta - id_delta)':   dod_crps,
    }, title=f'Burgers -- Physics Benefit Analysis (seed={SEED})')
    print("Negative Delta CRPS = physics improves. Positive DoD = OOD benefits more from physics.")

    _id_pb = [{'data_only': {'crps_a': ci_in_do_pb['estimate'],
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
        metric_label='Delta CRPS (physics - data-only)',
        condition_labels=('in-domain', 'OOD'),
        save_path=FIGURE_DIR / 'physics_benefit_comparison.png',
    )

    # ## Save Structured Result

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
        problem="burgers",
        experiment_type="comparison",
        timestamp=datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        seed=SEED,
        test_idx=TEST_IDX,
        conditions={k: _build_condition(k, v) for k, v in conditions_raw.items()},
        prior=build_prior_result(prior_metrics_in),
        prior_ood=build_prior_result(prior_metrics_ood),
        map_time_s=_map_in_time_s + _map_ood_time_s,
        total_time_s=time.time() - _t_total_start,
    )

    out_path = save_experiment_result(experiment)
    print(f"Saved structured result to: {out_path}")


    # ## 9. Plots

    plot_burgers_std_comparison(
        x_mesh_np,
        std_a=np.array(res_ood_do['a_std']),
        std_b=np.array(res_ood_phys['a_std']),
        label_a='OOD Data-Only',
        label_b='OOD Physics',
        save_path=FIGURE_DIR / 'std_comparison_ood.png',
    )

    plot_burgers_std_comparison(
        x_mesh_np,
        std_a=np.array(res_in_do['a_std']),
        std_b=np.array(res_ood_do['a_std']),
        label_a='In-Domain Data-Only',
        label_b='OOD Data-Only',
        save_path=FIGURE_DIR / 'std_comparison_in_vs_ood.png',
    )

    plot_calibration_overlay(
        [
            (res_in_do['cal_levels'],   res_in_do['cal_empirical'],   'ID DO'),
            (res_in_phys['cal_levels'], res_in_phys['cal_empirical'], 'ID Phys'),
            (res_ood_do['cal_levels'],  res_ood_do['cal_empirical'],  'OOD DO'),
            (res_ood_phys['cal_levels'],res_ood_phys['cal_empirical'],'OOD Phys'),
        ],
        save_path=FIGURE_DIR / 'calibration_overlay.png',
    )

    for res, label, fname, a_map_ref, u_map_ref, a_true_ref_np, u_true_ref_np, x_obs_ref in [
        (res_in_do, 'ID Data-Only', 'field_comparison_in_do.png',
         a_map_in, u_map_in, np.array(a_true_in[0, :, 0]), np.array(u_true_in[0, :, 0]), np.array(x_obs_in[0])),
        (res_in_phys, 'ID Physics-Informed', 'field_comparison_in_phys.png',
         a_map_in, u_map_in, np.array(a_true_in[0, :, 0]), np.array(u_true_in[0, :, 0]), np.array(x_obs_in[0])),
        (res_ood_do, 'OOD Data-Only', 'field_comparison_ood_do.png',
         a_map_ood, u_map_ood, np.array(a_true_ood[0, :, 0]), np.array(u_true_ood[0, :, 0]), np.array(x_obs_ood[0])),
        (res_ood_phys, 'OOD Physics-Informed', 'field_comparison_ood_phys.png',
         a_map_ood, u_map_ood, np.array(a_true_ood[0, :, 0]), np.array(u_true_ood[0, :, 0]), np.array(x_obs_ood[0])),
    ]:
        u_std_np = np.std(res['u_samples'], axis=0)
        plot_burgers_field_comparison(
            x_mesh=x_mesh_np,
            t_mesh=t_mesh,
            a_true=a_true_ref_np,
            a_map=np.array(a_map_ref[:, 0]),
            a_mean=res['a_mean'],
            a_std=res['a_std'],
            u_true=u_true_ref_np,
            u_map=u_map_ref,
            u_mean=res['u_mean'],
            u_std=u_std_np,
            obs_coords=x_obs_ref,
            save_path=FIGURE_DIR / fname,
        )

    # ## 10. Diagnostics

    beta_true_in = np.array(problem_in.models['enc'].apply({'params': params['enc']}, a_true_in)[0])
    beta_true_ood = np.array(problem_in.models['enc'].apply({'params': params['enc']}, a_true_ood)[0])

    for res, bt_np, tag in [
        (res_in_do,   beta_true_in,  'in_do'),
        (res_in_phys, beta_true_in,  'in_phys'),
        (res_ood_do,  beta_true_ood, 'ood_do'),
        (res_ood_phys,beta_true_ood, 'ood_phys'),
    ]:
        print(f"\n{res['label']}:")
        print(f"  ESS min={res['ess_min']:.1f}  R-hat max={res['rhat_max']:.4f}  divs={res['n_div']}")
        print(f"  a_err={res['a_err']:.4f}  coverage={res['coverage_95']:.2%}")

        plot_trace(
            res['beta_for_trace'], bt_np, num_warmup=0,
            save_path=FIGURE_DIR / f'trace_{tag}.png',
        )

# %% [markdown]
# ## Cross-Seed Aggregation Summary

# %%
print_cross_seed_summary("ood", "burgers")
