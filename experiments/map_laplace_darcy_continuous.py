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
# # MAP and Laplace Approximation: Darcy Continuous
#
# - PDE: $-\nabla \cdot (a \nabla u) = 10$, continuous coefficient function
# - Latent dimension: $d = 6$

# %%
import sys, itertools
sys.path.insert(0, 'experiment_utils')
from _slurm import parse_slurm_task

PARAMETER_GRID = [
    {"seed": s, "test_idx": t}
    for s, t in itertools.product([42, 123, 7], [0, 1, 2])
]
_params, _task_id = parse_slurm_task(PARAMETER_GRID)

# %%
sys.path.insert(0, '..')
import load_this_before_everything_else

import jax
import jax.numpy as jnp
from jax import random
import numpy as np
import time
from pathlib import Path
from datetime import datetime

from src.problems.darcy_continuous import DarcyContinuous, mollifier
from src.evaluation.metrics import rmse
from src.evaluation.laplace import compute_hessian, sample_laplace
from src.solver.config import InversionConfig, LossWeights, OptimizerConfig, SchedulerConfig

from experiment_utils import (
    crps_ensemble, compute_calibration, ci_width_95, nll_score,
    compute_error_std_correlation,
    build_laplace_result, save_experiment_result,
    load_problem, run_map_estimation, make_igno_loss_fn,
    print_cross_seed_summary,
)
from results_schema import ExperimentResult

SEEDS = [42, 123, 7]
if _task_id is not None:
    SEEDS = [PARAMETER_GRID[_task_id]["seed"]]

print(f"JAX: {jax.__version__}")
print(f"Devices: {jax.devices()}")

# %% [markdown]
# ## 1. Load Trained Model

# %%
CHECKPOINT_PATH = Path("../runs/final_darcy_continuous/weights/best.pt")
TEST_DATA_PATH = "../data/darcy_continuous/smh_test_in.mat"

problem = DarcyContinuous(seed=42, test_data_path=TEST_DATA_PATH)
params = load_problem(problem, CHECKPOINT_PATH)

print(f"Latent dim: {problem.BETA_SIZE}")

# %% [markdown]
# ## 2. Prepare Observations

# %%
TEST_IDX = 0
if _task_id is not None:
    TEST_IDX = PARAMETER_GRID[_task_id]["test_idx"]
N_OBS = 100

n_points = problem.get_n_points()

# %% [markdown]
# ## 3. Inversion Config

# %%
inv_config = InversionConfig(
    epochs=1000,
    loss_weights=LossWeights(pde=1.0, data=50.0),
    optimizer=OptimizerConfig(type='Adam', lr=0.01),
    scheduler=SchedulerConfig(type='StepLR', step_size=125, gamma=0.8),
)

# %% [markdown]
# ## 4. Per-Seed Loop

# %%
NUM_SAMPLES = 2000
NUM_CHAINS = 4

for SEED in SEEDS:
    print(f"\n{'='*60}")
    print(f"SEED = {SEED}")
    print(f"{'='*60}")

    FIGURE_DIR = Path(f'figures/map_laplace_darcy_continuous/seed_{SEED}')
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    # ### Observations (this seed)

    rng = random.PRNGKey(SEED)
    rng, key = random.split(rng)

    obs_indices = problem.sample_observation_indices(n_points, N_OBS, 'random', key)

    obs_data = problem.prepare_observations(
        sample_indices=[TEST_IDX],
        obs_indices=obs_indices,
    )

    x_full = obs_data['x_full']
    x_obs = obs_data['x_obs']
    u_obs = obs_data['u_obs']
    a_true = obs_data['a_true']
    u_true = obs_data['u_true']

    print(f"x_obs: {x_obs.shape}, u_obs: {u_obs.shape}")
    print(f"a_true range: [{float(a_true.min()):.3f}, {float(a_true.max()):.3f}]")

    # ### MAP Baseline

    map_result = run_map_estimation(problem, params, x_obs, u_obs, x_full, inv_config, rng)
    beta_map = map_result['beta_map']
    a_map = map_result['a_map']
    u_map = map_result['u_map']
    rng = map_result['rng']

    rmse_map_a = rmse(a_map, a_true[0])
    rmse_map_u = rmse(u_map, u_true[0])
    print(f"\nMAP RMSE: a={rmse_map_a:.6f}, u={rmse_map_u:.6f}")

    from src.utils.PlotFigure import Plot
    h = map_result['loss_history']
    Plot.show_loss(
        [h['total'], h['weighted_pde'], h['weighted_data']],
        ['Total', f'PDE (x{inv_config.loss_weights.pde})', f'Data (x{inv_config.loss_weights.data})'],
        save_path=str(FIGURE_DIR / 'map_loss_curves.png'),
    )

    # ### Laplace Approximation (Hessian of IGNO objective at IGNO MAP)

    rng, hess_key = random.split(rng)
    igno_loss_fn = make_igno_loss_fn(problem, params, inv_config, x_obs, u_obs, hess_key)

    LA_N_SAMPLES = NUM_SAMPLES * NUM_CHAINS
    LA_REG_LAMBDA = 1e-4

    H, h_diag = compute_hessian(igno_loss_fn, beta_map[0], reg_lambda=LA_REG_LAMBDA)

    rng, la_key = random.split(rng)
    t0 = time.time()
    la_samples, frac_clip = sample_laplace(beta_map[0], H, LA_N_SAMPLES, la_key)
    sampling_time = time.time() - t0

    grad_fn = jax.grad(igno_loss_fn)
    grad_at_map = grad_fn(beta_map[0])
    grad_norm = float(jnp.linalg.norm(grad_at_map))

    print(f"  Laplace: frac_clipped={frac_clip:.4f}, "
          f"hessian_time={h_diag['hessian_time_s']:.1f}s")

    # Decode Laplace samples
    beta_la = la_samples
    x_la = jnp.tile(x_full, (LA_N_SAMPLES, 1, 1))

    la_a_pred = problem.models['a'].apply({'params': params['a']}, x_la, beta_la)
    la_a_pred = la_a_pred[..., None] if la_a_pred.ndim == 2 else la_a_pred

    la_u_pred = problem.models['u'].apply({'params': params['u']}, x_la, beta_la)
    if la_u_pred.ndim == 2:
        la_u_pred = la_u_pred[..., None]
    la_u_pred = mollifier(la_u_pred.squeeze(-1), x_la)

    la_a_true_np = np.array(a_true[0, :, 0])
    la_u_true_np = np.array(u_true[0, :, 0])
    la_a_samples_np = np.array(la_a_pred[:, :, 0])
    la_u_samples_np = np.array(la_u_pred[:, :, 0])
    la_a_mean_np = np.mean(la_a_samples_np, axis=0)
    la_a_std_np = np.std(la_a_samples_np, axis=0)
    la_u_mean_np = np.mean(la_u_samples_np, axis=0)

    la_rmse_a = rmse(jnp.array(la_a_mean_np), jnp.array(la_a_true_np))
    la_rmse_u = rmse(jnp.array(la_u_mean_np), jnp.array(la_u_true_np))
    la_crps_a = float(np.mean(crps_ensemble(la_a_samples_np, la_a_true_np)))
    la_nll_a = nll_score(la_a_samples_np, la_a_true_np)
    la_cal_levels, la_cal_empirical = compute_calibration(la_a_samples_np, la_a_true_np)
    la_ci_w = ci_width_95(la_a_samples_np)
    la_sharpness = float(np.mean(la_a_std_np))

    la_sp_rho, la_sp_p = compute_error_std_correlation(la_a_true_np, la_a_mean_np, la_a_std_np)

    print(f"  Laplace metrics: a_err={la_rmse_a:.6f}, u_err={la_rmse_u:.6f}, "
          f"crps_a={la_crps_a:.6f}, cov95={float(la_cal_empirical[-1]):.4f}, "
          f"ci_w={la_ci_w:.6f}, sharpness={la_sharpness:.6f}")

    # ### Save Structured Result

    la_run_result = {
        "n_samples": LA_N_SAMPLES, "map_max_iter": 1000,
        "hessian_reg_lambda": LA_REG_LAMBDA,
        "neg_log_posterior_at_map": float(igno_loss_fn(beta_map[0])),
        "grad_norm_at_map": grad_norm,
        "map_converged": True,
        "hessian_min_eigenvalue": h_diag['min_eigenvalue_raw'],
        "hessian_condition_number": h_diag['condition_number'],
        "n_negative_eigenvalues": h_diag['n_negative_eigenvalues'],
        "fraction_clipped": frac_clip,
        "a_err": la_rmse_a, "u_err": float(la_rmse_u),
        "crps_a": la_crps_a, "nll_a": la_nll_a,
        "coverage_95": float(la_cal_empirical[-1]),
        "ci_width": float(la_ci_w), "mean_std": la_sharpness,
        "cal_levels": la_cal_levels, "cal_empirical": la_cal_empirical,
        "map_a_err": float(rmse_map_a), "map_u_err": float(rmse_map_u),
        "spearman_rho_error_std": la_sp_rho, "spearman_pvalue_error_std": la_sp_p,
        "map_time_s": map_result['time_s'], "hessian_time_s": h_diag['hessian_time_s'],
        "sampling_time_s": sampling_time,
    }
    la_result_obj = build_laplace_result(la_run_result)

    experiment = ExperimentResult(
        experiment="map_laplace",
        problem="darcy_continuous",
        experiment_type="single",
        timestamp=datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        seed=SEED,
        test_idx=TEST_IDX,
        condition=None,
        prior=None,
        laplace=la_result_obj,
    )

    out_path = save_experiment_result(experiment)
    print(f"Saved structured result to: {out_path}")

# %% [markdown]
# ## Cross-Seed Aggregation Summary

# %%
print_cross_seed_summary("map_laplace", "darcy_continuous")
