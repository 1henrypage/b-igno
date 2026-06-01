#!/usr/bin/env python3
"""
Evaluation script for inverse problems in JAX.

Runs IGNO-style gradient inversion on test set and computes metrics.
All models are loaded from pretrained checkpoint into the ProblemInstance.

Usage:
    python evaluate.py --config configs/evaluation/example_evaluate.yaml
    python evaluate.py --config configs/evaluation/example_evaluate.yaml --n-obs 25
    python evaluate.py --config configs/evaluation/example_evaluate.yaml --snr-db 50
    python evaluate.py --config configs/evaluation/example_evaluate.yaml --batch-size 64
"""
import load_this_before_everything_else
import argparse
import json
import sys
import os
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

import jax
import jax.numpy as jnp
from jax import random
from tqdm import trange

from src.solver.config import TrainingConfig
from src.problems import create_problem
from src.evaluation import IGNOInverter, compute_all_metrics, aggregate_metrics


def save_results(results: dict, config: TrainingConfig, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    snr_str = f"snr{config.evaluation.snr_db}" if config.evaluation.snr_db else "clean"
    filename = f"eval_{config.evaluation.method}_nobs{config.evaluation.n_obs}_{snr_str}_{timestamp}.json"

    output_path = output_dir / filename

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")
    return output_path


def print_results(results: dict):
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    for field in ['u', 'a']:
        if f'{field}_metrics' in results:
            print(f"\n{field.upper()} Metrics:")
            metrics = results[f'{field}_metrics']
            for metric_name, values in metrics.items():
                print(f"  {metric_name}:")
                print(f"    mean: {values['mean']:.6f}")
                print(f"    std:  {values['std']:.6f}")
                print(f"    min:  {values['min']:.6f}")
                print(f"    max:  {values['max']:.6f}")


def evaluate(config: TrainingConfig, verbose: bool = True):
    """Run evaluation on test set."""
    eval_cfg = config.evaluation

    ckpt_path = config.get_pretrained_path()
    if ckpt_path is None:
        raise ValueError("pretrained.path must be set in config for evaluation")

    # load_train_data=False: test-only eval needs no training set
    print(f"Creating problem: {config.problem.type}")
    problem = create_problem(config, load_train_data=False)

    # Flax params must be initialized before checkpoint overwrite
    print("Initializing model architectures...")
    sample_inputs = problem.get_sample_inputs(batch_size=1)
    problem.initialize_models(sample_inputs)

    print(f"\nLoading checkpoint: {ckpt_path}")
    problem.load_checkpoint(ckpt_path)

    rng = random.PRNGKey(config.seed)

    if eval_cfg.method == 'igno':
        print("\nUsing IGNO gradient-based inversion")
        rng, inverter_rng = random.split(rng)
        inverter = IGNOInverter(problem, inverter_rng)
    elif eval_cfg.method == 'mcmc':
        raise NotImplementedError(
            "MCMC-based inversion not yet implemented. "
            "This will be added for Bayesian uncertainty quantification."
        )
    else:
        raise ValueError(f"Unknown evaluation method: {eval_cfg.method}")

    # Fixed observation indices shared across all test samples
    n_points = problem.get_n_points()
    n_test = problem.get_n_test_samples()

    rng, obs_rng = random.split(rng)
    obs_indices = problem.sample_observation_indices(
        n_total=n_points,
        n_obs=eval_cfg.n_obs,
        method=eval_cfg.obs_sampling,
        rng=obs_rng,
    )

    print(f"Observation indices shape: {obs_indices.shape}")

    # Clamp batch_size to n_test
    batch_size = eval_cfg.batch_size
    if batch_size is None:
        batch_size = n_test
    batch_size = min(batch_size, n_test)

    print(f"\nEvaluation setup:")
    print(f"  Test samples: {n_test}")
    print(f"  Batch size: {batch_size}")
    print(f"  Grid points: {n_points}")
    print(f"  Observations: {eval_cfg.n_obs} ({eval_cfg.obs_sampling} sampling)")
    print(f"  SNR: {eval_cfg.snr_db} dB" if eval_cfg.snr_db else "  SNR: Clean (no noise)")
    print(f"  Inversion epochs: {eval_cfg.inversion.epochs}")
    print(f"  Loss weights: PDE={eval_cfg.inversion.loss_weights.pde}, Data={eval_cfg.inversion.loss_weights.data}")

    all_metrics_u = []
    all_metrics_a = []
    per_sample_results = []

    n_batches = (n_test + batch_size - 1) // batch_size
    print(f"\nRunning inversion on {n_test} samples in {n_batches} batches...")

    for batch_start in trange(0, n_test, batch_size, desc="Batches", disable=not verbose):
        batch_end = min(batch_start + batch_size, n_test)
        batch_indices = list(range(batch_start, batch_end))
        current_batch_size = len(batch_indices)

        rng, noise_rng = random.split(rng)
        obs_data = problem.prepare_observations(
            sample_indices=batch_indices,
            obs_indices=obs_indices,
            snr_db=eval_cfg.snr_db,
            rng=noise_rng if eval_cfg.snr_db else None,
        )

        betas_opt = inverter.invert(
            x_obs=obs_data['x_obs'],
            u_obs=obs_data['u_obs'],
            x_full=obs_data['x_full'],
            config=eval_cfg.inversion,
            verbose=False,  # Don't spam per-batch epoch progress
        )

        preds = problem.predict_from_beta(
            problem.params,
            betas_opt,
            obs_data['x_full']
        )

        for i, sample_idx in enumerate(batch_indices):
            metrics_u = compute_all_metrics(preds['u_pred'][i], obs_data['u_true'][i])
            metrics_a = compute_all_metrics(preds['a_pred'][i], obs_data['a_true'][i])

            all_metrics_u.append(metrics_u)
            all_metrics_a.append(metrics_a)

            per_sample_results.append({
                'sample_idx': sample_idx,
                'u_metrics': metrics_u,
                'a_metrics': metrics_a,
            })

    aggregated_u = aggregate_metrics(all_metrics_u)
    aggregated_a = aggregate_metrics(all_metrics_a)

    results = {
        'config': {
            'problem_type': config.problem.type,
            'method': eval_cfg.method,
            'n_obs': eval_cfg.n_obs,
            'obs_sampling': eval_cfg.obs_sampling,
            'snr_db': eval_cfg.snr_db,
            'batch_size': batch_size,
            'inversion_epochs': eval_cfg.inversion.epochs,
            'loss_weights': {
                'pde': eval_cfg.inversion.loss_weights.pde,
                'data': eval_cfg.inversion.loss_weights.data,
            },
        },
        'n_samples': n_test,
        'u_metrics': aggregated_u,
        'a_metrics': aggregated_a,
        'per_sample': per_sample_results,
    }

    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate inverse problem (JAX)')
    parser.add_argument('--config', required=True, help='Path to config YAML')
    parser.add_argument('--pretrained', type=str, help='Override pretrained path')
    parser.add_argument('--method', choices=['igno', 'mcmc'], help='Inversion method')
    parser.add_argument('--n-obs', type=int, help='Number of observations')
    parser.add_argument('--snr-db', type=float, help='SNR in dB (use 0 for clean)')
    parser.add_argument('--inversion-epochs', type=int, help='Inversion epochs')
    parser.add_argument('--batch-size', type=int, help='Batch size for evaluation')
    parser.add_argument('--seed', type=int, help='Random seed')
    parser.add_argument('--quiet', action='store_true', help='Less verbose output')

    args = parser.parse_args()

    config = TrainingConfig.load(args.config)

    # Apply CLI overrides
    if args.pretrained:
        config.pretrained = {'path': args.pretrained}
    if args.method:
        config.evaluation.method = args.method
    if args.n_obs:
        config.evaluation.n_obs = args.n_obs
    if args.snr_db is not None:
        config.evaluation.snr_db = args.snr_db if args.snr_db > 0 else None
    if args.inversion_epochs:
        config.evaluation.inversion.epochs = args.inversion_epochs
    if args.batch_size:
        config.evaluation.batch_size = args.batch_size
    if args.seed:
        config.seed = args.seed

    print(f"Config: {args.config}")
    print(f"Problem: {config.problem.type}")
    print(f"JAX devices: {jax.devices()}")

    print(f"x64 enabled: {jax.config.jax_enable_x64}")
    print(f"Matmul precision: {jax.config.jax_default_matmul_precision}")
    print(f"NVIDIA_TF32_OVERRIDE: {os.environ.get('NVIDIA_TF32_OVERRIDE', 'not set')}")

    results = evaluate(config, verbose=not args.quiet)
    print_results(results)

    output_dir = Path(config.evaluation.results_dir)
    save_results(results, config, output_dir)


if __name__ == '__main__':
    main()