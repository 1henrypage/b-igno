#!/usr/bin/env python3
"""
Main entry point for IGNO training (JAX version).

Trains encoder + decoders + NF jointly in a single phase.
Models are owned by ProblemInstance. Trainer orchestrates training.

Usage:
    python training.py --config configs/training/example_train.yaml
    python training.py --config configs/training/example_train.yaml --epochs 5000
    python training.py --config configs/training/example_train.yaml --pretrained path/to/checkpoint.pkl

JAX-specific options:
    python training.py --config configs/training/example_train.yaml --platform cpu
    python training.py --config configs/training/example_train.yaml --platform gpu
    python training.py --config configs/training/example_train.yaml --disable-jit
"""
import load_this_before_everything_else
import argparse
import sys
import os
from pathlib import Path
from typing import Optional, Dict, Any


def set_platform(platform: str):
    """Must be called before importing JAX."""
    if platform:
        os.environ['JAX_PLATFORMS'] = platform


def main():
    parser = argparse.ArgumentParser(description='Train IGNO models (JAX)')

    # Mutually exclusive: normal training vs NF retrain
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--config', help='Path to config YAML (normal training)')
    mode_group.add_argument('--retrain-nf', metavar='RUN_DIR',
                            help='Retrain NF on an existing run directory in-place')

    parser.add_argument('--pretrained', type=str, help='Path to pretrained checkpoint')
    parser.add_argument('--seed', type=int, help='Random seed')
    parser.add_argument('--epochs', type=int, help='Override number of epochs')
    parser.add_argument('--batch-size', type=int, help='Override batch size')
    parser.add_argument('--lr', type=float, help='Override learning rate')
    parser.add_argument('--nf-mode', choices=['joint', 'separate'],
                        help='NF training mode override (joint or separate)')
    parser.add_argument('--dry-run', action='store_true', help='Print config and exit')

    parser.add_argument('--platform', type=str, choices=['cpu', 'gpu', 'tpu'],
                        help='JAX platform (cpu, gpu, tpu)')
    parser.add_argument('--disable-jit', action='store_true',
                        help='Disable JIT compilation (for debugging)')
    parser.add_argument('--debug-nans', action='store_true',
                        help='Enable NaN debugging (slower)')

    args = parser.parse_args()

    if args.platform:
        set_platform(args.platform)

    import jax

    if args.disable_jit:
        jax.config.update("jax_disable_jit", True)
        print("JIT compilation disabled")

    if args.debug_nans:
        jax.config.update("jax_debug_nans", True)
        print("NaN debugging enabled")

    print(f"JAX version: {jax.__version__}")
    print(f"JAX devices: {jax.devices()}")
    print(f"JAX default backend: {jax.default_backend()}")
    print(f"x64 enabled: {jax.config.jax_enable_x64}")
    print(f"Matmul precision: {jax.config.jax_default_matmul_precision}")
    print(f"NVIDIA_TF32_OVERRIDE: {os.environ.get('NVIDIA_TF32_OVERRIDE', 'not set')}")

    sys.path.insert(0, str(Path(__file__).parent))
    from src.solver.config import TrainingConfig
    from src.solver.trainer import IGNOTrainer
    from src.problems import create_problem, ProblemInstance

    if args.retrain_nf:
        run_dir = Path(args.retrain_nf)
        results = run_retrain_nf(run_dir)
        print(f"\nNF retrain complete!")
        print(f"  NF final loss: {results['final_nf_loss']:.6f}")
        print(f"  NF time: {results['time']:.1f}s")
        return

    config = TrainingConfig.load(args.config)

    # Apply CLI overrides
    if args.pretrained:
        config.pretrained = {'path': args.pretrained}
    if args.seed:
        config.seed = args.seed
    if args.epochs:
        config.training.epochs = args.epochs
    if args.batch_size:
        config.training.batch_size = args.batch_size
    if args.lr:
        config.training.optimizer.lr = args.lr
    if args.nf_mode:
        config.training.nf.mode = args.nf_mode

    print(f"\n{'=' * 60}")
    print(f"IGNO Training (JAX)")
    print(f"{'=' * 60}")
    print(f"Config: {args.config}")
    print(f"Seed: {config.seed}")
    print(f"Problem: {config.problem.type}")
    print(f"  Train data: {config.problem.train_data}")
    print(f"  Test data: {config.problem.test_data}")
    print(f"Training:")
    print(f"  Epochs: {config.training.epochs}")
    print(f"  Batch size: {config.training.batch_size}")
    print(f"  Optimizer: {config.training.optimizer.type}, lr={config.training.optimizer.lr}")
    print(f"  Loss weights: pde={config.training.loss_weights.pde}, data={config.training.loss_weights.data}")
    print(f"  NF mode: {config.training.nf.mode}", end="")
    if config.training.nf.mode == 'joint':
        print(f" (loss_weight={config.training.nf.loss_weight})")
    else:
        print(f" (epochs={config.training.nf.epochs}, batch={config.training.nf.batch_size})")
    if config.pretrained:
        print(f"Pretrained: {config.pretrained.get('path')}")
    print(f"{'=' * 60}\n")

    if args.dry_run:
        print("[DRY RUN] Config loaded successfully. Exiting.")
        return

    results = run_training(config, pretrained_path=Path(args.pretrained) if args.pretrained else None)

    print(f"\nTraining complete!")
    print(f"  Best error: {results['best_error']:.6f}")
    print(f"  Time: {results['time']:.1f}s")
    if 'nf_training' in results:
        print(f"  NF final loss: {results['nf_training']['final_nf_loss']:.6f}")
        print(f"  NF time: {results['nf_training']['time']:.1f}s")


def run_retrain_nf(run_dir: Path) -> Dict[str, Any]:
    """The config is read from <run_dir>/config.yaml.  The NF must be in 'separate'
    mode — an error is raised otherwise.  All non-NF model weights are loaded from
    <run_dir>/weights/best.pt; the NF is re-initialised from the current class
    constants so you can change NF_NUM_FLOWS / NF_HIDDEN_DIM before calling this.
    """
    from src.solver.config import TrainingConfig
    from src.solver.trainer import IGNOTrainer
    from src.problems import create_problem

    run_dir = Path(run_dir)
    config_path = run_dir / "config.yaml"
    if not config_path.exists():
        raise RuntimeError(f"No config.yaml found in {run_dir}")

    config = TrainingConfig.load(config_path)

    if config.training.nf.mode != 'separate':
        raise RuntimeError(
            f"NF mode in config is '{config.training.nf.mode}', expected 'separate'. "
            "Set nf.mode = 'separate' in the config before retraining the NF."
        )

    print(f"\n{'=' * 60}")
    print(f"NF Retrain on existing run")
    print(f"{'=' * 60}")
    print(f"Run dir:  {run_dir}")
    print(f"Problem:  {config.problem.type}")
    print(f"NF epochs: {config.training.nf.epochs}, batch={config.training.nf.batch_size}")
    print(f"{'=' * 60}\n")

    problem = create_problem(config)
    trainer = IGNOTrainer(problem)
    trainer.setup_retrain_nf(config, run_dir)

    try:
        results = trainer.train_nf_separate(config)
    finally:
        trainer.close()

    print(f"\nDone. Checkpoints updated in: {run_dir / 'weights'}")
    return results


def run_training(
        config: 'TrainingConfig',
        problem: Optional['ProblemInstance'] = None,
        pretrained_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run IGNO training."""
    # Import here to allow platform selection before import
    from src.solver.trainer import IGNOTrainer
    from src.problems import create_problem

    if problem is None:
        print(f"Creating problem: {config.problem.type}")
        problem = create_problem(config)

    trainer = IGNOTrainer(problem)
    trainer.setup(config, pretrained_path=pretrained_path)

    try:
        results = trainer.train(config)

        if config.training.nf.mode == 'separate':
            nf_results = trainer.train_nf_separate(config)
            results['nf_training'] = nf_results
    finally:
        trainer.close()

    print(f"\nDone. Results saved to: {trainer.run_dir}")
    return results


if __name__ == '__main__':
    main()
