"""Training loop for ProblemInstance subclasses."""

import jax
import jax.numpy as jnp
from jax import random, jit, value_and_grad
import optax
import time
import numpy as np
from typing import Optional, Dict, Any, Tuple, List
from tqdm import trange
from datetime import datetime
from pathlib import Path
from functools import partial

from tensorboardX import SummaryWriter

from src.solver.config import TrainingConfig
from src.problems import ProblemInstance
from src.utils.solver_utils import create_train_state, create_data_batches, get_optimizer, get_scheduler


class IGNOTrainer:
    """JAX trainer for IGNO. Works with any ProblemInstance subclass."""

    def __init__(self, problem: ProblemInstance):
        self.problem = problem
        self.dtype = problem.dtype

        # Training state (will be created in setup)
        self.train_state: Optional[Dict] = None
        self.writer: Optional[SummaryWriter] = None

        self.run_dir: Optional[Path] = None
        self.weights_dir: Optional[Path] = None
        self.tb_dir: Optional[Path] = None

    def _create_run_dir(self, config: TrainingConfig) -> Path:
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        run_dir = Path(config.artifact_root) / f"{timestamp}_{config.run_name}"
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def _setup_directories(self) -> None:
        if self.run_dir is None:
            raise RuntimeError("run_dir must be set")

        self.weights_dir = self.run_dir / "weights"
        self.tb_dir = self.run_dir / "tensorboard"

        self.weights_dir.mkdir(parents=True, exist_ok=True)
        self.tb_dir.mkdir(parents=True, exist_ok=True)

    def _setup_tensorboard(self) -> None:
        self.writer = SummaryWriter(log_dir=str(self.tb_dir))

    def _log(self, tag: str, value: float, step: int) -> None:
        if self.writer:
            self.writer.add_scalar(tag, value, step)

    def setup(self, config: TrainingConfig, pretrained_path: Optional[Path] = None) -> None:
        self.problem.pre_train_check()

        # Create run directory
        self.run_dir = self._create_run_dir(config)
        self.problem.run_dir = self.run_dir

        # Setup directories and tensorboard
        self._setup_directories()
        self._setup_tensorboard()

        cfg = config.training

        # Set NF loss weight based on NF training mode
        nf_cfg = cfg.nf
        if nf_cfg.mode == 'separate':
            cfg.loss_weights.nf = 0.0
        else:
            cfg.loss_weights.nf = nf_cfg.loss_weight

        sample_inputs = self.problem.get_sample_inputs(cfg.batch_size)

        batch_keys = self.problem.get_batch_keys()
        first_key = batch_keys[0]
        n_train_samples = len(self.problem.train_data[first_key])
        batches_per_epoch = (n_train_samples + cfg.batch_size - 1) // cfg.batch_size
        num_steps = cfg.epochs * batches_per_epoch

        weight_decay_groups = self.problem.get_weight_decay_groups()

        self.train_state = create_train_state(
            models=self.problem.models,
            rng=self.problem.rng,
            sample_inputs=sample_inputs,
            weight_decay_groups=weight_decay_groups,
            optimizer_config=cfg.optimizer,
            scheduler_config=cfg.scheduler,
            num_steps=num_steps,
            epochs=cfg.epochs,
        )

        # Share params dict between problem and train_state so both stay in sync
        self.problem.params = self.train_state['params']

        # Load pretrained AFTER create_train_state so checkpoint params aren't overwritten
        ckpt_path = pretrained_path or config.get_pretrained_path()
        if ckpt_path:
            if not ckpt_path.exists():
                raise RuntimeError("Couldn't find pretrained checkpoint")
            print(f"Loading pretrained: {ckpt_path}")
            self.problem.load_checkpoint(ckpt_path)

        # Save config
        config.save(self.run_dir / "config.yaml")
        print(f"Run directory: {self.run_dir}")

    def setup_retrain_nf(self, config: TrainingConfig, run_dir: Path) -> None:
        """Setup trainer for in-place NF retraining on an existing run.

        Loads all non-NF model weights from <run_dir>/weights/best.pt and
        leaves the NF params freshly initialised (from current class constants).
        """
        self.problem.pre_train_check()

        # Point at the existing run dir — no new timestamped directory
        self.run_dir = run_dir
        self.problem.run_dir = run_dir

        self._setup_directories()
        self._setup_tensorboard()

        cfg = config.training

        # NF is always 'separate' here (caller already verified)
        cfg.loss_weights.nf = 0.0

        sample_inputs = self.problem.get_sample_inputs(cfg.batch_size)
        batch_keys = self.problem.get_batch_keys()
        first_key = batch_keys[0]
        n_train_samples = len(self.problem.train_data[first_key])
        batches_per_epoch = (n_train_samples + cfg.batch_size - 1) // cfg.batch_size
        num_steps = cfg.epochs * batches_per_epoch
        weight_decay_groups = self.problem.get_weight_decay_groups()

        # Initialise all model params fresh (NF gets new architecture from class constants)
        self.train_state = create_train_state(
            models=self.problem.models,
            rng=self.problem.rng,
            sample_inputs=sample_inputs,
            weight_decay_groups=weight_decay_groups,
            optimizer_config=cfg.optimizer,
            scheduler_config=cfg.scheduler,
            num_steps=num_steps,
            epochs=cfg.epochs,
        )
        self.problem.params = self.train_state['params']

        # Load all models EXCEPT 'nf' so the fresh NF params are preserved
        ckpt_path = run_dir / "weights" / "best.pt"
        if not ckpt_path.exists():
            raise RuntimeError(f"Checkpoint not found: {ckpt_path}")

        models_to_load = [m for m in self.problem.models.keys() if m != 'nf']
        self.problem.load_checkpoint(ckpt_path, models_to_load=models_to_load)

        print(f"Run directory: {self.run_dir}")

    def train(self, config: TrainingConfig) -> Dict[str, Any]:
        cfg = config.training
        train_data = self.problem.get_train_data()
        test_data = self.problem.get_test_data()

        batch_keys = self.problem.get_batch_keys()
        train_step_fn = self._create_train_step(cfg)
        eval_step_fn = self._create_eval_step()

        t_start = time.time()
        best_error = float('inf')

        nf_mode = cfg.nf.mode
        print("\n" + "=" * 60)
        if nf_mode == 'separate':
            print("IGNO Training (encoder + decoders) — NF trained separately")
        else:
            print("IGNO Joint Training (encoder + decoders + NF)")
        print("=" * 60)
        print(f"Loss weights: pde={cfg.loss_weights.pde}, data={cfg.loss_weights.data}, nf={cfg.loss_weights.nf}")
        print(f"Epochs: {cfg.epochs}, Batch size: {cfg.batch_size}")
        print(f"Models: {list(self.problem.models.keys())}")
        print(f"Weight decay groups: {self.problem.get_weight_decay_groups()}")
        if nf_mode == 'separate':
            print(f"NF separate training: {cfg.nf.epochs} epochs, batch={cfg.nf.batch_size}")
        print("=" * 60 + "\n")

        for epoch in trange(cfg.epochs, desc="Training"):
            self.train_state['rng'], data_rng = random.split(self.train_state['rng'])
            train_arrays = [train_data[k] for k in batch_keys]
            train_batches, n_train = create_data_batches(
                *train_arrays,
                batch_size=cfg.batch_size,
                shuffle=True,
                rng=data_rng
            )

            loss_sum = 0.
            pde_sum = 0.
            data_sum = 0.
            nf_sum = 0.

            for batch_tuple in train_batches:
                batch = {k: v for k, v in zip(batch_keys, batch_tuple)}
                self.train_state['rng'], step_rng = random.split(self.train_state['rng'])

                new_params, new_opt_states, metrics = train_step_fn(
                    self.train_state['params'],
                    self.train_state['opt_states'],
                    batch,
                    step_rng
                )
                self.train_state['params'] = new_params
                self.train_state['opt_states'] = new_opt_states

                loss_sum += float(metrics['loss'])
                pde_sum += float(metrics['loss_pde'])
                data_sum += float(metrics['loss_data'])
                nf_sum += float(metrics['loss_nf'])

            self.problem.params = self.train_state['params']

            test_arrays = [test_data[k] for k in batch_keys]
            test_batches, n_test = create_data_batches(
                *test_arrays,
                batch_size=cfg.batch_size,
                shuffle=False
            )

            error_sum = 0.
            test_nf_sum = 0.

            for batch_tuple in test_batches:
                batch = {k: v for k, v in zip(batch_keys, batch_tuple)}
                metrics = eval_step_fn(
                    self.train_state['params'],
                    batch
                )
                error_sum += float(metrics['error'])
                test_nf_sum += float(metrics['nf_loss'])

            avg_loss = loss_sum / n_train
            avg_error = error_sum / n_test
            avg_nf = nf_sum / n_train
            avg_test_nf = test_nf_sum / n_test

            self._log("train/loss", avg_loss, epoch)
            self._log("train/pde", pde_sum / n_train, epoch)
            self._log("train/data", data_sum / n_train, epoch)
            self._log("train/nf", avg_nf, epoch)
            self._log("test/error", avg_error, epoch)
            self._log("test/nf", avg_test_nf, epoch)

            if avg_error < best_error:
                best_error = avg_error
                self.problem.save_checkpoint(
                    self.weights_dir / 'best.pt',
                    epoch=epoch,
                    opt_states=self.train_state['opt_states'],
                    metric=float(avg_error),
                    metric_name='error'
                )

            if (epoch + 1) % cfg.epoch_show == 0:
                print(f"\nEpoch {epoch + 1}:")
                print(f"  Loss: {avg_loss:.4f} (pde={pde_sum / n_train:.4f}, "
                      f"data={data_sum / n_train:.4f}, nf={avg_nf:.4f})")
                print(f"  Test Error: {avg_error:.4f}, Test NF NLL: {avg_test_nf:.4f}")

                first_batch = {k: train_data[k][:cfg.batch_size] for k in batch_keys}
                self.problem.run_diagnostics(
                    epoch=epoch,
                    params=self.train_state['params'],
                    sample_data=first_batch,
                    logger=self._log
                )

            self.train_state['step'] += 1

        self.problem.save_checkpoint(
            self.weights_dir / 'last.pt',
            epoch=cfg.epochs - 1,
            opt_states=self.train_state['opt_states']
        )

        total_time = time.time() - t_start
        print(f"\nTraining completed in {total_time:.1f}s")
        print(f"Best error: {best_error:.4f}")
        print(f"Checkpoints saved to: {self.weights_dir}")

        return {
            'best_error': best_error,
            'time': total_time,
        }

    def _create_train_step(self, cfg):
        problem = self.problem
        weights = cfg.loss_weights

        optimizers = self.train_state['optimizers']

        @jit
        def train_step(params, opt_states, batch, rng):
            def loss_fn(params_dict):
                total_loss, metrics = problem.compute_training_losses(
                    params_dict, batch, rng, weights
                )
                return total_loss, metrics

            (loss, metrics), grads = value_and_grad(loss_fn, has_aux=True)(params)

            new_params = {}
            new_opt_states = {}
            for name in params.keys():
                updates, new_opt_state = optimizers[name].update(
                    grads[name],
                    opt_states[name],
                    params[name]
                )
                new_params[name] = optax.apply_updates(params[name], updates)
                new_opt_states[name] = new_opt_state

            return new_params, new_opt_states, metrics

        return train_step

    def _create_eval_step(self):
        problem = self.problem

        @jit
        def eval_step(params, batch):
            return problem.compute_eval_metrics(params, batch)

        return eval_step

    def _collect_all_betas(self, train_data: Dict, batch_keys: List[str], batch_size: int) -> jnp.ndarray:
        """Collect encoder outputs (betas) for all training samples."""
        model_enc = self.problem.models['enc']
        params = self.train_state['params']

        if 'a' not in train_data:
            raise ValueError("train_data must contain key 'a' for beta collection")

        a_data = train_data['a']
        n_samples = a_data.shape[0]
        all_betas = []

        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            a_batch = a_data[start:end]
            beta_batch = model_enc.apply({'params': params['enc']}, a_batch)
            all_betas.append(beta_batch)

        return jnp.concatenate(all_betas, axis=0)

    def train_nf_separate(self, config: TrainingConfig) -> Dict[str, Any]:
        """Run separate NF training phase on frozen encoder outputs."""
        nf_cfg = config.training.nf
        train_data = self.problem.get_train_data()
        batch_keys = self.problem.get_batch_keys()

        test_data = self.problem.get_test_data()

        print(f"Collecting encoder outputs for {len(train_data['a'])} training samples...")
        all_betas = self._collect_all_betas(train_data, batch_keys, config.training.batch_size)
        n_betas = all_betas.shape[0]
        print(f"Collected {n_betas} betas of shape {all_betas.shape[1:]}")

        print(f"Collecting encoder outputs for {len(test_data['a'])} test samples...")
        all_test_betas = self._collect_all_betas(test_data, batch_keys, config.training.batch_size)
        print(f"Collected {all_test_betas.shape[0]} test betas")

        # Create NF optimizer and scheduler
        nf_epochs = nf_cfg.epochs
        nf_batch_size = nf_cfg.batch_size
        batches_per_epoch = (n_betas + nf_batch_size - 1) // nf_batch_size
        num_steps = nf_epochs * batches_per_epoch

        nf_schedule = get_scheduler(nf_cfg.scheduler, nf_cfg.optimizer, num_steps, nf_epochs)
        nf_optimizer = get_optimizer(
            nf_cfg.optimizer,
            learning_rate=nf_schedule if nf_schedule is not None else nf_cfg.optimizer.lr
        )

        model_nf = self.problem.models['nf']
        nf_params = self.train_state['params']['nf']
        nf_opt_state = nf_optimizer.init(nf_params)

        @jit
        def nf_train_step(nf_params, nf_opt_state, beta_batch):
            def nf_loss_fn(p):
                return model_nf.apply({'params': p}, beta_batch, method=model_nf.loss)
            loss, grads = value_and_grad(nf_loss_fn)(nf_params)
            updates, new_state = nf_optimizer.update(grads, nf_opt_state, nf_params)
            return optax.apply_updates(nf_params, updates), new_state, loss

        print("\n" + "=" * 60)
        print("Phase 2: Separate NF Training")
        print("=" * 60)
        print(f"Epochs: {nf_epochs}, Batch size: {nf_batch_size}")
        print(f"Optimizer: {nf_cfg.optimizer.type}, lr={nf_cfg.optimizer.lr}, wd={nf_cfg.optimizer.weight_decay}")
        print(f"Scheduler: {nf_cfg.scheduler.type}, step={nf_cfg.scheduler.step_size}, gamma={nf_cfg.scheduler.gamma}")
        print("=" * 60 + "\n")

        t_start = time.time()
        rng = self.train_state['rng']

        best_test_loss = float('inf')
        best_nf_params = nf_params

        for epoch in trange(nf_epochs, desc="NF Training"):
            rng, data_rng = random.split(rng)
            indices = random.permutation(data_rng, n_betas)

            epoch_loss = 0.0
            n_batches = 0
            for start in range(0, n_betas, nf_batch_size):
                end = min(start + nf_batch_size, n_betas)
                batch_idx = indices[start:end]
                beta_batch = all_betas[batch_idx]

                nf_params, nf_opt_state, loss = nf_train_step(nf_params, nf_opt_state, beta_batch)
                epoch_loss += float(loss)
                n_batches += 1

            avg_loss = epoch_loss / n_batches
            self._log("nf_separate/loss", avg_loss, epoch)

            test_loss, n_test_batches = 0.0, 0
            for start in range(0, all_test_betas.shape[0], nf_batch_size):
                test_batch = all_test_betas[start:start + nf_batch_size]
                test_loss += float(model_nf.apply({'params': nf_params}, test_batch, method=model_nf.loss))
                n_test_batches += 1
            avg_test_loss = test_loss / n_test_batches
            self._log("nf_separate/test_loss", avg_test_loss, epoch)

            if avg_test_loss < best_test_loss:
                best_test_loss = avg_test_loss
                best_nf_params = jax.tree.map(lambda x: x.copy(), nf_params)

        # Write final NF params back and save last.pt
        self.train_state['params']['nf'] = nf_params
        self.train_state['rng'] = rng
        self.problem.params = self.train_state['params']

        self.problem.save_checkpoint(
            self.weights_dir / 'last.pt',
            epoch=config.training.epochs - 1,
            opt_states=self.train_state['opt_states']
        )

        # Save best NF params to best.pt
        best_pt = self.weights_dir / 'best.pt'
        if best_pt.exists():
            from src.utils.solver_utils import load_checkpoint, save_checkpoint
            state, metadata = load_checkpoint(best_pt)
            state['params']['nf'] = best_nf_params
            save_checkpoint(best_pt, state, metadata)

        total_time = time.time() - t_start
        final_loss = avg_loss
        print(f"\nNF training completed in {total_time:.1f}s, final loss: {final_loss:.4f}")

        return {
            'final_nf_loss': final_loss,
            'time': total_time,
        }

    def close(self) -> None:
        if self.writer:
            self.writer.close()
            self.writer = None
