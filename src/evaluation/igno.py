import jax
import jax.numpy as jnp
from jax import random, jit, value_and_grad
import optax
from tqdm import trange
from typing import Dict, Any, Optional
from functools import partial

from src.problems import ProblemInstance
from src.solver.config import InversionConfig


class IGNOInverter:
    def __init__(self, problem: ProblemInstance, rng: jax.Array):
        self.problem = problem
        self.rng = rng
        self.frozen_params = problem.params
        self.loss_history = None
        problem.setup_inversion_grid(n_mesh_or_grid=7)

    def invert(
            self,
            x_obs: jnp.ndarray,
            u_obs: jnp.ndarray,
            x_full: jnp.ndarray,
            config: InversionConfig,
            verbose: bool = False,
    ) -> jnp.ndarray:
        batch_size = x_obs.shape[0]

        # Initialize at the mean of NF samples (matches reference implementation)
        self.rng, init_rng = random.split(self.rng)
        nf_samples = self.problem.sample_latent_from_nf(
            self.frozen_params, num_samples=1000, rng=init_rng
        )  # (1000, BETA_SIZE)
        beta_mean = jnp.mean(nf_samples, axis=0)  # (BETA_SIZE,)
        beta_init = jnp.tile(beta_mean[None, :], (batch_size, 1))  # (batch, BETA_SIZE)

        optimizer = self._create_optimizer(config)
        opt_state = optimizer.init(beta_init)

        weights = config.loss_weights
        target_type = getattr(self.problem, 'data_target_type', 'u')

        print(f"Loss weights: pde={weights.pde}, data={weights.data}, target={target_type}")

        def loss_fn(beta, rng_key):
            loss_pde = self.problem.loss_pde_from_beta(self.frozen_params, beta, rng_key)
            loss_data = self.problem.loss_data_from_beta(
                self.frozen_params, beta, x_obs, u_obs, target_type=target_type
            )
            total_loss = weights.pde * loss_pde + weights.data * loss_data
            return total_loss, {'loss_pde': loss_pde, 'loss_data': loss_data}

        @jit
        def update_step(beta, opt_state, rng_key):
            (loss, aux), grads = value_and_grad(loss_fn, has_aux=True)(beta, rng_key)
            updates, new_opt_state = optimizer.update(grads, opt_state, beta)
            new_beta = optax.apply_updates(beta, updates)
            return new_beta, new_opt_state, loss, aux

        beta = beta_init
        history_total, history_pde, history_data = [], [], []
        iterator = trange(config.epochs, desc="Inverting", disable=not verbose)
        for epoch in iterator:
            self.rng, step_rng = random.split(self.rng)
            beta, opt_state, loss, aux = update_step(beta, opt_state, step_rng)

            history_total.append(float(loss))
            history_pde.append(float(aux['loss_pde']))
            history_data.append(float(aux['loss_data']))

            if verbose and (epoch + 1) % 100 == 0:
                iterator.set_postfix({
                    'loss': f'{float(loss):.4f}',
                    'pde': f'{float(aux["loss_pde"]):.4f}',
                    'data': f'{float(aux["loss_data"]):.4f}',
                })

        self.rng, final_rng = random.split(self.rng)
        final_loss, final_aux = loss_fn(beta, final_rng)
        print(f"Final: loss_pde={float(final_aux['loss_pde']):.6f}, loss_data={float(final_aux['loss_data']):.6f}")

        pde_arr = jnp.array(history_pde)
        data_arr = jnp.array(history_data)
        self.loss_history = {
            'total': jnp.array(history_total),
            'pde': pde_arr,
            'data': data_arr,
            'weighted_pde': pde_arr * weights.pde,
            'weighted_data': data_arr * weights.data,
        }

        return beta

    def _create_optimizer(self, config: InversionConfig) -> optax.GradientTransformation:
        opt_cfg = config.optimizer
        lr = opt_cfg.lr

        if config.scheduler is not None and config.scheduler.type is not None:
            sched_cfg = config.scheduler

            if sched_cfg.type == 'StepLR':
                schedule = optax.exponential_decay(
                    init_value=lr,
                    transition_steps=sched_cfg.step_size,
                    decay_rate=sched_cfg.gamma,
                    staircase=True
                )
                lr = schedule
            elif sched_cfg.type == 'CosineAnnealing':
                schedule = optax.cosine_decay_schedule(
                    init_value=lr,
                    decay_steps=config.epochs,
                    alpha=sched_cfg.eta_min / lr if sched_cfg.eta_min else 0.0
                )
                lr = schedule

        OPTIMIZERS = {
            'Adam': optax.adam,
            'AdamW': optax.adamw,
            'SGD': optax.sgd,
        }

        opt_type = opt_cfg.type
        if opt_type not in OPTIMIZERS:
            raise ValueError(f"Unknown optimizer: {opt_type}")

        return OPTIMIZERS[opt_type](learning_rate=lr)
