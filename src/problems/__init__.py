"""Abstract base class for inverse PDE problem instances."""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Type, Callable, Literal, List, Any, Tuple
from datetime import datetime
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from jax import random, jit
from flax import linen as nn

from src.utils.Losses import MyError, MyLoss
from src.utils.misc_utils import setup_seed
from src.utils.TestFun_ParticleWNN import TestFun_ParticleWNN

_PROBLEM_REGISTRY: Dict[str, Type['ProblemInstance']] = {}


def register_problem(name: str) -> Callable:
    def decorator(cls: Type['ProblemInstance']) -> Type['ProblemInstance']:
        _PROBLEM_REGISTRY[name] = cls
        return cls

    return decorator


def get_problem_class(name: str) -> Type['ProblemInstance']:
    if name not in _PROBLEM_REGISTRY:
        raise ValueError(f"Unknown problem: '{name}'. Available: {list(_PROBLEM_REGISTRY.keys())}")
    return _PROBLEM_REGISTRY[name]


def list_problems() -> list:
    return list(_PROBLEM_REGISTRY.keys())


class ProblemInstance(ABC):
    """
    Abstract base for PDE problems in JAX.

    Key principles:
    - Models are Flax modules (Python objects)
    - Parameters stored separately (functional)
    - Loss functions are JIT-compiled pure functions
    - No mutable state during computation

    Training uses:
    - loss_pde(params, a, rng): Encodes a -> beta, then computes PDE loss
    - loss_data(params, x, a, u): Encodes a -> beta, then computes data loss
    - NF loss computed on stop_gradient(beta) in trainer

    Inversion uses:
    - loss_pde_from_beta(params, beta, rng): PDE loss directly from beta
    - loss_data_from_beta(params, beta, x, target, target_type): Data loss from beta
    - sample_latent_from_nf(params, num_samples, rng): Sample from NF prior
    - log_prob_latent(params, beta): Compute log p(beta) for MCMC
    """

    def __init__(
            self,
            seed: int,
            dtype: jnp.dtype = jnp.float32,
            train_data_path: str = None,
            test_data_path: str = None,
    ) -> None:
        self.rng = setup_seed(seed)
        self.seed = seed
        self.dtype = dtype
        self.run_dir: Optional[Path] = None

        # Data paths
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path

        self.models: Dict[str, nn.Module] = {}
        self.params: Dict[str, Any] = {}
        self.train_data: Dict[str, jnp.ndarray] = {}
        self.test_data: Dict[str, jnp.ndarray] = {}

        self.get_loss = None
        self.get_error = None
        self.init_error()
        self.init_loss()

        # Inversion grid (set by setup_inversion_grid; None = use training grid)
        self.inv_int_grid = None
        self.inv_v = None
        self.inv_dv_dr = None
        self.inv_n_grid = None

    @abstractmethod
    def _build_models(self) -> Dict[str, nn.Module]:
        """Build all Flax modules for this problem."""
        raise NotImplementedError

    @abstractmethod
    def get_sample_inputs(self, batch_size: int) -> Dict[str, Dict[str, jax.Array]]:
        """Return sample inputs for each model (used for Flax parameter init)."""
        raise NotImplementedError

    @abstractmethod
    def get_weight_decay_groups(self) -> Dict[str, bool]:
        """Map model name -> whether it gets weight decay."""
        raise NotImplementedError

    def get_batch_keys(self) -> List[str]:
        """Return keys to extract from train_data for batching. Override if needed."""
        return ['a', 'u', 'x']

    def initialize_models(
            self,
            sample_inputs: Dict[str, Dict[str, jax.Array]]
    ) -> None:
        """Initialize model parameters."""
        self.rng, *init_rngs = random.split(self.rng, len(self.models) + 1)

        for (name, model), init_rng in zip(self.models.items(), init_rngs):
            if name in sample_inputs:
                variables = model.init(init_rng, **sample_inputs[name])
                self.params[name] = variables['params']
                print(f"  Initialized {name}: {self._count_params(self.params[name]):,} params")
            else:
                raise ValueError(f"No sample input for model '{name}'")

    def _count_params(self, params) -> int:
        """Count parameters in pytree"""
        return sum(x.size for x in jax.tree_util.tree_leaves(params))

    def compute_training_losses(
            self,
            params: Dict[str, Any],
            batch: Dict[str, jnp.ndarray],
            rng: jax.Array,
            loss_weights: Any
    ) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
        """Main training interface: computes PDE + data + NF losses for a batch."""
        a = batch['a']
        u = batch['u']
        x = batch['x']

        beta = self.models['enc'].apply({'params': params['enc']}, a)

        loss_pde = self.loss_pde(params, a, rng)
        loss_data = self.loss_data(params, x, a, u)

        # NF loss on DETACHED beta (key insight: gradients don't flow to encoder)
        beta_detached = jax.lax.stop_gradient(beta)
        loss_nf = self.models['nf'].apply(
            {'params': params['nf']},
            beta_detached,
            method=self.models['nf'].loss
        )

        total_loss = loss_weights.pde * loss_pde + loss_weights.data * loss_data + loss_weights.nf * loss_nf

        metrics = {
            'loss': total_loss,
            'loss_pde': loss_pde,
            'loss_data': loss_data,
            'loss_nf': loss_nf,
        }

        return total_loss, metrics

    def compute_eval_metrics(
            self,
            params: Dict[str, Any],
            batch: Dict[str, jnp.ndarray],
    ) -> Dict[str, jnp.ndarray]:
        """Compute evaluation metrics for a batch."""
        a = batch['a']
        u = batch['u']
        x = batch['x']

        error = self.error(params, x, a, u)

        beta = self.models['enc'].apply({'params': params['enc']}, a)
        nf_loss = self.models['nf'].apply(
            {'params': params['nf']},
            beta,
            method=self.models['nf'].loss
        )

        return {
            'error': jnp.mean(error),
            'nf_loss': nf_loss,
        }

    def run_diagnostics(
            self,
            epoch: int,
            params: Dict[str, Any],
            sample_data: Dict[str, jnp.ndarray],
            logger: Callable[[str, float, int], None]
    ) -> None:
        """Run problem-specific diagnostics. Default: NF health check."""
        self._run_nf_diagnostics(epoch, params, sample_data['a'], logger)

    def _run_nf_diagnostics(self, epoch, params, a_sample, logger):
        nf = self.models['nf']
        enc = self.models['enc']

        beta = enc.apply({'params': params['enc']}, a_sample)
        n_samples, n_dims = beta.shape

        # Forward through NF: beta -> z
        z_out, log_det_fwd = nf.apply(
            {'params': params['nf']},
            beta,
            method=nf.__call__
        )

        # z-space statistics
        z_mean_total = float(jnp.mean(z_out))
        z_std_total = float(jnp.std(z_out))
        z_min = float(jnp.min(z_out))
        z_max = float(jnp.max(z_out))

        # Per-dimension statistics for z
        z_mean_per_dim = jnp.mean(z_out, axis=0)
        z_std_per_dim = jnp.std(z_out, axis=0)
        z_dead_dims = int(jnp.sum(z_std_per_dim < 0.05))  # Collapsed dimensions
        z_dim_std_min = float(jnp.min(z_std_per_dim))
        z_dim_std_max = float(jnp.max(z_std_per_dim))

        # beta (encoded) statistics
        beta_mean = float(jnp.mean(beta))
        beta_std = float(jnp.std(beta))
        beta_min = float(jnp.min(beta))
        beta_max = float(jnp.max(beta))

        # Per-dimension statistics for encoded beta
        beta_mean_per_dim = jnp.mean(beta, axis=0)
        beta_std_per_dim = jnp.std(beta, axis=0)
        beta_dead_dims = int(jnp.sum(beta_std_per_dim < 0.05))
        beta_dim_std_min = float(jnp.min(beta_std_per_dim))
        beta_dim_std_max = float(jnp.max(beta_std_per_dim))

        # Check how much of [-1,1] is being utilized
        beta_utilization = float((beta_max - beta_min) / 2.0)  # 1.0 = full range

        # Distribution shape: check for clustering near boundaries or center
        beta_near_boundary = float(jnp.mean(jnp.abs(beta) > 0.9))  # Fraction near ±1
        beta_near_center = float(jnp.mean(jnp.abs(beta) < 0.1))    # Fraction near 0

        # invertibility check
        beta_rec, _ = nf.apply(
            {'params': params['nf']},
            z_out,
            method=nf.inverse
        )
        rec_err = float(jnp.mean(jnp.abs(beta - beta_rec)))
        rec_err_max = float(jnp.max(jnp.abs(beta - beta_rec)))

        # sampling check
        self.rng, sample_key = random.split(self.rng)
        nf_alpha = nf.alpha
        z_prior_01 = random.beta(sample_key, a=nf_alpha, b=nf_alpha, shape=(n_samples, n_dims))
        z_prior = 2.0 * z_prior_01 - 1.0

        beta_sampled, log_det_inv = nf.apply(
            {'params': params['nf']},
            z_prior,
            method=nf.inverse
        )

        # Sampled beta statistics
        beta_sampled_mean = float(jnp.mean(beta_sampled))
        beta_sampled_std = float(jnp.std(beta_sampled))
        beta_sampled_min = float(jnp.min(beta_sampled))
        beta_sampled_max = float(jnp.max(beta_sampled))

        beta_sampled_std_per_dim = jnp.std(beta_sampled, axis=0)
        beta_sampled_dead_dims = int(jnp.sum(beta_sampled_std_per_dim < 0.05))

        # Distribution match: compare encoded vs sampled distributions
        dist_mean_diff = float(jnp.mean(jnp.abs(jnp.mean(beta, axis=0) - jnp.mean(beta_sampled, axis=0))))
        dist_std_diff = float(jnp.mean(jnp.abs(jnp.std(beta, axis=0) - jnp.std(beta_sampled, axis=0))))

        # log-det analysis
        ldj_mean = float(jnp.mean(log_det_fwd))
        ldj_std = float(jnp.std(log_det_fwd))
        ldj_min = float(jnp.min(log_det_fwd))
        ldj_max = float(jnp.max(log_det_fwd))

        # Check for exploding/vanishing log-det
        ldj_inv_mean = float(jnp.mean(log_det_inv))

        # boundary violations
        beta_outside = jnp.abs(beta) > 1.0
        frac_violations_enc = float(jnp.mean(beta_outside))
        dims_with_violations_enc = int(jnp.sum(jnp.any(beta_outside, axis=0)))

        beta_sampled_outside = jnp.abs(beta_sampled) > 1.0
        frac_violations_sampled = float(jnp.mean(beta_sampled_outside))
        dims_with_violations_sampled = int(jnp.sum(jnp.any(beta_sampled_outside, axis=0)))

        logger("nf_health/z_mean", z_mean_total, epoch)
        logger("nf_health/z_std", z_std_total, epoch)
        logger("nf_health/z_min", z_min, epoch)
        logger("nf_health/z_max", z_max, epoch)
        logger("nf_health/z_dead_dims", float(z_dead_dims), epoch)
        logger("nf_health/z_dim_std_min", z_dim_std_min, epoch)
        logger("nf_health/z_dim_std_max", z_dim_std_max, epoch)

        logger("nf_health/beta_enc_mean", beta_mean, epoch)
        logger("nf_health/beta_enc_std", beta_std, epoch)
        logger("nf_health/beta_enc_min", beta_min, epoch)
        logger("nf_health/beta_enc_max", beta_max, epoch)
        logger("nf_health/beta_enc_dead_dims", float(beta_dead_dims), epoch)
        logger("nf_health/beta_enc_utilization", beta_utilization, epoch)
        logger("nf_health/beta_enc_near_boundary", beta_near_boundary, epoch)
        logger("nf_health/beta_enc_near_center", beta_near_center, epoch)

        logger("nf_health/beta_sampled_mean", beta_sampled_mean, epoch)
        logger("nf_health/beta_sampled_std", beta_sampled_std, epoch)
        logger("nf_health/beta_sampled_min", beta_sampled_min, epoch)
        logger("nf_health/beta_sampled_max", beta_sampled_max, epoch)
        logger("nf_health/beta_sampled_dead_dims", float(beta_sampled_dead_dims), epoch)

        logger("nf_health/dist_mean_diff", dist_mean_diff, epoch)
        logger("nf_health/dist_std_diff", dist_std_diff, epoch)

        logger("nf_health/rec_error_mean", rec_err, epoch)
        logger("nf_health/rec_error_max", rec_err_max, epoch)

        logger("nf_health/log_det_fwd_mean", ldj_mean, epoch)
        logger("nf_health/log_det_fwd_std", ldj_std, epoch)
        logger("nf_health/log_det_fwd_min", ldj_min, epoch)
        logger("nf_health/log_det_fwd_max", ldj_max, epoch)
        logger("nf_health/log_det_inv_mean", ldj_inv_mean, epoch)

        print(f"  [NF Health] Roundtrip Err: mean={rec_err:.2e}, max={rec_err_max:.2e}")
        print(f"  [NF Health] Z-Space: Mean={z_mean_total:.3f}, Std={z_std_total:.3f}, "
              f"Range=[{z_min:.3f}, {z_max:.3f}], Dead dims={z_dead_dims}/{n_dims}")
        print(f"  [NF Health] Beta (enc): Mean={beta_mean:.3f}, Std={beta_std:.3f}, "
              f"Range=[{beta_min:.3f}, {beta_max:.3f}]")
        print(f"             Utilization={beta_utilization*100:.1f}%, "
              f"Near boundary={beta_near_boundary*100:.1f}%, "
              f"Near center={beta_near_center*100:.1f}%, "
              f"Dead dims={beta_dead_dims}/{n_dims}")
        print(f"  [NF Health] Beta (sampled): Mean={beta_sampled_mean:.3f}, Std={beta_sampled_std:.3f}, "
              f"Range=[{beta_sampled_min:.3f}, {beta_sampled_max:.3f}], Dead dims={beta_sampled_dead_dims}/{n_dims}")
        print(f"  [NF Health] Distribution match: Mean diff={dist_mean_diff:.4f}, Std diff={dist_std_diff:.4f}")
        print(f"  [NF Health] Log-det: mean={ldj_mean:.2f}, std={ldj_std:.2f}, range=[{ldj_min:.2f}, {ldj_max:.2f}]")

        if rec_err > 1e-3:
            print("  WARNING: Poor invertibility (mean).")
        if rec_err_max > 1e-2:
            print("  WARNING: Poor invertibility (max).")
        if frac_violations_enc > 0.01:
            print(f"  WARNING: {frac_violations_enc*100:.1f}% of encoded betas outside [-1,1].")
        if frac_violations_sampled > 0.01:
            print(f"  WARNING: {frac_violations_sampled*100:.1f}% of NF-sampled betas outside [-1,1].")
        if z_dead_dims > 0:
            print(f"  WARNING: {z_dead_dims} dead dimensions in Z-space (std < 0.05).")
        if beta_dead_dims > 0:
            print(f"  WARNING: {beta_dead_dims} dead dimensions in encoded beta (std < 0.05).")
        if beta_sampled_dead_dims > 0:
            print(f"  WARNING: {beta_sampled_dead_dims} dead dimensions in sampled beta (std < 0.05).")
        if beta_utilization < 0.5:
            print(f"  WARNING: Low beta utilization ({beta_utilization*100:.1f}%). Encoder may be collapsing.")
        if dist_mean_diff > 0.1:
            print(f"  WARNING: Large mean difference between encoded and sampled betas.")
        if dist_std_diff > 0.1:
            print(f"  WARNING: Large std difference between encoded and sampled betas.")
        if abs(ldj_mean) > 200:
            print(f"  WARNING: Log-det magnitude very large. Potential numerical issues.")

    def setup_inversion_grid(self, n_mesh_or_grid: int = 7) -> None:
        """Precompute test function grid for inversion (may differ from training grid).

        DarcyContinuous uses n=9 for training but n=7 for inversion. DP and EIT use n=7 for both.
        """
        inv_int_grid, inv_v, inv_dv_dr = TestFun_ParticleWNN(
            fun_type='Wendland',
            dim=2,
            n_mesh_or_grid=n_mesh_or_grid,
        ).get_testFun()
        self.inv_int_grid = inv_int_grid
        self.inv_v = inv_v
        self.inv_dv_dr = inv_dv_dr
        self.inv_n_grid = inv_int_grid.shape[0]
        print(f"  Inversion grid: n_mesh_or_grid={n_mesh_or_grid}, "
              f"n_grid={self.inv_n_grid}")

    def save_checkpoint(
            self,
            path: Path,
            epoch: int,
            opt_states: Optional[Dict] = None,
            metric: Optional[float] = None,
            metric_name: Optional[str] = None,
            extra: Optional[Dict] = None,
    ) -> Path:
        from src.utils.solver_utils import save_checkpoint as save_ckpt

        state = {
            'params': self.params,
            'epoch': epoch,
            'timestamp': datetime.now().isoformat(),
        }

        if opt_states is not None:
            state['opt_states'] = opt_states

        metadata = {}
        if metric is not None:
            metadata['metric'] = float(metric)
            metadata['metric_name'] = metric_name
        if extra is not None:
            metadata.update(extra)

        save_ckpt(path, state, metadata)
        return path

    def load_checkpoint(
            self,
            path: Path,
            models_to_load: Optional[List[str]] = None,
            load_opt_states: bool = False,
    ) -> Dict[str, Any]:
        from src.utils.solver_utils import load_checkpoint as load_ckpt

        state, metadata = load_ckpt(path)

        print(f"Loading checkpoint: {path}")

        # Load parameters
        to_load = models_to_load if models_to_load else list(self.models.keys())
        for name in to_load:
            if name in state['params']:
                self.params[name] = state['params'][name]
                print(f"  Loaded {name}")

        result = {'epoch': state.get('epoch', 0), **metadata}

        if load_opt_states and 'opt_states' in state:
            result['opt_states'] = state['opt_states']

        return result

    def sample_latent_from_nf(
            self,
            params: Dict[str, Any],
            num_samples: int,
            rng: jax.Array
    ) -> jnp.ndarray:
        """Sample from NF prior. Returns (num_samples, latent_dim)."""
        nf = self.models['nf']
        return nf.apply(
            {'params': params['nf']},
            rng,
            num_samples,
            method=nf.sample
        )

    def log_prob_latent(
            self,
            params: Dict[str, Any],
            beta: jnp.ndarray
    ) -> jnp.ndarray:
        """Log p(beta) under the NF prior. Returns (batch,) log probabilities."""
        nf = self.models['nf']
        return nf.apply(
            {'params': params['nf']},
            beta,
            method=nf.log_prob
        )

    @abstractmethod
    def loss_pde(
            self,
            params: Dict[str, Any],
            a: jnp.ndarray,
            rng: jax.Array
    ) -> jnp.ndarray:
        raise NotImplementedError

    @abstractmethod
    def loss_data(
            self,
            params: Dict[str, Any],
            x: jnp.ndarray,
            a: jnp.ndarray,
            u: jnp.ndarray
    ) -> jnp.ndarray:
        raise NotImplementedError

    @abstractmethod
    def error(
            self,
            params: Dict[str, Any],
            x: jnp.ndarray,
            a: jnp.ndarray,
            u: jnp.ndarray
    ) -> jnp.ndarray:
        raise NotImplementedError

    # --- From-beta methods (for inversion) ---

    @abstractmethod
    def loss_pde_from_beta(
            self,
            params: Dict[str, Any],
            beta: jnp.ndarray,
            rng: jax.Array
    ) -> jnp.ndarray:
        raise NotImplementedError

    @abstractmethod
    def loss_data_from_beta(
            self,
            params: Dict[str, Any],
            beta: jnp.ndarray,
            x: jnp.ndarray,
            target: jnp.ndarray,
            target_type: Literal['a', 'u']
    ) -> jnp.ndarray:
        raise NotImplementedError

    @abstractmethod
    def error_from_beta(
            self,
            params: Dict[str, Any],
            beta: jnp.ndarray,
            x: jnp.ndarray,
            target: jnp.ndarray,
            target_type: Literal['a', 'u']
    ) -> jnp.ndarray:
        raise NotImplementedError

    def predict_from_beta(
            self,
            params: Dict[str, Any],
            beta: jnp.ndarray,
            x: jnp.ndarray
    ) -> Dict[str, jnp.ndarray]:
        """Predict u and a fields from beta. Returns dict with 'u_pred' and 'a_pred'."""
        raise NotImplementedError

    # --- Observation utilities ---

    def add_noise_snr(
            self,
            signal: jnp.ndarray,
            snr_db: float,
            rng: jax.Array
    ) -> jnp.ndarray:
        """Add Gaussian noise to achieve target SNR in dB."""
        signal_power = jnp.mean(signal ** 2)
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise_std = jnp.sqrt(noise_power)
        return signal + random.normal(rng, signal.shape) * noise_std

    def sample_observation_indices(
            self,
            n_total: int,
            n_obs: int,
            method: str = "random",
            rng: jax.Array = None
    ) -> jnp.ndarray:
        """Sample observation point indices ("random" or "grid")."""

        if method == "random":
            if rng is None:
                raise ValueError("rng required for random sampling")
            indices = random.choice(rng, n_total, (n_obs,), replace=False)
            return jnp.sort(indices)
        elif method == "grid":
            step = max(1, n_total // n_obs)
            return jnp.arange(0, n_total, step)[:n_obs]
        else:
            raise ValueError(f"Unknown sampling method: {method}")

    @abstractmethod
    def prepare_observations(
            self,
            sample_indices: List[int],
            obs_indices: jnp.ndarray,
            snr_db: float = None,
            rng: jax.Array = None
    ) -> Dict[str, jnp.ndarray]:
        raise NotImplementedError

    # --- Utilities ---

    def get_train_data(self) -> Dict[str, jnp.ndarray]:
        return self.train_data

    def get_test_data(self) -> Dict[str, jnp.ndarray]:
        return self.test_data

    def init_error(
            self,
            err_type: str = 'lp_rel',
            d: int = 2,
            p: int = 2
    ) -> None:
        self.get_error = MyError(d=d, p=p)(err_type)

    def init_loss(self, loss_type: str = 'mse_org') -> None:
        self.get_loss = MyLoss()(loss_type)

    def pre_train_check(self) -> None:
        if self.get_loss is None or self.get_error is None:
            self.init_loss()
            self.init_error()

    @abstractmethod
    def get_n_test_samples(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def get_n_points(self) -> int:
        raise NotImplementedError


def create_problem(config, load_train_data: bool = True) -> ProblemInstance:
    """Create problem from config."""
    problem_cls = get_problem_class(config.problem.type)
    return problem_cls(
        seed=config.seed,
        train_data_path=config.problem.train_data if load_train_data else None,
        test_data_path=config.problem.test_data,
    )


# Auto-register problems
from src.problems.darcy_continuous import DarcyContinuous
from src.problems.darcy_piecewise import DarcyPiecewise, DarcyPiecewise5v100, DarcyPiecewise5v1000
from src.problems.eit import EIT
from src.problems.burgers import Burgers
