"""Darcy flow with continuous (smooth) coefficient field: -div(a * grad(u)) = f.

The coefficient a is represented via RBF interpolation during training and
by the decoder G_theta_a(beta) during inversion.  Weak PDE residual is
evaluated with 2D Wendland test functions and JAX autodiff.
"""

import jax
import jax.numpy as jnp
import h5py
from jax import random, grad, jit, vmap, value_and_grad
from flax import linen as nn
import numpy as np
from functools import partial
from typing import Dict, List, Literal, Tuple, Any, Optional
from pathlib import Path

from src.components.encoder import EncoderCNNet2dTanh
from src.components.nf import RealNVP
from src.problems import ProblemInstance, register_problem
from src.utils.GenPoints import Point2D
from src.utils.TestFun_ParticleWNN import TestFun_ParticleWNN
from src.utils.misc_utils import np2jax
from src.utils.RBFInterpolatorMesh import RBFInterpolator
from src.utils.solver_utils import get_model


# Pure functions for PDE computation (JIT-compilable)

def mollifier(u: jax.Array, x: jax.Array) -> jax.Array:
    """Apply mollifier: u * sin(πx) * sin(πy), returning shape (..., 1)."""
    pi = jnp.pi
    result = u * jnp.sin(pi * x[..., 0]) * jnp.sin(pi * x[..., 1])
    return result[..., None]


def mollifier_no_expand(u: jax.Array, x: jax.Array) -> jax.Array:
    """Apply mollifier without dimension expansion (for internal gradient computation)."""
    pi = jnp.pi
    return u * jnp.sin(pi * x[..., 0]) * jnp.sin(pi * x[..., 1])


def compute_u_and_grad(
        params_u: Any,
        model_u: nn.Module,
        x: jax.Array,
        beta: jax.Array
) -> Tuple[jax.Array, jax.Array]:
    """Compute mollified u and grad(u) w.r.t. x per-point via vmap.

    Returns u (n_points,) and du_dx (n_points, 2).
    """

    def u_at_point(x_single):
        x_batch = x_single[None, None, :]  # (1, 1, 2)
        beta_batch = beta[None, :]
        u_val = model_u.apply({'params': params_u}, x_batch, beta_batch)
        u_val = u_val[0, 0]
        u_val = u_val * jnp.sin(jnp.pi * x_single[0]) * jnp.sin(jnp.pi * x_single[1])
        return u_val

    def value_and_grad_at_point(x_single):
        return jax.value_and_grad(u_at_point)(x_single)

    u_vals, du_vals = vmap(value_and_grad_at_point)(x)

    return u_vals, du_vals


def compute_pde_residual_single_sample(
        params_u: Any,
        model_u: nn.Module,
        beta: jax.Array,
        xc: jax.Array,
        R: jax.Array,
        int_grid: jax.Array,
        v: jax.Array,
        dv_dr: jax.Array,
        a_vals: jax.Array,
        n_grid: int,
) -> jax.Array:
    """Compute PDE residual for single sample.

    Weak formulation: ∫ a * grad(u) · grad(v) dx = ∫ f * v dx
    Returns unsquared residual (nc,); caller takes L2 norm.
    """
    nc = xc.shape[0]

    x = int_grid[None, :, :] * R + xc  # (nc, n_grid, 2)
    x_flat = x.reshape(-1, 2)

    # dv/dx = (1/R) * dv/dr
    dv = (dv_dr[None, :, :] / R).reshape(-1, 2)
    v_flat = jnp.tile(v, (nc, 1, 1)).reshape(-1, 1)

    def compute_for_center(center_idx):
        x_center = x[center_idx]
        return compute_u_and_grad(params_u, model_u, x_center, beta)
    u_all, du_all = vmap(compute_for_center)(jnp.arange(nc))

    du_flat = du_all.reshape(-1, 2)  # (nc*n_grid, 2)

    left = jnp.sum(a_vals * du_flat * dv, axis=-1)
    left = left.reshape(nc, n_grid).mean(axis=-1)

    right = (10.0 * v_flat).reshape(nc, n_grid).mean(axis=-1)

    return left - right


@register_problem("darcy_continuous")
class DarcyContinuous(ProblemInstance):
    BETA_SIZE = 6
    HIDDEN_SIZE = 100
    NF_NUM_FLOWS = 2
    NF_HIDDEN_DIM = 32
    NF_ALPHA = 8.0

    def __init__(
            self,
            seed: int,
            dtype: jnp.dtype = jnp.float32,
            train_data_path: str = None,
            test_data_path: str = None,
    ):
        super().__init__(
            seed=seed,
            dtype=dtype,
            train_data_path=train_data_path,
            test_data_path=test_data_path,
        )

        print("Loading data...")
        if self.train_data_path:
            self.train_data, self.gridx_train = self._load_data(self.train_data_path)
            print(f"  Train: a={self.train_data['a'].shape}, u={self.train_data['u'].shape}")

            # RBF interpolator for a during PDE evaluation
            self.fun_a = RBFInterpolator(
                x_mesh=self.gridx_train,
                kernel='gaussian',
                eps=25.,
                smoothing=0.,
                degree=6,
            )

        if self.test_data_path:
            self.test_data, self.gridx_test = self._load_data(self.test_data_path)
            print(f"  Test: a={self.test_data['a'].shape}, u={self.test_data['u'].shape}")

        print("Setting up grids and test functions...")

        self.genPoint = Point2D(
            x_lb=[0., 0.],
            x_ub=[1., 1.],
            random_seed=self.seed
        )

        int_grid, v, dv_dr = TestFun_ParticleWNN(
            fun_type='Wendland',
            dim=2,
            n_mesh_or_grid=9,
        ).get_testFun()

        self.int_grid = int_grid
        self.v = v
        self.dv_dr = dv_dr
        self.n_grid = int_grid.shape[0]

        print(f"  int_grid: {self.int_grid.shape}, v: {self.v.shape}")

        print("Building models...")
        self.models = self._build_models()

    def _load_data(self, path: str) -> Tuple[Dict, jax.Array]:
        """Load coefficient and solution fields from HDF5.

        For continuous Darcy, data contains:
        - coeff: Continuous coefficient field (N, 29, 29)
        - sol: Solution field (N, 29, 29)
        - X, Y: Mesh grids
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Data path not found: {path}")

        data = h5py.File(path, mode='r')

        a = np2jax(np.array(data["coeff"]).T, self.dtype)
        u = np2jax(np.array(data["sol"]).T, self.dtype)

        X, Y = np.array(data['X']).T, np.array(data['Y']).T
        mesh = np2jax(np.vstack([X.ravel(), Y.ravel()]).T, self.dtype)
        gridx = mesh.reshape(-1, 2)

        ndata = a.shape[0]
        a = a.reshape(ndata, -1, 1)
        x = jnp.tile(gridx[None, :, :], (ndata, 1, 1))
        u = u.reshape(ndata, -1, 1)

        return {'a': a, 'u': u, 'x': x}, gridx

    def _build_models(self) -> Dict[str, nn.Module]:
        net_type = 'MultiONetBatch'

        conv_arch = [1, 64, 64, 64]
        fc_arch = [64 * 2 * 2, 128, 64, self.BETA_SIZE]
        model_enc = EncoderCNNet2dTanh(
            conv_arch=conv_arch,
            fc_arch=fc_arch,
            activation_conv='SiLU',
            activation_fc='SiLU',
            nx_size=29,
            ny_size=29,
            kernel_size=(3, 3),
            stride=2,
            dtype=self.dtype
        )

        trunk_layers = [self.HIDDEN_SIZE] * 6
        branch_layers = [self.HIDDEN_SIZE] * 6

        model_a = get_model(
            x_in_size=2,
            beta_in_size=self.BETA_SIZE,
            trunk_layers=trunk_layers,
            branch_layers=branch_layers,
            activation_trunk='Tanh_Sin',
            activation_branch='Tanh_Sin',
            net_type=net_type,
            sum_layers=5,
            dtype=self.dtype
        )

        model_u = get_model(
            x_in_size=2,
            beta_in_size=self.BETA_SIZE,
            trunk_layers=trunk_layers,
            branch_layers=branch_layers,
            activation_trunk='Tanh_Sin',
            activation_branch='Tanh_Sin',
            net_type=net_type,
            sum_layers=5,
            dtype=self.dtype
        )

        model_nf = RealNVP(
            dim=self.BETA_SIZE,
            num_flows=self.NF_NUM_FLOWS,
            hidden_dim=self.NF_HIDDEN_DIM,
            alpha=self.NF_ALPHA,
        )

        return {
            'enc': model_enc,
            'u': model_u,
            'a': model_a,
            'nf': model_nf,
        }

    def get_sample_inputs(self, batch_size: int) -> Dict[str, Dict[str, jax.Array]]:
        if self.train_data:
            sample_a = self.train_data['a'][:batch_size]
            sample_x = self.train_data['x'][:batch_size]
        else:
            sample_a = self.test_data['a'][:batch_size]
            sample_x = self.test_data['x'][:batch_size]

        sample_beta = jnp.ones((batch_size, self.BETA_SIZE), dtype=self.dtype)

        return {
            'enc': {'x': sample_a},
            'u': {'x': sample_x, 'a': sample_beta},
            'a': {'x': sample_x, 'a': sample_beta},
            'nf': {'x': sample_beta},
        }

    def get_weight_decay_groups(self) -> Dict[str, bool]:
        return {
            'enc': False,
            'u': True,
            'a': True,
            'nf': False,
        }

    def loss_pde(
            self,
            params: Dict[str, Any],
            a: jnp.ndarray,
            rng: jax.Array
    ) -> jnp.ndarray:
        """PDE residual loss during training. Uses RBF interpolation on the true coefficient field."""
        nc = 100
        n_batch = a.shape[0]

        beta = self.models['enc'].apply({'params': params['enc']}, a)

        rng, subkey = random.split(rng)
        xc, R = self.genPoint.weight_centers(n_center=nc, R_max=1e-4, R_min=1e-4, key=subkey)

        x = self.int_grid[None, :, :] * R + xc  # (nc, n_grid, 2)
        x_flat = jnp.tile(x.reshape(-1, 2)[None, :, :], (n_batch, 1, 1))
        a_vals = self.fun_a(x_flat, a)  # (n_batch, nc*n_grid, 1)

        def compute_residual_for_sample(beta_single, a_single):
            return compute_pde_residual_single_sample(
                params['u'], self.models['u'], beta_single,
                xc, R, self.int_grid, self.v, self.dv_dr, a_single, self.n_grid,
            )

        residuals = vmap(compute_residual_for_sample)(beta, a_vals)  # (n_batch, nc)
        residuals_squared = residuals ** 2

        loss_per_sample = jnp.linalg.norm(residuals, axis=1)  # (n_batch,)
        mse_loss = jnp.mean(loss_per_sample)

        residuals_flat = residuals_squared.reshape(-1)
        top_k_vals, _ = jax.lax.top_k(residuals_flat, nc * 10)
        top_k_loss = jnp.sum(top_k_vals)

        return mse_loss + top_k_loss

    def loss_data(
            self,
            params: Dict[str, Any],
            x: jnp.ndarray,
            a: jnp.ndarray,
            u: jnp.ndarray
    ) -> jnp.ndarray:
        beta = self.models['enc'].apply({'params': params['enc']}, a)
        a_pred = self.models['a'].apply({'params': params['a']}, x, beta)
        a_target = a.squeeze(-1) if a.ndim == 3 else a
        return self.get_loss(a_pred, a_target)

    def compute_training_losses(
            self,
            params: Dict[str, Any],
            batch: Dict[str, jax.Array],
            rng: jax.Array,
            loss_weights: Any
    ) -> Tuple[jax.Array, Dict[str, jax.Array]]:
        """Training losses with beta encoded once and reused."""
        a = batch['a']
        x = batch['x']
        n_batch = a.shape[0]

        beta = self.models['enc'].apply({'params': params['enc']}, a)

        # PDE loss (inlined from loss_pde, reusing beta)
        nc = 100
        rng, subkey = random.split(rng)
        xc, R = self.genPoint.weight_centers(n_center=nc, R_max=1e-4, R_min=1e-4, key=subkey)
        x_int = self.int_grid[None, :, :] * R + xc
        x_flat = jnp.tile(x_int.reshape(-1, 2)[None, :, :], (n_batch, 1, 1))
        a_vals = self.fun_a(x_flat, a)

        def compute_residual_for_sample(beta_single, a_single):
            return compute_pde_residual_single_sample(
                params['u'], self.models['u'], beta_single,
                xc, R, self.int_grid, self.v, self.dv_dr, a_single, self.n_grid,
            )

        residuals = vmap(compute_residual_for_sample)(beta, a_vals)
        residuals_squared = residuals ** 2
        residuals_flat = residuals_squared.reshape(-1)
        top_k_vals, _ = jax.lax.top_k(residuals_flat, nc * 10)
        loss_pde = jnp.mean(jnp.linalg.norm(residuals, axis=1)) + jnp.sum(top_k_vals)

        # Data loss (inlined from loss_data, reusing beta)
        a_pred = self.models['a'].apply({'params': params['a']}, x, beta)
        a_target = a.squeeze(-1) if a.ndim == 3 else a
        loss_data = self.get_loss(a_pred, a_target)

        # NF loss on stop_gradient(beta)
        loss_nf = self.models['nf'].apply(
            {'params': params['nf']},
            jax.lax.stop_gradient(beta),
            method=self.models['nf'].loss
        )

        total_loss = (
            loss_weights.pde * loss_pde +
            loss_weights.data * loss_data +
            loss_weights.nf * loss_nf
        )

        metrics = {
            'loss': total_loss,
            'loss_pde': loss_pde,
            'loss_data': loss_data,
            'loss_nf': loss_nf,
        }

        return total_loss, metrics

    def error(
            self,
            params: Dict[str, Any],
            x: jnp.ndarray,
            a: jnp.ndarray,
            u: jnp.ndarray
    ) -> jnp.ndarray:
        beta = self.models['enc'].apply({'params': params['enc']}, a)
        u_pred = self.models['u'].apply({'params': params['u']}, x, beta)
        u_pred = u_pred[..., None] if u_pred.ndim == 2 else u_pred
        u_pred = mollifier(u_pred.squeeze(-1), x)
        return self.get_error(u_pred, u)

    # From-beta methods (for inversion)

    def loss_pde_from_beta(
            self,
            params: Dict[str, Any],
            beta: jnp.ndarray,
            rng: jax.Array
    ) -> jnp.ndarray:
        """PDE loss from beta. Uses the decoded coefficient field (for inversion)."""
        nc = 100
        n_batch = beta.shape[0]

        # Fall back to training grid if no inversion grid set
        int_grid = self.inv_int_grid if self.inv_int_grid is not None else self.int_grid
        v = self.inv_v if self.inv_v is not None else self.v
        dv_dr = self.inv_dv_dr if self.inv_dv_dr is not None else self.dv_dr
        n_grid = self.inv_n_grid if self.inv_n_grid is not None else self.n_grid

        rng, subkey = random.split(rng)
        xc, R = self.genPoint.weight_centers(n_center=nc, R_max=1e-4, R_min=1e-4, key=subkey)

        x = int_grid[None, :, :] * R + xc  # (nc, n_grid, 2)
        x_flat = jnp.tile(x.reshape(-1, 2)[None, :, :], (n_batch, 1, 1))

        a_decoded = self.models['a'].apply({'params': params['a']}, x_flat, beta)
        a_decoded = a_decoded[..., None]  # (n_batch, nc*n_grid, 1)

        def compute_residual_for_sample(beta_single, a_single):
            return compute_pde_residual_single_sample(
                params['u'], self.models['u'], beta_single,
                xc, R, int_grid, v, dv_dr, a_single, n_grid,
            )

        residuals = vmap(compute_residual_for_sample)(beta, a_decoded)  # (n_batch, nc)
        loss_per_sample = jnp.linalg.norm(residuals, axis=1)  # L2 norm, matching PyTorch
        return jnp.mean(loss_per_sample)

    def loss_data_from_beta(
            self,
            params: Dict[str, Any],
            beta: jnp.ndarray,
            x: jnp.ndarray,
            target: jnp.ndarray,
            target_type: Literal['a', 'u']
    ) -> jnp.ndarray:
        """Data loss from beta (for inversion). Supports 'a' (MSE) and 'u' (relative L2)."""
        if target_type == 'a':
            pred = self.models['a'].apply({'params': params['a']}, x, beta)
            target_flat = target.squeeze(-1) if target.ndim == 3 else target
            return self.get_loss(pred, target_flat)

        elif target_type == 'u':
            pred = self.models['u'].apply({'params': params['u']}, x, beta)
            pred = pred[..., None] if pred.ndim == 2 else pred
            pred = mollifier(pred.squeeze(-1), x)
            # Relative loss per sample
            loss_per_sample = jnp.linalg.norm(pred - target, axis=1) / jnp.linalg.norm(target, axis=1)
            return jnp.mean(loss_per_sample)

        else:
            raise ValueError(f"Unknown target_type: {target_type}")

    def error_from_beta(
            self,
            params: Dict[str, Any],
            beta: jnp.ndarray,
            x: jnp.ndarray,
            target: jnp.ndarray,
            target_type: Literal['a', 'u']
    ) -> jnp.ndarray:
        if target_type == 'u':
            pred = self.models['u'].apply({'params': params['u']}, x, beta)
            pred = pred[..., None] if pred.ndim == 2 else pred
            pred = mollifier(pred.squeeze(-1), x)

        elif target_type == 'a':
            pred = self.models['a'].apply({'params': params['a']}, x, beta)
            if target.ndim == 3 and target.shape[-1] == 1:
                target = target.squeeze(-1)
            pred = pred[..., None] if pred.ndim == 2 else pred

        else:
            raise ValueError(f"Unknown target_type: {target_type}")

        return self.get_error(pred, target)

    def predict_from_beta(
            self,
            params: Dict[str, Any],
            beta: jnp.ndarray,
            x: jnp.ndarray
    ) -> Dict[str, jnp.ndarray]:
        """Decode beta to u and a predictions at coordinates x."""
        u_pred = self.models['u'].apply({'params': params['u']}, x, beta)
        u_pred = u_pred[..., None] if u_pred.ndim == 2 else u_pred
        u_pred = mollifier(u_pred.squeeze(-1), x)

        a_pred = self.models['a'].apply({'params': params['a']}, x, beta)
        a_pred = a_pred[..., None]

        return {
            'u_pred': u_pred,
            'a_pred': a_pred,
        }

    def prepare_observations(
            self,
            sample_indices: List[int],
            obs_indices: jnp.ndarray,
            snr_db: float = None,
            rng: jax.Array = None
    ) -> Dict[str, jnp.ndarray]:
        indices = jnp.array(sample_indices)
        a_true = self.test_data['a'][indices]
        u_true = self.test_data['u'][indices]
        x_full = self.test_data['x'][indices]

        x_obs = x_full[:, obs_indices, :]
        u_obs = u_true[:, obs_indices, :]

        if snr_db is not None and rng is not None:
            u_obs = self.add_noise_snr(u_obs, snr_db, rng)

        return {
            'x_full': x_full,
            'x_obs': x_obs,
            'u_obs': u_obs,
            'u_true': u_true,
            'a_true': a_true,
        }

    def get_n_test_samples(self) -> int:
        return len(self.test_data['a'])

    def get_n_points(self) -> int:
        return self.test_data['x'].shape[1]
