"""1D viscous Burgers equation: du/dt + u*du/dx = (0.1/pi) * d^2u/dx^2.

Domain x in [-1,1], t in [0,1].  Homogeneous Dirichlet BCs enforced by a
cos(pi*x/2) mollifier.  The inverse problem recovers the initial condition
a(x) = u(x,0) from sparse observations along the trajectory.
"""

import jax
import jax.numpy as jnp
import h5py
from jax import random, vmap
from flax import linen as nn
import numpy as np
from typing import Dict, List, Literal, Tuple, Any
from pathlib import Path

from src.components.encoder import EncoderFCNetTanh
from src.components.nf import RealNVP
from src.problems import ProblemInstance, register_problem
from src.utils.GenPoints import Point1DTime
from src.utils.TestFun_ParticleWNN import TestFun_ParticleWNN
from src.utils.misc_utils import np2jax
from src.utils.solver_utils import get_model


# Pure functions for Burgers PDE computation (JIT-compilable)

def mollifier_burgers(u: jax.Array, xt: jax.Array) -> jax.Array:
    """Apply Burgers mollifier: u * sin(πx/2 + π/2), vanishes at x=±1. Returns (..., 1)."""
    x = xt[..., 0]
    result = u * jnp.sin(jnp.pi * x / 2. + jnp.pi / 2.)
    return result[..., None]


def compute_u_and_grad_burgers(
        params_u: Any,
        model_u: nn.Module,
        xt: jax.Array,
        beta: jax.Array
) -> Tuple[jax.Array, jax.Array]:
    """Compute mollified u and grad(u) w.r.t. (x,t) per-point via vmap.

    Mollifier is applied inside autodiff so gradients are correct.
    Returns u (n_points,) and du_dxt (n_points, 2) with columns [du/dx, du/dt].
    """

    def u_at_point(xt_single):
        xt_batch = xt_single[None, None, :]  # (1, 1, 2)
        beta_batch = beta[None, :]
        u_val = model_u.apply({'params': params_u}, xt_batch, beta_batch)
        u_val = u_val[0, 0]
        u_val = u_val * jnp.sin(jnp.pi * xt_single[0] / 2. + jnp.pi / 2.)
        return u_val

    def value_and_grad_at_point(xt_single):
        return jax.value_and_grad(u_at_point)(xt_single)

    u_vals, du_vals = vmap(value_and_grad_at_point)(xt)

    return u_vals, du_vals


def compute_pde_residual_single_sample_burgers(
        params_u: Any,
        model_u: nn.Module,
        beta: jax.Array,
        xc: jax.Array,
        tc: jax.Array,
        R: jax.Array,
        int_grid: jax.Array,
        v: jax.Array,
        dv_dr: jax.Array,
        n_grid: int,
        lamda: float,
) -> jax.Array:
    """Compute Burgers PDE residual for a single sample using weak formulation.

    Weak form: ∫(du/dt * v + u * du/dx * v + λ * du/dx * dv/dx) = 0
    Returns residual per center (nc,).
    """
    nc = xc.shape[0]

    x_points = int_grid[None, :, :] * R + xc  # (nc, n_grid, 1)
    t_points = jnp.broadcast_to(tc, (nc, n_grid, 1))
    xt_points = jnp.concatenate([x_points, t_points], axis=-1)  # (nc, n_grid, 2)

    # dv/dx = (1/R) * dv/dr
    dv_physical = dv_dr[None, :, :] / R  # (nc, n_grid, 1)

    def compute_for_center(center_idx):
        xt = xt_points[center_idx]
        return compute_u_and_grad_burgers(params_u, model_u, xt, beta)

    u_all, du_all = vmap(compute_for_center)(jnp.arange(nc))

    dux = du_all[..., 0]  # du/dx
    dut = du_all[..., 1]  # du/dt

    v_flat = jnp.broadcast_to(v[None, :, 0], (nc, n_grid))
    dv_flat = dv_physical[..., 0]

    res = dut * v_flat + u_all * dux * v_flat + lamda * dux * dv_flat
    return jnp.mean(res, axis=-1)


@register_problem("burgers")
class Burgers(ProblemInstance):

    BETA_SIZE = 16
    HIDDEN_SIZE = 100
    NF_NUM_FLOWS = 3
    NF_HIDDEN_DIM = 56
    NF_ALPHA = 10.0
    LAMDA = 0.1 / np.pi  # viscosity coefficient

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
            self.train_data, self.gridxt_train, self.xt_init_train, self.x_mesh, self.t_mesh = \
                self._load_data(self.train_data_path)
            print(f"  Train: a={self.train_data['a'].shape}, u={self.train_data['u'].shape}")

        if self.test_data_path:
            self.test_data, self.gridxt_test, self.xt_init_test, self.x_mesh, self.t_mesh = \
                self._load_data(self.test_data_path)
            print(f"  Test: a={self.test_data['a'].shape}, u={self.test_data['u'].shape}")

        # Use whichever grid is available
        self.gridxt = self.gridxt_train if self.train_data_path else self.gridxt_test
        self.xt_init = self.xt_init_train if self.train_data_path else self.xt_init_test

        self.n_mesh = int(self.x_mesh.shape[0])
        self.n_time = int(self.t_mesh.shape[0])
        print("Setting up grids and test functions...")

        self.genPoint = Point1DTime(
            x_lb=-1., x_ub=1.,
            t_lb=0., t_ub=1.,
            random_seed=self.seed,
        )

        # 1D Wendland test functions (dim=1, n_mesh_or_grid=10, matching reference)
        int_grid, v, dv_dr = TestFun_ParticleWNN(
            fun_type='Wendland',
            dim=1,
            n_mesh_or_grid=10,
        ).get_testFun()

        self.int_grid = int_grid  # (n_grid, 1)
        self.v = v                # (n_grid, 1)
        self.dv_dr = dv_dr        # (n_grid, 1)
        self.n_grid = int_grid.shape[0]

        print(f"  int_grid: {self.int_grid.shape}, v: {self.v.shape}")

        print("Building models...")
        self.models = self._build_models()

    def _load_data(self, path: str) -> Tuple[Dict, jax.Array, jax.Array, np.ndarray, np.ndarray]:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Data path not found: {path}")

        data = h5py.File(path, mode='r')

        # Initial conditions: (n_mesh, n_samples) -> (n_samples, n_mesh)
        a = np.array(data["u0"]).T

        # Solution: (n_mesh, n_time, n_samples) -> (n_samples, n_time, n_mesh) -> (n_samples, n_time*n_mesh)
        u_sol = np.array(data["u_sol"]).T  # reverses all axes: (n_samples, n_time, n_mesh)
        ndata = u_sol.shape[0]

        x_mesh = np.array(data['x_mesh']).flatten()  # (n_mesh,)
        t_mesh = np.array(data['t_mesh']).flatten()  # (n_time,)

        data.close()

        # Build space-time meshgrid: X varies fast (columns), T varies slow (rows)
        X, T = np.meshgrid(x_mesh, t_mesh)
        gridxt = np2jax(np.stack([X.ravel(), T.ravel()], axis=1), self.dtype)  # (n_mesh*n_time, 2)

        # Initial condition points: spatial mesh at t=0
        xt_init = np2jax(
            np.stack([x_mesh, np.zeros_like(x_mesh)], axis=1),
            self.dtype
        )  # (n_mesh, 2)

        a = np2jax(a.reshape(ndata, -1, 1), self.dtype)     # (N, n_mesh, 1)
        u = np2jax(u_sol.reshape(ndata, -1, 1), self.dtype)  # (N, n_mesh*n_time, 1)
        x = np2jax(
            np.tile(x_mesh[None, :, None], (ndata, 1, 1)),
            self.dtype
        )  # (N, n_mesh, 1)

        return {'a': a, 'u': u, 'x': x}, gridxt, xt_init, x_mesh, t_mesh

    def _build_models(self) -> Dict[str, nn.Module]:
        net_type = 'MultiONetBatch'

        layers_enc = [self.n_mesh, 128, 64, self.BETA_SIZE]
        model_enc = EncoderFCNetTanh(
            layers_list=layers_enc,
            activation='ELU',
            dtype=self.dtype,
        )

        # Solution decoder: MultiONet with (x,t) input
        trunk_layers = [self.HIDDEN_SIZE] * 6
        branch_layers = [self.HIDDEN_SIZE] * 6

        model_u = get_model(
            x_in_size=2,  # (x, t)
            beta_in_size=self.BETA_SIZE,
            trunk_layers=trunk_layers,
            branch_layers=branch_layers,
            activation_trunk='Tanh_Sin',
            activation_branch='Tanh_Sin',
            net_type=net_type,
            sum_layers=5,
            dtype=self.dtype,
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
            'nf': model_nf,
        }

    def get_sample_inputs(self, batch_size: int) -> Dict[str, Dict[str, jax.Array]]:
        if self.train_data:
            sample_a = self.train_data['a'][:batch_size]
        else:
            sample_a = self.test_data['a'][:batch_size]

        sample_beta = jnp.ones((batch_size, self.BETA_SIZE), dtype=self.dtype)
        sample_xt = jnp.tile(self.gridxt[None, :, :], (batch_size, 1, 1))

        return {
            'enc': {'x': sample_a},
            'u': {'x': sample_xt, 'a': sample_beta},
            'nf': {'x': sample_beta},
        }

    def get_weight_decay_groups(self) -> Dict[str, bool]:
        return {
            'enc': True,
            'u': True,
            'nf': False,
        }

    def loss_pde(
            self,
            params: Dict[str, Any],
            a: jnp.ndarray,
            rng: jax.Array
    ) -> jnp.ndarray:
        """PDE residual loss for Burgers equation (weak formulation)."""
        nc = 100
        n_batch = a.shape[0]

        beta = self.models['enc'].apply({'params': params['enc']}, a)

        rng, subkey = random.split(rng)
        xc, tc, R = self.genPoint.weight_centers(n_center=nc, R_max=1e-4, R_min=1e-4, key=subkey)

        def compute_residual_for_sample(beta_single):
            return compute_pde_residual_single_sample_burgers(
                params['u'], self.models['u'], beta_single,
                xc, tc, R, self.int_grid, self.v, self.dv_dr, self.n_grid, self.LAMDA,
            )

        residuals = vmap(compute_residual_for_sample)(beta)  # (n_batch, nc)
        residuals_squared = residuals ** 2

        loss_per_sample = jnp.linalg.norm(residuals, axis=1)
        mse_loss = jnp.mean(loss_per_sample)

        residuals_flat = residuals_squared.reshape(-1)
        top_k_vals, _ = jax.lax.top_k(residuals_flat, min(nc * 25, residuals_flat.shape[0]))
        top_k_loss = jnp.sum(top_k_vals)

        return mse_loss + top_k_loss

    def loss_data(
            self,
            params: Dict[str, Any],
            x: jnp.ndarray,
            a: jnp.ndarray,
            u: jnp.ndarray
    ) -> jnp.ndarray:
        """Data loss: initial condition matching u(x, t=0, beta) ≈ a(x)."""
        n_batch = a.shape[0]
        beta = self.models['enc'].apply({'params': params['enc']}, a)

        xt = jnp.tile(self.xt_init[None, :, :], (n_batch, 1, 1))  # (n_batch, n_mesh, 2)
        u_init = self.models['u'].apply({'params': params['u']}, xt, beta)
        u_init = mollifier_burgers(u_init, xt)  # (n_batch, n_mesh, 1)

        a_target = a.squeeze(-1) if a.ndim == 3 else a  # (n_batch, n_mesh)
        return self.get_loss(u_init.squeeze(-1), a_target)

    def compute_training_losses(
            self,
            params: Dict[str, Any],
            batch: Dict[str, jax.Array],
            rng: jax.Array,
            loss_weights: Any
    ) -> Tuple[jax.Array, Dict[str, jax.Array]]:
        """Training losses with beta encoded once and reused."""
        a = batch['a']
        n_batch = a.shape[0]

        beta = self.models['enc'].apply({'params': params['enc']}, a)

        # PDE loss
        nc = 100
        rng, subkey = random.split(rng)
        xc, tc, R = self.genPoint.weight_centers(n_center=nc, R_max=1e-4, R_min=1e-4, key=subkey)

        def compute_residual_for_sample(beta_single):
            return compute_pde_residual_single_sample_burgers(
                params['u'], self.models['u'], beta_single,
                xc, tc, R, self.int_grid, self.v, self.dv_dr, self.n_grid, self.LAMDA,
            )

        residuals = vmap(compute_residual_for_sample)(beta)
        residuals_squared = residuals ** 2
        residuals_flat = residuals_squared.reshape(-1)
        top_k_vals, _ = jax.lax.top_k(residuals_flat, min(nc * 25, residuals_flat.shape[0]))
        loss_pde = jnp.mean(jnp.linalg.norm(residuals, axis=1)) + jnp.sum(top_k_vals)

        xt = jnp.tile(self.xt_init[None, :, :], (n_batch, 1, 1))
        u_init = self.models['u'].apply({'params': params['u']}, xt, beta)
        u_init = mollifier_burgers(u_init, xt)
        a_target = a.squeeze(-1) if a.ndim == 3 else a
        loss_data = self.get_loss(u_init.squeeze(-1), a_target)

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
        """Compute solution error at full space-time grid."""
        n_batch = a.shape[0]
        beta = self.models['enc'].apply({'params': params['enc']}, a)

        xt = jnp.tile(self.gridxt[None, :, :], (n_batch, 1, 1))  # (n_batch, n_mesh*n_time, 2)
        u_pred = self.models['u'].apply({'params': params['u']}, xt, beta)
        u_pred = mollifier_burgers(u_pred, xt)  # (n_batch, n_mesh*n_time, 1)

        return self.get_error(u_pred, u)

    def loss_pde_from_beta(
            self,
            params: Dict[str, Any],
            beta: jnp.ndarray,
            rng: jax.Array
    ) -> jnp.ndarray:
        nc = 100

        # Fall back to training grid if no inversion grid set
        int_grid = self.inv_int_grid if self.inv_int_grid is not None else self.int_grid
        v = self.inv_v if self.inv_v is not None else self.v
        dv_dr = self.inv_dv_dr if self.inv_dv_dr is not None else self.dv_dr
        n_grid = self.inv_n_grid if self.inv_n_grid is not None else self.n_grid

        rng, subkey = random.split(rng)
        xc, tc, R = self.genPoint.weight_centers(n_center=nc, R_max=1e-4, R_min=1e-4, key=subkey)

        def compute_residual_for_sample(beta_single):
            return compute_pde_residual_single_sample_burgers(
                params['u'], self.models['u'], beta_single,
                xc, tc, R, int_grid, v, dv_dr, n_grid, self.LAMDA,
            )

        residuals = vmap(compute_residual_for_sample)(beta)
        loss_per_sample = jnp.linalg.norm(residuals, axis=1)
        return jnp.mean(loss_per_sample)

    def loss_data_from_beta(
            self,
            params: Dict[str, Any],
            beta: jnp.ndarray,
            x: jnp.ndarray,
            target: jnp.ndarray,
            target_type: Literal['a', 'u']
    ) -> jnp.ndarray:
        """Data loss from beta directly.

        target_type='u': compare u predictions at given (x,t) coords to observed u
        target_type='a': compare u(x, t=0) to initial condition
        """
        n_batch = beta.shape[0]

        if target_type == 'u':
            pred = self.models['u'].apply({'params': params['u']}, x, beta)
            pred = mollifier_burgers(pred, x)
            # Relative loss per sample
            loss_per_sample = jnp.linalg.norm(pred - target, axis=1) / jnp.linalg.norm(target, axis=1)
            return jnp.mean(loss_per_sample)

        elif target_type == 'a':
            # Evaluate u at t=0 to get initial condition
            xt = jnp.tile(self.xt_init[None, :, :], (n_batch, 1, 1))
            u_init = self.models['u'].apply({'params': params['u']}, xt, beta)
            u_init = mollifier_burgers(u_init, xt)
            target_flat = target.squeeze(-1) if target.ndim == 3 else target
            return self.get_loss(u_init.squeeze(-1), target_flat)

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
        """Compute error from beta directly."""
        n_batch = beta.shape[0]

        if target_type == 'u':
            pred = self.models['u'].apply({'params': params['u']}, x, beta)
            pred = mollifier_burgers(pred, x)

        elif target_type == 'a':
            xt = jnp.tile(self.xt_init[None, :, :], (n_batch, 1, 1))
            pred = self.models['u'].apply({'params': params['u']}, xt, beta)
            pred = mollifier_burgers(pred, xt)
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
        """Predict u at given (x,t) coords and initial condition a = u(x, t=0)."""
        n_batch = beta.shape[0]

        u_pred = self.models['u'].apply({'params': params['u']}, x, beta)
        u_pred = mollifier_burgers(u_pred, x)

        xt_init = jnp.tile(self.xt_init[None, :, :], (n_batch, 1, 1))
        a_pred = self.models['u'].apply({'params': params['u']}, xt_init, beta)
        a_pred = mollifier_burgers(a_pred, xt_init)

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

        n_batch = len(sample_indices)
        x_full = jnp.tile(self.gridxt[None, :, :], (n_batch, 1, 1))

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

    def setup_inversion_grid(self, n_mesh_or_grid: int = 7) -> None:
        inv_int_grid, inv_v, inv_dv_dr = TestFun_ParticleWNN(
            fun_type='Wendland',
            dim=1,
            n_mesh_or_grid=n_mesh_or_grid,
        ).get_testFun()
        self.inv_int_grid = inv_int_grid
        self.inv_v = inv_v
        self.inv_dv_dr = inv_dv_dr
        self.inv_n_grid = inv_int_grid.shape[0]
        print(f"  Inversion grid: n_mesh_or_grid={n_mesh_or_grid}, "
              f"n_grid={self.inv_n_grid}")

    def get_n_test_samples(self) -> int:
        return len(self.test_data['a'])

    def get_n_points(self) -> int:
        """Number of space-time grid points per sample."""
        return self.gridxt.shape[0]
