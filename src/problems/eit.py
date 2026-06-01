"""Electrical impedance tomography (EIT): -div(a * grad(u)) = 0.

Analytic coefficient a(x,y) parameterised by 6 latent dims; 20 boundary
conditions encoded as one-hot vectors concatenated to the latent code
(beta = [beta_a; one_hot(g_l)]).  The normalizing flow operates only on
the 6-dim coefficient subspace.
"""

import jax
import jax.numpy as jnp
import h5py
from jax import random, grad, jit, vmap, value_and_grad, lax
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
from src.utils.solver_utils import get_model


# Pure functions for EIT-specific computations (JIT-compilable)

def one_hot_g_l(g_l: jax.Array) -> jax.Array:
    """One-hot encode boundary condition labels (batch, 1) → (batch, 20)."""
    # Convert from 1-indexed to 0-indexed
    g_l_idx = (g_l.squeeze(-1).astype(jnp.int32) - 1)
    return jax.nn.one_hot(g_l_idx, 20)


def compute_analytic_a(
        x: jax.Array,
        coe_val: jax.Array,
        coe_mask: jax.Array
) -> jax.Array:
    """Compute a(x,y) = Σ_k c_mask[k] * exp(c_val[k] * sin(kπx) * sin(kπy)).

    Handles both unbatched (n_points, 2) and batched (batch, n_points, 2) x.
    """
    pi = jnp.pi

    # Handle both batched and unbatched cases
    if x.ndim == 2:
        # Unbatched: x (n_points, 2), coe_val (4,), coe_mask (4,)
        x_coord = x[:, 0]  # (n_points,)
        y_coord = x[:, 1]  # (n_points,)

        a = jnp.zeros_like(x_coord)  # (n_points,)
        for k in range(1, 5):
            term = jnp.exp(
                coe_val[k-1] * jnp.sin(k * pi * x_coord) * jnp.sin(k * pi * y_coord)
            )
            a = a + coe_mask[k-1] * term

        return a[..., None]  # (n_points, 1)

    else:
        # Batched: x (batch, n_points, 2), coe_val (batch, 4), coe_mask (batch, 4)
        x_coord = x[..., 0]  # (batch, n_points)
        y_coord = x[..., 1]  # (batch, n_points)

        a = jnp.zeros_like(x_coord)  # (batch, n_points)
        for k in range(1, 5):
            term = jnp.exp(
                coe_val[:, k-1:k] * jnp.sin(k * pi * x_coord) * jnp.sin(k * pi * y_coord)
            )
            a = a + coe_mask[:, k-1:k] * term

        return a[..., None]  # (batch, n_points, 1)


def eit_boundary_function(x: jax.Array, g_l_scalar: float) -> jax.Array:
    """g(x, y) = cos(2π * (x*cos(θ) + y*sin(θ))), θ = π*l/20."""
    theta = jnp.pi * g_l_scalar / 20.0
    proj = x[..., 0] * jnp.cos(theta) + x[..., 1] * jnp.sin(theta)
    return jnp.cos(2.0 * jnp.pi * proj)


def mollifier_eit(u_raw: jax.Array, x: jax.Array, g_l: jax.Array) -> jax.Array:
    """Apply EIT mollifier: u = u_raw * sin(πx)*sin(πy) + g(x,y), returning (..., 1)."""
    pi = jnp.pi
    sin_term = jnp.sin(pi * x[..., 0]) * jnp.sin(pi * x[..., 1])

    # Handle batched vs unbatched g_l
    if jnp.ndim(g_l) == 2:
        # Batched: vmap directly over array slices (avoids index tracing overhead)
        def apply_boundary_direct(u_raw_i, x_i, sin_term_i, g_l_i):
            g_vals = eit_boundary_function(x_i, g_l_i[0])
            return u_raw_i * sin_term_i + g_vals

        result = vmap(apply_boundary_direct)(u_raw, x, sin_term, g_l)
    else:
        # Unbatched: g_l is scalar
        g_vals = eit_boundary_function(x, g_l)
        result = u_raw * sin_term + g_vals

    return result[..., None]


def compute_u_and_grad_eit(
        params_u: Any,
        model_u: nn.Module,
        x: jax.Array,
        beta: jax.Array,
        g_l_scalar: float
) -> Tuple[jax.Array, jax.Array]:
    """Compute mollified u and grad(u) for EIT per-point via vmap.

    g(x,y) depends on x, so the mollifier is applied inside the per-point autodiff.
    Returns u (n_points,) and du_dx (n_points, 2).
    """

    def u_at_point(x_single):
        x_batch = x_single[None, None, :]
        beta_batch = beta[None, :]

        u_val = model_u.apply({'params': params_u}, x_batch, beta_batch)
        u_val = u_val[0, 0]

        # Apply EIT mollifier
        pi = jnp.pi
        sin_term = jnp.sin(pi * x_single[0]) * jnp.sin(pi * x_single[1])
        g_val = eit_boundary_function(x_single, g_l_scalar)

        return u_val * sin_term + g_val

    def value_and_grad_at_point(x_single):
        return jax.value_and_grad(u_at_point)(x_single)

    u_vals, du_vals = vmap(value_and_grad_at_point)(x)
    return u_vals, du_vals


def compute_pde_residual_eit_single_sample(
        params_u: Any,
        model_u: nn.Module,
        beta_u: jax.Array,
        g_l_scalar: float,
        xc: jax.Array,
        R: jax.Array,
        int_grid: jax.Array,
        v: jax.Array,
        dv_dr: jax.Array,
        a_vals: jax.Array,
        n_grid: int
) -> jax.Array:
    """Compute PDE residual for single EIT sample.

    Weak formulation: ∫ a * grad(u) · grad(v) dx = 0 (no source term).
    Returns residual (nc,).
    """
    nc = xc.shape[0]

    x = int_grid[None, :, :] * R + xc  # (nc, n_grid, 2)
    x_flat = x.reshape(-1, 2)

    # dv/dx = (1/R) * dv/dr
    dv = (dv_dr[None, :, :] / R).reshape(-1, 2)

    def compute_for_center(center_idx):
        x_center = x[center_idx]
        u_vals, du_vals = compute_u_and_grad_eit(params_u, model_u, x_center, beta_u, g_l_scalar)
        return u_vals, du_vals

    u_all, du_all = vmap(compute_for_center)(jnp.arange(nc))
    du_flat = du_all.reshape(-1, 2)

    integrand = jnp.sum(a_vals * du_flat * dv, axis=-1)
    residual = integrand.reshape(nc, n_grid).mean(axis=-1)

    return residual


@register_problem("eit")
class EIT(ProblemInstance):
    BETA_SIZE_A = 6   # Latent dim for coefficient
    BETA_SIZE_G = 20  # One-hot dim for boundary condition
    BETA_SIZE_U = 26  # Total: 6 + 20
    BETA_SIZE = BETA_SIZE_A  # Alias for igno.py compatibility

    # Inversion target type (Neumann boundary flux observations)
    data_target_type = 'neumann'
    HIDDEN_SIZE = 100
    NF_NUM_FLOWS = 3
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

        # Current boundary condition (for inversion)
        self._current_g_l = None

        # Boundary geometry (set after loading test data)
        self._boundary_indices = None     # numpy int array (n_bd,)
        self._boundary_normals = None     # numpy float array (n_bd, 2)
        self._boundary_normals_jax = None  # cached JAX array (n_bd, 2)
        self._active_boundary_normals_jax = None  # subsampled normals for current inversion

        # Load data
        print("Loading data...")
        if self.train_data_path:
            self.train_data, self.gridx_train = self._load_data(self.train_data_path, is_train=True)
            print(f"  Train: a={self.train_data['a'].shape}, u={self.train_data['u'].shape}, g_l={self.train_data['g_l'].shape}")

        if self.test_data_path:
            self.test_data, self.gridx_test = self._load_data(self.test_data_path, is_train=False)
            print(f"  Test: a={self.test_data['a'].shape}, u={self.test_data.get('u', 'N/A')}")
            self._setup_boundary_points(np.array(self.gridx_test))

        # Setup grids & test functions
        print("Setting up grids and test functions...")

        self.genPoint = Point2D(
            x_lb=[0., 0.],
            x_ub=[1., 1.],
            random_seed=self.seed
        )

        int_grid, v, dv_dr = TestFun_ParticleWNN(
            fun_type='Wendland',
            dim=2,
            n_mesh_or_grid=7,
        ).get_testFun()

        self.int_grid = int_grid
        self.v = v
        self.dv_dr = dv_dr
        self.n_grid = int_grid.shape[0]

        print(f"  int_grid: {self.int_grid.shape}, v: {self.v.shape}")

        # Build models
        print("Building models...")
        self.models = self._build_models()

    def _load_data(self, path: str, is_train: bool) -> Tuple[Dict, jax.Array]:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Data path not found: {path}")

        data = h5py.File(path, mode='r')

        coe_val = np2jax(np.array(data["coe_val"]).T, self.dtype)  # (N, 4)
        coe_mask = np2jax(np.array(data["coe_mask"]).T, self.dtype)  # (N, 4)
        g_l = np2jax(np.array(data["g_l"]).T, self.dtype)  # (N, 1)

        # Try to load mesh
        if 'X' in data and 'Y' in data:
            X, Y = np.array(data['X']).T, np.array(data['Y']).T
            mesh = np2jax(np.vstack([X.ravel(), Y.ravel()]).T, self.dtype)
            gridx = mesh.reshape(-1, 2)
        else:
            # Generate default 32x32 grid
            print(f"  Generating default 32x32 grid")
            x_1d = np.linspace(0, 1, 32)
            y_1d = np.linspace(0, 1, 32)
            X, Y = np.meshgrid(x_1d, y_1d)
            mesh = np2jax(np.vstack([X.ravel(), Y.ravel()]).T, self.dtype)
            gridx = mesh.reshape(-1, 2)

        ndata = coe_val.shape[0]

        x_grid = jnp.tile(gridx[None, :, :], (ndata, 1, 1))  # (N, n_points, 2)
        a = compute_analytic_a(x_grid, coe_val, coe_mask)  # (N, n_points, 1)

        result_dict = {
            'a': a,
            'x': x_grid,
            'g_l': g_l,
            'coe_val': coe_val,
            'coe_mask': coe_mask,
        }

        # Load u_sol if available (test data)
        if 'u_sol' in data or 'sol' in data:
            u_key = 'u_sol' if 'u_sol' in data else 'sol'
            u = np2jax(np.array(data[u_key]).T, self.dtype)
            u = u.reshape(ndata, -1, 1)
            result_dict['u'] = u

        # Load gradient fields for Neumann boundary observations (test/inverse data only)
        if not is_train and 'dux_sol' in data:
            dux = np2jax(np.array(data['dux_sol']).T, self.dtype)
            dux = dux.reshape(ndata, -1, 1)
            result_dict['dux'] = dux

            duy = np2jax(np.array(data['duy_sol']).T, self.dtype)
            duy = duy.reshape(ndata, -1, 1)
            result_dict['duy'] = duy

        return result_dict, gridx

    def _setup_boundary_points(self, gridx_np: np.ndarray) -> None:
        """Extract boundary point indices and outward normals from the 32×32 grid.

        Orders counter-clockwise: left (x=0, +y) -> top (y=1, +x) -> right (x=1, -y) -> bottom (y=0, -x).
        Corners are assigned to the first face that claims them to avoid duplicates.

        Args:
            gridx_np: Grid coordinates (n_points, 2) as numpy array
        """
        tol = 1e-5  # float32 safe; linspace(0,1,32) step is ~0.032
        x = gridx_np[:, 0]
        y = gridx_np[:, 1]

        is_left = x < tol
        is_right = x > 1 - tol
        is_bottom = y < tol
        is_top = y > 1 - tol

        # Left: x=0, all y (includes corners at (0,0) and (0,1))
        left_mask = is_left
        left_idx = np.where(left_mask)[0][np.argsort(y[left_mask])]

        # Top: y=1, x>0 (excludes top-left corner already in left)
        top_mask = is_top & ~is_left
        top_idx = np.where(top_mask)[0][np.argsort(x[top_mask])]

        # Right: x=1, y<1 (excludes top-right corner already in top)
        right_mask = is_right & ~is_top
        right_idx = np.where(right_mask)[0][np.argsort(y[right_mask])[::-1]]

        # Bottom: y=0, 0<x<1 (excludes both bottom corners)
        bottom_mask = is_bottom & ~is_left & ~is_right
        bottom_idx = np.where(bottom_mask)[0][np.argsort(x[bottom_mask])[::-1]]

        all_idx = np.concatenate([left_idx, top_idx, right_idx, bottom_idx])

        n_left = len(left_idx)
        n_top = len(top_idx)
        n_right = len(right_idx)
        n_bottom = len(bottom_idx)

        normals = np.zeros((len(all_idx), 2))
        normals[:n_left] = [-1, 0]
        normals[n_left:n_left + n_top] = [0, 1]
        normals[n_left + n_top:n_left + n_top + n_right] = [1, 0]
        normals[n_left + n_top + n_right:] = [0, -1]

        self._boundary_indices = all_idx
        self._boundary_normals = normals
        self._boundary_normals_jax = jnp.array(normals, dtype=self.dtype)
        print(f"  Boundary points: {len(all_idx)} (left={n_left}, top={n_top}, right={n_right}, bottom={n_bottom})")

    def _build_models(self) -> Dict[str, nn.Module]:
        net_type = 'MultiONetBatch'

        # Encoder for coefficient (CNN)
        conv_arch = [1, 64, 64, 64, 64]
        fc_arch = [64, 32, self.BETA_SIZE_A]
        model_enc = EncoderCNNet2dTanh(
            conv_arch=conv_arch,
            fc_arch=fc_arch,
            activation_conv='SiLU',
            activation_fc='SiLU',
            nx_size=32,
            ny_size=32,
            kernel_size=(3, 3),
            stride=2,
            dtype=self.dtype
        )

        trunk_layers = [self.HIDDEN_SIZE] * 5
        branch_layers = [self.HIDDEN_SIZE] * 5

        model_u = get_model(
            x_in_size=2,
            beta_in_size=self.BETA_SIZE_U,
            trunk_layers=trunk_layers,
            branch_layers=branch_layers,
            activation_trunk='Tanh_Sin',
            activation_branch='Tanh_Sin',
            net_type=net_type,
            sum_layers=4,
            dtype=self.dtype
        )

        model_a = get_model(
            x_in_size=2,
            beta_in_size=self.BETA_SIZE_A,
            trunk_layers=trunk_layers,
            branch_layers=branch_layers,
            activation_trunk='Tanh_Sin',
            activation_branch='Tanh_Sin',
            net_type=net_type,
            sum_layers=4,
            dtype=self.dtype
        )

        model_nf = RealNVP(
            dim=self.BETA_SIZE_A,
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

        sample_beta_a = jnp.ones((batch_size, self.BETA_SIZE_A), dtype=self.dtype)
        sample_beta_u = jnp.ones((batch_size, self.BETA_SIZE_U), dtype=self.dtype)

        return {
            'enc': {'x': sample_a},
            'u': {'x': sample_x, 'a': sample_beta_u},
            'a': {'x': sample_x, 'a': sample_beta_a},
            'nf': {'x': sample_beta_a},
        }

    def get_weight_decay_groups(self) -> Dict[str, bool]:
        return {
            'enc': False,
            'u': True,
            'a': True,
            'nf': False,
        }

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
        g_l = batch['g_l']
        coe_val = batch['coe_val']
        coe_mask = batch['coe_mask']
        n_batch = a.shape[0]

        beta_a = self.models['enc'].apply({'params': params['enc']}, a)

        g_l_onehot = one_hot_g_l(g_l)  # (batch, 20)
        beta_u = jnp.concatenate([beta_a, g_l_onehot], axis=-1)  # (batch, 26)

        # PDE loss
        nc = 100
        rng, subkey = random.split(rng)
        xc, R = self.genPoint.weight_centers(
            n_center=nc,
            R_max=1e-4,
            R_min=1e-4,
            key=subkey
        )

        x_grid = self.int_grid[None, :, :] * R + xc
        x_flat = x_grid.reshape(-1, 2)

        def compute_residual_for_sample(beta_u_single, g_l_single, coe_val_single, coe_mask_single):
            a_vals = compute_analytic_a(x_flat, coe_val_single, coe_mask_single)
            a_vals = lax.stop_gradient(a_vals)

            return compute_pde_residual_eit_single_sample(
                params['u'],
                self.models['u'],
                beta_u_single,
                g_l_single[0],  # Scalar
                xc, R,
                self.int_grid, self.v, self.dv_dr,
                a_vals,
                self.n_grid
            )

        residuals = vmap(compute_residual_for_sample)(beta_u, g_l, coe_val, coe_mask)
        residuals_squared = residuals ** 2
        mse_loss = jnp.mean(residuals_squared)

        residuals_flat = residuals_squared.reshape(-1)
        top_k_vals, _ = jax.lax.top_k(residuals_flat, nc * 50)
        top_k_loss = jnp.mean(top_k_vals) * jnp.sqrt(nc)

        loss_pde = mse_loss + top_k_loss

        a_pred = self.models['a'].apply({'params': params['a']}, x, beta_a)
        loss_data = jnp.mean((a_pred - a.squeeze(-1)) ** 2)

        loss_nf = self.models['nf'].apply(
            {'params': params['nf']},
            lax.stop_gradient(beta_a),
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

    def compute_eval_metrics(
            self,
            params: Dict[str, Any],
            batch: Dict[str, jax.Array],
    ) -> Dict[str, jax.Array]:
        """Compute evaluation metrics: u relative L2 error (matching reference)."""
        a = batch['a']
        x = batch['x']
        g_l = batch['g_l']
        u = batch['u']

        beta_a = self.models['enc'].apply({'params': params['enc']}, a)

        g_l_onehot = one_hot_g_l(g_l)
        beta_u = jnp.concatenate([beta_a, g_l_onehot], axis=-1)

        u_pred = self.models['u'].apply({'params': params['u']}, x, beta_u)
        u_pred = mollifier_eit(u_pred, x, g_l)

        u_error = self.get_error(u_pred, u)

        nf_loss = self.models['nf'].apply(
            {'params': params['nf']},
            lax.stop_gradient(beta_a),
            method=self.models['nf'].loss
        )

        return {
            'error': jnp.mean(u_error),
            'nf_loss': nf_loss,
        }

    def loss_pde(
            self,
            params: Dict[str, Any],
            a: jnp.ndarray,
            rng: jax.Array
    ) -> jnp.ndarray:
        """Not used - compute_training_losses is overridden"""
        raise NotImplementedError("Use compute_training_losses instead")

    def loss_data(
            self,
            params: Dict[str, Any],
            x: jnp.ndarray,
            a: jnp.ndarray,
            u: jnp.ndarray
    ) -> jnp.ndarray:
        """Not used - compute_training_losses is overridden"""
        raise NotImplementedError("Use compute_training_losses instead")

    def error(
            self,
            params: Dict[str, Any],
            x: jnp.ndarray,
            a: jnp.ndarray,
            u: jnp.ndarray
    ) -> jnp.ndarray:
        """Not used - compute_eval_metrics is overridden"""
        raise NotImplementedError("Use compute_eval_metrics instead")

    def loss_pde_from_beta(
            self,
            params: Dict[str, Any],
            beta: jnp.ndarray,
            rng: jax.Array
    ) -> jnp.ndarray:
        nc = 100
        n_batch = beta.shape[0]

        # Combine beta_a with current g_l
        if self._current_g_l is None:
            raise RuntimeError("Must call prepare_observations first to set _current_g_l")

        g_l_onehot = one_hot_g_l(self._current_g_l)
        g_l_onehot = jnp.broadcast_to(g_l_onehot, (beta.shape[0], g_l_onehot.shape[-1]))
        beta_u = jnp.concatenate([beta, g_l_onehot], axis=-1)

        rng, subkey = random.split(rng)
        xc, R = self.genPoint.weight_centers(
            n_center=nc,
            R_max=1e-4,
            R_min=1e-4,
            key=subkey
        )

        # Fall back to training grid if no inversion grid set
        int_grid = self.inv_int_grid if self.inv_int_grid is not None else self.int_grid
        v = self.inv_v if self.inv_v is not None else self.v
        dv_dr = self.inv_dv_dr if self.inv_dv_dr is not None else self.dv_dr
        n_grid = self.inv_n_grid if self.inv_n_grid is not None else self.n_grid

        x_grid = int_grid[None, :, :] * R + xc
        x_flat = x_grid.reshape(-1, 2)
        x_batch = jnp.tile(x_flat[None, :, :], (n_batch, 1, 1))

        a_decoded = self.models['a'].apply({'params': params['a']}, x_batch, beta)
        a_decoded = a_decoded[..., None]

        def compute_residual_for_sample(beta_u_single, a_single, g_l_single):
            return compute_pde_residual_eit_single_sample(
                params['u'],
                self.models['u'],
                beta_u_single,
                g_l_single[0],
                xc, R,
                int_grid, v, dv_dr,
                a_single,
                n_grid
            )

        residuals = vmap(compute_residual_for_sample)(beta_u, a_decoded, self._current_g_l)
        loss_per_sample = jnp.linalg.norm(residuals, axis=1)
        return jnp.mean(loss_per_sample)

    def loss_data_from_beta(
            self,
            params: Dict[str, Any],
            beta: jnp.ndarray,
            x: jnp.ndarray,
            target: jnp.ndarray,
            target_type: Literal['a', 'u', 'neumann']
    ) -> jnp.ndarray:
        if target_type == 'a':
            pred = self.models['a'].apply({'params': params['a']}, x, beta)
            target_flat = target.squeeze(-1) if target.ndim == 3 else target
            return self.get_loss(pred, target_flat)

        elif target_type == 'u':
            if self._current_g_l is None:
                raise RuntimeError("Must call prepare_observations first to set _current_g_l")

            g_l_onehot = one_hot_g_l(self._current_g_l)
            g_l_onehot = jnp.broadcast_to(g_l_onehot, (beta.shape[0], g_l_onehot.shape[-1]))
            beta_u = jnp.concatenate([beta, g_l_onehot], axis=-1)

            pred = self.models['u'].apply({'params': params['u']}, x, beta_u)
            pred = pred[..., None] if pred.ndim == 2 else pred
            g_l_tiled = jnp.broadcast_to(self._current_g_l, (beta.shape[0], self._current_g_l.shape[-1]))
            pred = mollifier_eit(pred.squeeze(-1), x, g_l_tiled)

            # Relative loss per sample
            loss_per_sample = jnp.linalg.norm(pred - target, axis=1) / jnp.linalg.norm(target, axis=1)
            return jnp.mean(loss_per_sample)

        elif target_type == 'neumann':
            # Neumann boundary flux: a * (grad(u) · n)
            # x: (batch, n_bd, 2) — boundary coordinates
            # target: (batch, n_bd, 1) — true Neumann flux
            if self._current_g_l is None:
                raise RuntimeError("Must call prepare_observations first to set _current_g_l")
            if self._boundary_normals is None:
                raise RuntimeError("Boundary geometry not set up. Need test data with gradient fields.")

            # Use active (possibly subsampled) normals set by prepare_observations
            normals = self._active_boundary_normals_jax if self._active_boundary_normals_jax is not None else self._boundary_normals_jax  # (n_bd, 2)

            g_l_onehot = one_hot_g_l(self._current_g_l)
            g_l_onehot = jnp.broadcast_to(g_l_onehot, (beta.shape[0], g_l_onehot.shape[-1]))
            beta_u = jnp.concatenate([beta, g_l_onehot], axis=-1)  # (batch, 26)

            def compute_neumann_for_sample(beta_u_single, beta_a_single, x_single, g_l_single):
                g_l_scalar = g_l_single[0]
                # Compute grad(u) at boundary points
                _, du_vals = compute_u_and_grad_eit(
                    params['u'], self.models['u'], x_single, beta_u_single, g_l_scalar
                )
                # du_vals: (n_bd, 2)
                # Predict a at boundary points
                a_vals = self.models['a'].apply(
                    {'params': params['a']}, x_single[None, :, :], beta_a_single[None, :]
                )
                a_vals = a_vals[0]  # (n_bd,)
                obs_pred = a_vals * (du_vals[:, 0] * normals[:, 0] + du_vals[:, 1] * normals[:, 1])
                return obs_pred  # (n_bd,)

            obs_pred = vmap(compute_neumann_for_sample)(beta_u, beta, x, self._current_g_l)
            # obs_pred: (batch, n_bd)
            target_flat = target.squeeze(-1)  # (batch, n_bd)
            loss_per_sample = (
                jnp.linalg.norm(obs_pred - target_flat, axis=1) /
                (jnp.linalg.norm(target_flat, axis=1) + 1e-8)
            )
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
            if self._current_g_l is None:
                raise RuntimeError("Must call prepare_observations first to set _current_g_l")

            g_l_onehot = one_hot_g_l(self._current_g_l)
            g_l_onehot = jnp.broadcast_to(g_l_onehot, (beta.shape[0], g_l_onehot.shape[-1]))
            beta_u = jnp.concatenate([beta, g_l_onehot], axis=-1)

            pred = self.models['u'].apply({'params': params['u']}, x, beta_u)
            pred = pred[..., None] if pred.ndim == 2 else pred
            g_l_tiled = jnp.broadcast_to(self._current_g_l, (beta.shape[0], self._current_g_l.shape[-1]))
            pred = mollifier_eit(pred.squeeze(-1), x, g_l_tiled)

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
        if self._current_g_l is None:
            raise RuntimeError("Must call prepare_observations first to set _current_g_l")

        g_l_onehot = one_hot_g_l(self._current_g_l)
        g_l_onehot = jnp.broadcast_to(g_l_onehot, (beta.shape[0], g_l_onehot.shape[-1]))
        beta_u = jnp.concatenate([beta, g_l_onehot], axis=-1)

        u_pred = self.models['u'].apply({'params': params['u']}, x, beta_u)
        u_pred = u_pred[..., None] if u_pred.ndim == 2 else u_pred
        g_l_tiled = jnp.broadcast_to(self._current_g_l, (beta.shape[0], self._current_g_l.shape[-1]))
        u_pred = mollifier_eit(u_pred.squeeze(-1), x, g_l_tiled)

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
        """Prepare Neumann boundary flux observations for test samples.

        EIT observations are boundary flux: a * (grad(u) · n) at the 32×32 grid boundary.
        The obs_indices parameter is ignored (boundary geometry is fixed).
        Also sets self._current_g_l for inversion context.
        """
        indices = jnp.array(sample_indices)
        a_true = self.test_data['a'][indices]
        x_full = self.test_data['x'][indices]
        g_l = self.test_data['g_l'][indices]

        # Store current g_l for inversion methods
        self._current_g_l = g_l

        result = {
            'x_full': x_full,
            'a_true': a_true,
            'g_l': g_l,
        }

        # Load full u solution for evaluation metrics
        if 'u' in self.test_data:
            result['u_true'] = self.test_data['u'][indices]

        # Compute Neumann boundary flux observations from stored gradient fields
        if 'dux' in self.test_data and self._boundary_indices is not None:
            dux_true = self.test_data['dux'][indices]
            duy_true = self.test_data['duy'][indices]

            bd_idx = self._boundary_indices
            n_bd = len(bd_idx)

            # Subsample boundary points if obs_indices requests fewer than all boundary points
            n_obs_requested = len(obs_indices)
            if n_obs_requested < n_bd:
                # Uniform spacing along boundary for even spatial coverage
                step = n_bd / n_obs_requested
                subsample = np.array([int(i * step) for i in range(n_obs_requested)])
                bd_idx = bd_idx[subsample]
                nx = self._boundary_normals_jax[subsample, 0:1]
                ny = self._boundary_normals_jax[subsample, 1:2]
            else:
                nx = self._boundary_normals_jax[:, 0:1]
                ny = self._boundary_normals_jax[:, 1:2]

            # Cache active normals for loss_data_from_beta during inversion
            self._active_boundary_normals_jax = jnp.concatenate([nx, ny], axis=1)

            dux_bd = dux_true[:, bd_idx, :]   # (batch, n_bd, 1)
            duy_bd = duy_true[:, bd_idx, :]
            a_bd = a_true[:, bd_idx, :]        # (batch, n_bd, 1)

            # Neumann flux: a * (du/dx * nx + du/dy * ny)
            neumann_obs = a_bd * (dux_bd * nx + duy_bd * ny)  # (batch, n_bd, 1)

            if snr_db is not None and rng is not None:
                neumann_obs = self.add_noise_snr(neumann_obs, snr_db, rng)

            x_bd = x_full[:, bd_idx, :]  # (batch, n_bd, 2)
            result['x_obs'] = x_bd
            result['u_obs'] = neumann_obs
        else:
            # Fallback for data without gradient fields
            result['x_obs'] = x_full[:, obs_indices, :]
            if 'u' in self.test_data:
                u_obs = result['u_true'][:, obs_indices, :]
                if snr_db is not None and rng is not None:
                    u_obs = self.add_noise_snr(u_obs, snr_db, rng)
                result['u_obs'] = u_obs

        return result

    def get_n_test_samples(self) -> int:
        return len(self.test_data['a'])

    def get_n_points(self) -> int:
        return self.test_data['x'].shape[1]

    def get_batch_keys(self) -> List[str]:
        return ['a', 'x', 'g_l', 'coe_val', 'coe_mask', 'u']
