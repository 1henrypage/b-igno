import jax
import jax.numpy as jnp
from jax import vmap, jit
from typing import Literal
from functools import partial


@partial(jit, static_argnums=(2,))
def compute_norm_single(y_pred: jnp.ndarray, y_true: jnp.ndarray, p: int) -> jnp.ndarray:
    diff = y_pred - y_true
    return jnp.linalg.norm(diff, ord=p)


# Vectorize over batch dimension
compute_norm_batched = vmap(compute_norm_single, in_axes=(0, 0, None))


@partial(jit, static_argnums=(3, 4))
def compute_norm_with_mesh_scaling_single(
        y_pred: jnp.ndarray,
        y_true: jnp.ndarray,
        n_points: int,
        d: int,
        p: int
) -> jnp.ndarray:
    diff = y_pred - y_true
    h = 1.0 / (n_points - 1.0)
    norm = jnp.linalg.norm(diff, ord=p)
    return (h ** (d / p)) * norm


# Vectorize over batch
compute_norm_with_mesh_scaling_batched = vmap(
    compute_norm_with_mesh_scaling_single,
    in_axes=(0, 0, None, None, None)
)


class MyError:
    """Error metrics with proper vectorization"""

    def __init__(self, d: int = 2, p: int = 2, size_average: bool = True, reduction: bool = True):
        assert d > 0 and p > 0
        self.d = d
        self.p = p
        self.size_average = size_average
        self.reduction = reduction
        self.eps = 1e-8

    def _reduce(self, values: jnp.ndarray):
        if not self.reduction:
            return values
        return jnp.mean(values) if self.size_average else jnp.sum(values)

    @partial(jit, static_argnums=(0,))
    def LP_abs(self, y_pred: jnp.ndarray, y_true: jnp.ndarray) -> jnp.ndarray:
        """Absolute Lp error with mesh scaling"""
        assert y_pred.shape == y_true.shape
        batch_size = y_true.shape[0]
        n_points = y_true.shape[1]

        y_pred_flat = y_pred.reshape(batch_size, -1)
        y_true_flat = y_true.reshape(batch_size, -1)
        total_norm = compute_norm_with_mesh_scaling_batched(
            y_pred_flat, y_true_flat, n_points, self.d, self.p
        )

        return self._reduce(total_norm)

    @partial(jit, static_argnums=(0,))
    def Lp_rel(self, y_pred: jnp.ndarray, y_true: jnp.ndarray) -> jnp.ndarray:
        """Relative Lp error"""
        assert y_pred.shape == y_true.shape
        batch_size = y_true.shape[0]

        y_pred_flat = y_pred.reshape(batch_size, -1)
        y_true_flat = y_true.reshape(batch_size, -1)
        diff_norm = compute_norm_batched(y_pred_flat - y_true_flat, jnp.zeros_like(y_true_flat), self.p)
        y_norm = compute_norm_batched(y_true_flat, jnp.zeros_like(y_true_flat), self.p) + self.eps

        return self._reduce(diff_norm / y_norm)

    def __call__(self, err_type: Literal['lp_abs', 'lp_rel']):
        if err_type == 'lp_abs':
            return self.LP_abs
        elif err_type == 'lp_rel':
            return self.Lp_rel
        else:
            raise NotImplementedError(f"Error type '{err_type}' is not defined.")


class MyLoss:
    """Loss functions with proper vectorization"""

    def __init__(self, size_average: bool = True, reduction: bool = True):
        self.size_average = size_average
        self.reduction = reduction
        self.eps = 1e-6

    def _reduce(self, values: jnp.ndarray):
        if not self.reduction:
            return values
        return jnp.mean(values) if self.size_average else jnp.sum(values)

    @partial(jit, static_argnums=(0,))
    def mse_org(self, y_pred: jnp.ndarray, y_true: jnp.ndarray) -> jnp.ndarray:
        """MSE loss without relative scaling"""
        assert y_pred.shape == y_true.shape
        batch_size = y_true.shape[0]

        y_pred_flat = y_pred.reshape(batch_size, -1)
        y_true_flat = y_true.reshape(batch_size, -1)

        diff_norm = compute_norm_batched(y_pred_flat - y_true_flat, jnp.zeros_like(y_true_flat), 2)
        return self._reduce(diff_norm)

    @partial(jit, static_argnums=(0,))
    def mse_rel(self, y_pred: jnp.ndarray, y_true: jnp.ndarray) -> jnp.ndarray:
        """MSE loss with relative scaling"""
        assert y_pred.shape == y_true.shape
        batch_size = y_true.shape[0]

        y_pred_flat = y_pred.reshape(batch_size, -1)
        y_true_flat = y_true.reshape(batch_size, -1)

        diff_norm = compute_norm_batched(y_pred_flat - y_true_flat, jnp.zeros_like(y_true_flat), 2)
        y_norm = compute_norm_batched(y_true_flat, jnp.zeros_like(y_true_flat), 2) + self.eps

        return self._reduce(diff_norm / y_norm)

    def __call__(self, loss_type: Literal['mse_org', 'mse_rel']):
        if loss_type == 'mse_org':
            return self.mse_org
        elif loss_type == 'mse_rel':
            return self.mse_rel
        else:
            raise NotImplementedError(f"Loss type '{loss_type}' is not defined.")
