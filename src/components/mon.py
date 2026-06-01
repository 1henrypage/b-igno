# Multi-Output DeepONet architectures

import jax
import jax.numpy as jnp
from jax import vmap
from flax import linen as nn
from typing import Callable, List
from functools import partial

from .activation import FunActivation

from jax.nn.initializers import variance_scaling, uniform

_pytorch_kernel_init = variance_scaling(1.0/3.0, "fan_in", "uniform")
_pytorch_bias_init = uniform(scale=0.1)


class MultiONetBatch(nn.Module):
    """Multi-Output Operator Network."""
    in_size_x: int
    in_size_a: int
    trunk_layers: List[int]
    branch_layers: List[int]
    activation_trunk: str | Callable = 'SiLU_Sin'
    activation_branch: str | Callable = 'SiLU'
    sum_layers: int = 4
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        assert self.sum_layers < len(self.branch_layers)
        self.l = self.sum_layers

        if isinstance(self.activation_trunk, str):
            self.act_trunk = FunActivation()(self.activation_trunk)
        else:
            self.act_trunk = self.activation_trunk

        if isinstance(self.activation_branch, str):
            self.act_branch = FunActivation()(self.activation_branch)
        else:
            self.act_branch = self.activation_branch

        self.fc_trunk_in = nn.Dense(
            self.trunk_layers[0],
            dtype=self.dtype,
            kernel_init=_pytorch_kernel_init,
            bias_init=_pytorch_bias_init,
            name='trunk_in'
        )
        self.trunk_net = [
            nn.Dense(
                hidden,
                dtype=self.dtype,
                kernel_init=_pytorch_kernel_init,
                bias_init=_pytorch_bias_init,
                name=f'trunk_{i}'
            )
            for i, hidden in enumerate(self.trunk_layers[1:])
        ]

        self.fc_branch_in = nn.Dense(
            self.branch_layers[0],
            dtype=self.dtype,
            kernel_init=_pytorch_kernel_init,
            bias_init=_pytorch_bias_init,
            name='branch_in'
        )
        self.branch_net = [
            nn.Dense(
                hidden,
                dtype=self.dtype,
                kernel_init=_pytorch_kernel_init,
                bias_init=_pytorch_bias_init,
                name=f'branch_{i}'
            )
            for i, hidden in enumerate(self.branch_layers[1:])
        ]

        # Learnable weights and bias for final layers
        self.w_init = lambda key, shape: jnp.full(shape, 0.01)
        self.b_init = lambda key, shape: jnp.zeros(shape)

    @nn.compact
    def __call__(self, x: jax.Array, a: jax.Array) -> jax.Array:
        assert x.shape[0] == a.shape[0], "Batch sizes must match"

        w = [self.param(f'w_{i}', self.w_init, (1,)) for i in range(self.l)]
        b = self.param('b', self.b_init, (1,))

        x_trunk = self.act_trunk(self.fc_trunk_in(x))
        a_branch = self.act_branch(self.fc_branch_in(a))

        for net_t, net_b in zip(self.trunk_net[:-self.l], self.branch_net[:-self.l]):
            x_trunk = self.act_trunk(net_t(x_trunk))
            a_branch = self.act_branch(net_b(a_branch))

        # Sum over final layers with learnable weights
        out = 0.
        for net_t, net_b, weight in zip(
                self.trunk_net[-self.l:],
                self.branch_net[-self.l:],
                w
        ):
            x_trunk = self.act_trunk(net_t(x_trunk))
            a_branch = self.act_branch(net_b(a_branch))
            # Einstein summation: (batch, n_mesh, hidden) * (batch, hidden) -> (batch, n_mesh)
            out = out + jnp.einsum('bnh,bh->bn', x_trunk, a_branch) * weight[0]

        out = out / self.l + b[0]

        return out


class MultiONetBatch_X(nn.Module):
    """Multi-Output Operator Network with separate latent and output projections."""
    in_size_x: int
    in_size_a: int
    latent_size: int
    out_size: int
    trunk_layers: List[int]
    branch_layers: List[int]
    activation_trunk: str | Callable = 'SiLU_Sin'
    activation_branch: str | Callable = 'SiLU'
    sum_layers: int = 4
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        assert self.sum_layers < len(self.branch_layers)
        self.l = self.sum_layers

        if isinstance(self.activation_trunk, str):
            self.act_trunk = FunActivation()(self.activation_trunk)
        else:
            self.act_trunk = self.activation_trunk

        if isinstance(self.activation_branch, str):
            self.act_branch = FunActivation()(self.activation_branch)
        else:
            self.act_branch = self.activation_branch

        self.fc_trunk_in = nn.Dense(
            self.trunk_layers[0],
            dtype=self.dtype,
            kernel_init=_pytorch_kernel_init,
            bias_init=_pytorch_bias_init,
            name='trunk_in'
        )
        self.trunk_net = [
            nn.Dense(
                hidden,
                dtype=self.dtype,
                kernel_init=_pytorch_kernel_init,
                bias_init=_pytorch_bias_init,
                name=f'trunk_{i}'
            )
            for i, hidden in enumerate(self.trunk_layers[1:])
        ]

        self.fc_branch_in = nn.Dense(
            self.branch_layers[0],
            dtype=self.dtype,
            kernel_init=_pytorch_kernel_init,
            bias_init=_pytorch_bias_init,
            name='branch_in'
        )
        self.branch_net = [
            nn.Dense(hidden, dtype=self.dtype, kernel_init=_pytorch_kernel_init, bias_init=_pytorch_bias_init, name=f'branch_{i}')
            for i, hidden in enumerate(self.branch_layers[1:])
        ]

        self.fc_out = nn.Dense(self.out_size, dtype=self.dtype, kernel_init=_pytorch_kernel_init, bias_init=_pytorch_bias_init, name='out')

    @nn.compact
    def __call__(self, x: jax.Array, a: jax.Array) -> jax.Array:
        assert x.shape[0] == a.shape[0], "Batch sizes must match"

        x_trunk = self.act_trunk(self.fc_trunk_in(x))
        a_branch = self.act_branch(self.fc_branch_in(a))

        for net_t, net_b in zip(self.trunk_net[:-self.l], self.branch_net[:-self.l]):
            x_trunk = self.act_trunk(net_t(x_trunk))
            a_branch = self.act_branch(net_b(a_branch))

        # Sum over final layers
        out = 0.
        for net_t, net_b in zip(self.trunk_net[-self.l:], self.branch_net[-self.l:]):
            x_trunk = self.act_trunk(net_t(x_trunk))
            a_branch = self.act_branch(net_b(a_branch))
            # (batch, n_mesh, hidden) * (batch, latent_size, hidden) -> (batch, n_mesh, latent_size)
            out = out + jnp.einsum('bnh,bmh->bnm', x_trunk, a_branch)

        # (batch, n_mesh, latent_size) -> (batch, n_mesh, out_size)
        out = self.fc_out(out / self.l)

        return out


class MultiONetCartesianProd(nn.Module):
    """More efficient for cases where trunk and branch can be evaluated separately
    and combined via outer product.
    """
    in_size_x: int
    in_size_a: int
    trunk_layers: List[int]
    branch_layers: List[int]
    activation_trunk: str | Callable = 'SiLU_Sin'
    activation_branch: str | Callable = 'SiLU'
    sum_layers: int = 4
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        assert self.sum_layers < len(self.branch_layers)
        self.l = self.sum_layers

        if isinstance(self.activation_trunk, str):
            self.act_trunk = FunActivation()(self.activation_trunk)
        else:
            self.act_trunk = self.activation_trunk

        if isinstance(self.activation_branch, str):
            self.act_branch = FunActivation()(self.activation_branch)
        else:
            self.act_branch = self.activation_branch

        self.fc_trunk_in = nn.Dense(self.trunk_layers[0], dtype=self.dtype, kernel_init=_pytorch_kernel_init, bias_init=_pytorch_bias_init, name='trunk_in')
        self.trunk_net = [
            nn.Dense(hidden, dtype=self.dtype, kernel_init=_pytorch_kernel_init, bias_init=_pytorch_bias_init, name=f'trunk_{i}')
            for i, hidden in enumerate(self.trunk_layers[1:])
        ]

        self.fc_branch_in = nn.Dense(self.branch_layers[0], dtype=self.dtype, kernel_init=_pytorch_kernel_init, bias_init=_pytorch_bias_init, name='branch_in')
        self.branch_net = [
            nn.Dense(hidden, dtype=self.dtype, kernel_init=_pytorch_kernel_init, bias_init=_pytorch_bias_init, name=f'branch_{i}')
            for i, hidden in enumerate(self.branch_layers[1:])
        ]

        self.w_init = lambda key, shape: jnp.full(shape, 0.01)
        self.b_init = lambda key, shape: jnp.zeros(shape)

    @nn.compact
    def __call__(self, x: jax.Array, a: jax.Array) -> jax.Array:
        """Forward pass with Cartesian product

        x has no batch dimension (mesh_size, dx); a has batch (n_batch, latent_size).
        """
        w = [self.param(f'w_{i}', self.w_init, (1,)) for i in range(self.l)]
        b = self.param('b', self.b_init, (1,))

        x_trunk = self.act_trunk(self.fc_trunk_in(x))
        a_branch = self.act_branch(self.fc_branch_in(a))

        for net_t, net_b in zip(self.trunk_net[:-self.l], self.branch_net[:-self.l]):
            x_trunk = self.act_trunk(net_t(x_trunk))
            a_branch = self.act_branch(net_b(a_branch))

        # Sum over final layers - Cartesian product
        out = 0.
        for net_t, net_b, weight in zip(
                self.trunk_net[-self.l:],
                self.branch_net[-self.l:],
                w
        ):
            x_trunk = self.act_trunk(net_t(x_trunk))
            a_branch = self.act_branch(net_b(a_branch))
            # (batch, hidden) * (mesh, hidden) -> (batch, mesh)
            out = out + jnp.einsum('bh,mh->bm', a_branch, x_trunk) * weight[0]

        out = out / self.l + b[0]

        return out


class MultiONetCartesianProd_X(nn.Module):
    """Multi-Input & Multi-Output with Cartesian product structure"""
    in_size_x: int
    in_size_a: int
    latent_size: int
    out_size: int
    trunk_layers: List[int]
    branch_layers: List[int]
    activation_trunk: str | Callable = 'SiLU_Sin'
    activation_branch: str | Callable = 'SiLU'
    sum_layers: int = 4
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        assert self.sum_layers < len(self.branch_layers)
        self.l = self.sum_layers

        if isinstance(self.activation_trunk, str):
            self.act_trunk = FunActivation()(self.activation_trunk)
        else:
            self.act_trunk = self.activation_trunk

        if isinstance(self.activation_branch, str):
            self.act_branch = FunActivation()(self.activation_branch)
        else:
            self.act_branch = self.activation_branch

        self.fc_trunk_in = nn.Dense(self.trunk_layers[0], dtype=self.dtype, kernel_init=_pytorch_kernel_init, bias_init=_pytorch_bias_init, name='trunk_in')
        self.trunk_net = [
            nn.Dense(hidden, dtype=self.dtype, kernel_init=_pytorch_kernel_init, bias_init=_pytorch_bias_init, name=f'trunk_{i}')
            for i, hidden in enumerate(self.trunk_layers[1:])
        ]

        self.fc_branch_in = nn.Dense(self.branch_layers[0], dtype=self.dtype, kernel_init=_pytorch_kernel_init, bias_init=_pytorch_bias_init, name='branch_in')
        self.branch_net = [
            nn.Dense(hidden, dtype=self.dtype, kernel_init=_pytorch_kernel_init, bias_init=_pytorch_bias_init, name=f'branch_{i}')
            for i, hidden in enumerate(self.branch_layers[1:])
        ]

        self.fc_out = nn.Dense(self.out_size, dtype=self.dtype, kernel_init=_pytorch_kernel_init, bias_init=_pytorch_bias_init, name='out')

    @nn.compact
    def __call__(self, x: jax.Array, a: jax.Array) -> jax.Array:
        """Forward pass with Cartesian product

        x has no batch dimension (mesh_size, dx); a has batch (n_batch, latent_size, da).
        """
        x_trunk = self.act_trunk(self.fc_trunk_in(x))
        a_branch = self.act_branch(self.fc_branch_in(a))

        for net_t, net_b in zip(self.trunk_net[:-self.l], self.branch_net[:-self.l]):
            x_trunk = self.act_trunk(net_t(x_trunk))
            a_branch = self.act_branch(net_b(a_branch))

        # Sum over final layers - Cartesian product
        out = 0.
        for net_t, net_b in zip(self.trunk_net[-self.l:], self.branch_net[-self.l:]):
            x_trunk = self.act_trunk(net_t(x_trunk))
            a_branch = self.act_branch(net_b(a_branch))
            # (batch, latent_size, hidden) * (mesh, hidden) -> (batch, mesh, latent_size)
            out = out + jnp.einsum('bmh,nh->bnm', a_branch, x_trunk)

        # (batch, mesh, latent_size) -> (batch, mesh, out_size)
        out = self.fc_out(out / self.l)

        return out
