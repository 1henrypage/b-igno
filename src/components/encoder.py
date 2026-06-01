# NOTE: JAX Conv uses channels-LAST: (batch, height, width, channels)

import jax
import jax.numpy as jnp
from jax import random
from flax import linen as nn
from typing import Tuple, Callable

from .activation import FunActivation
from .fcn import FCNet
from .cnn import CNNet1d, CNNet2d


class EncoderFCNet(nn.Module):
    layers_list: list[int]
    activation: str | Callable
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        if isinstance(self.activation, str):
            self.act_fn = FunActivation()(self.activation)
        else:
            self.act_fn = self.activation

        self.net = FCNet(
            layers_list=self.layers_list,
            activation=self.act_fn,
            dtype=self.dtype
        )

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        x = x.reshape(x.shape[0], -1)
        x = self.net(x)
        return x


class EncoderFCNet_VAE(nn.Module):
    """Variational autoencoder with reparameterization trick"""
    layers_list: list
    activation: str | Callable
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        if isinstance(self.activation, str):
            self.act_fn = FunActivation()(self.activation)
        else:
            self.act_fn = self.activation

        self.net_mu = FCNet(
            layers_list=self.layers_list,
            activation=self.act_fn,
            dtype=self.dtype
        )
        self.net_log_var = FCNet(
            layers_list=self.layers_list,
            activation=self.act_fn,
            dtype=self.dtype
        )

    def reparam(self, rng: jax.Array, mu: jax.Array, log_var: jax.Array) -> jax.Array:
        std = jnp.exp(0.5 * log_var)
        eps = random.normal(rng, log_var.shape)
        return mu + std * eps

    @nn.compact
    def __call__(
            self,
            x: jax.Array,
            rng: jax.Array = None
    ) -> Tuple[jax.Array, jax.Array, jax.Array]:
        x = x.reshape(x.shape[0], -1)
        mu = self.net_mu(x)
        log_var = self.net_log_var(x)

        if rng is not None:
            beta = self.reparam(rng, mu, log_var)
        else:
            beta = mu

        return beta, mu, log_var


class EncoderFCNetTanh(nn.Module):
    """Fully connected encoder with tanh output"""
    layers_list: list[int]
    activation: str | Callable
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.encoder = EncoderFCNet(
            layers_list=self.layers_list,
            activation=self.activation,
            dtype=self.dtype
        )

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        beta = self.encoder(x)
        return jnp.tanh(beta)


class EncoderCNNet1d(nn.Module):
    conv_arch: list[int]
    fc_arch: list[int]
    activation_conv: str | Callable
    activation_fc: str | Callable
    kernel_size: int = 5
    stride: int = 3
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.in_channel = self.conv_arch[0]
        self.net = CNNet1d(
            conv_arch=self.conv_arch,
            fc_arch=self.fc_arch,
            activation_conv=self.activation_conv,
            activation_fc=self.activation_fc,
            kernel_size=self.kernel_size,
            stride=self.stride,
            dtype=self.dtype
        )

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        x = self.net(x)
        return x


class EncoderCNNet2d(nn.Module):
    conv_arch: list[int]
    fc_arch: list[int]
    activation_conv: str | Callable
    activation_fc: str | Callable
    nx_size: int
    ny_size: int
    kernel_size: Tuple[int, int] = (5, 5)
    stride: int = 3
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.in_channel = self.conv_arch[0]
        self.net = CNNet2d(
            conv_arch=self.conv_arch,
            fc_arch=self.fc_arch,
            activation_conv=self.activation_conv,
            activation_fc=self.activation_fc,
            kernel_size=self.kernel_size,
            stride=self.stride,
            dtype=self.dtype
        )

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        # (batch, ny*nx, ch) -> (batch, ny, nx, ch)
        x = x.reshape(-1, self.ny_size, self.nx_size, self.in_channel)
        x = self.net(x)
        return x


class EncoderCNNet2dTanh(nn.Module):
    """2D CNN encoder with tanh output"""
    conv_arch: list[int]
    fc_arch: list[int]
    activation_conv: str | Callable
    activation_fc: str | Callable
    nx_size: int
    ny_size: int
    kernel_size: Tuple[int, int] = (5, 5)
    stride: int = 3
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.encoder = EncoderCNNet2d(
            conv_arch=self.conv_arch,
            fc_arch=self.fc_arch,
            activation_conv=self.activation_conv,
            activation_fc=self.activation_fc,
            nx_size=self.nx_size,
            ny_size=self.ny_size,
            kernel_size=self.kernel_size,
            stride=self.stride,
            dtype=self.dtype
        )

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        beta = self.encoder(x)
        beta_out = jnp.tanh(beta)
        return beta_out
