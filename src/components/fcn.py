import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import List, Callable
from .activation import FunActivation

from jax.nn.initializers import variance_scaling, uniform

# Define PyTorch-compatible initializers
_pytorch_kernel_init = variance_scaling(1.0/3.0, "fan_in", "uniform")
_pytorch_bias_init = uniform(scale=0.1)


class FCNet(nn.Module):
    layers_list: List[int]
    activation: str | Callable = 'Tanh'
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        if isinstance(self.activation, str):
            self.act_fn = FunActivation()(self.activation)
        else:
            self.act_fn = self.activation

        self.layers = [
            nn.Dense(
                out_features,
                dtype=self.dtype,
                kernel_init=_pytorch_kernel_init,
                bias_init=_pytorch_bias_init,
                name=f'layer_{i}'
            )
            for i, (in_features, out_features) in enumerate(
                zip(self.layers_list[:-1], self.layers_list[1:])
            )
        ]

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        for layer in self.layers[:-1]:
            x = layer(x)
            x = self.act_fn(x)
        x = self.layers[-1](x)
        return x