import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Callable


class Sinc(nn.Module):
    """Sinc activation: x * sin(x)"""
    
    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        return x * jnp.sin(x)


class Tanh_Sin(nn.Module):
    """Tanh-Sin activation: tanh(sin(π(x+1))) + x"""
    
    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        return jnp.tanh(jnp.sin(jnp.pi * (x + 1))) + x


class SiLU_Sin(nn.Module):
    """SiLU-Sin activation: silu(sin(π(x+1))) + x"""
    
    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        return jax.nn.silu(jnp.sin(jnp.pi * (x + 1))) + x


class SiLU_Id(nn.Module):
    """SiLU with identity residual: silu(x) + x"""
    
    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        return jax.nn.silu(x) + x


class FunActivation:
    """Returns Flax modules or pure functions by name (case-insensitive)."""

    def __init__(self):
        self.activation = {
            'identity': lambda: lambda x: x,
            'relu': lambda: jax.nn.relu,
            'elu': lambda: jax.nn.elu,
            'softplus': lambda: jax.nn.softplus,
            'sigmoid': lambda: jax.nn.sigmoid,
            'tanh': lambda: jnp.tanh,
            'silu': lambda: jax.nn.silu,
            'sinc': Sinc,
            'tanh_sin': Tanh_Sin,
            'silu_sin': SiLU_Sin,
            'silu_id': SiLU_Id,
        }
    
    def __call__(self, type: str) -> Callable[[jax.Array], jax.Array] | nn.Module:
        """
        Returns either a Flax module (for custom activations) or a pure function 
        (for standard JAX activations).
        """
        key = type.lower()
        if key not in self.activation:
            raise ValueError(
                f"Activation '{type}' not found. "
                f"Available: {list(self.activation.keys())}"
            )
        return self.activation[key]()
