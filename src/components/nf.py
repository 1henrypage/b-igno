# Neural Spline Flow coupling layers

import jax
import jax.numpy as jnp
from jax import random, jit
from jax.scipy.stats import beta as beta_dist
from flax import linen as nn
from typing import Tuple
from functools import partial
from jax.nn.initializers import variance_scaling, uniform


# PyTorch-compatible initializers

_pytorch_kernel_init = variance_scaling(1.0/3.0, "fan_in", "uniform")
_pytorch_bias_init = uniform(scale=0.1)


# Rational Quadratic Spline Functions (Pure functional, JIT-compilable)

DEFAULT_MIN_BIN_WIDTH = 1e-3
DEFAULT_MIN_BIN_HEIGHT = 1e-3
DEFAULT_MIN_DERIVATIVE = 1e-3


@jit
def searchsorted(bin_locations: jax.Array, inputs: jax.Array, eps: float = 1e-6) -> jax.Array:
    """Bin index lookup via cumulative comparison."""
    bin_locations = bin_locations.at[..., -1].add(eps)
    return jnp.sum(inputs[..., None] >= bin_locations, axis=-1) - 1


@partial(jit, static_argnums=(4, 5, 6, 7, 8))
def unconstrained_RQS(
        inputs: jax.Array,
        unnormalized_widths: jax.Array,
        unnormalized_heights: jax.Array,
        unnormalized_derivatives: jax.Array,
        inverse: bool = False,
        tail_bound: float = 1.,
        min_bin_width: float = DEFAULT_MIN_BIN_WIDTH,
        min_bin_height: float = DEFAULT_MIN_BIN_HEIGHT,
        min_derivative: float = DEFAULT_MIN_DERIVATIVE
) -> Tuple[jax.Array, jax.Array]:
    """Unconstrained rational quadratic spline transformation"""
    inside_intvl_mask = (inputs >= -tail_bound) & (inputs <= tail_bound)

    # Pad derivatives: (batch, dim, K-1) -> (batch, dim, K+1)
    unnormalized_derivatives = jnp.pad(
        unnormalized_derivatives,
        ((0, 0), (0, 0), (1, 1)),  # pad last axis
        mode='constant'
    )
    constant = jnp.log(jnp.exp(1 - min_derivative) - 1)
    unnormalized_derivatives = unnormalized_derivatives.at[:, :, 0].set(constant)
    unnormalized_derivatives = unnormalized_derivatives.at[:, :, -1].set(constant)

    # Apply RQS to all points (we'll mask later)
    outputs_inside, logabsdet_inside = RQS(
        inputs=inputs,
        unnormalized_widths=unnormalized_widths,
        unnormalized_heights=unnormalized_heights,
        unnormalized_derivatives=unnormalized_derivatives,
        inverse=inverse,
        left=-tail_bound,
        right=tail_bound,
        bottom=-tail_bound,
        top=tail_bound,
        min_bin_width=min_bin_width,
        min_bin_height=min_bin_height,
        min_derivative=min_derivative
    )

    outputs = jnp.where(inside_intvl_mask, outputs_inside, inputs)
    logabsdet = jnp.where(inside_intvl_mask, logabsdet_inside, 0.)

    return outputs, logabsdet


@partial(jit, static_argnums=(4, 5, 6, 7, 8, 9, 10, 11))
def RQS(
        inputs: jax.Array,
        unnormalized_widths: jax.Array,
        unnormalized_heights: jax.Array,
        unnormalized_derivatives: jax.Array,
        inverse: bool = False,
        left: float = 0.,
        right: float = 1.,
        bottom: float = 0.,
        top: float = 1.,
        min_bin_width: float = DEFAULT_MIN_BIN_WIDTH,
        min_bin_height: float = DEFAULT_MIN_BIN_HEIGHT,
        min_derivative: float = DEFAULT_MIN_DERIVATIVE
) -> Tuple[jax.Array, jax.Array]:
    """Rational quadratic spline transformation"""
    num_bins = unnormalized_widths.shape[-1]

    widths = jax.nn.softmax(unnormalized_widths, axis=-1)
    widths = min_bin_width + (1 - min_bin_width * num_bins) * widths
    cumwidths = jnp.cumsum(widths, axis=-1)
    cumwidths = jnp.pad(cumwidths, ((0, 0), (0, 0), (1, 0)), mode='constant', constant_values=0.0)
    cumwidths = (right - left) * cumwidths + left
    cumwidths = cumwidths.at[:, :, 0].set(left)
    cumwidths = cumwidths.at[:, :, -1].set(right)
    widths = cumwidths[..., 1:] - cumwidths[..., :-1]

    derivatives = min_derivative + jax.nn.softplus(unnormalized_derivatives)

    heights = jax.nn.softmax(unnormalized_heights, axis=-1)
    heights = min_bin_height + (1 - min_bin_height * num_bins) * heights
    cumheights = jnp.cumsum(heights, axis=-1)
    cumheights = jnp.pad(cumheights, ((0, 0), (0, 0), (1, 0)), mode='constant', constant_values=0.0)
    cumheights = (top - bottom) * cumheights + bottom
    cumheights = cumheights.at[:, :, 0].set(bottom)
    cumheights = cumheights.at[:, :, -1].set(top)
    heights = cumheights[..., 1:] - cumheights[..., :-1]

    # Find bin indices
    # inputs: (batch, dim), cumwidths/cumheights: (batch, dim, K+1)
    if inverse:
        bin_idx = searchsorted(cumheights, inputs)[..., None]  # (batch, dim, 1)
    else:
        bin_idx = searchsorted(cumwidths, inputs)[..., None]  # (batch, dim, 1)

    input_cumwidths = jnp.take_along_axis(cumwidths, bin_idx, axis=-1)[..., 0]
    input_bin_widths = jnp.take_along_axis(widths, bin_idx, axis=-1)[..., 0]
    input_cumheights = jnp.take_along_axis(cumheights, bin_idx, axis=-1)[..., 0]
    delta = heights / widths
    input_delta = jnp.take_along_axis(delta, bin_idx, axis=-1)[..., 0]
    input_derivatives = jnp.take_along_axis(derivatives, bin_idx, axis=-1)[..., 0]
    input_derivatives_plus_one = jnp.take_along_axis(derivatives[..., 1:], bin_idx, axis=-1)[..., 0]
    input_heights = jnp.take_along_axis(heights, bin_idx, axis=-1)[..., 0]

    if inverse:
        a = (((inputs - input_cumheights) *
              (input_derivatives + input_derivatives_plus_one - 2 * input_delta) +
              input_heights * (input_delta - input_derivatives)))
        b = (input_heights * input_derivatives -
             (inputs - input_cumheights) *
             (input_derivatives + input_derivatives_plus_one - 2 * input_delta))
        c = -input_delta * (inputs - input_cumheights)

        discriminant = b ** 2 - 4 * a * c
        discriminant = jnp.maximum(discriminant, 0.)

        root = (2 * c) / (-b - jnp.sqrt(discriminant))
        outputs = root * input_bin_widths + input_cumwidths

        theta_one_minus_theta = root * (1 - root)
        denominator = input_delta + ((input_derivatives + input_derivatives_plus_one - 2 * input_delta) *
                                     theta_one_minus_theta)
        derivative_numerator = input_delta ** 2 * (
                input_derivatives_plus_one * root ** 2 +
                2 * input_delta * theta_one_minus_theta +
                input_derivatives * (1 - root) ** 2
        )
        logabsdet = jnp.log(derivative_numerator) - 2 * jnp.log(denominator)
        return outputs, -logabsdet
    else:
        theta = (inputs - input_cumwidths) / input_bin_widths
        theta_one_minus_theta = theta * (1 - theta)

        numerator = input_heights * (input_delta * theta ** 2 + input_derivatives * theta_one_minus_theta)
        denominator = input_delta + ((input_derivatives + input_derivatives_plus_one - 2 * input_delta) *
                                     theta_one_minus_theta)
        outputs = input_cumheights + numerator / denominator

        derivative_numerator = input_delta ** 2 * (
                input_derivatives_plus_one * theta ** 2 +
                2 * input_delta * theta_one_minus_theta +
                input_derivatives * (1 - theta) ** 2
        )
        logabsdet = jnp.log(derivative_numerator) - 2 * jnp.log(denominator)
        return outputs, logabsdet


class FCNN(nn.Module):
    """2-hidden-layer feedforward network with SiLU activation"""
    in_dim: int
    out_dim: int
    hidden_dim: int

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        x = nn.Dense(self.hidden_dim, kernel_init=_pytorch_kernel_init, bias_init=_pytorch_bias_init)(x)
        x = jax.nn.silu(x)
        x = nn.Dense(self.hidden_dim, kernel_init=_pytorch_kernel_init, bias_init=_pytorch_bias_init)(x)
        x = jax.nn.silu(x)
        x = nn.Dense(self.out_dim, kernel_init=_pytorch_kernel_init, bias_init=_pytorch_bias_init)(x)
        return x


class NSF_CL(nn.Module):
    dim: int
    hidden_dim: int
    K: int = 5
    B: float = 1.0

    def setup(self):
        self.f1 = FCNN(self.dim // 2, (3 * self.K - 1) * (self.dim - self.dim // 2), self.hidden_dim)
        self.f2 = FCNN(self.dim - self.dim // 2, (3 * self.K - 1) * (self.dim // 2), self.hidden_dim)

    def __call__(self, x: jax.Array) -> Tuple[jax.Array, jax.Array]:
        log_det = jnp.zeros(x.shape[0])
        lower, upper = x[:, :self.dim // 2], x[:, self.dim // 2:]

        # Transform upper given lower
        out = self.f1(lower).reshape(-1, self.dim - self.dim // 2, 3 * self.K - 1)
        W, H, D = jnp.split(out, [self.K, 2 * self.K], axis=2)

        upper, ld = unconstrained_RQS(upper, W, H, D, inverse=False, tail_bound=self.B)
        log_det = log_det + jnp.sum(ld, axis=1)

        # Transform lower given upper
        out = self.f2(upper).reshape(-1, self.dim // 2, 3 * self.K - 1)
        W, H, D = jnp.split(out, [self.K, 2 * self.K], axis=2)

        lower, ld = unconstrained_RQS(lower, W, H, D, inverse=False, tail_bound=self.B)
        log_det = log_det + jnp.sum(ld, axis=1)

        return jnp.concatenate([lower, upper], axis=1), log_det

    def inverse(self, z: jax.Array) -> Tuple[jax.Array, jax.Array]:
        log_det = jnp.zeros(z.shape[0])
        lower, upper = z[:, :self.dim // 2], z[:, self.dim // 2:]

        # Inverse transform lower given upper
        out = self.f2(upper).reshape(-1, self.dim // 2, 3 * self.K - 1)
        W, H, D = jnp.split(out, [self.K, 2 * self.K], axis=2)

        lower, ld = unconstrained_RQS(lower, W, H, D, inverse=True, tail_bound=self.B)
        log_det = log_det + jnp.sum(ld, axis=1)

        # Inverse transform upper given lower
        out = self.f1(lower).reshape(-1, self.dim - self.dim // 2, 3 * self.K - 1)
        W, H, D = jnp.split(out, [self.K, 2 * self.K], axis=2)

        upper, ld = unconstrained_RQS(upper, W, H, D, inverse=True, tail_bound=self.B)
        log_det = log_det + jnp.sum(ld, axis=1)

        return jnp.concatenate([lower, upper], axis=1), log_det


class RealNVP(nn.Module):
    """Normalizing Flow using Neural Spline Flows

    Maps between latent space β ∈ [-1,1]^d and Beta(α,α) base distribution on [-1,1].
    """
    dim: int
    num_flows: int
    hidden_dim: int
    alpha: float
    K: int = 5
    B: float = 1.0  # Tail bound matches [-1, 1]

    def setup(self):
        self.flows = [
            NSF_CL(self.dim, self.hidden_dim, self.K, self.B)
            for _ in range(self.num_flows)
        ]

        # Fixed permutation (reversal - its own inverse)
        self.perm = jnp.arange(self.dim - 1, -1, -1, dtype=jnp.int32)

    def __call__(self, x: jax.Array) -> Tuple[jax.Array, jax.Array]:
        """Forward: data x -> latent z"""
        log_det_total = jnp.zeros(x.shape[0])

        for i, flow in enumerate(self.flows):
            x, log_det = flow(x)
            log_det_total = log_det_total + log_det
            if i < len(self.flows) - 1:
                x = x[:, self.perm]

        return x, log_det_total

    def inverse(self, z: jax.Array) -> Tuple[jax.Array, jax.Array]:
        """Inverse: latent z -> data x"""
        log_det_total = jnp.zeros(z.shape[0])

        for i, flow in enumerate(reversed(self.flows)):
            if i > 0:
                z = z[:, self.perm]
            z, log_det = flow.inverse(z)
            log_det_total = log_det_total + log_det

        return z, log_det_total

    def _beta_log_prob(self, z: jax.Array) -> jax.Array:
        """Log prob of scaled symmetric Beta(α,α) on [-1, 1]"""
        # Transform from [-1, 1] to [0, 1]
        z_01 = (z + 1.0) / 2.0

        # Clamp to avoid numerical issues at boundaries
        z_01 = jnp.clip(z_01, 1e-6, 1.0 - 1e-6)

        # Beta log prob + Jacobian for the scaling (d/dz of (z+1)/2 = 1/2, so log|J| = -log(2))
        log_prob = beta_dist.logpdf(z_01, self.alpha, self.alpha) - jnp.log(2.0)

        return log_prob

    def log_prob(self, x: jax.Array) -> jax.Array:
        """Compute log p(x) using change of variables"""
        z, log_det = self(x)
        log_pz = self._beta_log_prob(z).sum(axis=1)
        return log_pz + log_det

    def loss(self, x: jax.Array) -> jax.Array:
        """Negative log-likelihood loss"""
        return -self.log_prob(x).mean()

    def sample(self, rng: jax.Array, num_samples: int) -> jax.Array:
        z_01 = random.beta(rng, a=self.alpha, b=self.alpha, shape=(num_samples, self.dim))
        z = 2.0 * z_01 - 1.0
        x, _ = self.inverse(z)
        return x
