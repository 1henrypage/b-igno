"""
Laplace approximation utilities for posterior inference over the IGNO latent space.

Given a MAP estimate beta* (from the IGNO inverter) and a loss function U(beta),
computes the Hessian H = d²U/d(beta)² at beta* and samples from the Laplace
approximation q(beta) = N(beta*, H^{-1}).
"""
import time
from typing import Callable, Dict, Tuple

import jax
import jax.numpy as jnp


def compute_hessian(
    neg_log_posterior_fn: Callable[[jnp.ndarray], float],
    beta_map: jnp.ndarray,
    reg_lambda: float = 1e-4,
) -> Tuple[jnp.ndarray, Dict]:
    """Compute the Hessian of U at beta_map with PD correction.

    Args:
        neg_log_posterior_fn: U(beta), scalar-valued.
        beta_map: MAP estimate, shape (d,).
        reg_lambda: Tikhonov regularisation added to diagonal.

    Returns:
        (H, diagnostics) where H is the (d, d) positive-definite precision
        matrix and diagnostics records eigenvalue information.
    """
    t0 = time.time()
    hessian_fn = jax.jit(jax.hessian(neg_log_posterior_fn))
    H_raw = hessian_fn(beta_map)
    hessian_time = time.time() - t0

    d = beta_map.shape[0]

    # Symmetrise (numerical safety)
    H = 0.5 * (H_raw + H_raw.T)

    # Record pre-regularisation diagnostics
    eigvals_raw = jnp.linalg.eigvalsh(H)
    min_eig_raw = float(eigvals_raw[0])
    max_eig_raw = float(eigvals_raw[-1])
    n_neg = int(jnp.sum(eigvals_raw < 0))

    # Tikhonov regularisation
    H = H + reg_lambda * jnp.eye(d)

    # Check PD via Cholesky; fallback to eigenvalue clipping if needed
    try:
        jnp.linalg.cholesky(H)
        pd_method = "tikhonov"
    except Exception:
        eigvals, eigvecs = jnp.linalg.eigh(H)
        eps = max(1e-6, float(jnp.max(jnp.abs(eigvals))) * 1e-6)
        eigvals = jnp.maximum(eigvals, eps)
        H = eigvecs @ jnp.diag(eigvals) @ eigvecs.T
        pd_method = "eigenvalue_clip"

    cond = float(max_eig_raw / max(abs(min_eig_raw), 1e-30))

    diagnostics = {
        "min_eigenvalue_raw": min_eig_raw,
        "max_eigenvalue_raw": max_eig_raw,
        "n_negative_eigenvalues": n_neg,
        "condition_number": cond,
        "reg_lambda": reg_lambda,
        "pd_method": pd_method,
        "hessian_time_s": hessian_time,
    }
    return H, diagnostics


def sample_laplace(
    beta_map: jnp.ndarray,
    H: jnp.ndarray,
    n_samples: int,
    rng_key: jax.Array,
) -> Tuple[jnp.ndarray, float]:
    """Sample from the Laplace posterior N(beta_map, H^{-1}).

    Samples are clipped to [-1, 1] (the NF support).

    Args:
        beta_map: MAP estimate, shape (d,).
        H: precision matrix (positive definite), shape (d, d).
        n_samples: number of posterior samples to draw.
        rng_key: JAX PRNG key.

    Returns:
        (samples, fraction_clipped) where samples has shape (n_samples, d).
    """
    Sigma = jnp.linalg.inv(H)
    Sigma = 0.5 * (Sigma + Sigma.T)  # symmetrise (numerical safety)

    samples = jax.random.multivariate_normal(rng_key, beta_map, Sigma, shape=(n_samples,))

    out_of_bounds = jnp.any((samples < -1.0) | (samples > 1.0), axis=-1)
    fraction_clipped = float(jnp.mean(out_of_bounds))

    samples = jnp.clip(samples, -1.0, 1.0)
    return samples, fraction_clipped
