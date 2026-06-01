import jax
import jax.numpy as jnp
from jax import random
import numpy as np
from pathlib import Path
import subprocess


def setup_seed(seed: int):
    # JAX uses explicit PRNG keys, but we can set numpy seed
    np.random.seed(seed)
    # Return a base PRNG key for JAX
    return random.PRNGKey(seed)


def get_project_root() -> Path:
    return Path(
        subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"],
            stderr=subprocess.DEVNULL,
        )
        .decode()
        .strip()
    )

def np2jax(x: np.ndarray, dtype=jnp.float32) -> jnp.ndarray:
    return jnp.array(x, dtype=dtype)


def jax2np(x: jnp.ndarray) -> np.ndarray:
    return np.array(x)

