"""
B-IGNO Configuration - Import this FIRST in every script.

Sets numerical precision settings for reproducibility across different
hardware (local GPU vs SLURM cluster). Must be imported before any other
JAX-related imports.

Usage:
    import load_this_before_everything_else  # must be first line!
    import jax
    import flax
    # ... rest of imports
"""

import os

# Disable TF32 on Ampere+ GPUs (A40, A100, RTX 30xx/40xx)
# TF32 reduces mantissa precision from 23 to 10 bits, causing numerical
# instability in PDE gradient computations
os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
print("HIGH PRECISION MODE ACTIVATED!!!")

import jax

# Enable 64-bit mode - affects intermediate computations and RNG
# Critical for matching results between JAX versions (0.4.26 vs 0.4.35)
jax.config.update("jax_enable_x64", True)

# Force full float32 precision for matmuls (no TensorFloat32)
jax.config.update("jax_default_matmul_precision", "float32")


def print_config():
    print("=" * 50)
    print("B-IGNO Configuration")
    print("=" * 50)
    print(f"JAX version: {jax.__version__}")
    print(f"Devices: {jax.devices()}")
    print(f"x64 enabled: {jax.config.jax_enable_x64}")
    print(f"Matmul precision: {jax.config.jax_default_matmul_precision}")
    print(f"NVIDIA_TF32_OVERRIDE: {os.environ.get('NVIDIA_TF32_OVERRIDE', 'not set')}")
    print("=" * 50)


if __name__ == "__main__":
    print_config()
