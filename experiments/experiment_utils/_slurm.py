# WARNING: stdlib-only. Do NOT import numpy, JAX, or anything from this package.
# Do NOT add to __init__.py — must be imported directly.
import os
import sys
import argparse


def parse_slurm_task(parameter_grid):
    """Parse --task-id / --print-array-size / SLURM_ARRAY_TASK_ID.

    Must be called BEFORE any JAX import. Handles three modes:
    1. --print-array-size: prints len(grid) and exits (for SLURM job submission)
    2. --task-id N or SLURM_ARRAY_TASK_ID=N: returns grid[N] params
    3. Interactive (neither): returns None

    Args:
        parameter_grid: list of dicts, one per array task.
            Examples: [{"seed": 42, "test_idx": 0}, ...]
                      [{"seed": 42, "snr": 20}, ...]

    Returns:
        (task_params, task_id) where task_params is grid[task_id] or None
        if running interactively.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", type=int, default=None)
    parser.add_argument("--print-array-size", action="store_true")
    args, _ = parser.parse_known_args()

    if args.print_array_size:
        print(len(parameter_grid))
        sys.exit(0)

    task_id = args.task_id
    if task_id is None:
        env_id = os.environ.get("SLURM_ARRAY_TASK_ID")
        if env_id is not None:
            task_id = int(env_id)

    task_params = parameter_grid[task_id] if task_id is not None else None
    return task_params, task_id
