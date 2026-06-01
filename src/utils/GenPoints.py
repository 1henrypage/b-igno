# JAX version with JIT-compatible LHS sampling

import jax
import jax.numpy as jnp
from jax import random
import matplotlib.pyplot as plt
from typing import Literal, Tuple, Optional
from functools import partial


# 1D pure functions

@partial(jax.jit, static_argnums=(1, 2, 3))
def inner_point_1d_mesh(
        key: jax.Array,
        num_sample: int,
        x_lb: float = 0.,
        x_ub: float = 1.
) -> jnp.ndarray:
    """Generate mesh points in 1D"""
    return jnp.linspace(x_lb, x_ub, num_sample).reshape(-1, 1)


@partial(jax.jit, static_argnums=(1, 2, 3))
def inner_point_1d_uniform(
        key: jax.Array,
        num_sample: int,
        x_lb: float = 0.,
        x_ub: float = 1.
) -> jnp.ndarray:
    """Generate uniform random points in 1D"""
    return random.uniform(key, shape=(num_sample, 1), minval=x_lb, maxval=x_ub)


# 2D pure functions

@partial(jax.jit, static_argnums=(1, 2, 3))
def inner_point_2d_mesh(
        key: jax.Array,
        num_sample: int,
        x_lb: Tuple[float, float] = (0., 0.),
        x_ub: Tuple[float, float] = (1., 1.)
) -> jnp.ndarray:
    """Generate mesh points in 2D"""
    x_mesh = jnp.linspace(x_lb[0], x_ub[0], num_sample)
    y_mesh = jnp.linspace(x_lb[1], x_ub[1], num_sample)
    x_mesh, y_mesh = jnp.meshgrid(x_mesh, y_mesh)
    return jnp.stack([x_mesh.flatten(), y_mesh.flatten()], axis=1)


@partial(jax.jit, static_argnums=(1, 2, 3))
def inner_point_2d_uniform(
        key: jax.Array,
        num_sample: int,
        x_lb: Tuple[float, float] = (0., 0.),
        x_ub: Tuple[float, float] = (1., 1.)
) -> jnp.ndarray:
    """Generate uniform random points in 2D"""
    lb = jnp.array(x_lb)
    ub = jnp.array(x_ub)
    return random.uniform(key, shape=(num_sample, 2), minval=lb, maxval=ub)


@partial(jax.jit, static_argnums=(1, 4))
def inner_point_sphere_muller(
        key: jax.Array,
        num_sample: int,
        xc: jnp.ndarray,
        radius: float,
        dim: int = 2
) -> jnp.ndarray:
    """Generate points inside a sphere using Muller method"""
    key1, key2 = random.split(key)
    x = random.normal(key1, shape=(num_sample, dim))
    r = random.uniform(key2, shape=(num_sample, 1)) ** 0.5
    x = (x * r) / jnp.sqrt(jnp.sum(x ** 2, axis=1, keepdims=True))
    return x * radius + xc


@partial(jax.jit, static_argnums=(1,))
def inner_point_sphere_mesh(
        key: jax.Array,
        num_sample: int,
        xc: jnp.ndarray,
        radius: float
) -> jnp.ndarray:
    """Generate points inside a sphere using mesh.

    Returns a padded fixed-size array; points outside the sphere are masked to (0,0) + xc.
    """
    x_mesh, y_mesh = jnp.meshgrid(
        jnp.linspace(-1., 1., num_sample),
        jnp.linspace(-1., 1., num_sample)
    )
    grid = jnp.stack([x_mesh.reshape(-1), y_mesh.reshape(-1)], axis=1)

    # For JIT compatibility, we keep all points but zero out those outside
    mask = jnp.linalg.norm(grid, axis=1, keepdims=True) < 1.
    x = jnp.where(mask, grid, 0.0)
    return x * radius + xc


def inner_point_sphere_mesh_variable(
        key: jax.Array,
        num_sample: int,
        xc: jnp.ndarray,
        radius: float
) -> jnp.ndarray:
    """Generate points inside sphere - NOT JIT compatible (variable size output)"""
    x_mesh, y_mesh = jnp.meshgrid(
        jnp.linspace(-1., 1., num_sample),
        jnp.linspace(-1., 1., num_sample)
    )
    grid = jnp.stack([x_mesh.reshape(-1), y_mesh.reshape(-1)], axis=1)
    mask = jnp.linalg.norm(grid, axis=1) < 1.
    x = grid[mask, :]
    return x * radius + xc


@partial(jax.jit, static_argnums=(1, 2, 3))
def boundary_point_2d_mesh(
        key: jax.Array,
        num_each_edge: int,
        x_lb: Tuple[float, float] = (0., 0.),
        x_ub: Tuple[float, float] = (1., 1.)
) -> jnp.ndarray:
    """Generate boundary points using mesh"""
    lb = jnp.array(x_lb)
    ub = jnp.array(x_ub)

    x_mesh = jnp.linspace(lb[0], ub[0], num_each_edge)
    y_mesh = jnp.linspace(lb[1], ub[1], num_each_edge)

    # Bottom edge (y = lb[1])
    bottom = jnp.stack([x_mesh, jnp.full(num_each_edge, lb[1])], axis=1)
    # Top edge (y = ub[1])
    top = jnp.stack([x_mesh, jnp.full(num_each_edge, ub[1])], axis=1)
    # Left edge (x = lb[0])
    left = jnp.stack([jnp.full(num_each_edge, lb[0]), y_mesh], axis=1)
    # Right edge (x = ub[0])
    right = jnp.stack([jnp.full(num_each_edge, ub[0]), y_mesh], axis=1)

    return jnp.concatenate([bottom, top, left, right], axis=0)


@partial(jax.jit, static_argnums=(1, 2, 3))
def boundary_point_2d_uniform(
        key: jax.Array,
        num_each_edge: int,
        x_lb: Tuple[float, float] = (0., 0.),
        x_ub: Tuple[float, float] = (1., 1.)
) -> jnp.ndarray:
    """Generate boundary points uniformly"""
    lb = jnp.array(x_lb)
    ub = jnp.array(x_ub)

    keys = random.split(key, 8)

    # Bottom edge (y = lb[1])
    bottom_x = random.uniform(keys[0], shape=(num_each_edge,), minval=lb[0], maxval=ub[0])
    bottom = jnp.stack([bottom_x, jnp.full(num_each_edge, lb[1])], axis=1)

    # Top edge (y = ub[1])
    top_x = random.uniform(keys[1], shape=(num_each_edge,), minval=lb[0], maxval=ub[0])
    top = jnp.stack([top_x, jnp.full(num_each_edge, ub[1])], axis=1)

    # Left edge (x = lb[0])
    left_y = random.uniform(keys[2], shape=(num_each_edge,), minval=lb[1], maxval=ub[1])
    left = jnp.stack([jnp.full(num_each_edge, lb[0]), left_y], axis=1)

    # Right edge (x = ub[0])
    right_y = random.uniform(keys[3], shape=(num_each_edge,), minval=lb[1], maxval=ub[1])
    right = jnp.stack([jnp.full(num_each_edge, ub[0]), right_y], axis=1)

    return jnp.concatenate([bottom, top, left, right], axis=0)


@partial(jax.jit, static_argnums=(1,))
def boundary_point_sphere_muller(
        key: jax.Array,
        num_sample: int,
        xc: jnp.ndarray,
        radius: float
) -> jnp.ndarray:
    """Generate points on sphere surface using Muller"""
    x = random.normal(key, shape=(num_sample, 2))
    x = x / jnp.sqrt(jnp.sum(x ** 2, axis=1, keepdims=True))
    return x * radius + xc


@partial(jax.jit, static_argnums=(1,))
def boundary_point_sphere_mesh(
        key: jax.Array,
        num_sample: int,
        xc: jnp.ndarray,
        radius: float
) -> jnp.ndarray:
    """Generate points on sphere surface using mesh"""
    theta = jnp.linspace(0., 2. * jnp.pi, num_sample)
    x = jnp.stack([jnp.cos(theta), jnp.sin(theta)], axis=1)
    return x * radius + xc


@partial(jax.jit, static_argnums=(1, 2, 3, 4, 5))
def weight_centers_uniform(
        key: jax.Array,
        n_center: int,
        x_lb: Tuple[float, float],
        x_ub: Tuple[float, float],
        R_max: float = 1e-4,
        R_min: float = 1e-4
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Generate centers of compact support regions using UNIFORM sampling

    Returns:
        xc: size(n_center, 1, 2)
        R: size(n_center, 1, 1)
    """
    lb = jnp.array(x_lb)
    ub = jnp.array(x_ub)

    key1, key2 = random.split(key)
    R = random.uniform(key1, shape=(n_center, 1), minval=R_min, maxval=R_max)

    lb_adj = lb + R
    ub_adj = ub - R
    xc = random.uniform(key2, shape=(n_center, 2), minval=0., maxval=1.)
    xc = xc * (ub_adj - lb_adj) + lb_adj

    return xc.reshape(-1, 1, 2), R.reshape(-1, 1, 1)


@partial(jax.jit, static_argnums=(1, 2, 3, 4, 5))
def weight_centers_lhs(
        key: jax.Array,
        n_center: int,
        x_lb: tuple[float, float],
        x_ub: tuple[float, float],
        r_max: float = 1e-4,
        r_min: float = 1e-4
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """generate centers using latin hypercube sampling

    returns:
        xc: size(n_center, 1, 2)
        r: size(n_center, 1, 1)
    """
    lb = jnp.array(x_lb)
    ub = jnp.array(x_ub)

    key1, key2, key3, key4, key5 = random.split(key, 5)
    r = random.uniform(key1, shape=(n_center, 1), minval=r_min, maxval=r_max)

    # adjust bounds based on radius
    lb_adj = lb + r
    ub_adj = ub - r

    # Latin hypercube sampling: one sample per stratum in each dimension
    perm_0 = random.permutation(key2, n_center)
    u_0 = random.uniform(key3, shape=(n_center,))
    samples_0 = (perm_0 + u_0) / n_center  # in [0, 1]
    perm_1 = random.permutation(key4, n_center)
    u_1 = random.uniform(key5, shape=(n_center,))
    samples_1 = (perm_1 + u_1) / n_center  # in [0, 1]
    xc_unit = jnp.stack([samples_0, samples_1], axis=1)  # (n_center, 2) in [0,1]^2
    xc = xc_unit * (ub_adj - lb_adj) + lb_adj

    return xc.reshape(-1, 1, 2), r.reshape(-1, 1, 1)


# Default: LHS for better spatial coverage
weight_centers = weight_centers_lhs


@partial(jax.jit, static_argnums=(1,))
def integral_grid(
        key: jax.Array,
        n_mesh: int = 9
) -> jnp.ndarray:
    """Meshgrid for calculating integrals in [-1,1]^2.

    Returns a fixed-size array; points outside the unit circle are zeroed.
    """
    x_mesh, y_mesh = jnp.meshgrid(
        jnp.linspace(-1., 1., n_mesh),
        jnp.linspace(-1., 1., n_mesh)
    )
    grid = jnp.stack([x_mesh.reshape(-1), y_mesh.reshape(-1)], axis=1)

    # For JIT compatibility, keep fixed size but mask invalid points
    mask = jnp.linalg.norm(grid, axis=1, keepdims=True) < 1.
    return jnp.where(mask, grid, 0.0)


def integral_grid_variable(
        key: jax.Array,
        n_mesh: int = 9
) -> jnp.ndarray:
    """Meshgrid for integrals - NOT JIT compatible (variable size)"""
    x_mesh, y_mesh = jnp.meshgrid(
        jnp.linspace(-1., 1., n_mesh),
        jnp.linspace(-1., 1., n_mesh)
    )
    grid = jnp.stack([x_mesh.reshape(-1), y_mesh.reshape(-1)], axis=1)
    mask = jnp.linalg.norm(grid, axis=1) < 1.
    return grid[mask, :]


# 1D+time pure functions

@partial(jax.jit, static_argnums=(1, 2, 3, 4, 5, 6, 7))
def weight_centers_1d_time_lhs(
        key: jax.Array,
        n_center: int,
        x_lb: float = -1.,
        x_ub: float = 1.,
        t_lb: float = 0.,
        t_ub: float = 1.,
        R_max: float = 1e-4,
        R_min: float = 1e-4,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Generate 1D spatial + temporal centers using LHS

    Spatial centers are LHS-sampled within [x_lb+R, x_ub-R].
    Temporal centers are linspace in [t_lb, t_ub] (matches reference behavior).

    Returns:
        xc: Spatial centers, shape (n_center, 1, 1)
        tc: Temporal centers, shape (n_center, 1, 1)
        R: Radii, shape (n_center, 1, 1)
    """
    key1, key2, key3 = random.split(key, 3)

    # Radii
    R = random.uniform(key1, shape=(n_center, 1), minval=R_min, maxval=R_max)

    # Spatial centers via LHS in adjusted bounds
    lb_adj = x_lb + R
    ub_adj = x_ub - R

    perm = random.permutation(key2, n_center)
    u = random.uniform(key3, shape=(n_center,))
    samples = (perm + u) / n_center  # in [0, 1]
    xc = samples[:, None] * (ub_adj - lb_adj) + lb_adj  # (n_center, 1)

    # Temporal centers via linspace (matches reference GenPoints_Time.py)
    tc = jnp.linspace(t_lb, t_ub, n_center)[:, None]  # (n_center, 1)

    return xc.reshape(-1, 1, 1), tc.reshape(-1, 1, 1), R.reshape(-1, 1, 1)


# Class-based API (backwards compatibility)

class Point1DTime:
    """Wrapper class for 1D spatial + temporal point generation

    Used for time-dependent PDEs (e.g., Burgers equation) where
    collocation centers need both spatial and temporal coordinates.
    """

    def __init__(
            self,
            x_lb: float = -1.,
            x_ub: float = 1.,
            t_lb: float = 0.,
            t_ub: float = 1.,
            dataType=jnp.float32,
            random_seed: int | None = None
    ):
        self.x_lb = x_lb
        self.x_ub = x_ub
        self.t_lb = t_lb
        self.t_ub = t_ub
        self.dtype = dataType
        self.key = random.PRNGKey(random_seed if random_seed is not None else 0)

    def weight_centers(
            self,
            n_center: int,
            R_max: float = 1e-4,
            R_min: float = 1e-4,
            key: Optional[jax.Array] = None,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Generate space-time centers for compact support regions

        Args:
            n_center: Number of centers
            R_max: Maximum radius
            R_min: Minimum radius
            key: Optional PRNG key. If None, uses internal state.

        Returns:
            xc: Spatial centers, shape (n_center, 1, 1)
            tc: Temporal centers, shape (n_center, 1, 1)
            R: Radii, shape (n_center, 1, 1)
        """
        if key is None:
            self.key, key = random.split(self.key)

        xc, tc, R = weight_centers_1d_time_lhs(
            key, n_center,
            self.x_lb, self.x_ub,
            self.t_lb, self.t_ub,
            R_max, R_min
        )
        return xc.astype(self.dtype), tc.astype(self.dtype), R.astype(self.dtype)


class Point1D:
    """Wrapper class for 1D point generation - maintains stateful key"""

    def __init__(
            self,
            x_lb: float = 0.,
            x_ub: float = 1.,
            dataType=jnp.float32,
            random_seed: int | None = None
    ):
        self.lb = x_lb
        self.ub = x_ub
        self.dtype = dataType
        self.key = random.PRNGKey(random_seed if random_seed is not None else 0)

    def inner_point(
            self,
            num_sample: int = 100,
            method: Literal['mesh', 'uniform'] = 'uniform',
            key: Optional[jax.Array] = None
    ) -> jnp.ndarray:
        """Generate points - dispatches to pure functions

        Args:
            num_sample: Number of points to generate
            method: 'mesh' or 'uniform'
            key: Optional PRNG key. If None, uses internal state.
        """
        if key is None:
            self.key, key = random.split(self.key)

        if method == 'mesh':
            return inner_point_1d_mesh(
                key, num_sample, self.lb, self.ub
            ).astype(self.dtype)
        elif method == 'uniform':
            return inner_point_1d_uniform(
                key, num_sample, self.lb, self.ub
            ).astype(self.dtype)
        else:
            raise NotImplementedError(f"Unknown method: {method}")


class Point2D:
    """Wrapper class for 2D point generation - maintains stateful key

    All methods accept optional `key` parameter for JIT compatibility.
    When called from JIT-compiled code, pass the key explicitly.
    """

    def __init__(
            self,
            x_lb: list[float] = [0., 0.],
            x_ub: list[float] = [1., 1.],
            dataType=jnp.float32,
            random_seed: int | None = None
    ):
        self.lb = tuple(x_lb)
        self.ub = tuple(x_ub)
        self.dtype = dataType
        self.key = random.PRNGKey(random_seed if random_seed is not None else 0)

    def inner_point(
            self,
            num_sample_or_mesh: int,
            method: Literal['mesh', 'uniform'] = 'uniform',
            key: Optional[jax.Array] = None
    ) -> jnp.ndarray:
        """Points inside the domain - dispatches to pure functions

        Args:
            num_sample_or_mesh: Number of points or mesh size
            method: 'mesh' or 'uniform'
            key: Optional PRNG key. If None, uses internal state.
        """
        if key is None:
            self.key, key = random.split(self.key)

        if method == 'mesh':
            return inner_point_2d_mesh(
                key, num_sample_or_mesh, self.lb, self.ub
            ).astype(self.dtype)
        elif method == 'uniform':
            return inner_point_2d_uniform(
                key, num_sample_or_mesh, self.lb, self.ub
            ).astype(self.dtype)
        else:
            raise NotImplementedError(f"Unknown method: {method}")

    def inner_point_sphere(
            self,
            num_sample: int,
            xc: jnp.ndarray,
            radius: float,
            method: Literal['muller', 'mesh'] = 'muller',
            key: Optional[jax.Array] = None
    ) -> jnp.ndarray:
        """Points inside a sphere - dispatches to pure functions

        Args:
            num_sample: Number of points
            xc: Center of sphere
            radius: Radius of sphere
            method: 'muller' or 'mesh'
            key: Optional PRNG key. If None, uses internal state.
        """
        if key is None:
            self.key, key = random.split(self.key)

        if method == 'muller':
            return inner_point_sphere_muller(
                key, num_sample, xc, radius
            ).astype(self.dtype)
        elif method == 'mesh':
            return inner_point_sphere_mesh(
                key, num_sample, xc, radius
            ).astype(self.dtype)
        else:
            raise NotImplementedError(f"Unknown method: {method}")

    def boundary_point(
            self,
            num_each_edge: int,
            method: Literal['mesh', 'uniform'] = 'uniform',
            key: Optional[jax.Array] = None
    ) -> jnp.ndarray:
        """Points on the boundary - dispatches to pure functions

        Args:
            num_each_edge: Number of points per edge
            method: 'mesh' or 'uniform'
            key: Optional PRNG key. If None, uses internal state.
        """
        if key is None:
            self.key, key = random.split(self.key)

        if method == 'mesh':
            return boundary_point_2d_mesh(
                key, num_each_edge, self.lb, self.ub
            ).astype(self.dtype)
        elif method == 'uniform':
            return boundary_point_2d_uniform(
                key, num_each_edge, self.lb, self.ub
            ).astype(self.dtype)
        else:
            raise NotImplementedError(f"Unknown method: {method}")

    def boundary_point_sphere(
            self,
            num_sample: int,
            xc: jnp.ndarray,
            radius: float,
            method: Literal['muller', 'mesh'] = 'mesh',
            key: Optional[jax.Array] = None
    ) -> jnp.ndarray:
        """Points on sphere surface - dispatches to pure functions

        Args:
            num_sample: Number of points
            xc: Center of sphere
            radius: Radius of sphere
            method: 'muller' or 'mesh'
            key: Optional PRNG key. If None, uses internal state.
        """
        if key is None:
            self.key, key = random.split(self.key)

        if method == 'muller':
            return boundary_point_sphere_muller(
                key, num_sample, xc, radius
            ).astype(self.dtype)
        elif method == 'mesh':
            return boundary_point_sphere_mesh(
                key, num_sample, xc, radius
            ).astype(self.dtype)
        else:
            raise NotImplementedError(f"Unknown method: {method}")

    def weight_centers(
            self,
            n_center: int,
            R_max: float = 1e-4,
            R_min: float = 1e-4,
            key: Optional[jax.Array] = None,
            use_lhs: bool = True  # Default to LHS for better coverage
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Generate centers of compact support regions

        Args:
            n_center: Number of centers
            R_max: Maximum radius
            R_min: Minimum radius
            key: Optional PRNG key. If None, uses internal state.
            use_lhs: If True, use LHS sampling (better coverage); if False, use uniform.

        Returns:
            xc: Centers, shape (n_center, 1, 2)
            R: Radii, shape (n_center, 1, 1)
        """
        if key is None:
            self.key, key = random.split(self.key)

        # Choose sampling method
        if use_lhs:
            xc, R = weight_centers_lhs(key, n_center, self.lb, self.ub, R_max, R_min)
        else:
            xc, R = weight_centers_uniform(key, n_center, self.lb, self.ub, R_max, R_min)

        return xc.astype(self.dtype), R.astype(self.dtype)

    def integral_grid(
            self,
            n_mesh_or_grid: int = 9,
            key: Optional[jax.Array] = None,
            variable_size: bool = True
    ) -> jnp.ndarray:
        """Meshgrid for calculating integrals in [-1,1]^2

        Args:
            n_mesh_or_grid: Mesh size
            key: Optional PRNG key (not used for mesh, but kept for API consistency)
            variable_size: If True, return only points inside unit circle (NOT JIT compatible)
                          If False, return fixed-size array with zeros outside (JIT compatible)
        """
        if key is None:
            key = self.key  # Don't update state since this is deterministic

        if variable_size:
            return integral_grid_variable(key, n_mesh_or_grid).astype(self.dtype)
        else:
            return integral_grid(key, n_mesh_or_grid).astype(self.dtype)

    def showPoint(self, point_dict: dict, title: str = ''):
        """Visualize the generated points"""
        fig = plt.figure(figsize=(9, 6))
        for name in point_dict.keys():
            x = point_dict[name][:, 0]
            y = point_dict[name][:, 1]
            plt.scatter(x, y, label=name, alpha=0.7)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(title)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        fig.tight_layout()
        plt.show()

