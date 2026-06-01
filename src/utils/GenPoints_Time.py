import jax
import jax.numpy as jnp
from jax import random
from scipy.stats import qmc
import numpy as np
import matplotlib.pyplot as plt
from typing import Literal, Tuple
from functools import partial


# 1D pure functions

@partial(jax.jit, static_argnums=(1, 2, 3, 4, 5))
def inner_point_1d_mesh(
        key: jax.Array,
        nx: int,
        nt: int,
        x_lb: float,
        x_ub: float,
        t0: float = 0.,
        tT: float = 1.
) -> jnp.ndarray:
    """Generate mesh points in 1D space-time"""
    X = jnp.linspace(x_lb, x_ub, nx)
    T = jnp.linspace(t0, tT, nt)
    xx, tt = jnp.meshgrid(X, T)
    return jnp.stack([xx.flatten(), tt.flatten()], axis=1)


@partial(jax.jit, static_argnums=(1, 2, 3, 4, 5))
def inner_point_1d_uniform(
        key: jax.Array,
        nx: int,
        nt: int,
        x_lb: float,
        x_ub: float,
        t0: float = 0.,
        tT: float = 1.
) -> jnp.ndarray:
    """Generate uniform points in 1D space-time"""
    T = jnp.linspace(t0, tT, nt).repeat(nx, axis=0)
    X = random.uniform(key, shape=(nx * nt,), minval=x_lb, maxval=x_ub)
    return jnp.stack([X.flatten(), T.flatten()], axis=1)


@partial(jax.jit, static_argnums=(1, 4, 5))
def boundary_point_1d(
        key: jax.Array,
        num_sample: int,
        x_lb: float,
        x_ub: float,
        t0: float = 0.,
        tT: float = 1.
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Generate boundary points in 1D space-time"""
    T = jnp.linspace(t0, tT, num_sample)
    # Lower boundary
    X_lb = jnp.full((num_sample,), x_lb)
    XT_lb = jnp.stack([X_lb, T], axis=1)
    # Upper boundary
    X_ub = jnp.full((num_sample,), x_ub)
    XT_ub = jnp.stack([X_ub, T], axis=1)
    return XT_lb, XT_ub


@partial(jax.jit, static_argnums=(1, 2, 4, 5))
def init_point_1d_mesh(
        key: jax.Array,
        num_sample: int,
        x_lb: float,
        x_ub: float,
        t_stamp: float = 0.
) -> jnp.ndarray:
    """Generate initial condition points"""
    X = jnp.linspace(x_lb, x_ub, num_sample)
    T = jnp.full((num_sample,), t_stamp)
    return jnp.stack([X, T], axis=1)


@partial(jax.jit, static_argnums=(1, 2, 4, 5))
def init_point_1d_uniform(
        key: jax.Array,
        num_sample: int,
        x_lb: float,
        x_ub: float,
        t_stamp: float = 0.
) -> jnp.ndarray:
    """Generate initial condition points uniformly"""
    X = random.uniform(key, shape=(num_sample,), minval=x_lb, maxval=x_ub)
    T = jnp.full((num_sample,), t_stamp)
    return jnp.stack([X, T], axis=1)


@partial(jax.jit, static_argnums=(1, 2, 5, 6))
def weight_centers_1d(
        key: jax.Array,
        n_center: int,
        nt: int,
        x_lb: float,
        x_ub: float,
        R_max: float = 1e-4,
        R_min: float = 1e-4,
        t0: float = 0.,
        tT: float = 1.
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Generate centers for 1D space-time test functions

    Returns:
        xc: size(n_center*nt, 1, 1)
        tc: size(n_center*nt, 1, 1)
        R: size(n_center*nt, 1, 1)
    """
    key1, key2 = random.split(key)

    total = n_center * nt
    R = random.uniform(key1, shape=(total, 1), minval=R_min, maxval=R_max)

    lb = x_lb + R
    ub = x_ub - R

    X = random.uniform(key2, shape=(total, 1))
    xc = X * (ub - lb) + lb
    T = jnp.linspace(t0, tT, nt).repeat(n_center, axis=0)
    tc = T.reshape(-1, 1)

    return xc.reshape(-1, 1, 1), tc.reshape(-1, 1, 1), R.reshape(-1, 1, 1)


@partial(jax.jit, static_argnums=(1,))
def integral_grid_1d(
        key: jax.Array,
        n_mesh: int = 9
) -> jnp.ndarray:
    """1D integral grid"""
    return jnp.linspace(-1., 1., n_mesh).reshape(-1, 1)


# 2D pure functions

@partial(jax.jit, static_argnums=(1, 2, 3, 4))
def inner_point_2d_mesh(
        key: jax.Array,
        nx: int,
        nt: int,
        x_lb: Tuple[float, float],
        x_ub: Tuple[float, float],
        t0: float = 0.,
        tT: float = 1.
) -> jnp.ndarray:
    """Generate mesh points in 2D space-time"""
    lb = jnp.array(x_lb)
    ub = jnp.array(x_ub)

    x_mesh = jnp.linspace(lb[0], ub[0], nx)
    y_mesh = jnp.linspace(lb[1], ub[1], nx)
    x_mesh, y_mesh = jnp.meshgrid(x_mesh, y_mesh)
    X = jnp.stack([x_mesh.flatten(), y_mesh.flatten()], axis=1)

    T = jnp.linspace(t0, tT, nt).repeat(X.shape[0], axis=0)

    XT = jnp.concatenate([jnp.tile(X, (nt, 1)), T.reshape(-1, 1)], axis=-1)
    return XT


@partial(jax.jit, static_argnums=(1, 2, 3, 4, 7))
def boundary_point_2d(
        key: jax.Array,
        nx_each_edge: int,
        nt: int,
        x_lb: Tuple[float, float],
        x_ub: Tuple[float, float],
        t0: float = 0.,
        tT: float = 1.,
        method: str = 'mesh'
) -> jnp.ndarray:
    """Generate boundary points in 2D space-time"""
    lb = jnp.array(x_lb)
    ub = jnp.array(x_ub)

    T = jnp.linspace(t0, tT, nt).repeat(nx_each_edge * 4, axis=0)

    X_bd = []
    for _ in range(nt):
        for d in range(2):
            # Generate mesh points
            x_mesh = jnp.linspace(lb[0], ub[0], nx_each_edge)
            y_mesh = jnp.linspace(lb[1], ub[1], nx_each_edge)
            X_lb = jnp.stack([x_mesh, y_mesh], axis=1)
            X_ub = jnp.stack([x_mesh, y_mesh], axis=1)

            X_lb = X_lb.at[:, d].set(lb[d])
            X_bd.append(X_lb)
            X_ub = X_ub.at[:, d].set(ub[d])
            X_bd.append(X_ub)

    X_bd = jnp.concatenate(X_bd, axis=0)
    XT = jnp.concatenate([X_bd, T.reshape(-1, 1)], axis=-1)
    return XT


@partial(jax.jit, static_argnums=(1, 2, 3, 5))
def init_point_2d_mesh(
        key: jax.Array,
        nx: int,
        x_lb: Tuple[float, float],
        x_ub: Tuple[float, float],
        t_stamp: float = 0.
) -> jnp.ndarray:
    """Generate initial condition points in 2D"""
    lb = jnp.array(x_lb)
    ub = jnp.array(x_ub)

    x_mesh = jnp.linspace(lb[0], ub[0], nx)
    y_mesh = jnp.linspace(lb[1], ub[1], nx)
    x_mesh, y_mesh = jnp.meshgrid(x_mesh, y_mesh)
    X = jnp.stack([x_mesh.flatten(), y_mesh.flatten()], axis=1)

    T = jnp.full((X.shape[0],), t_stamp)
    XT = jnp.concatenate([X, T.reshape(-1, 1)], axis=-1)
    return XT


@partial(jax.jit, static_argnums=(1, 2, 3, 4, 7))
def weight_centers_2d(
        key: jax.Array,
        n_center: int,
        nt: int,
        x_lb: Tuple[float, float],
        x_ub: Tuple[float, float],
        R_max: float = 1e-4,
        R_min: float = 1e-4,
        t0: float = 0.,
        tT: float = 1.
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Generate centers for 2D space-time test functions

    Returns:
        xc: size(n_center*nt, 1, 2)
        tc: size(n_center*nt, 1, 1)
        R: size(n_center*nt, 1, 1)
    """
    lb = jnp.array(x_lb)
    ub = jnp.array(x_ub)

    key1, key2 = random.split(key)

    total = n_center * nt
    R = random.uniform(key1, shape=(total, 1), minval=R_min, maxval=R_max)

    lb_adj = lb + R
    ub_adj = ub - R

    X = random.uniform(key2, shape=(total, 2))
    xc = X * (ub_adj - lb_adj) + lb_adj

    T = jnp.linspace(t0, tT, nt).repeat(n_center, axis=0)
    tc = T.reshape(-1, 1)

    return xc.reshape(-1, 1, 2), tc.reshape(-1, 1, 1), R.reshape(-1, 1, 1)


# Class-based API (backwards compatibility)

class Point1D:
    """Wrapper for 1D space-time point generation"""

    def __init__(
            self,
            x_lb: list[float] = [0.],
            x_ub: list[float] = [1.],
            dataType=jnp.float32,
            random_seed: int | None = None
    ):
        self.lb = x_lb[0] if isinstance(x_lb, list) else x_lb
        self.ub = x_ub[0] if isinstance(x_ub, list) else x_ub
        self.dtype = dataType
        self.key = random.PRNGKey(random_seed if random_seed is not None else 0)
        # Keep scipy for LHS
        self.lhs_t = qmc.LatinHypercube(1, seed=random_seed)
        self.lhs_x = qmc.LatinHypercube(1, seed=random_seed + 10086 if random_seed else 10086)

    def inner_point(
            self,
            nx: int,
            nt: int,
            method: Literal['mesh', 'uniform', 'hypercube'] = 'hypercube',
            t0: list[float] = [0.],
            tT: list[float] = [1.]
    ) -> jnp.ndarray:
        """Inner points"""
        t0_val = t0[0] if isinstance(t0, list) else t0
        tT_val = tT[0] if isinstance(tT, list) else tT

        if method == 'mesh':
            return inner_point_1d_mesh(
                self.key, nx, nt, self.lb, self.ub, t0_val, tT_val
            ).astype(self.dtype)
        elif method == 'uniform':
            self.key, subkey = random.split(self.key)
            return inner_point_1d_uniform(
                subkey, nx, nt, self.lb, self.ub, t0_val, tT_val
            ).astype(self.dtype)
        elif method == 'hypercube':
            # Use scipy for LHS
            T = np.linspace(t0_val, tT_val, nt).repeat(nx, axis=0)
            X = qmc.scale(self.lhs_x.random(nx * nt), self.lb, self.ub)
            XT = np.vstack((X.flatten(), T.flatten())).T
            return jnp.array(XT, dtype=self.dtype)
        else:
            raise NotImplementedError

    def boundary_point(
            self,
            num_sample: int,
            t0: list[float] = [0.],
            tT: list[float] = [1.]
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Boundary points"""
        t0_val = t0[0] if isinstance(t0, list) else t0
        tT_val = tT[0] if isinstance(tT, list) else tT
        return boundary_point_1d(self.key, num_sample, self.lb, self.ub, t0_val, tT_val)

    def init_point(
            self,
            num_sample: int,
            t_stamp: list[float] = [0.],
            method: Literal['mesh', 'uniform', 'hypercube'] = 'mesh'
    ) -> jnp.ndarray:
        """Initial condition points"""
        t_val = t_stamp[0] if isinstance(t_stamp, list) else t_stamp

        if method == 'mesh':
            return init_point_1d_mesh(
                self.key, num_sample, self.lb, self.ub, t_val
            ).astype(self.dtype)
        elif method == 'uniform':
            self.key, subkey = random.split(self.key)
            return init_point_1d_uniform(
                subkey, num_sample, self.lb, self.ub, t_val
            ).astype(self.dtype)
        elif method == 'hypercube':
            X = qmc.scale(self.lhs_x.random(num_sample), self.lb, self.ub)
            T = np.full((num_sample, 1), t_val)
            XT = np.hstack((X, T))
            return jnp.array(XT, dtype=self.dtype)
        else:
            raise NotImplementedError

    def weight_centers(
            self,
            n_center: int,
            nt: int,
            Rmax: float = 1e-4,
            Rmin: float = 1e-4,
            method: Literal['mesh', 'uniform', 'hypercube'] = 'hypercube',
            t0: list[float] = [0.],
            tT: list[float] = [1.]
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Generate weight centers"""
        t0_val = t0[0] if isinstance(t0, list) else t0
        tT_val = tT[0] if isinstance(tT, list) else tT

        # Validation
        if Rmax < Rmin:
            raise ValueError('R_max should be larger than R_min.')
        if (2. * Rmax) > (self.ub - self.lb):
            raise ValueError('R_max is too large.')
        if (Rmin) < 1e-4 and self.dtype is jnp.float32:
            raise ValueError('R_min<1e-4 when data_type is jnp.float32!')
        if (Rmin) < 1e-10 and self.dtype is jnp.float64:
            raise ValueError('R_min<1e-10 when data_type is jnp.float64!')

        if method == 'hypercube':
            # Use scipy for LHS - spatial is LHS, temporal is regular grid
            R = np.random.uniform(Rmin, Rmax, [n_center * nt, 1])
            T = np.linspace(t0_val, tT_val, nt).repeat(n_center, axis=0)
            X = self.lhs_x.random(n_center * nt)

            lb = self.lb + R
            ub = self.ub - R
            xc = X * (ub - lb) + lb

            return (
                jnp.array(xc, dtype=self.dtype).reshape(-1, 1, 1),
                jnp.array(T, dtype=self.dtype).reshape(-1, 1, 1),
                jnp.array(R, dtype=self.dtype).reshape(-1, 1, 1)
            )
        else:
            # Use JAX version
            self.key, subkey = random.split(self.key)
            return weight_centers_1d(
                subkey, n_center, nt, self.lb, self.ub, Rmax, Rmin, t0_val, tT_val
            )

    def integral_grid(self, n_mesh_or_grid: int = 9, method: str = 'mesh') -> jnp.ndarray:
        """Integral grid"""
        if method == 'mesh':
            return integral_grid_1d(self.key, n_mesh_or_grid).astype(self.dtype)
        else:
            raise NotImplementedError


class Point2D:
    """Wrapper for 2D space-time point generation"""

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
        # Keep scipy for LHS
        self.lhs_t = qmc.LatinHypercube(1, seed=random_seed)
        self.lhs_x = qmc.LatinHypercube(2, seed=random_seed)

    def inner_point(
            self,
            nx: int,
            nt: int,
            method: Literal['mesh', 'hypercube'] = 'hypercube',
            t0: list[float] = [0.],
            tT: list[float] = [1.]
    ) -> jnp.ndarray:
        """Inner points"""
        t0_val = t0[0] if isinstance(t0, list) else t0
        tT_val = tT[0] if isinstance(tT, list) else tT

        if method == 'mesh':
            return inner_point_2d_mesh(
                self.key, nx, nt, self.lb, self.ub, t0_val, tT_val
            ).astype(self.dtype)
        elif method == 'hypercube':
            # Use scipy for LHS
            T = np.linspace(t0_val, tT_val, nt).repeat(nx, axis=0)
            X = qmc.scale(self.lhs_x.random(nx * nt), self.lb, self.ub)
            XT = np.concatenate([X, T.reshape(-1, 1)], axis=-1)
            return jnp.array(XT, dtype=self.dtype)
        else:
            raise NotImplementedError

    def boundary_point(
            self,
            nx_each_edge: int,
            nt: int,
            t0: list[float] = [0.],
            tT: list[float] = [1.],
            method: Literal['mesh', 'uniform', 'hypercube'] = 'hypercube'
    ) -> jnp.ndarray:
        """Boundary points"""
        t0_val = t0[0] if isinstance(t0, list) else t0
        tT_val = tT[0] if isinstance(tT, list) else tT

        if method == 'mesh':
            return boundary_point_2d(
                self.key, nx_each_edge, nt, self.lb, self.ub, t0_val, tT_val, method
            ).astype(self.dtype)
        elif method in ['uniform', 'hypercube']:
            # Use scipy for LHS/uniform
            T = np.linspace(t0_val, tT_val, nt).repeat(nx_each_edge * 4, axis=0)
            X_bd = []
            for _ in range(nt):
                for d in range(2):
                    if method == 'hypercube':
                        X_lb = qmc.scale(self.lhs_x.random(nx_each_edge), self.lb, self.ub)
                        X_ub = qmc.scale(self.lhs_x.random(nx_each_edge), self.lb, self.ub)
                    else:  # uniform
                        X_lb = np.random.uniform(self.lb, self.ub, (nx_each_edge, 2))
                        X_ub = np.random.uniform(self.lb, self.ub, (nx_each_edge, 2))
                    X_lb[:, d] = self.lb[d]
                    X_bd.append(X_lb)
                    X_ub[:, d] = self.ub[d]
                    X_bd.append(X_ub)
            X_bd = np.concatenate(X_bd, axis=0)
            XT = np.concatenate([X_bd, T.reshape(-1, 1)], axis=-1)
            return jnp.array(XT, dtype=self.dtype)
        else:
            raise NotImplementedError

    def init_point(
            self,
            nx_or_mesh: int,
            t_stamp: list[float] = [0.],
            method: Literal['mesh', 'uniform', 'hypercube'] = 'mesh'
    ) -> jnp.ndarray:
        """Initial condition points"""
        t_val = t_stamp[0] if isinstance(t_stamp, list) else t_stamp

        if method == 'mesh':
            return init_point_2d_mesh(
                self.key, nx_or_mesh, self.lb, self.ub, t_val
            ).astype(self.dtype)
        elif method in ['uniform', 'hypercube']:
            if method == 'hypercube':
                X = qmc.scale(self.lhs_x.random(nx_or_mesh), self.lb, self.ub)
            else:
                x_rand = np.random.uniform(self.lb[0], self.ub[0], nx_or_mesh)
                y_rand = np.random.uniform(self.lb[1], self.ub[1], nx_or_mesh)
                X = np.stack([x_rand, y_rand], axis=1)
            T = np.full((X.shape[0],), t_val)
            XT = np.concatenate([X, T.reshape(-1, 1)], axis=-1)
            return jnp.array(XT, dtype=self.dtype)
        else:
            raise NotImplementedError

    def weight_centers(
            self,
            n_center: int,
            nt: int,
            Rmax: float = 1e-4,
            Rmin: float = 1e-4,
            t0: list[float] = [0.],
            tT: list[float] = [1.]
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Generate weight centers"""
        t0_val = t0[0] if isinstance(t0, list) else t0
        tT_val = tT[0] if isinstance(tT, list) else tT

        # Validation
        if Rmax < Rmin:
            raise ValueError('R_max should be larger than R_min.')
        lb_arr = jnp.array(self.lb)
        ub_arr = jnp.array(self.ub)
        if (2. * Rmax) > jnp.min(ub_arr - lb_arr):
            raise ValueError('R_max is too large.')
        if (Rmin) < 1e-4 and self.dtype is jnp.float32:
            raise ValueError('R_min<1e-4 when data_type is jnp.float32!')
        if (Rmin) < 1e-10 and self.dtype is jnp.float64:
            raise ValueError('R_min<1e-10 when data_type is jnp.float64!')

        # Use scipy for LHS
        R = np.random.uniform(Rmin, Rmax, [n_center * nt, 1])
        lb = np.array(self.lb) + R
        ub = np.array(self.ub) - R

        T = np.linspace(t0_val, tT_val, nt).repeat(n_center, axis=0)
        X = self.lhs_x.random(n_center * nt)
        xc = X * (ub - lb) + lb

        return (
            jnp.array(xc, dtype=self.dtype).reshape(-1, 1, 2),
            jnp.array(T, dtype=self.dtype).reshape(-1, 1, 1),
            jnp.array(R, dtype=self.dtype).reshape(-1, 1, 1)
        )

    def integral_grid(self, n_mesh_or_grid: int = 9, method: str = 'mesh') -> jnp.ndarray:
        """Integral grid"""
        if method == 'mesh':
            x_mesh, y_mesh = jnp.meshgrid(
                jnp.linspace(-1., 1., n_mesh_or_grid),
                jnp.linspace(-1., 1., n_mesh_or_grid)
            )
            grid = jnp.stack([x_mesh.reshape(-1), y_mesh.reshape(-1)], axis=1)
            mask = jnp.linalg.norm(grid, axis=1) < 1.
            return grid[mask, :].astype(self.dtype)
        else:
            raise NotImplementedError