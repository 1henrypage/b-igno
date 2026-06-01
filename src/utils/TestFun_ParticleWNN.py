import jax
import jax.numpy as jnp
from jax import grad
import math
from typing import Literal, Tuple
from functools import partial


class TestFun_ParticleWNN:
    """Test functions for Particle Weak Neural Network

    Supports: Bump, Wendland, Cosin, Wendland with power k
    """

    def __init__(
            self,
            fun_type: Literal['Cosin', 'Bump', 'Wendland', 'Wendland_k'] = 'Cosin',
            dim: int = 1,
            n_mesh_or_grid: int = 9,
            grid_method: Literal['mesh'] = 'mesh',
            dataType=jnp.float32
    ):
        self._dim = dim
        self._eps = jnp.finfo(jnp.float32).eps
        self._n_mesh_or_grid = n_mesh_or_grid
        self._grid_method = grid_method
        self._dtype = dataType

        fun_dict = {
            "Cosin": self._Cosin,
            "Bump": self._Bump,
            "Wendland": self._Wendland,
            "Wendland_k": self._Wend_powerK
        }

        if fun_type in fun_dict.keys():
            self.testFun = fun_dict[fun_type]
        else:
            raise NotImplementedError(f'No {fun_type} test function type.')

    def _dist(self, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.linalg.norm(x, axis=-1, keepdims=True)

    def _Bump(self, x_mesh: jnp.ndarray, dim: int = 1) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Bump test function"""
        r = 1. - jnp.maximum(0., 1. - self._dist(x_mesh))
        r2 = r * r
        r4 = r2 * r2

        v = jnp.exp(1. - 1. / (1. - r2 + self._eps))
        dv_dr_divide_by_r = v * (-2.) / ((1. - r2) ** 2 + self._eps)

        if dim == 1:
            dv = dv_dr_divide_by_r * r * jnp.sign(x_mesh)
        else:
            dv = dv_dr_divide_by_r * x_mesh

        return v, dv

    def _Wendland(self, x_mesh: jnp.ndarray, dim: int = 1) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Wendland test function"""
        l = math.floor(dim / 2) + 3

        r = 1. - jnp.maximum(0., 1. - self._dist(x_mesh))
        r2 = r * r

        v = ((1 - r) ** (l + 2) *
             ((l ** 2 + 4. * l + 3.) * r2 + (3. * l + 6.) * r + 3.) / 3.)

        dv_dr_divide_by_r = ((1 - r) ** (l + 1) *
                             (-(l ** 3 + 8. * l ** 2 + 19. * l + 12) * r -
                              (l ** 2 + 7. * l + 12)) / 3.)

        if dim == 1:
            dv = dv_dr_divide_by_r * r * jnp.sign(x_mesh)
        else:
            dv = dv_dr_divide_by_r * x_mesh

        return v, dv

    def _Cosin(self, x_mesh: jnp.ndarray, dim: int = 1) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Cosine test function"""
        r = 1. - jnp.maximum(0., 1. - self._dist(x_mesh))
        v = (1. - jnp.cos(jnp.pi * (r + 1.))) / jnp.pi

        dv_dr_divide_by_r = jnp.sin(jnp.pi * (r + 1.)) / (r + self._eps)

        if dim == 1:
            dv = dv_dr_divide_by_r * r * jnp.sign(x_mesh)
        else:
            dv = dv_dr_divide_by_r * x_mesh

        return v, dv

    def _Wend_powerK(
            self,
            x_mesh: jnp.ndarray,
            dim: int = 1,
            k: int = 4
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Wendland with power k test function (k >= 2)"""
        l = math.floor(dim / 2) + 3

        r = 1. - jnp.maximum(0., 1. - self._dist(x_mesh))
        r2 = r * r

        v_wend = ((1 - r) ** (l + 2) *
                  ((l ** 2 + 4. * l + 3.) * r2 + (3. * l + 6.) * r + 3.) / 3.)
        dv_dr_divide_by_r_wend = ((1 - r) ** (l + 1) *
                                  (-(l ** 3 + 8. * l ** 2 + 19. * l + 12) * r -
                                   (l ** 2 + 7. * l + 12)) / 3.)

        v = v_wend ** k
        dv_dr_divide_by_r = k * v_wend ** (k - 1) * dv_dr_divide_by_r_wend

        if dim == 1:
            dv = dv_dr_divide_by_r * r * jnp.sign(x_mesh)
        else:
            dv = dv_dr_divide_by_r * x_mesh

        return v, dv

    def integral_grid(
            self,
            n_mesh_or_grid: int,
            method: Literal['mesh'] = 'mesh',
            dtype=jnp.float32
    ) -> jnp.ndarray:
        """Generate integration grid in [-1, 1]^d

        Args:
            n_mesh_or_grid: Number of grid points
            method: Grid generation method
            dtype: Data type

        Returns:
            Grid points, shape (?, d)
        """
        if method == 'mesh':
            if self._dim == 1:
                grid_scaled = jnp.linspace(-1., 1., n_mesh_or_grid).reshape(-1, 1)
            elif self._dim == 2:
                x_mesh, y_mesh = jnp.meshgrid(
                    jnp.linspace(-1., 1., n_mesh_or_grid),
                    jnp.linspace(-1., 1., n_mesh_or_grid)
                )
                grid = jnp.stack([x_mesh.reshape(-1), y_mesh.reshape(-1)], axis=1)

                # Filter points inside unit circle
                mask = jnp.linalg.norm(grid, axis=1) < 1.
                grid_scaled = grid[mask, :]
            else:
                raise NotImplementedError(f'dim>{self._dim} is not available')
        else:
            raise NotImplementedError(f'No {method} method.')

        return grid_scaled.astype(dtype)

    def get_testFun(
            self,
            grids: jnp.ndarray = None
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Get test function values and gradients

        Args:
            grids: Query points, shape (?, d). If None, use default grid.

        Returns:
            grids: Grid points
            v: Function values
            dv: Gradients
        """
        if grids is None:
            grids = self.integral_grid(
                self._n_mesh_or_grid,
                self._grid_method,
                self._dtype
            )

        v, dv = self.testFun(grids, self._dim)

        return grids, v, dv