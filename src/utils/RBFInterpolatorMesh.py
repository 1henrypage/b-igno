import jax
import jax.numpy as jnp
from jax import jit, vmap
from typing import Literal
import math
from itertools import combinations_with_replacement
from functools import partial


class RadialFun:
    """Radial basis functions - all static, JIT-friendly"""

    @staticmethod
    @jit
    def linear(r: jnp.ndarray) -> jnp.ndarray:
        return r

    @staticmethod
    @jit
    def thin_plate_spline(r: jnp.ndarray, min_eps: float = 1e-7) -> jnp.ndarray:
        return jnp.maximum(r, min_eps)

    @staticmethod
    @jit
    def cubic(r: jnp.ndarray) -> jnp.ndarray:
        return r ** 3

    @staticmethod
    @jit
    def quintic(r: jnp.ndarray) -> jnp.ndarray:
        return -r ** 5

    @staticmethod
    @jit
    def multiquadric(r: jnp.ndarray) -> jnp.ndarray:
        return -jnp.sqrt(r ** 2 + 1)

    @staticmethod
    @jit
    def inverse_multiquadric(r: jnp.ndarray) -> jnp.ndarray:
        return 1 / jnp.sqrt(r ** 2 + 1)

    @staticmethod
    @jit
    def inverse_quadratic(r: jnp.ndarray) -> jnp.ndarray:
        return 1 / (r ** 2 + 1)

    @staticmethod
    @jit
    def gaussian(r: jnp.ndarray) -> jnp.ndarray:
        return jnp.exp(-r ** 2)


# Pure functional versions that can be vmapped

def kernel_matrix_single(x_eps: jnp.ndarray, x_eps_base: jnp.ndarray, kernel_fn) -> jnp.ndarray:
    # distances: (nx, n_mesh)
    distances = jnp.linalg.norm(x_eps[:, None, :] - x_eps_base[None, :, :], axis=-1)
    return kernel_fn(distances)


@jit
def polynomial_matrix_single(x_hat: jnp.ndarray, powers: jnp.ndarray) -> jnp.ndarray:
    # x_: (nx, n_monos, dx)
    x_ = x_hat[:, None, :] ** powers[None, :, :]
    # Product over dx: (nx, n_monos)
    return jnp.prod(x_, axis=-1)


polynomial_matrix_batched = vmap(polynomial_matrix_single, in_axes=(0, None))


@partial(jit, static_argnums=(3,))
def solve_rbf_system_single(
        lhs: jnp.ndarray,
        a_values: jnp.ndarray,
        n_mesh: int,
        n_monos: int
) -> jnp.ndarray:
    rhs = jnp.zeros((n_mesh + n_monos, 1), dtype=a_values.dtype)
    rhs = rhs.at[:n_mesh, :].set(a_values)
    return jnp.linalg.solve(lhs, rhs)


# Vectorize over batch dimension
solve_rbf_system_batched = vmap(solve_rbf_system_single, in_axes=(None, 0, None, None))


@partial(jit, static_argnums=())
def interpolate_single(
        vec: jnp.ndarray,
        coeff: jnp.ndarray
) -> jnp.ndarray:
    return vec @ coeff


# Vectorize over batch dimension
interpolate_batched = vmap(interpolate_single, in_axes=(0, 0))


class RBFInterpolator:
    """RBF interpolator — precomputes the system matrix at init, solves per call."""

    def __init__(
            self,
            x_mesh: jnp.ndarray,
            kernel: str = "thin_plate_spline",
            eps: float = None,
            degree: int = None,
            smoothing: float = 0.,
            dtype=jnp.float32
    ):
        assert x_mesh.ndim == 2

        self.x_mesh = x_mesh
        self.n_mesh, self.dx = x_mesh.shape
        self.dtype = dtype

        # Setup kernel function
        scale_fun = {"linear", "thin_plate_spline", "cubic", "quintic"}
        self.kernel_name = kernel

        # Get kernel function
        kernel_fns = {
            "linear": RadialFun.linear,
            "thin_plate_spline": RadialFun.thin_plate_spline,
            "cubic": RadialFun.cubic,
            "quintic": RadialFun.quintic,
            "multiquadric": RadialFun.multiquadric,
            "inverse_multiquadric": RadialFun.inverse_multiquadric,
            "inverse_quadratic": RadialFun.inverse_quadratic,
            "gaussian": RadialFun.gaussian
        }
        self.kernel_fun = kernel_fns[kernel]

        # Setup eps
        if eps is None:
            if kernel in scale_fun:
                self.eps = 1.
            else:
                raise ValueError('Require eps for the kernel.')
        else:
            self.eps = float(eps)

        # Setup smoothing
        if isinstance(smoothing, (int, float)):
            self.smoothing = jnp.full((self.n_mesh,), smoothing, dtype=dtype)
        else:
            self.smoothing = jnp.array(smoothing, dtype=dtype)

        # Setup degree
        min_degree_dict = {
            "multiquadric": 0,
            "linear": 0,
            "thin_plate_spline": 1,
            "cubic": 1,
            "quintic": 2
        }
        min_degree = min_degree_dict.get(kernel, -1)

        if degree is None:
            degree = max(min_degree, 0)
        else:
            degree = int(degree)
            if degree < -1:
                raise ValueError('degree must be at least -1')
            elif degree < min_degree:
                raise ValueError(f'degree must be larger than {min_degree}')

        # Setup powers
        self.powers = self.monomial_powers(self.dx, degree)
        self.n_monos = self.powers.shape[0]
        if self.n_monos > self.n_mesh:
            raise ValueError('The data is not compatible with the requested degree')

        # Build the interpolation matrix (done once at initialization)
        self.lhs, self.shift, self.scale = self.build()
        self.x_eps_mesh = self.x_mesh * self.eps

    def __call__(self, x: jnp.ndarray, a_batch: jnp.ndarray) -> jnp.ndarray:
        assert x.ndim == 3 and a_batch.ndim == 3
        assert a_batch.shape[1] == self.n_mesh

        # Capture constants in closure to avoid tracing issues
        lhs = self.lhs
        n_mesh = self.n_mesh
        n_monos = self.n_monos
        x_eps_mesh = self.x_eps_mesh
        kernel_fun = self.kernel_fun
        powers = self.powers
        shift = self.shift
        scale = self.scale
        eps = self.eps

        def solve_single(a_values):
            """Solve for single sample - n_mesh/n_monos captured via closure"""
            rhs = jnp.zeros((n_mesh + n_monos, 1), dtype=a_values.dtype)
            rhs = rhs.at[:n_mesh, :].set(a_values)
            return jnp.linalg.solve(lhs, rhs)

        def kernel_single(x_e):
            """Kernel matrix for single sample"""
            distances = jnp.linalg.norm(x_e[:, None, :] - x_eps_mesh[None, :, :], axis=-1)
            return kernel_fun(distances)

        def poly_single(x_h):
            """Polynomial matrix for single sample"""
            x_ = x_h[:, None, :] ** powers[None, :, :]
            return jnp.prod(x_, axis=-1)

        # Solve for coefficients (vectorized over batch)
        coeff = vmap(solve_single)(a_batch)

        # Prepare query points
        x_eps = x * eps
        x_hat = (x - shift) / scale

        # Compute kernel matrix (vectorized over batch)
        kv = vmap(kernel_single)(x_eps)

        # Compute polynomial matrix (vectorized over batch)
        pmat = vmap(poly_single)(x_hat)

        # Concatenate
        vec = jnp.concatenate([kv, pmat], axis=-1)

        # Interpolate (vectorized over batch)
        a_pred = vmap(lambda v, c: v @ c)(vec, coeff)

        return a_pred


    def build(self):
        mins = jnp.min(self.x_mesh, axis=0)
        maxs = jnp.max(self.x_mesh, axis=0)
        shift = (maxs + mins) / 2.
        scale = (maxs - mins) / 2.
        scale = jnp.where(scale == 0., 1., scale)

        x_eps = self.x_mesh * self.eps
        x_hat = (self.x_mesh - shift) / scale

        lhs = jnp.zeros((self.n_mesh + self.n_monos, self.n_mesh + self.n_monos), dtype=self.dtype)

        kernel_mat = kernel_matrix_single(x_eps, x_eps, self.kernel_fun)
        lhs = lhs.at[:self.n_mesh, :self.n_mesh].set(kernel_mat)

        pmat = polynomial_matrix_single(x_hat, self.powers)
        lhs = lhs.at[:self.n_mesh, self.n_mesh:].set(pmat)
        lhs = lhs.at[self.n_mesh:, :self.n_mesh].set(pmat.T)

        lhs = lhs.at[:self.n_mesh, :self.n_mesh].add(jnp.diag(self.smoothing))

        return lhs, shift, scale

    @staticmethod
    def monomial_powers(dx: int, degree: int) -> jnp.ndarray:
        n_monos = math.comb(degree + dx, dx)
        out = jnp.zeros((n_monos, dx), dtype=jnp.int32)
        count = 0
        for deg in range(degree + 1):
            for mono in combinations_with_replacement(range(dx), deg):
                for var in mono:
                    out = out.at[count, var].add(1)
                count += 1
        return out
