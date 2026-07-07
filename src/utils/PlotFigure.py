import matplotlib
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import scienceplots

plt.rcParams['axes.titlesize'] = 26
plt.rcParams['axes.labelsize'] = 24
plt.rcParams['xtick.labelsize'] = 18
plt.rcParams['ytick.labelsize'] = 18
plt.rcParams['legend.fontsize'] = 24

colormap = 'jet'


def _to_numpy(arr):
    if isinstance(arr, jnp.ndarray):
        return np.array(arr)
    return arr


class Plot:
    """Plotting utilities that accept both JAX and numpy arrays"""

    @staticmethod
    def show_error(
            time_list: list[jnp.ndarray | np.ndarray],
            error_list: list[jnp.ndarray | np.ndarray],
            label_list: list[str],
            save_path: str = None,
            title: str = None
    ) -> None:
        with plt.style.context(['science', 'no-latex']):
            plt.figure(figsize=(7, 5))
            for time, error, label in zip(time_list, error_list, label_list):
                time_np = _to_numpy(time)
                error_np = _to_numpy(error)
                plt.semilogy(time_np.ravel(), error_np.ravel(), linewidth=1.5, label=label)
            plt.xlabel('time(s)')
            plt.ylabel(r'Relative error (avg)')
            plt.tight_layout()
            plt.legend()

        if title is not None:
            plt.title(title)

        if save_path is not None:
            plt.savefig(save_path)
            plt.close()

        plt.show()

    @staticmethod
    def show_loss(
            loss_list: list[jnp.ndarray | np.ndarray],
            label_list: list[str],
            save_path: str = None
    ) -> None:
        with plt.style.context(['science', 'no-latex']):
            plt.figure(figsize=(7, 5))
            for loss, label in zip(loss_list, label_list):
                loss_np = _to_numpy(loss)
                plt.semilogy(loss_np.ravel(), linewidth=1.5, label=label)
            plt.xlabel('iter')
            plt.ylabel('loss')
            plt.tight_layout()
            plt.legend()

        if save_path is not None:
            plt.savefig(save_path)
            plt.close()

        plt.show()

    @staticmethod
    def show_1d_list(
            x_list,
            y_list,
            label_list,
            title=None,
            pause=None,
            save_path=None,
            point_list=None,
            lb=-1.,
            ub=1.
    ):
        """Plot 1D functions"""
        x_plot = np.linspace(lb, ub, 100)
        plt.figure(figsize=(8, 5))

        if type(x_list).__name__ != 'list':
            x_list = [x_list] * len(y_list)

        with plt.style.context(['science', 'no-latex']):
            for x, y, label in zip(x_list, y_list, label_list):
                x_np = _to_numpy(x)
                y_np = _to_numpy(y)
                y_plot = griddata(x_np.flatten(), y_np.flatten(), x_plot, method='cubic')
                plt.plot(x_plot, y_plot, '-.', linewidth=3., label=label)

            # Plot the points
            if point_list is not None:
                for points in point_list:
                    points_np = _to_numpy(points)
                    plt.scatter(points_np[:, 0], np.zeros_like(points_np[:, 0]), s=20, lw=1.)

            if title is not None:
                plt.title(title)

            plt.xlabel('x')
            plt.ylabel('y')
            plt.tight_layout()
            plt.legend()

        if save_path is not None:
            plt.savefig(save_path)
            plt.close()

        if pause is not None:
            plt.pause(pause)
            plt.close()

        plt.show()

    @staticmethod
    def show_1dt(
            xt,
            u,
            title=None,
            pause=None,
            save_path=None,
            point_list=None,
            t0=0.,
            tT=1.,
            lb=-1.,
            ub=1.
    ):
        """Plot 1D space-time solution"""
        xt_np = _to_numpy(xt)
        u_np = _to_numpy(u)

        mesh = np.meshgrid(np.linspace(lb, ub, 100), np.linspace(t0, tT, 200))
        x_plot, t_plot = mesh[0], mesh[1]
        x, t = xt_np[..., 0], xt_np[..., -1]

        fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(8, 3))

        z_plot = griddata((t.flatten(), x.flatten()), np.ravel(u_np), (t_plot, x_plot), method='cubic')
        cntr = axs.contourf(t_plot, x_plot, z_plot, levels=14, cmap=colormap)
        fig.colorbar(cntr, ax=axs)
        axs.set_xlabel('t')
        axs.set_ylabel('x')

        if point_list is not None:
            for points in point_list:
                points_np = _to_numpy(points)
                plt.scatter(points_np[:, -1], points_np[:, 0], s=20, lw=2., marker='o')

        if title is not None:
            axs.set_title(title)

        if save_path is not None:
            plt.savefig(save_path)
            plt.close()

        if pause is not None:
            plt.pause(pause)
            plt.close()

        plt.show()

    @staticmethod
    def show_2d(
            x,
            y,
            title=None,
            pause=None,
            save_path=None,
            point_list=None,
            lb=-1.,
            ub=1.
    ):
        """Plot 2D field"""
        x_np = _to_numpy(x)
        y_np = _to_numpy(y)

        if isinstance(lb, list):
            mesh = np.meshgrid(
                np.linspace(lb[0], ub[0], 100),
                np.linspace(lb[1], ub[1], 100)
            )
        else:
            mesh = np.meshgrid(
                np.linspace(lb, ub, 100),
                np.linspace(lb, ub, 100)
            )
        x_plot, y_plot = mesh[0], mesh[1]

        with plt.style.context(['science', 'no-latex']):
            fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(7, 5))

            z_plot = griddata((x_np[:, 0], x_np[:, 1]), np.ravel(y_np), (x_plot, y_plot), method='cubic')
            cntr = axs.contourf(x_plot, y_plot, z_plot, levels=14, cmap=colormap)
            fig.colorbar(cntr, ax=axs)

            if point_list is not None:
                for points in point_list:
                    points_np = _to_numpy(points)
                    plt.scatter(points_np[:, 0], points_np[:, 1], s=20, lw=2., marker='o')

            axs.set_xlabel('x')
            axs.set_ylabel('y')
            plt.tight_layout()

        if title is not None:
            axs.set_title(title)

        if save_path is not None:
            plt.savefig(save_path)
            plt.close()

        if pause is not None:
            plt.pause(pause)
            plt.close()

        plt.show()

    @staticmethod
    def show_2d_list(
            x_list,
            y_list,
            label_list,
            pause=None,
            save_path=None,
            lb=-1.,
            ub=1.
    ):
        """Plot list of 2D fields"""
        n_col = len(y_list)
        if type(x_list).__name__ != 'list':
            x_list = [x_list] * n_col

        if isinstance(lb, list):
            mesh = np.meshgrid(
                np.linspace(lb[0], ub[0], 100),
                np.linspace(lb[1], ub[1], 100)
            )
        else:
            mesh = np.meshgrid(
                np.linspace(lb, ub, 100),
                np.linspace(lb, ub, 100)
            )
        x_plot, y_plot = mesh[0], mesh[1]

        with plt.style.context(['science', 'no-latex']):
            fig, axs = plt.subplots(nrows=1, ncols=n_col, figsize=(6 * n_col, 5))
            for i, x, y, title in zip(range(n_col), x_list, y_list, label_list):
                x_np = _to_numpy(x)
                y_np = _to_numpy(y)

                z_plot = griddata(
                    (x_np[:, 0], x_np[:, 1]),
                    np.ravel(y_np),
                    (x_plot, y_plot),
                    method='cubic'
                )
                cntr = axs.flat[i].contourf(x_plot, y_plot, z_plot, levels=14, cmap=colormap)
                fig.colorbar(cntr, ax=axs.flat[i])

                axs.flat[i].set_title(title)
                axs.flat[i].set_xlabel('x')
                axs.flat[i].set_ylabel('y')

            plt.tight_layout()

        if save_path is not None:
            plt.savefig(save_path)
            plt.close()

        if pause is not None:
            plt.pause(pause)
            plt.close()

        plt.show()

    @staticmethod
    def contourf_on_ax(ax, fig, x, values, cmap='jet', vmin=None, vmax=None, lb=0., ub=1., levels=100, norm=None):
        """Render a 2D scatter field as smooth contourf using cubic griddata interpolation.

        Args:
            ax: matplotlib axis
            fig: parent figure (for colorbar)
            x: (n_points, 2) coordinate array
            values: (n_points,) field values
            cmap: colormap string or Colormap object
            vmin, vmax: color scale limits (None = auto)
            lb, ub: spatial domain bounds
            levels: number of contour levels
            norm: optional matplotlib Normalize instance (e.g. BoundaryNorm for discrete colormaps)

        Returns:
            contour object from ax.contourf
        """
        x_np = _to_numpy(x)
        v_np = _to_numpy(values).ravel()

        mesh = np.meshgrid(np.linspace(lb, ub, 100), np.linspace(lb, ub, 100))
        x_plot, y_plot = mesh[0], mesh[1]

        z_plot = griddata((x_np[:, 0], x_np[:, 1]), v_np, (x_plot, y_plot), method='cubic')
        kwargs = dict(cmap=cmap)
        if norm is not None:
            kwargs['norm'] = norm
            kwargs['levels'] = norm.boundaries
        else:
            kwargs['levels'] = levels
            if vmin is not None:
                kwargs['vmin'] = vmin
            if vmax is not None:
                kwargs['vmax'] = vmax
        cntr = ax.contourf(x_plot, y_plot, z_plot, **kwargs)
        cbar = fig.colorbar(cntr, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=11)
        return cntr

    @staticmethod
    def imshow_on_ax(ax, fig, grid_2d, cmap='jet', vmin=None, vmax=None, lb=0., ub=1., norm=None):
        """Render a pre-gridded 2D array with nearest-neighbour imshow (no interpolation artifacts).

        Use this for piecewise-constant fields where cubic griddata creates Gibbs-like streaks.

        Args:
            ax: matplotlib axis
            fig: parent figure (for colorbar)
            grid_2d: (H, W) 2D array already on a regular grid (row 0 = top of domain)
            cmap: colormap string or Colormap object
            vmin, vmax: color scale limits (None = auto)
            lb, ub: spatial domain bounds (used for extent)
            norm: optional matplotlib Normalize instance

        Returns:
            AxesImage object from ax.imshow
        """
        g = _to_numpy(grid_2d)
        kwargs = dict(
            cmap=cmap,
            interpolation='nearest',
            origin='lower',
            extent=[lb, ub, lb, ub],
            aspect='equal',
        )
        if norm is not None:
            kwargs['norm'] = norm
        else:
            if vmin is not None:
                kwargs['vmin'] = vmin
            if vmax is not None:
                kwargs['vmax'] = vmax
        im = ax.imshow(g, **kwargs)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=11)
        return im

    @staticmethod
    def show_3d_list(
            x_list: jnp.ndarray | np.ndarray | list,
            Z_list: list,
            label_list: list,
            pause: float = None,
            save_path: str = None,
            lb=-1.,
            ub=1.
    ) -> None:
        """Plot 3D surfaces"""
        n_col = int(np.ceil(len(Z_list) / 2))
        fig, axs = plt.subplots(
            nrows=n_col,
            ncols=2,
            figsize=(11, 6),
            subplot_kw={"projection": "3d"}
        )

        if isinstance(lb, list):
            mesh = np.meshgrid(
                np.linspace(lb[0], ub[0], 100),
                np.linspace(lb[1], ub[1], 100)
            )
        else:
            mesh = np.meshgrid(
                np.linspace(lb, ub, 100),
                np.linspace(lb, ub, 100)
            )
        x_plot, y_plot = mesh[0], mesh[1]

        if type(x_list).__name__ != 'list':
            x_list = [x_list] * len(Z_list)

        for i in range(len(Z_list)):
            x_np = _to_numpy(x_list[i])
            Z_np = _to_numpy(Z_list[i])

            z_plot = griddata(
                (x_np[:, 0], x_np[:, 1]),
                np.ravel(Z_np),
                (x_plot, y_plot),
                method='linear'
            )
            axs.flat[i].plot_surface(
                x_plot, y_plot, z_plot,
                cmap=cm.coolwarm,
                linewidth=0,
                antialiased=False
            )

            axs.flat[i].set_title(label_list[i])
            axs.flat[i].set_xlabel('x')
            axs.flat[i].set_ylabel('y')
            axs.flat[i].zaxis.set_major_locator(LinearLocator(5))
            axs.flat[i].zaxis.set_major_formatter('{x:.02f}')

        plt.tight_layout()

        if save_path is not None:
            plt.savefig(save_path)
            plt.close()

        if pause is not None:
            plt.pause(pause)
            plt.close()

        plt.show()
