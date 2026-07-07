"""Publication-quality plotting functions."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from scipy.stats import norm, pearsonr, spearmanr

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.utils.PlotFigure import Plot

from ._metrics import mcmc_reliability_flag

_PIECEWISE_A_CMAP = ListedColormap(['#4477AA', '#EE7733'])  # blue, orange
_UNCERTAINTY_CMAP = 'inferno'


def _piecewise_a_norm(bounds: tuple):
    """Create BoundaryNorm for a 2-value piecewise coefficient field."""
    low, high = bounds
    midpoint = (low + high) / 2
    return BoundaryNorm([low, midpoint, high], _PIECEWISE_A_CMAP.N)

def _use_science_style():
    """Context-manager-compatible style list."""
    available = plt.style.available
    styles = []
    if 'science' in available:
        styles.append('science')
    if 'no-latex' in available:
        styles.append('no-latex')
    return styles if styles else ['default']


def _reshape_scatter_to_grid(values: np.ndarray, grid_shape: Tuple[int, int]) -> np.ndarray:
    """Reshape flat scatter values to 2D grid for imshow rendering."""
    return values.reshape(grid_shape)


def _save_contourf(save_dir: Path, name: str, x, values, cmap='jet',
                   lb=0., ub=1., vmin=None, vmax=None, norm=None, overlay_fn=None,
                   grid_shape: Optional[Tuple[int, int]] = None):
    """Save a single field panel as an individual figure.

    When grid_shape and norm are both provided, uses imshow_on_ax for piecewise fields.
    Otherwise uses contourf_on_ax with cubic interpolation.
    """
    with plt.style.context(_use_science_style()):
        fig, ax = plt.subplots(figsize=(4, 4))
        if grid_shape is not None and norm is not None:
            Plot.imshow_on_ax(ax, fig, _reshape_scatter_to_grid(values, grid_shape),
                              cmap=cmap, vmin=vmin, vmax=vmax, lb=lb, ub=ub, norm=norm)
        else:
            Plot.contourf_on_ax(ax, fig, x, values, cmap=cmap, vmin=vmin, vmax=vmax,
                                lb=lb, ub=ub, norm=norm)
        ax.set_aspect('equal')
        ax.tick_params(labelsize=13)
        if overlay_fn:
            overlay_fn(ax)
        fig.savefig(save_dir / f'{name}.png', dpi=200, bbox_inches='tight')
        plt.close(fig)


def plot_metric_convergence(
    convergence_results: list,
    labels: Optional[List[str]] = None,
    save_path: Optional[Path] = None,
):
    """Plot posterior metrics vs number of samples (2x3 grid, 5 panels used)."""
    linestyles = ['-', '--', ':', '-.']

    panel_specs = [
        ('a_err',       'Rel. L2 (a)',              'C0'),
        ('coverage_95', '95% CI coverage',           'C2'),
        ('ci_width',    'Mean CI width (a)',          'C1'),
        ('crps_a',      'CRPS (a)',                   'C5'),
        ('mean_std',    'Mean posterior std (a)',      'C3'),
    ]

    with plt.style.context(_use_science_style()):
        fig, axes = plt.subplots(2, 3, figsize=(16, 9))
        axes[1, 2].set_visible(False)

        grid_positions = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)]

        for idx, ((key, ylabel, color), (r, c)) in enumerate(
            zip(panel_specs, grid_positions)
        ):
            ax = axes[r, c]
            for j, res in enumerate(convergence_results):
                ls = linestyles[j % len(linestyles)]
                lbl = labels[j] if labels else None
                ax.plot(res['sample_counts'], res[key], ls=ls, color=color, label=lbl)
            if key == 'coverage_95':
                ax.axhline(0.95, ls='--', color='grey', lw=1, alpha=0.6)
            ax.set_xlabel('Number of posterior samples', fontsize=14)
            ax.set_ylabel(ylabel, fontsize=14)
            ax.tick_params(labelsize=13)
            if labels and len(convergence_results) > 1:
                ax.legend(fontsize=9, framealpha=0.3, facecolor='white', edgecolor='black')

        plt.tight_layout()

        if save_path:
            save_path = Path(save_path)
            fig.savefig(save_path, dpi=200, bbox_inches='tight')

            save_dir = save_path.parent / save_path.stem
            save_dir.mkdir(parents=True, exist_ok=True)

            fname_map = {
                'a_err': 'a_err', 'coverage_95': 'coverage',
                'ci_width': 'ci_width', 'crps_a': 'crps', 'mean_std': 'mean_std',
            }
            for key, ylabel, color in panel_specs:
                with plt.style.context(_use_science_style()):
                    f2, ax2 = plt.subplots(figsize=(5, 4))
                    for j, res in enumerate(convergence_results):
                        ls = linestyles[j % len(linestyles)]
                        lbl = labels[j] if labels else None
                        ax2.plot(res['sample_counts'], res[key],
                                 ls=ls, color=color, label=lbl)
                    if key == 'coverage_95':
                        ax2.axhline(0.95, ls='--', color='grey', lw=1, alpha=0.6)
                    ax2.set_xlabel('Number of posterior samples', fontsize=14)
                    ax2.set_ylabel(ylabel, fontsize=14)
                    ax2.tick_params(labelsize=13)
                    if labels and len(convergence_results) > 1:
                        ax2.legend(fontsize=9, framealpha=0.3, facecolor='white', edgecolor='black')
                    f2.savefig(save_dir / f'{fname_map[key]}.png',
                               dpi=200, bbox_inches='tight')
                    plt.close(f2)

        plt.show()


def plot_field_comparison(
    x: np.ndarray,
    a_true: np.ndarray,
    a_map: np.ndarray,
    a_mean: np.ndarray,
    a_std: np.ndarray,
    grid_shape: Tuple[int, int],
    save_path: Optional[Path] = None,
    cmap: str = 'jet',
    lb: float = 0.,
    ub: float = 1.,
    u_true: Optional[np.ndarray] = None,
    u_map: Optional[np.ndarray] = None,
    u_mean: Optional[np.ndarray] = None,
    u_std: Optional[np.ndarray] = None,
    obs_coords: Optional[np.ndarray] = None,
    show_abs_error: bool = True,
    piecewise_a_bounds: Optional[Tuple[float, float]] = None,
):
    """Field comparison with scientific subplot labels.

    When show_abs_error is True (default), shows 5 columns:
      (a) Ground truth a  (b) MAP a  (c) Posterior mean a  (d) |Error| a  (e) Posterior std a
      (f) Ground truth u  (g) MAP u  (h) Posterior mean u  (i) Posterior std u  [hidden]

    When show_abs_error is False, shows 4 columns (no |Error| panel).

    When u_* arrays are provided, adds a second row for the solution field.
    If obs_coords is provided, overlays black dots on the u Truth panel.

    Args:
        x: (n_points, 2) coordinates
        a_true, a_map, a_mean, a_std: (n_points,) coefficient field values
        grid_shape: (rows, cols) grid shape (unused, kept for API compatibility)
        save_path: if provided, saves figure
        cmap: colormap for field values
        lb, ub: spatial domain bounds for interpolation
        u_true, u_map, u_mean, u_std: optional (n_points,) solution field values
        obs_coords: optional (n_obs, 2) sensor coordinates for black dot overlay
        show_abs_error: if True, add |a_true - a_mean| panel (default True)
        piecewise_a_bounds: if provided as (k_low, k_high), use discrete 2-color
            map for coefficient panels.
    """
    has_u = u_true is not None
    n_rows = 2 if has_u else 1
    n_cols = 5 if show_abs_error else 4

    with plt.style.context(_use_science_style()):
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.2 * n_cols, 3.5 * n_rows))
        if n_rows == 1:
            axes = axes[np.newaxis, :]

        vmin_a = min(a_true.min(), a_map.min(), a_mean.min())
        vmax_a = max(a_true.max(), a_map.max(), a_mean.max())

        if show_abs_error:
            abs_err = np.abs(a_true - a_mean)
            a_fields = [a_true, a_map, a_mean, abs_err, a_std]
            a_labels = ['(a)', '(b)', '(c)', '(d)', '(e)']
        else:
            a_fields = [a_true, a_map, a_mean, a_std]
            a_labels = ['(a)', '(b)', '(c)', '(d)']

        n_field = 3  # first 3 panels always use field cmap
        for i, (ax, field) in enumerate(zip(axes[0], a_fields)):
            if piecewise_a_bounds is not None and i < n_field:
                Plot.imshow_on_ax(ax, fig, _reshape_scatter_to_grid(field, grid_shape),
                                  cmap=_PIECEWISE_A_CMAP, norm=_piecewise_a_norm(piecewise_a_bounds), lb=lb, ub=ub)
            elif i < n_field:
                Plot.contourf_on_ax(ax, fig, x, field, cmap=cmap, vmin=vmin_a, vmax=vmax_a, lb=lb, ub=ub)
            elif show_abs_error and i == n_field:
                Plot.contourf_on_ax(ax, fig, x, field, cmap=cmap, vmin=0, lb=lb, ub=ub)
            else:
                Plot.contourf_on_ax(ax, fig, x, field, cmap=_UNCERTAINTY_CMAP, vmin=0, lb=lb, ub=ub)
            ax.set_aspect('equal')
            ax.tick_params(labelsize=13)

        if has_u:
            vmin_u = min(u_true.min(), u_map.min(), u_mean.min())
            vmax_u = max(u_true.max(), u_map.max(), u_mean.max())

            u_fields = [u_true, u_map, u_mean, u_std]

            for i, (ax, field) in enumerate(zip(axes[1], u_fields)):
                if i < 3:
                    Plot.contourf_on_ax(ax, fig, x, field, cmap=cmap, vmin=vmin_u, vmax=vmax_u, lb=lb, ub=ub)
                else:
                    Plot.contourf_on_ax(ax, fig, x, field, cmap=_UNCERTAINTY_CMAP, vmin=0, lb=lb, ub=ub)
                ax.set_aspect('equal')
                ax.tick_params(labelsize=13)

            # Overlay observation locations on u Truth panel
            if obs_coords is not None:
                axes[1, 0].scatter(obs_coords[:, 0], obs_coords[:, 1],
                                   c='k', s=10, zorder=5)

            # Hide unused 5th panel in u row when showing abs error
            if show_abs_error:
                axes[1, 4].set_visible(False)

        plt.tight_layout()
        if save_path:
            save_path = Path(save_path)
            save_dir = save_path.parent / save_path.stem
            save_dir.mkdir(parents=True, exist_ok=True)

            # Save a row individual panels
            a_names = ['a_true', 'a_map', 'a_mean', 'a_abs_error', 'a_std'] if show_abs_error \
                else ['a_true', 'a_map', 'a_mean', 'a_std']
            for i, (name, field) in enumerate(zip(a_names, a_fields)):
                if piecewise_a_bounds is not None and i < n_field:
                    _save_contourf(save_dir, name, x, field, cmap=_PIECEWISE_A_CMAP,
                                   norm=_piecewise_a_norm(piecewise_a_bounds), lb=lb, ub=ub, grid_shape=grid_shape)
                elif i < n_field:
                    _save_contourf(save_dir, name, x, field, cmap=cmap,
                                   vmin=vmin_a, vmax=vmax_a, lb=lb, ub=ub)
                elif show_abs_error and i == n_field:
                    _save_contourf(save_dir, name, x, field, cmap=cmap, vmin=0, lb=lb, ub=ub)
                else:
                    _save_contourf(save_dir, name, x, field, cmap=_UNCERTAINTY_CMAP, vmin=0, lb=lb, ub=ub)

            if has_u:
                u_names = ['u_true', 'u_map', 'u_mean', 'u_std']
                for i, (name, field) in enumerate(zip(u_names, u_fields)):
                    def _obs_overlay(ax, _oc=obs_coords, _i=i):
                        if _oc is not None and _i == 0:
                            ax.scatter(_oc[:, 0], _oc[:, 1], c='k', s=10, zorder=5)
                    if i < 3:
                        _save_contourf(save_dir, name, x, field, cmap=cmap,
                                       vmin=vmin_u, vmax=vmax_u, lb=lb, ub=ub,
                                       overlay_fn=_obs_overlay)
                    else:
                        _save_contourf(save_dir, name, x, field, cmap=_UNCERTAINTY_CMAP, vmin=0,
                                       lb=lb, ub=ub, overlay_fn=_obs_overlay)
        plt.show()


def plot_calibration(
    nominal: np.ndarray,
    empirical: np.ndarray,
    save_path: Optional[Path] = None,
    label: str = 'Observed',
    ax: Optional[plt.Axes] = None,
):
    """Calibration plot: empirical vs nominal coverage with diagonal reference.

    If ax is provided, plots on that axis (for overlaying multiple curves).
    """
    own_fig = ax is None
    if own_fig:
        with plt.style.context(_use_science_style()):
            fig, ax = plt.subplots(figsize=(4, 4))

    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Ideal')
    ax.plot(nominal, empirical, 'o-', markersize=5, label=label)
    ax.set_xlabel('Nominal Coverage', fontsize=14)
    ax.set_ylabel('Empirical Coverage', fontsize=14)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.tick_params(labelsize=13)
    ax.legend(fontsize=9, loc='lower right', framealpha=0.3, facecolor='white', edgecolor='black')

    if own_fig:
        plt.tight_layout()
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.show()


def plot_posterior_gallery(
    x: np.ndarray,
    a_samples: np.ndarray,
    grid_shape: Tuple[int, int],
    a_true: Optional[np.ndarray] = None,
    n_show: int = 6,
    save_path: Optional[Path] = None,
    cmap: str = 'jet',
    lb: float = 0.,
    ub: float = 1.,
    piecewise_a_bounds: Optional[Tuple[float, float]] = None,
):
    """Gallery of posterior coefficient samples with (a)-(f) labels.

    Args:
        x: (n_points, 2) coordinates
        a_samples: (n_samples, n_points) decoded coefficient samples
        grid_shape: unused
        a_true: (n_points,) for setting consistent color range
        n_show: number of samples to show (default 6)
        save_path: if provided, saves figure
        lb, ub: spatial domain bounds for interpolation
        piecewise_a_bounds: if provided as (k_low, k_high), use discrete 2-color map
    """
    # (a)-(f): Individual posterior coefficient samples
    with plt.style.context(_use_science_style()):
        n_rows = (n_show + 2) // 3
        fig, axes = plt.subplots(n_rows, 3, figsize=(12, 4 * n_rows))
        axes = np.atleast_2d(axes)

        n_total = a_samples.shape[0]
        idxs = np.linspace(0, n_total - 1, n_show, dtype=int)

        if a_true is not None:
            vmin, vmax = a_true.min(), a_true.max()
        else:
            vmin, vmax = a_samples.min(), a_samples.max()

        for k, idx in enumerate(idxs):
            ax = axes[k // 3, k % 3]
            if piecewise_a_bounds is not None:
                Plot.imshow_on_ax(ax, fig, _reshape_scatter_to_grid(a_samples[idx], grid_shape),
                                  cmap=_PIECEWISE_A_CMAP, norm=_piecewise_a_norm(piecewise_a_bounds), lb=lb, ub=ub)
            else:
                Plot.contourf_on_ax(ax, fig, x, a_samples[idx], cmap=cmap, vmin=vmin, vmax=vmax, lb=lb, ub=ub)
            ax.set_aspect('equal')
            ax.tick_params(labelsize=13)

        # Hide unused axes
        for k in range(n_show, n_rows * 3):
            axes[k // 3, k % 3].set_visible(False)

        plt.tight_layout()
        if save_path:
            save_path = Path(save_path)
            save_dir = save_path.parent / save_path.stem
            save_dir.mkdir(parents=True, exist_ok=True)
            for k, idx in enumerate(idxs):
                name = f'sample_{k:02d}'
                if piecewise_a_bounds is not None:
                    _save_contourf(save_dir, name, x, a_samples[idx],
                                   cmap=_PIECEWISE_A_CMAP, norm=_piecewise_a_norm(piecewise_a_bounds), lb=lb, ub=ub,
                                   grid_shape=grid_shape)
                else:
                    _save_contourf(save_dir, name, x, a_samples[idx],
                                   cmap=cmap, vmin=vmin, vmax=vmax, lb=lb, ub=ub)
        plt.show()


def plot_posterior_predictive(
    obs_values: np.ndarray,
    pred_samples: np.ndarray,
    obs_label: str = 'Observed',
    save_path: Optional[Path] = None,
):
    """Posterior predictive check: observed vs predicted scatter with identity line.

    Args:
        obs_values: (n_obs,) observed values
        pred_samples: (n_samples, n_obs) predicted values
        obs_label: unused (kept for API compatibility)
        save_path: if provided, saves figure
    """
    with plt.style.context(_use_science_style()):
        pred_mean = np.mean(pred_samples, axis=0)
        pred_lo = np.percentile(pred_samples, 2.5, axis=0)
        pred_hi = np.percentile(pred_samples, 97.5, axis=0)

        fig, ax = plt.subplots(figsize=(5, 5))
        yerr_lo = np.maximum(pred_mean - pred_lo, 0)
        yerr_hi = np.maximum(pred_hi - pred_mean, 0)
        ax.errorbar(obs_values, pred_mean,
                     yerr=[yerr_lo, yerr_hi],
                     fmt='none', alpha=0.3, color='b', label='95% CI', zorder=1)
        ax.scatter(obs_values, pred_mean, s=8, alpha=0.7, color='C0',
                   label='Posterior mean', zorder=2)

        lims = [min(obs_values.min(), pred_mean.min()),
                max(obs_values.max(), pred_mean.max())]
        margin = (lims[1] - lims[0]) * 0.05
        lims = [lims[0] - margin, lims[1] + margin]
        ax.plot(lims, lims, 'k--', lw=1, label='Identity')
        ax.set_xlim(lims)
        ax.set_ylim(lims)

        ax.set_xlabel('Observed', fontsize=14)
        ax.set_ylabel('Predicted', fontsize=14)
        ax.tick_params(labelsize=13)
        ax.legend(fontsize=9, loc='upper left', framealpha=0.3, facecolor='white', edgecolor='black')
        ax.set_aspect('equal')

        plt.tight_layout()
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.show()


def plot_trace(
    beta_samples: np.ndarray,
    beta_true: Optional[np.ndarray] = None,
    num_warmup: int = 0,
    save_path: Optional[Path] = None,
):
    """Trace plots for latent dimensions with warmup shading and true value legend.

    Args:
        beta_samples: (n_samples, d) MCMC samples (warmup + post-warmup if num_warmup > 0)
        beta_true: (d,) true encoded beta (optional)
        num_warmup: number of warmup samples at start of chain (shaded region)
        save_path: if provided, saves figure
    """
    with plt.style.context(_use_science_style()):
        d = beta_samples.shape[1]
        n_cols = min(4, d)
        n_rows = (d + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 2.5 * n_rows))
        if d == 1:
            axes = np.array([[axes]])
        axes = np.atleast_2d(axes)

        for i in range(d):
            ax = axes[i // n_cols, i % n_cols]
            ax.plot(beta_samples[:, i], alpha=0.7, lw=0.5)
            if num_warmup > 0:
                ax.axvspan(0, num_warmup, alpha=0.15, color='grey', label='Warmup')
                ax.axvline(num_warmup, color='grey', ls=':', lw=0.8)
            if beta_true is not None:
                ax.axhline(float(beta_true[i]), c='r', ls='--', lw=1,
                           label=r'True $\beta$')
            ax.set_ylabel(f'$\\beta_{{{i}}}$', fontsize=12)
            ax.tick_params(labelsize=13)
            # Show legend on every subplot
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(fontsize=8, loc='upper right', framealpha=0.3, facecolor='white', edgecolor='black')

        # Hide unused axes
        for i in range(d, n_rows * n_cols):
            axes[i // n_cols, i % n_cols].set_visible(False)

        plt.tight_layout()
        if save_path:
            save_path = Path(save_path)
            save_dir = save_path.parent / save_path.stem
            save_dir.mkdir(parents=True, exist_ok=True)
            for i in range(d):
                with plt.style.context(_use_science_style()):
                    f2, ax2 = plt.subplots(figsize=(4, 2.5))
                    ax2.plot(beta_samples[:, i], alpha=0.7, lw=0.5)
                    if num_warmup > 0:
                        ax2.axvspan(0, num_warmup, alpha=0.15, color='grey', label='Warmup')
                        ax2.axvline(num_warmup, color='grey', ls=':', lw=0.8)
                    if beta_true is not None:
                        ax2.axhline(float(beta_true[i]), c='r', ls='--', lw=1,
                                    label=r'True $\beta$')
                    ax2.set_ylabel(f'$\\beta_{{{i}}}$', fontsize=12)
                    ax2.tick_params(labelsize=13)
                    handles, lbls = ax2.get_legend_handles_labels()
                    if handles:
                        ax2.legend(fontsize=8, loc='upper right', framealpha=0.3, facecolor='white', edgecolor='black')
                    f2.savefig(save_dir / f'beta_{i:02d}.png', dpi=200, bbox_inches='tight')
                    plt.close(f2)
        plt.show()


def plot_metrics_table(
    metrics: Dict[str, float],
    title: str = 'Metrics Summary',
    diagnostics: Optional[Dict] = None,
):
    """Print a formatted metrics comparison table.

    Args:
        metrics: dict of metric_name -> value
        title: table title
        diagnostics: optional dict with keys 'ess_min', 'rhat_max', 'n_div',
                     'total_samples'. When provided, prints a reliability banner.
    """
    print(f"\n{'=' * 50}")
    print(f"  {title}")
    print(f"{'=' * 50}")
    for name, value in metrics.items():
        if isinstance(value, float):
            print(f"  {name:30s}: {value:.6f}")
        else:
            print(f"  {name:30s}: {value}")
    if diagnostics is not None:
        flag, explanation = mcmc_reliability_flag(
            ess_min=diagnostics.get('ess_min', float('inf')),
            rhat_max=diagnostics.get('rhat_max', 1.0),
            n_divergences=diagnostics.get('n_div', 0),
            total_samples=diagnostics.get('total_samples', 1),
        )
        marker = '!!' if flag == 'FAIL' else ('!' if flag == 'WARN' else '')
        print(f"  {'-' * 48}")
        print(f"  {marker} RELIABILITY: [{flag}] {explanation}")
    print(f"{'=' * 50}\n")


# ---------------------------------------------------------------------------
# RQ2-specific: rho sweep plot
# ---------------------------------------------------------------------------

def plot_rho_sweep(
    sweep_results: List[Dict],
    baseline: Dict,
    save_path: Optional[Path] = None,
    a_metric_key: str = 'a_err',
    a_metric_label: str = 'Rel. L2 (a)',
):
    """Multi-panel plot: a_err/icorr, u_err, coverage, divergences, CRPS vs rho_pde.

    Args:
        sweep_results: list of dicts with keys: rho_pde, a_err (or icorr), u_err,
                       coverage, n_div, crps_a
        baseline: dict with same keys for data-only reference
        save_path: if provided, saves figure
        a_metric_key: key for coefficient accuracy metric (default 'a_err')
        a_metric_label: display label for coefficient metric
    """
    with plt.style.context(_use_science_style()):
        rhos = [r['rho_pde'] for r in sweep_results]
        fig, axes = plt.subplots(2, 3, figsize=(16, 9))

        # a_err / icorr
        ax = axes[0, 0]
        ax.semilogx(rhos, [r[a_metric_key] for r in sweep_results], 'o-', color='C0')
        if a_metric_key in baseline:
            ax.axhline(baseline[a_metric_key], ls='--', color='grey', label='data-only')
        ax.set_xlabel(r'$\rho_{\mathrm{pde}}$')
        ax.set_ylabel(a_metric_label)
        ax.invert_xaxis()
        ax.legend(fontsize=9, framealpha=0.3, facecolor='white', edgecolor='black')

        # u_err (optional — skipped if sweep_results don't contain the key)
        ax = axes[0, 1]
        if sweep_results and 'u_err' in sweep_results[0]:
            ax.semilogx(rhos, [r['u_err'] for r in sweep_results], 'o-', color='C1')
            if 'u_err' in baseline:
                ax.axhline(baseline['u_err'], ls='--', color='grey', label='data-only')
            ax.set_xlabel(r'$\rho_{\mathrm{pde}}$')
            ax.set_ylabel('u_err (rel. L2)')
            ax.invert_xaxis()
            ax.legend(fontsize=9, framealpha=0.3, facecolor='white', edgecolor='black')
        else:
            ax.set_visible(False)

        # Coverage
        ax = axes[0, 2]
        ax.semilogx(rhos, [r['coverage'] * 100 for r in sweep_results], 'o-', color='C2')
        ax.axhline(95, ls='-', color='green', lw=2, label='ideal 95%')
        if 'coverage' in baseline:
            ax.axhline(baseline['coverage'] * 100, ls='--', color='grey', label='data-only')
        ax.set_xlabel(r'$\rho_{\mathrm{pde}}$')
        ax.set_ylabel('95% CI coverage (%)')
        ax.invert_xaxis()
        ax.legend(fontsize=9, framealpha=0.3, facecolor='white', edgecolor='black')

        # Divergences
        ax = axes[1, 0]
        ax.semilogx(rhos, [r['n_div'] for r in sweep_results], 'o-', color='C4')
        ax.axhline(0, ls='-', color='green', lw=1)
        ax.set_xlabel(r'$\rho_{\mathrm{pde}}$')
        ax.set_ylabel('# divergences')
        ax.invert_xaxis()

        # CRPS
        ax = axes[1, 1]
        crps_key = 'crps_a' if 'crps_a' in sweep_results[0] else 'mean_log_pde'
        ax.semilogx(rhos, [r[crps_key] for r in sweep_results], 'o-', color='C5')
        if crps_key in baseline:
            ax.axhline(baseline[crps_key], ls='--', color='grey', label='data-only')
        ax.set_xlabel(r'$\rho_{\mathrm{pde}}$')
        ax.set_ylabel('CRPS (a)' if crps_key == 'crps_a' else 'Mean log p(R|beta)')
        ax.invert_xaxis()
        if crps_key in baseline:
            ax.legend(fontsize=9, framealpha=0.3, facecolor='white', edgecolor='black')

        # Hide unused panel
        axes[1, 2].set_visible(False)

        plt.tight_layout()
        if save_path:
            save_path = Path(save_path)
            save_dir = save_path.parent / save_path.stem
            save_dir.mkdir(parents=True, exist_ok=True)

            panels_data = [
                ('a_metric', rhos, [r[a_metric_key] for r in sweep_results], a_metric_label,
                 baseline.get(a_metric_key), 'C0', True),
                *([('u_err', rhos, [r['u_err'] for r in sweep_results], 'u_err (rel. L2)',
                    baseline.get('u_err'), 'C1', True)]
                  if sweep_results and 'u_err' in sweep_results[0] else []),
                ('coverage', rhos, [r['coverage'] * 100 for r in sweep_results], '95% CI coverage (%)',
                 baseline.get('coverage', None), 'C2', True),
                ('divergences', rhos, [r['n_div'] for r in sweep_results], '# divergences',
                 None, 'C4', True),
                ('crps', rhos, [r[crps_key] for r in sweep_results],
                 'CRPS (a)' if crps_key == 'crps_a' else 'Mean log p(R|beta)',
                 baseline.get(crps_key), 'C5', True),
            ]
            for fname, xs, ys, ylabel, bl_val, color, inv in panels_data:
                with plt.style.context(_use_science_style()):
                    f2, ax2 = plt.subplots(figsize=(5, 4))
                    ax2.semilogx(xs, ys, 'o-', color=color)
                    if fname == 'coverage':
                        ax2.axhline(95, ls='-', color='green', lw=2, label='ideal 95%')
                        if bl_val is not None:
                            ax2.axhline(bl_val * 100, ls='--', color='grey', label='data-only')
                    elif fname == 'divergences':
                        ax2.axhline(0, ls='-', color='green', lw=1)
                    elif bl_val is not None:
                        ax2.axhline(bl_val, ls='--', color='grey', label='data-only')
                    ax2.set_xlabel(r'$\rho_{\mathrm{pde}}$')
                    ax2.set_ylabel(ylabel)
                    if inv:
                        ax2.invert_xaxis()
                    handles, lbls = ax2.get_legend_handles_labels()
                    if handles:
                        ax2.legend(fontsize=9, framealpha=0.3, facecolor='white', edgecolor='black')
                    f2.savefig(save_dir / f'{fname}.png', dpi=200, bbox_inches='tight')
                    plt.close(f2)
        plt.show()


# ---------------------------------------------------------------------------
# RQ2-specific: sharpness-calibration trade-off scatter
# ---------------------------------------------------------------------------

def plot_sharpness_calibration_tradeoff(
    sweep_results: List[Dict],
    baseline: Optional[Dict] = None,
    save_path: Optional[Path] = None,
):
    """Scatter plot of CI width (sharpness) vs calibration error.

    Visualises the trade-off between sharpness and calibration across
    different rho_pde values.  Ideal region is bottom-left (tight CI AND
    well-calibrated).

    Args:
        sweep_results: list of dicts, each with keys ``rho_pde`` (float),
            ``ci_width`` (float), ``coverage`` (float, 0-1).
        baseline: optional dict with the same keys for the data-only
            reference point.
        save_path: if provided, saves the composite figure and an individual
            panel PNG in a subdirectory following the ``plot_rho_sweep``
            convention.
    """
    with plt.style.context(_use_science_style()):
        fig, ax = plt.subplots(figsize=(5, 4))

        xs = [r['ci_width'] for r in sweep_results]
        ys = [abs(r['coverage'] - 0.95) for r in sweep_results]
        rhos = [r['rho_pde'] for r in sweep_results]

        ax.plot(xs, ys, 'o-', color='C0', zorder=3)

        for x, y, rho in zip(xs, ys, rhos):
            ax.annotate(
                f'{rho:g}', (x, y),
                textcoords='offset points', xytext=(6, 4),
                fontsize=14, color='C0',
            )

        if baseline is not None and 'ci_width' in baseline and 'coverage' in baseline:
            bx = baseline['ci_width']
            by = abs(baseline['coverage'] - 0.95)
            ax.plot(bx, by, 'r*', markersize=12, zorder=4, label='data-only')
            ax.legend(fontsize=9, framealpha=0.3, facecolor='white', edgecolor='black')

        ax.set_xlabel('Mean 95% CI width')
        ax.set_ylabel(r'Calibration error $|\mathrm{coverage} - 0.95|$')
        plt.tight_layout()

        if save_path:
            save_path = Path(save_path)
            save_dir = save_path.parent / save_path.stem
            save_dir.mkdir(parents=True, exist_ok=True)
            # Individual panel
            with plt.style.context(_use_science_style()):
                f2, ax2 = plt.subplots(figsize=(5, 4))
                ax2.plot(xs, ys, 'o-', color='C0', zorder=3)
                for x, y, rho in zip(xs, ys, rhos):
                    ax2.annotate(
                        f'{rho:g}', (x, y),
                        textcoords='offset points', xytext=(6, 4),
                        fontsize=14, color='C0',
                    )
                if baseline is not None and 'ci_width' in baseline and 'coverage' in baseline:
                    bx = baseline['ci_width']
                    by = abs(baseline['coverage'] - 0.95)
                    ax2.plot(bx, by, 'r*', markersize=12, zorder=4, label='data-only')
                    ax2.legend(fontsize=9, framealpha=0.3, facecolor='white', edgecolor='black')
                ax2.set_xlabel('Mean 95% CI width')
                ax2.set_ylabel(r'Calibration error $|\mathrm{coverage} - 0.95|$')
                f2.savefig(save_dir / 'sharpness_calibration.png', dpi=200, bbox_inches='tight')
                plt.close(f2)
        plt.show()


# ---------------------------------------------------------------------------
# RQ2-specific: std comparison
# ---------------------------------------------------------------------------

def plot_std_comparison(x, std_data_only, std_physics, grid_shape=None,
                        save_path=None, lb=0., ub=1.):
    """Side-by-side posterior std maps: Data-Only vs Physics-Informed."""
    plot_std_comparison_generic(
        x, std_data_only, std_physics,
        label_a='Data-Only', label_b='Physics-Informed',
        save_path=save_path, lb=lb, ub=ub,
    )


# ---------------------------------------------------------------------------
# RQ2-specific: metrics comparison table
# ---------------------------------------------------------------------------

def plot_metrics_comparison_table(
    data_only_metrics: Dict[str, float],
    physics_metrics: Dict[str, float],
    title: str = 'Data-Only vs Physics-Informed',
):
    """Two-column formatted comparison table.

    Args:
        data_only_metrics: dict of metric_name -> value for data-only run
        physics_metrics: dict of metric_name -> value for physics-informed run
        title: table title
    """
    all_keys = list(dict.fromkeys(list(data_only_metrics.keys()) + list(physics_metrics.keys())))

    print(f"\n{'=' * 65}")
    print(f"  {title}")
    print(f"{'=' * 65}")
    print(f"  {'Metric':<25s} {'Data-Only':>16s} {'Physics-Inf':>16s}")
    print(f"  {'-' * 59}")

    for key in all_keys:
        dv = data_only_metrics.get(key, None)
        pv = physics_metrics.get(key, None)

        def _fmt(v):
            if v is None:
                return '-'
            if isinstance(v, float):
                return f"{v:.6f}"
            return str(v)

        print(f"  {key:<25s} {_fmt(dv):>16s} {_fmt(pv):>16s}")

    print(f"{'=' * 65}\n")


# ---------------------------------------------------------------------------
# RQ3-specific: generic std comparison
# ---------------------------------------------------------------------------

def plot_std_comparison_generic(
    x: np.ndarray,
    std_a: np.ndarray,
    std_b: np.ndarray,
    label_a: str,
    label_b: str,
    save_path: Optional[Path] = None,
    lb: float = 0.,
    ub: float = 1.,
    **kwargs,
):
    """Side-by-side posterior std maps: (a) std_a, (b) std_b, (c) difference."""
    with plt.style.context(_use_science_style()):
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))

        vmax = max(std_a.max(), std_b.max())

        Plot.contourf_on_ax(axes[0], fig, x, std_a, cmap=_UNCERTAINTY_CMAP, vmin=0, vmax=vmax, lb=lb, ub=ub)
        axes[0].set_aspect('equal')
        axes[0].tick_params(labelsize=13)

        Plot.contourf_on_ax(axes[1], fig, x, std_b, cmap=_UNCERTAINTY_CMAP, vmin=0, vmax=vmax, lb=lb, ub=ub)
        axes[1].set_aspect('equal')
        axes[1].tick_params(labelsize=13)

        diff = std_a - std_b
        Plot.contourf_on_ax(axes[2], fig, x, diff, cmap='RdBu', lb=lb, ub=ub)
        axes[2].set_aspect('equal')
        axes[2].tick_params(labelsize=13)

        plt.tight_layout()
        if save_path:
            save_path = Path(save_path)
            save_dir = save_path.parent / save_path.stem
            save_dir.mkdir(parents=True, exist_ok=True)
            _save_contourf(save_dir, 'std_a', x, std_a, cmap=_UNCERTAINTY_CMAP, vmin=0, vmax=vmax, lb=lb, ub=ub)
            _save_contourf(save_dir, 'std_b', x, std_b, cmap=_UNCERTAINTY_CMAP, vmin=0, vmax=vmax, lb=lb, ub=ub)
            _save_contourf(save_dir, 'diff', x, diff, cmap='RdBu', lb=lb, ub=ub)
        plt.show()


# ---------------------------------------------------------------------------
# RQ3-specific: 4-way metrics comparison table
# ---------------------------------------------------------------------------

def plot_metrics_comparison_table_4way(
    metrics_in_do: Dict[str, float],
    metrics_in_phys: Dict[str, float],
    metrics_ood_do: Dict[str, float],
    metrics_ood_phys: Dict[str, float],
    title: str = 'In-Domain vs OOD x Data-Only vs Physics',
    col_labels: Optional[List[str]] = None,
):
    """Four-column formatted comparison table.

    Args:
        metrics_in_do: column 1 metrics
        metrics_in_phys: column 2 metrics
        metrics_ood_do: column 3 metrics
        metrics_ood_phys: column 4 metrics
        title: table title
        col_labels: optional list of 4 column header strings
    """
    if col_labels is None:
        col_labels = ['In-DO', 'In-Phys', 'OOD-DO', 'OOD-Phys']
    c1, c2, c3, c4 = col_labels

    all_keys = list(dict.fromkeys(
        list(metrics_in_do.keys()) + list(metrics_in_phys.keys()) +
        list(metrics_ood_do.keys()) + list(metrics_ood_phys.keys())
    ))

    print(f"\n{'=' * 95}")
    print(f"  {title}")
    print(f"{'=' * 95}")
    print(f"  {'Metric':<20s} {c1:>16s} {c2:>16s} {c3:>16s} {c4:>16s}")
    print(f"  {'-' * 84}")

    def _fmt(v):
        if v is None:
            return '-'
        if isinstance(v, float):
            return f"{v:.6f}"
        return str(v)

    for key in all_keys:
        v1 = metrics_in_do.get(key, None)
        v2 = metrics_in_phys.get(key, None)
        v3 = metrics_ood_do.get(key, None)
        v4 = metrics_ood_phys.get(key, None)
        print(f"  {key:<20s} {_fmt(v1):>16s} {_fmt(v2):>16s} {_fmt(v3):>16s} {_fmt(v4):>16s}")

    print(f"{'=' * 95}\n")


# ---------------------------------------------------------------------------
# RQ3-specific: calibration overlay
# ---------------------------------------------------------------------------

def plot_calibration_overlay(
    calibration_list: List[Tuple[np.ndarray, np.ndarray, str]],
    save_path: Optional[Path] = None,
):
    """Overlay multiple calibration curves on one plot.

    Args:
        calibration_list: list of (nominal, empirical, label) tuples
        save_path: if provided, saves figure
    """
    markers = ['o', 's', '^', 'D', 'v', 'p', '*']
    with plt.style.context(_use_science_style()):
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Ideal')

        for i, (nominal, empirical, label) in enumerate(calibration_list):
            marker = markers[i % len(markers)]
            ax.plot(nominal, empirical, f'{marker}-', markersize=5, label=label)

        ax.set_xlabel('Nominal Coverage', fontsize=14)
        ax.set_ylabel('Empirical Coverage', fontsize=14)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.05)
        ax.tick_params(labelsize=13)
        ax.legend(fontsize=9, loc='lower right', framealpha=0.3, facecolor='white', edgecolor='black')

        plt.tight_layout()
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.show()


# ---------------------------------------------------------------------------
# RQ4/RQ5: sweep summary table
# ---------------------------------------------------------------------------

def plot_sweep_summary_table(
    results_list: List[Dict],
    condition_labels: List[str],
    title: str = 'Sweep Summary',
    use_icorr: bool = False,
    num_chains: int = 4,
    num_samples: int = 2000,
):
    """Formatted table: rows = conditions, columns = key metrics + reliability flag.

    Args:
        results_list: list of result dicts with keys: crps_a, coverage, ci_width,
                      a_err or icorr, mean_std, n_div, ess_min (optional), rhat_max (optional)
        condition_labels: display label for each condition
        title: table title
        use_icorr: if True, show I_corr instead of Rel.L2(a)
        num_chains: number of MCMC chains (for divergence fraction calculation)
        num_samples: samples per chain (for divergence fraction calculation)
    """
    if use_icorr:
        columns = ['I_corr', 'CRPS(a)', 'Coverage', 'CI Width', 'Sharpness', 'Diverg.', 'ESS min', 'R-hat', 'Flag']
        keys = ['icorr', 'crps_a', 'coverage_95', 'ci_width', 'mean_std', 'n_div', 'ess_min', 'rhat_max', '_flag']
    else:
        columns = ['CRPS(a)', 'Coverage', 'CI Width', 'Sharpness', 'Rel.L2(a)', 'Diverg.', 'ESS min', 'R-hat', 'Flag']
        keys = ['crps_a', 'coverage_95', 'ci_width', 'mean_std', 'a_err', 'n_div', 'ess_min', 'rhat_max', '_flag']

    # Flag column is wider
    col_widths = [12] * (len(columns) - 1) + [8]
    header = f"  {'Condition':<22s}" + "".join(f"{c:>{w}s}" for c, w in zip(columns, col_widths))
    total_width = 22 + sum(col_widths)
    sep = "  " + "-" * total_width

    print(f"\n{'=' * (total_width + 2)}")
    print(f"  {title}")
    print(f"{'=' * (total_width + 2)}")
    print(header)
    print(sep)

    total_samples = num_chains * num_samples
    any_warn = False
    any_fail = False

    for label, res in zip(condition_labels, results_list):
        ess_min = res.get('ess_min', float('inf'))
        rhat_max = res.get('rhat_max', 1.0)
        n_div = res.get('n_div', 0)
        flag, _ = mcmc_reliability_flag(ess_min, rhat_max, n_div, total_samples)
        if flag == 'FAIL':
            any_fail = True
        elif flag == 'WARN':
            any_warn = True

        vals = []
        for k, w in zip(keys, col_widths):
            if k == '_flag':
                marker = '**' if flag == 'FAIL' else ('*' if flag == 'WARN' else '')
                vals.append(f"{(marker + flag):>{w}s}")
                continue
            v = res.get(k, None)
            if v is None:
                vals.append(f"{'-':>{w}s}")
            elif k == 'coverage_95':
                vals.append(f"{v:>{w - 1}.2%} ")
            elif k in ('n_div',):
                vals.append(f"{int(v):>{w}d}")
            elif k in ('ess_min',):
                vals.append(f"{v:>{w}.1f}")
            elif k in ('rhat_max',):
                vals.append(f"{v:>{w}.4f}")
            else:
                vals.append(f"{v:>{w}.6f}")
        print(f"  {label:<22s}" + "".join(vals))

    print(f"{'=' * (total_width + 2)}")
    if any_fail:
        print("  ** FAIL: chains did not converge — do not draw conclusions from these conditions")
    if any_warn:
        print("  *  WARN: marginal convergence — interpret with caution")
    print()


# ---------------------------------------------------------------------------
# RQ4: noise sweep plot
# ---------------------------------------------------------------------------

def _plot_sweep_common(
    sweep_results: List[Dict],
    x_vals: list,
    x_label: str,
    baseline: Optional[Dict] = None,
    invert_x: bool = False,
    save_path: Optional[Path] = None,
    a_metric_key: str = 'a_err',
    a_metric_label: str = 'Rel. L2 (a)',
):
    """Shared implementation for noise/sensor sweep plots (2x3 grid, 5 panels)."""
    with plt.style.context(_use_science_style()):
        fig, axes = plt.subplots(2, 3, figsize=(16, 9))

        panels = [
            (axes[0, 0], a_metric_key, a_metric_label, 'C0'),
            (axes[0, 1], 'coverage_95', '95% CI coverage', 'C2'),
            (axes[0, 2], 'ci_width', 'Mean CI width (a)', 'C1'),
            (axes[1, 0], 'n_div', '# divergences', 'C4'),
            (axes[1, 1], 'crps_a', 'CRPS (a)', 'C5'),
        ]

        for ax, key, ylabel, color in panels:
            vals = [r[key] * 100 if key == 'coverage_95' else r[key] for r in sweep_results]
            ax.plot(x_vals, vals, 'o-', color=color)

            if baseline and key in baseline:
                bv = baseline[key] * 100 if key == 'coverage_95' else baseline[key]
                ax.axhline(bv, ls='--', color='grey', label='clean baseline')
                ax.legend(fontsize=9, framealpha=0.3, facecolor='white', edgecolor='black')

            if key == 'coverage_95':
                ax.axhline(95, ls='-', color='green', lw=1.5, alpha=0.5)

            ax.set_xlabel(x_label, fontsize=14)
            ax.set_ylabel(ylabel, fontsize=14)
            ax.tick_params(labelsize=13)
            if invert_x:
                ax.invert_xaxis()

        axes[1, 2].set_visible(False)
        plt.tight_layout()

        if save_path:
            save_path = Path(save_path)
            save_dir = save_path.parent / save_path.stem
            save_dir.mkdir(parents=True, exist_ok=True)
            panel_specs = [
                (a_metric_key, a_metric_label, 'C0'),
                ('coverage_95', '95% CI coverage (%)', 'C2'),
                ('ci_width', 'Mean CI width (a)', 'C1'),
                ('n_div', '# divergences', 'C4'),
                ('crps_a', 'CRPS (a)', 'C5'),
            ]
            fname_map = {a_metric_key: 'a_metric', 'coverage_95': 'coverage',
                         'ci_width': 'ci_width', 'n_div': 'divergences', 'crps_a': 'crps'}
            for key, ylabel, color in panel_specs:
                vals = [r[key] * 100 if key == 'coverage_95' else r[key] for r in sweep_results]
                with plt.style.context(_use_science_style()):
                    f2, ax2 = plt.subplots(figsize=(5, 4))
                    ax2.plot(x_vals, vals, 'o-', color=color)
                    if baseline and key in baseline:
                        bv = baseline[key] * 100 if key == 'coverage_95' else baseline[key]
                        ax2.axhline(bv, ls='--', color='grey', label='clean baseline')
                        ax2.legend(fontsize=9, framealpha=0.3, facecolor='white', edgecolor='black')
                    if key == 'coverage_95':
                        ax2.axhline(95, ls='-', color='green', lw=1.5, alpha=0.5)
                    ax2.set_xlabel(x_label, fontsize=14)
                    ax2.set_ylabel(ylabel, fontsize=14)
                    ax2.tick_params(labelsize=13)
                    if invert_x:
                        ax2.invert_xaxis()
                    f2.savefig(save_dir / f'{fname_map[key]}.png', dpi=200, bbox_inches='tight')
                    plt.close(f2)
        plt.show()


def plot_noise_sweep(sweep_results, baseline=None, save_path=None,
                     a_metric_key='a_err', a_metric_label='Rel. L2 (a)'):
    """2x3 panel: metrics vs SNR (dB)."""
    _plot_sweep_common(
        sweep_results, [r['snr_db'] for r in sweep_results], 'SNR (dB)',
        baseline=baseline, invert_x=True, save_path=save_path,
        a_metric_key=a_metric_key, a_metric_label=a_metric_label,
    )


def plot_sensor_sweep(sweep_results, save_path=None,
                      a_metric_key='a_err', a_metric_label='Rel. L2 (a)'):
    """2x3 panel: metrics vs number of sensors."""
    _plot_sweep_common(
        sweep_results, [r['n_obs'] for r in sweep_results], 'Number of sensors',
        save_path=save_path, a_metric_key=a_metric_key, a_metric_label=a_metric_label,
    )


# ---------------------------------------------------------------------------
# RQ4/RQ5: multi-panel posterior std maps
# ---------------------------------------------------------------------------

def plot_std_multi_panel(
    x: np.ndarray,
    stds: List[np.ndarray],
    labels: List[str],
    grid_shape: Tuple[int, int],
    suptitle: str = 'Posterior Std Comparison',
    save_path: Optional[Path] = None,
    lb: float = 0.,
    ub: float = 1.,
):
    """1xN panel showing posterior std maps with (a)(b)(c)... labels.

    Panels labeled (a), (b), (c), ... — caller should document what each letter represents.

    Args:
        x: (n_points, 2) coordinates
        stds: list of (n_points,) posterior std arrays
        labels: label for each panel (unused, kept for API compat; use markdown cell)
        grid_shape: unused
        suptitle: unused (kept for API compatibility)
        save_path: if provided, saves figure
        lb, ub: spatial domain bounds for interpolation
    """
    n = len(stds)
    with plt.style.context(_use_science_style()):
        fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4))
        if n == 1:
            axes = [axes]

        vmax = max(s.max() for s in stds)

        for i, (ax, std) in enumerate(zip(axes, stds)):
            Plot.contourf_on_ax(ax, fig, x, std, cmap=_UNCERTAINTY_CMAP, vmin=0, vmax=vmax, lb=lb, ub=ub)
            ax.set_aspect('equal')
            ax.tick_params(labelsize=13)

        plt.tight_layout()
        if save_path:
            save_path = Path(save_path)
            save_dir = save_path.parent / save_path.stem
            save_dir.mkdir(parents=True, exist_ok=True)
            for i, std in enumerate(stds):
                _save_contourf(save_dir, f'std_{i:02d}', x, std,
                               cmap=_UNCERTAINTY_CMAP, vmin=0, vmax=vmax, lb=lb, ub=ub)
        plt.show()


# ---------------------------------------------------------------------------
# Uncertainty investigation
# ---------------------------------------------------------------------------

def plot_uncertainty_investigation(
    x: np.ndarray,
    a_std: np.ndarray,
    obs_coords: np.ndarray,
    grid_shape: Tuple[int, int],
    save_path: Optional[Path] = None,
    lb: float = 0.,
    ub: float = 1.,
):
    """Investigate correlation between sensor placement and posterior uncertainty.

    Panels:
      (a) Posterior std with sensor locations overlaid
      (b) Distance-to-nearest-sensor vs posterior std, with Pearson correlation

    Args:
        x: (n_points, 2) spatial coordinates
        a_std: (n_points,) posterior standard deviation
        obs_coords: (n_obs, 2) sensor coordinates
        grid_shape: unused
        save_path: if provided, saves figure
        lb, ub: spatial domain bounds for interpolation
    """
    with plt.style.context(_use_science_style()):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        # (a) Posterior std with sensor overlay
        ax = axes[0]
        Plot.contourf_on_ax(ax, fig, x, a_std, cmap=_UNCERTAINTY_CMAP, vmin=0, lb=lb, ub=ub)
        ax.scatter(obs_coords[:, 0], obs_coords[:, 1], c='k', s=12, zorder=5)
        ax.set_aspect('equal')
        ax.tick_params(labelsize=13)

        # (b) Distance-to-nearest-sensor vs posterior std
        ax = axes[1]
        # Compute min distance from each grid point to any sensor
        dists = np.sqrt(((x[:, None, :] - obs_coords[None, :, :]) ** 2).sum(axis=2))
        min_dist = dists.min(axis=1)  # (n_points,)

        ax.scatter(min_dist, a_std, s=3, alpha=0.4, color='C0')
        r_val, p_val = pearsonr(min_dist, a_std)
        ax.set_xlabel('Distance to nearest sensor', fontsize=14)
        ax.set_ylabel('Posterior std', fontsize=14)
        ax.tick_params(labelsize=13)
        ax.text(0.05, 0.95, f'Pearson r = {r_val:.3f}\np = {p_val:.2e}',
                transform=ax.transAxes, fontsize=12, va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()
        if save_path:
            save_path = Path(save_path)
            save_dir = save_path.parent / save_path.stem
            save_dir.mkdir(parents=True, exist_ok=True)
            _save_contourf(save_dir, 'std_with_sensors', x, a_std,
                           cmap=_UNCERTAINTY_CMAP, vmin=0, lb=lb, ub=ub,
                           overlay_fn=lambda ax2: ax2.scatter(
                               obs_coords[:, 0], obs_coords[:, 1], c='k', s=12, zorder=5))
            with plt.style.context(_use_science_style()):
                f2, ax2 = plt.subplots(figsize=(5, 4))
                ax2.scatter(min_dist, a_std, s=3, alpha=0.4, color='C0')
                ax2.set_xlabel('Distance to nearest sensor', fontsize=14)
                ax2.set_ylabel('Posterior std', fontsize=14)
                ax2.tick_params(labelsize=13)
                ax2.text(0.05, 0.95, f'Pearson r = {r_val:.3f}\np = {p_val:.2e}',
                         transform=ax2.transAxes, fontsize=12, va='top',
                         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                f2.savefig(save_dir / 'distance_vs_std.png', dpi=200, bbox_inches='tight')
                plt.close(f2)
        plt.show()


# ---------------------------------------------------------------------------
# EIT: ground truth and observation data visualization
# ---------------------------------------------------------------------------

def plot_eit_ground_truth(
    x: np.ndarray,
    a_true: np.ndarray,
    u_true: Optional[np.ndarray] = None,
    save_path: Optional[Path] = None,
    cmap: str = 'jet',
    lb: float = 0.,
    ub: float = 1.,
):
    """2D contourf visualization of EIT ground truth fields.

    Mirrors the author's Plot.show_2d_list pattern. Shows:
      (a) Ground truth coefficient a
      (b) Ground truth solution u (if provided)

    Args:
        x: (n_points, 2) coordinates
        a_true: (n_points,) ground truth coefficient values
        u_true: (n_points,) ground truth solution values (optional)
        save_path: if provided, saves figure
        cmap: colormap (default 'jet')
        lb, ub: spatial domain bounds for interpolation
    """
    n_cols = 2 if u_true is not None else 1
    with plt.style.context(_use_science_style()):
        fig, axes = plt.subplots(1, n_cols, figsize=(4.5 * n_cols, 4))
        if n_cols == 1:
            axes = [axes]

        Plot.contourf_on_ax(axes[0], fig, x, a_true, cmap=cmap, lb=lb, ub=ub)
        axes[0].set_aspect('equal')
        axes[0].tick_params(labelsize=13)

        if u_true is not None:
            Plot.contourf_on_ax(axes[1], fig, x, u_true, cmap=cmap, lb=lb, ub=ub)
            axes[1].set_aspect('equal')
            axes[1].tick_params(labelsize=13)

        plt.tight_layout()
        if save_path:
            save_path = Path(save_path)
            save_dir = save_path.parent / save_path.stem
            save_dir.mkdir(parents=True, exist_ok=True)
            _save_contourf(save_dir, 'a_true', x, a_true, cmap=cmap, lb=lb, ub=ub)
            if u_true is not None:
                _save_contourf(save_dir, 'u_true', x, u_true, cmap=cmap, lb=lb, ub=ub)
        plt.show()


def plot_eit_observation_data(
    x_bd: np.ndarray,
    g_l: int,
    neumann_obs: np.ndarray,
    neumann_clean: Optional[np.ndarray] = None,
    save_path: Optional[Path] = None,
):
    """1D line plots of EIT boundary condition and Neumann flux observations.

    Mirrors the author's 3-panel boundary plot. Shows:
      (a) Boundary condition g(x) = cos(2π(x·cos(θ) + y·sin(θ))), θ = πl/20
      (b) Clean Neumann flux a*(∇u·n) (if neumann_clean provided, else same as obs)
      (c) Observed Neumann flux (with noise if neumann_clean was provided)

    Args:
        x_bd: (n_bd, 2) boundary point coordinates (ordered left→top→right→bottom)
        g_l: boundary condition index l (1-indexed, used to compute θ = πl/20)
        neumann_obs: (n_bd,) Neumann boundary flux (may be noisy)
        neumann_clean: (n_bd,) noise-free Neumann flux (optional; if None, panel (b) = panel (c))
        save_path: if provided, saves figure
    """
    theta = np.pi * g_l / 20.0
    g_vals = np.cos(2.0 * np.pi * (x_bd[:, 0] * np.cos(theta) + x_bd[:, 1] * np.sin(theta)))

    with plt.style.context(_use_science_style()):
        fig, axes = plt.subplots(1, 3, figsize=(14, 3))

        axes[0].plot(g_vals)
        axes[0].set_title(f'g (l={g_l})', fontsize=14)
        axes[0].set_xlabel('Boundary index', fontsize=12)
        axes[0].tick_params(labelsize=13)

        clean = neumann_clean if neumann_clean is not None else neumann_obs
        axes[1].plot(clean)
        axes[1].set_title('Neumann flux (clean)', fontsize=14)
        axes[1].set_xlabel('Boundary index', fontsize=12)
        axes[1].tick_params(labelsize=13)

        axes[2].plot(neumann_obs)
        noise_label = 'Neumann flux (observed)' if neumann_clean is None else 'Neumann flux (noisy)'
        axes[2].set_title(noise_label, fontsize=14)
        axes[2].set_xlabel('Boundary index', fontsize=12)
        axes[2].tick_params(labelsize=13)

        plt.tight_layout()
        if save_path:
            save_path = Path(save_path)
            save_dir = save_path.parent / save_path.stem
            save_dir.mkdir(parents=True, exist_ok=True)
            panel_data = [
                ('g_boundary', g_vals, f'g (l={g_l})'),
                ('neumann_clean', clean, 'Neumann flux (clean)'),
                ('neumann_observed', neumann_obs, noise_label),
            ]
            for fname, data, title in panel_data:
                with plt.style.context(_use_science_style()):
                    f2, ax2 = plt.subplots(figsize=(5, 3))
                    ax2.plot(data)
                    ax2.set_title(title, fontsize=14)
                    ax2.set_xlabel('Boundary index', fontsize=12)
                    ax2.tick_params(labelsize=13)
                    f2.savefig(save_dir / f'{fname}.png', dpi=200, bbox_inches='tight')
                    plt.close(f2)
        plt.show()


# ---------------------------------------------------------------------------
# Burgers: field comparison, posterior gallery, uncertainty investigation
# ---------------------------------------------------------------------------

def plot_burgers_field_comparison(
    x_mesh: np.ndarray,
    t_mesh: np.ndarray,
    a_true: np.ndarray,
    a_map: np.ndarray,
    a_mean: np.ndarray,
    a_std: np.ndarray,
    u_true: np.ndarray,
    u_map: np.ndarray,
    u_mean: np.ndarray,
    u_std: np.ndarray,
    obs_coords: Optional[np.ndarray] = None,
    save_path: Optional[Path] = None,
    cmap: str = 'jet',
):
    """Field comparison for Burgers: 1D line plots for a(x) and 2D heatmaps for u(x,t).

    Top row (3 panels): initial condition a(x)
      (a) Truth (red dashed) vs MAP (blue)
      (b) Truth (red dashed) vs Posterior mean (blue) +/- 2 sigma shaded band
      (c) Posterior std sigma(x)

    Bottom row (4 panels): space-time solution u(x,t) as pcolormesh heatmaps
      (d) Ground truth  (e) MAP  (f) Posterior mean  (g) Posterior std

    Args:
        x_mesh: (n_mesh,) spatial grid
        t_mesh: (n_time,) temporal grid
        a_true, a_map, a_mean, a_std: (n_mesh,) initial condition arrays
        u_true, u_map, u_mean, u_std: (n_mesh*n_time,) solution arrays (row-major meshgrid order)
        obs_coords: (n_obs, 2) [x, t] sensor locations, overlaid on truth u panel
        save_path: if provided, saves individual panels to save_dir/
        cmap: colormap for u heatmaps (default 'jet')
    """
    n_mesh = len(x_mesh)
    n_time = len(t_mesh)

    # Reshape solution from (n_mesh*n_time,) to (n_time, n_mesh) for pcolormesh.
    # gridxt is built as meshgrid(x_mesh, t_mesh) which puts t on rows (slow axis).
    def _reshape(u): return u.reshape(n_time, n_mesh)

    u_true_2d = _reshape(u_true)
    u_map_2d = _reshape(u_map)
    u_mean_2d = _reshape(u_mean)
    u_std_2d = _reshape(u_std)

    vmin_u = min(u_true_2d.min(), u_map_2d.min(), u_mean_2d.min())
    vmax_u = max(u_true_2d.max(), u_map_2d.max(), u_mean_2d.max())

    X, T = np.meshgrid(x_mesh, t_mesh)

    with plt.style.context(_use_science_style()):
        fig, axes = plt.subplots(2, 4, figsize=(16, 7))

        # --- Top row: a(x) line plots ---
        # (a) Truth vs MAP
        ax = axes[0, 0]
        ax.plot(x_mesh, a_true, 'r--', lw=1.2, label='Truth')
        ax.plot(x_mesh, a_map, 'b-', lw=1.0, label='MAP')
        ax.set_xlabel('$x$', fontsize=12)
        ax.set_ylabel('$a(x)$', fontsize=12)
        ax.legend(fontsize=9, loc='upper right', framealpha=0.3, facecolor='white', edgecolor='black')
        ax.tick_params(labelsize=13)
        ax.set_xlim(x_mesh[0], x_mesh[-1])

        # (b) Truth vs Posterior mean +/- 2sigma
        ax = axes[0, 1]
        ax.plot(x_mesh, a_true, 'r--', lw=1.2, label='Truth')
        ax.plot(x_mesh, a_mean, 'b-', lw=1.0, label='Post. mean')
        ax.fill_between(x_mesh, a_mean - 2 * a_std, a_mean + 2 * a_std,
                         alpha=0.25, color='b', label='$\\pm 2\\sigma$')
        ax.set_xlabel('$x$', fontsize=12)
        ax.set_ylabel('$a(x)$', fontsize=12)
        ax.legend(fontsize=9, loc='upper right', framealpha=0.3, facecolor='white', edgecolor='black')
        ax.tick_params(labelsize=13)
        ax.set_xlim(x_mesh[0], x_mesh[-1])

        # (c) Posterior std
        ax = axes[0, 2]
        ax.plot(x_mesh, a_std, 'k-', lw=1.0)
        ax.set_xlabel('$x$', fontsize=12)
        ax.set_ylabel('$\\sigma_a(x)$', fontsize=12)
        ax.tick_params(labelsize=13)
        ax.set_xlim(x_mesh[0], x_mesh[-1])
        ax.set_ylim(bottom=0)

        axes[0, 3].set_visible(False)

        # --- Bottom row: u(x,t) pcolormesh ---
        u_panels = [
            (axes[1, 0], u_true_2d,  cmap,  vmin_u, vmax_u),
            (axes[1, 1], u_map_2d,   cmap,  vmin_u, vmax_u),
            (axes[1, 2], u_mean_2d,  cmap,  vmin_u, vmax_u),
            (axes[1, 3], u_std_2d,  _UNCERTAINTY_CMAP, 0,      None),
        ]
        for ax, data, cm, vmin, vmax in u_panels:
            im = ax.pcolormesh(X, T, data, cmap=cm, vmin=vmin, vmax=vmax, shading='auto')
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_xlabel('$x$', fontsize=12)
            ax.set_ylabel('$t$', fontsize=12)
            ax.tick_params(labelsize=13)

        # Overlay sensors on truth panel
        if obs_coords is not None:
            axes[1, 0].scatter(obs_coords[:, 0], obs_coords[:, 1],
                               c='k', s=8, zorder=5)

        plt.tight_layout()

        if save_path:
            save_path = Path(save_path)
            save_dir = save_path.parent / save_path.stem
            save_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_dir / 'combined.png', dpi=200, bbox_inches='tight')

            # Individual a panels
            _save_burgers_a_lines(save_dir, 'a_truth_vs_map', x_mesh,
                                   [('r--', 'Truth', a_true), ('b-', 'MAP', a_map)])
            _save_burgers_a_band(save_dir, 'a_truth_vs_mean', x_mesh, a_true, a_mean, a_std)
            _save_burgers_a_std_panel(save_dir, 'a_std', x_mesh, a_std)

            # Individual u panels
            for fname, data, cm, vmin, vmax in [
                ('u_true',  u_true_2d,  cmap,  vmin_u, vmax_u),
                ('u_map',   u_map_2d,   cmap,  vmin_u, vmax_u),
                ('u_mean',  u_mean_2d,  cmap,  vmin_u, vmax_u),
                ('u_std',   u_std_2d,  _UNCERTAINTY_CMAP, 0,      None),
            ]:
                obs = obs_coords if fname == 'u_true' else None
                _save_burgers_u_panel(save_dir, fname, X, T, data, cm, vmin, vmax, obs)

        plt.show()


def _save_burgers_a_lines(save_dir, fname, x_mesh, lines):
    with plt.style.context(_use_science_style()):
        f, ax = plt.subplots(figsize=(5, 3))
        for style, label, vals in lines:
            ax.plot(x_mesh, vals, style, lw=1.2, label=label)
        ax.set_xlabel('$x$', fontsize=12)
        ax.set_ylabel('$a(x)$', fontsize=12)
        ax.legend(fontsize=9, framealpha=0.3, facecolor='white', edgecolor='black')
        ax.tick_params(labelsize=13)
        f.savefig(save_dir / f'{fname}.png', dpi=200, bbox_inches='tight')
        plt.close(f)


def _save_burgers_a_band(save_dir, fname, x_mesh, a_true, a_mean, a_std):
    with plt.style.context(_use_science_style()):
        f, ax = plt.subplots(figsize=(5, 3))
        ax.plot(x_mesh, a_true, 'r--', lw=1.2, label='Truth')
        ax.plot(x_mesh, a_mean, 'b-', lw=1.0, label='Post. mean')
        ax.fill_between(x_mesh, a_mean - 2 * a_std, a_mean + 2 * a_std,
                         alpha=0.25, color='b', label='$\\pm 2\\sigma$')
        ax.set_xlabel('$x$', fontsize=12)
        ax.set_ylabel('$a(x)$', fontsize=12)
        ax.legend(fontsize=9, framealpha=0.3, facecolor='white', edgecolor='black')
        ax.tick_params(labelsize=13)
        f.savefig(save_dir / f'{fname}.png', dpi=200, bbox_inches='tight')
        plt.close(f)


def _save_burgers_a_std_panel(save_dir, fname, x_mesh, a_std):
    with plt.style.context(_use_science_style()):
        f, ax = plt.subplots(figsize=(5, 3))
        ax.plot(x_mesh, a_std, 'k-', lw=1.0)
        ax.set_xlabel('$x$', fontsize=12)
        ax.set_ylabel('$\\sigma_a(x)$', fontsize=12)
        ax.tick_params(labelsize=13)
        ax.set_ylim(bottom=0)
        f.savefig(save_dir / f'{fname}.png', dpi=200, bbox_inches='tight')
        plt.close(f)


def _save_burgers_u_panel(save_dir, fname, X, T, data, cmap, vmin, vmax, obs_coords):
    with plt.style.context(_use_science_style()):
        f, ax = plt.subplots(figsize=(5, 3.5))
        im = ax.pcolormesh(X, T, data, cmap=cmap, vmin=vmin, vmax=vmax, shading='auto')
        f.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xlabel('$x$', fontsize=12)
        ax.set_ylabel('$t$', fontsize=12)
        ax.tick_params(labelsize=13)
        if obs_coords is not None:
            ax.scatter(obs_coords[:, 0], obs_coords[:, 1], c='k', s=8, zorder=5)
        f.savefig(save_dir / f'{fname}.png', dpi=200, bbox_inches='tight')
        plt.close(f)


def plot_burgers_posterior_gallery(
    x_mesh: np.ndarray,
    a_samples: np.ndarray,
    a_true: Optional[np.ndarray] = None,
    n_show: int = 6,
    save_path: Optional[Path] = None,
):
    """Gallery of Burgers posterior initial-condition samples as 1D line plots.

    Args:
        x_mesh: (n_mesh,) spatial grid
        a_samples: (n_samples, n_mesh) decoded initial condition samples
        a_true: (n_mesh,) ground truth (overlaid in red dashed on each panel)
        n_show: number of evenly spaced samples to show (default 6)
        save_path: if provided, saves individual panels to save_dir/
    """
    n_total = a_samples.shape[0]
    idxs = np.linspace(0, n_total - 1, n_show, dtype=int)

    all_vals = a_samples[idxs]
    ymin = float(all_vals.min()) - 0.1 * abs(float(all_vals.min()))
    ymax = float(all_vals.max()) + 0.1 * abs(float(all_vals.max()))
    if a_true is not None:
        ymin = min(ymin, float(a_true.min()) - 0.1 * abs(float(a_true.min())))
        ymax = max(ymax, float(a_true.max()) + 0.1 * abs(float(a_true.max())))

    n_rows = (n_show + 2) // 3
    with plt.style.context(_use_science_style()):
        fig, axes = plt.subplots(n_rows, 3, figsize=(12, 3.5 * n_rows))
        axes = np.atleast_2d(axes)

        for k, idx in enumerate(idxs):
            ax = axes[k // 3, k % 3]
            ax.plot(x_mesh, a_samples[idx], 'b-', lw=0.9, label='Sample')
            if a_true is not None:
                ax.plot(x_mesh, a_true, 'r--', lw=1.0, label='Truth')
            ax.set_ylim(ymin, ymax)
            ax.set_xlabel('$x$', fontsize=12)
            ax.set_ylabel('$a(x)$', fontsize=12)
            ax.tick_params(labelsize=13)
            if k == 0:
                ax.legend(fontsize=9, loc='upper right', framealpha=0.3, facecolor='white', edgecolor='black')

        for k in range(n_show, n_rows * 3):
            axes[k // 3, k % 3].set_visible(False)

        plt.tight_layout()
        if save_path:
            save_path = Path(save_path)
            save_dir = save_path.parent / save_path.stem
            save_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_dir / 'combined.png', dpi=200, bbox_inches='tight')
            for k, idx in enumerate(idxs):
                with plt.style.context(_use_science_style()):
                    f2, ax2 = plt.subplots(figsize=(4, 3))
                    ax2.plot(x_mesh, a_samples[idx], 'b-', lw=0.9, label='Sample')
                    if a_true is not None:
                        ax2.plot(x_mesh, a_true, 'r--', lw=1.0, label='Truth')
                    ax2.set_ylim(ymin, ymax)
                    ax2.set_xlabel('$x$', fontsize=12)
                    ax2.set_ylabel('$a(x)$', fontsize=12)
                    ax2.tick_params(labelsize=13)
                    ax2.legend(fontsize=9, framealpha=0.3, facecolor='white', edgecolor='black')
                    f2.savefig(save_dir / f'sample_{k:02d}.png', dpi=200, bbox_inches='tight')
                    plt.close(f2)
        plt.show()


def plot_burgers_uncertainty_investigation(
    x_mesh: np.ndarray,
    a_std: np.ndarray,
    obs_xt: np.ndarray,
    save_path: Optional[Path] = None,
):
    """Investigate correlation between sensor placement and posterior uncertainty for Burgers.

    The initial condition a(x) depends only on space, so uncertainty is analysed
    in the x-dimension only. Sensor (x,t) locations are projected to their x-coordinates.

    Panels:
      (a) a(x) posterior std with vertical lines at sensor x-coordinates
      (b) Min spatial distance to nearest sensor vs posterior std, with Pearson r

    Args:
        x_mesh: (n_mesh,) spatial grid
        a_std: (n_mesh,) posterior standard deviation of a(x)
        obs_xt: (n_obs, 2) [x, t] sensor locations in space-time
        save_path: if provided, saves figure
    """
    sensor_x = obs_xt[:, 0]  # project sensors to x-axis

    # Min spatial distance from each grid point to nearest sensor (in x only)
    min_dist = np.min(np.abs(x_mesh[:, None] - sensor_x[None, :]), axis=1)

    with plt.style.context(_use_science_style()):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        # (a) Posterior std with sensor projection
        ax = axes[0]
        ax.plot(x_mesh, a_std, 'k-', lw=1.0)
        for sx in sensor_x:
            ax.axvline(sx, color='C0', alpha=0.15, lw=0.6)
        ax.set_xlabel('$x$', fontsize=14)
        ax.set_ylabel('Posterior std $\\sigma_a(x)$', fontsize=14)
        ax.tick_params(labelsize=13)
        ax.set_xlim(x_mesh[0], x_mesh[-1])
        ax.set_ylim(bottom=0)

        # (b) Distance vs std scatter
        ax = axes[1]
        ax.scatter(min_dist, a_std, s=6, alpha=0.5, color='C0')
        r_val, p_val = pearsonr(min_dist, a_std)
        ax.set_xlabel('Min distance to nearest sensor ($x$)', fontsize=14)
        ax.set_ylabel('Posterior std $\\sigma_a(x)$', fontsize=14)
        ax.tick_params(labelsize=13)
        ax.text(0.05, 0.95, f'Pearson r = {r_val:.3f}\np = {p_val:.2e}',
                transform=ax.transAxes, fontsize=12, va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()
        if save_path:
            save_path = Path(save_path)
            save_dir = save_path.parent / save_path.stem
            save_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_dir / 'combined.png', dpi=200, bbox_inches='tight')
            with plt.style.context(_use_science_style()):
                f2, ax2 = plt.subplots(figsize=(5, 3.5))
                ax2.plot(x_mesh, a_std, 'k-', lw=1.0)
                for sx in sensor_x:
                    ax2.axvline(sx, color='C0', alpha=0.15, lw=0.6)
                ax2.set_xlabel('$x$', fontsize=14)
                ax2.set_ylabel('Posterior std $\\sigma_a(x)$', fontsize=14)
                ax2.tick_params(labelsize=13)
                f2.savefig(save_dir / 'std_with_sensors.png', dpi=200, bbox_inches='tight')
                plt.close(f2)
            with plt.style.context(_use_science_style()):
                f3, ax3 = plt.subplots(figsize=(5, 3.5))
                ax3.scatter(min_dist, a_std, s=6, alpha=0.5, color='C0')
                ax3.set_xlabel('Min distance to nearest sensor ($x$)', fontsize=14)
                ax3.set_ylabel('Posterior std $\\sigma_a(x)$', fontsize=14)
                ax3.tick_params(labelsize=13)
                ax3.text(0.05, 0.95, f'Pearson r = {r_val:.3f}\np = {p_val:.2e}',
                         transform=ax3.transAxes, fontsize=12, va='top',
                         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                f3.savefig(save_dir / 'distance_vs_std.png', dpi=200, bbox_inches='tight')
                plt.close(f3)
        plt.show()


def compute_error_std_correlation(
    a_true: np.ndarray,
    a_mean: np.ndarray,
    a_std: np.ndarray,
    save_path: Optional[Path] = None,
) -> Tuple[float, float]:
    """Spearman rank correlation between pointwise error and posterior std.

    Tests whether high posterior uncertainty predicts where actual errors are
    (Gap #6: 'Is the uncertainty informative?').

    Args:
        a_true: ground truth coefficient field, shape (n_points,) or (n_points, 1)
        a_mean: posterior mean coefficient field, same shape
        a_std: posterior std coefficient field, same shape
        save_path: if provided, saves scatter plot to this path

    Returns:
        (rho, p_value) from scipy.stats.spearmanr
    """
    a_true_flat = np.asarray(a_true).ravel()
    a_mean_flat = np.asarray(a_mean).ravel()
    a_std_flat = np.asarray(a_std).ravel()

    abs_error = np.abs(a_mean_flat - a_true_flat)
    rho, p_value = spearmanr(abs_error, a_std_flat)

    if save_path is not None:
        save_path = Path(save_path)
        save_dir = save_path.parent
        save_dir.mkdir(parents=True, exist_ok=True)
        with plt.style.context(_use_science_style()):
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.scatter(abs_error, a_std_flat, s=3, alpha=0.4, color='C0')
            ax.set_xlabel('|Posterior mean - Truth|', fontsize=14)
            ax.set_ylabel('Posterior std', fontsize=14)
            ax.tick_params(labelsize=13)
            ax.text(0.05, 0.95,
                    f'Spearman $\\rho$ = {rho:.3f}\np = {p_value:.2e}',
                    transform=ax.transAxes, fontsize=12, va='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            fig.savefig(save_path, dpi=200, bbox_inches='tight')
            plt.close(fig)

    return float(rho), float(p_value)


def plot_burgers_std_comparison(
    x_mesh: np.ndarray,
    std_a: np.ndarray,
    std_b: np.ndarray,
    label_a: str,
    label_b: str,
    save_path: Optional[Path] = None,
):
    """Side-by-side 1D posterior std comparison for two Burgers conditions.

    Three panels: (a) condition A std, (b) condition B std, (c) difference B - A.

    Args:
        x_mesh: (n_mesh,) spatial grid
        std_a: (n_mesh,) posterior std for condition A
        std_b: (n_mesh,) posterior std for condition B
        label_a: label for condition A
        label_b: label for condition B
        save_path: if provided, saves figure
    """
    with plt.style.context(_use_science_style()):
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
        vmax = max(std_a.max(), std_b.max()) * 1.05
        for ax, std, label in [(axes[0], std_a, label_a), (axes[1], std_b, label_b)]:
            ax.plot(x_mesh, std, 'C3', lw=1.5)
            ax.set_title(label, fontsize=14)
            ax.set_xlabel('$x$', fontsize=12)
            ax.set_ylabel('Posterior std $a(x)$', fontsize=12)
            ax.set_ylim(0, vmax)
            ax.tick_params(labelsize=13)
        axes[2].plot(x_mesh, std_b - std_a, 'C1', lw=1.5)
        axes[2].axhline(0, ls='--', color='grey', lw=0.8)
        axes[2].set_title(f'{label_b} \u2212 {label_a}', fontsize=14)
        axes[2].set_xlabel('$x$', fontsize=12)
        axes[2].set_ylabel('\u0394 std', fontsize=12)
        axes[2].tick_params(labelsize=13)
        plt.tight_layout()
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.show()


def plot_burgers_std_multi_panel(
    x_mesh: np.ndarray,
    stds: List[np.ndarray],
    labels: List[str],
    save_path: Optional[Path] = None,
):
    """1xN panels of posterior std of a(x) as 1D line plots with a shared y-axis.

    Used by RQ4 (noise sweep) and RQ5 (sensor sweep).

    Args:
        x_mesh: (n_mesh,) spatial grid
        stds: list of (n_mesh,) posterior std arrays, one per condition
        labels: list of condition labels (same length as stds)
        save_path: if provided, saves combined figure and individual panels
    """
    n = len(stds)
    with plt.style.context(_use_science_style()):
        fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 3.5), sharey=True)
        if n == 1:
            axes = [axes]

        vmax = max(s.max() for s in stds)

        for i, (ax, std, label) in enumerate(zip(axes, stds, labels)):
            ax.plot(x_mesh, std, 'C3', lw=1.5)
            ax.set_title(label, fontsize=14)
            ax.set_xlabel('$x$', fontsize=12)
            ax.set_ylim(0, vmax * 1.1)
            ax.tick_params(labelsize=13)
            if i == 0:
                ax.set_ylabel('Posterior std $a(x)$', fontsize=12)

        plt.tight_layout()
        if save_path:
            save_path = Path(save_path)
            save_dir = save_path.parent / save_path.stem
            save_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path.parent / (save_path.stem + '.png'), dpi=200, bbox_inches='tight')
            for i, (std, label) in enumerate(zip(stds, labels)):
                with plt.style.context(_use_science_style()):
                    fig2, ax2 = plt.subplots(figsize=(4.5, 3.5))
                    ax2.plot(x_mesh, std, 'C3', lw=1.5)
                    ax2.set_title(label, fontsize=14)
                    ax2.set_xlabel('$x$', fontsize=12)
                    ax2.set_ylabel('Posterior std $a(x)$', fontsize=12)
                    ax2.set_ylim(0, vmax * 1.1)
                    ax2.tick_params(labelsize=13)
                    fig2.savefig(save_dir / f'std_{i:02d}.png', dpi=200, bbox_inches='tight')
                    plt.close(fig2)
        plt.show()

def _significance_stars(p_value: float) -> str:
    """Return significance stars for a p-value (bootstrap: p from CI exclusion)."""
    if p_value < 0.001:
        return '***'
    elif p_value < 0.01:
        return '**'
    elif p_value < 0.05:
        return '*'
    return 'ns'


def plot_physics_benefit_sweep(
    sweep_pairs: List[Dict],
    sweep_var: str,
    sweep_values: List,
    metric_key: str = 'a_err',
    metric_label: str = 'Rel. L2 (a)',
    sweep_label: Optional[str] = None,
    save_path: Optional[Path] = None,
):
    """Paired line plot: data-only vs physics metric across a sweep dimension.

    Shows two lines (data-only and physics-augmented) on the same axes with a
    shaded region between them to highlight the benefit region.  A second panel
    shows coverage so readers can check calibration simultaneously.

    Args:
        sweep_pairs: list of dicts, one per sweep point, each with keys:
            ``data_only`` (dict with metric_key) and ``physics`` (dict with metric_key),
            plus ``coverage_data`` / ``coverage_physics`` (fractions).
        sweep_var: axis label for the sweep dimension (e.g. ``'SNR (dB)'``).
        sweep_values: x-axis tick values, same length as ``sweep_pairs``.
        metric_key: result dict key for the accuracy metric.
        metric_label: y-axis label for the accuracy panel.
        sweep_label: override x-axis label (defaults to ``sweep_var``).
        save_path: if given, writes ``<stem>/physics_benefit_sweep.png`` and
            individual panel PNGs.
    """
    xs = list(sweep_values)
    data_vals = [p['data_only'][metric_key] for p in sweep_pairs]
    phys_vals = [p['physics'][metric_key] for p in sweep_pairs]
    cov_data = [p['data_only'].get('coverage', np.nan) * 100 for p in sweep_pairs]
    cov_phys = [p['physics'].get('coverage', np.nan) * 100 for p in sweep_pairs]
    x_label = sweep_label or sweep_var

    with plt.style.context(_use_science_style()):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

        # Panel 1 — accuracy metric
        ax = axes[0]
        ax.plot(xs, data_vals, 'o--', color='C0', label='data-only')
        ax.plot(xs, phys_vals, 's-', color='C1', label='physics')
        # shade region where physics < data-only (beneficial)
        ax.fill_between(xs, phys_vals, data_vals,
                        where=[p <= d for p, d in zip(phys_vals, data_vals)],
                        alpha=0.15, color='C1', label='benefit region')
        ax.set_xlabel(x_label, fontsize=14)
        ax.set_ylabel(metric_label, fontsize=14)
        ax.legend(fontsize=9, framealpha=0.3, facecolor='white', edgecolor='black')
        ax.tick_params(labelsize=13)

        # Panel 2 — coverage
        ax2 = axes[1]
        ax2.plot(xs, cov_data, 'o--', color='C0', label='data-only')
        ax2.plot(xs, cov_phys, 's-', color='C1', label='physics')
        ax2.axhline(95, ls='-', color='green', lw=1.5, alpha=0.5, label='95% target')
        ax2.set_xlabel(x_label, fontsize=14)
        ax2.set_ylabel('95% CI coverage (%)', fontsize=14)
        ax2.legend(fontsize=9, framealpha=0.3, facecolor='white', edgecolor='black')
        ax2.tick_params(labelsize=13)

        plt.tight_layout()

        if save_path:
            save_path = Path(save_path)
            save_dir = save_path.parent / save_path.stem
            save_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_dir / 'physics_benefit_sweep.png', dpi=200, bbox_inches='tight')
            # Individual panels
            for panel_name, d_vals, p_vals, ylabel, is_cov in [
                (metric_key, data_vals, phys_vals, metric_label, False),
                ('coverage', cov_data, cov_phys, '95% CI coverage (%)', True),
            ]:
                with plt.style.context(_use_science_style()):
                    f2, a2 = plt.subplots(figsize=(5, 4))
                    a2.plot(xs, d_vals, 'o--', color='C0', label='data-only')
                    a2.plot(xs, p_vals, 's-', color='C1', label='physics')
                    if is_cov:
                        a2.axhline(95, ls='-', color='green', lw=1.5, alpha=0.5)
                    else:
                        a2.fill_between(xs, p_vals, d_vals,
                                        where=[p <= d for p, d in zip(p_vals, d_vals)],
                                        alpha=0.15, color='C1')
                    a2.set_xlabel(x_label, fontsize=14)
                    a2.set_ylabel(ylabel, fontsize=14)
                    a2.legend(fontsize=9, framealpha=0.3, facecolor='white', edgecolor='black')
                    a2.tick_params(labelsize=13)
                    f2.savefig(save_dir / f'{panel_name}.png', dpi=200, bbox_inches='tight')
                    plt.close(f2)

        plt.show()


def plot_delta_metrics_sweep(
    sweep_pairs: List[Dict],
    sweep_var: str,
    sweep_values: List,
    metric_key: str = 'a_err',
    metric_label: str = 'Δ Rel. L2 (a)',
    sweep_label: Optional[str] = None,
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
    save_path: Optional[Path] = None,
):
    """Bar chart of delta(metric) = physics − data_only at each sweep point.

    Each bar shows the mean difference with bootstrap CI error bars.  Bars
    are annotated with significance stars when the CI excludes zero.

    Args:
        sweep_pairs: list of dicts, one per sweep point.  Each dict must have
            keys ``data_only`` and ``physics``, each being a dict with:
            - ``metric_key``: scalar metric value
            - ``bootstrap_lo`` / ``bootstrap_hi``: bootstrap CI bounds for the
              scalar metric (used to compute delta CI via error propagation).
            Alternatively, if ``delta_ci_lo`` / ``delta_ci_hi`` are present at
            the top level of each pair dict they are used directly.
        sweep_var: dimension name (e.g. ``'SNR (dB)'``).
        sweep_values: x-axis positions (same length as ``sweep_pairs``).
        metric_key: key for the metric to difference.
        metric_label: y-axis label.
        sweep_label: override x-axis label.
        n_bootstrap: unused (reserved for future per-pair resampling).
        ci_level: confidence level for significance annotation threshold.
        save_path: if given, saves figure.
    """
    xs = np.arange(len(sweep_values))
    x_labels = [str(v) for v in sweep_values]
    x_label = sweep_label or sweep_var

    deltas = []
    ci_los = []
    ci_his = []
    for pair in sweep_pairs:
        d = pair['data_only'][metric_key]
        p = pair['physics'][metric_key]
        delta = p - d
        deltas.append(delta)
        # Use precomputed delta CI if available, else fall back to individual CIs
        if 'delta_ci_lo' in pair and 'delta_ci_hi' in pair:
            ci_los.append(pair['delta_ci_lo'])
            ci_his.append(pair['delta_ci_hi'])
        else:
            d_lo = pair['data_only'].get('bootstrap_lo', d)
            d_hi = pair['data_only'].get('bootstrap_hi', d)
            p_lo = pair['physics'].get('bootstrap_lo', p)
            p_hi = pair['physics'].get('bootstrap_hi', p)
            # Conservative: add half-widths in quadrature
            hw_d = (d_hi - d_lo) / 2
            hw_p = (p_hi - p_lo) / 2
            hw = np.sqrt(hw_d**2 + hw_p**2)
            ci_los.append(delta - hw)
            ci_his.append(delta + hw)

    err_lo = np.array(deltas) - np.array(ci_los)
    err_hi = np.array(ci_his) - np.array(deltas)

    colors = ['C1' if d < 0 else 'C0' for d in deltas]  # C1=orange=beneficial

    with plt.style.context(_use_science_style()):
        fig, ax = plt.subplots(figsize=(max(6, 2 * len(xs)), 4.5))
        bars = ax.bar(xs, deltas, color=colors, alpha=0.75, width=0.5)
        ax.errorbar(xs, deltas, yerr=[err_lo, err_hi],
                    fmt='none', color='black', capsize=4, lw=1.2)
        ax.axhline(0, color='black', lw=1.0, ls='--')

        # Significance annotation (approx p from CI under normal assumption)
        z_crit = norm.ppf((1 + ci_level) / 2)
        for i, (delta, lo, hi) in enumerate(zip(deltas, ci_los, ci_his)):
            hw = (hi - lo) / 2
            se = hw / z_crit if hw > 0 else 0.0
            z = abs(delta) / se if se > 0 else 0.0
            p_approx = 2 * norm.sf(z) if se > 0 else 1.0
            stars = _significance_stars(p_approx)
            y_ann = hi + (max(np.abs(deltas)) * 0.03)
            ax.text(xs[i], y_ann, stars, ha='center', va='bottom', fontsize=12)

        ax.set_xticks(xs)
        ax.set_xticklabels(x_labels, fontsize=12)
        ax.set_xlabel(x_label, fontsize=14)
        ax.set_ylabel(metric_label, fontsize=14)
        ax.tick_params(labelsize=13)
        plt.tight_layout()

        if save_path:
            save_path = Path(save_path)
            save_dir = save_path.parent / save_path.stem
            save_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_dir / 'delta_metrics_sweep.png', dpi=200, bbox_inches='tight')

        plt.show()


def plot_physics_benefit_comparison(
    id_results: List[Dict],
    ood_results: List[Dict],
    metric_key: str = 'a_err',
    metric_label: str = 'Δ Rel. L2 (a)',
    condition_labels: Tuple[str, str] = ('in-domain', 'OOD'),
    save_path: Optional[Path] = None,
):
    """Side-by-side delta(physics) bars: in-domain vs OOD (for hardened RQ3).

    Tests whether the physics benefit (delta = physics − data_only) is larger
    on OOD data than in-domain data.  Each pair of bars represents one test
    case; error bars are bootstrap CIs (propagated or direct).

    Args:
        id_results: list of result dicts for in-domain cases.  Each dict has
            ``data_only`` and ``physics`` sub-dicts with ``metric_key`` and
            optional ``bootstrap_lo`` / ``bootstrap_hi`` / ``delta_ci_lo`` /
            ``delta_ci_hi``.
        ood_results: same structure, same length as ``id_results``.
        metric_key: key for the metric to delta.
        metric_label: y-axis label.
        condition_labels: display names for the two bars per group.
        save_path: if given, saves ``<stem>/physics_benefit_comparison.png``.
    """
    n = len(id_results)
    assert len(ood_results) == n, "id_results and ood_results must have the same length"

    def _delta_and_ci(pair):
        d = pair['data_only'][metric_key]
        p = pair['physics'][metric_key]
        delta = p - d
        if 'delta_ci_lo' in pair and 'delta_ci_hi' in pair:
            return delta, pair['delta_ci_lo'], pair['delta_ci_hi']
        d_lo = pair['data_only'].get('bootstrap_lo', d)
        d_hi = pair['data_only'].get('bootstrap_hi', d)
        p_lo = pair['physics'].get('bootstrap_lo', p)
        p_hi = pair['physics'].get('bootstrap_hi', p)
        hw = np.sqrt(((d_hi - d_lo) / 2)**2 + ((p_hi - p_lo) / 2)**2)
        return delta, delta - hw, delta + hw

    id_deltas, id_los, id_his = zip(*[_delta_and_ci(r) for r in id_results])
    ood_deltas, ood_los, ood_his = zip(*[_delta_and_ci(r) for r in ood_results])

    xs = np.arange(n)
    width = 0.35

    with plt.style.context(_use_science_style()):
        fig, ax = plt.subplots(figsize=(max(6, 2.5 * n), 4.5))

        id_err_lo = np.array(id_deltas) - np.array(id_los)
        id_err_hi = np.array(id_his) - np.array(id_deltas)
        ood_err_lo = np.array(ood_deltas) - np.array(ood_los)
        ood_err_hi = np.array(ood_his) - np.array(ood_deltas)

        ax.bar(xs - width / 2, id_deltas, width, color='C0', alpha=0.75,
               label=condition_labels[0])
        ax.errorbar(xs - width / 2, id_deltas, yerr=[id_err_lo, id_err_hi],
                    fmt='none', color='black', capsize=3, lw=1.2)

        ax.bar(xs + width / 2, ood_deltas, width, color='C3', alpha=0.75,
               label=condition_labels[1])
        ax.errorbar(xs + width / 2, ood_deltas, yerr=[ood_err_lo, ood_err_hi],
                    fmt='none', color='black', capsize=3, lw=1.2)

        # Annotate significance for OOD bar (approx p from CI under normal assumption)
        z_crit_95 = norm.ppf(0.975)
        for i, (delta, lo, hi) in enumerate(zip(ood_deltas, ood_los, ood_his)):
            hw = (hi - lo) / 2
            se = hw / z_crit_95 if hw > 0 else 0.0
            z = abs(delta) / se if se > 0 else 0.0
            p_approx = 2 * norm.sf(z) if se > 0 else 1.0
            stars = _significance_stars(p_approx)
            y_top = hi + (max(np.abs(list(id_deltas) + list(ood_deltas))) * 0.04)
            ax.text(xs[i] + width / 2, y_top, stars, ha='center', va='bottom', fontsize=12)

        ax.axhline(0, color='black', lw=1.0, ls='--')
        ax.set_xticks(xs)
        ax.set_xticklabels([f'case {i+1}' for i in range(n)], fontsize=12)
        ax.set_xlabel('Test case', fontsize=14)
        ax.set_ylabel(metric_label, fontsize=14)
        ax.legend(fontsize=9, framealpha=0.3, facecolor='white', edgecolor='black')
        ax.tick_params(labelsize=13)
        plt.tight_layout()

        if save_path:
            save_path = Path(save_path)
            save_dir = save_path.parent / save_path.stem
            save_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_dir / 'physics_benefit_comparison.png', dpi=200, bbox_inches='tight')

        plt.show()
