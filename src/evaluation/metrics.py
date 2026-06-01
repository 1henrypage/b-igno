"""Evaluation metrics for inverse problems: relative RMSE, I_corr, and relative L2."""
import jax.numpy as jnp
from typing import Dict, List

from src.utils.Losses import MyError


def rmse(pred: jnp.ndarray, true: jnp.ndarray) -> float:
    """
    Relative Root Mean Square Error (IGNO Eq. 14).

    RMSE = sqrt(sum((a_rec - a_true)^2) / sum(a_true^2))
    """
    pred_flat = pred.flatten().astype(jnp.float32)
    true_flat = true.flatten().astype(jnp.float32)

    numerator = jnp.sum((pred_flat - true_flat) ** 2)
    denominator = jnp.sum(true_flat ** 2)

    safe_denom = jnp.where(denominator < 1e-10, 1e-10, denominator)
    result = jnp.sqrt(numerator / safe_denom)

    return float(result)


def relative_l2(pred: jnp.ndarray, true: jnp.ndarray, p: int = 2) -> float:
    """
    Relative L_p error

    rel_L2 = ||pred - true||_p / ||true||_p
    """
    error_fn = MyError(d=2, p=p, size_average=True, reduction=True)
    return float(error_fn.Lp_rel(pred, true))


def cross_correlation(pred: jnp.ndarray, true: jnp.ndarray,
                      k_low: float = 5.0, k_high: float = 10.0) -> float:
    """
    Cross-correlation indicator I_corr (IGNO Eq. 15).

    For piecewise constant coefficients with values {k_low, k_high}.
    Rescales via (x - k_low) / (k_high - k_low) to map {k_low, k_high} → {0, 1}, then computes:

    I_corr = sum((pred_scaled * true_scaled)^2) /
             sqrt(sum(pred_scaled^2) * sum(true_scaled^2))

    Matches reference implementation (yaohua/IGNO get_Icorr).
    Only meaningful for the darcy_piecewise problem.
    """
    pred_flat = pred.flatten().astype(jnp.float32)
    true_flat = true.flatten().astype(jnp.float32)

    k_range = k_high - k_low
    pred_scaled = (pred_flat - k_low) / k_range
    true_scaled = (true_flat - k_low) / k_range

    norm_pred = jnp.sum(pred_scaled ** 2)
    norm_true = jnp.sum(true_scaled ** 2)

    numerator = jnp.sum((pred_scaled * true_scaled) ** 2)
    denominator = jnp.sqrt(norm_pred * norm_true)

    safe_denom = jnp.where(denominator < 1e-10, 1e-10, denominator)

    return float(numerator / safe_denom)


def compute_all_metrics(pred: jnp.ndarray, true: jnp.ndarray,
                        k_low: float = 5.0, k_high: float = 10.0) -> Dict[str, float]:
    return {
        'rmse': rmse(pred, true),
        'relative_l2': relative_l2(pred, true),
        'cross_correlation': cross_correlation(pred, true, k_low=k_low, k_high=k_high),
    }


def aggregate_metrics(metrics_list: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """Return mean, std, min, max for each metric across a list of per-sample dicts."""
    if not metrics_list:
        return {}

    keys = metrics_list[0].keys()
    result = {}

    for key in keys:
        values = [m[key] for m in metrics_list]
        n = len(values)
        mean_val = sum(values) / n
        std_val = (sum((v - mean_val) ** 2 for v in values) / n) ** 0.5

        result[key] = {
            'mean': mean_val,
            'std': std_val,
            'min': min(values),
            'max': max(values),
        }

    return result