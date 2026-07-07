"""Result dataclasses for posterior inference experiments.

MCMCResult holds per-run diagnostics and metrics.  ExperimentResult is the
top-level container whose experiment_type ('single', 'comparison', or 'sweep')
determines which payload fields are populated.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# MCMCResult
# ---------------------------------------------------------------------------

@dataclass
class MCMCResult:
    """Diagnostics and posterior metrics for a single MCMC run.

    All fields that contain numpy arrays in the raw notebooks are stored as
    plain Python lists so the dataclass is directly JSON-serialisable.
    """

    # --- MCMC configuration ---
    sigma: float
    num_warmup: int
    num_samples: int
    num_chains: int

    # --- MCMC diagnostics ---
    ess_min: float
    rhat_max: float
    rhat_mean: float
    n_div: int
    reliability_flag: str          # 'PASS', 'WARN', or 'FAIL'
    reliability_explanation: str

    # --- Posterior metrics (coefficient field a) ---
    a_err: float
    crps_a: float
    coverage_95: float
    ci_width: float
    mean_std: float

    # --- Calibration curve ---
    cal_levels: List[float]
    cal_empirical: List[float]

    # --- Posterior metrics (additional scoring rules) ---
    nll_a: Optional[float] = None

    # --- Posterior metrics (solution field u) ---
    u_err: Optional[float] = None

    # --- Posterior predictive check ---
    chi2_ppc: Optional[float] = None
    chi2_ppc_pvalue: Optional[float] = None

    # --- MAP baseline metrics ---
    map_a_err: Optional[float] = None
    map_u_err: Optional[float] = None

    # --- Condition metadata (optional, used in comparison/sweep) ---
    label: str = ""

    # --- Runtime (seconds, populated by timing instrumentation) ---
    warmup_time_s: Optional[float] = None
    sampling_time_s: Optional[float] = None
    step_time_s: Optional[float] = None

    # --- Spatial error-uncertainty correlation (Gap #6) ---
    spearman_rho_error_std: Optional[float] = None
    spearman_pvalue_error_std: Optional[float] = None

    # --- Per-sample accuracy (piecewise problems only) ---
    a_err_per_sample: Optional[float] = None


# ---------------------------------------------------------------------------
# LaplaceResult
# ---------------------------------------------------------------------------

@dataclass
class LaplaceResult:
    """Diagnostics and posterior metrics for a Laplace approximation run.

    The Laplace approximation fits q(beta) = N(beta_MAP, H^{-1}) where H is
    the Hessian of the negative log-posterior at the MAP estimate.
    """

    # --- Configuration ---
    n_samples: int
    map_max_iter: int
    hessian_reg_lambda: float

    # --- MAP diagnostics ---
    neg_log_posterior_at_map: float
    grad_norm_at_map: float
    map_converged: bool

    # --- Hessian diagnostics ---
    hessian_min_eigenvalue: float
    hessian_condition_number: float
    n_negative_eigenvalues: int
    fraction_clipped: float

    # --- Posterior metrics (coefficient field a) ---
    a_err: float
    crps_a: float
    coverage_95: float
    ci_width: float
    mean_std: float

    # --- Calibration curve ---
    cal_levels: List[float]
    cal_empirical: List[float]

    # --- Configuration (optional) ---
    sigma: Optional[float] = None

    # --- Additional scoring rules ---
    nll_a: Optional[float] = None

    # --- Solution field metrics ---
    u_err: Optional[float] = None

    # --- Posterior predictive check ---
    chi2_ppc: Optional[float] = None
    chi2_ppc_pvalue: Optional[float] = None

    # --- MAP point-estimate errors (from Laplace MAP, not IGNOInverter) ---
    map_a_err: Optional[float] = None
    map_u_err: Optional[float] = None

    # --- Condition metadata ---
    label: str = ""

    # --- Runtime (seconds) ---
    map_time_s: Optional[float] = None
    hessian_time_s: Optional[float] = None
    sampling_time_s: Optional[float] = None

    # --- Spatial error-uncertainty correlation ---
    spearman_rho_error_std: Optional[float] = None
    spearman_pvalue_error_std: Optional[float] = None


# ---------------------------------------------------------------------------
# PriorPredictiveResult
# ---------------------------------------------------------------------------

@dataclass
class PriorPredictiveResult:
    """Metrics from NF prior sampling (no data conditioning)."""

    n_prior: int
    a_err: float
    crps_a: float
    coverage_95: float
    ci_width: float
    mean_std: float


# ---------------------------------------------------------------------------
# ExperimentResult
# ---------------------------------------------------------------------------

@dataclass
class ExperimentResult:
    """Top-level container for one experiment run.

    ``experiment_type`` determines which payload fields are populated:

    * ``"single"``     — ``condition`` holds the one MCMCResult
    * ``"comparison"`` — ``conditions`` maps condition name → MCMCResult
    * ``"sweep"``      — ``baseline`` + ``sweep_conditions`` list the results;
                         ``sweep_var`` names the swept quantity (e.g. "snr_db")
    """

    experiment: str
    problem: str
    experiment_type: str           # "single" | "comparison" | "sweep"
    timestamp: str
    seed: Optional[int] = None
    test_idx: Optional[int] = None

    # --- Prior predictive baseline (Gap #4) ---
    prior: Optional[PriorPredictiveResult] = None
    prior_ood: Optional[PriorPredictiveResult] = None

    # --- Single condition (RQ1 / RQ3) ---
    condition: Optional[MCMCResult] = None

    # --- Comparison (RQ2): named conditions ---
    conditions: Optional[Dict[str, MCMCResult]] = None

    # --- Sweep (RQ4 / RQ5) ---
    sweep_var: Optional[str] = None
    baseline: Optional[MCMCResult] = None
    sweep_conditions: Optional[List[MCMCResult]] = field(default=None)

    # --- Laplace approximation baseline ---
    laplace: Optional[LaplaceResult] = None

    # --- Runtime (seconds, populated by timing instrumentation) ---
    map_time_s: Optional[float] = None
    total_time_s: Optional[float] = None

    # --- Sweep provenance (for condition-split sweep scripts) ---
    sweep_value: Optional[float] = None
    is_sweep_baseline: Optional[bool] = None

    def __post_init__(self) -> None:
        valid = {"single", "comparison", "sweep"}
        if self.experiment_type not in valid:
            raise ValueError(
                f"experiment_type must be one of {valid}, got {self.experiment_type!r}"
            )
