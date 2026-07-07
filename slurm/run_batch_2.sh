#!/bin/bash
# =============================================================================
# Batch 2: ood + noise_sweep + sensor_sweep (12 jobs)
#
# Usage:
#   ./run_batch_2.sh                     # all in this batch
#   ./run_batch_2.sh ood_burgers         # pattern match
#   ./run_batch_2.sh --dry-run           # preview
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONTAINER="${SCRIPT_DIR}/jigno.sif"

BATCH_LABEL="Batch 2: ood + noise_sweep + sensor_sweep"
ALL_SCRIPTS=(
    ${SCRIPT_DIR}/experiments/ood_{burgers,darcy_continuous,darcy_piecewise,eit}.py
    ${SCRIPT_DIR}/experiments/noise_sweep_*.py
    ${SCRIPT_DIR}/experiments/sensor_sweep_{burgers,darcy_continuous,darcy_piecewise,eit}.py
)

source "${SCRIPT_DIR}/slurm/slurm_common.sh"
