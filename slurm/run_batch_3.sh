#!/bin/bash
# =============================================================================
# Batch 3: physics (4 jobs)
#
# Usage:
#   ./run_batch_3.sh                           # all in this batch
#   ./run_batch_3.sh physics_darcy             # pattern match
#   ./run_batch_3.sh --dry-run                 # preview
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONTAINER="${SCRIPT_DIR}/jigno.sif"

BATCH_LABEL="Batch 3: physics"
ALL_SCRIPTS=(
    ${SCRIPT_DIR}/experiments/physics_{burgers,darcy_continuous,darcy_piecewise,eit}.py
)

source "${SCRIPT_DIR}/slurm/slurm_common.sh"
