#!/bin/bash
# =============================================================================
# Batch 1: baseline + map_laplace (14 jobs)
#
# Usage:
#   ./run_batch_1.sh                     # all in this batch
#   ./run_batch_1.sh baseline_darcy      # pattern match
#   ./run_batch_1.sh --dry-run           # preview
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONTAINER="${SCRIPT_DIR}/jigno.sif"

BATCH_LABEL="Batch 1: baseline + map_laplace"
ALL_SCRIPTS=(
    ${SCRIPT_DIR}/experiments/baseline_*.py
    ${SCRIPT_DIR}/experiments/map_laplace_*.py
)

source "${SCRIPT_DIR}/slurm/slurm_common.sh"
