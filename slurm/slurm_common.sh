#!/usr/bin/env false
# =============================================================================
# slurm_common.sh — shared SLURM submission logic
#
# Sourced by run_experiments.sh and run_batch_*.sh. Do not execute directly.
#
# Callers must set before sourcing:
#   BATCH_LABEL   — display label (e.g. "Batch 1: baseline + map_laplace")
#   ALL_SCRIPTS   — array of experiment .py globs
#   SCRIPT_DIR    — project root (from BASH_SOURCE[0])
#   CONTAINER     — path to jigno.sif
# =============================================================================

DRY_RUN=false
TASK_SINGLE=""
TASK_RANGE=""
PATTERNS=()

for arg in "$@"; do
    case "$arg" in
        --dry-run)  DRY_RUN=true ;;
        --task=*)   TASK_SINGLE="${arg#--task=}" ;;
        --tasks=*)  TASK_RANGE="${arg#--tasks=}" ;;
        *)          PATTERNS+=("$arg") ;;
    esac
done

if [[ -n "$TASK_SINGLE" && -n "$TASK_RANGE" ]]; then
    echo "ERROR: --task and --tasks are mutually exclusive"
    exit 1
fi
if [[ -n "$TASK_SINGLE" && ! "$TASK_SINGLE" =~ ^[0-9]+$ ]]; then
    echo "ERROR: --task requires a non-negative integer (got: $TASK_SINGLE)"
    exit 1
fi
if [[ -n "$TASK_RANGE" && ! "$TASK_RANGE" =~ ^[0-9]+-[0-9]+$ ]]; then
    echo "ERROR: --tasks requires a range in M-N form (got: $TASK_RANGE)"
    exit 1
fi
if [[ -n "$TASK_RANGE" ]]; then
    _rs="${TASK_RANGE%%-*}"
    _re="${TASK_RANGE#*-}"
    if [[ "$_rs" -gt "$_re" ]]; then
        echo "ERROR: --tasks range start must be <= end (got: $TASK_RANGE)"
        exit 1
    fi
fi

MATCHED=()

if [[ ${#PATTERNS[@]} -eq 0 ]]; then
    MATCHED=("${ALL_SCRIPTS[@]}")
else
    for py in "${ALL_SCRIPTS[@]}"; do
        for pat in "${PATTERNS[@]}"; do
            if [[ "$(basename "$py")" == *"$pat"* ]]; then
                MATCHED+=("$py")
                break
            fi
        done
    done
fi

if [[ ${#MATCHED[@]} -eq 0 ]]; then
    echo "No scripts matched patterns: ${PATTERNS[*]}"
    exit 1
fi

if [[ "$DRY_RUN" == false && ! -f "$CONTAINER" ]]; then
    echo "ERROR: container not found: $CONTAINER"
    echo "Run ./slurm/build.sh first"
    exit 1
fi

mkdir -p "${SCRIPT_DIR}/slurm_logs" "${SCRIPT_DIR}/experiments/results"

# ---------------------------------------------------------------------------
# Generate short SLURM job name from script filename
# ---------------------------------------------------------------------------
job_name() {
    local base
    base="$(basename "$1" .py)"
    local prefix problem short
    case "$base" in
        *_darcy_continuous) problem="darcy_continuous"; prefix="${base%_darcy_continuous}" ;;
        *_darcy_piecewise_5v10)   problem="darcy_piecewise_5v10";   prefix="${base%_darcy_piecewise_5v10}" ;;
        *_darcy_piecewise_5v100)  problem="darcy_piecewise_5v100";  prefix="${base%_darcy_piecewise_5v100}" ;;
        *_darcy_piecewise_5v1000) problem="darcy_piecewise_5v1000"; prefix="${base%_darcy_piecewise_5v1000}" ;;
        *_darcy_piecewise)  problem="darcy_piecewise";  prefix="${base%_darcy_piecewise}" ;;
        *_burgers)          problem="burgers";           prefix="${base%_burgers}" ;;
        *_eit)              problem="eit";               prefix="${base%_eit}" ;;
        *)                  problem="$base";             prefix="unknown" ;;
    esac
    case "$problem" in
        darcy_continuous) short="dc" ;;
        darcy_piecewise_5v10)   short="dp10"  ;;
        darcy_piecewise_5v100)  short="dp100" ;;
        darcy_piecewise_5v1000) short="dp1k"  ;;
        darcy_piecewise)  short="dp" ;;
        burgers)          short="bg" ;;
        eit)              short="eit" ;;
        *)                short="$problem" ;;
    esac
    echo "jigno-${prefix}-${short}"
}

# ---------------------------------------------------------------------------
# Time limits: ~25% above measured wall times where data exists.
# Estimates and previously-timed-out experiments left conservative.
# Burgers bumped to account for tune_sigma pilot chain overhead (~15-30min).
# ---------------------------------------------------------------------------
get_time_limit() {
    local base
    base="$(basename "$1" .py)"
    case "$base" in
        # baseline: per-task (1 seed per task)
        baseline_darcy_continuous)  echo "02:30:00" ;;   # ~2h measured
        baseline_eit)               echo "01:30:00" ;;   # ~1.3h measured
        baseline_darcy_piecewise)   echo "02:30:00" ;;   # ~2h measured
        baseline_burgers)           echo "10:00:00" ;;   # ~7.3h + tune_sigma overhead

        # physics: per-task (1 seed per task)
        physics_darcy_continuous)  echo "10:00:00" ;;    # test_idx=1 timed out at 5h
        physics_eit)               echo "06:00:00" ;;    # ~5.3h measured
        physics_darcy_piecewise)   echo "12:00:00" ;;    # task6 (seed=7,test0) timed out at 10h
        physics_burgers)           echo "10:00:00" ;;    # ~7.3h + tune_sigma overhead

        # ood: per-task (1 seed per task)
        ood_darcy_continuous)  echo "12:00:00" ;;        # timed out at 8h previously
        ood_eit)               echo "12:00:00" ;;        # ~10.7h measured
        ood_darcy_piecewise)   echo "36:00:00" ;;        # timed out at 20h previously
        ood_burgers)           echo "16:00:00" ;;        # estimate, kept conservative

        # noise_sweep: per-task (array of 15: 3 seeds × 5 noise levels)
        noise_sweep_darcy_continuous)  echo "02:00:00" ;;   # timed out at 1h previously
        noise_sweep_eit)               echo "01:00:00" ;;   # 15dB SNR timed out at 45min (task4, seed=42)
        noise_sweep_darcy_piecewise)   echo "01:30:00" ;;   # ~1:06h measured
        noise_sweep_burgers)           echo "04:00:00" ;;   # estimate, kept conservative

        # sensor_sweep: per-task (array of 9: 3 seeds × 3 sensor counts)
        sensor_sweep_darcy_continuous)  echo "02:00:00" ;;  # timed out at 1h previously
        sensor_sweep_eit)               echo "00:30:00" ;;  # ~10min measured
        sensor_sweep_darcy_piecewise)   echo "01:30:00" ;;  # ~1:06h measured
        sensor_sweep_burgers)           echo "04:00:00" ;;  # estimate, kept conservative

        # physics_noise_sweep: per-task (array of 15: 3 seeds × 5 noise levels, 2 methods each)
        physics_noise_sweep_darcy_continuous)  echo "04:00:00" ;;   # ~1-3h range, kept conservative
        physics_noise_sweep_eit)               echo "08:00:00" ;;   # SNR=50 timed out at 4h (job 12677400), previously at 2:30h
        physics_noise_sweep_darcy_piecewise)   echo "20:00:00" ;;   # SNR=50,35 timed out at 10h
        physics_noise_sweep_burgers)           echo "08:00:00" ;;   # estimate, kept conservative

        # physics_sensor_sweep: per-task (array of 9: 3 seeds × 3 sensor counts, 2 methods each)
        physics_sensor_sweep_darcy_continuous)  echo "04:00:00" ;;  # timed out at 1h (job 12664417)
        physics_sensor_sweep_eit)               echo "05:00:00" ;;  # up to 3:48h measured
        physics_sensor_sweep_darcy_piecewise)   echo "10:00:00" ;;  # timed out at 4h
        physics_sensor_sweep_burgers)           echo "08:00:00" ;;  # estimate, kept conservative

        # Contrast variants: darcy_piecewise 5v10, 5v100, 5v1000
        baseline_darcy_piecewise_5v10)    echo "02:30:00" ;;  # follows base DP
        baseline_darcy_piecewise_5v100)   echo "10:00:00" ;;  # raised from 6h, kept
        baseline_darcy_piecewise_5v1000)  echo "02:30:00" ;;  # follows base DP
        sensor_sweep_darcy_piecewise_5v10)    echo "01:30:00" ;;  # follows base DP
        sensor_sweep_darcy_piecewise_5v100)   echo "01:30:00" ;;  # follows base DP
        sensor_sweep_darcy_piecewise_5v1000)  echo "01:30:00" ;;  # follows base DP
        ood_darcy_piecewise_5v10)    echo "16:00:00" ;;  # follows base DP ood
        ood_darcy_piecewise_5v100)   echo "16:00:00" ;;  # follows base DP ood
        ood_darcy_piecewise_5v1000)  echo "16:00:00" ;;  # follows base DP ood

        # map_laplace: MAP + Laplace only, <2min compute per seed
        map_laplace_darcy_continuous)       echo "00:30:00" ;;
        map_laplace_eit)                    echo "00:30:00" ;;
        map_laplace_darcy_piecewise)        echo "00:30:00" ;;
        map_laplace_burgers)                echo "00:30:00" ;;
        map_laplace_darcy_piecewise_5v10)   echo "00:30:00" ;;
        map_laplace_darcy_piecewise_5v100)  echo "00:30:00" ;;
        map_laplace_darcy_piecewise_5v1000) echo "00:30:00" ;;

        *)                     echo "08:00:00" ;;   # safe default
    esac
}

# QoS based on required wall time (short<=4h, medium<=36h, long<=7d)
get_qos() {
    local tlimit="$1"
    local hours=$((10#${tlimit%%:*}))
    if [[ "$hours" -le 3 ]]; then
        echo "short"
    elif [[ "$hours" -eq 4 && "${tlimit#*:}" == "00:00" ]]; then
        echo "short"
    else
        echo "medium"
    fi
}

# ---------------------------------------------------------------------------
# Submit jobs
# ---------------------------------------------------------------------------
echo "=== ${BATCH_LABEL} ==="
echo "Scripts: ${#MATCHED[@]}"
echo ""

SUBMITTED=()

for py in "${MATCHED[@]}"; do
    name="$(job_name "$py")"
    pyfile="$(basename "$py")"
    timelimit="$(get_time_limit "$py")"
    qos="$(get_qos "$timelimit")"

    ARRAY_SIZE=$(cd "${SCRIPT_DIR}/experiments" && python3 "${pyfile}" --print-array-size 2>/dev/null || echo "1")

    if [[ -n "$TASK_SINGLE" ]]; then
        if [[ "$TASK_SINGLE" -ge "$ARRAY_SIZE" ]]; then
            echo "ERROR: --task=$TASK_SINGLE out of range for $pyfile (array size: $ARRAY_SIZE, valid: 0-$((ARRAY_SIZE - 1)))"
            exit 1
        fi
        ARRAY_FLAG="--array=$TASK_SINGLE"
        LOG_PATTERN="%A_%a"
    elif [[ -n "$TASK_RANGE" ]]; then
        _re="${TASK_RANGE#*-}"
        if [[ "$_re" -ge "$ARRAY_SIZE" ]]; then
            echo "ERROR: --tasks=$TASK_RANGE out of range for $pyfile (array size: $ARRAY_SIZE, valid: 0-$((ARRAY_SIZE - 1)))"
            exit 1
        fi
        ARRAY_FLAG="--array=$TASK_RANGE"
        LOG_PATTERN="%A_%a"
    elif [[ "$ARRAY_SIZE" -gt 1 ]] 2>/dev/null; then
        ARRAY_FLAG="--array=0-$((ARRAY_SIZE - 1))"
        LOG_PATTERN="%A_%a"
    else
        ARRAY_SIZE=1
        ARRAY_FLAG=""
        LOG_PATTERN="%j"
    fi

    if [[ "$DRY_RUN" == true ]]; then
        if [[ -n "$ARRAY_FLAG" ]]; then
            echo "[dry-run] would submit ${ARRAY_FLAG}: $pyfile  (time: ${timelimit}, job: $name, qos: $qos)"
        else
            echo "[dry-run] would submit: $pyfile  (job: $name, time: $timelimit, qos: $qos)"
        fi
        continue
    fi

    JOB_ID=$(sbatch --parsable \
        --job-name="$name" \
        --partition=general,insy \
        --qos="$qos" \
        --requeue \
        --time="$timelimit" \
        --ntasks=1 \
        --cpus-per-task=1 \
        --mem=8G \
        --gres=gpu:a40:1 \
        ${ARRAY_FLAG:+"$ARRAY_FLAG"} \
        --output="${SCRIPT_DIR}/slurm_logs/slurm_job_${LOG_PATTERN}.out" \
        --error="${SCRIPT_DIR}/slurm_logs/slurm_job_${LOG_PATTERN}.err" \
        --mail-type=END,FAIL \
        --mail-user=h.page@student.tudelft.nl \
        --export="PYFILE=${pyfile}" \
        <<'EOF'
#!/bin/bash
set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR}"
CONTAINER="${PROJECT_DIR}/jigno.sif"

cd "$PROJECT_DIR"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Script: ${PYFILE}"
nvidia-smi

apptainer exec --nv -C \
    --bind "${PROJECT_DIR}:/workspace" \
    --pwd /workspace \
    --env XLA_FLAGS="--xla_gpu_cuda_data_dir=/usr/local/cuda" \
    --env UV_CACHE_DIR=/workspace/.uv_cache \
    --env UV_PYTHON_INSTALL_DIR=/workspace/.uv_python \
    --env SLURM_ARRAY_TASK_ID="${SLURM_ARRAY_TASK_ID:-0}" \
    "$CONTAINER" \
    bash -c "
        TASK_NBFILE=\"${PYFILE%.py}_task\${SLURM_ARRAY_TASK_ID}.ipynb\" && \
        uv run jupytext --to notebook -o \"experiments/\${TASK_NBFILE}\" experiments/${PYFILE} && \
        uv run --extra slurm jupyter nbconvert \
            --to notebook --execute \
            --ExecutePreprocessor.timeout=-1 \
            --output-dir=experiments/results \
            \"experiments/\${TASK_NBFILE}\" && \
        rm -f \"experiments/\${TASK_NBFILE}\"
    "
EOF
    )

    SUBMITTED+=("$JOB_ID $name $pyfile $timelimit $qos $ARRAY_SIZE")
    if [[ "$ARRAY_SIZE" -gt 1 ]]; then
        echo "Submitted $JOB_ID  $name  ($pyfile, array[$ARRAY_SIZE], time: $timelimit/task, qos: $qos)"
    else
        echo "Submitted $JOB_ID  $name  ($pyfile, time: $timelimit, qos: $qos)"
    fi
done

if [[ "$DRY_RUN" == false && ${#SUBMITTED[@]} -gt 0 ]]; then
    echo ""
    echo "=== Summary ==="
    printf "%-10s %-18s %-10s %-8s %s\n" "JOB_ID" "NAME" "TIME" "TASKS" "SCRIPT"
    for entry in "${SUBMITTED[@]}"; do
        read -r job_id jname script tlimit jqos arr_size <<< "$entry"
        printf "%-10s %-18s %-10s %-8s %s\n" "$job_id" "$jname" "$tlimit" "${arr_size}" "$script"
    done
fi
