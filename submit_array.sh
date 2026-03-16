#!/bin/bash
#SBATCH --job-name=vampire_sweep
#SBATCH --account=cheme
#SBATCH --partition=compute
#SBATCH --array=1-600
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=logs/vampire_%A_%a.out
#SBATCH --error=logs/vampire_%A_%a.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=nlsschim@uw.edu

# ── Environment ───────────────────────────────────────────────────────────────
set -euo pipefail

eval "$(micromamba shell hook --shell bash)"
micromamba activate /mmfs1/gscratch/cheme/nlsschim/vampire

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_PATH="/mmfs1/gscratch/cheme/nlsschim/data/cross_species/tifs/specific_studies/adult_wm_gm"
OUTPUT_ROOT="/mmfs1/gscratch/cheme/nlsschim/output/all_species_gm_wm/3_to_7_clusters"
PARAMS_FILE="/mmfs1/gscratch/cheme/nlsschim/TURMorIC/scripts/params.txt"
SCRIPT="/mmfs1/gscratch/cheme/nlsschim/TURMorIC/scripts/run_kfold_vampire.py"

# ── Create log directory ──────────────────────────────────────────────────────
mkdir -p logs

# ── Read parameter line for this array task ───────────────────────────────────
PARAM_LINE=$(grep -v '^\s*#' "$PARAMS_FILE" | grep -v '^\s*$' | sed -n "${SLURM_ARRAY_TASK_ID}p")

if [[ -z "$PARAM_LINE" ]]; then
    echo "ERROR: No parameter line found for task ${SLURM_ARRAY_TASK_ID}"
    exit 1
fi

# Parse columns (tab-separated)
NUM_CLUSTERS=$(echo "$PARAM_LINE" | awk '{print $1}')
NUM_PC=$(echo "$PARAM_LINE"       | awk '{print $2}')
NUM_POINTS=$(echo "$PARAM_LINE"   | awk '{print $3}')
N_FOLDS=$(echo "$PARAM_LINE"      | awk '{print $4}')
RANDOM_STATE=$(echo "$PARAM_LINE" | awk '{print $5}')
RUN_LABEL=$(echo "$PARAM_LINE"    | awk '{print $6}')

# Guard against empty RUN_LABEL which would cause all runs to write to the same dir
if [[ -z "$RUN_LABEL" ]]; then
    echo "ERROR: RUN_LABEL is empty — check params.txt parsing"
    echo "PARAM_LINE was: '${PARAM_LINE}'"
    exit 1
fi

OUTPUT_DIR="${OUTPUT_ROOT}/${RUN_LABEL}"

echo "=================================================="
echo "SLURM array task : ${SLURM_ARRAY_TASK_ID}"
echo "Run label        : ${RUN_LABEL}"
echo "num_clusters     : ${NUM_CLUSTERS}"
echo "num_pc           : ${NUM_PC}"
echo "num_points       : ${NUM_POINTS}"
echo "n_folds          : ${N_FOLDS}"
echo "random_state     : ${RANDOM_STATE}"
echo "output_dir       : ${OUTPUT_DIR}"
echo "params_file      : ${PARAMS_FILE}"
echo "script           : ${SCRIPT}"
echo "=================================================="

# Verify params file is readable
if [[ ! -f "$PARAMS_FILE" ]]; then
    echo "ERROR: params file not found at ${PARAMS_FILE}"
    exit 1
fi

# Verify script exists
if [[ ! -f "$SCRIPT" ]]; then
    echo "ERROR: run script not found at ${SCRIPT}"
    exit 1
fi

# Create output directory explicitly before python runs
mkdir -p "$OUTPUT_DIR"
echo "Output directory created: ${OUTPUT_DIR}"

# ── Run pipeline ──────────────────────────────────────────────────────────────
python "$SCRIPT" \
    --data-path       "$DATA_PATH"    \
    --output-base     "$OUTPUT_DIR"   \
    --num-clusters    "$NUM_CLUSTERS" \
    --num-pc          "$NUM_PC"       \
    --num-points      "$NUM_POINTS"   \
    --n-folds         "$N_FOLDS"      \
    --random-state    "$RANDOM_STATE" \
    --log-level       INFO

echo "Task ${SLURM_ARRAY_TASK_ID} (${RUN_LABEL}) completed successfully."
