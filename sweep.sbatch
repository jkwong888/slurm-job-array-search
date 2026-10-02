#!/bin/bash
#SBATCH --job-name=train-search
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --gpus=1
#SBATCH --time=01:00:00
# Pass --array at submission time; submit_sweep.sh can calculate it from JSON.
#SBATCH --output=train-search-%A_%a.log

set -euo pipefail

TRAIN_IMAGE=${TRAIN_IMAGE:-cr.uk-south1.nebius.cloud#e03s4vtyhkerraexbw/train_search:cu130}
# Results and the prepared dataset live on shared storage outside the image.
SWEEP_DIR=${SWEEP_DIR:-/data/test}
DATASET_PATH=${DATASET_PATH:-/data/datasets/california_housing}
SEARCH_CONFIG=${SEARCH_CONFIG:-}

if [[ "$SWEEP_DIR" != /* ]]; then
    echo "SWEEP_DIR must be an absolute path" >&2
    exit 2
fi
if [[ "$DATASET_PATH" != /* || ! -d "$DATASET_PATH" ]]; then
    echo "DATASET_PATH must name an existing absolute-path dataset directory: $DATASET_PATH" >&2
    exit 2
fi
for split in train test validate; do
    if [[ ! -f "${DATASET_PATH}/${split}.parquet" ]]; then
        echo "missing dataset split: ${DATASET_PATH}/${split}.parquet" >&2
        exit 2
    fi
done
if [[ "$SEARCH_CONFIG" != /* || ! -f "$SEARCH_CONFIG" ]]; then
    echo "SEARCH_CONFIG must name an existing absolute-path JSON file: $SEARCH_CONFIG" >&2
    exit 2
fi
if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    echo "submit this sweep with sbatch --array=... or submit_sweep.sh" >&2
    exit 2
fi

mkdir -p "$SWEEP_DIR"
SEARCH_DIR=$(dirname "$SEARCH_CONFIG")
SEARCH_NAME=$(basename "$SEARCH_CONFIG")

echo "Host GPU allocation:"
# Check the allocation before Pyxis starts the container.
srun nvidia-smi -L

# Every task reads the same dataset and search plan. The training script
# creates /results/<array-job-id>/<task-id>/ for its model and metrics.
srun \
    --container-image="$TRAIN_IMAGE" \
    --container-mounts="$SWEEP_DIR:/results,$DATASET_PATH:/dataset:ro,$SEARCH_DIR:/search:ro" \
    --no-container-entrypoint \
    python /app/train_search.py \
    --dataset-path "/dataset" \
    --search-config "/search/$SEARCH_NAME" \
    --device cuda \
    --trial-index "$SLURM_ARRAY_TASK_ID" \
    --output-dir "/results/${SLURM_ARRAY_JOB_ID}"
