#!/usr/bin/env bash
# Submit one Slurm array task for each trial in an external search JSON file.

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: submit_sweep.sh --search-config PATH [--max-parallel N] [--dry-run]

PATH must be visible on the Slurm compute nodes. The batch script mounts the
configuration read-only inside each training container.
EOF
}

die() {
    echo "submit_sweep: $*" >&2
    exit 2
}

SEARCH_CONFIG=""
MAX_PARALLEL=4
DRY_RUN=0

# Parse submission settings on the login node, before asking Slurm for jobs.
while (($#)); do
    case "$1" in
        --search-config)
            (($# >= 2)) || die "--search-config needs a path"
            SEARCH_CONFIG=$2
            shift 2
            ;;
        --max-parallel)
            (($# >= 2)) || die "--max-parallel needs a number"
            MAX_PARALLEL=$2
            shift 2
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            die "unknown argument: $1 (use --help for usage)"
            ;;
    esac
done

[[ -n "$SEARCH_CONFIG" ]] || die "--search-config is required"
[[ -f "$SEARCH_CONFIG" ]] || die "configuration file does not exist: $SEARCH_CONFIG"
[[ "$MAX_PARALLEL" =~ ^[1-9][0-9]*$ ]] || die "--max-parallel must be a positive integer"
command -v jq >/dev/null || die "jq is required to read the search configuration"

# Convert to an absolute path. sbatch exports it to the compute node, where
# sweep.sbatch checks and mounts this same host path into the container.
CONFIG_DIR=$(cd -- "$(dirname -- "$SEARCH_CONFIG")" && pwd -P)
SEARCH_CONFIG="$CONFIG_DIR/$(basename -- "$SEARCH_CONFIG")"
export SEARCH_CONFIG

# Slurm reads #SBATCH directives before running the batch script, so the array
# size must be supplied on the sbatch command line. Check the basic JSON shape
# here; train_search.py performs the full parameter validation in each task.
if ! jq -e '.format_version == 1 and (.trials | type) == "array" and (.trials | length) > 0' \
    "$SEARCH_CONFIG" >/dev/null; then
    die "expected a version 1 search JSON file with a nonempty trials array"
fi
TRIAL_COUNT=$(jq -r '.trials | length' "$SEARCH_CONFIG")
LAST_INDEX=$((TRIAL_COUNT - 1))
ARRAY_RANGE="0-${LAST_INDEX}%${MAX_PARALLEL}"
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)

if ((DRY_RUN)); then
    echo "Would launch a Slurm job array from: $SEARCH_CONFIG"
else
    echo "Launching a Slurm job array from: $SEARCH_CONFIG"
fi
echo "Trial count: $TRIAL_COUNT (task IDs 0-$LAST_INDEX)"
echo "Maximum concurrent tasks: $MAX_PARALLEL"
echo "Hyperparameters by task:"
jq -r '.trials | to_entries[] |
    "  task \(.key): learning_rate=\(.value.learning_rate) batch_size=\(.value.batch_size) hidden_width=\(.value.hidden_width)"' \
    "$SEARCH_CONFIG"

if ((DRY_RUN)); then
    echo "Dry run: sbatch --array=$ARRAY_RANGE --export=ALL $SCRIPT_DIR/sweep.sbatch"
    exit 0
fi

command -v sbatch >/dev/null || die "sbatch is not available on this host"
sbatch --array="$ARRAY_RANGE" --export=ALL "$SCRIPT_DIR/sweep.sbatch"
