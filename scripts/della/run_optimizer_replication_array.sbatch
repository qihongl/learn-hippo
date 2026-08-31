#!/bin/bash
#SBATCH --job-name=em-replicate
#SBATCH --account=knorman
#SBATCH --qos=short
#SBATCH --array=0-9%10
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=slurm-logs/em-replicate-%A_%a.out
#SBATCH --error=slurm-logs/em-replicate-%A_%a.err

set -euo pipefail

: "${LEARN_HIPPO_REPO:?Set LEARN_HIPPO_REPO to the remote repository checkout.}"
: "${LEARN_HIPPO_RUN_ROOT:?Set LEARN_HIPPO_RUN_ROOT to a fresh scratch run directory.}"
: "${LEARN_HIPPO_PYTHON:?Set LEARN_HIPPO_PYTHON to the project Python executable.}"

manifest="$LEARN_HIPPO_REPO/scripts/della/optimizer_replication_array.tsv"
record=$(sed -n "$((SLURM_ARRAY_TASK_ID + 2))p" "$manifest")
IFS=$'\t' read -r experiment config seed <<< "$record"

if [[ -z "$experiment" || -z "$config" || -z "$seed" ]]; then
  echo "No manifest record for array index $SLURM_ARRAY_TASK_ID" >&2
  exit 2
fi

result_directory="$LEARN_HIPPO_RUN_ROOT/results/$experiment"
environment_directory="$LEARN_HIPPO_RUN_ROOT/environment"
mkdir -p "$result_directory" "$environment_directory"
cd "$LEARN_HIPPO_REPO"

export PYTHONPATH="$LEARN_HIPPO_REPO/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"

environment_file="$environment_directory/environment-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.txt"
{
  "$LEARN_HIPPO_PYTHON" --version
  uname -a
  git rev-parse HEAD
  git status --short
  sha256sum "$config"
  "$LEARN_HIPPO_PYTHON" -m pip freeze
} > "$environment_file" 2>&1 || true

"$LEARN_HIPPO_PYTHON" -m boundary_em.run_paper_sampled_hazard \
  "$config" \
  --seed "$seed" \
  --output-directory "$result_directory"
