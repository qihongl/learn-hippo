#!/bin/bash
#SBATCH --job-name=boundary-em
#SBATCH --array=0-19%10
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=slurm-logs/boundary-em-%A_%a.out

set -euo pipefail

: "${LEARN_HIPPO_REPO:?Set LEARN_HIPPO_REPO to the remote repository checkout.}"
: "${LEARN_HIPPO_RUN_ROOT:?Set LEARN_HIPPO_RUN_ROOT to a fresh scratch run directory.}"
: "${LEARN_HIPPO_PYTHON:?Set LEARN_HIPPO_PYTHON to an existing Python executable with the project dependencies.}"

manifest="$LEARN_HIPPO_REPO/scripts/della/paper_task_encoding_array.tsv"
record=$(sed -n "$((SLURM_ARRAY_TASK_ID + 2))p" "$manifest")
IFS=$'\t' read -r experiment config seed <<< "$record"

if [[ -z "$experiment" || -z "$config" || -z "$seed" ]]; then
  echo "No manifest record for array index $SLURM_ARRAY_TASK_ID" >&2
  exit 2
fi

task_output="$LEARN_HIPPO_RUN_ROOT/$experiment"
mkdir -p "$task_output" "$LEARN_HIPPO_RUN_ROOT/environment"
cd "$LEARN_HIPPO_REPO"

export PYTHONPATH="$LEARN_HIPPO_REPO/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"

"$LEARN_HIPPO_PYTHON" --version \
  > "$LEARN_HIPPO_RUN_ROOT/environment/python-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.txt"
"$LEARN_HIPPO_PYTHON" -m pip freeze \
  > "$LEARN_HIPPO_RUN_ROOT/environment/packages-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.txt"

"$LEARN_HIPPO_PYTHON" -m boundary_em.run_paper_sampled_hazard \
  "$config" \
  --seed "$seed" \
  --output-directory "$task_output"
