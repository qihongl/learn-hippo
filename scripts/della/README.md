# Della execution notes

These files parallelize independent model seeds. They do not request a GPU because
the current implementation is CPU-only. Do not run training on a login node.

## Required account-specific paths

The submitter must set three variables rather than editing scientific configuration:

- `LEARN_HIPPO_REPO`: a small checkout under the account's home directory;
- `LEARN_HIPPO_RUN_ROOT`: a new directory under
  `/scratch/gpfs/<ResearchGroup>/<NetID>/learn-hippo/runs/<date>-<git-sha>`;
- `LEARN_HIPPO_PYTHON`: an existing Python executable containing the dependencies.

The research-group name and NetID are intentionally not guessed. Scratch is for the
active jobs and is not backed up. After validation, copy compact final results to the
research group's `/projects` space or TigerData and fetch every seed JSON, Slurm log,
and environment record into a new immutable local results directory.

## Submission

From the fresh scratch run directory, make the log directory and submit the versioned
script:

```bash
mkdir -p slurm-logs
export LEARN_HIPPO_REPO=/home/<NetID>/learn-hippo
export LEARN_HIPPO_RUN_ROOT="$PWD"
export LEARN_HIPPO_PYTHON=/path/to/existing/python
sbatch "$LEARN_HIPPO_REPO/scripts/della/run_paper_task_encoding_array.sbatch"
```

Use `squeue -u "$USER"` to monitor the array and `jobstats <jobid>` after tasks have
run long enough to estimate their actual CPU and memory use. The array currently has
20 records and permits at most 10 simultaneous tasks.

## Optimizer-stability screen

The post-report stability experiment has its own 12-task manifest and submission
script. It uses paired seeds across four cells and writes all large artifacts below
the supplied scratch run root:

```bash
mkdir -p "$LEARN_HIPPO_RUN_ROOT/slurm-logs"
sbatch --export=ALL,LEARN_HIPPO_REPO,LEARN_HIPPO_RUN_ROOT,LEARN_HIPPO_PYTHON \
  "$LEARN_HIPPO_REPO/scripts/della/run_optimizer_stability_array.sbatch"
```

For the KNORMAN allocation, use a fresh directory below
`/scratch/gpfs/KNORMAN/qlu/learn-hippo/runs/<date>-<git-sha>/optimizer-stability`.
