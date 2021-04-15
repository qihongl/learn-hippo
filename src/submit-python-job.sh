#!/bin/bash
#SBATCH -t 71:55:00
#SBATCH -c 1
#SBATCH --mem-per-cpu 4G

#SBATCH --job-name=mvpa-schema
#SBATCH --output slurm_log/mvpa-run-%j.log

#module load anaconda

DATADIR=/tigress/qlu/logs/learn-hippocampus/log

echo $(date)

srun python -u mvpa-run.py --def_prob ${1}

echo $(date)
