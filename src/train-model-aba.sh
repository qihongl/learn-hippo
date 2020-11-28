#!/bin/bash
#SBATCH -t 19:55:00
#SBATCH -c 1
#SBATCH --mem-per-cpu 4G

#SBATCH --job-name=lcarnn
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=qlu@princeton.edu
#SBATCH --output slurm_log/lcarnn-%j.log

#module load anaconda
DATADIR=/tigress/qlu/logs/learn-hippocampus/log

echo $(date)

srun python -u train-aba.py --exp_name ${1} \
   --subj_id ${2} --penalty ${3} --n_epoch ${4} --sup_epoch ${5} \
   --log_root $DATADIR

echo $(date)
