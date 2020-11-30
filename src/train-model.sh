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

srun python -u train-sl.py --exp_name ${1} --subj_id ${2} --penalty ${3}  \
    --n_epoch ${4} --sup_epoch ${5} --similarity_max ${6} --similarity_min ${7} \
    --penalty_random ${8} --def_prob ${9} --n_def_tps ${10} --attach_cond ${11} \
    --log_root $DATADIR

echo $(date)
