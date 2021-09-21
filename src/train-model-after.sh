#!/bin/bash
#SBATCH -t 19:55:00
#SBATCH -c 1
#SBATCH --mem-per-cpu 4G

#SBATCH --job-name=lcarnn
#SBATCH --output slurm_log/lcarnn-%j.log

#module load anaconda

DATADIR=/tigress/qlu/logs/learn-hippocampus/log

echo $(date)

srun python -u train-sl-after.py --exp_name ${1} --subj_id ${2} --penalty ${3}  \
    --n_epoch ${4} --sup_epoch ${5} --similarity_max ${6} --similarity_min ${7} \
    --penalty_random ${8} --def_prob ${9} --n_def_tps ${10} --cmpt ${11} --attach_cond ${12} \
    --enc_size ${13} --dict_len ${14} \
    --log_root $DATADIR

echo $(date)
