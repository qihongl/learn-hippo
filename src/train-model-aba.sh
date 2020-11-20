#!/bin/bash
#SBATCH -t 13:55:00
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
   --p_rm_ob_enc ${6} --p_rm_ob_rcl ${7} --n_event_remember ${8} \
   --similarity_max ${9} --similarity_min ${10} --penalty_random ${11} \
   --def_prob ${12} --n_def_tps ${13} --cmpt ${14} --n_event_remember_aba ${15} \
   --log_root $DATADIR

echo $(date)
