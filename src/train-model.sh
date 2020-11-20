train-sl.sh
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

srun python -u train-sl.py --exp_name ${1} \
    --subj_id ${2} --penalty ${3}  \
    --n_epoch ${4} --sup_epoch ${5} --p_rm_ob_enc ${6} --p_rm_ob_rcl ${7} \
    --similarity_max ${8} --similarity_min ${9} \
    --penalty_random ${10} --def_prob ${11} --n_def_tps ${12} --cmpt ${13} \
    --attach_cond ${14} \
    --log_root $DATADIR

echo $(date)
