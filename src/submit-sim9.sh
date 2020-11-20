#!/bin/bash
exp_name=vary-schema-level
n_epoch=1000
n_def_tps=8
similarity_max=.9
similarity_min=0
p_rm_ob_enc=0.3
p_rm_ob_rcl=0.0
sup_epoch=600
penalty_random=0
attach_cond=0

for subj_id in {0..15}
do
   for penalty in 0 2 4
   do
       for def_prob in .25 .35 .45 .55 .65 .75 .85 .95
       do
           sbatch train-model.sh ${exp_name} \
               ${subj_id} ${penalty} \
               ${n_epoch} ${sup_epoch} ${p_rm_ob_enc} ${p_rm_ob_rcl} \
               ${similarity_max} ${similarity_min} \
               ${penalty_random} ${def_prob} ${n_def_tps} ${attach_cond}
       done
   done
done
