#!/bin/bash
exp_name=vary-schema-level-ig.3-pfixed
n_epoch=1000
n_def_tps=8
similarity_max=.9
similarity_min=0
sup_epoch=600
penalty_random=0
attach_cond=0
cmpt=.8
enc_size=16
dict_len=2

for subj_id in {0..15}
do
   for penalty in 0 2 4
   do
       for def_prob in .25 .35 .45 .55 .65 .75 .85 .95
       do
           sbatch train-model-after.sh ${exp_name} ${subj_id} ${penalty} \
               ${n_epoch} ${sup_epoch} ${similarity_max} ${similarity_min} \
               ${penalty_random} ${def_prob} ${n_def_tps} ${cmpt} ${attach_cond} \
               ${enc_size} ${dict_len}

       done
   done
done
