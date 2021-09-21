#!/bin/bash

exp_name=vary-test-penalty-after-ig.3-enc8d4

n_def_tps=0
similarity_max=.9
similarity_min=0
sup_epoch=600
n_epoch=1000
penalty_random=1
attach_cond=0
cmpt=.8

enc_size=8
dict_len=4

for subj_id in {0..15}
do
   for penalty in 4
   do
       for def_prob in .25
       do
           sbatch train-model-after.sh ${exp_name} \
               ${subj_id} ${penalty} ${n_epoch} ${sup_epoch} \
               ${similarity_max} ${similarity_min} \
               ${penalty_random} ${def_prob} ${n_def_tps} ${cmpt} ${attach_cond} \
               ${enc_size} ${dict_len}
       done
   done
done
