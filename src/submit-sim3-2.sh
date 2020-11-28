#!/bin/bash

exp_name=low-similarity

n_def_tps=0
similarity_max=.4
similarity_min=0
sup_epoch=600
n_epoch=1000
penalty_random=1
attach_cond=0

for subj_id in {0..15}
do
   for penalty in 4
   do
       for def_prob in .25
       do
           sbatch train-model.sh ${exp_name} \
               ${subj_id} ${penalty} ${n_epoch} ${sup_epoch} \
               ${similarity_max} ${similarity_min} \
               ${penalty_random} ${def_prob} ${n_def_tps} ${attach_cond}
       done
   done
done
