#!/bin/bash

exp_name=vary-test-penalty
n_epoch=1000
def_prob=.25
n_def_tps=0
similarity_max=.9
similarity_min=0
penalty = 4
p_rm_ob_enc=0.3
p_rm_ob_rcl=0.0
sup_epoch=600
cmpt=.8
n_event_remember=2
n_event_remember_aba=2

for subj_id in {0..15}
do
   sbatch train-model-aba.sh $exp_name \
       ${subj_id} ${penalty} ${n_epoch} ${sup_epoch} \
       ${p_rm_ob_enc} ${p_rm_ob_rcl} ${n_event_remember} \
       ${similarity_max} ${similarity_min} \
       ${def_prob} ${n_def_tps} ${cmpt} ${n_event_remember_aba}

done
