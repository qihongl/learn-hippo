#!/bin/bash

exp_name=vary-test-penalty
n_epoch=1000
sup_epoch=600
penalty=4

for subj_id in {0..15}
do
   sbatch train-model-aba.sh $exp_name \
       ${subj_id} ${penalty} ${n_epoch} ${sup_epoch}
done
