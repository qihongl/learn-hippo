#!/bin/bash

exp_name=penalty-random-discrete-onehot-new
n_epoch=900
n_branch=4
p_rm_ob_enc=0.3
p_rm_ob_rcl=0.3
n_event_remember=4
pad_len=-1
eta=.1
lr=5e-4
sup_epoch=600
sim_cap=.5
penalty_random=1
penalty_discrete=1
penalty_onehot=1

for subj_id in 0 1
do
    for penalty in 0 2 4
    do
        for n_param in 16
        do
            for enc_size in 16
            do
                for n_hidden in 194
                do
                    for n_hidden_dec in 128
                    do
                        sbatch train.sh $exp_name \
                            ${subj_id} ${penalty} ${n_param} ${n_branch} \
                            ${n_hidden} ${n_hidden_dec} ${eta} ${lr} \
                            $n_epoch ${sup_epoch} \
                            ${p_rm_ob_enc} ${p_rm_ob_rcl} ${n_event_remember} \
                            ${pad_len} ${enc_size} ${sim_cap} \
                            ${penalty_random} ${penalty_discrete} ${penalty_onehot}
                    done
                done
            done
        done
    done
done

