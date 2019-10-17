#!/bin/bash

exp_name=penalty-random-discrete
n_epoch=1000
n_branch=4
def_prob=.25

p_rm_ob_enc=0.3
p_rm_ob_rcl=0.3
pad_len=-1
eta=.1
lr=7e-4
sup_epoch=600
sim_cap=.75

penalty_random=0
penalty_discrete=1
penalty_onehot=0

normalize_return=1

n_event_remember=2

for subj_id in {0..4}
do
    for penalty in 4
    do
        for n_param in 16
        do
            for enc_size in 16
            do
                for n_event_remember_aba in 2 4
                do
                    for n_hidden in 194
                    do
                        for n_hidden_dec in 128
                        do
                            sbatch train-aba.sh $exp_name \
                                ${subj_id} ${penalty} ${n_param} ${n_branch} \
                                ${n_hidden} ${n_hidden_dec} ${eta} ${lr} \
                                $n_epoch ${sup_epoch} \
                                ${p_rm_ob_enc} ${p_rm_ob_rcl} ${n_event_remember} \
                                ${pad_len} ${enc_size} ${sim_cap} \
                                ${penalty_random} ${penalty_discrete} ${penalty_onehot}\
                                ${normalize_return} ${def_prob} ${n_event_remember_aba}
                        done
                    done
                done
            done
        done
    done
done

