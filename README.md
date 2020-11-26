## About this repo

This repo hosts the python code for the following paper: 
> Learning to use episodic memory for event prediction (2020) Qihong Lu, Uri Hasson, Kenneth A Norman. bioRxiv.

This document contains an instruction about how to replicate all results figures in the paper. Most simulations can be replicated in a day if you have access to a cluster. 

Feel free to contact me if you have any question / comment. Thank you very much ahead! 

## Dependencies 

I used python 3.6.9 for this project. The main dependencies are pytorch, numpy, scikit-learn, scipy, matplotlib, seaborn, dabest, bidict. I think the code should work as long as your packages are relatively up to date, but just in case, the full dependencies and their version information are listed in this [txt file](https://github.com/qihongl/learn-hippo/blob/master/dep.txt). 

I used a cluster to parallelize model training, so that the training step for most simulations can be done in a day. The cluster I used at Princeton uses [Slurm](https://slurm.schedmd.com/documentation.html). So depends on where you are, the job submission files I provided might not work for you. However, the conversion should be relatively simple. 

And we will try to host pre-trained weights somewhere asap. 

## General procedure 

### 0. Download the code
First, you need to clone this repo: 
```sh
git clone https://https://github.com/qihongl/learn-hippo
```
### 1. Model training 
I provided job submission files for all simulations: `src/submit-sim*.sh`. For example, [src/submit-sim2.sh](https://github.com/qihongl/learn-hippo/blob/master/src/submit-sim2.sh) is the job submission file for simulation 2. To execute this file, simply go to the `src/` folder and type 

```sh
./submit-sim2.sh
```

Executing this file will submit 15 jobs to train 15 models in parallel with the specified simulation parameters. `submit-sim2.sh` will specify the parameter in [train-model.sh](https://github.com/qihongl/learn-hippo/blob/master/src/train-model.sh), then it will submits a python job with the following command: 

```sh
srun python -u train-sl.py --exp_name ${1} --subj_id ${2} --penalty ${3}  \
    --n_epoch ${4} --sup_epoch ${5} --p_rm_ob_enc ${6} --p_rm_ob_rcl ${7} \
    --similarity_max ${8} --similarity_min ${9} --penalty_random ${10} \
    --def_prob ${11} --n_def_tps ${12} --attach_cond ${13} \
    --log_root $DATADIR
```

The code block attached above also clarify how to train a model on any platform with any parameter configuration. Suppose you want to train the model with some parameter configuation `{1}`, `{2}`, ... `{13}`, simply run `python train-sl.py --exp_name ${1} --subj_id ${2} ... --attach_cond ${13}`. What these variable names correspond to are explained in this [wiki document](url). 


### 2. Model evaluation 
The model training script will evaluate the model on a test set by default. However, some simulations simply test previously trained models on some other data set or test previously trained models with their hippocampal module removed. So we need a way to evalute trained models on some test set with arbitrary condition. 

To evaluate some trained model, go to `src/` and type: 

```sh
./python eval-group.py
```

### 3.Visualize the data 

To visualize some basic results, go to `src/` and type: 

```sh
python vis-data.py
```

### Contact

Qihong Lu (qlu@princeton.edu)
