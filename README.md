## About this repo

This repo hosts the python code for the following paper: 
[Learning to use episodic memory for event prediction (2020) Qihong Lu, Uri Hasson, Kenneth A Norman. bioRxiv.](url)

This document contains an instruction about how to replicate all results figures in the paper. Most simulations can be replicated in a day if you have access to a cluster. 

Feel free to contact me if you have any question / comment. Thank you very much ahead! 

## Dependencies 

I used python 3.6.9 for this project. The main dependencies are pytorch, numpy, scikit-learn, scipy, matplotlib, seaborn, dabest, bidict. I think the code should work as long as your packages are relatively up to date, but just in case, the full dependencies and their version information are listed in this [txt file](https://github.com/qihongl/learn-hippo/blob/master/dep.txt). 

I used a cluster to parallelize model training, so that the training step for most simulations can be done in a day. The cluster I used at Princeton uses [Slurm](https://slurm.schedmd.com/documentation.html). So depends on where you are, the job submission files I provided might not work for you. However, the conversion should be relatively simple. 

And we will try to host pre-trained weights somewhere asap. 

## General procedure 

Here we introduce the general procedure of how to replicate any simulation in the paper and explain the logic of the code. We will use simulation 2 as an example, since many simulations depends on it. 

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

Executing this file will submit 15 jobs to train 15 models in parallel with the specified simulation parameters. Specifically, `submit-sim2.sh` will fill in the parameters in [train-model.sh](https://github.com/qihongl/learn-hippo/blob/master/src/train-model.sh), then `train-mode.sh` will submits a python job with the following command: 

```sh
srun python -u train-sl.py --exp_name ${1} --subj_id ${2} --penalty ${3}  \
    --n_epoch ${4} --sup_epoch ${5} --p_rm_ob_enc ${6} --p_rm_ob_rcl ${7} \
    --similarity_max ${8} --similarity_min ${9} --penalty_random ${10} \
    --def_prob ${11} --n_def_tps ${12} --attach_cond ${13} \
    --log_root $DATADIR
```

The code block attached above also clarifies how to train a model on any platform with any parameter configuration. Namely, suppose you want to train the model with some parameter configuation `exp_name = {1}`, `subj_id = {2}`, `penalty = {4}`... `attach_cond = {13}`, simply run `python train-sl.py --exp_name ${1} --subj_id ${2} --penalty ${4} ... --attach_cond ${13}`. 

Here's a brief summary of what these parameters mean: 
```
exp_name - the name of the experiment, only affects the directory name where the data will be saved
subj_ids - the id of the subject, only affects the directory name where the data will be saved
penalty - if penalty_random is 0 (false), then this is the penalty of making a prediction mistake; else penalty_random is 1, then the penalty value will be sampled from uniform[0, penalty]. 
n_epoch - the total number of training epoch 
sup_epoch - the number of supervised pre-training epoch 
p_rm_ob_enc - the probability of withholding observation before part 2
p_rm_ob_rcl - the probability of withholding observation during part 2
similarity_max - the maximum event similarity in a single trial of experiment
similarity_min - the minimum event similarity in a single trial of experiment
penalty_random - see description for penalty 
def_prob - the probability that the prototypical event happens
n_def_tps - the number of time points with a prototypical event
attach_cond - if 1 (true), attach the familiarity signal to the input; if (0) false, doesn't affect the input at all
```

A more detailed description of all parameters are [here](url). 

### 2. Model evaluation 
The model training script will evaluate the model on a test set by default. However, some simulations simply test previously trained models on some other data set or test previously trained models with their hippocampal module removed. So we need a way to evalute trained models on some test set with arbitrary condition. 

To evaluate some trained model, go to `src/` and configure the simulation parameters in `eval-group.py` to specify which simulation are you running, then run the evaluation script: 

```sh
python eval-group.py
```

This script will use the input variables to locate the pre-trained models, test those models on a test set, then save the data. Note that the input variables here must match the input variables use in model training (step 1), otherwise the script won't be able locate the pre-trained models. 

### 3.Visualize the data 

To visualize some basic results, go to `src/` and configure the simulation parameters in `eval-group.py` to specify which simulation are you running. Then run the visualization script: 

```sh
python vis-data.py
```

Almost all figures (except for Figure 8, which involves MVPA decoding) from simulation 1 to 5 can be created using this script, with the 3 steps discussion above. Then for other simulations, please refer to the detail instruction for each simulation.  

Note that this script will use the input variables to locate the data and then make plots. Note that the input variables here must match the input variables use in model training (step 1), otherwise the script won't be able locate the data. 

## Detailed instruction for all simulations 

### Simulation 1 

For this simulation, simply train the model and visualize the data
```sh
./submit-sim1.sh
python vis-data.py
```

### Simulation 2

For this simulation, simply train the model and visualize the data
```sh
./submit-sim2.sh
python vis-data.py
```

### Simulation 3 

For this simulation, train the model and visualize the data
```sh
./submit-sim3-1.sh
./submit-sim3-2.sh
python vis-data.py
```

### Simulation 4

For this simulation, train the model and visualize the data
```sh
./submit-sim4.sh
python vis-data.py
```

### Simulation 5 

This simulation re-use the models trained in simulation 2. First, re-evaluate the model by setting the encoding size the `n_param / 2`, which will let the model to encode episodic memories midway through an event sequence. Then you can visualize the data to see that their performance is worse. 
```sh
python vis-data.py
```

### Simulation 6

This simulation re-use the models trained in simulation 2. First, re-evaluate the model: 
```sh
python eval-group.py
```


Then run the following code the run the inter-subject analysis and plot the data: 
```sh
python vis-isc.py
```


### Simulation 7 

This simulation re-use the models trained in simulation 2. First, re-evaluate the model: 
```sh
python eval-group.py
```
Then run the following code the run the inter-subject analysis and plot the data: 
```sh
python vis-zuo-scramble.py
```

### Simulation 8

In this simulation, we need to train the models in simulation 2 further on a new task. 

```sh
./submit-sim8.sh
```

Then visualize the data 
```sh
python vis-aba.py
```

This script performs MVPA analysis and plots the result
```sh
python mvpa-aba.py
```


### Simulation 9 

For this simulation, train the model and visualize the data
```sh
./submit-sim9.sh
python vis-data.py
```

To perform MVPA analysis and visualize the data
```sh
python mvpa-run.py
python mvpa-plot.py
```


### Contact

Qihong Lu (qlu@princeton.edu)
