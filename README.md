## About this repo

This repo hosts the python code for the following paper: 
[Learning to use episodic memory for event prediction (2020) Qihong Lu, Uri Hasson, Kenneth A Norman. bioRxiv.](url)

If you have access to a cluster, most simulations can be replicated in a day. 

Feel free to contact me if you have any question / comment. Thank you very much ahead! 

## Playing with some pre-trained models

If you would like to play with the model, here's a [Code Ocean capsule](https://codeocean.com/) we created, which contains the code in this repo and some pretrained weights.  

[URL]()

You can run Simulation 1, 2, 4, and 5 and see the results 10 minutes. Once you are in the Code Ocean capsule, simply click **reproducible run**. To specify which simulation you would like to run, go to `code/src/demo.py`, on line 31, change `simulation_id` to 1, 2, 4, or 5, and then click **reproducible run** again. Once you are done, click the figures listed in the timeline (right hand side, under reproducible run). 

## Dependencies 

This is a python-based project. The list of dependencies and their version information are listed [here](https://github.com/qihongl/learn-hippo/blob/master/requirement.txt). 
The code should work as long as the version of your packages are close to what I used. 

I used a cluster to parallelize model training, so that most simulations took less than a day. The cluster I used at Princeton uses [Slurm](https://slurm.schedmd.com/documentation.html). So depends on where you are, the job submission files 
(e.g. 
[src/submit-sim1.sh](https://github.com/qihongl/learn-hippo/blob/master/src/submit-sim1.sh)
[src/train-model.sh](https://github.com/qihongl/learn-hippo/blob/master/src/train-model.sh)
)
I wrote might not work for you. 
However, these scripts simply documents the simulation parameters I used and it is relatively simple to adapt them for your cluster/platform (see the section on **Replicate the simulation results - general guidelines**). 

## What's in the repo

Here's the structure of this repo

```sh
├── demo-log    # some data for the demo
├── log         # the logging directory 
├── figs        # figure directory, most scripts save figs to here 
└── src         # source code
    ├── analysis        # helper functions for analyzing the data 
    │   ├── __init__.py
    │   ├── behav.py
    │   ├── general.py
    │   ├── neural.py
    │   ├── preprocessing.py
    │   ├── task.py
    │   └── utils.py
    ├── examples        # some simple demos 
    │   ├── event-empirical-similarity.py
    │   ├── event-similarity-cap.py
    │   ├── event-similarity.py
    │   ├── memory_benefit.py
    │   ├── schema-regularity.py
    │   ├── stimuli-representation.py
    │   ├── true-uncertainty.py
    │   └── zuo-2019-control-patient-isc-logic.py
    ├── models         # the components of the model 
    │   ├── A2C.py             # the standard actor-critic algorithm 
    │   ├── EM.py              # the episodic memory module
    │   ├── LCALSTM.py         # the model 
    │   ├── LCA_pytorch.py     # a pytorch implementation of leaking competing accumulator model
    │   ├── _rl_helpers.py     # some RL helpers 
    │   ├── initializer.py     # weight initializer
    │   ├── metrics.py         # some helper functions for metrics 
    │   ├── __init__.py    
    ├── task            # the definition of the task 
    │   ├── Schema.py           
    │   ├── SequenceLearning.py
    │   ├── StimSampler.py
    │   ├── utils.py
    │   └── __init__.py
    ├── utils           # general utility functions
    │   ├── constants.py
    │   ├── io.py
    │   ├── params.py
    │   ├── utils.py
    │   └── __init__.py    
    ├── vis             # code for visualizing the data
    │   ├── _utils.py
    │   ├── _vis.py
    │   └── __init__.py    
    ├── submit-sim1.sh              # the script for submitting jobs for simulation 1, triggers train-model.sh
    ├── submit-sim2.sh              # the script for submitting jobs for simulation 2, triggers train-model.sh
    ├── submit-sim3-1.sh            # the script for submitting jobs for simulation 3 (part one), triggers train-model.sh
    ├── submit-sim3-2.sh            # the script for submitting jobs for simulation 3 (part two), triggers train-model.sh
    ├── submit-sim4.sh              # the script for submitting jobs for simulation 4, triggers train-model.sh
    ├── submit-sim8.sh              # the script for submitting jobs for simulation 8, triggers train-model-aba.sh
    ├── submit-sim9.sh              # the script for submitting jobs for simulation 9, triggers train-model.sh
    ├── train-model-aba.sh          # submit a python job to train a model on the ABA experiment by Chang et al. 2020
    ├── train-model.sh              # submit a python job to train a model for the twilight zone experiment by Chen et al. 2016
    ├── demo.py                     # a demo for Code Ocean
    ├── eval-group.py               # evaluate a group of models 
    ├── exp_aba.py                  # definition of the ABA experiment by Chang et al. 2020
    ├── exp_tz.py                   # definition of the twilight zone experiment by Chen et al. 2016
    ├── mvpa-aba.py                 # run MVPA analysis on the ABA experiment 
    ├── mvpa-plot.py                # plot MVPA results from mvpa-run.py
    ├── mvpa-run.py                 # run MVPA analysis (mainly for simulation 9, but useful for other simulations except for simulation 8)
    ├── train-aba.py                # train the model for simulation 8
    ├── train-sl.py                 # train the model (except for simulation 8)
    ├── vis-aba.py                  # visualize the data for simulation 8
    ├── vis-data.py                 # visualize some basic results  
    ├── vis-inpt-by-schematicity.py # visualize the effect of schema level on input gate values (see simulation 9)
    ├── vis-isc.py                  # visualize the ISC analysis (see simulation 6)
    ├── vis-policy-adjustment.py    # visualize the how the model adjusts its policy according to the penalty level 
    ├── vis-policy-diff.py          # visualize the how the model models trained in different penalty levels respond differently
    └── vis-zuo-scramble.py         # visualize the results for the scrambling analysis (see simulation 7)
```    

## Replicate the simulation results - general guidelines

Here we introduce the general procedure of how to replicate any simulation in the paper. We will use simulation 2 as an example. 

### 0. Download the code
First, you need to clone this repo: 
```sh
git clone https://https://github.com/qihongl/learn-hippo
```
### 1. Model training 
I provided job submission files for all simulations: `src/submit-sim*.sh`. 
For example, [src/submit-sim2.sh](https://github.com/qihongl/learn-hippo/blob/master/src/submit-sim2.sh) is the job submission file for simulation 2. Executing this file will train 15 models in parallel with the specified simulation parameters. Note that the job of `submit-sim*.sh` is to specify simulation parameters and trigger 
[train-model.sh](https://github.com/qihongl/learn-hippo/blob/master/src/train-model.sh). 
Then `train-model.sh` takes those simulation parameters and run a python program that trains the model, which is general across simulations. 

Several things to check before you run this script. 

1. In `train-model.sh` ([line 11](https://github.com/qihongl/learn-hippo/blob/master/src/train-model.sh#L11)), you need to set the `DATADIR`, the logging directory, to something that exisits on your machine. Then the data (e.g. trained network weights) will be saved to this directory. Later on, other programs, such as the code for visualizing data, will need to access this directory to find the weights for the trained models. 

2. In `train-model.sh` ([line 7](https://github.com/qihongl/learn-hippo/blob/master/src/train-model.sh#L7)), you need to set where to log the output (from the python script), to a directory that exists on your machine. This is useful if you want to inspect the training process. 

To train models for simulation 2, simply go to the `src/` folder and type the following: 

```sh
./submit-sim2.sh
```

This will trigger `train-mode.sh` and it will submits a python job with the following command with the specified simulation parameters: 

```sh
srun python -u train-sl.py --exp_name ${1} --subj_id ${2} --penalty ${3}  \
    --n_epoch ${4} --sup_epoch ${5} --p_rm_ob_enc ${6} --p_rm_ob_rcl ${7} \
    --similarity_max ${8} --similarity_min ${9} --penalty_random ${10} \
    --def_prob ${11} --n_def_tps ${12} --attach_cond ${13} \
    --log_root $DATADIR
```

The code above clarifies how to train a model on any platform with any parameter configuration. Suppose you want to train the model with some parameter configuation `exp_name = {1}`, `subj_id = {2}`, `penalty = {4}`... `attach_cond = {13}`, simply run `python train-sl.py --exp_name ${1} --subj_id ${2} --penalty ${4} ... --attach_cond ${13}`. This works on your laptop too. 

Here's a brief summary of what these parameters mean: 

`exp_name` - the name of the experiment, only affects the directory name where the data will be saved

`subj_ids` - the id of the subject, only affects the directory name where the data will be saved

`penalty` - if penalty_random is 0 (false), then this is the penalty of making a prediction mistake; else penalty_random is 1, then the penalty value will be sampled from uniformly from 0 up to the input value. 

`n_epoch` - the total number of training epoch 

`sup_epoch` - the number of supervised pre-training epoch 

`p_rm_ob_enc` - the probability of withholding observation before part 2

`p_rm_ob_rcl` - the probability of withholding observation during part 2

`similarity_max` - the maximum event similarity in a single trial of experiment

`similarity_min` - the minimum event similarity in a single trial of experiment

`penalty_random` - see description for penalty 

`def_prob` - the probability that the prototypical event happens

`n_def_tps` - the number of time points with a prototypical event

`attach_cond` - if 1 (true), attach the familiarity signal to the input; if (0) false, doesn't affect the input at all


### 2. Model evaluation 

`eval-group.py` evaluates some pre-trained models on some tasks with the specified simulation parameters. Actually, the training script evaluates the model on some test set by default, but in some simulations, I test the pre-trained models on some new tasks that the model hasn't been trained on. `eval-group.py` is a generic evaluation script that allows me to do that. 

To evaluate some trained model, go to `src/` and configure the simulation parameters in `eval-group.py` to specify which simulation are you running, then run the evaluation script: 

```sh
python eval-group.py
```

This script will use the input simulation parameters to locate the pre-trained models, test those models on a test set, then save the data. Note that the input simulation parameters here must match the simulation parameters you used for model training (step 1), otherwise this script won't be able locate the pre-trained models. 

### 3.Visualize the data 

To visualize the results, go to `src/` and configure the simulation parameters in `vis-data.py` to specify the simulation parameters. Then run the visualization script: 

```sh
python vis-data.py
```

Note that this script will use the input simulation parameters to locate the data and then make plots. So input simulation parameters here must match the simulation parameters used for model training (step 1), otherwise the script won't be able locate the data. 

## Specific instruction for each simulation

This section lists the scripts you need to replicate every simulation in the paper. Note that when you use the python scripts to visualize the data (e.g. `vis-*.py`), the input parameters must match the parameters used in the training scripts. This enable the python script to find the location of the saved data. 

### Simulation 1 

First, train the models 
```sh
./submit-sim1.sh
```
Visualize the data (make sure the input parameters in this python script match what's used in `submit-sim1.sh`)
```sh
python vis-data.py
```


### Simulation 2

First, train the models
```sh
./submit-sim2.sh
```
Visualize the data (make sure the input parameters in this python script match what's used in `submit-sim2.sh`)

```sh
python vis-data.py
```


### Simulation 3 

First, train the models for in the high event similarity environment: 
```sh
./submit-sim3-1.sh
```
Visualize the data (make sure the input parameters in this python script match what's used in `submit-sim3-1.sh`)

```sh
python vis-data.py
```

Then train the models for in the low event similarity environment: 
```sh
./submit-sim3-2.sh
```
Visualize the data (make sure the input parameters in this python script match what's used in `submit-sim3-2.sh`)

```sh
python vis-data.py
```

### Simulation 4

First, train the models whlie providing the familiarity signal
```sh
./submit-sim4.sh
```
Visualize the data (make sure the input parameters in this python script match what's used in `submit-sim4.sh`)
```sh
python vis-data.py
```

Compare these results to what you got from simulation 2 to see the effect of having the familiarity signal. 

To reverse the familiarity signal, open `eval-group.py`, change the variable `attach_cond` to `-1`. Then run 
```sh
python eval-group.py
```

and visualize the data again. 


### Simulation 5 

This simulation re-use the models trained in simulation 2. First, re-evaluate the model by setting the `enc_size` to `8`, which will let the model to encode episodic memories midway through an event sequence. 
```sh
python eval-group.py
```

Then you can visualize the data: 
```sh
python vis-data.py
```

### Simulation 6

This simulation re-use the models trained in simulation 2. First, you need to re-evaluate the model on RM, DM, NM condition separately. The simulation parameters in `eval-group.py` is configured to do this, so simply run
```sh
python eval-group.py
```

Then run the following code the run the inter-subject analysis: 
```sh
python vis-isc.py
```


### Simulation 7 

This simulation re-use the models trained in simulation 2. First, you need re-evaluate the model on temporally scrambled stimuli while having their hippocampus turned off, The simulation parameters in `eval-group.py` is configured to do this, so simply run 
```sh
python eval-group.py
```
Then run the following code to run the inter-subject analysis: 
```sh
python vis-zuo-scramble.py
```

### Simulation 8

In this simulation, we need to train the models in simulation 2 further on a new task (make sure the input parameters in this python script match what's used in `submit-sim2.sh`)

```sh
./submit-sim8.sh
```

Then visualize the data (make sure the input parameters in this python script match what's used in `submit-sim2.sh`) 
```sh
python vis-aba.py
```

This script performs MVPA, you need to evaluate the model on a larger dataset with `eval-group-aba.py` and then run `mvpa-aba.py` to plot the result
```sh
python eval-group-aba.py
python mvpa-aba.py
```


### Simulation 9 

For this simulation, train the model and visualize the data
```sh
./submit-sim9.sh
```

Visualize the data (make sure the input parameters in this python script match what's used in `submit-sim9.sh`)
```sh
python vis-data.py
```

To perform MVPA and visualize the data 
```sh
python mvpa-run.py
python mvpa-plot.py
```


### Contact

Qihong Lu (qlu@princeton.edu)
