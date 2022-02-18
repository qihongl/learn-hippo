## About this repo

This repo hosts the python code for the following paper: 

[Lu, Q., Hasson, U., & Norman, K. A. (2022). A neural network model of when to retrieve and encode episodic memories. eLife](https://elifesciences.org/articles/74445)

If you have access to a cluster, most simulations can be replicated in a day. 

Feel free to contact me if you have any questions / comments. Thank you very much in advance! 

## Playing with some pre-trained models

If you would like to play with the model, here's a [Code Ocean capsule](https://codeocean.com/capsule/3639589/tree) we created, which contains the code in this repo and some pretrained weights.  

You can qualitatively replicate most results and see the plots in 5 minutes. Once you are in the Code Ocean capsule, simply click **reproducible run**. To specify which simulation you would like to run, go to `code/src/demo.py`, on line 33, change `simulation_id` to 1 or 2, and then click **reproducible run** again. You can also modify the level penalty at test (e.g. line 39 - 41). Once you are done, you can check the figures listed in the timeline (right hand side, under reproducible run). 

## Dependencies 

This is a python-based project. The list of dependencies and their version information is provided [here](https://github.com/qihongl/learn-hippo/blob/master/requirement.txt). 
The code should work as long as the versions of your packages are close to what I used. 

I used a cluster to parallelize model training, so that most simulations took less than a day. The cluster I used at Princeton uses [Slurm](https://slurm.schedmd.com/documentation.html). So depends on where you are, the job submission files 
(e.g. 
[src/submit-vary-test-penalty.sh](https://github.com/qihongl/learn-hippo/blob/master/src/submit-vary-test-penalty.sh)
[src/train-model.sh](https://github.com/qihongl/learn-hippo/blob/master/src/train-model.sh)
)
I wrote might not work for you. 
However, these scripts simply document the simulation parameters I used and it is relatively simple to adapt them for your cluster/platform (see the section on **Replicate the simulation results - general guidelines**). 

## What's in the repo

Here's the structure of this repo

```sh
├── demo-log    # some data for the demo
├── log         # the logging directory 
├── figs        # figure directory, most scripts save figs to here 
└── src         # source code
    ├── analysis        # helper functions for analyzing the data 
    │   ├── __init__.py
    │   ├── behav.py
    │   ├── general.py
    │   ├── neural.py
    │   ├── preprocessing.py
    │   ├── task.py
    │   └── utils.py
    ├── examples        # some simple demos 
    │   ├── event-empirical-similarity.py
    │   ├── event-similarity-cap.py
    │   ├── event-similarity.py
    │   ├── memory_benefit.py
    │   ├── schema-regularity.py
    │   ├── stimuli-representation.py
    │   ├── true-uncertainty.py
    │   └── zuo-2019-control-patient-isc-logic.py
    ├── models         # the components of the model 
    │   ├── A2C.py                          # the standard actor-critic algorithm 
    │   ├── EM.py                           # the episodic memory module
    │   ├── LCALSTM.py                      # the model 
    │   ├── LCALSTM-after.py                # the postgating model     
    │   ├── LCA_pytorch.py                  # a pytorch implementation of leaky competing accumulator model
    │   ├── _rl_helpers.py                  # some RL helpers 
    │   ├── initializer.py                  # weight initializer
    │   ├── metrics.py                      # some helper functions for metrics 
    │   ├── __init__.py    
    ├── task            # the definition of the task 
    │   ├── Schema.py           
    │   ├── StimSampler.py
    │   ├── SequenceLearning.py    
    │   ├── utils.py
    │   └── __init__.py
    ├── utils           # general utility functions
    │   ├── constants.py
    │   ├── io.py
    │   ├── params.py
    │   ├── utils.py
    │   └── __init__.py    
    ├── vis             # code for visualizing the data
    │   ├── _utils.py
    │   ├── _vis.py
    │   └── __init__.py    
    ├── submit-vary-train-penalty.sh            # train models with fixed penalty level, triggers train-model.sh
    ├── submit-vary-test-penalty.sh             # train models with varying penalty level, triggers train-model.sh
    ├── submit-vary-test-penalty-postgate.sh    # train postgating models with varying penalty level, triggers train-model-after.sh
    ├── submit-vary-test-penalty-fixobs.sh      # train models with varying penalty level and fix the observation order, triggers train-model.sh
    ├── submit-familiarity.sh                   # train models with familiarity signal, triggers train-model.sh
    ├── submit-similarity-high.sh               # train models in a high similarity env, part of similarity x penalty experiment, triggers train-model.sh
    ├── submit-similarity-low.sh                # train models in a low similarity env, part of similarity x penalty experiment, triggers train-model.sh
    ├── submit-vary-schema-level.sh             # train models with varying schema level, triggers train-model.sh
    ├── submit-vary-schema-level-postgate.sh    # train postgating models with varying schema level, triggers train-model-after.sh
    ├── train-model.sh                          # submit a python job to train a model
    ├── train-model-after.sh                    # submit a python job to train a postgating model    
    ├── demo.py                                 # a demo for Code Ocean
    ├── eval-group.py                           # evaluate a group of models 
    ├── exp_tz.py                               # definition of the Twilight Zone experiment by Chen et al. 2016
    ├── train-sl.py                             # train the model on the event prediction task 
    ├── train-sl-after.py                       # train the postgating model on the event prediction task 
    ├── vis-compare-encpol.py                   # compute the performance of different encoding policies 
    ├── vis-cosine-sim.py                       # visualize the cosine similarity between cell state and memories over time 
    ├── vis-data.py                             # visualize the basic results  
    ├── vis-data-after.py                       # visualize the basic results for the postgating model
    ├── vis-inpt-by-schematicity.py             # visualize the effect of schema level on input gate values (see simulation 9)
    ├── vis-isc.py                              # visualize the ISC analysis
    ├── vis-policy-adjustment.py                # visualize the how the model adjusts its policy according to the penalty level 
    ├── vis-policy-diff.py                      # visualize the how the model models trained in different penalty levels respond differently
    └── vis-policy-adj-similarity.py    
    
```    

## Replicate the simulation results - general guidelines

Here we introduce the general procedure for how to replicate any simulation in the paper. We will use simulation 2 as an example. 

### 0. Download the code
First, you need to clone this repo: 
```sh
git clone https://https://github.com/qihongl/learn-hippo
```
### 1. Model training 
I provided job submission files for all simulations: `src/submit-sim*.sh`. 
For example, [src/submit-vary-test-penalty.sh](https://github.com/qihongl/learn-hippo/blob/master/src/submit-vary-test-penalty.sh) is the job submission file to train models with varying penalty level at test. Executing this file will train 15 models in parallel with the specified simulation parameters. Note that the job of `submit-*.sh` is to specify simulation parameters and trigger 
[train-model.sh](https://github.com/qihongl/learn-hippo/blob/master/src/train-model.sh). 
Then `train-model.sh` takes those simulation parameters and runs a python program that trains the model, which is general across simulations. 

Several things to check before you run this script. 

1. In `train-model.sh` ([line 11](https://github.com/qihongl/learn-hippo/blob/master/src/train-model.sh#L11)), you need to set the `DATADIR`, the logging directory, to something that exisits on your machine. Then the data (e.g. trained network weights) will be saved to this directory. Later on, other programs, such as the code for visualizing data, will need to access this directory to find the weights for the trained models. 

2. In `train-model.sh` ([line 7](https://github.com/qihongl/learn-hippo/blob/master/src/train-model.sh#L7)), you need to set where to log the output (from the python script) to a directory that exists on your machine. This is useful if you want to inspect the training process. 

To train models for simulation 2, simply go to the `src/` folder and type the following: 

```sh
./submit-vary-test-penalty.sh
```

This will trigger `train-model.sh` and it will submit a python job with the following command with the specified simulation parameters: 

```sh
srun python -u train-sl.py --exp_name ${1} --subj_id ${2} --penalty ${3}  \
    --n_epoch ${4} --sup_epoch ${5} --similarity_max ${6} --similarity_min ${7} \
    --penalty_random ${8} --def_prob ${9} --n_def_tps ${10} --cmpt ${11} --attach_cond ${12} \
    --enc_size ${13} --dict_len ${14} --noRL ${15} \
    --log_root $DATADIR
```

The code above clarifies how to train a model on any platform with any parameter configuration. Suppose you want to train the model with some parameter configuration `exp_name = {1}`, `subj_id = {2}`, `penalty = {4}`... `noRL = {15}`, simply run `python train-sl.py --exp_name ${1} --subj_id ${2} --penalty ${4} ... --noRL = {15}`. Though this works on your laptop too, it is gonna be tedious and error prone. That why I wrote the `submit-*.sh` to do this systematically. 

Here's a brief summary of what these parameters mean: 

`exp_name` - the name of the experiment, only affects the directory name where the data will be saved

`subj_ids` - the id of the subject, only affects the directory name where the data will be saved

`penalty` - if penalty_random is 0 (false), then this is the penalty of making a prediction mistake; else if penalty_random is 1, then the penalty value will be sampled uniformly from 0 up to the input value. 

`n_epoch` - the total number of training epochs 

`sup_epoch` - the number of supervised pre-training epochs 

`p_rm_ob_enc` - the probability of withholding observation before part 2

`p_rm_ob_rcl` - the probability of withholding observation during part 2

`similarity_max` - the maximum event similarity in a single trial of experiment

`similarity_min` - the minimum event similarity in a single trial of experiment

`penalty_random` - see description for penalty 

`def_prob` - the probability that the prototypical event happens

`n_def_tps` - the number of time points with a prototypical event

`cmpt` - the level of competition 

`attach_cond` - if 1 (true), attach the familiarity signal to the input; if (0) false, doesn't affect the input at all

`enc_size` - defines the frequency of episodic encoding. e.g. if it is the same as `n_param`, then the model selectively encodes at event boundaries. if it is `n_param/2` then it also encodes midway

`dict_len` - the size of the episodic memory buffer. since the buffer is a queue, when there is an overflow, the earliest memory will be removed


### 2. Model evaluation 

`eval-group.py` evaluates some pre-trained models on some tasks with the specified simulation parameters. Actually, the training script evaluates the model on some test sets by default, but in some simulations, I test the pre-trained models on some new tasks that the model hasn't been trained on. `eval-group.py` is a generic evaluation script that allows me to do that. 

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

Note that this script will use the input simulation parameters to locate the data and then make plots. So input simulation parameters here must match the simulation parameters used for model training (step 1), otherwise the script won't be able to locate the data. 

## Specific instructions for each simulation

This section lists the scripts you need to replicate every simulation in the paper. Note that when you use the python scripts to visualize the data (e.g. `vis-*.py`), the input parameters must match the parameters used in the training scripts. This enables the python script to find the location of the saved data. 

### Simulation 1 - recall policy 

First, train the models 
```sh
./submit-vary-test-penalty.sh
```
Visualize the data (make sure the input parameters in this python script match what's used in `submit-vary-test-penalty.sh`)
```sh
python vis-data.py
```

#### Simulation 1 - variants

1. to train the model on a fixed the penalty level, which means the level of penalty is fixed to be the same for both training and testing (instead of varying the penalty at test), train the models with ... 
```sh
./submit-vary-train-penalty.sh
```
And then visualize the data with `vis-data.py`

2. to train the postgaing model with varying test penalty, train the models with ... 
```sh
./submit-vary-train-penalty.sh
```
And then visualize the data with `vis-data.py`

3. to explore the interaction between the level of event similarity and penalty level, train the model with 
```sh
./submit-similarity-low.sh
./submit-similarity-high.sh
```
And then visualize the data with `vis-data.py`

4. to explore effect of the familiarity signal, train the model with 
```sh
./submit-familiarity.sh
```
Then visualize the data with `vis-data.py`. 

To reverse the familiarity signal, open `eval-group.py`, change the variable `attach_cond` to `-1`. Then run 
```sh
python eval-group.py
```
and visualize the data again. 



5. To vary the schema level, run 
```sh
./submit-vary-schema-level.sh
```

Visualize the data: 
```sh
python vis-data.py
```


6. To perform the intersubject correlation analysis (appendix). First, you need to re-evaluate the model on RM, DM, NM condition separately. The simulation parameters in `eval-group.py` are configured to do this, so simply run
```sh
python eval-group.py
```

Then run the following code to run the inter-subject analysis: 
```sh
python vis-isc.py
```


### Simulation 2 - selective encoding at event boundaries 

This simulation re-uses the models trained in simulation 1. First, re-evaluate the model by setting the `enc_size` to `8` and run ... 
```sh
python eval-group.py
```
... which will let the model to encode episodic memories midway through an event sequence. 

Then you can visualize the data: 
```sh
python vis-data.py
```
This gives you Figure 4B-E. And after that, you can run `vis-compare-encpol.py` to get Figure 4A. 



### Contact

Qihong Lu (qlu@princeton.edu)


