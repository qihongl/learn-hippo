# Follow-up design: learning an event-boundary encoding policy

## 1. Research question and status

The follow-up asks whether optimization can teach a shared online encoding policy to
store episodic memories selectively at event endpoints on the original
event-prediction task. The first 10-seed diagnostic established two facts: forced
endpoint encoding is useful, and the current sparse actor uses episodic memory but
does not become endpoint-selective. That run received only 9,600 free-policy event
sequences and 600 batch-level parameter updates. The eLife model's reinforcement-
learning phase received 102,400 event sequences and updated parameters after every
sequence. The earlier negative result therefore identifies a learning problem but
does not constitute a training-budget-matched failure.

Architecture development is exploratory. After the model class, configuration,
training duration, and analysis are fixed, a new seed namespace will provide a
confirmatory replication. The confirmatory result will be reported even if it
contradicts development behavior.

## 2. Hypothesis and alternative outcomes

The primary hypothesis is that a policy which receives only the accumulated current
situation can learn to delay encoding until that situation is complete, because a
complete trace improves later DM prediction and creates less retrieval competition
than incomplete traces. The predicted pattern is endpoint encoding probability at
least 0.80, endpoint-minus-nonendpoint probability at least 0.50, and a causal DM
reward advantage over never and matched-random-one encoding.

Three alternative outcomes have distinct interpretations. If even a directly
parameterized temporal policy fails under exact counterfactual reward, the
optimization objective or implementation is defective. If exact counterfactual
optimization succeeds but sampled actor-critic learning fails, high-variance delayed
credit assignment is the limiting factor. If an actor using the exact situation
record succeeds but an actor using a recurrent state fails, neural situation
learning—not the reward objective—is the limiting factor. A stable nonboundary
solution after the prespecified budget is evidence against reliable discovery under
that method; it is not evidence that boundary encoding is never learnable.

## 3. Synthetic task and partitions

The primary task is the tested `released_code` generator: 16 features, four feature
values, the original 37-dimensional input, random observation order, fixed query
order, the `a1`--`b1`--`b2` sequence, and the original RM/DM/NM manipulations. The
training mixture is 0.25 RM, 0.25 DM, and 0.50 NM. Mappings, observation orders,
delays, and removal patterns are resampled. The `paper_text` generator is a separate
sensitivity. No human data are used.

Initial optimization audits may train on DM alone, where episodic information is
most identifiable. A positive curriculum result is not the final result. The final
basic model must train from random initialization on the full mixture without a
future-target, condition, boundary, countdown, or completion label.

Exploratory model seeds, fixed development trial banks, and untouched confirmatory
model and trial seeds use nonoverlapping namespaces. Within seed, all baselines and
ablations share trials. Weights are frozen at every held-out evaluation.

## 4. Models and staged protocol

### Stage 0: measurement and convergence infrastructure

The runner will retain every training update and evaluate fixed development trials
every 10 epochs. Checkpoint evaluation must be deterministic and must not alter
weights, optimizer state, or random-number streams used for subsequent training.
Training and held-out curves will include reward, endpoint selectivity, encoding
count, actor and critic losses, entropy, learning rate, and gradient norm. This stage
also reconstructs all curves supported by the original 10-seed records.

### Stage 1: temporal-hazard optimization audit

Sixteen online encoding hazards parameterize the probability of encoding at each
successive observation, conditional on not having encoded earlier. These parameters
are shared across `a1` and `b1`. Exact counterfactual enumeration measures reward for
possible encoding-time pairs and optimizes their probability-weighted expectation.
There is no supervised endpoint target. Training begins with DM and one trace per
event. This is an optimizer and objective audit rather than the model-level claim.

### Stage 2: neural actor with exact situation state

A small multilayer perceptron replaces the temporal parameters. It receives only the
80-dimensional exact situation state already used by the diagnostic model. Time,
completion, and event identity are not appended. The actor is first optimized using
the counterfactual objective, then from a fresh initialization with sampled
actor-critic learning. Success of the counterfactual model establishes basic
computational feasibility; success of the sampled model establishes the stronger
delayed-reward result.

### Stage 3: full mixture and historical capacity

The neural actor progresses from DM-only and one trace per event to the full mixture
and two total trace slots for the entire trial. The latter prevents the environment
from reserving a slot at an event transition. Models trained with a DM curriculum
and models trained from scratch on the mixture are separate. Only from-scratch
success supports the primary basic claim. Four- and 40-slot capacities diagnose
whether the learned policy depends on a storage bottleneck.

### Stage 4: recurrent situation model

After exact-state success, a GRU or LSTM receives the original 37-dimensional input.
It is pretrained for 600 epochs of 256 sequences, matching the paper's supervised
exposure. The initial objective decodes accumulated observed feature values and
predicts queried values without accessing future values. The recurrent model is
frozen after it meets its held-out representation precondition and after forced
endpoint encoding remains better than the declared alternatives. The encoding actor
then trains for 400 epochs of 256 sequences. A next-state-only pretraining condition
tests whether explicit reconstruction supervision was necessary.

### Stage 5: locked confirmation

The successful architecture and training protocol are committed before confirmation.
Twenty untouched model seeds are trained for the fixed duration and evaluated on 512
new trials per condition. There is no seed filtering or confirmatory retuning.

## 5. Training budget and convergence

One epoch contains 256 newly generated sequences. The primary policy budget is 400
epochs, 102,400 sequences, batch size 16, and 6,400 parameter updates. Development
evaluation occurs every 10 epochs. Exploratory early success requires every primary
criterion for five consecutive checkpoints. Confirmation always reaches epoch 400.

An exploratory run may extend once to 800 epochs only when the endpoint gap improves
by at least 0.05 or held-out reward improves by at least 0.005 from epochs 201--300
to epochs 301--400. Otherwise the run is classified as plateaued or moving toward a
wrong solution. A batch-size-one, 102,400-update sensitivity is allowed for three
seeds only if event-matched training shows a continuing favorable trend; it is not a
general rescue search. The final checkpoint, not the most favorable checkpoint, is
the primary estimate.

## 6. Outcomes and success criteria

The model seed is the inferential unit. The basic exact-state model must satisfy all
of these requirements across at least 10 seeds:

1. mean endpoint encoding probability is at least 0.80;
2. mean endpoint-minus-nonendpoint probability is at least 0.50 and its 95% bootstrap
   interval is above zero;
3. at least 80% of seeds have a positive endpoint gap;
4. DM reward exceeds both never and matched-random-one encoding with intervals above
   zero;
5. retrieval-off removes at least 80% of the learned DM benefit;
6. target-memory removal eliminates the benefit, whereas distractor removal does not;
7. endpoint preference is present separately in `a1` and `b1`; and
8. all effects use unseen mappings with frozen weights.

The recurrent-state extension uses the same policy criteria and must first decode at
least 99% of already observed feature values on held-out sequences. Accuracy on
unobserved values is not counted. Results below a threshold, mixed positive and
negative criteria, or intervals spanning the required effect are ambiguous rather
than successful.

## 7. Baselines and ablations

Forced endpoint, midpoint, midpoint-plus-endpoint, dense, never, and matched-random-
one schedules remain mandatory. A supervised endpoint actor is a realizability
ceiling only. Causal ablations disable retrieval, remove target traces, remove
distractor traces, and disable the content-match gate. Mechanistic comparisons test
counterfactual versus sampled credit, one-per-event versus two-total capacity,
DM curriculum versus full-mixture-from-scratch training, exact versus recurrent
situation states, and 600/1,600/3,200/6,400-update budgets. All comparisons receive
paired trial banks and retain individual seeds.

## 8. Risks, limitations, and stopping diagnoses

Counterfactual enumeration uses privileged access to simulated outcomes during
training and is not proposed as a biological learning mechanism. It is valuable
because it distinguishes the reward surface from sampling variance. One-per-event
capacity also presupposes segmentation and is a debugging condition. The two-total-
trace model is required before making the basic claim. The content gate was selected
exploratorily in the first diagnostic, so a later confirmation must freeze it or
learn it on a disjoint retrieval objective.

Future relevance remains naturally unobservable and is not treated as a task flaw.
The policy must learn a rule that is useful on average without a prospective target
label. Even a successful synthetic model demonstrates feasibility rather than the
human mechanism; model retrieval is an analogue of reinstatement, not human gaze or
hippocampal activity.

## 9. Compute, provenance, and reporting

The first 100 updates of every new implementation provide a measured throughput
benchmark. Exploratory temporal and exact-state work is capped at 24 CPU-hours;
full-mixture development at another 24 CPU-hours; recurrent development at 72
CPU-hours; and locked confirmation at 96 CPU-hours. A projected overrun pauses the
stage for user review instead of silently reducing seeds or trials.

Every stage has a versioned YAML configuration, configuration hash, complete
per-seed/checkpoint JSON, progress and decision-log entry, deterministic tests, and
its own commit and push. Generated checkpoint tensors remain outside Git. The final
paper-style report will distinguish explanatory diagrams from measured synthetic
results and will show complete learning curves, seed distributions, failures,
deviations, and limitations.
