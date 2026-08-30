# Learning an episodic encoding policy on the original prediction task

## Research question and falsifiable hypothesis

Lu, Hasson, and Norman (2022) showed that forcing a model to encode episodic
memories only at event endpoints produced better later prediction than forcing it
to encode both midway through and at event endpoints. The model did not discover
either schedule. The present experiment asks whether optimization can discover the
endpoint-selective policy on the same event-prediction task.

The hypothesis is conditional and architecture-specific. First, with the new
differentiable retrieval system fixed, forced endpoint encoding must outperform
never encoding, midpoint-plus-endpoint encoding, and dense encoding during `b2` of
the distant-memory (DM) condition. This establishes that the task and model jointly
create an incentive for selectivity. Conditional on that precondition, delayed
prediction reward should train a boundary-blind Bernoulli encoding gate to assign a
higher encoding probability to the endpoint of `b1` than to its other time points.

Success requires all of the following across at least 10 retained model seeds:

1. the confidence interval for endpoint-minus-nonendpoint encoding probability is
   above zero;
2. the learned policy improves held-out DM `b2` prediction over never encoding and
   recovers at least 80% of the forced-random-to-forced-endpoint reward gap;
3. disabling episodic retrieval removes the learned policy's DM benefit; and
4. the weights remain frozen while the model forms and uses new episodic memories
   for unseen 16-feature situation mappings.

These criteria test emergence, functional benefit, and causal dependence. A single
attractive encoding-probability curve is insufficient.

## Exact task

The task is defined in `source_audit.md` and `data_contract.md`. Each event contains
16 situation features, each with one of four values. At each time point the model
observes one randomly selected feature and receives a fixed-order query about the
next state. The 37-dimensional input contains a one-hot observed-feature identity,
one-hot observed value, one-hot queried-feature identity, and the current error
penalty. The output has four specific next-state responses and one “don't know”
response. A correct response earns +1, an incorrect response earns the negative
penalty, and “don't know” earns 0.

A trial contains a distractor event `a1`, a first target event `b1`, and a second
target event `b2`. In recent memory (RM), `b1` and `b2` share a situation and working
memory is retained. In DM they share a situation but working memory is reset before
`b2`. In no memory (NM), the three events use unrelated situations and working
memory is reset. In RM and DM, episodic memory may contain both a lure from `a1` and
a target from `b1`. The benefit of episodic memory is therefore most identifiable in
early DM `b2`, where target information is absent from working memory but available
in an episodic trace.

The primary `released_code` profile reproduces the executable generator, including
its 0–4 delay and historical random-count missing-observation procedure. The
`paper_text` profile uses the published 0–3 delay and independent 0.30 removal. The
primary claim will be based on `released_code`; `paper_text` is a non-pooled
sensitivity analysis. Meta-test uses delay zero, no observation removal, penalty 2,
and new situations, matching the published evaluation.

The encoding policy receives no added boundary label, completion mask, time counter,
or condition label. It acts from the recurrent situation model generated from the
original inputs. Because query order is fixed and event length is constrained, the
original task itself contains temporal information from which the endpoint can be
inferred. Success therefore means learning when to encode within a segmented event;
it does not mean discovering event boundaries in an unsegmented stream.

## Model mechanism

The new model retains the three functional components needed by the hypothesis:

- A recurrent controller integrates observations into a situation model and
  predicts the next state.
- Episodic encoding copies the current recurrent state into a finite memory.
- Differentiable episodic retrieval computes cosine similarities between the current
  state and stored traces, converts them to competitive attention weights, and adds
  their weighted combination back into the recurrent state before prediction.

The implementation may use a smaller recurrent layer than the original 194-unit
LSTM because the user specified task rather than architectural identity. Network
width, attention temperature, and capacity are configuration values, not hidden
assumptions. The primary learned encoding gate is a Bernoulli actor operating at
each eligible time point of `a1` and `b1`. Its binary action creates or skips a
distinct episodic snapshot. The action is optimized with actor-critic gradients from
later task reward. Retrieval remains differentiable; the binary encoding decision is
not misdescribed as ordinary backpropagation through a fractional memory.

The primary capacity is large enough to retain every possible trace from the two
remembered events, preventing selective encoding from winning merely through reduced
eviction. With delay up to four, that ceiling is 40 traces. Forced-policy evaluation
also reproduces the historical capacity convention of two endpoint traces versus
four midpoint-plus-endpoint traces. A fixed two-slot memory is a sensitivity
analysis, not the primary result.

## Staged optimization

### Stage 0: environment and response oracle

Deterministic tests verify all task invariants: input dimensions, fixed query order,
independent observation permutations, shared versus unrelated situations, delay
placement, missing observations, reward values, and RM/DM/NM memory availability. A
response oracle that sees the queried situation value must achieve perfect valid-step
accuracy, and a random response must approach 0.25.

### Stage 1: recurrent prediction without episodic memory

The controller is trained on RM sequences with episodic memory disabled. This stage
establishes that the model can integrate observations, represent the situation, and
solve ordinary within-event prediction. It cannot teach an encoding policy because
no encoding action or episodic retrieval is present.

### Stage 2: supervised differentiable retrieval

Endpoint encoding is forced while the retrieval and prediction pathway is trained on
RM, DM, and NM. The required check is that episodic retrieval improves early DM `b2`
prediction relative to retrieval-off without creating a false benefit in NM. A
memory-identity diagnostic must show more target than lure contribution. If this
stage fails, encoding-policy learning is not attempted because the policy would have
no useful downstream mechanism to control.

### Stage 3: forced-schedule validation and value learning

With the predictor and retriever fixed, endpoint-only, midpoint-only,
midpoint-plus-endpoint, dense, never, and matched-random encoding schedules are
evaluated on paired trials. Endpoint-only must be optimal or tied-optimal and must
beat never and dense encoding in DM. The value estimator is then trained under
forced exploration across encoding and skipping actions while the encoding actor
remains fixed. This stage gives the critic experience with delayed consequences
without allowing a premature actor collapse.

### Stage 4: free encoding selection

The forced schedule is removed. The Bernoulli actor chooses freely at every eligible
time point and receives only later prediction reward. Controller and retrieval
weights remain frozen in the primary gate-only condition, isolating encoding-policy
learning. An optional joint fine-tuning condition is reported separately. If the
gate-only model succeeds but joint training fails, the result diagnoses an
encoding–retrieval coordination problem rather than robust end-to-end discovery.

Straight-through or soft encoding is allowed only as an optimization diagnostic.
Fractional traces do not establish selective episodic snapshots and cannot satisfy
the primary success criterion.

## Baselines, interventions, and ablations

Required forced baselines are endpoint-only, midpoint-only,
midpoint-plus-endpoint, dense, never, and matched-random encoding. A supervised
endpoint classifier is a realizability ceiling and is never evidence that delayed
reward taught the policy.

Required causal interventions are retrieval-off, shuffled memory identities,
target-memory removal, and moving a learned endpoint action to a midpoint while
matching encoding count. Retrieval temperature, memory capacity, and the
`paper_text` generator are sensitivity analyses. An oracle that retrieves the newest
or correct target trace tests whether encoding selectivity is only compensating for
competitive retrieval. If dense encoding becomes optimal under that oracle, the
report must state that selectivity depends on retrieval limitations.

The faithful task contains no encoding cost. Limited-capacity and encoding-cost
variants may be run only after the faithful result is frozen and must be labeled
secondary. They cannot turn a negative exact-task result into a positive primary
claim.

## Outcomes and analysis

Prediction outcomes are valid-step accuracy, task reward, error rate, “don't know”
rate, and early-`b2` DM accuracy. Encoding outcomes are endpoint and nonendpoint
probabilities, their paired difference, number of encodings per event, and the full
time course. Retrieval outcomes are target and lure attention, retrieved-state norm,
and the performance loss when retrieval is disabled.

The model seed is the inferential unit. All seeds are shown. Baselines share trial
banks within seed. Estimates include mean, standard deviation, paired seed values,
and fixed-seed bootstrap confidence intervals. Configuration selection uses
exploratory validation seeds only. The confirmatory configuration and test seed
namespace are committed before test execution.

## Failure diagnoses and limitations

- If the task oracle fails, the generator is wrong.
- If recurrent prediction fails, the controller has not learned the task.
- If forced endpoint retrieval fails, the episodic pathway is not usable.
- If dense encoding matches endpoint encoding, the architecture supplies no reward
  for selectivity; learned dense encoding is then rational rather than a policy
  failure.
- If the supervised endpoint ceiling succeeds but free reinforcement learning fails,
  delayed credit assignment is implicated.
- If only a boundary label, explicit time counter, encoding cost, or limited capacity
  yields selectivity, the exact-task hypothesis is unsupported.

Even a positive result would establish computational feasibility only. The task has
fixed event segmentation and synthetic states; recurrent activity is an analogue of
a human situation model, and model retrieval is not equivalent to hippocampal
reinstatement. The result would show that one stated architecture can optimize an
encoding policy under one task distribution, not that human episodic encoding is
globally optimal or learned by the same algorithm.

## Reproducibility and compute budget

All reported runs use YAML configurations, deterministic seed namespaces,
configuration hashes, seed-level JSON, and frozen evaluation. Checkpoints and large
arrays remain outside Git; compact metrics, configurations, tests, decisions, and
figures are versioned.

| Work | Initial budget |
|---|---:|
| Task and model behavior tests | less than 2 CPU minutes |
| One-seed stage smoke run | up to 30 CPU minutes |
| Exploratory optimization, 3 seeds | up to 3 CPU hours |
| Frozen evaluation, at least 10 seeds | up to 10 CPU hours |
| Required baselines and causal ablations | up to 4 additional CPU hours |

The staged design permits early stopping for diagnosis: later stages are not run if
their stated precondition fails. Runtime and hardware are recorded from measured
executions rather than inferred after the fact.
