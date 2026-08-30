# Learning to Encode at Event Boundaries

## Research question and claims

The 2022 `learn-hippo` simulations showed that an endpoint-only episodic encoding
schedule could outperform a schedule that also stored a midway representation.
The schedules were imposed by the experimenter. This experiment asks the stronger
question: can optimization discover selective boundary encoding when a model is
rewarded only for accurate delayed prediction and is not given an explicit boundary
label?

The primary hypothesis is that a stochastic write policy coupled to differentiable
episodic retrieval will learn a higher probability of writing complete endpoint
states than incomplete states. The policy should close at least 80% of the held-out
reward gap between a matched random-write baseline and a hand-coded endpoint oracle.
Endpoint selectivity must be positive in at least 12 of 15 independent training
seeds, with a paired 95% confidence interval above zero. Moving the learned endpoint
writes to matched mid-event positions must reduce performance. These criteria test
policy emergence, functional value, and causal specificity rather than relying on
one favorable learning curve.

This is a computational-feasibility study on a deliberately simplified synthetic
task. It does not test whether human event boundaries are discovered, whether the
policy is biologically implemented, or whether neural attention is human memory
retrieval. A negative result will be preserved and diagnosed.

## Task and theoretical precondition

Each synthetic event has a random event cue and four independently sampled binary
features. During study, the features are revealed one at a time in randomized order.
The currently accumulated event state contains observed feature values and an
observation mask. The write policy may inspect that ordinary state but receives no
`is_boundary`, remaining-time, or endpoint variable. Null intervening steps vary
the physical duration of an event without changing its semantic completion rule.

After study, transient controller state is reset. At delayed test, the model sees
the event cue and the first two features from the original reveal sequence. It
queries episodic memory and predicts the remaining features. Memory keys combine
the event cue with the partial event state; memory values contain the accumulated
feature representation. Consequently, a stored snapshot matching the partial test
cue can competitively block a less similar but more complete endpoint snapshot.
When only the endpoint is stored, it is the best available target and supplies the
missing independent features. This mirrors the proposed mechanism in the original
paper while removing unrelated event-graph complexity.

Before policy learning, every deterministic write schedule will be evaluated. With
four informative positions there are only 16 semantic schedules, so enumeration is
exact. The task passes the theoretical precondition only if endpoint-only encoding
is optimal or tied-optimal and outperforms always-write and matched-rate random
encoding. Policy training will not begin until deterministic tests verify this
property. If the precondition fails, task parameters may be changed using oracle
results alone; each change must be committed and explained before any learned-policy
result is observed.

## Model

The episodic memory is a differentiable key-value store. Retrieval computes cosine
similarities between a test query and stored keys, applies a temperature-controlled
softmax, and returns the attention-weighted sum of stored values. The complete
attention distribution is recorded. A zero-memory read returns a declared zero
vector. The primary fixed-capacity condition allocates enough slots for every
possible study write, eliminating eviction as an explanation for selective encoding.

The write policy is a small shared multilayer perceptron with a Bernoulli output at
each study opportunity. It sees the partial feature state, observation mask, and
event cue, but no boundary label or absolute remaining time. Its sampled binary
action either appends a distinct episodic trace or skips the write. Delayed test
reward is propagated to study actions with an actor-critic estimator. Retrieval and
prediction remain differentiable; the discrete storage decision is optimized with
policy gradients so that episodic traces remain distinct rather than fractional
mixtures.

A straight-through differentiable write relaxation is a secondary diagnostic, not
the primary evidential model. A soft model can store fractions at every step and may
therefore solve a different problem. A supervised endpoint classifier is used only
as a realizability ceiling and never as evidence that delayed reward taught the
policy.

## Staged protocol

1. **Environment and oracle validation.** Test feature independence, cue reuse,
   state reset, absence of a boundary input, memory capacity, and exact schedule
   enumeration. Compare endpoint, always, never, midpoint, and matched-random
   schedules.
2. **Differentiable retrieval.** Under forced endpoint writing, require held-out
   reconstruction performance substantially above the no-memory baseline and high
   attention on the correct event. Then expose retrieval to heterogeneous schedules
   to rule out a schedule-distribution artifact.
3. **Write-policy realizability.** Train a supervised diagnostic gate to verify that
   the allowed state supports endpoint discrimination. Its labels are unavailable to
   all reinforcement-learning runs.
4. **Gate-only delayed learning.** Freeze or use the validated retrieval system and
   train the Bernoulli write policy with delayed test reward. No write cost is used
   in the primary condition, so selectivity must arise from interference.
5. **Joint and optimization diagnostics.** Compare gate-only training, alternating
   retrieval/gate updates, joint end-to-end training, straight-through writes, and
   from-scratch training. Staged-only success will be reported as evidence of a
   coordination problem rather than robust spontaneous discovery.
6. **Generalization and causal tests.** Evaluate unseen event contents, reveal
   orders, cue vectors, and longer null-step patterns. Clamp the learned gate to
   endpoint, midpoint, always, never, and matched random schedules. Displace its
   endpoint writes while matching write count.

Model weights are frozen throughout held-out evaluation. Test episodes are generated
from seed ranges disjoint from training and validation. Hyperparameter selection uses
validation data only.

## Baselines and capacity variants

Required baselines are endpoint-only, always-write, never-write, midpoint-only,
midpoint-plus-endpoint, uniform periodic, matched-rate random, supervised endpoint
ceiling, and exhaustive hindsight oracle. Mechanistic ablations remove competitive
retrieval, replace retrieval with the correct endpoint, shuffle event-memory
correspondence, or eliminate null-step length variation.

Two capacity conventions will be tested during the oracle stage. The historical
convention allocates `events remembered × snapshots per event`, so denser schedules
receive more physical slots while retaining the same number of events. The fixed
convention gives every policy the same maximum slot count. The primary learned-policy
run will use whichever convention yields the simplest stable implementation after
both oracle comparisons; the decision must be recorded before policy optimization.
Both oracle results remain reportable.

## Outcomes and analysis

The primary performance outcome is delayed feature-prediction reward, equivalently
reported as mean squared error and binary feature accuracy. Policy outcomes are
endpoint minus non-endpoint write probability, boundary-detection AUC, and writes per
event. Retrieval outcomes are correct-event attention, attention on incomplete versus
endpoint traces, and the target-completeness of the retrieved value. Robustness is
summarized by seed variance, worst-seed performance, and out-of-distribution null-step
length performance.

Fifteen independent model seeds are the unit of replication. Baselines share paired
held-out episode banks within each seed. Reports will show all seed values, means,
standard deviations, paired effect sizes, and bootstrap confidence intervals over
seeds. No seed will be filtered. The confirmatory test set is evaluated once after
configuration selection is frozen.

## Success and failure interpretation

Full success requires the oracle precondition, the prespecified 15-seed policy
criteria, performance loss under write displacement, and above-chance transfer to
held-out event realizations and longer null-step patterns. If the endpoint oracle
does not win, the task does not instantiate the theoretical claim. If the oracle and
supervised gate succeed but reinforcement learning fails, delayed credit assignment
is implicated. If gate-only training succeeds but joint training fails, encoding and
retrieval coordination is implicated. If always-write wins once retrieval is made
flexible, selective encoding is unnecessary for that architecture. If only fractional
soft writes succeed, the evidence does not establish selective episodic snapshots.

## Compute budget and reproducibility

Smoke tests use three seeds and small episode batches. Confirmatory runs use 15 seeds,
with one frozen configuration for the primary model and required baselines. The task
is intentionally short enough to run on CPU; available PyTorch acceleration may be
used without changing configurations. Runtime, device, package versions, Git commit,
and complete seed metrics will be saved. Source code, tests, configurations, compact
JSON/CSV metrics, figures, and reports are versioned. Checkpoints and large arrays are
excluded. Each scientifically coherent milestone is committed and pushed before the
next stage begins.
