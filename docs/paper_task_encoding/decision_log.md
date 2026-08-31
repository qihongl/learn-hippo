# Decision log: exact prediction task

## 2026-08-30 — Scientific target

- Only the task must match the 2022 eLife simulation; the model architecture may be
  simplified.
- No new event-boundary cue, completion mask, countdown, or condition label will be
  given to the encoding policy.
- The encoding policy is learned during meta-training and frozen during evaluation
  on new situations.
- Success requires learned endpoint selectivity, prediction benefit, and a causal
  loss of that benefit when episodic retrieval is disabled, across at least 10 model
  seeds.
- Encoding-cost and capacity-limited variants are secondary diagnostics only.

## 2026-08-30 — Source discrepancy policy

- The released-code behavior is the primary executable profile because it is the
  best available record of what the simulations ran.
- The prose-defined 0–3 delay and independent 0.30 feature removal form a separate
  `paper_text` sensitivity profile.
- The profiles will not be pooled, and differences will be reported rather than
  silently reconciled.

## 2026-08-30 — Model and optimization scope

- The model uses a recurrent situation model, distinct episodic snapshots,
  differentiable competitive retrieval, and a Bernoulli encoding actor trained from
  delayed prediction reward.
- The discrete encoding action is trained with actor-critic gradients; only episodic
  retrieval is differentiable end to end.
- Predictor and retrieval learning precede encoding-policy learning. Gate-only free
  selection is the primary test; joint training is an explicitly labeled diagnostic.
- The primary faithful condition has no encoding cost and enough capacity to retain
  all eligible traces from two events. Historical and fixed-capacity conventions are
  reported as sensitivity analyses.

## 2026-08-30 — Prediction-model precondition and simplification

- An exploratory recurrent-model run did not satisfy the preregistered precondition:
  after 1,000 prediction updates and 1,000 forced-retrieval updates, endpoint-only
  encoding reached expected reward -0.635 on 128 new DM trials, compared with -0.625
  for dense encoding and -0.682 for no encoding. These are diagnostic, not final,
  results.
- The main obstacle was prediction and retrieval quality, not evidence about a learned
  encoding policy. We therefore introduced a structured situation model that directly
  accumulates the feature-value observations already present in the original
  37-dimensional input. It receives no boundary label, completion indicator,
  countdown, or condition label.
- This simplification isolates the stated question: whether delayed prediction reward
  can train an encoding policy when the prediction and retrieval system is already
  competent. The recurrent model remains a documented negative comparison.
- Before optimizing the encoding policy, fixed endpoint encoding was verified to beat
  fixed midpoint, midpoint-plus-endpoint, and dense encoding when each schedule was
  applied identically to distracting and target events on new DM trials.

## 2026-08-30 — Shared-policy correction

- The first exploratory free-policy run incorrectly optimized encoding only in the
  target event while forcing endpoint encoding in the distracting event. This made
  dense late target memories artificially useful and is not evidence about the
  proposed shared encoding policy.
- The corrected objective applies one shared encoding network independently to every
  time point in both events. Both sets of sampled actions receive the same delayed
  prediction outcome, and both distracting and target traces enter retrieval.
- With this correction, a 1,000-update single-seed diagnostic showed partial temporal
  sensitivity but not successful selective encoding: endpoint probability was 0.864,
  mean nonendpoint probability was 0.667, the model encoded 10.90 times per event,
  and held-out expected reward was 0.394 versus approximately 0.493 for no encoding.
  This run is a failure under the preregistered prediction-benefit criterion.

## 2026-08-30 — Fully differentiable encoding diagnostic

- To separate discrete-policy gradient variance from the scientific objective, each
  memory was given a continuous encoding strength. The strength enters retrieval as
  a differentiable prior, so delayed prediction reward can update the encoding
  network directly. No task input or boundary cue was added.
- A 500-update single-seed diagnostic also failed. It assigned its largest strength
  to the first observation, with endpoint strength 0.009 and mean nonendpoint
  strength 0.033 on new DM trials. Retrieval reduced expected reward from 0.488 when
  disabled to 0.426 when enabled.
- A gradient audit explained this local solution. Complete target memories become
  more useful toward the endpoint, but complete distracting memories become almost
  equally harmful. Because the shared policy sees statistically indistinguishable
  first presentations, their boundary gradients nearly cancel.
- More importantly, under 2,048 exploratory trials from the released meta-training
  mixture, fixed no encoding achieved expected reward 0.695, compared with -0.097
  for endpoint-only and 0.011 for dense encoding in the structured model. In the 50%
  NM condition, unconditional retrieval of irrelevant memories drove endpoint reward
  to -0.959. Therefore endpoint encoding is not globally optimal for a model that
  retrieves unconditionally; the original endpoint-versus-midpoint comparison does
  not establish otherwise.
- The next architectural check is content-based retrieval suppression. It may use
  similarity between the current partial event and stored situations, but not the
  latent RM/DM/NM condition. If this does not make endpoint encoding beneficial under
  the original mixture, the result will be treated as a task-identification failure.

## 2026-08-30 — Content-gated retrieval and stopping diagnosis

- A content-based retrieval gate now compares the partial current situation with
  stored situations. It receives no condition label. Exploratory fixed-schedule
  sweeps selected a conservative match threshold of 0.60, sharpness 30, and retrieval
  strength 0.20 before the final policy diagnostic.
- On 768 exploratory released-profile training trials, endpoint-only expected reward
  was 0.711, compared with 0.693 for midpoint-only, 0.698 for
  midpoint-plus-endpoint, 0.687 for dense, and 0.687 for never encoding. Thus the
  retriever can make the endpoint schedule useful without changing the task.
- A shared discrete actor initialized uniformly at 0.02 encoding probability was then
  trained for 1,500 free-selection updates after 100 critic-only updates. It converged
  from the opposite starting point to the same qualitative failure: endpoint
  probability 0.0015, nonendpoint probability about 0.079, and an early encoding band
  over observations 1–7. Held-out DM reward was 0.488, indistinguishable in practical
  terms from retrieval disabled/no encoding.
- The stopping diagnosis is therefore not merely high-variance policy gradients or an
  overly dense initialization. Under the exact task, a shared online policy receives
  no prospective signal that distinguishes a future target event from a distractor,
  and multiple nonboundary schedules lie in easier local optima. The published task
  supports endpoint superiority among selected fixed schedules, but does not uniquely
  identify endpoint encoding as the globally learnable unconstrained policy.

## 2026-08-31 — Approved training-budget-matched follow-up

- The user approved a new gated plan after auditing the original paper's 600
  supervised and 400 A2C epochs, with 256 event sequences per epoch.
- The earlier 600-update encoding run is retained as a limited-budget diagnostic.
  The new primary policy budget is 400 epochs, 102,400 sequences, and 6,400
  batch-level updates, with checkpointed held-out learning curves.
- Future relevance remains naturally unobservable and is treated as a scientific
  constraint, not a limitation to remove. No prospective target label will be added.
- One trace per event and counterfactual reward enumeration are debugging stages.
  The basic claim requires a neural actor using the exact accumulated state; the
  stronger result requires sampled online learning and two total trace slots.
- Only after exact-state success will a pretrained recurrent situation model replace
  exact feature recording. Exploratory and locked confirmatory seeds remain separate.

## 2026-08-31 — Checkpoint measurement contract implemented

- Fixed development trials can now be evaluated during free-policy training without
  advancing the training random-number stream or changing the model's parameters or
  training mode. The final update is always evaluated even when it is not an exact
  multiple of the checkpoint interval.
- A checkpoint row contains both its held-out results and the matching training row,
  plus epoch, update, total sequences processed, and measured evaluation runtime.
  Optional weight files are referenced by paths in the JSON and remain ignored by
  Git.
- Two requested independent reviews were launched but both agents lost their
  connections before returning findings. The same contract and code-quality checks
  were completed locally before the full test and lint gates.

## 2026-08-31 — Temporal audit uses the task's 16-step evaluation form

- Stage 1 uses the exact released-code evaluation generator: 16 observations,
  zero added delay, no removed observations, and a fixed error penalty of two. This
  makes the approved 16-hazard debugging model well-defined. It is not silently
  treated as the final variable-delay training task.
- The policy parameterizes the probability of first encoding at each time given that
  it has not encoded earlier. The seventeenth outcome is never encoding. Its initial
  distribution is uniform over these 17 outcomes, so no encoding time is favored.
- Completed trial outcomes are used only to build an exact delayed-reward objective;
  the optimizer receives no endpoint target. The same hazards govern both events.

## 2026-08-31 — Exact shared temporal objective passes

- Across the three declared exploratory seeds, the exact optimizer assigned a mean
  probability of 0.9979 to endpoint encoding on independent DM mappings. Learned
  expected reward exceeded both never and matched-random-one encoding in every seed.
- For every seed, endpoint was the best deterministic time when one time had to be
  shared across events. Without the shared-policy constraint, the best pair was
  never encoding in `a1` and endpoint encoding in `b1`. This is expected because the
  former is a distractor and the latter is later relevant; the policy never receives
  those identities prospectively.
- The positive result is limited to a 16-parameter temporal schedule, one trace per
  event, exact counterfactual credit, DM trials, and the zero-delay/no-removal
  evaluation form. It diagnoses a valid objective and optimizer. It does not yet show
  that a state-dependent neural policy can discover the rule.

## 2026-08-31 — Neural exact-state development bank

- The neural actor receives the accumulated 64 feature-value entries and 16 query-
  context entries already produced by the structured model. It receives no separate
  completion count, event identity, time, boundary, or future-relevance variable.
- A first 16-mapping, 1,000-update pilot achieved training reward 0.701 but only
  0.236 endpoint probability on 16 new mappings. This is recorded as a small-bank
  overfitting diagnostic, not evidence of success.
- The next declared run uses 256 unique training mappings and 128 fixed development
  mappings. Its 6,400 batches of 16 give the approved 102,400 example exposures, but
  the 256 exact counterfactual examples are reused. This is explicitly an
  architecture-development approximation; it does not satisfy the later requirement
  for 102,400 newly generated sampled-RL sequences.
