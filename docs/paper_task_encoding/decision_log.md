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
