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
