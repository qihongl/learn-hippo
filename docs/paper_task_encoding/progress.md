# Progress ledger

- 2026-08-30: traced the published task and archived source, recorded the shared task
  contract, and exposed two paper/code differences in delay and feature removal.
- 2026-08-30: froze the data contract, success criteria, staged optimization,
  baselines, causal tests, and failure interpretations before implementation.
- 2026-08-30: added the modern exact-task generator and five deterministic behavior
  tests covering the 37-dimensional input, source profiles, RM/DM/NM situation
  relationships, and boundary-blind deterministic sampling.
- 2026-08-30: added the recurrent situation model, five-response prediction head,
  differentiable competitive episodic retrieval, and an encoding actor that receives
  only the recurrent situation model. Three model behavior tests pass.
- 2026-08-30: implemented complete `a1`–`b1`–`b2` rollouts, forced encoding
  schedules, exact expected task reward, RM/DM/NM working-memory resets, and
  profile-specific lure availability. Five rollout behavior tests pass.
- 2026-08-30: added supervised prediction/retrieval objectives and a reproducible
  training stage that can run with episodic memory absent or under a forced encoding
  schedule. A repeated exact trial is learnable in the training test.
- 2026-08-30: added delayed encoding optimization with a separate critic-only forced
  exploration stage and a free Bernoulli actor-critic stage. Tests verify that the
  forced stage leaves the actor unchanged and both stages freeze the complete
  prediction and retrieval system.
- 2026-08-30: the recurrent predictor failed the fixed-schedule precondition in an
  exploratory high-budget run. Added a boundary-blind structured situation model to
  isolate encoding-policy learning. On new DM trials its fixed endpoint schedule
  beats midpoint, midpoint-plus-endpoint, dense, and no-encoding alternatives when
  the same schedule controls both distracting and target events.
- 2026-08-30: corrected the free-policy objective so the learned policy controls both
  distracting and target events. A regression test verifies that both sampled action
  sequences produce the corresponding memory traces. The corrected discrete
  actor-critic learned higher endpoint than nonendpoint probability in one diagnostic
  seed, but encoded too densely and performed below the no-encoding baseline.
- 2026-08-30: added continuous differentiable encoding strengths and verified that
  delayed prediction reward has gradients to encoding in both events. This optimizer
  also failed, favoring the first observation. A fixed-schedule audit exposed the
  deeper cause: with unconditional retrieval on the full released training mixture,
  no encoding strongly outperforms endpoint encoding because irrelevant memories are
  damaging in NM trials.
- 2026-08-30: added content-based retrieval suppression without exposing the hidden
  memory condition. Conservative retrieval made endpoint-only the best declared
  fixed schedule, but a sparse shared actor again learned an early encoding band and
  no held-out DM benefit. This satisfies the stopping condition with a replicated
  task-identification and optimization diagnosis rather than a positive result.
- 2026-08-30: added versioned smoke and 10-seed diagnostic-replication
  configurations plus a runner that freezes each trained model, evaluates new
  mappings in RM/DM/NM, pairs retrieval-on and retrieval-off actions, retains all
  forced baselines, and writes provenance-rich machine-readable results.
- 2026-08-31: aggregated every declared seed, added forced-schedule and causal-memory
  ablations, and completed a self-contained paper-style HTML report with four
  reproducible explanatory and measured-data figures. The report preserves the
  negative policy result and separates it from successful memory use.
- 2026-08-31: audited the original 600-epoch supervised and 400-epoch A2C budget,
  documented which current learning curves can be reconstructed, and froze the
  user-approved follow-up contract, success criteria, convergence rules, capacity
  ladder, recurrent extension, and confirmatory split before new implementation.
- 2026-08-31: implemented deterministic fixed-bank checkpoint evaluation and
  complete per-update exposure records. Checkpoints retain epoch, update, sequences
  processed, contemporaneous training metrics, held-out schedule/ablation results,
  evaluation runtime, and optional weights; tests verify that evaluation does not
  change model parameters, mode, random streams, or the subsequent training result.
- 2026-08-31: implemented the Stage 1 temporal-hazard audit. Sixteen online
  first-encoding hazards are shared across the distracting and target events; a
  17-by-17 reward table enumerates each pair of encoding times plus never encoding.
  Unit tests verify the literal online distribution, reward gradients, endpoint
  rollout equality, endpoint learnability, and reproducible result serialization.
- 2026-08-31: completed all three declared temporal-audit seeds. On 64 unseen DM
  mappings per seed, mean endpoint encoding probability was 0.9979 and the endpoint
  gap was 0.9977. Mean learned reward was 0.6551, compared with 0.4896 for never and
  0.5145 for matched-random-one encoding. From update 800 to 1,000, endpoint
  probability changed by 0.0010 and reward by 0.00011, indicating practical
  convergence. The neural exact-state policy is therefore the next gate.
