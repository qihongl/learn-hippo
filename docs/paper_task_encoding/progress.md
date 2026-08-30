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
