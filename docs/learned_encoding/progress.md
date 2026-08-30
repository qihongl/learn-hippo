# Progress Ledger

This file records pushed scientific milestones. Detailed implementation and result
decisions belong in `decision_log.md`.

- 2026-08-30: pinned upstream revision and preregistered the simplified task,
  data-generating contract, baselines, success criteria, and failure interpretations.
- 2026-08-30: added the isolated modern Python environment, deterministic event
  generator, variable-duration null steps, differentiable key-value read, and exact
  fixed-schedule evaluator. Seven deterministic tests pass before measured runs.
- 2026-08-30: added a versioned oracle configuration and an auditable JSON runner
  that records synthetic-data provenance, package version, Git revision, all 16
  schedules, named baselines, and both capacity conventions.
- 2026-08-30: executed the 256-episode exhaustive oracle audit from `d59be78`.
  Endpoint-only was uniquely optimal under both capacity conventions; fixed
  four-slot capacity was selected before policy learning.
- 2026-08-30: implemented the boundary-blind Bernoulli actor-critic, discrete
  memory-controlled rollouts, delayed-reward objective, deterministic training, and
  frozen held-out evaluation. Fourteen tests cover the complete learning seam.
