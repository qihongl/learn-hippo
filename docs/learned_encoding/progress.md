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
