# Decision Log: Learned Boundary Encoding

## 2026-08-30 — Repository and provenance

- Work proceeds in a separate clone of `qihongl/learn-hippo`, not inside
  `learn-ei`, on branch `codex/learned-encoding-policy`.
- The upstream starting point is pinned at
  `6a4a1be4fd6780d4c8413ffc6b1facade4741135`.
- The legacy repository is treated as the historical reference. New minimal code
  will be isolated from its legacy training stack so simplifying assumptions remain
  explicit and tests can run on a current Python environment.

## 2026-08-30 — Scientific scope

- A four-feature event is acceptable for primary evaluation because the user
  explicitly permits a shortened task.
- Boundary usefulness must be established by exact schedule enumeration before
  learned-policy training.
- The primary policy receives no explicit boundary bit. It may inspect accumulated
  feature state and observation mask.
- The primary storage action is a distinct Bernoulli write trained by actor-critic.
  Straight-through or soft writes are optimization diagnostics.
- No write cost is included in the primary task. Selectivity must be useful because
  incomplete traces interfere with delayed retrieval.
- Both historical and fixed-slot capacity conventions will be checked. The simpler
  stable convention will be selected and logged before policy training.

## Fixed completion criteria

- Endpoint-only encoding is optimal or tied-optimal under exhaustive schedule
  evaluation.
- Fifteen independent policy seeds are retained.
- Learned endpoint selectivity is positive in at least 12 seeds and its paired 95%
  confidence interval excludes zero.
- The learned policy closes at least 80% of the matched-random-to-endpoint-oracle
  held-out reward gap.
- Moving learned endpoint writes to mid-event positions reduces held-out reward.
- Failure to meet any criterion is reported with a staged diagnosis rather than
  hidden by changing the task after confirmatory evaluation.

## 2026-08-30 — Capacity convention selected after oracle audit

- The fixed four-slot and historical schedule-scaled conventions produced identical
  rankings and rewards in the one-event oracle audit because neither evicted an
  eligible trace.
- Fixed four-slot capacity is selected for learned-policy training because it gives
  every policy the same physical resource and is simpler to vectorize.
- This choice was made at commit `d59be78`, before any learned write policy was run.
