# Results: Learned Boundary Encoding

> **Provenance:** measured executions of a controlled synthetic task. These are not
> human data and are not illustrative or fabricated numbers. This document is
> updated stage by stage; no learned-policy result existed when the oracle section
> was written.

## Stage 0: environment and oracle validation

- **Configuration:** `configs/learned_encoding/oracle.yaml`
- **Machine-readable result:** `outputs/learned_encoding/oracle_results.json`
- **Code revision used for execution:** `d59be7811fbb6687d4f9c640e526217d8cc8e198`
- **Episodes:** 256 deterministic synthetic event seeds, 10000–10255
- **Schedules:** all 16 binary write masks over four informative positions

Endpoint-only writing (`0001`) was the unique optimal deterministic schedule. Its
mean delayed feature-prediction reward was 1.000. The second-ranked schedule was
lower by 0.267. This ranking was identical under fixed four-slot capacity and the
historical schedule-scaled capacity convention.

| Fixed-capacity policy | Mean reward | SD across episode seeds |
|---|---:|---:|
| Endpoint only | 1.000 | 0.000 |
| Matched random one-write | 0.375 | 0.415 |
| Always write | 0.241 | < 0.001 |
| Midpoint plus endpoint | 0.140 | < 0.001 |
| Midpoint only | 0.000 | 0.000 |
| Never write | 0.000 | 0.000 |

The near-zero episode variance for deterministic schedules is expected: independent
binary feature values change content but not the task's similarity geometry or the
fraction of held-out features recovered. The matched-random value pools the four
possible one-write positions across episodes, so its dispersion reflects schedule
position rather than unstable event sampling.

This stage establishes the required optimization target, not learning. The endpoint
contains all four independent features. The delayed query matches the incomplete
two-feature state exactly, so storing that state competitively blocks the less
similar but complete endpoint. Endpoint-only storage avoids this interference.
Because both tested capacity conventions retain every eligible trace, the observed
advantage is not caused by endpoint-only encoding receiving more slots or avoiding
eviction.

Fixed four-slot capacity is used for subsequent policy experiments because it keeps
physical memory resources identical across policies. The historical convention is
retained as a documented oracle diagnostic.

## Exploratory actor-critic smoke test

- **Configuration:** `configs/learned_encoding/smoke.yaml`
- **Code revision:** `d6ad84833e52b276c93c9f976905844ac099f2e2`
- **Model seeds:** 0, 1, and 2
- **Held-out validation episodes:** 256 per model seed
- **Status:** exploratory configuration check; not confirmatory evidence

All three actor-critic seeds learned a highly selective write policy without a
boundary label or write cost. Mean stochastic validation reward was 0.982 ± 0.012
across the three model seeds. Endpoint-minus-nonendpoint write probability was
0.983 ± 0.008, and boundary AUC was 1.000 for every seed. The policy made 1.007 ±
0.016 writes per event. Thresholded deterministic evaluation wrote once at the
endpoint and achieved reward 1.000 for all three seeds.

These runs used validation seeds and were inspected before freezing the confirmatory
configuration. They establish that the plain delayed actor-critic optimization is
viable in the simplified task, but they are not counted toward the preregistered
15-seed success criterion.

## Confirmatory learned-policy result

- **Configuration:** `configs/learned_encoding/reported.yaml`
- **Seed-level outputs:** `outputs/learned_encoding/reported/`
- **Aggregate:** `outputs/learned_encoding/reported_summary.json`
- **Model seeds:** 100–114, all retained
- **Held-out test episodes:** 1,024 per model seed
- **Training:** 300 actor-critic updates, unchanged from the exploratory smoke run

The learned stochastic policy achieved mean held-out reward 0.987 ± 0.004 across
15 model seeds (fixed-seed bootstrap 95% interval [0.986, 0.989]). Mean
endpoint-minus-nonendpoint write probability was 0.985 ± 0.006, interval [0.982,
0.988]. All 15 seeds had positive endpoint selectivity, exceeding the preregistered
requirement of 12. Thresholding write probability at 0.5 produced exactly one
endpoint write and reward 1.000 for every seed.

Relative to the paired matched-random one-write baseline, the stochastic policy
closed 0.980 of the reward gap to the endpoint oracle, interval [0.977, 0.983],
exceeding the required 0.80. Moving the deterministic learned endpoint write to the
midpoint reduced reward from 1.000 to 0.000 for every seed; the paired displacement
loss was therefore 1.000, interval [1.000, 1.000]. All four confirmatory criteria
passed under the analysis frozen before test execution.

| Method or intervention | Reward mean | Model-seed SD |
|---|---:|---:|
| Learned stochastic policy | 0.987 | 0.004 |
| Learned deterministic policy | 1.000 | 0.000 |
| Endpoint-only oracle | 1.000 | 0.000 |
| Matched random one-write | 0.371 | 0.015 |
| Always write | 0.241 | 0.000 |
| Midpoint plus endpoint | 0.140 | 0.000 |
| Displaced learned write | 0.000 | 0.000 |

### Duration generalization

Training events had four informative states. The OOD test inserted three randomly
placed null states, increasing duration to seven while preserving the semantic rule
that the endpoint is the first state with all four features accumulated. The policy
was not retrained. OOD stochastic reward was 0.982 ± 0.009, interval [0.978, 0.986],
and endpoint selectivity was 0.986 ± 0.005. Thus the policy did not merely write at
the fourth physical timestep; it generalized using the accumulated state available
to the gate.

### Interpretation boundary

This is positive evidence for computational feasibility in the simplified task:
delayed prediction reward trained a discrete write gate to use a differentiable
content-addressable memory selectively at semantic completion. The memory read is
differentiable, but the discrete write action is trained with actor-critic rather
than by ordinary backpropagation through a fractional memory update.

The task was deliberately constructed so an exact partial-state memory competes
with the less similar complete endpoint. It therefore establishes that optimization
can discover the policy when endpoint storage is independently known to be optimal;
it does not establish that endpoint storage is optimal in arbitrary memory systems.
The event representation and observation mask are explicit, the task contains one
event at a time, and the primary retriever has no learned key projection. These
simplifications make the credit-assignment claim identifiable but limit direct
comparison with the original multi-event LSTM simulations and with human cognition.
