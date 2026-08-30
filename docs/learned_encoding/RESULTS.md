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
