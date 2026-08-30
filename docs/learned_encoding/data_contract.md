# Data Contract: Learned Boundary Encoding

## Task binding

- **Research question:** Can delayed prediction reward teach a differentiable
  episodic-memory agent to store complete event-boundary states selectively?
- **Task type:** synthetic sequential event study followed by delayed feature
  prediction.
- **Target variable:** independently sampled event features withheld at delayed test.

## Dataset route

- **Mode:** controlled synthetic generator.
- **Source:** a new minimal generator derived from the computational logic, but not
  copied implementation, of Lu, Hasson, and Norman (2022).
- **Original repository:** https://github.com/qihongl/learn-hippo
- **Pinned source revision:** `6a4a1be4fd6780d4c8413ffc6b1facade4741135`.
- **Paper:** https://doi.org/10.7554/eLife.74445
- **Snapshot date:** 2026-08-30.
- **Licence:** MIT for the repository; generated synthetic episodes contain no
  external or human data.

## Generated records

Each episode contains random event cues, four independent binary feature values,
random feature-reveal orders, declared null-step placements, study states, sampled
write actions, memory keys and values, delayed queries, feature targets, retrieval
attention, and prediction reward. No human pilot data, proposal data, or external
behavioral observations are used.

## Split protocol

- Training, validation, and confirmatory test episodes are generated from disjoint
  deterministic seed namespaces.
- Fifteen model-training seeds are retained in full.
- Validation episodes are used for configuration selection.
- Confirmatory episodes use unseen event cues, feature vectors, reveal orders, and
  null-step patterns; model weights are frozen.
- Baselines within a model seed use the same held-out episode bank for paired
  comparisons.
- Exact seed ranges and episode counts will be frozen in versioned YAML before the
  first confirmatory run.

The main leakage risks are reusing a pseudorandom stream between training and test,
exposing an endpoint flag to the policy, or selecting a task parameter after viewing
learned-policy test performance. Tests and saved configuration hashes must address
all three.

## Preprocessing contract

Event cues are seeded random vectors. Feature values are balanced independent draws
from `{-1, +1}`. Partial states use zero for unrevealed values and a separate binary
observation mask. Inputs are generated directly at model scale; there is no fitted
normalization. The policy may receive the current partial state and mask but not an
explicit boundary label, remaining-step count, target feature values, or test reward
before acting.

## Provenance and publication

All numerical results are measured executions of a synthetic environment. They must
be labeled **measured synthetic simulations**, never human data and never fabricated
or illustrative results. Each aggregate must trace to configuration- and seed-level
files. The final report must identify the generator version, Git commit, hardware,
and package versions and must preserve negative and failed configurations.
