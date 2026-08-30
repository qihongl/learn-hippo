# Data contract: learned encoding on the 2022 prediction task

## Task binding

- **Research question:** Can delayed next-state prediction reward train a model with
  differentiable episodic retrieval to encode selectively at event boundaries on
  the task used by Lu, Hasson, and Norman (2022)?
- **Task type:** controlled synthetic sequential prediction with within-trial
  working-memory and episodic-memory manipulations.
- **Target:** the four-valued next state specified by each fixed-order query.
- **Primary evidence:** held-out DM prediction reward and learned encoding timing.

## Data route

- **Mode:** controlled synthetic generator.
- **Source:** modern, deterministic implementation of the task defined by the eLife
  article and archived `learn-hippo` source revision.
- **Paper:** https://doi.org/10.7554/eLife.74445
- **Archived source:**
  https://archive.softwareheritage.org/swh:1:rev:6a4a1be4fd6780d4c8413ffc6b1facade4741135
- **Local source revision:** `6a4a1be4fd6780d4c8413ffc6b1facade4741135`.
- **Access:** public; repository code is MIT licensed.
- **Human data:** none.

## Generated records

Every generated trial records the profile name, model seed, trial seed, condition,
penalty, situation values, distractor situation, independent observation orders,
observation-removal indicators, delay, input tensors, query targets, valid-prediction
mask, encoding eligibility, imposed or learned encoding decisions, episodic-memory
identity, retrieval weights, predictions, and rewards. Compact aggregate files may
omit large tensors, but seed-level metrics and configuration hashes must remain.

## Profiles

- `released_code`: primary profile reproducing the executable generator, including
  delay 0–4 and its historical random-count observation removal.
- `paper_text`: sensitivity profile reproducing the prose description, including
  delay 0–3 and independent 0.30 observation removal.
- Meta-test for both profiles uses delay 0, no observation removal, penalty 2, new
  situations, and frozen model weights.

Neither profile may be retuned after its held-out test results are inspected. Results
from the two profiles must not be pooled.

## Split protocol

- Generator, optimizer, exploratory validation, and confirmatory evaluation use
  disjoint deterministic seed namespaces.
- The situation distribution is shared, but exact 16-feature mappings, distractor
  mappings, observation orders, conditions, and delays are newly sampled.
- Exploratory seeds may select optimization settings. Confirmatory test seeds may be
  evaluated only after the configuration hash and success checks are frozen.
- At least 10 independent model-training seeds are retained; no seed is filtered.
- Within a model seed, every baseline and ablation uses the same trial bank.
- All model weights are frozen during held-out evaluation. Formation and retrieval
  of new episodic memories within a test trial remain enabled as the task requires.

## Leakage and invariants

The encoding policy may use the recurrent situation model produced from the original
37-dimensional input. It may not receive a new boundary flag, elapsed-time counter,
remaining-time counter, condition label, target value before prediction, future
observation order, or reward before acting. RM/DM/NM is an environment manipulation,
not an input label. Forced schedules are evaluated separately and never provide
supervision to the free encoding actor.

The main leakage risks are reuse of trial seeds across splits, accidental exposure
of the final-time indicator, updating model weights during meta-test, or allowing
the target `b1` memory in NM. Deterministic behavior tests must cover each risk.

## Provenance and publication

Numbers produced by executed runs are **measured synthetic simulations**. They are
not human data and must never be labeled as empirical behavioral evidence. Simulated
placeholders are prohibited. Every reported number must trace to a configuration,
source revision, model seed, and machine-readable output. Failures and deviations
remain reportable rather than being removed during tuning.
