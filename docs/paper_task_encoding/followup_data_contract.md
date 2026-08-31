# Data contract: reward-learned boundary encoding follow-up

## Task binding

- **Question:** Can prediction reward teach a shared online policy to encode an
  episodic memory selectively at the endpoint of an event on the Lu, Hasson, and
  Norman (2022) prediction task?
- **Task type:** controlled synthetic sequential prediction with differentiable
  episodic retrieval and discrete online encoding decisions.
- **Primary task profile:** `released_code`, as specified in `data_contract.md` and
  `source_audit.md`.
- **Sensitivity profile:** `paper_text`; its results remain separate from the
  primary profile.
- **Primary outcome:** endpoint encoding probability minus mean nonendpoint
  encoding probability on unseen distant-memory (DM) mappings.
- **Supporting outcomes:** prediction reward relative to never and matched-random
  encoding, and loss of the learned benefit when retrieval is disabled.

## Data route

- **Mode:** controlled synthetic generator; no human observations are used.
- **Source:** `src/boundary_em/paper_task.py`, independently validated against the
  eLife article and archived `learn-hippo` source revision
  `6a4a1be4fd6780d4c8413ffc6b1facade4741135`.
- **Paper:** https://doi.org/10.7554/eLife.74445
- **Access:** public; the repository is MIT licensed.
- **Provenance label:** executed outputs are *measured synthetic simulations*, not
  human behavioral measurements and not simulated placeholder numbers.

## Experiment partitions

The generator has no finite dataset to divide. Separation is implemented with
deterministic, disjoint seed namespaces:

- behavior-test seeds verify invariants and never enter estimation;
- exploratory training seeds are used to develop the optimization method;
- development-evaluation seeds form a fixed checkpoint bank used only to diagnose
  learning and convergence;
- confirmatory model seeds and trial seeds remain untouched until the architecture,
  training budget, metrics, and configuration hash are frozen;
- sensitivity seeds are separate from the primary `released_code` run.

The post-report optimizer-stability screen uses paired exploratory seeds 970--972
and a fixed evaluation namespace beginning at 99,000,000. These remain separate
from all prior development seeds and from any later ten- or twenty-seed confirmation.

Every checkpoint for a model seed is evaluated on the same development trial bank,
making changes over training paired rather than confounded by newly sampled trials.
The final test bank is newly sampled. All model weights are frozen during every
development and test evaluation.

## Training exposure

One follow-up epoch contains 256 newly generated event sequences, matching the
convention in the original paper. The main encoding-policy budget is 400 epochs, or
102,400 sequences. With batch size 16 this gives 6,400 parameter updates. A
batch-size-one sensitivity is permitted only under the conditional rule in
`followup_experiment_design.md`; it is not silently substituted after a failure.

The initial debugging curriculum may use only DM trials. The final exact-state and
neural-state claims require the original 0.25 RM / 0.25 DM / 0.50 NM mixture. Results
from the curriculum and full mixture are stored and reported separately.

## Model-input contract

The encoding policy may use only a state computed from the observations available
so far and the internal state of its own episodic-memory system. It may not receive:

- a boundary or endpoint flag;
- elapsed or remaining time;
- a separately supplied observed-feature count or completion mask;
- the RM/DM/NM condition;
- a label identifying a future target or distractor;
- a future observation, query, reward, or retrieval outcome.

During initial debugging, the situation state records observed features exactly.
Unobserved rows remain zero, so completeness can in principle be inferred from the
record itself; it is not supplied as an additional input. A later recurrent model
must construct its state from the original 37-dimensional sequence.

Counterfactual reward enumeration is an exploratory training algorithm. It may use
completed simulated outcomes to compute a lower-variance training objective, just
as reinforcement learning uses eventual reward, but it does not change the policy's
online inputs at evaluation. It never supplies an endpoint target.

## Capacity contract

- A one-trace-per-event capacity is an explicitly labeled debugging condition.
- The final basic exact-state model receives two total trace slots across the two
  encoding events, with no slot reserved for either event.
- Four- and 40-slot runs are sensitivity analyses.
- A capacity change cannot replace a failed primary two-slot result.

## Records retained

Every model seed retains the complete per-update training history. Every checkpoint
record contains epoch, update, sequences processed, learning rate, training reward,
actor and critic losses, entropy, gradient norm, encodings per event, endpoint and
nonendpoint probabilities, and evaluation runtime. Development checkpoints add
held-out reward, the complete time-resolved encoding distribution, forced baselines,
and retrieval-off results. Checkpoint weights are kept outside Git; compact metrics
and their configuration hashes are versioned.

## Leakage and publication rules

No model or trial seed is filtered. Hyperparameters may be changed using exploratory
and development results only. Once the confirmatory configuration is committed, no
choice may depend on confirmatory outputs. Exploratory, confirmatory, task-profile,
capacity, curriculum, and recurrent-model results are never pooled. Negative and
ambiguous outcomes remain in the record.
