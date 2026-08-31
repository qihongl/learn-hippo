# Locked replication: constant-rate batch-32 full-mixture policy

## Question and status

The three-seed optimizer screen selected constant learning rate with a batch of 32
sequences. This replication asks whether that final-window result generalizes to ten
fresh model seeds without another method choice. It is a locked replication of one
development cell, not a new search and not the final 20-seed recurrent-model
confirmation described in the broader plan.

## Fixed task and model

The simulation remains the released-code 0.25 recent-memory, 0.25 distant-memory,
and 0.50 no-relevant-memory mixture. Situation mappings, observation orders,
delays, removed observations, and penalties are newly sampled. The online encoding
policy receives the 80-dimensional exact accumulated situation state. It receives
no time, event-boundary, event-identity, memory-condition, or future-relevance
input. Differentiable content-based retrieval, the two global memory slots, and the
one-trace-per-event debugging decision rule are unchanged.

The policy starts at encoding probability 0.05. Random forced exploration trains
the value estimator for 25 epochs, or 6,400 sequences. Free policy learning then
uses the selected constant 0.001 learning rate, batch 32, and 3,200 updates over 400
epochs, or 102,400 new sequences. Condition-centered retrospective credit and the
fixed distant-memory-to-mixture schedule are unchanged. Evaluation occurs every ten
epochs on 128 frozen unseen trials per condition from a new trial namespace
beginning at 101,000,000. The final checkpoint is primary; no best checkpoint,
extension, restart, or seed replacement is allowed.

## Seeds and outcomes

Model seeds 980--989 are new and were not used for method selection. Every seed and
checkpoint trajectory is retained. Primary outcomes are endpoint probability,
endpoint-minus-nonendpoint gap, separate endpoint probabilities in events `a1` and
`b1`, the final-five-checkpoint rule, the largest post-epoch-200 checkpoint drop,
DM prediction reward, and target- and distractor-memory removal. Never encoding,
matched-random-one encoding, and forced endpoint encoding remain paired baselines.

The replication passes only if both levels below pass:

1. the existing aggregate ten-seed audit passes: mean endpoint probability at
   least 0.80; mean gap at least 0.50 with its bootstrap interval above zero; at
   least 80% positive-gap seeds; prediction reward above never and random with
   intervals above zero; target-memory removal eliminates at least 80% of the
   benefit; distractor removal preserves benefit; endpoint preference appears in
   both events; and evaluation uses unseen mappings without forbidden inputs; and
2. at least eight of ten individual seeds have final endpoint probability at least
   0.80, endpoint gap at least 0.50, endpoint probabilities at least 0.80 in both
   events, all selectivity thresholds at each of the final five checkpoints, reward
   above never and random, at most 20% target-memory benefit remaining, and positive
   reward after distractor removal relative to never.

Eight of ten is the reliability threshold inherited from the earlier multiseed
contract. Results below it are failures, even if the mean reward is useful. Earlier
excursions remain reported and qualify any convergence interpretation, but they do
not change the locked pass rule after the fact.

## Compute and next gate

Ten independent CPU tasks run concurrently on Della under the `knorman` account.
Each requests one CPU, 4 GB, and two hours. Large raw JSON, Slurm logs, and
environment manifests are written to a fresh immutable directory below
`/scratch/gpfs/KNORMAN/qlu/learn-hippo/runs/<date>-<git-sha>/optimizer-replication`.
Every remote file is fetched and checksum-verified before analysis; compact
summaries and the audit are versioned.

If the replication fails, optimizer development stops at this gate and the failure
diagnosis is reported. If it passes, the next experiment removes the privileged
condition-centered retrospective label while keeping the task and successful
optimizer fixed. Unreserved-memory training and recurrent situation learning remain
later gates and are not implied by this result.
