# Predeclared experiment: stabilizing full-mixture boundary encoding

## 1. Research question and hypothesis

The fixed-duration distant-memory (DM) experiment established that sampled delayed
prediction reward can teach the exact-state neural policy to encode selectively at
event endpoints. The unresolved question is whether the same policy can retain that
solution when trained on the released 0.25 recent-memory (RM), 0.25 DM, and 0.50
no-relevant-memory (NM) mixture. The best previous intervention combined a low
initial encoding probability, retrospective condition-centered reward, and gradual
DM-to-mixture training. It found endpoint encoding in all three fresh seeds by epoch
390, but one seed moved to a pre-endpoint policy during the final ten epochs. The
final-checkpoint audit therefore failed.

This experiment tests a narrow optimization hypothesis: the selected method finds
the boundary solution, but constant-rate, small-batch policy updates do not retain it
reliably. We predict that reducing the learning rate during the second half of
training and/or doubling the sampled batch will reduce late checkpoint-to-checkpoint
transitions. The primary expected effect is a higher minimum final endpoint
probability across paired seeds, not merely a higher mean or a favorable earlier
checkpoint. No task outcome, model input, episodic-memory rule, reward, or total
sequence exposure changes.

## 2. Synthetic data and partitions

The experiment uses the existing `released_code` synthetic generator documented in
`followup_data_contract.md`. Every sequence contains the original 37-dimensional
input and the $a1$--$b1$--$b2$ event structure. Observation order, situation
mappings, delays, penalties, and removed observations are newly sampled. Training
conditions follow the unchanged RM/DM/NM/NM cycle, which implements weights 0.25,
0.25, and 0.50. No human data are inspected or used.

Seeds 970, 971, and 972 are paired exploratory model seeds shared by all four
optimizer cells. These seeds have not appeared in earlier experiments. Each seed
receives new training mappings on every sequence. Checkpoint evaluation uses 128
fixed unseen trials per condition from a separate namespace beginning at 99,000,000.
Weights are frozen during evaluation, and evaluation does not advance the training
random-number stream. The final checkpoint is the primary result even when an
earlier checkpoint is better.

## 3. Model and fixed learning mechanism

The model and retrieval system are unchanged from the selected 400-epoch follow-up.
The encoding policy is a two-hidden-layer neural network that receives the
80-dimensional accumulated situation state. It receives no endpoint, time,
condition, event-identity, or future-relevance input. The memory store has two global
first-in, first-out slots, while the current debugging decision rule permits at most
one trace from each encoding event. Differentiable content-based retrieval uses the
same temperature, similarity threshold, sharpness, and retrieval strength as the
previous experiment.

Every cell begins with 25 epochs, or 6,400 sequences, of random forced exploration
to train the value estimator while leaving the encoding policy fixed. It then uses
400 epochs, or 102,400 newly generated sequences, of sampled free-policy learning.
The initial encoding probability is 0.05. A condition-centered baseline may use the
condition observed after $b2$ to center the completed reward, but condition is never
available to the online encoding policy. The training distribution changes
deterministically from DM-only to the full unchanged mixture over the 400 free-policy
epochs.

## 4. Paired two-by-two intervention

The screen crosses two predeclared factors:

| Factor | Level 1 | Level 2 |
|---|---|---|
| Policy learning-rate schedule | constant 0.001 | 0.001 through epoch 200, then cosine decay to 0.00005 at epoch 400 |
| Sampled batch | 16 sequences and 6,400 updates | 32 sequences and 3,200 updates |

Both batch levels therefore see exactly 102,400 free-policy sequences. Checkpoints
occur every ten epochs: every 160 updates for batch 16 and every 80 updates for
batch 32. The larger batch is a variance intervention, not an increase in data or
compute budget. The cosine schedule is fixed by training progress alone; it does not
observe endpoint probability, reward, or convergence and cannot preserve a model by
selecting its best checkpoint.

The constant/batch-16 cell is a paired replication of the prior selected method on
new seeds. It is retained even though it is expected to be less stable. No cell is
stopped early, restarted, extended, or assigned additional seeds after its trajectory
is seen.

## 5. Outcomes, baselines, and selection rule

The model seed is the unit of replication. For each cell, the report will retain all
three final endpoint probabilities, endpoint-minus-nonendpoint gaps, separate $a1$
and $b1$ endpoint probabilities, final-five-checkpoint stability, the largest
post-epoch-200 drop between adjacent checkpoints, and complete learning curves.
Prediction outcomes include learned DM reward and paired forced-endpoint, never,
matched-random-one, target-memory-removal, and distractor-memory-removal baselines.
Runtime and peak Slurm resource use are cost outcomes.

A screen cell passes only if all three seeds meet all of the following at epoch 400:

1. endpoint probability is at least 0.80 and endpoint gap is at least 0.50;
2. endpoint preference is present separately in $a1$ and $b1$;
3. the endpoint and gap criteria hold at each of the final five checkpoints;
4. learned DM reward exceeds never and matched-random-one encoding;
5. removing the target memory removes at least 80% of the learned benefit, while
   removing the distracting memory preserves it; and
6. no forbidden online policy input or best-checkpoint selection is used.

If multiple cells pass, the cell with the highest minimum final endpoint probability
across seeds advances. A difference smaller than 0.02 is treated as a tie; ties favor
batch 16 because it changes fewer properties of the earlier method, then favor the
decaying schedule because it directly addresses late instability. If no cell passes,
none advances and the result is a failed stability intervention. A passing cell is
still exploratory: it must next pass ten fresh full-mixture seeds before the
unreserved-memory actor or recurrent situation model is implemented.

## 6. Compute, storage, and provenance

The four cells by three seeds form a 12-task CPU Slurm array on Della. Each task
requests one CPU, 4 GB memory, and 90 minutes under the `knorman` account. The code
runs from a Git checkout in `/home/qlu/learn-hippo` at one committed revision. A
project-specific Python environment and all large or transient artifacts live below
`/scratch/gpfs/KNORMAN/qlu/learn-hippo`.

The immutable run root is
`/scratch/gpfs/KNORMAN/qlu/learn-hippo/runs/<date>-<git-sha>/optimizer-stability`.
Each array task writes a separate raw JSON and Slurm log. Environment records,
configuration hashes, Git revision, and `sacct` state are retained. After completion,
every file is fetched into a fresh immutable local result directory before compact
summaries are versioned. Scratch is treated as temporary rather than archival.

Expected cost is approximately 25--35 minutes and 0.4 GB per task, or about six CPU
hours total. With twelve concurrent tasks, compute wall time should be below one
hour after scheduling. Any cluster rejection or projected budget overrun pauses the
experiment rather than changing seeds, exposure, or evaluation trials.

## 7. Risks and stopping interpretation

Three seeds are enough for a bounded method-development screen but not a final
reliability claim. Condition-centered credit remains a privileged retrospective
signal and must later be replaced with an observable-outcome baseline. The current
one-trace-per-event decision rule still assumes event segmentation. Most importantly,
a positive optimization screen would show that stability can be engineered; it
would not establish that the solution is biologically natural or reliably learned
across twenty seeds. Conversely, a negative screen would not prove that all
optimization methods must fail. It would show that these two targeted, predeclared
variance controls do not solve the measured instability within the reference
400-epoch budget.

## 8. Outcome

The constant-rate, batch-32 cell was the only cell to pass the frozen all-seed
rule. Its seed-level final endpoint probabilities were 0.9929, 0.9969, and 0.9965;
all three also passed the final-five-checkpoint, event-specific, reward, and memory-
removal criteria. The constant-rate, batch-16 cell failed because one seed was not
selective throughout the final five checkpoints, despite ending at 0.9500. Each
cosine-decay cell failed one seed, with final endpoint probabilities of 0.00005 and
0.0460. The deterministic rule therefore selects constant rate with batch 32.

This is partial support for the optimization hypothesis. The larger batch improved
the final window, whereas learning-rate decay could preserve a wrong basin. It did
not eliminate policy excursions: selected-cell seed 970 dropped from 0.9405 at
epoch 230 to 0.0065 at epoch 240 before recovering. Accordingly, the cell advances
only to the planned ten-seed replication and is not described as converged or as a
confirmed full-mixture result.

The 12-task Della array used one CPU per task and at most 0.37 GB. Eleven tasks
completed in 42:43--54:50. One task timed out under the predeclared 90-minute limit
and was rerun unchanged with a two-hour limit, completing in 91:37. All 51 result,
log, and environment files were fetched and checksum-verified; the original timeout
was retained rather than overwritten.
