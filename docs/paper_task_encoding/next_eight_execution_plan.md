# Execution plan: next eight experiments

## Purpose and fixed scientific question

The goal is to determine whether delayed event-prediction reward can teach a neural
policy with differentiable episodic retrieval to encode selectively at event
endpoints. The policy never receives an endpoint label, a condition label, future
relevance, elapsed time, or a separate completion signal. Future relevance being
unknown during the first part of a trial is part of the natural memory problem, not
a defect in the task. The task generator, reward, and primary two-trace capacity are
not changed to rescue learning.

The eight steps below are ordered diagnostic tests. Each step has a stopping rule so
that later architectural complexity is added only when the simpler model has passed
the relevant test. Every measured run retains all declared seeds, complete learning
curves, configuration hashes, software provenance, and frozen evaluation on unseen
mappings. One epoch is 256 newly generated event sequences.

## Step 1 — execution and measurement contract

Freeze the seed namespaces, 400-epoch budget, evaluation bank, output layout, local
benchmark, and Della job-array manifest. Verify that a command can direct each
seed-level result to a fresh run directory. The local laptop is the reference machine
for debugging; Della is an optional throughput tool. Passing this step means the
fixed and variable DM configurations each declare ten unique seeds, checkpoints occur
every ten epochs, and tests verify the manifest/configuration correspondence.

## Step 2 — fixed-duration DM replication

Train ten fresh sampled-reward actors for 25 forced-exploration epochs followed by
400 free-policy epochs on the fixed 16-observation DM debugging task. This is a
replication of the seed-610 positive result, not a new architecture search. Report
the final and time-resolved endpoint probability, endpoint-minus-nonendpoint gap,
reward against never and matched-random encoding, target-memory removal, distractor-
memory removal, convergence, and total runtime for every seed. The preregistered
multiseed criteria in `followup_experiment_design.md` determine success.

## Step 3 — variable-duration DM replication

Repeat Step 2 with the released-code delay and missing-observation sampling restored.
Ten new seeds train for the same 400 epochs and are evaluated on paired unseen DM
trials with variable event lengths. This directly tests whether the earlier
100-epoch seed-720 result is reproducible and stable. No hyperparameter is selected
from these ten results. Failure here localizes the problem to temporal variability;
success moves the focus to the mixed RM/DM/NM objective.

## Step 4 — exact audit of the mixed objective

Before another neural rescue attempt, optimize one shared 16-position temporal
encoding distribution against an exactly enumerated RM/DM/NM reward surface. The
policy is shared across the first and second events and has no access to condition.
Use multiple random initializations and fresh trial banks. Compare the best learned
distribution, the deterministic endpoint schedule, never encoding, and matched-
random encoding. If endpoint encoding is not a stable optimum even in this small
model, the full-mixture objective itself—not neural state representation—is the
central obstacle.

## Step 5 — bounded credit-assignment interventions

If Step 4 confirms a usable endpoint optimum, test a small predeclared factorial on
the unchanged full mixture: neutral versus low initial encoding probability,
ordinary mixture reward versus a condition-centered retrospective advantage, and
direct mixture training versus slow DM-to-mixture annealing. These changes affect
only how delayed outcomes are assigned to past actions; they add no online input and
do not reveal future relevance. Use exploratory seeds and equal budgets. Choose a
single method only by the declared endpoint and reward criteria, retaining every
failed cell.

## Step 6 — observable-progress neural policy

Compare the unconstrained multilayer perceptron with a policy whose encoding hazard
is constrained to increase as the accumulated situation record becomes more
complete. Completeness is computed from the same observed feature rows already in
the state; it is not an endpoint flag and can remain below one when observations are
missing. The test asks whether a simple inductive bias can make delayed credit
reliable without identifying an event boundary for the model. Report whether the
constraint improves full-mixture selectivity and whether it harms RM or NM reward.

## Step 7 — remove event-reserved memory slots

Replace the debugging convention of one trace reserved for each event with two
unreserved trace slots for the entire trial. The policy must decide whether the
second trace is worth using; an event transition does not clear or reserve storage.
First verify forced endpoint, never, and random schedules under this capacity. Then
train the best method from Steps 5–6 if one passed. Four- and 40-slot variants are
diagnostic sensitivities only. A positive one-per-event result cannot substitute for
failure with two total slots.

## Step 8 — recurrent-state gate and locked conclusion

Add a GRU or LSTM only if the exact situation-state model passes the full-mixture and
two-total-slot criteria. Pretrain it for 600 epochs to reconstruct observed features
and predict queried values from the original 37-dimensional input. Require at least
99% held-out decoding of values that have actually been observed and verify that
forced endpoint encoding still helps before training the actor for 400 epochs. If
the gate passes, freeze the complete protocol and run 20 untouched seeds. If an
earlier gate fails, Step 8 is still executed by recording the failure diagnosis and
formally deferring the recurrent extension rather than spending compute on a harder
model with an unresolved simpler failure.

## Compute placement and budget

The local reference machine is an Apple M5 Pro MacBook Pro with 48 GB memory. Existing
measured runs on this implementation took about 13.1 minutes for one fixed-DM seed
and 15.4 minutes for one full-mixture seed. A 100-epoch variable-DM seed took about
5.7 minutes; the 400-epoch version is expected to take roughly 18–22 minutes. Thus
Steps 2 and 3 together should require about 5.2–5.9 CPU-hours if run sequentially,
or about 1.5–2.5 hours of wall time with three or four single-threaded processes.
These are estimates; observed per-seed runtime is recorded.

Della becomes useful when ten or twenty independent seeds are ready, because a Slurm
array can run them concurrently. It is not needed for debugging and the current code
does not benefit from a GPU. A conservative array task requests one CPU, 4 GB memory,
and 90 minutes. Active runs belong in
`/scratch/gpfs/<ResearchGroup>/<NetID>/learn-hippo/runs/<date>-<git-sha>`; scratch is
not backed up. Code may live in the account's home directory. After validation,
important compact results should move to the group's backed-up `/projects` space or
TigerData and must also be fetched locally. The account-specific research group,
NetID, and Python environment remain explicit setup requirements.

The projected exploratory budget is at most 24 CPU-hours for Steps 1–4, another 24
CPU-hours for Steps 5–7, and up to 72 CPU-hours for the recurrent extension. Any
projected overrun pauses the relevant stage rather than reducing seeds or evaluation
trials silently.

## Execution outcome

All eight decision steps were executed through their declared gates. Step 1 froze
the configuration, seed, output, and Della contracts. Step 2 passed across all ten
fixed-duration DM seeds. Step 3 failed the reliability criterion despite useful
memory in all ten variable-duration seeds. Step 4 showed that the endpoint is the
best deterministic shared time in the full mixture, but five exact optimizations
stopped one or two observations early. Step 5 found a promising credit intervention,
then rejected it at 400 epochs because one of three fresh policies catastrophically
left the endpoint solution at the final checkpoint. Step 6's observable-progress
policy failed. Step 7 verified global two-slot storage and the capacity sufficiency
of forced endpoints; learned unreserved selection was gated off by Step 5. Step 8
therefore records the full-task optimization diagnosis and formally defers recurrent
state learning and locked confirmation.

Measured laptop costs replaced the advance estimates. The fixed ten-seed batch used
3.57 total process-hours (21.4 minutes per seed), the variable batch 4.68 hours
(28.1 minutes per seed), and the selected full-mixture follow-up 1.22 hours (24.4
minutes per seed). Processes used about 0.30--0.36 GB each. The workload fits the
48 GB laptop; Della is useful only to reduce wall time for future locked seed arrays.

After these eight gates, Della access was configured under the `knorman` account,
with large artifacts below `/scratch/gpfs/KNORMAN/qlu`. A 12-task follow-up screen
confirmed the throughput benefit but also showed substantial node-speed variation:
eleven tasks completed in 42:43--54:50, while one unchanged seed required a
successful 91:37 retry after the original 90-minute limit proved too short. Peak
memory remained below 0.4 GB. The selected constant-rate, batch-32 cell now advances
to ten fresh seeds; this does not retroactively change the eight-step outcomes.
