# Results: learning an encoding policy on the 2022 prediction task

## Outcome

The basic fixed-duration feasibility result is positive across ten fresh model
seeds. A neural policy using the exact accumulated situation state learned selective
event-boundary encoding from sampled delayed prediction reward on newly generated
DM sequences. Mean endpoint probability was 0.999895 and all ten seeds met the
five-checkpoint stability rule. Mean reward was 0.6616 versus 0.4870 for never
encoding and 0.5139 for one random encoding. Removing the target memory reduced
reward to 0.4833. Bootstrap intervals for the endpoint gap and both reward
advantages were above zero. The endpoint was not forced or supplied as a target.

The ten-seed variable-delay, missing-observation DM replication did not pass the
endpoint criterion. Mean endpoint probability was 0.7055 (SD 0.4346): seven seeds
finished above 0.80, but only six met the five-checkpoint stability rule, while three
ended at 0.211, 0.046, or 0.000. Mean reward was nevertheless 0.7291 versus 0.6097
for never and 0.6343 for matched random encoding, with intervals above zero. Thus
variable timing permits boundary learning but does not yield reliable discovery.

The stronger full-task result is negative. From-scratch sampled learning on the
released 0.25 RM / 0.25 DM / 0.50 NM mixture produced endpoint probability 0.0027,
below nonendpoint probability 0.0253. Exact counterfactual credit and a DM
curriculum also produced broad late-event policies. In five exact shared-temporal
audits, the endpoint was the best deterministic time but every gradient run stopped
one or two steps early. Forced endpoint encoding remained valuable in DM, so this is
a discovery and stability failure rather than an absent endpoint advantage.

A predeclared two-seed, eight-cell credit screen found one promising but unstable
intervention. Low initial encoding, a condition-centered retrospective
reward baseline, and gradual DM-to-mixture training reached mean endpoint
probability 0.7869 at epoch 100 (individual seeds 0.8593 and 0.7144). Its matched
direct-mixture cell reached zero. In the selected 400-epoch follow-up, two fresh
seeds ended at 0.9979 and 0.9991 and were stable, but a third changed from 0.9991 at
epoch 390 to effectively zero at epoch 400. The three-seed mean was 0.6657 (SD
0.5765), so the predeclared audit failed. Mean reward nevertheless exceeded never
and random encoding, showing again that prediction performance alone does not
establish a boundary policy.

A subsequent predeclared optimizer-stability screen crossed constant versus
second-half cosine-decayed learning rates with batches of 16 versus 32 while
holding 400 epochs and 102,400 sequences fixed. Only constant rate with batch 32
passed all three paired seeds. Final endpoint probabilities were 0.9929, 0.9969,
and 0.9965; mean DM reward was 0.7328 versus 0.6072 for never and 0.6330 for random
encoding, and target-memory removal reduced it to 0.6010. This is a positive
three-seed development result, not confirmation. Two selected-cell trajectories
had large earlier excursions, and each cosine cell retained a wrong solution in
one seed. The selected cell therefore advanced unchanged to ten fresh seeds.

That locked replication failed. Across fresh seeds 980--989, mean endpoint
probability was 0.7656 (SD 0.4084), below 0.80, and only six seeds passed the full
individual rule rather than the required eight. Two seeds ended near zero, one at
0.6824, and one at 0.9742 but failed the final-five stability rule. Mean DM reward
was nevertheless 0.7261 versus 0.6031 for never and 0.6274 for random encoding;
target-memory removal reduced it to 0.6013. The learner reliably acquired useful
episodic information selection, but not reliable event-boundary selectivity.

Constraining encoding probability to increase with accumulated observed-feature and
query progress did not help. With the otherwise selected credit and curriculum
settings, both paired seeds remained at endpoint probability 0.021 throughout the
100-epoch screen. Observable progress is therefore not a sufficient replacement for
the unconstrained situation-state policy.

## Diagnosis and scope

Variable timing and missing observations are not an absolute obstacle: most DM-only
seeds learned with both, but multiseed reliability was poor. The full condition
mixture poses a stronger problem, even with fixed duration. RM and NM comprise 75%
of training and make episodic encoding unnecessary or slightly harmful, while the
useful DM signal is sparse and delayed. Removing sampling variance does not by
itself escape the nonboundary solution.

These are exploratory synthetic simulations. The fixed- and variable-duration DM
results use ten seeds each; only the fixed-duration condition passes all criteria.
The original selected full-mixture follow-up and the progress-policy screen failed.
The batch-32 screen passed three development seeds but failed its locked ten-seed
replication and still showed nonmonotonic trajectories. The memory store has two
global slots, but the decision procedure permits only one trace per event;
unreserved-capacity and recurrent training remain gated. The result demonstrates
basic fixed-duration DM feasibility, not a reliable full-task effect or a human
mechanism. Complete methods, learning curves, limitations, compute requirements,
and next decisions are in
[`experiment_report.html`](experiment_report.html).
