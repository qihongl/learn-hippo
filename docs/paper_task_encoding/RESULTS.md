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
The selected full-mixture method and the progress-policy screen failed. The memory
store has two global slots, but the decision procedure permits only one trace per
event, so unreserved-capacity training remains gated. Because the exact-state model
did not pass the stable full-mixture gate, the recurrent extension was formally
deferred. The result demonstrates basic computational feasibility, not a confirmed
full-task effect or a human mechanism. Complete methods, learning curves,
limitations, compute requirements, and next experiments are in
[`experiment_report.html`](experiment_report.html).
