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

A variable-delay, missing-observation DM diagnostic also reached endpoint
probability 0.8928 and reward 0.7404, close to the forced-endpoint value of 0.7425,
but it had not met the five-checkpoint convergence rule.

The stronger full-task result is negative. From-scratch sampled learning on the
released 0.25 RM / 0.25 DM / 0.50 NM mixture produced endpoint probability 0.0027,
below nonendpoint probability 0.0253. Exact counterfactual credit also failed
(0.0069 versus 0.0212), and a DM curriculum produced a broad late-event policy.
Forced endpoint encoding remained valuable in DM, so this is a
discovery/local-optimum failure rather than an absent endpoint advantage.

A predeclared two-seed, eight-cell credit screen found one promising but not yet
successful intervention. Low initial encoding, a condition-centered retrospective
reward baseline, and gradual DM-to-mixture training reached mean endpoint
probability 0.7869 at epoch 100 (individual seeds 0.8593 and 0.7144). The matched
direct-mixture cell reached zero. No cell passed both seeds, so the screen is not a
positive full-task result; only the selected cell proceeds to the fixed 400-epoch
budget.

## Diagnosis and scope

Variable timing and missing observations are not the principal obstacle: DM-only
learning succeeded with both. The full condition mixture is sufficient to block
discovery, even with fixed duration. RM and NM comprise 75% of training and make
episodic encoding unnecessary or slightly harmful, while the useful DM signal is
sparse and delayed. Removing sampling variance does not by itself escape the
nonboundary solution.

These are exploratory synthetic simulations. The fixed-duration DM result now uses
ten seeds, but the variable-DM replication and selected 400-epoch full-mixture
intervention remain in progress. The situation record is exact and one memory slot
is reserved for each earlier event. The result demonstrates basic computational
feasibility, not a confirmed full-task effect or a human mechanism. Complete
methods, learning curves, limitations, and next experiments are in
[`experiment_report.html`](experiment_report.html).
