# Results: learning an encoding policy on the 2022 prediction task

## Outcome

The basic feasibility result is positive. A neural policy using the exact
accumulated situation state learned selective event-boundary encoding from sampled
delayed prediction reward on newly generated DM sequences. On 128 unseen mappings,
endpoint probability was 0.9993, mean probability per earlier time was 0.00002, and
reward was 0.6684 versus 0.4849 for never encoding and 0.5129 for one random
encoding. Removing the target memory reduced reward to 0.4831. The endpoint was not
forced or supplied as a target.

A variable-delay, missing-observation DM diagnostic also reached endpoint
probability 0.8928 and reward 0.7404, close to the forced-endpoint value of 0.7425,
but it had not met the five-checkpoint convergence rule.

The stronger full-task result is negative. From-scratch sampled learning on the
released 0.25 RM / 0.25 DM / 0.50 NM mixture produced endpoint probability 0.0027,
below nonendpoint probability 0.0253. Exact counterfactual credit also failed
(0.0069 versus 0.0212), and a DM curriculum produced a broad late-event policy.
Forced endpoint encoding remained valuable in DM, so this is a
discovery/local-optimum failure rather than an absent endpoint advantage.

## Diagnosis and scope

Variable timing and missing observations are not the principal obstacle: DM-only
learning succeeded with both. The full condition mixture is sufficient to block
discovery, even with fixed duration. RM and NM comprise 75% of training and make
episodic encoding unnecessary or slightly harmful, while the useful DM signal is
sparse and delayed. Removing sampling variance does not by itself escape the
nonboundary solution.

These are exploratory synthetic simulations. The decisive positive and negative
neural runs currently use one seed each, the situation record is exact, and one
memory slot is reserved for each earlier event. The result demonstrates basic
computational feasibility, not a confirmed full-task effect or a human mechanism.
Complete methods, learning curves, limitations, and next experiments are in
[`experiment_report.html`](experiment_report.html).
