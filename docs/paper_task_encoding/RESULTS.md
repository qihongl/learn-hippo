# Results: learned encoding on the exact 2022 prediction task

## Outcome

The main success criterion failed. Across 10 retained model seeds, learned
endpoint-minus-nonendpoint encoding probability was -0.00030, with 95% bootstrap
interval [-0.00092, 0.00026]; only 4 of 10 seeds had a positive difference. The
model learned sparse, nearly time-uniform encoding rather than selective encoding at
the event boundary.

The episodic pathway was nevertheless functional. On unseen distant-memory trials,
learned-policy expected reward was 0.5013 versus 0.4881 for never encoding, a paired
benefit of 0.01319 [0.01023, 0.01580]. Disabling retrieval removed exactly that
benefit. Removing target memories reduced reward to 0.4880, and disabling the
content-match retrieval gate reduced it to 0.3789.

Forced endpoint-only encoding achieved 0.6623 in distant memory, exceeding
midpoint-only (0.5164), midpoint-plus-endpoint (0.5483), dense (0.5483), matched
random one-encoding (0.5138), and never encoding (0.4881). These policies were
imposed by the experimenter; they are a task and model precondition, not evidence
that the actor learned a boundary policy.

## Diagnosis

The exact task makes complete endpoint memories useful but does not make the
endpoint policy reliably discoverable from delayed reward. The shared policy sees no
prospective label distinguishing a future target from a distractor at first
presentation. Multiple sparse, early, and dense-late policies form easier solutions.
Discrete dense initialization, continuous differentiable encoding strengths, and
sparse discrete initialization all failed to produce selective endpoint encoding.

The reported run is a diagnostic replication, not a preregistered confirmation,
because the structured situation representation and conservative content gate were
selected after exploratory failures. Complete methods, limitations, figures, and
next steps are in [`experiment_report.html`](experiment_report.html).
