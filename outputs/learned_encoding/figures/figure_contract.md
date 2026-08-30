# Figure Contract

## Figure 1 — Task and computational flow

- **Core conclusion:** A delayed prediction can credit an earlier discrete encoding
  action because the selected event snapshot is later reinstated through
  differentiable episodic retrieval.
- **Archetype:** schematic-led composite.
- **Target:** self-contained HTML paper and full-width manuscript figure.
- **Final size:** 7.2 × 5.3 inches; minimum text 8 pt.
- **Panel map:** (a) stimulus/state representation, (b) study-reset-query sequence,
  (c) actor-critic and differentiable-memory flow.
- **Hero evidence:** complete endpoint versus incomplete midway trace.
- **Validation evidence:** no explicit boundary input; reward arrives after query.
- **Source:** task and model definitions in `src/boundary_em/`.
- **Reviewer risk:** a reader could mistake the observation mask for latent boundary
  discovery. The schematic must label it as an accumulated feature mask.

## Figure 2 — Policy emergence

- **Core conclusion:** Delayed actor-critic optimization changes an initially
  indiscriminate gate into an almost endpoint-exclusive encoding policy.
- **Archetype:** quantitative grid.
- **Target:** full-width manuscript figure.
- **Final size:** 7.2 × 3.0 inches; minimum text 8 pt.
- **Panel map:** (a) unsmoothed mean training reward with model-seed SD, (b) encoding
  probability by semantic progress with bootstrap 95% intervals.
- **Hero evidence:** probability jump at progress four.
- **Statistics:** 15 independent model seeds; raw 300-update curves and fixed-seed
  bootstrap intervals.
- **Reviewer risk:** training data are not held-out evidence; panel (a) must be
  labeled optimization dynamics, while panel (b) uses new mechanism episodes.

## Figure 3 — Confirmatory performance and causality

- **Core conclusion:** The learned policy approaches the forced endpoint-only policy
  and loses its benefit when its endpoint encoding is moved to the midpoint.
- **Archetype:** quantitative grid.
- **Target:** full-width manuscript figure.
- **Final size:** 7.2 × 3.2 inches; minimum text 8 pt.
- **Panel map:** (a) all model-seed rewards for learned and baseline policies, (b)
  paired deterministic learned versus displaced reward, (c) per-seed gap closure.
- **Hero evidence:** 15 paired learned-policy seeds and a forced displacement
  intervention.
- **Statistics:** individual model-seed points, model-seed mean, bootstrap 95% CI.
- **Reviewer risk:** baseline geometry is deliberately favorable to endpoint storage;
  the caption must state the task is controlled synthetic and theoretically audited.

## Figure 4 — Generalization and boundary conditions

- **Core conclusion:** The policy transfers to longer event duration and depends on
  the completion mask, while forced latest-memory retrieval removes the need for
  selective encoding.
- **Archetype:** quantitative grid.
- **Target:** full-width manuscript figure.
- **Final size:** 7.2 × 3.2 inches; minimum text 8 pt.
- **Panel map:** (a) paired primary versus OOD-duration reward, (b) policy-input
  ablations, (c) fixed-schedule reward over retrieval temperature plus latest-trace
  oracle annotation.
- **Hero evidence:** mask-only versus state/cue-only and latest-trace dense reward.
- **Statistics:** 15 model seeds with bootstrap 95% CI; retrieval sweep uses 512
  controlled episodes per temperature.
- **Reviewer risk:** post-confirmatory analyses are exploratory; every caption and
  panel annotation must say so.

## Shared visual policy

- Open Science palette, redundant marker/line encodings, single y-axis per panel.
- Vector PDF and SVG plus 300-dpi PNG preview; editable SVG text and embedded PDF
  fonts.
- White background; no 3-D, gradients, rainbow scales, or hidden axis truncation.
- Captions identify these as measured executions of a synthetic task, not human data.
