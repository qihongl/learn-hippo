# Decision log: exact prediction task

## 2026-08-30 — Scientific target

- Only the task must match the 2022 eLife simulation; the model architecture may be
  simplified.
- No new event-boundary cue, completion mask, countdown, or condition label will be
  given to the encoding policy.
- The encoding policy is learned during meta-training and frozen during evaluation
  on new situations.
- Success requires learned endpoint selectivity, prediction benefit, and a causal
  loss of that benefit when episodic retrieval is disabled, across at least 10 model
  seeds.
- Encoding-cost and capacity-limited variants are secondary diagnostics only.

## 2026-08-30 — Source discrepancy policy

- The released-code behavior is the primary executable profile because it is the
  best available record of what the simulations ran.
- The prose-defined 0–3 delay and independent 0.30 feature removal form a separate
  `paper_text` sensitivity profile.
- The profiles will not be pooled, and differences will be reported rather than
  silently reconciled.

## 2026-08-30 — Model and optimization scope

- The model uses a recurrent situation model, distinct episodic snapshots,
  differentiable competitive retrieval, and a Bernoulli encoding actor trained from
  delayed prediction reward.
- The discrete encoding action is trained with actor-critic gradients; only episodic
  retrieval is differentiable end to end.
- Predictor and retrieval learning precede encoding-policy learning. Gate-only free
  selection is the primary test; joint training is an explicitly labeled diagnostic.
- The primary faithful condition has no encoding cost and enough capacity to retain
  all eligible traces from two events. Historical and fixed-capacity conventions are
  reported as sensitivity analyses.

## 2026-08-30 — Prediction-model precondition and simplification

- An exploratory recurrent-model run did not satisfy the preregistered precondition:
  after 1,000 prediction updates and 1,000 forced-retrieval updates, endpoint-only
  encoding reached expected reward -0.635 on 128 new DM trials, compared with -0.625
  for dense encoding and -0.682 for no encoding. These are diagnostic, not final,
  results.
- The main obstacle was prediction and retrieval quality, not evidence about a learned
  encoding policy. We therefore introduced a structured situation model that directly
  accumulates the feature-value observations already present in the original
  37-dimensional input. It receives no boundary label, completion indicator,
  countdown, or condition label.
- This simplification isolates the stated question: whether delayed prediction reward
  can train an encoding policy when the prediction and retrieval system is already
  competent. The recurrent model remains a documented negative comparison.
- Before optimizing the encoding policy, fixed endpoint encoding was verified to beat
  fixed midpoint, midpoint-plus-endpoint, and dense encoding when each schedule was
  applied identically to distracting and target events on new DM trials.

## 2026-08-30 — Shared-policy correction

- The first exploratory free-policy run incorrectly optimized encoding only in the
  target event while forcing endpoint encoding in the distracting event. This made
  dense late target memories artificially useful and is not evidence about the
  proposed shared encoding policy.
- The corrected objective applies one shared encoding network independently to every
  time point in both events. Both sets of sampled actions receive the same delayed
  prediction outcome, and both distracting and target traces enter retrieval.
- With this correction, a 1,000-update single-seed diagnostic showed partial temporal
  sensitivity but not successful selective encoding: endpoint probability was 0.864,
  mean nonendpoint probability was 0.667, the model encoded 10.90 times per event,
  and held-out expected reward was 0.394 versus approximately 0.493 for no encoding.
  This run is a failure under the preregistered prediction-benefit criterion.

## 2026-08-30 — Fully differentiable encoding diagnostic

- To separate discrete-policy gradient variance from the scientific objective, each
  memory was given a continuous encoding strength. The strength enters retrieval as
  a differentiable prior, so delayed prediction reward can update the encoding
  network directly. No task input or boundary cue was added.
- A 500-update single-seed diagnostic also failed. It assigned its largest strength
  to the first observation, with endpoint strength 0.009 and mean nonendpoint
  strength 0.033 on new DM trials. Retrieval reduced expected reward from 0.488 when
  disabled to 0.426 when enabled.
- A gradient audit explained this local solution. Complete target memories become
  more useful toward the endpoint, but complete distracting memories become almost
  equally harmful. Because the shared policy sees statistically indistinguishable
  first presentations, their boundary gradients nearly cancel.
- More importantly, under 2,048 exploratory trials from the released meta-training
  mixture, fixed no encoding achieved expected reward 0.695, compared with -0.097
  for endpoint-only and 0.011 for dense encoding in the structured model. In the 50%
  NM condition, unconditional retrieval of irrelevant memories drove endpoint reward
  to -0.959. Therefore endpoint encoding is not globally optimal for a model that
  retrieves unconditionally; the original endpoint-versus-midpoint comparison does
  not establish otherwise.
- The next architectural check is content-based retrieval suppression. It may use
  similarity between the current partial event and stored situations, but not the
  latent RM/DM/NM condition. If this does not make endpoint encoding beneficial under
  the original mixture, the result will be treated as a task-identification failure.

## 2026-08-30 — Content-gated retrieval and stopping diagnosis

- A content-based retrieval gate now compares the partial current situation with
  stored situations. It receives no condition label. Exploratory fixed-schedule
  sweeps selected a conservative match threshold of 0.60, sharpness 30, and retrieval
  strength 0.20 before the final policy diagnostic.
- On 768 exploratory released-profile training trials, endpoint-only expected reward
  was 0.711, compared with 0.693 for midpoint-only, 0.698 for
  midpoint-plus-endpoint, 0.687 for dense, and 0.687 for never encoding. Thus the
  retriever can make the endpoint schedule useful without changing the task.
- A shared discrete actor initialized uniformly at 0.02 encoding probability was then
  trained for 1,500 free-selection updates after 100 critic-only updates. It converged
  from the opposite starting point to the same qualitative failure: endpoint
  probability 0.0015, nonendpoint probability about 0.079, and an early encoding band
  over observations 1–7. Held-out DM reward was 0.488, indistinguishable in practical
  terms from retrieval disabled/no encoding.
- The stopping diagnosis is therefore not merely high-variance policy gradients or an
  overly dense initialization. Under the exact task, a shared online policy receives
  no prospective signal that distinguishes a future target event from a distractor,
  and multiple nonboundary schedules lie in easier local optima. The published task
  supports endpoint superiority among selected fixed schedules, but does not uniquely
  identify endpoint encoding as the globally learnable unconstrained policy.

## 2026-08-31 — Approved training-budget-matched follow-up

- The user approved a new gated plan after auditing the original paper's 600
  supervised and 400 A2C epochs, with 256 event sequences per epoch.
- The earlier 600-update encoding run is retained as a limited-budget diagnostic.
  The new primary policy budget is 400 epochs, 102,400 sequences, and 6,400
  batch-level updates, with checkpointed held-out learning curves.
- Future relevance remains naturally unobservable and is treated as a scientific
  constraint, not a limitation to remove. No prospective target label will be added.
- One trace per event and counterfactual reward enumeration are debugging stages.
  The basic claim requires a neural actor using the exact accumulated state; the
  stronger result requires sampled online learning and two total trace slots.
- Only after exact-state success will a pretrained recurrent situation model replace
  exact feature recording. Exploratory and locked confirmatory seeds remain separate.

## 2026-08-31 — Checkpoint measurement contract implemented

- Fixed development trials can now be evaluated during free-policy training without
  advancing the training random-number stream or changing the model's parameters or
  training mode. The final update is always evaluated even when it is not an exact
  multiple of the checkpoint interval.
- A checkpoint row contains both its held-out results and the matching training row,
  plus epoch, update, total sequences processed, and measured evaluation runtime.
  Optional weight files are referenced by paths in the JSON and remain ignored by
  Git.
- Two requested independent reviews were launched but both agents lost their
  connections before returning findings. The same contract and code-quality checks
  were completed locally before the full test and lint gates.

## 2026-08-31 — Temporal audit uses the task's 16-step evaluation form

- Stage 1 uses the exact released-code evaluation generator: 16 observations,
  zero added delay, no removed observations, and a fixed error penalty of two. This
  makes the approved 16-hazard debugging model well-defined. It is not silently
  treated as the final variable-delay training task.
- The policy parameterizes the probability of first encoding at each time given that
  it has not encoded earlier. The seventeenth outcome is never encoding. Its initial
  distribution is uniform over these 17 outcomes, so no encoding time is favored.
- Completed trial outcomes are used only to build an exact delayed-reward objective;
  the optimizer receives no endpoint target. The same hazards govern both events.

## 2026-08-31 — Exact shared temporal objective passes

- Across the three declared exploratory seeds, the exact optimizer assigned a mean
  probability of 0.9979 to endpoint encoding on independent DM mappings. Learned
  expected reward exceeded both never and matched-random-one encoding in every seed.
- For every seed, endpoint was the best deterministic time when one time had to be
  shared across events. Without the shared-policy constraint, the best pair was
  never encoding in `a1` and endpoint encoding in `b1`. This is expected because the
  former is a distractor and the latter is later relevant; the policy never receives
  those identities prospectively.
- The positive result is limited to a 16-parameter temporal schedule, one trace per
  event, exact counterfactual credit, DM trials, and the zero-delay/no-removal
  evaluation form. It diagnoses a valid objective and optimizer. It does not yet show
  that a state-dependent neural policy can discover the rule.

## 2026-08-31 — Neural exact-state development bank

- The neural actor receives the accumulated 64 feature-value entries and 16 query-
  context entries already produced by the structured model. It receives no separate
  completion count, event identity, time, boundary, or future-relevance variable.
- A first 16-mapping, 1,000-update pilot achieved training reward 0.701 but only
  0.236 endpoint probability on 16 new mappings. This is recorded as a small-bank
  overfitting diagnostic, not evidence of success.
- The next declared run uses 256 unique training mappings and 128 fixed development
  mappings. Its 6,400 batches of 16 give the approved 102,400 example exposures, but
  the 256 exact counterfactual examples are reused. This is explicitly an
  architecture-development approximation; it does not satisfy the later requirement
  for 102,400 newly generated sampled-RL sequences.

## 2026-08-31 — Neural exact-state feasibility succeeds

- Seed 510 learned a shared observation-dependent encoding rule with no endpoint
  target: on 128 held-out mappings, endpoint probability was 0.9872 and nonendpoint
  probability was 0.00034. Both `a1` and `b1` separately expressed the preference.
- The learned expected reward exceeded never and matched-random-one encoding. Removing
  the target memory erased the advantage; removing the distracting memory improved
  reward. These interventions show that prediction benefit came from target memory,
  while the shared policy still incurs a small distractor cost.
- Held-out endpoint probability was above 0.88 for nearly all 10-epoch checkpoints
  but briefly fell to 0.669 at epoch 280 before recovering. It stabilized from epoch
  360 through 400, changing by less than 0.001. The result is therefore converged at
  the final checkpoint but the optimization path is not monotonic.
- This establishes the approved basic computational feasibility result only in the
  Stage 2 debugging condition. The next decisive comparison is a fresh actor trained
  from sampled delayed rewards; after that come variable delay/removal, the full
  RM/DM/NM mixture, and two total memory slots.

## 2026-08-31 — Sampled-credit development protocol

- A fresh neural actor starts at an encoding hazard of 0.05. It may encode once per
  event; after encoding, no further action is available in that event. This enforces
  the same one-trace-per-event debugging convention as the exact-reward comparison.
- The critic first receives 6,400 newly generated DM sequences under random forced
  choices. The free actor then receives 102,400 newly generated DM sequences over
  400 epochs. Only sampled delayed prediction rewards train the actor; exact
  counterfactual outcomes are used on a fixed held-out bank for measurement.
- The initial free-policy learning rate is 0.001, lower than the earlier 0.003
  discrete-policy diagnostic because the exact-reward run showed occasional large
  policy excursions. Entropy coefficient 0.001 preserves exploration without making
  dense repeated encoding possible.

## 2026-08-31 — Sampled delayed reward learns the boundary rule

- Seed 610 learned endpoint-selective encoding from sampled delayed prediction reward
  on fresh training mappings. No counterfactual reward table, endpoint target, or
  future relevance entered actor training. On new mappings, endpoint probability was
  0.9993 and the endpoint gap was 0.9992.
- Held-out reward exceeded never and matched-random-one encoding. Removing the target
  memory erased the benefit; removing the distractor improved it. The same policy
  encoded endpoints in both events despite their later roles being unobservable.
- The rule emerged rapidly (endpoint probability 0.912 at epoch 30), but the path had
  two large temporary failures at epochs 90 and 340. Both recovered under continued
  training, and epochs 360–400 all exceeded 0.99. The final result is stable by the
  prespecified five-checkpoint rule, but optimization robustness requires multiseed
  evaluation.
- This achieves the stronger Stage 2 goal in the simplified DM, zero-delay/no-removal,
  one-per-event condition. The next gate restores the released-code training delays,
  removed observations, and RM/DM/NM mixture while retaining the same online policy.

## 2026-08-31 — Full released-task gate

- The next fresh actor trains from scratch on the released 0.25 RM / 0.25 DM / 0.50
  NM mixture. Training now samples the released 0–4 delays and feature-removal rule;
  the policy still receives only its current accumulated situation state.
- Held-out evaluation uses 64 new training-mode trials in each condition, preserving
  variable and unequal event durations, removed features, and random penalties. A
  generalized rectangular counterfactual evaluator supports unequal `a1` and `b1`
  lengths without altering their online states.
- This remains the one-trace-per-event capacity gate. It asks whether the successful
  sampled learner survives the exact task distribution before removing the reserved
  event slots.

## 2026-08-31 — Full task fails through an early delay-linked policy

- Seed 710 did not learn endpoint encoding on the released full mixture. Held-out DM
  endpoint probability was 0.0027, lower than nonendpoint probability 0.0253, and its
  reward was below matched-random-one and forced endpoint encoding.
- The negative result is not explained by endpoint encoding lacking value in DM:
  forced endpoint reward was 0.7040 versus 0.5826 for never encoding. In RM, however,
  never encoding slightly beat endpoint encoding; NM was nearly indifferent. The
  weighted shared endpoint schedule remains better overall, but its informative DM
  advantage competes with more frequent conditions that reward using working memory
  or avoiding irrelevant episodic traces.
- The learned time distribution depends strongly on delay. With zero delay it mostly
  never encodes; with positive delays it places substantial probability on the first
  few steps and almost none on the endpoint. Thus restoring both delay cues and the
  condition mixture reintroduced a nonboundary local solution.
- DM endpoint selectivity was already absent at epoch 10 and did not show a favorable
  trend through epoch 400. The extension rule is not met. Two 100-epoch factorial
  diagnostics will vary delay/removal and condition mixture separately; they are
  diagnostic rather than rescue tuning.

## 2026-08-31 — Factorial diagnosis identifies the condition mixture

- Holding DM constant while restoring released delays and feature removal produced a
  strong boundary rule by epoch 100: endpoint probability 0.893 and reward within
  0.0021 of the forced-endpoint ceiling. The curve had a collapse at epoch 60 and had
  not met the five-checkpoint stability rule, so this is feasibility rather than a
  converged final result.
- Holding event duration fixed while restoring the full condition mixture failed.
  The actor's endpoint probability fell from about 0.02 to 0.003 and never-encoding
  probability reached 0.940, even though forced endpoint encoding improved DM reward
  by 0.194.
- Therefore variable timing and feature removal are not the principal obstacle. The
  sampled actor cannot reliably extract the sparse DM advantage from the more common
  RM/NM trials, where episodic encoding is unneeded or slightly harmful. The next
  bounded test uses exact counterfactual credit on the unchanged full mixture; a
  success would specifically diagnose sampling variance, while a failure would
  implicate the shared neural representation or mixed objective.

## 2026-08-31 — Exact mixed credit does not escape the nonboundary basin

- A fresh neural actor received exact probability-weighted rewards for all encoding-
  time pairs on 256 full-mixture training examples. It still converged to low endpoint
  probability: 0.0069 in held-out DM, below nonendpoint probability 0.0212.
- The forced endpoint schedule remained substantially better than never in DM, and
  its mixture-weighted reward was also better than never. Thus the failure is not an
  absent endpoint optimum. Removing action-sampling variance was insufficient to
  reach it from the boundary-neutral initialization.
- A DM curriculum is now the narrowest justified intervention. It changes the order
  of experience rather than policy inputs or task outcomes: first enter the verified
  boundary basin using DM, then continue optimization on the unchanged full mixture.
  Curriculum and from-scratch results will remain separate.

## 2026-08-31 — DM curriculum produces a late-event policy, not a boundary policy

- Exact DM curriculum training on variable-delay trials never reached the declared
  endpoint threshold. It preferred a distribution over several late observations,
  with held-out endpoint probability between 0.26 and 0.37 across checkpoints.
- Full-mixture continuation preserved useful late-event memories but reduced final
  endpoint probability to 0.0078. DM reward remained above never and matched random,
  showing that the model learned adaptive information selection, but not the proposed
  selective boundary-encoding rule.
- A recurrent situation model is therefore not the next rational implementation
  step: it adds a representation-learning problem while the exact observable state
  already exhibits the mixed-objective failure. The report will separate the positive
  DM-only feasibility result from this negative full-task diagnosis.

## 2026-08-31 — Reporting conclusion and next gate

- The report's headline claim is deliberately two-part: learned boundary encoding is
  computationally feasible in DM-only versions of the released task, but emergence
  from the complete RM/DM/NM mixture is not supported.
- A learned endpoint result means the free neural policy assigned its own probability
  to the boundary; forced endpoint schedules remain labeled as baselines throughout.
- Future relevance being unavailable at encoding remains a natural task constraint,
  not a limitation. No future-target or condition label will be added to rescue the
  mixed objective.
- Multiseed DM confirmation and a full-mixture temporal-policy audit precede any
  recurrent representation extension. This preserves the approved gate: do not add
  representation complexity while the exact observable state still fails the mixed
  objective.

## 2026-08-31 — Exact temporal mixture audit finds a late-event basin

- Five fresh exact-objective audits used the fixed 16-observation RM/DM/NM mixture,
  the original 0.25/0.25/0.50 weights, independent reward banks, and random
  perturbations of a time-uniform initial encoding distribution. The policy was
  shared across conditions and events and received neither condition nor boundary.
- In every seed, the endpoint was the best deterministic shared encoding time and
  exceeded never encoding on the mixture reward surface. Nevertheless, none of the
  five gradient runs reached it. Mean endpoint probability was 0.00015 and the mean
  endpoint gap was -0.0665.
- Three runs placed more than 0.994 probability on observation 15 of 16; two placed
  more than 0.994 on observation 14. Mean learned reward was 0.6570, below the
  deterministic endpoint value of 0.6653 but above never encoding at 0.6213.
- The exact mixed objective therefore contains the desired solution but does not
  make it a reliably reachable solution. The next intervention is restricted to
  initialization, retrospective credit assignment, and curriculum order. No policy
  input or task outcome will change.

## 2026-08-31 — Fixed-duration DM learning replicates across ten seeds

- Ten fresh actors received 25 forced-exploration epochs and 400 sampled free-policy
  epochs on newly generated fixed-duration DM mappings. No seed was removed and all
  checkpoint evaluations used unseen mappings with frozen weights.
- Mean endpoint probability was 0.999895 (SD 0.000034), and all ten seeds met the
  endpoint and endpoint-gap thresholds at each of their last five checkpoints. The
  bootstrap interval for the endpoint gap was [0.999875, 0.999915].
- Mean learned reward was 0.6616, versus 0.4870 for never and 0.5139 for matched
  random encoding. Paired bootstrap intervals were [0.1718, 0.1775] above never and
  [0.1455, 0.1498] above matched random. Target-memory removal reduced reward to
  0.4833, whereas distractor removal preserved the benefit.
- This passes every declared criterion for the one-trace-per-event, fixed-duration
  DM debugging condition. It does not resolve variable duration, the RM/DM/NM
  mixture, or unreserved memory capacity.

## 2026-08-31 — Bounded credit factorial identifies a promising curriculum

- Eight paired 100-epoch cells crossed low (0.05) versus neutral (0.50) initial
  encoding probability, the ordinary state critic versus a condition-centered
  retrospective reward baseline, and direct mixture versus gradual DM-to-mixture
  training. Conditions were used only to assign delayed credit after reward; the
  online policy still received no condition or future-relevance input.
- No cell passed both model seeds. The factorial is therefore a negative screen, not
  evidence that the full task is solved.
- The clearly best cell combined low initialization, condition-centered credit, and
  gradual mixture training. Its mean endpoint probability was 0.7869 at epoch 100
  (seed values 0.8593 and 0.7144), and its endpoint gap was 0.7760. The same credit
  method trained directly on the mixture reached endpoint probability 0.0000.
- The selected curve rose from mean endpoint probability 0.1509 at epoch 70 to
  0.8094 at epoch 90 and 0.7869 at epoch 100. Because the favorable trend occurs at
  the end of the screen, only this cell receives a fresh 400-epoch exploratory run;
  the other seven cells stop.

## 2026-08-31 — Observable progress is not a sufficient policy constraint

- A paired screen replaced the unconstrained state multilayer perceptron with a
  logistic policy whose encoding probability could only increase with the fraction
  of observed feature rows and query rows accumulated in the existing situation
  state. These quantities were computed from the state, not supplied as endpoint or
  time inputs.
- The screen used the same seeds, trials, low initialization, condition-centered
  retrospective credit, gradual mixture schedule, and 100-epoch budget as the best
  unconstrained factorial cell.
- Both seeds remained at endpoint probability 0.021 throughout all ten held-out
  checkpoints; the mean endpoint gap was -0.0116. The progress policy gained a small
  DM reward over never encoding but remained below matched random and forced
  endpoint encoding.
- This failure is retained without tuning the progress definition or slope
  initialization after observing the result. The unconstrained selected method
  remains the only full-mixture candidate receiving 400 epochs.

## 2026-08-31 — Variable-duration DM learning is feasible but unreliable

- Ten fresh actors received the same 25 forced-exploration and 400 free-policy
  epochs as the fixed-duration replication, with the released 0–4 delay and removed-
  observation sampling restored during both training and held-out evaluation.
- Mean endpoint probability was 0.7055 (SD 0.4346), below the declared 0.80
  threshold. Seven seeds finished above 0.80, six met the last-five-checkpoint
  stability rule, and three ended at 0.2111, 0.0464, or 0.0000. Mean endpoint
  probability was similar in the two events (0.7074 for `a1`, 0.7037 for `b1`).
- Mean reward was 0.7291 versus 0.6097 for never and 0.6343 for matched random.
  Paired bootstrap intervals for both advantages were above zero, and target-memory
  removal eliminated the benefit. Several failed endpoint seeds still used useful
  late-event memories.
- Variable duration therefore does not make boundary learning impossible, but the
  sampled optimizer is not reliable enough for the declared claim. This negative
  multiseed result replaces the earlier single-seed 100-epoch feasibility result.

## 2026-08-31 — Selected full-mixture method fails the stability gate

- The single selected factorial cell received the declared 400-epoch budget on
  three fresh seeds. No checkpoint or seed was selected for reporting.
- Seeds 961 and 962 finished with endpoint probabilities 0.9979 and 0.9991 and met
  the final-five-checkpoint rule. Seed 960 reached 0.9991 at epoch 390 but changed
  to a broad pre-endpoint policy during the last ten epochs; its endpoint
  probability at epoch 400 was effectively zero.
- The final mean endpoint probability was 0.6657 (SD 0.5765), below the 0.80
  criterion. Mean reward was 0.7209 versus 0.6074 for never and 0.6288 for random
  encoding, and target-memory removal erased the gain. Reward is therefore not a
  substitute for temporal selectivity.
- The full-mixture result remains negative. Additional constant-rate epochs are not
  authorized because the observed problem is catastrophic policy instability, not
  insufficient exposure along a favorable convergent curve.

## 2026-08-31 — Capacity and recurrent extensions are formally gated

- The memory implementation is a global two-slot FIFO store. A deterministic test
  verifies that multiple `a1` encodings compete with a later `b1` encoding; slots
  are not labeled or reserved by the store.
- The current sampled decision procedure nevertheless stops after the first
  encoding in each event, so the learned policy has not been tested with two fully
  unreserved choices. Forced endpoint, never, and random baselines already use the
  global two-slot store and confirm that two endpoint traces fit and improve DM
  reward.
- The approved plan required stable full-mixture learning before training the harder
  unreserved policy, and required both results before recurrent state learning.
  Because the selected method failed, the unreserved actor, recurrent pretraining,
  and locked 20-seed confirmation are deferred rather than misreported as failed.

## 2026-09-01 — Batch 32 advances from the optimizer-stability screen

- The predeclared four-cell screen paired seeds 970--972 across constant versus
  second-half cosine-decayed learning rates and batches of 16 versus 32. Every cell
  used 400 policy epochs and exactly 102,400 fresh training sequences.
- Only the constant-rate, batch-32 cell passed the strict rule in all three seeds.
  Final endpoint probabilities were 0.9929, 0.9969, and 0.9965, and all three met
  the final-five-checkpoint, event-specific, reward, and memory-removal criteria.
- The result is not treated as convergence: two selected-cell seeds had large
  earlier excursions, including a fall from 0.9405 to 0.0065 between epochs 230 and
  240. Cosine decay failed one seed at each batch size, apparently preserving a
  wrong basin rather than ensuring discovery.
- Per the rule committed before outcomes were opened, the batch-32 cell advances to
  ten fresh full-mixture seeds. No unreserved-memory or recurrent-model work begins
  before that replication passes.

## 2026-09-01 — Locked batch-32 replication fails

- All ten fresh seeds 980--989 completed the unchanged 400-epoch protocol. The
  aggregate endpoint mean was 0.7656 (SD 0.4084), below the declared 0.80 threshold,
  and six rather than eight seeds passed every individual criterion.
- Seeds 981 and 985 ended near zero, seed 982 ended at 0.6824, and seed 983 ended at
  0.9742 but did not remain selective through the final five checkpoints. Eight of
  ten runs had a post-epoch-200 endpoint drop greater than 0.08.
- Mean DM reward still exceeded never and matched-random encoding, and target-memory
  removal eliminated the benefit. The model therefore learned useful episodic
  information selection more reliably than boundary-selective timing.
- The optimizer line stops at the predeclared gate. Condition-label removal,
  unreserved memory, recurrent representation learning, and 20-seed confirmation do
  not advance. Any new objective requires a separately approved plan and new seeds.
