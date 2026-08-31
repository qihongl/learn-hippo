# Progress ledger

- 2026-08-30: traced the published task and archived source, recorded the shared task
  contract, and exposed two paper/code differences in delay and feature removal.
- 2026-08-30: froze the data contract, success criteria, staged optimization,
  baselines, causal tests, and failure interpretations before implementation.
- 2026-08-30: added the modern exact-task generator and five deterministic behavior
  tests covering the 37-dimensional input, source profiles, RM/DM/NM situation
  relationships, and boundary-blind deterministic sampling.
- 2026-08-30: added the recurrent situation model, five-response prediction head,
  differentiable competitive episodic retrieval, and an encoding actor that receives
  only the recurrent situation model. Three model behavior tests pass.
- 2026-08-30: implemented complete `a1`–`b1`–`b2` rollouts, forced encoding
  schedules, exact expected task reward, RM/DM/NM working-memory resets, and
  profile-specific lure availability. Five rollout behavior tests pass.
- 2026-08-30: added supervised prediction/retrieval objectives and a reproducible
  training stage that can run with episodic memory absent or under a forced encoding
  schedule. A repeated exact trial is learnable in the training test.
- 2026-08-30: added delayed encoding optimization with a separate critic-only forced
  exploration stage and a free Bernoulli actor-critic stage. Tests verify that the
  forced stage leaves the actor unchanged and both stages freeze the complete
  prediction and retrieval system.
- 2026-08-30: the recurrent predictor failed the fixed-schedule precondition in an
  exploratory high-budget run. Added a boundary-blind structured situation model to
  isolate encoding-policy learning. On new DM trials its fixed endpoint schedule
  beats midpoint, midpoint-plus-endpoint, dense, and no-encoding alternatives when
  the same schedule controls both distracting and target events.
- 2026-08-30: corrected the free-policy objective so the learned policy controls both
  distracting and target events. A regression test verifies that both sampled action
  sequences produce the corresponding memory traces. The corrected discrete
  actor-critic learned higher endpoint than nonendpoint probability in one diagnostic
  seed, but encoded too densely and performed below the no-encoding baseline.
- 2026-08-30: added continuous differentiable encoding strengths and verified that
  delayed prediction reward has gradients to encoding in both events. This optimizer
  also failed, favoring the first observation. A fixed-schedule audit exposed the
  deeper cause: with unconditional retrieval on the full released training mixture,
  no encoding strongly outperforms endpoint encoding because irrelevant memories are
  damaging in NM trials.
- 2026-08-30: added content-based retrieval suppression without exposing the hidden
  memory condition. Conservative retrieval made endpoint-only the best declared
  fixed schedule, but a sparse shared actor again learned an early encoding band and
  no held-out DM benefit. This satisfies the stopping condition with a replicated
  task-identification and optimization diagnosis rather than a positive result.
- 2026-08-30: added versioned smoke and 10-seed diagnostic-replication
  configurations plus a runner that freezes each trained model, evaluates new
  mappings in RM/DM/NM, pairs retrieval-on and retrieval-off actions, retains all
  forced baselines, and writes provenance-rich machine-readable results.
- 2026-08-31: aggregated every declared seed, added forced-schedule and causal-memory
  ablations, and completed a self-contained paper-style HTML report with four
  reproducible explanatory and measured-data figures. The report preserves the
  negative policy result and separates it from successful memory use.
- 2026-08-31: audited the original 600-epoch supervised and 400-epoch A2C budget,
  documented which current learning curves can be reconstructed, and froze the
  user-approved follow-up contract, success criteria, convergence rules, capacity
  ladder, recurrent extension, and confirmatory split before new implementation.
- 2026-08-31: implemented deterministic fixed-bank checkpoint evaluation and
  complete per-update exposure records. Checkpoints retain epoch, update, sequences
  processed, contemporaneous training metrics, held-out schedule/ablation results,
  evaluation runtime, and optional weights; tests verify that evaluation does not
  change model parameters, mode, random streams, or the subsequent training result.
- 2026-08-31: implemented the Stage 1 temporal-hazard audit. Sixteen online
  first-encoding hazards are shared across the distracting and target events; a
  17-by-17 reward table enumerates each pair of encoding times plus never encoding.
  Unit tests verify the literal online distribution, reward gradients, endpoint
  rollout equality, endpoint learnability, and reproducible result serialization.
- 2026-08-31: completed all three declared temporal-audit seeds. On 64 unseen DM
  mappings per seed, mean endpoint encoding probability was 0.9979 and the endpoint
  gap was 0.9977. Mean learned reward was 0.6551, compared with 0.4896 for never and
  0.5145 for matched-random-one encoding. From update 800 to 1,000, endpoint
  probability changed by 0.0010 and reward by 0.00011, indicating practical
  convergence. The neural exact-state policy is therefore the next gate.
- 2026-08-31: implemented vectorized exact-reward optimization for the shared neural
  actor using only accumulated 80-dimensional situation states. A 16-mapping pilot
  showed partial held-out selectivity (endpoint 0.236 versus nonendpoint 0.029) but
  overfit its small bank. The declared development run therefore increases the bank
  to 256 mappings, evaluates 128 disjoint mappings, and supplies the full 102,400
  sampled-example exposures with held-out checks every 10 epochs.
- 2026-08-31: the declared neural exact-state development run succeeded on 128
  unseen DM mappings. At epoch 400, endpoint probability was 0.9872 (0.9896 in `a1`,
  0.9849 in `b1`) versus mean nonendpoint probability 0.00034. Learned reward was
  0.6611 versus 0.4896 for never and 0.5161 for matched random. Removing target
  memory reduced reward to 0.4869; removing distractor memory raised it to 0.6742.
  The last five checkpoints were stable, although an epoch-280 optimization
  excursion makes the learning path non-monotonic. This is the first positive basic
  feasibility result, not yet a multiseed or sampled-RL result.
- 2026-08-31: implemented the stronger sampled-credit stage. Each policy decision is
  made sequentially from the current situation state and stops after the first
  encoding in an event. Tests verify the one-encoding constraint, valid encoding-time
  distributions, actor freezing during forced value learning, actor updating during
  free learning, high seed safety, and reproducible checkpoint records.
- 2026-08-31: the declared 400-epoch sampled-reward run succeeded on 128 unseen DM
  mappings. Final endpoint probability was 0.9993 (0.9994 in `a1`, 0.9991 in `b1`)
  versus nonendpoint probability 0.000021. Reward was 0.6684 versus 0.4849 for never
  and 0.5129 for matched random; target-memory removal reduced it to 0.4831. The
  endpoint preference first exceeded 0.90 at epoch 30. Large temporary excursions at
  epochs 90 and 340 recovered, and the last five checkpoints all exceeded 0.99.
- 2026-08-31: generalized counterfactual evaluation to rectangular reward surfaces
  for unequal event delays and grouped time-resolved encoding probabilities by event
  length. The next declared fresh actor restores training-mode delays, feature
  removal, random penalties, and the full 0.25 RM / 0.25 DM / 0.50 NM mixture; held-
  out banks retain these manipulations separately in all three conditions.
- 2026-08-31: the full released-task run failed after its fixed 400 epochs. On held-
  out variable-delay DM trials, endpoint probability was 0.0027 versus nonendpoint
  0.0253; learned reward 0.5850 was below matched random 0.6092 and forced endpoint
  0.7040. RM and NM favored near-no encoding, and the actor learned an early delay-
  linked pattern. The DM endpoint gap was negative at every 10-epoch checkpoint and
  showed no favorable late trend, so the prespecified extension criterion was not
  met. Factorial DM-variable-delay and mixture-fixed-duration diagnostics are next.
- 2026-08-31: factorial diagnostics isolated the failure. DM-only training with the
  released delays and feature removal reached endpoint probability 0.893 and reward
  0.740 versus forced endpoint 0.743 after 100 epochs, although its curve remained
  unstable. Conversely, the full mixture with fixed 16-step events moved toward
  never encoding (endpoint 0.003, never 0.940). Thus variable timing is learnable;
  the RM/NM mixture is sufficient to overwhelm sampled boundary credit.
- 2026-08-31: extended exact counterfactual neural optimization to mixed variable-
  length examples and condition-separated evaluation. The declared bounded test uses
  the unchanged full training-mode mixture, 256 unique counterfactual examples, and
  100 epochs; it distinguishes sampling variance from a mixed-objective or neural-
  representation failure without adding any policy input.
- 2026-08-31: exact counterfactual credit also failed from scratch on the full
  variable-delay mixture. At epoch 100, DM endpoint probability was 0.0069 versus
  nonendpoint 0.0212; reward 0.626 was below matched random 0.637 and forced endpoint
  0.738. Endpoint selectivity was absent at all checkpoints. Sampling variance is
  therefore not the sole problem: the shared neural mixed objective has a strong
  nonboundary basin despite a better endpoint solution.
- 2026-08-31: added an explicitly labeled two-stage exact-reward curriculum. The
  shared actor first receives 100 epochs of DM-only experience, then 100 epochs of
  the unchanged full mixture. Tests verify that curriculum and mixture histories and
  held-out checkpoints remain separately serialized; no policy input or reward is
  altered.
- 2026-08-31: the DM-to-mixture exact curriculum failed the selectivity criterion.
  DM pretraining produced endpoint probabilities of only 0.26–0.37; after mixture
  continuation, endpoint probability was 0.0078 versus nonendpoint 0.0582. The final
  late-event policy improved DM reward to 0.659 over never 0.582 and random 0.601 but
  remained below forced endpoint 0.707. Adding a recurrent state is deferred because
  the simpler exact state has not solved the full mixed objective.
- 2026-08-31: rebuilt the self-contained paper-style HTML around the complete
  follow-up evidence. The report now defines the task and model from first
  principles, explains missing observations without calling them a boundary mask,
  distinguishes learned and forced endpoint policies, reports the original 600+400
  epoch reference, shows checkpointed learning curves, and separates positive
  DM-only feasibility from the negative full-mixture diagnosis. Four publication-
  style figures are generated directly from versioned compact summaries.
- 2026-08-31: executed five exact shared temporal-policy audits on the fixed-duration
  RM/DM/NM mixture. The endpoint was the best deterministic shared time in every
  held-out reward surface, but all five random starts converged one or two
  observations early. Mean endpoint probability was 0.00015. This separates the
  existence of an endpoint optimum from its accessibility under gradient
  optimization and motivates only bounded credit/initialization interventions.
- 2026-08-31: completed ten fresh 400-epoch fixed-duration DM replications. Every
  seed learned selective endpoint encoding and was stable over its last five held-
  out checkpoints. Mean endpoint probability was 0.999895 and mean reward exceeded
  never and matched-random encoding with paired bootstrap intervals above zero.
  Target-memory removal eliminated the benefit. The full predeclared success audit
  passed for this debugging condition.
- 2026-08-31: completed the paired eight-cell full-mixture credit screen. No cell
  passed both seeds. Low initialization plus condition-centered retrospective credit
  and gradual DM-to-mixture training was uniquely promising, reaching mean endpoint
  probability 0.7869 at epoch 100; its direct-mixture counterpart reached zero. The
  selected cell alone advances to the standard 400-epoch budget.
- 2026-08-31: tested a paired monotonic observable-progress policy using the selected
  credit and curriculum settings. Both seeds remained at endpoint probability 0.021
  for all 100 epochs. The constraint did not improve delayed boundary credit and is
  not tuned further.
- 2026-08-31: completed ten 400-epoch variable-duration DM replications. Mean
  endpoint probability was 0.7055 with large between-seed variability; seven seeds
  finished above 0.80 and six were stable over the last five checkpoints. All seeds
  gained reward from episodic memory, but the full policy success audit failed.
- 2026-08-31: completed the selected 400-epoch full-mixture follow-up on three fresh
  seeds. Two policies finished stably at endpoint probabilities 0.9979 and 0.9991.
  The third was at 0.9991 at epoch 390 but collapsed to effectively zero at epoch
  400. The final mean was 0.6657, so the predeclared audit failed despite a reliable
  prediction-reward benefit.
- 2026-08-31: verified that the underlying two-slot memory is global FIFO rather
  than event-specific. Forced endpoint evaluation shows that two slots suffice, but
  training a policy that can spend both within one event was gated on stable
  full-mixture learning and was therefore deferred.
- 2026-08-31: applied the recurrent-state gate. Because the exact situation-state
  model did not pass both the stable full-mixture and unreserved-capacity criteria,
  the 600-epoch recurrent pretraining and 20-seed confirmation were not run.
- 2026-08-31: rebuilt the paper-style report with multiseed learning curves,
  seed-level outcomes, the exact mixed-objective audit, the credit factorial, actual
  laptop runtimes, Della placement instructions, and explicit limitations.
