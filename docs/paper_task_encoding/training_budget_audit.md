# Training-budget audit

## Original eLife model

Lu, Hasson, and Norman (2022) used **1,000 meta-training epochs**: **600
supervised-pretraining epochs**, followed by **400 advantage actor-critic (A2C)
epochs**. Each epoch contained **256 newly generated event sequences**. Thus, one
model saw 153,600 supervised examples and 102,400 A2C examples, or 256,000
examples in total. The paper says that 1,000 epochs was chosen to ensure that the
learning curves converged; it does not report a formal stopping rule. During
supervised pretraining, only the recent-memory (RM) condition was used, episodic
memory and the “don’t know” response were disabled, and the network learned
next-state prediction. During A2C, all weights could change. Both phases used Adam
with an initial learning rate of 7e-4. See [Methods, PDF p. 21](https://elifesciences.org/articles/74445.pdf#page=21)
and [Appendix 7, PDF p. 43](https://elifesciences.org/articles/74445.pdf#page=43).

The released code confirms and sharpens this description:

- The reported job sets `sup_epoch=600` and `n_epoch=1000`
  ([`src/submit-vary-test-penalty.sh`, lines 8–14](https://github.com/qihongl/learn-hippo/blob/6a4a1be4fd6780d4c8413ffc6b1facade4741135/src/submit-vary-test-penalty.sh#L8-L14)).
- `n_examples` defaults to 256, and epochs 0–599 use supervised learning while
  epochs 600–999 use A2C
  ([`src/train-sl.py`, lines 50–52 and 159–172](https://github.com/qihongl/learn-hippo/blob/6a4a1be4fd6780d4c8413ffc6b1facade4741135/src/train-sl.py#L50-L52)).
- The program loops over the 256 event sequences and calls `optimizer.step()`
  after each sequence. One original-paper epoch therefore contains 256 parameter
  updates, not one update on a batch of 256
  ([`src/exp_tz.py`, lines 19–33 and 116–140](https://github.com/qihongl/learn-hippo/blob/6a4a1be4fd6780d4c8413ffc6b1facade4741135/src/exp_tz.py#L19-L33)).
- Supervised examples are RM only; A2C samples RM, distant-memory (DM), and
  no-memory (NM) with probabilities 0.25, 0.25, and 0.50
  ([`src/exp_tz.py`, lines 184–193](https://github.com/qihongl/learn-hippo/blob/6a4a1be4fd6780d4c8413ffc6b1facade4741135/src/exp_tz.py#L184-L193),
  [`src/utils/constants.py`, lines 15–18](https://github.com/qihongl/learn-hippo/blob/6a4a1be4fd6780d4c8413ffc6b1facade4741135/src/utils/constants.py#L15-L18)).

One detail differs between prose and code: the paper describes cross-entropy
pretraining, whereas the release applies mean-squared error to the non-“don’t
know” prediction probabilities and one-hot targets
([`src/exp_tz.py`, lines 97–99](https://github.com/qihongl/learn-hippo/blob/6a4a1be4fd6780d4c8413ffc6b1facade4741135/src/exp_tz.py#L97-L99)).

This budget is only an informative reference for the new model. In the original
selective-encoding simulation, encoding time was imposed by `enc_size`; the
trained model was re-evaluated with endpoint-only versus
midpoint-plus-endpoint schedules. The network did **not** learn when to encode
([README, lines 287–299](https://github.com/qihongl/learn-hippo/blob/6a4a1be4fd6780d4c8413ffc6b1facade4741135/README.md#L287-L299),
[`src/exp_tz.py`, lines 177–200](https://github.com/qihongl/learn-hippo/blob/6a4a1be4fd6780d4c8413ffc6b1facade4741135/src/exp_tz.py#L177-L200)).

## Comparison with the present diagnostic run

For each seed, the present run used 50 forced-exploration critic updates with 16
trials per update, followed by 600 free-selection actor-critic updates with 16
trials per update. This is 10,400 training trials and 650 parameter updates per
seed. The condition mixture matches the paper's 0.25 RM / 0.25 DM / 0.50 NM
mixture. The settings are in
[`configs/paper_task_encoding/reported_failure.yaml`](../../configs/paper_task_encoding/reported_failure.yaml).

The present model therefore received about 24.6 times fewer event sequences than
the original model overall. Considering only the policy phase, free selection
received 9,600 trials versus 102,400 A2C event sequences in the paper (10.7 times
fewer). Optimization steps are less comparable because the original code updated
after every sequence, whereas the present code averages 16 trials before each
update. The phases also train different mechanisms: the original supervised
phase learned the recurrent prediction representation, whereas present forced
exploration trains only the value estimator for the encoding actor.

## Can the current learning curves be reconstructed?

**Yes for training; no for a held-out learning curve.** Each of the ten
seed-specific JSON files under `outputs/paper_task_encoding/reported/` retains all
50 forced-value records and all 600 free-selection records. Every record contains
training-batch reward, actor and critic losses, entropy, gradient norm, encodings
per event, and endpoint and nonendpoint encoding probabilities. These data are
sufficient to plot per-update training curves with seed-level uncertainty. The
first 100 versus last 100 free-selection updates have almost identical mean
training reward across seeds (0.6895 versus 0.6884), although this noisy on-policy
quantity is not by itself a convergence test.

Only the final frozen model was evaluated on held-out trials: 256 trials for each
of RM, DM, and NM. The files contain final aggregate statistics but no
intermediate held-out evaluations, per-trial evaluation values, or model
checkpoints. A retrospective figure can therefore show the training curves and
one final held-out point, but it cannot reconstruct held-out reward or endpoint
selectivity across training. The next run should evaluate a fixed held-out set at
prespecified checkpoints and retain those metrics—and preferably checkpoint
weights—so convergence can be judged independently of the training batches.
