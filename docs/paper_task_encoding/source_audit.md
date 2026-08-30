# Source audit: the 2022 event-prediction task

This note freezes what “the exact task” means before implementation. The primary
sources are the [eLife article](https://doi.org/10.7554/eLife.74445) and the
[archived released code](https://github.com/qihongl/learn-hippo/tree/6a4a1be4fd6780d4c8413ffc6b1facade4741135).
The task definition is separated from the original model architecture: the present
study preserves the task while replacing the imposed encoding schedule with a
learned encoding policy.

## Task contract shared by the paper and code

| Component | Frozen behavior |
|---|---|
| Event | One event contains 16 ordered queries about 16 situation features. Each queried feature has one of four values, giving four possible next states and chance prediction accuracy of 0.25. |
| Situation | A situation is the complete set of 16 feature values controlling how the event unfolds. A situation model is the model's internal representation of those values. |
| Observation | At each event time point, one situation feature identity and its value are observed. Feature observation order is independently randomized in each event. |
| Query | Query identities occur in the fixed order 0 through 15. Predicting the queried feature value is equivalent to predicting the next state. |
| Input | Observed feature identity (16-dimensional one-hot), observed value (4-dimensional one-hot), queried feature identity (16-dimensional one-hot), and scalar error penalty: 37 dimensions. |
| Output | Four specific next-state responses plus a fifth “don't know” response. Correct prediction earns +1, an error earns the negative penalty, and “don't know” earns 0. |
| Trial | The model processes a distractor event, `a1`, and two events, `b1` and `b2`, that share a situation in RM and DM. In NM the three events have unrelated situations. |
| RM | The situation model in working memory is retained between `b1` and `b2`; target and lure episodic memories are available. |
| DM | Working memory is reset between `b1` and `b2`; target and lure episodic memories are available. |
| NM | Working memory is reset and no relevant `b1` episodic memory is available during `b2`. |
| Evaluation | Neocortical/model weights are frozen. New situations are generated and new episodic memories may be formed within a test trial. |

The article states these task properties in the sections “Training environment,”
“Stimulus representation,” and “Model testing.” The released implementation is in
[`SequenceLearning.py`](https://github.com/qihongl/learn-hippo/blob/6a4a1be4fd6780d4c8413ffc6b1facade4741135/src/task/SequenceLearning.py),
[`StimSampler.py`](https://github.com/qihongl/learn-hippo/blob/6a4a1be4fd6780d4c8413ffc6b1facade4741135/src/task/StimSampler.py), and
[`exp_tz.py`](https://github.com/qihongl/learn-hippo/blob/6a4a1be4fd6780d4c8413ffc6b1facade4741135/src/exp_tz.py).

## Original encoding comparison

The 2022 model did **not** learn an encoding policy. The experimenter imposed:

- endpoint-only encoding: one episodic snapshot at the final time point of each
  event; and
- midpoint-plus-endpoint encoding: snapshots midway through and at the end of each
  event.

The endpoint-only condition performed better because incomplete midpoint memories
competed with more informative endpoint memories during retrieval. The released
evaluation used the historical capacity convention: two slots for endpoint-only
encoding and four slots for midpoint-plus-endpoint encoding, retaining two events in
both cases. This follows `n_event_remember * n_segments` in
[`params.py`](https://github.com/qihongl/learn-hippo/blob/6a4a1be4fd6780d4c8413ffc6b1facade4741135/src/utils/params.py).

The new model will receive an encoding opportunity at every time point in `a1` and
`b1`. It will receive no added boundary label, countdown, or completion mask. The
fixed query order and recurrent history remain available because they are part of
the original task. Forced endpoint and midpoint schedules are baselines, not learned
models.

## Published-description versus released-code differences

Two task details disagree. They will be represented as named configuration profiles
rather than silently reconciled.

| Detail | Published description | Released code | Decision |
|---|---|---|---|
| Training delay | Prediction queries delayed by a uniformly sampled 0–3 time points. | With 16 features, `max_pad_len = n_param // 3 - 1 = 4`; inclusive integer sampling therefore gives 0–4. | `released_code` is the primary executable profile; `paper_text` with 0–3 is a sensitivity analysis. |
| Missing observations in `b1` | Each observation independently has 0.30 probability of removal, so 70% are expected to remain. | With `n_rm_fixed=False`, the code samples and rounds one uniform removal count between 0 and 4.8. This removes about 2.40 of 16 observations on average, or about 15%, and does not use independent Bernoulli removal. | Preserve the released behavior in `released_code`; implement independent 0.30 removal in `paper_text`. |
| NM traces | Endpoint traces are formed for all three unrelated events, so `b1` is another lure for `b2`. | Encoding is disabled for the current `b1` whenever the sampled condition is NM; an older unrelated `a1` trace remains. | Preserve one lure in `released_code` and both unrelated traces in `paper_text`; neither profile contains a relevant target. |

There are also training-description differences that do not change the task itself:
the paper describes cross-entropy supervised initialization, whereas the released
training loop uses mean-squared error on the four specific-response probabilities;
and the paper says every training epoch includes all three conditions, whereas the
code samples RM/DM/NM with probabilities 0.25/0.25/0.50 across 256 events. These
differences will be documented in configurations. The new staged optimizer is not
claimed to reproduce the original optimizer.

## Frozen original settings

- Main task: `T = 16`, `B = 4`, random observation order, fixed query order.
- Uniform schema condition: `def_prob = 0.25`, `n_def_tps = 0`.
- Training penalties: sampled from 0 through 4; primary test penalty: 2.
- Training conditions: RM/DM/NM probabilities 0.25/0.25/0.50 in the released code.
- Original model reference values, used only for context: LSTM width 194, decision
  width 128, cosine/LCA competition 0.8, Adam learning rate `7e-4`, 600 supervised
  epochs followed by 400 A2C epochs, 256 events per epoch.
- Simulation 2 model count and evaluation convention: the released analysis uses 15
  model seeds and 256 test examples per condition.

The present study may use a smaller network or a staged optimizer because the user
specified that the task, rather than the original architecture, must be exact. Every
such change will be named in the experiment design and cannot be described as an
exact reproduction of the 2022 model.
