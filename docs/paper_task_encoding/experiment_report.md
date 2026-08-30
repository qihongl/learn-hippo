# Learning to Encode at Event Boundaries on the Original Event-Prediction Task

## A differentiable episodic-memory diagnostic of Lu, Hasson, and Norman (2022)

> **Report status:** complete diagnostic replication with measured synthetic
> simulations, 2026-08-30
>
> **Implementation and report:** Codex, with scientific direction from Qihong Lu
>
> **Scope:** the task follows the released code for the prediction simulation in
> Lu, Hasson, and Norman (2022); the model architecture is new.
>
> **Review notice:** human review by a domain expert is strongly recommended before
> scientific publication. The numbers below are measurements from synthetic model
> simulations, not human behavioral data.

## Abstract

Lu, Hasson, and Norman (2022) argued that selectively **encoding** an episodic
memory at an **event boundary** can improve later event prediction because an
endpoint memory contains a more complete situation model and avoids competition
from incomplete memories. Their simulations imposed the encoding times; the model
did not learn them. We tested whether delayed prediction reward can teach this
policy on the same event-prediction task. Each event contains 16 situation features,
each with four possible values. The model observes the features in random order,
predicts values in a fixed query order, and can encode its current situation model
at every time point. No boundary flag, countdown, completion mask, or memory-condition
label is added. A differentiable, content-addressable episodic memory retrieves a
competitive mixture of encoded situation models, while a shared Bernoulli policy
learns discrete encode-or-skip decisions with actor--critic optimization. Forced
endpoint-only encoding was a strong solution in held-out distant-memory trials
(expected reward 0.662 versus 0.488 for never encoding), so the architecture made
boundary memory useful. Nevertheless, across 10 independently trained models, the
learned endpoint-minus-nonendpoint encoding probability was -0.00030, with 95%
bootstrap interval [-0.00092, 0.00026]; only 4 of 10 models had a positive gap.
The learned policy did use episodic memory: it improved distant-memory reward by
0.0132 [0.0102, 0.0158], and disabling retrieval removed that benefit. The result is
therefore specific: optimization learned a sparse, nearly time-uniform encoding
policy, not selective boundary encoding. The exact task demonstrates the value of a
forced boundary policy but does not, under this model and learning signal, make that
policy reliably discoverable.

## 1. Question and relation to the original paper

An **event** is a temporally extended sequence in which a stable latent situation
determines what states will occur. A **situation** is the collection of feature
values that characterizes the current event. A **situation model** is the model's
currently available representation of those values. **Working memory** carries the
situation model through ongoing computation, whereas **episodic memory** stores a
snapshot that can persist after working memory is reset. We use **encoding** for
creating such a snapshot and **retrieval** for reinstating stored information. These
terms follow the [eLife paper](https://doi.org/10.7554/eLife.74445); we avoid the
computer-science shorthand “write” and “read.”

The original simulation asked the model to predict the next state of an event. It
learned when to retrieve episodic information, but the experimenter specified when
episodic memories were encoded. The critical comparison imposed either one encoding
at each event endpoint or two encodings, one at the midpoint and one at the endpoint.
Endpoint-only encoding performed better because the incomplete midpoint memory
could compete with the complete endpoint memory during retrieval. That result showed
that the endpoint policy can be advantageous. It did **not** show that delayed
prediction reward can teach an unconstrained model to discover the policy.

The present question is narrower and causal: if the same task gives a model an
encoding opportunity at every time point, will optimization assign higher encoding
probability to the endpoint than to nonendpoint time points? A positive answer
requires three observations on unseen situations: endpoint probability must exceed
nonendpoint probability, the learned memories must improve prediction over never
encoding, and disabling retrieval must remove that prediction benefit.

## 2. The exact prediction task

### 2.1 Stimuli and predictions

Each situation contains 16 categorical features, denoted $F_1,\ldots,F_{16}$.
Every feature has one of four values, represented by an integer from 0 to 3. At each
time point, the original model input is

\[
u_t=[o_t;v_t;q_t;\lambda_t] \in \mathbb{R}^{37},
\]

where $o_t$ is a 16-unit one-hot vector naming the observed feature, $v_t$ is a
four-unit one-hot vector giving that feature's observed value, $q_t$ is a 16-unit
one-hot vector naming the feature whose value must be predicted, and $\lambda_t$
is the penalty for an incorrect specific prediction. “One-hot” means that exactly
one unit is 1 and the others are 0. The output has five responses: the four possible
feature values and a **don't-know response**, which declines to make a specific
prediction.

![](../../outputs/paper_task_encoding/figures/fig_01_task.svg)

**Figure 1. Exact task representation.** **a**, A situation is a mapping from 16
features to four possible values. The 37-dimensional model input is the concatenation
of the observed feature, observed value, queried feature, and error penalty. The
displayed mapping is an explanatory schematic. **b**, A trial contains a distracting
event $a1$, a first target event $b1$, and a prediction event $b2$. The diagram
defines the three memory conditions used below.

The 16 features are observed once each in a new random order. Queries follow the
fixed order $F_1,\ldots,F_{16}$, shifted by a delay. In the released implementation,
the meta-training delay is sampled from 0 through 4; held-out evaluation uses delay
0. The endpoint is the final time point, after all observations and delayed queries
have occurred. Crucially, the model receives no extra endpoint marker. There is also
no “mask” in this task: nothing explicitly states that all features are complete.
The encoding network can only use the situation representation accumulated from the
original input history.

For a queried feature with correct value $y_t$, a specific correct response earns
+1, a specific incorrect response earns $-\lambda_t$, and don't know earns 0. If
the model's response probabilities are $\pi_t$, its differentiable expected reward
is

\[
r_t=\pi_t(y_t)-\lambda_t\sum_{k\neq y_t,\,k\leq 4}\pi_t(k).
\]

The don't-know probability does not appear in the equation because its reward is
zero. Training can therefore favor a calibrated don't-know response when errors are
costly.

### 2.2 Trial sequence and memory conditions

Every trial contains three events. The first, $a1$, is a distracting situation.
The second, $b1$, is the first presentation of a target situation. The third,
$b2$, supplies the predictions used for delayed reward. The conditions are:

::: {.concept-table}
| Condition | Situation relationship | Working memory before $b2$ | Relevant episodic memory |
|---|---|---|---|
| **RM: recent memory** | $b1=b2$ | retained | available but usually unnecessary |
| **DM: distant memory** | $b1=b2$ | reset | the $b1$ trace can restore missing information |
| **NM: no relevant memory** | $a1$, $b1$, and $b2$ are unrelated | reset | unavailable under the released-code convention |
:::

Meta-training uses RM, DM, and NM in proportions 0.25, 0.25, and 0.50. New
situations and observation orders are sampled for every trial. Evaluation uses new
mappings and freezes all model weights.

### 2.3 Released-code profile versus paper prose

We treated the archived released code as the primary executable specification. It
differs from the paper prose in two small but material ways. First, the code samples
delay 0--4, whereas the paper reports 0--3. Second, the paper describes independent
removal of 30% of observations during training, whereas the code samples a rounded
number between 0 and 4.8 observations, removing about 2.4 of 16 on average. We did
not silently reconcile these definitions; the primary runs use the code behavior.
The complete audit is in
[`source_audit.md`](source_audit.md).

## 3. Model and computational flow

### 3.1 Structured situation model

The task is exact, but the model is not the 2022 LSTM. An initial LSTM implementation
failed a required precondition: after 1,000 prediction updates and 1,000 forced
retrieval updates, forced endpoint-only reward was -0.635, compared with -0.625 for
dense encoding and -0.682 for never encoding on 128 new DM trials. Because the
predictor and retriever were not yet competent enough to isolate encoding-policy
learning, we replaced the LSTM with a **structured situation model**. This model
accumulates the feature--value relations already present in the original 37-unit
input; it does not receive a boundary label.

Let $S_t\in\{0,1\}^{16\times4}$ be the accumulated feature--value table. It is
updated by

\[
S_t=\max\left(S_{t-1},\;o_tv_t^\top\right),
\]

where $o_tv_t^\top$ is the outer product placing the current observed value in the
row of the observed feature, and the maximum is elementwise. A 16-unit query-history
vector accumulates which original queries have occurred. Flattening $S_t$ and
concatenating that history gives an 80-dimensional situation state $x_t$. The
history makes event progress inferable from the original sequence, but it does not
directly label the final time point.

![](../../outputs/paper_task_encoding/figures/fig_02_model.svg)

**Figure 2. Model and optimization.** **a**, The encoding actor receives the
structured situation model, chooses encode or skip, and stores selected snapshots.
During $b2$, the partial current situation cues competitive retrieval; retrieved
content modifies the five-way prediction. No boundary, countdown, or RM/DM/NM label
enters the actor. **b**, Optimization proceeds from task/oracle validation through
forced value learning to free selection. Held-out mappings never update model
weights. This is an explanatory schematic, not a result plot.

### 3.2 Encoding policy

A shared neural network controls encoding in both $a1$ and $b1$. “Shared” means
that the same parameters are applied to both events; the model cannot encode target
events densely while treating distracting events differently based on a hidden
identity label. The actor computes

\[
p_t=\sigma(f_\theta(x_t)), \qquad z_t\sim\operatorname{Bernoulli}(p_t),
\]

where $f_\theta$ is a two-hidden-layer neural network, $\sigma$ is the logistic
function, $p_t$ is the probability of encoding, and $z_t$ is the binary
encode-or-skip action. When $z_t=1$, the current 80-dimensional state becomes a
distinct episodic trace. The primary capacity is 40 traces, enough to retain every
possible state from $a1$ and $b1$ even at delay 4. Consequently, selective
encoding cannot win merely by avoiding eviction.

### 3.3 Differentiable retrieval

During $b2$, cosine similarity compares the current situation state with each
encoded trace $m_i$. The attention assigned to trace $i$ is

\[
\alpha_i=\frac{\exp(\cos(x_t,m_i)/\tau)}
{\sum_j\exp(\cos(x_t,m_j)/\tau)}, \qquad
e_t=\sum_i\alpha_i m_i,
\]

where $\tau=0.05$ is the retrieval temperature, $\alpha_i$ is the competitive
retrieval weight, and $e_t$ is the retrieved situation state. This operation is
**differentiable**: prediction gradients can change retrieval-related parameters.
The discrete encoding action is trained with a policy gradient rather than by
pretending that a fractional snapshot is a discrete episodic memory.

Unconditional retrieval was strongly harmful in NM trials during exploration.
Therefore a content-match gate compares only the currently observed feature--value
content with stored content:

\[
g_t=0.2\,\sigma\!\left(30\,[\max_i\cos(S_t,M_i)-0.60]\right).
\]

Here $M_i$ is the feature--value portion of memory $m_i$, and $g_t$ scales the
retrieved state before prediction. The threshold 0.60 and strength 0.20 were selected
on exploratory fixed-schedule trials and frozen before the 10-seed diagnostic run.
The gate uses similarity, not the hidden memory condition. Model retrieval is an
analogue of reinstatement; it is not equivalent to human gaze or hippocampal
retrieval.

### 3.4 Forced exploration and free selection

Before free selection, random encoding actions train a **critic**, a neural network
that estimates later prediction reward from the current situation state. In the free
stage, the actor and critic are trained while the predictor and retriever are frozen.
For a trial-level delayed reward $R$, critic estimate $V(x_t)$, and entropy bonus
$H$, the actor loss is

\[
\mathcal{L}_{\mathrm{actor}}
=-\frac{1}{T}\sum_t\log P_\theta(z_t\mid x_t)[R-V(x_t)]-\eta H,
\]

where $T$ is the number of eligible encoding decisions and $\eta=0.002$. The same
later $b2$ prediction outcome supplies credit to all earlier choices in $a1$ and
$b1$.

## 4. Experimental protocol

The reported configuration used 10 independent model seeds (300--309), none of
which was excluded. Every model received 50 forced-value updates and 600 free-policy
updates, with 16 trials per update. Initial encoding probability was 0.02 at every
time point, imposing sparsity but no temporal preference. The 0.25/0.25/0.50
RM/DM/NM mixture was implemented as a repeated RM, DM, NM, NM cycle. Each frozen
model was evaluated on 256 new trials per condition. Baselines and ablations shared
the same trial bank within every model seed.

This is a **diagnostic replication**, not a preregistered confirmatory test. The
structured representation, content threshold, retrieval strength, and sparse
initialization were selected after exploratory runs. The evaluation seed namespace
was separate, and all declared seeds were retained, but the architectural choices
had already been informed by earlier behavior.

### 4.1 Learned and forced policies

::: {.concept-table}
| Report label | Learned? | Definition |
|---|:---:|---|
| Learned policy | **Yes** | The actor samples encode or skip from its optimized $p_t$ at every eligible time point. |
| Endpoint only | No--forced | Encode only the final state of both $a1$ and $b1$. |
| Midpoint only | No--forced | Encode only the declared midpoint of both events. |
| Midpoint + endpoint | No--forced | Encode both the midpoint and final state. |
| Dense | No--forced | Encode every eligible state. |
| Never | No--forced | Encode no episodic state. |
| Matched random one | No--forced | Encode one uniformly sampled time point per event. |
:::

The endpoint-only bar in Figure 4 is therefore an experimenter-imposed comparison,
not a second model that learned to encode at event boundaries.

### 4.2 Success criteria and uncertainty

The model seed is the unit of replication. We required at least 10 seeds, positive
endpoint-minus-nonendpoint encoding probability, positive DM reward relative to
never encoding, and loss of that benefit when retrieval was disabled. Ninety-five
percent intervals resample the 10 model seeds 10,000 times using bootstrap seed
20,260,830. The configuration and every seed-level JSON file are retained in the
repository.

## 5. Results

### 5.1 The architecture makes endpoint encoding useful

Forced endpoint-only encoding achieved expected reward 0.6623 in held-out DM
trials. This exceeded midpoint-only (0.5164), midpoint-plus-endpoint (0.5483), dense
(0.5483), matched random one-encoding (0.5138), and never encoding (0.4881). Thus the
task and retriever satisfy the essential precondition: a complete endpoint memory
can improve prediction, and competing incomplete memories reduce that advantage.
These schedules were imposed and cannot demonstrate learning.

### 5.2 The learned policy is sparse but not boundary-selective

The learned endpoint probability in DM was 0.01633, while mean nonendpoint
probability was 0.01663. Their paired difference was -0.00030 ± 0.00101 across model
seeds, with 95% interval [-0.00092, 0.00026]. Only 4 of 10 seeds had a positive
difference. The model encoded about 0.270 states per event, consistent with a sparse,
nearly time-uniform policy rather than a single endpoint action.

![](../../outputs/paper_task_encoding/figures/fig_03_learning.svg)

**Figure 3. The actor did not learn endpoint selectivity.** **a**, Mean endpoint and
mean nonendpoint probabilities during 600 free-selection updates; shading is one
standard deviation across 10 model seeds. **b**, Encoding probability across the 16
steps of unseen DM events; the band is a 95% bootstrap interval over model seeds.
**c**, Endpoint-minus-nonendpoint probability for every model seed and the mean with
95% interval. Quantitative marks are measured synthetic simulation results.

### 5.3 The learned memories nevertheless have a small causal benefit

Learned-policy DM reward was 0.5013 ± 0.0047, compared with 0.4881 for never
encoding. The paired benefit was 0.01319, interval [0.01023, 0.01580], and was
positive in all 10 seeds. Disabling retrieval while preserving the sampled encoding
actions returned reward to 0.4881, so the same 0.01319 benefit disappeared. The model
therefore used episodic information; its failure is about *when* it encoded, not
whether the memory system influenced prediction.

Removing target $b1$ memories reduced DM reward to 0.4880, whereas removing the
distracting $a1$ memories left reward at 0.5034. Disabling the content-match gate
reduced reward to 0.3789. These interventions show that the learned benefit depends
on occasionally storing target content and retrieving conservatively. Because the
learned probability is nearly flat over time, it does not depend on preferentially
storing the endpoint.

![](../../outputs/paper_task_encoding/figures/fig_04_results.svg)

**Figure 4. Forced endpoint encoding is effective, but the learned benefit has a
different mechanism.** **a**, Forced schedules in held-out DM trials; endpoint-only
is the best declared schedule. **b**, Paired learned-policy ablations for all model
seeds. Removing target memory or retrieval eliminates the small benefit; removing
the lure does not, and unconditional retrieval is harmful. Quantitative marks are
measured synthetic simulation results.

::: {.concept-table}
| Quantity | Across-seed mean | 95% bootstrap interval | Criterion |
|---|---:|---:|---|
| Endpoint minus nonendpoint probability | -0.00030 | [-0.00092, 0.00026] | **failed** |
| DM learned reward minus never encoding | 0.01319 | [0.01023, 0.01580] | passed |
| DM learned reward minus retrieval off | 0.01319 | [0.01023, 0.01580] | passed |
| Seeds with positive endpoint selectivity | 4 / 10 | -- | failed |
:::

The complete success criterion therefore failed. It would be incorrect to conclude
that this differentiable-memory model learned to encode episodic memories
selectively at event boundaries.

## 6. Why learning failed

The negative result survived three diagnostic changes. Starting the discrete actor
near 0.5 produced a dense late-event policy rather than endpoint-only encoding.
Giving memories continuous encoding strengths and optimizing them directly through
differentiable retrieval favored the first observation, showing that binary
policy-gradient variance was not the only obstacle. Starting the corrected shared
actor sparsely at 0.02 produced the reported near-uniform policy. A longer
single-seed exploratory run of 1,500 free updates also shifted toward an early
encoding band rather than the endpoint.

The credit structure explains these solutions. At $a1$ and $b1$, the current
event has no observable prospective label saying whether it will later be a
distractor, a useful target, or irrelevant. A complete target memory becomes more
useful toward its boundary, but a complete distracting memory also becomes a
stronger competitor. The shared policy must apply the same temporal rule to both.
The small global advantage of forced endpoint-only encoding therefore sits in a
reward landscape with easier sparse-random, early, and dense-late solutions. Fixed
schedule comparisons can establish that endpoint encoding is valuable; they do not
guarantee that an unconstrained online learner can discover it from delayed reward.

This distinction sharpens the theoretical claim. “Endpoint encoding is better than
midpoint-plus-endpoint encoding under competitive retrieval” is supported by the
forced comparison. “Endpoint encoding is the policy that optimization will learn on
this task” is not supported by the present simulation.

## 7. Limitations and proposed next steps

::: {.limitations-table}
| Limitation | What it means | Why it matters | Proposed next step |
|---|---|---|---|
| Structured predictor rather than the original LSTM | Feature--value observations are accumulated by a transparent 80-dimensional state instead of learned recurrent dynamics. | The result isolates encoding but does not show what the original recurrent architecture would learn. | Pretrain the released predictor/retrieval architecture to the forced-endpoint precondition, freeze it, and optimize only a shared encoder. |
| Content-match retrieval gate was selected exploratorily | Threshold 0.60 and retrieval strength 0.20 were chosen after inspecting fixed-schedule behavior. | The final run is diagnostic, not confirmatory; the gate may shape which encoding schedules are useful. | Preregister the gate or learn it on a disjoint retrieval objective, then evaluate new model and trial seeds once. |
| Future relevance is unobservable during first presentation | $a1$ and $b1$ are drawn from the same situation distribution, yet only $b1$ may later support $b2$. | A shared encoder cannot know at encoding time which event will become a target, weakening boundary-specific credit. | Use a continuous stream in which every event can later be queried, so the same endpoint rule has symmetric future value, without adding a target label. |
| Sparse initialization and 600 free updates | The reported models begin near never encoding and receive a finite optimization budget. | A very slow transition to endpoint selectivity is not logically excluded. | Run prespecified longer budgets with learning-curve stopping rules; the 1,500-update exploratory failure suggests, but does not prove, that time alone is insufficient. |
| Primary capacity differs from the historical comparison | We allow 40 traces with no eviction; the original forced endpoint and midpoint-plus-endpoint policies used two and four slots, respectively. | Large capacity prevents a trivial storage-budget explanation but changes retrieval competition. | Repeat after freezing the primary conclusion with the historical policy-matched capacities and a fixed two-slot capacity, labeled as sensitivities. |
| Released code and prose differ | The primary profile uses delay 0--4 and roughly 15% mean removal; the paper describes delay 0--3 and independent 30% removal. | “Exact task” has two nonidentical authoritative descriptions. | Run the already specified `paper_text` sensitivity without pooling it with the released-code result. |
| Synthetic modeling only | Situation states are abstract categorical variables and model retrieval is not human gaze or hippocampal activity. | Computational feasibility or failure cannot establish the human mechanism. | Derive behavioral predictions that distinguish learned retrieval, boundary encoding, and nonspecific sparse encoding, then test them in human data separately. |
| Not preregistered confirmation | Architectural corrections followed exploratory failures, although reported seeds were all retained. | Bootstrap intervals quantify seed variation, not researcher degrees of freedom. | Treat this report as hypothesis refinement and preregister the next architecture, training budget, and stopping rule. |
:::

## 8. Feasibility of the original simulation setting

Using the original **task** is feasible and has now been done: the 16-feature,
four-value representation, 37-unit input, five responses, random observation order,
delayed fixed queries, penalties, missing observations, and RM/DM/NM trial structure
are implemented and tested. Using the original **model architecture** is also
computationally feasible, but the independent LSTM did not yet satisfy the fixed
endpoint retrieval precondition. Training a free encoding actor on top of that weak
predictor would confound failure of prediction, retrieval, and encoding.

The most direct next experiment is therefore not a larger unconstrained joint model.
It is a staged transplant: reproduce or reuse the released predictor, demonstrate
forced endpoint benefit on held-out DM situations, freeze prediction and retrieval,
and then train only the shared encoding policy. The event stream should also remove
the asymmetric “distractor now, possible target later” credit problem by making each
event equally eligible for a future query. That preserves the theoretical advantage
of complete event models without giving the encoder a boundary label or a prospective
target label.

## 9. Conclusion

This study reaches a clean negative conclusion. On the exact released
event-prediction task, a differentiable episodic-memory system can use encoded
information to improve delayed prediction, and forced endpoint-only encoding is
substantially better than the paper's midpoint and dense alternatives. Yet the free
actor does not learn to encode selectively at event boundaries across 10 model
seeds. Its encoding probability remains sparse and nearly uniform over time.

The result does not refute the narrower argument that complete endpoint memories are
better retrieval targets than incomplete memories. It limits the stronger optimal-
policy claim: the advantage shown by forced schedules is not sufficient to make
endpoint encoding reliably emerge through optimization in this online task. A next
test should preserve the original prediction problem while making future query value
symmetric across events and separating competent prediction/retrieval from encoding
policy learning.

## 10. Reproducibility and provenance

The implementation, tests, configuration, seed-level records, and aggregate metrics
are versioned. The primary files are:

- configuration: [`configs/paper_task_encoding/reported_failure.yaml`](../../configs/paper_task_encoding/reported_failure.yaml);
- aggregate results: [`outputs/paper_task_encoding/reported_summary.json`](../../outputs/paper_task_encoding/reported_summary.json);
- seed records: [`outputs/paper_task_encoding/reported/`](../../outputs/paper_task_encoding/reported/);
- task generator: [`src/boundary_em/paper_task.py`](../../src/boundary_em/paper_task.py);
- model: [`src/boundary_em/structured_paper_model.py`](../../src/boundary_em/structured_paper_model.py);
- policy training: [`src/boundary_em/paper_policy_training.py`](../../src/boundary_em/paper_policy_training.py);
- test suite: [`tests/`](../../tests/).

Each number in the result figures is generated from the seed JSON files. The figures
are reproducible from the scripts saved beside them. Model weights were frozen in
held-out evaluation, mappings and observation orders were newly sampled, and all 10
declared model seeds were retained.

### References

Lu, Q., Hasson, U., & Norman, K. A. (2022). A neural network model of
differentiation and integration of hippocampal and neocortical memory systems.
*eLife, 11*, e74445. [Article](https://doi.org/10.7554/eLife.74445) ·
[Released code](https://github.com/qihongl/learn-hippo)
