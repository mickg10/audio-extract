# Trained Voice-Removal Judge: Data, Architecture, Validation, and Parameter-Tuning Plan

> **Repository:** `mickg10/audio-extract`  
> **Issue:** #1  
> **Mission:** train a judge that estimates voice-removal quality without an accompaniment reference at inference, then use that judge to select separation parameters  
> **Status:** proposed implementation and acceptance plan  
> **Date:** 2026-08-07

---

## Executive decision

The reframed mission is correct:

> Train a judge of how well the requested voice was removed, obtain the data needed to train and validate it, and use the trained judge to select parameters.

The primary judge should **not** be:

- a scalar heuristic based on ensemble disagreement;
- SAM Audio Judge by itself;
- a candidate-only classifier;
- a model trained from unlimited self-remixes while treating those remixes as independent real works.

The recommended design is a **task-conditioned, source-aware, multi-stream, multi-head student judge**.

At inference it receives:

```text
original mixture M
candidate retained output Y
removed signal D = M − Y
task ontology
optional member estimates / diagnostic features
```

It does **not** receive the exact target accompaniment.

At training time it learns from privileged exact-reference labels computed from known retained and removed sources.

The judge predicts separate distributions for:

```text
retained-solo voice
event-correlated accompaniment holes
unexplained artifacts
secondary fidelity damage
out-of-domain / uncertainty
```

Parameter selection is constrained and lexicographic:

```text
satisfy voice-residue ceiling
satisfy event-hole ceiling
satisfy severe-artifact ceiling
then minimize secondary fidelity loss
```

The current data is enough to train a meaningful prototype. It is not yet enough to declare the judge trusted for soloist-plus-orchestra parameter tuning, because the nominal “~36 works” are heavily concentrated in choir, a-cappella, organ, and a small number of recording sessions. The binding constraint is the number of **independent, in-domain soloist-plus-orchestra works and production regimes**, not the number of synthetic cases.

---

# 1. Terminology

## Exact-reference teacher

For a training case with known retained accompaniment \(A\), known removed source \(V\), and candidate output \(Y\), the teacher computes exact labels.

## Reference-free student

“Reference-free” means no exact target \(A\) is required at inference.

The student should still receive the original mixture. A candidate waveform alone cannot reliably distinguish:

```text
a naturally sparse orchestral passage
from
a separator-created spectral hole
```

The correct inference contract is therefore source-aware:

\[
J(M,Y,D,\text{task}),
\qquad D=M-Y.
\]

This is analogous to current reference-free separation evaluators that use the mixture and separated track together.

## Parameter-selection judge

The judge is not merely a report metric. It must preserve the ordering of candidate recipes well enough to choose parameters.

That makes **within-work ranking and top-one regret** more important than small absolute score error.

---

# 2. Exact training labels

For a local time-frequency tile \(q\), fit:

\[
Y_q
=
\alpha_q A_q
+
\beta_q V_q
+
R_q.
\]

Use weighted complex ridge regression:

\[
\begin{bmatrix}
\alpha_q\\
\beta_q
\end{bmatrix}
=
\left(X_q^\ast W_qX_q+\lambda I\right)^{-1}
X_q^\ast W_qY_q,
\qquad
X_q=[A_q\;\;V_q].
\]

## Accompaniment retention and event holes

\[
H_q
=
\left[-20\log_{10}|\alpha_q|\right]_+.
\]

Also retain:

\[
D_q^{(A)}=|\alpha_q-1|.
\]

The first is an audible gain-hole label. The second includes gain and phase error.

## Retained voice

Coefficient:

\[
L_q^{(\mathrm{coef})}=|\beta_q|.
\]

Energy relative to retained accompaniment:

\[
L_q^{(\mathrm{energy})}
=
\frac{
|\beta_q|^2\|V_q\|_2^2
}{
|\alpha_q|^2\|A_q\|_2^2+\epsilon
}.
\]

Both labels are required. The same retained fraction is much more audible when the singer dominates the local mixture.

## Orthogonal artifacts

\[
D_q^{(\perp)}
=
\frac{\|R_q\|_2}
{\|A_q\|_2+\epsilon}.
\]

This represents output energy explained by neither the correct accompaniment nor the removed voice.

## Identifiability

Store:

\[
\kappa_q=\operatorname{cond}(X_q^\ast W_qX_q).
\]

Tiles with excessive condition number are unavailable or high-uncertainty; they must not be assigned falsely precise labels.

## Additional exact labels

Keep:

- event-hole depth;
- event-hole area;
- event-hole duration;
- recovery;
- SI-SDR;
- SI-SIR / SI-SAR where meaningful;
- multiresolution STFT distance;
- auditory-band envelope error;
- transient loss and excess;
- hall-tail decay error;
- stereo width and coherence error.

Do not compress these into one teacher score.

---

# 3. Student input architecture

## Required streams

For each 4–8 second passage:

1. **Mixture** \(M\)
2. **Candidate accompaniment** \(Y\)
3. **Removed signal** \(D=M-Y\)

Optional streams:

4. individual member vocal estimates for an ensemble;
5. task text or categorical ontology;
6. deterministic features and availability masks.

## Why the removed signal matters

The difference contains direct evidence about:

- what the separator removed;
- whether orchestral transients were assigned to the singer;
- whether voice-like harmonics remain in the candidate.

A student that sees only \(Y\) has less information and will confuse content with damage.

## Task conditioning

At minimum:

```text
all_voices_vs_nonvocal
soloist_vs_rest
role_specific_voice_removal
choir_vs_rest
```

For `soloist_vs_rest`, the retained output may legitimately contain chorus or other singers. The judge must not classify those retained voices as leakage.

Example task record:

```yaml
task: soloist_vs_rest
removed:
  kind: solo_voice
  role: Gilda
  voice_type: soprano
retained:
  - orchestra
  - chorus
  - other_soloists
```

## Temporal granularity

Do not train only on whole-track averages.

Use:

- 4–8 second passages;
- vocal-event-centered windows;
- no-vocal controls;
- phrase-end hall tails;
- the worst passages from each candidate.

The model outputs passage-level distributions. Track-level risk aggregates the worst few passages rather than the mean.

---

# 4. Model ladder

## Baseline A — gradient-boosted trees

Use the existing deterministic features plus corrected event-conditioned features.

Model:

```text
LightGBM / CatBoost / HistGradientBoosting
```

Benefits:

- fastest implementation;
- interpretable ablations;
- suitable for small work count;
- establishes whether the current features contain transferable signal.

This model is a baseline, not the final architecture.

## Baseline B — frozen source-aware judge

Run SAM Audio Judge with positive task descriptions.

Examples:

```text
retained target: "orchestral music"
removed target:  "operatic singing"
```

For chorus-retaining tasks:

```text
retained target: "orchestra and operatic chorus"
removed target:  "solo operatic soprano"
```

Store recall, precision, faithfulness, and overall separately.

SAJ is an auxiliary benchmark and potential feature source, not the sole optimization target.

## Primary student — frozen audio encoder plus small multi-stream head

Start with one music-capable frozen encoder:

```text
MuQ or MERT
```

BEATs/M2D can be compared as ablations.

Compute frame embeddings for \(M\), \(Y\), and \(D\). Form interaction features:

\[
z_t=
[
e_M,\,
e_Y,\,
e_D,\,
e_M-e_Y,\,
e_M\odot e_Y
].
\]

Concatenate:

- deterministic feature vector;
- feature-availability mask;
- task embedding;
- domain metadata.

Use a small temporal attention or Conformer/BLSTM head, followed by separate defect heads.

Do not fine-tune a large encoder initially. The current independent-work count is too small. Fine-tuning or LoRA is a later experiment after the frozen-head model has plateaued.

## Optional auxiliary features

SAJ dimensions may be added only if leave-one-corpus-out ablation shows incremental information.

Ensemble-member disagreement belongs primarily in the **uncertainty head**, not the quality head. Several models can agree and still be wrong.

Finite-difference probes are selective audit tools for high-uncertainty cases, not the primary inference path.

---

# 5. Multi-task and ranking losses

For defect \(d\):

\[
\mathcal L_{\mathrm{reg}}
=
\sum_d w_d
\operatorname{Huber}
(\hat y_d,y_d).
\]

Add quantile heads:

\[
\mathcal L_{\mathrm{quantile}}
=
\sum_{d,\tau}
\operatorname{Pinball}_{\tau}
(\hat y_{d,\tau},y_d).
\]

For two candidate recipes \(i,j\) on the same work and passage, construct exact preference labels from the constrained teacher objective.

Pairwise loss:

\[
\mathcal L_{\mathrm{rank}}
=
-\log \sigma
\left(
s_j-s_i
\right)
\]

when candidate \(i\) is better than \(j\).

Final loss:

\[
\mathcal L
=
\mathcal L_{\mathrm{reg}}
+
\lambda_r\mathcal L_{\mathrm{rank}}
+
\lambda_q\mathcal L_{\mathrm{quantile}}.
\]

The ranking term is essential because the downstream use is recipe selection.

## Group balancing

Sample hierarchically:

```text
work uniformly
→ passage/event uniformly
→ candidate recipe uniformly
```

Do not let thousands of self-remixes from one track outweigh one independent work.

---

# 6. Data inventory: what the current “36 works” really means

The new acquisition is useful, but the nominal count is not the effective in-domain count.

## Current strengths

- Bologna and Aalto: direct soloist-plus-orchestra exact references.
- Spheres/PHENICX plus VocalSet: controllable synthetic soloist-plus-orchestra.
- Cantoría: exact/near-exact voice-versus-organ and SATB material.
- Choir datasets: retained-voice task and multi-voice stress.
- 3D-MARCo/Spheres RIRs: measured-room rendering.
- Saraga/Jingju: deliberate OOD tests.
- Library self-remixes: target-domain adaptation and stress generation.

## Current limitation

The inventory is dominated by:

```text
14 Cantoría works sharing a quartet/corpus regime
~11 a-cappella choir works
organ accompaniment
synthetic cross-products
only about 7 directly relevant solo-voice-plus-orchestra works
```

Those sources are not interchangeable.

Fourteen pieces recorded by the same ensemble/session add musical diversity, but much less acoustic-domain independence than fourteen unrelated singers, venues, and mastering chains.

The “100+ synthesizable works” are combinations of a smaller number of singers, orchestral donors, and RIR banks. They are valuable examples, not 100 independent deployment domains.

## Binding constraint

The binding constraint is:

> independent, in-domain soloist-plus-orchestra works spanning singers, voice types, venues, orchestration, reverberation, and mastering.

Synthetic case count is not the binding constraint.

---

# 7. Practical minimum data targets

There is no universal sample count independent of architecture and target domain. Use learning curves and group-held-out performance. The following are practical acquisition targets for a frozen-encoder/small-head judge.

## Prototype

Current pool:

```text
~36 heterogeneous works
+ synthetic combinations
+ library self-remixes
```

This is sufficient to:

- build the data pipeline;
- train tree and frozen-encoder baselines;
- detect shortcut learning;
- run leave-one-work and leave-one-corpus experiments;
- estimate whether the mission is feasible.

It is not enough to trust soloist-plus-orchestra tuning.

## Personal-library beta

Target at least:

```text
40 independent in-domain soloist-plus-orchestra works total
24 training
8 validation
8 untouched final test
```

Also require approximately:

```text
>=10 singers
>=3 voice classes/ranges
>=4 venue/mastering regimes
>=3 dense-orchestra/chorus regimes
```

This supports a beta judge with uncertainty and a fixed alternate output. It does not establish a tight population-risk guarantee.

## Strong internal judge

Recommended:

```text
60–80 independent in-domain works
15–20 untouched final-test works
```

with full work/singer/session grouping.

## Strong failure-rate statement

If the final work-level parameter-selection procedure makes zero catastrophic selections, approximately 29 independent accepted test works are required before the elementary one-sided 95% upper bound drops below 10%.

The judge may be useful well before this bar, but the claim must be narrower.

## Highest-value acquisition

Professionally mastered same-take soloist-plus-orchestra pairs have the largest marginal value.

Cantolopera-style pairs are much more important now than acquiring another hundred synthetic combinations of the same donors.

---

# 8. Split design

## Group identifiers

Every example must include:

```text
work_id
singer_id
recording_session_id
venue_id
mastering_chain_id
corpus_id
vocal_donor_id
orchestra_donor_id
RIR_bank_id
task
```

## Required evaluations

### Leave-one-work-out

Tests work generalization inside known corpora.

### Leave-one-corpus/session-out

Tests whether the judge memorized recording-chain signatures.

### Leave-one-singer-out

Tests voice generalization.

### Leave-one-orchestra-donor-out

Required for synthetic mixtures.

### Leave-one-model-family-out

Tests whether the judge evaluates audio rather than memorizing separator artifacts.

### Unseen recipe test

Hold out some overlaps, weights, and ensemble constructions entirely.

## Target-library self-remix

Use library self-remix data for transductive adaptation or uncertainty calibration.

Do not use those same cases as the final proof that the judge works on the real vocal passages of that track.

A safe first adaptation is:

```text
global judge
+ per-track affine calibration
+ per-track uncertainty widening
```

Do not fine-tune the whole encoder on one track’s remixes.

---

# 9. Self-remix validity

Self-remix cases are legitimate exact-reference cases for the constructed mixture when the accompaniment control is genuinely vocal-free.

They are not unlimited independent works.

## Current circularity risk

The current no-vocal scan relies on separated vocal and accompaniment stems. A separator can miss quiet or reverberant voice and cause a contaminated span to be labeled vocal-free.

For training-grade controls require at least:

- agreement of two architecture-diverse vocal estimates;
- low task-conditioned voice probability;
- low harmonic/coherence evidence against the removed-source estimate;
- no known retained/removed-role ambiguity;
- a margin substantially stricter than the mining threshold.

Maintain:

```text
training_grade_control
screening_only_control
```

If a control is uncertain, do not use it as exact teacher data.

## Domain randomization

For synthetic training vary:

- source-specific measured RIRs;
- voice/orchestra level;
- singer placement;
- section placement;
- stereo perspective;
- EQ;
- compression;
- limiting;
- saturation;
- sample rate;
- codec;
- separator chunk boundaries.

Keep exact-linear and paired-master labels in separate strata.

---

# 10. Adversarial review of the current `judge_features.py`

The file is a useful baseline, not a sufficient judge input contract.

## `singing_voice_prob_on_instrumental`

It is currently pYIN voiced probability.

That is not singing-voice probability. It will respond strongly to violin, flute, oboe, horn, and many retained chorus passages.

Rename it and replace or augment it with a trained, task-aware voice detector.

## Harmonic salience

Running F0 tracking on the candidate itself can confuse orchestral harmonics with residual voice.

Condition harmonic evidence on:

- F0/events inferred from the original mixture;
- the removed signal \(D=M-Y\);
- a provisional soloist estimate;
- the task ontology.

## Global aggregation

Whole-clip means can miss one catastrophic four-second phrase.

Return distributions and event-level sequences:

```text
median
p90
maximum
top-three mean
```

## Missing features

Missing optional inputs currently produce numeric zero.

Add explicit availability masks. Zero disagreement and unavailable disagreement are not the same observation.

## Mixture-reduction features

Broadband and high-frequency reduction relative to the mixture are confounded by correct voice removal.

Make them event-conditioned and interpret them through the multi-stream model, not as direct quality scores.

## Ensemble disagreement

Use it as uncertainty evidence. Low disagreement does not imply high quality.

## High-frequency artifact proxies

HF modulation and musical-noise heuristics are content-dependent. Cymbals, brass, strings, and applause can trigger them.

They need candidate-versus-mixture context and held-out real-orchestra validation.

## Stereo

Do not reduce the principal learned representation to mono. Include left/right or mid/side embeddings.

## Tests

The current noise-versus-harmonic-comb fixture proves direction on one synthetic case. Add:

- solo violin;
- flute/oboe;
- retained chorus;
- brass;
- applause;
- hall tail;
- correct voice removal;
- residual voice;
- orchestral gouge;
- sample-rate mismatch;
- component-wise median stereo artifacts.

---

# 11. Transfer guard

Self-remix training must transfer to reverberant and mastered real tracks.

## Training

Use explicit domain labels and group-balanced domain randomization.

Do not attempt to erase all domain information. The correct severity mapping can genuinely differ by regime.

A hierarchy is preferable:

```text
global representation
+ task head
+ domain calibration head
```

## Paired-master anchors

Use paired mastered works for:

- pairwise recipe ordering;
- transfer calibration;
- ranking loss;
- final held-out testing.

Even when the pair is not a mathematically exact source sum, it supplies a legitimate desired-output target.

## Group-robust optimization

Report and optimize worst-group performance across:

```text
dry
measured hall
paired master
dense orchestra
chorus retained
soloist only
```

Do not let synthetic dry cases dominate the loss.

## OOD behavior

Jingju, Saraga, unseen venues, and unseen model families should increase uncertainty.

The judge must be allowed to abstain or fall back to the fixed median/MDX pair.

---

# 12. Parameter-tuning loop

## Global warm start

Start every track with:

```text
median(MDX23C, Mel-Band, BS) residual
MDX23C residual
```

## Bounded local search

Generate roughly 12–20 candidate recipes on mined passages:

- overlap;
- small weight grid;
- median/geometric median when available;
- model subset;
- native versus residual where genuinely distinct;
- one cleanup transform only when diagnosed.

## Split passages inside the track

Use:

```text
tuning passages
audit passages
```

The optimizer sees tuning passages. The selected recipe must pass on audit passages.

This reduces direct Goodhart overfitting to the judge.

## Uncertainty-aware constrained choice

For candidate \(c\), use upper prediction bounds:

\[
U_{\mathrm{voice}}(c),\;
U_{\mathrm{hole}}(c),\;
U_{\mathrm{artifact}}(c).
\]

Feasible candidates satisfy:

\[
U_{\mathrm{voice}}(c)\le\tau_v,
\]

\[
U_{\mathrm{hole}}(c)\le\tau_h,
\]

\[
U_{\mathrm{artifact}}(c)\le\tau_a.
\]

Among feasible candidates minimize the upper bound on secondary fidelity loss.

If none are feasible:

```text
use fixed fallback
or
mark no_acceptable_candidate / amber
```

Do not minimize one weighted average that allows less voice to compensate for catastrophic orchestral holes.

## Multi-fidelity execution

1. Evaluate all candidates on 8–12 mined passages.
2. Keep top two.
3. Render top two full-track.
4. Re-score passages cut from the full-track artifacts.
5. Package primary plus alternate.

---

# 13. Acceptance gate for the judge

Reproducing one bake-off ranking is necessary but not sufficient.

## Baselines it must beat

- ensemble-disagreement heuristic;
- handcrafted tree baseline;
- SAJ-only ranking;
- fixed median recipe;
- fixed MDX23C recipe.

## Held-out work metrics

### Pairwise ranking

Target provisional gate:

```text
>=85% pairwise accuracy
or Kendall tau >=0.60
```

within held-out works.

### Top-one regret

Let \(c^\star\) be exact-oracle best and \(\hat c\) judge-selected.

Track normalized regret:

\[
\operatorname{regret}
=
L(\hat c)-L(c^\star).
\]

Provisional beta gate:

```text
median normalized regret <= 0.05
p90 normalized regret <= 0.15
```

### Critical constraint violations

A judge-selected candidate must not violate exact voice/hole/artifact ceilings.

For beta:

```text
zero catastrophic violations on at least 12 untouched in-domain works
```

This is an engineering promotion gate, not yet a tight statistical guarantee.

### Uncertainty calibration

A nominal 90% prediction interval should have approximately 85–95% empirical coverage on held-out groups.

Selective-risk curves must improve as low-confidence cases are abstained.

### OOD

The model must identify or widen uncertainty on:

- unseen corpus;
- unseen singer;
- unseen hall/mastering regime;
- unseen separator family;
- OOD vocal tradition.

### Invariances

- one global level match must not reverse ranking;
- time alignment correction must restore score;
- model/recipe names are not features;
- channel-coordinate transforms do not create spurious confidence;
- identical audio receives identical predictions.

## Promotion stages

### Shadow

Judge ranks candidates but does not affect delivery.

### Beta tuner

Judge chooses among a bounded candidate set; alternate is always packaged.

### Trusted tuner

Only after held-out work gates, uncertainty calibration, and transfer tests pass.

---

# 14. Implementation modules

```text
audio_extract/
  judge_dataset.py
  judge_labels.py
  judge_features.py
  judge_embeddings.py
  judge_model.py
  judge_train.py
  judge_eval.py
  tune_parameters.py
```

## Dataset table

Use Parquet for examples:

```text
example_id
work_id
group ids
task
domain
mixture artifact
candidate artifact
removed artifact
recipe id
passage times
exact labels
handcrafted features
embedding ids
availability masks
```

## Model artifacts

```text
judge-model.json
judge-model.safetensors
feature-schema.json
normalization.json
training-groups.json
validation-report.json
```

Everything is content-addressed.

---

# 15. Fast implementation plan

## Days 1–2

- correct and version `judge_features`;
- implement exact \(\alpha,\beta,R\) labels;
- build Parquet dataset;
- implement work-balanced splits;
- train gradient-boosted baseline.

## Days 3–5

- extract frozen MuQ/MERT embeddings for \(M,Y,D\);
- train multi-head MLP/attention student;
- add pairwise-ranking loss;
- compare against SAJ-only and tree baseline.

## Days 6–7

- leave-one-work/corpus/singer evaluation;
- top-one-regret evaluation;
- uncertainty calibration;
- OOD tests;
- shadow-mode ranking of the 23-track library.

## Week 2

- acquire/purchase more in-domain mastered opera pairs;
- freeze untouched test works;
- promote to beta tuner only if gates pass;
- run bounded per-track parameter search.

---

# 16. Direct answers

## Which architecture should be primary?

A learned multi-stream regressor/ranker using \(M\), \(Y\), and \(M-Y\), with frozen music embeddings plus deterministic features.

SAJ is auxiliary. Finite-difference probes are selective audits.

## Why are more works needed?

To make the reference-free mapping generalize across content, singers, rooms, and mastering—not merely to create a population-risk certificate.

The independent in-domain work count is the binding constraint.

## Is the current data enough?

Enough for a prototype and learning-curve study.

Not enough for a trusted soloist-plus-orchestra tuner because only a small subset is truly in-domain and independent.

## Is the synthesizable pool enough?

Enough to teach distortion geometry and populate parameter variation.

Not enough to validate transfer. The ~36 anchors are more important, and within those, the directly relevant soloist-plus-orchestra anchors are the scarce resource.

## Minimum target

For a beta frozen-encoder judge:

```text
~40 independent in-domain works
with 8 untouched final-test works
```

For a stronger internal judge:

```text
60–80 in-domain works
with 15–20 untouched works
```

For a 10% zero-failure upper-bound style claim:

```text
~29 independent accepted test outcomes with zero catastrophic failures
```

## Should tuning be global or per-track?

Global recipe as warm start; bounded per-track selection among a small candidate set.

## Is self-remix legitimate?

Yes, as exact data for the constructed mixture when the control is genuinely vocal-free.

No, it is not unlimited independent evidence, and it cannot by itself prove transfer to real vocal passages.

---

# Bottom line

The trained judge should be:

```text
source-aware
task-conditioned
multi-stream
multi-head
work-balanced
uncertainty-aware
ranking-trained
validated by top-one regret
```

The most valuable next data is not another synthetic cross-product. It is more independent, professionally mastered, same-take soloist-plus-orchestra pairs.

The current 36-work pool is enough to build the judge. The next acquisition tranche determines whether it can be trusted to tune real opera.
