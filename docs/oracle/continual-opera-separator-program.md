# Continual Opera Separator Program
## Data, adaptation, anti-forgetting, and champion/challenger machinery

> **Repository:** `mickg10/audio-extract`  
> **Primary task:** remove featured operatic soloists while preserving orchestra, chorus when explicitly retained, transients, stereo image, dynamics, and hall tails  
> **Operating model:** perpetual improvement as new multitrack, paired-master, room, score, and unlabeled target data arrive  
> **Immediate active experiment:** `opera-htdemucs-vocals-ft-001`  
> **Status:** research and implementation program; this document does not block the active Gate-2 pilot

---

# 1. Product invariant

The product is not a source-separation leaderboard score. It is a production-usable accompaniment:

\[
\hat A = M-\hat V
\]

where:

- \(M\) is the original mastered mixture;
- \(\hat V\) is the estimated featured-soloist signal;
- \(\hat A\) remains on the exact original sample grid.

The product fails in two asymmetric ways:

\[
\text{under-removal:}\quad \hat V \text{ misses voice } \Rightarrow \hat A \text{ retains soloist}
\]

\[
\text{over-removal:}\quad \hat V \text{ steals orchestra } \Rightarrow \hat A \text{ contains holes}
\]

The development priority is:

```text
1. prevent orchestral theft and event holes
2. satisfy a retained-soloist ceiling
3. preserve artifacts, timbre, transients, stereo, and hall
4. reduce compute only after quality
```

A small amount of retained soloist may sometimes be less damaging than a large orchestral gouge, but neither may be hidden by a global average.

---

# 2. Current champion and active challenger

## Current production baseline

```text
median of aligned MDX23C + Mel-Band + BS-RoFormer vocal estimates
→ mixture minus median vocal
```

## Current single-model fallback

```text
MDX23C residual
```

## Active trainable challenger

Load the released, vocal-specialized HTDemucs checkpoint:

```text
04573f0d
```

without reconstructing its architecture manually.

Use the unchanged four-source model package, but construct the product only from its vocal output:

```python
V_hat = outputs["vocals"]
A_hat = mixture - V_hat
```

The other three outputs are not part of the accompaniment construction.

The first continuation run compares:

```text
step 0
step 100
step 500
step 2000
```

against:

```text
untouched 04573f0d
untouched 955717e8
MDX23C residual
current median ensemble
```

No learned judge, Pi planner, accelerator port, or expanded zoo blocks this result.

---

# 3. Why cross-genre multitracks can help

“Genre” is only a loose label. Auxiliary data are useful when they cover a missing factor in the target problem.

## 3.1 Useful transfer factors

### Voice factors

```text
soprano / mezzo / tenor / baritone / bass
wide vibrato
portamento
coloratura
sustained high notes
soft onset
abrupt forte
solo
duet
chorus
backing voices
```

### Orchestral false-positive confusers

```text
solo violin
viola
cello upper register
oboe
clarinet
flute
horn
trumpet
organ
high string sections
brass tuttis
retained chorus
```

### Acoustic factors

```text
anechoic
close spot microphones
main stereo pairs
ambient microphones
source-dependent bleed
short rooms
long halls
stage-position differences
audience/noise
```

### Production factors

```text
linear stem sums
studio mastering
smooth EQ
bus compression
limiting
saturation
codec processing
independently mastered full/base pairs
```

### Task factors

```text
featured soloists vs rest
all voices vs non-vocal
choir vs rest
one duet role vs rest
```

A jazz singer with acoustic big band, musical-theatre singer with orchestra, or classical instrumental multitrack may therefore add more value than a much larger electronic-pop corpus.

## 3.2 How cross-genre data can hurt

Auxiliary data are dangerous when they introduce:

- incompatible source ontology;
- dataset/codec shortcuts;
- random stem mixtures with implausible harmony, timing, and dynamics;
- synthetic-only room behavior;
- one corpus dominating crop count;
- leakage between derivatives of one source across splits;
- a mastering signature correlated with the target label;
- catastrophic forgetting of the pretrained separator’s broad boundary.

The rule is:

> Auxiliary data must earn inclusion by improving held-out opera/classical work-level outcomes under equal compute.

---

# 4. Source ontology

Every row must explicitly state:

```yaml
task: featured_soloists_vs_rest

removed_sources:
  - featured_solo_voice

retained_sources:
  - orchestra
  - chorus
  - explicitly_retained_supporting_roles
```

The first student supports one frozen task:

```text
featured_soloists_vs_rest
```

An unconditioned model must not receive contradictory labels such as “chorus is vocal” in one row and “chorus is accompaniment” in another without a task-conditioning mechanism.

Datasets with incompatible labels have three options:

1. remap their stems to the frozen task when semantically valid;
2. use them only as hard negatives or representation replay;
3. reserve them for a future task-conditioned checkpoint.

---

# 5. Dataset registry and eligibility

Each corpus version is immutable and records:

```text
container and decoded-PCM hashes
license and permitted use
source roles
work / singer / session / ensemble / room / mastering IDs
sample grid and alignment
integrity class
task ontology
available target axes
split role
```

## Integrity classes

### `linear_exact`

Known source decomposition under a declared rendering recipe:

\[
M=A+V
\]

Eligible for exact source-coordinate and waveform supervision.

### `same_take_paired_target`

The desired accompaniment is available from the same performance, but mastering differs.

Eligible for accompaniment-target supervision with a restricted mastering bridge and uncertainty weighting.

Not eligible for:

```text
V = M - A_pair
```

as a clean vocal target.

### `same_performance_bleed`

Simultaneous microphones or spot mixes contain natural cross-source bleed.

Eligible for weak transfer supervision and realism evaluation.

### `matched_program`

Different performance or re-recording.

Eligible for representation/domain exposure, not same-take source truth.

### `donor`

Isolated voice, instrument, RIR, or room material used to construct exact conditions.

---

# 6. Measuring marginal value of a new corpus

A new corpus should not be ranked by track count alone.

Represent its coverage as a factor vector:

\[
x_d =
[
\text{voice classes},
\text{confuser instruments},
\text{room classes},
\text{mastering classes},
\text{task classes}
].
\]

Define:

- \(I_d\): label-integrity score;
- \(N_d\): novelty relative to current corpus;
- \(E_d\): exposure to current champion’s known failures;
- \(P_d\): proximity to target deployment;
- \(C_d\): acquisition, licensing, storage, and processing cost.

A practical acquisition score is:

\[
S_d =
\frac{
I_d\left(
\lambda_NN_d+
\lambda_EE_d+
\lambda_PP_d
\right)
}{
C_d+\epsilon
}.
\]

This is a ranking heuristic, not a statistical guarantee.

For a batch of candidate corpora, use facility-location or D-optimal selection on factor vectors so the selected tranche expands coverage rather than duplicating it.

---

# 7. Training sampler

Sample hierarchically:

```text
corpus
→ work/session group
→ event class
→ crop
```

Never uniformly sample all crops; a long or heavily augmented work must not dominate.

## Gate-2 pilot proportions

```text
45% exact classical/opera M,A,V
20% exact no-vocal classical/acoustic hard negatives
10% exact vocal-only classical/operatic examples
15% Cantolopera paired-target examples
10% broad multitrack replay or parent-model distillation
```

The proportions are preregistered pilot values.

## Curriculum

### Phase A — preserve the broad boundary

```text
exact opera/classical
no-vocal hard negatives
vocal-only controls
broad replay
```

### Phase B — increase target difficulty

```text
dense orchestration
high V/A ratio
abrupt forte
soprano/string and tenor/brass overlap
long hall tails
```

### Phase C — weak real-production transfer

```text
strong and weak Cantolopera pairs
real bleed
mastering variation
```

Do not begin with weak paired data dominating the optimizer.

---

# 8. Hard-example prioritized replay

After each complete-work evaluation, assign each exact event a priority:

\[
p_i =
\left(
\epsilon+
L_{\text{critical},i}
\right)^\alpha.
\]

Normalize within work, then across works, so one difficult recording cannot dominate the entire run.

Use an exponential moving average of event loss across checkpoints and augmentations. A one-off noisy event is not enough to become permanently high-priority.

Critical event priority combines:

```text
orchestral-hole depth/area
retained-solo audibility
orthogonal artifacts
hall/stereo regression
```

This directly concentrates later training on the Verdi/Nessun-Dorma-type failures rather than on easy average crops.

---

# 9. Target-gradient alignment audit for auxiliary domains

Cross-genre data should be tested by its effect on the opera objective, not by its own training loss.

For auxiliary domain \(d\), compute a batch gradient:

\[
g_d=\nabla_\theta L_d.
\]

On a fixed opera-development batch, compute:

\[
g_T=\nabla_\theta L_T.
\]

Monitor:

\[
a_d=
\frac{
g_d^\top g_T
}{
\|g_d\|\|g_T\|+\epsilon
}.
\]

Interpretation:

```text
persistent positive alignment:
    likely helpful transfer

near zero:
    mainly replay/regularization

persistent negative alignment:
    target conflict
```

Do not adapt weights from the final holdout.

Use this audit on a frozen development set and confirm with equal-compute ablations. Gradient alignment is diagnostic; held-out work-level results remain authoritative.

---

# 10. Exact loss family

For exact data:

```python
V_hat = model(M)["vocals"]
A_hat = M - V_hat
```

Use:

\[
L_{\text{exact}}
=
\lambda_{vc}D_{\text{complex-MRSTFT}}(\hat V,V)
+
\lambda_{ac}D_{\text{complex-MRSTFT}}(\hat A,A)
+
\lambda_{aw}\|\hat A-A\|_1
+
\lambda_eL_{\text{event}}
+
\lambda_sL_{\text{stereo}}
+
\lambda_hL_{\text{hall}}.
\]

Because:

\[
\hat A-A=-(\hat V-V),
\]

identical waveform norms on both stems are redundant. Use complementary domains and event structure.

## No-vocal controls

\[
M=A,\quad V=0
\]

\[
L_{\text{no-vocal}}
=
D(\hat V,0).
\]

Use high weight. These examples teach the separator not to call orchestra “solo voice.”

## Vocal-only controls

\[
M=V,\quad A=0
\]

\[
L_{\text{vocal-only}}
=
D(\hat V,V).
\]

---

# 11. Differentiable source-coordinate penalties

For a local complex tile:

\[
Y=\hat A,\qquad X=[A\;\;V].
\]

Estimate:

\[
c=
\begin{bmatrix}
\alpha\\
\beta
\end{bmatrix}
=
(X^\ast WX+\lambda I)^{-1}X^\ast WY.
\]

Then:

\[
R=Y-\alpha A-\beta V.
\]

Penalize:

\[
L_\alpha=|\alpha-1|^2
\]

\[
L_\beta=|\beta|^2
\]

\[
L_{\beta,\text{aud}}
=
\frac{
\|\beta V\|_2^2
}{
\|\alpha A\|_2^2+\epsilon
}
\]

\[
L_R=
\frac{
\|R\|_2^2
}{
\|A\|_2^2+\epsilon
}.
\]

Use:

\[
\kappa=\operatorname{cond}(X^\ast WX)
\]

to mask or downweight ill-conditioned tiles.

This loss explicitly separates:

```text
orchestral transfer
retained solo voice
unexplained artifacts
```

rather than asking one SDR number to represent all three.

---

# 12. Cantolopera mastering bridge

The paired accompaniment target is valuable, but the bridge must be too weak to absorb the singer.

## 12.1 Inactive-region selection

Use agreement among:

```text
multiple vocal estimators
independent vocal activity detector
score/lyrics timing where available
pair-difference evidence
```

Split contiguous inactive regions into bridge-fit and bridge-validation blocks.

## 12.2 Allowed bridge

```text
sample-rate ratio correction
sub-sample delay
fixed per-channel gain
smooth low-order linear-phase EQ
optional fixed mid/side gain
```

## 12.3 Forbidden bridge

```text
time-varying gain
dynamic compression inversion
framewise EQ
nonlinear neural matching
source-dependent adaptive filtering
```

## 12.4 Robust frequency-domain estimator

On fit frames, estimate a cross-spectral transfer:

\[
H(f)
=
\operatorname{robust-median}_t
\frac{
S_{MA}(f,t)
}{
S_{AA}(f,t)+\lambda
}.
\]

Retain only:

```text
constant delay / linear phase
smooth log-magnitude curve
```

Fit the smooth magnitude on a small number of log-frequency knots.

## 12.5 Validation and uncertainty

On held-out inactive regions, estimate residual variance by band:

\[
\sigma_f^2
=
\operatorname{Var}
\left[
M_f-H(f)A_f
\right].
\]

Use weak-pair loss weights:

\[
w_f=
\operatorname{clip}
\left(
\frac{1}{\sigma_f^2+\epsilon},
w_{\min},
w_{\max}
\right).
\]

Thus uncertain mastering bands contribute less without pretending the target is exact.

Never supervise:

```text
V_hat ≈ M - H(A_pair)
```

as a clean vocal target.

---

# 13. Anti-forgetting mechanisms

Full fine-tuning is the first pilot because it gives the strongest adaptation test.

If it improves opera but damages broad no-vocal controls, compare:

## 13.1 Replay

Retain a small, diverse replay set from the parent model’s broad domain.

## 13.2 Parent-weight regularization

\[
L_{\text{L2-SP}}
=
\lambda_{\text{SP}}
\|\theta-\theta_0\|_2^2.
\]

## 13.3 Parent-output distillation

On replay examples:

\[
L_{\text{distill}}
=
D(f_\theta(M),f_{\theta_0}(M)).
\]

Use only where the parent is not known to fail the opera-specific objective.

## 13.4 LoRA / adapters

A recent singing-voice adaptation study reports that full fine-tuning reached the highest target performance, while LoRA preserved more of the source-domain capability with only a small trainable-parameter increase.

Therefore LoRA is a planned anti-forgetting challenger, not the first experiment.

## 13.5 Zero-initialized correction model

If whole-model continuation is unstable, train:

\[
\hat V'=\hat V_0+\Delta_\psi(M,\hat V_0),
\]

\[
\hat A'=M-\hat V'.
\]

Initialize:

\[
\Delta_\psi=0.
\]

Regularize:

\[
\|\Delta_\psi\|_1
\]

so the student changes only the parent’s identified operatic errors.

This creates exact step-0 parent parity and reduces catastrophic drift.

---

# 14. Unlabeled target-track adaptation

The original user tracks have no clean accompaniment. They can still supply low-weight, label-free consistency constraints.

Let \(F_A\) and \(F_V\) be the accompaniment and vocal outputs.

## 14.1 Vocal perturbation response

Add a small, room-matched vocal probe \(v\):

\[
M_\pm=M\pm\alpha v.
\]

Require:

\[
F_A(M_+)-F_A(M_-)\approx0
\]

and:

\[
F_V(M_+)-F_V(M_-)\approx2\alpha v.
\]

## 14.2 Orchestral perturbation response

For a small orchestral probe \(a\):

\[
M_\pm=M\pm\alpha a.
\]

Require:

\[
F_A(M_+)-F_A(M_-)\approx2\alpha a
\]

and:

\[
F_V(M_+)-F_V(M_-)\approx0.
\]

Use several low amplitudes and reject probes that clip or enter a strongly nonlinear regime.

These are local finite-difference response constraints, not claims about the physical mastering process.

They should enter only after the supervised continuation passes Gate 2.

---

# 15. Teacher-consensus distillation on unlabeled tracks

The existing MDX/Mel/BS ensemble can provide pseudo-targets only where it is trustworthy.

For each tile, calculate:

```text
member agreement
parent/ensemble agreement
vocal activity
orchestral-confuser risk
```

Distill only when:

```text
agreement is high
no-vocal-confuser risk is low
task ontology is clear
```

Skip uncertain tiles.

Exact and paired targets always outrank pseudo-targets.

This uses the existing median as a coverage teacher without importing its known holes everywhere.

---

# 16. Equal-compute auxiliary-data ablation

After the active pretrained HTDemucs pilot, compare:

## A0 — target-only continuation

```text
exact opera/classical
no-vocal classical
vocal-only classical
Cantolopera weak pairs
```

## A1 — targeted acoustic/classical auxiliary data

Add:

```text
classical instrumental multitracks
acoustic vocal multitracks
choir/solo role data
measured source-specific RIRs
```

## A2 — broad replay

Add a small, capped broad multitrack replay tranche.

All arms use:

```text
same parent checkpoint
same optimizer updates
same seeds
same target-domain batches
same evaluation works
```

Promotion depends only on held-out opera/classical work-level results.

---

# 17. Architecture ladder

Do not oscillate architectures without a falsified hypothesis.

## Stage 1 — pretrained HTDemucs continuation

Active now.

## Stage 2 — LoRA/adapter continuation

Use only if full fine-tuning improves opera but forgets broad source boundaries.

## Stage 3 — zero-initialized correction student

Use if whole-model continuation is unstable or data-limited.

## Stage 4 — pretrained RoFormer continuation

MSST supports checkpoint-based training for Mel-Band/BS-RoFormer. Use this when the HTDemucs family fails the dense-operatic frontier, not merely because RoFormer has a higher generic leaderboard score.

## Stage 5 — score-informed protective model

Where aligned scores/MIDI are available, penalize removal of expected orchestral harmonics and use score-derived activity to protect instrumental partials.

## Stage 6 — task-conditioned separator

A query-conditioned model becomes justified when the product must support:

```text
soloist vs chorus
one duet role vs another
choir vs orchestra
```

without separate checkpoints.

Banquet/QSCNet-style single-decoder conditioning and open-vocabulary target-separation work are relevant research directions, but they do not block Stage 1.

---

# 18. Validation hierarchy

## 18.1 Development evidence

- Bologna/Aalto `test-v1` works;
- exact synthetic and source-specific-RIR renders;
- held-out Cantolopera volumes;
- no-vocal confuser set.

These are useful but some are already tuning-contaminated.

## 18.2 Prospective holdout

Freeze a newly acquired work/session before inspecting any challenger output.

All derivatives of that source remain in the same group.

## 18.3 Complete-work evaluation

Every checkpoint renders complete works.

Report:

```text
event-hole depth / area / duration
retained-solo coefficient
retained-solo energy relative to accompaniment
orthogonal artifact ratio
no-vocal false-positive energy
spectral error
transient loss and excess
stereo width/coherence
hall-tail error
sample-grid parity
```

## 18.4 Original hard library

Every promising challenger renders:

```text
Verdi ensembles
Nessun Dorma
Donna
doll_draft
La Wally
```

These are shadow deployment checks, not exact-reference thresholds.

## 18.5 Promotion

A challenger advances only when it is Pareto-better on critical target-domain axes or is a validated specialist for a defined regime.

Improvements on auxiliary genres alone do not count.

---

# 19. Champion/challenger registry

Every experiment records:

```text
parent model hash
corpus manifest hash
split manifest hash
resolved config
loss implementation revision
optimizer and scheduler state
RNG states
code commit
hardware/runtime fingerprint
checkpoint hashes
evaluation report hashes
```

A rejected challenger remains in the registry with its failure map.

This prevents repeated rediscovery of failed ideas.

---

# 20. Rotating holdouts

A holdout ceases to be untouched when its results influence:

```text
architecture
loss
augmentation
threshold
sampler
early stopping
hyperparameters
```

It then becomes:

```text
test-v1
```

A new prospective holdout is frozen before the next promotion claim.

This is the mechanism that makes the program perpetual without repeatedly overfitting one famous Verdi excerpt.

---

# 21. Immediate experiment queue

## Experiment 1 — active Gate 2

```text
04573f0d → 100 / 500 / 2000-step opera continuation
```

## Experiment 2 — targeted auxiliary ablation

```text
A0 target-only
A1 + classical/acoustic hard negatives and measured RIRs
A2 + capped broad replay
```

## Experiment 3 — anti-forgetting challenger

Triggered only if full fine-tuning improves opera but damages controls:

```text
LoRA or zero-initialized correction model
```

## Experiment 4 — unlabeled target consistency

Triggered only after supervised Gate 2:

```text
finite-difference vocal/orchestra response training
```

## Experiment 5 — architecture transition

Triggered only if the HTDemucs family fails:

```text
pretrained Mel-Band/BS-RoFormer continuation through MSST
```

---

# 22. Program-level success

There is no terminal “done.”

At every champion version, the repository must still produce:

```text
full-length Bologna Verdi accompaniment
full-length Bologna Puccini accompaniment
full-length Bologna Donizetti accompaniment
full-length Aalto Mozart accompaniment
original hard-library outputs
complete reproducibility reports
```

New data or methods are accepted only when they improve that measurable frontier without hiding regressions.

The permanent operating loop is:

```text
acquire
→ quarantine
→ audit
→ map factor coverage
→ freeze role and split
→ train challenger
→ exact work-level evaluation
→ original-track shadow render
→ promote or reject
→ rotate holdout
→ repeat
```

---

# 23. Primary research references

- HTDemucs and extra-data/source-specific fine-tuning:  
  <https://arxiv.org/abs/2211.08553>
- Official Demucs training and released checkpoint genealogy:  
  <https://github.com/facebookresearch/demucs/blob/main/docs/training.md>
- Demucs use of extra unlabeled/remixed music:  
  <https://arxiv.org/abs/1909.01174>
- MixIT for unlabeled-mixture adaptation:  
  <https://arxiv.org/abs/2006.12701>
- RemixIT teacher/student bootstrapped domain adaptation:  
  <https://arxiv.org/abs/2202.08862>
- Slakh2100 large synthetic multitrack data:  
  <https://arxiv.org/abs/1909.08494>
- Score-informed synthetic-to-real classical separation:  
  <https://arxiv.org/abs/2503.07352>
- SynthSOD and the synthetic/real orchestral gap:  
  <https://arxiv.org/abs/2409.10995>
- The Spheres orchestral multitrack/RIR dataset:  
  <https://arxiv.org/abs/2511.21247>
- Banquet query-based music source separation:  
  <https://arxiv.org/abs/2406.18747>
- Conditioned UNet/QSCNet:  
  <https://arxiv.org/abs/2512.15532>
- Domain adaptation from speech enhancement to singing separation; full fine-tuning vs LoRA:  
  <https://arxiv.org/abs/2607.11630>

---

*This program is intentionally broader than the immediate 2,000-step pilot, but none of its later stages may delay that pilot.*
