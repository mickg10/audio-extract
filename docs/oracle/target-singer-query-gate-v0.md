# Target-singer query gate v0

> Research lane only. This design does not replace the certified O1/O2/O3 rerun or the current median/MDX delivery lane.

## Product semantics

The target is not generic `vocals`.

For

\[
M=A+V_t+V_o+C,
\]

where `V_t` is the featured target singer, `V_o` another soloist, and `C` chorus, the requested accompaniment is

\[
A_{product}=M-V_t=A+V_o+C.
\]

A generic vocal separator cannot infer which singer to remove. The long-term extractor therefore receives a target-singer audio enrollment `q`:

\[
e_q=E_{singer}(q),
\qquad
\hat V_t=F(M,e_q),
\qquad
\hat A=M-\hat V_t.
\]

## Lowest-risk first model

Freeze every separator member and train only a low-resolution, singer-conditioned convex gate.

Candidate target-vocal estimates:

\[
\hat V_1,\ldots,\hat V_K.
\]

Gate:

\[
w_{t,b}=G_\psi(M,\hat V_1,\ldots,\hat V_K,e_q),
\qquad w_{t,b}\in\Delta_K.
\]

Output:

\[
\hat V_t=\sum_iw_{i,t,b}\hat V_i,
\qquad
\hat A=M-\hat V_t.
\]

The first implementation uses one shared stereo weight vector per time/frequency cell and cannot extrapolate outside the candidate convex hull.

## Exact parent parity

Choose a conservative parent, initially the current waveform median or the certified-envelope-selected parent.

Use a route scale `s`:

\[
w=(1-s)e_c+s\,\operatorname{softmax}(z),
\qquad 0\le s\le1.
\]

At step zero:

```text
s = 0
output = conservative parent through an exact bypass
```

After parity is recorded, start training with a small positive `s` and ramp it under the experiment configuration. With `s=0`, the learned branch intentionally receives no gradient.

## Inputs

Low-resolution features may include:

```text
mixture spectral/temporal features
candidate vocal features
candidate disagreement
current objective metric features
target-singer embedding
vocal activity
orchestral density
room/hall descriptors
task embedding
```

The singer encoder is frozen in v0.

## Enrollment hierarchy

Preferred target-singer query sources:

1. licensed clean solo recording of the same singer;
2. another aria by the same singer;
3. isolated multitrack stem;
4. sparse, high-confidence passage from the same recording;
5. consensus vocal estimate from an easy passage;
6. album-level singer profile assembled across tracks.

Enrollment and target passages must be disjoint.

## Exact scene construction

For target singer `s_t`:

\[
M=A+V_t+V_o+C,
\]

with targets

\[
V_{target}=V_t,
\qquad
A_{target}=A+V_o+C.
\]

Training scenes include:

```text
target soloist + orchestra
target soloist + another soloist
target soloist + chorus
target soloist + orchestra + chorus
same-voice-type competing singers
source-specific RIRs and microphone conditions
```

## Query-absent controls

For an enrollment singer absent from the mixture:

\[
V_{target}=0.
\]

This prevents the model from removing any similar soprano, tenor, or chorus merely because a singer query exists.

Counterfactual tests on one mixture:

```text
query target singer
query other singer present
query absent singer
query chorus
```

## Losses

Use the versioned stable residual objective once it is reviewed:

```text
one mixture-consistent source residual
mixture/reference-normalized MR-STFT
source-coordinate alpha/beta/R
A-only and V-only controls
accompaniment stereo/hall terms
```

Add:

\[
L_{id}=d(E_{singer}(\hat V_t),e_q),
\]

on sufficiently active target regions,

\[
L_{absent}=D(\hat V_t,0),
\]

for query-absent controls, and gate regularization

\[
L_g=\lambda_c\|w-e_c\|_1+
\lambda_tTV_t(w)+
\lambda_fTV_f(w).
\]

## Data roles

- Bologna/Aalto exact scenes: target-singer and complete-work evaluation.
- MoisesDB detailed roles: lead singer as target; choir/background vocals retained.
- URMP/Spheres: no-target classical hard negatives and room rendering.
- isolated singer corpora: enrollment and singer-identity pretraining, subject to rights.
- Cantolopera: base/inactive hard negatives plus bridge-weighted accompaniment supervision; `full-base` is not an exact clean target singer.

All singer, work, session, room, donor, and mastering groups remain atomic.

## Promotion gate

The v0 gate advances only if it:

1. preserves exact step-zero parent output;
2. beats the current median champion on complete held-out works;
3. improves target-singer removal by at least 1.5 dB on one critical axis;
4. regresses the opposing orchestral-hole axis by no more than 0.5 dB;
5. retains non-target singers and chorus;
6. keeps query-absent/no-vocal false positives within the frozen limit;
7. preserves stereo, hall, transient, artifact, and seam limits;
8. survives singer-held-out and work-held-out evaluation.

## Escalation

```text
large certified candidate-routing gap:
    train this frozen-member query gate

small candidate gap but target role ambiguity remains:
    zero-initialized query-conditioned correction student

common exact failure cells outside the candidate hull:
    full query-conditioned separator or a new pretrained family

source assignment succeeds but accompaniment remains damaged:
    separate restoration student
```
