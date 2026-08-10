# D0/R0 grouped counterfactual-risk dataset contract v1

Status: **preregistered dormant CPU research contract**. This document does not authorize separator inference, model training, routing, rendering, or production promotion.

## Question

Compare two equal-compute students on one frozen exact dataset:

- **D0 — structured-route imitation.** Predict the exact globally decoded O2 route on high-confidence training examples. The label is the route produced by the frozen structured policy, never an independently chosen local argmin.
- **R0 — counterfactual-risk prediction.** Predict every frozen candidate × defect outcome and apply the same frozen constrained Potts decoder used to define the exact route.

The comparison asks whether retaining the complete counterfactual supervision tensor lowers held-out complete-route regret and catastrophic false-safe selection relative to compressing each example into a route label.

## One calibration unit means one independent family

Cells, bands, windows, resolutions, crops, remixes, and query variants derived from the same underlying recording family are not independent calibration examples. Every row carries a content-bound `group_family` with:

```text
work_id
recording_session_id
target_singer_id
source_family_sha256
query_condition_sha256
```

The **source-family identity** is the anti-leakage boundary. Every derivative of one source performance—alternate crops, resolutions, bands, synthetic remixes, gain changes, channel transforms, cached aliases, and repeated renders—must remain in one train/calibration/test partition. Different group-name strings do not permit a shared source family to cross partitions.

Unless a separately preregistered query-generalization study says otherwise, all query variants for one source family remain together.

## Exact row

Each exact row binds:

1. group-family identity;
2. exact mixture, accompaniment-truth, and vocal-truth decoded-PCM identities;
3. complete cell geometry and spectral-grid identity;
4. one ordered frozen candidate panel, with recipe and decoded-PCM identity for every candidate;
5. one ordered metric contract—name, unit, and `lower_is_better` direction;
6. finite nonnegative `K × D` exact risks;
7. an equally shaped availability mask; unavailable values are stored as zero but **must never be interpreted as safe evidence**;
8. one finite inference-available feature vector and feature-contract identity;
9. exact metric-contract and structured-route-policy identities.

The row identity binds the canonical hashes of the feature vector, risk tensor, and availability mask. Candidate order, metric order, cell geometry, source identity, or query identity changes produce another row identity.

## Inference boundary

The inference projection contains only:

```text
row identity
group/query identity
mixture identity
cell geometry
ordered candidate identities
feature contract
inference-available feature vector
```

It never contains clean accompaniment, clean vocal truth, exact risks, availability labels, exact teacher routes, or oracle margins.

## Frozen split manifest

The split manifest is written before fitting and binds:

```text
dataset semantic SHA-256
train group-family identities
calibration group-family identities
test group-family identities
split algorithm identity
split seed identity
source commit
```

Validation requires:

- pairwise-disjoint group sets;
- exact partition of all dataset groups;
- no shared `source_family_sha256` across partitions;
- canonical ordering and no duplicate groups;
- no model selection, early stopping, threshold selection, metric scaling, or architecture selection on the calibration or test partitions.

Hyperparameter/model selection must be completed using training-internal folds and bound by a separate selection-manifest identity before the calibration set is opened.

## Group-simultaneous split conformal upper risks

For independent calibration group \(g\), frozen point predictor \(\hat r\), exact risk \(r\), and frozen positive metric scale \(s_d\), define

\[
S_g = \max_{i\in g,\,k,\,d}
      \frac{r_{ikd}-\hat r_{ikd}}{s_d}.
\]

With \(G\) exchangeable calibration groups and requested coverage \(1-\alpha\), use rank

\[
q = \lceil (G+1)(1-\alpha) \rceil.
\]

A finite certificate is refused when \(q>G\). Otherwise the simultaneous offset is the `q`th smallest group score and

\[
U_{ikd}=\max(0,\hat r_{ikd}) + \max(0,S_{(q)})s_d.
\]

Clamping a negative offset to zero only enlarges every upper bound relative to using the negative order statistic, so it preserves the one-sided coverage claim. The claim is simultaneous over all cells, candidates, and calibrated metrics of a new exchangeable group—not over arbitrary future distribution shift.

The calibration certificate must bind the exchangeability scope, group-family definition, candidate/metric/feature contracts, predictor coefficients or state, training/calibration/split manifests, frozen metric scales, calibration algorithm revision, requested and finite-sample guaranteed coverage, group count, conformal rank, offset, and source commit.

## Equal-compute comparison

D0 and R0 use exactly the same:

- train/calibration/test group-family split;
- inference feature tensor and query identity;
- ordered candidate panel;
- structured route policy, hard thresholds, and smoothness;
- optimization-step, wall-clock, parameter, and hyperparameter-search budget;
- checkpoint-selection rule fixed before calibration;
- test-time decoder and conservative parent fallback;
- complete-work evaluation code and exact truth.

D0 receives only the frozen structured-route target and its preregistered high-confidence mask. R0 receives the full `K × D` exact-risk tensor and availability mask. Neither arm receives test truth or calibration scores while fitting.

## Primary evaluation

Report by held-out group and in aggregate:

```text
complete-route exact regret versus the frozen exact oracle
catastrophic false-safe count
critical-gate violation count
coverage and abstention rate
worst-work regret
route-switch count and boundary diagnostics
query-conditioned target-singer correctness
```

Unavailable critical evidence makes a candidate infeasible. It never contributes zero risk.

Hall is a secondary prediction head in the first CPU-only D0/R0 comparison, but hall evidence is mandatory, availability-bearing, complete-work non-regression evidence before any promotion or release. A good training score cannot waive the hall, stereo, transient, seam, or artifact release gates.

## Decision rule

The numerical promotion thresholds must live in a separate immutable preregistration. At minimum:

- no catastrophic false-safe event may be hidden by averaging;
- a result cannot promote from sensitivity-only, calibration-only, or partial-work evidence;
- R0 must be compared with D0 and the frozen production parent on the identical test groups;
- all complete-work release gates remain mandatory;
- if every frozen candidate fails the same critical cells, the router is representationally insufficient and work moves to the target-singer correction/separator track rather than relaxing thresholds.

## Deliberate non-scope

- no neural architecture choice;
- no GPU job;
- no audio render;
- no production selector;
- no replacement of the verified median/MDX release;
- no interpretation of crop accuracy as complete-route safety.
