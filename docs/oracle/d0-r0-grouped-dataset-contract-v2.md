# D0/R0 grouped exact dataset contract v2

Status: **authoritative dormant CPU research contract**. V1 is retained only as review history. This document does not authorize dataset export, fitting, inference, routing, rendering, or production promotion.

## Corrected candidate identity

A candidate panel has two levels of identity.

### Stable ordered slot

The cross-work axis is a stable semantic slot:

```text
slot_id
model bundle SHA-256
adapter bundle SHA-256
construction-contract SHA-256
query-contract SHA-256
output role
```

This is the common `K` axis for both D0 and R0.

### Per-source artifact binding

Each source family realizes every slot as a different exact artifact:

```text
slot identity
source-family certificate SHA-256
recipe identity
artifact decoded-PCM SHA-256
recipe semantic SHA-256
verified recipe→slot projection
verifier commit
passed status
```

Source-specific recipe and PCM identities are expected to vary by work. They are row-identity-bearing but are not the cross-work panel identity.

## Closed-world source family

`source_family_sha256` is not a caller-chosen group string. It is the semantic SHA-256 of a strict v2 certificate containing:

- full `(recipe_id, artifact_pcm_sha256)` references for mixture, A truth, V truth, and every derivative;
- the complete derivation DAG;
- the closed-world discovery-manifest identity;
- permitted spectral grids and query conditions;
- derivation-policy identity and verifier commit.

The verifier globally proves acyclicity and checks **every parent path**. Candidate and feature artifacts must terminate only at the exact mixture root. A valid parent cannot hide another parent derived from clean truth. Full artifact references disambiguate no-vocal cases where mixture PCM equals accompaniment-truth PCM.

Every candidate artifact binding in a dataset row must also appear as a candidate node in that row’s strict source-family certificate.

## Exact row

A v2 row binds:

1. group-family certificate, work/session/singer/query identity;
2. exact mixture, A-truth, and V-truth PCM identities;
3. complete cell geometry and spectral-grid identity;
4. one stable candidate-panel identity;
5. one verified per-source artifact for every slot, in panel order;
6. ordered metric names, units, and lower-is-better directions;
7. finite nonnegative `K × D` exact risks and an equally shaped availability mask;
8. one finite inference-available feature vector;
9. feature, metric, and structured-route-policy identities.

Unavailable entries are stored as zero only to obtain canonical finite JSON. The mask is separately identity-bearing and every feasibility helper treats unavailable critical evidence as infeasible.

## Inference projection

The inference record contains the mixture/cell/query identities, stable panel, per-source candidate artifacts, feature contract, and feature vector. It excludes clean A/V truth, exact risks, availability labels, exact routes, and oracle margins.

## Group split

All derivatives of one source-family certificate remain in one partition, including crops, bands, resolutions, remixes, gains, transforms, aliases, and query variants unless a separately frozen query-generalization study says otherwise.

The split manifest exactly partitions group-family identities into nonempty train/calibration/test sets and separately rejects any shared source-family certificate across those sets. Model and hyperparameter selection must be completed on training-internal folds and bound before calibration is opened.

## D0 versus R0

- **D0:** predict the route produced by the frozen global structured oracle on the preregistered high-confidence population. Never train on independent local argmins.
- **R0:** predict the complete candidate × defect tensor, calibrate upper risks by independent source families, and apply the identical frozen decoder.

Both arms receive the same stable panel, per-source artifacts, features, query identity, splits, compute/search budget, checkpoint rule, decoder, thresholds, and complete-work evaluation.

## Evaluation

Primary evidence is complete-route exact regret, catastrophic false-safe count, critical-gate violations, coverage/abstention, worst-work regret, route switching/boundaries, and target-singer correctness. Crop accuracy is diagnostic only.

Hall remains a secondary prediction head in the first CPU-only comparison, but availability-bearing complete-work hall, stereo, transient, artifact, and seam non-regression gates remain mandatory for promotion or release.

If all frozen slots fail the same critical cells, the panel is representationally insufficient. The response is a target-singer correction/separator experiment—not relaxed thresholds or post-hoc group definitions.
