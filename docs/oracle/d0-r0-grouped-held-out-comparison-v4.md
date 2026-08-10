# D0/R0 grouped held-out comparison v4

> Repository: `mickg10/audio-extract`  
> Branch: `bigoracle/d0-r0-grouped-comparison-artifacts-20260810`  
> Draft PR: `#25`  
> Status: dormant CPU-only research; no promotion decision

## Purpose

Compare two frozen learned-routing strategies on the same independent held-out
source families:

```text
D0 — structured-route imitation
R0 — per-candidate risk prediction + deterministic decoder
```

The artifact measures complete-route exact outcomes. It does not train either
arm, render audio, or select a winner.

## Public artifact boundary

The v3 Python report remains the source semantic model. V4 converts a validated
v3 report into canonical immutable UTF-8 bytes and groups both arms under each
held-out unit:

```text
held-out unit
  ├── D0 complete route lineage and evaluation
  └── R0 complete route lineage and evaluation
```

The public payload is:

```json
{
  "schema": "audio-extract/d0-r0-grouped-comparison-report/v4",
  "status": "COMPLETE_NO_PROMOTION_DECISION",
  "promotion_decision": null,
  "paired_units": []
}
```

Raw NumPy label arrays never enter the issued payload. Once canonical bytes are
created, later mutation of a caller-owned array cannot change the artifact SHA.
Revalidation against a mutated source object fails, while the issued bytes remain
stable.

## Exact-source lineage

Partition generation and complete-route evaluation may legitimately produce two
different reports:

```text
exact risk-evidence artifact
exact source/geometry report
```

V4 does not require those report hashes to be equal. The
`ExactSourceLineageV2` record proves that both refer to the same held-out unit,
source family, truth manifest, and decoded mixture/accompaniment/vocal parents.

## Pre-execution partition manifest

Before an arm runs, `InferenceInputPartitionManifestV2` freezes:

- held-out unit, source family, target query and query-quality stratum;
- D0/R0 arm identity;
- dataset, split and test subset;
- candidate panel, feature contract and metric contract;
- exact partition-certificate identity;
- spectral grid and physical resolution;
- time/frequency range identities;
- rational-measure contract;
- exact-source lineage;
- builder commit.

The resulting semantic SHA must equal
`FrozenArmInferenceRunV3.inference_input_manifest_sha256`.

## Multiple certified resolutions

A source family may have valid certificates at several resolutions. The v2
attestation registry permits those certificates to coexist. Every unit/arm cell
selects exactly one content-bound manifest and certificate, and both D0 and R0
must select the same certificate for a given held-out unit.

Unused alternative resolutions do not invalidate the registry.

## Route and partition attestation

For every unit × arm cell, the registry binds:

```text
pre-execution input manifest
selected partition certificate
arm inference run
route or abstention output
route artifact manifest
route submission
label shape and label SHA
```

A route label grid must have shape:

\[
(T_{\mathrm{cells}}, B_{\mathrm{bands}})
\]

from the selected certificate. An abstention must expose no labels.

The registry rejects reuse of one input manifest, route artifact, route output,
submission, or attestation across matrix cells.

## Diagnostics

Diagnostics v4 derive every denominator from the certificate selected by the
attestation registry.

For a rectangular grid with `T` time cells and `B` bands:

\[
N_t=(T-1)B,
\qquad
N_f=T(B-1).
\]

Every submitted route—safe, catastrophic, or incomplete—is checked against
these limits. Safe-route summaries then report pooled and mean-per-unit switch
rates.

Rational cell measures affect exact risk aggregation, not unweighted graph-edge
counts used for route switching.

## External schema

The paired v4 JSON Schema adds constraints that cannot be represented reliably
when D0 and R0 are serialized as unrelated rows:

- all nested identities in the D0 branch are D0;
- all nested identities in the R0 branch are R0;
- route evaluations require route lineage;
- abstention evaluations require abstention lineage;
- all oracle-available outcomes require a numeric oracle objective;
- critical and secondary violation kinds are disjoint;
- oracle availability must be two-sided within a paired unit;
- one-sided oracle-unavailable status-cross-tab entries are forbidden;
- D0 and R0 summaries remain pinned to their names.

The schema references the closed v3 object definitions for the embedded route
lineage and evaluation records.

## Promotion boundary

The artifact remains descriptive:

```text
coverage
abstention
catastrophic false-safe count
incomplete required evidence
safe-route exact regret
paired regret differences
normalized switching
```

A separate preregistered policy must interpret these measurements. V4 cannot
write a winner or a production promotion decision.

## Validation gate

Before integration into PR #23:

1. exact-head adversarial review;
2. focused tests for v4 report, source lineage, partition attestation and
   diagnostics;
3. complete repository pytest and Ruff;
4. compileall and diff check;
5. real v4 schema round-trip with the v3 schema registered;
6. rejection tests for arm swaps, one-sided oracle availability, wrong violation
   kinds, route/abstention mismatch, wrong partition, wrong shape and mutable
   source arrays;
7. zero unresolved findings on the authoritative v4 files.

No fitting, calibration execution, GPU job, audio render, production selector,
or winner is authorized by this document.
