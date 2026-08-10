# D0/R0 route-partition attestation v1

> Separate artifact branch: `bigoracle/d0-r0-grouped-comparison-artifacts-20260810`  
> Draft PR: `#25`  
> Status: dormant CPU-only research evidence; no promotion decision

## Problem closed by this layer

A route output can have the same matrix shape as a certified exact partition but
still refer to another spectral grid, resolution, range set, or inference input.
Shape-only normalization is therefore insufficient.

## Per-route attestation

`RoutePartitionAttestationV1` binds one report cell to:

```text
held-out unit + D0/R0 arm
route artifact manifest
route output
route submission
inference-input manifest
CellPartitionCertificate
spectral grid and resolution
exact label grid shape/hash
source and verifier commits
```

For a route:

```text
route_output.labels_shape
submission.labels_shape
attestation.labels_shape
    == (certificate.time_cell_count, certificate.band_count)
```

and all label hashes must agree.

For an abstention, all three label fields are absent.

## Complete registry

`RoutePartitionAttestationRegistryV1` requires exactly one attestation for every:

```text
held-out unit × {D0,R0}
```

It binds the grouped report and the complete geometry manifest, refuses missing
or duplicate cells, and revalidates every route and partition parent.

## Certified envelope

`CertifiedGroupedDiagnosticsEnvelopeV1` binds:

```text
grouped report SHA
geometry-manifest SHA
grouped diagnostics SHA
route-partition attestation-registry SHA
verifier commit
```

Its only legal status is:

```text
COMPLETE_NO_PROMOTION_DECISION
```

and its serialized output contains:

```json
"promotion_decision": null
```

## Non-scope

The attestation proves artifact identity and grid consistency. It does not prove
model quality, fit a student, run a separator, render audio, or choose D0/R0.
