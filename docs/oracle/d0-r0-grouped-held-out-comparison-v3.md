# D0/R0 grouped held-out comparison v3

> Branch: `bigoracle/d0-r0-grouped-comparison-artifacts-20260810`  
> Draft PR: `#25`  
> Status: dormant CPU-only research artifact; no promotion decision

## Purpose

Compare the two preregistered learned-routing arms on the same independent
held-out source families:

```text
D0 — imitate the frozen structured exact route
R0 — predict per-candidate defects, calibrate upper risks, then decode
```

The report measures complete-route exact behavior. It does not train either arm,
render audio, select a winner, or change the verified median/MDX release.

## Transitive route provenance

V3 requires this complete chain for every held-out unit and arm:

```text
preregistered arm
  → verified held-out inference-run manifest
  → route/abstention output
  → content-bearing route artifact manifest
  → ArmRouteSubmissionV1.route_evidence_sha256
  → CompleteRouteEvaluationV1.submission_sha256
  → grouped comparison row
```

The inference-run manifest binds:

- preregistration and held-out unit;
- source family, query condition, and query-quality stratum;
- frozen arm identity and artifact manifest;
- compute budget, checkpoint rule, and inference policy;
- dataset, split, test subset, candidate panel, feature and metric contracts;
- immutable inference-input manifest and runner bundle;
- source and runner commits.

The route artifact embeds that run and the actual route output. A submission is
accepted only when its `route_evidence_sha256` equals the semantic SHA of the
artifact manifest. The complete-route evaluation must name the resulting
submission SHA.

## Status consistency

For an oracle-available unit:

```text
ROUTE_SAFE
ROUTE_CATASTROPHIC_FALSE_SAFE
ROUTE_INCOMPLETE_REQUIRED_EVIDENCE
    require submission.status == ROUTE

ABSTAIN
    requires submission.status == ABSTAIN
```

`ORACLE_UNAVAILABLE` may accompany either submission kind because no exact route
comparison is possible.

## Matrix invariants

The completed matrix must contain exactly one row for every:

```text
held-out unit × {D0,R0}
```

It rejects reuse of one:

```text
inference-run SHA
route-output SHA
route-artifact-manifest SHA
submission SHA
```

across matrix cells. Route-label patterns themselves may coincide across works.

## Certified normalized diagnostics

`counterfactual_risk_grouped_diagnostics_v2.py` consumes the v3 report and one
content-bearing exact partition certificate per held-out unit.

Each partition certificate binds:

- unit and source-family identity;
- exact panel, preflight, and oracle decision;
- spectral grid;
- partition contract and partition artifact;
- time-cell and band counts;
- source and verifier commits.

Switch-count feasibility is checked for **every submitted route**, including
catastrophic and incomplete-evidence routes, before safe-only normalized
summaries are computed.

## Outputs

The authoritative report remains:

```json
{
  "schema": "audio-extract/d0-r0-grouped-comparison-report/v3",
  "status": "COMPLETE_NO_PROMOTION_DECISION",
  "promotion_decision": null
}
```

It may report:

- route and safe-route coverage;
- abstention;
- catastrophic false-safe outcomes;
- incomplete required evidence;
- complete-route exact regret for safe routes;
- paired R0-D0 regret differences;
- status cross-tabulation;
- normalized temporal/frequency switching;
- fail-closed status dominance.

Promotion belongs to a separate preregistered policy.

## Current gate

Before v3 can be copied into PR #23:

1. exact-head Codex review;
2. focused v3 comparison and diagnostics tests;
3. complete repository pytest and Ruff;
4. compileall and diff check;
5. synthetic report round-trip through the closed v3 JSON Schema;
6. zero unresolved review threads on the authoritative v3 files.

No fitting, calibration run, GPU job, audio rendering, or production selector is
authorized by this artifact.
