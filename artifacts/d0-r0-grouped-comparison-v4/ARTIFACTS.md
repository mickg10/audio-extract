# D0/R0 grouped-comparison artifact index v4

## GitHub location

```text
repository: mickg10/audio-extract
branch:     bigoracle/d0-r0-grouped-comparison-artifacts-20260810
draft PR:   #25
base:       bigoracle/counterfactual-risk-router-v0-20260809
```

This is a separate research-artifact branch. It is not the production branch and
contains no arm-promotion decision.

## Authoritative implementation

```text
audio_extract/counterfactual_risk_grouped_report_artifact_v4.py
audio_extract/counterfactual_risk_inference_partition_manifest_v2.py
audio_extract/counterfactual_risk_route_partition_attestation_v2.py
audio_extract/counterfactual_risk_grouped_diagnostics_v4.py
```

## Authoritative external schema

```text
schemas/d0-r0-grouped-comparison-report-v4.schema.json
```

The schema reuses the closed v3 nested definitions and adds paired D0/R0
constraints.

## Focused tests

```text
tests/test_counterfactual_risk_grouped_report_artifact_v4.py
tests/test_counterfactual_risk_inference_partition_v2.py
tests/test_d0_r0_grouped_comparison_schema_v4.py
```

## Design record

```text
docs/oracle/d0-r0-grouped-held-out-comparison-v4.md
artifacts/d0-r0-grouped-comparison-v4/VALIDATION.md
```

## Artifact lineage

```text
validated v3 comparison object
→ immutable paired v4 UTF-8 report bytes

exact risk evidence + partition source report
→ content-bearing exact-source lineage
→ pre-execution partition input manifest
→ held-out arm inference run
→ route/abstention artifact and submission
→ resolved route/partition attestation
→ normalized diagnostics v4
```

## Non-scope

```text
no model fitting
no calibration execution
no GPU job
no audio rendering
no production selector
no winner or promotion decision
```

The verified median/MDX release remains the operational fallback.
