# D0/R0 grouped-comparison artifact index

## GitHub location

```text
repository: mickg10/audio-extract
branch:     bigoracle/d0-r0-grouped-comparison-artifacts-20260810
draft PR:   #25
base:       bigoracle/counterfactual-risk-router-v0-20260809
```

This is a separate research-artifact branch. It is not the production branch and
must not be interpreted as a promotion decision.

## Authoritative source contracts

```text
audio_extract/counterfactual_risk_grouped_comparison_v3.py
audio_extract/counterfactual_risk_grouped_diagnostics_v3.py
audio_extract/counterfactual_risk_route_partition_attestation_v1.py
schemas/d0-r0-grouped-comparison-report-v3.schema.json
```

## Authoritative tests

```text
tests/test_counterfactual_risk_grouped_comparison_v3.py
tests/test_counterfactual_risk_grouped_diagnostics_v3.py
tests/test_counterfactual_risk_route_partition_attestation_v1.py
```

## Design and review records

```text
docs/oracle/d0-r0-grouped-held-out-comparison-v3.md
docs/oracle/d0-r0-route-partition-attestation-v1.md
docs/oracle/pr25-p1-repair-contract.md
artifacts/d0-r0-grouped-comparison-v3/VALIDATION.md
```

## Lineage represented by the artifacts

```text
preregistered D0/R0 arm
→ held-out inference run
→ route or abstention output
→ route artifact manifest
→ route submission
→ exact complete-route evaluation
→ grouped held-out report
→ complete certified cell partition
→ per-route route/partition attestation
→ certified non-promoting diagnostics envelope
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
