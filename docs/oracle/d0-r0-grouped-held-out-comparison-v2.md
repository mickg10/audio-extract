# D0/R0 grouped held-out comparison v2

> **Separate artifact branch:** `bigoracle/d0-r0-grouped-comparison-artifacts-20260810`  
> **Stacked draft:** PR #25  
> **Authoritative grouped comparison:** `counterfactual_risk_grouped_comparison_v2.py`  
> **V1 status:** review history only

## Why v2 exists

Codex found four adjacent weaknesses in the first aggregation contract:

1. a complete-route evaluation could be moved between two held-out units that
   happened to reuse the same exact evidence;
2. a sibling arm SHA did not prove the evaluated submission came from the
   preregistered artifact/checkpoint/run;
3. non-oracle statuses could carry a missing oracle objective and still enter
   coverage denominators;
4. Python equality allowed type-changing summary mutations such as `2.0 == 2`
   and `True == 1`.

V2 closes all four without changing the scientific comparison or introducing a
promotion decision.

## Content-bearing route provenance

Every work/arm record now contains `ArmRouteProvenanceV2`, which binds:

```text
held-out unit SHA
group-family SHA
query condition
query-quality stratum
arm ID
frozen arm-run SHA
artifact-manifest SHA
equal-compute budget SHA
checkpoint-rule SHA
inference-policy SHA
source commit
route-artifact SHA
verifier commit
the exact ArmRouteSubmissionV1 object and its semantic SHA
```

`WorkArmEvaluationV2` verifies that:

```text
complete_route_evaluation.submission_sha256
    == route_provenance.submission.sha256
```

and that the evaluation's exact evidence, policy, arm, unit, query, and source
family all match the frozen preregistration.

Thus a route cannot be reassigned to another work, singer, query, source family,
model checkpoint, or arm by changing a sibling label.

## Oracle availability semantics

V2 treats oracle availability as an explicit invariant:

```text
status == ORACLE_UNAVAILABLE
    iff exact_oracle_objective is null

all other statuses
    require a finite nonnegative exact_oracle_objective
```

D0 and R0 must carry the same exact oracle objective for one held-out unit.
Coverage denominators are therefore never increased by an evaluation whose
oracle was not actually available.

## Strict summaries

`ArmSummaryV2` and `PairwiseSummaryV2` require exact field types and reject
booleans masquerading as integers or integers/floats crossing the schema
boundary.

They independently enforce:

```text
oracle_available + oracle_unavailable == unit_count
route + abstention == oracle_available
safe + catastrophic + incomplete == route
paired regret classes == both_safe
status cross-tab count == unit_count
```

Rate fields are recomputed from counts. Regret and switch aggregates are present
only when their source rows exist. Every stored summary is recomputed from the
work-level evidence during validation.

## Regression coverage

The v2 focused suite covers:

- unit swaps when exact evidence is intentionally shared;
- forged arm-run and route-artifact identities;
- a complete evaluation naming another submission;
- missing oracle objectives for ABSTAIN, catastrophic, and incomplete rows;
- an ORACLE_UNAVAILABLE row carrying an invented objective;
- float and bool summary-count substitutions;
- cross-arm oracle-objective drift;
- missing and duplicate matrix cells;
- all-oracle-unavailable `None` coverage rather than fake zero coverage;
- recomputed summary tamper refusal.

The isolated interface-compatible harness passed:

```text
13 passed
```

This remains focused artifact validation. Exact-head full repository validation
is required before v2 replaces v1 in PR #23.

## Output semantics

The v2 report remains strictly non-promoting:

```text
schema: audio-extract/d0-r0-grouped-comparison-report/v2
status: COMPLETE_NO_PROMOTION_DECISION
promotion_decision: null
```

A separate, preregistered promotion policy may consume a verified v2 report.
The report itself never chooses D0, R0, a separator model, or a production route.
