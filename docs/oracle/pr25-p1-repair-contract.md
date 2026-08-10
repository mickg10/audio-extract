# PR #25 provenance repair contract

> Separate branch: `bigoracle/d0-r0-grouped-comparison-artifacts-20260810`  
> Draft PR: `#25`  
> Scope: grouped held-out D0/R0 comparison artifacts only

## Review findings

The first grouped-comparison implementation had two substantive attribution
weaknesses:

1. an evaluation could be reassigned to another held-out unit when both units
   reused the same exact evidence hashes;
2. the expected arm identity was asserted beside the submission rather than
   proven through a route artifact produced by that arm.

A later review also found that a `ROUTE` evaluation could be paired with an
`ABSTAIN` submission, or vice versa.

## Binding repair

The authoritative v3 contract requires:

```text
FrozenArmInferenceRunV3
    binds unit + query + arm + compute/checkpoint/inference contracts

RouteOutputV3
    binds the actual route or abstention payload

RouteArtifactManifestV3
    embeds the verified inference run and route output

ArmRouteProvenanceV3
    requires submission.route_evidence_sha256 == route_manifest.sha256

WorkArmEvaluationV3
    requires evaluation.submission_sha256 == submission.sha256
    and requires evaluation status to match submission status
```

The route artifact manifest, not a free-standing sibling hash, is the transitive
parent of the evaluated submission.

## Required anti-substitution checks

Every work/arm row must reject:

- a different held-out unit, source family, query, or quality stratum;
- a different arm/checkpoint artifact manifest;
- a different compute budget, checkpoint rule, or inference policy;
- a different dataset, split, test subset, panel, feature, or metric contract;
- a different inference-input manifest or runner bundle;
- a route output different from the submitted labels/status/reason;
- a submission that does not name the route artifact manifest;
- an evaluation that does not name that submission;
- a route evaluation paired with an abstaining submission;
- an abstention evaluation paired with a route submission.

`ORACLE_UNAVAILABLE` may preserve either submitted route or abstention because no
exact comparison is possible.

## Matrix-level anti-reuse

The completed matrix rejects reuse of one:

```text
inference-run identity
route-output identity
route-artifact-manifest identity
submission identity
```

across held-out unit/arm cells.

Identical label patterns are not forbidden; two independent works may
legitimately receive the same route pattern.

## Diagnostics repair

The normalized diagnostics now require an embedded exact partition certificate
per unit, binding:

```text
unit and group family
exact panel / preflight / oracle
spectral grid
partition contract and artifact
cell/band counts
source and verifier commits
```

Every submitted route status is checked against available time/frequency
boundaries before safe-only summaries are aggregated.

## Validation gate

The branch remains draft until the authoritative v3 files pass:

```bash
uv run --extra dev pytest -q \
  tests/test_counterfactual_risk_grouped_comparison_v3.py \
  tests/test_counterfactual_risk_grouped_diagnostics_v2.py
uv run --extra dev pytest -q
uv run --extra dev ruff check .
python -m compileall -q audio_extract tests
git diff --check
```

No training, GPU job, audio render, production selection, or promotion decision
is authorized by this document.

## External artifact bundle

The local repair bundle prepared before integration had SHA-256:

```text
3fdf14900ccf1841cb23bb01c44f65ff30d5e300c5a41c2e7ca7664eea9c968a
```

Its substantive contract is now represented by version-controlled GitHub files
on this branch; the binary ZIP itself is not required for repository execution.
