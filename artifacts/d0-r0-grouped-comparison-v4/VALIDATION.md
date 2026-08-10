# D0/R0 grouped comparison v4 — validation record

## Branch and exact-head rule

```text
repository: mickg10/audio-extract
branch:     bigoracle/d0-r0-grouped-comparison-artifacts-20260810
draft PR:   #25
```

The validator must detach the exact PR head named in the accompanying review or
validation request. Results from an ancestor or a subsequently moved head are
stale and do not satisfy this record.

## Authoritative files

```text
audio_extract/counterfactual_risk_grouped_report_artifact_v4.py
audio_extract/counterfactual_risk_inference_partition_manifest_v2.py
audio_extract/counterfactual_risk_source_lineage_v3.py
audio_extract/counterfactual_risk_route_partition_attestation_v2.py
audio_extract/counterfactual_risk_grouped_diagnostics_v4.py

schemas/d0-r0-grouped-comparison-report-v4.schema.json

tests/test_counterfactual_risk_grouped_report_artifact_v4.py
tests/test_counterfactual_risk_inference_partition_v2.py
tests/test_counterfactual_risk_source_lineage_v3.py
tests/test_d0_r0_grouped_comparison_schema_v4.py
```

## Current evidence status

```text
implementation uploaded: yes
focused exact-head repository run: pending implementer / GitHub Actions
complete repository pytest: pending implementer / GitHub Actions
repository-wide Ruff: pending implementer / GitHub Actions
compileall/diff check: pending implementer / GitHub Actions
v4/v3 schema round-trip: pending implementer / GitHub Actions
Codex exact-head review: requested after upload
promotion decision: none
```

No pass count is claimed until these commands run against one exact GitHub head
in the repository's real environment.

## Required commands

```bash
git fetch origin bigoracle/d0-r0-grouped-comparison-artifacts-20260810
git switch --detach <exact-head-from-current-review-request>

uv run --extra dev pytest -q \
  tests/test_counterfactual_risk_grouped_report_artifact_v4.py \
  tests/test_counterfactual_risk_inference_partition_v2.py \
  tests/test_counterfactual_risk_source_lineage_v3.py

# The schema test requires jsonschema without changing the frozen runtime path.
uv run --with 'jsonschema>=4.23' --extra dev pytest -q \
  tests/test_d0_r0_grouped_comparison_schema_v4.py

uv run --extra dev pytest -q
uv run --with 'ruff>=0.9' ruff check .
python -m compileall -q audio_extract tests
python -m json.tool \
  schemas/d0-r0-grouped-comparison-report-v3.schema.json >/dev/null
python -m json.tool \
  schemas/d0-r0-grouped-comparison-report-v4.schema.json >/dev/null
git diff --check
```

## Required semantic checks

- issue one immutable paired v4 artifact from real repository classes;
- validate it with the v4 schema while registering the sibling v3 schema;
- require the public root to carry both `source_v3_verifier_commit` and the
  frozen `paired_regret_tolerance`;
- mutate the original NumPy route labels and prove the issued bytes/SHA remain
  unchanged while source revalidation fails;
- reject reassignment of a complete D0 or R0 branch to another paired unit;
- reject nested D0/R0 arm substitutions;
- reject route evaluation backed by abstention lineage and the converse;
- reject missing exact-oracle objectives for every oracle-available status;
- reject critical/secondary violation-kind swaps;
- reject nonpositive above-threshold values at the schema boundary;
- reject `critical_above_threshold` unless semantic `value > threshold` holds;
- reject one-sided oracle-unavailable paired units and status-cross-tab entries;
- derive shared truth/M/A/V parents from the two content-bearing source reports,
  not caller-supplied sibling hashes;
- accept different exact-risk and partition-source report hashes only when those
  reports share the frozen truth/PCM parents;
- accept unused alternative certified resolutions;
- reject D0/R0 selecting different partitions for the same held-out unit;
- reject a run whose input-manifest SHA does not equal the content-bearing
  pre-execution manifest;
- reject route shapes or switch counts outside the selected certificate grid;
- reconstruct and verify the embedded source-v3 report SHA from the paired
  public payload and stored source verifier commit;
- recompute all arm and pairwise summaries from the paired work evidence.

## Blocking state

Any failed, skipped, or stale-head check keeps PR #25 draft and isolated.

```text
no D0/R0 fitting
no conformal calibration execution
no GPU job
no audio rendering
no merge into PR #23
no production selector
no promotion decision
```

## Portable repair-bundle reference

```text
SHA-256:
3fdf14900ccf1841cb23bb01c44f65ff30d5e300c5a41c2e7ca7664eea9c968a
```

The executable source of truth is the GitHub branch, not the portable ZIP.
