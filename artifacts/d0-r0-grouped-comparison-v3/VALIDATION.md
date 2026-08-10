# D0/R0 grouped comparison v3 — validation record

## Branch

```text
bigoracle/d0-r0-grouped-comparison-artifacts-20260810
```

## Authoritative files

```text
audio_extract/counterfactual_risk_grouped_comparison_v3.py
tests/test_counterfactual_risk_grouped_comparison_v3.py

audio_extract/counterfactual_risk_grouped_diagnostics_v3.py
tests/test_counterfactual_risk_grouped_diagnostics_v3.py

audio_extract/counterfactual_risk_route_partition_attestation_v1.py
tests/test_counterfactual_risk_route_partition_attestation_v1.py

schemas/d0-r0-grouped-comparison-report-v3.schema.json
docs/oracle/d0-r0-grouped-held-out-comparison-v3.md
docs/oracle/d0-r0-route-partition-attestation-v1.md
docs/oracle/pr25-p1-repair-contract.md
```

Earlier versions remain review history only.

## Current evidence status

```text
implementation uploaded: yes
focused exact-head repository run: pending implementer
complete repository pytest: pending implementer
repository-wide Ruff: pending implementer
compileall/diff check: pending implementer
synthetic schema round-trip: pending implementer
Codex exact-head review: requested after upload
promotion decision: none
```

No pass count is claimed here until the tests are executed in the repository's
real `uv` environment against one exact branch head.

## Required commands

```bash
git fetch origin bigoracle/d0-r0-grouped-comparison-artifacts-20260810
git switch --detach <exact-head>

uv run --extra dev pytest -q \
  tests/test_counterfactual_risk_grouped_comparison_v3.py \
  tests/test_counterfactual_risk_grouped_diagnostics_v3.py \
  tests/test_counterfactual_risk_route_partition_attestation_v1.py

uv run --extra dev pytest -q
uv run --extra dev ruff check .
python -m compileall -q audio_extract tests
python -m json.tool \
  schemas/d0-r0-grouped-comparison-report-v3.schema.json >/dev/null
git diff --check
```

## Blocking semantics under test

- route artifact manifest transitively binds frozen arm and held-out inference;
- route output exactly matches the submitted route/abstention;
- submission names the route artifact manifest;
- complete-route evaluation names the submission;
- route/abstention statuses agree;
- every matrix cell has unique inference-run, output, artifact, and submission
  identities;
- geometry uses the complete existing `CellPartitionCertificate`, including
  row-major entries, rational measures, range hashes, and exact source report;
- every registry certificate is bound exactly once to a held-out unit;
- every route artifact is independently attested to the selected partition,
  inference-input manifest, grid dimensions, and label identity;
- route shapes must equal the certified `(time_cell_count, band_count)`;
- impossible switch counts are rejected for safe, catastrophic, and incomplete
  route statuses;
- stored summary types are validated before recomputed equality;
- schema-only validation pins D0/R0 summaries and couples route/abstention status
  to label presence;
- external v3 JSON is closed and round-trips from the real Python classes.

## Local repair-bundle reference

```text
SHA-256:
3fdf14900ccf1841cb23bb01c44f65ff30d5e300c5a41c2e7ca7664eea9c968a
```

The executable source of truth is the GitHub branch, not the local ZIP.
