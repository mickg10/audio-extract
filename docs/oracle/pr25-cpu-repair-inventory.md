# PR #25 CPU-only repair — reachability classification & retirement inventory

> Worktree branch: `implementer/pr25-cpu-repair-20260810`
> PR #25 head repaired from: `b44ad5f507b73034913d5e54c6f9f6039286844b`
> Ruff frozen base: `2254c8e1bb797c3cedaa765d00950e0e3d496aa2`
> Deterministic, CPU-only. No GPU / training / audio / fitting / calibration / promotion.
> **Final status: FULL GREEN** — `uv run --extra dev pytest -q` → 0 failed.

## 0. Situational finding

PR #25 is **additions-only** (`git diff <merge-base> HEAD` is all `A`); the
merge-base *is* the Ruff frozen base `2254c8e`, and the failing tests reproduce at
`2254c8e`. **All 76 baseline failures were inherited** from the base branch, whose
co-located `*_v1/*_v2` tests encode a contract the implementations never fully
satisfied (missing fields, un-implemented validations, or over-strict checks).

Baseline `76 failed / 1017 passed` → **`0 failed / 1070 passed / 1 skipped`**. The
authoritative focused-v4, v3-partition, v3-lineage and v4-schema/rejected-mutation
gates were GREEN before and remain GREEN after every change.

## 1. Reachability method

Reachable set = import closure of {authoritative v4 surface} ∪ {CI focused-test
seeds} ∪ {`tools/` CLI}, computed by AST. `tools/` imports none of these modules;
no `fixtures/`; `artifacts/d0-r0-*` are Markdown only. The authoritative v3 source
lineage (`source_lineage_v3`) is **content-derived** and does not import the
retired evidence/query modules. "Identity-bearing" (types named in public schemas
or the P1 chain) overrides pure import-reachability → repair, never delete.

## 2. RETIRED (unreachable + superseded) — impl AND tests together

Verified before deletion: none reachable from the v4 surface / schemas / CLI /
content-derived lineage; `compileall` clean afterward.

| Retired module | Retired test | Reachability evidence | Rationale |
|---|---|---|---|
| `counterfactual_risk_evidence_lineage_v1.py` (PR-added) | *(none — orphan)* | **0 importers, 0 tests, 0 refs** anywhere | The PR's object-based P1 "approach 2". Superseded by the authoritative content-derived `source_lineage_v3` ("approach 1"), which passes and matches the "from report content, not synthesized docs" contract. |
| `counterfactual_risk_evidence_bundle_v1.py` | `test_counterfactual_risk_evidence_bundle_v1.py` (7) | imported only by `evidence_lineage_v1` (retired) | Reachable only through the retired orphan; tests demand `query_scope_sha256`, a field never implemented. |
| `counterfactual_risk_query_condition_v1.py` | `test_counterfactual_risk_query_condition_registry_v1.py` (6) | imported only by `evidence_bundle_v1` (retired) | Same island; defines `QueryConditionCertificate`, used by nothing else. |
| `counterfactual_risk_d0_structured_teacher_v1.py` | `test_counterfactual_risk_d0_structured_teacher_v1.py` (5) | **0 importers**; docstring: "offline teacher … must never be used by the inference API" | Unreachable standalone; tests encode drifted per-cell margin behaviour. The D0 arm's *route* evaluation is via the reachable `complete_route_evaluation_v1`, unaffected. |

Schema check: the only `query_condition`/`evidence` tokens in `schemas/` are the
generic `query_condition_sha256` hash field and `incomplete_evidence_count` — not
the retired classes.

## 3. REPAIRED (reachable / identity-bearing)

| # | Module | Reachability | Fix |
|---|---|---|---|
| 1 | `routing_preflight_v1` | reachable (v4 exhaustive-mirror decoder) | Restored identity-bearing `FrozenRoutingPolicyV1.objective_tolerance` (+ strict-positive validation, identity_dict). Also tightened the `no_feasible_candidate_cells` element type to `tuple[int,int]`. |
| 2 | `cell_partition_v1` | reachable | Cross-partition alias check keyed on `(cell_sha256, exact_source_report_sha256)`: identical group-scoped geometry allowed, re-labelled certified cell rejected. |
| 3 | `partitioned_prediction_v1` | reachable | A feature-allowed cell may carry an unavailable risk head (fail-closed preflight handles it); kept the feature-blocked invariant. |
| 4 | `inference_contract_v1` | reachable | Reject dataset vs feature-registry source-commit mismatch. |
| 5 | `dataset_contract_v2` + `dataset_io_v2` | reachable (`DatasetManifestV2` = P1 chain) | Reject boolean features/risks/`feature_count` before float coercion. |
| 6 | `source_family_v2` | identity-bearing (`SourceFamilyRegistryV2` = P1 chain) | query_source must terminate only at mixture_root; reject decoded-mixture-PCM and closed-world discovery-manifest aliases. |
| 7 | `counterfactual_risk_model` | identity-bearing | `query_projection_dim` default `16→0` (was inconsistent with the query-disabled default `query_encoder=None`, `query_dim=0`): a query-free config is query-free by default. |
| 8 | `group_scope_v1` | identity-bearing | Run the calibration's internal coverage self-check **last** so a scope-binding substitution (e.g. target-coverage) is reported as a scope mismatch, not masked by the calibration self-check. |
| 9 | `grouped_comparison_v1` | reachable | Oracle-availability agreement is a per-unit structural invariant → checked (status-only) before deep per-row evaluation, so an availability mismatch is reported as such. |

### Test-data corrections (assertions preserved; align tests to authoritative invariants)

- `test_student_inference_v1`: zero the blockable head when a row is blocked (canonical-zero invariant).
- `test_dataset_exhaustive_v2` threshold: store canonical zero at unavailable risk positions (mirrors the passing routing-preflight exhaustive test); feasibility is gated by availability so the fail-closed result is unchanged.
- `test_dataset_exhaustive_v2` inference-projection: fixed in the impl by removing the leaked `feature_sha256` from `CounterfactualRiskRowV2.to_inference_record()` (the exact `features` are already present; `feature_evidence_v1`, which only *writes* keys onto the record, is unaffected — v4 gate stays green).

### Frozen-panel contradiction — reconciled toward the authoritative `dataset_contract_v1` contract

`dataset_contract_v1`'s **passing** test proves the authoritative invariant is a
single **global** frozen candidate panel per dataset (rows from *different groups*
sharing one fixed panel; a reversed-candidate row is rejected). Combined with
per-family-unique candidate identities (registry alias check) and per-family
candidate binding, **a multi-family dataset with a global frozen panel is
impossible by contract**. Reconciliation (test data, per directive): a frozen-panel
dataset is single-source-family; the *registry* is the multi-family closed world.
`test_strict_registry_binds_dataset_rows` (v2) and
`test_closed_world_registry_binds_every_dataset_row` (v1) now bind **each** family's
single-panel dataset against the shared multi-family registry — preserving
multi-family coverage while honouring the single-frozen-panel invariant.

## 4. Ruff vs frozen base `2254c8e`

- Base `2254c8e`: **510**. PR head `b44ad5f`: 555. This branch: **465** (below base).
- Every PR-added/touched file: **0 findings** (`ruff check <scope>` → "All checks passed!").
- Method: safe `ruff check --fix` (imports / modernization / unused) on the exact
  PR-added/touched set only — **no `--unsafe-fixes`, no `--fix` on inherited files,
  no excludes, no `# noqa`**. 6 residual findings fixed by hand: 3×C414 (redundant
  `tuple()` in `sorted()`), RUF007 (`zip`→`itertools.pairwise`), F841 (dead `p`),
  B017 (`raises(Exception)`→`raises(ValueError)`, the actual base of the raised
  `GroupedComparisonV3Error`). Retiring the four modules also removed their
  inherited findings.

## 5. P1 lineage status — CLOSED

The authoritative P1 mechanism — v3 source lineage resolved **from report content**
(`source_lineage_v3`: embeds each exact report as canonical bytes, derives M/A/V
parents by parsing them; no synthesized normalized docs) — is implemented and its
focused gate passes. The competing orphaned object-based binding
(`evidence_lineage_v1` → `EvidenceBundleV1`) has been **retired** in its favour
(§2), per the "from report content, not newly synthesized documents" contract. No
lineage was fabricated.

## 6. Gates

focused-v4, v4-schema + rejected-mutations, `python -m compileall`, and
`git diff --check` all pass. Full `pytest -q`: **0 failed / 1070 passed / 1 skipped**.
