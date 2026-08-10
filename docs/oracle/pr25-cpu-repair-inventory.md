# PR #25 CPU-only repair — reachability classification & retirement inventory

> Worktree branch: `implementer/pr25-cpu-repair-20260810`
> PR #25 head repaired from: `b44ad5f507b73034913d5e54c6f9f6039286844b`
> Ruff frozen base: `2254c8e1bb797c3cedaa765d00950e0e3d496aa2`
> Deterministic, CPU-only. No GPU / training / audio / fitting / calibration / promotion.

## 0. Situational finding (important context for the Oracle)

The full-suite failure set is **not introduced by PR #25**. PR #25 is an
**additions-only** branch (every path in `git diff <merge-base> HEAD` is `A`); the
merge-base *is* the Ruff frozen base `2254c8e`. Re-running the failing tests at
`2254c8e` reproduces them. **All 76 baseline failures are inherited from the base
branch** `bigoracle/counterfactual-risk-router-v0-20260809`, where the co-located
`*_v1/*_v2` tests were (mostly) authored *after* their implementation modules and
encode a contract the implementations never fully satisfied (API fields that were
never added, validations that were never implemented, or checks that were added
too strictly).

Baseline: `76 failed / 1017 passed`. After this repair: **`25 failed / 1067 passed`**
(51 failures resolved). The authoritative focused-v4, v3-partition, v3-lineage and
v4-schema/rejected-mutation gates were GREEN before and remain GREEN after every
change.

## 1. Reachability method

Reachable set = import closure of {authoritative v4 surface} ∪ {CI focused-test
seeds} ∪ {`tools/` CLI}, computed by AST over `audio_extract/`.

- Authoritative v4 surface: `counterfactual_risk_grouped_report_artifact_v4`,
  `grouped_diagnostics_v4`, `source_lineage_v3`, `route_partition_attestation_v2`,
  `inference_partition_manifest_v2`.
- `tools/` imports NONE of the counterfactual-risk modules (no CLI reach).
- No `fixtures/` dirs; `artifacts/d0-r0-*` are Markdown only (no code lineage).
- The authoritative **v3 source lineage is content-derived** (`source_lineage_v3`
  parses embedded report bytes); it does **not** import `evidence_bundle_v1` /
  `query_condition_v1` / `evidence_lineage_v1`.

"Identity-bearing" overrides pure import-reachability: modules that define types
named in public schemas or the P1 lineage chain (`DatasetManifestV2`,
`SourceFamilyRegistryV2`, `CellPartitionRegistry`, the counterfactual model) are
treated as **repair, never delete**, even when not imported by the v4 report path.

## 2. REPAIRED (reachable / identity-bearing) — 7 root causes, 51 failures

All repairs verified to keep the focused-v4 + schema gates green; each touched
file's Ruff finding count is unchanged vs the frozen base (no new lint).

| # | Root cause (module) | Reachability | Fix | Failures cleared |
|---|---|---|---|---|
| 1 | `objective_tolerance` missing from `FrozenRoutingPolicyV1` (`counterfactual_risk_routing_preflight_v1.py`) | REACHABLE (`…v4 → grouped_comparison_v3 → complete_route_evaluation_v1 → exhaustive_decoder_v1`) | Restored the identity-bearing `objective_tolerance` field (the v4 "frozen uniqueness tolerance") + validation ("strictly positive") + `identity_dict`. The exhaustive-mirror decoder reads `policy.objective_tolerance`, so the class was internally broken. | ~33 (exhaustive_decoder_v1, complete_route_evaluation_v1, + d0_teacher setup) |
| 2 | Registry rejected the same cell geometry across group-scoped partitions (`counterfactual_risk_cell_partition_v1.py`) | REACHABLE (v4 closure) | Changed the cross-partition alias check from global `cell → partition_key` to `(cell_sha256, exact_source_report_sha256) → partition_key`. Reconciles `test_repeated_cell_geometry_across_groups_remains_group_scoped` (same grid ⇒ allowed) with `test_registry_rejects_…cross_partition_cell_alias` (same source report re-labelled ⇒ rejected). | 8 (partitioned_prediction_v1) |
| 3 | Panel forbade a feature-allowed cell from carrying an unavailable risk head (`counterfactual_risk_partitioned_prediction_v1.py`) | REACHABLE (v4 closure) | Removed the over-strict allowed-branch; kept the feature-blocked invariant. `row_prediction_allowed` is a FEATURE gate; the fail-closed preflight (which reads per-head `available`) is designed to mark missing required heads infeasible. | (part of routing_preflight / complete_route_evaluation / d0_teacher) |
| 4 | `InferenceManifestV1.build` did not cross-check dataset vs feature-registry source commit (`counterfactual_risk_inference_contract_v1.py`) | REACHABLE (v4 closure) | Added the "different source commits" guard. | 1 |
| 5 | `test_all_feature_blocked_rows_never_call_the_student` built `second` with a non-zero feature at a to-be-blocked head (inconsistent test data vs the correct, separately-relied-upon canonical-zero invariant) (`tests/test_counterfactual_risk_student_inference_v1.py`) | REACHABLE | Zeroed the blockable head when the row is blocked; test assertions preserved. | 1 |
| 6 | Numeric domains accepted JSON/`np.bool_` in feature/risk vectors & `feature_count` (`counterfactual_risk_dataset_contract_v2.py`, `counterfactual_risk_dataset_io_v2.py`) | REACHABLE (`DatasetManifestV2` in v4 + P1 chain) | Reject booleans before float coercion (`_features`, `_risks`, IO `_reject_booleans`, top-level `feature_count`). | 4 |
| 7 | `source_family_v2` lacked: query_source mixture-only ancestry, decoded-mixture-PCM alias, closed-world discovery-manifest alias (`counterfactual_risk_source_family_v2.py`) | Identity-bearing (`SourceFamilyRegistryV2` = P1 chain) | Added query_source to the "terminate only at mixture_root" rule; added `artifact_pcm_sha256` and closed-world `discovery_manifest_sha256` uniqueness checks. | 4 (aliases_v2 ×2, query_provenance_v2 ×2) |

## 3. REMAINING FAILURES (25) — classification + the decisions gpt56 must own

### 3a. Unreachable, P1-entangled cluster — 13 failures — **BLOCKING (design)**

`counterfactual_risk_evidence_bundle_v1` (7) and `counterfactual_risk_query_condition_v1`
(6, test `…query_condition_registry_v1`) are **unreachable** except through
`counterfactual_risk_evidence_lineage_v1` — a PR-#25-added module that is an
**orphan**: 0 imports, 0 tests, 0 references anywhere in `audio_extract/`, `tests/`,
`tools/`, `docs/`, `schemas/`.

- The failing tests demand `QueryConditionCertificate(query_scope_sha256=…)`, a
  field **never present in the implementation** (`git log -S` empty). No module
  reads it. Making them pass is **feature completion**, not a mechanical repair.
- `evidence_lineage_v1` is the HEAD commit "bind held-out lineage to
  EvidenceBundleV1" — i.e. the PR's **object-based P1 approach ("approach 2")**,
  which *competes with* the authoritative content-derived `source_lineage_v3`
  ("approach 1", which passes). Its docstring explicitly argues against pure
  report-projection authority.

**Decision required (gpt56):** retire the cluster
(`evidence_lineage_v1` + `evidence_bundle_v1` + `query_condition_v1` + their tests,
together — rule 2) in favour of the authoritative content-derived `source_lineage_v3`;
**or** complete/​integrate approach-2 (implement `query_scope_sha256`, wire and test
`evidence_lineage_v1`). This is a P1 design decision + touches lineage code, so per
the P1 contract and the guardrails I did **not** fabricate the field, delete lineage
code, or force it. See §5.

### 3b. Unreachable, standalone — 5 failures — **flag (retire vs invest)**

`counterfactual_risk_d0_structured_teacher_v1` (5). Unreachable (nothing imports it;
explicitly "an offline teacher contract … must never be used by the inference API").
Remaining failures are **deep behavioural drift** in the structured per-cell margin /
feasibility computation (`[[False,True]]==[[False,False]]`, `READY_FOR_D0_TRAINING`
vs `UNAVAILABLE_…_MARGIN_CELLS`, regex), not a field/API fix. Rule 2 would authorise
retiring it (impl+test); but it is not clearly *superseded* (it is the D0 arm's
target generator), so per "prefer repair / flag rather than guess" I flag it.

### 3c. Reachable / identity-bearing but genuine design ambiguity — 7 failures — **BLOCKING/flag**

- **Frozen-panel contradiction — 2** (`source_family_v1` ×1, `source_family_v2` ×1).
  Both build a **multi-family** v1 `DatasetManifest` (per-family candidate panels)
  and expect it valid. But `dataset_contract_v1`'s **own passing** test
  (`…:129`, "disagree on frozen panel") requires all rows to share one frozen
  candidate panel and rejects even reversed-candidate rows. Same `DatasetManifest.build`,
  contradictory contracts. Cannot satisfy both without a semantic ruling: is a v1
  dataset single-panel (frozen) or multi-family (per-family candidates)?
- **`dataset_exhaustive_v2` — 2.** One is a fail-closed threshold-state check; the
  other (`test_inference_projection_keyset_is_disjoint_from_offline_truth_keyset`)
  demands `inference_records_v2` DROP fields (`metric_contract_sha256`,
  `route_policy_sha256`, `feature_sha256`, …) that `InferenceRowV1` currently emits
  and that the **v4 path consumes** — a structural change with v4-regression risk.
- **`counterfactual_risk_model` — 1.** `test_query_free_default_contract_is_valid`:
  a query-free contract is rejected ("query-disabled models require zero query
  dimensions"). Needs a model-contract semantics decision (query-dim defaulting).
- **`group_scope_v1` — 1.** `test_bound_student_refuses_…`: the *valid-setup*
  `student.validate()` fails "guaranteed group coverage is below the target" — a
  conformal-calibration coverage-math question, not a mechanical fix.
- **`grouped_comparison_v1` — 1.** `test_oracle_unavailable_status_must_agree_between_arms`
  builds a ROUTE_SAFE R0 row with `exact_oracle_objective=None` and expects a
  grouped-level "availability differs" error, but the lower-level
  `complete_route_evaluation` "safe route lacks exact objective/regret" fires first.
  Needs a validation-ordering / status-semantics ruling.

## 4. Ruff status vs frozen base `2254c8e`

- Base `2254c8e`: **510** findings (`ruff 0.16.2`, `ruff check .`).
- PR head `b44ad5f`: **555** (the PR's own +45, all inside PR-added files).
- This repair branch: **555** — **my edits added ZERO findings** (verified per
  touched file: base-count == branch-count for every file I changed).
- Rule-5 requirement "zero in PR-added files ⇒ repo ≤ 510" (removing the +45 inside
  the new v1–v4 files) is **not yet done** — it is deliberately deferred because the
  set of PR-added files to clean depends on the §3a retirement decision (retired
  files' findings vanish; cleaning files that may be deleted is wasted). No
  `--fix`/exclude has been applied so nothing is hidden.

## 5. P1 lineage status (honest)

The **authoritative** P1 mechanism — v3 source lineage resolved **from report
content** (`counterfactual_risk_source_lineage_v3`, which embeds each exact
verification report as canonical bytes and derives M/A/V parents by parsing them) —
is **implemented and its focused gate passes** (`test_…source_lineage_v3`,
`test_…inference_partition_v3`). It does not synthesise new normalized source docs.

The unresolved P1 item is **not** the content-derived path; it is the **coexistence**
of the PR's alternative object-based binding (`evidence_lineage_v1` → the frozen
`EvidenceBundleV1 → DatasetManifestV2 → SourceFamilyRegistryV2 / CellPartitionRegistry`
chain), which is orphaned + untested and whose only-dependencies' tests demand an
unimplemented `query_scope_sha256`. Closing it is the §3a design decision. Per the
P1 contract I did **not** fabricate lineage or delete lineage code; status =
**blocked on gpt56's approach-1-vs-2 ruling.**
