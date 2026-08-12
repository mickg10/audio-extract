# PR #25 CPU-only repair — reachability, retirement & Oracle-decision inventory

> Worktree branch: `implementer/pr25-cpu-repair-20260810`
> PR #25 head repaired from: `b44ad5f507b73034913d5e54c6f9f6039286844b`
> Ruff frozen base: `2254c8e1bb797c3cedaa765d00950e0e3d496aa2`
> Deterministic, CPU-only. No GPU / training / audio / fitting / calibration / promotion / merge.
> **Status: FULL local GREEN**; PR-added/touched files Ruff-clean; repo Ruff ≤ 510.

This revision implements gpt56's binding decisions (issue #1 comments 5247359793 /
5248527429) exactly, after the previous green was rejected for violating them.

**Latest head** additionally implements gpt56's three adjudicated decisions on
draft PR #26 (issue #1 comment 5267006554):

1. **Ruff hard ratchet in CI** (§6): the workflow's single `ruff check .` step is
   replaced by two *binding* checks with the pinned Ruff `0.16.2` — (a) zero
   findings on the deterministically-enumerated PR-touched Python files, and
   (b) a repo-wide ratchet requiring `head_count (466) <= base_count (510)`, with
   the base count re-derived in a throwaway base-SHA worktree and asserted equal
   to the frozen `510` (tamper-evident). Base SHA, Ruff version, base/head counts
   and the touched-file list are written to `validation-logs/`; malformed/missing
   baseline or an unenumerable diff fails closed.
2. **Frozen panel** (§4): gpt56 CONFIRMED the recipe-level-identity
   implementation; verified the coded `DatasetManifest.validate` matches it
   exactly (ordered candidate-RECIPE panel; per-family immutable artifacts;
   availability masks; reorder/substitute changes dataset identity). No change.
3. **D0 teacher contract split** (§3): the per-panel teacher now emits
   `READY_FOR_GROUPED_ASSEMBLY` (never a training-ready status) and binds its
   exact group identity; a new grouped dataset/training-manifest boundary
   (`counterfactual_risk_grouped_training_manifest_v1`) is the *only* emitter of
   `READY_FOR_D0_TRAINING`, gated by the finite-group inequality
   `ceil((n_groups + 1) * (1 - alpha)) <= n_groups` with `alpha`/`n_groups`
   identity-bearing; unsatisfiable → uncertifiable / fail-closed.

## 1. P1 provenance — BOTH content-derivation AND frozen-bundle membership

`counterfactual_risk_source_lineage_v3.ExactSourceLineageV3` keeps content
derivation (M/A/V parents parsed from canonical report bytes; report byte-hash
must equal the frozen unit/certificate identity) **and** now adds a bundle-
membership resolver `validate_bundle_membership(bundle, exact_evidence_bundle_sha256,
unit, certificate)` that binds ALL of:

- **`exact_evidence_bundle_sha256`**: the referenced `EvidenceBundleV1` must exist
  and `bundle.sha256` must equal the declared identity;
- **exact-report membership**: the partition-source report byte-hash equals the
  member certificate's `exact_source_report_sha256` (and that certificate is a
  registry member, §below), so the exact report is a bundle member;
- **M/A/V parent membership**: the derived mixture/accompaniment/vocal parents must
  equal the frozen `SourceFamilyCertificateV2` roots the bundle certifies;
- **held-out unit + source-family membership**: the unit's group family is a member
  of the bundle dataset and its source family is a registry member;
- **partition-certificate membership**: the selected `CellPartitionCertificate` is a
  member of the bundle's `CellPartitionRegistry`.

A synthesized-but-internally-valid report that is NOT a member fails closed.

**Adversarial provenance tests** (in `tests/test_counterfactual_risk_source_lineage_v3.py`,
part of the focused v3-lineage CI gate) — each asserts rejection:

- `test_bundle_member_lineage_resolves_content_and_membership` (positive control);
- `test_synthesized_internally_valid_nonmember_report_is_rejected` — (a) a fully
  internally-valid synthesized report that is not a bundle member;
- `test_copied_reused_evidence_hash_is_rejected` — (b) a copied/reused evidence hash;
- `test_substituted_partition_certificate_is_rejected` — (c) an alternate/substituted
  partition certificate;
- `test_mismatched_bundle_identity_is_rejected` — (d) a mismatched bundle.

`EvidenceBundleV1`, `DatasetManifestV2`, `SourceFamilyRegistryV2`,
`CellPartitionRegistry` are retained/repaired as the identity-bearing primitives the
resolver binds against. `EvidenceBundleV1` was made **query-free** (the query
registry/conditioning was removed; the study is query-free with zero query dims).

## 2. RETIRED (deleted impl AND tests together) — exact paths

- `audio_extract/counterfactual_risk_evidence_lineage_v1.py` — orphan object-based
  P1 "approach 2" (0 imports/refs/tests); superseded by the content-derived +
  membership `source_lineage_v3`; not a competing lineage authority.
- `audio_extract/counterfactual_risk_query_condition_v1.py` and
  `tests/test_counterfactual_risk_query_condition_registry_v1.py` — query
  conditioning; the study is query-free (no fabricated `query_scope_sha256`). It was
  imported only by `evidence_bundle_v1`, now query-free.

(`evidence_lineage_v1` had no test file.)

## 3. PRESERVED + REPAIRED

- **`counterfactual_risk_d0_structured_teacher_v1`** (+ its test) — kept OUT of the
  inference import path (imports only decoder/preflight/panel types; nothing in the
  inference runtime imports it). Behavioral-margin contract unchanged: a cell is
  teacher-eligible only when every required critical/secondary exact risk is
  AVAILABLE; feasible = every required risk ≤ its frozen threshold; behavioral
  margin = min signed `(threshold − risk)` across required constraints (negative
  margins retained/informative). **gpt56 decision #3 (contracts split):** the
  per-panel teacher's ready status is renamed `READY_FOR_D0_TRAINING` →
  **`READY_FOR_GROUPED_ASSEMBLY`** (module-level constant `READY_FOR_GROUPED_ASSEMBLY`);
  it now binds its **exact group identity** (`group_family_sha256`, derived from the
  panel, hashed into the teacher identity) and must NOT claim conformal/training
  coverage from one panel (one panel = one exchangeable group). The prior "flag"
  about the under-specified single-panel coverage gate is thereby RESOLVED.

- **NEW `counterfactual_risk_grouped_training_manifest_v1`** (+ its test) — the
  grouped dataset/training-manifest boundary and the **only** emitter of a
  training-ready status. `GroupedTrainingManifestV1.build(members, alpha=…)`
  aggregates `GroupedAssemblyMemberV1` records (each derived from a real
  `READY_FOR_GROUPED_ASSEMBLY` teacher via `from_teacher`, so group identity and the
  content-addressed teacher artifact always describe the same panel), counts the
  DISTINCT group identities `n_groups`, and applies the finite-group conformal
  inequality `ceil((n_groups + 1) * (1 − alpha)) <= n_groups`. Satisfiable →
  `READY_FOR_D0_TRAINING`; unsatisfiable → `UNCERTIFIABLE_INSUFFICIENT_INDEPENDENT_GROUPS`
  (fail-closed, never training-ready). `alpha` and `n_groups` are identity-bearing
  (in the hashed identity), plus `target_coverage`, `conformal_rank`, sorted
  `group_family_sha256s`, and member identities. Repeated group identities count
  ONCE; non-ready members, duplicate teacher artifacts, malformed `alpha`
  (outside `(0, 0.5)`), out-of-order members and empty manifests all fail closed.
  Kept OUT of the inference import path (imports only the teacher module).

- **Reachable/identity-bearing repairs retained from the prior pass**:
  `routing_preflight_v1` (`objective_tolerance`; `no_feasible_candidate_cells`
  `tuple[int,int]`), `cell_partition_v1` (group-scoped cross-partition alias),
  `partitioned_prediction_v1` (feature-allowed cell may carry an unavailable head),
  `inference_contract_v1` (source-commit check), `dataset_contract_v2`+`dataset_io_v2`
  (boolean-domain rejection), `source_family_v2` (query_source mixture-only ancestry,
  decoded-mixture-PCM + closed-world discovery-manifest aliases).

## 4. Dataset panel — one globally-ordered frozen candidate-recipe panel

`dataset_contract_v1`'s frozen-panel check now compares the **ordered candidate-
recipe panel** (recipe-id order) rather than full recipe+decoded-artifact identities;
order remains identity-bearing (a reversed panel is still rejected — the
`dataset_contract_v1` order test passes). Multiple source families in one dataset
share that one ordered recipe panel while each decodes its own per-family artifact
PCM; family-specific outputs are expressed by the per-row availability mask, never
by mutating/reordering the panel. The source-family fixtures were corrected to build
this shared recipe panel with per-family artifacts (`test_source_family_v1/v2`).

**Flag (single genuinely under-specified point):** gpt56 said "fix the fixtures, not
the invariant." A *fixture-only* fix is provably impossible: the registry
cross-family alias check forbids a candidate identity in multiple families, and the
dataset↔family binding requires each row's candidates to be members of that row's
family — so a multi-family dataset can NEVER share a full recipe+artifact candidate
panel. Honoring gpt56's stated invariant ("SAME ordered candidate-RECIPE panel;
family-specific outputs use MASKS") therefore required aligning the coded check to
compare the ordered RECIPE panel (its stated granularity), a minimal,
order-preserving, backward-compatible relaxation. Flagged for review.

**RESOLVED — gpt56 CONFIRMED (PR #26, comment 5267006554):** the recipe-level
identity is the intended design — "one globally ordered panel of candidate
recipes/model identities per dataset version; per-family immutable artifacts per
panel member; availability masks for missing artifacts; reordering/substituting a
recipe changes dataset identity." Verified `DatasetManifest.validate` /
`identity_dict` match this wording exactly (ordered `recipe_id` panel compared
across rows; per-row `artifact_pcm_sha256`; per-row availability mask; identity
carries the ordered `candidate_panel` + `ordered_row_ids`). Backed by
`test_candidate_order_is_identity_bearing_and_cannot_mix_inside_dataset`. No code
change required.

## 5. The four semantic decisions

- **`dataset_exhaustive_v2`**: `to_inference_record` keeps provenance hashes
  (metric-contract, route-policy, feature-identity `feature_sha256`) and excludes
  truth-bearing values; the over-broad keyset test was rewritten to forbid only
  truth-bearing keys. Unavailable exact-risk positions store canonical zero;
  empty/nonbinding threshold fails closed.
- **model query-free default**: `RiskModelConfig.query_projection_dim` default
  `16→0` so a query-disabled config materializes zero query dimensions and is valid;
  query-disabled + nonzero remains invalid.
- **group coverage**: the finite-group conformal safety gate and the
  `guaranteed_group_coverage ≥ target_coverage` self-check are UNTOUCHED; the fixture
  substitution `0.91→0.80` (certifiable with the fixture's groups, ≤ guaranteed,
  mismatching scope 0.90) exercises the scope-binding "target coverage" mismatch. A
  singleton `group_score_mode` case was replaced by an explicit invalid-mode
  fail-closed assertion.
- **grouped comparison**: each arm is validated locally before pairwise; the test now
  builds two individually-valid arms (D0=`ORACLE_UNAVAILABLE`, R0=`ABSTAIN` with
  objective None) that agree on the objective drift check but differ on
  `ORACLE_UNAVAILABLE`, reaching the pairwise "availability differs" rejection.

## 6. Ruff

- Base `2254c8e` = 510. This branch = **466** (≤ base). Every PR-added/touched
  Python file: **0 findings** (`ruff check <scope>` → "All checks passed!").
- Method: safe `ruff check --fix` on the PR-added/touched scope only (imports /
  modernization / unused) + by-hand C414/F841/B017 — **no `--unsafe-fixes`, no
  `# noqa`, no excludes, no `--fix` on inherited files**.
- The remaining 466 are inherited base debt in NON-PR files, predominantly
  non-autofixable. **Not rewritten**, per gpt56's "do not rewrite unrelated
  inherited lint debt."

**gpt56 decision #1 — CI Ruff hard ratchet (pinned `ruff==0.16.2`).** Empirically
verified the pin reproduces the frozen recording: at base SHA `2254c8e` the default
ruleset (no repo Ruff config exists) yields exactly **510** under `0.16.2`
(≤ `0.15` yields 258 — `0.16` stabilized many rules), and the branch head yields
**466**. The workflow's single `ruff check .` step is replaced by two binding steps:
  - **PR touched-file Ruff (zero findings):** enumerate PR-added/modified `*.py` via
    `git diff --name-only $(git merge-base 2254c8e HEAD)..HEAD -- '*.py'`, existing
    files only, and require **0 findings** under the pin.
  - **Repository-wide Ruff ratchet:** re-derive the base count in a throwaway
    base-SHA worktree with the pin and assert it equals the recorded **510**
    (tamper-evident), compute the deterministic head count, and require
    `head_count <= 510`.
  Both write `validation-logs/` evidence (base SHA, Ruff version, base/head counts,
  touched-file list). Missing/malformed baseline, base-count drift, or an
  unenumerable diff **fail closed**. No blanket exclude, no `# noqa`, no
  `--unsafe-fixes`, no rewriting inherited debt. This resolves the previously-red
  `Repository-wide Ruff` step (466 ≤ 510) at the exact head.

## 7. Gates

Full `pytest -q` GREEN (**1107 passed / 1 skipped**); focused v4 / v3-partition /
v3-lineage(+adversarial) / v4-schema+rejected-mutations GREEN; d0-structured-teacher
and NEW grouped-training-manifest gates GREEN; `compileall` GREEN;
`git diff --check` GREEN. Touched-file Ruff (pin `0.16.2`) = **0**; repo-wide Ruff
head = **466 ≤ 510**.
