import hashlib

import numpy as np
import pytest

from audio_extract.counterfactual_risk_dataset_contract_v1 import (
    CellGeometry,
    GroupFamilyIdentity,
)
from audio_extract.counterfactual_risk_dataset_contract_v2 import (
    CandidateArtifactBinding,
    CandidatePanel,
    CandidateSlot,
    CounterfactualRiskRowV2,
    DatasetManifestV2,
)
from audio_extract.counterfactual_risk_feature_evidence_v1 import (
    FeatureEvidenceCertificate,
    FeatureEvidenceRegistry,
)
from audio_extract.counterfactual_risk_inference_contract_v1 import (
    InferenceAuditBinding,
    InferenceContractError,
    InferenceManifestV1,
    InferenceRowV1,
)


def sha(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode()).hexdigest()


def git(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()


SOURCE_COMMIT = git("source")
FEATURE_CONTRACT = sha("feature-contract")
METRIC_CONTRACT = sha("metric-contract")
ROUTE_POLICY = sha("route-policy")


def panel() -> CandidatePanel:
    return CandidatePanel(
        (
            CandidateSlot(
                "slot-a",
                sha("model-a"),
                sha("adapter-a"),
                sha("construction-a"),
                sha("query-contract"),
            ),
            CandidateSlot(
                "slot-b",
                sha("model-b"),
                sha("adapter-b"),
                sha("construction-b"),
                sha("query-contract"),
            ),
        )
    )


def exact_row(
    index: int,
    *,
    exact_risks=None,
    accompaniment_truth=None,
    vocal_truth=None,
    features=None,
    query=None,
) -> CounterfactualRiskRowV2:
    candidate_panel = panel()
    source_family = sha(f"source-family-{index}")
    artifacts = tuple(
        CandidateArtifactBinding(
            slot_id=slot.slot_id,
            slot_sha256=slot.sha256,
            recipe_id=sha(f"recipe-{index}-{slot.slot_id}"),
            artifact_pcm_sha256=sha(f"pcm-{index}-{slot.slot_id}"),
            recipe_semantic_sha256=sha(f"semantic-{index}-{slot.slot_id}"),
            recipe_slot_projection_sha256=slot.sha256,
            source_family_sha256=source_family,
            verifier_commit=git("artifact-verifier"),
        )
        for slot in candidate_panel.slots
    )
    return CounterfactualRiskRowV2(
        group_family=GroupFamilyIdentity(
            work_id=f"work-{index}",
            recording_session_id=f"session-{index}",
            target_singer_id=f"singer-{index}",
            source_family_sha256=source_family,
            query_condition_sha256=query or sha(f"query-{index}"),
        ),
        mixture_pcm_sha256=sha(f"mixture-{index}"),
        accompaniment_truth_pcm_sha256=(
            accompaniment_truth or sha(f"a-truth-{index}")
        ),
        vocal_truth_pcm_sha256=vocal_truth or sha(f"v-truth-{index}"),
        cell=CellGeometry(
            sample_rate_hz=44_100,
            resolution_ms=500,
            start_frame=index * 1000,
            end_frame=(index + 1) * 1000,
            band_low_hz=0,
            band_high_hz=500,
            spectral_grid_sha256=sha("grid"),
        ),
        candidate_panel=candidate_panel,
        candidate_artifacts=artifacts,
        metric_names=("voice", "hole", "artifact"),
        metric_units=("ratio", "ratio", "ratio"),
        metric_directions=(
            "lower_is_better",
            "lower_is_better",
            "lower_is_better",
        ),
        features=np.asarray(
            features if features is not None else [float(index), 0.5, -0.25],
            dtype=np.float64,
        ),
        exact_risks=np.asarray(
            exact_risks
            if exact_risks is not None
            else [[0.1, 0.2, 0.3], [0.2, 0.1, 0.4]],
            dtype=np.float64,
        ),
        available=np.ones((2, 3), dtype=bool),
        feature_contract_sha256=FEATURE_CONTRACT,
        metric_contract_sha256=METRIC_CONTRACT,
        route_policy_sha256=ROUTE_POLICY,
    )


def dataset(rows=None, *, source_commit=SOURCE_COMMIT) -> DatasetManifestV2:
    return DatasetManifestV2.build(
        rows or (exact_row(0), exact_row(1)),
        source_commit=source_commit,
    )


def evidence(
    row: CounterfactualRiskRowV2,
    *,
    available=(True, True, True),
    required=(True, True, True),
) -> FeatureEvidenceCertificate:
    return FeatureEvidenceCertificate(
        row_id=row.row_id,
        source_family_sha256=row.group_family.source_family_sha256,
        feature_contract_sha256=row.feature_contract_sha256,
        feature_sha256=row.feature_sha256,
        available=tuple(available),
        required=tuple(required),
        missing_value_policy_sha256=sha("missing-policy"),
        extraction_report_sha256=sha(f"report-{row.row_id}"),
        extractor_bundle_sha256=sha("extractor"),
        verifier_commit=git("feature-verifier"),
    )


def feature_registry(
    data: DatasetManifestV2,
    *,
    masks=None,
    source_commit=None,
) -> FeatureEvidenceRegistry:
    mask_values = masks or {}
    certificates = []
    for row in data.rows:
        values = mask_values.get(row.group_family.work_id, {})
        certificates.append(evidence(row, **values))
    return FeatureEvidenceRegistry.build(
        tuple(certificates),
        source_commit=source_commit or data.source_commit,
    )


def recursive_keys(value):
    if isinstance(value, dict):
        result = set(value)
        for child in value.values():
            result.update(recursive_keys(child))
        return result
    if isinstance(value, list):
        result = set()
        for child in value:
            result.update(recursive_keys(child))
        return result
    return set()


def test_model_input_is_truth_free_and_audit_binding_is_separate():
    data = dataset()
    registry = feature_registry(data)
    manifest = InferenceManifestV1.build(data, registry)
    document = manifest.model_input_document()
    keys = recursive_keys(document)
    forbidden = {
        "offline_dataset_sha256",
        "offline_row_id",
        "feature_evidence_sha256",
        "accompaniment_truth_pcm_sha256",
        "vocal_truth_pcm_sha256",
        "exact_risks",
        "available",
        "exact_risks_sha256",
        "availability_sha256",
        "teacher_route",
        "oracle_margin",
    }
    assert keys.isdisjoint(forbidden)
    assert document["metric_names"] == ["voice", "hole", "artifact"]
    assert document["metric_contract_sha256"] == METRIC_CONTRACT
    assert document["route_policy_sha256"] == ROUTE_POLICY
    assert len(manifest.audit_bindings) == len(manifest.rows) == 2
    assert manifest.model_input_sha256.startswith("sha256:")
    assert manifest.sha256.startswith("sha256:")


def test_exact_label_and_clean_truth_changes_do_not_change_model_input_identity():
    first_rows = (exact_row(0), exact_row(1))
    first_data = dataset(first_rows)
    first_manifest = InferenceManifestV1.build(
        first_data, feature_registry(first_data)
    )

    changed_rows = (
        exact_row(
            0,
            exact_risks=[[9.0, 8.0, 7.0], [6.0, 5.0, 4.0]],
            accompaniment_truth=sha("different-a-truth"),
            vocal_truth=sha("different-v-truth"),
        ),
        exact_row(1),
    )
    changed_data = dataset(changed_rows)
    changed_manifest = InferenceManifestV1.build(
        changed_data, feature_registry(changed_data)
    )

    assert first_manifest.model_input_sha256 == (
        changed_manifest.model_input_sha256
    )
    assert first_manifest.offline_dataset_sha256 != (
        changed_manifest.offline_dataset_sha256
    )
    assert first_manifest.sha256 != changed_manifest.sha256


def test_feature_query_and_candidate_changes_are_model_input_identity_bearing():
    base_data = dataset()
    base = InferenceManifestV1.build(base_data, feature_registry(base_data))

    feature_rows = (exact_row(0, features=[9.0, 0.5, -0.25]), exact_row(1))
    feature_data = dataset(feature_rows)
    feature_manifest = InferenceManifestV1.build(
        feature_data, feature_registry(feature_data)
    )

    query_rows = (exact_row(0, query=sha("different-query")), exact_row(1))
    query_data = dataset(query_rows)
    query_manifest = InferenceManifestV1.build(
        query_data, feature_registry(query_data)
    )

    candidate = base_data.rows[0]
    first_binding = candidate.candidate_artifacts[0]
    changed_binding = CandidateArtifactBinding(
        **{
            **first_binding.__dict__,
            "recipe_id": sha("different-recipe"),
            "artifact_pcm_sha256": sha("different-pcm"),
            "recipe_semantic_sha256": sha("different-semantic"),
        }
    )
    changed_row = CounterfactualRiskRowV2(
        **{
            **candidate.__dict__,
            "candidate_artifacts": (
                changed_binding,
                candidate.candidate_artifacts[1],
            ),
        }
    )
    candidate_data = dataset((changed_row, base_data.rows[1]))
    candidate_manifest = InferenceManifestV1.build(
        candidate_data, feature_registry(candidate_data)
    )

    assert len(
        {
            base.model_input_sha256,
            feature_manifest.model_input_sha256,
            query_manifest.model_input_sha256,
            candidate_manifest.model_input_sha256,
        }
    ) == 4


def test_feature_masks_and_prediction_allowed_are_model_facing():
    value = exact_row(0, features=[0.0, 0.5, -0.25])
    data = dataset((value, exact_row(1)))
    registry = feature_registry(
        data,
        masks={
            "work-0": {
                "available": (False, True, True),
                "required": (True, True, True),
            }
        },
    )
    manifest = InferenceManifestV1.build(data, registry)
    row_document = next(
        item
        for item in manifest.model_input_document()["rows"]
        if item["work_id"] == "work-0"
    )
    assert row_document["feature_available"] == [False, True, True]
    assert row_document["feature_required"] == [True, True, True]
    assert row_document["prediction_allowed"] is False


def test_nonzero_unavailable_feature_is_refused_before_projection():
    value = exact_row(0, features=[7.0, 0.5, -0.25])
    data = dataset((value, exact_row(1)))
    registry = feature_registry(
        data,
        masks={
            "work-0": {
                "available": (False, True, True),
                "required": (True, True, True),
            }
        },
    )
    with pytest.raises(Exception, match="canonical zero"):
        InferenceManifestV1.build(data, registry)


def test_feature_registry_and_dataset_must_use_one_source_commit():
    data = dataset()
    registry = feature_registry(data, source_commit=git("other-source"))
    with pytest.raises(
        InferenceContractError,
        match="different source commits",
    ):
        InferenceManifestV1.build(data, registry)


def test_audit_binding_order_and_uniqueness_are_fail_closed():
    data = dataset()
    manifest = InferenceManifestV1.build(data, feature_registry(data))
    reversed_audits = tuple(reversed(manifest.audit_bindings))
    with pytest.raises(
        InferenceContractError,
        match="audit binding names a different inference row",
    ):
        InferenceManifestV1(
            **{**manifest.__dict__, "audit_bindings": reversed_audits}
        ).validate()

    first = manifest.audit_bindings[0]
    duplicate_offline = InferenceAuditBinding(
        inference_row_sha256=manifest.audit_bindings[1].inference_row_sha256,
        offline_row_id=first.offline_row_id,
        feature_evidence_sha256=(
            manifest.audit_bindings[1].feature_evidence_sha256
        ),
    )
    with pytest.raises(
        InferenceContractError,
        match="offline exact row is bound to multiple inference rows",
    ):
        InferenceManifestV1(
            **{
                **manifest.__dict__,
                "audit_bindings": (first, duplicate_offline),
            }
        ).validate()


def test_rows_must_share_model_facing_contracts_and_canonical_order():
    data = dataset()
    manifest = InferenceManifestV1.build(data, feature_registry(data))
    changed_row = InferenceRowV1(
        **{
            **manifest.rows[1].__dict__,
            "metric_contract_sha256": sha("other-metric-contract"),
        }
    )
    changed_rows = tuple(
        sorted(
            (manifest.rows[0], changed_row),
            key=lambda row: row.sha256(candidate_panel=manifest.candidate_panel),
        )
    )
    changed_audits = []
    by_offline = {
        binding.offline_row_id: binding
        for binding in manifest.audit_bindings
    }
    # Rebind audit identities to avoid making the order check hide the contract
    # mismatch under test.
    for index, row in enumerate(changed_rows):
        old = manifest.audit_bindings[index]
        changed_audits.append(
            InferenceAuditBinding(
                inference_row_sha256=row.sha256(
                    candidate_panel=manifest.candidate_panel
                ),
                offline_row_id=old.offline_row_id,
                feature_evidence_sha256=old.feature_evidence_sha256,
            )
        )
    with pytest.raises(
        InferenceContractError,
        match="disagree on model-facing contracts",
    ):
        InferenceManifestV1(
            **{
                **manifest.__dict__,
                "rows": changed_rows,
                "audit_bindings": tuple(changed_audits),
            }
        ).validate()

    with pytest.raises(
        InferenceContractError,
        match="canonical inference-SHA order",
    ):
        InferenceManifestV1(
            **{
                **manifest.__dict__,
                "rows": tuple(reversed(manifest.rows)),
                "audit_bindings": tuple(
                    reversed(manifest.audit_bindings)
                ),
            }
        ).validate()
