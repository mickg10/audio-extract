import hashlib
import itertools

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
    FeatureEvidenceError,
    FeatureEvidenceRegistry,
    inference_records_with_feature_evidence,
    validate_dataset_feature_evidence,
)


def sha(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode()).hexdigest()


def git(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()


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


def row(index: int = 0, *, features=None) -> CounterfactualRiskRowV2:
    candidate_panel = panel()
    source_family = sha(f"source-{index}")
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
            query_condition_sha256=sha(f"query-{index}"),
        ),
        mixture_pcm_sha256=sha(f"mixture-{index}"),
        accompaniment_truth_pcm_sha256=sha(f"a-{index}"),
        vocal_truth_pcm_sha256=sha(f"v-{index}"),
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
            features if features is not None else [0.25, -0.5, 1.5],
            dtype=np.float64,
        ),
        exact_risks=np.asarray(
            [[0.1, 0.2, 0.3], [0.2, 0.1, 0.4]], dtype=np.float64
        ),
        available=np.ones((2, 3), dtype=bool),
        feature_contract_sha256=sha("features"),
        metric_contract_sha256=sha("metrics"),
        route_policy_sha256=sha("route-policy"),
    )


def dataset(rows=None) -> DatasetManifestV2:
    return DatasetManifestV2.build(
        rows or (row(0),), source_commit=git("dataset")
    )


def evidence(
    value: CounterfactualRiskRowV2,
    *,
    available=(True, True, True),
    required=(True, True, True),
    source_family_sha256=None,
    feature_contract_sha256=None,
    feature_sha256=None,
    status="passed",
) -> FeatureEvidenceCertificate:
    return FeatureEvidenceCertificate(
        row_id=value.row_id,
        source_family_sha256=(
            source_family_sha256
            or value.group_family.source_family_sha256
        ),
        feature_contract_sha256=(
            feature_contract_sha256 or value.feature_contract_sha256
        ),
        feature_sha256=feature_sha256 or value.feature_sha256,
        available=tuple(available),
        required=tuple(required),
        missing_value_policy_sha256=sha("missing-value-policy"),
        extraction_report_sha256=sha(f"extraction-report-{value.row_id}"),
        extractor_bundle_sha256=sha("extractor-bundle"),
        verifier_commit=git("feature-verifier"),
        status=status,
    )


def registry(*certificates: FeatureEvidenceCertificate) -> FeatureEvidenceRegistry:
    return FeatureEvidenceRegistry.build(
        certificates, source_commit=git("feature-registry")
    )


def test_valid_feature_evidence_binds_dataset_and_inference_projection():
    value = row()
    data = dataset((value,))
    certificate = evidence(value)
    feature_registry = registry(certificate)
    validate_dataset_feature_evidence(data, feature_registry)
    records = inference_records_with_feature_evidence(data, feature_registry)
    assert len(records) == 1
    record = records[0]
    assert record["prediction_allowed"] is True
    assert record["feature_available"] == [True, True, True]
    assert record["feature_required"] == [True, True, True]
    assert record["feature_evidence_sha256"] == certificate.sha256
    assert record["feature_availability_sha256"] == (
        certificate.availability_sha256
    )
    assert "exact_risks" not in record
    assert "available" not in record
    assert "accompaniment_truth_pcm_sha256" not in record
    assert "vocal_truth_pcm_sha256" not in record


def test_missing_and_unused_feature_evidence_are_refused():
    first = row(0)
    second = row(1)
    data = dataset((first, second))
    with pytest.raises(FeatureEvidenceError, match="lacks a feature-evidence"):
        validate_dataset_feature_evidence(data, registry(evidence(first)))

    single = dataset((first,))
    with pytest.raises(FeatureEvidenceError, match="unused by the dataset"):
        validate_dataset_feature_evidence(
            single,
            registry(evidence(first), evidence(second)),
        )


def test_source_contract_and_feature_hash_must_match_the_row():
    value = row()
    data = dataset((value,))
    cases = (
        (
            evidence(value, source_family_sha256=sha("other-source")),
            "different source family",
        ),
        (
            evidence(value, feature_contract_sha256=sha("other-contract")),
            "different feature contract",
        ),
        (
            evidence(value, feature_sha256=sha("other-features")),
            "different feature bytes",
        ),
    )
    for certificate, message in cases:
        with pytest.raises(FeatureEvidenceError, match=message):
            validate_dataset_feature_evidence(data, registry(certificate))


def test_unavailable_feature_must_be_stored_as_canonical_zero():
    damaged = row(features=[0.25, 9.0, 1.5])
    certificate = evidence(
        damaged,
        available=(True, False, True),
        required=(True, False, True),
    )
    with pytest.raises(
        FeatureEvidenceError,
        match="unavailable feature entries must be stored as canonical zero",
    ):
        validate_dataset_feature_evidence(
            dataset((damaged,)), registry(certificate)
        )

    valid = row(features=[0.25, 0.0, 1.5])
    valid_certificate = evidence(
        valid,
        available=(True, False, True),
        required=(True, False, True),
    )
    validate_dataset_feature_evidence(
        dataset((valid,)), registry(valid_certificate)
    )


def test_required_missing_feature_abstains_but_optional_missing_feature_does_not():
    value = row(features=[0.25, 0.0, 1.5])
    required_missing = evidence(
        value,
        available=(True, False, True),
        required=(True, True, True),
    )
    optional_missing = evidence(
        value,
        available=(True, False, True),
        required=(True, False, True),
    )
    assert required_missing.prediction_allowed is False
    assert optional_missing.prediction_allowed is True

    required_record = inference_records_with_feature_evidence(
        dataset((value,)), registry(required_missing)
    )[0]
    optional_record = inference_records_with_feature_evidence(
        dataset((value,)), registry(optional_missing)
    )[0]
    assert required_record["prediction_allowed"] is False
    assert optional_record["prediction_allowed"] is True


def test_registry_refuses_multiple_required_masks_for_one_feature_contract():
    first = row(0)
    second = row(1)
    first_certificate = evidence(
        first,
        available=(True, True, True),
        required=(True, True, False),
    )
    second_certificate = evidence(
        second,
        available=(True, True, True),
        required=(True, False, True),
    )
    with pytest.raises(
        FeatureEvidenceError,
        match="multiple required-feature masks",
    ):
        registry(first_certificate, second_certificate)


def test_masks_must_be_nonempty_boolean_and_match_feature_count():
    value = row()
    with pytest.raises(FeatureEvidenceError, match="contain only booleans"):
        evidence(value, available=(1, True, True)).validate()
    with pytest.raises(FeatureEvidenceError, match="must be non-empty"):
        evidence(value, available=(), required=()).validate()
    with pytest.raises(FeatureEvidenceError, match="differ in length"):
        evidence(
            value,
            available=(True, True),
            required=(True, True, True),
        ).validate()
    with pytest.raises(FeatureEvidenceError, match="length differs"):
        evidence(
            value,
            available=(True, True),
            required=(True, True),
        ).validate(feature_count=3)


def test_failed_certificate_is_refused():
    value = row()
    with pytest.raises(FeatureEvidenceError, match="status must be passed"):
        evidence(value, status="failed").validate()


def test_exhaustive_prediction_allowed_matches_required_availability_rule():
    value = row()
    checked = 0
    for availability in itertools.product((False, True), repeat=3):
        for required in itertools.product((False, True), repeat=3):
            certificate = evidence(
                value,
                available=availability,
                required=required,
            )
            expected = all(
                is_available or not is_required
                for is_available, is_required in zip(
                    availability, required
                )
            )
            assert certificate.prediction_allowed is expected
            checked += 1
    assert checked == 2 ** 6


def test_feature_evidence_identity_binds_masks_policy_extractor_and_report():
    value = row()
    original = evidence(value)
    changed_availability = FeatureEvidenceCertificate(
        **{
            **original.__dict__,
            "available": (True, False, True),
            "required": (True, False, True),
        }
    )
    changed_policy = FeatureEvidenceCertificate(
        **{
            **original.__dict__,
            "missing_value_policy_sha256": sha("other-policy"),
        }
    )
    changed_extractor = FeatureEvidenceCertificate(
        **{
            **original.__dict__,
            "extractor_bundle_sha256": sha("other-extractor"),
        }
    )
    changed_report = FeatureEvidenceCertificate(
        **{
            **original.__dict__,
            "extraction_report_sha256": sha("other-report"),
        }
    )
    assert len(
        {
            original.sha256,
            changed_availability.sha256,
            changed_policy.sha256,
            changed_extractor.sha256,
            changed_report.sha256,
        }
    ) == 5
