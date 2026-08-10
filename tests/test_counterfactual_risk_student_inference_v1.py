import hashlib
import inspect

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
    InferenceManifestV1,
)
from audio_extract.counterfactual_risk_student_inference_v1 import (
    PredictedUpperRiskManifestV1,
    StudentInferenceError,
    predict_upper_fail_closed,
)


def sha(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode()).hexdigest()


def git(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()


SOURCE_COMMIT = git("source")
FEATURE_CONTRACT = sha("feature-contract")
METRIC_CONTRACT = sha("metric-contract")
ROUTE_POLICY = sha("route-policy")
POLICY = sha("prediction-policy")


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


def exact_row(index: int, *, features) -> CounterfactualRiskRowV2:
    candidate_panel = panel()
    source = sha(f"source-{index}")
    artifacts = tuple(
        CandidateArtifactBinding(
            slot_id=slot.slot_id,
            slot_sha256=slot.sha256,
            recipe_id=sha(f"recipe-{index}-{slot.slot_id}"),
            artifact_pcm_sha256=sha(f"pcm-{index}-{slot.slot_id}"),
            recipe_semantic_sha256=sha(f"semantic-{index}-{slot.slot_id}"),
            recipe_slot_projection_sha256=slot.sha256,
            source_family_sha256=source,
            verifier_commit=git("artifact-verifier"),
        )
        for slot in candidate_panel.slots
    )
    return CounterfactualRiskRowV2(
        group_family=GroupFamilyIdentity(
            work_id=f"work-{index}",
            recording_session_id=f"session-{index}",
            target_singer_id=f"singer-{index}",
            source_family_sha256=source,
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
        metric_names=("voice", "hole"),
        metric_units=("ratio", "ratio"),
        metric_directions=("lower_is_better", "lower_is_better"),
        features=np.asarray(features, dtype=np.float64),
        exact_risks=np.asarray([[0.1, 0.2], [0.2, 0.1]]),
        available=np.ones((2, 2), dtype=bool),
        feature_contract_sha256=FEATURE_CONTRACT,
        metric_contract_sha256=METRIC_CONTRACT,
        route_policy_sha256=ROUTE_POLICY,
    )


def inference_manifest(*, block_first=True, block_second=False):
    # A feature-blocked position must be stored as canonical zero, so the
    # blockable head of each row is zero whenever that row is blocked.
    first = exact_row(0, features=[0.0, 0.5])
    second = exact_row(1, features=[0.0 if block_second else 1.0, 1.5])
    dataset = DatasetManifestV2.build(
        (first, second), source_commit=SOURCE_COMMIT
    )
    certificates = []
    for row, blocked in ((first, block_first), (second, block_second)):
        certificates.append(
            FeatureEvidenceCertificate(
                row_id=row.row_id,
                source_family_sha256=row.group_family.source_family_sha256,
                feature_contract_sha256=row.feature_contract_sha256,
                feature_sha256=row.feature_sha256,
                available=(not blocked, True),
                required=(True, True),
                missing_value_policy_sha256=sha("missing-policy"),
                extraction_report_sha256=sha(f"report-{row.row_id}"),
                extractor_bundle_sha256=sha("extractor"),
                verifier_commit=git("feature-verifier"),
            )
        )
    registry = FeatureEvidenceRegistry.build(
        tuple(certificates), source_commit=SOURCE_COMMIT
    )
    return InferenceManifestV1.build(dataset, registry)


class FakeStudent:
    def __init__(
        self,
        manifest,
        *,
        model_sha=None,
        shape_mode="valid",
        value_offset=0.0,
    ):
        self.candidate_count = 2
        self.metric_count = 2
        self.feature_count = 2
        self.candidate_panel_sha256 = manifest.candidate_panel.sha256
        self.feature_contract_sha256 = manifest.rows[0].feature_contract_sha256
        self.metric_contract_sha256 = manifest.rows[0].metric_contract_sha256
        self._sha256 = model_sha or sha("student")
        self.shape_mode = shape_mode
        self.value_offset = value_offset
        self.calls = []

    @property
    def sha256(self):
        return self._sha256

    def validate(self):
        return None

    def predict_upper(self, features):
        values = np.asarray(features, dtype=np.float64)
        self.calls.append(values.copy())
        result = np.empty((len(values), 2, 2), dtype=np.float64)
        for index, row in enumerate(values):
            base = float(row.sum()) + self.value_offset
            result[index] = ((base + 0.1, base + 0.2), (base + 0.3, base + 0.4))
        if self.shape_mode == "wrong":
            return result[:, :1]
        if self.shape_mode == "negative":
            result[0, 0, 0] = -1.0
        if self.shape_mode == "nan":
            result[0, 0, 0] = np.nan
        return result


class NeverCallStudent(FakeStudent):
    def predict_upper(self, features):
        raise AssertionError("blocked rows must not reach the student")


def test_only_feature_complete_rows_reach_the_student():
    inference = inference_manifest(block_first=True, block_second=False)
    student = FakeStudent(inference)
    result = predict_upper_fail_closed(
        student, inference, prediction_policy_sha256=POLICY
    )
    assert len(student.calls) == 1
    assert student.calls[0].shape == (1, 2)
    assert result.upper.shape == (2, 2, 2)
    assert result.available.shape == result.upper.shape
    for index, allowed in enumerate(result.row_prediction_allowed):
        if allowed:
            assert result.available[index].all()
            assert np.all(result.upper[index] > 0)
        else:
            assert not result.available[index].any()
            assert np.all(result.upper[index] == 0.0)
    assert result.model_input_sha256 == inference.model_input_sha256


def test_all_feature_blocked_rows_never_call_the_student():
    inference = inference_manifest(block_first=True, block_second=True)
    student = NeverCallStudent(inference)
    result = predict_upper_fail_closed(
        student, inference, prediction_policy_sha256=POLICY
    )
    assert result.row_prediction_allowed == (False, False)
    assert not result.available.any()
    assert np.all(result.upper == 0.0)


def test_student_contract_mismatches_are_refused_before_prediction():
    inference = inference_manifest()
    mutations = (
        ("candidate_panel_sha256", sha("other-panel"), "candidate panel"),
        ("feature_contract_sha256", sha("other-features"), "feature contract"),
        ("metric_contract_sha256", sha("other-metrics"), "metric contract"),
        ("candidate_count", 3, "candidate count"),
        ("metric_count", 3, "metric count"),
        ("feature_count", 3, "feature count"),
    )
    for attribute, value, expected in mutations:
        student = FakeStudent(inference)
        setattr(student, attribute, value)
        with pytest.raises(StudentInferenceError, match=expected):
            predict_upper_fail_closed(
                student, inference, prediction_policy_sha256=POLICY
            )
        assert student.calls == []


def test_wrong_shape_negative_and_nonfinite_predictions_are_refused():
    inference = inference_manifest()
    for mode, message in (
        ("wrong", "shape differs"),
        ("negative", "finite and non-negative"),
        ("nan", "finite and non-negative"),
    ):
        with pytest.raises(StudentInferenceError, match=message):
            predict_upper_fail_closed(
                FakeStudent(inference, shape_mode=mode),
                inference,
                prediction_policy_sha256=POLICY,
            )


def test_prediction_manifest_identity_binds_model_policy_inputs_outputs_and_mask():
    inference = inference_manifest()
    first = predict_upper_fail_closed(
        FakeStudent(inference),
        inference,
        prediction_policy_sha256=POLICY,
    )
    changed_model = predict_upper_fail_closed(
        FakeStudent(inference, model_sha=sha("other-student")),
        inference,
        prediction_policy_sha256=POLICY,
    )
    changed_policy = predict_upper_fail_closed(
        FakeStudent(inference),
        inference,
        prediction_policy_sha256=sha("other-policy"),
    )
    changed_values = predict_upper_fail_closed(
        FakeStudent(inference, value_offset=2.0),
        inference,
        prediction_policy_sha256=POLICY,
    )
    changed_mask_inference = inference_manifest(
        block_first=False, block_second=False
    )
    changed_mask = predict_upper_fail_closed(
        FakeStudent(changed_mask_inference),
        changed_mask_inference,
        prediction_policy_sha256=POLICY,
    )
    assert len(
        {
            first.sha256,
            changed_model.sha256,
            changed_policy.sha256,
            changed_values.sha256,
            changed_mask.sha256,
        }
    ) == 5


def test_output_arrays_are_read_only_and_unavailable_storage_is_canonical():
    inference = inference_manifest()
    result = predict_upper_fail_closed(
        FakeStudent(inference),
        inference,
        prediction_policy_sha256=POLICY,
    )
    assert not result.upper.flags.writeable
    assert not result.available.flags.writeable
    with pytest.raises(ValueError):
        result.upper[0, 0, 0] = 99.0
    with pytest.raises(ValueError):
        result.available[0, 0, 0] = True


def test_manifest_validation_rejects_mask_semantic_contradictions():
    inference = inference_manifest()
    result = predict_upper_fail_closed(
        FakeStudent(inference),
        inference,
        prediction_policy_sha256=POLICY,
    )
    blocked_index = result.row_prediction_allowed.index(False)
    allowed_index = result.row_prediction_allowed.index(True)

    available_on_blocked = np.asarray(result.available).copy()
    available_on_blocked[blocked_index] = True
    damaged = PredictedUpperRiskManifestV1(
        **{**result.__dict__, "available": available_on_blocked}
    )
    with pytest.raises(StudentInferenceError, match="feature-blocked"):
        damaged.validate()

    partial_allowed = np.asarray(result.available).copy()
    partial_allowed[allowed_index, 0, 0] = False
    upper = np.asarray(result.upper).copy()
    upper[allowed_index, 0, 0] = 0.0
    damaged = PredictedUpperRiskManifestV1(
        **{**result.__dict__, "available": partial_allowed, "upper": upper}
    )
    with pytest.raises(StudentInferenceError, match="allowed prediction row"):
        damaged.validate()

    nonzero_unavailable = np.asarray(result.upper).copy()
    nonzero_unavailable[blocked_index, 0, 0] = 1.0
    damaged = PredictedUpperRiskManifestV1(
        **{**result.__dict__, "upper": nonzero_unavailable}
    )
    with pytest.raises(StudentInferenceError, match="canonical zero"):
        damaged.validate()


def test_prediction_api_accepts_no_exact_dataset_or_truth_arguments():
    parameters = inspect.signature(predict_upper_fail_closed).parameters
    assert set(parameters) == {
        "student",
        "inference",
        "prediction_policy_sha256",
    }
    assert not {
        "dataset",
        "exact_risks",
        "accompaniment_truth",
        "vocal_truth",
        "oracle_route",
    } & set(parameters)
