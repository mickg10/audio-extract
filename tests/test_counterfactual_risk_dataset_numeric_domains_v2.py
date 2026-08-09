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
    CounterfactualRiskDatasetV2Error,
    CounterfactualRiskRowV2,
    DatasetManifestV2,
)
from audio_extract.counterfactual_risk_dataset_io_v2 import (
    CounterfactualRiskDatasetIOV2Error,
    parse_dataset_document_v2,
)


def sha(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode()).hexdigest()


def git(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()


def valid_row() -> CounterfactualRiskRowV2:
    slots = (
        CandidateSlot(
            "a",
            sha("model-a"),
            sha("adapter-a"),
            sha("construction-a"),
            sha("query-contract"),
        ),
        CandidateSlot(
            "b",
            sha("model-b"),
            sha("adapter-b"),
            sha("construction-b"),
            sha("query-contract"),
        ),
    )
    panel = CandidatePanel(slots)
    source = sha("source")
    artifacts = tuple(
        CandidateArtifactBinding(
            slot_id=slot.slot_id,
            slot_sha256=slot.sha256,
            recipe_id=sha(f"recipe-{slot.slot_id}"),
            artifact_pcm_sha256=sha(f"pcm-{slot.slot_id}"),
            recipe_semantic_sha256=sha(f"semantic-{slot.slot_id}"),
            recipe_slot_projection_sha256=slot.sha256,
            source_family_sha256=source,
            verifier_commit=git("verifier"),
        )
        for slot in slots
    )
    return CounterfactualRiskRowV2(
        group_family=GroupFamilyIdentity(
            work_id="work",
            recording_session_id="session",
            target_singer_id="singer",
            source_family_sha256=source,
            query_condition_sha256=sha("query"),
        ),
        mixture_pcm_sha256=sha("mixture"),
        accompaniment_truth_pcm_sha256=sha("a-truth"),
        vocal_truth_pcm_sha256=sha("v-truth"),
        cell=CellGeometry(
            sample_rate_hz=44_100,
            resolution_ms=500,
            start_frame=0,
            end_frame=1000,
            band_low_hz=0,
            band_high_hz=500,
            spectral_grid_sha256=sha("grid"),
        ),
        candidate_panel=panel,
        candidate_artifacts=artifacts,
        metric_names=("voice", "hole", "artifact"),
        metric_units=("ratio", "ratio", "ratio"),
        metric_directions=(
            "lower_is_better",
            "lower_is_better",
            "lower_is_better",
        ),
        features=np.asarray([0.1, 0.2]),
        exact_risks=np.asarray([[0.1, 0.2, 0.3], [0.2, 0.1, 0.4]]),
        available=np.ones((2, 3), dtype=bool),
        feature_contract_sha256=sha("features"),
        metric_contract_sha256=sha("metrics"),
        route_policy_sha256=sha("policy"),
    )


def dataset_document():
    row = valid_row()
    return DatasetManifestV2.build(
        [row], source_commit=git("dataset")
    ).to_document()


def test_boolean_feature_array_is_not_a_numeric_feature_vector():
    row = valid_row()
    damaged = CounterfactualRiskRowV2(
        **{**row.__dict__, "features": np.asarray([True, False])}
    )
    with pytest.raises(
        CounterfactualRiskDatasetV2Error,
        match="features must be numeric and not boolean",
    ):
        damaged.validate()


def test_boolean_exact_risk_array_is_not_a_numeric_risk_tensor():
    row = valid_row()
    damaged = CounterfactualRiskRowV2(
        **{
            **row.__dict__,
            "exact_risks": np.asarray(
                [[True, False, True], [False, True, False]], dtype=bool
            ),
        }
    )
    with pytest.raises(
        CounterfactualRiskDatasetV2Error,
        match="exact risks must be numeric and not boolean",
    ):
        damaged.validate()


def test_json_boolean_features_and_risks_are_refused_before_float_coercion():
    document = dataset_document()
    document["rows"][0]["features"][0] = True
    with pytest.raises(
        CounterfactualRiskDatasetIOV2Error,
        match="features contains a boolean",
    ):
        parse_dataset_document_v2(document)

    document = dataset_document()
    document["rows"][0]["exact_risks"][0][0] = False
    with pytest.raises(
        CounterfactualRiskDatasetIOV2Error,
        match="exact_risks contains a boolean",
    ):
        parse_dataset_document_v2(document)


def test_top_level_feature_count_boolean_is_refused_not_equal_to_one():
    document = dataset_document()
    # Python's True == 1, so the parser must enforce the JSON scalar type before
    # comparing this field with recomputed semantics.
    document["feature_count"] = True
    with pytest.raises(
        CounterfactualRiskDatasetIOV2Error,
        match="feature_count must be a positive integer, not boolean",
    ):
        parse_dataset_document_v2(document)
