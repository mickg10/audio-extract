import hashlib
import json

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
from audio_extract.counterfactual_risk_dataset_io_v2 import (
    CounterfactualRiskDatasetIOV2Error,
    dataset_container_sha256_v2,
    dump_dataset_json_v2,
    load_dataset_path_v2,
    loads_dataset_json_v2,
    parse_dataset_document_v2,
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


def row(index: int, candidate_panel: CandidatePanel):
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
        metric_names=("voice", "hole", "artifact"),
        metric_units=("ratio", "ratio", "ratio"),
        metric_directions=(
            "lower_is_better",
            "lower_is_better",
            "lower_is_better",
        ),
        features=np.asarray([float(index), 0.5, -0.25]),
        exact_risks=np.asarray([[0.1, 0.2, 0.3], [0.2, 0.1, 0.4]]),
        available=np.ones((2, 3), dtype=bool),
        feature_contract_sha256=sha("features"),
        metric_contract_sha256=sha("metrics"),
        route_policy_sha256=sha("route-policy"),
    )


def dataset():
    candidate_panel = panel()
    return DatasetManifestV2.build(
        (row(0, candidate_panel), row(1, candidate_panel)),
        source_commit=git("dataset"),
    )


def test_canonical_json_roundtrip_recomputes_every_semantic_hash():
    original = dataset()
    payload = dump_dataset_json_v2(original)
    restored = parse_dataset_document_v2(loads_dataset_json_v2(payload))
    assert restored.sha256 == original.sha256
    assert restored.to_document() == original.to_document()
    assert payload.endswith(b"\n")
    assert b" " not in payload


def test_exact_container_hash_is_separate_from_semantic_identity():
    original = dataset()
    canonical = dump_dataset_json_v2(original)
    pretty = json.dumps(original.to_document(), indent=2).encode() + b"\n"
    first = parse_dataset_document_v2(loads_dataset_json_v2(canonical))
    second = parse_dataset_document_v2(loads_dataset_json_v2(pretty))
    assert first.sha256 == second.sha256
    assert dataset_container_sha256_v2(canonical) != (
        dataset_container_sha256_v2(pretty)
    )


def test_duplicate_json_key_and_nonfinite_constant_are_refused():
    with pytest.raises(
        CounterfactualRiskDatasetIOV2Error,
        match="duplicate JSON object key",
    ):
        loads_dataset_json_v2('{"schema":"a","schema":"b"}')
    with pytest.raises(
        CounterfactualRiskDatasetIOV2Error,
        match="non-finite JSON number",
    ):
        loads_dataset_json_v2('{"value":NaN}')


def test_unknown_top_level_row_panel_and_artifact_fields_are_refused():
    document = dataset().to_document()
    document["extra"] = True
    with pytest.raises(
        CounterfactualRiskDatasetIOV2Error,
        match="dataset document keys differ",
    ):
        parse_dataset_document_v2(document)

    document = dataset().to_document()
    document["candidate_panel"]["extra"] = True
    with pytest.raises(
        CounterfactualRiskDatasetIOV2Error,
        match="candidate_panel keys differ",
    ):
        parse_dataset_document_v2(document)

    document = dataset().to_document()
    document["rows"][0]["extra"] = True
    with pytest.raises(
        CounterfactualRiskDatasetIOV2Error,
        match="rows\[0\] keys differ",
    ):
        parse_dataset_document_v2(document)

    document = dataset().to_document()
    document["rows"][0]["candidate_artifacts"][0]["extra"] = True
    with pytest.raises(
        CounterfactualRiskDatasetIOV2Error,
        match="candidate_artifacts\[0\] keys differ",
    ):
        parse_dataset_document_v2(document)


def test_stale_dataset_row_array_and_slot_hashes_are_refused():
    document = dataset().to_document()
    document["dataset_sha256"] = sha("stale-dataset")
    with pytest.raises(
        CounterfactualRiskDatasetIOV2Error,
        match="top-level semantics differ",
    ):
        parse_dataset_document_v2(document)

    document = dataset().to_document()
    document["rows"][0]["row_id"] = sha("stale-row")
    with pytest.raises(
        CounterfactualRiskDatasetIOV2Error,
        match="does not match recomputed semantics",
    ):
        parse_dataset_document_v2(document)

    document = dataset().to_document()
    document["rows"][0]["features"][0] += 1.0
    with pytest.raises(
        CounterfactualRiskDatasetIOV2Error,
        match="does not match recomputed semantics",
    ):
        parse_dataset_document_v2(document)

    document = dataset().to_document()
    document["rows"][0]["candidate_artifacts"][0][
        "recipe_slot_projection_sha256"
    ] = sha("wrong-slot")
    with pytest.raises(
        Exception,
        match="slot projection differs",
    ):
        parse_dataset_document_v2(document)


def test_row_order_and_top_level_panel_disagreement_are_refused():
    document = dataset().to_document()
    document["rows"] = list(reversed(document["rows"]))
    with pytest.raises(
        Exception,
        match="canonical row-ID order",
    ):
        parse_dataset_document_v2(document)

    document = dataset().to_document()
    document["candidate_panel"]["ordered_slots"][0][
        "model_bundle_sha256"
    ] = sha("different-model")
    with pytest.raises(Exception):
        parse_dataset_document_v2(document)


def test_integer_availability_mask_is_refused_not_coerced_to_boolean():
    document = dataset().to_document()
    document["rows"][0]["available"] = [[1, 1, 1], [1, 1, 1]]
    with pytest.raises(Exception, match="availability must be a boolean array"):
        parse_dataset_document_v2(document)


def test_path_loader_returns_exact_container_identity(tmp_path):
    original = dataset()
    payload = dump_dataset_json_v2(original)
    path = tmp_path / "dataset.json"
    path.write_bytes(payload)
    restored, container_sha = load_dataset_path_v2(path)
    assert restored.sha256 == original.sha256
    assert container_sha == dataset_container_sha256_v2(payload)
