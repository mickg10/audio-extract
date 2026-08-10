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
    SplitManifestV2,
    critical_candidate_feasibility_v2,
    inference_records_v2,
)


def sha(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def git(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def panel(reverse: bool = False) -> CandidatePanel:
    slots = (
        CandidateSlot(
            slot_id="median_mdx_mel_bs",
            model_bundle_sha256=sha("model:median"),
            adapter_bundle_sha256=sha("adapter:median"),
            construction_contract_sha256=sha("construction:median"),
            query_contract_sha256=sha("query:none"),
        ),
        CandidateSlot(
            slot_id="residual_mdx23c",
            model_bundle_sha256=sha("model:mdx"),
            adapter_bundle_sha256=sha("adapter:mdx"),
            construction_contract_sha256=sha("construction:residual"),
            query_contract_sha256=sha("query:none"),
        ),
    )
    return CandidatePanel(tuple(reversed(slots)) if reverse else slots)


def family(index: int, *, source: str | None = None, query: str | None = None):
    return GroupFamilyIdentity(
        work_id=f"work-{index}",
        recording_session_id=f"session-{index}",
        target_singer_id=f"singer-{index}",
        source_family_sha256=sha(source or f"source-{index}"),
        query_condition_sha256=sha(query or f"query-{index}"),
    )


def artifact_binding(slot: CandidateSlot, source_family: str, index: int):
    return CandidateArtifactBinding(
        slot_id=slot.slot_id,
        slot_sha256=slot.sha256,
        recipe_id=sha(f"recipe:{source_family}:{slot.slot_id}:{index}"),
        artifact_pcm_sha256=sha(f"pcm:{source_family}:{slot.slot_id}:{index}"),
        recipe_semantic_sha256=sha(
            f"recipe-semantic:{source_family}:{slot.slot_id}:{index}"
        ),
        recipe_slot_projection_sha256=slot.sha256,
        source_family_sha256=source_family,
        verifier_commit=git("artifact-verifier"),
    )


def row(
    index: int,
    *,
    candidate_panel: CandidatePanel | None = None,
    group_family: GroupFamilyIdentity | None = None,
    artifacts=None,
    exact_risks=None,
    available=None,
):
    candidate_panel = candidate_panel or panel()
    group_family = group_family or family(index)
    candidate_artifacts = tuple(
        artifacts
        if artifacts is not None
        else [
            artifact_binding(
                slot,
                group_family.source_family_sha256,
                index,
            )
            for slot in candidate_panel.slots
        ]
    )
    risks = np.asarray(
        exact_risks
        if exact_risks is not None
        else [
            [0.2 + 0.01 * index, 0.3, 0.4],
            [0.6, 0.1 + 0.01 * index, 0.2],
        ],
        dtype=np.float64,
    )
    availability = np.asarray(
        available if available is not None else np.ones((2, 3), dtype=bool),
        dtype=bool,
    )
    return CounterfactualRiskRowV2(
        group_family=group_family,
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
        candidate_artifacts=candidate_artifacts,
        metric_names=("voice", "hole", "artifact"),
        metric_units=("ratio", "ratio", "ratio"),
        metric_directions=(
            "lower_is_better",
            "lower_is_better",
            "lower_is_better",
        ),
        features=np.asarray([float(index), 0.5, -0.25], dtype=np.float64),
        exact_risks=risks,
        available=availability,
        feature_contract_sha256=sha("features"),
        metric_contract_sha256=sha("metrics"),
        route_policy_sha256=sha("route-policy"),
    )


def dataset(rows=None):
    return DatasetManifestV2.build(
        rows or [row(0), row(1), row(2)],
        source_commit=git("dataset"),
    )


def split_for(value: DatasetManifestV2):
    groups = value.group_family_sha256s
    return SplitManifestV2(
        dataset_sha256=value.sha256,
        train_group_family_sha256s=(groups[0],),
        calibration_group_family_sha256s=(groups[1],),
        test_group_family_sha256s=tuple(groups[2:]),
        split_algorithm="group-family-hash/v2",
        split_seed_sha256=sha("split-seed"),
        selection_manifest_sha256=sha("selection"),
        source_commit=git("dataset"),
    )


def test_stable_panel_accepts_different_per_source_artifacts():
    value = dataset()
    value.validate()
    first_artifacts = tuple(
        item.artifact_pcm_sha256 for item in value.rows[0].candidate_artifacts
    )
    assert any(
        tuple(item.artifact_pcm_sha256 for item in row_value.candidate_artifacts)
        != first_artifacts
        for row_value in value.rows[1:]
    )
    assert len({row_value.candidate_panel.sha256 for row_value in value.rows}) == 1


def test_dataset_build_is_row_order_invariant():
    forward = dataset([row(0), row(1), row(2)])
    backward = dataset([row(2), row(1), row(0)])
    assert forward.sha256 == backward.sha256
    assert [item.row_id for item in forward.rows] == sorted(
        item.row_id for item in forward.rows
    )


def test_candidate_slot_order_is_identity_bearing_and_mixed_panels_are_refused():
    normal = row(0)
    reversed_panel = panel(reverse=True)
    reversed_row = row(0, candidate_panel=reversed_panel)
    assert normal.candidate_panel.sha256 != reversed_row.candidate_panel.sha256
    assert normal.row_id != reversed_row.row_id
    with pytest.raises(
        CounterfactualRiskDatasetV2Error,
        match="disagree on stable panel",
    ):
        dataset([normal, row(1, candidate_panel=reversed_panel), row(2)])


def test_artifact_binding_must_match_slot_projection_and_source_family():
    original = row(0)
    first = original.candidate_artifacts[0]
    bad_projection = CandidateArtifactBinding(
        **{
            **first.__dict__,
            "recipe_slot_projection_sha256": sha("wrong-slot"),
        }
    )
    with pytest.raises(
        CounterfactualRiskDatasetV2Error,
        match="slot projection differs",
    ):
        row(0, artifacts=(bad_projection, original.candidate_artifacts[1])).validate()

    bad_family = CandidateArtifactBinding(
        **{**first.__dict__, "source_family_sha256": sha("other-family")}
    )
    with pytest.raises(
        CounterfactualRiskDatasetV2Error,
        match="different source family",
    ):
        row(0, artifacts=(bad_family, original.candidate_artifacts[1])).validate()

    failed = CandidateArtifactBinding(**{**first.__dict__, "status": "failed"})
    with pytest.raises(
        CounterfactualRiskDatasetV2Error,
        match="status must be passed",
    ):
        row(0, artifacts=(failed, original.candidate_artifacts[1])).validate()


def test_artifact_axis_must_follow_panel_slot_order():
    original = row(0)
    with pytest.raises(
        CounterfactualRiskDatasetV2Error,
        match="slot_id differs from panel order",
    ):
        row(
            0,
            artifacts=tuple(reversed(original.candidate_artifacts)),
        ).validate()


def test_duplicate_stable_slot_and_duplicate_source_artifact_are_refused():
    first = panel().slots[0]
    duplicate_panel = CandidatePanel((first, first))
    with pytest.raises(
        CounterfactualRiskDatasetV2Error,
        match="slot IDs must be unique",
    ):
        duplicate_panel.validate()

    original = row(0)
    duplicate_artifact = CandidateArtifactBinding(
        **{
            **original.candidate_artifacts[1].__dict__,
            "recipe_id": original.candidate_artifacts[0].recipe_id,
            "artifact_pcm_sha256": (
                original.candidate_artifacts[0].artifact_pcm_sha256
            ),
        }
    )
    with pytest.raises(
        CounterfactualRiskDatasetV2Error,
        match="recipe/PCM identities are duplicated",
    ):
        row(
            0,
            artifacts=(original.candidate_artifacts[0], duplicate_artifact),
        ).validate()


def test_availability_is_identity_bearing_and_missing_is_infeasible():
    present = row(
        0,
        exact_risks=[[0.0, 0.3, 0.4], [0.6, 0.1, 0.2]],
    )
    mask = np.ones((2, 3), dtype=bool)
    mask[0, 0] = False
    missing = row(
        0,
        exact_risks=[[0.0, 0.3, 0.4], [0.6, 0.1, 0.2]],
        available=mask,
    )
    assert present.row_id != missing.row_id
    assert critical_candidate_feasibility_v2(
        missing, {"voice": 0.5}
    ).tolist() == [False, False]


def test_unavailable_nonzero_value_is_refused():
    mask = np.ones((2, 3), dtype=bool)
    mask[0, 1] = False
    damaged = row(
        0,
        exact_risks=[[0.2, 9.0, 0.4], [0.6, 0.1, 0.2]],
        available=mask,
    )
    with pytest.raises(
        CounterfactualRiskDatasetV2Error,
        match="unavailable exact-risk entries must be stored as zero",
    ):
        damaged.validate()


def test_valid_split_is_exact_and_derivative_family_leak_is_refused():
    value = dataset()
    split_for(value).validate(value)

    shared = sha("shared-source")
    rows = [
        row(0, group_family=family(0, source="shared-source", query="q0")),
        row(1, group_family=family(1, source="source-1", query="q1")),
        row(2, group_family=family(2, source="source-2", query="q2")),
        row(3, group_family=family(3, source="shared-source", query="q3")),
    ]
    assert rows[0].group_family.source_family_sha256 == shared
    leaky = dataset(rows)
    by_work = {
        item.group_family.work_id: item.group_family.sha256 for item in rows
    }
    split = SplitManifestV2(
        dataset_sha256=leaky.sha256,
        train_group_family_sha256s=(by_work["work-0"],),
        calibration_group_family_sha256s=(by_work["work-1"],),
        test_group_family_sha256s=tuple(
            sorted((by_work["work-2"], by_work["work-3"]))
        ),
        split_algorithm="group-family-hash/v2",
        split_seed_sha256=sha("split-seed"),
        selection_manifest_sha256=sha("selection"),
        source_commit=git("dataset"),
    )
    with pytest.raises(
        CounterfactualRiskDatasetV2Error,
        match="source family has derivatives in multiple partitions",
    ):
        split.validate(leaky)


def test_inference_projection_contains_per_source_artifacts_but_no_truth_labels():
    value = dataset()
    records = inference_records_v2(value)
    assert len(records) == len(value.rows)
    for record in records:
        assert "features" in record
        assert "candidate_artifacts" in record
        assert "mixture_pcm_sha256" in record
        assert "accompaniment_truth_pcm_sha256" not in record
        assert "vocal_truth_pcm_sha256" not in record
        assert "exact_risks" not in record
        assert "available" not in record
        assert "teacher_route" not in record


def test_row_identity_binds_per_source_recipe_and_artifact():
    first = row(0)
    artifact = first.candidate_artifacts[0]
    changed = CandidateArtifactBinding(
        **{
            **artifact.__dict__,
            "recipe_id": sha("changed-recipe"),
            "recipe_semantic_sha256": sha("changed-recipe-semantic"),
        }
    )
    second = row(
        0,
        artifacts=(changed, first.candidate_artifacts[1]),
    )
    assert first.candidate_panel.sha256 == second.candidate_panel.sha256
    assert first.row_id != second.row_id
