import numpy as np
import pytest

from audio_extract.counterfactual_risk_dataset_contract_v1 import (
    CandidateIdentity,
    CellGeometry,
    CounterfactualRiskDatasetError,
    CounterfactualRiskRow,
    DatasetManifest,
    GroupFamilyIdentity,
    SplitManifest,
    critical_candidate_feasibility,
    inference_records,
)


def sha(digit: str) -> str:
    return "sha256:" + digit * 64


def git(digit: str) -> str:
    return digit * 40


def candidates(reverse: bool = False):
    result = (
        CandidateIdentity(sha("1"), sha("3")),
        CandidateIdentity(sha("2"), sha("4")),
    )
    return tuple(reversed(result)) if reverse else result


def group(index: int, *, source: str | None = None, query: str | None = None):
    return GroupFamilyIdentity(
        work_id=f"work-{index}",
        recording_session_id=f"session-{index}",
        target_singer_id=f"singer-{index}",
        source_family_sha256=sha(source or format(index + 5, "x")),
        query_condition_sha256=sha(query or format(index + 9, "x")),
    )


def row(
    index: int,
    *,
    family: GroupFamilyIdentity | None = None,
    reverse_candidates: bool = False,
    exact_risks=None,
    available=None,
):
    risk_values = np.asarray(
        exact_risks
        if exact_risks is not None
        else [[0.20 + 0.01 * index, 0.30, 0.40], [0.60, 0.10, 0.20]],
        dtype=np.float64,
    )
    availability = np.asarray(
        available if available is not None else np.ones((2, 3), dtype=bool),
        dtype=bool,
    )
    return CounterfactualRiskRow(
        group_family=family or group(index),
        mixture_pcm_sha256=sha("a"),
        accompaniment_truth_pcm_sha256=sha("b"),
        vocal_truth_pcm_sha256=sha("c"),
        cell=CellGeometry(
            sample_rate_hz=44_100,
            resolution_ms=500,
            start_frame=index * 1000,
            end_frame=(index + 1) * 1000,
            band_low_hz=0,
            band_high_hz=500,
            spectral_grid_sha256=sha("d"),
        ),
        candidates=candidates(reverse_candidates),
        metric_names=("voice", "hole", "artifact"),
        metric_units=("ratio", "ratio", "ratio"),
        metric_directions=(
            "lower_is_better",
            "lower_is_better",
            "lower_is_better",
        ),
        features=np.asarray([float(index), 0.5, -0.25], dtype=np.float64),
        exact_risks=risk_values,
        available=availability,
        feature_contract_sha256=sha("e"),
        metric_contract_sha256=sha("f"),
        route_policy_sha256=sha("0"),
    )


def dataset(rows=None):
    return DatasetManifest.build(
        rows or [row(0), row(1), row(2)],
        source_commit=git("1"),
    )


def split_for(value: DatasetManifest):
    groups = value.group_family_sha256s
    return SplitManifest(
        dataset_sha256=value.sha256,
        train_group_family_sha256s=(groups[0],),
        calibration_group_family_sha256s=(groups[1],),
        test_group_family_sha256s=tuple(groups[2:]),
        split_algorithm="group-family-hash/v1",
        split_seed_sha256=sha("2"),
        selection_manifest_sha256=sha("3"),
        source_commit=git("1"),
    )


def test_dataset_build_is_order_invariant_and_identity_complete():
    forward = dataset([row(0), row(1), row(2)])
    backward = dataset([row(2), row(1), row(0)])
    assert forward.sha256 == backward.sha256
    assert [item.row_id for item in forward.rows] == sorted(
        item.row_id for item in forward.rows
    )
    assert forward.to_document()["dataset_sha256"] == forward.sha256


def test_candidate_order_is_identity_bearing_and_cannot_mix_inside_dataset():
    normal = row(0)
    reversed_panel = row(0, reverse_candidates=True)
    assert normal.row_id != reversed_panel.row_id
    with pytest.raises(
        CounterfactualRiskDatasetError,
        match="disagree on frozen panel",
    ):
        dataset([normal, row(1, reverse_candidates=True), row(2)])


def test_availability_is_identity_bearing_and_missing_never_means_safe_zero():
    present = row(0, exact_risks=[[0.0, 0.3, 0.4], [0.6, 0.1, 0.2]])
    mask = np.ones((2, 3), dtype=bool)
    mask[0, 0] = False
    missing = row(
        0,
        exact_risks=[[0.0, 0.3, 0.4], [0.6, 0.1, 0.2]],
        available=mask,
    )
    assert present.row_id != missing.row_id
    feasible = critical_candidate_feasibility(missing, {"voice": 0.5})
    assert feasible.tolist() == [False, False]


def test_unavailable_entry_with_hidden_nonzero_value_is_refused():
    mask = np.ones((2, 3), dtype=bool)
    mask[0, 1] = False
    damaged = row(
        0,
        exact_risks=[[0.2, 99.0, 0.4], [0.6, 0.1, 0.2]],
        available=mask,
    )
    with pytest.raises(
        CounterfactualRiskDatasetError,
        match="unavailable exact-risk entries must be stored as zero",
    ):
        damaged.validate()


def test_valid_split_is_an_exact_disjoint_group_partition():
    value = dataset()
    split = split_for(value)
    split.validate(value)
    assert split.sha256(value).startswith("sha256:")


def test_split_refuses_derivatives_of_one_source_family_across_partitions():
    shared_source = "5"
    rows = [
        row(0, family=group(0, source=shared_source, query="9")),
        row(1, family=group(1, source="6", query="a")),
        row(2, family=group(2, source="7", query="b")),
        row(3, family=group(3, source=shared_source, query="c")),
    ]
    value = dataset(rows)
    by_index = {item.group_family.work_id: item.group_family.sha256 for item in rows}
    split = SplitManifest(
        dataset_sha256=value.sha256,
        train_group_family_sha256s=tuple(sorted((by_index["work-0"],))),
        calibration_group_family_sha256s=tuple(sorted((by_index["work-1"],))),
        test_group_family_sha256s=tuple(
            sorted((by_index["work-2"], by_index["work-3"]))
        ),
        split_algorithm="group-family-hash/v1",
        split_seed_sha256=sha("2"),
        selection_manifest_sha256=sha("3"),
        source_commit=git("1"),
    )
    with pytest.raises(
        CounterfactualRiskDatasetError,
        match="source family has derivatives in multiple partitions",
    ):
        split.validate(value)


def test_split_refuses_missing_and_duplicate_group_assignments():
    value = dataset()
    groups = value.group_family_sha256s
    missing = SplitManifest(
        dataset_sha256=value.sha256,
        train_group_family_sha256s=(groups[0],),
        calibration_group_family_sha256s=(groups[1],),
        test_group_family_sha256s=(sha("9"),),
        split_algorithm="group-family-hash/v1",
        split_seed_sha256=sha("2"),
        selection_manifest_sha256=sha("3"),
        source_commit=git("1"),
    )
    with pytest.raises(
        CounterfactualRiskDatasetError,
        match="do not exactly partition",
    ):
        missing.validate(value)

    overlap = SplitManifest(
        dataset_sha256=value.sha256,
        train_group_family_sha256s=(groups[0],),
        calibration_group_family_sha256s=(groups[0],),
        test_group_family_sha256s=tuple(groups[1:]),
        split_algorithm="group-family-hash/v1",
        split_seed_sha256=sha("2"),
        selection_manifest_sha256=sha("3"),
        source_commit=git("1"),
    )
    with pytest.raises(
        CounterfactualRiskDatasetError,
        match="partitions overlap",
    ):
        overlap.validate(value)


def test_inference_projection_exposes_features_but_no_clean_truth_or_labels():
    value = dataset()
    records = inference_records(value)
    assert len(records) == len(value.rows)
    for record in records:
        assert "features" in record
        assert "mixture_pcm_sha256" in record
        assert "accompaniment_truth_pcm_sha256" not in record
        assert "vocal_truth_pcm_sha256" not in record
        assert "exact_risks" not in record
        assert "available" not in record
        assert "teacher_route" not in record


def test_unknown_inference_group_and_unknown_critical_metric_are_refused():
    value = dataset()
    with pytest.raises(
        CounterfactualRiskDatasetError,
        match="absent from the dataset",
    ):
        inference_records(value, group_family_sha256s=(sha("8"),))
    with pytest.raises(
        CounterfactualRiskDatasetError,
        match="unknown critical metric",
    ):
        critical_candidate_feasibility(value.rows[0], {"other": 1.0})


def test_cell_geometry_and_query_condition_are_row_identity_bearing():
    first = row(0)
    other_query = row(0, family=group(0, source="5", query="8"))
    shifted = CounterfactualRiskRow(
        **{
            **first.__dict__,
            "cell": CellGeometry(
                **{**first.cell.__dict__, "start_frame": 1, "end_frame": 1001}
            ),
        }
    )
    assert first.row_id != other_query.row_id
    assert first.row_id != shifted.row_id
