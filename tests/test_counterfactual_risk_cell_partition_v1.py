import hashlib
from fractions import Fraction

import numpy as np
import pytest

from audio_extract.counterfactual_risk_cell_partition_v1 import (
    CellPartitionCertificate,
    CellPartitionEntry,
    CellPartitionError,
    CellPartitionRegistry,
    RationalMeasure,
    normalized_row_measures,
    validate_dataset_cell_partitions,
)
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


def group() -> GroupFamilyIdentity:
    return GroupFamilyIdentity(
        work_id="work",
        recording_session_id="session",
        target_singer_id="singer",
        source_family_sha256=sha("source-family"),
        query_condition_sha256=sha("query"),
    )


def exact_row(time_index: int, band_index: int) -> CounterfactualRiskRowV2:
    candidate_panel = panel()
    group_family = group()
    artifacts = tuple(
        CandidateArtifactBinding(
            slot_id=slot.slot_id,
            slot_sha256=slot.sha256,
            recipe_id=sha(f"recipe-{slot.slot_id}"),
            artifact_pcm_sha256=sha(f"pcm-{slot.slot_id}"),
            recipe_semantic_sha256=sha(f"semantic-{slot.slot_id}"),
            recipe_slot_projection_sha256=slot.sha256,
            source_family_sha256=group_family.source_family_sha256,
            verifier_commit=git("artifact-verifier"),
        )
        for slot in candidate_panel.slots
    )
    index = 2 * time_index + band_index
    return CounterfactualRiskRowV2(
        group_family=group_family,
        mixture_pcm_sha256=sha("mixture"),
        accompaniment_truth_pcm_sha256=sha("a-truth"),
        vocal_truth_pcm_sha256=sha("v-truth"),
        cell=CellGeometry(
            sample_rate_hz=44_100,
            resolution_ms=500,
            start_frame=time_index * 1000,
            end_frame=(time_index + 1) * 1000,
            band_low_hz=band_index * 500,
            band_high_hz=(band_index + 1) * 500,
            spectral_grid_sha256=sha("grid"),
        ),
        candidate_panel=candidate_panel,
        candidate_artifacts=artifacts,
        metric_names=("voice", "hole"),
        metric_units=("ratio", "ratio"),
        metric_directions=("lower_is_better", "lower_is_better"),
        features=np.asarray([float(index), 1.0]),
        exact_risks=np.asarray([[0.1, 0.2], [0.2, 0.1]]),
        available=np.ones((2, 2), dtype=bool),
        feature_contract_sha256=sha("features"),
        metric_contract_sha256=sha("metrics"),
        route_policy_sha256=sha("policy"),
    )


def complete_rows():
    return tuple(
        exact_row(time_index, band_index)
        for time_index in range(2)
        for band_index in range(2)
    )


def dataset(rows=None):
    return DatasetManifestV2.build(
        rows or complete_rows(), source_commit=git("dataset")
    )


def certificate(rows=None, *, measures=(1, 2, 3, 4)):
    values = rows or complete_rows()
    entries = tuple(
        CellPartitionEntry(
            time_index=time_index,
            band_index=band_index,
            cell_sha256=values[2 * time_index + band_index].cell.sha256,
            measure=RationalMeasure(measures[2 * time_index + band_index]),
        )
        for time_index in range(2)
        for band_index in range(2)
    )
    return CellPartitionCertificate(
        group_family_sha256=group().sha256,
        spectral_grid_sha256=sha("grid"),
        resolution_ms=500,
        time_cell_count=2,
        band_count=2,
        entries=entries,
        time_ranges_sha256=sha("time-ranges"),
        frequency_ranges_sha256=sha("frequency-ranges"),
        measure_contract_sha256=sha("measure-contract"),
        exact_source_report_sha256=sha("source-report"),
        verifier_commit=git("partition-verifier"),
    )


def registry(cert=None):
    return CellPartitionRegistry.build(
        (cert or certificate(),), source_commit=git("registry")
    )


def test_complete_rectangular_partition_binds_dataset_and_exact_measure():
    value = dataset()
    partitions = registry()
    validate_dataset_cell_partitions(value, partitions)
    weights = normalized_row_measures(value, partitions)
    by_cell = {row.cell.sha256: weights[row.row_id] for row in value.rows}
    expected = {
        row.cell.sha256: Fraction(index + 1, 10)
        for index, row in enumerate(complete_rows())
    }
    assert by_cell == expected
    assert sum(weights.values(), start=Fraction(0, 1)) == 1


def test_missing_or_extra_dataset_cell_is_refused():
    rows = complete_rows()
    with pytest.raises(CellPartitionError, match="does not exactly cover"):
        validate_dataset_cell_partitions(
            dataset(rows[:-1]),
            registry(certificate(rows)),
        )

    extra = CounterfactualRiskRowV2(
        **{
            **rows[-1].__dict__,
            "cell": CellGeometry(
                **{
                    **rows[-1].cell.__dict__,
                    "start_frame": 2000,
                    "end_frame": 3000,
                }
            ),
        }
    )
    with pytest.raises(CellPartitionError, match="no matching certified cell partition|absent"):
        validate_dataset_cell_partitions(
            dataset((*rows, extra)),
            registry(certificate(rows)),
        )


def test_noncanonical_or_incomplete_rectangular_entries_are_refused():
    base = certificate()
    reversed_entries = tuple(reversed(base.entries))
    with pytest.raises(CellPartitionError, match="row-major coverage"):
        CellPartitionCertificate(
            **{**base.__dict__, "entries": reversed_entries}
        ).validate()

    with pytest.raises(CellPartitionError, match="entry count differs"):
        CellPartitionCertificate(
            **{**base.__dict__, "entries": base.entries[:-1]}
        ).validate()

    duplicate = tuple(
        [base.entries[0], base.entries[0], *base.entries[2:]]
    )
    with pytest.raises(CellPartitionError):
        CellPartitionCertificate(
            **{**base.__dict__, "entries": duplicate}
        ).validate()


def test_measure_must_be_positive_reduced_rational():
    with pytest.raises(CellPartitionError, match="positive reduced rational"):
        RationalMeasure(2, 2).validate()
    with pytest.raises(CellPartitionError, match=">= 1"):
        RationalMeasure(0, 1).validate()
    assert RationalMeasure.from_fraction(Fraction(3, 7)).fraction == Fraction(3, 7)


def test_scaled_physical_measures_have_same_normalized_weights_but_new_identity():
    first = certificate(measures=(1, 2, 3, 4))
    second = certificate(measures=(10, 20, 30, 40))
    first_weights = [
        first.normalized_measure(entry.cell_sha256) for entry in first.entries
    ]
    second_weights = [
        second.normalized_measure(entry.cell_sha256) for entry in second.entries
    ]
    assert first_weights == second_weights
    assert first.sha256 != second.sha256


def test_registry_rejects_duplicate_partition_key_and_cross_partition_cell_alias():
    first = certificate()
    with pytest.raises(CellPartitionError, match="partition keys must be unique"):
        CellPartitionRegistry.build(
            (first, first), source_commit=git("registry")
        )

    rows = complete_rows()
    other_group = GroupFamilyIdentity(
        work_id="other",
        recording_session_id="other-session",
        target_singer_id="other-singer",
        source_family_sha256=sha("other-source"),
        query_condition_sha256=sha("other-query"),
    )
    second = CellPartitionCertificate(
        **{
            **first.__dict__,
            "group_family_sha256": other_group.sha256,
            "spectral_grid_sha256": sha("other-grid"),
        }
    )
    with pytest.raises(CellPartitionError, match="appears in multiple"):
        CellPartitionRegistry.build(
            (first, second), source_commit=git("registry")
        )


def test_partition_for_absent_group_and_row_without_partition_are_refused():
    value = dataset()
    base = certificate()
    unrelated = CellPartitionCertificate(
        **{
            **base.__dict__,
            "group_family_sha256": sha("absent-group"),
            "spectral_grid_sha256": sha("other-grid"),
        }
    )
    with pytest.raises(CellPartitionError, match="no matching certified"):
        validate_dataset_cell_partitions(
            value,
            CellPartitionRegistry.build(
                (unrelated,), source_commit=git("registry")
            ),
        )
