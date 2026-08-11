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
    CounterfactualRiskDatasetV2Error,
    CounterfactualRiskRowV2,
    DatasetManifestV2,
    SplitManifestV2,
    critical_candidate_feasibility_v2,
    inference_records_v2,
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


def row(index: int, source_name: str, *, risks=None, available=None):
    candidate_panel = panel()
    source = sha(source_name)
    artifacts = tuple(
        CandidateArtifactBinding(
            slot_id=slot.slot_id,
            slot_sha256=slot.sha256,
            recipe_id=sha(f"recipe-{index}-{slot.slot_id}"),
            artifact_pcm_sha256=sha(f"pcm-{index}-{slot.slot_id}"),
            recipe_semantic_sha256=sha(f"semantic-{index}-{slot.slot_id}"),
            recipe_slot_projection_sha256=slot.sha256,
            source_family_sha256=source,
            verifier_commit=git("verifier"),
        )
        for slot in candidate_panel.slots
    )
    risk_values = np.asarray(
        risks if risks is not None else [[0.1, 0.2], [0.2, 0.1]],
        dtype=np.float64,
    )
    mask = np.asarray(
        available if available is not None else np.ones((2, 2), dtype=bool),
        dtype=bool,
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
            start_frame=index * 100,
            end_frame=(index + 1) * 100,
            band_low_hz=0,
            band_high_hz=500,
            spectral_grid_sha256=sha("grid"),
        ),
        candidate_panel=candidate_panel,
        candidate_artifacts=artifacts,
        metric_names=("voice", "hole"),
        metric_units=("ratio", "ratio"),
        metric_directions=("lower_is_better", "lower_is_better"),
        features=np.asarray([float(index), 1.0]),
        exact_risks=risk_values,
        available=mask,
        feature_contract_sha256=sha("features"),
        metric_contract_sha256=sha("metrics"),
        route_policy_sha256=sha("policy"),
    )


def split_dataset():
    # Groups 0 and 3 are derivatives of one exact source family despite having
    # different work/session/query group identities.
    rows = (
        row(0, "shared-source"),
        row(1, "source-one"),
        row(2, "source-two"),
        row(3, "shared-source"),
    )
    dataset = DatasetManifestV2.build(rows, source_commit=git("dataset"))
    group_by_work = {
        item.group_family.work_id: item.group_family.sha256 for item in rows
    }
    source_by_group = {
        item.group_family.sha256: item.group_family.source_family_sha256
        for item in rows
    }
    ordered_groups = tuple(
        group_by_work[f"work-{index}"] for index in range(len(rows))
    )
    return dataset, ordered_groups, source_by_group


def test_exhaustive_four_group_split_acceptance_matches_abstract_invariant():
    dataset, groups, source_by_group = split_dataset()
    checked = 0
    for assignment in itertools.product(range(3), repeat=len(groups)):
        partitions = tuple(
            tuple(sorted(group for group, side in zip(groups, assignment) if side == index))
            for index in range(3)
        )
        split = SplitManifestV2(
            dataset_sha256=dataset.sha256,
            train_group_family_sha256s=partitions[0],
            calibration_group_family_sha256s=partitions[1],
            test_group_family_sha256s=partitions[2],
            split_algorithm="exhaustive-assignment/v2",
            split_seed_sha256=sha("seed"),
            selection_manifest_sha256=sha("selection"),
            source_commit=git("dataset"),
        )
        nonempty = all(partitions)
        source_partition: dict[str, int] = {}
        source_consistent = True
        for group, side in zip(groups, assignment):
            source = source_by_group[group]
            previous = source_partition.setdefault(source, side)
            source_consistent &= previous == side
        expected = nonempty and source_consistent
        if expected:
            split.validate(dataset)
        else:
            with pytest.raises(CounterfactualRiskDatasetV2Error):
                split.validate(dataset)
        checked += 1
    assert checked == 3 ** 4


def test_exhaustive_availability_and_threshold_states_match_fail_closed_rule():
    checked = 0
    threshold = 0.5
    for risk_bits in itertools.product((0.25, 0.75), repeat=4):
        risks = np.asarray(risk_bits, dtype=np.float64).reshape(2, 2)
        for mask_bits in itertools.product((False, True), repeat=4):
            mask = np.asarray(mask_bits, dtype=bool).reshape(2, 2)
            # Unavailable exact-risk entries are stored as canonical zero (the
            # dataset-v2 invariant); feasibility is gated by availability, so the
            # expected fail-closed result is unchanged.
            value = row(
                0, "source", risks=np.where(mask, risks, 0.0), available=mask
            )
            observed = critical_candidate_feasibility_v2(
                value, {"voice": threshold, "hole": threshold}
            )
            expected = np.all(mask & (risks <= threshold), axis=1)
            assert np.array_equal(observed, expected)
            checked += 1
    assert checked == 2 ** 8


def test_inference_projection_keyset_is_disjoint_from_offline_truth_keyset():
    value = DatasetManifestV2.build(
        (row(0, "source"),), source_commit=git("dataset")
    )
    record = inference_records_v2(value)[0]
    # The inference projection must exclude only truth-bearing values (exact
    # risks and their identity, clean accompaniment/vocal truth, availability of
    # exact risks).  Provenance hashes the router legitimately needs -- feature
    # identity, metric-contract and route-policy -- are KEPT, not forbidden.
    truth_bearing_only = {
        "accompaniment_truth_pcm_sha256",
        "vocal_truth_pcm_sha256",
        "exact_risks",
        "exact_risks_sha256",
        "available",
        "availability_sha256",
    }
    assert set(record).isdisjoint(truth_bearing_only)
