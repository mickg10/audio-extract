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
    SplitManifestV2,
)
from audio_extract.counterfactual_risk_group_scope_v1 import (
    BoundGroupedRiskStudentV1,
    GroupCalibrationScopeV1,
    GroupScopeError,
    GroupSubsetManifestV1,
)
from audio_extract.group_conformal_risk_student import (
    GroupConformalRiskError,
    fit_grouped_linear_risk_student,
)


def sha(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode()).hexdigest()


def git(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()


SOURCE_COMMIT = git("source")
FEATURE_CONTRACT = sha("features")
METRIC_CONTRACT = sha("metrics")


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
    group_index: int,
    *,
    cell_index: int = 0,
    source_name: str | None = None,
) -> CounterfactualRiskRowV2:
    candidate_panel = panel()
    source_name = source_name or f"source-{group_index}"
    source_family = sha(source_name)
    artifacts = tuple(
        CandidateArtifactBinding(
            slot_id=slot.slot_id,
            slot_sha256=slot.sha256,
            recipe_id=sha(f"recipe-{group_index}-{slot.slot_id}"),
            artifact_pcm_sha256=sha(f"pcm-{group_index}-{slot.slot_id}"),
            recipe_semantic_sha256=sha(
                f"semantic-{group_index}-{slot.slot_id}"
            ),
            recipe_slot_projection_sha256=slot.sha256,
            source_family_sha256=source_family,
            verifier_commit=git("artifact-verifier"),
        )
        for slot in candidate_panel.slots
    )
    x0 = float(group_index) / 10.0
    x1 = float(cell_index) / 10.0
    return CounterfactualRiskRowV2(
        group_family=GroupFamilyIdentity(
            work_id=f"work-{group_index}",
            recording_session_id=f"session-{group_index}",
            target_singer_id=f"singer-{group_index}",
            source_family_sha256=source_family,
            query_condition_sha256=sha(f"query-{group_index}"),
        ),
        mixture_pcm_sha256=sha(f"mixture-{group_index}"),
        accompaniment_truth_pcm_sha256=sha(f"a-{group_index}"),
        vocal_truth_pcm_sha256=sha(f"v-{group_index}"),
        cell=CellGeometry(
            sample_rate_hz=44_100,
            resolution_ms=500,
            start_frame=cell_index * 1000,
            end_frame=(cell_index + 1) * 1000,
            band_low_hz=0,
            band_high_hz=500,
            spectral_grid_sha256=sha("grid"),
        ),
        candidate_panel=candidate_panel,
        candidate_artifacts=artifacts,
        metric_names=("voice", "hole"),
        metric_units=("ratio", "ratio"),
        metric_directions=("lower_is_better", "lower_is_better"),
        features=np.asarray([x0, x1], dtype=np.float64),
        exact_risks=np.asarray(
            [
                [0.1 + 0.2 * x0, 0.2 + 0.1 * x1],
                [0.3 + 0.1 * x1, 0.1 + 0.2 * x0],
            ],
            dtype=np.float64,
        ),
        available=np.ones((2, 2), dtype=bool),
        feature_contract_sha256=FEATURE_CONTRACT,
        metric_contract_sha256=METRIC_CONTRACT,
        route_policy_sha256=sha("route-policy"),
    )


def dataset_and_split(*, calibration_groups: int = 9, duplicate_cell=False):
    rows = [exact_row(0)]
    rows.extend(exact_row(index + 1) for index in range(calibration_groups))
    rows.append(exact_row(calibration_groups + 1))
    if duplicate_cell:
        # A second cell from one calibration source family must not create a
        # second exchangeability unit.
        rows.append(exact_row(1, cell_index=1))
    dataset = DatasetManifestV2.build(rows, source_commit=SOURCE_COMMIT)
    by_work = {
        row.group_family.work_id: row.group_family.sha256
        for row in dataset.rows
    }
    split = SplitManifestV2(
        dataset_sha256=dataset.sha256,
        train_group_family_sha256s=(by_work["work-0"],),
        calibration_group_family_sha256s=tuple(
            sorted(
                by_work[f"work-{index + 1}"]
                for index in range(calibration_groups)
            )
        ),
        test_group_family_sha256s=(
            by_work[f"work-{calibration_groups + 1}"],
        ),
        split_algorithm="source-family-frozen/v1",
        split_seed_sha256=sha("seed"),
        selection_manifest_sha256=sha("selection"),
        source_commit=SOURCE_COMMIT,
    )
    split.validate(dataset)
    return dataset, split


def scope(dataset, split, *, target_coverage=0.90):
    return GroupCalibrationScopeV1.build(
        dataset,
        split,
        source_registry_sha256=sha("source-registry"),
        partition_registry_sha256=sha("partition-registry"),
        feature_registry_sha256=sha("feature-registry"),
        metric_scales_sha256=sha("metric-scales"),
        target_coverage=target_coverage,
        exchangeability_policy_sha256=sha("exchangeability-policy"),
        hyperparameter_selection_manifest_sha256=sha("hyperparameters"),
        query_quality_strata_policy_sha256=sha("query-strata"),
        source_commit=SOURCE_COMMIT,
        verifier_commit=git("scope-verifier"),
    )


def rows_for_groups(dataset, group_sha256s):
    group_set = set(group_sha256s)
    return tuple(
        row for row in dataset.rows
        if row.group_family.sha256 in group_set
    )


def fitted_student(dataset, split, calibration_scope):
    train_rows = rows_for_groups(
        dataset, split.train_group_family_sha256s
    )
    calibration_rows = rows_for_groups(
        dataset, split.calibration_group_family_sha256s
    )
    return fit_grouped_linear_risk_student(
        np.asarray([row.features for row in train_rows]),
        np.asarray([row.exact_risks for row in train_rows]),
        [row.group_family.source_family_sha256 for row in train_rows],
        np.asarray([row.features for row in calibration_rows]),
        np.asarray([row.exact_risks for row in calibration_rows]),
        [
            row.group_family.source_family_sha256
            for row in calibration_rows
        ],
        metric_scales=(1.0, 1.0),
        target_coverage=calibration_scope.target_coverage,
        ridge=1e-6,
        candidate_panel_sha256=dataset.rows[0].candidate_panel.sha256,
        feature_contract_sha256=FEATURE_CONTRACT,
        metric_contract_sha256=METRIC_CONTRACT,
        training_manifest_sha256=calibration_scope.train.sha256,
        calibration_manifest_sha256=(
            calibration_scope.calibration.sha256
        ),
        split_manifest_sha256=split.sha256(dataset),
        code_commit=SOURCE_COMMIT,
    )


def test_valid_scope_binds_source_family_units_and_existing_student_certificate():
    dataset, split = dataset_and_split()
    calibration_scope = scope(dataset, split)
    student = fitted_student(dataset, split, calibration_scope)
    bound = BoundGroupedRiskStudentV1(student, calibration_scope)
    bound.validate(dataset, split)
    assert calibration_scope.calibration_group_count == 9
    assert calibration_scope.conformal_rank == 9
    assert student.calibration.calibration_group_count == 9
    assert bound.sha256(dataset, split).startswith("sha256:")


def test_extra_cells_and_query_rows_do_not_inflate_independent_group_count():
    dataset, split = dataset_and_split(duplicate_cell=True)
    calibration_scope = scope(dataset, split)
    calibration_rows = rows_for_groups(
        dataset, split.calibration_group_family_sha256s
    )
    assert len(calibration_rows) == 10
    assert calibration_scope.calibration_group_count == 9
    assert len(
        {
            row.group_family.source_family_sha256
            for row in calibration_rows
        }
    ) == 9
    student = fitted_student(dataset, split, calibration_scope)
    BoundGroupedRiskStudentV1(student, calibration_scope).validate(
        dataset, split
    )


def test_eight_source_families_cannot_claim_ninety_percent_finite_coverage():
    dataset, split = dataset_and_split(calibration_groups=8)
    with pytest.raises(
        GroupScopeError,
        match="insufficient independent source families",
    ):
        scope(dataset, split, target_coverage=0.90)


def test_subset_manifest_is_exactly_reconstructed_from_dataset_and_split():
    dataset, split = dataset_and_split()
    manifest = GroupSubsetManifestV1.build(
        dataset, split, partition="calibration"
    )
    changed = GroupSubsetManifestV1(
        **{
            **manifest.__dict__,
            "ordered_row_ids": manifest.ordered_row_ids[:-1],
        }
    )
    with pytest.raises(GroupScopeError, match="differs from exact"):
        changed.validate(dataset, split)


def test_bound_student_refuses_manifest_coverage_count_rank_and_code_substitution():
    dataset, split = dataset_and_split()
    calibration_scope = scope(dataset, split)
    student = fitted_student(dataset, split, calibration_scope)
    calibration = student.calibration
    mutations = (
        ("training_manifest_sha256", sha("other-training"), "training subset"),
        (
            "calibration_manifest_sha256",
            sha("other-calibration"),
            "calibration subset",
        ),
        ("split_manifest_sha256", sha("other-split"), "split"),
        # A certifiable target (ceil((9+1)*0.80)=8 <= 9) that stays <= the fitted
        # guaranteed coverage but mismatches the frozen scope target (0.90): this
        # exercises the scope-binding mismatch, not the finite-group coverage gate.
        ("target_coverage", 0.80, "target coverage"),
        ("calibration_group_count", 10, "calibration group count"),
        ("conformal_rank", 8, "conformal rank"),
        # group_score_mode is a singleton ("max_cells_candidates_metrics/v1"),
        # so no valid-but-different substitution exists: an invalid mode is
        # refused fail-closed by the calibration mode safety check (kept intact),
        # not by the scope-binding comparison, so it is not a scope-substitution
        # case (see the mode fail-closed check below).
        ("code_commit", git("other-code"), "code commit"),
    )
    for field, value, expected in mutations:
        changed_calibration = calibration.__class__(
            **{**calibration.__dict__, field: value}
        )
        changed_student = student.__class__(
            **{**student.__dict__, "calibration": changed_calibration}
        )
        with pytest.raises(GroupScopeError, match=expected):
            BoundGroupedRiskStudentV1(
                changed_student, calibration_scope
            ).validate(dataset, split)

    # A substituted group-score mode is a singleton, so it cannot be a valid
    # scope-binding mismatch; it is refused fail-closed by the calibration's mode
    # safety check.
    mode_substituted = student.__class__(
        **{
            **student.__dict__,
            "calibration": calibration.__class__(
                **{**calibration.__dict__, "group_score_mode": "other"}
            ),
        }
    )
    with pytest.raises(GroupConformalRiskError, match="unknown group score mode"):
        BoundGroupedRiskStudentV1(
            mode_substituted, calibration_scope
        ).validate(dataset, split)


def test_scope_identity_binds_exchangeability_hyperparameters_strata_and_components():
    dataset, split = dataset_and_split()
    original = scope(dataset, split)
    changed_exchangeability = GroupCalibrationScopeV1(
        **{
            **original.__dict__,
            "exchangeability_policy_sha256": sha("other-exchangeability"),
        }
    )
    changed_hyperparameters = GroupCalibrationScopeV1(
        **{
            **original.__dict__,
            "hyperparameter_selection_manifest_sha256": sha(
                "other-hyperparameters"
            ),
        }
    )
    changed_strata = GroupCalibrationScopeV1(
        **{
            **original.__dict__,
            "query_quality_strata_policy_sha256": sha("other-strata"),
        }
    )
    changed_registry = GroupCalibrationScopeV1(
        **{
            **original.__dict__,
            "source_registry_sha256": sha("other-source-registry"),
        }
    )
    assert len(
        {
            original.sha256,
            changed_exchangeability.sha256,
            changed_hyperparameters.sha256,
            changed_strata.sha256,
            changed_registry.sha256,
        }
    ) == 5


def test_scope_source_commit_and_status_are_fail_closed():
    dataset, split = dataset_and_split()
    original = scope(dataset, split)
    with pytest.raises(GroupScopeError, match="different source commits"):
        GroupCalibrationScopeV1(
            **{**original.__dict__, "source_commit": git("other-source")}
        ).validate(dataset, split)
    with pytest.raises(GroupScopeError, match="status must be passed"):
        GroupCalibrationScopeV1(
            **{**original.__dict__, "status": "failed"}
        ).validate(dataset, split)
