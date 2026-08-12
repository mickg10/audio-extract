import hashlib

import numpy as np
import pytest

from audio_extract.counterfactual_risk_cell_partition_v1 import RationalMeasure
from audio_extract.counterfactual_risk_d0_structured_teacher_v1 import (
    READY_FOR_GROUPED_ASSEMBLY,
    build_d0_structured_teacher,
)
from audio_extract.counterfactual_risk_dataset_contract_v1 import (
    GroupFamilyIdentity,
)
from audio_extract.counterfactual_risk_grouped_training_manifest_v1 import (
    READY_FOR_D0_TRAINING,
    UNCERTIFIABLE_INSUFFICIENT_INDEPENDENT_GROUPS,
    GroupedAssemblyMemberV1,
    GroupedTrainingManifestError,
    GroupedTrainingManifestV1,
)
from audio_extract.counterfactual_risk_partitioned_prediction_v1 import (
    PartitionedUpperRiskPanelV1,
)
from audio_extract.counterfactual_risk_routing_preflight_v1 import (
    FrozenRoutingPolicyV1,
    preflight_partitioned_upper_risks,
)


def sha(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode()).hexdigest()


def git(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()


_DEFAULT_EVIDENCE = sha("exact-evidence")


def policy(**kwargs) -> FrozenRoutingPolicyV1:
    values = {
        "critical_thresholds": (("voice", 1.0), ("hole", 1.0)),
        "secondary_weights": (("artifact", 0.1),),
        "critical_slack_weight": 1.0,
        "temporal_switch_penalty": 0.05,
        "frequency_switch_penalty": 0.04,
        "feasibility_tolerance": 0.0,
        "objective_tolerance": 1e-10,
        "whole_track_abstention": True,
        "require_secondary_evidence": True,
    }
    values.update(kwargs)
    return FrozenRoutingPolicyV1(**values)


def panel(routing_policy, *, work_id="work") -> PartitionedUpperRiskPanelV1:
    group = GroupFamilyIdentity(
        work_id=work_id,
        recording_session_id="session",
        target_singer_id="singer",
        source_family_sha256=sha(f"source-family-{work_id}"),
        query_condition_sha256=sha("query"),
    )
    risk = np.asarray(
        [
            [
                [[0.5, 0.2, 3.0], [0.2, 1.0, 1.0]],
                [[1.1, 0.2, 0.5], [0.2, 0.3, 0.2]],
            ]
        ],
        dtype=np.float64,
    )
    mask = np.ones_like(risk, dtype=bool)
    return PartitionedUpperRiskPanelV1(
        group_family_sha256=group.sha256,
        work_id=group.work_id,
        recording_session_id=group.recording_session_id,
        target_singer_id=group.target_singer_id,
        source_family_sha256=group.source_family_sha256,
        query_condition_sha256=group.query_condition_sha256,
        mixture_pcm_sha256=sha("mixture"),
        spectral_grid_sha256=sha("grid"),
        resolution_ms=500,
        time_cell_count=1,
        band_count=2,
        partition_certificate_sha256=sha("partition"),
        prediction_manifest_sha256=sha("exact-risk-panel"),
        model_sha256=sha("exact-oracle"),
        model_input_sha256=sha("exact-input"),
        prediction_policy_sha256=sha("exact-policy"),
        source_commit=git("source"),
        candidate_panel_sha256=sha("candidate-panel"),
        candidate_slot_sha256s=(sha("slot-a"), sha("slot-b")),
        feature_contract_sha256=sha("features"),
        metric_contract_sha256=sha("metrics"),
        route_policy_sha256=routing_policy.sha256,
        metric_names=("voice", "hole", "artifact"),
        metric_units=("ratio", "ratio", "ratio"),
        metric_directions=(
            "lower_is_better",
            "lower_is_better",
            "lower_is_better",
        ),
        inference_row_sha256s=(sha("row-0"), sha("row-1")),
        cell_sha256s=(sha("cell-0"), sha("cell-1")),
        row_prediction_allowed=(True, True),
        measures=(RationalMeasure(1), RationalMeasure(2)),
        upper=risk,
        available=mask,
    )


def ready_teacher(*, work_id="work", evidence=_DEFAULT_EVIDENCE):
    """Return one grouped-assembly-ready teacher and its candidate count."""

    routing_policy = policy()
    value = panel(routing_policy, work_id=work_id)
    preflight = preflight_partitioned_upper_risks(value, routing_policy)
    teacher = build_d0_structured_teacher(
        value,
        preflight,
        routing_policy,
        exact_evidence_sha256=evidence,
        minimum_structured_margin=0.1,
    )
    assert teacher.status == READY_FOR_GROUPED_ASSEMBLY
    return teacher, len(value.candidate_slot_sha256s)


def unavailable_teacher():
    routing_policy = policy(
        secondary_weights=(),
        temporal_switch_penalty=0.0,
        frequency_switch_penalty=0.0,
    )
    value = panel(routing_policy)
    tied = PartitionedUpperRiskPanelV1(
        **{**value.__dict__, "upper": np.full((1, 2, 2, 3), 0.2)}
    )
    preflight = preflight_partitioned_upper_risks(tied, routing_policy)
    teacher = build_d0_structured_teacher(
        tied,
        preflight,
        routing_policy,
        exact_evidence_sha256=_DEFAULT_EVIDENCE,
        minimum_structured_margin=0.1,
    )
    assert teacher.status == "UNAVAILABLE_NONUNIQUE_ROUTE"
    return teacher, len(tied.candidate_slot_sha256s)


def test_distinct_independent_groups_certify_d0_training():
    # Two independent groups with coverage 1 - 0.34 = 0.66:
    # ceil((2 + 1) * 0.66) = ceil(1.98) = 2 <= 2 -> certifiable.
    teachers = [
        ready_teacher(work_id="work-a"),
        ready_teacher(work_id="work-b"),
    ]
    manifest = GroupedTrainingManifestV1.from_teachers(teachers, alpha=0.34)
    assert manifest.n_groups == 2
    assert manifest.conformal_rank == 2
    assert manifest.certifiable
    assert manifest.status == READY_FOR_D0_TRAINING
    assert manifest.training_ready()
    identity = manifest.identity_dict()
    # alpha and n_groups are identity-bearing.
    assert identity["alpha"] == pytest.approx(0.34)
    assert identity["n_groups"] == 2
    assert identity["target_coverage"] == pytest.approx(0.66)


def test_same_members_stricter_alpha_becomes_uncertifiable():
    teachers = [
        ready_teacher(work_id="work-a"),
        ready_teacher(work_id="work-b"),
    ]
    lenient = GroupedTrainingManifestV1.from_teachers(teachers, alpha=0.34)
    # coverage 0.9: ceil((2 + 1) * 0.9) = ceil(2.7) = 3 > 2 -> uncertifiable.
    strict = GroupedTrainingManifestV1.from_teachers(teachers, alpha=0.1)
    assert strict.n_groups == 2
    assert strict.conformal_rank == 3
    assert not strict.certifiable
    assert strict.status == UNCERTIFIABLE_INSUFFICIENT_INDEPENDENT_GROUPS
    assert not strict.training_ready()
    # alpha is identity-bearing: the two manifests are distinct artifacts.
    assert lenient.sha256 != strict.sha256


def test_n_groups_drives_certifiability_at_fixed_alpha():
    # coverage 0.75 (alpha 0.25):
    #   n=2 -> ceil(3 * 0.75) = ceil(2.25) = 3 > 2 -> uncertifiable
    #   n=3 -> ceil(4 * 0.75) = 3 <= 3 -> certifiable
    two = GroupedTrainingManifestV1.from_teachers(
        [ready_teacher(work_id="work-a"), ready_teacher(work_id="work-b")],
        alpha=0.25,
    )
    three = GroupedTrainingManifestV1.from_teachers(
        [
            ready_teacher(work_id="work-a"),
            ready_teacher(work_id="work-b"),
            ready_teacher(work_id="work-c"),
        ],
        alpha=0.25,
    )
    assert two.status == UNCERTIFIABLE_INSUFFICIENT_INDEPENDENT_GROUPS
    assert three.status == READY_FOR_D0_TRAINING
    # n_groups is identity-bearing.
    assert two.sha256 != three.sha256
    assert two.identity_dict()["n_groups"] == 2
    assert three.identity_dict()["n_groups"] == 3


def test_repeated_group_identity_counts_once():
    # Two grouped-assembly-ready teachers for the SAME group (distinct exact
    # evidence -> distinct immutable artifacts) supply only ONE independent
    # group, so the finite-group gate cannot certify from them alone.
    teachers = [
        ready_teacher(work_id="work-a", evidence=sha("evidence-1")),
        ready_teacher(work_id="work-a", evidence=sha("evidence-2")),
    ]
    manifest = GroupedTrainingManifestV1.from_teachers(teachers, alpha=0.34)
    assert len(manifest.members) == 2
    assert manifest.n_groups == 1
    assert manifest.status == UNCERTIFIABLE_INSUFFICIENT_INDEPENDENT_GROUPS
    assert not manifest.training_ready()


def test_non_ready_teacher_cannot_join_the_manifest():
    teacher, candidate_count = unavailable_teacher()
    with pytest.raises(
        GroupedTrainingManifestError, match="grouped-assembly-ready"
    ):
        GroupedAssemblyMemberV1.from_teacher(
            teacher, candidate_count=candidate_count
        )


def test_manifest_rejects_a_non_ready_member_status():
    teacher, candidate_count = ready_teacher()
    member = GroupedAssemblyMemberV1.from_teacher(
        teacher, candidate_count=candidate_count
    )
    corrupt = GroupedAssemblyMemberV1(
        **{**member.__dict__, "status": "UNAVAILABLE_NO_HIGH_MARGIN_CELLS"}
    )
    with pytest.raises(
        GroupedTrainingManifestError, match="grouped-assembly-ready"
    ):
        GroupedTrainingManifestV1(members=(corrupt,), alpha=0.34).validate()


def test_duplicate_teacher_artifact_is_refused():
    teacher, candidate_count = ready_teacher()
    member = GroupedAssemblyMemberV1.from_teacher(
        teacher, candidate_count=candidate_count
    )
    with pytest.raises(GroupedTrainingManifestError, match="repeats a per-panel"):
        GroupedTrainingManifestV1.build([member, member], alpha=0.34)


def test_out_of_order_members_and_empty_manifest_fail_closed():
    teachers = [
        ready_teacher(work_id="work-a"),
        ready_teacher(work_id="work-b"),
    ]
    members = [
        GroupedAssemblyMemberV1.from_teacher(t, candidate_count=c)
        for t, c in teachers
    ]
    ordered = sorted(members, key=lambda m: m.teacher_sha256)
    with pytest.raises(GroupedTrainingManifestError, match="canonical"):
        GroupedTrainingManifestV1(
            members=tuple(reversed(ordered)), alpha=0.34
        ).validate()
    with pytest.raises(GroupedTrainingManifestError, match="no members"):
        GroupedTrainingManifestV1(members=(), alpha=0.34).validate()


@pytest.mark.parametrize("alpha", [0.0, -0.1, 0.5, 0.9, float("nan")])
def test_malformed_alpha_fails_closed(alpha):
    teacher, candidate_count = ready_teacher()
    member = GroupedAssemblyMemberV1.from_teacher(
        teacher, candidate_count=candidate_count
    )
    with pytest.raises(GroupedTrainingManifestError, match="alpha"):
        GroupedTrainingManifestV1(members=(member,), alpha=alpha).validate()


def test_only_this_boundary_emits_a_training_ready_status():
    # The per-panel teacher never emits the training-ready status; the two
    # contracts use disjoint ready vocabularies.
    teacher, _ = ready_teacher()
    assert teacher.status == READY_FOR_GROUPED_ASSEMBLY
    assert teacher.status != READY_FOR_D0_TRAINING
    assert READY_FOR_D0_TRAINING != READY_FOR_GROUPED_ASSEMBLY
