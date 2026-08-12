import dataclasses
import hashlib
import json

import numpy as np
import pytest

from audio_extract.counterfactual_risk_candidate_projection_v1 import (
    CandidateProjectionRegistry,
    CandidateSlotProjectionCertificate,
)
from audio_extract.counterfactual_risk_cell_partition_v1 import (
    CellPartitionCertificate,
    CellPartitionEntry,
    CellPartitionRegistry,
    RationalMeasure,
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
    SplitManifestV2,
)
from audio_extract.counterfactual_risk_evidence_bundle_v1 import EvidenceBundleV1
from audio_extract.counterfactual_risk_feature_evidence_v1 import (
    FeatureEvidenceCertificate,
    FeatureEvidenceRegistry,
)
from audio_extract.counterfactual_risk_grouped_comparison_v1 import HeldOutUnitV1
from audio_extract.counterfactual_risk_source_family_v2 import (
    ArtifactNode,
    ArtifactRef,
    SourceFamilyCertificateV2,
    SourceFamilyRegistryV2,
)
from audio_extract.counterfactual_risk_source_lineage_v3 import (
    ExactSourceLineageV3,
    ExactSourceLineageV3Error,
    ExactSourceReportDocumentV3,
)


def sha(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode()).hexdigest()


def git(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()


def context(name="opera"):
    seed = HeldOutUnitV1(
        work_id=name,
        recording_session_id=f"session-{name}",
        target_singer_id=f"singer-{name}",
        source_family_sha256=sha(f"source-{name}"),
        query_condition_sha256=sha(f"query-{name}"),
        query_quality_stratum_sha256=sha(f"quality-{name}"),
        exact_evidence_sha256=sha("placeholder"),
        exact_panel_sha256=sha(f"panel-{name}"),
        exact_preflight_sha256=sha(f"preflight-{name}"),
        exact_oracle_decision_sha256=sha(f"oracle-{name}"),
    )
    parents = {
        "truth_manifest_sha256": sha(f"truth-{name}"),
        "mixture_pcm_sha256": sha(f"mixture-{name}"),
        "accompaniment_pcm_sha256": sha(f"accompaniment-{name}"),
        "vocal_pcm_sha256": sha(f"vocal-{name}"),
    }
    risk = ExactSourceReportDocumentV3.build(
        role="risk_evidence",
        group_family_sha256=seed.group_family_sha256,
        source_artifact_sha256=sha("risk-artifact"),
        source_artifact_schema="audio-extract/exact-risk-evidence/v1",
        verifier_commit=git("risk-verifier"),
        **parents,
    )
    unit = dataclasses.replace(seed, exact_evidence_sha256=risk.sha256)
    partition = ExactSourceReportDocumentV3.build(
        role="partition_source",
        group_family_sha256=unit.group_family_sha256,
        source_artifact_sha256=sha("partition-artifact"),
        source_artifact_schema="audio-extract/cell-partition-source/v1",
        verifier_commit=git("partition-verifier"),
        **parents,
    )
    certificate = CellPartitionCertificate(
        group_family_sha256=unit.group_family_sha256,
        spectral_grid_sha256=sha("grid"),
        resolution_ms=500,
        time_cell_count=1,
        band_count=1,
        entries=(
            CellPartitionEntry(0, 0, sha("cell"), RationalMeasure(1)),
        ),
        time_ranges_sha256=sha("time-ranges"),
        frequency_ranges_sha256=sha("frequency-ranges"),
        measure_contract_sha256=sha("measure-contract"),
        exact_source_report_sha256=partition.sha256,
        verifier_commit=git("partition-verifier"),
    )
    lineage = ExactSourceLineageV3(
        risk_evidence_report=risk,
        partition_source_report=partition,
        lineage_verifier_commit=git("lineage-verifier"),
    )
    return unit, certificate, lineage, parents


def test_distinct_report_hashes_validate_when_derived_parents_match():
    unit, certificate, lineage, parents = context()
    assert lineage.risk_evidence_report.sha256 != (
        lineage.partition_source_report.sha256
    )
    lineage.validate(unit, certificate)
    assert lineage.parent_tuple == (
        parents["truth_manifest_sha256"],
        parents["mixture_pcm_sha256"],
        parents["accompaniment_pcm_sha256"],
        parents["vocal_pcm_sha256"],
    )


def test_changed_parent_changes_report_identity_and_is_rejected():
    unit, certificate, lineage, parents = context()
    changed = ExactSourceReportDocumentV3.build(
        role="partition_source",
        group_family_sha256=unit.group_family_sha256,
        source_artifact_sha256=sha("partition-artifact"),
        source_artifact_schema="audio-extract/cell-partition-source/v1",
        truth_manifest_sha256=sha("another-truth"),
        mixture_pcm_sha256=parents["mixture_pcm_sha256"],
        accompaniment_pcm_sha256=parents["accompaniment_pcm_sha256"],
        vocal_pcm_sha256=parents["vocal_pcm_sha256"],
        verifier_commit=certificate.verifier_commit,
    )
    with pytest.raises(
        ExactSourceLineageV3Error,
        match="partition source report bytes differ",
    ):
        dataclasses.replace(
            lineage,
            partition_source_report=changed,
        ).validate(unit, certificate)


def test_report_role_cannot_be_substituted():
    unit, certificate, lineage, parents = context()
    wrong_role = ExactSourceReportDocumentV3.build(
        role="risk_evidence",
        group_family_sha256=unit.group_family_sha256,
        source_artifact_sha256=sha("partition-artifact"),
        source_artifact_schema="audio-extract/cell-partition-source/v1",
        verifier_commit=certificate.verifier_commit,
        **parents,
    )
    wrong_certificate = dataclasses.replace(
        certificate,
        exact_source_report_sha256=wrong_role.sha256,
    )
    with pytest.raises(
        ExactSourceLineageV3Error,
        match="partition source document has the wrong role",
    ):
        dataclasses.replace(
            lineage,
            partition_source_report=wrong_role,
        ).validate(unit, wrong_certificate)


def test_noncanonical_report_bytes_are_refused():
    unit, _certificate, lineage, _parents = context()
    value = lineage.risk_evidence_report.report()
    pretty = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    with pytest.raises(
        ExactSourceLineageV3Error,
        match="not canonical",
    ):
        ExactSourceReportDocumentV3(pretty).validate()
    assert unit.exact_evidence_sha256 != (
        "sha256:" + hashlib.sha256(pretty).hexdigest()
    )


# --------------------------------------------------------------------------
# Bundle-membership resolver: content-derivation ALONE is insufficient.
# These fixtures build a REAL frozen evidence bundle whose held-out unit,
# source family, and partition certificate are genuine members, then prove
# the resolver fails closed on synthesized / copied / substituted / mismatched
# inputs.
# --------------------------------------------------------------------------

SOURCE_COMMIT = git("bundle-source")
FEATURE_CONTRACT = sha("feature-contract")
METRIC_CONTRACT = sha("metric-contract")
ROUTE_POLICY = sha("route-policy")


def _panel() -> CandidatePanel:
    return CandidatePanel(
        (
            CandidateSlot(
                "slot-a", sha("model-a"), sha("adapter-a"),
                sha("construction-a"), sha("query-contract"),
            ),
            CandidateSlot(
                "slot-b", sha("model-b"), sha("adapter-b"),
                sha("construction-b"), sha("query-contract"),
            ),
        )
    )


def _group(index: int, panel: CandidatePanel):
    """One fully-consistent bundle group whose partition certificate and
    held-out unit name REAL exact-source reports deriving the group's roots."""

    name = f"lineage-group-{index}"
    mixture = ArtifactNode(
        "mixture_root",
        ArtifactRef(sha(f"{name}:mix-recipe"), sha(f"{name}:mix-pcm")),
    )
    accompaniment = ArtifactNode(
        "accompaniment_truth",
        ArtifactRef(sha(f"{name}:a-recipe"), sha(f"{name}:a-pcm")),
    )
    vocal = ArtifactNode(
        "vocal_truth",
        ArtifactRef(sha(f"{name}:v-recipe"), sha(f"{name}:v-pcm")),
    )
    candidate_nodes = tuple(
        ArtifactNode(
            "candidate",
            ArtifactRef(
                sha(f"{name}:{slot.slot_id}:recipe"),
                sha(f"{name}:{slot.slot_id}:pcm"),
            ),
            (mixture.identity,),
        )
        for slot in panel.slots
    )
    query_condition = sha(f"{name}:query")
    family = SourceFamilyCertificateV2(
        mixture_root=mixture,
        accompaniment_truth=accompaniment,
        vocal_truth=vocal,
        derived_artifacts=tuple(
            sorted(candidate_nodes, key=lambda item: item.identity)
        ),
        spectral_grid_sha256s=(sha(f"{name}:grid"),),
        query_condition_sha256s=(query_condition,),
        discovery_manifest_sha256=sha(f"{name}:inventory"),
        derivation_policy_sha256=sha("family-derivation-policy"),
        verifier_commit=git("family-verifier"),
    )
    group_family = GroupFamilyIdentity(
        work_id=f"work-{index}",
        recording_session_id=f"session-{index}",
        target_singer_id=f"singer-{index}",
        source_family_sha256=family.sha256,
        query_condition_sha256=query_condition,
    )
    parents = {
        "truth_manifest_sha256": sha(f"{name}:truth"),
        "mixture_pcm_sha256": mixture.identity.artifact_pcm_sha256,
        "accompaniment_pcm_sha256": accompaniment.identity.artifact_pcm_sha256,
        "vocal_pcm_sha256": vocal.identity.artifact_pcm_sha256,
    }
    risk_report = ExactSourceReportDocumentV3.build(
        role="risk_evidence",
        group_family_sha256=group_family.sha256,
        source_artifact_sha256=sha(f"{name}:risk-artifact"),
        source_artifact_schema="audio-extract/exact-risk-evidence/v1",
        verifier_commit=git("risk-verifier"),
        **parents,
    )
    partition_report = ExactSourceReportDocumentV3.build(
        role="partition_source",
        group_family_sha256=group_family.sha256,
        source_artifact_sha256=sha(f"{name}:partition-artifact"),
        source_artifact_schema="audio-extract/cell-partition-source/v1",
        verifier_commit=git("partition-verifier"),
        **parents,
    )
    cell = CellGeometry(
        sample_rate_hz=44_100,
        resolution_ms=500,
        start_frame=0,
        end_frame=1000,
        band_low_hz=0,
        band_high_hz=500,
        spectral_grid_sha256=family.spectral_grid_sha256s[0],
    )
    node_by_recipe = {n.identity.recipe_id: n for n in candidate_nodes}
    bindings = []
    projections = []
    for slot in panel.slots:
        node = node_by_recipe[sha(f"{name}:{slot.slot_id}:recipe")]
        recipe_semantic = sha(f"{name}:{slot.slot_id}:semantic")
        bindings.append(
            CandidateArtifactBinding(
                slot_id=slot.slot_id,
                slot_sha256=slot.sha256,
                recipe_id=node.identity.recipe_id,
                artifact_pcm_sha256=node.identity.artifact_pcm_sha256,
                recipe_semantic_sha256=recipe_semantic,
                recipe_slot_projection_sha256=slot.sha256,
                source_family_sha256=family.sha256,
                verifier_commit=git("projection-verifier"),
            )
        )
        projections.append(
            CandidateSlotProjectionCertificate(
                source_family_sha256=family.sha256,
                source_root=mixture.identity,
                candidate=node.identity,
                slot_sha256=slot.sha256,
                slot_id=slot.slot_id,
                model_bundle_sha256=slot.model_bundle_sha256,
                adapter_bundle_sha256=slot.adapter_bundle_sha256,
                construction_contract_sha256=slot.construction_contract_sha256,
                query_contract_sha256=slot.query_contract_sha256,
                output_role=slot.output_role,
                recipe_semantic_sha256=recipe_semantic,
                recipe_container_sha256=sha(f"{name}:{slot.slot_id}:container"),
                effective_defaults_sha256=sha(f"{name}:{slot.slot_id}:defaults"),
                source_lineage_sha256=sha(f"{name}:{slot.slot_id}:lineage"),
                verification_manifest_sha256=sha(
                    f"{name}:{slot.slot_id}:verification"
                ),
                verifier_commit=git("projection-verifier"),
            )
        )
    row = CounterfactualRiskRowV2(
        group_family=group_family,
        mixture_pcm_sha256=mixture.identity.artifact_pcm_sha256,
        accompaniment_truth_pcm_sha256=(
            accompaniment.identity.artifact_pcm_sha256
        ),
        vocal_truth_pcm_sha256=vocal.identity.artifact_pcm_sha256,
        cell=cell,
        candidate_panel=panel,
        candidate_artifacts=tuple(bindings),
        metric_names=("voice", "hole", "artifact"),
        metric_units=("ratio", "ratio", "ratio"),
        metric_directions=("lower_is_better",) * 3,
        features=np.asarray([float(index), 0.5, -0.25]),
        exact_risks=np.asarray(
            [[0.1, 0.2, 0.3], [0.2, 0.1, 0.4]], dtype=np.float64
        ),
        available=np.ones((2, 3), dtype=bool),
        feature_contract_sha256=FEATURE_CONTRACT,
        metric_contract_sha256=METRIC_CONTRACT,
        route_policy_sha256=ROUTE_POLICY,
    )
    partition_cert = CellPartitionCertificate(
        group_family_sha256=group_family.sha256,
        spectral_grid_sha256=cell.spectral_grid_sha256,
        resolution_ms=cell.resolution_ms,
        time_cell_count=1,
        band_count=1,
        entries=(
            CellPartitionEntry(0, 0, cell.sha256, RationalMeasure(index + 1)),
        ),
        time_ranges_sha256=sha(f"{name}:time-ranges"),
        frequency_ranges_sha256=sha(f"{name}:frequency-ranges"),
        measure_contract_sha256=sha("measure-contract"),
        exact_source_report_sha256=partition_report.sha256,
        verifier_commit=git("partition-verifier"),
    )
    feature_cert = FeatureEvidenceCertificate(
        row_id=row.row_id,
        source_family_sha256=family.sha256,
        feature_contract_sha256=FEATURE_CONTRACT,
        feature_sha256=row.feature_sha256,
        available=(True, True, True),
        required=(True, True, True),
        missing_value_policy_sha256=sha("missing-value-policy"),
        extraction_report_sha256=sha(f"{name}:feature-report"),
        extractor_bundle_sha256=sha("feature-extractor"),
        verifier_commit=git("feature-verifier"),
    )
    unit = HeldOutUnitV1(
        work_id=f"work-{index}",
        recording_session_id=f"session-{index}",
        target_singer_id=f"singer-{index}",
        source_family_sha256=family.sha256,
        query_condition_sha256=query_condition,
        query_quality_stratum_sha256=sha(f"{name}:quality"),
        exact_evidence_sha256=risk_report.sha256,
        exact_panel_sha256=sha(f"{name}:panel"),
        exact_preflight_sha256=sha(f"{name}:preflight"),
        exact_oracle_decision_sha256=sha(f"{name}:oracle"),
    )
    lineage = ExactSourceLineageV3(
        risk_evidence_report=risk_report,
        partition_source_report=partition_report,
        lineage_verifier_commit=git("lineage-verifier"),
    )
    return {
        "row": row,
        "family": family,
        "projections": tuple(projections),
        "partition_cert": partition_cert,
        "feature_cert": feature_cert,
        "unit": unit,
        "lineage": lineage,
        "parents": parents,
    }


def member_bundle():
    panel = _panel()
    groups = [_group(i, panel) for i in range(3)]
    dataset = DatasetManifestV2.build(
        tuple(g["row"] for g in groups), source_commit=SOURCE_COMMIT
    )
    order = dataset.group_family_sha256s
    split = SplitManifestV2(
        dataset_sha256=dataset.sha256,
        train_group_family_sha256s=(order[0],),
        calibration_group_family_sha256s=(order[1],),
        test_group_family_sha256s=(order[2],),
        split_algorithm="frozen-source-family-hash/v1",
        split_seed_sha256=sha("split-seed"),
        selection_manifest_sha256=sha("selection-manifest"),
        source_commit=SOURCE_COMMIT,
    )
    bundle = EvidenceBundleV1(
        dataset=dataset,
        split=split,
        source_registry=SourceFamilyRegistryV2.build(
            tuple(g["family"] for g in groups), source_commit=SOURCE_COMMIT
        ),
        projection_registry=CandidateProjectionRegistry.build(
            tuple(c for g in groups for c in g["projections"]),
            source_commit=SOURCE_COMMIT,
        ),
        partition_registry=CellPartitionRegistry.build(
            tuple(g["partition_cert"] for g in groups),
            source_commit=SOURCE_COMMIT,
        ),
        feature_registry=FeatureEvidenceRegistry.build(
            tuple(g["feature_cert"] for g in groups),
            source_commit=SOURCE_COMMIT,
        ),
        dataset_container_sha256=sha("dataset-container"),
        dataset_schema_sha256=sha("dataset-schema"),
        source_family_schema_sha256=sha("source-family-schema"),
        projection_schema_sha256=sha("projection-schema"),
        partition_schema_sha256=sha("partition-schema"),
        feature_evidence_schema_sha256=sha("feature-schema"),
        split_policy_sha256=sha("split-policy"),
        export_policy_sha256=sha("export-policy"),
        source_commit=SOURCE_COMMIT,
        verifier_commit=git("bundle-verifier"),
    )
    return bundle, groups


def member_context():
    bundle, groups = member_bundle()
    held_out = groups[0]
    return bundle, held_out["unit"], held_out["partition_cert"], held_out["lineage"]


def test_bundle_member_lineage_resolves_content_and_membership():
    bundle, unit, certificate, lineage = member_context()
    bundle.validate()
    # Content-derivation alone passes for the member.
    lineage.validate(unit, certificate)
    # And so does the stronger frozen-bundle membership binding.
    lineage.validate_bundle_membership(
        bundle, bundle.sha256, unit, certificate
    )


def test_synthesized_internally_valid_nonmember_report_is_rejected():
    # `context()` builds a fully internally-valid, canonical, synthesized
    # lineage whose reports are NOT members of the frozen bundle.  Content
    # derivation accepts it; membership must fail closed.
    bundle, _u, _c, _l = member_context()
    synth_unit, synth_cert, synth_lineage, _parents = context("intruder")
    synth_lineage.validate(synth_unit, synth_cert)  # internally valid
    with pytest.raises(
        ExactSourceLineageV3Error,
        match="not a member of the evidence bundle dataset",
    ):
        synth_lineage.validate_bundle_membership(
            bundle, bundle.sha256, synth_unit, synth_cert
        )


def test_copied_reused_evidence_hash_is_rejected():
    # A caller copies a real member's risk-evidence hash onto a unit whose
    # actual report is a different (synthesized) document.
    bundle, unit, certificate, lineage = member_context()
    forged_unit = dataclasses.replace(
        unit, exact_evidence_sha256=lineage.partition_source_report.sha256
    )
    with pytest.raises(
        ExactSourceLineageV3Error,
        match="risk evidence report bytes differ from the frozen unit",
    ):
        lineage.validate_bundle_membership(
            bundle, bundle.sha256, forged_unit, certificate
        )


def test_substituted_partition_certificate_is_rejected():
    # A certificate that names the same exact partition-source report but is
    # NOT the frozen registry member (different measure/time-range identity).
    bundle, unit, certificate, lineage = member_context()
    substitute = dataclasses.replace(
        certificate,
        time_ranges_sha256=sha("substitute-time-ranges"),
    )
    assert substitute.sha256 != certificate.sha256
    with pytest.raises(
        ExactSourceLineageV3Error,
        match="not a member of the bundle partition registry",
    ):
        lineage.validate_bundle_membership(
            bundle, bundle.sha256, unit, substitute
        )


def test_mismatched_bundle_identity_is_rejected():
    bundle, unit, certificate, lineage = member_context()
    with pytest.raises(
        ExactSourceLineageV3Error,
        match="does not match the declared exact_evidence_bundle_sha256",
    ):
        lineage.validate_bundle_membership(
            bundle, sha("a-different-bundle"), unit, certificate
        )
