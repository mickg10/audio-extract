import hashlib

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
from audio_extract.counterfactual_risk_evidence_bundle_v1 import (
    EvidenceBundleError,
    EvidenceBundleV1,
)
from audio_extract.counterfactual_risk_feature_evidence_v1 import (
    FeatureEvidenceCertificate,
    FeatureEvidenceRegistry,
)
from audio_extract.counterfactual_risk_query_condition_v1 import (
    QueryConditionCertificate,
    QueryConditionRegistry,
)
from audio_extract.counterfactual_risk_source_family_v2 import (
    ArtifactNode,
    ArtifactRef,
    SourceFamilyCertificateV2,
    SourceFamilyRegistryV2,
)


def sha(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode()).hexdigest()


def git(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()


SOURCE_COMMIT = git("bundle-source")
VERIFIER_COMMIT = git("bundle-verifier")
FEATURE_CONTRACT = sha("feature-contract")
METRIC_CONTRACT = sha("metric-contract")
ROUTE_POLICY = sha("route-policy")


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


def build_group(index: int, candidate_panel: CandidatePanel):
    name = f"group-{index}"
    target = f"singer-{index}"
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
        for slot in candidate_panel.slots
    )
    query_source = ArtifactNode(
        "query_source",
        ArtifactRef(sha(f"{name}:query-recipe"), sha(f"{name}:query-pcm")),
        (mixture.identity,),
    )
    query_scope = sha(f"{name}:query-scope")
    family = SourceFamilyCertificateV2(
        mixture_root=mixture,
        accompaniment_truth=accompaniment,
        vocal_truth=vocal,
        derived_artifacts=tuple(
            sorted((*candidate_nodes, query_source), key=lambda item: item.identity)
        ),
        spectral_grid_sha256s=(sha(f"{name}:grid"),),
        query_condition_sha256s=(query_scope,),
        discovery_manifest_sha256=sha(f"{name}:inventory"),
        derivation_policy_sha256=sha("family-derivation-policy"),
        verifier_commit=git("family-verifier"),
    )
    query = QueryConditionCertificate(
        source_family_sha256=family.sha256,
        query_scope_sha256=query_scope,
        target_singer_id=target,
        target_identity_sha256=sha(f"target:{target}"),
        query_source=query_source.identity,
        query_encoder_bundle_sha256=sha("query-encoder"),
        query_feature_contract_sha256=FEATURE_CONTRACT,
        query_embedding_sha256=sha(f"{name}:query-embedding"),
        query_segment_manifest_sha256=sha(f"{name}:query-segments"),
        query_projection_sha256=sha(f"{name}:query-projection"),
        quality_class="mixture_derived_verified",
        quality_policy_sha256=sha("query-quality-policy"),
        quality_report_sha256=sha(f"{name}:query-quality-report"),
        verifier_commit=git("query-verifier"),
    )
    group_family = GroupFamilyIdentity(
        work_id=f"work-{index}",
        recording_session_id=f"session-{index}",
        target_singer_id=target,
        source_family_sha256=family.sha256,
        query_condition_sha256=query.sha256,
    )
    node_by_recipe = {
        node.identity.recipe_id: node for node in candidate_nodes
    }
    bindings = []
    projections = []
    for slot in candidate_panel.slots:
        node = node_by_recipe[sha(f"{name}:{slot.slot_id}:recipe")]
        recipe_semantic = sha(f"{name}:{slot.slot_id}:semantic")
        binding = CandidateArtifactBinding(
            slot_id=slot.slot_id,
            slot_sha256=slot.sha256,
            recipe_id=node.identity.recipe_id,
            artifact_pcm_sha256=node.identity.artifact_pcm_sha256,
            recipe_semantic_sha256=recipe_semantic,
            recipe_slot_projection_sha256=slot.sha256,
            source_family_sha256=family.sha256,
            verifier_commit=git("projection-verifier"),
        )
        bindings.append(binding)
        projections.append(
            CandidateSlotProjectionCertificate(
                source_family_sha256=family.sha256,
                source_root=mixture.identity,
                candidate=node.identity,
                slot_sha256=slot.sha256,
                slot_id=slot.slot_id,
                model_bundle_sha256=slot.model_bundle_sha256,
                adapter_bundle_sha256=slot.adapter_bundle_sha256,
                construction_contract_sha256=(
                    slot.construction_contract_sha256
                ),
                query_contract_sha256=slot.query_contract_sha256,
                output_role=slot.output_role,
                recipe_semantic_sha256=recipe_semantic,
                recipe_container_sha256=sha(
                    f"{name}:{slot.slot_id}:container"
                ),
                effective_defaults_sha256=sha(
                    f"{name}:{slot.slot_id}:defaults"
                ),
                source_lineage_sha256=sha(
                    f"{name}:{slot.slot_id}:lineage"
                ),
                verification_manifest_sha256=sha(
                    f"{name}:{slot.slot_id}:verification"
                ),
                verifier_commit=git("projection-verifier"),
            )
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
    row = CounterfactualRiskRowV2(
        group_family=group_family,
        mixture_pcm_sha256=mixture.identity.artifact_pcm_sha256,
        accompaniment_truth_pcm_sha256=(
            accompaniment.identity.artifact_pcm_sha256
        ),
        vocal_truth_pcm_sha256=vocal.identity.artifact_pcm_sha256,
        cell=cell,
        candidate_panel=candidate_panel,
        candidate_artifacts=tuple(bindings),
        metric_names=("voice", "hole", "artifact"),
        metric_units=("ratio", "ratio", "ratio"),
        metric_directions=(
            "lower_is_better",
            "lower_is_better",
            "lower_is_better",
        ),
        features=np.asarray([float(index), 0.5, -0.25]),
        exact_risks=np.asarray(
            [[0.1, 0.2, 0.3], [0.2, 0.1, 0.4]], dtype=np.float64
        ),
        available=np.ones((2, 3), dtype=bool),
        feature_contract_sha256=FEATURE_CONTRACT,
        metric_contract_sha256=METRIC_CONTRACT,
        route_policy_sha256=ROUTE_POLICY,
    )
    partition = CellPartitionCertificate(
        group_family_sha256=group_family.sha256,
        spectral_grid_sha256=cell.spectral_grid_sha256,
        resolution_ms=cell.resolution_ms,
        time_cell_count=1,
        band_count=1,
        entries=(
            CellPartitionEntry(
                time_index=0,
                band_index=0,
                cell_sha256=cell.sha256,
                measure=RationalMeasure(index + 1),
            ),
        ),
        time_ranges_sha256=sha(f"{name}:time-ranges"),
        frequency_ranges_sha256=sha(f"{name}:frequency-ranges"),
        measure_contract_sha256=sha("measure-contract"),
        exact_source_report_sha256=sha(f"{name}:source-report"),
        verifier_commit=git("partition-verifier"),
    )
    feature = FeatureEvidenceCertificate(
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
    return row, family, query, tuple(projections), partition, feature


def valid_bundle() -> EvidenceBundleV1:
    candidate_panel = panel()
    components = [build_group(index, candidate_panel) for index in range(3)]
    rows = tuple(item[0] for item in components)
    dataset = DatasetManifestV2.build(rows, source_commit=SOURCE_COMMIT)
    groups = dataset.group_family_sha256s
    split = SplitManifestV2(
        dataset_sha256=dataset.sha256,
        train_group_family_sha256s=(groups[0],),
        calibration_group_family_sha256s=(groups[1],),
        test_group_family_sha256s=(groups[2],),
        split_algorithm="frozen-source-family-hash/v1",
        split_seed_sha256=sha("split-seed"),
        selection_manifest_sha256=sha("selection-manifest"),
        source_commit=SOURCE_COMMIT,
    )
    return EvidenceBundleV1(
        dataset=dataset,
        split=split,
        source_registry=SourceFamilyRegistryV2.build(
            tuple(item[1] for item in components),
            source_commit=SOURCE_COMMIT,
        ),
        query_registry=QueryConditionRegistry.build(
            tuple(item[2] for item in components),
            source_commit=SOURCE_COMMIT,
        ),
        projection_registry=CandidateProjectionRegistry.build(
            tuple(
                certificate
                for item in components
                for certificate in item[3]
            ),
            source_commit=SOURCE_COMMIT,
        ),
        partition_registry=CellPartitionRegistry.build(
            tuple(item[4] for item in components),
            source_commit=SOURCE_COMMIT,
        ),
        feature_registry=FeatureEvidenceRegistry.build(
            tuple(item[5] for item in components),
            source_commit=SOURCE_COMMIT,
        ),
        dataset_container_sha256=sha("dataset-container"),
        dataset_schema_sha256=sha("dataset-schema"),
        source_family_schema_sha256=sha("source-family-schema"),
        query_schema_sha256=sha("query-schema"),
        projection_schema_sha256=sha("projection-schema"),
        partition_schema_sha256=sha("partition-schema"),
        feature_evidence_schema_sha256=sha("feature-schema"),
        split_policy_sha256=sha("split-policy"),
        export_policy_sha256=sha("export-policy"),
        source_commit=SOURCE_COMMIT,
        verifier_commit=VERIFIER_COMMIT,
    )


def test_valid_bundle_forces_every_semantic_boundary_and_has_one_identity():
    bundle = valid_bundle()
    bundle.validate()
    identity = bundle.identity_dict()
    assert bundle.sha256.startswith("sha256:")
    assert identity["row_count"] == 3
    assert identity["group_family_count"] == 3
    assert identity["candidate_slot_count"] == 2
    assert identity["metric_count"] == 3
    assert identity["feature_count"] == 3


def test_component_source_commit_mismatch_is_refused():
    bundle = valid_bundle()
    changed_registry = FeatureEvidenceRegistry(
        certificates=bundle.feature_registry.certificates,
        source_commit=git("other-source"),
    )
    changed = EvidenceBundleV1(
        **{**bundle.__dict__, "feature_registry": changed_registry}
    )
    with pytest.raises(
        EvidenceBundleError,
        match="different source commits.*feature_registry",
    ):
        changed.validate()


def test_missing_projection_or_partition_cannot_be_omitted_from_bundle_gate():
    bundle = valid_bundle()
    missing_projection = CandidateProjectionRegistry.build(
        bundle.projection_registry.certificates[:-1],
        source_commit=SOURCE_COMMIT,
    )
    changed = EvidenceBundleV1(
        **{**bundle.__dict__, "projection_registry": missing_projection}
    )
    with pytest.raises(Exception, match="lacks an independent slot projection"):
        changed.validate()

    missing_partition = CellPartitionRegistry.build(
        bundle.partition_registry.certificates[:-1],
        source_commit=SOURCE_COMMIT,
    )
    changed = EvidenceBundleV1(
        **{**bundle.__dict__, "partition_registry": missing_partition}
    )
    with pytest.raises(Exception, match="no matching certified cell partition"):
        changed.validate()


def test_query_and_feature_certificates_are_mandatory_at_bundle_boundary():
    bundle = valid_bundle()
    missing_query = QueryConditionRegistry.build(
        bundle.query_registry.certificates[:-1],
        source_commit=SOURCE_COMMIT,
    )
    changed = EvidenceBundleV1(
        **{**bundle.__dict__, "query_registry": missing_query}
    )
    with pytest.raises(Exception, match="unregistered query-condition"):
        changed.validate()

    missing_feature = FeatureEvidenceRegistry.build(
        bundle.feature_registry.certificates[:-1],
        source_commit=SOURCE_COMMIT,
    )
    changed = EvidenceBundleV1(
        **{**bundle.__dict__, "feature_registry": missing_feature}
    )
    with pytest.raises(Exception, match="lacks a feature-evidence"):
        changed.validate()


def test_split_must_name_exact_dataset_and_every_partition():
    bundle = valid_bundle()
    bad_split = SplitManifestV2(
        **{**bundle.split.__dict__, "dataset_sha256": sha("other-dataset")}
    )
    changed = EvidenceBundleV1(**{**bundle.__dict__, "split": bad_split})
    with pytest.raises(Exception, match="different dataset"):
        changed.validate()


def test_bundle_identity_binds_container_schemas_and_policies():
    bundle = valid_bundle()
    changed_container = EvidenceBundleV1(
        **{
            **bundle.__dict__,
            "dataset_container_sha256": sha("other-container"),
        }
    )
    changed_schema = EvidenceBundleV1(
        **{
            **bundle.__dict__,
            "dataset_schema_sha256": sha("other-schema"),
        }
    )
    changed_policy = EvidenceBundleV1(
        **{**bundle.__dict__, "export_policy_sha256": sha("other-policy")}
    )
    assert len(
        {
            bundle.sha256,
            changed_container.sha256,
            changed_schema.sha256,
            changed_policy.sha256,
        }
    ) == 4


def test_bundle_status_purpose_and_digest_syntax_are_fail_closed():
    bundle = valid_bundle()
    with pytest.raises(EvidenceBundleError, match="status must be passed"):
        EvidenceBundleV1(**{**bundle.__dict__, "status": "failed"}).validate()
    with pytest.raises(EvidenceBundleError, match="unknown evidence bundle purpose"):
        EvidenceBundleV1(**{**bundle.__dict__, "purpose": "other"}).validate()
    with pytest.raises(EvidenceBundleError, match="canonical sha256"):
        EvidenceBundleV1(
            **{**bundle.__dict__, "dataset_container_sha256": "bad"}
        ).validate()
