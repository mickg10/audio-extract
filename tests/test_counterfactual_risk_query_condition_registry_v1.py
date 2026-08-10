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
)
from audio_extract.counterfactual_risk_query_condition_v1 import (
    QueryConditionCertificate,
    QueryConditionError,
    QueryConditionRegistry,
    validate_dataset_query_conditions,
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


def source_family(name: str = "work"):
    candidate_panel = panel()
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
    candidates = tuple(
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
    derived = tuple(
        sorted((*candidates, query_source), key=lambda item: item.identity)
    )
    query_scope = sha(f"{name}:query-scope")
    family = SourceFamilyCertificateV2(
        mixture_root=mixture,
        accompaniment_truth=accompaniment,
        vocal_truth=vocal,
        derived_artifacts=derived,
        spectral_grid_sha256s=(sha(f"{name}:grid"),),
        query_condition_sha256s=(query_scope,),
        discovery_manifest_sha256=sha(f"{name}:inventory"),
        derivation_policy_sha256=sha("derivation-policy"),
        verifier_commit=git("source-verifier"),
    )
    return candidate_panel, family, query_source, query_scope


def query_certificate(
    family: SourceFamilyCertificateV2,
    query_source: ArtifactNode,
    query_scope: str,
    *,
    target_singer_id: str = "singer-work",
    source_family_sha256: str | None = None,
    feature_contract_sha256: str | None = None,
    quality_class: str = "mixture_derived_verified",
):
    return QueryConditionCertificate(
        source_family_sha256=source_family_sha256 or family.sha256,
        query_scope_sha256=query_scope,
        target_singer_id=target_singer_id,
        target_identity_sha256=sha(f"target:{target_singer_id}"),
        query_source=query_source.identity,
        query_encoder_bundle_sha256=sha("query-encoder"),
        query_feature_contract_sha256=(
            feature_contract_sha256 or sha("features")
        ),
        query_embedding_sha256=sha(f"embedding:{target_singer_id}"),
        query_segment_manifest_sha256=sha("query-segments"),
        query_projection_sha256=sha("query-projection"),
        quality_class=quality_class,
        quality_policy_sha256=sha("quality-policy"),
        quality_report_sha256=sha(f"quality-report:{target_singer_id}"),
        verifier_commit=git("query-verifier"),
    )


def row_for(
    candidate_panel: CandidatePanel,
    family: SourceFamilyCertificateV2,
    query: QueryConditionCertificate,
    *,
    target_singer_id: str = "singer-work",
):
    candidate_nodes = {
        node.identity.recipe_id: node
        for node in family.derived_artifacts
        if node.role == "candidate"
    }
    artifacts = []
    for slot in candidate_panel.slots:
        node = candidate_nodes[sha(f"work:{slot.slot_id}:recipe")]
        artifacts.append(
            CandidateArtifactBinding(
                slot_id=slot.slot_id,
                slot_sha256=slot.sha256,
                recipe_id=node.identity.recipe_id,
                artifact_pcm_sha256=node.identity.artifact_pcm_sha256,
                recipe_semantic_sha256=sha(f"semantic:{slot.slot_id}"),
                recipe_slot_projection_sha256=slot.sha256,
                source_family_sha256=family.sha256,
                verifier_commit=git("artifact-verifier"),
            )
        )
    return CounterfactualRiskRowV2(
        group_family=GroupFamilyIdentity(
            work_id="work",
            recording_session_id="session-work",
            target_singer_id=target_singer_id,
            source_family_sha256=family.sha256,
            query_condition_sha256=query.sha256,
        ),
        mixture_pcm_sha256=family.mixture_root.identity.artifact_pcm_sha256,
        accompaniment_truth_pcm_sha256=(
            family.accompaniment_truth.identity.artifact_pcm_sha256
        ),
        vocal_truth_pcm_sha256=family.vocal_truth.identity.artifact_pcm_sha256,
        cell=CellGeometry(
            sample_rate_hz=44_100,
            resolution_ms=500,
            start_frame=0,
            end_frame=1000,
            band_low_hz=0,
            band_high_hz=500,
            spectral_grid_sha256=family.spectral_grid_sha256s[0],
        ),
        candidate_panel=candidate_panel,
        candidate_artifacts=tuple(artifacts),
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
        route_policy_sha256=sha("route-policy"),
    )


def setup():
    candidate_panel, family, query_source, query_scope = source_family()
    query = query_certificate(family, query_source, query_scope)
    row = row_for(candidate_panel, family, query)
    dataset = DatasetManifestV2.build((row,), source_commit=git("dataset"))
    source_registry = SourceFamilyRegistryV2.build(
        (family,), source_commit=git("source-registry")
    )
    query_registry = QueryConditionRegistry.build(
        (query,), source_commit=git("query-registry")
    )
    return dataset, source_registry, query_registry, family, query_source, query_scope


def test_content_bearing_query_certificate_binds_dataset_without_hash_cycle():
    dataset, source_registry, query_registry, family, _, _ = setup()
    validate_dataset_query_conditions(
        dataset, source_registry, query_registry
    )
    query = query_registry.certificates[0]
    assert query.sha256 == dataset.rows[0].group_family.query_condition_sha256
    assert query.source_family_sha256 == family.sha256
    assert query.sha256 not in family.query_condition_sha256s


def test_unregistered_concrete_query_certificate_is_refused():
    dataset, source_registry, query_registry, _, _, _ = setup()
    row = dataset.rows[0]
    changed_group = GroupFamilyIdentity(
        **{
            **row.group_family.__dict__,
            "query_condition_sha256": sha("unregistered-query"),
        }
    )
    changed = CounterfactualRiskRowV2(
        **{**row.__dict__, "group_family": changed_group}
    )
    changed_dataset = DatasetManifestV2.build(
        (changed,), source_commit=git("dataset")
    )
    with pytest.raises(QueryConditionError, match="unregistered query-condition"):
        validate_dataset_query_conditions(
            changed_dataset, source_registry, query_registry
        )


def test_query_target_source_and_feature_contract_must_match_row():
    _, source_registry, _, family, query_source, query_scope = setup()
    candidate_panel = panel()

    wrong_target = query_certificate(
        family, query_source, query_scope, target_singer_id="other-singer"
    )
    wrong_target_dataset = DatasetManifestV2.build(
        (row_for(candidate_panel, family, wrong_target),),
        source_commit=git("dataset"),
    )
    with pytest.raises(QueryConditionError, match="different target singer"):
        validate_dataset_query_conditions(
            wrong_target_dataset,
            source_registry,
            QueryConditionRegistry.build(
                (wrong_target,), source_commit=git("query-registry")
            ),
        )

    wrong_source = query_certificate(
        family,
        query_source,
        query_scope,
        source_family_sha256=sha("other-source-family"),
    )
    wrong_source_dataset = DatasetManifestV2.build(
        (row_for(candidate_panel, family, wrong_source),),
        source_commit=git("dataset"),
    )
    with pytest.raises(QueryConditionError, match="different source family"):
        validate_dataset_query_conditions(
            wrong_source_dataset,
            source_registry,
            QueryConditionRegistry.build(
                (wrong_source,), source_commit=git("query-registry")
            ),
        )

    wrong_features = query_certificate(
        family,
        query_source,
        query_scope,
        feature_contract_sha256=sha("other-features"),
    )
    wrong_features_dataset = DatasetManifestV2.build(
        (row_for(candidate_panel, family, wrong_features),),
        source_commit=git("dataset"),
    )
    with pytest.raises(QueryConditionError, match="different inference feature"):
        validate_dataset_query_conditions(
            wrong_features_dataset,
            source_registry,
            QueryConditionRegistry.build(
                (wrong_features,), source_commit=git("query-registry")
            ),
        )


def test_query_source_must_be_declared_query_node_and_scope_must_be_registered():
    _, source_registry, _, family, _, query_scope = setup()
    candidate_panel = panel()
    candidate_node = next(
        node for node in family.derived_artifacts if node.role == "candidate"
    )
    wrong_role = query_certificate(
        family, candidate_node, query_scope
    )
    wrong_role_dataset = DatasetManifestV2.build(
        (row_for(candidate_panel, family, wrong_role),),
        source_commit=git("dataset"),
    )
    with pytest.raises(QueryConditionError, match="strict family query inventory"):
        validate_dataset_query_conditions(
            wrong_role_dataset,
            source_registry,
            QueryConditionRegistry.build(
                (wrong_role,), source_commit=git("query-registry")
            ),
        )

    query_source = next(
        node for node in family.derived_artifacts if node.role == "query_source"
    )
    wrong_scope = query_certificate(
        family, query_source, sha("other-scope")
    )
    wrong_scope_dataset = DatasetManifestV2.build(
        (row_for(candidate_panel, family, wrong_scope),),
        source_commit=git("dataset"),
    )
    with pytest.raises(QueryConditionError, match="scope is absent"):
        validate_dataset_query_conditions(
            wrong_scope_dataset,
            source_registry,
            QueryConditionRegistry.build(
                (wrong_scope,), source_commit=git("query-registry")
            ),
        )


def test_low_confidence_query_and_unused_registry_entry_are_refused():
    dataset, source_registry, query_registry, family, query_source, query_scope = setup()
    low = query_certificate(
        family,
        query_source,
        query_scope,
        target_singer_id="low-singer",
        quality_class="mixture_derived_low_confidence",
    )
    with pytest.raises(QueryConditionError, match="only a verified"):
        QueryConditionRegistry.build(
            (low,), source_commit=git("query-registry")
        )

    unused = query_certificate(
        family,
        query_source,
        query_scope,
        target_singer_id="unused-singer",
    )
    expanded = QueryConditionRegistry.build(
        (query_registry.certificates[0], unused),
        source_commit=git("query-registry"),
    )
    with pytest.raises(QueryConditionError, match="unused by the dataset"):
        validate_dataset_query_conditions(
            dataset, source_registry, expanded
        )


def test_query_identity_binds_embedding_scope_quality_and_target():
    _, _, _, family, query_source, query_scope = setup()
    original = query_certificate(family, query_source, query_scope)
    changed_scope = QueryConditionCertificate(
        **{**original.__dict__, "query_scope_sha256": sha("changed-scope")}
    )
    changed_embedding = QueryConditionCertificate(
        **{
            **original.__dict__,
            "query_embedding_sha256": sha("changed-embedding"),
        }
    )
    changed_target = QueryConditionCertificate(
        **{
            **original.__dict__,
            "target_identity_sha256": sha("changed-target"),
        }
    )
    assert len(
        {
            original.sha256,
            changed_scope.sha256,
            changed_embedding.sha256,
            changed_target.sha256,
        }
    ) == 4
