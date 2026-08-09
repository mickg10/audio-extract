import hashlib

import numpy as np
import pytest

from audio_extract.counterfactual_risk_candidate_projection_v1 import (
    CandidateProjectionError,
    CandidateProjectionRegistry,
    CandidateSlotProjectionCertificate,
    validate_dataset_candidate_projections,
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


def source_family(name: str, candidate_panel: CandidatePanel):
    mixture = ArtifactNode(
        "mixture_root",
        ArtifactRef(sha(f"{name}:mixture-recipe"), sha(f"{name}:mixture-pcm")),
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
        sorted(
            (
                ArtifactNode(
                    "candidate",
                    ArtifactRef(
                        sha(f"{name}:{slot.slot_id}:recipe"),
                        sha(f"{name}:{slot.slot_id}:pcm"),
                    ),
                    (mixture.identity,),
                )
                for slot in candidate_panel.slots
            ),
            key=lambda value: value.identity,
        )
    )
    return SourceFamilyCertificateV2(
        mixture_root=mixture,
        accompaniment_truth=accompaniment,
        vocal_truth=vocal,
        derived_artifacts=candidates,
        spectral_grid_sha256s=(sha(f"{name}:grid"),),
        query_condition_sha256s=(sha(f"{name}:query"),),
        discovery_manifest_sha256=sha(f"{name}:inventory"),
        derivation_policy_sha256=sha("derivation-policy"),
        verifier_commit=git("source-family-verifier"),
    )


def row_for(
    name: str,
    candidate_panel: CandidatePanel,
    family: SourceFamilyCertificateV2,
):
    nodes = {node.identity.recipe_id: node for node in family.derived_artifacts}
    bindings = []
    for slot in candidate_panel.slots:
        node = nodes[sha(f"{name}:{slot.slot_id}:recipe")]
        bindings.append(
            CandidateArtifactBinding(
                slot_id=slot.slot_id,
                slot_sha256=slot.sha256,
                recipe_id=node.identity.recipe_id,
                artifact_pcm_sha256=node.identity.artifact_pcm_sha256,
                recipe_semantic_sha256=sha(
                    f"{name}:{slot.slot_id}:recipe-semantic"
                ),
                recipe_slot_projection_sha256=slot.sha256,
                source_family_sha256=family.sha256,
                verifier_commit=git("projection-verifier"),
            )
        )
    return CounterfactualRiskRowV2(
        group_family=GroupFamilyIdentity(
            work_id=name,
            recording_session_id=f"session-{name}",
            target_singer_id=f"singer-{name}",
            source_family_sha256=family.sha256,
            query_condition_sha256=family.query_condition_sha256s[0],
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
        candidate_artifacts=tuple(bindings),
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


def projection(
    slot: CandidateSlot,
    binding: CandidateArtifactBinding,
    family: SourceFamilyCertificateV2,
):
    return CandidateSlotProjectionCertificate(
        source_family_sha256=family.sha256,
        source_root=family.mixture_root.identity,
        candidate=ArtifactRef(binding.recipe_id, binding.artifact_pcm_sha256),
        slot_sha256=slot.sha256,
        slot_id=slot.slot_id,
        model_bundle_sha256=slot.model_bundle_sha256,
        adapter_bundle_sha256=slot.adapter_bundle_sha256,
        construction_contract_sha256=slot.construction_contract_sha256,
        query_contract_sha256=slot.query_contract_sha256,
        output_role=slot.output_role,
        recipe_semantic_sha256=binding.recipe_semantic_sha256,
        recipe_container_sha256=sha(f"container:{binding.recipe_id}"),
        effective_defaults_sha256=sha(f"defaults:{binding.recipe_id}"),
        source_lineage_sha256=sha(f"lineage:{binding.recipe_id}"),
        verification_manifest_sha256=sha(f"verify:{binding.recipe_id}"),
        verifier_commit=binding.verifier_commit,
    )


def setup():
    candidate_panel = panel()
    family = source_family("work", candidate_panel)
    row = row_for("work", candidate_panel, family)
    dataset = DatasetManifestV2.build((row,), source_commit=git("dataset"))
    source_registry = SourceFamilyRegistryV2.build(
        (family,), source_commit=git("source-registry")
    )
    certificates = tuple(
        projection(slot, binding, family)
        for slot, binding in zip(
            candidate_panel.slots, row.candidate_artifacts
        )
    )
    projection_registry = CandidateProjectionRegistry.build(
        certificates, source_commit=git("projection-registry")
    )
    return dataset, source_registry, projection_registry, family


def test_dataset_artifacts_require_independent_content_bearing_projections():
    dataset, source_registry, projections, _ = setup()
    validate_dataset_candidate_projections(
        dataset, source_registry, projections
    )
    assert projections.sha256.startswith("sha256:")


def test_missing_projection_certificate_is_refused():
    dataset, source_registry, projections, _ = setup()
    missing = CandidateProjectionRegistry.build(
        projections.certificates[:1],
        source_commit=git("projection-registry"),
    )
    with pytest.raises(
        CandidateProjectionError,
        match="lacks an independent slot projection certificate",
    ):
        validate_dataset_candidate_projections(
            dataset, source_registry, missing
        )


def test_projection_must_match_every_stable_slot_component():
    dataset, source_registry, projections, _ = setup()
    first = projections.certificates[0]
    wrong = CandidateSlotProjectionCertificate(
        **{**first.__dict__, "model_bundle_sha256": sha("other-model")}
    )
    registry = CandidateProjectionRegistry.build(
        tuple(sorted((wrong, projections.certificates[1]), key=lambda value: value.certificate_key)),
        source_commit=git("projection-registry"),
    )
    with pytest.raises(
        CandidateProjectionError,
        match="differs from stable slot fields",
    ):
        validate_dataset_candidate_projections(
            dataset, source_registry, registry
        )


def test_projection_source_root_must_equal_strict_family_mixture_root():
    dataset, source_registry, projections, _ = setup()
    first = projections.certificates[0]
    wrong = CandidateSlotProjectionCertificate(
        **{
            **first.__dict__,
            "source_root": ArtifactRef(
                sha("other-root-recipe"), sha("other-root-pcm")
            ),
        }
    )
    registry = CandidateProjectionRegistry.build(
        tuple(sorted((wrong, projections.certificates[1]), key=lambda value: value.certificate_key)),
        source_commit=git("projection-registry"),
    )
    with pytest.raises(
        CandidateProjectionError,
        match="source root differs",
    ):
        validate_dataset_candidate_projections(
            dataset, source_registry, registry
        )


def test_row_recipe_semantics_and_verifier_commit_must_match_certificate():
    dataset, source_registry, projections, _ = setup()
    row = dataset.rows[0]
    first = row.candidate_artifacts[0]
    bad_semantic = CandidateArtifactBinding(
        **{**first.__dict__, "recipe_semantic_sha256": sha("other-semantic")}
    )
    changed_row = CounterfactualRiskRowV2(
        **{
            **row.__dict__,
            "candidate_artifacts": (bad_semantic, row.candidate_artifacts[1]),
        }
    )
    changed_dataset = DatasetManifestV2.build(
        (changed_row,), source_commit=git("dataset")
    )
    with pytest.raises(
        CandidateProjectionError,
        match="recipe semantic identity differs",
    ):
        validate_dataset_candidate_projections(
            changed_dataset, source_registry, projections
        )

    bad_commit = CandidateArtifactBinding(
        **{**first.__dict__, "verifier_commit": git("other-verifier")}
    )
    changed_row = CounterfactualRiskRowV2(
        **{
            **row.__dict__,
            "candidate_artifacts": (bad_commit, row.candidate_artifacts[1]),
        }
    )
    changed_dataset = DatasetManifestV2.build(
        (changed_row,), source_commit=git("dataset")
    )
    with pytest.raises(
        CandidateProjectionError,
        match="verifier commit differs",
    ):
        validate_dataset_candidate_projections(
            changed_dataset, source_registry, projections
        )


def test_registry_refuses_unused_or_duplicate_candidate_projection():
    dataset, source_registry, projections, family = setup()
    row = dataset.rows[0]
    slot = row.candidate_panel.slots[0]
    binding = row.candidate_artifacts[0]
    unused_binding = CandidateArtifactBinding(
        **{
            **binding.__dict__,
            "recipe_id": sha("unused-recipe"),
            "artifact_pcm_sha256": sha("unused-pcm"),
            "recipe_semantic_sha256": sha("unused-semantic"),
        }
    )
    unused = projection(slot, unused_binding, family)
    expanded = CandidateProjectionRegistry.build(
        tuple(
            sorted(
                (*projections.certificates, unused),
                key=lambda value: value.certificate_key,
            )
        ),
        source_commit=git("projection-registry"),
    )
    with pytest.raises(CandidateProjectionError, match="unused by the dataset"):
        validate_dataset_candidate_projections(
            dataset, source_registry, expanded
        )

    first = projections.certificates[0]
    duplicate = CandidateSlotProjectionCertificate(
        **{
            **first.__dict__,
            "slot_sha256": projections.certificates[1].slot_sha256,
            "slot_id": projections.certificates[1].slot_id,
            "model_bundle_sha256": projections.certificates[1].model_bundle_sha256,
            "adapter_bundle_sha256": projections.certificates[1].adapter_bundle_sha256,
            "construction_contract_sha256": projections.certificates[1].construction_contract_sha256,
            "query_contract_sha256": projections.certificates[1].query_contract_sha256,
        }
    )
    with pytest.raises(
        CandidateProjectionError,
        match="multiple slot projections",
    ):
        CandidateProjectionRegistry.build(
            (first, duplicate), source_commit=git("projection-registry")
        )
