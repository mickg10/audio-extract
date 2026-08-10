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
from audio_extract.counterfactual_risk_dataset_source_binding_v2 import (
    validate_dataset_v2_family_registry,
)
from audio_extract.counterfactual_risk_source_family_v2 import (
    ArtifactNode,
    ArtifactRef,
    SourceFamilyCertificateV2,
    SourceFamilyRegistryV2,
    SourceFamilyV2Error,
)


def sha(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode()).hexdigest()


def git(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()


def stable_panel() -> CandidatePanel:
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


def source_family(name: str, panel: CandidatePanel):
    mixture = ArtifactNode(
        "mixture_root", ArtifactRef(sha(f"{name}:mix-recipe"), sha(f"{name}:mix"))
    )
    accompaniment = ArtifactNode(
        "accompaniment_truth",
        ArtifactRef(sha(f"{name}:a-recipe"), sha(f"{name}:a")),
    )
    vocal = ArtifactNode(
        "vocal_truth", ArtifactRef(sha(f"{name}:v-recipe"), sha(f"{name}:v"))
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
                for slot in panel.slots
            ),
            key=lambda node: node.identity,
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
        verifier_commit=git("family-verifier"),
    )


def exact_row(
    name: str,
    index: int,
    panel: CandidatePanel,
    family: SourceFamilyCertificateV2,
):
    nodes = {node.identity.recipe_id: node for node in family.derived_artifacts}
    artifacts = []
    for slot in panel.slots:
        recipe = sha(f"{name}:{slot.slot_id}:recipe")
        node = nodes[recipe]
        artifacts.append(
            CandidateArtifactBinding(
                slot_id=slot.slot_id,
                slot_sha256=slot.sha256,
                recipe_id=node.identity.recipe_id,
                artifact_pcm_sha256=node.identity.artifact_pcm_sha256,
                recipe_semantic_sha256=sha(f"{name}:{slot.slot_id}:semantic"),
                recipe_slot_projection_sha256=slot.sha256,
                source_family_sha256=family.sha256,
                verifier_commit=git("artifact-verifier"),
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
            start_frame=index * 100,
            end_frame=(index + 1) * 100,
            band_low_hz=0,
            band_high_hz=500,
            spectral_grid_sha256=family.spectral_grid_sha256s[0],
        ),
        candidate_panel=panel,
        candidate_artifacts=tuple(artifacts),
        metric_names=("voice", "hole", "artifact"),
        metric_units=("ratio", "ratio", "ratio"),
        metric_directions=(
            "lower_is_better",
            "lower_is_better",
            "lower_is_better",
        ),
        features=np.asarray([float(index), 0.5]),
        exact_risks=np.asarray([[0.1, 0.2, 0.3], [0.2, 0.1, 0.4]]),
        available=np.ones((2, 3), dtype=bool),
        feature_contract_sha256=sha("features"),
        metric_contract_sha256=sha("metrics"),
        route_policy_sha256=sha("route-policy"),
    )


def setup_two():
    panel = stable_panel()
    first = source_family("first", panel)
    second = source_family("second", panel)
    rows = (
        exact_row("first", 0, panel, first),
        exact_row("second", 1, panel, second),
    )
    dataset = DatasetManifestV2.build(rows, source_commit=git("dataset"))
    registry = SourceFamilyRegistryV2.build(
        (first, second), source_commit=git("registry")
    )
    return dataset, registry


def test_two_works_share_slots_but_bind_different_strict_artifacts():
    dataset, registry = setup_two()
    validate_dataset_v2_family_registry(dataset, registry)
    assert dataset.rows[0].candidate_panel.sha256 == (
        dataset.rows[1].candidate_panel.sha256
    )
    assert {
        tuple(a.artifact_pcm_sha256 for a in row.candidate_artifacts)
        for row in dataset.rows
    }.__len__() == 2


def test_missing_candidate_from_family_inventory_is_refused():
    dataset, registry = setup_two()
    row = dataset.rows[0]
    first = row.candidate_artifacts[0]
    forged = CandidateArtifactBinding(
        **{
            **first.__dict__,
            "recipe_id": sha("forged-recipe"),
            "artifact_pcm_sha256": sha("forged-pcm"),
            "recipe_semantic_sha256": sha("forged-semantic"),
        }
    )
    changed = CounterfactualRiskRowV2(
        **{
            **row.__dict__,
            "candidate_artifacts": (forged, row.candidate_artifacts[1]),
        }
    )
    damaged = DatasetManifestV2.build(
        (changed, dataset.rows[1]), source_commit=git("dataset")
    )
    with pytest.raises(SourceFamilyV2Error, match="absent from the strict family"):
        validate_dataset_v2_family_registry(damaged, registry)


def test_root_grid_and_query_mismatch_are_refused():
    dataset, registry = setup_two()
    row = dataset.rows[0]
    bad_mix = CounterfactualRiskRowV2(
        **{**row.__dict__, "mixture_pcm_sha256": sha("other-mixture")}
    )
    with pytest.raises(SourceFamilyV2Error, match="mixture differs"):
        validate_dataset_v2_family_registry(
            DatasetManifestV2.build(
                (bad_mix, dataset.rows[1]), source_commit=git("dataset")
            ),
            registry,
        )

    bad_grid = CounterfactualRiskRowV2(
        **{
            **row.__dict__,
            "cell": CellGeometry(
                **{**row.cell.__dict__, "spectral_grid_sha256": sha("other-grid")}
            ),
        }
    )
    with pytest.raises(SourceFamilyV2Error, match="unregistered spectral grid"):
        validate_dataset_v2_family_registry(
            DatasetManifestV2.build(
                (bad_grid, dataset.rows[1]), source_commit=git("dataset")
            ),
            registry,
        )

    bad_group = GroupFamilyIdentity(
        **{**row.group_family.__dict__, "query_condition_sha256": sha("other-query")}
    )
    bad_query = CounterfactualRiskRowV2(
        **{**row.__dict__, "group_family": bad_group}
    )
    # Artifact bindings name the original source family and remain structurally valid;
    # the strict family verifier is responsible for the query inventory check.
    with pytest.raises(SourceFamilyV2Error, match="unregistered query condition"):
        validate_dataset_v2_family_registry(
            DatasetManifestV2.build(
                (bad_query, dataset.rows[1]), source_commit=git("dataset")
            ),
            registry,
        )
