import hashlib

import numpy as np
import pytest

from audio_extract.counterfactual_risk_dataset_contract_v1 import (
    CandidateIdentity,
    CellGeometry,
    CounterfactualRiskRow,
    DatasetManifest,
    GroupFamilyIdentity,
)
from audio_extract.counterfactual_risk_source_family_v1 import (
    SourceArtifact,
    SourceFamilyCertificate,
    SourceFamilyError,
    SourceFamilyRegistry,
    validate_dataset_family_registry,
)


def sha(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def git(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def artifact(role: str, name: str, *, parents=(), pcm_name=None):
    return SourceArtifact(
        role=role,
        recipe_ids=(sha(f"recipe:{name}"),),
        artifact_pcm_sha256=sha(f"pcm:{pcm_name or name}"),
        parent_pcm_sha256s=tuple(sorted(parents)),
    )


def certificate(name: str, *, closed_world=True, derived=None):
    mixture = artifact("mixture_root", f"{name}:mixture")
    accompaniment = artifact("accompaniment_truth", f"{name}:a")
    vocal = artifact("vocal_truth", f"{name}:v")
    # One globally-ordered frozen candidate-recipe panel shared across families
    # ("candidate-0"/"candidate-1" recipes), each decoding its own per-family
    # artifact PCM.
    candidates = (
        artifact(
            "candidate",
            "candidate-0",
            parents=(mixture.artifact_pcm_sha256,),
            pcm_name=f"{name}:candidate-0",
        ),
        artifact(
            "candidate",
            "candidate-1",
            parents=(mixture.artifact_pcm_sha256,),
            pcm_name=f"{name}:candidate-1",
        ),
    )
    return SourceFamilyCertificate(
        mixture_root=mixture,
        accompaniment_truth=accompaniment,
        vocal_truth=vocal,
        derived_artifacts=tuple(derived) if derived is not None else candidates,
        spectral_grid_sha256s=(sha(f"{name}:grid"),),
        query_condition_sha256s=(sha(f"{name}:query"),),
        discovery_manifest_sha256=sha(f"{name}:discovery"),
        derivation_policy_sha256=sha(f"{name}:policy"),
        verifier_commit=git(f"{name}:verifier"),
        closed_world=closed_world,
    )


def row_for(value: SourceFamilyCertificate, index: int = 0):
    candidates = tuple(
        CandidateIdentity(
            artifact.recipe_ids[0],
            artifact.artifact_pcm_sha256,
        )
        for artifact in value.derived_artifacts
        if artifact.role == "candidate"
    )
    return CounterfactualRiskRow(
        group_family=GroupFamilyIdentity(
            work_id=f"work-{index}",
            recording_session_id=f"session-{index}",
            target_singer_id=f"singer-{index}",
            source_family_sha256=value.sha256,
            query_condition_sha256=value.query_condition_sha256s[0],
        ),
        mixture_pcm_sha256=value.mixture_root.artifact_pcm_sha256,
        accompaniment_truth_pcm_sha256=(
            value.accompaniment_truth.artifact_pcm_sha256
        ),
        vocal_truth_pcm_sha256=value.vocal_truth.artifact_pcm_sha256,
        cell=CellGeometry(
            sample_rate_hz=44_100,
            resolution_ms=500,
            start_frame=index * 100,
            end_frame=(index + 1) * 100,
            band_low_hz=0,
            band_high_hz=500,
            spectral_grid_sha256=value.spectral_grid_sha256s[0],
        ),
        candidates=candidates,
        metric_names=("voice", "hole", "artifact"),
        metric_units=("ratio", "ratio", "ratio"),
        metric_directions=(
            "lower_is_better",
            "lower_is_better",
            "lower_is_better",
        ),
        features=np.asarray([0.1, 0.2], dtype=np.float64),
        exact_risks=np.asarray(
            [[0.1, 0.2, 0.3], [0.2, 0.1, 0.4]], dtype=np.float64
        ),
        available=np.ones((2, 3), dtype=bool),
        feature_contract_sha256=sha("features"),
        metric_contract_sha256=sha("metrics"),
        route_policy_sha256=sha("route-policy"),
    )


def dataset_for(*values: SourceFamilyCertificate):
    return DatasetManifest.build(
        [row_for(value, index) for index, value in enumerate(values)],
        source_commit=git("dataset"),
    )


def registry_for(*values: SourceFamilyCertificate):
    return SourceFamilyRegistry.build(values, source_commit=git("registry"))


def test_closed_world_registry_binds_every_dataset_row():
    first = certificate("first")
    second = certificate("second")
    dataset = dataset_for(first, second)
    registry = registry_for(first, second)
    validate_dataset_family_registry(dataset, registry)
    assert registry.sha256.startswith("sha256:")
    assert first.sha256 != second.sha256


def test_arbitrary_source_family_string_cannot_bypass_registry():
    value = certificate("family")
    original = row_for(value)
    forged = CounterfactualRiskRow(
        **{
            **original.__dict__,
            "group_family": GroupFamilyIdentity(
                **{
                    **original.group_family.__dict__,
                    "source_family_sha256": sha("invented-family"),
                }
            ),
        }
    )
    dataset = DatasetManifest.build([forged], source_commit=git("dataset"))
    with pytest.raises(
        SourceFamilyError,
        match="unregistered source-family certificate",
    ):
        validate_dataset_family_registry(dataset, registry_for(value))


def test_open_inventory_and_unknown_parent_are_refused():
    with pytest.raises(SourceFamilyError, match="closed-world inventory"):
        certificate("open", closed_world=False).validate()

    base = certificate("orphan")
    orphan = artifact(
        "candidate",
        "orphan:candidate",
        parents=(sha("missing-parent"),),
    )
    damaged = SourceFamilyCertificate(
        **{**base.__dict__, "derived_artifacts": (orphan,)}
    )
    with pytest.raises(SourceFamilyError, match="unknown parents"):
        damaged.validate()


def test_cycle_in_derivation_graph_is_refused():
    base = certificate("cycle")
    first_pcm = sha("pcm:cycle:first")
    second_pcm = sha("pcm:cycle:second")
    first = SourceArtifact(
        role="candidate",
        recipe_ids=(sha("recipe:cycle:first"),),
        artifact_pcm_sha256=first_pcm,
        parent_pcm_sha256s=(second_pcm,),
    )
    second = SourceArtifact(
        role="candidate",
        recipe_ids=(sha("recipe:cycle:second"),),
        artifact_pcm_sha256=second_pcm,
        parent_pcm_sha256s=(first_pcm,),
    )
    damaged = SourceFamilyCertificate(
        **{**base.__dict__, "derived_artifacts": (first, second)}
    )
    with pytest.raises(SourceFamilyError, match="contains a cycle"):
        damaged.validate()


def test_candidate_must_trace_to_exact_mixture_root():
    base = certificate("truth-parent")
    damaged_candidate = artifact(
        "candidate",
        "truth-parent:candidate",
        parents=(base.accompaniment_truth.artifact_pcm_sha256,),
    )
    damaged = SourceFamilyCertificate(
        **{**base.__dict__, "derived_artifacts": (damaged_candidate,)}
    )
    with pytest.raises(
        SourceFamilyError,
        match="does not trace to the exact mixture root",
    ):
        damaged.validate()


def test_row_candidate_grid_query_and_truth_must_match_certificate():
    value = certificate("match")
    original = row_for(value)
    registry = registry_for(value)

    bad_candidate = CounterfactualRiskRow(
        **{
            **original.__dict__,
            "candidates": (
                CandidateIdentity(sha("unknown-recipe"), original.candidates[0].artifact_pcm_sha256),
                original.candidates[1],
            ),
        }
    )
    with pytest.raises(SourceFamilyError, match="candidate is absent"):
        validate_dataset_family_registry(
            DatasetManifest.build([bad_candidate], source_commit=git("dataset")),
            registry,
        )

    bad_grid = CounterfactualRiskRow(
        **{
            **original.__dict__,
            "cell": CellGeometry(
                **{
                    **original.cell.__dict__,
                    "spectral_grid_sha256": sha("other-grid"),
                }
            ),
        }
    )
    with pytest.raises(SourceFamilyError, match="unregistered spectral grid"):
        validate_dataset_family_registry(
            DatasetManifest.build([bad_grid], source_commit=git("dataset")),
            registry,
        )

    bad_query = CounterfactualRiskRow(
        **{
            **original.__dict__,
            "group_family": GroupFamilyIdentity(
                **{
                    **original.group_family.__dict__,
                    "query_condition_sha256": sha("other-query"),
                }
            ),
        }
    )
    with pytest.raises(SourceFamilyError, match="unregistered query condition"):
        validate_dataset_family_registry(
            DatasetManifest.build([bad_query], source_commit=git("dataset")),
            registry,
        )

    bad_truth = CounterfactualRiskRow(
        **{
            **original.__dict__,
            "vocal_truth_pcm_sha256": sha("other-vocal"),
        }
    )
    with pytest.raises(SourceFamilyError, match="vocal truth differs"):
        validate_dataset_family_registry(
            DatasetManifest.build([bad_truth], source_commit=git("dataset")),
            registry,
        )


def test_candidate_recipe_pcm_pair_cannot_appear_in_multiple_families():
    first = certificate("first")
    second = certificate("second")
    duplicate = second.derived_artifacts[0]
    duplicate = SourceArtifact(
        **{
            **duplicate.__dict__,
            "recipe_ids": first.derived_artifacts[0].recipe_ids,
            "artifact_pcm_sha256": (
                first.derived_artifacts[0].artifact_pcm_sha256
            ),
        }
    )
    second = SourceFamilyCertificate(
        **{
            **second.__dict__,
            "derived_artifacts": (duplicate, second.derived_artifacts[1]),
        }
    )
    with pytest.raises(
        SourceFamilyError,
        match="candidate recipe/PCM identity appears in multiple families",
    ):
        registry_for(first, second)


def test_certificate_identity_binds_inventory_policy_and_query_scope():
    value = certificate("identity")
    changed_inventory = SourceFamilyCertificate(
        **{
            **value.__dict__,
            "discovery_manifest_sha256": sha("different-discovery"),
        }
    )
    changed_query = SourceFamilyCertificate(
        **{
            **value.__dict__,
            "query_condition_sha256s": (sha("different-query"),),
        }
    )
    assert value.sha256 != changed_inventory.sha256
    assert value.sha256 != changed_query.sha256
