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
from audio_extract.counterfactual_risk_source_family_v2 import (
    ArtifactNode,
    ArtifactRef,
    SourceFamilyCertificateV2,
    SourceFamilyRegistryV2,
    SourceFamilyV2Error,
    validate_dataset_family_registry_v2,
)


def sha(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def git(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def ref(name: str, *, pcm_name: str | None = None) -> ArtifactRef:
    return ArtifactRef(
        recipe_id=sha(f"recipe:{name}"),
        artifact_pcm_sha256=sha(f"pcm:{pcm_name or name}"),
    )


def node(role: str, name: str, *, parents=(), pcm_name: str | None = None):
    return ArtifactNode(
        role=role,
        identity=ref(name, pcm_name=pcm_name),
        parents=tuple(sorted(parents)),
    )


def certificate(
    name: str,
    *,
    no_vocal_alias: bool = False,
    derived=None,
    closed_world: bool = True,
):
    mixture = node("mixture_root", f"{name}:mixture")
    accompaniment = node(
        "accompaniment_truth",
        f"{name}:a",
        pcm_name=(f"{name}:mixture" if no_vocal_alias else None),
    )
    vocal = node("vocal_truth", f"{name}:v")
    # One globally-ordered frozen candidate-recipe panel shared across all
    # families ("candidate-0"/"candidate-1" recipes); each family decodes its
    # own per-family artifact PCM for those recipes.
    default_derived = (
        node(
            "candidate",
            "candidate-0",
            parents=(mixture.identity,),
            pcm_name=f"{name}:candidate-0",
        ),
        node(
            "candidate",
            "candidate-1",
            parents=(mixture.identity,),
            pcm_name=f"{name}:candidate-1",
        ),
    )
    values = tuple(default_derived if derived is None else derived)
    values = tuple(sorted(values, key=lambda value: value.identity))
    return SourceFamilyCertificateV2(
        mixture_root=mixture,
        accompaniment_truth=accompaniment,
        vocal_truth=vocal,
        derived_artifacts=values,
        spectral_grid_sha256s=(sha(f"{name}:grid"),),
        query_condition_sha256s=(sha(f"{name}:query"),),
        discovery_manifest_sha256=sha(f"{name}:discovery"),
        derivation_policy_sha256=sha(f"{name}:policy"),
        verifier_commit=git(f"{name}:verifier"),
        closed_world=closed_world,
    )


def row_for(value: SourceFamilyCertificateV2, index: int = 0):
    candidate_nodes = tuple(
        item for item in value.derived_artifacts if item.role == "candidate"
    )
    candidate_ids = tuple(
        CandidateIdentity(
            item.identity.recipe_id,
            item.identity.artifact_pcm_sha256,
        )
        for item in candidate_nodes
    )
    risks = np.asarray(
        [
            [0.1 + 0.01 * candidate_index, 0.2, 0.3]
            for candidate_index in range(len(candidate_ids))
        ],
        dtype=np.float64,
    )
    return CounterfactualRiskRow(
        group_family=GroupFamilyIdentity(
            work_id=f"work-{index}",
            recording_session_id=f"session-{index}",
            target_singer_id=f"singer-{index}",
            source_family_sha256=value.sha256,
            query_condition_sha256=value.query_condition_sha256s[0],
        ),
        mixture_pcm_sha256=value.mixture_root.identity.artifact_pcm_sha256,
        accompaniment_truth_pcm_sha256=(
            value.accompaniment_truth.identity.artifact_pcm_sha256
        ),
        vocal_truth_pcm_sha256=value.vocal_truth.identity.artifact_pcm_sha256,
        cell=CellGeometry(
            sample_rate_hz=44_100,
            resolution_ms=500,
            start_frame=index * 100,
            end_frame=(index + 1) * 100,
            band_low_hz=0,
            band_high_hz=500,
            spectral_grid_sha256=value.spectral_grid_sha256s[0],
        ),
        candidates=candidate_ids,
        metric_names=("voice", "hole", "artifact"),
        metric_units=("ratio", "ratio", "ratio"),
        metric_directions=(
            "lower_is_better",
            "lower_is_better",
            "lower_is_better",
        ),
        features=np.asarray([0.1, 0.2], dtype=np.float64),
        exact_risks=risks,
        available=np.ones_like(risks, dtype=bool),
        feature_contract_sha256=sha("features"),
        metric_contract_sha256=sha("metrics"),
        route_policy_sha256=sha("route-policy"),
    )


def dataset_for(*values: SourceFamilyCertificateV2):
    return DatasetManifest.build(
        [row_for(value, index) for index, value in enumerate(values)],
        source_commit=git("dataset"),
    )


def registry_for(*values: SourceFamilyCertificateV2):
    return SourceFamilyRegistryV2.build(values, source_commit=git("registry"))


def test_strict_registry_binds_dataset_rows():
    first = certificate("first")
    second = certificate("second")
    validate_dataset_family_registry_v2(
        dataset_for(first, second),
        registry_for(first, second),
    )


def test_multi_parent_convergent_ensemble_is_valid_when_every_path_is_mixture_only():
    base = certificate("ensemble")
    first, second = base.derived_artifacts
    ensemble = node(
        "candidate",
        "ensemble:combined",
        parents=(first.identity, second.identity),
    )
    value = SourceFamilyCertificateV2(
        **{
            **base.__dict__,
            "derived_artifacts": tuple(
                sorted((first, second, ensemble), key=lambda item: item.identity)
            ),
        }
    )
    value.validate()
    validate_dataset_family_registry_v2(
        dataset_for(value),
        registry_for(value),
    )


def test_one_valid_parent_cannot_hide_a_clean_truth_parent():
    base = certificate("mixed-parent")
    mixture_child = base.derived_artifacts[0]
    contaminated = node(
        "candidate",
        "mixed-parent:combined",
        parents=(
            mixture_child.identity,
            base.accompaniment_truth.identity,
        ),
    )
    value = SourceFamilyCertificateV2(
        **{
            **base.__dict__,
            "derived_artifacts": tuple(
                sorted(
                    (*base.derived_artifacts, contaminated),
                    key=lambda item: item.identity,
                )
            ),
        }
    )
    with pytest.raises(
        SourceFamilyV2Error,
        match="ancestry must terminate only at mixture_root",
    ):
        value.validate()


def test_global_cycle_is_refused_even_when_an_unrelated_path_reaches_mixture():
    base = certificate("cycle")
    first_ref = ref("cycle:first")
    second_ref = ref("cycle:second")
    first = ArtifactNode(
        role="candidate",
        identity=first_ref,
        parents=tuple(sorted((base.mixture_root.identity, second_ref))),
    )
    second = ArtifactNode(
        role="candidate",
        identity=second_ref,
        parents=(first_ref,),
    )
    value = SourceFamilyCertificateV2(
        **{
            **base.__dict__,
            "derived_artifacts": tuple(
                sorted((first, second), key=lambda item: item.identity)
            ),
        }
    )
    with pytest.raises(SourceFamilyV2Error, match="contains a cycle"):
        value.validate()


def test_no_vocal_pcm_alias_is_disambiguated_by_full_artifact_reference():
    value = certificate("no-vocal", no_vocal_alias=True)
    assert (
        value.mixture_root.identity.artifact_pcm_sha256
        == value.accompaniment_truth.identity.artifact_pcm_sha256
    )
    assert value.mixture_root.identity != value.accompaniment_truth.identity
    value.validate()
    validate_dataset_family_registry_v2(
        dataset_for(value),
        registry_for(value),
    )


def test_orphan_parent_and_noncanonical_node_order_are_refused():
    base = certificate("orphan")
    orphan = node(
        "candidate",
        "orphan:derived",
        parents=(ref("orphan:missing"),),
    )
    value = SourceFamilyCertificateV2(
        **{**base.__dict__, "derived_artifacts": (orphan,)}
    )
    with pytest.raises(SourceFamilyV2Error, match="unknown parent references"):
        value.validate()

    first, second = base.derived_artifacts
    reversed_value = SourceFamilyCertificateV2(
        **{**base.__dict__, "derived_artifacts": (second, first)}
    )
    if second.identity < first.identity:
        reversed_value = SourceFamilyCertificateV2(
            **{**base.__dict__, "derived_artifacts": (first, second)}
        )
    with pytest.raises(SourceFamilyV2Error, match="canonically ordered"):
        reversed_value.validate()


def test_unregistered_family_candidate_grid_query_and_truth_are_refused():
    value = certificate("binding")
    original = row_for(value)
    registry = registry_for(value)

    forged_family = CounterfactualRiskRow(
        **{
            **original.__dict__,
            "group_family": GroupFamilyIdentity(
                **{
                    **original.group_family.__dict__,
                    "source_family_sha256": sha("forged-family"),
                }
            ),
        }
    )
    with pytest.raises(SourceFamilyV2Error, match="unregistered source-family"):
        validate_dataset_family_registry_v2(
            DatasetManifest.build([forged_family], source_commit=git("dataset")),
            registry,
        )

    forged_candidate = CounterfactualRiskRow(
        **{
            **original.__dict__,
            "candidates": (
                CandidateIdentity(
                    sha("forged-recipe"),
                    original.candidates[0].artifact_pcm_sha256,
                ),
                *original.candidates[1:],
            ),
        }
    )
    with pytest.raises(SourceFamilyV2Error, match="candidate is absent"):
        validate_dataset_family_registry_v2(
            DatasetManifest.build([forged_candidate], source_commit=git("dataset")),
            registry,
        )

    forged_grid = CounterfactualRiskRow(
        **{
            **original.__dict__,
            "cell": CellGeometry(
                **{
                    **original.cell.__dict__,
                    "spectral_grid_sha256": sha("forged-grid"),
                }
            ),
        }
    )
    with pytest.raises(SourceFamilyV2Error, match="unregistered spectral grid"):
        validate_dataset_family_registry_v2(
            DatasetManifest.build([forged_grid], source_commit=git("dataset")),
            registry,
        )

    forged_query = CounterfactualRiskRow(
        **{
            **original.__dict__,
            "group_family": GroupFamilyIdentity(
                **{
                    **original.group_family.__dict__,
                    "query_condition_sha256": sha("forged-query"),
                }
            ),
        }
    )
    with pytest.raises(SourceFamilyV2Error, match="unregistered query condition"):
        validate_dataset_family_registry_v2(
            DatasetManifest.build([forged_query], source_commit=git("dataset")),
            registry,
        )

    forged_truth = CounterfactualRiskRow(
        **{
            **original.__dict__,
            "accompaniment_truth_pcm_sha256": sha("forged-a"),
        }
    )
    with pytest.raises(SourceFamilyV2Error, match="accompaniment truth differs"):
        validate_dataset_family_registry_v2(
            DatasetManifest.build([forged_truth], source_commit=git("dataset")),
            registry,
        )


def test_candidate_artifact_reference_cannot_cross_families():
    first = certificate("first")
    second = certificate("second")
    duplicate = ArtifactNode(
        role="candidate",
        identity=first.derived_artifacts[0].identity,
        parents=(second.mixture_root.identity,),
    )
    second = SourceFamilyCertificateV2(
        **{
            **second.__dict__,
            "derived_artifacts": tuple(
                sorted(
                    (duplicate, second.derived_artifacts[1]),
                    key=lambda item: item.identity,
                )
            ),
        }
    )
    with pytest.raises(
        SourceFamilyV2Error,
        match="candidate artifact identity appears in multiple families",
    ):
        registry_for(first, second)


def test_certificate_identity_binds_closed_world_inventory_and_query_scope():
    value = certificate("identity")
    changed_inventory = SourceFamilyCertificateV2(
        **{
            **value.__dict__,
            "discovery_manifest_sha256": sha("different-discovery"),
        }
    )
    changed_query = SourceFamilyCertificateV2(
        **{
            **value.__dict__,
            "query_condition_sha256s": (sha("different-query"),),
        }
    )
    assert value.sha256 != changed_inventory.sha256
    assert value.sha256 != changed_query.sha256
