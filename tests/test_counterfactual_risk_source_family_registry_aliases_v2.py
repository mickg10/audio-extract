import hashlib

import pytest

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


def make_certificate(name: str) -> SourceFamilyCertificateV2:
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
        sorted(
            (
                ArtifactNode(
                    "candidate",
                    ArtifactRef(
                        sha(f"{name}:candidate-{index}:recipe"),
                        sha(f"{name}:candidate-{index}:pcm"),
                    ),
                    (mixture.identity,),
                )
                for index in range(2)
            ),
            key=lambda item: item.identity,
        )
    )
    return SourceFamilyCertificateV2(
        mixture_root=mixture,
        accompaniment_truth=accompaniment,
        vocal_truth=vocal,
        derived_artifacts=candidates,
        spectral_grid_sha256s=(sha(f"{name}:grid"),),
        query_condition_sha256s=(sha(f"{name}:query"),),
        discovery_manifest_sha256=sha(f"{name}:discovery"),
        derivation_policy_sha256=sha("policy"),
        verifier_commit=git("verifier"),
    )


def rebuild_candidates(
    certificate: SourceFamilyCertificateV2,
    mixture: ArtifactNode,
) -> tuple[ArtifactNode, ...]:
    return tuple(
        sorted(
            (
                ArtifactNode(
                    node.role,
                    node.identity,
                    (mixture.identity,),
                )
                for node in certificate.derived_artifacts
            ),
            key=lambda item: item.identity,
        )
    )


def test_same_decoded_mixture_cannot_be_split_by_alternate_recipe_identity():
    first = make_certificate("first")
    second = make_certificate("second")
    aliased_mix = ArtifactNode(
        "mixture_root",
        ArtifactRef(
            sha("second:alternate-mix-recipe"),
            first.mixture_root.identity.artifact_pcm_sha256,
        ),
    )
    second = SourceFamilyCertificateV2(
        **{
            **second.__dict__,
            "mixture_root": aliased_mix,
            "derived_artifacts": rebuild_candidates(second, aliased_mix),
        }
    )
    with pytest.raises(
        SourceFamilyV2Error,
        match="decoded mixture PCM appears in multiple source families",
    ):
        SourceFamilyRegistryV2.build(
            (first, second), source_commit=git("registry")
        )


def test_same_closed_world_inventory_cannot_issue_multiple_family_certificates():
    first = make_certificate("first")
    second = make_certificate("second")
    second = SourceFamilyCertificateV2(
        **{
            **second.__dict__,
            "discovery_manifest_sha256": first.discovery_manifest_sha256,
        }
    )
    with pytest.raises(
        SourceFamilyV2Error,
        match="closed-world discovery manifest appears in multiple source families",
    ):
        SourceFamilyRegistryV2.build(
            (first, second), source_commit=git("registry")
        )


def test_distinct_mixture_and_inventory_families_remain_valid():
    first = make_certificate("first")
    second = make_certificate("second")
    registry = SourceFamilyRegistryV2.build(
        (first, second), source_commit=git("registry")
    )
    assert len(registry.certificates) == 2
