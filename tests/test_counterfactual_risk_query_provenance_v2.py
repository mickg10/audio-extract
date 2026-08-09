import hashlib

import pytest

from audio_extract.counterfactual_risk_source_family_v2 import (
    ArtifactNode,
    ArtifactRef,
    SourceFamilyCertificateV2,
    SourceFamilyV2Error,
)


def sha(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode()).hexdigest()


def git(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()


def ref(name: str) -> ArtifactRef:
    return ArtifactRef(sha(f"recipe:{name}"), sha(f"pcm:{name}"))


def base_certificate(query_parent: ArtifactRef) -> SourceFamilyCertificateV2:
    mixture = ArtifactNode("mixture_root", ref("mixture"))
    accompaniment = ArtifactNode("accompaniment_truth", ref("a-truth"))
    vocal = ArtifactNode("vocal_truth", ref("v-truth"))
    candidates = (
        ArtifactNode("candidate", ref("candidate-a"), (mixture.identity,)),
        ArtifactNode("candidate", ref("candidate-b"), (mixture.identity,)),
    )
    query = ArtifactNode("query_source", ref("query"), (query_parent,))
    derived = tuple(sorted((*candidates, query), key=lambda item: item.identity))
    return SourceFamilyCertificateV2(
        mixture_root=mixture,
        accompaniment_truth=accompaniment,
        vocal_truth=vocal,
        derived_artifacts=derived,
        spectral_grid_sha256s=(sha("grid"),),
        query_condition_sha256s=(sha("query-condition"),),
        discovery_manifest_sha256=sha("inventory"),
        derivation_policy_sha256=sha("policy"),
        verifier_commit=git("verifier"),
    )


def test_mixture_derived_query_source_is_inference_safe():
    provisional = base_certificate(ref("placeholder"))
    value = base_certificate(provisional.mixture_root.identity)
    value.validate()


def test_clean_vocal_truth_cannot_be_query_source_at_inference():
    provisional = base_certificate(ref("placeholder"))
    value = base_certificate(provisional.vocal_truth.identity)
    with pytest.raises(
        SourceFamilyV2Error,
        match="query_source ancestry must terminate only at mixture_root",
    ):
        value.validate()


def test_clean_accompaniment_truth_cannot_be_query_source_at_inference():
    provisional = base_certificate(ref("placeholder"))
    value = base_certificate(provisional.accompaniment_truth.identity)
    with pytest.raises(
        SourceFamilyV2Error,
        match="query_source ancestry must terminate only at mixture_root",
    ):
        value.validate()
