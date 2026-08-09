"""Bind stable-slot D0/R0 dataset v2 rows to strict source-family v2 graphs."""

from __future__ import annotations

from .counterfactual_risk_dataset_contract_v2 import DatasetManifestV2
from .counterfactual_risk_source_family_v2 import (
    ArtifactRef,
    SourceFamilyRegistryV2,
    SourceFamilyV2Error,
)


def validate_dataset_v2_family_registry(
    dataset: DatasetManifestV2,
    registry: SourceFamilyRegistryV2,
) -> None:
    """Verify roots, grid/query scope, and every per-source candidate artifact."""

    dataset.validate()
    registry.validate()
    certificates = registry.by_sha256()
    for row in dataset.rows:
        family_sha = row.group_family.source_family_sha256
        certificate = certificates.get(family_sha)
        if certificate is None:
            raise SourceFamilyV2Error(
                "dataset row names an unregistered source-family certificate"
            )
        if row.mixture_pcm_sha256 != (
            certificate.mixture_root.identity.artifact_pcm_sha256
        ):
            raise SourceFamilyV2Error("dataset mixture differs from family root")
        if row.accompaniment_truth_pcm_sha256 != (
            certificate.accompaniment_truth.identity.artifact_pcm_sha256
        ):
            raise SourceFamilyV2Error(
                "dataset accompaniment truth differs from family certificate"
            )
        if row.vocal_truth_pcm_sha256 != (
            certificate.vocal_truth.identity.artifact_pcm_sha256
        ):
            raise SourceFamilyV2Error(
                "dataset vocal truth differs from family certificate"
            )
        if row.cell.spectral_grid_sha256 not in certificate.spectral_grid_sha256s:
            raise SourceFamilyV2Error(
                "dataset cell uses an unregistered spectral grid"
            )
        if row.group_family.query_condition_sha256 not in (
            certificate.query_condition_sha256s
        ):
            raise SourceFamilyV2Error(
                "dataset row uses an unregistered query condition"
            )
        for artifact in row.candidate_artifacts:
            ref = ArtifactRef(
                recipe_id=artifact.recipe_id,
                artifact_pcm_sha256=artifact.artifact_pcm_sha256,
            )
            if not certificate.contains_candidate(ref):
                raise SourceFamilyV2Error(
                    "dataset candidate artifact is absent from the strict family inventory"
                )
